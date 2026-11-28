"""Actual-worker MEGA qualification: one combined return, no formal mutation."""
import argparse
from contextlib import ExitStack,nullcontext
import datetime as dt
import math
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import uuid
from .common import ROOT,atomic_json,digest,fingerprint,lock,read,revision,worker_failure

MODES=('reference','eager','graph','interrupt','resume','graph-interrupt','graph-resume')
STEPS=8


def window(state,rows,logical):
    """Use one matched even prefix; preserve the epoch and source cursor."""
    if state['training_complete']:return 0
    if not 0<=state['cursor']<=rows or type(logical) is not int or logical<1:raise ValueError('Invalid pending sample window')
    available=(rows-state['cursor'])//logical
    return min(STEPS,available)//2*2


def failure_detail(output,error):
    """Cover setup/import failures too; retain native evidence separately."""
    from .mega_refinement import error_chain
    worker_failure(output,error)
    detail=read(output/'failure-detail.json') if (output/'failure-detail.json').exists() else {}
    atomic_json(output/'failure-detail.json',dict(detail,chain=error_chain(error),
        native_error=read(output/'worker-error.json')))


def binding():
    files=('mega_qualification.py','mega_execution.py','mega_refinement.py','mega_fft.py','execution_plugins.py','worker.py','models.py')
    return {n:digest(ROOT/'experiments/baseline_delivery'/n) for n in files}


def child_spec(suite,mode):
    if mode not in MODES:raise ValueError('Unknown qualification mode')
    target='experiments.baseline_delivery.mega_qualification'
    return dict(target=target,cwd=str(ROOT),argv=[str(ROOT/'.venv-mega/bin/python'),'-B','-m',target,
        '--request',str(suite/'request.json'),'--mode',mode,'--output',str(suite/mode)])


def technical(results):
    return set(results)==set(MODES) and all(r.get('passed') is True for r in results.values()) and all(results[m].get('recovery_passed') is True for m in ('resume','graph-resume'))


def native(path,mode,output):
    request=read(path);suite=path.parent
    if request['source_revision']!=revision() or request['binding']!=binding():raise ValueError('Qualification code changed')
    if output.resolve()!=(suite/mode).resolve():raise ValueError('Qualification output differs')
    import torch
    import numpy as np
    from . import worker as w
    from .execution_plugins import Hooks,create
    from .migration import checked_pointer,copy_verified
    from .logical_batch import make_controller,planned_batch_size
    from .data import Split
    from .attention_update_native import NativeUpdate
    from .key_bias_worker_admission import scientific
    from .key_bias_stability_native import compare_values
    from .diagnostic_watch import mark
    from .mega_refinement import error_chain
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=read(suite/'source/config.json');pointer=read(suite/'source/checkpoints/latest.json')
    base=w.load_checkpoint(suite/'source',pointer['contract'])
    controller=make_controller(config['adaptive_memory'],config['batch_size'],base['adaptive_memory'])
    logical=planned_batch_size(controller,config,base['step']);micro=controller.micro_batch
    mark('mega_joint_data',flush=True)
    train=Split(request['data'],config['task'],'train',request['cache'],config['data_hashes']['train'])
    steps=window(base,len(train),logical)
    atomic_json(output/'setup.json',dict(phase='pending_window',source_step=base['step'],
        cursor=base['cursor'],train_rows=len(train),logical_batch=logical,micro_batch=micro,planned_steps=steps))
    if steps<2:
        atomic_json(output/'result.json',dict(mode=mode,request_sha256=digest(path),skipped=True,
            reason='Fewer than two pending full batches before epoch end',planned_steps=0,
            temporary_updates=0,formal_updates=0))
        return
    resuming=mode.endswith('resume');interrupting=mode.endswith('interrupt')
    prior=suite/('graph-interrupt' if mode.startswith('graph') else 'interrupt') if resuming else suite/'source'
    cp,generation=checked_pointer(prior)
    copy_verified(prior,output,{'checkpoints/'+generation.name:cp['sha256']})
    atomic_json(output/'checkpoints/latest.json',{**cp,'previous':[]})
    before=w.load_checkpoint(output,cp['contract'])
    hasher=SimpleNamespace(t=torch,np=np)
    state_hash=lambda state:NativeUpdate.state_hash(hasher,state)
    policy=None;updates=[]
    selection=None if mode=='reference' else dict(plugin='mega.fft_tail_bf16',version=1,graph=mode.startswith('graph'))
    def install(model):
        nonlocal policy
        policy=create(selection,config,torch,model)
        return policy if policy else nullcontext()
    def configure(c,state):
        # Instance-local finite control only. Keep original tier indices/sizes;
        # do not let timing-driven tuning change the A/B sample groups.
        c.observe=lambda *args,**kwargs:None
        c.policy['status_interval']=.001
    def progress(h):
        actual=h.get('adaptive_memory',{})
        used_micro=actual.get('used_micro_batch');used_logical=actual.get('used_logical_batch',h.get('logical_batch'))
        if used_micro!=micro or used_logical!=logical:raise ValueError('Qualification batch changed during retry')
        updates.append(dict(step=h['optimizer_step'],cursor=h['next_row_cursor'],loss=h['loss'],
            used_micro_batch=used_micro,used_logical_batch=used_logical))
    def before_save(state):
        state['mega_execution']=dict(selection=selection,binding=request['binding'])
    def after_save(state):
        atomic_json(output/'mega-execution.json',dict(policy.status() if policy else dict(plugin=None),step=state['step'],binding=request['binding']))
        environment=output/'environment.json'
        if policy and environment.exists():
            value=read(environment);value.update(precision=policy.status()['precision'],execution_plugin=selection)
            atomic_json(environment,value)
    handlers={};w.PAUSE=False
    for name in ('SIGINT','SIGTERM','SIGHUP'):
        if hasattr(signal,name):
            sig=getattr(signal,name);handlers[sig]=signal.signal(sig,w.request_pause)
    limit=base['step']+(steps//2 if interrupting else steps)
    try:
        mark('mega_joint_worker',flush=True)
        with lock(output/'worker.lock'):
            code=w.run(config,request['data'],request['cache'],output,'cuda:0',resume=True,stop_after=limit,
                execution=Hooks(model_context=install,configure_controller=configure,on_progress=progress,
                    acceleration_state=lambda saved:{**(saved or {}),'mode':'eager','benchmarked':True,'reason':'mega_joint_eager_outer','benchmark':None},
                    before_save=before_save,after_save=after_save))
        if code!=75:raise ValueError('Finite actual worker did not save pause')
        final=w.load_checkpoint(output,cp['contract']);count=limit-before['step']
        if (final['step']!=limit or final['epoch']!=base['epoch'] or final['cursor']!=before['cursor']+count*logical
                or final['sample_clock']['progress']!=before['sample_clock']['progress']+count*logical
                or len(updates)!=count or any(not math.isfinite(u['loss']) for u in updates)):
            raise ValueError('Finite actual worker budget/row clock differs')
        clean=lambda state:{k:v for k,v in scientific(state).items() if k!='mega_execution'}
        result=dict(passed=True,mode=mode,planned_steps=steps,request_sha256=digest(path),source_step=base['step'],step=final['step'],
            loaded_scientific_sha256=state_hash(clean(before)),scientific_sha256=state_hash(clean(final)),
            checkpoint_sha256=checked_pointer(output)[0]['sha256'],updates=updates,micro_batch=micro,logical_batch=logical,
            training_seconds=final['training_seconds']-before['training_seconds'],temporary_updates=count,formal_updates=0,
            execution=policy.status() if policy else dict(plugin=None),
            row_ids_sha256=state_hash(final['permutation'][base['cursor']:final['cursor']]))
        if resuming:
            if result['loaded_scientific_sha256']!=read(prior/'result.json')['scientific_sha256']:raise ValueError('Resume start differs')
            uninterrupted=w.load_checkpoint(suite/('graph' if mode.startswith('graph') else 'eager'),pointer['contract'])
            compare_values(hasher,clean(uninterrupted),clean(final))
            result['recovery_passed']=True
        # All routes use original FP32 evaluation, fixed FIRST validation rows;
        # no test split or validation-based checkpoint/hyperparameter selection.
        if mode in ('reference','eager','graph'):
            val=Split(request['data'],config['task'],'val',request['cache'],config['data_hashes']['val'])
            class View:
                def __len__(self):return min(256,len(val))
                def batch(self,ids,device):return val.batch(ids,device)
            model=w.Classifier(config,request['cache']).cuda();model.load_state_dict(final['model'])
            result['validation']=w.evaluate(model,View(),'cuda:0',micro)
            result['validation_scope']=dict(split='val',first_rows=min(256,len(val)),test_access=False,selection=False)
        if policy and (policy.calls<=0 or policy.fft.calls<=0 or policy.islands.calls<=0):raise ValueError('Plugin not observed')
        atomic_json(output/'result.json',result)
    except BaseException as error:
        atomic_json(output/'failure-detail.json',dict(chain=error_chain(error)));raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def campaign(args):
    from .bootstrap import check_exchange
    from .mega_diagnostic import select_job,snapshot
    from .runner import discover_gpus
    from .attention_diagnostic import gpu_preflight
    from .diagnostic_watch import Supervisor
    from .return_artifacts import upload_declared
    check_exchange();revision();gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1 or not gpu_preflight(inventory[gpus[0]])['idle']:raise ValueError('Pause original runner first; one idle GPU required')
    source=args.source_suite.resolve();suite=ROOT/'runs'/('mega-qualify-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    with ExitStack() as locks:
        for p in (source/'progress-supervisor.lock',source/'suite.lock'):locks.enter_context(lock(p))
        job=select_job(source,args.job)
        if job is None:print('No pending MEGA checkpoint; qualification skipped.',flush=True);return 0
        locks.enter_context(lock(source/'jobs'/job/'worker.lock'))
        request=snapshot(source,suite,job,args.cache)
        request.update(binding=binding(),qualification_steps=STEPS,graph_default=False)
        atomic_json(suite/'request.json',request)
        print('MEGA actual-worker qualification: '+str(suite)+' | '+job,flush=True)
        supervisor=Supervisor(900,300);report=dict(schema='qkan-mega-qualification.v1',results={},failures={},
            formal_policy_changed=False,quality_admitted=False,formal_updates=0,child_executions=supervisor.records)
        safe=True;plan=None
        try:
            for mode in MODES:
                if mode.endswith('resume') and ('graph-interrupt' if mode.startswith('graph') else 'interrupt') not in report['results']:
                    report['failures'][mode]=dict(reason='Interrupted checkpoint unavailable');continue
                print('MEGA actual worker: '+mode,flush=True)
                prior_count=len(supervisor.records)
                try:
                    code=supervisor.run(child_spec(suite,mode),suite/(mode+'.log'),dict(os.environ,
                        CUDA_VISIBLE_DEVICES=gpus[0],CUBLAS_WORKSPACE_CONFIG=':4096:8',QKAN_MEGA_SOURCE=str(ROOT/'.bootstrap/strong-upstream/mega')))
                    if len(supervisor.records)!=prior_count+1 or not supervisor.records[-1]['termination']['group_verified']:
                        safe=False;raise ValueError('Worker cleanup unverified')
                    if code!=0:raise RuntimeError('Qualification child failed')
                    value=read(suite/mode/'result.json')
                    if mode=='reference' and value.get('skipped') is True:
                        if (value.get('request_sha256')!=digest(suite/'request.json') or value.get('temporary_updates')!=0
                                or value.get('formal_updates')!=0):raise ValueError('Skip identity differs')
                        report['skipped']=value;break
                    steps=value.get('planned_steps')
                    if type(steps) is not int or steps not in range(2,STEPS+1,2):raise ValueError('Finite window budget differs')
                    if plan is None:plan=steps
                    elif steps!=plan:raise ValueError('Child windows differ')
                    report['planned_steps']=plan
                    expected=steps//2 if mode.endswith(('interrupt','resume')) else steps
                    if (value['request_sha256']!=digest(suite/'request.json') or value['mode']!=mode or value['temporary_updates']!=expected
                            or value['formal_updates']!=0 or value['step']!=request['checkpoint']['step']+(steps//2 if mode.endswith('interrupt') else steps)):raise ValueError('Worker result identity differs')
                    report['results'][mode]=value
                except Exception as error:
                    # Child main already wrote native evidence here. Preserve it;
                    # never replace its traceback with the parent's generic error.
                    error_output=suite/'parent-errors'/mode
                    worker_failure(error_output,error)
                    report['failures'][mode]=dict(error=read(error_output/'worker-error.json'))
                    if (suite/mode/'worker-error.json').exists():report['failures'][mode]['native_error']=read(suite/mode/'worker-error.json')
                    if (suite/mode/'failure-detail.json').exists():report['failures'][mode]['detail']=read(suite/mode/'failure-detail.json')
                    if (suite/mode/'setup.json').exists():report['failures'][mode]['setup']=read(suite/mode/'setup.json')
                    if len(supervisor.records)!=prior_count+1 or not supervisor.records[-1]['termination']['group_verified']:safe=False
                    if not safe:break
                    if mode=='reference':
                        report['skipped_remaining']='Shared reference/setup failed; retain evidence before further candidates'
                        break
            report['technical_passed']=technical(report['results'])
            if all(m in report['results'] for m in ('reference','eager','graph')):
                rows=report['results'];ref=rows['reference']
                report['matched_rows']=len({rows[m]['row_ids_sha256'] for m in ('reference','eager','graph')})==1
                report['comparison']={m:dict(training_losses=[u['loss'] for u in rows[m]['updates']],
                    validation=rows[m]['validation'],throughput_gain=ref['training_seconds']/rows[m]['training_seconds']-1) for m in ('eager','graph')}
                report['technical_passed'] &= report['matched_rows']
                report['graph_incremental_throughput_gain']=rows['eager']['training_seconds']/rows['graph']['training_seconds']-1
        finally:
            report['request']=request
            report['source_unchanged']=all(digest(p)==sha for p,sha in request['source_files'].items())
            for mode in MODES:
                for p in (suite/mode/'checkpoints').glob('*.pt'):p.unlink()
            for p in (suite/'source/checkpoints').glob('*.pt'):p.unlink()
            report['owned_checkpoints_removed']=not any(suite.rglob('*.pt'))
            atomic_json(suite/'qualification.json',report);upload_declared(suite,'mega-qualification',['qualification.json'])
    print('Qualification returned; technical checks and validation evidence remain separate. Original policy retained.',flush=True)
    return 0 if safe and report['source_unchanged'] and report['owned_checkpoints_removed'] else 76


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--source-suite',type=Path);g.add_argument('--request',type=Path)
    p.add_argument('--cache',type=Path);p.add_argument('--gpu',default='0');p.add_argument('--job',default='auto')
    p.add_argument('--mode',choices=MODES);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.request:
        if a.mode is None or a.output is None:p.error('Child mode/output required')
        try:native(a.request,a.mode,a.output);return 0
        except BaseException as error:
            failure_detail(a.request.parent/a.mode,error);raise
    if a.cache is None:p.error('cache required')
    return campaign(a)

if __name__=='__main__':raise SystemExit(main())
