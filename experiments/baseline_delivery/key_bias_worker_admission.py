"""Final-worker migration, two-step fresh-process recovery and private real trace.

Four owned children; four total optimizer updates on disposable copies. Never
launches a formal queue. All native imports occur only in the collaborator child.
"""
import argparse
import copy
from contextlib import ExitStack
import datetime as dt
from pathlib import Path
import os
import sys
from types import SimpleNamespace
import uuid

from .common import ROOT,atomic_json,digest,fingerprint,read,revision,lock,worker_failure
from . import key_bias_training as strategy

TARGET='experiments.baseline_delivery.key_bias_worker_admission'
SCHEMA='qkan-key-bias-worker-admission.v1'
MODES=('migration','continuous','interrupt','resume')
EXCLUDE=('training_seconds','adaptive_memory','adaptive_peak_bytes','torch_acceleration','rollback_device')


def scientific(state):
    """Keep every persisted field except explicit execution/time telemetry.

    Controller timing/headroom is process dependent; the candidate pins physical
    and logical sizes before every update. No optimizer/sample/RNG field omitted.
    """
    return {k:v for k,v in state.items() if k not in EXCLUDE}


def bind_pilot(folder):
    path=Path(folder)/'key-bias-stability.json';review=read(strategy.REVIEW)
    if (path.is_symlink() or digest(path)!=review['source_report']['sha256']
            or path.stat().st_size!=review['source_report']['bytes']):
        raise ValueError('Audited immutable successful pilot required')
    value=read(path)
    if (value.get('passed') is not True or value.get('candidate_admitted') is not False
            or value.get('original_checkpoint_and_metadata_unchanged') is not True):
        raise ValueError('Pilot gates/source preservation differ')
    return dict(suite=str(Path(folder).resolve()),sha256=digest(path),review_sha256=digest(strategy.REVIEW))


def completed(source,target=None):
    from .migration import completed_files,copy_verified
    source=Path(source);states=read(source/'status.json');result={}
    for job in read(source/'suite.json')['jobs']:
        name=job['id'];folder=source/'jobs'/name
        if states[name]['state']!='computed': continue
        config=read(folder/'config.json')
        if config['model']!='transformer' or config['task'] not in ('listops','text'):
            raise ValueError('Completed source roster changed; review latest source')
        record=completed_files(folder,config)
        if target is not None: copy_verified(folder,Path(target)/name,record['files'])
        result[name]=record
    if len(result)!=2: raise ValueError('Expected two original completed Transformer results')
    return result


def accept(value,request,request_sha,mode):
    if (value.get('schema')!=SCHEMA or value.get('request_sha256')!=request_sha
            or value.get('mode')!=mode or value.get('implementation')!=request['implementation']
            or value.get('source_checkpoint_sha256')!=request['checkpoint']['sha256']
            or value.get('passed') is not True or value.get('formal_training_started') is not False
            or value.get('candidate_admitted') is not False):
        raise ValueError('Final-worker child identity/result differs')
    expected={'migration':0,'continuous':2,'interrupt':1,'resume':2}[mode]
    if (value.get('step')!=request['checkpoint']['step']+expected
            or len(value.get('updates',[]))!=expected
            or any(v['samples']!=128 or v['micro_batch']!=16 for v in value['updates'])):
        raise ValueError('Final-worker update budget/coverage differs')
    return value


def agreement(results):
    a,b,c=(results[n] for n in ('continuous','interrupt','resume'))
    if (a['scientific_state_sha256']!=c['scientific_state_sha256']
            or a['component_sha256']!=c['component_sha256'] or a['updates']!=c['updates']
            or c['updates'][:1]!=b['updates'] or c['loaded_state_sha256']!=b['scientific_state_sha256']
            or not c.get('trace',{}).get('validation',{}).get('passed')):
        raise ValueError('Actual worker fresh-process recovery/trace differs')
    if 'migration' in results and (a['loaded_state_sha256']!=results['migration']['scientific_state_sha256']
            or b['loaded_state_sha256']!=a['loaded_state_sha256']):
        raise ValueError('Recovery must start from the actual worker-validated zero-update migration copy')
    return dict(passed=True,continuous_updates=2,fresh_process_updates=[1,1],
        scientific_state_sha256=a['scientific_state_sha256'],component_sha256=a['component_sha256'],
        excluded_execution_telemetry=list(EXCLUDE),trace=c['trace'])


def native(path,output,mode):
    from .attention_precision_diagnostic import unchanged,JOB
    from .migration import checked_pointer
    from .data import Split,vocabulary
    from . import worker
    import torch
    import numpy as np
    from .attention_update_native import NativeUpdate
    r=read(path);output=Path(output)
    if (r['source_revision']!=revision() or r['implementation']!=strategy.binding()
            or bind_pilot(r['pilot']['suite'])!=r['pilot'] or completed(r['source_suite'])!=r['completed']
            or torch.__version__!='2.7.1+cu126' or np.__version__!='2.2.6'
            or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('Final-worker source/code/pilot/completed/runtime binding differs')
    admission=strategy.Admission(r,path,output,mode)
    limit=r['checkpoint']['step']+(1 if mode=='interrupt' else 2)
    admission.validate(r['target_config'],output,'cuda:0',True,limit,False,False,None)
    if output.exists(): raise ValueError('Fresh owned worker output required')
    output.mkdir(parents=True)
    hasher=SimpleNamespace(t=torch,np=np)
    state_hash=lambda x:NativeUpdate.state_hash(hasher,x)
    try:
        config=r['target_config'];old=r['source_config']
        train=Split(r['data'],'retrieval','train',r['cache'],old['data_hashes']['train'])
        val=Split(r['data'],'retrieval','val',r['cache'],old['data_hashes']['val'])
        _,dictionary=vocabulary(r['data'],'retrieval')
        def contract(c): return fingerprint(dict(config=c,train=train.identity,val=val.identity,dictionary=dictionary))
        if contract(old)!=r['checkpoint']['contract']: raise ValueError('Original data/dictionary contract differs')
        target_contract=contract(config)
        if mode!='migration':
            prior_mode='interrupt' if mode=='resume' else 'migration'
            prior=Path(output).parent/prior_mode
            receipt=accept(read(prior/'result.json'),r,digest(path),prior_mode)
            pointer,generation=checked_pointer(prior)
            if pointer['sha256']!=receipt['checkpoint_sha256']: raise ValueError('Interrupted receipt differs')
            state=worker.load_checkpoint(prior,target_contract)
            strategy.check_checkpoint(state,config,admission.expected)
            loaded=state_hash(scientific(state))
            if loaded!=receipt['scientific_state_sha256']: raise ValueError('Prepared full values differ')
        else:
            source=Path(r['source_suite'])/'jobs'/JOB
            pointer,generation=checked_pointer(source)
            if any(pointer[k]!=r['checkpoint'][k] for k in ('sha256','file','contract','step')):
                raise ValueError('Original latest checkpoint changed')
            state=worker.load_checkpoint(source,contract(old));loaded=state_hash(state)
            if (state.get('training_strategy') is not None or state.get('training_complete')
                    or state['permutation'] is None or state['cursor']+256>len(train)
                    or state['scheduler']!=state.get('sample_clock')
                    or state['scheduler']['progress']!=state['epoch']*len(train)+state['cursor']):
                raise ValueError('Original active full sample-clock checkpoint/256 pending rows required')
            original=copy.deepcopy(state)
            state['contract']=target_contract
            state['training_strategy']=admission.expected
            # Parameter membership is supplied explicitly below from the actual
            # architecture; buffers must not be mistaken for leaf gradients.
            from .models import Classifier
            model=Classifier(config,r['cache'])
            state['strategy_gradients']={n:None for n,_ in model.named_parameters()};del model
            state['strategy_updates']=[]
            previous=state.get('torch_acceleration')
            state.pop('torch_acceleration',None)
            state['migration']=dict(source_checkpoint_sha256=r['checkpoint']['sha256'],
                source_contract=original['contract'],target_contract=target_contract,
                previous_migration=original.get('migration'),previous_torch_acceleration=previous,
                transition=admission.expected['transition'],optimizer_updates_performed=0)
            # Verify every inherited scientific value before publishing the copy.
            for key in original:
                if key not in ('contract','migration','torch_acceleration'):
                    if state_hash(original[key])!=state_hash(state[key]): raise ValueError('Migration changed '+key)
        worker.save_checkpoint(output,state)
        if state_hash(state)!=state_hash(worker.load_checkpoint(output,target_contract)):
            raise ValueError('Prepared full-state copy roundtrip differs')
        worker.PAUSE=False
        with lock(output/'worker.lock'):
            code=worker.run(config,r['data'],r['cache'],output,'cuda:0',resume=True,stop_after=limit,
                            strategy_admission=admission)
        if code!=(0 if mode=='migration' else 75): raise ValueError('Final worker failed finite stop')
        final=worker.load_checkpoint(output,target_contract);strategy.check_checkpoint(final,config,admission.expected)
        if final['epoch']!=state['epoch'] or final['cursor']!=state['cursor']+(0 if mode=='migration' else (1 if mode in ('interrupt','resume') else 2)*128):
            raise ValueError('Worker sampler progression differs')
        if final['sample_clock']['progress']!=state['sample_clock']['progress']+final['cursor']-state['cursor']:
            raise ValueError('Worker schedule progression differs')
        components={k:state_hash(v) for k,v in scientific(final).items()}
        pointer,_=checked_pointer(output)
        result=dict(schema=SCHEMA,mode=mode,request_sha256=digest(path),implementation=r['implementation'],
            source_checkpoint_sha256=r['checkpoint']['sha256'],checkpoint_sha256=pointer['sha256'],
            passed=True,step=final['step'],cursor=final['cursor'],loaded_state_sha256=loaded,
            scientific_state_sha256=state_hash(scientific(final)),component_sha256=components,
            updates=final['strategy_updates'],candidate_admitted=False,formal_training_started=False,
            worker_exit_status=code,excluded_execution_telemetry=list(EXCLUDE))
        if mode=='migration':
            validation=read(output/'migration-validation.json')
            if (validation.get('passed') is not True or validation.get('optimizer_steps_performed')!=0
                    or final['step']!=state['step']): raise ValueError('Final-worker zero-update migration differs')
            result['migration_validation']=validation
        if mode=='resume': result['trace']=read(output/'strategy-trace.json')
        atomic_json(output/'result.json',result)
    finally:
        unchanged(r)
        if completed(r['source_suite'])!=r['completed']: raise ValueError('Completed results changed')


def execute(args,folder,report):
    from .attention_precision_diagnostic import bind_source,unchanged,JOB
    from .attention_diagnostic import gpu_preflight
    from .runner import discover_gpus
    from .diagnostic_watch import Supervisor
    gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1: raise ValueError('One GPU required')
    gpu=gpus[0];report.update(gpu_identity=inventory[gpu],gpu_preflight=gpu_preflight(inventory[gpu]))
    if not report['gpu_preflight']['idle']: raise ValueError('GPU busy; preserve existing work')
    with ExitStack() as held:
        source=args.source_suite.resolve()
        for p in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/JOB/'worker.lock'):
            held.enter_context(lock(p))
        bound=bind_source(source,args.cache)
        r=dict(schema='qkan-key-bias-worker-input.v1',**bound,source_revision=revision(),
            implementation=strategy.binding(),target_config=strategy.target_config(bound['source_config']),
            pilot=bind_pilot(args.pilot_from),completed=completed(source,folder/'preserved-completed'))
        path=folder/'worker-input.json';atomic_json(path,r)
        supervisor=Supervisor(args.child_timeout,args.stall_timeout)
        report.update(request_sha256=digest(path),source_checkpoint=bound['checkpoint'],
            completed=r['completed'],pilot=r['pilot'],child_executions=supervisor.records,results={})
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',QKAN_DIAGNOSTIC_STAGE_FENCES='0')
        try:
            for mode in MODES:
                output=folder/mode;print('Final-worker child: '+mode,flush=True)
                code=supervisor.run(dict(target=TARGET,cwd=str(ROOT),argv=[sys.executable,'-B','-m',TARGET,
                    '--worker','--request',str(path),'--output',str(output),'--mode',mode]),folder/(mode+'.log'),env)
                if (output/'worker-error.json').is_file():
                    report.setdefault('partial_diagnostics',{})[mode]=read(output/'worker-error.json')
                if code!=0 or not (output/'result.json').is_file(): raise RuntimeError('Final-worker admission child failed')
                report['results'][mode]=accept(read(output/'result.json'),r,digest(path),mode)
            report['recovery']=agreement(report['results'])
            if completed(source,folder/'preserved-completed')!=r['completed']: raise ValueError('Preserved copy changed')
            report.update(passed=True,status='final_worker_preflight_passed_review_pending')
        finally:
            unchanged(bound)
            if completed(source)!=r['completed']: raise ValueError('Original completed results changed')
            report['original_checkpoint_and_metadata_unchanged']=True
            # Delete only verified disposable generations; retain compact receipts,
            # the immutable completed copies and private trace for review.
            removed=[]
            for mode in MODES:
                checkpoint=folder/mode/'checkpoints'
                if not checkpoint.exists(): continue
                for p in checkpoint.glob('step-*.pt'):
                    removed.append(dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=digest(p)))
                    p.unlink()
            report['disposable_checkpoints']=dict(removed=removed,retained_generations=0,
                continuation_source='original source only; no diagnostic checkpoint admitted')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('source-suite','cache','pilot-from','request','output'): parser.add_argument('--'+n,type=Path)
    parser.add_argument('--gpu',default='0');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--child-timeout',type=float,default=900)
    parser.add_argument('--stall-timeout',type=float,default=300);args=parser.parse_args()
    if args.worker:
        if any(getattr(args,n) is None for n in ('request','output','mode')): parser.error('Request/output/mode required')
        try: native(args.request,args.output,args.mode)
        except BaseException as error: worker_failure(args.output,error);raise
        return 0
    if any(getattr(args,n) is None for n in ('source_suite','cache','pilot_from')):
        parser.error('Original source/cache/completed pilot required')
    from .bootstrap import check_exchange
    from .return_artifacts import upload_declared
    from .diagnostic_watch import Supervisor
    Supervisor(args.child_timeout,args.stall_timeout);check_exchange()
    folder=ROOT/'runs'/('key-bias-worker-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    report=dict(schema=SCHEMA,source_revision=revision(),implementation=strategy.binding(),passed=False,
        status='failed',candidate_admitted=False,formal_training_started=False,training_samples_committed=0)
    error=None;print('Final-worker admission Suite: '+str(folder),flush=True)
    try: execute(args,folder,report)
    except BaseException as caught:
        error=caught;report.update(passed=False,status='failed')
        worker_failure(folder,caught);report['failure_locations']=read(folder/'worker-error.json')
    atomic_json(folder/'key-bias-worker-admission.json',report)
    try: upload_declared(folder,'key-bias-worker-admission',['key-bias-worker-admission.json'])
    except Exception:
        print('Return pending; report/retry manifest retained: '+str(folder),flush=True)
        if error is None: raise
    if error is not None: raise error
    print('Final-worker preflight complete; formal queue unchanged. See key-bias-worker-admission.json.',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
