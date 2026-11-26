"""Bounded MEGA FP32 FFT comparison; collaborator CUDA only, zero formal updates."""
import argparse
from contextlib import ExitStack
import copy
import datetime as dt
import math
import os
from pathlib import Path
import sys
import time
import uuid
from .common import ROOT,atomic_json,digest,fingerprint,lock,read,revision,worker_failure

ORDER=(False,True,True,False)*2


def snapshot(source,suite,job,cache):
    from .migration import checked_pointer,copy_verified
    source=source.resolve();folder=source/'jobs'/job
    states=read(source/'status.json')
    if any(v['state']=='running' for v in states.values()):raise ValueError('Pause original runner and wait for saved exit first')
    if states[job]['state'] not in ('pending','paused') or (folder/'summary.json').exists():raise ValueError('Unfinished MEGA checkpoint required')
    config=read(folder/'config.json')
    if (config['model']!='mega' or job!='mega-'+config['task']+'-s42'
            or not config.get('adaptive_memory') or config.get('training_strategy')):
        raise ValueError('Original adaptive MEGA configuration required')
    pointer,generation=checked_pointer(folder)
    files={'config.json':digest(folder/'config.json'),'checkpoints/latest.json':digest(folder/'checkpoints/latest.json'),
        'checkpoints/'+generation.name:pointer['sha256']}
    copy_verified(folder,suite/'source',files)
    data=Path(read(source/'suite.json')['data']).resolve()
    if not data.is_dir() or not cache.is_dir():raise ValueError('Original data/cache required')
    request=dict(schema='qkan-mega-fft-request.v1',source_revision=revision(),job=job,
        config_sha256=fingerprint(config),checkpoint=pointer,data=str(data),cache=str(cache.resolve()),
        source_files={str(folder/name):sha for name,sha in files.items()},formal_release_allowed=False)
    atomic_json(suite/'request.json',request);return request


def child_spec(suite):
    target='experiments.baseline_delivery.mega_diagnostic'
    return dict(target=target,cwd=str(ROOT),argv=[str(ROOT/'.venv-mega/bin/python'),'-B','-m',target,
        '--request',str(suite/'request.json'),'--output',str(suite/'child')])


def native(request_path,output):
    import numpy as np
    import torch
    from .models import Classifier,optimizer
    from .worker import load_checkpoint,rng_state,restore_rng
    from .data import Split,vocabulary
    from .logical_batch import make_controller,planned_batch_size,SampleClock,TorchSampleSchedule
    from .mamba_diagnostic import optimizer_steps,check_optimizer_steps,restore_diagnostic_state,summarize
    from .mamba_memory_benchmark import numeric_summary
    from .runner import compare_states
    from .diagnostic_watch import mark
    from .mega_fft import PowerTwoEMA,SOURCE_SHA
    request=read(request_path);suite=request_path.parent;config=read(suite/'source/config.json')
    if request['source_revision']!=revision() or fingerprint(config)!=request['config_sha256']:
        raise ValueError('Diagnostic revision/config changed')
    torch.set_num_threads(int(os.environ.get('QKAN_CPU_THREADS','1')))
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    mark('mega_data_identity',flush=True)
    train=Split(request['data'],config['task'],'train',request['cache'],config['data_hashes']['train'])
    val=Split(request['data'],config['task'],'val',request['cache'],config['data_hashes']['val'])
    vocab,dict_sha=vocabulary(request['data'],config['task'])
    if vocab!=config['vocabulary']:raise ValueError('Vocabulary changed')
    state=load_checkpoint(suite/'source',fingerprint(dict(config=config,train=train.identity,val=val.identity,dictionary=dict_sha)))
    if state['runtime']!={'torch':torch.__version__,'numpy':np.__version__,'device_type':'cuda'} or state['training_complete']:
        raise ValueError('Original MEGA runtime and pending training state required')
    controller=make_controller(config['adaptive_memory'],config['batch_size'],state['adaptive_memory'])
    micro=controller.micro_batch;logical=planned_batch_size(controller,config,state['step'])
    if state['cursor']+logical>len(train) or logical%micro:raise ValueError('Checkpoint must have a complete pending logical batch')
    ids=state['permutation'][state['cursor']:state['cursor']+logical].numpy().copy()
    model=Classifier(config,request['cache']).cuda().train();opt=optimizer(model,config)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('FP32 parameters required')
    clock_state=state['sample_clock']
    schedule=TorchSampleSchedule(opt,SampleClock(clock_state['total'],clock_state['warmup'],clock_state['base_rates'],
        saved=clock_state,reference_batch=clock_state['reference_batch']))
    def host(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:host(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return type(value)(host(v) for v in value)
        return copy.deepcopy(value)
    frozen=optimizer_steps(state['optimizer']);plugin=PowerTwoEMA(model);budget=0;step_proofs=[]
    def restore():
        restore_diagnostic_state(model,opt,schedule,state,host,restore_rng)
        check_optimizer_steps(frozen,state['optimizer']);check_optimizer_steps(frozen,opt.state_dict())
    def exact():
        for key,value in dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),sample_clock=schedule.state_dict()).items():
            compare_states(state[key],host(value))
    def update(candidate):
        nonlocal budget
        opt.zero_grad(set_to_none=True);total=0.
        arrays=train.numpy_batch(ids,validate=True)
        batch={k:torch.as_tensor(v).pin_memory().to('cuda:0',non_blocking=True) for k,v in arrays.items()}
        with plugin.activate(candidate),torch.autocast('cuda',enabled=False):
            for start in range(0,logical,micro):
                part={k:v[start:start+micro] for k,v in batch.items()}
                loss=torch.nn.functional.cross_entropy(model(part,validated_inputs=True).float(),part['labels'],reduction='sum')/logical
                loss.backward();total=total+loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
        opt.step();schedule.step(logical);torch.cuda.synchronize();budget+=1
        value=float(total)
        if not math.isfinite(value):raise ValueError('Nonfinite diagnostic loss')
        return value
    def capture():return host(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
        scheduler=schedule.state_dict(),gradients={n:p.grad for n,p in model.named_parameters()}))
    def measure(candidate,label):
        restore();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();mark('mega_'+label,flush=True)
        start=time.perf_counter();loss=update(candidate);elapsed=time.perf_counter()-start
        step_proofs.append(check_optimizer_steps(frozen,opt.state_dict(),1))
        return dict(seconds=elapsed,loss=loss,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
    try:
        restore();exact()
        # Three observations also warm both FFT plans; copies excluded from timing.
        measure(False,'reference_a');left=capture()
        measure(False,'reference_b');repeat=capture()
        measure(True,'candidate');right=capture()
        numerics=dict(repeat=numeric_summary(torch,state['model'],left,repeat),
            candidate=numeric_summary(torch,state['model'],left,right))
        times={False:[],True:[]};observations=[]
        for i,candidate in enumerate(ORDER):
            row=measure(candidate,'timing_'+str(i));times[candidate].append(row['seconds']);observations.append(dict(row,candidate=candidate))
        restore();exact()
        if plugin.calls<=0 or not any(a.split('->')[0]!=a.split('->')[1] for a in plugin.lengths):
            raise ValueError('Candidate did not execute a changed FFT length')
        report=dict(schema='qkan-mega-fft-child.v1',measurement_completed=True,request_sha256=digest(request_path),
            checkpoint_sha256=request['checkpoint']['sha256'],source_step=state['step'],micro_batch=micro,logical_batch=logical,
            precision='fp32',backend='eager',fft_source_sha256=SOURCE_SHA,fft_calls=plugin.calls,fft_lengths=plugin.lengths,
            timing=summarize(times[False],times[True]),observations=observations,numerics=numerics,
            frozen_adam_steps=frozen,measured_adam_steps=step_proofs,rollback_exact=True,disposable_updates=budget,
            formal_updates_committed=0,formal_release_allowed=False,
            timing_scope='CPU gather/H2D/forward/backward/clip/Adam/sample schedule/synchronize; restore and numerical copies excluded',
            limits='One pending logical batch; no convergence, recovery admission or sustained speed claim')
        atomic_json(output/'result.json',report)
    finally:plugin.close()


def campaign(args):
    from .bootstrap import check_exchange
    from .runner import discover_gpus
    from .attention_diagnostic import gpu_preflight
    from .diagnostic_watch import Supervisor
    from .return_artifacts import upload_declared
    check_exchange();revision();gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1 or not gpu_preflight(inventory[gpus[0]])['idle']:raise ValueError('Pause original runner; one idle GPU required')
    source=args.source_suite.resolve();suite=ROOT/'runs'/('mega-fft-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    supervisor=Supervisor(900,300);report=dict(schema='qkan-mega-fft.v1',measurement_completed=False,formal_release_allowed=False)
    with ExitStack() as held:
        for path in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/args.job/'worker.lock'):held.enter_context(lock(path))
        request=snapshot(source,suite,args.job,args.cache)
        print('MEGA FFT diagnostic Suite: '+str(suite),flush=True)
        try:
            code=supervisor.run(child_spec(suite),suite/'worker.log',dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[0],
                CUBLAS_WORKSPACE_CONFIG=':4096:8',QKAN_MEGA_SOURCE=str(ROOT/'.bootstrap/strong-upstream/mega')))
            if code==0:
                result=read(suite/'child/result.json')
                if (result.get('request_sha256')!=digest(suite/'request.json') or result.get('rollback_exact') is not True
                        or result.get('disposable_updates')!=11 or result.get('formal_updates_committed')!=0):raise ValueError('Child proof differs')
                report.update(measurement_completed=True,result=result)
            else:report.update(child_exit_status=code)
        except BaseException as error:
            worker_failure(suite,error);report['error']=read(suite/'worker-error.json')
        finally:
            report.update(request=request,child_executions=supervisor.records,gpu_identity=inventory[gpus[0]],
                source_unchanged=all(digest(p)==sha for p,sha in request['source_files'].items()))
            for path in (suite/'source/checkpoints').glob('*.pt'):path.unlink()
            report['copied_checkpoint_removed']=not any((suite/'source/checkpoints').glob('*.pt'))
            child_error=suite/'child/worker-error.json'
            if child_error.exists():report['child_error']=read(child_error)
            atomic_json(suite/'mega-fft.json',report);upload_declared(suite,'mega-fft',['mega-fft.json'])
    success=(report['measurement_completed'] and report['source_unchanged'] and report['copied_checkpoint_removed']
        and len(supervisor.records)==1 and supervisor.records[0]['termination']['group_verified'])
    if success:print('FFT throughput gain: '+str(round(100*report['result']['timing']['throughput_gain'],2))+'%; formal policy unchanged.',flush=True)
    else:print('MEGA FFT FAILED/incomplete; compact diagnostics returned; original state preserved.',flush=True)
    return 0 if success else 76


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--source-suite',type=Path);g.add_argument('--request',type=Path)
    p.add_argument('--cache',type=Path);p.add_argument('--gpu',default='0');p.add_argument('--output',type=Path)
    p.add_argument('--job',choices=['mega-'+task+'-s42' for task in ('listops','text','image','pathfinder','retrieval')],default='mega-listops-s42')
    a=p.parse_args()
    if a.request:
        if a.output!=a.request.parent/'child':p.error('Child output must match the owned diagnostic folder')
        try:native(a.request,a.output);return 0
        except BaseException as error:worker_failure(a.output,error);raise
    if a.cache is None:p.error('cache required')
    return campaign(a)


if __name__=='__main__':raise SystemExit(main())
