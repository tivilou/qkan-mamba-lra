"""Stable worker plus explicit FP32 execution plugin; no native CLI imports."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

from .common import atomic_json, digest, fingerprint, lock, read, worker_failure
from . import mamba_fused_formal as formal


def scientific(state):
    from .key_bias_worker_admission import scientific as select
    return {k:v for k,v in select(state).items() if k!='mamba_fused_execution'}


def worker_core():
    from . import worker
    return worker


def native(path,job,mode):
    r=formal.verify(path,ready=mode=='formal');suite=Path(path).parent
    if job not in r['candidates'] or mode not in (*formal.MODES,'formal'):raise ValueError('Bound Mamba route required')
    import torch
    import numpy as np
    from .mamba_convolution import prepare_native
    from .execution_plugins import Hooks,create,eager_state,reset_measurements
    prepare_native(torch,r)
    w=worker_core()
    from .attention_update_native import NativeUpdate
    from .migration import checked_pointer,copy_verified
    from .key_bias_recovery import prepare_worker_output
    from .key_bias_stability_native import compare_values
    hasher=SimpleNamespace(t=torch,np=np)
    hash_state=lambda state:NativeUpdate.state_hash(hasher,state)
    finite=mode!='formal';source=suite/'jobs'/job
    output=suite/'preflight'/mode if finite else source
    config=read(source/'config.json');base=r['candidates'][job]['checkpoint']
    if finite and (job!=formal.JOB or base is None):raise ValueError('Recovery uses the reviewed active job')
    before=None;loaded=None
    if finite:
        prepare_worker_output(output)
        prior=suite/'preflight/interrupt' if mode=='resume' else source
        pointer,generation=checked_pointer(prior)
        if mode=='resume' and pointer['sha256']!=read(prior/'result.json')['checkpoint_sha256']:
            raise ValueError('Interrupted checkpoint changed')
        copy_verified(prior,output,{'checkpoints/'+generation.name:pointer['sha256']})
        atomic_json(output/'checkpoints/latest.json',{**pointer,'previous':[]})
        before=w.load_checkpoint(output,pointer['contract'])
        if hash_state(before)!=hash_state(w.load_checkpoint(prior,pointer['contract'])):raise ValueError('Full state copy differs')
        loaded=hash_state(scientific(before))
    latest=checked_pointer(output)[0] if (output/'checkpoints/latest.json').exists() else None
    if latest:
        state=w.load_checkpoint(output,latest['contract']);marker=state.get('mamba_fused_execution')
        if marker!=formal.lineage(r) and (base is None or latest['sha256']!=base['sha256']):
            raise ValueError('Advanced checkpoint lacks matching fused lineage')
    policy=None
    def install(model):
        nonlocal policy
        policy=create(r['execution_plugin'],config,torch,model)
        return policy
    def before_save(state):
        state['mamba_fused_execution']=formal.lineage(r)
    def configure_controller(controller,state):
        if state.get('mamba_fused_execution')!=formal.lineage(r):reset_measurements(controller)
    def after_save(state):
        atomic_json(output/'mamba-fused-execution.json',dict(policy=policy.mechanism,
            observed_fused_calls=policy.calls,pid=os.getpid(),updated_at=time.time(),step=state['step'],
            implementation=r['binding'],evaluation='original_unfused',graph=False))
    def publish(value):
        if policy is not None:
            atomic_json(output/'mamba-fused-execution.json',dict(policy=policy.mechanism,
                observed_fused_calls=policy.calls,pid=os.getpid(),updated_at=time.time(),
                step=value.get('optimizer_step'),implementation=r['binding'],evaluation='original_unfused',graph=False))
    limit=base['step']+(1 if mode=='interrupt' else 2) if finite else None
    handlers={};w.PAUSE=False
    try:
        for name in ('SIGINT','SIGTERM','SIGHUP'):
            if hasattr(signal,name):
                sig=getattr(signal,name);handlers[sig]=signal.signal(sig,w.request_pause)
        with lock(output/'worker.lock'):
            hooks=Hooks(model_context=install,acceleration_state=eager_state,before_save=before_save,
                configure_controller=configure_controller,after_save=after_save,on_progress=publish)
            code=w.run(config,r['data'],r['cache'],output,'cuda:0',resume=latest is not None,stop_after=limit,execution=hooks)
        if policy is None or policy.calls<=0:raise ValueError('No actual fused training calls')
        if not finite:return code
        if code!=75:raise ValueError('Finite worker did not save a paused checkpoint')
        pointer,_=checked_pointer(output);final=w.load_checkpoint(output,pointer['contract'])
        count=1 if mode in ('interrupt','resume') else 2
        if (final['step']!=before['step']+count or final['epoch']!=before['epoch']
                or final['cursor']!=before['cursor']+32*count
                or final['sample_clock']['progress']!=before['sample_clock']['progress']+32*count):
            raise ValueError('Finite step/sample cursor differs')
        result=dict(passed=True,mode=mode,job=job,input_sha256=digest(path),step=final['step'],
            checkpoint_sha256=pointer['sha256'],loaded_scientific_sha256=loaded,
            scientific_sha256=hash_state(scientific(final)),calls=policy.calls,
            formal_updates_committed=0,copy_roundtrip_exact=True)
        if mode=='resume':
            prior=read(suite/'preflight/interrupt/result.json')
            if loaded!=prior['scientific_sha256']:raise ValueError('Resume scientific start differs')
            other=w.load_checkpoint(suite/'preflight/continuous',base['contract'])
            compare_values(hasher,scientific(other),scientific(final))
            result['recovery']=dict(passed=True,rtol=1e-5,atol=1e-6,
                rng_rows_clocks_controls='exact',all_persisted_scientific_fields=True)
        atomic_json(output/'result.json',result);return 0
    finally:
        if policy is not None:policy.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        formal.unchanged(r)


def checked_output(path,job,mode,output):
    expected=path.parent/('jobs/'+job if mode=='formal' else 'preflight/'+mode)
    if output.is_symlink() or output.resolve()!=expected.resolve():raise ValueError('Worker output differs from bound Suite/mode')
    return expected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=(*formal.MODES,'formal'),required=True);a=p.parse_args()
    try:
        checked_output(a.request,a.job,a.mode,a.output)
        return native(a.request,a.job,a.mode)
    except BaseException as error:
        worker_failure(a.request.parent/('preflight/'+a.mode if a.mode!='formal' else 'jobs/'+a.job),error);raise


if __name__=='__main__':raise SystemExit(main())
