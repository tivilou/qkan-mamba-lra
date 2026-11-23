"""Finite Mamba update comparisons; no formal continuation or source mutation."""
import argparse
from contextlib import ExitStack, contextmanager
import copy
import datetime as dt
import math
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

from .common import ROOT, atomic_json, digest, fingerprint, lock, read, revision

JOBS = tuple('mamba1-'+task+'-s42' for task in ('retrieval','image','pathfinder'))
ROUTES = ('fp32_eager','bf16_eager','fp32_graph','bf16_graph')


class InputProjectionBF16:
    """Disposable input-projection policy, preserving the pinned native path.

    Direct weight matmul bypasses Linear.forward in upstream Mamba. Keep the
    original forward except that one expression and all parameter objects.
    """
    source_sha256='a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3'

    def __init__(self,torch,model):
        import importlib
        import inspect
        import textwrap
        module=importlib.import_module('mamba_ssm.modules.mamba_simple')
        if digest(Path(module.__file__))!=self.source_sha256:
            raise ValueError('Projection candidate requires the reviewed pinned Mamba source')
        mixers=[b.mixer for b in model.blocks]
        if (not mixers or any(type(m) is not module.Mamba
                or any(p.dtype!=torch.float32 for p in m.parameters()) for m in mixers)):
            raise ValueError('Pinned Mamba instances with FP32 parameters required')
        self.active=False;self.calls=0;self.installed=[]
        def project(weight,values):
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                projected=weight @ values
            if projected.dtype!=torch.bfloat16:raise ValueError('BF16 projection did not execute')
            self.calls+=1
            return projected.float()
        # Reuse the entire pinned forward, including its optional-convolution
        # branch. Linear.forward hooks miss this direct weight access.
        source=textwrap.dedent(inspect.getsource(module.Mamba.forward))
        expression='self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)")'
        if source.count(expression)!=1:raise ValueError('Pinned projection expression boundary changed')
        namespace=dict(vars(module),_qkan_input_projection_bf16=project)
        exec(compile(source.replace(expression,
            '_qkan_input_projection_bf16(self.in_proj.weight, rearrange(hidden_states, "b l d -> d (b l)"))'),
            module.__file__,'exec'),namespace)
        projected_forward=namespace['forward']
        self.mechanism=dict(policy='mamba-in-proj-bf16-inner-fp32-v1',
            upstream_source_sha256=self.source_sha256,blocks=len(mixers),graph=False,
            bf16_operations=['in_proj.weight @ hidden_states'],
            fp32_operations=['projection bias','conv1d','x_proj','dt_proj','selective_scan','out_proj'],
            parameter_dtype='float32',optimizer='original_fp32_adamw',
            causal_conv1d_available=module.causal_conv1d_fn is not None,
            inner_paths=['fused' if m.use_fast_path and module.causal_conv1d_fn is not None
                else 'original_unfused' for m in mixers],
            source_expression_replacements=1)
        for mixer in mixers:
            original=mixer.forward
            def forward(hidden_states,inference_params=None,_m=mixer,_original=original):
                if not self.active:return _original(hidden_states,inference_params=inference_params)
                if inference_params is not None or hidden_states.dtype!=torch.float32:
                    raise ValueError('Projection candidate supports FP32 training inputs only')
                with torch.autocast('cuda',enabled=False):
                    out=projected_forward(_m,hidden_states,inference_params=inference_params)
                if out.dtype!=torch.float32:raise ValueError('Original inner output must remain FP32')
                return out
            self.installed.append((mixer,'forward' in mixer.__dict__,original))
            mixer.forward=forward

    @contextmanager
    def activate(self):
        previous=self.active;self.active=True
        try:yield
        finally:self.active=previous

    def close(self):
        self.active=False
        for mixer,owned,original in self.installed:
            if owned:mixer.forward=original
            else:del mixer.forward
        self.installed=[]


def summarize(reference, candidate):
    """Paired complete-update wall times; capture/setup are separate costs."""
    if len(reference)!=4 or len(candidate)!=4 or any(
            not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0
            for v in (*reference,*candidate)):
        raise ValueError('Four positive finite matched measurements required')
    gain=statistics.median(reference)/statistics.median(candidate)-1
    return dict(reference_median_seconds=statistics.median(reference),
        candidate_median_seconds=statistics.median(candidate),throughput_gain=gain,
        positive_pairs=sum(b<a for a,b in zip(reference,candidate)),
        timing_promising=gain>=.03 and sum(b<a for a,b in zip(reference,candidate))>=3)


def child_spec(suite, route):
    if route not in ROUTES: raise ValueError('Unknown Mamba comparison route')
    return dict(target='experiments.baseline_delivery.mamba_diagnostic',cwd=str(ROOT),argv=[
        sys.executable,'-B','-m','experiments.baseline_delivery.mamba_diagnostic',
        '--request',str(suite/'request.json'),'--route',route,'--output',str(suite/route)])


def snapshot(source, suite, job, cache):
    """Copy only the selected paused job, not the other 24 jobs/weights."""
    from .migration import checked_pointer, copy_verified
    source=source.resolve(); folder=source/'jobs'/job
    if not source.is_dir() or source.is_symlink(): raise ValueError('Existing source Suite required')
    with ExitStack() as held:
        for path in (source/'progress-supervisor.lock',source/'suite.lock',folder/'worker.lock'):
            held.enter_context(lock(path))
        states=read(source/'status.json')
        if any(row['state']=='running' for row in states.values()):
            raise ValueError('Pause the one-command runner and wait for its exit first')
        if states[job]['state'] not in ('pending','paused') or (folder/'summary.json').exists():
            raise ValueError('Select an unfinished Mamba job')
        config=read(folder/'config.json')
        if (config['model']!='mamba1' or job!='mamba1-'+config['task']+'-s42'
                or config.get('training_strategy') or not config.get('adaptive_memory')):
            raise ValueError('Original adaptive Mamba configuration required')
        pointer,generation=checked_pointer(folder)
        files={'config.json':digest(folder/'config.json'),
            'checkpoints/latest.json':digest(folder/'checkpoints/latest.json'),
            'checkpoints/'+generation.name:pointer['sha256']}
        copy_verified(folder,suite/'source',files)
        manifest=read(source/'suite.json')
        data=Path(manifest['data']).resolve()
        if not data.is_dir() or not cache.is_dir(): raise ValueError('Existing data/cache required')
        request=dict(schema='qkan-mamba-diagnostic.v1',source_suite=str(source),job=job,
            source_files={str(folder/name):sha for name,sha in files.items()},
            config_sha256=fingerprint(config),checkpoint=pointer,data=str(data),cache=str(cache.resolve()),
            source_revision=revision(),formal_release_allowed=False)
        atomic_json(suite/'request.json',request)
        return request


def native(request_path, route, validation=None):
    # Native imports occur exclusively in a supervised collaborator child.
    import numpy as np
    import torch
    from .worker import load_checkpoint, rng_state, restore_rng
    from .models import Classifier, optimizer
    from .data import Split, vocabulary
    from .logical_batch import make_controller, planned_batch_size, SampleClock, TorchSampleSchedule
    from .runner import compare_states
    from .torch_acceleration import RecoveryMismatch
    from .diagnostic_watch import mark
    suite=request_path.parent; request=read(request_path); config=read(suite/'source/config.json')
    if request['source_revision']!=revision() or fingerprint(config)!=request['config_sha256']:
        raise ValueError('Diagnostic source changed')
    torch.set_num_threads(int(os.environ.get('QKAN_CPU_THREADS','1')))
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    mark('mamba_restore',flush=True)
    mark('mamba_data_identity',flush=True)
    train=Split(request['data'],config['task'],'train',request['cache'],config['data_hashes']['train'])
    val=Split(request['data'],config['task'],'val',request['cache'],config['data_hashes']['val'])
    vocab,dict_sha=vocabulary(request['data'],config['task'])
    if vocab!=config['vocabulary']: raise ValueError('Vocabulary changed')
    contract=fingerprint(dict(config=config,train=train.identity,val=val.identity,dictionary=dict_sha))
    mark('mamba_checkpoint_load',flush=True)
    state=load_checkpoint(suite/'source',contract)
    if state['runtime']!={'torch':torch.__version__,'numpy':np.__version__,'device_type':'cuda'}:
        raise ValueError('Use the original Mamba Torch/NumPy runtime')
    if state['training_complete'] or state['cursor']>=len(train):
        raise ValueError('Use a latest checkpoint with a pending training update')
    mark('mamba_model_prepare',flush=True)
    model=Classifier(config,request['cache']).cuda().train(); opt=optimizer(model,config)
    controller=make_controller(config['adaptive_memory'],config['batch_size'],state['adaptive_memory'])
    micro=controller.micro_batch; logical=planned_batch_size(controller,config,state['step'])
    ids=state['permutation'][state['cursor']:min(len(train),state['cursor']+logical)].numpy().copy()
    if len(ids)!=logical or logical%micro: raise ValueError('Complete logical update required for matched Graph shapes')
    clock_state=state['sample_clock']
    clock=SampleClock(clock_state['total'],clock_state['warmup'],clock_state['base_rates'],
        saved=clock_state,reference_batch=clock_state['reference_batch'])
    schedule=TorchSampleSchedule(opt,clock)
    def host(value):
        if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
        if isinstance(value,dict): return {k:host(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)): return type(value)(host(v) for v in value)
        return copy.deepcopy(value)
    def restore():
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        schedule.load_state_dict(clock_state);opt.zero_grad(set_to_none=True);restore_rng(state['rng'])
    def exact():
        for name,value in dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state()).items():
            compare_states(state[name],host(value))
        compare_states(clock_state,schedule.state_dict())
    mark('mamba_state_roundtrip',flush=True)
    restore();exact()
    inputs=tuple(sorted(k for k in train.numpy_batch(ids) if k!='labels'))
    bf16=route.startswith('bf16'); graph=route.endswith('graph'); replays=0
    projection=InputProjectionBF16(torch,model) if route=='projection_bf16_eager' else None
    class Adapter(torch.nn.Module):
        def __init__(self): super().__init__();self.base=model
        def forward(self,*values):
            with ExitStack() as stack:
                stack.enter_context(torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16,cache_enabled=False))
                if projection is not None:stack.enter_context(projection.activate())
                return self.base(dict(zip(inputs,values)),validated_inputs=True)
    forward=Adapter();setup=time.perf_counter()
    if graph:
        mark('mamba_graph_capture',flush=True)
        batch=train.batch(ids[:micro],'cuda:0')
        forward=torch.cuda.make_graphed_callables(forward,tuple(batch[k] for k in inputs),num_warmup_iters=3)
        torch.cuda.synchronize();restore();exact()
    setup=time.perf_counter()-setup
    def update(use_candidate, observe=None):
        nonlocal replays
        arrays=train.numpy_batch(ids,validate=True)
        staged={k:torch.as_tensor(v).pin_memory().to('cuda:0',non_blocking=True) for k,v in arrays.items()}
        opt.zero_grad(set_to_none=True);loss_sum=0.
        for start in range(0,len(ids),micro):
            batch={k:v[start:start+micro] for k,v in staged.items()}
            if use_candidate:
                logits=forward(*(batch[k] for k in inputs));replays+=int(graph)
            else: logits=model(batch,validated_inputs=True)
            loss=torch.nn.functional.cross_entropy(logits.float(),batch['labels'],reduction='sum')/len(ids)
            loss.backward();loss_sum+=loss.detach()
            # Graph backward returns static storage; detach accumulated gradients.
            if graph and use_candidate:
                for parameter in model.parameters():
                    if parameter.grad is not None:parameter.grad=parameter.grad.clone()
        if observe is not None: observe()
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
        opt.step();schedule.step(len(ids))
        torch.cuda.synchronize()
        return loss_sum
    def output(): return host(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),scheduler=schedule.state_dict()))
    if validation is not None:
        try:
            return validation(dict(torch=torch,np=np,model=model,opt=opt,schedule=schedule,state=state,
                micro=micro,logical=logical,ids=ids,train=train,host=host,restore=restore,exact=exact,
                update=update,output=output,setup_seconds=setup,projection=projection))
        finally:
            if projection is not None:projection.close()
    # Uninstrumented warmups, restored before every measurement.
    mark('mamba_warmup',flush=True)
    for candidate in (False,True):restore();update(candidate)
    times={False:[],True:[]};results={};losses={}
    for index,candidate in enumerate((False,True,True,False)*2):
        restore();torch.cuda.synchronize();mark('mamba_timing_'+str(index),flush=True)
        started=time.perf_counter(); loss=update(candidate)
        times[candidate].append(time.perf_counter()-started)
        if candidate not in results:
            results[candidate]=output();losses[candidate]=float(loss)
    left=results[False];right=results[True]
    def relative(a,b):
        pairs=[(a[k].double(),b[k].double()) for k in a if a[k].is_floating_point()]
        numerator=sum(float(((x-y)**2).sum()) for x,y in pairs)
        denominator=sum(float((x**2).sum()) for x,y in pairs)
        return math.sqrt(numerator/max(denominator,1e-30))
    model_relative_l2=relative(left['model'],right['model'])
    delta_left={k:v-state['model'][k] for k,v in left['model'].items()}
    delta_right={k:v-state['model'][k] for k,v in right['model'].items()}
    rng_exact=True
    try:compare_states(left['rng'],right['rng'])
    except RecoveryMismatch:rng_exact=False
    full_state_exact=True
    try:compare_states(left,right)
    except RecoveryMismatch:full_state_exact=False
    restore();exact()
    report=dict(schema='qkan-mamba-route.v1',route=route,request_sha256=digest(request_path),
        micro_batch=micro,logical_batch=logical,samples=len(ids),step=state['step'],
        model_relative_l2=model_relative_l2,update_delta_relative_l2=relative(delta_left,delta_right),
        losses={str(k):v for k,v in losses.items()},rng_exact=rng_exact,full_state_exact=full_state_exact,
        setup_seconds=setup,graph_replays=replays,reference_seconds=times[False],candidate_seconds=times[True],
        timing=summarize(times[False],times[True]),rollback_exact=True,
        formal_updates_committed=0,formal_release_allowed=False,
        mechanism=dict(mamba_fast_path_flags=[getattr(b.mixer,'use_fast_path',None) for b in model.blocks],
            precision='autocast_bf16_fp32_parameters' if bf16 else 'fp32',graph=graph),
        claim='Timing candidate only; BF16 numerics and fresh-process recovery still require review.')
    atomic_json(suite/(route+'.json'),report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-suite',type=Path);p.add_argument('--cache',type=Path)
    p.add_argument('--job',choices=JOBS,default=JOBS[0]);p.add_argument('--gpu',default='0')
    p.add_argument('--request',type=Path);p.add_argument('--route',choices=ROUTES)
    p.add_argument('--output',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.request:
        if not args.route or args.output!=args.request.parent/args.route:p.error('Bound child route/output required')
        try: native(args.request,args.route)
        except Exception as error:
            from .common import worker_failure
            worker_failure(args.output,error)
            raise
        return 0
    if not args.source_suite or not args.cache:p.error('source-suite and cache required')
    from .bootstrap import check_exchange
    from .runner import discover_gpus
    from .attention_diagnostic import gpu_preflight
    from .diagnostic_watch import Supervisor
    check_exchange();gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1 or not gpu_preflight(inventory[gpus[0]])['idle']:
        raise ValueError('Pause and wait for the original runner; selected GPU must be idle')
    suite=ROOT/'runs'/('mamba-speed-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    supervisor=Supervisor(900,300);report=dict(schema='qkan-mamba-speed.v1',routes={},formal_release_allowed=False)
    with ExitStack() as held:
        source=args.source_suite.resolve()
        # Hold the source locks for the entire diagnostic, including all children.
        # Snapshot acquires these itself first, then locks are reacquired and bytes checked.
        request=snapshot(source,suite,args.job,args.cache)
        for path in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/args.job/'worker.lock'):
            held.enter_context(lock(path))
        if any(digest(path)!=sha for path,sha in request['source_files'].items()):
            raise ValueError('Source advanced while acquiring diagnostic locks; take a fresh snapshot')
        print('Mamba diagnostic Suite: '+str(suite),flush=True)
        try:
            for route in ROUTES:
                print('Mamba short comparison: '+route,flush=True)
                spec=child_spec(suite,route)
                code=supervisor.run(spec,suite/(route+'.log'),{**os.environ,'CUDA_VISIBLE_DEVICES':gpus[0],
                    'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
                path=suite/(route+'.json')
                value=read(path) if code==0 and path.exists() else dict(status='failed',exit_status=code)
                failure=suite/route/'worker-error.json'
                if code!=0 and failure.is_file():value['error']=read(failure)
                report['routes'][route]=value
                if code!=0:
                    print('Mamba route FAILED: '+route+' | exit '+str(code)+' | '+
                        value.get('error',{}).get('error_type','inspect local log')+' | '+str(suite/(route+'.log')),flush=True)
                # Candidate errors get one isolated process; no repeat or fallback claim.
                if route==ROUTES[0] and code!=0:break
        finally:
            report.update(source_unchanged=all(digest(path)==sha for path,sha in request['source_files'].items()),
                request=request,child_executions=supervisor.records,gpu_identity=inventory[gpus[0]])
            atomic_json(suite/'mamba-speed.json',report)
            from .return_artifacts import upload_declared
            upload_declared(suite,'mamba-speed',['mamba-speed.json'])
    complete=all('timing' in v for v in report['routes'].values()) and len(report['routes'])==4
    print(('Four comparisons completed and returned.' if complete else
           'Comparisons FAILED/incomplete; partial diagnostics returned, no speed result established.')+
          ' Original training remains paused. No formal policy changed.',flush=True)
    return 0 if complete else 76


if __name__=='__main__':raise SystemExit(main())
