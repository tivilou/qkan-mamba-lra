"""Bounded MEGA acceleration comparison; collaborator CUDA only, zero formal updates."""
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
METHODS=('fft','bf16','tf32','graph','micro-down','micro-up','resident')
REFINEMENTS=('fft+island-tail','fft+island-ffn','fft+island-projections','mask-fp32','graph-fixed','fft+graph-fixed','bf16+graph-fixed','fft+bf16+graph-fixed','fft+island-ffn+graph-fixed')
GRAPH_REPAIR=('mask-fp32+kernel-grid','graph-fixed+kernel-grid','fft+graph-fixed+kernel-grid','fft+island-tail+graph-fixed+kernel-grid','ffn-graph','fft+island-tail+ffn-graph')
GRAPH_DIAGNOSE=('graph-probe','ffn-graph','graph-fixed+kernel-grid')
STACKS=('fft+bf16','fft+graph','bf16+graph','fft+bf16+graph')
JOBS=tuple('mega-'+task+'-s42' for task in ('listops','text','image','pathfinder','retrieval'))


def phase_name(route,stage):
    settings(route)
    if not stage or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in stage):raise ValueError('Invalid MEGA stage')
    return 'mega_'+route.replace('+','_').replace('-','_')+'_'+stage


def settings(route):
    parts=route.split('+')
    if not parts or len(parts)!=len(set(parts)) or any(p not in METHODS+('island-tail','island-ffn','island-projections','mask-fp32','graph-fixed','kernel-grid','ffn-graph','graph-probe') for p in parts):raise ValueError('Unknown MEGA candidate')
    if 'micro-down' in parts and 'micro-up' in parts:raise ValueError('Conflicting micro tiers')
    if sum(p.startswith('island-') for p in parts)>1 or ('bf16' in parts and any(p.startswith('island-') for p in parts)):raise ValueError('Conflicting precision tiers')
    return set(parts)


def combined(results):
    winners=[name for name in METHODS if results.get(name,{}).get('timing',{}).get('timing_promising')]
    if 'micro-up' in winners and 'micro-down' in winners:
        remove=min(('micro-up','micro-down'),key=lambda n:results[n]['timing']['throughput_gain']);winners.remove(remove)
    if 'bf16' in winners and 'tf32' in winners:winners.remove('tf32')
    return '+'.join(winners) if len(winners)>1 else None


def select_job(source,requested='auto'):
    """Caller holds Suite locks. Never restart a completed or unstarted job."""
    states=read(source/'status.json')
    if any(v['state']=='running' for v in states.values()):raise ValueError('Pause original runner and wait for saved exit first')
    if requested!='auto' and requested not in JOBS:raise ValueError('Unknown MEGA job')
    candidates=JOBS if requested=='auto' else (requested,)
    # Non-Retrieval first; prefer the paused current task within that group.
    candidates=sorted(candidates,key=lambda job:('-retrieval-' in job,states.get(job,{}).get('state')!='paused',JOBS.index(job)))
    for job in candidates:
        folder=source/'jobs'/job
        if (states.get(job,{}).get('state') in ('pending','paused')
                and not (folder/'summary.json').exists() and (folder/'checkpoints/latest.json').is_file()):
            return job
    if requested!='auto':raise ValueError('Selected MEGA job has no unfinished checkpoint; use --job auto')
    return None


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


def child_spec(suite,route=None):
    if route:settings(route)
    target='experiments.baseline_delivery.mega_diagnostic'
    spec=dict(target=target,cwd=str(ROOT),argv=[str(ROOT/'.venv-mega/bin/python'),'-B','-m',target,
        '--request',str(suite/'request.json'),'--output',str(suite/(route or 'child'))])
    if route:spec['argv'].extend(['--route',route])
    return spec


def native(request_path,output,route='fft'):
    flags=settings(route)
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
    if route=='graph-probe':
        from .mega_refinement import graph_probe,error_chain
        try:
            mark('mega_graph_probe',flush=True)
            probe=graph_probe(torch)
            atomic_json(output/'result.json',dict(route=route,request_sha256=digest(request_path),
                kind='graph-probe',formal_updates_committed=0,disposable_updates=0,**probe))
            return
        except BaseException as error:
            atomic_json(output/'failure-detail.json',dict(phase='linear_graph_probe',chain=error_chain(error)))
            raise
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
    candidate_micro=micro
    if 'micro-up' in flags:candidate_micro=min(logical,micro*2)
    if 'micro-down' in flags:candidate_micro=max(1,micro//2)
    if logical%candidate_micro:raise ValueError('Candidate micro must divide fixed logical batch')
    if flags.intersection(('micro-up','micro-down')) and candidate_micro==micro:raise ValueError('Micro tier already at boundary')
    frozen=optimizer_steps(state['optimizer']);plugin=PowerTwoEMA(model);budget=0;step_proofs=[]
    from .mega_refinement import PrecisionIslands,GraphPadding,KernelGrid,FFNGraph,make_graph
    from contextlib import nullcontext
    tier=next((f[len('island-'):] for f in flags if f.startswith('island-')),None)
    islands=PrecisionIslands(model,tier,torch) if tier else None
    padding=GraphPadding(model.encoder) if flags.intersection(('graph-fixed','mask-fp32')) else None
    resident=None;graphed=None;graph_calls=0;stage='prepare'
    graph_requested=bool(flags.intersection(('graph','graph-fixed')))
    def phase(name):
        nonlocal stage
        stage=name;mark(phase_name(route,name),flush=True)
    grid=KernelGrid(model,torch) if 'kernel-grid' in flags else None
    ffn_graph=FFNGraph(model,torch,phase) if 'ffn-graph' in flags else None
    def load_batch():
        arrays=train.numpy_batch(ids,validate=True)
        return {k:torch.as_tensor(v).pin_memory().to('cuda:0',non_blocking=True) for k,v in arrays.items()}
    if 'resident' in flags:
        resident=load_batch();torch.cuda.synchronize()
    def restore():
        restore_diagnostic_state(model,opt,schedule,state,host,restore_rng)
        check_optimizer_steps(frozen,state['optimizer']);check_optimizer_steps(frozen,opt.state_dict())
    def exact():
        for key,value in dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),sample_clock=schedule.state_dict()).items():
            compare_states(state[key],host(value))
    def update(candidate):
        nonlocal budget,graphed,graph_calls
        opt.zero_grad(set_to_none=True);total=0.
        batch=resident if candidate and resident is not None else load_batch()
        size=candidate_micro if candidate else micro
        tf32=candidate and 'tf32' in flags
        torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=tf32
        with (grid.activate(candidate) if grid else nullcontext()), (ffn_graph.activate(candidate) if ffn_graph else nullcontext()), plugin.activate(candidate and 'fft' in flags), (islands.activate(candidate) if islands else nullcontext()), (padding.activate(candidate) if padding else nullcontext()), torch.autocast('cuda',enabled=candidate and 'bf16' in flags,dtype=torch.bfloat16,cache_enabled=False):
            for start in range(0,logical,size):
                part={k:v[start:start+size] for k,v in batch.items()}
                if candidate and graph_requested:
                    keys=tuple(k for k in part if k!='labels');values=tuple(part[k] for k in keys)
                    if graphed is None:
                        class Adapter(torch.nn.Module):
                            def __init__(self):super().__init__();self.base=model
                            def forward(self,*args):return self.base(dict(zip(keys,args)),validated_inputs=True)
                        phase('graph_capture');before=rng_state();torch.cuda.synchronize()
                        graphed=make_graph(torch,Adapter(),values)
                        torch.cuda.synchronize();restore_rng(before);opt.zero_grad(set_to_none=True)
                    phase('graph_replay');logits=graphed(*values);graph_calls+=1
                else:logits=model(part,validated_inputs=True)
                loss=torch.nn.functional.cross_entropy(logits.float(),part['labels'],reduction='sum')/logical
                loss.backward();total=total+loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
        opt.step();schedule.step(logical);torch.cuda.synchronize();budget+=1
        value=float(total)
        if not math.isfinite(value):raise ValueError('Nonfinite diagnostic loss')
        return value
    def capture():return host(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
        scheduler=schedule.state_dict(),gradients={n:p.grad for n,p in model.named_parameters()}))
    def measure(candidate,label):
        restore();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();phase(label)
        start=time.perf_counter();loss=update(candidate);elapsed=time.perf_counter()-start
        step_proofs.append(check_optimizer_steps(frozen,opt.state_dict(),1))
        atomic_json(output/'last-measurement.json',dict(route=route,phase=label,disposable_updates=budget,seconds=elapsed))
        return dict(seconds=elapsed,loss=loss,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
    try:
        restore();exact()
        # Three observations also warm both FFT plans; copies excluded from timing.
        setup_observations=[measure(False,'reference_a')];left=capture()
        setup_observations.append(measure(False,'reference_b'));repeat=capture()
        setup_observations.append(measure(True,'candidate'));right=capture()
        numerics=dict(repeat=numeric_summary(torch,state['model'],left,repeat),
            candidate=numeric_summary(torch,state['model'],left,right))
        atomic_json(output/'numeric-observation.json',numerics)
        times={False:[],True:[]};observations=[]
        for i,candidate in enumerate(ORDER):
            row=measure(candidate,'timing_'+str(i));times[candidate].append(row['seconds']);observations.append(dict(row,candidate=candidate))
        restore();exact()
        if 'fft' in flags and (plugin.calls<=0 or not any(a.split('->')[0]!=a.split('->')[1] for a in plugin.lengths)):
            raise ValueError('Candidate did not execute a changed FFT length')
        if islands and islands.calls<=0:raise ValueError('Precision island did not execute')
        if graph_requested and graph_calls<=0:raise ValueError('Graph did not replay')
        if padding and padding.calls<=0:raise ValueError('Padding fix did not execute')
        report=dict(schema='qkan-mega-fft-child.v1',route=route,measurement_completed=True,request_sha256=digest(request_path),
            checkpoint_sha256=request['checkpoint']['sha256'],source_step=state['step'],micro_batch=micro,logical_batch=logical,
            precision='bf16-autocast-fp32-master' if 'bf16' in flags else ('selective-bf16-fp32-master' if islands else 'fp32'),backend='ffn-graph' if ffn_graph else ('graph' if graph_requested else 'eager'),
            tf32='tf32' in flags,candidate_micro_batch=candidate_micro,gpu_resident_batch='resident' in flags,graph_calls=graph_calls,fft_source_sha256=SOURCE_SHA,fft_calls=plugin.calls,fft_lengths=plugin.lengths,
            timing=summarize(times[False],times[True]),setup_observations=setup_observations,observations=observations,numerics=numerics,
            frozen_adam_steps=frozen,measured_adam_steps=step_proofs,rollback_exact=True,disposable_updates=budget,
            formal_updates_committed=0,formal_release_allowed=False,
            timing_scope='forward/backward/clip/Adam/sample schedule/synchronize; CPU gather/H2D included except resident candidate; restore and numerical copies excluded',
            resident_limit='One repeated batch cached on GPU; upper-bound transfer-elimination screen, not multi-batch prefetch throughput',
            limits='One pending logical batch; no convergence, recovery admission or sustained speed claim')
        if grid and grid.calls<=0:raise ValueError('Kernel grid did not execute')
        if ffn_graph and ffn_graph.calls<=0:raise ValueError('FFN graph did not replay')
        report.update(kernel_grid_builds=grid.builds if grid else 0,ffn_graph_calls=ffn_graph.calls if ffn_graph else 0,ffn_graph_captures=ffn_graph.captures if ffn_graph else 0)
        report.update(precision_islands=islands.names if islands else [],island_calls=islands.calls if islands else 0,padding_fix_calls=padding.calls if padding else 0)
        atomic_json(output/'result.json',report)
    except BaseException as error:
        from .mega_refinement import error_category,native_trace,error_chain
        atomic_json(output/'failure-detail.json',dict(category=error_category(error),phase=stage,error_type=type(error).__name__,framework_frames=native_trace(error),chain=error_chain(error)))
        raise
    finally:
        if ffn_graph:ffn_graph.close()
        if grid:grid.close()
        if islands:islands.close()
        if padding:padding.close()
        plugin.close();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def campaign(args):
    from .bootstrap import check_exchange
    from .runner import discover_gpus
    from .attention_diagnostic import gpu_preflight
    from .diagnostic_watch import Supervisor
    from .return_artifacts import upload_declared
    check_exchange();revision();gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1 or not gpu_preflight(inventory[gpus[0]])['idle']:raise ValueError('Pause original runner; one idle GPU required')
    source=args.source_suite.resolve();suite=ROOT/'runs'/('mega-fft-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    supervisor=Supervisor(300,180);report=dict(schema='qkan-mega-fft.v1',measurement_completed=False,formal_release_allowed=False)
    with ExitStack() as held:
        for path in (source/'progress-supervisor.lock',source/'suite.lock'):held.enter_context(lock(path))
        job=select_job(source,args.job)
        if job is None:
            print('No unfinished MEGA checkpoint; diagnostic skipped. Continue original training queue.',flush=True)
            return 0
        held.enter_context(lock(source/'jobs'/job/'worker.lock'))
        print('MEGA diagnostic selected job: '+job,flush=True)
        request=snapshot(source,suite,job,args.cache)
        print('MEGA FFT diagnostic Suite: '+str(suite),flush=True)
        results={};failures={};attempted=[];unsafe=False
        refine=getattr(args,'refine',False)
        graph_repair=getattr(args,'graph_repair',False)
        graph_diagnose=getattr(args,'graph_diagnose',False)
        routes=list(GRAPH_DIAGNOSE) if graph_diagnose else (list(GRAPH_REPAIR) if graph_repair else (list(REFINEMENTS) if refine else (list(METHODS+STACKS) if getattr(args,'all_methods',False) else ['fft'])))
        report.update(results=results,failures=failures,attempted=attempted,all_methods=getattr(args,'all_methods',False),refine=refine,graph_repair=graph_repair,graph_diagnose=graph_diagnose)
        try:
            index=0
            while index<len(routes):
                route=routes[index];index+=1;attempted.append(route)
                print('MEGA candidate '+str(index)+': '+route,flush=True)
                try:
                    code=supervisor.run(child_spec(suite,route),suite/(route+'.log'),dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[0],
                        CUBLAS_WORKSPACE_CONFIG=':4096:8',QKAN_MEGA_SOURCE=str(ROOT/'.bootstrap/strong-upstream/mega')))
                    if len(supervisor.records)!=index or not supervisor.records[-1]['termination']['group_verified']:
                        unsafe=True;raise ValueError('Child group cleanup unverified')
                    if code!=0:raise RuntimeError('Candidate failed')
                    result=read(suite/route/'result.json')
                    if result.get('route')!=route or result.get('request_sha256')!=digest(suite/'request.json') or result.get('formal_updates_committed')!=0:
                        raise ValueError('Child identity differs')
                    if route=='graph-probe':
                        if result.get('probe_passed') is not True or result.get('disposable_updates')!=0:raise ValueError('Probe proof differs')
                    elif result.get('rollback_exact') is not True or result.get('disposable_updates')!=11:raise ValueError('Child proof differs')
                    results[route]=result
                except Exception as error:
                    worker_failure(suite/route,error)
                    failures[route]=dict(error=read(suite/route/'worker-error.json'))
                    native_error=suite/route/'native-error.json'
                    if native_error.exists():failures[route]['native']=read(native_error)
                    for name in ('last-measurement.json','numeric-observation.json','failure-detail.json'):
                        if (suite/route/name).exists():failures[route][name]=read(suite/route/name)
                    if len(supervisor.records)!=index or not supervisor.records[-1]['termination']['group_verified']:unsafe=True
                    if unsafe:break
                    if route=='graph-probe':
                        report['skipped']={r:'Linear capture prerequisite failed' for r in routes[index:]}
                        routes=routes[:index]
                if not graph_diagnose and not refine and not graph_repair and index==len(METHODS+STACKS) and getattr(args,'all_methods',False):
                    combo=combined(results)
                    if combo and combo not in routes:routes.append(combo)
            report.update(measurement_completed=any('timing' in r for r in results.values()),campaign_completed=not unsafe and index==len(routes),
                result=results.get('fft'),formal_updates_committed=0)
        except BaseException as error:
            worker_failure(suite,error);report['error']=read(suite/'worker-error.json')
        finally:
            report.update(request=request,child_executions=supervisor.records,gpu_identity=inventory[gpus[0]],
                source_unchanged=all(digest(p)==sha for p,sha in request['source_files'].items()))
            for path in (suite/'source/checkpoints').glob('*.pt'):path.unlink()
            report['copied_checkpoint_removed']=not any((suite/'source/checkpoints').glob('*.pt'))
            atomic_json(suite/'mega-fft.json',report);upload_declared(suite,'mega-fft',['mega-fft.json'])
    success=(report.get('campaign_completed',False) and report['source_unchanged'] and report['copied_checkpoint_removed']
        and len(supervisor.records)==len(attempted) and all(r['termination']['group_verified'] for r in supervisor.records)
        and (graph_diagnose or graph_repair or refine or getattr(args,'all_methods',False) or bool(results)))
    for name,result in results.items():
        if 'timing' not in result:
            print(name+': capture/replay probe passed; no speed measurement.',flush=True);continue
        print(name+': '+str(round(100*result['timing']['throughput_gain'],2))+'% throughput; formal policy unchanged.',flush=True)
    print('MEGA campaign returned: '+str(sum('timing' in r for r in results.values()))+' measured, '+str(len(failures))+' failed; original strategy retained.',flush=True)

    return 0 if success else 76


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--source-suite',type=Path);g.add_argument('--request',type=Path)
    p.add_argument('--cache',type=Path);p.add_argument('--gpu',default='0');p.add_argument('--output',type=Path)
    p.add_argument('--graph-diagnose',action='store_true',help='Linear capture prerequisite, then partial and whole MEGA graphs; one return')
    p.add_argument('--graph-repair',action='store_true',help='Kernel-grid and partial FFN graph repair candidates only')
    p.add_argument('--refine',action='store_true',help='Selective BF16 precision islands and repaired Graph candidates only')
    p.add_argument('--all-methods',action='store_true',help='All isolated candidates, then compatible promising combination; one return')
    p.add_argument('--route',default='fft',help=argparse.SUPPRESS)
    p.add_argument('--job',choices=['auto',*JOBS],default='auto',help='Default: unfinished MEGA with checkpoint, non-Retrieval first')
    a=p.parse_args()
    if sum((a.refine,a.all_methods,a.graph_repair,a.graph_diagnose))>1:p.error('Select one diagnostic campaign mode')
    if a.request:
        settings(a.route)
        if a.output!=a.request.parent/a.route and a.output!=a.request.parent/'child':p.error('Child output must match the owned diagnostic folder')
        try:native(a.request,a.output,a.route);return 0
        except BaseException as error:
            worker_failure(a.output,error)
            atomic_json(a.output/'native-error.json',dict(error=read(a.output/'worker-error.json'),
                cuda_oom=isinstance(error,RuntimeError) and 'out of memory' in str(error).lower(),
                graph_requested=bool(settings(a.route).intersection(('graph','graph-fixed','ffn-graph'))),route=a.route))
            raise
    if a.cache is None:p.error('cache required')
    return campaign(a)


if __name__=='__main__':raise SystemExit(main())
