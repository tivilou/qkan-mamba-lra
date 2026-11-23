"""Bounded BF16 numerical attribution and fresh-process recovery, no timing sweep."""
import argparse
from contextlib import ExitStack
import math
import os
from pathlib import Path
import sys
import time
import uuid

from .common import ROOT,atomic_json,digest,fingerprint,lock,read,revision,worker_failure
from . import mamba_diagnostic as speed

STEPS=8
MODES=('reference','bf16_eager','bf16_eager_interrupt','bf16_eager_resume',
       'bf16_graph','bf16_graph_interrupt','bf16_graph_resume')
FP32_MODES=('fp32_repeat_a','fp32_repeat_b')
NOISE_MODES=('fp32_noise_a','fp32_noise_b','bf16_eager_noise_a','bf16_eager_noise_b',
    'bf16_graph_noise_a','bf16_graph_noise_b')
PROJECTION_MODES=('fp32_noise_a','fp32_noise_b','projection_noise_a','projection_noise_b')
CONV_MODES=('fp32_noise_a','fp32_noise_b','conv_noise_a','conv_noise_b')
REVIEW=ROOT/'docs/evidence/mamba-speed-native-review-20261123.json'
REPEAT_REVIEW=ROOT/'docs/evidence/mamba-validation-native-review-20261123.json'


def native_route(mode):
    if mode.startswith('conv_'):return 'conv_fused_fp32'
    if mode.startswith('projection_'):return 'projection_bf16_eager'
    return ('fp32_eager' if mode=='reference' or mode.startswith('fp32_') else
        'bf16_graph' if mode.startswith('bf16_graph') else 'bf16_eager')


def spec(suite,mode):
    if mode not in MODES+FP32_MODES+NOISE_MODES+PROJECTION_MODES+CONV_MODES:raise ValueError('Unknown finite validation mode')
    return dict(target='experiments.baseline_delivery.mamba_validation',cwd=str(ROOT),argv=[
        sys.executable,'-B','-m','experiments.baseline_delivery.mamba_validation',
        '--request',str(suite/'request.json'),'--mode',mode,'--output',str(suite/mode)])


def accept(value,request_sha,mode):
    expected=2 if mode in FP32_MODES+NOISE_MODES+PROJECTION_MODES+CONV_MODES else 4 if mode.endswith('_interrupt') else STEPS
    if (value.get('schema')!='qkan-mamba-validation-child.v1' or value.get('mode')!=mode
            or value.get('request_sha256')!=request_sha or value.get('completed_updates')!=expected
            or value.get('source_restored_exact') is not True or value.get('formal_updates_committed')!=0
            or value.get('formal_release_allowed') is not False or not value.get('finite')):
        raise ValueError('Incomplete or mismatched finite validation receipt')
    return value


def recovery_agreement(continuous,interrupt,resume):
    return dict(exact=continuous['state_sha256']==resume['state_sha256']
        and continuous['components']==resume['components']
        and continuous['updates']==interrupt['updates']+resume['updates']
        and resume['starting_state_sha256']==interrupt['state_sha256'],
        continuous_state_sha256=continuous['state_sha256'],resumed_state_sha256=resume['state_sha256'])


def repeat_agreement(left,right):
    steps=[]
    for a,b in zip(left['updates'],right['updates']):
        steps.append(dict(index=a['index'],position_equal=(a['index'],a.get('step'))==(b['index'],b.get('step')),
            rows_equal=a['row_ids_sha256']==b['row_ids_sha256'],
            loss_equal=a['loss']==b['loss'],left_loss=a['loss'],right_loss=b['loss'],
            differing_components=sorted(k for k,v in a['components'].items() if b['components'].get(k)!=v)))
    initial=left['starting_state_sha256']==right['starting_state_sha256']
    exact=(initial and len(steps)==len(left['updates'])==len(right['updates'])==2
        and left['state_sha256']==right['state_sha256'] and left['components']==right['components']
        and all(x['position_equal'] and x['rows_equal'] and x['loss_equal'] and not x['differing_components'] for x in steps))
    return dict(exact=exact,starting_state_exact=initial,steps=steps,
        first_different_update_1based=next((x['index']+1 for x in steps
            if not x['position_equal'] or not x['rows_equal'] or not x['loss_equal'] or x['differing_components']),None))


def norm_summary(rows):
    left=math.sqrt(sum(r['reference_norm']**2 for r in rows))
    right=math.sqrt(sum(r['candidate_norm']**2 for r in rows))
    difference=math.sqrt(sum(r['difference_norm']**2 for r in rows))
    dot=sum(r['dot'] for r in rows)
    return dict(tensor_count=len(rows),elements=sum(r['elements'] for r in rows),
        reference_norm=left,candidate_norm=right,difference_norm=difference,
        relative_l2=difference/left if left else None,
        zero_reference=left==0,cosine=dot/(left*right) if left and right else None,
        max_absolute=max((r['max_absolute'] for r in rows),default=0.),
        worst_parameters=[{k:v for k,v in r.items() if k!='dot'} for r in
            sorted(rows,key=lambda r:r['difference_norm'],reverse=True)[:8]])


def tensor_errors(torch,left,right):
    rows=[];controls=[]
    def visit(a,b,path):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.shape!=b.shape or a.dtype!=b.dtype:
                raise ValueError('Numerical comparison tensor structure differs')
            if not a.is_floating_point():controls.append(bool(torch.equal(a,b)));return
            x=a.double().flatten();y=b.double().flatten()
            if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
                raise ValueError('Nonfinite measurement tensor')
            n=float(x.norm());m=float(y.norm());d=float((x-y).norm());dot=float((x*y).sum())
            rows.append(dict(parameter=path,elements=x.numel(),reference_norm=n,candidate_norm=m,
                difference_norm=d,relative_l2=d/n if n else None,
                max_absolute=float((x-y).abs().max()) if x.numel() else 0.,
                cosine=dot/(n*m) if n and m else None,dot=dot))
        elif isinstance(a,dict):
            if not isinstance(b,dict) or a.keys()!=b.keys():raise ValueError('Numerical comparison keys differ')
            for k,v in a.items():visit(v,b[k],path+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            if type(a) is not type(b) or len(a)!=len(b):raise ValueError('Numerical comparison sequence differs')
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
        else:controls.append(type(a) is type(b) and a==b)
    visit(left,right,'')
    return dict(norm_summary(rows),nonfloating_controls_equal=all(controls))


def numerical_comparison(torch,left,right):
    if len(left['samples'])!=2 or len(right['samples'])!=2:raise ValueError('Two measurement updates required')
    steps=[]
    for a,b in zip(left['samples'],right['samples']):
        invariants=(a['index']==b['index'] and a['row_ids_sha256']==b['row_ids_sha256']
            and a['invariants_sha256']==b['invariants_sha256'])
        metrics={k:tensor_errors(torch,a[k],b[k]) for k in ('raw_gradients','delta','parameters','optimizer','clipped_gradients')}
        steps.append(dict(index=a['index'],invariants_exact=invariants,
            loss_reference=a['loss'],loss_candidate=b['loss'],loss_absolute_difference=abs(a['loss']-b['loss']),metrics=metrics))
    return dict(starting_scientific_state_exact=left['starting_scientific_sha256']==right['starting_scientific_sha256'],
        invariants_exact=all(s['invariants_exact'] and all(m['nonfloating_controls_equal'] for m in s['metrics'].values()) for s in steps),steps=steps)


def measurements_complete(children,modes=NOISE_MODES):
    if set(children)!=set(modes):return False
    for mode,child in children.items():
        expected=({'repeat'} if mode.endswith('_b') else set())|({'vs_fp32_a'} if native_route(mode)!='fp32_eager' else set())
        values=child.get('numerical_comparisons',{})
        if set(values)!=expected:return False
        if any(not v.get('starting_scientific_state_exact') or not v.get('invariants_exact')
                or len(v.get('steps',[]))!=2 for v in values.values()):return False
    return True


def cleanup(suite,request):
    # Only exact owned filenames, never source Suite files.
    names=['reference-values.pt','bf16_eager-midpoint.pt','bf16_graph-midpoint.pt']
    names.extend(route+'-noise-values.pt' for route in ('fp32_eager','bf16_eager','bf16_graph','projection_bf16_eager','conv_fused_fp32'))
    if request.get('checkpoint'):names.append('source/checkpoints/'+Path(request['checkpoint']['file']).name)
    for name in names:
        path=suite/name
        if path.is_symlink():raise ValueError('Owned diagnostic binary became a symlink')
        path.unlink(missing_ok=True)
    return dict(private_binary_files_removed=names,remaining_owned_binaries=any((suite/n).exists() for n in names))


def operation(c,request_path,mode,output):
    from types import SimpleNamespace
    from .attention_update_native import NativeUpdate
    from .worker import restore_rng
    from .runner import compare_states
    from .diagnostic_watch import mark
    torch=c['torch'];suite=request_path.parent;state=c['state'];model=c['model']
    noise=mode in NOISE_MODES+PROJECTION_MODES+CONV_MODES;repeat=mode in FP32_MODES or noise
    reference_mode=native_route(mode)=='fp32_eager'
    request_sha=digest(request_path);base=native_route(mode)
    total=2 if repeat else STEPS
    start=4 if mode.endswith('_resume') else 0;end=4 if mode.endswith('_interrupt') else total
    if state['cursor']+total*c['logical']>len(c['train']):
        raise ValueError('Complete prefix updates must remain in the original epoch; source preserved')
    hasher=SimpleNamespace(t=torch,np=c['np'])
    state_hash=lambda value:NativeUpdate.state_hash(hasher,value)
    def payload(offset):
        return dict(**c['output'](),gradients=c['host']({n:p.grad for n,p in model.named_parameters()}),
            sampler=dict(epoch=state['epoch'],cursor=state['cursor']+offset*c['logical'],
                step=state['step']+offset,permutation=state['permutation']),
            micro=c['micro'],logical=c['logical'],route='fp32_reference' if reference_mode else base)
    def restore_payload(saved):
        model.load_state_dict(saved['model']);c['opt'].load_state_dict(saved['optimizer'])
        c['schedule'].load_state_dict(saved['scheduler'])
        for n,p in model.named_parameters():
            value=saved['gradients'][n];p.grad=None if value is None else value.to(p.device).clone()
        restore_rng(saved['rng'])
    def save_private(path,value):
        torch.save(value,path)
        if path.stat().st_size>128*1024**2:raise ValueError('Diagnostic binary exceeds 128 MiB')
        reopened=torch.load(path,map_location='cpu',weights_only=False)
        compare_states(value,reopened)
        atomic_json(path.with_suffix('.receipt.json'),dict(sha256=digest(path),
            state_sha256=state_hash(value),request_sha256=request_sha,mode=mode))
    def load_private(path,creator):
        record=read(path.with_suffix('.receipt.json'))
        if (path.is_symlink() or record['sha256']!=digest(path)
                or record['request_sha256']!=request_sha or record['mode']!=creator):
            raise ValueError('Disposable state identity changed')
        value=torch.load(path,map_location='cpu',weights_only=False)
        if record['state_sha256']!=state_hash(value):raise ValueError('Disposable state hash differs')
        return value
    midpoint=suite/(base+'-midpoint.pt');reference_path=suite/'reference-values.pt'
    c['restore']();c['exact']()
    if start:
        saved=load_private(midpoint,base+'_interrupt')
        if (saved['sampler']['cursor']!=state['cursor']+4*c['logical']
                or saved['sampler']['step']!=state['step']+4 or saved['route']!=base):
            raise ValueError('Fresh-process recovery cursor/route differs')
        restore_payload(saved)
        compare_states(saved,payload(start))
    initial=payload(start);starting_hash=state_hash(initial);updates=[];first=None;samples=[]
    starting_scientific_hash=state_hash({k:v for k,v in initial.items() if k!='route'})
    reference=load_private(reference_path,'reference') if mode in ('bf16_eager','bf16_graph') else None
    for index in range(start,end):
        mark('mamba_validate_'+mode+'_'+str(index),flush=True)
        ids=state['permutation'][state['cursor']+index*c['logical']:state['cursor']+(index+1)*c['logical']].numpy()
        c['ids'][:]=ids
        before=c['host'](dict(model.named_parameters())) if noise else None
        captured={}
        def observe():
            captured.update(c['host']({n:p.grad for n,p in model.named_parameters()}))
        loss=float(c['update'](not reference_mode,observe if index==0 or noise else None))
        if not math.isfinite(loss):raise ValueError('Nonfinite prefix loss')
        updates.append(dict(index=index,step=state['step']+index+1,
            row_ids_sha256=fingerprint([int(i) for i in ids]),loss=loss))
        if repeat:
            updates[-1]['components']={k:state_hash(v) for k,v in payload(index+1).items()}
        if noise:
            current=payload(index+1)
            samples.append(dict(index=index,row_ids_sha256=updates[-1]['row_ids_sha256'],loss=loss,
                invariants_sha256=state_hash(dict({k:current[k] for k in ('rng','scheduler','sampler','micro','logical')},
                    optimizer_steps={k:s['step'] for k,s in current['optimizer']['state'].items() if 'step' in s})),
                raw_gradients=captured,clipped_gradients=current['gradients'],parameters=current['model'],
                optimizer=current['optimizer'],delta={n:p.detach().cpu()-before[n] for n,p in model.named_parameters()}))
        if index==0:
            first=dict(gradients=captured,delta={n:p.detach().cpu()-state['model'][n]
                for n,p in model.named_parameters()})
    final=payload(end)
    def require_finite(value):
        if torch.is_tensor(value):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite diagnostic state')
        elif isinstance(value,dict):
            for child in value.values():require_finite(child)
        elif isinstance(value,(list,tuple)):
            for child in value:require_finite(child)
    require_finite(final)
    if noise:require_finite(samples)
    components={k:state_hash(v) for k,v in final.items()}
    attribution={}
    if reference is not None:
        for kind in ('gradients','delta'):
            rows=[]
            for name,left in reference['first'][kind].items():
                right=first[kind][name]
                if left is None or right is None:
                    rows.append(dict(parameter=name,missing_gradient_equal=left is right));continue
                a=left.double().flatten();b=right.double().flatten();norm=float(a.norm());other=float(b.norm())
                error=float((a-b).norm());dot=float((a*b).sum())
                rows.append(dict(parameter=name,elements=a.numel(),reference_norm=norm,
                    difference_norm=error,relative_l2=error/max(norm,1e-30),
                    cosine=dot/(norm*other) if norm and other else None,
                    max_absolute=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all())))
            attribution[kind]=rows
    if mode=='reference':save_private(reference_path,dict(first=first,updates=updates))
    if end==4:save_private(midpoint,final)
    comparisons={}
    if noise:
        measured=dict(starting_scientific_sha256=starting_scientific_hash,samples=samples)
        path=suite/(native_route(mode)+'-noise-values.pt')
        if mode.endswith('_a'):save_private(path,measured)
        else:comparisons['repeat']=numerical_comparison(torch,load_private(path,mode[:-1]+'a'),measured)
        if not reference_mode:
            comparisons['vs_fp32_a']=numerical_comparison(torch,
                load_private(suite/'fp32_eager-noise-values.pt','fp32_noise_a'),measured)
    timing=None;mechanism=None
    if mode.startswith(('projection_','conv_')):
        policy=c['projection']
        if policy.calls<=0:raise ValueError('Candidate policy did not execute')
        counter='observed_fused_calls' if mode.startswith('conv_') else 'observed_projection_calls'
        mechanism=dict(policy.mechanism,**{counter:policy.calls})
        if mode.endswith('_b'):
            # Reuse the restored first batch; exclude numerical copies/hashes from timing.
            c['ids'][:]=state['permutation'][state['cursor']:state['cursor']+c['logical']].numpy()
            for candidate in (False,True):c['restore']();c['update'](candidate)
            times={False:[],True:[]}
            for index,candidate in enumerate((False,True,True,False)*2):
                c['restore']();torch.cuda.synchronize();mark('mamba_'+('conv' if mode.startswith('conv_') else 'projection')+'_timing_'+str(index),flush=True)
                began=time.perf_counter();loss=float(c['update'](candidate))
                times[candidate].append(time.perf_counter()-began)
                if not math.isfinite(loss):raise ValueError('Nonfinite projection timing loss')
            timing=dict(reference_seconds=times[False],candidate_seconds=times[True],
                summary=speed.summarize(times[False],times[True]),disposable_updates=10,
                includes='CPU gather, H2D, forward/backward, clip, Adam and sample schedule',
                excludes='restore, hashes, CPU numerical copies and setup')
    # Restore the immutable source into this disposable process before claiming zero commits.
    c['restore']();c['exact']()
    atomic_json(output/'result.json',dict(schema='qkan-mamba-validation-child.v1',mode=mode,
        request_sha256=request_sha,completed_updates=end,updates=updates,finite=True,
        micro_batch=c['micro'],logical_batch=c['logical'],starting_state_sha256=starting_hash,
        state_sha256=state_hash(final),components=components,attribution=attribution,
        numerical_comparisons=comparisons,
        projection_mechanism=mechanism if mode.startswith('projection_') else None,
        convolution_mechanism=mechanism if mode.startswith('conv_') else None,paired_timing=timing,
        source_restored_exact=True,formal_updates_committed=0,formal_release_allowed=False,
        claim=('Two-update numerical measurements only; no error threshold or formal admission.' if noise else
            'Two-update original FP32 repeat control; no formal admission.' if repeat else
            'Fixed eight-update numerical/recovery evidence; not convergence or formal admission.')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-suite',type=Path);p.add_argument('--cache',type=Path)
    p.add_argument('--job',choices=speed.JOBS,default=speed.JOBS[0]);p.add_argument('--gpu',default='0')
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--fp32-repeat',action='store_true',help='Only two fresh original-FP32 two-update prefixes; four disposable updates')
    selection.add_argument('--measure-noise',action='store_true',help='FP32/BF16 eager/BF16 Graph, two fresh two-update prefixes each; measure errors only')
    selection.add_argument('--projection-bf16',action='store_true',help='Input projection BF16, fused inner FP32: four two-update prefixes plus ten timing updates; no Graph')
    selection.add_argument('--fused-fp32',action='store_true',help='Offline process-local causal convolution, original FP32 fused path; 18 disposable updates, no pip or formal restart')
    p.add_argument('--request',type=Path);p.add_argument('--mode',choices=MODES+FP32_MODES+NOISE_MODES+PROJECTION_MODES+CONV_MODES);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.request:
        if not a.mode or a.output!=a.request.parent/a.mode:p.error('Bound child mode/output required')
        route=native_route(a.mode)
        try:speed.native(a.request,route,validation=lambda c:operation(c,a.request,a.mode,a.output))
        except Exception as error:worker_failure(a.output,error);raise
        return 0
    if not a.source_suite or not a.cache:p.error('source-suite and cache required')
    from .bootstrap import check_exchange
    from .runner import discover_gpus
    from .attention_diagnostic import gpu_preflight
    from .diagnostic_watch import Supervisor
    check_exchange();gpus,inventory=discover_gpus(a.gpu)
    if len(gpus)!=1 or not gpu_preflight(inventory[gpus[0]])['idle']:raise ValueError('Original runner must be paused; choose one idle GPU')
    if not read(REVIEW)['routes']['bf16_eager']['timing']['timing_promising']:raise ValueError('Reviewed positive BF16 timing required')
    experiment='mamba-convolution' if a.fused_fp32 else 'mamba-projection' if a.projection_bf16 else 'mamba-noise' if a.measure_noise else 'mamba-repeat' if a.fp32_repeat else 'mamba-validation'
    modes=CONV_MODES if a.fused_fp32 else PROJECTION_MODES if a.projection_bf16 else NOISE_MODES if a.measure_noise else FP32_MODES if a.fp32_repeat else MODES
    suite=ROOT/'runs'/(experiment+'-'+uuid.uuid4().hex[:12]);supervisor=Supervisor(900,300)
    report=dict(schema='qkan-mamba-validation.v1',children={},recovery={},formal_release_allowed=False,
        reuse_speed_review_sha256=digest(REVIEW),prefix_steps=2 if a.fp32_repeat or a.measure_noise or a.projection_bf16 or a.fused_fp32 else STEPS,training_updates_committed=0)
    request=speed.snapshot(a.source_suite,suite,a.job,a.cache)
    try:
        if a.fp32_repeat or a.measure_noise or a.projection_bf16 or a.fused_fp32:
            reviewed=read(REPEAT_REVIEW)
            if request['checkpoint']['sha256']!=reviewed['evidence']['source_checkpoint_sha256']:
                raise ValueError('Latest checkpoint differs from reviewed BF16 input; source preserved, no rollback')
            report['reuse_validation_review_sha256']=digest(REPEAT_REVIEW)
        if a.fused_fp32:
            from .mamba_convolution import stage
            request['convolution_package']=stage()
            atomic_json(suite/'request.json',request)
        with ExitStack() as held:
            source=a.source_suite.resolve()
            for path in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/a.job/'worker.lock'):
                held.enter_context(lock(path))
            if any(digest(path)!=sha for path,sha in request['source_files'].items()):raise ValueError('Source advanced; use latest checkpoint')
            print('Mamba '+('offline fused FP32' if a.fused_fp32 else 'input projection BF16' if a.projection_bf16 else 'numerical measurements' if a.measure_noise else 'FP32 repeat control' if a.fp32_repeat else 'numerical/recovery')+' Suite: '+str(suite),flush=True)
            for mode in modes:
                print('Mamba finite validation: '+mode,flush=True)
                code=supervisor.run(spec(suite,mode),suite/(mode+'.log'),dict(os.environ,
                    CUDA_VISIBLE_DEVICES=gpus[0],CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                path=suite/mode/'result.json'
                if code!=0 or not path.is_file():
                    error=suite/mode/'worker-error.json'
                    report['children'][mode]=dict(exit_status=code,error=read(error) if error.is_file() else None)
                    print('Validation child FAILED: '+mode+'; '+str(suite/(mode+'.log')),flush=True);break
                report['children'][mode]=accept(read(path),digest(suite/'request.json'),mode)
            if a.fp32_repeat and all(k in report['children'] and 'state_sha256' in report['children'][k] for k in FP32_MODES):
                report['repeatability']=repeat_agreement(*(report['children'][k] for k in FP32_MODES))
            for route in ('bf16_eager','bf16_graph'):
                if all(k in report['children'] and 'state_sha256' in report['children'][k]
                        for k in (route,route+'_interrupt',route+'_resume')):
                    report['recovery'][route]=recovery_agreement(*(report['children'][k]
                        for k in (route,route+'_interrupt',route+'_resume')))
            report['source_unchanged']=all(digest(path)==sha for path,sha in request['source_files'].items())
            if a.measure_noise or a.projection_bf16 or a.fused_fp32:
                report['measurement_completed']=measurements_complete(report['children'],modes) and report['source_unchanged']
                report['numeric_acceptance']='not_evaluated; two repeats do not establish a noise bound'
            if a.projection_bf16 or a.fused_fp32:
                candidates=[report['children'].get(k,{}) for k in modes[2:]]
                field='convolution_mechanism' if a.fused_fp32 else 'projection_mechanism'
                counter='observed_fused_calls' if a.fused_fp32 else 'observed_projection_calls'
                report['measurement_completed'] &= all(
                    (c.get(field) or {}).get(counter,0)>0 for c in candidates)
                report['measurement_completed'] &= bool(candidates[-1].get('paired_timing'))
                report['disposable_update_budget']=18
    finally:
        report.update(request=request,child_executions=supervisor.records,gpu_identity=inventory[gpus[0]],
            cleanup=cleanup(suite,request))
        atomic_json(suite/(experiment+'.json'),report)
        from .return_artifacts import upload_declared
        upload_declared(suite,experiment,[experiment+'.json'])
    if a.measure_noise or a.projection_bf16 or a.fused_fp32:
        complete=report.get('measurement_completed') is True
        print(('Numerical measurements completed; inspect magnitudes, no numerical admission.' if complete else
            'Numerical measurements incomplete or invariant mismatch; inspect returned report.')+
            ' Original training remains paused; see '+experiment+'.json.',flush=True)
        return 0 if complete else 76
    if a.fp32_repeat:
        comparison=report.get('repeatability',{})
        complete=len(report['children'])==2 and comparison.get('exact') is True and report['source_unchanged']
        print(('FP32 two-by-two repeat is exact.' if complete else
            'FP32 repeat differs/incomplete; first differing update: '+str(comparison.get('first_different_update_1based')))+
            ' Original training remains paused; inspect returned mamba-repeat.json.',flush=True)
        return 0 if complete else 76
    complete=len(report['children'])==len(MODES) and len(report['recovery'])==2 and all(
        v['exact'] for v in report['recovery'].values()) and report['source_unchanged']
    print(('Finite prefix/recovery completed; numerical review pending.' if complete else
        'Finite validation FAILED/incomplete; inspect returned diagnostics.')+' Original training remains paused.',flush=True)
    return 0 if complete else 76


if __name__=='__main__':raise SystemExit(main())
