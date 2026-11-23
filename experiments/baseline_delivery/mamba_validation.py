"""Bounded BF16 numerical attribution and fresh-process recovery, no timing sweep."""
import argparse
from contextlib import ExitStack
import math
import os
from pathlib import Path
import sys
import uuid

from .common import ROOT,atomic_json,digest,fingerprint,lock,read,revision,worker_failure
from . import mamba_diagnostic as speed

STEPS=8
MODES=('reference','bf16_eager','bf16_eager_interrupt','bf16_eager_resume',
       'bf16_graph','bf16_graph_interrupt','bf16_graph_resume')
REVIEW=ROOT/'docs/evidence/mamba-speed-native-review-20261123.json'


def spec(suite,mode):
    if mode not in MODES:raise ValueError('Unknown finite validation mode')
    return dict(target='experiments.baseline_delivery.mamba_validation',cwd=str(ROOT),argv=[
        sys.executable,'-B','-m','experiments.baseline_delivery.mamba_validation',
        '--request',str(suite/'request.json'),'--mode',mode,'--output',str(suite/mode)])


def accept(value,request_sha,mode):
    expected=4 if mode.endswith('_interrupt') else STEPS
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


def cleanup(suite,request):
    # Only exact owned filenames, never source Suite files.
    names=['reference-values.pt','bf16_eager-midpoint.pt','bf16_graph-midpoint.pt']
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
    request_sha=digest(request_path);base='bf16_graph' if mode.startswith('bf16_graph') else 'bf16_eager'
    start=4 if mode.endswith('_resume') else 0;end=4 if mode.endswith('_interrupt') else STEPS
    if state['cursor']+STEPS*c['logical']>len(c['train']):
        raise ValueError('Eight complete updates must remain in the original epoch; source preserved')
    hasher=SimpleNamespace(t=torch,np=c['np'])
    state_hash=lambda value:NativeUpdate.state_hash(hasher,value)
    def payload(offset):
        return dict(**c['output'](),gradients=c['host']({n:p.grad for n,p in model.named_parameters()}),
            sampler=dict(epoch=state['epoch'],cursor=state['cursor']+offset*c['logical'],
                step=state['step']+offset,permutation=state['permutation']),
            micro=c['micro'],logical=c['logical'],route=base if mode!='reference' else 'fp32_reference')
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
    starting_hash=state_hash(payload(start));updates=[];first=None
    reference=load_private(reference_path,'reference') if mode in ('bf16_eager','bf16_graph') else None
    for index in range(start,end):
        mark('mamba_validate_'+mode+'_'+str(index),flush=True)
        ids=state['permutation'][state['cursor']+index*c['logical']:state['cursor']+(index+1)*c['logical']].numpy()
        c['ids'][:]=ids
        captured={}
        def observe():
            captured.update(c['host']({n:p.grad for n,p in model.named_parameters()}))
        loss=float(c['update'](mode!='reference',observe if index==0 else None))
        if not math.isfinite(loss):raise ValueError('Nonfinite prefix loss')
        updates.append(dict(index=index,step=state['step']+index+1,
            row_ids_sha256=fingerprint([int(i) for i in ids]),loss=loss))
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
    # Restore the immutable source into this disposable process before claiming zero commits.
    c['restore']();c['exact']()
    atomic_json(output/'result.json',dict(schema='qkan-mamba-validation-child.v1',mode=mode,
        request_sha256=request_sha,completed_updates=end,updates=updates,finite=True,
        micro_batch=c['micro'],logical_batch=c['logical'],starting_state_sha256=starting_hash,
        state_sha256=state_hash(final),components=components,attribution=attribution,
        source_restored_exact=True,formal_updates_committed=0,formal_release_allowed=False,
        claim='Fixed eight-update numerical/recovery evidence; not convergence or formal admission.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-suite',type=Path);p.add_argument('--cache',type=Path)
    p.add_argument('--job',choices=speed.JOBS,default=speed.JOBS[0]);p.add_argument('--gpu',default='0')
    p.add_argument('--request',type=Path);p.add_argument('--mode',choices=MODES);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.request:
        if not a.mode or a.output!=a.request.parent/a.mode:p.error('Bound child mode/output required')
        route='fp32_eager' if a.mode=='reference' else 'bf16_graph' if a.mode.startswith('bf16_graph') else 'bf16_eager'
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
    suite=ROOT/'runs'/('mamba-validation-'+uuid.uuid4().hex[:12]);supervisor=Supervisor(900,300)
    report=dict(schema='qkan-mamba-validation.v1',children={},recovery={},formal_release_allowed=False,
        reuse_speed_review_sha256=digest(REVIEW),prefix_steps=STEPS,training_updates_committed=0)
    request=speed.snapshot(a.source_suite,suite,a.job,a.cache)
    try:
        with ExitStack() as held:
            source=a.source_suite.resolve()
            for path in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/a.job/'worker.lock'):
                held.enter_context(lock(path))
            if any(digest(path)!=sha for path,sha in request['source_files'].items()):raise ValueError('Source advanced; use latest checkpoint')
            print('Mamba numerical/recovery Suite: '+str(suite),flush=True)
            for mode in MODES:
                print('Mamba finite validation: '+mode,flush=True)
                code=supervisor.run(spec(suite,mode),suite/(mode+'.log'),dict(os.environ,
                    CUDA_VISIBLE_DEVICES=gpus[0],CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                path=suite/mode/'result.json'
                if code!=0 or not path.is_file():
                    error=suite/mode/'worker-error.json'
                    report['children'][mode]=dict(exit_status=code,error=read(error) if error.is_file() else None)
                    print('Validation child FAILED: '+mode+'; '+str(suite/(mode+'.log')),flush=True);break
                report['children'][mode]=accept(read(path),digest(suite/'request.json'),mode)
            for route in ('bf16_eager','bf16_graph'):
                if all(k in report['children'] and 'state_sha256' in report['children'][k]
                        for k in (route,route+'_interrupt',route+'_resume')):
                    report['recovery'][route]=recovery_agreement(*(report['children'][k]
                        for k in (route,route+'_interrupt',route+'_resume')))
            report['source_unchanged']=all(digest(path)==sha for path,sha in request['source_files'].items())
    finally:
        report.update(request=request,child_executions=supervisor.records,gpu_identity=inventory[gpus[0]],
            cleanup=cleanup(suite,request))
        atomic_json(suite/'mamba-validation.json',report)
        from .return_artifacts import upload_declared
        upload_declared(suite,'mamba-validation',['mamba-validation.json'])
    complete=len(report['children'])==len(MODES) and len(report['recovery'])==2 and all(
        v['exact'] for v in report['recovery'].values()) and report['source_unchanged']
    print(('Finite prefix/recovery completed; numerical review pending.' if complete else
        'Finite validation FAILED/incomplete; inspect returned diagnostics.')+' Original training remains paused.',flush=True)
    return 0 if complete else 76


if __name__=='__main__':raise SystemExit(main())
