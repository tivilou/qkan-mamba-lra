"""FP32 fused Mamba fork: preserved roster, finite actual-worker recovery, original UI."""
import argparse
from contextlib import ExitStack
import copy
import datetime as dt
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid
from .common import ROOT,atomic_json,digest,fingerprint,lock,read,revision,worker_failure
from . import key_bias_formal as original
from .migration import checked_pointer,completed_files,copy_verified,verify_completed

JOB='mamba1-retrieval-s42'
MODES=('continuous','interrupt','resume')
SCHEMA='qkan-mamba-fused-formal.v2'
WORKER='experiments.baseline_delivery.mamba_fused_worker'
REVIEW=ROOT/'docs/evidence/mamba-convolution-native-review-20261124.json'


def binding():
    names=('mamba_fused_formal.py','mamba_fused_worker.py','mamba_convolution.py','mamba-conv-wheel.json',
        'execution_plugins.py','worker.py','models.py','adaptive_memory.py','logical_batch.py',
        'execution.py','performance.py','torch_acceleration.py','data.py')
    paths=[ROOT/'experiments/baseline_delivery'/n for n in names]
    paths += [REVIEW]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}


def lineage(r):return dict(policy='mamba-causal-conv-fused-fp32-v2',implementation=r['binding'])


def source_context(source):
    fused=source/'mamba-input.json'
    value=read(fused if fused.exists() else source/'graph-input.json')
    if fused.exists() and value.get('schema') not in ('qkan-mamba-fused-formal.v1',SCHEMA):
        raise ValueError('Unknown source execution schema')
    return value


def checked_executor(root,expected):
    root=Path(root).resolve()
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if head!=expected or dirty:raise ValueError('Original executor revision/files changed')
    return str(root)


def unchanged(r):
    if any(digest(p)!=sha for p,sha in r['source_files'].items()):raise ValueError('Paused source changed')


def verify(path,ready=False):
    path=original.confined(path,ROOT/'runs');r=read(path);suite=path.parent
    if (r['schema']!=SCHEMA or r['source_revision']!=revision() or r['binding']!=binding()
            or r['manifest_sha256']!=digest(suite/'suite.json')
            or r['plan_sha256']!=digest(suite/'migration-source/plan.json')):raise ValueError('Bound fused Suite changed')
    manifest=read(suite/'suite.json')
    if len(manifest['jobs'])!=25 or {j['id'] for j in manifest['jobs']}!=original.JOBS:raise ValueError('Roster changed')
    if r.get('execution_plugin')!=dict(plugin='mamba.fused_fp32',version=2):raise ValueError('Bound execution plugin changed')
    for job in manifest['jobs']:
        if fingerprint(read(suite/'jobs'/job['id']/'config.json'))!=job['config_sha256']:raise ValueError('Scientific config changed')
        if job.get('preserved_completion'):verify_completed(suite,job)
    if ready:
        receipt=read(suite/'mamba-ready.json');report=read(suite/'preflight/preflight.json')
        if report['input_sha256']!=digest(path) or set(report['results'])!=set(MODES) or not report['disposable_checkpoints_removed']:raise ValueError('Preflight binding/cleanup differs')
        for mode in MODES:accept(report['results'][mode],r,mode,digest(path))
        if len(report['child_executions'])!=3 or any(e['exit_code']!=0 or not e['termination']['group_verified'] for e in report['child_executions']):raise ValueError('Worker child proof differs')
        if (receipt['input_sha256']!=digest(path) or receipt['report_sha256']!=digest(suite/'preflight/preflight.json')
                or not report['passed'] or not report['results']['resume']['recovery']['passed']):raise ValueError('Worker recovery pending')
    unchanged(r);return r


def prepare(args,suite):
    from .mamba_convolution import stage
    source=args.source_suite.resolve();manifest=read(source/'suite.json');statuses=read(source/'status.json')
    if len(manifest['jobs'])!=25 or {j['id'] for j in manifest['jobs']}!=original.JOBS:raise ValueError('Original roster required')
    controller=source_context(source)
    checked_executor(source.parent.parent,controller['source_revision'])
    if (source/'mamba-input.json').exists():files_controller='mamba-input.json'
    else:files_controller='graph-input.json'
    review=read(REVIEW)
    if review['timing']['summary']['timing_promising'] is not True:raise ValueError('Reviewed speed evidence required')
    if str(args.cache.resolve())!=controller['cache']:raise ValueError('Use original cache')
    files={str(source/n):digest(source/n) for n in ('suite.json','status.json',files_controller)}
    jobs=[];states={};plan={};candidates={}
    for entry in manifest['jobs']:
        name=entry['id'];src=source/'jobs'/name;dst=suite/'jobs'/name;job=copy.deepcopy(entry)
        config=read(src/'config.json')
        if statuses[name]['state']=='computed':
            complete=completed_files(src,config);copy_verified(src,dst,complete['files'])
            job.update(preserved_completion=complete,execution_role='preserved_result',execution_root=None)
            plan[name]=dict(checkpoint=None,completion=complete);states[name]=dict(state='computed',preserved=True)
            files.update({str(src/n):sha for n,sha in complete['files'].items()})
        else:
            if statuses[name]['state'] not in ('paused','pending'):raise ValueError('Source must be fully paused')
            copy_verified(src,dst,{'config.json':digest(src/'config.json')});pointer=None
            files[str(src/'config.json')]=digest(src/'config.json')
            if (src/'checkpoints/latest.json').exists():
                pointer,generation=checked_pointer(src)
                copy_verified(src,dst,{'checkpoints/'+generation.name:pointer['sha256']})
                atomic_json(dst/'checkpoints/latest.json',{**pointer,'previous':[]})
                files[str(src/'checkpoints/latest.json')]=digest(src/'checkpoints/latest.json');files[str(generation)]=pointer['sha256']
            if config['model']=='transformer':raise ValueError('Existing completed Transformer cohort required')
            root=checked_executor(job['execution_root'],job['execution_revision'])
            if config['model']=='mamba1':
                if config['task'] not in ('retrieval','image','pathfinder') or config.get('training_strategy'):raise ValueError('Unexpected Mamba job')
                if (config.get('torch_acceleration') or {}).get('required') is not None:raise ValueError('Mamba fusion requires optional eager execution')
                candidates[name]=dict(execution_root=root,execution_revision=job['execution_revision'],checkpoint=pointer)
                job.update(execution_root=str(ROOT),execution_revision=revision(),execution_role='mamba_fused_fp32')
            plan[name]=dict(checkpoint=pointer,completion=None);states[name]=dict(state='paused' if pointer else 'pending',started_training=False)
        jobs.append(job)
    cp=candidates[JOB]['checkpoint'];expected=review['evidence']['source_checkpoint_sha256']
    if cp is None:raise ValueError('Latest full Mamba checkpoint required')
    if not (source/'mamba-input.json').exists() and cp['sha256']!=expected:
        raise ValueError('Use reviewed latest source checkpoint; advanced source requires a bounded reassessment')
    atomic_json(suite/'suite.json',{**manifest,'source_revision':revision(),'jobs':jobs})
    atomic_json(suite/'status.json',states)
    atomic_json(suite/'migration-source/plan.json',dict(jobs=plan,source_suite=str(source),optimizer_updates=0))
    inherited=read(source/'formal-input.json') if (source/'formal-input.json').exists() else controller
    r=dict(schema=SCHEMA,source_revision=revision(),binding=binding(),origin_suite=str(source),
        source_suite=controller['source_suite'],original_root=controller['original_root'],
        data=manifest['data'],cache=str(args.cache.resolve()),source_files=files,candidates=candidates,
        manifest_sha256=digest(suite/'suite.json'),plan_sha256=digest(suite/'migration-source/plan.json'),
        target_config=inherited.get('target_config'),checkpoint=inherited.get('checkpoint'),
        implementation=inherited.get('implementation'),convolution_package=stage(),
        execution_plugin=dict(plugin='mamba.fused_fp32',version=2))
    atomic_json(suite/'mamba-input.json',r);verify(suite/'mamba-input.json');return r


def child_spec(suite,job,mode='formal'):
    return dict(target=WORKER,cwd=str(ROOT),argv=[sys.executable,'-B','-m',WORKER,
        '--request',str(suite/'mamba-input.json'),'--job',job,
        '--output',str(suite/'jobs'/job if mode=='formal' else suite/'preflight'/mode),'--mode',mode])


def accept(result,r,mode,input_sha):
    expected=r['candidates'][JOB]['checkpoint']['step']+(1 if mode=='interrupt' else 2)
    if (not result['passed'] or result['mode']!=mode or result['job']!=JOB or result['input_sha256']!=input_sha
            or result['step']!=expected or result['calls']<=0 or not result['copy_roundtrip_exact']
            or result['formal_updates_committed']!=0):raise ValueError('Actual worker receipt differs')
    return result


def preflight(suite,r,gpu):
    from .diagnostic_watch import Supervisor
    from .return_artifacts import upload_declared
    folder=suite/'preflight';folder.mkdir(exist_ok=True);path=folder/'preflight.json'
    if path.exists():
        if not read(path).get('passed'):raise ValueError('Partial preflight retained; inspect its report')
        upload_declared(folder,'mamba-fused-worker',['preflight.json'],run_id=suite.name+'--preflight')
    else:
        if any((folder/m/'checkpoints').exists() for m in MODES):raise ValueError('Partial worker outputs retained; inspect before retry')
        supervisor=Supervisor(900,300);report=dict(passed=False,results={},child_executions=supervisor.records,
            input_sha256=digest(suite/'mamba-input.json'),formal_updates_committed=0,disposable_updates=4)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',QKAN_DIAGNOSTIC_STAGE_FENCES='0')
        try:
            for mode in MODES:
                atomic_json(folder/'progress.json',dict(mode=mode,completed=len(report['results'])));print('Actual FP32 fused worker: '+mode,flush=True)
                if supervisor.run(child_spec(suite,JOB,mode),folder/(mode+'.log'),env)!=0:raise RuntimeError('Mamba worker preflight child failed; inspect compact report')
                report['results'][mode]=accept(read(folder/mode/'result.json'),r,mode,report['input_sha256'])
            if any(not e.get('termination',{}).get('group_verified') or e['exit_code']!=0 for e in supervisor.records):raise ValueError('Preflight child group not clean')
            report['passed']=report['results']['resume']['recovery']['passed']
        finally:
            report['failure_locations']={m:read(folder/m/'worker-error.json') for m in MODES if (folder/m/'worker-error.json').exists()}
            for mode in MODES:
                for p in (folder/mode/'checkpoints').glob('step-*.pt'):
                    if p.is_symlink():raise ValueError('Disposable generation is a link')
                    p.unlink()
            report['disposable_checkpoints_removed']=True
            unchanged(r);atomic_json(path,report)
            upload_declared(folder,'mamba-fused-worker',['preflight.json'],run_id=suite.name+'--preflight')
    for mode in MODES:accept(read(path)['results'][mode],r,mode,digest(suite/'mamba-input.json'))
    if not read(path)['results']['resume']['recovery']['passed']:raise ValueError('Recovery failed')
    atomic_json(suite/'mamba-ready.json',dict(input_sha256=digest(suite/'mamba-input.json'),report_sha256=digest(path)))


def job_spec(suite,r,job):
    if job['id'] in r['candidates']:
        origin=r['candidates'][job['id']]
        checked_executor(origin['execution_root'],origin['execution_revision'])
        return child_spec(suite,job['id'])
    return original.job_spec(suite,r,job)


def validate_result(suite,r,job):
    value=original.validate_job_result(suite,r,job)
    if job['id'] in r['candidates']:
        proof=read(suite/'jobs'/job['id']/'mamba-fused-execution.json')
        if proof['observed_fused_calls']<=0 or proof['implementation']!=r['binding']:raise ValueError('Actual fused execution missing')
    return value


def execute(args,suite):
    path=suite/'mamba-input.json';source=Path(read(path)['origin_suite']) if path.exists() else args.source_suite.resolve()
    controller=source_context(source)
    with ExitStack() as held:
        for origin in dict.fromkeys((str(source),controller['origin_suite'],controller['source_suite'])):
            for p in (Path(origin)/'progress-supervisor.lock',Path(origin)/'suite.lock',
                    *[Path(origin)/'jobs'/j/'worker.lock' for j in sorted(original.JOBS)]):held.enter_context(lock(p))
        r=verify(path) if path.exists() else prepare(args,suite)
        if args.cache and str(args.cache.resolve())!=r['cache']:raise ValueError('Resume cache differs')
        if args.upload_only:
            from .return_artifacts import upload_declared
            upload_declared(suite/'preflight','mamba-fused-worker',['preflight.json'],run_id=suite.name+'--preflight')
            if (suite/'compute-complete.json').exists():
                from .runner import deliver
                deliver(suite)
            return 0
        from .runner import discover_gpus
        from .attention_diagnostic import gpu_preflight
        gpus,inventory=discover_gpus(args.gpu)
        if len(gpus)!=1 or not gpu_preflight(inventory[gpus[0]])['idle']:raise ValueError('One idle GPU required; original tasks preserved')
        if not (suite/'mamba-ready.json').exists():preflight(suite,r,gpus[0])
        verify(path,ready=True)
        if args.prepare_only:return 0
        statuses=read(suite/'status.json')
        for job in read(suite/'suite.json')['jobs']:
            if statuses[job['id']]['state']=='computed':validate_result(suite,r,job);continue
            code=original.run_job(suite,r,job,gpus[0],statuses,spec_builder=job_spec,result_validator=validate_result)
            if code!=0:return code
        from .return_artifacts import upload_declared
        event=suite/'events/execution'
        atomic_json(event/'execution.json',dict(input=r,ready=read(suite/'mamba-ready.json'),jobs={j:read(suite/'jobs'/j/'mamba-fused-execution.json') for j in r['candidates']}))
        upload_declared(event,'mamba-fused-formal',['execution.json'],run_id=suite.name+'--execution')
        atomic_json(suite/'compute-complete.json',dict(state='computed',jobs=25))
        from .runner import deliver
        deliver(suite);return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--source-suite',type=Path);g.add_argument('--resume',type=Path)
    p.add_argument('--cache',type=Path);p.add_argument('--gpu',default='0')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--upload-only',action='store_true');a=p.parse_args()
    if a.source_suite and a.cache is None:p.error('cache required')
    from .bootstrap import check_exchange
    check_exchange();revision()
    suite=original.confined(a.resume,ROOT/'runs') if a.resume else ROOT/'runs'/('mamba-fused-formal-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    suite.mkdir(parents=True,exist_ok=bool(a.resume));print('Formal continuation Suite: '+str(suite),flush=True)
    def pause(*_):raise KeyboardInterrupt
    for name in ('SIGINT','SIGTERM','SIGHUP'):
        if hasattr(signal,name):signal.signal(getattr(signal,name),pause)
    with lock(suite/'suite.lock'):
        try:return execute(a,suite)
        except BaseException as error:
            worker_failure(suite,error)
            from .return_artifacts import upload_declared
            event=suite/'events'/('failure-'+uuid.uuid4().hex[:8]);atomic_json(event/'failure.json',read(suite/'worker-error.json'))
            try:upload_declared(event,'mamba-fused-formal',['failure.json'],run_id=suite.name+'--'+event.name)
            except Exception:print('Compact failure retained for return retry.',flush=True)
            return 75 if isinstance(error,KeyboardInterrupt) else 2


if __name__=='__main__':raise SystemExit(main())
