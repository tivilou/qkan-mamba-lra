"""Prospective MEGA Text execution amendment; original Suite/core stay pinned.

Controller helpers are deliberately stdlib-only and loadable with runpy, so the
old dispatcher can keep its own package imports. Native execution uses Hooks.
"""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import FunctionType
import uuid

ROOT = Path(__file__).resolve().parents[2]
JOB = 'mega-text-s42'
POLICY = dict(plugin='mega.fft_tail_bf16', version=1, graph=False)
REVIEW = ROOT/'docs/evidence/mega-text-policy-switch-20261127.json'
RECORD = 'mega-policy.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    with Path(path).open('rb') as handle:
        value=hashlib.sha256()
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def binding():
    review=read(REVIEW)
    files=['experiments/baseline_delivery/'+n for n in review['qualified_binding']]
    files+=['experiments/baseline_delivery/mega_formal.py',REVIEW.relative_to(ROOT).as_posix()]
    return {n:digest(ROOT/n) for n in files}


def revision():
    value=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if dirty:raise ValueError('MEGA policy controller checkout has tracked changes')
    return value


def check_code(record):
    if record['code_root']!=str(ROOT) or record['code_revision']!=revision() or record['binding']!=binding():
        raise ValueError('Resume with the pinned MEGA policy controller checkout: '+record['code_root'])
    review=read(REVIEW)
    if any(digest(ROOT/'experiments/baseline_delivery'/n)!=sha for n,sha in review['qualified_binding'].items()):
        raise ValueError('Qualified MEGA science changed')


def check_record(root,suite):
    record=read(suite/RECORD);check_code(record)
    controller=read(suite/'mamba-input.json')
    if (record.get('schema')!='qkan-mega-policy.v1' or record['origin_root']!=str(root)
            or record['suite']!=str(suite) or record['job']!=JOB
            or record['mode'] not in ('fft-tail-bf16','original')
            or record['manifest_sha256']!=digest(suite/'suite.json')
            or record['config_sha256']!=fingerprint(read(suite/'jobs'/JOB/'config.json'))
            or any(record[k]!=controller[k] for k in ('data','cache'))
            or record['baseline_root']!=controller['original_root']):
        raise ValueError('MEGA amendment binding differs')
    return record


def activate(root,suite,mode):
    """Called under supervisor, Suite and selected worker locks; no model load."""
    from experiments.baseline_delivery.common import atomic_json
    from experiments.baseline_delivery.migration import checked_pointer,copy_verified
    folder=suite/'jobs'/JOB
    status=read(suite/'status.json')[JOB]
    if status['state'] not in ('paused','pending') or (folder/'summary.json').exists():
        raise ValueError('Pause the unfinished MEGA Text worker first; completed jobs are preserved')
    cp,generation=checked_pointer(folder)
    if (suite/RECORD).exists():
        record=check_record(root,suite)
    else:
        if mode=='original':raise ValueError('No MEGA policy amendment to roll back')
        review=read(REVIEW);config=read(folder/'config.json')
        if (fingerprint(config)!=review['config_sha256'] or config.get('model')!='mega'
                or config.get('task')!='text' or cp['contract']!=review['contract']
                or cp['step']<review['qualified_source_step']):
            raise ValueError('Use the reviewed MEGA Text configuration and latest full state')
        controller=read(suite/'mamba-input.json')
        event='mega-policy-'+uuid.uuid4().hex
        record=dict(schema='qkan-mega-policy.v1',amendment=event,job=JOB,
            suite=str(suite),origin_root=str(root),manifest_sha256=digest(suite/'suite.json'),
            config_sha256=fingerprint(config),code_root=str(ROOT),code_revision=revision(),binding=binding(),
            data=controller['data'],cache=controller['cache'],baseline_root=controller['original_root'],mode=mode,
            baseline_checkpoint=cp,quality_scope=review['quality_scope'],transitions=[])
        check_code(record)
        # Preserve the pre-switch generation independently of the worker's prune.
        archive=suite/'events'/event/'baseline'
        copy_verified(folder,archive,{'checkpoints/'+generation.name:cp['sha256']})
        atomic_json(archive/'checkpoints/latest.json',cp)
    record['mode']=mode
    record['transitions'].append(dict(mode=mode,checkpoint=cp,updated_at=time.time()))
    event=suite/'events'/record['amendment']
    atomic_json(event/('transition-'+str(len(record['transitions']))+'.json'),record)
    atomic_json(suite/RECORD,record)
    print('MEGA Text: '+mode+'; latest step '+str(cp['step'])+'; Graph off',flush=True)
    return record


def worker_spec(suite,record):
    return dict(target='experiments.baseline_delivery.mega_formal',cwd=record['code_root'],
        argv=[str(Path(record['code_root'])/'.venv-mega/bin/python'),'-B','-m',
            'experiments.baseline_delivery.mega_formal','--worker','--suite',str(suite)])


def private(function,**overrides):
    result=FunctionType(function.__code__,dict(function.__globals__,**overrides),
        function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def attach(target,root,suite,ordered_read):
    """Replace only this invocation's two dispatch seams, never module globals."""
    record=check_record(root,suite)
    def spec(s,r,job):
        return worker_spec(s,record) if job['id']==JOB else target.job_spec(s,r,job)
    def validate(s,r,job):
        result=target.validate_result(s,r,job)
        if job['id']==JOB:
            summary=read(s/'jobs'/JOB/'summary.json')
            if summary.get('execution_amendment',{}).get('amendment')!=record['amendment']:
                raise ValueError('Completed MEGA result lacks its prospective execution lineage')
        return result
    return private(target.execute,read=ordered_read,job_spec=spec,validate_result=validate)


def lineage(record):
    return dict(amendment=record['amendment'],mode=record['mode'],selection=POLICY if record['mode']=='fft-tail-bf16' else None,
        code_revision=record['code_revision'],binding=record['binding'],quality_scope=record['quality_scope'])


def native(suite):
    from .common import atomic_json,lock,worker_failure
    from .migration import checked_pointer
    record=check_record(suite.parent.parent,suite);folder=suite/'jobs'/JOB
    if record['data']!=read(suite/'mamba-input.json')['data'] or record['cache']!=read(suite/'mamba-input.json')['cache']:
        raise ValueError('MEGA data/cache binding differs')
    import torch
    from . import worker as w
    from .execution_plugins import Hooks,create,reset_measurements
    config=read(folder/'config.json')
    # Validate the original certificate with its own original code. The explicit
    # amendment below adds precision provenance; it does not forge an old seal.
    baseline=record['baseline_root']
    subprocess.check_call([str(ROOT/'.venv-mega/bin/python'),'-B','-c',
        'import sys; from experiments.baseline_delivery.runner import check_release; '
        'from experiments.baseline_delivery.common import read; check_release(read(sys.argv[1]))',
        str(folder/'config.json')],cwd=baseline,env=dict(os.environ,PYTHONPATH=baseline))
    policy=None;expected=lineage(record);loaded=False
    def install(model):
        nonlocal policy
        policy=create(expected['selection'],config,torch,model)
        return policy if policy else nullcontext()
    def configure(controller,state):
        nonlocal loaded
        loaded=True
        prior=state.get('mega_execution')
        cp,_=checked_pointer(folder)
        if cp['sha256']!=record['transitions'][-1]['checkpoint']['sha256'] and (
                not prior or prior.get('amendment')!=record['amendment'] or prior.get('binding')!=record['binding']):
            raise ValueError('Latest checkpoint MEGA lineage differs')
        if prior!=expected:reset_measurements(controller)
    def proof(step):
        status=policy.status() if policy else dict(plugin=None,graph=False,training_calls=0,fft_calls=0,island_calls=0)
        value=dict(status,**dict(pid=os.getpid(),updated_at=time.time(),step=step,amendment=record['amendment'],mode=record['mode']))
        atomic_json(folder/'mega-execution.json',value)
        path=folder/'environment.json'
        if path.exists():
            env=read(path);env.update(execution_amendment=expected,
                precision=status.get('precision','original float32; no mixed precision'))
            atomic_json(path,env)
    def before_save(state):state['mega_execution']=expected
    def after_save(state):proof(state['step'])
    def progress(heartbeat):proof(heartbeat['optimizer_step'])
    w.PAUSE=False
    for name in ('SIGTERM','SIGINT','SIGHUP'):
        if hasattr(signal,name):signal.signal(getattr(signal,name),w.request_pause)
    try:
        with lock(folder/'worker.lock'):
            code=w.run(config,Path(record['data']),Path(record['cache']),folder,'cuda:0',resume=True,
                execution=Hooks(model_context=install,configure_controller=configure,before_save=before_save,
                    after_save=after_save,on_progress=progress,
                    acceleration_state=lambda saved:{**(saved or {}),'mode':'eager','benchmarked':True,
                        'reason':'mega_fft_tail_bf16' if policy else 'mega_original_rollback','benchmark':None}))
            if not loaded:raise ValueError('MEGA full-state controller was not restored')
            if code==0:
                summary=read(folder/'summary.json');summary['execution_amendment']=expected
                atomic_json(folder/'summary.json',summary)
            return code
    except BaseException as error:
        worker_failure(folder,error);raise
    finally:
        # Compact automatic return; failed transport retains retryable local files.
        from .return_artifacts import upload_declared
        event=ROOT/'runs'/('mega-policy-status-'+uuid.uuid4().hex[:12])
        value=dict(policy=record,execution=read(folder/'mega-execution.json') if (folder/'mega-execution.json').exists() else None)
        if (folder/'worker-error.json').exists():value['failure']=read(folder/'worker-error.json')
        atomic_json(event/'execution.json',value)
        try:upload_declared(event,'mega-policy',['execution.json'],run_id=event.name)
        except Exception:print('MEGA execution receipt retained: '+str(event),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true',required=True)
    parser.add_argument('--suite',type=Path,required=True)
    args=parser.parse_args()
    try:raise SystemExit(native(args.suite.resolve()))
    except Exception as error:
        from .common import worker_failure
        from .mega_refinement import error_chain
        from .common import atomic_json
        output=args.suite.resolve()/'jobs'/JOB
        worker_failure(output,error)
        atomic_json(output/'mega-policy-error.json',dict(chain=error_chain(error)))
        raise
