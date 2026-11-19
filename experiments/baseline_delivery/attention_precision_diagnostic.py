"""Isolated BF16 efficient-attention mechanism observation, never training."""
import argparse
from contextlib import ExitStack
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .common import ROOT,atomic_json,digest,fingerprint,lock,read,revision
from .optimization_plugins import resolve

JOB='transformer-retrieval-s42'
SOURCE_REVISION='b07d98e17ad0f996dbd416b0f38e015df6af4f60'
TARGET='experiments.baseline_delivery.attention_precision_native'


def bind_source(source,cache):
    """Call while suite/supervisor/worker locks are held. No checkpoint copy."""
    from .migration import checked_pointer,validate_source_admission
    source=Path(source).resolve();root=source.parent.parent
    if source.parent.name!='runs': raise ValueError('Source suite must be under runs/')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip()
    if head!=SOURCE_REVISION or dirty: raise ValueError('Expected clean original b07 executor')
    validate_source_admission(source,head)
    suite=read(source/'suite.json');context=read(source/'context.json')
    folder=source/'jobs'/JOB;config=read(folder/'config.json')
    jobs=[row for row in suite['jobs'] if row['id']==JOB]
    if (len(jobs)!=1 or jobs[0]['config_sha256']!=fingerprint(config)
            or any(item.get('source_revision')!=head for item in (suite,context,config))
            or context['cache']!=str(Path(cache).resolve()) or config['model']!='transformer'
            or config['task']!='retrieval' or config['width']!=128 or config['heads']!=4
            or config['layers']!=4 or config.get('length',4000)!=4000):
        raise ValueError('Source/cache/model contract differs')
    if (folder/'summary.json').exists() or read(source/'status.json')[JOB]['state']=='computed':
        raise ValueError('Mechanism probe requires unfinished original Retrieval')
    pointer,generation=checked_pointer(folder)
    if pointer['schema']!='qkan-training-checkpoint.v1': raise ValueError('Full Torch checkpoint required')
    data=Path(suite['data'])
    if not data.is_absolute() or not data.is_dir(): raise ValueError('Original dataset missing')
    if not (Path(cache)/'arrays'/config['data_hashes']['train']/'complete.json').is_file():
        raise ValueError('Existing verified train array cache required')
    metadata={str(p):digest(p) for p in (source/'suite.json',source/'context.json',folder/'config.json',folder/'checkpoints/latest.json')}
    return dict(source_suite=str(source),source_config=config,source_config_sha256=fingerprint(config),
        checkpoint=dict(path=str(generation),**pointer),source_metadata_sha256=metadata,
        data=str(data),cache=str(Path(cache).resolve()))


def unchanged(binding):
    if (digest(binding['checkpoint']['path'])!=binding['checkpoint']['sha256']
            or any(digest(path)!=sha for path,sha in binding['source_metadata_sha256'].items())):
        raise ValueError('Original checkpoint or metadata changed')


def execute(args,folder,report):
    from .attention_diagnostic import gpu_preflight
    from .runner import discover_gpus
    from .diagnostic_watch import Supervisor
    selection=resolve(read(args.optimizations),scope='attention_probe')
    if not selection.bf16_amp: raise ValueError('Explicit BF16 preset required')
    gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1: raise ValueError('Choose exactly one physical GPU')
    gpu=gpus[0];report.update(gpu_identity=inventory[gpu],gpu_preflight=gpu_preflight(inventory[gpu]))
    if not report['gpu_preflight']['idle']: raise ValueError('Selected GPU has compute processes; existing work left running')
    source=args.source_suite.resolve()
    if source.parent.name!='runs' or not (source/'jobs'/JOB).is_dir():
        raise ValueError('Original source Suite/job missing')
    # Match the existing source supervisor/suite/worker exclusion protocol.
    # Nonblocking locks reject a live source; this command never stops it.
    with ExitStack() as held:
        for path in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/JOB/'worker.lock'):
            held.enter_context(lock(path))
        binding=bind_source(source,args.cache)
        request=dict(schema='qkan-attention-probe-input.v1',**binding,source_revision=report['source_revision'],
            optimization_config=read(args.optimizations),optimization_config_sha256=digest(args.optimizations),
            micro_batch=args.micro_batch)
        atomic_json(folder/'probe-input.json',request)
        supervisor=Supervisor(args.child_timeout,args.stall_timeout)
        report['child_executions']=supervisor.records
        output=folder/'probe';spec=dict(target=TARGET,cwd=str(ROOT),argv=[sys.executable,'-B','-m',TARGET,
            '--request',str(folder/'probe-input.json'),'--output',str(output)])
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        try:
            code=supervisor.run(spec,folder/'probe.log',env=env)
            if (output/'worker-error.json').is_file():
                report['native_failure_locations']=read(output/'worker-error.json')
            result=read(output/'result.json')
            if (code not in (0,76) or result.get('request_sha256')!=digest(folder/'probe-input.json')
                    or result.get('plugin_manifest')!=selection.manifest()
                    or result.get('checkpoint_sha256')!=binding['checkpoint']['sha256']
                    or result.get('rollback')!={'inputs_exact':True,'rng_exact':True,'leaf_gradients_unmodified':True}
                    or (code==0)!=bool(result.get('mechanism_passed'))):
                raise ValueError('Native attention report incomplete or identity differs')
            report.update(result,passed=code==0,schema='qkan-attention-precision.v1')
        finally: unchanged(binding)
        report['original_checkpoint_and_metadata_unchanged']=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-suite',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--optimizations',type=Path,required=True)
    parser.add_argument('--micro-batch',type=int,choices=(16,32),default=16)
    parser.add_argument('--child-timeout',type=float,default=900)
    parser.add_argument('--stall-timeout',type=float,default=300)
    args=parser.parse_args();selected=resolve(read(args.optimizations),scope='attention_probe')
    if not selected.bf16_amp: parser.error('Enable the scoped BF16 mechanism option')
    from .bootstrap import check_exchange
    check_exchange();source=revision()
    folder=ROOT/'runs'/('attention-precision-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(schema='qkan-attention-precision.v1',passed=False,status='failed',source_revision=source,
        formal_training_started=False,training_samples_committed=0,optimizer_updates=0,
        architecture_status='approved_pending_native_trace',native_trace_ready=False,
        scope='First layer/document1 train-prefix attention forward+backward; not full-update admission')
    report['plugin_files']={name:digest(Path(__file__).parent/name) for name in
        ('optimization_plugins.py','attention_precision.py','attention_precision_native.py','attention_precision_diagnostic.py')}
    report['plugin_tree_sha256']=fingerprint(report['plugin_files'])
    error=None
    print('BF16 attention diagnostic Suite: '+str(folder),flush=True)
    try: execute(args,folder,report)
    except BaseException as caught:
        error=caught;report.update(status='failed',passed=False,error_type=type(caught).__name__)
        from .common import worker_failure
        worker_failure(folder,caught)
        report['failure_locations']=read(folder/'worker-error.json')
    atomic_json(folder/'attention-precision.json',report)
    from .return_artifacts import upload_declared
    try: upload_declared(folder,'attention-precision',['attention-precision.json'])
    except Exception:
        print('Return pending; compact report and retry manifest retained: '+str(folder),flush=True)
        if error is None: raise
    if error is not None: raise error
    print('Attention mechanism observation complete; formal training was not started.',flush=True)
    return 0 if report['passed'] else 76


if __name__=='__main__': raise SystemExit(main())
