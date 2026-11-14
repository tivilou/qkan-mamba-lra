"""One opt-in parallel document candidate: exact recovery then real calibration.

Uses the immutable original checkpoint; never starts formal training.
"""
import argparse
import copy
import datetime as dt
import os
from pathlib import Path
import subprocess
import uuid

from .common import ROOT,atomic_json,child_spec,digest,fingerprint,lock,read,revision

JOB='transformer-retrieval-s42'


def gpu_preflight(identity):
    """Refuse competing compute processes; never stop an unowned old worker."""
    uuid=identity.split(',')[1].strip()
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    if result.returncode: raise ValueError('GPU compute-process inventory unavailable')
    pids=[]
    for line in result.stdout.splitlines():
        if not line.strip(): continue
        fields=[v.strip() for v in line.split(',')]
        if len(fields)!=2 or not fields[0].isdigit():
            raise ValueError('GPU compute-process inventory is not verifiable')
        if fields[1]==uuid: pids.append(int(fields[0]))
    return dict(idle=not pids,compute_pids=sorted(set(pids)))


def execute(args,suite,report):
    from .readiness import policy_check
    from .migration import snapshot,compatible
    from .runner import canary,discover_gpus
    from .torch_acceleration import policy,require_actual_backend
    from .acceleration_calibration import calibrate
    from .diagnostic_watch import Supervisor
    supervisor=Supervisor(getattr(args,'child_timeout',900),getattr(args,'stall_timeout',300))
    report['child_executions']=supervisor.records
    source=revision();implementation,review_policy=policy_check()
    plan=snapshot(args.migrate_from,suite,args.cache);record=plan['jobs'][JOB]
    if record.get('completion') or not record.get('checkpoint'):
        raise ValueError('Parallel candidate requires an unfinished full checkpoint')
    old=read(suite/'migration-source'/JOB/'config.json');config=copy.deepcopy(old)
    if not config.get('performance') or not config.get('adaptive_memory',{}).get('logical_tiers'):
        raise ValueError('Parallel candidate requires the original dynamic performance checkpoint')
    config.update(source_revision=source,torch_acceleration=policy('compile_graph','eager_parallel','graph'))
    compatible(old,config)
    data=Path(plan['data'])
    if not data.is_absolute() or not data.is_dir(): raise ValueError('Original local dataset unavailable')
    gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1: raise ValueError('Choose exactly one physical GPU')
    gpu=gpus[0]
    report['gpu_preflight']=gpu_preflight(inventory[gpu])
    if not report['gpu_preflight']['idle']:
        raise ValueError('Selected GPU still has compute processes; inspect listed PIDs before retrying')
    report.update(source_revision=source,job=JOB,config_sha256=fingerprint(config),gpu_identity=inventory[gpu],
        source_checkpoint_sha256=record['checkpoint']['sha256'],restored_step=record['checkpoint']['step'],
        preserved_completed=[name for name,row in plan['jobs'].items() if row.get('completion')])
    atomic_json(suite/'context.json',dict(source_revision=source,implementation_sha256=implementation,
        policy_sha256=review_policy,migration_sha256=digest(suite/'migration-source/plan.json'),
        purpose='parallel_attention_candidate_only',formal_release_allowed=False,
        diagnostic_stage_fences=bool(getattr(args,'stage_fences',False))))
    folder=suite/'preflight'/JOB;previous=os.environ.get('CUDA_VISIBLE_DEVICES')
    fence_key='QKAN_DIAGNOSTIC_STAGE_FENCES';previous_fences=os.environ.get(fence_key)
    fences=bool(getattr(args,'stage_fences',False));report['stage_fences']=fences
    print('Stage fences: synthetic recovery observation only; no timing or admission.' if fences else
          'Parallel document candidate: one Retrieval recovery/trace check, then original-checkpoint calibration.',flush=True)
    report['stage']='fresh_process_recovery';os.environ['CUDA_VISIBLE_DEVICES']=gpu
    os.environ[fence_key]='1' if fences else '0'
    try: receipt=canary(config,data,args.cache,folder,'cuda:0',announce=lambda message:print(message,flush=True),run=supervisor.run)
    finally:
        if previous is None: os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else: os.environ['CUDA_VISIBLE_DEVICES']=previous
        if previous_fences is None: os.environ.pop(fence_key,None)
        else: os.environ[fence_key]=previous_fences
    if (receipt.get('passed') is not True or receipt.get('fresh_process') is not True
            or receipt.get('trace_semantic_and_replay') is not True or receipt.get('test_access') is not False
            or receipt.get('config_sha256')!=fingerprint(config) or receipt.get('source_revision')!=source):
        raise ValueError('Parallel recovery receipt identity or trace differs')
    require_actual_backend(receipt,config)
    if fences:
        report.update(stage_observation=receipt,stage='complete',status='stage_fences_passed',
                      stage_observation_passed=True,passed=False)
        return
    report['recovery']=receipt;report['stage']='real_checkpoint_calibration'
    print('Exact recovery passed. Calibrating the original full update: numeric/RNG agreement, finite AB/BA timing, exact rollback.',flush=True)
    path=calibrate(suite,JOB,config,data,args.cache,gpu,inventory[gpu],record,supervisor.run,child_spec)
    report.update(calibration=read(path),stage='complete',status='candidate_passed',passed=True)


def failure_details(suite,error):
    # These existing worker diagnostics contain bounded numeric/control metadata,
    # never raw log text, input tensors, weights or traces.
    details={};suite=Path(suite)
    def compact(path):
        if not path.is_file(): return
        name=path.relative_to(suite).as_posix()
        if path.stat().st_size>512*1024:
            details[name]=dict(status='compact_report_size_exceeded');return
        try: details[name]=read(path)
        except (OSError,ValueError): details[name]=dict(status='compact_report_unreadable')
    for name in ('child-failure.json','recovery-validation.json'):
        compact(suite/'preflight'/JOB/name)
    if hasattr(error,'diagnostic'):
        name='child_execution' if error.diagnostic.get('schema')=='qkan-diagnostic-child.v1' else 'calibration'
        details[name]=error.diagnostic
    for name in ('uninterrupted','interrupted'):
        for filename in ('torch-acceleration.json','worker-error.json'):
            compact(suite/'preflight'/JOB/name/filename)
    for folder in sorted((suite/'calibration').glob('*/'+JOB))[:2]:
        for filename in ('acceleration-probe.json','torch-acceleration.json','worker-error.json','adaptive-memory.json'):
            compact(folder/filename)
    return details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migrate-from',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--child-timeout',type=float,default=900,help='Absolute seconds per diagnostic child; never formal training')
    parser.add_argument('--stall-timeout',type=float,default=300,help='Seconds without a new internal phase; heartbeat alone does not reset this')
    parser.add_argument('--stage-fences',action='store_true',help='Synthetic recovery stage localization only; adds GPU waits, skips calibration and cannot admit acceleration')
    args=parser.parse_args()
    from .diagnostic_watch import Supervisor
    Supervisor(args.child_timeout,args.stall_timeout)  # Reject bad limits before preparing a Suite.
    from .bootstrap import check_exchange
    check_exchange()
    suite=ROOT/'runs'/('parallel-attention-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    suite.mkdir(parents=True,exist_ok=False)
    print('Parallel-attention diagnostic Suite: '+str(suite),flush=True)
    report=dict(schema='qkan-parallel-attention-candidate.v1',status='failed',passed=False,
        stage='prepare',backend='eager_parallel',compiler_backend='eager',inductor_fusion=False,
        formal_release_allowed=False,formal_training_started=False,source_immutable=True,
        interpretation='Zero-commit calibration is declared only by its exact rollback report; synthetic canary updates stay in this disposable diagnostic Suite.')
    error=None
    with lock(suite/'suite.lock'):
        try: execute(args,suite,report)
        except BaseException as caught:
            error=caught;report.update(error_type=type(caught).__name__,diagnostics=failure_details(suite,caught))
        atomic_json(suite/'parallel-attention.json',report)
        from .return_artifacts import upload_declared
        try: upload_declared(suite,'parallel-attention',['parallel-attention.json'])
        except Exception:
            print('Return pending; report and delivery manifest retained. Retry upload only: '+str(suite),flush=True)
            if error is None: raise
        if error is not None: raise error
    print('Stage observation completed; acceleration admission and timing were not performed.' if args.stage_fences else
          'Parallel document candidate passed recovery and measured calibration; formal training was not started.',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
