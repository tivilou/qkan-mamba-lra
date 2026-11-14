"""Isolated two-layer Graph candidate; latest b07 checkpoint, no training launch."""
import argparse
import copy
import datetime as dt
import os
from pathlib import Path
import uuid

from .common import ROOT,atomic_json,child_spec,digest,fingerprint,lock,read,revision
from .attention_diagnostic import JOB,gpu_preflight,failure_details


def execute(args,suite,report):
    from .readiness import policy_check
    from .migration import snapshot,compatible,PARALLEL_REVISION
    from .runner import canary,discover_gpus
    from .torch_acceleration import policy,require_actual_backend
    from .acceleration_calibration import calibrate
    from .diagnostic_watch import Supervisor
    supervisor=Supervisor(args.child_timeout,args.stall_timeout)
    report['child_executions']=supervisor.records
    source=revision();implementation,review_policy=policy_check()
    plan=snapshot(args.migrate_from,suite,args.cache);record=plan['jobs'][JOB]
    old=read(suite/'migration-source'/JOB/'config.json')
    if old.get('source_revision')!=PARALLEL_REVISION or record.get('completion') or not record.get('checkpoint'):
        raise ValueError('Partition candidate requires the latest unfinished admitted b07 Retrieval checkpoint')
    config=copy.deepcopy(old)
    config.update(source_revision=source,torch_acceleration=policy('compile_graph','eager_partitioned','graph'))
    compatible(old,config)
    data=Path(plan['data'])
    if not data.is_absolute() or not data.is_dir(): raise ValueError('Original local dataset unavailable')
    gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1: raise ValueError('Choose exactly one physical GPU')
    gpu=gpus[0];report['gpu_preflight']=gpu_preflight(inventory[gpu])
    if not report['gpu_preflight']['idle']: raise ValueError('Selected GPU has compute processes; keep current training running')
    report.update(source_revision=source,job=JOB,config_sha256=fingerprint(config),gpu_identity=inventory[gpu],
        source_checkpoint_sha256=record['checkpoint']['sha256'],restored_step=record['checkpoint']['step'],
        preserved_completed=[name for name,row in plan['jobs'].items() if row.get('completion')])
    atomic_json(suite/'context.json',dict(source_revision=source,implementation_sha256=implementation,
        policy_sha256=review_policy,migration_sha256=digest(suite/'migration-source/plan.json'),
        purpose='partitioned_attention_candidate_only',formal_release_allowed=False,fixed_micro_batches=[16,32]))
    previous={key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','QKAN_DIAGNOSTIC_STAGE_FENCES')}
    report['recovery']={}
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=gpu;os.environ['QKAN_DIAGNOSTIC_STAGE_FENCES']='0'
        for micro in (16,32):
            report['stage']='fresh_process_recovery_micro'+str(micro)
            folder=suite/'preflight'/('micro'+str(micro))/JOB
            def fixed_run(spec,log,env=None):
                fixed={**spec,'argv':[*spec['argv'],'--probe-micro-batch',str(micro)]}
                return supervisor.run(fixed,log,env)
            print('Two-layer Graph fresh-process recovery; fixed micro '+str(micro),flush=True)
            receipt=canary(config,data,args.cache,folder,'cuda:0',
                announce=lambda message:print(message,flush=True),run=fixed_run)
            if (receipt.get('passed') is not True or receipt.get('fresh_process') is not True
                    or receipt.get('trace_semantic_and_replay') is not True or receipt.get('test_access') is not False
                    or receipt.get('config_sha256')!=fingerprint(config) or receipt.get('source_revision')!=source):
                raise ValueError('Partition recovery identity or trace differs')
            require_actual_backend(receipt,config)
            for name in ('uninterrupted','interrupted'):
                status=read(folder/name/'adaptive-memory.json')
                if status.get('last_update',{}).get('used_micro_batch')!=micro:
                    raise ValueError('Partition recovery did not retain the fixed microbatch')
            report['recovery'][str(micro)]=dict(receipt,fixed_micro_batch=micro)
        report['stage']='real_checkpoint_micro32_vs_incumbent16'
        print('Recovery passed. Comparing full logical128 updates: matched-micro eager, then incumbent Graph16; exact rollback.',flush=True)
        path=calibrate(suite,JOB,config,data,args.cache,gpu,inventory[gpu],record,supervisor.run,child_spec,
            probe_micro_batch=32,compare_incumbent=True)
        report.update(calibration=read(path),stage='complete',status='candidate_passed',passed=True)
    finally:
        for key,value in previous.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migrate-from',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--child-timeout',type=float,default=900)
    parser.add_argument('--stall-timeout',type=float,default=300)
    args=parser.parse_args()
    from .diagnostic_watch import Supervisor
    Supervisor(args.child_timeout,args.stall_timeout)
    from .bootstrap import check_exchange
    check_exchange()
    suite=ROOT/'runs'/('partitioned-attention-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    suite.mkdir(parents=True,exist_ok=False)
    print('Partitioned-attention diagnostic Suite: '+str(suite),flush=True)
    report=dict(schema='qkan-partitioned-attention-candidate.v1',status='failed',passed=False,stage='prepare',
        backend='eager_partitioned',compiler_backend='eager',inductor_fusion=False,segment_layers=2,
        formal_release_allowed=False,formal_training_started=False,source_immutable=True,
        interpretation='Diagnostic only. Zero-commit requires real-checkpoint rollback; synthetic canaries remain disposable.')
    error=None
    with lock(suite/'suite.lock'):
        try: execute(args,suite,report)
        except BaseException as caught:
            error=caught;details=failure_details(suite,caught)
            for micro in (16,32):
                folder=suite/'preflight'/('micro'+str(micro))/JOB
                for name in ('child-failure.json','recovery-validation.json',
                             'uninterrupted/worker-error.json','interrupted/worker-error.json',
                             'uninterrupted/torch-acceleration.json','interrupted/torch-acceleration.json'):
                    path=folder/name
                    if path.is_file() and path.stat().st_size<=512*1024:
                        details['micro'+str(micro)+'/'+name]=read(path)
            report.update(error_type=type(caught).__name__,diagnostics=details)
        atomic_json(suite/'partitioned-attention.json',report)
        from .return_artifacts import upload_declared
        try: upload_declared(suite,'partitioned-attention',['partitioned-attention.json'])
        except Exception:
            print('Return pending; report and delivery manifest retained: '+str(suite),flush=True)
            if error is None: raise
        if error is not None: raise error
    print('Partitioned Graph32 passed recovery and calibration against Graph16; formal training was not started.',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
