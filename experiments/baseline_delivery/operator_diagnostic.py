"""Collect one eager operator update from an immutable checkpoint, then return it.

No compiler/recovery benchmark loop, training continuation or release creation.
The real native worker runs only on the collaborator's selected CUDA device.
"""
import argparse
import copy
import datetime as dt
import os
from pathlib import Path
import uuid

from .common import ROOT,atomic_json,child_spec,digest,fingerprint,lock,read,revision,run_child


def observe_once(observe,capture,rollback,synchronize):
    from .torch_acceleration import fatal_cuda
    capture();synchronize()
    try: result=observe()
    except BaseException as error:
        if not fatal_cuda(error): rollback()
        raise
    rollback()  # Inexact restoration must prevent a zero-commit declaration.
    return result


def collect_child(suite,job,config,data,cache,gpu,inventory,record,run=run_child,build=child_spec):
    folder=Path(suite)/'calibration/operator'/job
    atomic_json(folder/'config.json',config)
    spec=build('--config',folder/'config.json','--data',data,'--cache',cache,
        '--output',folder,'--device','cuda:0','--acceleration-probe-source',suite,'--operator-profile-only')
    atomic_json(folder/'command.json',spec)
    code=run(spec,folder/'operator-profile.log',{**os.environ,'CUDA_VISIBLE_DEVICES':gpu})
    report=folder/'operator-profile.json'
    if code not in (0,76) or not report.is_file():
        raise ValueError('Operator child did not complete a verified rollback')
    value=read(report)
    if (value.get('schema')!='qkan-operator-profile.v1' or value.get('mode')!='eager'
            or value.get('config_sha256')!=fingerprint(config)
            or value.get('source_revision')!=config['source_revision']
            or value.get('source_checkpoint_sha256')!=record['checkpoint']['sha256']
            or value.get('restored_step')!=record['checkpoint']['step']
            or value.get('rollback_exact') is not True or value.get('samples_committed')!=0
            or value.get('optimizer_updates_committed')!=0 or value.get('sample_clock_reset') is not False
            or value.get('formal_release_allowed') is not False
            or value.get('maximum_updates')!=1 or (code==0)!=(value.get('status')=='collected')):
        raise ValueError('Operator report identity, status or rollback differs')
    return dict(value,job=job,gpu_identity=inventory)


def execute(args,suite):
    from .readiness import policy_check
    from .migration import snapshot,compatible
    from .runner import discover_gpus
    from .torch_acceleration import policy
    source=revision();implementation,review_policy=policy_check()
    plan=snapshot(args.migrate_from,suite,args.cache)
    record=plan['jobs'][args.job]
    if record.get('completion') or not record.get('checkpoint'):
        raise ValueError('Operator observation requires an unfinished full checkpoint')
    config=copy.deepcopy(read(suite/'migration-source'/args.job/'config.json'))
    if not config.get('performance') or not config.get('adaptive_memory',{}).get('logical_tiers'):
        raise ValueError('Operator observation requires the existing dynamic performance checkpoint')
    config.update(source_revision=source,torch_acceleration=policy('eager','eager'))
    compatible(read(suite/'migration-source'/args.job/'config.json'),config)
    data=Path(plan['data'])
    if not data.is_absolute() or not data.is_dir(): raise ValueError('Original local dataset unavailable')
    gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1: raise ValueError('Choose exactly one physical GPU')
    atomic_json(suite/'context.json',dict(source_revision=source,implementation_sha256=implementation,
        policy_sha256=review_policy,migration_sha256=digest(suite/'migration-source/plan.json'),
        purpose='operator_observation_only',formal_release_allowed=False))
    return collect_child(suite,args.job,config,data,args.cache,gpus[0],inventory[gpus[0]],record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--migrate-from',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--gpu',default='0')
    # Bound this diagnostic to the evidenced bottleneck; no cohort-wide sweep.
    parser.add_argument('--job',choices=['transformer-retrieval-s42'],default='transformer-retrieval-s42')
    args=parser.parse_args()
    from .bootstrap import check_exchange
    check_exchange()
    suite=ROOT/'runs'/('operator-profile-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    suite.mkdir(parents=True,exist_ok=False)
    print('Operator-only Suite: '+str(suite),flush=True)
    error=None
    with lock(suite/'suite.lock'):
        try: report=execute(args,suite)
        except BaseException as caught:
            error=caught
            report=dict(schema='qkan-operator-diagnostic-failure.v1',status='failed',
                error_type=type(caught).__name__,formal_release_allowed=False,
                interpretation='No successful rollback declaration; inspect local worker log. Original source remains immutable.')
        atomic_json(suite/'operator-profile.json',report)
        from .return_artifacts import upload_declared
        try: upload_declared(suite,'operator-profile',['operator-profile.json'])
        except Exception:
            print('Return pending; report and delivery manifest retained. Retry upload only: '+str(suite),flush=True)
            if error is None: raise
        if error is not None: raise error
    print('Operator observation: '+report['status']+'; formal training was not started.',flush=True)
    return 0 if report['status']=='collected' else 76


if __name__=='__main__': raise SystemExit(main())
