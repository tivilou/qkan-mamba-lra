"""Hash-bound real-checkpoint calibration. No framework imports or data export."""
import os
from pathlib import Path

from .common import atomic_json,digest,fingerprint,read
from .torch_acceleration import RecoveryMismatch,allowed


def validate_context(suite,config,source,implementation,policy):
    context=read(Path(suite)/'context.json')
    if (context['source_revision']!=source or config['source_revision']!=source
            or context['implementation_sha256']!=implementation or context['policy_sha256']!=policy
            or context.get('migration_sha256')!=digest(Path(suite)/'migration-source/plan.json')):
        raise ValueError('Real-checkpoint calibration context changed')


def finish_probe(config,state,selected,diagnostics,rollback,runtime):
    """Restore and verify all update state before declaring a zero-commit trial."""
    rollback()  # Must raise on any inexact model/Adam/scheduler/RNG restoration.
    benchmark=selected.get('benchmark') or {}
    passed=(selected['mode']==config['torch_acceleration']['mode']
        and allowed(selected['mode'],config['torch_acceleration'].get('required'))
        and selected.get('benchmarked') is True and selected.get('reason')=='benchmark_gain'
        and (selected['mode']!='compile_graph' or runtime.get('graph_replays',0)>0)
        and 0<benchmark.get('accelerated_seconds',0)<benchmark.get('eager_seconds',0))
    return dict(schema='qkan-real-checkpoint-acceleration.v1',passed=passed,
        config_sha256=fingerprint(config),source_revision=config['source_revision'],
        source_checkpoint_sha256=state['migration']['source_checkpoint_sha256'],restored_step=state['step'],
        samples_committed=0,optimizer_updates_committed=0,sample_clock_reset=False,rollback_exact=True,
        backend=config['torch_acceleration']['backend'],actual=selected,diagnostics=diagnostics,benchmark=benchmark,runtime=runtime)


def calibrate(suite,job_id,config,data,cache,gpu,inventory,record,run,build):
    """One bounded child per profile/device, using an immutable migration snapshot."""
    suite=Path(suite)
    if not record.get('checkpoint') or config['torch_acceleration']['mode']=='eager': return None
    key=fingerprint(dict(config=config,gpu=inventory,checkpoint=record['checkpoint']))[:20]
    folder=suite/'calibration'/key/job_id;report=folder/'acceleration-probe.json'
    if not report.exists():
        atomic_json(folder/'config.json',config)
        spec=build('--config',folder/'config.json','--data',data,'--cache',cache,'--output',folder,
            '--device','cuda:0','--acceleration-probe-source',suite)
        code=run(spec,folder/'calibration.log',{**os.environ,'CUDA_VISIBLE_DEVICES':gpu})
        # A model/data/CUDA-context fault without a completed rollback is fatal.
        if code not in (0,76) or not report.exists():
            raise ValueError('Real-checkpoint calibration did not complete safely: '+job_id)
        value=read(report)
        if (code==0)!=bool(value.get('passed')): raise ValueError('Calibration exit/report disagreement')
        atomic_json(report,dict(value,gpu_identity=inventory))
    value=read(report)
    if (value.get('schema')!='qkan-real-checkpoint-acceleration.v1'
            or value.get('config_sha256')!=fingerprint(config)
            or value.get('source_revision')!=config['source_revision']
            or value.get('source_checkpoint_sha256')!=record['checkpoint']['sha256']
            or value.get('restored_step')!=record['checkpoint']['step']
            or value.get('gpu_identity')!=inventory or value.get('samples_committed')!=0
            or value.get('optimizer_updates_committed')!=0 or value.get('sample_clock_reset') is not False
            or value.get('rollback_exact') is not True or value.get('backend')!=config['torch_acceleration']['backend']):
        raise ValueError('Real-checkpoint calibration identity or rollback differs')
    benchmark=value.get('benchmark') or {};actual=value.get('actual') or {}
    if (not value.get('passed') or actual.get('mode')!=config['torch_acceleration']['mode']
            or actual.get('reason')!='benchmark_gain' or actual.get('benchmarked') is not True
            or (actual.get('mode')=='compile_graph' and value.get('runtime',{}).get('graph_replays',0)<=0)
            or not 0<benchmark.get('accelerated_seconds',0)<benchmark.get('eager_seconds',0)):
        error=RecoveryMismatch('Real-checkpoint numerical/performance calibration rejected '+job_id)
        error.diagnostic=value;raise error
    return report
