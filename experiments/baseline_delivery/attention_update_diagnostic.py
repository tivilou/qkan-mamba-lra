"""Isolated optional BF16 real-loss update/recovery/trace/timing diagnostic."""
import argparse
from contextlib import ExitStack
import datetime as dt
import os
from pathlib import Path
import sys
import uuid

from .common import ROOT,atomic_json,digest,fingerprint,lock,read,revision,worker_failure
from .optimization_plugins import resolve
from .attention_update import POLICY,validate_request
from .attention_precision_diagnostic import JOB,bind_source,unchanged

TARGET='experiments.baseline_delivery.attention_update_native'
FILES=('optimization_plugins.py','attention_update.py','attention_update_native.py',
    'attention_update_trace.py','attention_update_diagnostic.py')
DOSSIER=ROOT/'docs/baselines/attention-update-mechanism-recovery.json'
PLAN=ROOT/'docs/baselines/attention-update-plugin-plan.json'


def accept_child(result,request,request_sha,mode,code):
    if (code not in (0,76) or result.get('schema')!='qkan-attention-update-native.v1'
            or result.get('request_sha256')!=request_sha or result.get('mode')!=mode
            or result.get('checkpoint_sha256')!=request['checkpoint']['sha256']
            or result.get('policy_sha256')!=fingerprint(POLICY)
            or result.get('plugin_manifest')!=resolve(request['optimization_config'],scope='attention_update').manifest()
            or result.get('formal_training_started') is not False
            or result.get('training_samples_committed')!=0
            or result.get('rollback')!=dict(full_state_exact=True,committed_samples=0,committed_updates=0)
            or (code==0)!=bool(result.get('passed'))):
        raise ValueError('Native full-update result identity/completeness differs')
    return result


def read_child_result(output,request,request_sha,mode,code,report):
    output=Path(output);result_path=output/'result.json'
    failure_path=output/'worker-error.json'
    if failure_path.is_file(): report['native_failure_locations']=read(failure_path)
    if code not in (0,76) or not result_path.is_file():
        report['native_child_failure']=dict(mode=mode,exit_code=code,
            result_present=result_path.is_file(),failure_locations=report.get('native_failure_locations'))
        raise RuntimeError('BF16 full-update child '+mode+' exited '+str(code)+
            '; inspect native_failure_locations and child_executions in attention-update.json')
    return accept_child(read(result_path),request,request_sha,mode,code)


def recovery_agreement(results):
    a,b,c=(results[name] for name in ('uninterrupted','interrupt','resume'))
    if (not all(row['passed'] for row in (a,b,c))
            or a['full_state_sha256']!=c['full_state_sha256']
            or a['step']!=c['step'] or a['cursor']!=c['cursor']
            or a['schedule_progress']!=c['schedule_progress']
            or a['step']!=b['step']+1 or a['cursor']!=b['cursor']+128
            or a['schedule_progress']!=b['schedule_progress']+128
            or a['updates']!=b['updates']+c['updates']):
        raise ValueError('Fresh-process exact BF16 real-loss recovery differs')
    return dict(passed=True,updates=2,fresh_process=True,full_state_sha256=a['full_state_sha256'],
        scope='Model/gradients/Adam/sample schedule/sampler cursor/permutation/epoch statistics/RNG/precision policy',
        checkpoint_schema='qkan-bf16-diagnostic-checkpoint.v1; never formal training state')


def remove_disposable_checkpoint(folder,result):
    root=Path(folder).resolve();receipt=result['diagnostic_checkpoint']
    if receipt.get('file')!='diagnostic-state.pt': raise ValueError('Unexpected disposable state filename')
    target=root/receipt['file'];sidecar=target.with_suffix('.json')
    if (target.is_symlink() or sidecar.is_symlink() or not target.resolve().is_relative_to(root)
            or target.stat().st_size!=receipt['bytes'] or digest(target)!=receipt['sha256']
            or read(sidecar)!={key:value for key,value in receipt.items() if key!='file'}):
        raise ValueError('Preserve changed diagnostic state instead of deleting it')
    reclaimed=target.stat().st_size+sidecar.stat().st_size
    target.unlink();sidecar.unlink()
    if target.exists() or sidecar.exists(): raise ValueError('Disposable checkpoint cleanup incomplete')
    return dict(removed=True,bytes=reclaimed,scope='Only this diagnostic child checkpoint and matching receipt')


def execute(args,folder,report):
    from .attention_diagnostic import gpu_preflight
    from .runner import discover_gpus
    from .diagnostic_watch import Supervisor
    selection=resolve(read(args.optimizations),scope='attention_update')
    if not selection.bf16_amp: raise ValueError('Explicit BF16 update preset required')
    plan=read(PLAN)
    if (plan['recovery_contract']['dossier_sha256']!=digest(DOSSIER)
            or plan['recovery_contract']['candidate_id']!='bf16-attention-full-update-v1'):
        raise ValueError('Recovery dossier binding differs')
    gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1: raise ValueError('Exactly one physical GPU required')
    gpu=gpus[0];report.update(gpu_identity=inventory[gpu],gpu_preflight=gpu_preflight(inventory[gpu]))
    if not report['gpu_preflight']['idle']: raise ValueError('GPU busy; existing processes left running')
    source=args.source_suite.resolve()
    if source.parent.name!='runs' or not (source/'jobs'/JOB).is_dir(): raise ValueError('Original source Suite missing')
    with ExitStack() as held:
        for path in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/JOB/'worker.lock'):
            held.enter_context(lock(path))
        binding=bind_source(source,args.cache)
        request=dict(schema='qkan-attention-update-input.v1',**binding,source_revision=report['source_revision'],
            optimization_config=read(args.optimizations),optimization_config_sha256=digest(args.optimizations),
            policy=POLICY,dossier_sha256=digest(DOSSIER),plugin_plan_sha256=digest(PLAN))
        validate_request(request,report['source_revision']);request_path=folder/'update-input.json';atomic_json(request_path,request)
        request_sha=digest(request_path);supervisor=Supervisor(args.child_timeout,args.stall_timeout)
        report['child_executions']=supervisor.records;results={}
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',QKAN_DIAGNOSTIC_STAGE_FENCES='0')
        try:
            for mode in ('uninterrupted','interrupt','resume','benchmark'):
                output=folder/mode
                argv=[sys.executable,'-B','-m',TARGET,'--request',str(request_path),'--output',str(output),'--mode',mode]
                if mode=='resume': argv+=['--recovery-state',str(folder/'interrupt/diagnostic-state.pt')]
                print('BF16 full-update child: '+mode,flush=True)
                try: code=supervisor.run(dict(target=TARGET,cwd=str(ROOT),argv=argv),folder/(mode+'.log'),env=env)
                finally:
                    if (output/'worker-error.json').is_file(): report['native_failure_locations']=read(output/'worker-error.json')
                result=read_child_result(output,request,request_sha,mode,code,report)
                results[mode]=result;report['results']=results
                if mode=='resume':
                    report['recovery']=recovery_agreement(results)
                    report['disposable_checkpoint']=remove_disposable_checkpoint(folder/'interrupt',results['interrupt'])
                if code!=0: break
            report['passed']=bool(report.get('recovery',{}).get('passed') and results.get('benchmark',{}).get('passed'))
            report['status']='diagnostic_passed' if report['passed'] else 'diagnostic_rejected'
            report['native_trace_ready']=bool(results.get('benchmark',{}).get('trace',{}).get('passed'))
            report['architecture_status']='native_diagnostic_passed_formal_pending' if report['passed'] else 'approved_pending_native_checks'
        finally:
            unchanged(binding)
            report['original_checkpoint_and_metadata_unchanged']=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-suite',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--gpu',default='0');parser.add_argument('--optimizations',type=Path,required=True)
    parser.add_argument('--child-timeout',type=float,default=900);parser.add_argument('--stall-timeout',type=float,default=300)
    args=parser.parse_args();selected=resolve(read(args.optimizations),scope='attention_update')
    if not selected.bf16_amp: parser.error('Explicit BF16 update preset required')
    from .bootstrap import check_exchange
    check_exchange();source=revision()
    folder=ROOT/'runs'/('attention-update-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(schema='qkan-attention-update.v1',source_revision=source,policy=POLICY,passed=False,status='failed',
        formal_training_started=False,training_samples_committed=0,optimizer_updates_committed=0,
        native_trace_ready=False,architecture_status='approved_pending_native_checks',
        comparison='Matched micro16/logical128 eager; original dropout for recovery/timing; dropout0 numerical/trace/trajectory; no Graph or convergence claim',
        dossier_sha256=digest(DOSSIER),plugin_plan_sha256=digest(PLAN),optimization_config_sha256=digest(args.optimizations),
        plugin_files={name:digest(Path(__file__).parent/name) for name in FILES})
    report['plugin_tree_sha256']=fingerprint(report['plugin_files']);error=None
    print('BF16 full-update diagnostic Suite: '+str(folder),flush=True)
    try: execute(args,folder,report)
    except BaseException as caught:
        error=caught;report.update(passed=False,status='failed',error_type=type(caught).__name__)
        worker_failure(folder,caught);report['failure_locations']=read(folder/'worker-error.json')
    atomic_json(folder/'attention-update.json',report)
    from .return_artifacts import upload_declared
    try: upload_declared(folder,'attention-update',['attention-update.json'])
    except Exception:
        print('Compact report return pending; retry manifest retained: '+str(folder),flush=True)
        if error is None: raise
    if error is not None: raise error
    print('Full-update diagnostic complete; formal queue unchanged. See attention-update.json.',flush=True)
    return 0 if report['passed'] else 76


if __name__=='__main__': raise SystemExit(main())
