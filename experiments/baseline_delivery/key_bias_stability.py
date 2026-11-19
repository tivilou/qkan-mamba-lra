"""64step train-only pilot and fresh matched incumbent comparisons; formal off."""
import argparse
from contextlib import ExitStack
import datetime as dt
import os
from pathlib import Path
import sys
import uuid
from . import key_bias_recovery as recovery,key_bias_benchmark as benchmark,key_bias_stability_native as operations
from .common import ROOT,atomic_json,digest,fingerprint,read,revision,lock,worker_failure
from .attention_update import POLICY

TARGET='experiments.baseline_delivery.key_bias_stability'
SCHEMA='qkan-key-bias-stability.v1'
PLAN=ROOT/'docs/baselines/attention-key-bias-stability-plan.json'
REVIEW=ROOT/'docs/evidence/key-bias-benchmark-native-review-20261116.json'
REUSE_REVIEW=ROOT/'docs/evidence/key-bias-stability-partial-review-20261117.json'


def bind_reuse(folder,request):
    """Reuse only the six authenticated, reviewed children; never an unfinished child."""
    folder=Path(folder).resolve();path=folder/'key-bias-stability.json';input_path=folder/'stability-input.json'
    review=read(REUSE_REVIEW)
    if (path.is_symlink() or input_path.is_symlink() or digest(path)!=review['source_report']['sha256']
            or path.stat().st_size!=review['source_report']['bytes'] or digest(input_path)!=review['request_sha256']):
        raise ValueError('Reviewed immutable partial pilot/request required')
    prior=read(path);old=read(input_path)
    # Only the diagnostic launcher/observation code changed. The executed protocol,
    # source, native model/update/gradient policy and incumbent engines stay pinned.
    same=(*recovery.BINDING_KEYS,'policy','optimization_config','optimization_config_sha256',
        'key_bias_config','key_bias_config_sha256','key_bias_manifest','screen_binding','dossier_sha256',
        'plugin_plan_sha256','plugin_files','benchmark_evidence')
    old_binding=review['pilot_binding'];current=request['pilot_binding']
    if (prior['source_revision']!=review['source_revision'] or old['source_revision']!=review['source_revision']
            or old['pilot_binding']!=old_binding or prior['pilot_binding']!=old_binding
            or prior['plugin_files']!=request['plugin_files'] or any(old.get(k)!=request.get(k) for k in same)
            or any(current[k]!=old_binding[k] for k in ('plan_sha256','review_sha256'))
            or set(current['files'])!=set(old_binding['files'])
            or any(current['files'][k]!=v for k,v in old_binding['files'].items()
                   if k not in ('key_bias_stability.py','key_bias_stability_native.py'))):
        raise ValueError('Reviewed prefix/source/unchanged implementation differs')
    if (set(prior['results'])!=set(operations.MODES[:-1]) or prior['passed'] is not False
            or prior['candidate_admitted'] is not False or prior['formal_training_started'] is not False
            or prior['training_samples_committed']!=0 or prior['original_checkpoint_and_metadata_unchanged'] is not True):
        raise ValueError('Expected closed partial pilot with exactly six completed children')
    for mode,value in prior['results'].items():
        child_accept(value,old,review['request_sha256'],mode)
        if value['passed'] is not True: raise ValueError('Only successful prefix children may be reused')
    gate=operations.gate(prior['results']);agreement=operations.agreement(prior['results'])
    graph=prior['results']['graph16']
    timing=operations.compare_summary(graph['timing_records'])
    from .partition_comparison import require_steady
    for row in graph['timing_records']:
        if row['route']=='incumbent':
            require_steady(row['steady_before'])
            if row['steady_before']!=row['steady_after']: raise ValueError('Prior capture changed inside timing')
    if (not gate['passed'] or gate!=prior['stability_gate'] or agreement!=prior['recovery']
            or timing!=graph['timing'] or timing['passed'] is not True or not prior['disposable_checkpoint']['removed']):
        raise ValueError('Reviewed prefix gates/recovery/Graph16 proof differs')
    return prior,dict(suite=str(folder),sha256=digest(path),request_sha256=digest(input_path),
        review_sha256=digest(REUSE_REVIEW),reused_modes=list(operations.MODES[:-1]),rerun_modes=['partition32'])


def binding():
    files={n:digest(Path(__file__).parent/n) for n in ('key_bias_stability.py','key_bias_stability_native.py',
        'key_bias_benchmark.py','torch_acceleration.py','parallel_attention.py','partitioned_attention.py','partition_comparison.py')}
    return dict(files=files,plan_sha256=digest(PLAN),review_sha256=digest(REVIEW),reuse_review_sha256=digest(REUSE_REVIEW))


def bind_benchmark(folder,source):
    folder=Path(folder).resolve();path=folder/'key-bias-benchmark.json';review=read(REVIEW)
    if path.is_symlink() or digest(path)!=review['source_report']['sha256'] or path.stat().st_size!=review['source_report']['bytes']:
        raise ValueError('Reviewed immutable benchmark required')
    report=read(path)
    if (report['source_revision']!=review['source_revision'] or report['passed'] is not True
            or report['candidate_admitted'] is not False or report['screen_binding']!=review['screen_binding']
            or report['result']['timing']!=review['timing'] or report['result']['trajectory']!=review['trajectory']
            or report['screen_binding']!=recovery.bind_screen(source['screen_binding']['suite'],source)
            or report['preflight_binding']!=benchmark.bind_preflight(report['preflight_binding']['suite'],source)):
        raise ValueError('Prior benchmark/source differs')
    return dict(suite=str(folder),sha256=digest(path),review_sha256=digest(REVIEW))


def child_accept(value,request,request_sha,mode):
    if (value.get('schema')!=recovery.SCHEMA or value.get('mode')!=mode or value.get('request_sha256')!=request_sha
            or value.get('key_bias_manifest')!=recovery.manifest() or value.get('checkpoint_sha256')!=request['checkpoint']['sha256']
            or value.get('checkpoint_step')!=request['checkpoint']['step'] or value.get('plugin_tree_sha256')!=fingerprint(request['plugin_files'])
            or value.get('rollback')!=recovery.ROLLBACK or value.get('candidate_admitted') is not False
            or value.get('formal_training_started') is not False or value.get('training_samples_committed')!=0):
        raise ValueError('Pilot child identity/rollback differs')
    return value


def execute(args,folder,report):
    from .attention_precision_diagnostic import bind_source,unchanged,JOB
    from .attention_diagnostic import gpu_preflight
    from .runner import discover_gpus
    from .diagnostic_watch import Supervisor
    from .attention_update_diagnostic import remove_disposable_checkpoint
    gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1: raise ValueError('One GPU required')
    gpu=gpus[0];report.update(gpu_identity=inventory[gpu],gpu_preflight=gpu_preflight(inventory[gpu]))
    if not report['gpu_preflight']['idle']: raise ValueError('GPU busy; preserve work')
    with ExitStack() as held:
        source=args.source_suite.resolve()
        for path in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/JOB/'worker.lock'): held.enter_context(lock(path))
        original=bind_source(source,args.cache);screen=recovery.bind_screen(args.screen_from,original)
        prior=bind_benchmark(args.benchmark_from,dict(**original,screen_binding=screen))
        request=dict(schema='qkan-attention-update-input.v1',**original,source_revision=revision(),policy=POLICY,
            optimization_config=read(recovery.BF16),optimization_config_sha256=digest(recovery.BF16),
            key_bias_config=read(recovery.PRESET),key_bias_config_sha256=digest(recovery.PRESET),key_bias_manifest=recovery.manifest(),
            screen_binding=screen,dossier_sha256=digest(recovery.DOSSIER),plugin_plan_sha256=digest(recovery.PLAN),
            plugin_files=recovery.code_binding(),pilot_binding=binding(),benchmark_evidence=prior)
        reused=None
        if getattr(args,'reuse_from',None) is not None:
            reused,request['reuse_evidence']=bind_reuse(args.reuse_from,request)
        path=folder/'stability-input.json';atomic_json(path,request)
        supervisor=Supervisor(args.child_timeout,args.stall_timeout);report.update(child_executions=supervisor.records,
            benchmark_evidence=prior,screen_binding=screen,results={})
        modes=operations.MODES
        if reused is not None:
            for key in ('results','stability_gate','recovery','disposable_checkpoint'):
                report[key]=reused[key]
            report.update(reuse_evidence=request['reuse_evidence'],reused_child_executions=reused['child_executions'][:6])
            modes=('partition32',)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',QKAN_DIAGNOSTIC_STAGE_FENCES='0')
        try:
            for mode in modes:
                output=folder/mode;print('Stability child: '+mode,flush=True)
                code=supervisor.run(dict(target=TARGET,cwd=str(ROOT),argv=[sys.executable,'-B','-m',TARGET,
                    '--worker','--request',str(path),'--output',str(output),'--mode',mode]),folder/(mode+'.log'),env)
                for name in ('stability.json','comparison.json','worker-error.json'):
                    if (output/name).is_file(): report.setdefault('partial_children',{}).setdefault(mode,{})[name]=read(output/name)
                if code!=0 or not (output/'result.json').is_file(): raise RuntimeError('Pilot child failed; inspect partial diagnostics')
                value=child_accept(read(output/'result.json'),request,digest(path),mode);report['results'][mode]=value
                report.get('partial_children',{}).pop(mode,None)
                if not value['passed'] and mode not in ('graph16','partition32'):
                    report.update(status='gate_rejected');return
                if mode=='bf16_analytic':
                    report['stability_gate']=operations.gate(report['results'])
                    if not report['stability_gate']['passed']: report.update(status='gate_rejected');return
                if mode=='resume':
                    report['recovery']=operations.agreement(report['results'])
                    receipt=dict(report['results']['interrupt']['diagnostic_checkpoint']);receipt.pop('complete_values_reopened_exact')
                    report['disposable_checkpoint']=remove_disposable_checkpoint(folder/'interrupt',dict(diagnostic_checkpoint=receipt))
            passed=all(v['passed'] for v in report['results'].values())
            report.update(passed=passed,status='pilot_passed_formal_pending' if passed else 'gate_rejected')
        finally:
            unchanged(original)
            if bind_benchmark(args.benchmark_from,dict(**original,screen_binding=screen))!=prior: raise ValueError('Prior evidence changed')
            if reused is not None and bind_reuse(args.reuse_from,request)[1]!=request['reuse_evidence']:
                raise ValueError('Reused evidence changed')
            report['original_checkpoint_and_metadata_unchanged']=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source-suite','cache','screen-from','benchmark-from','reuse-from','request','output'): p.add_argument('--'+n,type=Path)
    p.add_argument('--gpu',default='0');p.add_argument('--worker',action='store_true');p.add_argument('--mode',choices=operations.MODES)
    p.add_argument('--child-timeout',type=float,default=900);p.add_argument('--stall-timeout',type=float,default=300);args=p.parse_args()
    if args.worker:
        if any(getattr(args,n) is None for n in ('request','output','mode')): p.error('Request/output/mode required')
        recovery.prepare_worker_output(args.output)
        try:
            request=read(args.request)
            if request.get('pilot_binding')!=binding(): raise ValueError('Pilot implementation/plan/review differs')
            if bind_benchmark(request['benchmark_evidence']['suite'],request)!=request['benchmark_evidence']: raise ValueError('Prior evidence differs')
            if 'reuse_evidence' in request:
                if args.mode!='partition32' or bind_reuse(request['reuse_evidence']['suite'],request)[1]!=request['reuse_evidence']:
                    raise ValueError('Retry must bind the reviewed prefix and run partition32 only')
            def run(runtime,output):
                if args.mode in ('graph16','partition32'): return operations.comparison(runtime,output,args.mode)
                reference=None
                if args.mode in ('fp32_analytic','bf16_analytic'):
                    path=Path(output).parent/'fp32/result.json'
                    if path.is_symlink(): raise ValueError('Reference child must be local owned output')
                    original=child_accept(read(path),request,digest(args.request),'fp32')
                    if original['passed'] is not True or len(original['updates'])!=64: raise ValueError('Completed reference64 required')
                    reference=original['updates']
                return operations.stability(runtime,output,args.mode,digest(args.request),reference=reference)
            recovery.native(args.request,args.output,args.mode,operation=run)
        except BaseException as error: worker_failure(args.output,error);raise
        return 0
    if any(getattr(args,n) is None for n in ('source_suite','cache','screen_from','benchmark_from')): p.error('Original source/cache/completed screen/benchmark required')
    from .bootstrap import check_exchange
    from .return_artifacts import upload_declared
    from .diagnostic_watch import Supervisor
    Supervisor(args.child_timeout,args.stall_timeout);check_exchange()
    folder=ROOT/'runs'/('key-bias-stability-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
    report=dict(schema=SCHEMA,source_revision=revision(),pilot_binding=binding(),plugin_files=recovery.code_binding(),
        status='failed',passed=False,candidate_admitted=False,formal_training_started=False,training_samples_committed=0)
    error=None;print('Stability Suite: '+str(folder),flush=True)
    try: execute(args,folder,report)
    except BaseException as caught:
        error=caught;worker_failure(folder,caught);report['failure_locations']=read(folder/'worker-error.json')
    atomic_json(folder/'key-bias-stability.json',report)
    try: upload_declared(folder,'key-bias-stability',['key-bias-stability.json'])
    except Exception:
        print('Return pending;report and retry manifest retained: '+str(folder),flush=True)
        if error is None: raise
    if error is not None: raise error
    print('Pilot complete;formal admission remains closed.See key-bias-stability.json.',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
