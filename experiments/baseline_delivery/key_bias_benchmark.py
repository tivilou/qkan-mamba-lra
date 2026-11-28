"""Isolated post-preflight trajectory/profile/full-update timing; no training entry."""
import argparse
from contextlib import ExitStack
import datetime as dt
import math
import os
from pathlib import Path
import sys
import time
import uuid
from . import key_bias_recovery as recovery
from .common import ROOT,atomic_json,digest,fingerprint,read,revision,lock,worker_failure
from .key_bias_consistency import analytic_key_bias,validate_observation,manifest
from .attention_update import POLICY,timing_summary

TARGET='experiments.baseline_delivery.key_bias_benchmark'
SCHEMA='qkan-key-bias-benchmark.v1'
PLAN=ROOT/'docs/baselines/attention-key-bias-benchmark-plan.json'
REVIEW=ROOT/'docs/evidence/key-bias-recovery-native-review-20261116.json'
ROUTES=('fp32','fp32_analytic','bf16_analytic')
ORDER=('fp32','bf16_analytic','bf16_analytic','fp32')*2


def binding():
    return dict(module_sha256=digest(Path(__file__)),plan_sha256=digest(PLAN),review_sha256=digest(REVIEW))


def bind_preflight(folder,source):
    folder=Path(folder).resolve();path=folder/'key-bias-recovery.json';review=read(REVIEW)
    if path.is_symlink() or digest(path)!=review['source_report']['sha256'] or path.stat().st_size!=review['source_report']['bytes']:
        raise ValueError('Reviewed immutable native preflight required')
    r=read(path)
    if (r['source_revision']!=review['source_revision'] or not r['passed'] or not r['native_trace_ready']
            or r['candidate_admitted'] is not False or r['results']['trace']['trace']!=review['trace']
            or recovery.agreement(r['results'])!=review['recovery']
            or r['screen_binding']!=review['source_binding']
            or r['screen_binding']!=recovery.bind_screen(source['screen_binding']['suite'],source)):
        raise ValueError('Native preflight/source/trace differs')
    return dict(suite=str(folder),sha256=digest(path),review_sha256=digest(REVIEW))


def update(runtime,route,**kwargs):
    if route not in ROUTES: raise ValueError('Unknown route')
    with analytic_key_bias(runtime.t,runtime.model,route!='fp32') as observed:
        value=runtime.update(route=='bf16_analytic',**kwargs)
    if route!='fp32': validate_observation(observed)
    return value


def trajectory_gate(values):
    reference=values['fp32'];differences={}
    for route in ROUTES:
        rows=values[route]
        if len(rows)!=4 or [v['row_ids_sha256'] for v in rows]!=[v['row_ids_sha256'] for v in reference]:
            raise ValueError('Four matched trajectory batches required')
        losses=[v['loss'] for v in rows]
        if any(not math.isfinite(v) for v in losses): raise ValueError('Nonfinite trajectory loss')
        differences[route]=[abs(a['loss']-b['loss']) for a,b in zip(reference,rows)]
    return dict(passed=all(max(v)<=POLICY['trajectory_loss_absolute'] for v in differences.values()),
        loss_absolute_limit=POLICY['trajectory_loss_absolute'],loss_absolute_differences=differences,
        updates_per_route=4,dropout=0,convergence_certified=False)


def kernel_gate(names):
    forward=any('_scaled_dot_product_efficient_attention' in n and 'backward' not in n for n in names)
    backward=any('_scaled_dot_product_efficient_attention_backward' in n for n in names)
    other=any('_scaled_dot_product_' in n and any(k in n for k in ('flash','cudnn','math')) for n in names)
    if not forward or not backward or other: raise ValueError('Efficient SDPA forward/backward proof required')
    return dict(efficient_forward=True,efficient_backward=True,other_sdpa=False,
        scope='ATen profiler operators, not exported CUDA kernel names')


def benchmark(runtime,output):
    from .diagnostic_watch import mark
    t=runtime.t;report=dict(passed=False,schema=SCHEMA,stage='trajectory',policy=read(PLAN))
    original_hash=runtime.state_hash(runtime.base)
    try:
        values={}
        for route in ROUTES:
            runtime.restore(runtime.base);values[route]=[]
            for index in range(4):
                mark('key_bias_trajectory_'+route+'_'+str(index),flush=True)
                values[route].append(update(runtime,route,dropout_off=True))
        report.update(trajectory=trajectory_gate(values),trajectory_updates=values)
        if not report['trajectory']['passed']: return report
        report['stage']='profile';proof={};memory={}
        for route in ('fp32','bf16_analytic'):
            runtime.restore(runtime.base);t.cuda.synchronize();t.cuda.reset_peak_memory_stats()
            start=t.cuda.memory_allocated();mark('key_bias_profile_'+route,flush=True)
            with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,t.profiler.ProfilerActivity.CUDA]) as profile:
                value=update(runtime,route)
            proof[route]=dict(**kernel_gate([e.key for e in profile.key_averages()]),attention=value.get('attention'))
            memory[route]=dict(start_allocated=start,peak_allocated=t.cuda.max_memory_allocated(),
                peak_reserved=t.cuda.max_memory_reserved(),scope='Shared allocator full-update peak; not isolated minimum')
            del profile
        report.update(kernel_proof=proof,memory=memory,stage='warmup')
        for route in ('fp32','bf16_analytic'):
            runtime.restore(runtime.base)
            for index in range(2):
                mark('key_bias_warm_'+route+'_'+str(index),flush=True);update(runtime,route)
        samples={'fp32':[],'bf16':[]};records=[];report['stage']='timing'
        for index,route in enumerate(ORDER):
            runtime.restore(runtime.base);t.cuda.synchronize()
            mark('key_bias_timing_'+str(index)+'_'+route,flush=True)
            start=time.perf_counter();value=update(runtime,route);t.cuda.synchronize();elapsed=time.perf_counter()-start
            samples['fp32' if route=='fp32' else 'bf16'].append(elapsed)
            records.append(dict(route=route,seconds=elapsed,update=value,
                full_state_sha256=runtime.state_hash(runtime.snapshot())))
        if len({v['update']['row_ids_sha256'] for v in records})!=1: raise ValueError('Timed batch identity differs')
        for route in ('fp32','bf16_analytic'):
            rows=[v['update'] for v in records if v['route']==route]
            if any(v!=rows[0] for v in rows): raise ValueError('Timed same-route replay differs')
            if len({v['full_state_sha256'] for v in records if v['route']==route})!=1:
                raise ValueError('Timed full-state replay differs')
        summary=timing_summary(samples)
        paired=[records[i]['seconds']/records[i+1]['seconds']-1 if records[i]['route']=='fp32'
            else records[i+1]['seconds']/records[i]['seconds']-1 for i in range(0,8,2)]
        report.update(timing=summary,timing_records=records,paired_throughput_gain=paired,
            timing_scope='Original dropout; gather/H2D/forward/CE/backward/gradient hook/clip/Adam/schedule/finite checks; restore/warm/profile excluded',
            stage='complete',passed=summary['gain_passed'] and all(v>0 for v in paired))
        return report
    finally:
        runtime.restore(runtime.base)
        if runtime.state_hash(runtime.snapshot())!=original_hash: raise ValueError('Full-state rollback differs')
        report['rollback']=dict(recovery.ROLLBACK);atomic_json(Path(output)/'benchmark.json',report)


def execute(args,folder,report):
    from .attention_precision_diagnostic import bind_source,unchanged,JOB
    from .attention_diagnostic import gpu_preflight
    from .runner import discover_gpus
    from .diagnostic_watch import Supervisor
    gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1: raise ValueError('One GPU required')
    gpu=gpus[0];report.update(gpu_identity=inventory[gpu],gpu_preflight=gpu_preflight(inventory[gpu]))
    if not report['gpu_preflight']['idle']: raise ValueError('GPU busy; preserve current work')
    with ExitStack() as held:
        source=args.source_suite.resolve()
        for path in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/JOB/'worker.lock'): held.enter_context(lock(path))
        source_binding=bind_source(source,args.cache);screen=recovery.bind_screen(args.screen_from,source_binding)
        preflight=bind_preflight(args.preflight_from,dict(**source_binding,screen_binding=screen))
        request=dict(schema='qkan-attention-update-input.v1',**source_binding,source_revision=revision(),policy=POLICY,
            optimization_config=read(recovery.BF16),optimization_config_sha256=digest(recovery.BF16),
            key_bias_config=read(recovery.PRESET),key_bias_config_sha256=digest(recovery.PRESET),
            key_bias_manifest=manifest(),screen_binding=screen,dossier_sha256=digest(recovery.DOSSIER),
            plugin_plan_sha256=digest(recovery.PLAN),plugin_files=recovery.code_binding(),
            benchmark_binding=binding(),preflight_binding=preflight)
        path=folder/'benchmark-input.json';atomic_json(path,request);report.update(preflight_binding=preflight,screen_binding=screen)
        supervisor=Supervisor(args.child_timeout,args.stall_timeout);report['child_executions']=supervisor.records
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',QKAN_DIAGNOSTIC_STAGE_FENCES='0')
        output=folder/'benchmark'
        try:
            code=supervisor.run(dict(target=TARGET,cwd=str(ROOT),argv=[sys.executable,'-B','-m',TARGET,
                '--worker','--request',str(path),'--output',str(output)]),folder/'benchmark.log',env)
            if (output/'benchmark.json').is_file(): report['partial_benchmark']=read(output/'benchmark.json')
            if (output/'worker-error.json').is_file(): report['native_failure_locations']=read(output/'worker-error.json')
            if code!=0 or not (output/'result.json').is_file(): raise RuntimeError('Benchmark child failed; inspect compact diagnostics')
            result=read(output/'result.json')
            if (result['schema']!=recovery.SCHEMA or result['mode']!='benchmark' or result['request_sha256']!=digest(path)
                    or result['checkpoint_sha256']!=request['checkpoint']['sha256']
                    or result['plugin_tree_sha256']!=fingerprint(request['plugin_files'])
                    or result['rollback']!=recovery.ROLLBACK or result['candidate_admitted'] is not False
                    or result['formal_training_started'] is not False or result['training_samples_committed']!=0):
                raise ValueError('Native benchmark identity/rollback differs')
            report.update(result=result,passed=result['passed'],status='passed' if result['passed'] else 'gate_rejected')
        finally:
            unchanged(source_binding)
            if bind_preflight(args.preflight_from,dict(**source_binding,screen_binding=screen))!=preflight: raise ValueError('Prior preflight changed')
            report['original_checkpoint_and_metadata_unchanged']=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-suite','cache','screen-from','preflight-from','request','output'): p.add_argument('--'+name,type=Path)
    p.add_argument('--gpu',default='0');p.add_argument('--child-timeout',type=float,default=900)
    p.add_argument('--stall-timeout',type=float,default=300);p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.worker:
        if args.request is None or args.output is None: p.error('Request/output required')
        recovery.prepare_worker_output(args.output)
        try:
            request=read(args.request)
            if request.get('benchmark_binding')!=binding(): raise ValueError('Benchmark code/plan/review differs')
            if bind_preflight(request['preflight_binding']['suite'],request)!=request['preflight_binding']: raise ValueError('Preflight differs')
            recovery.native(args.request,args.output,'benchmark',operation=benchmark)
        except BaseException as error: worker_failure(args.output,error);raise
        return 0
    if any(getattr(args,n) is None for n in ('source_suite','cache','screen_from','preflight_from')): p.error('Source/cache/completed screen/preflight required')
    from .bootstrap import check_exchange
    from .return_artifacts import upload_declared
    from .diagnostic_watch import Supervisor
    Supervisor(args.child_timeout,args.stall_timeout);check_exchange()
    folder=ROOT/'runs'/('key-bias-benchmark-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
    report=dict(schema=SCHEMA,source_revision=revision(),benchmark_binding=binding(),plugin_files=recovery.code_binding(),
        status='failed',passed=False,candidate_admitted=False,formal_training_started=False,training_samples_committed=0)
    print('Analytic K-bias benchmark Suite: '+str(folder),flush=True);error=None
    try: execute(args,folder,report)
    except BaseException as caught:
        error=caught;worker_failure(folder,caught);report['failure_locations']=read(folder/'worker-error.json')
    atomic_json(folder/'key-bias-benchmark.json',report)
    try: upload_declared(folder,'key-bias-benchmark',['key-bias-benchmark.json'])
    except Exception:
        print('Return pending; report and retry manifest retained: '+str(folder),flush=True)
        if error is None: raise
    if error is not None: raise error
    print('Benchmark complete; formal admission remains closed. See key-bias-benchmark.json.',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
