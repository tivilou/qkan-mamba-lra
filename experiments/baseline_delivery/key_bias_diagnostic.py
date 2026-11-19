"""Isolated collaborator numerical screen for the analytic K-bias derivative."""
import argparse
from contextlib import ExitStack
import datetime as dt
import math
import os
from pathlib import Path
import sys
import uuid

from .common import ROOT,atomic_json,digest,fingerprint,lock,read,revision,worker_failure
from .key_bias_consistency import CANDIDATE,NAMES,analytic_key_bias,manifest,resolve,validate_observation
from .attention_update import POLICY,numerical_gate,validate_request

TARGET='experiments.baseline_delivery.key_bias_diagnostic'
DOSSIER=ROOT/'docs/baselines/attention-attribution-key-bias-recovery.json'
PLAN=ROOT/'docs/baselines/attention-attribution-key-bias-plan.json'
PRESET=ROOT/'experiments/baseline_delivery/optimization-presets/key-bias-consistency.json'
BF16=ROOT/'experiments/baseline_delivery/optimization-presets/bf16-attention-update.json'
FILES=('key_bias_consistency.py','key_bias_diagnostic.py','attention_update.py','attention_update_native.py',
    'attention_attribution.py','attention_precision_diagnostic.py','attention_bias_postprocess.py','diagnostic_watch.py')
SCHEMA='qkan-key-bias-screen.v1'
BINDING_KEYS=('source_suite','source_config','source_config_sha256','checkpoint','source_metadata_sha256','data','cache')


def bind_evidence(suite,binding):
    from .attention_bias_postprocess import bind
    prior,values,identity=bind(suite);request_path=Path(suite)/'update-input.json';request=read(request_path)
    if (digest(request_path)!=prior['results']['attribution']['request_sha256']
            or any(request.get(k)!=binding[k] for k in BINDING_KEYS)):
        raise ValueError('Original checkpoint/cursor/config/data differs from measured attribution')
    return dict(suite=str(Path(suite).resolve()),**identity),values


def validate_screen(request,source_revision):
    validate_request(request,source_revision)
    plan=read(PLAN)
    if (not resolve(request.get('key_bias_config')) or request.get('key_bias_manifest')!=manifest()
            or request.get('dossier_sha256')!=digest(DOSSIER) or request.get('plugin_plan_sha256')!=digest(PLAN)
            or plan['recovery_contract']['dossier_sha256']!=digest(DOSSIER)
            or plan['recovery_contract']['candidate_id']!=CANDIDATE
            or request.get('plugin_files')!={n:digest(Path(__file__).parent/n) for n in FILES}):
        raise ValueError('Key-bias screen configuration/code/recovery contract differs')
    return True


def compare_unchanged(torch,reference,observed,*,corrected):
    from .attention_attribution import FIELDS
    if (reference['row_ids_sha256']!=observed['row_ids_sha256']
            or reference['loss']!=observed['loss'] or reference['correct']!=observed['correct']
            or not torch.equal(reference['logits'],observed['logits'])
            or reference['optimizer_groups']!=observed['optimizer_groups']
            or any(reference['optimizer_steps_'+when]!=observed['optimizer_steps_'+when] for when in ('before','after'))):
        raise ValueError('Forward/sample/Adam policy differs from saved route')
    if len(reference['gradients'])!=53 or sum(v.numel() for v in reference['gradients'].values())!=807170:
        raise ValueError('Original full parameter population differs')
    for field in FIELDS:
        a,b=reference[field],observed[field]
        if set(a)!=set(b): raise ValueError('Parameter population changed')
        for name in a:
            if not corrected:
                if not torch.equal(a[name],b[name]): raise ValueError('Fresh original FP32 update differs from saved evidence')
            elif field in ('exp_avg_before','exp_avg_sq_before'):
                if not torch.equal(a[name],b[name]): raise ValueError('Original Adam history changed')
            elif name not in NAMES:
                if not torch.equal(a[name],b[name]): raise ValueError('Non-bias gradient/moment/delta changed')
            elif field in ('gradients','clipped_gradients'):
                if (not torch.equal(a[name][:128],b[name][:128]) or not torch.equal(a[name][256:],b[name][256:])
                        or bool((b[name][128:256]!=0).any())):
                    raise ValueError('Q/V gradient changed or K gradient not zero')
            else:
                if not torch.equal(a[name][:128],b[name][:128]) or not torch.equal(a[name][256:],b[name][256:]):
                    raise ValueError('Q/V moment/delta changed')
    return dict(forward_exact=True,sample_identity_exact=True,all53_parameters_retained=True,
        original_adam_groups_steps_and_starting_moments_exact=True,non_k_values_exact=True,
        k_gradient_zero=bool(corrected))


def screen(runtime,previous,output):
    from .attention_attribution import capture_update,compare,scalar_pair,aggregate
    from .diagnostic_watch import mark
    torch=runtime.t;output=Path(output);base_hash=runtime.state_hash(runtime.base)
    pairs={};proofs={};hooks={};callbacks={};result=dict(screen_complete=False,candidate_admitted=False)
    try:
        for route,enabled,corrected in [('fp32',False,False),('fp32_analytic',False,True),('bf16_analytic',True,True)]:
            mark('key_bias_'+route,flush=True)
            runtime.restore(runtime.base)
            if runtime.state_hash(runtime.snapshot())!=base_hash: raise ValueError('Screen starting state differs')
            callbacks[route]={n:[] for n in NAMES}
            def observe(name,raw,changed):
                callbacks[route][name].append(dict(raw=runtime.clone(raw),corrected=runtime.clone(changed)))
            with analytic_key_bias(torch,runtime.model,corrected,observe=observe) as observation:
                pairs[route]=capture_update(runtime,enabled)
            if corrected: validate_observation(observation)
            hooks[route]=observation
            proofs[route]=compare_unchanged(torch,previous['bf16' if enabled else 'fp32'],pairs[route],corrected=corrected)
        metrics={};totals={};key_deltas={}
        for route in ('fp32_analytic','bf16_analytic'):
            metrics[route]=runtime.metrics(pairs['fp32'],pairs[route])
            summary=compare(torch,pairs['fp32'],pairs[route])
            totals[route]={f:v['total'] for f,v in summary['fields'].items()}
            key_deltas[route]={name:scalar_pair(torch,pairs['fp32']['deltas'][name][128:256],
                pairs[route]['deltas'][name][128:256]) for name in NAMES}
        # Save complete actual updates, never a training checkpoint; raw values stay private.
        payload=dict(routes=pairs,callbacks=callbacks)
        path=output/'paired-updates.pt';torch.save(payload,path)
        if path.stat().st_size>128*1024**2:
            path.unlink();raise ValueError('Private screen values exceeded128MiB bound')
        h=digest(path);loaded=torch.load(path,map_location='cpu',weights_only=True)
        if runtime.state_hash(loaded)!=runtime.state_hash(payload) or digest(path)!=h:
            raise ValueError('Reopened private screen values differ')
        callback_summaries={route:{name:aggregate([scalar_pair(torch,v['raw'][128:256],v['corrected'][128:256])
            for v in values]) for name,values in callbacks[route].items()} for route in ('fp32_analytic','bf16_analytic')}
        result.update(screen_complete=True,numerical=metrics,global_fields=totals,key_bias_deltas=key_deltas,
            numerical_passed={route:numerical_gate(v) for route,v in metrics.items()},
            saved_route_agreement=proofs,hook_observations=hooks,raw_key_callback_summaries=callback_summaries,
            private_values=dict(file=path.name,bytes=path.stat().st_size,sha256=h,complete_values_reopened_exact=True),
            routes=['fp32','fp32_analytic','bf16_analytic'],updates_per_route=1,dropout=0,
            interpretation='New gradient policy: analytic zero only in the K-bias null direction; full original reference and all parameters retained. Numeric screen only, not recovery/trace/timing/convergence or formal admission.')
        return result
    finally:
        runtime.restore(runtime.base)
        if runtime.state_hash(runtime.snapshot())!=base_hash: raise ValueError('Full-state screen rollback differs')
        result['rollback']=dict(full_state_exact=True,committed_samples=0,committed_updates=0)
        atomic_json(output/'screen.json',result)


def native(request_path,output):
    from .attention_precision_diagnostic import unchanged,JOB,SOURCE_REVISION
    from .attention_update_native import NativeUpdate
    from .diagnostic_watch import mark
    request=read(request_path);validate_screen(request,revision())
    binding={k:request[k] for k in BINDING_KEYS};source=Path(request['source_suite']);p=request['checkpoint']
    identity,values=bind_evidence(request['evidence_binding']['suite'],binding)
    if (identity!=request['evidence_binding'] or request['source_config'].get('source_revision')!=SOURCE_REVISION
            or read(source/'jobs'/JOB/'config.json')!=request['source_config']
            or p['path']!=str(source/'jobs'/JOB/'checkpoints'/p['file']) or Path(p['path']).is_symlink()):
        raise ValueError('Original source identity differs')
    unchanged(binding)
    import torch
    import torch.nn.attention
    import numpy as np
    if (torch.__version__!='2.7.1+cu126' or np.__version__!='2.2.6'
            or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('Existing pinned CUDA BF16 environment required')
    torch.set_num_threads(1);seed=request['source_config']['seed']
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    fastpath=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
    runtime=None
    try:
        mark('key_bias_source_load',flush=True)
        previous=torch.load(values,map_location='cpu',weights_only=True)
        runtime=NativeUpdate(torch,np,request)
        result=screen(runtime,previous,output)
        if bind_evidence(identity['suite'],binding)[0]!=identity: raise ValueError('Original private evidence changed')
        result.update(schema=SCHEMA,request_sha256=digest(request_path),checkpoint_sha256=p['sha256'],
            checkpoint_step=p['step'],key_bias_manifest=manifest(),formal_training_started=False,
            training_samples_committed=0,native_trace_ready=False,
            environment=dict(torch=str(torch.__version__),numpy=str(np.__version__),deterministic=True,tf32=False))
        atomic_json(Path(output)/'result.json',result)
    finally:
        if runtime is not None:
            runtime.restore(runtime.base)
            if runtime.state_hash(runtime.snapshot())!=runtime.state_hash(runtime.base): raise ValueError('Final rollback differs')
        torch.backends.mha.set_fastpath_enabled(fastpath)
        unchanged(binding)
        if bind_evidence(identity['suite'],binding)[0]!=identity: raise ValueError('Private source evidence changed')


def accept(result,request,request_sha,code):
    if (code!=0 or result.get('schema')!=SCHEMA or result.get('request_sha256')!=request_sha
            or result.get('checkpoint_sha256')!=request['checkpoint']['sha256']
            or result.get('key_bias_manifest')!=manifest() or result.get('screen_complete') is not True
            or result.get('candidate_admitted') is not False or result.get('native_trace_ready') is not False
            or result.get('formal_training_started') is not False or result.get('training_samples_committed')!=0
            or result.get('routes')!=['fp32','fp32_analytic','bf16_analytic'] or result.get('updates_per_route')!=1
            or result.get('rollback')!=dict(full_state_exact=True,committed_samples=0,committed_updates=0)):
        raise ValueError('Native key-bias screen incomplete or identity differs')
    for route in ('fp32_analytic','bf16_analytic'):
        if numerical_gate(result['numerical'][route])!=result['numerical_passed'][route]:
            raise ValueError('Global numerical gate changed')
        validate_observation(result['hook_observations'][route])
        for field,metric in [('gradients','gradient_relative_l2'),('deltas','adam_delta_relative_l2')]:
            total=result['global_fields'][route][field]
            if (total['parameter_count']!=53 or total['numel']!=807170
                    or not math.isclose(total['relative_l2'],result['numerical'][route][metric],rel_tol=1e-10,abs_tol=1e-12)):
                raise ValueError('Full parameter population/global metric reconstruction differs')
        for name in NAMES:
            v=result['raw_key_callback_summaries'][route][name]
            if (v['parameter_count']!=8 or v['numel']!=1024 or v['candidate_l2']!=0 or v['reference_l2']<=0):
                raise ValueError('Raw/corrected callback evidence incomplete')
    if not result['private_values'].get('complete_values_reopened_exact') or not 0<result['private_values']['bytes']<=128*1024**2:
        raise ValueError('Complete private paired values receipt required')
    return result


def execute(args,folder,report):
    from .attention_precision_diagnostic import bind_source,unchanged,JOB
    from .attention_diagnostic import gpu_preflight
    from .runner import discover_gpus
    from .diagnostic_watch import Supervisor
    gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1: raise ValueError('Exactly one physical GPU required')
    gpu=gpus[0];report.update(gpu_identity=inventory[gpu],gpu_preflight=gpu_preflight(inventory[gpu]))
    if not report['gpu_preflight']['idle']: raise ValueError('GPU busy; retain existing work')
    source=args.source_suite.resolve()
    if source.parent.name!='runs' or not (source/'jobs'/JOB).is_dir(): raise ValueError('Original source Suite missing')
    with ExitStack() as held:
        for path in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/JOB/'worker.lock'):
            held.enter_context(lock(path))
        binding=bind_source(source,args.cache);identity,_=bind_evidence(args.attribution_from,binding)
        request=dict(schema='qkan-attention-update-input.v1',**binding,source_revision=report['source_revision'],
            policy=POLICY,optimization_config=read(BF16),optimization_config_sha256=digest(BF16),
            key_bias_config=read(args.optimizations),key_bias_manifest=manifest(),evidence_binding=identity,
            dossier_sha256=digest(DOSSIER),plugin_plan_sha256=digest(PLAN),plugin_files=report['plugin_files'])
        validate_screen(request,report['source_revision'])
        path=folder/'screen-input.json';atomic_json(path,request);report['evidence_binding']=identity
        supervisor=Supervisor(args.child_timeout,args.stall_timeout);report['child_executions']=supervisor.records
        output=folder/'screen';output.mkdir()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',QKAN_DIAGNOSTIC_STAGE_FENCES='0')
        try:
            code=supervisor.run(dict(target=TARGET,cwd=str(ROOT),argv=[sys.executable,'-B','-m',TARGET,
                '--worker','--request',str(path),'--output',str(output)]),folder/'screen.log',env)
            error=output/'worker-error.json'
            if error.is_file(): report['native_failure_locations']=read(error)
            if not (output/'result.json').is_file(): raise RuntimeError('Native screen result missing; inspect retained child diagnostics')
            report['result']=accept(read(output/'result.json'),request,digest(path),code)
            report.update(screen_complete=True,numerical_passed=report['result']['numerical_passed'],
                status='screen_complete',candidate_admitted=False,native_trace_ready=False)
        finally:
            unchanged(binding)
            if bind_evidence(args.attribution_from,binding)[0]!=identity: raise ValueError('Private input evidence changed')
            report['original_checkpoint_and_metadata_unchanged']=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-suite',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--gpu',default='0')
    p.add_argument('--attribution-from',type=Path);p.add_argument('--optimizations',type=Path)
    p.add_argument('--child-timeout',type=float,default=900);p.add_argument('--stall-timeout',type=float,default=300)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--request',type=Path,help=argparse.SUPPRESS);p.add_argument('--output',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        if args.request is None or args.output is None: p.error('Worker request and output required')
        try: native(args.request,args.output)
        except BaseException as error: worker_failure(args.output,error);raise
        return 0
    if any(getattr(args,k) is None for k in ('source_suite','cache','attribution_from','optimizations')):
        p.error('Source Suite,cache,attribution-from and explicit optimizations required')
    if not resolve(read(args.optimizations)): p.error('Explicit enabled numerical policy required')
    from .bootstrap import check_exchange
    from .diagnostic_watch import Supervisor
    from .return_artifacts import upload_declared
    Supervisor(args.child_timeout,args.stall_timeout);check_exchange();source=revision()
    folder=ROOT/'runs'/('key-bias-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,source_revision=source,status='failed',screen_complete=False,candidate_admitted=False,
        formal_training_started=False,training_samples_committed=0,native_trace_ready=False,
        key_bias_manifest=manifest(),dossier_sha256=digest(DOSSIER),plugin_plan_sha256=digest(PLAN),
        optimization_config_sha256=digest(args.optimizations),plugin_files={n:digest(Path(__file__).parent/n) for n in FILES})
    report['plugin_tree_sha256']=fingerprint(report['plugin_files']);error=None
    print('Key-bias numerical screen Suite: '+str(folder),flush=True)
    try: execute(args,folder,report)
    except BaseException as caught:
        error=caught;report['error_type']=type(caught).__name__;worker_failure(folder,caught)
        report['failure_locations']=read(folder/'worker-error.json')
    atomic_json(folder/'key-bias.json',report)
    try: upload_declared(folder,'key-bias',['key-bias.json'])
    except Exception:
        print('Return pending; report and artifact-delivery.json retained: '+str(folder),flush=True)
        if error is None: raise
    if error is not None: raise error
    print('Numerical screen complete: '+str(report['numerical_passed'])+'; candidate admission remains closed. See key-bias.json.',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
