"""Isolated analytic-K fresh-process recovery and paired trace preflight.

Native frameworks are imported only in supervised collaborator children.
"""
import argparse
from contextlib import ExitStack
import datetime as dt
import os
from pathlib import Path
import sys
import uuid

from .common import ROOT,atomic_json,digest,fingerprint,lock,read,revision,worker_failure
from .attention_update import POLICY,validate_request
from .key_bias_consistency import analytic_key_bias,manifest,resolve,validate_observation

TARGET='experiments.baseline_delivery.key_bias_recovery'
SCHEMA='qkan-key-bias-recovery.v1'
STATE_SCHEMA='qkan-key-bias-private-state.v1'
PLAN=ROOT/'docs/baselines/attention-attribution-key-bias-recovery-plan.json'
DOSSIER=ROOT/'docs/baselines/attention-attribution-key-bias-recovery.json'
PRESET=ROOT/'experiments/baseline_delivery/optimization-presets/key-bias-consistency.json'
BF16=ROOT/'experiments/baseline_delivery/optimization-presets/bf16-attention-update.json'
REVIEW=ROOT/'docs/evidence/key-bias-native-review-20261116.json'
FILES=('key_bias_recovery.py','key_bias_trace.py','key_bias_consistency.py','key_bias_diagnostic.py',
    'attention_update.py','attention_update_native.py','attention_update_trace.py','attention_attribution.py',
    'attention_precision_diagnostic.py','attention_bias_postprocess.py','diagnostic_watch.py','trace.py')
MODES=('uninterrupted','interrupt','resume','trace')
ROLLBACK=dict(full_state_exact=True,committed_samples=0,committed_updates=0)
BINDING_KEYS=('source_suite','source_config','source_config_sha256','checkpoint','source_metadata_sha256','data','cache')


def code_binding():
    values={n:digest(Path(__file__).parent/n) for n in FILES}
    for name in ('tools/sample_trace_contract/validate.py','tools/sample_trace_contract/schema.json'):
        if (ROOT/name).is_file(): values[name]=digest(ROOT/name)
    return values


def bind_screen(folder,binding):
    from .key_bias_diagnostic import bind_evidence
    folder=Path(folder).resolve();report_path=folder/'key-bias.json';input_path=folder/'screen-input.json'
    review=read(REVIEW)
    if (report_path.is_symlink() or input_path.is_symlink()
            or digest(report_path)!=review['source_report']['sha256']
            or report_path.stat().st_size!=review['source_report']['bytes']):
        raise ValueError('Reviewed immutable numerical screen required')
    report=read(report_path);request=read(input_path)
    if (report['source_revision']!=review['source_revision'] or not report['screen_complete']
            or report['candidate_admitted'] is not False
            or report['numerical_passed']!=dict(fp32_analytic=True,bf16_analytic=True)
            or report['result']['request_sha256']!=digest(input_path)
            or any(request.get(k)!=binding[k] for k in BINDING_KEYS)):
        raise ValueError('Numerical screen/source binding differs')
    evidence,_=bind_evidence(report['evidence_binding']['suite'],binding)
    if evidence!=report['evidence_binding']: raise ValueError('Original attribution changed')
    return dict(suite=str(folder),report_sha256=digest(report_path),request_sha256=digest(input_path),
        review_sha256=digest(REVIEW),attribution=evidence)


def validate(request,source_revision):
    validate_request(request,source_revision)
    plan=read(PLAN)
    if (not resolve(request.get('key_bias_config')) or request.get('key_bias_manifest')!=manifest()
            or request.get('dossier_sha256')!=digest(DOSSIER) or request.get('plugin_plan_sha256')!=digest(PLAN)
            or plan['recovery_contract']['dossier_sha256']!=digest(DOSSIER)
            or plan['recovery_contract']['candidate_id']!=manifest()['candidate_id']
            or request.get('plugin_files')!=code_binding()
            or request.get('key_bias_config_sha256')!=digest(PRESET)
            or request.get('optimization_config_sha256')!=digest(BF16)):
        raise ValueError('Recovery strategy/config/code/plan differs')
    return True


def state_envelope(runtime,request_sha):
    state=runtime.snapshot()
    return dict(schema=STATE_SCHEMA,strategy=manifest(),request_sha256=request_sha,
        source_checkpoint_sha256=runtime.request['checkpoint']['sha256'],
        baseline_state_sha256=runtime.state_hash(runtime.base),state=state,state_sha256=runtime.state_hash(state))


def save_state(runtime,path,request_sha):
    path=Path(path);payload=state_envelope(runtime,request_sha);runtime.t.save(payload,path)
    if path.stat().st_size>128*1024**2:
        path.unlink();raise ValueError('Private recovery state exceeded128MiB')
    receipt=dict(schema=STATE_SCHEMA,sha256=digest(path),bytes=path.stat().st_size,
        state_sha256=payload['state_sha256'],strategy=manifest(),request_sha256=request_sha,
        source_checkpoint_sha256=payload['source_checkpoint_sha256'],baseline_state_sha256=payload['baseline_state_sha256'])
    reopened=runtime.t.load(path,map_location='cpu',weights_only=False)
    if runtime.state_hash(reopened)!=runtime.state_hash(payload) or digest(path)!=receipt['sha256']:
        raise ValueError('Reopened recovery state differs')
    atomic_json(path.with_suffix('.json'),receipt)
    return dict(file=path.name,complete_values_reopened_exact=True,**receipt)


def load_state(runtime,path,output,request_sha):
    path=Path(path);expected=Path(output).parent/'interrupt/diagnostic-state.pt'
    if (path.is_symlink() or path.with_suffix('.json').is_symlink() or path.resolve()!=expected.resolve()):
        raise ValueError('Only explicit sibling diagnostic state may resume')
    receipt=read(path.with_suffix('.json'))
    expected_metadata=dict(schema=STATE_SCHEMA,strategy=manifest(),request_sha256=request_sha,
        source_checkpoint_sha256=runtime.request['checkpoint']['sha256'],baseline_state_sha256=runtime.state_hash(runtime.base))
    if (any(receipt.get(k)!=v for k,v in expected_metadata.items())
            or path.stat().st_size!=receipt.get('bytes') or not 0<path.stat().st_size<=128*1024**2
            or digest(path)!=receipt.get('sha256')):
        raise ValueError('Recovery receipt/strategy/source differs')
    # Full RNG state includes trusted checkpoint NumPy/Python objects. Read only
    # the locally created, hash-bound sibling after identity validation.
    payload=runtime.t.load(path,map_location='cpu',weights_only=False)
    if (any(payload.get(k)!=v for k,v in expected_metadata.items())
            or runtime.state_hash(payload['state'])!=payload.get('state_sha256')
            or payload['state_sha256']!=receipt['state_sha256']):
        raise ValueError('Private recovery payload/strategy differs')
    state=payload['state'];p=state['progress'];base=runtime.base['progress']
    if (p['step']!=base['step']+1 or p['cursor']!=base['cursor']+128
            or p['epoch']!=base['epoch']): raise ValueError('Recovery is not the single interrupted update')
    runtime.restore(state)
    if runtime.state_hash(runtime.snapshot())!=receipt['state_sha256']:
        raise ValueError('Restored full state differs from saved payload')
    return receipt['state_sha256']


def recovery_run(runtime,output,mode,request_sha,recovery_path=None):
    result={};base_hash=runtime.state_hash(runtime.base)
    try:
        result['starting_state_sha256']=runtime.state_hash(runtime.snapshot())
        if mode=='resume': result['loaded_state_sha256']=load_state(runtime,recovery_path,output,request_sha)
        updates=[];observations=[]
        for _ in range(2 if mode=='uninterrupted' else 1):
            with analytic_key_bias(runtime.t,runtime.model,True) as observed:
                updates.append(runtime.update(True))
            validate_observation(observed);observations.append(observed)
        state=runtime.snapshot()
        result.update(passed=True,full_state_sha256=runtime.state_hash(state),
            component_sha256={k:runtime.state_hash(v) for k,v in state.items()},updates=updates,
            hook_observations=observations,step=runtime.progress['step'],cursor=runtime.progress['cursor'],
            schedule_progress=runtime.schedule.state_dict()['progress'],dropout=runtime.config['dropout'])
        if mode=='interrupt': result['diagnostic_checkpoint']=save_state(runtime,Path(output)/'diagnostic-state.pt',request_sha)
        return result
    finally:
        runtime.restore(runtime.base)
        if runtime.state_hash(runtime.snapshot())!=base_hash: raise ValueError('New-policy recovery rollback differs')
        result['rollback']=dict(ROLLBACK);atomic_json(Path(output)/'recovery.json',result)


def agreement(results):
    a,b,c=(results[n] for n in MODES[:3])
    if (not all(v['passed'] for v in (a,b,c)) or a['full_state_sha256']!=c['full_state_sha256']
            or a['component_sha256']!=c['component_sha256'] or a['updates']!=b['updates']+c['updates']
            or len(a['updates'])!=2 or len(b['updates'])!=1 or len(c['updates'])!=1
            or len({v['starting_state_sha256'] for v in (a,b,c)})!=1
            or c['loaded_state_sha256']!=b['full_state_sha256']
            or a['step']!=c['step'] or a['step']!=b['step']+1
            or a['cursor']!=c['cursor'] or a['cursor']!=b['cursor']+128
            or a['schedule_progress']!=c['schedule_progress'] or a['schedule_progress']!=b['schedule_progress']+128
            or len({v['dropout'] for v in (a,b,c)})!=1):
        raise ValueError('New-policy continuous/interrupted fresh-process recovery differs')
    return dict(passed=True,fresh_process=True,updates=2,full_state_sha256=a['full_state_sha256'],
        component_sha256=a['component_sha256'],gradient_policy=manifest(),dropout=a['dropout'],
        checkpoint_schema=STATE_SCHEMA,scope='Model/gradients/Adam/sample schedule/permutation/cursor/epoch totals/RNG plus versioned gradient policy')


def accept(result,request,request_sha,mode,code):
    if (code!=0 or result.get('schema')!=SCHEMA or result.get('mode')!=mode or result.get('passed') is not True
            or result.get('request_sha256')!=request_sha or result.get('key_bias_manifest')!=manifest()
            or result.get('checkpoint_sha256')!=request['checkpoint']['sha256']
            or result.get('checkpoint_step')!=request['checkpoint']['step']
            or result.get('plugin_tree_sha256')!=fingerprint(request['plugin_files'])
            or result.get('rollback')!=ROLLBACK or result.get('candidate_admitted') is not False
            or result.get('formal_training_started') is not False or result.get('training_samples_committed')!=0):
        raise ValueError('New-policy child incomplete or identity differs')
    if mode!='trace':
        n=2 if mode=='uninterrupted' else 1
        if len(result['updates'])!=n or len(result['hook_observations'])!=n:
            raise ValueError('Recovery update coverage differs')
        for observation in result['hook_observations']: validate_observation(observation)
    else:
        trace=result['trace']
        if (trace.get('passed') is not True or trace.get('schema')!='sample-trace.v2'
                or set(trace['routes'])!={'fp32','bf16_analytic'} or trace.get('same_samples') is not True
                or trace.get('target_boundary_passed') is not True): raise ValueError('Paired trace readiness incomplete')
        for route in trace['routes'].values():
            if (route.get('passed') is not True or route.get('full_model_replay_exact') is not True
                    or route.get('structural_errors')!=0 or route.get('structural_warnings')!=0
                    or route.get('samples')!=2 or route.get('complete_values_reopened_exact') is not True):
                raise ValueError('Real complete-value trace/replay missing')
    return result


def native(request_path,output,mode,recovery_path=None,*,operation=None):
    from .attention_precision_diagnostic import unchanged,SOURCE_REVISION,JOB
    from .attention_update_native import NativeUpdate
    request=read(request_path);validate(request,revision());p=request['checkpoint'];source=Path(request['source_suite'])
    if (request['source_config'].get('source_revision')!=SOURCE_REVISION
            or read(source/'jobs'/JOB/'config.json')!=request['source_config']
            or p['path']!=str(source/'jobs'/JOB/'checkpoints'/p['file']) or Path(p['path']).is_symlink()):
        raise ValueError('Original source identity differs')
    binding={k:request[k] for k in BINDING_KEYS}
    if bind_screen(request['screen_binding']['suite'],binding)!=request['screen_binding']: raise ValueError('Prior screen changed')
    unchanged(binding)
    import torch
    import torch.nn.attention
    import numpy as np
    if (torch.__version__!='2.7.1+cu126' or np.__version__!='2.2.6'
            or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('Existing pinned CUDA BF16 runtime required')
    torch.set_num_threads(1);seed=request['source_config']['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    fastpath=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False);runtime=None
    try:
        runtime=NativeUpdate(torch,np,request);request_sha=digest(request_path)
        if operation is not None:
            result=operation(runtime,output)
        elif mode=='trace':
            from .key_bias_trace import collect_and_validate
            result=dict(passed=True,trace=collect_and_validate(runtime,output),rollback=dict(ROLLBACK))
        else: result=recovery_run(runtime,output,mode,request_sha,recovery_path)
        result.update(schema=SCHEMA,mode=mode,request_sha256=request_sha,key_bias_manifest=manifest(),
            checkpoint_sha256=p['sha256'],checkpoint_step=p['step'],plugin_tree_sha256=fingerprint(request['plugin_files']),
            candidate_admitted=False,formal_training_started=False,training_samples_committed=0,
            environment=dict(torch=str(torch.__version__),numpy=str(np.__version__),deterministic=True,tf32=False))
        atomic_json(Path(output)/'result.json',result)
    finally:
        if runtime is not None:
            runtime.restore(runtime.base)
            if runtime.state_hash(runtime.snapshot())!=runtime.state_hash(runtime.base): raise ValueError('Final full-state rollback differs')
        torch.backends.mha.set_fastpath_enabled(fastpath);unchanged(binding)
        if bind_screen(request['screen_binding']['suite'],binding)!=request['screen_binding']: raise ValueError('Prior evidence changed')


def execute(args,folder,report):
    from .attention_precision_diagnostic import bind_source,unchanged,JOB
    from .attention_diagnostic import gpu_preflight
    from .runner import discover_gpus
    from .diagnostic_watch import Supervisor
    from .attention_update_diagnostic import remove_disposable_checkpoint
    gpus,inventory=discover_gpus(args.gpu)
    if len(gpus)!=1: raise ValueError('Exactly one physical GPU required')
    gpu=gpus[0];report.update(gpu_identity=inventory[gpu],gpu_preflight=gpu_preflight(inventory[gpu]))
    if not report['gpu_preflight']['idle']: raise ValueError('GPU busy; retain existing work')
    source=args.source_suite.resolve()
    if source.parent.name!='runs' or not (source/'jobs'/JOB).is_dir(): raise ValueError('Original source Suite missing')
    with ExitStack() as held:
        for path in (source/'progress-supervisor.lock',source/'suite.lock',source/'jobs'/JOB/'worker.lock'):
            held.enter_context(lock(path))
        binding=bind_source(source,args.cache);screen_binding=bind_screen(args.screen_from,binding)
        request=dict(schema='qkan-attention-update-input.v1',**binding,source_revision=report['source_revision'],policy=POLICY,
            optimization_config=read(BF16),optimization_config_sha256=digest(BF16),key_bias_config=read(args.optimizations),
            key_bias_config_sha256=digest(args.optimizations),key_bias_manifest=manifest(),screen_binding=screen_binding,
            dossier_sha256=digest(DOSSIER),plugin_plan_sha256=digest(PLAN),plugin_files=report['plugin_files'])
        validate(request,report['source_revision']);path=folder/'recovery-input.json';atomic_json(path,request)
        supervisor=Supervisor(args.child_timeout,args.stall_timeout);report['child_executions']=supervisor.records
        report['screen_binding']=screen_binding;results={};report['results']=results
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',QKAN_DIAGNOSTIC_STAGE_FENCES='0')
        try:
            for mode in MODES:
                output=folder/mode;argv=[sys.executable,'-B','-m',TARGET,'--worker','--request',str(path),
                    '--output',str(output),'--mode',mode]
                if mode=='resume': argv+=['--recovery-state',str(folder/'interrupt/diagnostic-state.pt')]
                print('Analytic K-bias recovery child: '+mode,flush=True)
                code=supervisor.run(dict(target=TARGET,cwd=str(ROOT),argv=argv),folder/(mode+'.log'),env)
                if (output/'worker-error.json').is_file(): report['native_failure_locations']=read(output/'worker-error.json')
                if code!=0 or not (output/'result.json').is_file():
                    report['native_child_failure']=dict(mode=mode,exit_code=code,result_present=(output/'result.json').is_file())
                    raise RuntimeError('Recovery child failed; inspect compact native diagnostics')
                results[mode]=accept(read(output/'result.json'),request,digest(path),mode,code)
                if mode=='resume':
                    report['recovery']=agreement(results)
                    # Cleanup only after independently validated fresh-process recovery.
                    receipt=dict(results['interrupt']['diagnostic_checkpoint']);receipt.pop('complete_values_reopened_exact')
                    report['disposable_checkpoint']=remove_disposable_checkpoint(folder/'interrupt',dict(diagnostic_checkpoint=receipt))
            report.update(passed=True,status='recovery_trace_preflight_passed',native_trace_ready=True,
                architecture_status='native_fixture_validated_formal_pending')
        finally:
            unchanged(binding)
            if bind_screen(args.screen_from,binding)!=screen_binding: raise ValueError('Prior screen/evidence changed')
            report['original_checkpoint_and_metadata_unchanged']=True


def prepare_worker_output(output):
    """Allow the supervisor's existing diagnostics tree; preserve stale runs."""
    output=Path(output)
    if output.is_symlink(): raise ValueError('Worker output must be an owned directory')
    output.mkdir(parents=True,exist_ok=True)
    for entry in output.iterdir():
        if entry.name!='diagnostics' or entry.is_symlink() or not entry.is_dir():
            raise ValueError('Use a fresh output; only supervisor diagnostics may already exist')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-suite',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--gpu',default='0')
    p.add_argument('--screen-from',type=Path);p.add_argument('--optimizations',type=Path)
    p.add_argument('--child-timeout',type=float,default=900);p.add_argument('--stall-timeout',type=float,default=300)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--request',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--output',type=Path,help=argparse.SUPPRESS);p.add_argument('--mode',choices=MODES,help=argparse.SUPPRESS)
    p.add_argument('--recovery-state',type=Path,help=argparse.SUPPRESS);args=p.parse_args()
    if args.worker:
        if any(getattr(args,k) is None for k in ('request','output','mode')): p.error('Worker request/output/mode required')
        if (args.mode=='resume')!=(args.recovery_state is not None): p.error('Recovery state only for resume')
        prepare_worker_output(args.output)
        try: native(args.request,args.output,args.mode,args.recovery_state)
        except BaseException as error: worker_failure(args.output,error);raise
        return 0
    if any(getattr(args,k) is None for k in ('source_suite','cache','screen_from','optimizations')):
        p.error('Original source/cache,completed screen and explicit optimization config required')
    if not resolve(read(args.optimizations)): p.error('Explicit analytic gradient strategy required')
    from .bootstrap import check_exchange
    from .diagnostic_watch import Supervisor
    from .return_artifacts import upload_declared
    Supervisor(args.child_timeout,args.stall_timeout);check_exchange();folder=ROOT/'runs'/(
        'key-bias-recovery-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,source_revision=revision(),status='failed',passed=False,candidate_admitted=False,
        formal_training_started=False,training_samples_committed=0,native_trace_ready=False,
        architecture_status='approved_pending_native_fixture',key_bias_manifest=manifest(),plugin_files=code_binding(),
        dossier_sha256=digest(DOSSIER),plugin_plan_sha256=digest(PLAN),optimization_config_sha256=digest(args.optimizations))
    report['plugin_tree_sha256']=fingerprint(report['plugin_files']);error=None
    print('Analytic K-bias recovery Suite: '+str(folder),flush=True)
    try: execute(args,folder,report)
    except BaseException as caught:
        error=caught;report.update(passed=False,status='failed',error_type=type(caught).__name__)
        worker_failure(folder,caught);report['failure_locations']=read(folder/'worker-error.json')
    atomic_json(folder/'key-bias-recovery.json',report)
    try: upload_declared(folder,'key-bias-recovery',['key-bias-recovery.json'])
    except Exception:
        print('Return pending; compact report and retry manifest retained: '+str(folder),flush=True)
        if error is None: raise
    if error is not None: raise error
    print('New-policy recovery/trace preflight complete; formal admission remains closed. See key-bias-recovery.json.',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
