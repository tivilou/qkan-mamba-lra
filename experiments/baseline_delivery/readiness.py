"""One collaborator handoff: isolated setup, all gates, return, then formal run.

The committed release stays closed. A local release is minted only after every
declared runtime gate passes and binds the committed review policy and code.
"""
import argparse
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .common import ROOT, atomic_json, digest, fingerprint, lock, read, revision, run_child

MODELS=('transformer','mamba1','mega','s5','s4d_minimal')
TASKS=('listops','text','retrieval','image','pathfinder')


def compare_numpy(a,b):
    import numpy as np
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys(): raise ValueError('S5 recovery keys differ')
        for key in a: compare_numpy(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b): raise ValueError('S5 recovery lengths differ')
        for x,y in zip(a,b): compare_numpy(x,y)
    elif isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or not np.array_equal(a,b):
            raise ValueError('S5 recovery array differs')
    elif a!=b: raise ValueError('S5 recovery value differs')


def policy_check():
    from .runner import implementation_hash
    policy=read(ROOT/'experiments/baseline_delivery/fidelity-policy.json')
    if policy['status']!='reviewed_for_conditional_runtime_admission': raise ValueError('Review policy is not active')
    if policy['models']!=list(MODELS) or policy['tasks']!=list(TASKS): raise ValueError('Review policy coverage differs')
    for path,sha in policy['reviewed_artifacts'].items():
        if digest(ROOT/path)!=sha: raise ValueError('Reviewed artifact changed: '+path)
    data=read(ROOT/'docs/evidence/mega-document-v3-qualification.json')
    if data['status']!='qualified' or not data['formal_baseline_allowed']: raise ValueError('Dataset not qualified')
    return implementation_hash(),digest(ROOT/'experiments/baseline_delivery/fidelity-policy.json')


def validate_runtime_release(release,config):
    implementation,policy=policy_check()
    if release.get('implementation_sha256')!=implementation or release.get('policy_sha256')!=policy:
        raise ValueError('Runtime release policy or implementation changed')
    if release.get('source_revision')!=revision(): raise ValueError('Runtime release source changed')
    if set(release.get('certificates',{}))!=set(MODELS): raise ValueError('Runtime release cohort is incomplete')
    for name,entry in release['certificates'].items():
        path=ROOT/entry['path']
        if not path.resolve().is_relative_to((ROOT/'runs').resolve()) or digest(path)!=entry['sha256']:
            raise ValueError('Invalid local runtime certificate')
        certificate=read(path)
        if certificate['model']!=name or certificate['policy_sha256']!=policy:
            raise ValueError('Certificate model or policy mismatch')
        for artifact,sha in certificate['evidence'].items():
            if digest(ROOT/artifact)!=sha: raise ValueError('Runtime evidence changed')
    if config and config['model'] not in release['certificates']: raise ValueError('Model not admitted')


def execute(args,suite):
    from .runner import canary,config_for,discover_gpus,implementation_hash,execute as train
    from .sync import prepare
    from .bootstrap import check_core,check_mamba
    source=revision();implementation,policy=policy_check()
    from .migration import snapshot,plan_for,compatible,admit
    if args.migrate_from:
        if not args.adaptive_memory: raise ValueError('Migration requires explicit adaptive-memory policy')
        snapshot(args.migrate_from,suite,args.cache)
    migration=plan_for(suite)
    if migration and not args.adaptive_memory: raise ValueError('Migrated suite requires adaptive memory')
    context=suite/'context.json'
    identity=dict(source_revision=source,implementation_sha256=implementation,policy_sha256=policy,
                  cache=str(args.cache.resolve()),gpus=args.gpus)
    if args.adaptive_memory: identity['adaptive_memory']=True
    if args.dynamic_logical_batch: identity['dynamic_logical_batch']=True
    if args.performance: identity['performance']=True
    if migration: identity['migration_sha256']=digest(suite/'migration-source/plan.json')
    if context.exists() and read(context)!=identity: raise ValueError('Readiness resume identity changed')
    atomic_json(context,identity)
    setup={'target':'experiments.baseline_delivery.strong_environments','cwd':str(ROOT),
           'argv':[sys.executable,'-B','-m','experiments.baseline_delivery.strong_environments']}
    if run_child(setup,suite/'setup.log')!=0: raise ValueError('Isolated dependency/source setup failed; inspect setup.log locally')
    check_core();check_mamba()
    data=prepare(args.cache,None);gpus,inventory=discover_gpus(args.gpus)
    probe_evidence={}
    for model in MODELS:
        print('Source gate: '+model,flush=True)
        path=suite/(model+'-source-probe.json')
        if path.exists():
            value=read(path)
            if not value.get('passed') or value.get('implementation_sha256')!=implementation:
                raise ValueError('Stale source probe: '+model)
        else:
            strong=model in ('mega','s5')
            interpreter=str(ROOT/('.venv-'+model)/'bin/python') if strong else sys.executable
            module='experiments.baseline_delivery.strong_probe' if strong else 'experiments.baseline_delivery.acceptance'
            spec=dict(target=module,cwd=str(ROOT),argv=[interpreter,'-B','-m',module,
                '--model' if strong else '--probe',model,'--cache',str(args.cache.resolve()),
                '--device','cuda:0','--output',str(path)])
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus[0]}
            if run_child(spec,suite/(model+'-source-probe.log'),env)!=0: raise ValueError('Source forward/gradient/update gate failed: '+model)
        probe_evidence[model]={path.relative_to(ROOT).as_posix():digest(path)}
    jobs=[];statuses={}
    for model in MODELS:
        for task in TASKS:
            job_id=model+'-'+task+'-s42';job=suite/'jobs'/job_id;job.mkdir(parents=True,exist_ok=True)
            if migration and migration['jobs'][job_id].get('completion'):
                from .migration import preserve_completed
                jobs.append(preserve_completed(suite,job_id));statuses[job_id]=dict(state='computed',preserved=True)
                print('Preserved completed result: '+job_id+' | original revision '+migration['source_revision'],flush=True)
                continue
            config=config_for(task,model,42,data,source);cfg=job/'config.json'
            if args.adaptive_memory:
                from .adaptive_memory import default_policy
                config['adaptive_memory']=default_policy(config['batch_size'])
                if args.dynamic_logical_batch:
                    from .logical_batch import policy as logical_policy
                    config['adaptive_memory']=logical_policy(config['batch_size'])
            if args.performance:
                from .performance import policy as performance_policy
                config['performance']=performance_policy()
            if migration:
                compatible(read(suite/'migration-source'/job_id/'config.json'),config)
            if cfg.exists() and read(cfg)!=config: raise ValueError('Job configuration changed')
            atomic_json(cfg,config)
            jobs.append(dict(id=job_id,config_sha256=fingerprint(config)));statuses[job_id]=dict(state='pending')
            for gpu in gpus:
                folder=suite/'preflight'/fingerprint(inventory[gpu])[:12]/job_id
                if not (folder/'canary.json').exists():
                    print('Recovery/trace gate: '+job_id+' GPU '+gpu+
                        (' | logical32/64/128; fresh-process restore' if args.dynamic_logical_batch else ''),flush=True)
                    if folder.exists(): folder.rename(folder.with_name(folder.name+'.failed-'+uuid.uuid4().hex[:8]))
                    previous=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=gpu
                    try:
                        receipt=canary(config,data,args.cache,folder,'cuda:0')
                        receipt.update(gpu_identity=inventory[gpu],implementation_sha256=implementation)
                        atomic_json(folder/'canary.json',receipt)
                    finally:
                        if previous is None: os.environ.pop('CUDA_VISIBLE_DEVICES',None)
                        else: os.environ['CUDA_VISIBLE_DEVICES']=previous
                receipt=read(folder/'canary.json')
                if (not receipt['passed'] or not receipt['trace_semantic_and_replay'] or not receipt['fresh_process']
                    or receipt['test_access'] or receipt['config_sha256']!=fingerprint(config)
                    or receipt.get('implementation_sha256')!=implementation or receipt.get('gpu_identity')!=inventory[gpu]):
                    raise ValueError('Recovery/trace gate does not bind job/device: '+job_id)
                for name in ('canary.json','uninterrupted/trace-validation.json','interrupted/trace-validation.json'):
                    path=folder/name;probe_evidence[model][path.relative_to(ROOT).as_posix()]=digest(path)
                if model=='s5':
                    for name in ('recovery-validation.json','interrupted/restore-validation.json'):
                        path=folder/name
                        if not read(path).get('passed'): raise ValueError('S5 recovery/restore evidence failed')
                        probe_evidence[model][path.relative_to(ROOT).as_posix()]=digest(path)
    certificates={}
    for model,evidence in probe_evidence.items():
        certificate=suite/(model+'-certificate.json')
        atomic_json(certificate,dict(schema='qkan-runtime-fidelity.v1',model=model,
            status='source_faithful_adapted',implementation_sha256=implementation,
            protocol_sha256=digest(ROOT/'experiments/baseline_delivery/protocol.json'),policy_sha256=policy,
            source_revision=source,evidence=evidence,
            execution_variant='dynamic_logical_sample_clock' if args.dynamic_logical_batch else ('adaptive_physical' if args.adaptive_memory else 'fixed'),
            scope=('Pinned-source mechanism with reviewed dynamic logical batch/sample schedule; '
                   'not fixed-batch protocol equivalence. No author score reproduction or SOTA claim.'
                if args.dynamic_logical_batch else
                   'Direct pinned-source mechanism under explicitly reviewed common protocol. No author score reproduction or SOTA claim.')))
        certificates[model]=dict(path=certificate.relative_to(ROOT).as_posix(),sha256=digest(certificate))
    release=suite/'runtime-release.json'
    atomic_json(release,dict(schema='qkan-baseline-release.v1',status='released',cohort_frozen=True,
        formal_run_allowed=True,certificates=certificates,implementation_sha256=implementation,
        source_revision=source,policy_sha256=policy))
    validate_runtime_release(read(release),None)
    os.environ['QKAN_RUNTIME_RELEASE']=str(release)
    from .common import child_spec
    migration_report=admit(suite,{j['id']:read(suite/'jobs'/j['id']/'config.json') for j in jobs},
        data,args.cache,gpus,run_child,child_spec,inventory)
    atomic_json(suite/'suite.json',dict(schema='qkan-baseline-suite.v1',source_revision=source,
        protocol_sha256=digest(ROOT/'experiments/baseline_delivery/protocol.json'),models=list(MODELS),
        tasks=list(TASKS),seeds=[42],data=str(data),jobs=jobs,gpu_inventory=inventory))
    if not (suite/'status.json').exists(): atomic_json(suite/'status.json',statuses)
    # Compact evidence returned before long jobs start. No private inputs/traces.
    compact=dict(context=identity,source_probes={m:read(suite/(m+'-source-probe.json')) for m in MODELS},
        recovery=[read(p) for p in sorted((suite/'preflight').glob('*/*/canary.json'))],
        certificates={m:read(suite/(m+'-certificate.json')) for m in MODELS})
    if migration_report: compact['migration']=migration_report
    atomic_json(suite/'gate-evidence.json',compact)
    atomic_json(suite/'gate-report.json',dict(status='passed',source_revision=source,
        next='formal_training_ready',jobs=len(jobs),
        dataset_scope='internal noncommercial research; no input redistribution',evidence_sha256=digest(suite/'gate-evidence.json')))
    from .return_artifacts import upload_declared
    upload_declared(suite,'formal-readiness',['gate-report.json','gate-evidence.json'])
    if not args.gates_only: train(suite,gpus,args.cache,data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--gpus',default='0')
    p.add_argument('--resume',type=Path);p.add_argument('--gates-only',action='store_true')
    p.add_argument('--output',type=Path,help='Explicit NEW suite path under runs/')
    p.add_argument('--adaptive-memory',action='store_true',help='New adaptive trajectory; cannot resume a fixed-policy suite')
    p.add_argument('--migrate-from',type=Path,help='Fork paused admitted31639c0/adaptive96e542d, or dynamic9a147cd with --performance, into a new suite')
    p.add_argument('--dynamic-logical-batch',action='store_true',help='New sample-scheduled logical32/64/128 trajectory')
    p.add_argument('--performance',action='store_true',help='FP32 bounded lookahead, reusable pinned staging and first16 update profile')
    args=p.parse_args()
    if args.performance: args.dynamic_logical_batch=True
    if args.dynamic_logical_batch: args.adaptive_memory=True
    if args.migrate_from and (args.resume or not args.adaptive_memory):
        p.error('--migrate-from needs --adaptive-memory and a NEW suite, not --resume')
    if args.output and args.resume: p.error('--output and --resume are mutually exclusive')
    import signal
    def interrupted(*_): raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,interrupted)
    from .bootstrap import check_exchange
    check_exchange()
    suite=args.resume.resolve() if args.resume else (args.output.resolve() if args.output else ROOT/'runs'/('formal-ready-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]))
    if not suite.is_relative_to(ROOT/'runs'): raise ValueError('Readiness directory must be under project runs/')
    suite.mkdir(parents=True,exist_ok=bool(args.resume));print('Suite: '+str(suite),flush=True)
    with lock(suite/'suite.lock'):
        try: execute(args,suite)
        except BaseException as error:
            from .setup_diagnostics import summarize,redacted_tail
            # No raw child logs/exception text may leak input text or credentials.
            failure=suite/('failure-'+uuid.uuid4().hex[:12]);failure.mkdir()
            logs={p.relative_to(suite).as_posix():dict(sha256=digest(p),bytes=p.stat().st_size)
                  for p in suite.rglob('*.log')}
            atomic_json(failure/'failure.json',dict(status='failed',error_type=type(error).__name__,
                source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                suite=suite.relative_to(ROOT).as_posix(),local_logs=logs,
                recovery_diagnostics=[read(p) for p in sorted((suite/'preflight').glob('*/*/child-failure.json'))],
                recovery_comparisons=[dict(job=p.parent.name,report=read(p))
                    for p in sorted((suite/'preflight').glob('*/*/recovery-validation.json'))
                    if p.parent.name.startswith('s5-')],
                setup_diagnostic=summarize(suite/'setup.log'),formal_release_allowed=False))
            if args.adaptive_memory:
                failure_report=read(failure/'failure.json')
                failure_report['adaptive_memory']=[dict(job=p.parent.name,report=read(p))
                    for p in sorted((suite/'jobs').glob('*/adaptive-memory.json'))]
                atomic_json(failure/'failure.json',failure_report)
            if (suite/'migration-source/plan.json').exists():
                failure_report=read(failure/'failure.json')
                failure_report['migration_diagnostics']=[dict(job=p.parent.name,report=read(p))
                    for p in sorted((suite/'jobs').glob('*/worker-error.json'))]
                atomic_json(failure/'failure.json',failure_report)
            from .return_artifacts import upload_declared
            files=['failure.json']
            if redacted_tail(suite/'setup.log',failure/'setup-redacted.log'):
                files.append('setup-redacted.log')
            # Source probes use fixed synthetic inputs, never dataset samples.
            # Only these declared logs may accompany setup diagnostics; real
            # task/recovery/training logs and traces remain local.
            for model in MODELS:
                name=model+'-source-probe-redacted.log'
                if redacted_tail(suite/(model+'-source-probe.log'),failure/name):
                    files.append(name)
            try: upload_declared(failure,'formal-readiness',files,run_id=failure.name)
            except Exception: print('Failure receipt retained locally: '+str(failure),flush=True)
            raise


if __name__=='__main__': main()
