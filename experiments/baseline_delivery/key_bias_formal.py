"""Explicit mixed-executor formal continuation, with immutable source snapshots.

Only Transformer/Retrieval uses the reviewed BF16 analytic worker. Every other
job executes its original admitted code/configuration in the original checkout.
No dependency setup, scientific canary rerun, or ordinary-worker CLI unlock.
"""
import argparse
from contextlib import ExitStack
import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .common import ROOT, atomic_json, digest, fingerprint, lock, read, revision, spawn, stop_child, worker_failure
from . import key_bias_training as strategy
from .migration import PARALLEL_REVISION, checked_pointer, completed_files, copy_verified, snapshot, validate_source_admission

TARGET = 'experiments.baseline_delivery.key_bias_formal'
JOB = 'transformer-retrieval-s42'
REVIEW = ROOT/'docs/evidence/key-bias-worker-native-review-20261117.json'
PLAN = ROOT/'docs/baselines/attention-key-bias-formal-plan.json'
MODELS = ('transformer', 'mamba1', 'mega', 's5', 's4d_minimal')
TASKS = ('listops', 'text', 'retrieval', 'image', 'pathfinder')
JOBS = {m+'-'+t+'-s42' for m in MODELS for t in TASKS}
SCHEMA = 'qkan-key-bias-formal-input.v1'


def entry_binding():
    paths = (Path(__file__), ROOT/'scripts/continue_key_bias_training.sh', PLAN, REVIEW)
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def confined(path, root):
    path, root = Path(path).absolute(), Path(root).resolve()
    if not path.resolve().is_relative_to(root): raise ValueError('Formal path leaves its owned root')
    for part in (path, *path.parents):
        if part == root: break
        if part.is_symlink(): raise ValueError('Formal artifact path contains a link')
    return path


def reviewed_worker(folder):
    path = Path(folder)/'key-bias-worker-admission.json'
    review = read(REVIEW)
    if (path.is_symlink() or digest(path) != review['source_report']['sha256']
            or path.stat().st_size != review['source_report']['bytes']):
        raise ValueError('Reviewed actual-worker report required')
    report = read(path)
    if (not review['native_passed'] or not report['passed'] or report['candidate_admitted']
            or report['formal_training_started'] or not report['original_checkpoint_and_metadata_unchanged']
            or report['implementation'] != strategy.binding()):
        raise ValueError('Actual-worker implementation or admission changed')
    from .key_bias_worker_admission import agreement
    if agreement(report['results']) != report['recovery']:
        raise ValueError('Actual-worker recovery receipt differs')
    return report


def verify_input(path, ready=False):
    path = confined(path, ROOT/'runs'); suite = path.parent
    if not suite.name.startswith('key-bias-formal-'): raise ValueError('Owned formal suite required')
    r = read(path)
    if (r.get('schema') != SCHEMA or r['source_revision'] != revision()
            or r['entry_binding'] != entry_binding() or r['implementation'] != strategy.binding()
            or r['suite_manifest_sha256'] != digest(suite/'suite.json')
            or r['snapshot_sha256'] != digest(suite/'migration-source/plan.json')):
        raise ValueError('Formal entry, source snapshot or manifest changed')
    manifest = read(suite/'suite.json')
    if (len(manifest['jobs']) != 25 or {j['id'] for j in manifest['jobs']} != JOBS
            or manifest['models'] != list(MODELS) or manifest['tasks'] != list(TASKS)
            or manifest['seeds'] != [42] or manifest['data'] != r['data']):
        raise ValueError('Frozen 25-job roster/data changed')
    for job in manifest['jobs']:
        folder = confined(suite/'jobs'/job['id'], suite)
        if fingerprint(read(folder/'config.json')) != job['config_sha256']:
            raise ValueError('Formal job configuration changed')
    original = read(suite/'migration-source'/JOB/'config.json')
    expected = strategy.target_config(original); expected['source_revision'] = r['source_revision']
    if r['target_config'] != expected or read(suite/'jobs'/JOB/'config.json') != expected:
        raise ValueError('Candidate changes more than declared strategy/provenance')
    migration_input = suite/'key-bias-worker-migration/worker-input.json'
    if digest(migration_input) != r['migration_input_sha256']:
        raise ValueError('Zero-update migration input changed')
    if ready:
        value = read(suite/'formal-ready.json')
        if (value.get('passed') is not True or value['input_sha256'] != digest(path)
                or value['selection_sha256'] != digest(suite/'selection/result.json')
                or not read(suite/'selection/result.json')['passed']):
            raise ValueError('Formal zero-update preparation not complete')
    return r


def original_executor(r):
    root = Path(r['original_root'])
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if actual != PARALLEL_REVISION or dirty: raise ValueError('Original admitted executor changed')
    validate_source_admission(Path(r['source_suite']), PARALLEL_REVISION)
    return root


def prepare(args, suite):
    evidence = reviewed_worker(args.worker_from)
    plan = snapshot(args.migrate_from, suite, args.cache)
    if plan['source_revision'] != PARALLEL_REVISION or set(plan['jobs']) != JOBS:
        raise ValueError('Admitted b07 25-job source required')
    from .attention_precision_diagnostic import bind_source
    from .key_bias_worker_admission import completed
    bound = bind_source(args.migrate_from.resolve(), args.cache)
    cp = plan['jobs'][JOB]['checkpoint']
    if (cp is None or any(cp[k] != bound['checkpoint'][k] for k in ('file', 'sha256', 'contract', 'step'))
            or cp['contract'] != evidence['source_checkpoint']['contract']
            or cp['step'] < evidence['source_checkpoint']['step']):
        raise ValueError('Source changed during snapshot or moved behind audited progress')
    completed_records = completed(args.migrate_from)
    if completed_records != evidence['completed']: raise ValueError('Completed source results changed')
    source = revision(); config = strategy.target_config(bound['source_config'])
    config['source_revision'] = source  # Provenance only; numeric configuration is unchanged.
    request = dict(schema='qkan-key-bias-worker-input.v1', **bound, source_revision=source,
        implementation=strategy.binding(), target_config=config, pilot=evidence['pilot'], completed=completed_records)
    migration_path = suite/'key-bias-worker-migration/worker-input.json'
    atomic_json(migration_path, request)
    jobs = []; states = {}
    for old_job in read(args.migrate_from/'suite.json')['jobs']:
        name = old_job['id']; record = plan['jobs'][name]
        src = suite/'migration-source'/name; dst = suite/'jobs'/name
        dst.mkdir(parents=True)
        job = dict(id=name, config_sha256=record['config_sha256'], execution_revision=PARALLEL_REVISION,
                   execution_root=str(args.migrate_from.parent.parent.resolve()), execution_role='original_admitted')
        if record.get('completion'):
            from .migration import preserve_completed
            job.update(preserve_completed(suite, name)); states[name] = dict(state='computed', preserved=True)
            job.update(execution_revision=record['completion']['source_revision'], execution_root=None,
                       execution_role='preserved_result', preserved_from=str(args.migrate_from.resolve()))
        else:
            selected = config if name == JOB else read(src/'config.json')
            atomic_json(dst/'config.json', selected); job['config_sha256'] = fingerprint(selected)
            if name == JOB:
                job.update(execution_revision=source, execution_root=str(ROOT), execution_role='reviewed_bf16_analytic')
            elif record['checkpoint']:
                pointer, generation = checked_pointer(src)
                copy_verified(src, dst, {'checkpoints/'+generation.name: pointer['sha256']})
                atomic_json(dst/'checkpoints/latest.json', {**pointer, 'previous': []})
            states[name] = dict(state='pending', started_training=False)
        jobs.append(job)
    atomic_json(suite/'suite.json', dict(schema='qkan-baseline-suite.v1', source_revision=source,
        protocol_sha256=digest(ROOT/'experiments/baseline_delivery/protocol.json'), models=list(MODELS),
        tasks=list(TASKS), seeds=[42], data=bound['data'], jobs=jobs,
        execution='Explicit mixed executor; candidate lineage and original admitted executors kept separate'))
    atomic_json(suite/'status.json', states)
    r = dict(schema=SCHEMA, source_revision=source, implementation=strategy.binding(), entry_binding=entry_binding(),
        source_suite=str(args.migrate_from.resolve()), original_root=str(args.migrate_from.parent.parent.resolve()),
        data=bound['data'], cache=bound['cache'], checkpoint=bound['checkpoint'], target_config=config,
        worker_from=str(args.worker_from.resolve()), reviewed_report_sha256=digest(args.worker_from/'key-bias-worker-admission.json'),
        snapshot_sha256=digest(suite/'migration-source/plan.json'), suite_manifest_sha256=digest(suite/'suite.json'),
        migration_input_sha256=digest(migration_path), original_metadata_sha256=bound['source_metadata_sha256'])
    atomic_json(suite/'formal-input.json', r)
    return r


def recertify_selection(state, evaluate, hash_state, val_hash, val_rows):
    """Re-evaluate inherited best under the new evaluation route, not new weights.

    All scientific training fields, old history and the selected tensors remain
    exact. This is a validation metric transition, never an optimizer update.
    """
    protected = lambda value: {k: v for k, v in value.items() if k not in ('best_accuracy', 'migration')}
    before = hash_state(protected(state)); result = copy.deepcopy(state)
    metric = evaluate(result['best_weights']) if result['best_weights'] is not None else None
    if metric is not None:
        if (not math.isfinite(metric['loss']) or metric['loss'] < 0 or not math.isfinite(metric['accuracy'])
                or not 0 <= metric['accuracy'] <= 1 or metric['samples'] != val_rows):
            raise ValueError('Inherited best validation metric differs')
        result['best_accuracy'] = metric['accuracy']
    elif result['best_accuracy'] != -1 or result['best_epoch'] is not None:
        raise ValueError('Best selection metadata lacks its selected weights')
    transition = dict(schema='qkan-best-evaluation-transition.v1', source_best_accuracy=state['best_accuracy'],
        source_best_epoch=state['best_epoch'], candidate_validation=metric, validation_data_sha256=val_hash,
        criterion='maximum validation accuracy; earliest tie; test never selects',
        policy=strategy.POLICY['evaluation'], selected_weights_changed=False, training_updates=0,
        protected_state_sha256=before, prior_history_preserved=True, test_access=False)
    result['migration'] = {**result['migration'], 'selection_transition': transition}
    if hash_state(protected(result)) != before: raise ValueError('Selection recertification changed training state')
    return result, transition


def native_runtime():
    import torch
    import numpy as np
    if (torch.__version__ != '2.7.1+cu126' or np.__version__ != '2.2.6'
            or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('Reviewed candidate native runtime required')
    return torch, np


def native_selection(path, output):
    r = verify_input(path); suite = Path(path).parent; output = confined(output, suite)
    if output != suite/'selection' or (output/'checkpoints/latest.json').exists():
        raise ValueError('Selection preparation needs its new owned output')
    torch, np = native_runtime()
    from . import worker
    from .attention_update_native import NativeUpdate
    from .data import Split
    from .models import Classifier
    from .diagnostic_watch import mark
    from types import SimpleNamespace
    receipt = read(suite/'key-bias-worker-migration/migration/result.json')
    if (not receipt['passed'] or receipt['updates'] or receipt['source_checkpoint_sha256'] != r['checkpoint']['sha256']
            or receipt['implementation'] != r['implementation']):
        raise ValueError('Actual zero-update migration required before selection')
    source = suite/'key-bias-worker-migration/migration'
    pointer, _ = checked_pointer(source)
    if pointer['sha256'] != receipt['checkpoint_sha256']: raise ValueError('Zero-update state changed')
    state = worker.load_checkpoint(source, pointer['contract'])
    expected = strategy.envelope(r['target_config'], r['checkpoint']['sha256'], r['implementation'])
    strategy.check_checkpoint(state, r['target_config'], expected)
    hasher = SimpleNamespace(t=torch, np=np)
    hash_state = lambda value: NativeUpdate.state_hash(hasher, value)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    val = Split(r['data'], 'retrieval', 'val', r['cache'], r['target_config']['data_hashes']['val'])
    mark('formal_best_selection', flush=True)
    evaluation_proof = {}
    def evaluate(weights):
        rng = worker.rng_state()
        try:
            model = Classifier(r['target_config'], r['cache']).to('cuda:0')
            model.load_state_dict(weights, strict=True)
            with strategy.process(torch), strategy.attention_route(torch, model, True) as route:
                metric = worker.evaluate(model, val, 'cuda:0', 16)
            if (route['layer_calls'] != [2*math.ceil(len(val)/16)]*4
                    or set(route['qkv_dtypes']) != {'torch.bfloat16'}
                    or set(route['output_dtypes']) != {'torch.bfloat16'}):
                raise ValueError('Inherited best did not use the reviewed BF16 evaluation route')
            evaluation_proof.update(layer_calls=route['layer_calls'], qkv_dtypes=sorted(set(route['qkv_dtypes'])),
                output_dtypes=sorted(set(route['output_dtypes'])), samples=len(val), micro_batch=16)
            return metric
        finally: worker.restore_rng(rng)
    state, transition = recertify_selection(state, evaluate, hash_state,
        r['target_config']['data_hashes']['val'], len(val))
    worker.save_checkpoint(output, state)
    if hash_state(worker.load_checkpoint(output, pointer['contract'])) != hash_state(state):
        raise ValueError('Selection full checkpoint roundtrip differs')
    final, _ = checked_pointer(output)
    atomic_json(output/'result.json', dict(schema='qkan-formal-selection.v1', passed=True,
        input_sha256=digest(path), implementation=r['implementation'], checkpoint=final,
        source_checkpoint_sha256=r['checkpoint']['sha256'], step=state['step'], cursor=state['cursor'],
        samples_seen=state['sample_clock']['progress'], training_updates=0, transition=transition,
        full_state_roundtrip=True, evaluation_proof=evaluation_proof))


class FormalAdmission(strategy.Admission):
    """Separate capability; the finite Admission and ordinary CLI stay unchanged."""
    def __init__(self, path):
        self.path = Path(path); self.request = verify_input(path, ready=True)
        self.output = self.path.parent/'jobs'/JOB
        self.expected = strategy.envelope(self.request['target_config'],
            self.request['checkpoint']['sha256'], self.request['implementation'])
        self.check_only = False; self.trace = False

    def validate(self, config, output, device, resume, stop_after, synthetic, migration, probe):
        if (verify_input(self.path, ready=True) != self.request or config != self.request['target_config']
                or Path(output).resolve() != self.output.resolve() or device != 'cuda:0'
                or not resume or stop_after is not None or synthetic or migration or probe is not None):
            raise ValueError('Explicit bound formal candidate invocation required')
        strategy.select(config)


def native_train(path, output):
    admission = FormalAdmission(path); r = admission.request
    if Path(output).resolve() != admission.output: raise ValueError('Formal worker output differs')
    native_runtime()
    from . import worker
    for name in ('SIGTERM', 'SIGINT', 'SIGHUP'):
        if hasattr(signal, name): signal.signal(getattr(signal, name), worker.request_pause)
    worker.PAUSE = False
    with lock(Path(output)/'worker.lock'):
        return worker.run(r['target_config'], r['data'], r['cache'], output, 'cuda:0',
            resume=True, strategy_admission=admission)


def job_spec(suite, r, job):
    output = suite/'jobs'/job['id']; config = read(output/'config.json')
    if job['id'] == JOB:
        return dict(target=TARGET, cwd=str(ROOT), argv=[sys.executable, '-B', '-m', TARGET,
            '--worker', '--request', str(suite/'formal-input.json'), '--output', str(output)])
    if config.get('training_strategy') is not None or config['source_revision'] != PARALLEL_REVISION:
        raise ValueError('Unchanged job must use its original admitted policy')
    model = config['model']; module = 'experiments.baseline_delivery.'+('s5_worker' if model == 's5' else 'worker')
    interpreter = str(ROOT/('.venv-'+model)/'bin/python') if model in ('mega', 's5') else sys.executable
    args = ['--config', str(output/'config.json'), '--data', r['data'], '--cache', r['cache'],
            '--output', str(output), '--device', 'cuda:0']
    if (output/'checkpoints/latest.json').exists(): args.append('--resume')
    return dict(target=module, cwd=r['original_root'], argv=[interpreter, '-B', '-m', module, *args])


def checkpoint_required(suite, job, status):
    folder = suite/'jobs'/job['id']
    inherited = read(suite/'migration-source/plan.json')['jobs'][job['id']]['checkpoint']
    required = job['id'] == JOB or inherited is not None or status.get('started_training', False)
    if required and not (folder/'checkpoints/latest.json').exists():
        raise ValueError('Started/migrated job lost latest checkpoint; fresh fallback rejected')
    if (folder/'checkpoints/latest.json').exists(): checked_pointer(folder)


def validate_job_result(suite, r, job):
    folder = suite/'jobs'/job['id']
    if job.get('preserved_completion'):
        from .migration import verify_completed
        return verify_completed(suite, job)
    result = completed_files(folder, read(folder/'config.json'))
    if job['id'] == JOB and read(folder/'summary.json').get('training_strategy') != strategy.envelope(
            r['target_config'], r['checkpoint']['sha256'], r['implementation']):
        raise ValueError('Completed candidate lacks its reviewed strategy lineage')
    return result


def publish_status(suite, event):
    from .return_artifacts import upload_declared
    r = read(suite/'formal-input.json'); statuses = read(suite/'status.json'); results = {}
    for job in read(suite/'suite.json')['jobs']:
        path = suite/'jobs'/job['id']/'summary.json'
        if path.exists():
            summary = read(path)
            results[job['id']] = {k: summary[k] for k in ('source_revision', 'config_sha256', 'best_epoch',
                'validation', 'test', 'optimizer_steps', 'training_seconds') if k in summary}
    folder = suite/'events'/(event+'-'+uuid.uuid4().hex[:12]); folder.mkdir(parents=True)
    value = dict(schema='qkan-formal-continuation-status.v1', event=event, suite=str(suite),
        source_revision=r['source_revision'], input_sha256=digest(suite/'formal-input.json'),
        entry_binding=r['entry_binding'], worker_implementation=r['implementation'],
        source_checkpoint_sha256=r['checkpoint']['sha256'], jobs=statuses, results=results,
        formal_training_started=any(v.get('started_training') for v in statuses.values()),
        candidate_policy=strategy.POLICY, original_results_preserved=2)
    # Recoverable control metadata only: no environment values, arrays or weights.
    value['formal_input'] = r
    if (suite/'selection/result.json').exists(): value['selection_receipt'] = read(suite/'selection/result.json')
    value['migration_input'] = read(suite/'key-bias-worker-migration/worker-input.json')
    if (suite/'formal-ready.json').exists(): value['readiness'] = read(suite/'formal-ready.json')
    if (suite/'worker-error.json').exists(): value['failure_locations'] = read(suite/'worker-error.json')
    if (suite/'preparation-children.json').exists():
        value['preparation_children'] = read(suite/'preparation-children.json')
    value['worker_failure_locations'] = {name: read(suite/name/'worker-error.json')
        for name in ('key-bias-worker-migration/migration', 'selection', *['jobs/'+n for n in sorted(JOBS)])
        if (suite/name/'worker-error.json').is_file()}
    atomic_json(folder/'formal-status.json', value)
    upload_declared(folder, 'key-bias-formal', ['formal-status.json'], run_id=suite.name+'--'+folder.name)


def run_job(suite, r, job, gpu, statuses):
    folder = suite/'jobs'/job['id']; checkpoint_required(suite, job, statuses[job['id']])
    spec = job_spec(suite, r, job); atomic_json(folder/'command.json', spec)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    if job['id'] != JOB:
        env['QKAN_RUNTIME_RELEASE'] = str(Path(r['source_suite'])/'runtime-release.json')
        env['PYTHONPATH'] = r['original_root']
    started = time.time()
    statuses[job['id']] = dict(state='running', started_training=True, physical_gpu=gpu, started=started)
    atomic_json(suite/'status.json', statuses)
    code = None
    from .performance import ProfileReturn
    from .diagnostic_watch import terminate
    from scripts.formal_progress import Dashboard, Terminal
    profile = ProfileReturn(ROOT)
    dashboard = Dashboard(ROOT, suite); terminal = Terminal('auto')
    dashboard.launch = dict(started_at=started, runner_pid=os.getpid(), interval=30)
    with (folder/'worker.log').open('ab') as log:
        child = spawn(spec, log, env)
        try:
            statuses[job['id']]['pid'] = child.pid
            atomic_json(suite/'status.json', statuses)
            next_notice = 0
            while child.poll() is None:
                profile.tick(suite, [job])
                if time.monotonic() >= next_notice:
                    progress = read(folder/'heartbeat.json') if (folder/'heartbeat.json').exists() else {}
                    compact = {k: progress[k] for k in ('epoch', 'optimizer_step', 'next_row_cursor',
                        'samples_seen', 'sample_budget', 'loss') if k in progress}
                    elapsed = time.time()-progress.get('run_started_at', time.time())
                    seen = progress.get('samples_seen', 0)-progress.get('run_samples_start', 0)
                    if elapsed > 0 and seen > 0:
                        compact['samples_per_second'] = seen/elapsed
                        if 'sample_budget' in progress:
                            compact['eta_seconds'] = max(0, progress['sample_budget']-progress['samples_seen'])/(seen/elapsed)
                    notice = dict(job=job['id'], physical_gpu=gpu, progress=compact,
                        computed=sum(v['state'] == 'computed' for v in statuses.values()), total=25,
                        pending=[n for n, v in statuses.items() if v['state'] == 'pending'],
                        failed=[n for n, v in statuses.items() if v['state'] == 'failed'])
                    atomic_json(suite/'heartbeat.json', notice)
                    dashboard.launch['updated_at'] = time.time()
                    terminal.show(dashboard.render())
                    next_notice = time.monotonic()+30
                time.sleep(.5)
            code = child.returncode
        finally:
            terminal.close()
            stop_child(child)  # Existing safe-pause grace; no diagnostic time cap on long jobs.
            cleanup = terminate(child, grace=2)
            profile.close()
            code = child.returncode
            statuses[job['id']].update(state='paused' if code == 75 else 'failed', exit_code=code)
            atomic_json(suite/'status.json', statuses)
            atomic_json(folder/('formal-child-'+uuid.uuid4().hex[:12]+'.json'),
                dict(exit_status=code, termination=cleanup, command_sha256=digest(folder/'command.json')))
            if os.name != 'nt' and (not cleanup['parent_reaped'] or not cleanup['group_verified']):
                raise ValueError('Owned formal process group did not finish cleanup')
    if code == 0:
        validate_job_result(suite, r, job)
        statuses[job['id']].update(state='computed'); atomic_json(suite/'status.json', statuses)
    return code


def zero_preparation(suite, r, gpu):
    from .diagnostic_watch import Supervisor
    supervisor = Supervisor(900, 300)
    request = suite/'key-bias-worker-migration/worker-input.json'
    stages = (
        (suite/'key-bias-worker-migration/migration', 'experiments.baseline_delivery.key_bias_worker_admission',
         ['--worker', '--request', str(request), '--mode', 'migration']),
        (suite/'selection', TARGET, ['--selection-worker', '--request', str(suite/'formal-input.json')]),
    )
    try:
        for output, module, args in stages:
            result = output/'result.json'
            if result.exists():
                if read(result).get('passed') is not True: raise ValueError('Preparation receipt failed')
                checked_pointer(output); continue
            if (output/'checkpoints/latest.json').exists():
                raise ValueError('Partial preparation retained; inspect before restarting')
            spec = dict(target=module, cwd=str(ROOT), argv=[sys.executable, '-B', '-m', module,
                *args, '--output', str(output)])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', QKAN_DIAGNOSTIC_STAGE_FENCES='0')
            if supervisor.run(spec, output.with_suffix('.log'), env) != 0 or not result.exists():
                raise ValueError('Formal zero-update preparation failed')
    finally:
        records = suite/'preparation-children.json'
        atomic_json(records, (read(records) if records.exists() else [])+supervisor.records)
    from .key_bias_worker_admission import accept
    migration = suite/'key-bias-worker-migration/migration'
    migrated = accept(read(migration/'result.json'), read(request), digest(request), 'migration')
    if (checked_pointer(migration)[0]['sha256'] != migrated['checkpoint_sha256']
            or not migrated['migration_validation']['passed']
            or migrated['migration_validation']['optimizer_steps_performed'] != 0):
        raise ValueError('Prepared zero-update migration receipt differs')
    result = read(suite/'selection/result.json'); pointer, generation = checked_pointer(suite/'selection')
    if (result.get('input_sha256') != digest(suite/'formal-input.json') or not result['full_state_roundtrip']
            or result['training_updates'] != 0 or result['checkpoint'] != pointer
            or result['source_checkpoint_sha256'] != r['checkpoint']['sha256']
            or result['step'] != r['checkpoint']['step'] or result['cursor'] != migrated['cursor']
            or result['samples_seen'] != migrated['migration_validation']['sample_schedule']['samples_seen']):
        raise ValueError('Selection preparation identity differs')
    output = suite/'jobs'/JOB
    copy_verified(suite/'selection', output, {'checkpoints/'+generation.name: pointer['sha256']})
    if (output/'checkpoints/latest.json').exists():
        if checked_pointer(output)[0] != {**pointer, 'previous': []}:
            raise ValueError('Partial formal preparation has different state')
    else: atomic_json(output/'checkpoints/latest.json', {**pointer, 'previous': []})
    atomic_json(suite/'formal-ready.json', dict(schema='qkan-formal-continuation-ready.v1', passed=True,
        input_sha256=digest(suite/'formal-input.json'), selection_sha256=digest(suite/'selection/result.json'),
        candidate_initial_checkpoint=checked_pointer(output)[0], selection_transition=result['transition'],
        training_updates=0, inherited_worker_review_sha256=digest(REVIEW),
        entry_binding=r['entry_binding'], execution='candidate reviewed worker; 22 unchanged jobs original admitted executor; 2 preserved'))


def execute(args, suite):
    path = suite/'formal-input.json'
    r = verify_input(path) if path.exists() else prepare(args, suite)
    if args.cache and str(args.cache.resolve()) != r['cache']: raise ValueError('Resume cache changed')
    if args.upload_only:
        from .return_artifacts import upload_declared
        for event in sorted((suite/'events').glob('*')):
            if (event/'artifact-delivery.json').exists():
                upload_declared(event, 'key-bias-formal', ['formal-status.json'], run_id=suite.name+'--'+event.name)
        if (suite/'compute-complete.json').exists():
            from .runner import deliver
            deliver(suite)
        return 0
    original_executor(r)
    from .attention_diagnostic import gpu_preflight
    from .runner import discover_gpus
    gpus, inventory = discover_gpus(args.gpu)
    if len(gpus) != 1 or inventory[gpus[0]] != reviewed_worker(Path(r['worker_from']))['gpu_identity']:
        raise ValueError('Use the one reviewed physical GPU/runtime')
    gpu = gpus[0]
    if not gpu_preflight(inventory[gpu])['idle']: raise ValueError('Selected GPU busy; existing work retained')
    source = Path(r['source_suite'])
    with ExitStack() as held:
        for p in (source/'progress-supervisor.lock', source/'suite.lock', *[source/'jobs'/n/'worker.lock' for n in sorted(JOBS)]):
            held.enter_context(lock(p))
        if (digest(r['checkpoint']['path']) != r['checkpoint']['sha256']
                or any(digest(p) != h for p, h in r['original_metadata_sha256'].items())):
            raise ValueError('Original source changed since the formal fork')
        if not (suite/'formal-ready.json').exists(): zero_preparation(suite, r, gpu)
        verify_input(path, ready=True)
        publish_status(suite, 'ready')  # Required authenticated return before any long update.
        if args.prepare_only: return 0
        statuses = read(suite/'status.json')
        for job in read(suite/'suite.json')['jobs']:
            folder = suite/'jobs'/job['id']
            if (folder/'summary.json').exists():
                validate_job_result(suite, r, job)
                statuses[job['id']]['state'] = 'computed'; continue
            if statuses[job['id']]['state'] == 'computed': raise ValueError('Computed job lost result')
            code = run_job(suite, r, job, gpu, statuses)
            publish_status(suite, 'job-complete' if code == 0 else 'job-paused' if code == 75 else 'job-failed')
            if code != 0: return 75 if code == 75 else 2
        atomic_json(suite/'compute-complete.json', dict(state='computed', jobs=25))
        publish_status(suite, 'computed')
        from .runner import deliver
        deliver(suite)  # Existing selected-weight/full-state whitelist and hash receipts.
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--migrate-from', type=Path); group.add_argument('--resume', type=Path)
    p.add_argument('--worker-from', type=Path); p.add_argument('--cache', type=Path)
    p.add_argument('--gpu', default='0'); p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--upload-only', action='store_true', help='Retry immutable receipts/final delivery; never launch a job')
    p.add_argument('--worker', action='store_true'); p.add_argument('--selection-worker', action='store_true')
    p.add_argument('--request', type=Path); p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.worker or args.selection_worker:
        if (args.worker and args.selection_worker) or not args.request or not args.output:
            p.error('Choose one worker mode with request and output')
        try: return native_train(args.request, args.output) if args.worker else native_selection(args.request, args.output) or 0
        except BaseException as error: worker_failure(args.output, error); raise
    if not args.resume and not (args.migrate_from and args.worker_from and args.cache):
        p.error('New continuation requires migrate-from, worker-from and cache; otherwise resume the same suite')
    if args.upload_only and (not args.resume or args.prepare_only): p.error('Upload-only needs resume, without prepare-only')
    from .bootstrap import check_exchange
    check_exchange()
    if args.resume and (not args.resume.is_dir() or not (args.resume/'formal-input.json').is_file()):
        raise ValueError('Resume requires an existing initialized formal suite')
    suite = confined(args.resume, ROOT/'runs').resolve() if args.resume else ROOT/'runs'/('key-bias-formal-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    confined(suite, ROOT/'runs'); suite.mkdir(parents=True, exist_ok=bool(args.resume))
    print('Formal continuation Suite: '+str(suite), flush=True)
    def interrupted(*_): raise KeyboardInterrupt
    for name in ('SIGTERM', 'SIGINT', 'SIGHUP'):
        if hasattr(signal, name): signal.signal(getattr(signal, name), interrupted)
    with lock(suite/'suite.lock'):
        try: return execute(args, suite)
        except BaseException as error:
            worker_failure(suite, error)
            if (suite/'formal-input.json').exists():
                try: publish_status(suite, 'paused' if isinstance(error, KeyboardInterrupt) else 'failed')
                except Exception: print('Return pending; local event receipt retained: '+str(suite), flush=True)
            else:
                from .return_artifacts import upload_declared
                failure = suite/'early-failure'; failure.mkdir(exist_ok=True)
                atomic_json(failure/'failure.json', dict(schema='qkan-formal-early-failure.v1',
                    formal_training_started=False, failure_locations=read(suite/'worker-error.json')))
                try: upload_declared(failure, 'key-bias-formal', ['failure.json'], run_id=suite.name+'--early-failure')
                except Exception: print('Failure return pending; local receipt retained: '+str(failure), flush=True)
            return 75 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == '__main__': raise SystemExit(main())
