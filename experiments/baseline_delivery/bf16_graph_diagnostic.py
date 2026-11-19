"""BF16 analytic eager16 versus BF16 Graph16; paused formal source is read-only."""
import argparse
from contextlib import ExitStack
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .common import ROOT, atomic_json, digest, fingerprint, read, revision, lock, worker_failure
from . import key_bias_training as strategy
from .attention_precision_diagnostic import unchanged
from .migration import checked_pointer

TARGET = 'experiments.baseline_delivery.bf16_graph_diagnostic'
SCHEMA = 'qkan-bf16-graph-diagnostic.v1'
JOB = 'transformer-retrieval-s42'
PLAN = ROOT/'docs/baselines/bf16-graph-plan.json'
MODES = ('parity', 'continuous', 'interrupt', 'resume', 'timing')
CLOSED = dict(candidate_admitted=False, formal_training_started=False, training_samples_committed=0)
ROLLBACK = dict(full_state_exact=True, committed_samples=0, committed_updates=0)


def binding():
    names = ('bf16_graph_diagnostic.py', 'bf16_graph_native.py', 'bf16_graph_runtime.py',
             'parallel_attention.py', 'torch_acceleration.py', 'key_bias_benchmark.py',
             'key_bias_stability_native.py')
    return dict(files={n: digest(Path(__file__).parent/n) for n in names},
                strategy=strategy.binding(), plan_sha256=digest(PLAN))


def bind_source(suite, cache):
    """Verify old executor in place, independently of this new diagnostic revision."""
    from .key_bias_formal import SCHEMA as FORMAL_SCHEMA, JOBS, MODELS, TASKS, confined
    suite = Path(suite).absolute(); root = suite.parent.parent
    if suite.parent.name != 'runs' or not suite.name.startswith('key-bias-formal-'):
        raise ValueError('Existing BF16 formal Suite required')
    confined(suite, root/'runs')
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    path = suite/'formal-input.json'; r = read(path)
    if (r.get('schema') != FORMAL_SCHEMA or r['source_revision'] != head or dirty
            or r['implementation'] != strategy.binding()
            or r['suite_manifest_sha256'] != digest(suite/'suite.json')
            or r['snapshot_sha256'] != digest(suite/'migration-source/plan.json')
            or r['migration_input_sha256'] != digest(suite/'key-bias-worker-migration/worker-input.json')):
        raise ValueError('Original formal implementation/snapshot changed')
    from .key_bias_formal import entry_binding
    if set(r['entry_binding']) != set(entry_binding()): raise ValueError('Unknown formal entry binding')
    for name, sha in r['entry_binding'].items():
        if digest(confined(root/name, root)) != sha: raise ValueError('Original entry file changed')
    manifest = read(suite/'suite.json'); folder = suite/'jobs'/JOB
    if (len(manifest['jobs']) != 25 or {j['id'] for j in manifest['jobs']} != JOBS
            or manifest['models'] != list(MODELS) or manifest['tasks'] != list(TASKS)
            or manifest['seeds'] != [42] or manifest['data'] != r['data']):
        raise ValueError('Frozen roster/data differs')
    metadata = [path, suite/'suite.json', suite/'migration-source/plan.json',
                suite/'key-bias-worker-migration/worker-input.json', suite/'formal-ready.json',
                suite/'selection/result.json', suite/'status.json',
                suite/'migration-source'/JOB/'config.json']
    metadata.extend(confined(root/name, root) for name in r['entry_binding'])
    metadata.extend(root/'experiments/baseline_delivery'/name for name in strategy.FILES)
    for job in manifest['jobs']:
        config_path = confined(suite/'jobs'/job['id']/'config.json', suite)
        if fingerprint(read(config_path)) != job['config_sha256']: raise ValueError('Job config changed')
        metadata.append(config_path)
    config = read(folder/'config.json')
    original = read(suite/'migration-source'/JOB/'config.json')
    expected = strategy.target_config(original); expected['source_revision'] = head
    if config != r['target_config'] or config != expected or not strategy.select(config):
        raise ValueError('Exact existing BF16 candidate configuration required')
    ready = read(suite/'formal-ready.json')
    if (ready.get('passed') is not True or ready['input_sha256'] != digest(path)
            or ready['selection_sha256'] != digest(suite/'selection/result.json')
            or not read(suite/'selection/result.json')['passed']): raise ValueError('Formal preparation incomplete')
    if read(suite/'status.json')[JOB]['state'] != 'paused' or (folder/'summary.json').exists():
        raise ValueError('Safely paused, unfinished candidate required; existing work left running')
    pointer, generation = checked_pointer(folder)
    if pointer['schema'] != 'qkan-training-checkpoint.v1': raise ValueError('Full Torch checkpoint required')
    data = Path(r['data']); cache = Path(cache).resolve()
    if (str(cache) != str(Path(r['cache']).resolve()) or not data.is_absolute() or not data.is_dir()
            or not (cache/'arrays'/config['data_hashes']['train']/'complete.json').is_file()):
        raise ValueError('Use existing local data/cache')
    metadata.append(folder/'checkpoints/latest.json')
    # Preserve inherited completed artifacts; never evaluate or replace their weights.
    for name in ('transformer-listops-s42', 'transformer-text-s42'):
        from .migration import completed_files
        completed = suite/'jobs'/name
        files = completed_files(completed, read(completed/'config.json'))
        metadata.extend(confined(completed/name, completed) for name in files['files'])
    return dict(source_suite=str(suite), source_executor_revision=head, source_config=config,
        source_config_sha256=fingerprint(config), checkpoint=dict(path=str(generation), **pointer),
        source_metadata_sha256={str(p): digest(p) for p in metadata}, data=str(data), cache=str(cache))


def child_spec(request, output, mode):
    if mode not in MODES: raise ValueError('Unknown diagnostic mode')
    return dict(target=TARGET, cwd=str(ROOT), argv=[sys.executable, '-B', '-m', TARGET, '--worker',
        '--request', str(request), '--output', str(output), '--mode', mode])


def accept(result, request, request_sha, mode, code):
    if (code != 0 or result.get('schema') != SCHEMA or result.get('mode') != mode
            or result.get('request_sha256') != request_sha
            or result.get('checkpoint_sha256') != request['checkpoint']['sha256']
            or result.get('implementation') != request['implementation']
            or result.get('rollback') != ROLLBACK
            or any(type(result.get(k)) != type(v) or result[k] != v for k, v in CLOSED.items())):
        raise ValueError('Diagnostic child identity/rollback differs')
    if result.get('passed') is not True: raise ValueError('Diagnostic gate rejected; preserve eager formal strategy')


def agreement(results):
    a, b, c = (results[n] for n in ('continuous', 'interrupt', 'resume'))
    if (len(a['updates']) != 2 or len(b['updates']) != 1 or len(c['updates']) != 1
            or a['updates'] != b['updates'] + c['updates']
            or a['full_state_sha256'] != c['full_state_sha256']
            or a['component_sha256'] != c['component_sha256']
            or b['full_state_sha256'] != c['loaded_state_sha256']
            or len({r['starting_state_sha256'] for r in (a, b, c)}) != 1):
        raise ValueError('Graph continuous2 versus fresh-process1+1 differs')
    return dict(passed=True, fresh_process=True, updates=2, component_sha256=a['component_sha256'])


def remove_midpoint(folder, result):
    path = Path(folder)/'interrupt/diagnostic-state.pt'; receipt = result['diagnostic_checkpoint']
    if path.is_symlink() or path.stat().st_size != receipt['bytes'] or digest(path) != receipt['sha256']:
        raise ValueError('Owned midpoint cleanup identity differs')
    path.unlink()
    return dict(removed=True, sha256=receipt['sha256'], bytes=receipt['bytes'], retained_generations=0)


def execute(args, folder, report):
    from .attention_diagnostic import gpu_preflight
    from .runner import discover_gpus
    from .diagnostic_watch import Supervisor
    source = args.source_suite.absolute()
    if not (source/'formal-input.json').is_file(): raise ValueError('Existing formal Suite required')
    with ExitStack() as held:
        for path in (source/'progress-supervisor.lock', source/'suite.lock', source/'jobs'/JOB/'worker.lock'):
            held.enter_context(lock(path))
        bound = bind_source(source, args.cache)
        try:
            report['source_snapshot_sha256'] = fingerprint(bound['source_metadata_sha256'])
            gpus, inventory = discover_gpus(args.gpu)
            if len(gpus) != 1: raise ValueError('Exactly one idle GPU required')
            gpu = gpus[0]; report.update(gpu_identity=inventory[gpu], gpu_preflight=gpu_preflight(inventory[gpu]))
            if not report['gpu_preflight']['idle']: raise ValueError('GPU busy; preserve current work')
            request = dict(**bound, schema=SCHEMA, source_revision=revision(), implementation=binding())
            path = folder/'input.json'; atomic_json(path, request)
            report.update(source_checkpoint=bound['checkpoint'], source_executor_revision=bound['source_executor_revision'])
            supervisor = Supervisor(args.child_timeout, args.stall_timeout)
            report['child_executions'] = supervisor.records; report['results'] = {}
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', QKAN_DIAGNOSTIC_STAGE_FENCES='0')
            for mode in MODES:
                print('BF16 + Graph child: ' + mode, flush=True)
                output = folder/mode
                code = supervisor.run(child_spec(path, output, mode), folder/(mode+'.log'), env)
                for name in ('partial.json', 'worker-error.json'):
                    if (output/name).is_file(): report.setdefault('diagnostics', {})[mode+'/'+name] = read(output/name)
                if code != 0 or not (output/'result.json').is_file():
                    raise RuntimeError('BF16 Graph child failed; see compact diagnostics')
                result = read(output/'result.json'); report['results'][mode] = result
                accept(result, request, digest(path), mode, code)
                if mode == 'resume':
                    report['recovery'] = agreement(report['results'])
                    report['cleanup'] = remove_midpoint(folder, report['results']['interrupt'])
            report.update(passed=True, status='diagnostic_passed_formal_unchanged',
                disposable_optimizer_updates=16, processed_samples=2048, distinct_train_rows=256)
        finally:
            midpoint = folder/'interrupt/diagnostic-state.pt'
            if midpoint.is_file():
                report['private_midpoint_retention'] = dict(retained=True, bytes=midpoint.stat().st_size,
                    sha256=digest(midpoint), reason='Incomplete diagnostic; local only, never a formal source')
            unchanged(bound)
            report['original_checkpoint_and_metadata_unchanged'] = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-suite', 'cache', 'request', 'output'): p.add_argument('--'+name, type=Path)
    p.add_argument('--gpu', default='0'); p.add_argument('--child-timeout', type=float, default=900)
    p.add_argument('--stall-timeout', type=float, default=300); p.add_argument('--worker', action='store_true')
    p.add_argument('--mode', choices=MODES); args = p.parse_args()
    if args.worker:
        if args.request is None or args.output is None or args.mode is None: p.error('request/output/mode required')
        from .key_bias_recovery import prepare_worker_output
        prepare_worker_output(args.output)
        try:
            request = read(args.request)
            if (request.get('schema') != SCHEMA or request.get('implementation') != binding()
                    or request.get('source_revision') != revision()): raise ValueError('Code/plan/revision differs')
            unchanged(request)
            from .bf16_graph_native import native
            native(request, args.request, args.output, args.mode)
        except BaseException as error: worker_failure(args.output, error); raise
        return 0
    if args.source_suite is None or args.cache is None: p.error('source-suite/cache required')
    from .bootstrap import check_exchange
    from .return_artifacts import upload_declared
    from .diagnostic_watch import Supervisor
    Supervisor(args.child_timeout, args.stall_timeout); check_exchange()
    report = dict(schema=SCHEMA, source_revision=revision(), implementation=binding(), passed=False,
                  status='failed', **CLOSED)
    folder = ROOT/'runs'/('bf16-graph-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True); print('BF16 Graph diagnostic Suite: '+str(folder), flush=True); error = None
    try: execute(args, folder, report)
    except BaseException as caught:
        error = caught; worker_failure(folder, caught); report['failure_locations'] = read(folder/'worker-error.json')
    atomic_json(folder/'bf16-graph.json', report)
    try: upload_declared(folder, 'bf16-graph', ['bf16-graph.json'])
    except Exception:
        print('Return pending; local report/retry manifest retained: '+str(folder), flush=True)
        if error is None: raise
    if error is not None: raise error
    print('BF16 Graph diagnostic passed; formal executor unchanged. See bf16-graph.json.', flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
