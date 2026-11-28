"""Unified experiment operations; preserve pinned executors and original UI.

May be fetched alongside formal_progress.py into ignored .bootstrap/. No native
imports or install in this controller. Existing entrypoints remain compatible.
"""
import argparse
from contextlib import ExitStack
import datetime as dt
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
from types import FunctionType
import uuid

PROFILES = {
    'formal': [], 'adaptive': ['--adaptive-memory'],
    'dynamic': ['--adaptive-memory', '--dynamic-logical-batch'],
    'performance': ['--adaptive-memory', '--dynamic-logical-batch', '--performance'],
    'accelerated': ['--adaptive-memory', '--dynamic-logical-batch', '--performance', '--torch-acceleration'],
    'parallel': ['--adaptive-memory', '--dynamic-logical-batch', '--performance', '--torch-acceleration', '--parallel-retrieval-graph'],
    'key-bias': [], 'bf16-graph': [], 'mamba-fused': [],
}
DIAGNOSTICS = {
    'attention-update': 'attention_update_diagnostic', 'parallel-attention': 'attention_diagnostic',
    'partitioned-attention': 'partitioned_diagnostic', 'operator': 'operator_diagnostic',
    'precision': 'attention_precision_diagnostic', 'padding': 'padding_profile',
    'mamba-speed': 'mamba_diagnostic',
    'mega-fft': 'mega_diagnostic',
    'mega-qualify': 'mega_qualification',
    'roster-speed': 'roster_speed',
    'mamba-validate': 'mamba_validation',
    'bf16-graph': 'bf16_graph_diagnostic',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def confined_suite(root, suite, existing=True):
    path = Path(suite)
    if not path.is_absolute(): path = root/path
    path = path.absolute()
    if path.parent.resolve() != (root/'runs').resolve():
        raise ValueError('Suite must be directly under this executor runs/')
    if path.is_symlink() or path.resolve() != path:
        raise ValueError('Suite links/aliases are not accepted')
    if existing and not path.is_dir(): raise ValueError('Existing Suite required')
    return path


def validate_executor(root, suite):
    fused = suite/'mamba-input.json'
    graph = suite/'graph-input.json'
    formal = suite/'formal-input.json'
    if fused.exists():
        value = read(fused)
        if value.get('schema') not in ('qkan-mamba-fused-formal.v1','qkan-mamba-fused-formal.v2'):
            raise ValueError('Unknown Mamba fused Suite schema')
        expected = value['source_revision']
    elif graph.exists():
        value = read(graph)
        if value.get('schema') != 'qkan-bf16-graph-formal-input.v1':
            raise ValueError('Unknown Graph formal Suite schema')
        expected = value['source_revision']
    elif formal.exists():
        value = read(formal)
        if value.get('schema') != 'qkan-key-bias-formal-input.v1':
            raise ValueError('Unknown formal Suite schema')
        expected = value['source_revision']
    else:
        context, manifest, gate = (read(suite/n) for n in ('context.json', 'suite.json', 'gate-report.json'))
        expected = context['source_revision']
        if gate.get('status') != 'passed' or any(v['source_revision'] != expected for v in (manifest, gate)):
            raise ValueError('Legacy Suite must already be admitted')
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if actual != expected or dirty: raise ValueError('Use unchanged original executor revision '+expected)
    # The original child validates all scientific bindings, data and receipts.
    return 'mamba-fused' if fused.exists() else 'bf16-graph' if graph.exists() else 'key-bias' if formal.exists() else 'admitted'


def module(name, *args):
    return [sys.executable, '-B', '-m', 'experiments.baseline_delivery.'+name, *map(str, args)]


def queue_record(suite, task_order):
    record=dict(schema='qkan-task-order.v1', policy=task_order,
        manifest_sha256=hashlib.sha256((suite/'suite.json').read_bytes()).hexdigest())
    if task_order=='fastest-first':
        path=Path(__file__).resolve().parents[1]/'docs/baselines/runtime-order-20261128.json'
        evidence=read(path)
        if evidence['manifest_sha256']!=record['manifest_sha256']:raise ValueError('Runtime estimates belong to a different Suite')
        for job,entry in evidence['jobs'].items():
            if hashlib.sha256((suite/'jobs'/job/'config.json').read_bytes()).hexdigest()!=entry['config_sha256']:
                raise ValueError('Runtime estimate scientific configuration changed: '+job)
        record.update(estimates_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            remaining_seconds={name:v['remaining_seconds'] for name,v in evidence['jobs'].items()})
    return record


def queue_read(suite, original_read):
    """Private dispatcher view; validators continue reading the original bytes."""
    record = read(suite/'task-order.json')
    if record != queue_record(suite, record['policy']) or record['policy'] not in ('manifest', 'retrieval-last','fastest-first'):
        raise ValueError('Task order must match the original Suite manifest')
    completed=set()
    if record['policy']=='fastest-first':
        from experiments.baseline_delivery.migration import completed_files,verify_completed
        for job in original_read(suite/'suite.json')['jobs']:
            folder=suite/'jobs'/job['id']
            if (folder/'summary.json').exists():
                if job.get('preserved_completion'):verify_completed(suite,job)
                else:completed_files(folder,original_read(folder/'config.json'))
                completed.add(job['id'])
            elif job['id'] not in record['remaining_seconds']:
                raise ValueError('Unfinished job has no reviewed time estimate: '+job['id'])
    def ordered(path):
        value = original_read(path)
        if Path(path).resolve() == (suite/'suite.json').resolve() and record['policy'] == 'retrieval-last':
            value = dict(value, jobs=sorted(value['jobs'], key=lambda job: '-retrieval-s' in job['id']))
        if record['policy']=='fastest-first':
            if Path(path).resolve()==(suite/'suite.json').resolve():
                value=dict(value,jobs=sorted(value['jobs'],key=lambda j:0 if j['id'] in completed else record['remaining_seconds'][j['id']]))
            elif Path(path).resolve()==(suite/'status.json').resolve():
                # Result validation succeeded above; recover stale exit1 from a
                # post-summary transport failure without launching its worker.
                value={name:dict(state,**({'state':'computed'} if name in completed else {})) for name,state in value.items()}
        return value
    return ordered


def private_function(function, **overrides):
    """Bind only this controller call; never replace shared module globals."""
    result = FunctionType(function.__code__, dict(function.__globals__, **overrides),
        function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def mega_control(suite=None):
    # Load without importing the new experiments package into the pinned parent.
    path=Path(__file__).resolve().parents[1]/'experiments/baseline_delivery/mega_formal.py'
    if suite is not None and (suite/'mega-policy.json').exists():
        path=Path(read(suite/'mega-policy.json')['code_root'])/'experiments/baseline_delivery/mega_formal.py'
    return runpy.run_path(str(path))


def transport_spec(builder):
    """A private dispatch seam: run the same pinned worker with transport preload."""
    def wrapped(suite,record,job):
        spec=builder(suite,record,job)
        if spec.get('target')!='experiments.baseline_delivery.mega_formal':return spec
        argv=spec['argv'];index=argv.index('-m')
        if argv[index+1]!=spec['target']:raise ValueError('Pinned worker target differs')
        helper=Path(__file__).resolve().with_name('pinned_mega_worker.py')
        return dict(spec,argv=[*argv[:index],str(helper),'--root',spec['cwd'],'--',*argv[index+2:]],
            transport_wrapper=dict(path=str(helper),sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                scope='preload pinned exchange client; identical worker and science bindings'))
    return wrapped


def dispatch_queue(root, suite):
    kind = validate_executor(root, suite)
    targets = {'mamba-fused': 'mamba_fused_formal', 'bf16-graph': 'bf16_graph_formal',
               'key-bias': 'key_bias_formal'}
    if kind not in targets: raise ValueError('Task ordering requires a formal continuation Suite')
    sys.path.insert(0, str(root))
    target = importlib.import_module('experiments.baseline_delivery.'+targets[kind])
    execute = private_function(target.execute, read=queue_read(suite, target.read))
    if (suite/'mega-policy.json').exists():
        if kind!='mamba-fused':raise ValueError('MEGA amendment requires the reviewed Mamba continuation Suite')
        execute=mega_control(suite)['attach'](target,root,suite,queue_read(suite,target.read))
        execute=private_function(execute,job_spec=transport_spec(execute.__globals__['job_spec']))
    entry = private_function(target.main, execute=execute)
    previous = sys.argv
    try:
        sys.argv = [targets[kind], '--resume', str(suite)]
        return entry()
    finally:
        sys.argv = previous


def resume_command(root, suite, upload=False, task_order=None):
    kind = validate_executor(root, suite)
    if task_order is not None and not upload:
        if kind not in ('mamba-fused', 'bf16-graph', 'key-bias'):
            raise ValueError('Task ordering requires a formal continuation Suite')
        return [sys.executable, '-B', str(Path(__file__).resolve()), '--root', str(root),
                '_dispatch-queue', '--suite', str(suite)]
    if kind == 'mamba-fused':
        return module('mamba_fused_formal', '--resume', suite, *(['--upload-only'] if upload else []))
    if kind == 'bf16-graph':
        return module('bf16_graph_formal', '--resume', suite, *(['--upload-only'] if upload else []))
    if kind == 'key-bias':
        return module('key_bias_formal', '--resume', suite, *(['--upload-only'] if upload else []))
    if upload: return module('runner', '--mode', 'upload', '--resume', suite)
    return [sys.executable, '-B', str(root/'scripts/run_admitted.py'), '--root', str(root), '--resume', str(suite)]


def start_command(args, root, suite):
    graph_from = getattr(args, 'graph_from', None)
    if args.profile == 'mamba-fused':
        if not args.migrate_from or graph_from or args.worker_from or args.output:
            raise ValueError('mamba-fused requires migrate-from/cache and allocates its own Suite')
        return module('mamba_fused_formal', '--source-suite', args.migrate_from,
            '--cache', args.cache, '--gpu', args.gpu, '--prepare-only')
    if args.profile == 'bf16-graph':
        if not args.migrate_from or not graph_from or args.worker_from or args.output:
            raise ValueError('bf16-graph requires migrate-from/graph-from and allocates its own Suite')
        return module('bf16_graph_formal', '--source-suite', args.migrate_from,
            '--diagnostic-from', graph_from, '--cache', args.cache, '--gpu', args.gpu, '--prepare-only')
    if graph_from: raise ValueError('graph-from belongs to bf16-graph only')
    if args.profile == 'key-bias':
        if not args.migrate_from or not args.worker_from:
            raise ValueError('key-bias start requires migrate-from and worker-from')
        if args.output: raise ValueError('key-bias allocates its own new Suite; output is not accepted')
        return module('key_bias_formal', '--migrate-from', args.migrate_from,
            '--worker-from', args.worker_from, '--cache', args.cache, '--gpu', args.gpu, '--prepare-only')
    if args.worker_from: raise ValueError('worker-from belongs to key-bias only')
    if args.migrate_from and args.profile == 'formal': raise ValueError('Migration requires an explicit non-formal profile')
    command = module('readiness', '--output', suite, '--cache', args.cache, '--gpus', args.gpu,
                     '--gates-only', *PROFILES[args.profile])
    if args.migrate_from: command += ['--migrate-from', str(args.migrate_from)]
    return command


def display_module():
    path = Path(__file__).resolve().with_name('formal_progress.py')
    if not path.is_file(): raise ValueError('Keep experiment.py and formal_progress.py together')
    return runpy.run_path(str(path))


def discover_created(root, log, offset):
    with log.open('rb') as handle:
        handle.seek(offset); text = handle.read(65536).decode('utf8', errors='replace')
    for line in text.splitlines():
        if line.startswith('Formal continuation Suite: '):
            return confined_suite(root, line.removeprefix('Formal continuation Suite: '))
    return None


def require_linux():
    if os.name != 'posix': raise ValueError('Training supervision requires collaborator Linux runtime')


def supervise(root, suite, command, ui, interval=5, display='auto', after_prepare=False, task_order=None, mega_policy=None):
    """Own one parent, original UI, safe pause and logs; never own scientific state."""
    sys.path.insert(0, str(root))
    from experiments.baseline_delivery.common import lock, atomic_json
    require_linux()
    terminal = ui['Terminal'](display); child = None; pausing = False
    dashboard = None; started = time.time(); launch = uuid.uuid4().hex
    log = (suite/'progress-parent.log') if suite and suite.exists() else root/'.bootstrap'/('experiment-'+launch+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    def pause(*_):
        nonlocal pausing
        if not pausing:
            pausing = True
            if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)
    signals = {s: signal.signal(s, pause) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with ExitStack() as held:
            held.enter_context(lock(suite/'progress-supervisor.lock' if suite and suite.exists()
                                    else root/'.bootstrap/experiment-start.lock'))
            if suite and suite.exists():
                with lock(suite/'suite.lock'): pass
            if mega_policy is not None:
                validate_executor(root,suite)
                with lock(suite/'suite.lock'),lock(suite/'jobs/mega-text-s42/worker.lock'):
                    mega_control(suite)['activate'](root,suite,mega_policy)
            with log.open('ab') as output:
                offset = output.tell()
                while True:
                    if pausing: return 75
                    if task_order is not None and not after_prepare:
                        # Only scheduler metadata changes, while no formal parent owns the Suite.
                        with lock(suite/'suite.lock'):
                            atomic_json(suite/'task-order.json', queue_record(suite, task_order))
                    child = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                        start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                            PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                            XLA_PYTHON_CLIENT_PREALLOCATE='false',
                            QKAN_MEGA_SOURCE=str(root/'.bootstrap/strong-upstream/mega'),
                            QKAN_S5_SOURCE=str(root/'.bootstrap/strong-upstream/s5')))
                    if pausing and child.poll() is None: child.send_signal(signal.SIGTERM)
                    next_frame = 0
                    while child.poll() is None:
                        if suite is None: suite = discover_created(root, log, offset)
                        if time.monotonic() >= next_frame:
                            if suite and suite.exists():
                                if dashboard is None: dashboard = ui['Dashboard'](root, suite)
                                dashboard.launch = dict(started_at=started, runner_pid=child.pid,
                                    updated_at=time.time(), interval=interval)
                                terminal.show(dashboard.render()+'\n统一入口日志：'+str(log))
                                atomic_json(suite/'display-heartbeat.json', dict(
                                    schema='qkan-unified-supervisor.v1', started_at=started, updated_at=time.time(),
                                    runner_pid=child.pid, supervisor_pid=os.getpid(), interval=interval,
                                    pause_requested=pausing, log_path=str(log)))
                            else: terminal.show('正在准备正式实验；尚未确认训练派发。\n日志：'+str(log))
                            next_frame = time.monotonic()+interval
                        time.sleep(.2)
                    if pausing: return 75
                    if child.returncode != 0 or not after_prepare:
                        return child.returncode if child.returncode >= 0 else 128-child.returncode
                    if suite is None: suite = discover_created(root, log, offset)
                    if suite is None: raise ValueError('Preparation returned without an owned Suite')
                    held.enter_context(lock(suite/'progress-supervisor.lock'))
                    command = (resume_command(root, suite, task_order=task_order) if task_order is not None
                               else resume_command(root, suite))
                    after_prepare = False
    finally:
        terminal.close()
        try:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM); child.wait()
        finally:
            for sig, handler in signals.items(): signal.signal(sig, handler)
        print('实验入口已退出；日志：'+str(log), flush=True)
        if suite: print('Suite: '+str(suite), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd(), help='Original pinned executor; default current directory')
    subs = p.add_subparsers(dest='operation', required=True)
    start = subs.add_parser('start', help='Prepare once, then train with original progress dashboard')
    start.add_argument('--profile', choices=PROFILES, default='formal')
    start.add_argument('--cache', type=Path, required=True)
    start.add_argument('--migrate-from', type=Path); start.add_argument('--worker-from', type=Path)
    start.add_argument('--graph-from', type=Path, help='Already audited native BF16 Graph diagnostic Suite')
    start.add_argument('--output', type=Path); start.add_argument('--prepare-only', action='store_true')
    start.add_argument('--gpu', default='0')
    for name in ('resume', 'watch', 'upload'):
        sub = subs.add_parser(name)
        sub.add_argument('--suite', type=Path, required=True)
        if name == 'watch': sub.add_argument('--once', action='store_true')
    subs.choices['resume'].add_argument('--task-order', choices=('manifest', 'retrieval-last','fastest-first'),
        help='Use manifest, Retrieval last, or reviewed fastest remaining first; retain recorded choice on resume')
    subs.choices['resume'].add_argument('--mega-policy',choices=('fft-tail-bf16','original'),
        help='Explicit prospective MEGA Text switch or rollback; latest checkpoint and same dashboard')
    dispatch = subs.add_parser('_dispatch-queue', help=argparse.SUPPRESS)
    dispatch.add_argument('--suite', type=Path, required=True)
    diagnostic = subs.add_parser('diagnose', help='Explicit diagnostic only; never automatic during resume')
    diagnostic.add_argument('kind', choices=DIAGNOSTICS)
    diagnostic.add_argument('arguments', nargs=argparse.REMAINDER)
    for sub in (start, subs.choices['resume'], subs.choices['watch']):
        sub.add_argument('--display', choices=('auto', 'inplace', 'append'), default='auto')
        sub.add_argument('--interval', type=float, default=5)
    args = p.parse_args(argv); root = args.root.resolve()
    if hasattr(args, 'interval') and (not math.isfinite(args.interval) or not 1 <= args.interval <= 60):
        p.error('interval must be 1..60 seconds')
    try:
        if args.operation == 'diagnose':
            tail = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
            return subprocess.call(module(DIAGNOSTICS[args.kind], *tail), cwd=root)
        if args.operation == 'start':
            suite = None if args.profile in ('key-bias','bf16-graph','mamba-fused') else confined_suite(root,
                args.output or ('runs/formal-ready-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]), False)
            if suite and suite.exists(): raise ValueError('New start needs a NEW Suite; use resume for existing progress')
            command = start_command(args, root, suite)
            return supervise(root, suite, command, display_module(), args.interval, args.display, not args.prepare_only)
        suite = confined_suite(root, args.suite)
        if args.operation == '_dispatch-queue': return dispatch_queue(root, suite)
        if args.operation == 'watch':
            ui = display_module(); dashboard = ui['Dashboard'](root, suite); terminal = ui['Terminal'](args.display)
            try:
                while True:
                    if args.once: print(dashboard.render(), flush=True); return 0
                    terminal.show(dashboard.render()); time.sleep(args.interval)
            except KeyboardInterrupt: return 0
            finally: terminal.close()
        task_order = args.task_order if args.operation == 'resume' else None
        if args.operation == 'resume' and task_order is None and any(
                (suite/name).is_file() for name in ('mamba-input.json', 'graph-input.json', 'formal-input.json')):
            task_order = read(suite/'task-order.json')['policy'] if (suite/'task-order.json').exists() else 'retrieval-last'
        command = resume_command(root, suite, args.operation == 'upload', task_order=task_order)
        if args.operation == 'upload': return subprocess.call(command, cwd=root)
        return supervise(root, suite, command, display_module(), args.interval, args.display, task_order=task_order,
            **(dict(mega_policy=args.mega_policy) if args.mega_policy is not None else {}))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print('入口检查未通过：'+str(error), file=sys.stderr); return 2


if __name__ == '__main__': raise SystemExit(main())
