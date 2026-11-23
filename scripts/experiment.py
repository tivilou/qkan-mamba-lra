"""Unified experiment operations; preserve pinned executors and original UI.

May be fetched alongside formal_progress.py into ignored .bootstrap/. No native
imports or install in this controller. Existing entrypoints remain compatible.
"""
import argparse
from contextlib import ExitStack
import datetime as dt
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import uuid

PROFILES = {
    'formal': [], 'adaptive': ['--adaptive-memory'],
    'dynamic': ['--adaptive-memory', '--dynamic-logical-batch'],
    'performance': ['--adaptive-memory', '--dynamic-logical-batch', '--performance'],
    'accelerated': ['--adaptive-memory', '--dynamic-logical-batch', '--performance', '--torch-acceleration'],
    'parallel': ['--adaptive-memory', '--dynamic-logical-batch', '--performance', '--torch-acceleration', '--parallel-retrieval-graph'],
    'key-bias': [], 'bf16-graph': [],
}
DIAGNOSTICS = {
    'attention-update': 'attention_update_diagnostic', 'parallel-attention': 'attention_diagnostic',
    'partitioned-attention': 'partitioned_diagnostic', 'operator': 'operator_diagnostic',
    'precision': 'attention_precision_diagnostic', 'padding': 'padding_profile',
    'mamba-speed': 'mamba_diagnostic',
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
    graph = suite/'graph-input.json'
    formal = suite/'formal-input.json'
    if graph.exists():
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
    return 'bf16-graph' if graph.exists() else 'key-bias' if formal.exists() else 'admitted'


def module(name, *args):
    return [sys.executable, '-B', '-m', 'experiments.baseline_delivery.'+name, *map(str, args)]


def resume_command(root, suite, upload=False):
    kind = validate_executor(root, suite)
    if kind == 'bf16-graph':
        return module('bf16_graph_formal', '--resume', suite, *(['--upload-only'] if upload else []))
    if kind == 'key-bias':
        return module('key_bias_formal', '--resume', suite, *(['--upload-only'] if upload else []))
    if upload: return module('runner', '--mode', 'upload', '--resume', suite)
    return [sys.executable, '-B', str(root/'scripts/run_admitted.py'), '--root', str(root), '--resume', str(suite)]


def start_command(args, root, suite):
    graph_from = getattr(args, 'graph_from', None)
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


def supervise(root, suite, command, ui, interval=5, display='auto', after_prepare=False):
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
            with log.open('ab') as output:
                offset = output.tell()
                while True:
                    if pausing: return 75
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
                    command = resume_command(root, suite)
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
            suite = None if args.profile in ('key-bias','bf16-graph') else confined_suite(root,
                args.output or ('runs/formal-ready-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]), False)
            if suite and suite.exists(): raise ValueError('New start needs a NEW Suite; use resume for existing progress')
            command = start_command(args, root, suite)
            return supervise(root, suite, command, display_module(), args.interval, args.display, not args.prepare_only)
        suite = confined_suite(root, args.suite)
        if args.operation == 'watch':
            ui = display_module(); dashboard = ui['Dashboard'](root, suite); terminal = ui['Terminal'](args.display)
            try:
                while True:
                    if args.once: print(dashboard.render(), flush=True); return 0
                    terminal.show(dashboard.render()); time.sleep(args.interval)
            except KeyboardInterrupt: return 0
            finally: terminal.close()
        command = resume_command(root, suite, args.operation == 'upload')
        if args.operation == 'upload': return subprocess.call(command, cwd=root)
        return supervise(root, suite, command, display_module(), args.interval, args.display)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print('入口检查未通过：'+str(error), file=sys.stderr); return 2


if __name__ == '__main__': raise SystemExit(main())
