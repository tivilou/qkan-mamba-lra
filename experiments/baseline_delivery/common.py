from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKER = 'experiments.baseline_delivery.worker'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def dictionary_path(root,task):
    root=Path(root)
    if (root/'synthetic-canary.json').is_file():
        return root/'metadata'/(task+'-dictionary.json')
    entry=read(ROOT/'experiments/baseline_delivery/data-identities.json')['dictionaries'][task]
    return root/entry['path']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.pending')
    with temp.open('w', encoding='utf8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def worker_failure(output, error):
    """Return code locations only; never exception messages, locals or arrays."""
    import re
    name=type(error).__name__
    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,79}',name): name='Exception'
    frames=[];tb=error.__traceback__
    while tb is not None:
        path=Path(tb.tb_frame.f_code.co_filename).resolve()
        if path.is_relative_to(ROOT) and path.suffix=='.py':
            frames.append(dict(file=path.relative_to(ROOT).as_posix(),line=tb.tb_lineno))
        tb=tb.tb_next
    atomic_json(Path(output)/'worker-error.json',dict(schema='qkan-worker-error.v1',
        error_type=name,project_frames=frames[-16:],
        privacy='Exception text, source lines, locals, inputs and arrays excluded'))


def revision():
    result = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if dirty:
        raise ValueError('Tracked source has local changes; use a clean, pinned checkout')
    return result


@contextmanager
def lock(path):
    """Kernel lock: released after crashes; stale PID files never grant ownership."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def child_spec(*args):
    args=list(map(str,args)); target=WORKER; interpreter=sys.executable
    if '--config' in args:
        config=read(args[args.index('--config')+1]); model=config['model']
        if model in ('mega','s5'):
            interpreter=str(ROOT/('.venv-'+model)/'bin/python')
            if model=='s5': target='experiments.baseline_delivery.s5_worker'
    return {'target': target, 'cwd': str(ROOT),
            'argv': [interpreter, '-B', '-m', target, *args]}


def stop_child(child, grace=90):
    if child.poll() is not None:
        return
    import signal
    if os.name == 'nt':
        child.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        if os.name == 'nt':
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'], capture_output=True)
        else:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=15)


def spawn(spec, log, env=None):
    if Path(spec['argv'][0]).absolute()==(ROOT/'.venv-s5/bin/python').absolute():
        env=dict(os.environ if env is None else env)
        site=ROOT/'.venv-s5/lib/python3.10/site-packages'
        env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,sorted((site/'nvidia').glob('*/lib'))))
        nvcc=site/'nvidia/cuda_nvcc'
        env['PATH']=str(nvcc/'bin')+os.pathsep+env.get('PATH','')
        env['XLA_FLAGS']='--xla_gpu_cuda_data_dir='+str(nvcc)
        env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    kwargs = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    return subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=env,
                            stdout=log, stderr=subprocess.STDOUT, **kwargs)


def run_child(spec, log_path, env=None):
    log_path = Path(log_path); log_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(log_path.with_suffix('.command.json'), spec)
    with log_path.open('ab') as log:
        child = spawn(spec, log, env)
        try:
            return child.wait()
        finally:
            stop_child(child)
