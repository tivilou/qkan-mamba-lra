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
    return {'target': WORKER, 'cwd': str(ROOT),
            'argv': [sys.executable, '-B', '-m', WORKER, *map(str, args)]}


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
