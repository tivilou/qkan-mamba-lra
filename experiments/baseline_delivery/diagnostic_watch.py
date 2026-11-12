"""Opt-in, stdlib-only child supervision and bounded private stack collection.

Used by the parallel candidate diagnostic, never by the formal training runner.
The bootstrap installs faulthandler before importing the native worker. Phase
marks do not synchronize CUDA or inspect tensors/RNG/optimizer state. Explicit
synthetic stage-fence observation adds stream waits and cannot admit acceleration.
"""
from __future__ import annotations

import argparse
from collections import deque
import faulthandler
import math
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import threading
import time
import uuid

from .common import ROOT, atomic_json, read, spawn

_recorder = None


def safe_frame(filename, line, function):
    """Only code locations; never source lines, locals or arbitrary path text."""
    name = filename.replace('\\', '/')
    if '/experiments/baseline_delivery/' in name:
        name = 'experiments/baseline_delivery/' + name.rsplit('/experiments/baseline_delivery/', 1)[1]
    elif '/torch/' in name:
        name = 'torch/' + name.rsplit('/torch/', 1)[1]
    else:
        name = Path(name).name
    return dict(file=re.sub(r'[^a-zA-Z0-9_./-]', '_', name)[-200:], line=int(line),
                function=re.sub(r'[^a-zA-Z0-9_.<>-]', '_', function.strip())[:100])


def python_stacks():
    result=[]
    for ident, frame in list(sys._current_frames().items())[:8]:
        frames=[]
        while frame is not None and len(frames)<24:
            frames.append(safe_frame(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name))
            frame=frame.f_back
        result.append(dict(thread_id=ident, frames=frames))
    return result


def signal_stacks(path):
    """Parse a single faulthandler dump; raw text remains local and private."""
    try:
        with Path(path).open('rb') as handle: text=handle.read(128*1024).decode('utf8', errors='replace')
    except OSError:
        return []
    return [safe_frame(name, line, function) for name,line,function in
            re.findall(r'File "([^"\n]+)", line (\d+) in ([^\n]+)', text)[:64]]


class Recorder:
    def __init__(self, directory, run_id):
        self.directory=Path(directory); self.run_id=run_id
        self.events=deque(maxlen=64); self.mutex=threading.RLock()
        self.counter=0; self.phase='bootstrap'; self.phase_started=time.time()
        self.closed=threading.Event(); self.thread=None
        self.directory.mkdir(parents=True, exist_ok=True)

    def mark(self, phase, flush=False, document=None):
        if not re.fullmatch(r'[a-z0-9_]+', phase): raise ValueError('Invalid diagnostic phase')
        with self.mutex:
            self.counter+=1; self.phase=phase; self.phase_started=time.time()
            self.events.append(dict(phase=phase, at=self.phase_started, sequence=self.counter,
                                    **(dict(document=document) if document is not None else {})))
        if flush: self.publish()

    def publish(self):
        with self.mutex:
            value=dict(schema='qkan-diagnostic-progress.v1', run_id=self.run_id, pid=os.getpid(),
                       phase=self.phase, phase_started_at=self.phase_started,
                       progress_sequence=self.counter, observed_at=time.time(),
                       events=list(self.events), stacks=python_stacks())
            for attempt in range(5):
                try:
                    atomic_json(self.directory/'progress.json', value)
                    break
                except PermissionError:
                    # Windows readers briefly deny rename; the native executor
                    # is Linux. Retry only this bounded metadata publication.
                    if os.name!='nt' or attempt==4: raise
                    time.sleep(.01)

    def start(self):
        self.mark('worker_import', flush=True)
        def heartbeat():
            while not self.closed.wait(5): self.publish()
        self.thread=threading.Thread(target=heartbeat, name='diagnostic-heartbeat', daemon=True)
        self.thread.start()

    def close(self):
        self.closed.set()
        if self.thread: self.thread.join(timeout=1)


def mark(phase, *, flush=False, document=None):
    if _recorder is not None: _recorder.mark(phase, flush, document)


def cuda_fence(torch, phase, *, document=None):
    """Opt-in stage localization, never a performance measurement or admission.

    Only the supervised synthetic recovery worker may enable this mode. Publish
    before waiting so a GPU stall retains the exact stream-completion boundary.
    Ordinary diagnostics and training add no CUDA synchronization here.
    """
    if _recorder is None or os.environ.get('QKAN_DIAGNOSTIC_STAGE_FENCES')!='1': return
    mark(phase, flush=True, document=document)
    torch.cuda.current_stream().synchronize()
    mark(phase+'_complete', flush=True, document=document)


def process_group_members(pgid):
    """Linux live members of this Popen-created session, excluding zombies."""
    if not sys.platform.startswith('linux'): return None
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid and fields[0]!='Z': members.append(int(path.name))
        except (OSError, ValueError, IndexError): continue
    return sorted(members)


def terminate(child, grace=10):
    """Only the group created by spawn(); no name-based or GPU-wide killing."""
    escalated=False
    def send(sig):
        try: os.killpg(child.pid, sig)
        except ProcessLookupError: pass
    if os.name=='nt':
        # taskkill /T covers the child's descendants while its PID still exists.
        if child.poll() is None:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'], capture_output=True, timeout=grace+5)
        try: child.wait(timeout=grace)
        except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
        return dict(parent_reaped=child.poll() is not None, group_verified=False,
                    remaining_pids=None, verification='Linux process-group check required')
    send(signal.SIGTERM)
    end=time.monotonic()+grace
    while time.monotonic()<end:
        child.poll()
        members=process_group_members(child.pid)
        if members==[] and child.poll() is not None: break
        time.sleep(.1)
    members=process_group_members(child.pid)
    if members is None or members:
        escalated=True; send(signal.SIGKILL)
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired: pass
    end=time.monotonic()+5
    while process_group_members(child.pid) and time.monotonic()<end: time.sleep(.1)
    members=process_group_members(child.pid)
    return dict(parent_reaped=child.poll() is not None, group_verified=members==[] and child.poll() is not None,
                remaining_pids=members[:64] if members is not None else None, escalated=escalated)


class DiagnosticFailure(RuntimeError):
    def __init__(self, reason, diagnostic):
        super().__init__('Diagnostic child '+reason+'; inspect child-execution.json')
        self.diagnostic=diagnostic


class Supervisor:
    def __init__(self, child_timeout=900, stall_timeout=300, announce=print):
        for value in (child_timeout, stall_timeout):
            if isinstance(value,bool) or not math.isfinite(value) or value<=0:
                raise ValueError('Diagnostic timeouts must be finite and positive')
        self.child_timeout=child_timeout; self.stall_timeout=stall_timeout
        self.announce=announce; self.records=[]

    def run(self, spec, log_path, env=None):
        argv=list(map(str,spec['argv'])); module=argv.index('-m')+1
        target=argv[module]
        if target!=spec['target'] or not re.fullmatch(r'[a-zA-Z0-9_.]+',target):
            raise ValueError('Diagnostic worker target differs from launch specification')
        worker_args=argv[module+1:]
        if worker_args.count('--output')!=1: raise ValueError('Diagnostic worker needs one output directory')
        output=Path(worker_args[worker_args.index('--output')+1])
        run_id=uuid.uuid4().hex; directory=output/'diagnostics'/run_id
        directory.mkdir(parents=True,exist_ok=False)
        wrapped=dict(target='experiments.baseline_delivery.diagnostic_watch',cwd=spec['cwd'],
            argv=[*argv[:module], 'experiments.baseline_delivery.diagnostic_watch',
                  '--worker-module',target,'--run-id',run_id,'--directory',str(directory),'--',*worker_args])
        atomic_json(directory/'command.json',dict(launch=wrapped, worker=spec))
        log_path=Path(log_path); log_path.parent.mkdir(parents=True,exist_ok=True)
        started=time.monotonic(); advanced=started; sequence=None; next_notice=started
        last={}; reason=None; interruption=None; cleanup=None; dump_requested=False
        with log_path.open('ab') as log:
            child=spawn(wrapped,log,env)
            try:
                while child.poll() is None:
                    now=time.monotonic()
                    try:
                        progress=read(directory/'progress.json')
                        if progress.get('run_id')==run_id and progress.get('pid')==child.pid:
                            last=progress
                            if progress['progress_sequence']!=sequence:
                                sequence=progress['progress_sequence']; advanced=now
                    except (OSError,ValueError,KeyError): pass
                    if now>=next_notice:
                        self.announce('诊断 PID '+str(child.pid)+' | '+last.get('phase','worker_bootstrap')+
                            ' | 本阶段 '+str(int(now-advanced))+'s | 总计 '+str(int(now-started))+
                            's | 上限 '+str(self.child_timeout)+'s',flush=True)
                        next_notice=now+15
                    if now-started>=self.child_timeout: reason='child_timeout'; break
                    if now-advanced>=self.stall_timeout: reason='phase_timeout'; break
                    time.sleep(.2)
            except BaseException as error:
                reason='parent_interrupted'; interruption=error
            finally:
                if reason:
                    # faulthandler's C signal handler also works while Python's
                    # main thread is inside a native call. No SIGUSR2 without
                    # the bootstrap's matching registration receipt.
                    if os.name!='nt' and hasattr(signal,'SIGUSR2') and child.poll() is None:
                        try:
                            ready=read(directory/'stack-ready.json')
                            if ready==dict(pid=child.pid,run_id=run_id):
                                os.kill(child.pid,signal.SIGUSR2); dump_requested=True
                                time.sleep(1)
                        except (OSError,ValueError): pass
                    try: cleanup=terminate(child)
                    except Exception as error:
                        cleanup=dict(parent_reaped=child.poll() is not None,group_verified=False,
                                     remaining_pids=None,error_type=type(error).__name__)
                else:
                    members=process_group_members(child.pid)
                    if members:
                        try: cleanup=terminate(child)
                        except Exception as error:
                            cleanup=dict(parent_reaped=child.poll() is not None,group_verified=False,
                                         remaining_pids=None,error_type=type(error).__name__)
                    else: cleanup=dict(parent_reaped=True,group_verified=members==[],remaining_pids=members)
                # Reload after termination so a last heartbeat/phase is kept.
                try:
                    progress=read(directory/'progress.json')
                    if progress.get('run_id')==run_id and progress.get('pid')==child.pid: last=progress
                except (OSError,ValueError): pass
                record=dict(schema='qkan-diagnostic-child.v1',run_id=run_id,pid=child.pid,
                    outcome=reason or 'exited',
                    exit_code=child.poll(),elapsed_seconds=time.monotonic()-started,
                    child_timeout_seconds=self.child_timeout,stall_timeout_seconds=self.stall_timeout,
                    last_progress=last,signal_stack_requested=dump_requested,
                    signal_stack_frames=signal_stacks(directory/'stack-private.txt'),termination=cleanup,
                    privacy='Code locations only; no source lines, locals, tensors, raw log text or inputs')
                atomic_json(directory/'child-execution.json',record); self.records.append(record)
        if interruption: raise interruption
        if reason or not cleanup['parent_reaped'] or (sys.platform.startswith('linux') and not cleanup['group_verified']):
            raise DiagnosticFailure(reason or 'orphan_group',record)
        return child.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-module',required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('worker_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not re.fullmatch(r'[a-f0-9]{32}',args.run_id): parser.error('Invalid diagnostic run ID')
    global _recorder
    _recorder=Recorder(args.directory,args.run_id)
    stack=(args.directory/'stack-private.txt').open('xb')
    registered=False
    try:
        if os.name!='nt' and hasattr(signal,'SIGUSR2'):
            faulthandler.register(signal.SIGUSR2,file=stack,all_threads=True,chain=False);registered=True
            atomic_json(args.directory/'stack-ready.json',dict(pid=os.getpid(),run_id=args.run_id))
        _recorder.start()
        forwarded=args.worker_args[1:] if args.worker_args[:1]==['--'] else args.worker_args
        sys.argv=[args.worker_module,*forwarded]
        runpy.run_module(args.worker_module,run_name='__main__')
    finally:
        _recorder.close()
        if registered: faulthandler.unregister(signal.SIGUSR2)
        stack.close();_recorder=None


if __name__=='__main__':
    # Native modules import the canonical name; share this bootstrap's recorder
    # rather than creating an uninstrumented second module instance.
    sys.modules['experiments.baseline_delivery.diagnostic_watch']=sys.modules[__name__]
    main()
