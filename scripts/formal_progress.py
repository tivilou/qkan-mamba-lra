"""Readable supervisor for an already admitted, revision-pinned formal suite.

This file can be fetched into .bootstrap/ without changing the executor checkout.
Only the original readiness entry owns training, checkpoints and admission.
"""
import argparse
from collections import deque
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import unicodedata


def resume_command(root,suite,context,cache):
    entry=Path(__file__).resolve().with_name('run_admitted.py')
    if entry.is_file():
        return [sys.executable,'-B',str(entry),'--root',str(root),'--resume',str(suite)]
    command=['bash',str(root/'scripts/run_formal_readiness.sh'),'--cache',str(cache),
             '--gpus',context['gpus'],'--resume',str(suite)]
    for key in ('adaptive_memory','dynamic_logical_batch','performance','torch_acceleration'):
        if context.get(key): command.append('--'+key.replace('_','-'))
    for key in ('require_graph','require_acceleration'):
        for job in context.get(key,[]): command.extend(['--'+key.replace('_','-'),job])
    return command


def fit_line(text, columns):
    """Bound terminal cells, including Chinese, so redraw never wraps/scrolls."""
    output=[];used=0;limit=max(1,columns-1)
    for char in text:
        if ord(char)<32 or ord(char)==127: continue
        width=0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in 'WF' else 1)
        if used+width>limit:
            while output and used>limit-1:
                removed=output.pop();used-=0 if unicodedata.combining(removed) else (2 if unicodedata.east_asian_width(removed) in 'WF' else 1)
            return ''.join(output)+'…'
        output.append(char);used+=width
    return ''.join(output)


class Terminal:
    def __init__(self, mode):
        self.live=mode!='append' and sys.stdout.isatty() and os.environ.get('TERM','')!='dumb'
        self.entered=False

    def show(self, text):
        if not self.live:
            print(text,flush=True);return
        if not self.entered:
            sys.stdout.write('\x1b[?1049h\x1b[?25l')
            self.entered=True
        size=shutil.get_terminal_size(fallback=(100,30))
        lines=text.strip('\n').splitlines();height=max(1,size.lines-1)
        if len(lines)>height:
            lines=lines[:max(0,height-1)]+['窗口高度不足；放大窗口，或使用 --display append 查看完整快照。']
        frame='\n'.join(fit_line(line,size.columns) for line in lines)
        sys.stdout.write('\x1b[H\x1b[J'+frame);sys.stdout.flush()

    def close(self):
        if self.entered:
            sys.stdout.write('\x1b[?25h\x1b[?1049l');sys.stdout.flush()
            self.entered=False


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf8'))
    except (OSError,ValueError):
        return default


def duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds<0: return '待测'
    seconds=int(seconds);hours,seconds=divmod(seconds,3600);minutes,seconds=divmod(seconds,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class Throughput:
    """Count committed work against worker timestamps, independently of redraws."""
    def __init__(self,window=60.):
        self.window=window;self.points=deque(maxlen=256);self.origin=None;self.scope='本次观察'
        self.short=None

    def observe(self,count,stamp,now,origin=None):
        if (not isinstance(count,int) or count<0 or not isinstance(stamp,(int,float))
                or not math.isfinite(stamp) or stamp>now+1): return self.result(now)
        if self.points and stamp<self.points[-1][0]: return self.result(now)
        if self.points and count<self.points[-1][1]:
            self.points.clear();self.origin=None;self.short=None;self.scope='本次观察'
        if not self.points:
            self.origin=(stamp,count)
            if (origin and isinstance(origin[0],(int,float)) and math.isfinite(origin[0])
                    and isinstance(origin[1],int) and 0<=origin[1]<=count and origin[0]<=stamp):
                self.origin=origin;self.scope='本次进程'
        if not self.points or stamp>self.points[-1][0]:
            self.points.append((stamp,count))
            cutoff=stamp-self.window
            while len(self.points)>2 and self.points[1][0]<=cutoff: self.points.popleft()
            elapsed=stamp-self.points[0][0]
            if elapsed>0: self.short=(count-self.points[0][1])/elapsed
        return self.result(now)

    def result(self,now):
        if not self.points: return dict(short=None,average=None,scope=self.scope,age=None)
        stamp,count=self.points[-1];elapsed=now-self.origin[0]
        # Repeated redraws do not erase short throughput. Average includes waits.
        average=(count-self.origin[1])/elapsed if elapsed>0 and (count>self.origin[1] or self.short is not None) else None
        return dict(short=self.short,average=average,scope=self.scope,age=max(0,now-stamp))


def speed_text(value,unit,precision=2):
    return f'{value:.{precision}f} {unit}' if value is not None else '采集中'


def speed_lines(stats,unit,remaining,precision=2):
    stale=stats['age'] is not None and stats['age']>120
    label='最近测量，心跳已过期' if stale else '近60s平滑，窗口逐步积累'
    lines=[f"  短期平滑速度 {speed_text(stats['short'],unit,precision)}（{label}）",
           f"  长期平均速度 {speed_text(stats['average'],unit,precision)}（{stats['scope']}，含等待/保存）"]
    average=stats['average']
    eta=remaining/average if average and remaining>=0 and not stale else None
    estimate='暂停估计（心跳已过期）' if stale else (duration(eta) if eta is not None else '采集中')
    lines.append('  当前任务剩余 ETA '+estimate+'（按长期平均；不含后续验证/测试/回传及排队任务）')
    return lines


def backend_text(acceleration,admission):
    mode=acceleration['mode'];reason=acceleration.get('reason','unknown')
    attempts=admission.get('attempts',[]) if isinstance(admission,dict) else []
    if mode=='eager' and reason=='requested' and any(a.get('outcome')=='recovery_mismatch' for a in attempts):
        reason='编译/Graph未通过恢复校验，按已准入eager执行'
    compiler=admission.get('selected_backend') if isinstance(admission,dict) else None
    if compiler=='eager' and mode=='compile_graph': compiler='eager（Dynamo原生autograd＋CUDA Graph）'
    required=admission.get('required') if isinstance(admission,dict) else None
    return mode+(' / '+compiler if compiler and mode!='eager' else '')+' | '+reason+(' | 必须启用Graph' if required=='graph' else '')


def gpu_usage():
    """One bounded, read-only board query per frame; no CUDA/framework import."""
    try:
        result=subprocess.run(['nvidia-smi',
            '--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,check=True)
    except (OSError,subprocess.SubprocessError):
        return {},'不可用（nvidia-smi 缺失、超时或查询失败）'
    def number(value,maximum=None):
        try: parsed=float(value.strip())
        except ValueError: return None
        return parsed if math.isfinite(parsed) and parsed>=0 and (maximum is None or parsed<=maximum) else None
    metrics={}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row)!=4 or not row[0].strip().isdigit(): continue
        index,util,used,total=row
        metrics[str(int(index))]=(number(util,100),number(used),number(total))
    return metrics,None


def gpu_line(index,metrics,error):
    if index not in metrics:
        return f'GPU {index} 整卡占用：'+(error or '不可用（未返回该物理 GPU）')
    util,used,total=metrics[index]
    utilization=f'{util:.0f}%' if util is not None else '不可用'
    memory=(f'{used/1024:.2f}/{total/1024:.2f} GiB ({100*used/total:.1f}%)'
            if used is not None and total and used<=total else '不可用')
    return f'GPU {index} 整卡占用：利用率 {utilization} | 显存 {memory}'


class Dashboard:
    def __init__(self, root, suite):
        self.root=Path(root);self.suite=Path(suite);self.rates={}
        self.launch=None
        build=read(self.root/'docs/evidence/mega-document-v3-build.json',{})
        self.rows={t['task']:t['splits']['train']['arrays']['labels']['shape'][0]
                   for t in build.get('tasks',[])}

    def render(self, now=None):
        now=time.time() if now is None else now
        manifest=read(self.suite/'suite.json',{})
        states=read(self.suite/'status.json',{})
        jobs=manifest.get('jobs',[])
        names=[j['id'] for j in jobs]
        groups={name:[] for name in ('computed','running','pending','failed','paused')}
        for job in names:
            value=states.get(job,{}).get('state','pending')
            groups.setdefault(value,[]).append(job)
        clock=dt.datetime.fromtimestamp(now,dt.timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        lines=[f'\n[{clock}] 实验：{self.suite.name}',
               f"任务 {len(names)} | 已计算 {len(groups['computed'])} | 运行 {len(groups['running'])} | "
               f"排队 {len(groups['pending'])} | 失败 {len(groups['failed'])} | 暂停 {len(groups['paused'])}"]
        preserved=[job['id'] for job in jobs if job.get('preserved_completion')]
        if preserved:
            lines.append('沿用已完成结果（保留原版本，不重跑）：'+', '.join(preserved))
        gpu_ids={str(gpu) for gpu in manifest.get('gpu_inventory',{})}
        gpu_ids.update(str(states[job]['physical_gpu']) for job in groups['running']
                       if 'physical_gpu' in states[job])
        if gpu_ids:
            metrics,error=gpu_usage()
            for gpu in sorted(gpu_ids): lines.append(gpu_line(gpu,metrics,error))
        launch=self.launch or read(self.suite/'display-heartbeat.json',{})
        if self.launch is None:
            lines.append('模式：只读观察；不会自动启动或恢复暂停任务。')
        if launch:
            pid=launch.get('runner_pid');alive=False
            if isinstance(pid,int) and pid>0:
                try: os.kill(pid,0);alive=True
                except PermissionError: alive=True
                except OSError: pass
            fresh=now-launch.get('updated_at',0)<max(60,launch.get('interval',15)*3)
            if alive and fresh:
                elapsed=duration(now-launch['started_at']) if 'started_at' in launch else '未知'
                lines.append(f'恢复入口：PID {pid} 存活 | 入口耗时 {elapsed}')
                if not groups['running'] and not computed_suite(self.suite):
                    lines.append('当前阶段：恢复入口运行中；尚未派发训练。下方暂停/排队是上次保存的任务状态。')
                    lines.extend(startup_lines(self.suite,launch,now))
            elif self.launch is None and not groups['running']:
                lines.append('恢复入口：未确认存活；仅打开观察器不能让暂停任务继续。')
        elif not groups['running'] and groups['paused']:
            lines.append('恢复入口：无启动心跳；需要实际续跑入口，观察器本身不恢复训练。')
        if not jobs:
            probes=sum(bool(read(p,{}).get('passed')) for p in self.suite.glob('*-source-probe.json'))
            canaries=sum(bool(read(p,{}).get('passed')) for p in self.suite.glob('preflight/*/*/canary.json'))
            lines.append(f'当前阶段：准备/验收；源码检查 {probes}/5，任务验收 {canaries}/25。')
        for job in groups['running']:
            state=states[job];folder=self.suite/'jobs'/job
            config=read(folder/'config.json',{});hb=read(folder/'heartbeat.json',{})
            started=state.get('started',now);age=max(0,now-hb.get('updated_at',started))
            lines.append(f"GPU {state.get('physical_gpu','?')} | {job} | PID {state.get('pid','?')} | 本次进程耗时 {duration(now-started)}")
            if not hb or hb.get('updated_at',0)<started:
                lines.append('  初始化/恢复中，等待本次进程的第一个训练心跳；速度与 ETA 待测。')
                acceleration=read(folder/'torch-acceleration.json',{})
                if acceleration:
                    lines.append('  编译/Graph准备：'+acceleration.get('state',{}).get('phase',acceleration.get('state',{}).get('reason','待测'))+
                        '；首次编译/捕获及有界测速可能较慢，尚未计入样本进度。')
                continue
            adaptive=hb.get('adaptive_memory')
            if adaptive:
                lines.append(f"  显存适配：本步 micro-batch {adaptive['used_micro_batch']} | 下一步 {adaptive['micro_batch']} | "
                             f"累计 OOM 重试 {adaptive['oom_retries']} | {adaptive['event']}")
                if 'logical_batch' in adaptive:
                    lines.append(f"  逻辑 batch：本步 {adaptive['used_logical_batch']} | 下一步 {adaptive['logical_batch']} | 按样本调度学习率")
                acceleration=adaptive.get('torch_acceleration')
                if acceleration:
                    lines.append('  编译/Graph：'+backend_text(acceleration,read(folder/'acceleration-admission.json',{})))
            step=hb.get('optimizer_step');epoch=hb.get('epoch');epochs=config.get('epochs')
            n=self.rows.get(config.get('task'));batch_size=config.get('batch_size')
            batches=math.ceil(n/batch_size) if n and batch_size else None
            total=epochs*batches if epochs and batches else None
            dynamic=isinstance(hb.get('samples_seen'),int) and isinstance(hb.get('sample_budget'),int)
            if dynamic:
                seen=hb['samples_seen'];budget=hb['sample_budget']
                key=(job,state.get('pid'),started,'samples')
                meter=self.rates.setdefault(key,Throughput())
                origin=(hb.get('run_started_at'),hb.get('run_samples_start')) if 'run_samples_start' in hb else None
                stats=meter.observe(seen,hb.get('updated_at'),now,origin)
                loss=hb.get('loss');loss_text=f'{loss:.6g}' if isinstance(loss,(float,int)) else '?'
                lines.append(f'  训练更新 | Epoch {epoch+1 if isinstance(epoch,int) else "?"}/{epochs or "?"} | '
                    f'样本 {seen}/{budget} ({100*seen/budget:.2f}%) | 实际 Step {step}')
                lines.append(f'  loss {loss_text} | 心跳 {int(age)}s 前')
                lines.extend(speed_lines(stats,'samples/s',budget-seen))
                if age>120: lines.append('  提示：超过120s无新训练心跳；可能在编译/验证/保存，不能据此判定失败。')
                continue
            batch=step-epoch*batches if batches and isinstance(step,int) and isinstance(epoch,int) else None
            # The old worker has no phase event. Never label a quiet validation,
            # checkpoint write or compilation as a confirmed training stall.
            phase='训练更新'
            if batches and batch==batches: phase='本轮更新结束；验证/保存阶段未单独上报'
            if total and step>=total: phase='训练更新结束；最终评估/追踪阶段未单独上报'
            progress=f'{100*step/total:.2f}%' if total and isinstance(step,int) else '待测'
            lines.append(f'  {phase} | Epoch {epoch+1 if isinstance(epoch,int) else "?"}/{epochs or "?"} | '
                         f'Batch {batch if batch is not None else "?"}/{batches or "?"} | Step {step if step is not None else "?"}/{total or "?"} ({progress})')
            key=(job,state.get('pid'),started,'steps')
            meter=self.rates.setdefault(key,Throughput())
            origin=(hb.get('run_started_at'),hb.get('run_optimizer_step_start')) if 'run_optimizer_step_start' in hb else None
            stats=meter.observe(step,hb.get('updated_at'),now,origin)
            loss=hb.get('loss');loss_text=f'{loss:.6g}' if isinstance(loss,(float,int)) else '?'
            lines.append(f'  loss {loss_text} | 心跳 {int(age)}s 前')
            lines.extend(speed_lines(stats,'step/s',total-step if total and isinstance(step,int) else -1,3))
            if age>120:
                lines.append('  提示：超过 120s 无新训练心跳；可能在编译/验证/保存，不能据此判定失败。')
        for label,key in (('已计算','computed'),('排队','pending'),('失败','failed'),('暂停','paused')):
            lines.append(label+'：'+(', '.join(groups[key]) or '无'))
        delivered=(self.suite/'delivery-complete.json').exists()
        computed=(self.suite/'compute-complete.json').exists()
        lines.append('回传：'+('已完成' if delivered else ('计算完成，回传尚未完成' if computed else '等待全部任务计算完成')))
        lines.append('详情：各任务 jobs/<任务名>/worker.log；主入口日志 progress-parent.log。')
        return '\n'.join(lines)


def computed_suite(suite):
    return (suite/'compute-complete.json').exists()


def startup_lines(suite,launch,now):
    """Read only this launch's bounded log tails; never print raw child text."""
    lines=[]
    for name,label in (('setup.log','环境检查'),('progress-parent.log','入口日志')):
        path=suite/name
        try:
            stat=path.stat();offset=launch.get('log_offsets',{}).get(name)
            lines.append(f'  {label}：{stat.st_size} bytes | 最后写入 {int(max(0,now-stat.st_mtime))}s 前')
            if offset is None: continue
            with path.open('rb') as handle:
                handle.seek(max(offset,stat.st_size-65536))
                tail=handle.read(65536).decode('utf8',errors='replace')
            if name=='setup.log':
                stages=[s.removeprefix('QKAN_SETUP_STAGE ') for s in tail.splitlines()
                        if s.startswith('QKAN_SETUP_STAGE ')]
                allowed={'bootstrap_python'}|{m+':'+s for m in ('mega','s5')
                         for s in ('source','environment','dependencies','build','verify')}
                if stages and stages[-1] in allowed:
                    lines.append('  本次环境阶段：'+stages[-1]+'（进入阶段；不代表完成）')
            elif 'Verified identical array content despite ZIP encoding difference:' in tail:
                lines.append('  本次入口已进入数据一致性校验；全部完成后继续派发。')
        except OSError: pass
    lines.append('  暂无训练进度；请勿重复启动。若需排查，在另一窗口查看 progress-parent.log 与 setup.log。')
    return lines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--resume',type=Path,required=True)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--interval',type=float,default=15)
    parser.add_argument('--display',choices=('auto','inplace','append'),default='auto',
                        help='TTY: refresh in place; redirected output: append snapshots; append forces plain output')
    parser.add_argument('--watch-only',action='store_true',help='Read existing files without launching or writing anything')
    args=parser.parse_args()
    if not math.isfinite(args.interval) or not 1<=args.interval<=60: parser.error('interval must be 1..60 seconds')
    root=args.root.resolve();suite=(root/args.resume).resolve()
    if not suite.is_relative_to(root/'runs') or not suite.is_dir(): parser.error('resume must be an existing suite under runs/')
    dashboard=Dashboard(root,suite)
    terminal=Terminal(args.display)
    if args.watch_only:
        def leave_watch(*_): raise KeyboardInterrupt
        watch_signals={sig:signal.signal(sig,leave_watch) for sig in
                       (signal.SIGTERM,signal.SIGHUP)} if os.name=='posix' else {}
        try:
            while True:
                terminal.show(dashboard.render());time.sleep(args.interval)
        except KeyboardInterrupt: return 0
        finally:
            terminal.close()
            for sig,handler in watch_signals.items(): signal.signal(sig,handler)
    if os.name!='posix': parser.error('Training supervision requires the collaborator Linux runtime; watch-only supports other hosts')
    context=read(suite/'context.json');manifest=read(suite/'suite.json')
    report=read(suite/'gate-report.json')
    if not context or not manifest or not report or report.get('status')!='passed':
        parser.error('Only an already admitted suite can be resumed by this supervisor')
    expected=context['source_revision']
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if actual!=expected or manifest['source_revision']!=expected or report['source_revision']!=expected:
        parser.error('Executor revision must stay at '+expected+'; fetch this display script without pulling the checkout')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():
        parser.error('Executor tracked files changed; refusing to resume')
    cache=args.cache.resolve() if args.cache else Path(context['cache']).resolve()
    if str(cache)!=context['cache']: parser.error('Cache must match the original admitted suite')
    # Import only the original stdlib helpers. No model/framework is loaded.
    sys.path.insert(0,str(root))
    from experiments.baseline_delivery.common import lock,atomic_json
    with lock(suite/'progress-supervisor.lock'):
        # The original parent holds suite.lock for its complete lifetime.
        # Never start a second parent on an already running suite.
        try:
            with lock(suite/'suite.lock'): pass
        except OSError:
            parser.error('Original runner is still active; safely pause it and wait for the shell prompt first')
        command=resume_command(root,suite,context,cache)
        atomic_json(suite/('progress-launch-'+str(time.time_ns())+'.json'),dict(
            schema='qkan-progress-supervisor.v1',executor_revision=expected,
            display_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            admitted_entry_sha256=(hashlib.sha256(Path(command[2]).read_bytes()).hexdigest()
                if len(command)>2 and Path(command[2]).name=='run_admitted.py' else None),
            cwd=str(root),argv=command,started_at=time.time()))
        child=None;pause_requested=False
        def pause(*_):
            nonlocal pause_requested
            if not pause_requested:
                pause_requested=True
                print('\n请求安全暂停，等待原训练入口保存检查点并退出；请勿重复按键。',flush=True)
                if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)
        old={sig:signal.signal(sig,pause) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        try:
            print('接续原实验目录与原执行版本：'+expected+'\n'+str(suite),flush=True)
            with (suite/'progress-parent.log').open('ab') as log:
                if pause_requested: return 75
                started=time.time()
                offsets={name:(suite/name).stat().st_size if (suite/name).exists() else 0
                         for name in ('setup.log','progress-parent.log')}
                child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                dashboard.launch=dict(started_at=started,runner_pid=child.pid,interval=args.interval,
                                      log_offsets=offsets)
                last=0.
                while child.poll() is None:
                    if time.monotonic()-last>=args.interval:
                        dashboard.launch['updated_at']=time.time()
                        terminal.show(dashboard.render());last=time.monotonic()
                        atomic_json(suite/'display-heartbeat.json',dict(updated_at=time.time(),
                            supervisor_pid=os.getpid(),runner_pid=child.pid,pause_requested=pause_requested,
                            started_at=started,interval=args.interval,log_offsets=offsets,
                            meaning='Display supervisor alive; worker progress is in per-job heartbeat.json'))
                    time.sleep(.5)
                terminal.close()
                print(dashboard.render(),flush=True)
                code=child.returncode
                entry_status=read(suite/'entry-status.json',{})
                if (code not in (0,75) and entry_status.get('state')=='failed'
                        and entry_status.get('updated_at',0)>=started):
                    stage=entry_status.get('stage','unknown')
                    kind=entry_status.get('error_type','Exception')
                    print('入口失败阶段：'+stage+' | '+kind+' | 诊断：'+str(entry_status.get('diagnostic','entry-status.json')),flush=True)
                print('入口已退出：'+str(code)+'；'+('已完成计算与回传。' if code==0 and (suite/'delivery-complete.json').exists()
                      else '检查 progress-parent.log；退出/暂停不代表实验完成。'),flush=True)
                return code if code>=0 else 128-code
        finally:
            terminal.close()
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                child.wait()  # Let original runner finish its checkpoint/cleanup protocol.
            for sig,handler in old.items(): signal.signal(sig,handler)


if __name__=='__main__':
    raise SystemExit(main())
