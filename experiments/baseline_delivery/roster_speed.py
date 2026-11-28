"""Two bounded ListOps worker probes; stdlib controller, original pinned workers."""
import argparse
from contextlib import ExitStack
import datetime as dt
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from .common import ROOT,atomic_json,digest,lock,read,revision,spawn,stop_child,worker_failure

MODELS=('s5','s4d_minimal')


def require_linux():
    if os.name!='posix':raise ValueError('Run on the collaborator Linux GPU host')


def point(heartbeat,config):
    step=heartbeat['optimizer_step'];stamp=heartbeat['updated_at']
    samples=heartbeat.get('samples_seen',step*config['batch_size'])
    adaptive=heartbeat.get('adaptive_memory',{})
    batch=(adaptive.get('used_micro_batch',config['micro_batch']),
           adaptive.get('used_logical_batch',config['batch_size']))
    if not all(isinstance(n,(int,float)) and math.isfinite(n) and n>=0 for n in (step,stamp,samples)):
        raise ValueError('Invalid worker progress')
    return dict(step=step,at=stamp,samples=samples,batch=batch,
        oom_retries=adaptive.get('oom_retries',0),event=adaptive.get('event'),epoch=heartbeat.get('epoch',0))


class Windows:
    """Worker timestamps/counts only; a batch change restarts measurement."""
    def __init__(self,config,warmup=30,seconds=60):
        self.config=config;self.warmup=warmup;self.seconds=seconds
        self.first=None;self.anchor=None;self.last=None;self.windows=[];self.resets=0

    def observe(self,heartbeat):
        value=point(heartbeat,self.config)
        if self.last and value['step']<=self.last['step']:return
        changed=self.last and (value['batch']!=self.last['batch'] or value['oom_retries']!=self.last['oom_retries'])
        if changed:
            self.first=None;self.anchor=None;self.windows=[];self.resets+=1
        self.last=value
        if self.first is None:self.first=value
        if self.anchor is None:
            if value['at']-self.first['at']>=self.warmup and value['step']-self.first['step']>=5:self.anchor=value
            return
        elapsed=value['at']-self.anchor['at'];count=value['samples']-self.anchor['samples']
        if elapsed>=self.seconds and count>0:
            self.windows.append(dict(seconds=elapsed,samples=count,samples_per_second=count/elapsed,
                first=self.anchor,last=value))
            self.anchor=value

    def result(self,rows,epochs):
        windows=self.windows[:3]
        if len(windows)<3:
            anchor=self.anchor or self.first
            last=self.last
            if anchor and last and last['at']>anchor['at'] and last['samples']>anchor['samples']:
                seconds=last['at']-anchor['at'];count=last['samples']-anchor['samples']
                windows=windows+[dict(seconds=seconds,samples=count,samples_per_second=count/seconds,
                    first=anchor,last=last,partial=True,warmup_complete=self.anchor is not None)]
        rates=[w['samples_per_second'] for w in windows]
        if not rates:return dict(stable=False,windows=windows,estimates=None,batch_resets=self.resets)
        rate=sum(w['samples'] for w in windows)/sum(w['seconds'] for w in windows)
        spread=max(rates)/min(rates)-1
        stable=len(windows)==3 and not any(w.get('partial') for w in windows) and spread<=.15
        def estimate(n):
            return dict(training_samples=n,epochs=epochs,seconds=n*epochs/rate,
                observed_window_range_seconds=[n*epochs/max(rates),n*epochs/min(rates)])
        return dict(stable=stable,windows=windows,batch_resets=self.resets,samples_per_second=rate,
            relative_window_spread=spread,estimates=dict(full=estimate(rows),tenth=estimate(rows//10)),
            estimate_scope='Training cadence including periodic saves that occur inside windows; '
                'excludes startup, validation, test, final pause/save and return. '
                'Tenth is workload extrapolation, not measured subset training or accuracy. '
                'Observed window range is not a confidence interval.')


def make_spec(source,folder,job,limit):
    """Ask the unchanged original dispatcher for its admitted execution spec."""
    origin=source.parent.parent
    script=('import json,sys; from pathlib import Path; '
        'from experiments.baseline_delivery.key_bias_formal import job_spec; '
        'from experiments.baseline_delivery.common import read; '
        'print(json.dumps(job_spec(Path(sys.argv[1]),read(Path(sys.argv[2])/"mamba-input.json"),'
        '{"id":sys.argv[3]})))')
    value=subprocess.check_output([str(origin/'.venv-baselines/bin/python'),'-B','-c',script,
        str(folder),str(source),job],cwd=origin,text=True,
        env=dict(os.environ,PYTHONPATH=str(origin)),timeout=60)
    spec=__import__('json').loads(value)
    if '--resume' in spec['argv']:raise ValueError('Probe output must start fresh')
    spec['argv']+=['--stop-after',str(limit)]
    if job.startswith('s5-'):
        old=origin/'.venv-s5/bin/python';new=ROOT/'.venv-s5/bin/python'
        if not os.path.samefile(old,new):raise ValueError('Use the same original S5 environment')
        # common.spawn now applies the existing deterministic JAX/CUDA env setup.
        spec['argv'][0]=str(new)
    return spec


def peek(path):
    try:return read(path)
    except (OSError,ValueError):return None


def board(gpu):
    try:
        value=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=3)
        memory,util=map(float,value.strip().split(','))
        return dict(board_used_mib=memory,utilization_percent=util)
    except (OSError,ValueError,subprocess.SubprocessError):return {}


def probe(source,suite,model,args,rows,terminal,completed):
    from .diagnostic_watch import terminate
    job=model+'-listops-s42';output=suite/'jobs'/job
    config=read(source/'jobs'/job/'config.json');atomic_json(output/'config.json',config)
    largest=max((config.get('adaptive_memory') or {}).get('logical_tiers',[config['batch_size']]))
    # End strictly BEFORE the first validation/test pass, even at largest tier.
    limit=(rows-1)//largest
    if limit<10:raise ValueError('Insufficient first-epoch budget')
    spec=make_spec(source,suite,job,limit);atomic_json(output/'command.json',spec)
    controller=read(source/'mamba-input.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',
        QKAN_RUNTIME_RELEASE=str(Path(controller['source_suite'])/'runtime-release.json'),
        PYTHONPATH=controller['original_root'],PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        QKAN_S5_SOURCE=str(source.parent.parent/'.bootstrap/strong-upstream/s5'))
    meter=Windows(config,args.warmup_seconds,args.window_seconds)
    started=time.monotonic();first_update=None;requested=False;reason='worker_exit';peak=0;next_frame=0
    record={};cleanup=None
    with (output/'worker.log').open('ab') as log:
        child=spawn(spec,log,env)
        try:
            atomic_json(output/'child-start.json',dict(pid=child.pid))
            while child.poll() is None:
                now=time.monotonic();heartbeat=peek(output/'heartbeat.json')
                if heartbeat:
                    if first_update is None:first_update=now-started
                    meter.observe(heartbeat)
                if len(meter.windows)>=3:reason='three_windows';requested=True;break
                if now-started>=args.timeout:reason='time_limit';requested=True;break
                if now>=next_frame:
                    gpu=board(args.gpu);peak=max(peak,gpu.get('board_used_mib',0))
                    last=meter.last or {};phase='初始化 / 首次编译' if not last else '预热 / batch适配' if meter.anchor is None else '测量稳定窗口'
                    speed=meter.windows[-1]['samples_per_second'] if meter.windows else None
                    timing=(f'{speed:.2f} samples/s | 全量训练粗估 {rows*config["epochs"]/speed/3600:.2f}h（不含验证/测试）'
                        if speed else '速度 / 全量训练估时：待有效窗口')
                    terminal.show('\n'.join(['ListOps 双模型短时测速（正式队列保持暂停）',
                        f'模型 {model} | 已完成 {len(completed)}/2 | 阶段 {phase} | 耗时 {int(now-started)}s',
                        f'有效窗口 {len(meter.windows)}/3 | batch变化重置 {meter.resets} | GPU {args.gpu}',
                        f"Step {last.get('step','待测')} | 样本 {last.get('samples','待测')} | micro/逻辑batch {last.get('batch','待测')}",
                        timing,
                        f"显存整卡 {gpu.get('board_used_mib','待测')} MiB | 利用率 {gpu.get('utilization_percent','待测')}%",
                        '日志：'+str(output/'worker.log')]))
                    next_frame=now+5
                time.sleep(.5)
        finally:
            pause_started=time.monotonic()
            try:stop_child(child,grace=90)
            finally:
                cleanup=terminate(child,grace=2)
                code=child.returncode
                atomic_json(output/'child-execution.json',dict(pid=child.pid,exit_code=code,cleanup=cleanup))
    if not cleanup['group_verified']:raise RuntimeError('Probe process group cleanup unverified')
    finish=peek(output/'heartbeat.json')
    if finish:meter.observe(finish)
    record.update(meter.result(rows,config['epochs']),model=model,job=job,exit_code=code,
        stop_reason=reason,requested_pause=requested,first_heartbeat_seconds=first_update,
        startup_scope='Import/data/model/first compile + first update + heartbeat publication; not pure JIT time',
        wall_seconds=time.monotonic()-started,peak_observed_board_mib=peak,
        pause_save_cleanup_seconds=time.monotonic()-pause_started,
        memory_scope='Whole-board sampled peak every5s, not allocator high-water mark',
        final_progress=meter.last,config_sha256=digest(output/'config.json'),config=config,
        temporary_updates=(finish or {}).get('optimizer_step',0),formal_updates=0,cleanup=cleanup)
    record['passed']=code==75 and reason!='time_limit' and record['estimates'] is not None
    if (output/'summary.json').exists():raise RuntimeError('Probe unexpectedly reached final evaluation')
    if (output/'worker-error.json').exists():record['failure']=read(output/'worker-error.json')
    # Only disposable files in this owned probe's checkpoint directory.
    checkpoint=output/'checkpoints'
    if checkpoint.exists():
        if checkpoint.is_symlink():raise ValueError('Disposable checkpoints are linked')
        for path in checkpoint.iterdir():
            if path.is_symlink() or not path.is_file():raise ValueError('Unexpected disposable checkpoint entry')
            path.unlink()
    record['owned_checkpoints_removed']=True
    atomic_json(output/'probe-result.json',record)
    return record


def summary(report):
    lines=['# ListOps runtime estimate','', '| Model | Stable | samples/s | Full100epochs(h) | Tenth100epochs(h) |',
        '|---|---:|---:|---:|---:|']
    for model,result in report['results'].items():
        estimates=result.get('estimates')
        values=(f"{result['samples_per_second']:.2f}",f"{estimates['full']['seconds']/3600:.2f}",
            f"{estimates['tenth']['seconds']/3600:.2f}") if estimates else ('unavailable',)*3
        lines.append('| '+' | '.join((model,str(result.get('stable',False)),*values))+' |')
    lines+=['','Training-cadence extrapolation; periodic saves inside windows included. Startup, validation/test and final pause/return excluded.',
        'A tenth training subset was NOT trained. No accuracy or model-ranking conclusion.',
        'Fewer than3windows or >15% window spread: provisional estimate only.',
        'Formal queue remains paused; resume explicitly from its pinned policy controller.']
    return '\n'.join(lines)+'\n'


def campaign(args):
    from .bootstrap import check_exchange
    from .runner import discover_gpus
    from .attention_diagnostic import gpu_preflight
    from .return_artifacts import upload_declared
    from scripts.experiment import validate_executor
    from scripts.formal_progress import Terminal
    require_linux()
    check_exchange();rev=revision();source=args.source_suite.resolve()
    if validate_executor(source.parent.parent,source)!='mamba-fused':raise ValueError('Use original pinned continuation Suite')
    build=read(ROOT/'docs/evidence/mega-document-v3-build.json')
    rows=next(t['splits']['train']['arrays']['labels']['shape'][0] for t in build['tasks'] if t['task']=='listops')
    suite=ROOT/'runs'/('roster-speed-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    report=dict(schema='qkan-roster-speed.v1',controller_revision=rev,source_suite=str(source),train_rows=rows,
        results={},failures={},formal_updates=0,original_queue_resumed=False)
    terminal=Terminal(args.display)
    with ExitStack() as held:
        for p in (source/'progress-supervisor.lock',source/'suite.lock',
                *[source/'jobs'/(m+'-listops-s42')/'worker.lock' for m in MODELS]):held.enter_context(lock(p))
        if any(s['state']=='running' for s in read(source/'status.json').values()):raise ValueError('Pause original training and wait for saved exit first')
        gpus,inventory=discover_gpus(args.gpu)
        if len(gpus)!=1 or not gpu_preflight(inventory[gpus[0]])['idle']:raise ValueError('One idle GPU required')
        files=[source/n for n in ('suite.json','status.json','mamba-input.json')]
        for model in MODELS:
            folder=source/'jobs'/(model+'-listops-s42');files.append(folder/'config.json')
            files.extend(p for p in (folder/'checkpoints/latest.json',folder/'summary.json') if p.exists())
        hashes={str(p):digest(p) for p in files};suite.mkdir(parents=True)
        print('ListOps runtime probe Suite: '+str(suite),flush=True)
        try:
            for model in MODELS:
                try:
                    result=probe(source,suite,model,args,rows,terminal,report['results'])
                    report['results'][model]=result
                except KeyboardInterrupt:
                    report['interrupted']=True;break
                except Exception as error:
                    worker_failure(suite,error);report['failures'][model]=read(suite/'worker-error.json')
                    output=suite/'jobs'/(model+'-listops-s42')
                    proof=peek(output/'child-execution.json')
                    if (output/'child-start.json').exists() and not (proof or {}).get('cleanup',{}).get('group_verified'):
                        report['unsafe_remaining_processes']=True;break
                # Failures may continue only after the GPU is independently idle.
                if not gpu_preflight(inventory[gpus[0]])['idle']:
                    report['unsafe_remaining_processes']=True;break
        finally:
            terminal.close();report['source_unchanged']=all(digest(p)==sha for p,sha in hashes.items())
            report['source_files']=hashes
            atomic_json(suite/'report.json',report)
            (suite/'summary.md').write_text(summary(report),encoding='utf8')
            upload_declared(suite,'roster-speed',['report.json','summary.md'],run_id=suite.name)
    print(summary(report),flush=True)
    print('原正式队列保持暂停；报告：'+str(suite/'report.json'),flush=True)
    return 0 if (report['source_unchanged'] and not report.get('unsafe_remaining_processes')
        and set(report['results'])==set(MODELS) and all(v['passed'] for v in report['results'].values())) else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source-suite',type=Path)
    group.add_argument('--return-only',type=Path,help='Retry existing report upload; never rerun workers')
    parser.add_argument('--gpu',default='0');parser.add_argument('--display',choices=('auto','inplace','append'),default='auto')
    parser.add_argument('--warmup-seconds',type=float,default=30)
    parser.add_argument('--window-seconds',type=float,default=60)
    parser.add_argument('--timeout',type=float,default=1200,help='Per-model total wall cap, plus safe save/cleanup grace')
    args=parser.parse_args(argv)
    if not all(math.isfinite(n) and n>0 for n in (args.warmup_seconds,args.window_seconds,args.timeout)):
        parser.error('Durations must be positive and finite')
    if args.return_only:
        from .return_artifacts import upload_declared
        path=args.return_only.resolve()
        if path.parent!=(ROOT/'runs').resolve() or not path.name.startswith('roster-speed-'):raise ValueError('Owned probe Suite required')
        upload_declared(path,'roster-speed',['report.json','summary.md'],run_id=path.name);return 0
    previous={}
    def pause(*_):raise KeyboardInterrupt
    try:
        for name in ('SIGINT','SIGTERM','SIGHUP'):
            if hasattr(signal,name):
                sig=getattr(signal,name);previous[sig]=signal.signal(sig,pause)
        return campaign(args)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':raise SystemExit(main())
