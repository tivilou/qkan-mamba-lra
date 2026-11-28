"""Bounded baseline runtime probes; stdlib controller, original pinned workers."""
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
from types import SimpleNamespace
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


def final_training_limit(rows,config):
    """Allow epochs/validation in the disposable run, stop before final test.

    Use the largest possible logical batch, hence a LOWER bound on steps in
    epochs-1. It is not the old first-epoch cap and does not change config.
    """
    largest=max((config.get('adaptive_memory') or {}).get('logical_tiers',[config['batch_size']]))
    limit=math.ceil(rows/largest)*(config['epochs']-1)
    if limit<10:raise ValueError('Insufficient disposable training budget')
    return limit


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
            estimate_scope='Training cadence including periodic saves and epoch validation that occur inside windows; '
                'excludes startup, final validation/test, final pause/save and return. '
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
    job=getattr(args,'probe_job',model+'-listops-s42');output=suite/'jobs'/job
    config=read(source/'jobs'/job/'config.json');atomic_json(output/'config.json',config)
    limit=final_training_limit(rows,config)
    spec=make_spec(source,suite,job,limit);atomic_json(output/'command.json',spec)
    controller=read(source/'mamba-input.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',
        QKAN_RUNTIME_RELEASE=str(Path(controller['source_suite'])/'runtime-release.json'),
        PYTHONPATH=controller['original_root'],PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        QKAN_MEGA_SOURCE=str(source.parent.parent/'.bootstrap/strong-upstream/mega'),
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
                    terminal.show('\n'.join(['Baseline 队列短时测速（正式队列保持暂停）',
                        f'任务 {job} | 已完成 {len(completed)}/{getattr(args,"probe_total",2)} | 阶段 {phase} | 耗时 {int(now-started)}s',
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
        measurement_scope='Real worker training cadence; epoch validation and periodic saves inside windows included. '
            'Final validation/test/trace/return excluded. Other late-run costs may be unobserved.',
        crossed_epoch=bool(meter.last and meter.last['epoch']>0),stop_step_safety=limit,
        temporary_updates=(finish or {}).get('optimizer_step',0),formal_updates=0,cleanup=cleanup)
    record['measurement_complete']=len(meter.windows)>=3
    record['passed']=code==75 and reason!='time_limit' and record['measurement_complete'] and record['estimates'] is not None
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


def select_pending(source):
    """Include every unfinished roster job; summary validation outranks stale status."""
    from .migration import completed_files,verify_completed
    manifest=read(source/'suite.json');states=read(source/'status.json')
    chosen=[];done=[]
    for job in manifest['jobs']:
        name=job['id'];folder=source/'jobs'/name
        if (folder/'summary.json').exists() or states[name]['state']=='computed':
            if job.get('preserved_completion'):verify_completed(source,job)
            else:completed_files(folder,read(folder/'config.json'))
            done.append(name);continue
        if states[name]['state'] not in ('pending','paused','failed'):
            raise ValueError('Pause all source jobs first')
        chosen.append(job)
    return sorted(chosen,key=lambda j:'-retrieval-' in j['id']),done


def remaining_estimate(result,source,job,rows,config):
    heartbeat=peek(source/'jobs'/job/'heartbeat.json')
    progress=None
    if heartbeat:
        if 'samples_seen' in heartbeat:progress=heartbeat['samples_seen']
        elif 'next_row_cursor' in heartbeat:progress=heartbeat['epoch']*rows+heartbeat['next_row_cursor']
    if progress is None and not (source/'jobs'/job/'checkpoints/latest.json').exists():progress=0
    full=rows*config['epochs'];rate=result.get('samples_per_second')
    result['remaining']=dict(samples=max(0,full-progress) if progress is not None else None,
        seconds=max(0,full-progress)/rate if progress is not None and rate else None,
        progress_source='last saved heartbeat (approximate, not checkpoint deserialization)' if heartbeat else
            'not started' if progress==0 else 'unknown checkpoint progress; no guessed remainder')
    return result


def observed_source(source,name,rows,config):
    """Reuse a started job's OWN execution history, not unfused fresh Mamba."""
    folder=source/'jobs'/name;hb=peek(folder/'heartbeat.json')
    if not hb or 'samples_seen' not in hb or 'run_samples_start' not in hb:return None
    elapsed=hb['updated_at']-hb['run_started_at'];seen=hb['samples_seen']-hb['run_samples_start']
    if elapsed<60 or seen<=0:return None
    rate=seen/elapsed
    proof=peek(folder/'mamba-fused-execution.json') if config['model']=='mamba1' else peek(folder/'mega-execution.json')
    if config['model']=='mamba1' and (not proof or proof.get('observed_fused_calls',0)<=0):return None
    estimate=lambda n:dict(training_samples=n,epochs=config['epochs'],seconds=n*config['epochs']/rate)
    result=dict(passed=True,stable=False,provenance='existing same-job worker heartbeat; no new probe',
        samples_per_second=rate,observed_seconds=elapsed,observed_samples=seen,job=name,
        config_sha256=digest(folder/'config.json'),heartbeat_sha256=digest(folder/'heartbeat.json'),
        execution_proof=proof,estimates=dict(full=estimate(rows),tenth=estimate(rows//10)),
        formal_updates=0,limitation='Historical process average includes its startup/waits; stale relative to current hardware. '
            'No claim of a new three-window benchmark or current checkpoint-exact progress.')
    return remaining_estimate(result,source,name,rows,config)


def totals(report):
    values=list(report['results'].values())
    remaining=[r.get('remaining',{}).get('seconds') for r in values]
    full=[(r.get('estimates') or {}).get('full',{}).get('seconds') for r in values]
    return dict(estimated_jobs=sum(bool(r.get('estimates')) for r in values),selected_jobs=len(report['selected_jobs']),
        coverage_complete=len(values)==len(report['selected_jobs']) and all(v is not None for v in remaining)
            and all(r.get('estimates') for r in values),
        known_remaining_seconds=sum(v for v in remaining if v is not None),
        known_full_from_scratch_seconds=sum(v for v in full if v is not None),
        exclusions='Initialization, final evaluation/test/trace/return and unobserved validation overhead; '
            'epoch validation/checkpoint costs inside measured windows are included. Not a guaranteed total wall time.')


def summary(report):
    if report.get('all_pending'):
        lines=['# Unfinished baseline runtime estimates','',
            '| Job | Source | 3-window stable | samples/s | Full from scratch(h) | Remaining(h) |',
            '|---|---|---:|---:|---:|---:|']
        for name,result in report['results'].items():
            rate=result.get('samples_per_second');full=(result.get('estimates') or {}).get('full',{}).get('seconds')
            left=result.get('remaining',{}).get('seconds')
            fmt=lambda v:'unknown' if v is None else f'{v:.2f}'
            lines.append('| '+' | '.join((name,result.get('provenance','fresh isolated worker'),str(result.get('stable',False)),
                fmt(rate),fmt(full/3600 if full is not None else None),fmt(left/3600 if left is not None else None)))+' |')
        total=totals(report)
        lines+=['',f"Coverage {total['estimated_jobs']}/{total['selected_jobs']}; complete={total['coverage_complete']}",
            f"Known remaining training cadence: {total['known_remaining_seconds']/3600:.2f} h",
            f"Known full-from-scratch workload: {total['known_full_from_scratch_seconds']/3600:.2f} h",
            total['exclusions'],'Failures/unestimated jobs are NOT counted as zero cost.',
            'Original queue stays paused. Temporary probes are not formal training or accuracy evidence.']
        return '\n'.join(lines)+'\n'
    lines=['# ListOps runtime estimate','', '| Model | Stable | samples/s | Full100epochs(h) | Tenth100epochs(h) |',
        '|---|---:|---:|---:|---:|']
    for model,result in report['results'].items():
        estimates=result.get('estimates')
        values=(f"{result['samples_per_second']:.2f}",f"{estimates['full']['seconds']/3600:.2f}",
            f"{estimates['tenth']['seconds']/3600:.2f}") if estimates else ('unavailable',)*3
        lines.append('| '+' | '.join((model,str(result.get('stable',False)),*values))+' |')
    lines+=['','Training-cadence extrapolation; periodic saves/epoch validation inside windows included. Startup, final validation/test and final pause/return excluded.',
        'A tenth training subset was NOT trained. No accuracy or model-ranking conclusion.',
        'Fewer than3windows or >15% window spread: provisional estimate only.',
        'Formal queue remains paused; resume explicitly from its pinned policy controller.']
    return '\n'.join(lines)+'\n'


def campaign_pending(args):
    from .bootstrap import check_exchange
    from .runner import discover_gpus
    from .attention_diagnostic import gpu_preflight
    from .return_artifacts import upload_declared
    from scripts.experiment import validate_executor
    from scripts.formal_progress import Terminal
    require_linux();check_exchange();rev=revision();source=args.source_suite.resolve()
    if validate_executor(source.parent.parent,source)!='mamba-fused':raise ValueError('Use pinned original Suite')
    rows_by_task={t['task']:t['splits']['train']['arrays']['labels']['shape'][0]
        for t in read(ROOT/'docs/evidence/mega-document-v3-build.json')['tasks']}
    suite=ROOT/'runs'/('roster-speed-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    report=dict(schema='qkan-roster-speed.v2',all_pending=True,controller_revision=rev,source_suite=str(source),
        results={},failures={},formal_updates=0,original_queue_resumed=False)
    terminal=Terminal(args.display)
    with ExitStack() as held:
        names=[j['id'] for j in read(source/'suite.json')['jobs']]
        for p in (source/'progress-supervisor.lock',source/'suite.lock',*[source/'jobs'/n/'worker.lock' for n in sorted(names)]):
            held.enter_context(lock(p))
        jobs,done=select_pending(source);report.update(selected_jobs=[j['id'] for j in jobs],completed_skipped=done)
        if jobs:
            gpus,inventory=discover_gpus(args.gpu)
            if len(gpus)!=1 or not gpu_preflight(inventory[gpus[0]])['idle']:raise ValueError('One idle GPU required')
        files=[source/n for n in ('suite.json','status.json','mamba-input.json')]
        for name in names:
            folder=source/'jobs'/name;files.extend(p for p in [folder/'config.json',folder/'summary.json',
                folder/'checkpoints/latest.json',folder/'heartbeat.json'] if p.exists())
        hashes={str(p):digest(p) for p in files};suite.mkdir(parents=True)
        print('All pending runtime probe Suite: '+str(suite)+' | selected '+str(len(jobs)),flush=True)
        try:
            for job in jobs:
                name=job['id'];config=read(source/'jobs'/name/'config.json');rows=rows_by_task[config['task']]
                try:
                    existing=observed_source(source,name,rows,config)
                    if existing is not None:
                        report['results'][name]=existing
                    else:
                        if config['model'] not in ('mega','s5','s4d_minimal'):
                            raise ValueError('No matching execution observation for bound special worker; do not substitute an unfused worker')
                        options=SimpleNamespace(**vars(args));options.probe_job=name;options.probe_total=len(jobs)
                        value=probe(source,suite,config['model'],options,rows,terminal,report['results'])
                        report['results'][name]=remaining_estimate(value,source,name,rows,config)
                except KeyboardInterrupt:
                    report['interrupted']=True;break
                except Exception as error:
                    worker_failure(suite,error);report['failures'][name]=read(suite/'worker-error.json')
                    output=suite/'jobs'/name;proof=peek(output/'child-execution.json')
                    if (output/'child-start.json').exists() and not (proof or {}).get('cleanup',{}).get('group_verified'):
                        report['unsafe_remaining_processes']=True;break
                if not gpu_preflight(inventory[gpus[0]])['idle']:
                    report['unsafe_remaining_processes']=True;break
                # Recoverable local snapshot; only one immutable final upload.
                atomic_json(suite/'progress-report.json',report)
        finally:
            terminal.close();report['source_unchanged']=all(digest(p)==sha for p,sha in hashes.items())
            report['source_files']=hashes;report['totals']=totals(report)
            atomic_json(suite/'report.json',report);(suite/'summary.md').write_text(summary(report),encoding='utf8')
            upload_declared(suite,'roster-speed',['report.json','summary.md'],run_id=suite.name)
    print(summary(report),flush=True)
    return 0 if (report['source_unchanged'] and report['totals']['coverage_complete'] and
        not report.get('unsafe_remaining_processes') and all(r.get('passed') for r in report['results'].values())) else 2


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
    parser.add_argument('--timeout',type=float,default=2400,help='Per-job total wall cap, plus safe save/cleanup grace')
    parser.add_argument('--all-pending',action='store_true',help='Inventory and estimate every unfinished job; preserve completed results')
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
        return campaign_pending(args) if args.all_pending else campaign(args)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':raise SystemExit(main())
