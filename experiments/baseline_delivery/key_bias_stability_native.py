"""Bounded stability and incumbent controls. Frameworks supplied by native child."""
import math
from pathlib import Path
import statistics
import time
from .common import atomic_json,digest,read
from . import key_bias_recovery as recovery,key_bias_benchmark as benchmark
from .diagnostic_watch import mark

STEPS=64
BLOCK=8
MODES=('fp32','fp32_analytic','bf16_analytic','interrupt','resume','graph16','partition32')


def require_rows(runtime):
    if runtime.progress['cursor']+STEPS*128>len(runtime.split):
        raise ValueError('Need8192 remaining train rows in original epoch; preserve source, no reselection')


def gate(results):
    ref=results['fp32']['updates'];differences={}
    if len(ref)!=64: raise ValueError('Complete64step reference required')
    for name in benchmark.ROUTES:
        rows=results[name]['updates']
        if len(rows)!=64 or [u['row_ids_sha256'] for u in rows]!=[u['row_ids_sha256'] for u in ref]:
            raise ValueError('Matched64step train prefix required')
        blocks=[]
        for start in range(0,64,8):
            a=[u['loss'] for u in ref[start:start+8]];b=[u['loss'] for u in rows[start:start+8]]
            if any(not math.isfinite(v) or v<0 for v in a+b): raise ValueError('Invalid loss')
            original=statistics.mean(a);candidate=statistics.mean(b);limit=max(.10,.25*original)
            blocks.append(dict(start_update=start+1,stop_update=start+8,reference_mean=original,
                candidate_mean=candidate,excess=candidate-original,limit=limit,passed=candidate-original<=limit))
        differences[name]=blocks
    return dict(passed=all(v['passed'] for rows in differences.values() for v in rows),
        blocks=differences,criterion='8step mean excess<=max(.10,.25*reference_mean)',convergence_certified=False)


def load_midpoint(runtime,path,output,request_sha):
    path=Path(path);expected=Path(output).parent/'interrupt/diagnostic-state.pt';receipt=read(path.with_suffix('.json'))
    if path.is_symlink() or path.with_suffix('.json').is_symlink() or path.resolve()!=expected.resolve():
        raise ValueError('Only this diagnostic midpoint may resume')
    metadata=dict(schema=recovery.STATE_SCHEMA,strategy=recovery.manifest(),request_sha256=request_sha,
        source_checkpoint_sha256=runtime.request['checkpoint']['sha256'],baseline_state_sha256=runtime.state_hash(runtime.base))
    if (any(receipt.get(k)!=v for k,v in metadata.items()) or not 0<path.stat().st_size<=128*1024**2
            or path.stat().st_size!=receipt['bytes'] or digest(path)!=receipt['sha256']): raise ValueError('Midpoint receipt differs')
    payload=runtime.t.load(path,map_location='cpu',weights_only=False)
    if any(payload.get(k)!=v for k,v in metadata.items()) or runtime.state_hash(payload['state'])!=payload['state_sha256'] or payload['state_sha256']!=receipt['state_sha256']:
        raise ValueError('Midpoint payload differs')
    state=payload['state'];base=runtime.base
    if (state['progress']['step']!=base['progress']['step']+32 or state['progress']['cursor']!=base['progress']['cursor']+4096
            or state['progress']['epoch']!=base['progress']['epoch'] or state['scheduler']['progress']!=base['scheduler']['progress']+4096):
        raise ValueError('Expected exact32step midpoint')
    runtime.restore(state)
    if runtime.state_hash(runtime.snapshot())!=receipt['state_sha256']: raise ValueError('Loaded midpoint differs')
    return receipt['state_sha256']


def stability(runtime,output,mode,request_sha,reference=None):
    require_rows(runtime);report=dict(passed=False,updates=[],mode=mode,dropout=runtime.config['dropout'])
    base_hash=runtime.state_hash(runtime.base)
    try:
        report['starting_state_sha256']=runtime.state_hash(runtime.snapshot())
        if mode=='resume': report['loaded_state_sha256']=load_midpoint(runtime,Path(output).parent/'interrupt/diagnostic-state.pt',output,request_sha)
        route='bf16_analytic' if mode in ('interrupt','resume') else mode
        count=32 if mode in ('interrupt','resume') else 64
        for index in range(count):
            mark('key_bias_stability_'+mode+'_'+str(index),flush=True)
            value=benchmark.update(runtime,route)
            if 'attention' in value:
                a=value['attention'];value['attention']=dict(calls=a['calls'],layer_calls=a['layer_calls'],
                    qkv_dtypes=sorted(set(a['qkv_dtypes'])),output_dtypes=sorted(set(a['output_dtypes'])))
            report['updates'].append(value)
            atomic_json(Path(output)/'stability-progress.json',report)
            if reference is not None and (index+1)%8==0:
                a=reference[index-7:index+1];rows=report['updates'][-8:]
                if [u['row_ids_sha256'] for u in a]!=[u['row_ids_sha256'] for u in rows]: raise ValueError('Online block population differs')
                old=statistics.mean(u['loss'] for u in a);new=statistics.mean(u['loss'] for u in rows)
                if not math.isfinite(new) or new-old>max(.10,.25*old):
                    report.update(stop_reason='block_loss_gate',rejected_at_update=index+1);return report
        state=runtime.snapshot()
        expected=64 if mode=='resume' else count
        base=runtime.base
        if (runtime.progress['step']!=base['progress']['step']+expected
                or runtime.progress['cursor']!=base['progress']['cursor']+128*expected
                or runtime.progress['epoch']!=base['progress']['epoch']
                or runtime.schedule.state_dict()['progress']!=base['scheduler']['progress']+128*expected):
            raise ValueError('Pilot progress differs from frozen sample budget')
        report.update(passed=True,full_state_sha256=runtime.state_hash(state),
            component_sha256={k:runtime.state_hash(v) for k,v in state.items()},
            step=runtime.progress['step'],cursor=runtime.progress['cursor'],schedule_progress=runtime.schedule.state_dict()['progress'])
        if mode=='interrupt': report['diagnostic_checkpoint']=recovery.save_state(runtime,Path(output)/'diagnostic-state.pt',request_sha)
        return report
    finally:
        runtime.restore(runtime.base)
        if runtime.state_hash(runtime.snapshot())!=base_hash: raise ValueError('Stability rollback differs')
        report['rollback']=dict(recovery.ROLLBACK);atomic_json(Path(output)/'stability.json',report)


def agreement(results):
    a,b,c=(results[n] for n in ('bf16_analytic','interrupt','resume'))
    if (not all(v['passed'] for v in (a,b,c)) or len(a['updates'])!=64 or len(b['updates'])!=32 or len(c['updates'])!=32
            or a['updates']!=b['updates']+c['updates'] or a['full_state_sha256']!=c['full_state_sha256']
            or a['component_sha256']!=c['component_sha256'] or b['full_state_sha256']!=c['loaded_state_sha256']
            or len({v['starting_state_sha256'] for v in (a,b,c)})!=1
            or any(a[k]!=c[k] for k in ('step','cursor','schedule_progress'))): raise ValueError('64 vs32+32 recovery differs')
    return dict(passed=True,updates=64,midpoint=32,full_state_sha256=a['full_state_sha256'],component_sha256=a['component_sha256'])


def compare_values(runtime,left,right,path='state'):
    t=runtime.t;np=runtime.np
    if t.is_tensor(left):
        if not t.is_tensor(right) or left.shape!=right.shape or left.dtype!=right.dtype: raise ValueError('Tensor identity:'+path)
        exact='/rng/' in path or not left.is_floating_point()
        if not bool(t.equal(left,right) if exact else t.allclose(left,right,rtol=1e-5,atol=1e-6)):
            raise ValueError('Numerical parity:'+path)
    elif isinstance(left,np.ndarray):
        if not isinstance(right,np.ndarray) or not np.array_equal(left,right): raise ValueError('Array parity:'+path)
    elif type(left)!=type(right): raise ValueError('Type parity:'+path)
    elif isinstance(left,dict):
        if left.keys()!=right.keys(): raise ValueError('Keys parity:'+path)
        for k in left: compare_values(runtime,left[k],right[k],path+'/'+str(k))
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right): raise ValueError('Length parity:'+path)
        for i,(a,b) in enumerate(zip(left,right)): compare_values(runtime,a,b,path+'/'+str(i))
    elif isinstance(left,float) and '/rng/' not in path:
        if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(left,right,rel_tol=1e-5,abs_tol=1e-6): raise ValueError('Float parity:'+path)
    elif left!=right: raise ValueError('Control parity:'+path)


def compare_summary(records):
    order=('incumbent','candidate','candidate','incumbent')*2
    if tuple(v['route'] for v in records)!=order: raise ValueError('Balanced ABBAx2 order required')
    samples={n:[v['seconds'] for v in records if v['route']==n] for n in ('incumbent','candidate')}
    if any(not math.isfinite(v) or v<=0 for row in samples.values() for v in row): raise ValueError('Invalid timing')
    if len({v['update']['row_ids_sha256'] for v in records})!=1: raise ValueError('Timed sample differs')
    for name in samples:
        selected=[v for v in records if v['route']==name]
        if any(v['update']!=selected[0]['update'] or v['full_state_sha256']!=selected[0]['full_state_sha256'] for v in selected):
            raise ValueError('Same-route timed state/output differs')
    pairs=[]
    for i in range(0,8,2):
        a,c=records[i:i+2];old,new=(a,c) if a['route']=='incumbent' else(c,a)
        pairs.append(old['seconds']/new['seconds']-1)
    med={n:statistics.median(v) for n,v in samples.items()};gain=med['incumbent']/med['candidate']-1
    return dict(passed=gain>0 and all(v>0 for v in pairs),samples_seconds=samples,median_seconds=med,
        throughput_gain=gain,paired_throughput_gain=pairs,update_time_reduction=1-med['candidate']/med['incumbent'])


def comparison(runtime,output,mode):
    from .torch_acceleration import Executor,torch_runtime,policy
    from .partition_comparison import prepare_route,execution_validation,require_steady
    t=runtime.t;micro=16 if mode=='graph16' else 32;backend='eager_parallel' if micro==16 else'eager_partitioned'
    config=policy('compile_graph',backend,'graph');report=dict(passed=False,mode=mode,incumbent_backend=backend,incumbent_micro=micro,candidate_micro=16)
    graph=torch_runtime(t,runtime.model,runtime.rng_state,runtime.restore_rng,config);last={}
    executor=Executor(graph,config,saved=dict(mode='compile_graph',benchmarked=True,reason='matched_diagnostic',benchmark=None),
        read_state=lambda:dict(state=runtime.snapshot(),observed=last.get('observed')),
        compare=lambda a,b:compare_values(runtime,a,b))
    base_hash=runtime.state_hash(runtime.base);records=[]
    def run_graph():
        def callback():
            value=runtime.update(False,executor=executor,micro_batch=micro,retain=executor.validating)
            last['observed']=value
            return value
        return executor.run(callback,runtime.base,runtime.restore,lambda:runtime.opt.zero_grad(set_to_none=True),t.cuda.synchronize,
            lambda error:isinstance(error,t.cuda.OutOfMemoryError),execution_key=(micro,128))
    validation=lambda:execution_validation(executor,(micro,128))
    try:
        active=None
        for index,name in enumerate(('incumbent','candidate','candidate','incumbent')*2):
            if name!=active:
                graph.clear();runtime.restore(runtime.base)
                if name=='incumbent':
                    # A new capture generation must pass full-update parity before timing.
                    prepare_route(run_graph,validation,lambda:runtime.restore(runtime.base),t.cuda.synchronize)
                else:
                    for _ in range(2): runtime.restore(runtime.base);benchmark.update(runtime,'bf16_analytic')
                active=name
            runtime.restore(runtime.base);t.cuda.synchronize();before=validation() if name=='incumbent' else None
            if before is not None: require_steady(before)
            t.cuda.reset_peak_memory_stats();mark('key_bias_incumbent_'+mode+'_'+str(index),flush=True)
            start=time.perf_counter();value=run_graph() if name=='incumbent' else benchmark.update(runtime,'bf16_analytic')
            t.cuda.synchronize();elapsed=time.perf_counter()-start
            after=validation() if name=='incumbent' else None
            if before!=after: raise ValueError('Capture/validation changed during timing')
            if after is not None: require_steady(after)
            records.append(dict(route=name,seconds=elapsed,update=value,full_state_sha256=runtime.state_hash(runtime.snapshot()),
                steady_before=before,steady_after=after,peak_allocated=t.cuda.max_memory_allocated(),peak_reserved=t.cuda.max_memory_reserved()))
        report.update(timing=compare_summary(records),timing_records=records,passed=compare_summary(records)['passed'],
            scope='Same original state/logical128/dropout;CPU gather/H2D included;Graph native micro vs BF16eager16;one resident route;prepare/capture/parity/restore/hash excluded')
        return report
    finally:
        try: graph.clear()
        finally: runtime.restore(runtime.base)
        if runtime.state_hash(runtime.snapshot())!=base_hash: raise ValueError('Comparison rollback differs')
        report['rollback']=dict(recovery.ROLLBACK);atomic_json(Path(output)/'comparison.json',report)
