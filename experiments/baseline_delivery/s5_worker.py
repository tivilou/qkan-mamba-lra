"""Native JAX/Flax S5 worker. No Torch model or rewritten SSM is used."""
import argparse
from contextlib import ExitStack
import importlib.metadata
import math
import os
from pathlib import Path
import pickle
import signal
import time
import uuid

import numpy as np

from .common import ROOT, atomic_json, digest, fingerprint, lock, read
from .data import Split, TASKS, vocabulary
from .strong_sources import official_source

PAUSE = False


def pause(*_):
    global PAUSE
    PAUSE = True


def save_checkpoint(output, state):
    folder=Path(output)/'checkpoints'; folder.mkdir(exist_ok=True)
    name='step-'+str(state['step'])+'-'+uuid.uuid4().hex+'.pkl'
    temporary=folder/(name+'.pending')
    with temporary.open('xb') as handle:
        pickle.dump(state,handle,protocol=5);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,folder/name)
    old=read(folder/'latest.json') if (folder/'latest.json').exists() else None
    previous=[{'file':old['file'],'sha256':old['sha256']}] if old else []
    atomic_json(folder/'latest.json',dict(schema='qkan-s5-checkpoint.v1',file=name,
        sha256=digest(folder/name),contract=state['contract'],step=state['step'],previous=previous))
    if os.name!='nt':
        fd=os.open(folder,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    for path in folder.glob('step-*.pkl'):
        if path.name not in {name,*[p['file'] for p in previous]}: path.unlink()


def load_checkpoint(output, contract):
    pointer=read(Path(output)/'checkpoints/latest.json')
    if (pointer['schema']!='qkan-s5-checkpoint.v1' or pointer['contract']!=contract
            or Path(pointer['file']).name!=pointer['file']):
        raise ValueError('Invalid S5 checkpoint identity')
    path=Path(output)/'checkpoints'/pointer['file']
    if digest(path)!=pointer['sha256']: raise ValueError('S5 checkpoint hash mismatch')
    # Local runner-generated hash-bound state only; never accept remote pickle.
    with path.open('rb') as handle: state=pickle.load(handle)
    if state['contract']!=contract: raise ValueError('S5 state contract mismatch')
    return state


def build(modules, config):
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from .strong_sources import s5_encoder
    n=config['state_size']
    if n%2: raise ValueError('S5 conjugate state size must be even')
    lam,_,_,v,_=modules['initialization'].make_DPLR_HiPPO(n)
    v=v[:,:n//2];lam=lam[:n//2]
    ssm=dict(H=config['width'],P=n//2,Lambda_re_init=lam.real,Lambda_im_init=lam.imag,
        V=v,Vinv=v.conj().T,C_init='lecun_normal',discretization=config['discretization'],
        dt_min=.001,dt_max=.1,conj_sym=True,clip_eigs=False,bidirectional=config['bidirectional'])
    class Model(nn.Module):
        training: bool

        @nn.compact
        def __call__(self, batch):
            # vmap official per-sequence encoder; independent dropout keys per row.
            # LayerNorm avoids changing BatchNorm statistics via microbatching.
            template=s5_encoder(modules,ssm,dict(d_model=config['width'],n_layers=config['layers'],
                training=self.training,batchnorm=False,prenorm=True,
                dropout=config['dropout'],activation='half_glu1'))
            Batch=nn.vmap(type(template),in_axes=(0,0),out_axes=0,
                variable_axes={'params':None},split_rngs={'params':False,'dropout':True})
            encoder=Batch(**{key:getattr(template,key) for key in
                ('ssm','d_model','n_layers','training','batchnorm','prenorm','dropout','activation')},name='encoder')
            tokens=config['task'] not in ('image','pathfinder')
            embedding=nn.Embed(config['vocabulary'],config['width'],name='embedding') if tokens else None
            values={};pooled=[]
            for side,key in enumerate(('inputs1','inputs2') if config['task']=='retrieval' else ('inputs',)):
                x=batch[key];mask=x!=1 if tokens else jnp.ones(x.shape,dtype=bool)
                h=embedding(x) if tokens else x[...,None]
                h=jnp.where(mask[...,None],h,0)
                h=encoder(h,jnp.ones(x.shape,dtype=jnp.float32))
                h=jnp.where(mask[...,None],h,0)
                pooled.append(h.sum(1)/mask.sum(1)[:,None])
                values.update({f'input{side}':x,f'mask{side}':mask,f'encoder{side}':h})
            z=jnp.concatenate((abs(pooled[0]-pooled[1]),pooled[0]*pooled[1]),-1) if len(pooled)==2 else pooled[0]
            # Declare dot precision: JIT fusion and eager matmul may otherwise
            # choose different reduced-precision GPU kernels for this head.
            logits=nn.Dense(TASKS[config['task']][1],name='head',
                            precision=jax.lax.Precision.HIGHEST)(z)
            return logits,{**values,'pooled':z,'logits':logits}
    return Model


def run(args, modules):
    with ExitStack() as resources: return _run(args,modules,resources)


def _run(args, modules, resources):
    run_started_at=time.time()
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization, traverse_util
    from flax.training import train_state
    config=read(args.config);out=args.output;out.mkdir(parents=True,exist_ok=True)
    if config.get('batchnorm',False): raise ValueError('Only declared S5 LayerNorm protocol is supported')
    if args.device.startswith('cuda') and jax.default_backend()!='gpu': raise ValueError('JAX CUDA backend unavailable')
    train=Split(args.data,config['task'],'train',args.cache,config.get('data_hashes',{}).get('train'))
    val=Split(args.data,config['task'],'val',args.cache,config.get('data_hashes',{}).get('val'))
    from .logical_batch import canary_rows,canary_tier,planned_batch_size
    bounded_rows=canary_rows(config,args.synthetic)
    if args.synthetic:
        marker=read(args.data/'synthetic-canary.json')
        if marker.get('test_included') is not False or not marker.get('bounded') or config['epochs']!=2 or len(train)>bounded_rows or len(val)>2:
            raise ValueError('Invalid bounded S5 canary')
    vocab,dict_sha=vocabulary(args.data,config['task'])
    if vocab!=config['vocabulary']: raise ValueError('S5 vocabulary differs')
    contract=fingerprint(dict(config=config,train=train.identity,val=val.identity,dictionary=dict_sha))
    runtime={name:importlib.metadata.version(name) for name in ('jax','jaxlib','flax','optax','numpy')}
    runtime['backend']=jax.default_backend()
    runtime['classifier_dot_precision']='highest'
    runtime['training_state_placement']='jax-device-arrays-v1'
    runtime['xla_flags']=os.environ.get('XLA_FLAGS','')
    runtime['cublas_workspace_config']=os.environ.get('CUBLAS_WORKSPACE_CONFIG','')
    if args.device.startswith('cuda') and not (
            '--xla_gpu_deterministic_ops=true' in runtime['xla_flags'].split()
            and '--xla_gpu_autotune_level=0' in runtime['xla_flags'].split()
            and runtime['cublas_workspace_config']==':4096:8'):
        raise ValueError('S5 GPU requires declared deterministic launcher settings')
    atomic_json(out/'environment.json',dict(packages=runtime,devices=[str(d) for d in jax.devices()],
        precision='float32; no AMP; classifier dot HIGHEST in forward/reference/replay'))
    Model=build(modules,config);model=Model(training=True);evaluation=Model(training=False)
    rng=jax.random.PRNGKey(config['seed']);rng,init,drop=jax.random.split(rng,3)
    dummy=train.numpy_batch(np.arange(min(1 if config.get('adaptive_memory') else config['micro_batch'],len(train))))
    from flax.core import unfreeze
    params=unfreeze(model.init({'params':init,'dropout':drop},dummy)['params'])
    steps=config['epochs']*math.ceil(len(train)/config['batch_size']);warmup=min(config['warmup_steps'],steps-1)
    from .performance import Lookahead,Profile
    performance=bool(config.get('performance'))
    profile=Profile(limit=16 if performance and not (out/'performance-profile.json').exists() else 0)
    dynamic=bool((config.get('adaptive_memory') or {}).get('logical_tiers'));sample_clock=None
    def schedule(step):
        if sample_clock is not None: return jnp.asarray(sample_clock.rates()[0],dtype=jnp.float32)
        return config['lr']*jnp.where(step<warmup,(step+1)/max(1,warmup),
            .5*(1+jnp.cos(jnp.pi*jnp.minimum(1,(step-warmup)/max(1,steps-warmup)))))
    flat=traverse_util.flatten_dict(params)
    labels=traverse_util.unflatten_dict({key:'ssm' if key[-1] in
        ('B','Lambda_re','Lambda_im','log_step','norm') else 'regular' for key in flat})
    tx=optax.chain(optax.clip_by_global_norm(config['clip_norm']),optax.multi_transform({
        'ssm':optax.adam(lambda step:schedule(step)*config['ssm_lr_scale']),
        'regular':optax.adamw(schedule,weight_decay=config['weight_decay'])},labels))
    state=train_state.TrainState.create(apply_fn=model.apply,params=params,tx=tx)
    meta=dict(contract=contract,runtime=runtime,step=0,epoch=0,cursor=0,permutation=None,
        rng=np.asarray(rng),sampler=np.random.RandomState(config['seed']).get_state(),
        best_accuracy=-1.,best_epoch=None,best_weights=None,history=[],update_trace=[],
        epoch_loss=0.,epoch_correct=0,epoch_samples=0,training_seconds=0.,training_complete=False)
    migration_check_only=getattr(args,'migration_check_only',False)
    if args.resume or migration_check_only:
        if migration_check_only:
            if args.resume or args.synthetic or (out/'checkpoints/latest.json').exists():
                raise ValueError('Migration check requires a new formal job without a target checkpoint')
            from .migration import load_state
            meta=load_state(out,config,contract,lambda old:fingerprint(
                dict(config=old,train=train.identity,val=val.identity,dictionary=dict_sha)),load_checkpoint)
        else: meta=load_checkpoint(out,contract)
        if meta['runtime']!=runtime: raise ValueError('S5 runtime changed')
        state=serialization.from_state_dict(state,meta['train_state'])
    elif (out/'checkpoints/latest.json').exists(): raise ValueError('Existing S5 state requires --resume')
    run_samples_start=meta['epoch']*len(train)+meta['cursor']
    run_optimizer_step_start=meta['step']
    # Flax from_state_dict restores the supplied NumPy leaves as-is. Eager
    # Optax arithmetic on host moments/counters can differ from uninterrupted
    # JAX arithmetic. Canonicalize BOTH fresh and restored states before any
    # gradient/update; keep sampler and saved explicit RNG metadata on host.
    state=state.replace(step=jnp.asarray(state.step,dtype=jnp.int32))
    state=jax.device_put(jax.tree_util.tree_map(lambda value:jnp.asarray(value,dtype=value.dtype),state))
    if any(not isinstance(value,jax.Array) for value in jax.tree_util.tree_leaves(state)):
        raise ValueError('S5 training state has non-JAX leaves')
    if args.resume or migration_check_only:
        from .readiness import compare_numpy
        compare_numpy(meta['train_state'],jax.tree_util.tree_map(np.asarray,serialization.to_state_dict(state)))
        atomic_json(out/'restore-validation.json',dict(schema='qkan-s5-restore.v1',
            passed=True,state_placement='jax-device-arrays-v1',roundtrip='exact dtype/shape/value',
            scope='Parameters, optimizer moments/counters and training step before first resumed update'))
    adaptive=None
    if config.get('adaptive_memory'):
        if not args.device.startswith('cuda'): raise ValueError('Adaptive S5 requires CUDA')
        from .adaptive_memory import Controller,transaction,attach_journal,evaluation as adaptive_evaluate,jax_oom,board_memory
        from .logical_batch import make_controller,logical_size,SampleClock
        adaptive=make_controller(config['adaptive_memory'],config['batch_size'],meta.get('adaptive_memory'))
        from .execution import BatchPrefetch,StatusWriter
        prefetch=resources.enter_context((Lookahead if performance else BatchPrefetch)(lambda ids:train.numpy_batch(np.asarray(ids),validate=True)))
        status_writer=StatusWriter(atomic_json,adaptive.policy.get('status_interval',5.))
        attach_journal(adaptive,out,lambda:meta['step'])
        def cleanup_update():
            import gc
            gc.collect();jax.clear_caches()
        def memory():
            stats=jax.devices()[0].memory_stats() or {}
            # The process peak is conservative across tiers. Query global free
            # memory only at comparison windows, keeping normal updates cheap.
            result=dict(peak_bytes=stats.get('peak_bytes_in_use'),allocated_bytes=stats.get('bytes_in_use'))
            if not adaptive.state['warmup'] and len(adaptive.state['samples'])>=adaptive.policy['window']-1:
                result.update(board_memory())
                if result['peak_bytes'] is None and result.get('total_bytes'):
                    result['peak_bytes']=result['total_bytes']-result['free_bytes']
            return result
    if dynamic:
        # Optax schedule closures execute in the existing eager apply_gradients
        # path; gradient() is jitted separately and never captures this clock.
        anchor_rate=float(schedule(meta['step']))
        sample_clock=SampleClock(config['epochs']*len(train),min(config['warmup_steps']*config['batch_size'],
            max(0,config['epochs']*len(train)-config['batch_size'])),
            [config['lr']],meta['epoch']*len(train)+meta['cursor'],
            [anchor_rate] if (args.resume or migration_check_only) else None,
            saved=meta.get('sample_clock'),reference_batch=config['batch_size'])
        if sample_clock.state['progress']!=meta['epoch']*len(train)+meta['cursor']:
            raise ValueError('Sample clock and sampler cursor differ')
    if migration_check_only:
        if adaptive is None: raise ValueError('Migration target must enable adaptive memory')
        # The restored TrainState already passed exact host/device roundtrip above.
        meta['adaptive_memory']=adaptive.snapshot()
        if dynamic:
            meta['sample_clock']=sample_clock.snapshot()
            meta['migration']['sample_schedule']=dict(samples_seen=sample_clock.state['progress'],
                sample_budget=sample_clock.state['total'],anchor_rates=sample_clock.rates(),
                optimizer_counter='preserved; not rescaled',warmup_restart=False)
        save_checkpoint(out,meta)
        compare_numpy(meta,load_checkpoint(out,contract))
        atomic_json(out/'migration-validation.json',dict(passed=True,**meta['migration'],
            state_roundtrip='exact before first adaptive optimizer update',
            target_contract=contract,optimizer_steps_performed=0))
        return 0
    atomic_json(out/'config.json',config)
    @jax.jit
    def gradient(params,batch,key):
        def loss(p):
            logits,_=model.apply({'params':p},batch,rngs={'dropout':key})
            return optax.softmax_cross_entropy_with_integer_labels(logits,batch['labels']).mean(),(logits.argmax(-1)==batch['labels']).sum()
        return jax.value_and_grad(loss,has_aux=True)(params)
    @jax.jit
    def predict(params,batch): return evaluation.apply({'params':params},batch)
    @jax.jit
    def gradients_finite(grads):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
    def evaluate(params,split,micro_batch):
        loss=correct=0.
        for start in range(0,len(split),micro_batch):
            b=split.numpy_batch(np.arange(start,min(start+micro_batch,len(split))))
            logits,_=predict(params,b)
            loss+=float(optax.softmax_cross_entropy_with_integer_labels(logits,b['labels']).sum())
            correct+=int((logits.argmax(-1)==b['labels']).sum())
        return dict(loss=loss/len(split),accuracy=correct/len(split),samples=len(split))
    def evaluate_split(params,split):
        if adaptive:
            return adaptive_evaluate(adaptive,lambda micro:evaluate(params,split,micro),cleanup_update,jax_oom)
        return evaluate(params,split,config['micro_batch'])
    started=time.monotonic();last_save=started
    def save():
        nonlocal started,last_save
        now=time.monotonic();meta['training_seconds']+=now-started;started=last_save=now
        meta['train_state']=jax.tree_util.tree_map(np.asarray,serialization.to_state_dict(state))
        if adaptive: meta['adaptive_memory']=adaptive.snapshot()
        if dynamic: meta['sample_clock']=sample_clock.snapshot()
        save_checkpoint(out,meta)
    while meta['epoch']<config['epochs']:
        if meta['permutation'] is None:
            sampler=np.random.RandomState();sampler.set_state(meta['sampler'])
            meta['permutation']=sampler.permutation(len(train));meta['sampler']=sampler.get_state()
        while meta['cursor']<len(train):
            canary_tier(adaptive,config,meta['step'])
            batch_size=planned_batch_size(adaptive,config,meta['step'])
            end=min(meta['cursor']+batch_size,len(train));ids=meta['permutation'][meta['cursor']:end]
            update_started=time.perf_counter()
            profile.begin()
            retries_before=adaptive.state['oom_count'] if adaptive else 0
            if adaptive:
                with profile.phase('cpu_batch_wait',device=False): staged=prefetch.get(ids)
                if performance and end<len(train):
                    capacity=max(adaptive.policy.get('logical_tiers',[config['batch_size']]))
                    prefetch.submit(meta['permutation'][end:min(end+capacity,len(train))])
                device_batch=None
                try:
                    device_batch=jax.device_put(staged);jax.block_until_ready(device_batch)
                    staged=device_batch
                except Exception as error:
                    if not jax_oom(error): raise
                    device_batch=None
                if end<len(train) and not dynamic and not performance: prefetch.submit(meta['permutation'][end:min(end+config['batch_size'],len(train))])
            def update(micro_batch):
                grads=jax.tree_util.tree_map(jnp.zeros_like,state.params);loss=0.;correct=0;finite=True
                key=jnp.asarray(meta['rng'])
                for start in range(0,len(ids),micro_batch):
                    sub=ids[start:start+micro_batch]
                    batch=({name:values[start:start+micro_batch] for name,values in staged.items()}
                           if adaptive else train.numpy_batch(sub))
                    key,drop=jax.random.split(key)
                    (value,count),g=gradient(state.params,batch,drop)
                    if adaptive:
                        finite=finite & jnp.isfinite(value)
                        weight=len(sub)/len(ids);loss=loss+value*weight;correct=correct+count
                    else:
                        if not np.isfinite(float(value)): raise ValueError('Nonfinite S5 loss')
                        weight=len(sub)/len(ids);loss+=float(value)*weight;correct+=int(count)
                    grads=jax.tree_util.tree_map(lambda a,b:a+weight*b,grads,g)
                if adaptive:
                    finite=finite & gradients_finite(grads)
                if (adaptive and not bool(finite)) or (not adaptive and not all(
                        np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grads))):
                    raise ValueError('Nonfinite S5 gradient')
                new_state=state.apply_gradients(grads=grads)
                jax.block_until_ready(new_state)
                return new_state,float(loss),int(correct),np.asarray(key)
            adaptive_status={}
            if adaptive:
                # JAX state/optimizer arrays are immutable; update() commits
                # neither state nor RNG until all asynchronous work succeeds.
                (new_state,loss,correct,key),adaptive_status=transaction(adaptive,update,
                    lambda:None,lambda snapshot:None,cleanup_update,jax_oom,memory,lambda:None,
                    started_at=update_started,samples=len(ids))
                adaptive_status.update(rollback_device='immutable_jax_state',cpu_prefetch=1)
            else: new_state,loss,correct,key=update(config['micro_batch'])
            if profile.finish((adaptive.state['oom_count']-retries_before) if adaptive else 0):
                atomic_json(out/'performance-profile.json',dict(profile.report(),
                    config_sha256=fingerprint(config),source_revision=config['source_revision'],
                    framework='jax',phase_limit='Host update wall only; existing block_until_ready; no CUDA-event breakdown',
                    optimizer_step=meta['step']+1))
            state=new_state;meta['rng']=key;meta['cursor']=end;meta['step']+=1
            if dynamic:
                sample_clock.advance(len(ids))
                if end<len(train) and not performance:
                    next_batch=planned_batch_size(adaptive,config,meta['step'])
                    prefetch.submit(meta['permutation'][end:min(end+next_batch,len(train))])
            meta['epoch_loss']+=loss*len(ids);meta['epoch_correct']+=correct;meta['epoch_samples']+=len(ids)
            if args.synthetic: meta['update_trace'].append(dict(step=meta['step'],rows=ids.tolist(),loss=loss))
            heartbeat=dict(state='running',epoch=meta['epoch'],optimizer_step=meta['step'],loss=loss,updated_at=time.time(),
                run_started_at=run_started_at,run_samples_start=run_samples_start,
                run_optimizer_step_start=run_optimizer_step_start,
                **({'adaptive_memory':adaptive_status} if adaptive else {}))
            if dynamic: heartbeat.update(samples_seen=meta['epoch']*len(train)+end,
                sample_budget=config['epochs']*len(train),next_row_cursor=end,logical_batch=adaptive.logical_batch)
            if adaptive:
                status_writer.emit(lambda write:(write(out/'heartbeat.json',heartbeat),write(out/'adaptive-memory.json',
                    dict(policy=adaptive.policy,state=adaptive.snapshot(),last_update=adaptive_status,step=meta['step']))),
                    force=PAUSE or (args.stop_after is not None and meta['step']>=args.stop_after),
                    token=adaptive.state['transitions'])
            else: atomic_json(out/'heartbeat.json',heartbeat)
            if PAUSE or (args.stop_after is not None and meta['step']>=args.stop_after):
                save();atomic_json(out/'status.json',dict(state='paused',step=meta['step']));return 75
            if meta['step']%config['checkpoint_steps']==0 or time.monotonic()-last_save>=300: save()
        metric=evaluate_split(state.params,val)
        if metric['accuracy']>meta['best_accuracy']:
            meta.update(best_accuracy=metric['accuracy'],best_epoch=meta['epoch'],best_weights=jax.tree_util.tree_map(np.asarray,state.params))
        meta['history'].append(dict(epoch=meta['epoch'],train_loss=meta['epoch_loss']/meta['epoch_samples'],
            train_accuracy=meta['epoch_correct']/meta['epoch_samples'],validation=metric))
        meta.update(epoch=meta['epoch']+1,cursor=0,permutation=None,epoch_loss=0.,epoch_correct=0,epoch_samples=0);save()
    if not meta['training_complete']: meta['training_complete']=True;save()
    best=meta['best_weights'];metric=evaluate_split(best,val)
    if metric['accuracy']!=meta['best_accuracy']: raise ValueError('S5 selected validation does not reproduce')
    test=None if args.synthetic else evaluate_split(best,Split(args.data,config['task'],'test',args.cache,config['data_hashes']['test']))
    weights=out/'best-model.msgpack';weights.write_bytes(serialization.msgpack_serialize(best))
    ids=np.arange(min(2,len(val)));batch=val.numpy_batch(ids);_,values=predict(best,batch)
    @jax.jit
    def replay_head(encoded, masks, kernel, bias):
        # Reconstruct from decoded encoders, never the captured pooled/logits.
        # Match the forward's compiled batched reduction and explicit dot
        # precision while keeping this an independent implementation.
        parts=[h.sum(1)/mask.sum(1)[:,None] for h,mask in zip(encoded,masks)]
        z=jnp.concatenate((abs(parts[0]-parts[1]),parts[0]*parts[1]),-1) if len(parts)==2 else parts[0]
        return z,jnp.matmul(z,kernel,precision=jax.lax.Precision.HIGHEST)+bias

    def replay(rows):
        encoded=[];masks=[]
        for side in range(2 if config['task']=='retrieval' else 1):
            h=jnp.asarray(np.stack([r['encoder'+str(side)] for r in rows]));mask=np.stack([r['mask'+str(side)] for r in rows])
            if mask.dtype!=bool or np.any(mask.sum(1)==0) or np.any(np.asarray(h)[~mask]!=0): raise ValueError('Invalid S5 trace mask')
            encoded.append(h);masks.append(jnp.asarray(mask))
        pooled,logits=replay_head(tuple(encoded),tuple(masks),jnp.asarray(best['head']['kernel']),jnp.asarray(best['head']['bias']))
        return np.asarray(pooled),np.asarray(logits)
    from .trace import emit
    emit(jax.tree_util.tree_map(np.asarray,values),batch['labels'],ids,val,out,config,digest(weights),replay,'jax')
    atomic_json(out/'summary.json',dict(schema='qkan-baseline-result.v1',task=config['task'],model='s5',seed=config['seed'],
        source_revision=config['source_revision'],config_sha256=fingerprint(config),contract=contract,
        data_hashes=config.get('data_hashes',{}),best_epoch=meta['best_epoch'],validation=metric,test=test,
        history=meta['history'],optimizer_steps=meta['step'],parameters=sum(p.size for p in jax.tree_util.tree_leaves(best)),
        training_seconds=meta['training_seconds'],peak_gpu_bytes=None,synthetic=args.synthetic,formal=not args.synthetic,
        best_weights_sha256=digest(weights),**({'migration':meta['migration']} if meta.get('migration') else {}),
        **({'sample_clock':sample_clock.snapshot()} if dynamic else {}),
        **({'adaptive_memory':dict(policy=adaptive.policy,
            state=adaptive.snapshot(),trajectory=('dynamic logical batch/sample schedule; not fixed-batch equivalent'
                if dynamic else 'adaptive; not fixed-microbatch-equivalent'))} if adaptive else {})))
    atomic_json(out/'status.json',dict(state='computed',step=meta['step'],delivery='pending'));return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','data','cache','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--synthetic',action='store_true');p.add_argument('--stop-after',type=int)
    p.add_argument('--migration-check-only',action='store_true')
    args=p.parse_args()
    if not args.synthetic:
        from .runner import check_release
        check_release(read(args.config))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,pause)
    try:
        with lock(args.output/'worker.lock'),official_source('s5',os.environ['QKAN_S5_SOURCE']) as modules:
            return run(args,modules)
    except Exception as error:
        from .common import worker_failure
        worker_failure(args.output,error)
        raise


if __name__=='__main__': raise SystemExit(main())
