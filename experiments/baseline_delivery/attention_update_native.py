"""Collaborator-only complete-update diagnostic. Native imports are lazy.

This runner creates private disposable states; it never enters the formal queue.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import time

from .common import atomic_json,digest,fingerprint,read,revision,worker_failure
from .attention_update import POLICY,attention_route,numerical_gate,timing_summary,validate_request


class NativeUpdate:
    def __init__(self,torch,np,request):
        from .models import Classifier,optimizer
        from .data import Split
        from .logical_batch import SampleClock,TorchSampleSchedule
        from .worker import rng_state,restore_rng
        self.t,self.np,self.request=torch,np,request
        self.rng_state,self.restore_rng=rng_state,restore_rng
        self.config=request['source_config'];c=self.config;p=request['checkpoint']
        self.split=Split(request['data'],'retrieval','train',request['cache'],c['data_hashes']['train'])
        self.source=torch.load(p['path'],map_location='cpu',weights_only=False)
        s=self.source
        if (s.get('schema')!='qkan-training-state.v1' or s.get('contract')!=p['contract']
                or s.get('step')!=p['step'] or s.get('training_complete')
                or s.get('runtime',{}).get('torch')!=torch.__version__
                or s.get('runtime',{}).get('numpy')!=np.__version__
                or s.get('scheduler',{}).get('schema')!='qkan-sample-clock.v1'
                or s.get('sample_clock')!=s['scheduler']):
            raise ValueError('Expected original full sample-clock checkpoint and pinned runtime')
        if (s['permutation'] is None or s['permutation'].device.type!='cpu'
                or s['permutation'].dtype!=torch.int64 or len(s['permutation'])!=len(self.split)
                or not 0<=s['cursor']<=len(self.split)-POLICY['trajectory_updates']*POLICY['logical_batch']):
            raise ValueError('Need an original active permutation with at least512 rows before epoch end')
        self.model=Classifier(c,request['cache']).to('cuda:0')
        self.model.load_state_dict(s['model'],strict=True)
        if any(p.dtype!=torch.float32 for p in self.model.parameters()):
            raise ValueError('FP32 master parameters required')
        self.opt=optimizer(self.model,c);self.opt.load_state_dict(s['optimizer'])
        clock=SampleClock(c['epochs']*len(self.split),min(c['warmup_steps']*c['batch_size'],
            max(0,c['epochs']*len(self.split)-c['batch_size'])),
            [g.get('initial_lr',c['lr']) for g in self.opt.param_groups],
            s['epoch']*len(self.split)+s['cursor'],saved=s['sample_clock'],reference_batch=c['batch_size'])
        self.schedule=TorchSampleSchedule(self.opt,clock)
        self.progress={k:copy.deepcopy(s[k]) for k in ('step','epoch','cursor','epoch_loss','epoch_correct','epoch_samples')}
        self.permutation=s['permutation'].clone();self.restore_rng(s['rng'])
        self.opt.zero_grad(set_to_none=True);self.model.train()
        self.base=self.snapshot()

    def clone(self,value):
        if self.t.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value,self.np.ndarray): return value.copy()
        if isinstance(value,dict): return {key:self.clone(child) for key,child in value.items()}
        if isinstance(value,list): return [self.clone(child) for child in value]
        if isinstance(value,tuple): return tuple(self.clone(child) for child in value)
        return copy.deepcopy(value)

    def snapshot(self):
        self.t.cuda.synchronize()
        return self.clone(dict(model=self.model.state_dict(),optimizer=self.opt.state_dict(),
            scheduler=self.schedule.state_dict(),progress=self.progress,permutation=self.permutation,
            rng=self.rng_state(),gradients={n:p.grad for n,p in self.model.named_parameters()},
            precision=POLICY['precision'],policy=POLICY))

    def restore(self,state):
        self.t.cuda.synchronize()
        if state['policy']!=POLICY or state['precision']!=POLICY['precision']:
            raise ValueError('Recovery precision/policy changed')
        self.model.load_state_dict(state['model'],strict=True)
        self.opt.load_state_dict(self.clone(state['optimizer']))
        self.schedule.load_state_dict(self.clone(state['scheduler']))
        self.progress=self.clone(state['progress']);self.permutation=state['permutation'].clone()
        for name,p in self.model.named_parameters():
            value=state['gradients'][name];p.grad=None if value is None else value.to(p.device).clone()
        self.restore_rng(state['rng']);self.model.train()
        self.t.cuda.synchronize()

    def state_hash(self,state):
        h=hashlib.sha256()
        def visit(value):
            if self.t.is_tensor(value):
                h.update(b'tensor');h.update(str(value.dtype).encode());visit(tuple(value.shape))
                h.update(value.detach().cpu().contiguous().numpy().tobytes());return
            if isinstance(value,self.np.ndarray):
                h.update(b'array');h.update(value.dtype.str.encode());visit(tuple(value.shape));h.update(value.tobytes());return
            if isinstance(value,dict):
                h.update(b'dict')
                for key in sorted(value,key=lambda k:(type(k).__name__,repr(k))): visit(key);visit(value[key])
            elif isinstance(value,(list,tuple)):
                h.update(type(value).__name__.encode());h.update(str(len(value)).encode())
                for child in value: visit(child)
            else: h.update((type(value).__name__+':'+repr(value)+';').encode())
        visit(state);return h.hexdigest()

    @contextmanager
    def dropout(self,disabled):
        saved=[]
        if disabled:
            for module in self.model.modules():
                attr='p' if isinstance(module,self.t.nn.Dropout) else 'dropout' if isinstance(module,self.t.nn.MultiheadAttention) else None
                if attr is not None: saved.append((module,attr,getattr(module,attr)));setattr(module,attr,0.0)
        try: yield
        finally:
            for module,attr,value in saved: setattr(module,attr,value)

    def update(self,enabled,*,dropout_off=False,fp32_adapter=False,retain=False,executor=None,micro_batch=None):
        from .diagnostic_watch import mark
        torch=self.t;n=POLICY['logical_batch'];micro=POLICY['micro_batch'] if micro_batch is None else micro_batch;start=self.progress['cursor']
        if micro not in (16,32) or (executor is not None and (enabled or fp32_adapter)):
            raise ValueError('Isolated FP32 executor comparison requires micro16/32 and no attention adapter')
        if start+n>len(self.split): raise ValueError('Bounded diagnostic does not cross an epoch boundary')
        ids=self.permutation[start:start+n].numpy()
        # CPU selection plus transfer belongs to the timed complete update.
        arrays=self.split.numpy_batch(ids,validate=True)
        if executor is not None: executor.runtime.begin_update(micro)
        self.model.train();self.opt.zero_grad(set_to_none=True);loss_sum=0.;correct=0;outputs=[];finite=True
        with self.dropout(dropout_off), attention_route(torch,self.model,enabled,fp32_adapter=fp32_adapter) as stats:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                for offset in range(0,n,micro):
                    mark('attention_update_micro_'+str(offset//micro),flush=True)
                    batch={key:torch.as_tensor(value[offset:offset+micro],device='cuda:0') for key,value in arrays.items()}
                    logits=executor.forward(batch) if executor is not None else self.model(batch,validated_inputs=True)
                    loss=torch.nn.functional.cross_entropy(logits,batch['labels'],reduction='sum')/n
                    finite=finite & torch.isfinite(loss.detach())
                    loss_sum=loss_sum+loss.detach()*n
                    correct=correct+(logits.detach().argmax(-1)==batch['labels']).sum()
                    loss.backward()
                    if executor is not None and offset==0 and not executor.reference:
                        executor.runtime.isolate_gradients()
                    if retain: outputs.append(logits.detach().cpu())
                    del logits,loss,batch
        mark('attention_update_finite_clip',flush=True)
        if not bool(finite): raise ValueError('Nonfinite real loss')
        gradients={name:p.grad.detach().cpu().clone() for name,p in self.model.named_parameters() if p.grad is not None} if retain else None
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config['clip_norm'],error_if_nonfinite=True)
        self.opt.step();self.schedule.step(n)
        loss_sum=float(loss_sum);correct=int(correct)
        self.progress.update(step=self.progress['step']+1,cursor=start+n,
            epoch_loss=self.progress['epoch_loss']+loss_sum,
            epoch_correct=self.progress['epoch_correct']+correct,epoch_samples=self.progress['epoch_samples']+n)
        if not bool(torch.stack([torch.isfinite(p).all() for p in self.model.parameters()]).all()): raise ValueError('Nonfinite master parameter')
        torch.cuda.synchronize();mark('attention_update_complete',flush=True)
        result=dict(loss=loss_sum/n,correct=correct,rows=n,row_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
        if stats is not None:
            expected='torch.bfloat16' if enabled else 'torch.float32'
            if (stats['layer_calls']!=[2*(n//micro)]*4 or set(stats['qkv_dtypes'])!={expected}
                    or set(stats['output_dtypes'])!={expected}): raise ValueError('Incomplete attention dtype/call coverage')
            result['attention']=stats
        if retain:
            result.update(logits=torch.cat(outputs),gradients=gradients,
                deltas={name:p.detach().cpu()-self.base['model'][name] for name,p in self.model.named_parameters()})
        return result

    def relative(self,left,right):
        if isinstance(left,dict):
            if set(left)!=set(right): raise ValueError('Gradient/delta parameter coverage differs')
            keys=sorted(left);left=self.t.cat([left[k].reshape(-1).double() for k in keys]);right=self.t.cat([right[k].reshape(-1).double() for k in keys])
        else: left=left.double().reshape(-1);right=right.double().reshape(-1)
        if left.shape!=right.shape or not bool(self.t.isfinite(left).all() and self.t.isfinite(right).all()):
            raise ValueError('Invalid numerical comparison')
        return float((left-right).norm()/left.norm().clamp_min(1e-12))

    def metrics(self,reference,candidate):
        if reference['row_ids_sha256']!=candidate['row_ids_sha256']: raise ValueError('Compared rows changed')
        return dict(logits_relative_l2=self.relative(reference['logits'],candidate['logits']),
            loss_absolute=abs(reference['loss']-candidate['loss']),
            gradient_relative_l2=self.relative(reference['gradients'],candidate['gradients']),
            adam_delta_relative_l2=self.relative(reference['deltas'],candidate['deltas']))


def benchmark(runtime,output):
    from .diagnostic_watch import mark
    t=runtime.t;report=dict(passed=False,policy=POLICY)
    base_hash=runtime.state_hash(runtime.base)
    try:
        mark('attention_update_numerical',flush=True)
        runtime.restore(runtime.base);original=runtime.update(False,dropout_off=True,retain=True)
        runtime.restore(runtime.base);split_fp32=runtime.update(False,dropout_off=True,fp32_adapter=True,retain=True)
        parity=runtime.metrics(original,split_fp32)
        if max(parity.values())>1e-5: raise ValueError('FP32 instance adapter differs from original complete update')
        runtime.restore(runtime.base);candidate=runtime.update(True,dropout_off=True,retain=True)
        metrics=runtime.metrics(original,candidate)
        report.update(fp32_adapter_parity=parity,numerical=metrics,numerical_passed=numerical_gate(metrics))
        del original,split_fp32,candidate
        if not report['numerical_passed']: return report
        losses={};rows={}
        for route in ('fp32','bf16'):
            runtime.restore(runtime.base);losses[route]=[];rows[route]=[]
            for _ in range(POLICY['trajectory_updates']):
                value=runtime.update(route=='bf16',dropout_off=True)
                losses[route].append(value['loss']);rows[route].append(value['row_ids_sha256'])
        differences=[abs(a-b) for a,b in zip(losses['fp32'],losses['bf16'])]
        report['trajectory']=dict(updates=4,dropout=0,losses=losses,loss_absolute_differences=differences,
            rows_match=rows['fp32']==rows['bf16'],passed=rows['fp32']==rows['bf16'] and max(differences)<=POLICY['trajectory_loss_absolute'],
            convergence_certified=False)
        if not report['trajectory']['passed']: return report
        # Trace replays perform actual forwards/backwards from the same base state.
        from .attention_update_trace import collect_and_validate
        report['trace']=collect_and_validate(runtime,output)
        # Profile proof is outside timing. Only efficient SDPA is enabled for BOTH routes.
        proof={};memory={}
        for route in ('fp32','bf16'):
            runtime.restore(runtime.base);t.cuda.reset_peak_memory_stats()
            before=t.cuda.memory_allocated()
            with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU,t.profiler.ProfilerActivity.CUDA]) as profile:
                observed=runtime.update(route=='bf16')
            names=[event.key for event in profile.key_averages()]
            forward=any('_scaled_dot_product_efficient_attention' in name and 'backward' not in name for name in names)
            backward=any('_scaled_dot_product_efficient_attention_backward' in name for name in names)
            forbidden=any('_scaled_dot_product_' in name and any(word in name for word in ('flash','cudnn','math')) for name in names)
            if not forward or not backward or forbidden: raise ValueError('Requested efficient SDPA kernel proof failed')
            proof[route]=dict(efficient_forward=True,efficient_backward=True,other_sdpa=False,
                bf16_all_layer_calls=observed.get('attention') if route=='bf16' else None,
                level='ATen profiler operators; no CUDA kernel-name export')
            memory[route]=dict(start_allocated=before,peak_allocated=t.cuda.max_memory_allocated(),
                peak_reserved=t.cuda.max_memory_reserved(),scope='Shared warmed allocator; full-update peak, not isolated minimum')
            del profile
        # Warm and snapshot restore are explicitly outside measured intervals.
        for route in ('fp32','bf16'):
            runtime.restore(runtime.base)
            for _ in range(POLICY['warmup_updates']): runtime.update(route=='bf16')
        timings={'fp32':[],'bf16':[]};order=[]
        for pair in range(POLICY['timing_pairs']):
            for routes in (('fp32','bf16'),('bf16','fp32')):
                for route in routes:
                    runtime.restore(runtime.base);t.cuda.synchronize();start=time.perf_counter()
                    runtime.update(route=='bf16');t.cuda.synchronize()
                    timings[route].append(time.perf_counter()-start);order.append(route)
        report.update(timing=timing_summary(timings),timing_order=order,kernel_proof=proof,memory=memory,
            timing_scope='CPU minibatch gather/H2D, forward/real CE/backward, finite checks/clip, Adam/sample schedule; original dropout. Restore/warm/profile/trace excluded.')
        report['passed']=report['timing']['gain_passed'] and report['trace']['passed']
        return report
    finally:
        runtime.restore(runtime.base)
        restored=runtime.state_hash(runtime.snapshot())==base_hash
        if not restored: raise ValueError('Full-update rollback differs')
        report['rollback']=dict(full_state_exact=True,committed_samples=0,committed_updates=0)
        atomic_json(output/'benchmark.json',report)


def execute(request_path,output,mode,recovery_path=None):
    from .attention_precision_diagnostic import unchanged,SOURCE_REVISION,JOB
    from .diagnostic_watch import mark
    request=read(request_path);selection=validate_request(request,revision())
    if mode=='attribution':
        from .attention_attribution import bind_prior
        keys=('source_suite','source_config','source_config_sha256','checkpoint','source_metadata_sha256','data','cache')
        prior=request['attribution_binding']
        if bind_prior(prior['prior_suite'],{k:request[k] for k in keys})!=prior:
            raise ValueError('Attribution binding changed before native execution')
    elif 'attribution_binding' in request:
        raise ValueError('Attribution request cannot enter another native mode')
    source=Path(request['source_suite']);p=request['checkpoint'];config=request['source_config']
    if (source.parent.name!='runs' or config.get('source_revision')!=SOURCE_REVISION
            or read(source/'jobs'/JOB/'config.json')!=config
            or p['path']!=str(source/'jobs'/JOB/'checkpoints'/p['file'])
            or Path(p['path']).is_symlink()): raise ValueError('Original hash-bound source required')
    unchanged(request)
    import torch
    import torch.nn.attention
    import numpy as np
    if (torch.__version__.split('+')[0]!='2.7.1' or np.__version__!='2.2.6'
            or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('Pinned existing CUDA BF16 environment required')
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    fastpath=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
    runtime=None;result={}
    try:
        mark('attention_update_source_load',flush=True);runtime=NativeUpdate(torch,np,request)
        if mode=='attribution':
            from .attention_attribution import run
            result=run(runtime,output)
        elif mode=='benchmark': result=benchmark(runtime,output)
        else:
            if mode=='resume':
                if (recovery_path is None or recovery_path.is_symlink()
                        or recovery_path.resolve()!=(output.parent/'interrupt/diagnostic-state.pt').resolve()):
                    raise ValueError('Explicit sibling diagnostic recovery state required')
                receipt=read(recovery_path.with_suffix('.json'))
                if (receipt.get('schema')!='qkan-bf16-diagnostic-checkpoint.v1'
                        or receipt.get('request_sha256')!=digest(request_path)
                        or digest(recovery_path)!=receipt.get('sha256')): raise ValueError('Diagnostic recovery receipt differs')
                recovered=torch.load(recovery_path,map_location='cpu',weights_only=False)
                if runtime.state_hash(recovered)!=receipt['state_sha256']: raise ValueError('Diagnostic state identity differs')
                runtime.restore(recovered)
            try:
                count=2 if mode=='uninterrupted' else 1
                updates=[runtime.update(True) for _ in range(count)]
                final=runtime.snapshot();final_hash=runtime.state_hash(final)
                result=dict(passed=True,full_state_sha256=final_hash,updates=updates,
                    step=runtime.progress['step'],cursor=runtime.progress['cursor'],
                    schedule_progress=runtime.schedule.state_dict()['progress'])
                if mode=='interrupt':
                    checkpoint=output/'diagnostic-state.pt';torch.save(final,checkpoint)
                    if checkpoint.stat().st_size>128*1024**2:
                        checkpoint.unlink();raise ValueError('Diagnostic checkpoint exceeded128MiB bound')
                    receipt=dict(schema='qkan-bf16-diagnostic-checkpoint.v1',sha256=digest(checkpoint),
                        state_sha256=final_hash,request_sha256=digest(request_path),bytes=checkpoint.stat().st_size)
                    atomic_json(checkpoint.with_suffix('.json'),receipt)
                    result['diagnostic_checkpoint']=dict(file=checkpoint.name,**receipt)
                del final
            finally:
                runtime.restore(runtime.base)
                if runtime.state_hash(runtime.snapshot())!=runtime.state_hash(runtime.base): raise ValueError('Recovery child rollback differs')
                result['rollback']=dict(full_state_exact=True,committed_samples=0,committed_updates=0)
        result.update(schema='qkan-attention-update-native.v1',mode=mode,request_sha256=digest(request_path),
            checkpoint_sha256=p['sha256'],checkpoint_step=p['step'],plugin_manifest=selection.manifest(),
            policy_sha256=fingerprint(POLICY),formal_training_started=False,training_samples_committed=0,
            environment=dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),deterministic_algorithms=True,tf32=False))
        atomic_json(output/'result.json',result)
        return 0 if result['passed'] else 76
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath);unchanged(request)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('uninterrupted','interrupt','resume','benchmark','attribution'),required=True)
    parser.add_argument('--recovery-state',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if any((args.output/name).exists() for name in ('result.json','benchmark.json','diagnostic-state.pt','attribution.json','paired-parameters.pt')):
        parser.error('Use a fresh private diagnostic output directory')
    if (args.mode=='resume')!=(args.recovery_state is not None): parser.error('Recovery state only for fresh-process resume')
    try: return execute(args.request,args.output,args.mode,args.recovery_state)
    except BaseException as error:
        worker_failure(args.output,error);raise


if __name__=='__main__': raise SystemExit(main())
