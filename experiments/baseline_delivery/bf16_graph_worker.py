"""Actual unchanged worker with an explicitly admitted BF16 Graph execution context.

Only collaborator children enter native(); --help and imports are stdlib-only.
Finite modes run on disposable checkpoint copies and never seed the formal run.
"""
import argparse
from pathlib import Path
import signal
from types import SimpleNamespace

from .common import atomic_json, digest, read, lock, worker_failure
from . import bf16_graph_formal as formal
from . import key_bias_training as strategy
from .bf16_graph_training import Execution, POLICY
from .key_bias_worker_admission import scientific as original_scientific, EXCLUDE


def scientific(state):
    # One additional declared execution lineage field; keep every scientific key.
    return {k:v for k,v in original_scientific(state).items() if k != 'bf16_graph_execution'}


class Admission(strategy.Admission):
    def __init__(self, request, path, output, mode):
        self.request=request; self.path=Path(path); self.output=Path(output); self.mode=mode
        self.expected=request['expected_strategy']; self.check_only=False; self.trace=mode=='resume'
        self.execution=Execution(output,request['execution_binding'],enabled=mode!='reference')

    def validate(self,config,output,device,resume,stop_after,synthetic,migration,probe):
        r=self.request; suite=self.path.parent
        finite=self.mode in formal.MODES
        expected_output=suite/'preflight'/self.mode if finite else suite/'jobs'/formal.JOB
        limit=r['source_checkpoint']['step']+(1 if self.mode=='interrupt' else 2) if finite else None
        if (self.mode not in (*formal.MODES,'formal') or self.path.is_symlink()
                or Path(output).resolve()!=expected_output.resolve() or self.output.resolve()!=expected_output.resolve()
                or config!=r['target_config'] or not strategy.select(config) or device!='cuda:0'
                or not resume or synthetic or migration or probe is not None or stop_after!=limit
                or formal.verify(self.path,ready=not finite)!=r):
            raise ValueError('Bound actual BF16 Graph worker invocation required')
        formal.unchanged(r)

    def record(self,state,ids,loss,correct):
        super().record(state,ids,loss,correct)
        self.execution.publish(state)


def check_lineage(state,r,initial_sha):
    marker=state.get('bf16_graph_execution')
    if marker is None:
        if initial_sha!=r['source_checkpoint']['sha256']:
            raise ValueError('Unmarked advanced checkpoint rejected; use the copied origin generation')
    elif (marker.get('policy')!=POLICY or marker.get('implementation')!=r['execution_binding']
          or marker.get('enabled') is not True):
        raise ValueError('Checkpoint Graph execution lineage differs')


def native(path,output,mode):
    from .key_bias_formal import native_runtime
    torch,np=native_runtime()
    from . import worker
    from .attention_update_native import NativeUpdate
    from .key_bias_stability_native import compare_values
    from .key_bias_recovery import prepare_worker_output
    from .migration import checked_pointer,copy_verified
    from .diagnostic_watch import mark
    path=Path(path);output=Path(output);suite=path.parent
    r=formal.verify(path,ready=mode=='formal');formal.unchanged(r)
    admission=Admission(r,path,output,mode)
    finite=mode!='formal';limit=r['source_checkpoint']['step']+(1 if mode=='interrupt' else 2) if finite else None
    admission.validate(r['target_config'],output,'cuda:0',True,limit,False,False,None)
    hasher=SimpleNamespace(t=torch,np=np)
    state_hash=lambda value:NativeUpdate.state_hash(hasher,value)
    source=suite/'jobs'/formal.JOB
    source_pointer,_=checked_pointer(source)
    baseline=worker.load_checkpoint(source,r['source_checkpoint']['contract'])
    strategy.check_checkpoint(baseline,r['target_config'],admission.expected)
    if not finite: check_lineage(baseline,r,source_pointer['sha256'])
    try:
        if finite:
            prepare_worker_output(output)
            if (source_pointer['sha256']!=r['source_checkpoint']['sha256']
                    or baseline.get('training_complete') or baseline.get('permutation') is None
                    or baseline['cursor']+256>len(baseline['permutation'])):
                raise ValueError('Original active full-state source with256 remaining rows required')
            prior=suite/'preflight/interrupt' if mode=='resume' else source
            pointer,generation=checked_pointer(prior)
            if mode=='resume':
                receipt=formal.accept(read(prior/'result.json'),r,digest(path),'interrupt')
                if pointer['sha256']!=receipt['checkpoint_sha256']: raise ValueError('Interrupted generation differs')
            copy_verified(prior,output,{'checkpoints/'+generation.name:pointer['sha256']})
            atomic_json(output/'checkpoints/latest.json',{**pointer,'previous':[]})
            state=worker.load_checkpoint(output,r['source_checkpoint']['contract'])
            check_lineage(state,r,pointer['sha256'])
            # Reopen all serialized values, including gradients, RNG, Adam and selection.
            prior_state=worker.load_checkpoint(prior,r['source_checkpoint']['contract'])
            roundtrip=state_hash(state)==state_hash(prior_state)
            if not roundtrip: raise ValueError('Byte-copied full-state checkpoint roundtrip differs')
            loaded=state_hash(scientific(state))
        else:
            state=baseline;roundtrip=True
        worker.PAUSE=False
        old={}
        for name in ('SIGTERM','SIGINT','SIGHUP'):
            if hasattr(signal,name):
                sig=getattr(signal,name);old[sig]=signal.signal(sig,worker.request_pause)
        try:
            with lock(output/'worker.lock'), admission.execution.install():
                mark('bf16_graph_actual_worker_'+mode,flush=True)
                code=worker.run(r['target_config'],r['data'],r['cache'],output,'cuda:0',
                    resume=True,stop_after=limit,strategy_admission=admission)
        finally:
            for sig,handler in old.items(): signal.signal(sig,handler)
        if not finite:
            if code==0:
                summary=read(output/'summary.json')
                summary['bf16_graph_execution']=read(output/'bf16-graph-execution.json')
                atomic_json(output/'summary.json',summary)
            return code
        if code!=75: raise ValueError('Actual worker finite stop did not save a paused generation')
        final=worker.load_checkpoint(output,r['source_checkpoint']['contract'])
        strategy.check_checkpoint(final,r['target_config'],admission.expected)
        delta=128*(1 if mode in ('interrupt','resume') else 2)
        if (final['epoch']!=state['epoch'] or final['cursor']!=state['cursor']+delta
                or final['sample_clock']['progress']!=state['sample_clock']['progress']+delta
                or final['strategy_updates'][:len(baseline['strategy_updates'])]!=baseline['strategy_updates']):
            raise ValueError('Worker sampler/scheduler/history progression differs')
        pointer,_=checked_pointer(output)
        execution=read(output/'bf16-graph-execution.json')
        if execution['step']!=final['step']: raise ValueError('Actual execution receipt is stale')
        result=dict(schema=formal.REPORT_SCHEMA,passed=True,mode=mode,request_sha256=digest(path),
            execution_binding=r['execution_binding'],source_checkpoint_sha256=r['source_checkpoint']['sha256'],
            checkpoint_sha256=pointer['sha256'],step=final['step'],cursor=final['cursor'],
            sample_clock=final['sample_clock'],loaded_state_sha256=loaded,
            scientific_state_sha256=state_hash(scientific(final)),
            component_sha256={k:state_hash(v) for k,v in scientific(final).items()},
            updates=final['strategy_updates'][len(baseline['strategy_updates']):],execution=execution,
            copy_roundtrip_exact=roundtrip,worker_exit_status=code,formal_training_started=False,
            training_samples_committed=0,excluded_execution_telemetry=[*EXCLUDE,'bf16_graph_execution'])
        if mode=='resume':
            reference=worker.load_checkpoint(suite/'preflight/reference',r['source_checkpoint']['contract'])
            continuous=worker.load_checkpoint(suite/'preflight/continuous',r['source_checkpoint']['contract'])
            compare_values(hasher,scientific(reference),scientific(continuous))
            result['same_worker_parity']=dict(passed=True,rtol=1e-5,atol=1e-6,
                rng_and_controls='exact',covers='every persisted scientific field including gradients/history/Adam/selection')
            result['trace']=read(output/'strategy-trace.json')
            # Trace's own validator checks same model/gradients/RNG around real evaluation.
            if result['trace'].get('model_gradients_rng_unchanged') is not True:
                raise ValueError('Actual worker trace changed state')
        formal.accept(result,r,digest(path),mode)
        atomic_json(output/'result.json',result)
        return 0
    finally: formal.unchanged(r)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=(*formal.MODES,'formal'),required=True);args=p.parse_args()
    try: return native(args.request,args.output,args.mode)
    except BaseException as error:
        worker_failure(args.output,error)
        raise


if __name__=='__main__': raise SystemExit(main())
