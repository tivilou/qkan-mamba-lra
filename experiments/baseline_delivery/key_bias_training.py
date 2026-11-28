"""Versioned final-worker candidate. Native dependencies are injected, never imported.

This registry is separate from the historical diagnostic plugin registry. It is
closed to unbounded training until final-path admission is reviewed.
"""
from contextlib import contextmanager
import copy
import math
from pathlib import Path

from .common import ROOT, atomic_json, digest, fingerprint, read
from .key_bias_consistency import CANDIDATE, GRADIENT_POLICY, NAMES, analytic_key_bias
from .attention_update import attention_route

POLICY = dict(schema='qkan-training-strategy.v1', candidate_id=CANDIDATE,
    gradient_policy=GRADIENT_POLICY, precision='bf16_sdpa_fp32_master_v1',
    micro_batch=16, logical_batch=128, evaluation='bf16_attention_fp32_head_v1', graph=False)
FILES = ('worker.py', 'key_bias_training.py', 'key_bias_worker_admission.py',
         'attention_update.py', 'key_bias_consistency.py', 'models.py',
         'logical_batch.py', 'execution.py', 'trace.py', 'data.py', 'migration.py',
         'adaptive_memory.py', 'performance.py', 'common.py', 'diagnostic_watch.py',
         'attention_precision_diagnostic.py', 'attention_update_native.py', 'key_bias_recovery.py')
PLAN = ROOT/'docs/baselines/attention-key-bias-worker-plan.json'
REVIEW = ROOT/'docs/evidence/key-bias-stability-native-review-20261117.json'


def binding():
    return dict(files={n:digest(Path(__file__).parent/n) for n in FILES},
                trace_validator_sha256=digest(ROOT/'tools/sample_trace_contract/validate.py'),
                plan_sha256=digest(PLAN), pilot_review_sha256=digest(REVIEW))


def select(config):
    value=config.get('training_strategy')
    if value is None: return False
    if (value != POLICY or config.get('model')!='transformer' or config.get('task')!='retrieval'
            or (config.get('layers'),config.get('width'),config.get('heads'))!=(4,128,4)
            or config.get('length',4000)!=4000 or config.get('dropout')!=.1
            or config.get('torch_acceleration') or not config.get('adaptive_memory')
            or config.get('canary_logical_tiers') is not None):
        raise ValueError('Unsupported explicit training strategy/combination')
    return True


def target_config(source):
    if source.get('training_strategy') is not None: raise ValueError('Original FP32 source required')
    result=copy.deepcopy(source);result.pop('torch_acceleration',None)
    result['training_strategy']=dict(POLICY);result['micro_batch']=16
    result['adaptive_memory']['execution']['precision']=POLICY['precision']
    select(result)
    return result


def envelope(config,source_checkpoint,implementation):
    return dict(schema='qkan-training-strategy-checkpoint.v1', policy=dict(POLICY),
        config_sha256=fingerprint(config), source_checkpoint_sha256=source_checkpoint,
        implementation=implementation, transition='original_fp32_to_bf16_analytic_v1')


def check_checkpoint(state,config,expected):
    enabled=select(config)
    if not enabled:
        if state.get('training_strategy') is not None: raise ValueError('Candidate checkpoint requires candidate policy')
        return
    if state.get('training_strategy') != expected or not isinstance(state.get('strategy_gradients'),dict):
        raise ValueError('Checkpoint strategy/implementation/gradient state differs')


def pin(controller):
    if controller is None or 128 not in controller.policy.get('logical_tiers',[]):
        raise ValueError('Original sample-clock controller with logical128 required')
    controller.state['logical_index']=controller.policy['logical_tiers'].index(128)
    controller.state['index']=controller.policy['tiers'].index(16)
    controller.state['ceiling']=len(controller.policy['tiers'])-1
    controller.state.pop('capacity_ceiling',None)


@contextmanager
def process(torch):
    old=torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try: yield
    finally: torch.backends.mha.set_fastpath_enabled(old)


@contextmanager
def update(torch,model,micro,count):
    if micro!=16 or not 1<=count<=128: raise ValueError('Candidate requires eager16/logical128')
    with attention_route(torch,model,True) as forward, analytic_key_bias(torch,model,True) as gradient:
        yield
    calls=math.ceil(count/16)
    if (gradient['hook_calls']!={n:calls for n in NAMES} or not gradient['handles_removed']
            or forward['layer_calls']!=[2*calls]*4
            or set(forward['qkv_dtypes'])!={'torch.bfloat16'}
            or set(forward['output_dtypes'])!={'torch.bfloat16'}):
        raise ValueError('Final-worker BF16/hook coverage differs')


class Admission:
    """An owned, finite final-worker invocation; not a formal queue capability."""
    def __init__(self,request,path,output,mode):
        self.request=request;self.path=Path(path);self.output=Path(output);self.mode=mode
        self.check_only=mode=='migration'
        self.trace=mode=='resume'
        self.expected=envelope(request['target_config'],request['checkpoint']['sha256'],request['implementation'])

    def validate(self,config,output,device,resume,stop_after,synthetic,migration,probe):
        r=self.request;root=self.path.parent.resolve()
        if (r.get('schema')!='qkan-key-bias-worker-input.v1' or r['implementation']!=binding()
                or read(self.path)!=r or self.path.is_symlink() or not root.is_relative_to((ROOT/'runs').resolve())
                or not root.name.startswith('key-bias-worker-')
                or Path(output).resolve()!=self.output.resolve() or self.output.resolve()!=root/self.mode
                or self.mode not in ('migration','continuous','interrupt','resume')
                or config!=r['target_config'] or not select(config) or device!='cuda:0'
                or not resume or synthetic or migration or probe is not None
                or stop_after!=r['checkpoint']['step']+(1 if self.mode=='interrupt' else 2)):
            raise ValueError('Finite owned final-worker admission required')
        from .attention_precision_diagnostic import unchanged
        unchanged(r)

    def record(self,state,ids,loss,correct):
        if not math.isfinite(float(loss)): raise ValueError('Nonfinite candidate update')
        state.setdefault('strategy_updates',[]).append(dict(step=state['step'],
            row_ids_sha256=fingerprint([int(i) for i in ids]),samples=len(ids),
            loss=float(loss)/len(ids),correct=int(correct),micro_batch=16))

    def capture_trace(self,torch,model,split,config,checkpoint):
        from .worker import rng_state,restore_rng
        from .attention_update_native import NativeUpdate
        from types import SimpleNamespace
        import numpy as np
        hasher=SimpleNamespace(t=torch,np=np)
        snapshot=lambda:NativeUpdate.state_hash(hasher,dict(model=model.state_dict(),
            gradients={n:p.grad for n,p in model.named_parameters()},rng=rng_state()))
        before=snapshot();rng=rng_state();training=model.training;routes={}
        try:
            for route,enabled in (('fp32',False),('bf16_analytic',True)):
                with attention_route(torch,model,enabled):
                    routes[route]=self._capture(torch,model,split,config,checkpoint,self.output/('trace-'+route),route)
            if snapshot()!=before: raise ValueError('Final-worker trace changed model/gradients/RNG')
            if routes['fp32']['sample_ids']!=routes['bf16_analytic']['sample_ids']:
                raise ValueError('Paired final-worker trace samples differ')
            total=sum(p.stat().st_size for p in self.output.glob('trace-*/trace-values/*.npy'))
            if total>128*1024**2: raise ValueError('Final-worker private trace exceeds128MiB')
            return dict(validation=dict(passed=True),routes=routes,bytes=total,
                source='Same actual paused worker weights; original FP32 vs candidate BF16 evaluation',
                model_gradients_rng_unchanged=True,formal_model_case=False)
        finally:
            restore_rng(rng);model.train(training)

    def _capture(self,torch,model,split,config,checkpoint,output,route):
        """Use final worker's actual model/forward and the production value writer.

        Retag the writer's best-validation envelope: this is an unselected latest
        preflight state, never a best-model or utility result. Values stay private.
        """
        from .trace import capture,semantic_validate
        from tools.sample_trace_contract.validate import validate
        output.mkdir(parents=True)
        capture(model,split,output,config,checkpoint)
        path=output/'sample-trace.json';value=read(path)
        value['trace_id']=self.output.parent.name+'-latest-preflight'
        value['experiment']['training_strategy']=self.expected
        value['experiment']['code_revision']=self.request['source_revision']
        value['experiment']['counterfactual_evaluation_route']=route
        value['task_contract']['required_checkpoints']=['latest_preflight']
        value['task_contract']['checkpoint_policy']='Latest bounded final-worker checkpoint; no model selection'
        value['sample_selection']['rule']='First two frozen validation rows; fixed in plan before execution'
        for sample in value['samples']:
            sample['checkpoint']='latest_preflight'
            for stage in sample['stages']:
                if 'checkpoint' in stage: stage['checkpoint']='latest_preflight'
        errors,warnings=validate(value,require_complete=True)
        if errors or warnings: raise ValueError('Final-worker trace contract differs: '+str(errors+warnings))
        atomic_json(path,value);semantic_validate(path)
        for sample in value['samples']:
            for rep in sample['representations']:
                for ref in rep['artifact_refs']:
                    if digest(output/ref['path'])!=ref['sha256']: raise ValueError('Trace values changed')
        check=read(output/'trace-validation.json')
        if check.get('passed') is not True or check['checkpoint_sha256']!=checkpoint or check['samples']!=2:
            raise ValueError('Actual two-sample final-worker trace readiness differs')
        return dict(trace_sha256=digest(path),validation=read(output/'trace-validation.json'),
            sample_ids=[s['sample_id'] for s in value['samples']],checkpoint='latest_preflight',
            private=True,complete_values_reopened=True)
