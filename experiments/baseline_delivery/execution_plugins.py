"""Framework-free execution extensions; the worker owns training and persistence.

Only allowlisted factories create plugins. Hooks are passed per run; no module
globals are replaced. Scientific configuration and parameter names stay intact.
"""
from contextlib import nullcontext
from dataclasses import dataclass


REGISTRY = {
    'mamba.fused_fp32': dict(version=2, models=('mamba1',),
        tasks=('retrieval','image','pathfinder'), precision='fp32', backend='eager'),
}


def resolve(value, config):
    if value is None:return None
    if not isinstance(value,dict) or set(value)!={'plugin','version'}:
        raise ValueError('Execution plugin needs explicit plugin/version')
    spec=REGISTRY.get(value['plugin'])
    if (spec is None or type(value['version']) is not int or value['version']!=spec['version']
            or config.get('model') not in spec['models'] or config.get('task') not in spec['tasks']
            or config.get('training_strategy') or (config.get('torch_acceleration') or {}).get('required') is not None):
        raise ValueError('Unsupported execution plugin combination')
    return dict(value,precision=spec['precision'],backend=spec['backend'])


def create(value, config, torch, model):
    resolved=resolve(value,config)
    if resolved is None:return None
    from .mamba_convolution import InstanceFusedFP32
    return InstanceFusedFP32(torch,model)


def eager_state(saved):
    return {**(saved or {}),'mode':'eager','benchmarked':True,
        'reason':'measured_fp32_fused_eager','benchmark':None}


def reset_measurements(controller):
    """A new execution mechanism invalidates timings, not scientific progress."""
    controller.state.update(samples=[],trial_windows=[],baseline=None,previous=None,
        wait=0,warmup=controller.policy['warmup'])
    if 'previous_logical' in controller.state:controller.state['previous_logical']=None


@dataclass
class Hooks:
    """Optional stable worker boundary. Defaults preserve ordinary execution."""
    model_context: object = lambda model:nullcontext()
    acceleration_state: object = lambda saved:saved
    configure_controller: object = lambda controller,state:None
    before_save: object = lambda state:None
    after_save: object = lambda state:None
    on_progress: object = lambda heartbeat:None
