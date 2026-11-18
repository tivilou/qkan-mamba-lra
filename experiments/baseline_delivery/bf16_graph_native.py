"""Lazy native BF16 + Graph experiment; never imported on the owner test route."""
from pathlib import Path
import time

from .common import atomic_json, digest, read
from .bf16_graph_diagnostic import CLOSED, ROLLBACK, SCHEMA, binding
from .key_bias_consistency import analytic_key_bias, validate_observation
from .key_bias_stability_native import compare_values, compare_summary, ParityMismatch
from .diagnostic_watch import mark

ORDER = ('incumbent', 'candidate', 'candidate', 'incumbent')*2


def update(runtime, graph=None, retain=False):
    before = graph.status() if graph is not None else None
    with analytic_key_bias(runtime.t, runtime.model, True) as observed:
        value = runtime.update(graph is None, executor=graph, micro_batch=16, retain=retain)
    validate_observation(observed)
    if graph is not None:
        after = graph.status()
        if (not before['graph_resident'] or not after['graph_resident']
                or before['graph_builds'] != 1 or after['graph_builds'] != 1
                or after['forward_replays']-before['forward_replays'] != 16
                or after['backward_replays']-before['backward_replays'] != 16
                or after['replica_sync_rounds']-before['replica_sync_rounds'] != 8):
            raise ValueError('Resident BF16 Graph replay/sync coverage differs')
    value.pop('attention', None)  # Host capture/eager telemetry, not scientific state.
    return value, observed


def operation(runtime, graph, output, mode, request_sha):
    """Injectable full-update operation; all hashes/restore are outside timing."""
    from .key_bias_recovery import save_state, load_state
    report = dict(passed=False, mode=mode, stage='begin', actual_strategy='bf16_sdpa_fp32_master_analytic_k_bias',
        snapshot_adapter='legacy nine-component diagnostic envelope; not a formal checkpoint')
    base_hash = runtime.state_hash(runtime.base)
    try:
        if mode == 'parity':
            mark('bf16_graph_parity_eager', flush=True)
            eager, _ = update(runtime, retain=True); eager_state = runtime.snapshot()
            runtime.restore(runtime.base)
            mark('bf16_graph_parity_graph', flush=True)
            candidate, hook = update(runtime, graph, retain=True); graph_state = runtime.snapshot()
            compare_values(runtime, eager, candidate, 'update')
            compare_values(runtime, eager_state, graph_state, 'state')
            report.update(parity=dict(passed=True, rtol=1e-5, atol=1e-6, rng_and_controls='exact',
                covers='logits/all gradients/all parameter deltas/Adam/scheduler/permutation/progress/RNG'),
                hook_observation=hook, graph=graph.status(), passed=True)
        elif mode == 'timing':
            report['stage'] = 'warmup'
            for route in ('incumbent', 'candidate'):
                runtime.restore(runtime.base)
                update(runtime, graph if route == 'candidate' else None)
            records = []; report['timing_records'] = records; report['stage'] = 'timing'
            for index, route in enumerate(ORDER):
                runtime.restore(runtime.base); runtime.t.cuda.synchronize()
                before = graph.status()
                mark('bf16_graph_timing_'+str(index)+'_'+route, flush=True)
                start = time.perf_counter()
                value, _ = update(runtime, graph if route == 'candidate' else None)
                runtime.t.cuda.synchronize(); elapsed = time.perf_counter()-start
                records.append(dict(route=route, seconds=elapsed, update=value,
                    full_state_sha256=runtime.state_hash(runtime.snapshot()),
                    graph_before=before, graph_after=graph.status()))
                atomic_json(Path(output)/'partial.json', report)
            report.update(timing=compare_summary(records), graph=graph.status())
            report['passed'] = report['timing']['passed']
        else:
            report['starting_state_sha256'] = runtime.state_hash(runtime.snapshot())
            if mode == 'resume':
                report['loaded_state_sha256'] = load_state(runtime,
                    Path(output).parent/'interrupt/diagnostic-state.pt', output, request_sha)
            values = []; observations = []
            for index in range(2 if mode == 'continuous' else 1):
                mark('bf16_graph_'+mode+'_'+str(index), flush=True)
                value, hook = update(runtime, graph); values.append(value); observations.append(hook)
            state = runtime.snapshot(); count = 2 if mode in ('continuous', 'resume') else 1
            if (runtime.progress['step'] != runtime.base['progress']['step']+count
                    or runtime.progress['cursor'] != runtime.base['progress']['cursor']+128*count
                    or runtime.progress['epoch'] != runtime.base['progress']['epoch']
                    or state['scheduler']['progress'] != runtime.base['scheduler']['progress']+128*count):
                raise ValueError('Exact bounded update/sample clock differs')
            report.update(passed=True, updates=values, hook_observations=observations,
                full_state_sha256=runtime.state_hash(state),
                component_sha256={k: runtime.state_hash(v) for k, v in state.items()}, graph=graph.status())
            if mode == 'interrupt':
                report['diagnostic_checkpoint'] = save_state(runtime, Path(output)/'diagnostic-state.pt', request_sha)
        report['stage'] = 'complete'
        return report
    except ParityMismatch as error:
        report['parity_failure'] = error.report
        raise
    finally:
        runtime.restore(runtime.base)
        if runtime.state_hash(runtime.snapshot()) != base_hash: raise ValueError('Diagnostic full-state rollback differs')
        report['rollback'] = dict(ROLLBACK); atomic_json(Path(output)/'partial.json', report)


def native(request, request_path, output, mode):
    from . import key_bias_training as strategy
    from .attention_update_native import NativeUpdate
    from .bf16_graph_runtime import create
    from .attention_precision_diagnostic import unchanged
    import torch
    import torch.nn.attention
    import numpy as np
    if (torch.__version__ != '2.7.1+cu126' or np.__version__ != '2.2.6'
            or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('Existing pinned CUDA BF16 runtime required')
    if not strategy.select(request['source_config']): raise ValueError('Existing BF16 strategy required')
    torch.set_num_threads(1); torch.manual_seed(request['source_config']['seed'])
    torch.cuda.manual_seed_all(request['source_config']['seed'])
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    runtime = None; graph = None
    try:
        with strategy.process(torch):
            runtime = NativeUpdate(torch, np, request)
            formal = read(Path(request['source_suite'])/'formal-input.json')
            expected = strategy.envelope(request['source_config'], formal['checkpoint']['sha256'], formal['implementation'])
            strategy.check_checkpoint(runtime.source, request['source_config'], expected)
            gradients = runtime.source['strategy_gradients']
            if set(gradients) != {n for n, _ in runtime.model.named_parameters()}:
                raise ValueError('Complete inherited gradient population required')
            for name, parameter in runtime.model.named_parameters():
                value = gradients[name]
                parameter.grad = None if value is None else value.to(parameter.device).clone()
            runtime.base = runtime.snapshot(); initial = runtime.state_hash(runtime.base)
            graph = create(torch, runtime.model, runtime.rng_state, runtime.restore_rng)
            ids = runtime.permutation[runtime.progress['cursor']:runtime.progress['cursor']+16].numpy()
            arrays = runtime.split.numpy_batch(ids, validate=True)
            inputs = tuple(torch.as_tensor(arrays[name], device='cuda:0') for name in ('inputs1', 'inputs2'))
            mark('bf16_graph_prepare', flush=True); start = time.perf_counter()
            graph.prepare(inputs); prepare_seconds = time.perf_counter()-start
            runtime.restore(runtime.base)
            if runtime.state_hash(runtime.snapshot()) != initial: raise ValueError('Capture changed scientific state/RNG')
            result = operation(runtime, graph, output, mode, digest(request_path))
            result.update(schema=SCHEMA, mode=mode, request_sha256=digest(request_path), implementation=binding(),
                checkpoint_sha256=request['checkpoint']['sha256'], **CLOSED,
                preparation_seconds=prepare_seconds, preparation_outside_timing=True,
                environment=dict(torch=str(torch.__version__), numpy=str(np.__version__), deterministic=True, tf32=False))
            atomic_json(Path(output)/'result.json', result)
    finally:
        if graph is not None: graph.clear()
        if runtime is not None:
            runtime.restore(runtime.base)
            if runtime.state_hash(runtime.snapshot()) != runtime.state_hash(runtime.base):
                raise ValueError('Final diagnostic rollback differs')
        unchanged(request)
