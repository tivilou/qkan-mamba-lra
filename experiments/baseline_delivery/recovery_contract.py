"""Scoped numerical recovery contract; exact control state remains mandatory."""
from .recovery_audit import FIELDS, differences

POLICY = {
    'schema': 'qkan-recovery-policy.v1',
    'id': 'mamba1-cuda-fp32-v1',
    'rtol': 1e-5, 'atol': 1e-6,
    'scope': 'Mamba1 CUDA float32 bounded fresh-process canary only',
    'basis': 'Reuse the declared FP32 trace replay engineering budget; '
             'not a confidence interval or a measured noise upper bound.',
    'claim': 'Numerical continuation acceptance at the bounded checkpoint; '
             'no guarantee of bitwise or long-horizon trajectory equality.'}
NUMERICAL_FIELDS = {'model', 'optimizer', 'best_weights', 'history', 'update_trace', 'epoch_loss'}


def validate(left, right):
    import numpy as np
    import torch
    failures = []

    def visit(a, b, path, numerical):
        if type(a) is not type(b):
            failures.append(path); return
        if isinstance(a, dict):
            if a.keys() != b.keys(): failures.append(path); return
            for key in a: visit(a[key], b[key], path+'/'+str(key), numerical)
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b): failures.append(path); return
            for i, (x, y) in enumerate(zip(a, b)): visit(x, y, path+'/'+str(i), numerical)
        elif isinstance(a, (torch.Tensor, np.ndarray)):
            x = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
            y = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b
            if x.shape != y.shape or x.dtype != y.dtype:
                failures.append(path); return
            floating = x.dtype.kind in 'fc'
            if floating and (not np.isfinite(x).all() or not np.isfinite(y).all()):
                failures.append(path); return
            if floating and numerical:
                # Symmetric per-element criterion; never a global tensor-scale bound.
                delta = np.abs(x.astype(np.complex128 if x.dtype.kind == 'c' else np.float64)-y)
                bound = POLICY['atol']+POLICY['rtol']*np.maximum(np.abs(x), np.abs(y))
                equal = bool(np.all(delta <= bound))
            else:
                equal = bool(np.array_equal(x, y))
            if not equal: failures.append(path)
        elif isinstance(a, (float, np.floating)):
            finite = bool(np.isfinite(a) and np.isfinite(b))
            equal = abs(a-b) <= POLICY['atol']+POLICY['rtol']*max(abs(a), abs(b)) if numerical else a == b
            if not finite or not equal: failures.append(path)
        elif a != b:
            failures.append(path)

    leaves = []
    for field in FIELDS:
        visit(left[field], right[field], field, field in NUMERICAL_FIELDS)
        leaves.extend(differences(left[field], right[field], field))
    report = dict(policy=POLICY, passed=not failures, failed_paths=failures, leaves=leaves)
    return report
