"""Exact S5 continuation comparison with bounded, array-free diagnostics."""
import math

FIELDS=('train_state','rng','sampler','permutation','cursor','step','epoch',
        'best_weights','best_accuracy','best_epoch','history','update_trace',
        'epoch_loss','epoch_correct','epoch_samples','training_complete','runtime','contract')


def validate(left,right):
    import numpy as np
    mismatches=[];count=0;leaves=0;by_field={}

    def fail(path,reason,**details):
        nonlocal count
        count+=1
        field=path.split('/')[0]
        summary=by_field.setdefault(field,dict(mismatches=0,max_absolute_difference=None))
        summary['mismatches']+=1
        delta=details.get('max_absolute_difference')
        if delta is not None:
            previous=summary['max_absolute_difference']
            summary['max_absolute_difference']=delta if previous is None else max(previous,delta)
        if len(mismatches)<32: mismatches.append(dict(field=path,reason=reason,**details))

    def visit(a,b,path):
        nonlocal leaves
        if isinstance(a,dict):
            if not isinstance(b,dict) or a.keys()!=b.keys(): fail(path,'keys');return
            for key in a: visit(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,(tuple,list)):
            if type(a)!=type(b) or len(a)!=len(b): fail(path,'sequence structure');return
            for i,(x,y) in enumerate(zip(a,b)): visit(x,y,path+'/'+str(i))
        elif isinstance(a,np.ndarray):
            leaves+=1
            if not isinstance(b,np.ndarray) or a.shape!=b.shape or a.dtype!=b.dtype:
                fail(path,'array shape/dtype');return
            if a.dtype.kind in 'fc' and not (np.isfinite(a).all() and np.isfinite(b).all()):
                fail(path,'nonfinite array');return
            if not np.array_equal(a,b):
                details=dict(differing_elements=int(np.count_nonzero(a!=b)),elements=int(a.size),dtype=str(a.dtype))
                if a.dtype.kind in 'fc':
                    dtype=np.complex128 if a.dtype.kind=='c' else np.float64
                    delta=float(np.max(np.abs(a.astype(dtype)-b.astype(dtype))))
                    details['max_absolute_difference']=delta if math.isfinite(delta) else None
                fail(path,'array values',**details)
        else:
            leaves+=1
            if type(a)!=type(b) or a!=b: fail(path,'scalar type/value')

    for key in FIELDS:
        if key not in left or key not in right: fail(key,'missing field')
        else: visit(left[key],right[key],key)
    return dict(schema='qkan-s5-exact-recovery.v1',passed=count==0,agreement='exact',
        checked_fields=list(FIELDS),checked_leaves=leaves,mismatch_count=count,mismatches=mismatches,
        field_summary={key:by_field.get(key,dict(mismatches=0,max_absolute_difference=None)) for key in FIELDS},
        privacy='State field paths and aggregate differences only; no values, arrays, sample IDs, labels or inputs',
        tolerance='No numerical tolerance; dtype, shape and values must match')
