"""Measure repeat-run noise versus resume differences; never certify recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from .common import ROOT, atomic_json, child_spec, digest, read, revision, run_child

FIELDS = ('contract','runtime','model','optimizer','scheduler','rng','permutation',
          'cursor','step','epoch','best_weights','best_accuracy','best_epoch',
          'history','update_trace','epoch_loss','epoch_correct','epoch_samples',
          'training_complete','amp')
SCIENTIFIC_FILES = ('worker.py','models.py','data.py','trace.py','common.py')


def differences(left, right, path='state'):
    """Full leaf coverage; no tolerances and no implicit numerical pass flag."""
    import numpy as np
    import torch
    rows=[]
    def visit(a,b,p):
        if type(a) is not type(b):
            rows.append(dict(path=p,exact=False,kind='type_mismatch')); return
        if isinstance(a,(torch.Tensor,np.ndarray)):
            x=a.detach().cpu().numpy() if isinstance(a,torch.Tensor) else a
            y=b.detach().cpu().numpy() if isinstance(b,torch.Tensor) else b
            if x.shape!=y.shape or x.dtype!=y.dtype:
                rows.append(dict(path=p,exact=False,kind='shape_or_dtype_mismatch'));return
            floating=x.dtype.kind in 'fc'
            finite=bool(np.isfinite(x).all() and np.isfinite(y).all()) if floating else True
            exact=bool(np.array_equal(x,y)) and finite
            row=dict(path=p,kind='floating_array' if floating else 'discrete_array',
                     exact=exact,shape=list(x.shape),dtype=str(x.dtype),finite=finite,
                     differing_elements=int(np.count_nonzero(x!=y)))
            if floating and finite:
                d=np.abs(x.astype(np.complex128 if x.dtype.kind=='c' else np.float64)-y)
                row.update(max_abs=float(d.max()) if d.size else 0.,
                           rms=float(np.sqrt(np.mean(d*d))) if d.size else 0.,
                           reference_scale=float(max(np.abs(x).max(),np.abs(y).max())) if d.size else 0.)
            rows.append(row)
        elif isinstance(a,dict):
            if a.keys()!=b.keys(): rows.append(dict(path=p,exact=False,kind='key_mismatch'));return
            for k in a: visit(a[k],b[k],p+'/'+str(k))
        elif isinstance(a,(tuple,list)):
            if len(a)!=len(b): rows.append(dict(path=p,exact=False,kind='length_mismatch'));return
            for i,(x,y) in enumerate(zip(a,b)): visit(x,y,p+'/'+str(i))
        else:
            floating=isinstance(a,(float,np.floating))
            finite=bool(np.isfinite(a) and np.isfinite(b)) if floating else True
            row=dict(path=p,kind='floating_scalar' if floating else 'discrete_scalar',exact=bool(a==b) and finite,finite=finite)
            if floating and finite: row['max_abs']=float(abs(a-b))
            rows.append(row)
    visit(left,right,path)
    return rows


def compare(left,right):
    leaves=[]
    for key in FIELDS:
        leaves.extend(differences(left[key],right[key],key))
    return dict(exact=all(r['exact'] for r in leaves),
                all_finite=all(r.get('finite',True) for r in leaves),
                control_state_exact=all(r['exact'] for r in leaves if r['path'].split('/')[0]
                    in ('contract','runtime','rng','permutation','cursor','step','epoch','amp','training_complete')),
                differing_leaves=sum(not r['exact'] for r in leaves),
                max_abs=max((r.get('max_abs',0.) for r in leaves),default=0.),leaves=leaves)


def check_source(config):
    hashes={}
    for name in (*SCIENTIFIC_FILES,'protocol.json'):
        relative='experiments/baseline_delivery/'+name
        pinned=subprocess.check_output(['git','show',config['source_revision']+':'+relative],cwd=ROOT)
        current=(ROOT/relative).read_bytes()
        if pinned!=current:
            raise ValueError('Original scientific source changed: '+relative+'; preserve suite and investigate before rerunning')
        hashes[relative]=hashlib.sha256(current).hexdigest()
    return hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canary',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    folder=args.canary.resolve(); config=read(folder/'config.json')
    if config['epochs']!=2 or not (folder/'data/synthetic-canary.json').is_file():
        raise ValueError('Requires an existing bounded canary, not a formal training run')
    scientific_hashes=check_source(config)
    for name in ('uninterrupted','interrupted'):
        if not read(folder/name/'trace-validation.json')['passed']:
            raise ValueError('Existing run did not pass trace validation: '+name)
    audit=folder/('recovery-audit-'+uuid.uuid4().hex[:12]);audit.mkdir()
    output=audit/'repeat-uninterrupted'
    print('Audit: '+str(audit),flush=True)
    spec=child_spec('--config',folder/'config.json','--data',folder/'data',
                    '--cache',args.cache.resolve(),'--output',output,'--device',args.device,'--synthetic')
    env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONUNBUFFERED':'1'}
    if run_child(spec,audit/'repeat-uninterrupted.log',env)!=0:
        raise ValueError('Independent repeat failed; inspect '+str(audit/'repeat-uninterrupted.log'))
    # Load CPU checkpoint tensors only after the training child exits.
    from .worker import load_checkpoint
    states=[];checkpoints={}
    for name,path in (('original',folder/'uninterrupted'),('resumed',folder/'interrupted'),('repeat',output)):
        pointer=read(path/'checkpoints/latest.json')
        states.append(load_checkpoint(path,pointer['contract']))
        checkpoints[name]=pointer['sha256']
        if not read(path/'trace-validation.json')['passed']: raise ValueError('Trace validation failed: '+name)
    if len({s['contract'] for s in states})!=1:
        raise ValueError('Repeat and existing checkpoints have different scientific contracts')
    environments={name:read(path/'environment.json') for name,path in
                  (('original',folder/'uninterrupted'),('resumed',folder/'interrupted'),('repeat',output))}
    if environments['original']!=environments['repeat'] or environments['original']!=environments['resumed']:
        raise ValueError('Environment records differ across runs; do not attribute differences to kernel noise')
    comparisons={'repeat_vs_original':compare(states[0],states[2]),
                 'resumed_vs_original':compare(states[0],states[1]),
                 'resumed_vs_repeat':compare(states[2],states[1])}
    report=dict(schema='qkan-recovery-noise-audit.v1',status='diagnostic_only',formal=False,
                recovery_gate_changed=False,numerical_acceptance_criterion=None,
                audit_revision=revision(),producer_revision=config['source_revision'],
                scientific_source_sha256=scientific_hashes,config_sha256=digest(folder/'config.json'),
                model=config['model'],task=config['task'],device=args.device,
                checkpoint_sha256=checkpoints,comparisons=comparisons,
                environment_records_equal=True,
                limitations='One independent repeat diagnoses baseline nondeterminism; it does not establish a statistical noise bound or certify numerical recovery.')
    atomic_json(audit/'report.json',report)
    summary={key:{k:v for k,v in value.items() if k!='leaves'} for key,value in comparisons.items()}
    atomic_json(audit/'summary.json',dict(status='diagnostic_only',comparisons=summary))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('Report: '+str(audit/'report.json'),flush=True)
    print('Recovery gate unchanged; this diagnostic does not authorize formal training.',flush=True)
    from .return_artifacts import upload_declared
    try:
        upload_declared(audit,'recovery-audit',['report.json','summary.json'])
    except Exception as error:
        raise SystemExit('Report upload failed ('+type(error).__name__+
                         '); report retained at '+str(audit)+
                         '. Retry: python -m experiments.baseline_delivery.return_artifacts --directory '+str(audit))


if __name__=='__main__': main()
