"""Private selected-sample numeric evidence and a bounded coordinate reader."""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .common import atomic_json, digest, fingerprint, read


def display(label,ref):
    return {'summary':label,'default_view':'table',
            'visible_fields':[{'id':ref,'label':label,'help':label+'；数值来自所选样本的实际前向计算。','value_ref':ref}],
            'visualizations':[]}


def capture(model,split,output,config,checkpoint):
    output=Path(output); folder=output/'trace-values';folder.mkdir(exist_ok=True)
    model.eval();model.capture=True;model.captured=[]
    ids=np.arange(min(2,len(split)))
    device=next(model.parameters()).device
    batch=split.batch(ids,device)
    with torch.no_grad():
        logits,pooled=model(batch,representations=True)
    values={'pooled':pooled.cpu().numpy(),'logits':logits.cpu().numpy()}
    for side,observed in enumerate(model.captured):
        for name,tensor in observed.items():
            values[name+str(side)]=tensor.numpy()
    model.capture=False;model.captured=[]
    reps=list(values)
    contract={'task_type':'long-sequence classification','required_roles':['target','prediction'],
              'required_representations':reps,'required_value_representations':reps,
              'required_splits':['val'],'required_checkpoints':['best_validation'],
              'required_lanes':['evaluation'],'checkpoint_policy':'best validation accuracy; first tie retained'}
    samples=[]
    for i,row in enumerate(ids):
        representations=[]
        for name,array in values.items():
            value=np.ascontiguousarray(array[i]);path=folder/(str(i)+'-'+name+'.npy')
            np.save(path,value,allow_pickle=False)
            artifact={'path':path.relative_to(output).as_posix(),'sha256':digest(path),
                      'bytes':path.stat().st_size,'visibility':'private'}
            representations.append({'id':name,'stage_id':('preprocess' if name.startswith(('input','mask')) else 'forward'),'status':'observed',
                'meaning':name+' in actual selected-sample forward','shape':list(value.shape),'dtype':str(value.dtype),
                'axis_semantics':['axis '+str(a) for a in range(value.ndim)],'artifact_refs':[artifact],
                'value_capture':{'status':'complete','shape':list(value.shape),
                  'logical_element_count':value.size,'stored_element_count':value.size,'artifact_refs':[artifact],
                  'reader_projection':{'status':'available','kind':'coordinate_inspector','coordinate_access':'arbitrary',
                    'reader_ref':'python -m experiments.baseline_delivery.trace --trace sample-trace.json --sample '+str(i)+' --representation '+name,
                    'source_sha256':[artifact['sha256']]}},'display':display(name,name)})
        samples.append({'sample_id':'val:'+str(row),'lane':'evaluation','split':'val','checkpoint':'best_validation',
            'roles':{'target':int(batch['labels'][i]),'prediction':int(logits[i].argmax())},'representations':representations,
            'stages':[{'stage_id':'data','stage':'data','canonical_stage':'data','lane':'evaluation','status':'observed',
               'purpose':'Select a validation row by frozen split identity and row index',
               'input_refs':['source:'+split.identity+':row:'+str(row)],'output_refs':['dataset_row'],
               'display':display('验证集样本来源','dataset_row')},
              {'stage_id':'preprocess','stage':'preprocess','canonical_stage':'preprocess','lane':'evaluation','status':'observed',
               'purpose':'Integer tokens and padding mask, or fixed pixel conversion',
               'input_refs':['dataset_row'],'output_refs':[r for r in reps if r.startswith(('input','mask'))],
               'display':display('模型输入与掩码',next(r for r in reps if r.startswith('input')))},
              {'stage_id':'forward','stage':'scoring','canonical_stage':'scoring','lane':'evaluation','status':'observed',
               'checkpoint':'best_validation','purpose':'Encode raw permitted input, pool, then classify; target is evaluation-only',
               'input_refs':[r for r in reps if r.startswith(('input','mask'))],
               'output_refs':[r for r in reps if not r.startswith(('input','mask'))],
               'outputs':{'prediction':int(logits[i].argmax())},'display':display('分类前向与输出','logits'),
               'target_access':'post_generation_only'},
              {'stage_id':'evaluation','stage':'evaluation','canonical_stage':'evaluation','lane':'evaluation','status':'observed',
               'purpose':'Compare prediction with held-out target after scoring',
               'input_refs':['logits','source:validation-label:'+str(row)],'output_refs':['correct'],
               'outputs':{'correct':bool(logits[i].argmax()==batch['labels'][i])},
               'display':display('验证标签仅用于评估','correct')}]})
    trace={'schema_version':'sample-trace.v2','trace_id':output.name+'-best-val','trace_completeness':'complete',
       'experiment':{'run_id':output.name,'dataset':config['task'],'code_revision':config['source_revision'],
          'config_sha256':fingerprint(config),'model_identity':config['model'],'checkpoint_sha256':checkpoint,
          'privacy':'All trace payloads private; input arrays/text are excluded from automatic exchange.'},
       'sample_selection':{'rule':'first two validation row IDs, fixed before fitting','population_scope':'frozen validation split',
          'selected_count':len(samples),'selected_sample_ids':[s['sample_id'] for s in samples]},
       'coverage':{key:('observed' if key in ('data','preprocess','scoring','evaluation') else 'not_applicable')
          for key in ('data','preprocess','training','retrieval','scoring','selection','context','generation','evaluation','diagnosis')},
       'task_contract':contract,'samples':samples}
    atomic_json(output/'sample-trace.json',trace)
    from tools.sample_trace_contract.validate import validate
    errors,warnings=validate(trace,require_complete=True)
    if errors or warnings:
        raise ValueError('Trace structural contract failed: '+'; '.join(errors+warnings))
    semantic_validate(output/'sample-trace.json')
    # Decode complete captured encoder values and independently reconstruct the
    # classifier input/head. This compares replay with the producer's logits.
    for sample in samples:
        lookup={rep['id']:np.load(output/rep['artifact_refs'][0]['path'],allow_pickle=False) for rep in sample['representations']}
        encoded=[]
        for side in range(2 if config['task']=='retrieval' else 1):
            h,mask=lookup['encoder'+str(side)],lookup['mask'+str(side)]
            encoded.append((h*mask[:,None]).sum(0)/mask.sum())
        expected=np.concatenate((np.abs(encoded[0]-encoded[1]),encoded[0]*encoded[1])) if len(encoded)==2 else encoded[0]
        np.testing.assert_allclose(expected,lookup['pooled'],rtol=1e-5,atol=1e-6)
        with torch.no_grad():
            replay=model.head(torch.tensor(expected,device=device)).cpu().numpy()
        np.testing.assert_allclose(replay,lookup['logits'],rtol=1e-5,atol=1e-6)
    atomic_json(output/'trace-validation.json',{'schema':'qkan-trace-readiness.v1','passed':True,
        'semantic':'decoded full arrays and coordinate reader','replay':'pooled encoder -> pair head -> logits',
        'structural':'shared sample-trace.v2 validator require_complete; zero warnings',
        'checkpoint_sha256':checkpoint,'samples':len(samples),'privacy':'private; not automatically uploaded'})


def coordinate(trace_path,sample,representation,indices):
    path=Path(trace_path);trace=read(path)
    record=next(r for r in trace['samples'][sample]['representations'] if r['id']==representation)
    artifact=record['artifact_refs'][0]
    target=(path.parent/artifact['path']).resolve()
    if not target.is_relative_to(path.parent.resolve()) or digest(target)!=artifact['sha256']:
        raise ValueError('Unsafe or changed trace artifact')
    array=np.load(target,mmap_mode='r',allow_pickle=False)
    if len(indices)!=array.ndim or any(i<0 or i>=n for i,n in zip(indices,array.shape)):
        raise ValueError('Coordinate outside representation')
    return array[tuple(indices)].item()


def semantic_validate(path):
    trace=read(path);root=Path(path).parent
    required=set(trace['task_contract']['required_value_representations'])
    for index,sample in enumerate(trace['samples']):
        records={r['id']:r for r in sample['representations']}
        if set(records)!=required: raise ValueError('Missing trace representation')
        for name,record in records.items():
            artifact=record['artifact_refs'][0];value=np.load(root/artifact['path'],allow_pickle=False)
            if list(value.shape)!=record['shape'] or value.size!=record['value_capture']['stored_element_count']:
                raise ValueError('Trace shape/value coverage differs')
            for coords in ([0]*value.ndim,[n-1 for n in value.shape]):
                if coordinate(path,index,name,coords)!=value[tuple(coords)].item():
                    raise ValueError('Coordinate reader differs from actual values')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True);parser.add_argument('--sample',type=int,default=0)
    parser.add_argument('--representation',required=True);parser.add_argument('--indices',default='')
    args=parser.parse_args()
    print(coordinate(args.trace,args.sample,args.representation,[int(v) for v in args.indices.split(',') if v]))


if __name__=='__main__': main()
