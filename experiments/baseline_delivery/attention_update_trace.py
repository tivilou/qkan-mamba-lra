"""Private complete-value, paired train traces for the BF16 update diagnostic.

Lazy native imports; these artifacts are never automatically uploaded. This is
an original-latest-checkpoint diagnostic, not a best-validation case study.
"""
from pathlib import Path

from .common import atomic_json,digest,fingerprint,read
from .attention_update import attention_route,POLICY


def produce(runtime,enabled):
    t,np=runtime.t,runtime.np;runtime.restore(runtime.base)
    ids=np.arange(POLICY['trace_rows']);batch=runtime.split.batch(ids,'cuda:0')
    model=runtime.model;values={};source_dtypes={};hooks=[]

    def store(name,value):
        source_dtypes[name]=str(value.dtype)
        values[name]=value.detach().float().cpu().numpy().copy()

    def observe(layer,call,x,projected,actual,attention,mask):
        if layer!=0: return
        if call not in (0,1): raise ValueError('Unexpected trace document call')
        for key,value,cast in zip(('q','k','v'),projected,actual):
            name=key+str(call);store('projected_'+name,value);store('sdpa_'+name,cast)
            def gradient(grad,name=name): store('gradient_'+name,grad)
            hooks.append(value.register_hook(gradient))
        store('attention'+str(call),attention)

    previous_capture=getattr(model,'capture',False);previous_captured=getattr(model,'captured',[])
    try:
        model.capture=True;model.captured=[];model.train();runtime.opt.zero_grad(set_to_none=True)
        with runtime.dropout(True),attention_route(t,model,enabled,fp32_adapter=not enabled,observe=observe):
            logits,pooled=model(batch,representations=True,validated_inputs=True)
            losses=t.nn.functional.cross_entropy(logits,batch['labels'],reduction='none')
            losses.mean().backward()
        for side,observed in enumerate(model.captured):
            values['tokens'+str(side)]=observed['input'].numpy().copy()
            values['mask'+str(side)]=observed['mask'].numpy().copy()
            values['positions'+str(side)]=np.broadcast_to(np.arange(4000,dtype=np.int64),(len(ids),4000)).copy()
            values['encoder'+str(side)]=observed['encoder'].numpy().copy()
        store('pooled',pooled);store('logits',logits);store('loss',losses.unsqueeze(-1))
        grad=model.encoder.layers[0].self_attn.in_proj_weight.grad
        if grad is None or not bool(t.isfinite(grad).all()): raise ValueError('Trace real-loss parameter gradient missing')
        # Shared parameter gradient is the mean CE gradient over BOTH selected
        # rows, explicitly not a separate per-sample parameter gradient.
        values['batch_projection_gradient']=np.broadcast_to(grad.detach().cpu().numpy(),(len(ids),*grad.shape)).copy()
        expected={prefix+key+str(side) for prefix in ('projected_','sdpa_','gradient_')
            for key in ('q','k','v') for side in (0,1)}
        if not expected<=set(values) or len(model.captured)!=2:
            raise ValueError('Incomplete native train trace')
        return values,batch['labels'].cpu().numpy().copy(),ids,source_dtypes
    finally:
        for hook in hooks: hook.remove()
        model.capture=previous_capture;model.captured=previous_captured
        runtime.restore(runtime.base)


def emit(runtime,folder,route,values,labels,ids,source_dtypes):
    from .trace import display,coordinate
    from tools.sample_trace_contract.validate import validate
    np=runtime.np;folder=Path(folder);arrays=folder/'values';arrays.mkdir(parents=True,exist_ok=False)
    names=sorted(values);samples=[];total=0
    for i,row in enumerate(ids):
        reps=[]
        for name in names:
            value=np.ascontiguousarray(values[name][i]);path=arrays/(str(i)+'-'+name+'.npy')
            if value.size<1 or not bool(np.isfinite(value).all()): raise ValueError('Invalid complete trace values')
            if total+value.nbytes+1024>128*1024**2: raise ValueError('Private trace would exceed128MiB per route')
            np.save(path,value,allow_pickle=False);total+=path.stat().st_size
            if total>128*1024**2:
                path.unlink();raise ValueError('Private trace exceeds128MiB per route')
            artifact=dict(path=path.relative_to(folder).as_posix(),sha256=digest(path),bytes=path.stat().st_size,visibility='private')
            stage='preprocess' if name.startswith(('tokens','mask','positions')) else 'training' if name.startswith(('loss','gradient','batch_projection')) else 'scoring'
            meaning=name+' from actual '+route+' train forward/backward'
            if name=='batch_projection_gradient': meaning+='; shared mean-CE parameter gradient over fixed rows0 and1, repeated per row for lineage'
            if name.startswith('sdpa_'): meaning+='; original '+source_dtypes[name]+' values exactly stored as float32'
            reps.append(dict(id=name,stage_id=stage,status='observed',meaning=meaning,
                shape=list(value.shape),dtype=str(value.dtype),axis_semantics=['native axis '+str(j) for j in range(value.ndim)],
                artifact_refs=[artifact],value_capture=dict(status='complete',shape=list(value.shape),
                    logical_element_count=int(value.size),stored_element_count=int(value.size),artifact_refs=[artifact],
                    reader_projection=dict(status='available',kind='coordinate_inspector',coordinate_access='arbitrary',
                        reader_ref='python -m experiments.baseline_delivery.trace --trace sample-trace.json --sample '+str(i)+' --representation '+name,
                        source_sha256=[artifact['sha256']])),display=display(name,name)))
        stages=[]
        for stage,purpose,inputs,outputs in (
            ('data','Original train source row, selected before outputs; latest checkpoint, not best-validation',
             ['source:'+runtime.split.identity+':row:'+str(row)],['dataset_row']),
            ('preprocess','Original full4000 token coordinates and PAD masks, no trimming',
             ['dataset_row'],[n for n in names if n.startswith(('tokens','mask','positions'))]),
            ('scoring','FP32 original projection -> forced efficient SDPA -> encoder/pool/head; no target in forward',
             ['tokens0','tokens1','mask0','mask1'],[n for n in names if not n.startswith(('tokens','mask','positions','loss','gradient','batch_projection'))]),
            ('training','Train labels consumed only by mean CE after logits; gradients of actual loss, no optimizer commit',
             ['logits','source:train-label:'+str(row)],['loss','batch_projection_gradient']+[n for n in names if n.startswith('gradient_')])):
            stages.append(dict(stage_id=stage,stage=stage,canonical_stage=stage,lane='training',status='observed',
                purpose=purpose,input_refs=inputs,output_refs=outputs,target_access='training_only' if stage=='training' else 'outside_generation_input',
                display=display(purpose,outputs[0])))
        samples.append(dict(sample_id='train:'+str(row),lane='training',split='train',checkpoint='original_latest',
            roles=dict(target=int(labels[i]),prediction=int(values['logits'][i].argmax())),representations=reps,stages=stages))
    trace=dict(schema_version='sample-trace.v2',trace_id=folder.name,trace_completeness='complete',
        experiment=dict(run_id=folder.parent.parent.name,dataset='retrieval',code_revision=runtime.request['source_revision'],
            config_sha256=runtime.request['source_config_sha256'],model_identity='transformer/'+route,
            checkpoint_sha256=runtime.request['checkpoint']['sha256'],privacy='private; no automatic trace/tensor upload'),
        sample_selection=dict(rule='Fixed first2 train row IDs before outputs; both routes identical; diagnostic only',
            population_scope='original train split',selected_count=len(samples),selected_sample_ids=[s['sample_id'] for s in samples]),
        coverage={key:('observed' if key in ('data','preprocess','scoring','training') else 'not_applicable') for key in
            ('data','preprocess','training','retrieval','scoring','selection','context','generation','evaluation','diagnosis')},
        task_contract=dict(task_type='long-sequence classification training diagnostic',required_roles=['target','prediction'],
            required_representations=names,required_value_representations=names,required_splits=['train'],
            required_checkpoints=['original_latest'],required_lanes=['training'],
            checkpoint_policy='Original latest full checkpoint; no checkpoint/model selection; not best-validation'),samples=samples)
    errors,warnings=validate(trace,require_complete=True)
    if errors or warnings: raise ValueError('Native trace structural failure: '+'; '.join(errors+warnings))
    path=folder/'sample-trace.json';atomic_json(path,trace)
    # Reopen every complete value artifact and test arbitrary original coordinates.
    decoded=read(path)
    for i,sample in enumerate(decoded['samples']):
        for rep in sample['representations']:
            artifact=rep['artifact_refs'][0];target=(folder/artifact['path']).resolve()
            if (not target.is_relative_to(folder.resolve()) or digest(target)!=artifact['sha256']
                    or target.stat().st_size!=artifact['bytes']): raise ValueError('Private trace integrity differs')
            actual=np.load(target,allow_pickle=False);expected=values[rep['id']][i]
            if (list(actual.shape)!=rep['shape'] or str(actual.dtype)!=rep['dtype']
                    or actual.size!=rep['value_capture']['stored_element_count'] or not np.array_equal(actual,expected)):
                raise ValueError('Complete producer values differ from decoded trace')
            for coords in ([0]*actual.ndim,[n//2 for n in actual.shape],[n-1 for n in actual.shape]):
                if coordinate(path,i,rep['id'],coords)!=actual[tuple(coords)].item(): raise ValueError('Trace coordinate reader differs')
        for side in (0,1):
            if (not np.array_equal(values['tokens'+str(side)][i],runtime.split.arrays['inputs'+str(side+1)][row:=int(sample['sample_id'].split(':')[1])])
                    or not np.array_equal(values['mask'+str(side)][i],values['tokens'+str(side)][i]!=1)
                    or not np.array_equal(values['positions'+str(side)][i],np.arange(4000))):
                raise ValueError('Producer source-row/position/mask lineage differs')
    return dict(passed=True,schema='sample-trace.v2',sha256=digest(path),samples=2,
        value_artifacts=2*len(names),value_bytes=total,structural_errors=0,structural_warnings=0,
        semantic='Full producer arrays/hash/shape/dtype/source-row/mask/positions and first/middle/last coordinates checked',
        checkpoint_policy='original_latest; not best_validation',privacy='private values retained; aggregate readiness only returned')


def collect_and_validate(runtime,output):
    results={}
    for route in ('fp32','bf16'):
        values,labels,ids,dtypes=produce(runtime,route=='bf16')
        replay,labels_again,ids_again,dtypes_again=produce(runtime,route=='bf16')
        if (set(values)!=set(replay) or dtypes!=dtypes_again
                or not runtime.np.array_equal(labels,labels_again) or not runtime.np.array_equal(ids,ids_again)
                or any(not runtime.np.array_equal(values[name],replay[name]) for name in values)):
            raise ValueError('Same-route complete-value forward/real-loss/backward trace replay differs')
        results[route]=emit(runtime,output/('trace-'+route),route,values,labels,ids,dtypes)
        results[route]['full_model_replay_exact']=True
        del values,replay
    return dict(passed=True,routes=results,source_train_sha256=runtime.split.identity,
        role_mapping=dict(fp32='baseline',bf16='intervention'),selection='fixed train rows0,1',test_access=False)
