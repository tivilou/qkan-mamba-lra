"""Bounded, private producer-value traces for the analytic-K recovery preflight."""
import hashlib
import math
from pathlib import Path

from .common import atomic_json,digest,fingerprint,read
from .attention_update import POLICY,attention_route
from .key_bias_consistency import NAMES,analytic_key_bias,manifest


def adam_equations(before,after,gradient,beta1,beta2):
    """Independent scalar check over real saved packed moment values."""
    if not (len(before['m'])==len(before['v'])==len(after['m'])==len(after['v'])==len(gradient)==384):
        raise ValueError('Packed Adam moment coordinate coverage differs')
    if after['step']!=before['step']+1: raise ValueError('Adam step trace differs')
    for m,v,g,ma,va in zip(before['m'],before['v'],gradient,after['m'],after['v']):
        if not all(math.isfinite(x) for x in (m,v,g,ma,va)) or min(v,va)<0:
            raise ValueError('Nonfinite or negative Adam trace values')
        expected_m=beta1*m+(1-beta1)*g;expected_v=beta2*v+(1-beta2)*g*g
        if (not math.isclose(ma,expected_m,rel_tol=2e-5,abs_tol=1e-12)
                or not math.isclose(va,expected_v,rel_tol=2e-5,abs_tol=1e-18)):
            raise ValueError('Actual clipped gradient does not reconstruct Adam moments')
    return True


def validate_bias_lineage(np,values,groups,candidate):
    for layer in range(4):
        prefix='adam_'+str(layer)+'_';raw=values['gradient_batch_bias_raw_'+str(layer)][0]
        corrected=values['gradient_batch_bias_corrected_'+str(layer)][0]
        clipped=values['gradient_batch_bias_clipped_'+str(layer)][0]
        if (not np.array_equal(raw[:128],corrected[:128]) or not np.array_equal(raw[256:],corrected[256:])
                or candidate and bool((corrected[128:256]!=0).any())
                or not candidate and not np.array_equal(raw,corrected)):
            raise ValueError('Packed bias raw/corrected producer lineage differs')
        state={when:dict(m=values[prefix+'exp_avg_'+when][0].tolist(),v=values[prefix+'exp_avg_sq_'+when][0].tolist(),
            step=float(values[prefix+'step_'+when][0][0])) for when in ('before','after')}
        beta1,beta2=groups[str(layer)]['betas'];adam_equations(state['before'],state['after'],clipped.tolist(),beta1,beta2)
        if not np.array_equal(values['bias_'+str(layer)+'_delta'],values['bias_'+str(layer)+'_after']-values['bias_'+str(layer)+'_before']):
            raise ValueError('Actual bias delta does not match before/after parameters')
    return True


def selected_ids(runtime):
    start=runtime.base['progress']['cursor']
    ids=runtime.base['permutation'][start:start+POLICY['trace_rows']].numpy().copy()
    if len(ids)!=2 or len(set(map(int,ids)))!=2: raise ValueError('Exactly two distinct original-cursor rows required')
    return ids


def produce(runtime,route):
    t,np=runtime.t,runtime.np;runtime.restore(runtime.base);base_hash=runtime.state_hash(runtime.base)
    ids=selected_ids(runtime);batch=runtime.split.batch(ids,'cuda:0');model=runtime.model
    candidate=route=='bf16_analytic';adapter=route!='original';values={};dtypes={};hooks=[]
    parameters=dict(model.named_parameters());raw_callbacks={n:[] for n in NAMES};corrected_callbacks={n:[] for n in NAMES}
    groups={}
    for layer,name in enumerate(NAMES):
        group=[g for g in runtime.opt.param_groups if any(parameters[name] is p for p in g['params'])]
        if len(group)!=1 or group[0].get('amsgrad',False) or group[0].get('maximize',False):
            raise ValueError('Original Adam group identity differs')
        g=group[0];groups[str(layer)]=dict(lr=float(g['lr']),betas=list(g['betas']),eps=float(g['eps']),weight_decay=float(g['weight_decay']))
    old_capture=getattr(model,'capture',False);old_captured=getattr(model,'captured',[])

    def store(name,value):
        if name in values: raise ValueError('Repeated producer trace value '+name)
        dtypes[name]=str(value.dtype);values[name]=value.detach().float().cpu().numpy().copy()

    def shared(name,value):
        # Parameter-level values belong to the fixed two-row mean CE batch.
        # Repeat complete values per row for lineage; never call them per-row gradients.
        store(name,value)
        values[name]=np.broadcast_to(values[name],(len(ids),*values[name].shape)).copy()

    def callback(name,raw,corrected):
        raw_callbacks[name].append(runtime.clone(raw));corrected_callbacks[name].append(runtime.clone(corrected))

    def observe(layer,call,x,projected,actual,attention,mask):
        if layer!=0: return
        if call not in (0,1): raise ValueError('Trace first-layer document coverage differs')
        for key,value,cast in zip(('q','k','v'),projected,actual):
            name=key+str(call);store('projected_'+name,value);store('sdpa_'+name,cast)
            def gradient(grad,name=name): store('gradient_'+name,grad)
            hooks.append(value.register_hook(gradient))
        store('attention'+str(call),attention)

    def moment(when):
        for layer,name in enumerate(NAMES):
            p=parameters[name];state=runtime.opt.state[p]
            for key in ('exp_avg','exp_avg_sq','step'): shared('adam_'+str(layer)+'_'+key+'_'+when,state[key].reshape(-1))
            shared('bias_'+str(layer)+'_'+when,p)

    try:
        model.capture=True;model.captured=[];model.train();runtime.opt.zero_grad(set_to_none=True)
        moment('before')
        if not candidate:
            for name in NAMES:
                def unchanged(gradient,name=name): callback(name,gradient,gradient);return gradient
                if getattr(parameters[name],'_backward_hooks',None): raise ValueError('Existing bias hook conflicts with trace')
                hooks.append(parameters[name].register_hook(unchanged))
        with runtime.dropout(True),analytic_key_bias(t,model,candidate,observe=callback) as evidence:
            with attention_route(t,model,candidate,fp32_adapter=adapter and not candidate,observe=observe) as attention:
                with t.nn.attention.sdpa_kernel(t.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                    logits,pooled=model(batch,representations=True,validated_inputs=True)
                    losses=t.nn.functional.cross_entropy(logits,batch['labels'],reduction='none')
                    losses.mean().backward()
        if candidate and (evidence['hook_calls']!={n:1 for n in NAMES} or not evidence['handles_removed']
                or not evidence['qv_exact'] or not evidence['key_zero']): raise ValueError('Trace analytic callback coverage differs')
        if adapter:
            dtype='torch.bfloat16' if candidate else 'torch.float32'
            if (attention['layer_calls']!=[2]*4 or set(attention['qkv_dtypes'])!={dtype}
                    or set(attention['output_dtypes'])!={dtype}): raise ValueError('Trace attention dtype/call coverage differs')
        if len(model.captured)!=2: raise ValueError('Both document encoders required')
        for side,item in enumerate(model.captured):
            values['tokens'+str(side)]=item['input'].numpy().copy()
            values['mask'+str(side)]=item['mask'].numpy().copy()
            values['positions'+str(side)]=np.broadcast_to(np.arange(4000,dtype=np.int64),(len(ids),4000)).copy()
            values['encoder'+str(side)]=item['encoder'].numpy().copy()
        store('pooled',pooled);store('logits',logits);store('loss',losses.unsqueeze(-1))
        grad=model.encoder.layers[0].self_attn.in_proj_weight.grad
        if grad is None or not bool(t.isfinite(grad).all()): raise ValueError('Actual mean-CE projection gradient missing')
        shared('gradient_batch_projection',grad)
        gradients={n:runtime.clone(p.grad) for n,p in parameters.items() if p.grad is not None}
        if len(gradients)!=53 or sum(p.numel() for p in gradients.values())!=807170:
            raise ValueError('Trace complete parameter gradient coverage differs')
        for layer,name in enumerate(NAMES):
            if len(raw_callbacks[name])!=1 or len(corrected_callbacks[name])!=1:
                raise ValueError('Exactly one joint backward callback per bias required')
            raw,corrected=raw_callbacks[name][0],corrected_callbacks[name][0]
            if (not t.equal(raw[:128],corrected[:128]) or not t.equal(raw[256:],corrected[256:])
                    or not t.equal(gradients[name],corrected)
                    or candidate and bool((corrected[128:256]!=0).any())):
                raise ValueError('Trace raw/corrected gradient lineage differs')
            shared('gradient_batch_bias_raw_'+str(layer),raw)
            shared('gradient_batch_bias_corrected_'+str(layer),corrected)
        t.nn.utils.clip_grad_norm_(model.parameters(),runtime.config['clip_norm'],error_if_nonfinite=True)
        for layer,name in enumerate(NAMES): shared('gradient_batch_bias_clipped_'+str(layer),parameters[name].grad)
        runtime.opt.step();runtime.schedule.step(len(ids));moment('after')
        if not bool(t.stack([t.isfinite(p).all() for p in parameters.values()]).all()): raise ValueError('Nonfinite trace update')
        comparison=dict(logits=runtime.clone(logits),loss=float(losses.mean()),gradients=gradients,
            deltas={n:runtime.clone(p)-runtime.base['model'][n] for n,p in parameters.items()},
            row_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
        for layer,name in enumerate(NAMES): shared('bias_'+str(layer)+'_delta',comparison['deltas'][name])
        for layer in range(4):
            before=values['adam_'+str(layer)+'_step_before'];after=values['adam_'+str(layer)+'_step_after']
            if not np.array_equal(after,before+1): raise ValueError('Trace Adam step progression differs')
        validate_bias_lineage(np,values,groups,candidate)
        complete=runtime.state_hash(runtime.snapshot())
        return dict(values=values,labels=batch['labels'].detach().cpu().numpy().copy(),ids=ids,dtypes=dtypes,
            comparison=comparison,post_update_state_sha256=complete,attention=attention,
            hook_observation=evidence,adam_groups=groups,shared_values_scope='Fixed selected2 rows mean CE, not independent per-row parameter gradients')
    finally:
        for hook in reversed(hooks): hook.remove()
        model.capture=old_capture;model.captured=old_captured;runtime.restore(runtime.base)
        if runtime.state_hash(runtime.snapshot())!=base_hash: raise ValueError('Trace exact full-state rollback differs')


def training_value(name):
    return name.startswith(('loss','gradient_','adam_','bias_'))


def emit(runtime,folder,route,produced):
    from .trace import display,coordinate
    from tools.sample_trace_contract.validate import validate
    np=runtime.np;values=produced['values'];ids=produced['ids'];labels=produced['labels'];dtypes=produced['dtypes']
    folder=Path(folder);arrays=folder/'values';arrays.mkdir(parents=True,exist_ok=False);names=sorted(values);total=0;samples=[]
    expected={prefix+key+str(side) for prefix in ('projected_','sdpa_','gradient_') for key in ('q','k','v') for side in (0,1)}
    expected.update(prefix+str(side) for prefix in ('tokens','mask','positions','encoder','attention') for side in (0,1))
    expected.update(('pooled','logits','loss','gradient_batch_projection'))
    for layer in range(4):
        expected.update('gradient_batch_bias_'+kind+'_'+str(layer) for kind in ('raw','corrected','clipped'))
        expected.update('bias_'+str(layer)+'_'+when for when in ('before','after','delta'))
        expected.update('adam_'+str(layer)+'_'+key+'_'+when for key in ('exp_avg','exp_avg_sq','step') for when in ('before','after'))
    if not expected<=set(names): raise ValueError('Complete first-layer projected/SDPA/gradient coordinates required')
    validate_bias_lineage(np,values,produced['adam_groups'],route=='bf16_analytic')
    for i,row in enumerate(ids):
        reps=[]
        for name in names:
            value=np.ascontiguousarray(values[name][i]);path=arrays/(str(i)+'-'+name+'.npy')
            if not value.size or not bool(np.isfinite(value).all()): raise ValueError('Invalid complete producer values')
            if total+value.nbytes+1024>128*1024**2: raise ValueError('Trace exceeds128MiB per route')
            np.save(path,value,allow_pickle=False);total+=path.stat().st_size
            if total>128*1024**2: path.unlink();raise ValueError('Trace byte cap exceeded')
            artifact=dict(path=path.relative_to(folder).as_posix(),sha256=digest(path),bytes=path.stat().st_size,visibility='private')
            stage='preprocess' if name.startswith(('tokens','mask','positions')) else 'training' if training_value(name) else 'scoring'
            meaning=name+' from actual '+route+' selected-row forward/mean-CE backward/update'
            if name.startswith(('gradient_batch_','adam_','bias_')): meaning+='; shared joint two-row batch parameter value repeated per row; not a per-sample gradient'
            if name in dtypes: meaning+='; source dtype '+dtypes[name]+', stored exactly as float32'
            reps.append(dict(id=name,stage_id=stage,status='observed',meaning=meaning,shape=list(value.shape),dtype=str(value.dtype),
                axis_semantics=['native axis '+str(j) for j in range(value.ndim)],artifact_refs=[artifact],
                value_capture=dict(status='complete',shape=list(value.shape),logical_element_count=int(value.size),
                    stored_element_count=int(value.size),artifact_refs=[artifact],reader_projection=dict(status='available',
                        kind='coordinate_inspector',coordinate_access='arbitrary',reader_ref='python -m experiments.baseline_delivery.trace --trace sample-trace.json --sample '+str(i)+' --representation '+name,
                        source_sha256=[artifact['sha256']])),display=display(name,name)))
        stages=[]
        for stage,purpose,inputs,outputs in (
            ('data','Original cursor rows selected before outputs; original latest checkpoint, not best-validation',
                ['source:'+runtime.split.identity+':row:'+str(row)],['dataset_row']),
            ('preprocess','Full4000 token coordinates and PAD masks, no trimming',['dataset_row'],
                [n for n in names if n.startswith(('tokens','mask','positions'))]),
            ('scoring','FP32 projection -> forced efficient SDPA -> encoder/pool/head; no target input',
                ['tokens0','tokens1','mask0','mask1'],[n for n in names if not training_value(n) and not n.startswith(('tokens','mask','positions'))]),
            ('training','Two-row mean CE -> raw/corrected leaf gradients -> clip -> original AdamW moments/step; rollback after capture',
                ['logits','source:train-label:'+str(row)],[n for n in names if training_value(n)])):
            stages.append(dict(stage_id=stage,stage=stage,canonical_stage=stage,lane='training',status='observed',purpose=purpose,
                input_refs=inputs,output_refs=outputs,target_access='training_only' if stage=='training' else 'outside_generation_input',
                display=display(purpose,outputs[0])))
        samples.append(dict(sample_id='train:'+str(int(row)),lane='training',split='train',checkpoint='original_latest',
            roles=dict(target=int(labels[i]),prediction=int(values['logits'][i].argmax())),representations=reps,stages=stages))
    trace=dict(schema_version='sample-trace.v2',trace_id=folder.parent.parent.name+'-'+folder.name,trace_completeness='complete',
        experiment=dict(run_id=folder.parent.parent.name,dataset='retrieval',code_revision=runtime.request['source_revision'],
            config_sha256=runtime.request['source_config_sha256'],model_identity='transformer/'+route,
            checkpoint_sha256=runtime.request['checkpoint']['sha256'],privacy='private;no automatic value/trace upload',
            source_train_sha256=runtime.split.identity,seed=runtime.request['source_config']['seed'],
            plugin_tree_sha256=fingerprint(runtime.request['plugin_files']),
            gradient_config_sha256=runtime.request['key_bias_config_sha256'],
            environment=dict(torch='2.7.1+cu126',numpy='2.2.6',deterministic=True,tf32=False),
            key_bias_manifest=manifest() if route=='bf16_analytic' else dict(id='baseline.fp32_attention',version='1'),
            gradient_scope=produced['shared_values_scope'],adam_groups=produced['adam_groups'],post_update_state_sha256=produced['post_update_state_sha256']),
        sample_selection=dict(rule='First2 row IDs of the reviewed original cursor128 batch, fixed before outputs',
            population_scope='original train split/permutation/cursor',selected_count=2,selected_sample_ids=[s['sample_id'] for s in samples]),
        coverage={k:('observed' if k in ('data','preprocess','scoring','training') else 'not_applicable') for k in
            ('data','preprocess','training','retrieval','scoring','selection','context','generation','evaluation','diagnosis')},
        task_contract=dict(task_type='long-sequence classification training preflight',required_roles=['target','prediction'],
            required_representations=names,required_value_representations=names,required_splits=['train'],
            required_checkpoints=['original_latest'],required_lanes=['training'],checkpoint_policy='original latest reviewed14268;no best-validation model selection'),samples=samples)
    errors,warnings=validate(trace,require_complete=True)
    if errors or warnings: raise ValueError('Trace structural validation failed: '+'; '.join(errors+warnings))
    path=folder/'sample-trace.json';atomic_json(path,trace);decoded=read(path)
    for i,sample in enumerate(decoded['samples']):
        row=int(ids[i])
        for rep in sample['representations']:
            artifact=rep['artifact_refs'][0];target=(folder/artifact['path']).resolve()
            if (not target.is_relative_to(folder.resolve()) or digest(target)!=artifact['sha256']
                    or target.stat().st_size!=artifact['bytes']): raise ValueError('Trace artifact integrity differs')
            actual=np.load(target,allow_pickle=False)
            if (list(actual.shape)!=rep['shape'] or str(actual.dtype)!=rep['dtype'] or not np.array_equal(actual,values[rep['id']][i])
                    or actual.size!=rep['value_capture']['stored_element_count']): raise ValueError('Decoded complete producer values differ')
            for coords in ([0]*actual.ndim,[n//2 for n in actual.shape],[n-1 for n in actual.shape]):
                if coordinate(path,i,rep['id'],coords)!=actual[tuple(coords)].item(): raise ValueError('Real coordinate reader differs')
        for side in (0,1):
            if (not np.array_equal(values['tokens'+str(side)][i],runtime.split.arrays['inputs'+str(side+1)][row])
                    or not np.array_equal(values['mask'+str(side)][i],values['tokens'+str(side)][i]!=1)
                    or not np.array_equal(values['positions'+str(side)][i],np.arange(4000))):
                raise ValueError('Original producer row/mask/position lineage differs')
        if int(labels[i])!=int(runtime.split.arrays['labels'][row]): raise ValueError('Original train target lineage differs')
    return dict(passed=True,schema='sample-trace.v2',path=str(path),sha256=digest(path),samples=2,
        sample_ids=[s['sample_id'] for s in samples],value_artifacts=2*len(names),value_bytes=total,
        structural_errors=0,structural_warnings=0,complete_values_reopened_exact=True,
        semantic='Complete producer arrays/hash/shape/dtype/source rows/labels/PAD/positions and actual first/middle/last reader coordinates',
        gradient_scope=produced['shared_values_scope'],checkpoint_policy='original_latest;not best_validation',
        source_dtypes=dtypes,attention=produced['attention'],adam_moment_equations_passed=True,
        adam_equation_tolerance=dict(relative=2e-5,first_moment_absolute=1e-12,second_moment_absolute=1e-18),
        actual_bias_deltas_passed=True,privacy='Private complete values;compact readiness only returned')


def replay_equal(runtime,first,second):
    np=runtime.np
    if (first['dtypes']!=second['dtypes'] or first['post_update_state_sha256']!=second['post_update_state_sha256']
            or not np.array_equal(first['ids'],second['ids']) or not np.array_equal(first['labels'],second['labels'])
            or set(first['values'])!=set(second['values'])
            or any(not np.array_equal(first['values'][n],second['values'][n]) for n in first['values'])):
        raise ValueError('Same-route complete-value trace/full-state replay differs')
    return True


def collect_and_validate(runtime,output):
    from .diagnostic_watch import mark
    base_hash=runtime.state_hash(runtime.base);results={};selected=None;parity=None
    try:
        mark('key_bias_trace_original_parity',flush=True);original=produce(runtime,'original')
        baseline=produce(runtime,'fp32');parity=runtime.metrics(original['comparison'],baseline['comparison'])
        if max(parity.values())>1e-5: raise ValueError('Trace FP32 instrumentation adapter differs from original')
        del original
        for route in ('fp32','bf16_analytic'):
            mark('key_bias_trace_'+route,flush=True)
            first=baseline if route=='fp32' else produce(runtime,route)
            second=produce(runtime,route);replay_equal(runtime,first,second)
            identity=tuple(map(int,first['ids']))
            if selected is None: selected=identity;labels=first['labels'].copy()
            elif identity!=selected or not runtime.np.array_equal(labels,first['labels']): raise ValueError('Paired source samples/targets differ')
            results[route]=emit(runtime,Path(output)/('trace-'+route),route,first)
            results[route]['full_model_replay_exact']=True
            del first,second
            if route=='fp32': del baseline
        return dict(passed=True,schema='sample-trace.v2',routes=results,same_samples=True,
            fp32_adapter_parity=parity,source_train_sha256=runtime.split.identity,
            role_mapping=dict(fp32='baseline',bf16_analytic='intervention'),selection='First2 reviewed cursor128 rows',
            target_boundary_passed=True,test_access=False,optimizer_commits=0)
    finally:
        runtime.restore(runtime.base)
        if runtime.state_hash(runtime.snapshot())!=base_hash: raise ValueError('Trace collector full-state rollback differs')
