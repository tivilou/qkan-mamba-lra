"""Parameter-level Adam error measurements; no changed intervention or gate.

Framework supplied by the existing native diagnostic child. Complete paired
values stay private; only scalar summaries and integrity receipts are returned.
"""
import math
from pathlib import Path

from .common import ROOT,atomic_json,digest,read

CONTRACT=ROOT/'docs/baselines/attention-attribution-contract.json'
FIELDS=('gradients','clipped_gradients','deltas','exp_avg_before','exp_avg_after',
        'exp_avg_sq_before','exp_avg_sq_after')


def bind_prior(folder,binding):
    contract=read(CONTRACT);folder=Path(folder).resolve()
    report_path=folder/'attention-update.json';input_path=folder/'update-input.json'
    if (report_path.is_symlink() or input_path.is_symlink()
            or digest(report_path)!=contract['prior_report_sha256']):
        raise ValueError('Attribution requires the reviewed immutable rejected report')
    report=read(report_path);request=read(input_path)
    if (report['status']!='diagnostic_rejected' or report['passed'] is not False
            or report['recovery']['passed'] is not True
            or report['results']['benchmark']['request_sha256']!=digest(input_path)
            or any(request.get(key)!=value for key,value in binding.items())):
        raise ValueError('Original checkpoint, cursor, train data or prior request changed')
    return dict(prior_suite=str(folder),prior_report_sha256=digest(report_path),prior_request_sha256=digest(input_path),
        contract_sha256=digest(CONTRACT),prior_numerical=report['results']['benchmark']['numerical'])


def capture_update(runtime,enabled):
    """Observe the existing optimizer's single step, retaining its real values."""
    opt=runtime.opt;params=dict(runtime.model.named_parameters());captured={};calls=[]
    owned='step' in opt.__dict__;prior=opt.__dict__.get('step');original=opt.step
    def moments(when):
        for moment in ('exp_avg','exp_avg_sq'):
            captured[moment+'_'+when]={name:runtime.clone(opt.state[p][moment])
                for name,p in params.items() if p in opt.state and moment in opt.state[p]}
        captured['optimizer_steps_'+when]={n:float(opt.state[p]['step'])
            for n,p in params.items() if p in opt.state and 'step' in opt.state[p]}
    def step(*args,**kwargs):
        if calls: raise ValueError('Attribution allows exactly one optimizer update per route')
        calls.append(True)
        captured['optimizer_groups']=[dict(parameters=[n for n,p in params.items() if any(p is q for q in g['params'])],
            lr=float(g['lr']),eps=float(g['eps']),betas=list(g['betas']),weight_decay=float(g['weight_decay']),
            amsgrad=bool(g.get('amsgrad',False)),maximize=bool(g.get('maximize',False))) for g in opt.param_groups]
        captured['clipped_gradients']={n:runtime.clone(p.grad) for n,p in params.items() if p.grad is not None}
        moments('before');result=original(*args,**kwargs);moments('after');return result
    try:
        opt.step=step
        result=runtime.update(enabled,dropout_off=True,retain=True)
        if len(calls)!=1: raise ValueError('Optimizer update was not observed')
        result.update(captured)
        return result
    finally:
        if owned: opt.step=prior
        else: del opt.step


def scalar_pair(torch,left,right):
    if tuple(left.shape)!=tuple(right.shape): raise ValueError('Parameter shape differs')
    a=left.detach().cpu().double().reshape(-1);b=right.detach().cpu().double().reshape(-1)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()): raise ValueError('Nonfinite attribution values')
    d=b-a;aa=float((a*a).sum());bb=float((b*b).sum());dd=float((d*d).sum())
    dot=float((a*b).sum());norm=math.sqrt(aa);other=math.sqrt(bb)
    return dict(shape=list(left.shape),numel=a.numel(),reference_l2=norm,candidate_l2=other,
        difference_l2=math.sqrt(dd),reference_squared_l2=aa,candidate_squared_l2=bb,difference_squared_l2=dd,
        relative_l2=math.sqrt(dd)/max(norm,1e-12),denominator_clamped=norm<1e-12,
        max_absolute_difference=float(d.abs().max()) if a.numel() else 0.,
        cosine=dot/(norm*other) if norm and other else None,
        sign_disagreements=int(((a*b)<0).sum()),reference_zero_count=int((a==0).sum()),
        candidate_zero_count=int((b==0).sum()))


def aggregate(rows):
    aa=math.fsum(r['reference_squared_l2'] for r in rows)
    bb=math.fsum(r['candidate_squared_l2'] for r in rows)
    dd=math.fsum(r['difference_squared_l2'] for r in rows)
    return dict(reference_l2=math.sqrt(aa),candidate_l2=math.sqrt(bb),difference_l2=math.sqrt(dd),
        relative_l2=math.sqrt(dd)/max(math.sqrt(aa),1e-12),reference_squared_l2=aa,
        candidate_squared_l2=bb,difference_squared_l2=dd,parameter_count=len(rows),numel=sum(r['numel'] for r in rows))


def module_group(name):
    parts=name.split('.')
    return '.'.join(parts[:3]) if parts[:2]==['encoder','layers'] else parts[0]


def compare(torch,reference,candidate):
    if reference['row_ids_sha256']!=candidate['row_ids_sha256']: raise ValueError('Compared sample population changed')
    if reference['optimizer_groups']!=candidate['optimizer_groups']: raise ValueError('Optimizer policy differs across routes')
    if any(reference['optimizer_steps_'+when]!=candidate['optimizer_steps_'+when] for when in ('before','after')):
        raise ValueError('Optimizer step counters differ across routes')
    fields={}
    for field in FIELDS:
        left,right=reference[field],candidate[field]
        if set(left)!=set(right): raise ValueError('Parameter coverage differs: '+field)
        rows={n:scalar_pair(torch,left[n],right[n]) for n in sorted(left)}
        total=aggregate(list(rows.values()));groups={}
        for n,row in rows.items():
            denom=total['difference_squared_l2']
            row['global_error_energy_share']=row['difference_squared_l2']/denom if denom else 0.
            groups.setdefault(module_group(n),[]).append(row)
        grouped={name:aggregate(values) for name,values in sorted(groups.items())}
        for value in grouped.values():
            denom=total['difference_squared_l2']
            value['global_error_energy_share']=value['difference_squared_l2']/denom if denom else 0.
        fields[field]=dict(total=total,parameters=rows,groups=grouped,
            descending_error_contribution=sorted(rows,key=lambda n:(-rows[n]['difference_squared_l2'],n)))
    result=dict(fields=fields,optimizer_groups=reference['optimizer_groups'],
        optimizer_steps_before=reference['optimizer_steps_before'],optimizer_steps_after=reference['optimizer_steps_after'],
        preexisting_moments_equal=all(fields[f]['total']['difference_squared_l2']==0 for f in ('exp_avg_before','exp_avg_sq_before')),
        missing_before_moments={route:{moment:sorted(set(v['gradients'])-set(v[moment+'_before']))
            for moment in ('exp_avg','exp_avg_sq')} for route,v in [('fp32',reference),('bf16',candidate)]})
    if not result['preexisting_moments_equal']: raise ValueError('Optimizer starting moments differ')
    result['clip_norms']={route:dict(before=fields['gradients']['total'][key+'_l2'],
        after=fields['clipped_gradients']['total'][key+'_l2'],
        observed_norm_ratio=fields['clipped_gradients']['total'][key+'_l2']/max(fields['gradients']['total'][key+'_l2'],1e-12))
        for route,key in [('fp32','reference'),('bf16','candidate')]}
    return result


def run(runtime,output):
    from .attention_update import numerical_gate
    from .diagnostic_watch import mark
    t=runtime.t;output=Path(output);result=dict(passed=False,attribution_complete=False,candidate_admitted=False)
    base_hash=runtime.state_hash(runtime.base);pairs={};start_hashes={}
    try:
        for route in ('fp32','bf16'):
            mark('attention_attribution_'+route,flush=True)
            runtime.restore(runtime.base);start_hashes[route]=runtime.state_hash(runtime.snapshot())
            if start_hashes[route]!=base_hash: raise ValueError('Attribution start state differs')
            pairs[route]=capture_update(runtime,route=='bf16')
        metrics=runtime.metrics(pairs['fp32'],pairs['bf16'])
        summary=compare(t,pairs['fp32'],pairs['bf16'])
        for key,field in [('gradient_relative_l2','gradients'),('adam_delta_relative_l2','deltas')]:
            if not math.isclose(metrics[key],summary['fields'][field]['total']['relative_l2'],rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('Parameter summary does not reconstruct aggregate gate')
        # Complete paired values are private diagnostic evidence, never a formal
        # training checkpoint or an admission substitute for sample-trace.v2.
        path=output/'paired-parameters.pt';t.save(pairs,path)
        if path.stat().st_size>128*1024**2:
            path.unlink();raise ValueError('Private parameter evidence exceeded128MiB bound')
        h=digest(path);loaded=t.load(path,map_location='cpu',weights_only=False)
        if runtime.state_hash(loaded)!=runtime.state_hash(pairs) or digest(path)!=h:
            raise ValueError('Reopened paired parameter values differ')
        result.update(passed=True,attribution_complete=True,numerical=metrics,numerical_passed=numerical_gate(metrics),
            summary=summary,starting_state_sha256=start_hashes,row_ids_sha256=pairs['fp32']['row_ids_sha256'],
            routes=['fp32','bf16'],updates_per_route=1,dropout=0,
            private_values=dict(file=path.name,bytes=path.stat().st_size,sha256=h,complete_values_reopened_exact=True),
            interpretation='Diagnostic localization only; numerical pass is separate. Per-parameter ranks are post-hoc descriptive contributions, not causal ablations or convergence evidence.')
        return result
    finally:
        runtime.restore(runtime.base)
        if runtime.state_hash(runtime.snapshot())!=base_hash: raise ValueError('Attribution rollback differs')
        result['rollback']=dict(full_state_exact=True,committed_samples=0,committed_updates=0)
        atomic_json(output/'attribution.json',result)
