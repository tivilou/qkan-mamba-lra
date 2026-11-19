"""CPU-only Q/K/V bias analysis of the pinned, completed attribution file."""
import argparse
import datetime as dt
import math
import os
from pathlib import Path
import sys
import uuid

from .attention_attribution import FIELDS
from .common import ROOT, atomic_json, digest, read, revision, worker_failure

TARGET = 'experiments.baseline_delivery.attention_bias_postprocess'
REPORT_SHA = 'ce784d88f6b9269c5f321f2a890b15c3a9bd2060b0d8d630bc6216a646f7ef44'
VALUES_SHA = 'b0d0c8797a79865314ab37761ff52b3da0ef2f160c56994b8ad5aefd9bc6154f'
VALUES_BYTES = 45431227
NAMES = tuple('encoder.layers.'+str(i)+'.self_attn.in_proj_bias' for i in range(4))
SLICES = dict(Q=(0,128), K=(128,256), V=(256,384))
SCHEMA = 'qkan-attention-bias-postprocess.v1'


def bind(suite):
    suite=Path(suite).resolve();report=suite/'attention-attribution.json'
    values=suite/'attribution/paired-parameters.pt'
    if (report.stat().st_size>512*1024 or digest(report)!=REPORT_SHA
            or values.stat().st_size!=VALUES_BYTES or digest(values)!=VALUES_SHA):
        raise ValueError('Pinned attribution evidence differs; preserve it and stop')
    prior=read(report);child=prior['results']['attribution']
    if (prior['source_revision']!='ebe7e4732dc04264bbf257d2ec686657396cf44a'
            or prior['attribution_complete'] is not True or prior['candidate_admitted'] is not False
            or child['private_values']!=dict(file='paired-parameters.pt',bytes=VALUES_BYTES,
                sha256=VALUES_SHA,complete_values_reopened_exact=True)):
        raise ValueError('Reviewed attribution identity differs')
    return prior,values,dict(report_sha256=REPORT_SHA,values_sha256=VALUES_SHA,
        values_bytes=VALUES_BYTES,checkpoint_sha256=child['checkpoint_sha256'],
        checkpoint_step=child['checkpoint_step'],row_ids_sha256=child['row_ids_sha256'])


def stats(values):
    values=list(values)
    if not all(math.isfinite(x) for x in values): raise ValueError('Nonfinite evidence')
    squares=math.fsum(x*x for x in values)
    return dict(numel=len(values),l2=math.sqrt(squares),squared_l2=squares,
        zero_count=sum(x==0 for x in values),absolute_quantiles=quantiles([abs(x) for x in values]))


def quantiles(values):
    values=sorted(values)
    def at(q):
        if not values: return None
        p=(len(values)-1)*q;i=int(p);j=min(i+1,len(values)-1)
        return values[i]+(values[j]-values[i])*(p-i)
    return {name:at(q) for name,q in [('min',0),('p10',.1),('p50',.5),('p90',.9),('max',1)]}


def paired(left,right):
    if len(left)!=len(right): raise ValueError('Pair lengths differ')
    a,b=stats(left),stats(right);d=stats(y-x for x,y in zip(left,right))
    return dict(reference=a,candidate=b,difference=d,
        relative_l2=d['l2']/max(a['l2'],1e-12),denominator_clamped=a['l2']<1e-12,
        sign_disagreements=sum(x*y<0 for x,y in zip(left,right)))


def close(actual,expected):
    if not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-20):
        raise ValueError('Stored values do not reconstruct reviewed summary')


def group_for(route,name):
    groups=[g for g in route['optimizer_groups'] if name in g['parameters']]
    if len(groups)!=1: raise ValueError('Bias optimizer group coverage differs')
    g=groups[0];lr,eps,b1,b2=g['lr'],g['eps'],*g['betas']
    step=route['optimizer_steps_after'][name];before=route['optimizer_steps_before'][name]
    if (not all(math.isfinite(x) for x in (lr,eps,b1,b2,step,before,g['weight_decay']))
            or lr<=0 or eps<=0 or not 0<=b1<1 or not 0<=b2<1
            or step<1 or step!=int(step) or before!=step-1 or g['amsgrad'] or g['maximize']):
        raise ValueError('Unsupported or invalid observed Adam policy')
    return dict(lr=lr,eps=eps,betas=[b1,b2],step_before=before,step_after=step,
        weight_decay=g['weight_decay'],amsgrad=False,maximize=False)


def components(values,policy):
    """Float64 equation using saved real moments; no optimizer/model execution."""
    b1,b2=policy['betas'];step=policy['step_after'];eps=policy['eps'];lr=policy['lr']
    v=values['exp_avg_sq_after'];m=values['exp_avg_after'];grad=values['clipped_gradients']
    if any(x<0 for x in v+values['exp_avg_sq_before']): raise ValueError('Negative Adam variance')
    denom=[math.sqrt(x/(1-b2**step))+eps for x in v]
    update=[-lr*x/(1-b1**step)/d for x,d in zip(m,denom)]
    predicted_m=[b1*x+(1-b1)*g for x,g in zip(values['exp_avg_before'],grad)]
    predicted_v=[b2*x+(1-b2)*g*g for x,g in zip(values['exp_avg_sq_before'],grad)]
    return dict(denominator=denom,adaptive_update=update,
        first_moment_equation=paired(predicted_m,m),second_moment_equation=paired(predicted_v,v))


def slice_summary(left,right,policies,global_error,packed_error):
    fields={f:paired(left[f],right[f]) for f in FIELDS}
    a,b=components(left,policies[0]),components(right,policies[1])
    error=fields['deltas']['difference']['squared_l2']
    observed=[y-x for x,y in zip(left['deltas'],right['deltas'])]
    predicted=[y-x for x,y in zip(a['adaptive_update'],b['adaptive_update'])]
    residual=paired(observed,predicted)
    bands=[];eps=policies[0]['eps']
    # Bins are fixed before analysis and use only the reference denominator.
    for label,lo,hi in [('eps_to_2eps',0,2),('2eps_to_10eps',2,10),
                         ('10eps_to_100eps',10,100),('above_100eps',100,math.inf)]:
        indices=[i for i,d in enumerate(a['denominator']) if lo*eps<d<=hi*eps]
        delta=paired([left['deltas'][i] for i in indices],[right['deltas'][i] for i in indices])
        bands.append(dict(band=label,count=len(indices),delta=delta,
            global_error_energy_share=delta['difference']['squared_l2']/global_error if global_error else 0))
    return dict(fields=fields,global_error_energy_share=error/global_error if global_error else 0,
        packed_bias_error_energy_share=error/packed_error if packed_error else 0,
        denominator=paired(a['denominator'],b['denominator']),
        denominator_candidate_over_reference=quantiles([y/x for x,y in zip(a['denominator'],b['denominator'])]),
        adaptive_update_from_saved_moments=paired(a['adaptive_update'],b['adaptive_update']),
        observed_vs_adaptive_route_difference=residual,
        moment_equation_residuals={route:{key:c[key] for key in ('first_moment_equation','second_moment_equation')}
            for route,c in [('fp32',a),('bf16',b)]},reference_denominator_bands=bands)


def analyze(pairs,prior):
    if set(pairs)!={'fp32','bf16'}: raise ValueError('Both completed routes required')
    summary=prior['results']['attribution']['summary'];routes=[pairs[r] for r in ('fp32','bf16')]
    for r in routes:
        if (r['row_ids_sha256']!=prior['results']['attribution']['row_ids_sha256']
                or r['optimizer_groups']!=summary['optimizer_groups']
                or any(r['optimizer_steps_'+when]!=summary['optimizer_steps_'+when] for when in ('before','after'))):
            raise ValueError('Private values and reviewed policy/population differ')
    global_error=summary['fields']['deltas']['total']['difference_squared_l2']
    packed_error=math.fsum(summary['fields']['deltas']['parameters'][n]['difference_squared_l2'] for n in NAMES)
    layers={};pooled={r:{part:{f:[] for f in FIELDS} for part in SLICES} for r in ('fp32','bf16')}
    for name in NAMES:
        policies=[group_for(r,name) for r in routes]
        if policies[0]!=policies[1]: raise ValueError('Route Adam policy differs')
        vals=[]
        for r in routes:
            values={}
            for f in FIELDS:
                t=r[f][name]
                if tuple(t.shape)!=(384,) or t.device.type!='cpu' or str(t.dtype)!='torch.float32':
                    raise ValueError('Expected384 FP32 CPU bias values')
                values[f]=t.detach().double().reshape(-1).tolist()
                stats(values[f])
            vals.append(values)
        for f in FIELDS:
            measured=paired(vals[0][f],vals[1][f]);expected=summary['fields'][f]['parameters'][name]
            for label,key in [('reference','reference_squared_l2'),('candidate','candidate_squared_l2'),('difference','difference_squared_l2')]:
                close(measured[label]['squared_l2'],expected[key])
        parts={}
        for part,(start,end) in SLICES.items():
            selected=[{f:v[f][start:end] for f in FIELDS} for v in vals]
            parts[part]=slice_summary(*selected,policies,global_error,packed_error)
            for route,v in zip(('fp32','bf16'),selected):
                for f in FIELDS: pooled[route][part][f].extend(v[f])
        for f in FIELDS:
            close(math.fsum(parts[p]['fields'][f]['difference']['squared_l2'] for p in SLICES),
                  summary['fields'][f]['parameters'][name]['difference_squared_l2'])
        layers[name]=dict(policy=policies[0],slices=parts)
    # Pool only compatible policies; each segment aggregates all four layers.
    policies=[layers[n]['policy'] for n in NAMES]
    if any(p!=policies[0] for p in policies): raise ValueError('Cannot pool different Adam policies')
    pooled_summary={part:slice_summary(pooled['fp32'][part],pooled['bf16'][part],
        [policies[0],policies[0]],global_error,packed_error) for part in SLICES}
    close(math.fsum(v['fields']['deltas']['difference']['squared_l2'] for v in pooled_summary.values()),packed_error)
    return dict(layers=layers,pooled_qkv=pooled_summary,
        reviewed_global_delta=summary['fields']['deltas']['total'],
        packed_bias_error_energy_share=packed_error/global_error,
        coverage=dict(layers=4,packed_elements=1536,segment_width=128,pooled_elements_per_segment=512,
            slices={k:list(v) for k,v in SLICES.items()},all_fields_reconstruct_reviewed_packed_summaries=True),
        interpretation=[
            'Post-hoc localization only; Q/K/V shares and denominator bins never replace the fixed global gate.',
            'Denominator = sqrt(exp_avg_sq_after / (1-beta2**step_after)) + eps; adaptive update = -lr * m_hat / denominator.',
            'Float64 equations use saved FP32 moments; moment-equation residuals include original arithmetic rounding.',
            'Original weights are absent. Adaptive route-difference residual includes FP32 step rounding and unmodeled weight-decay arithmetic; it is not exact full-step reconstruction.',
            'Key-bias softmax cancellation is a hypothesis. These statistics alone do not prove causality or convergence.'])


def execute_cpu(suite,output):
    from .diagnostic_watch import mark
    prior,path,binding=bind(suite)
    mark('bias_cpu_import',flush=True)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    mark('bias_cpu_load',flush=True)
    pairs=torch.load(path,map_location='cpu',weights_only=True)
    mark('bias_cpu_reduce',flush=True)
    result=analyze(pairs,prior)
    # Check both originals again, including on exceptions in the caller.
    if bind(suite)[2]!=binding: raise ValueError('Evidence changed during CPU analysis')
    result.update(schema=SCHEMA,postprocess_complete=True,candidate_admitted=False,binding=binding,
        input_integrity=dict(before_after_sha256_equal=True,source_files_written=0),
        execution=dict(device='cpu',torch=str(torch.__version__),threads=1,
            forward_calls=0,backward_calls=0,optimizer_steps=0,training_samples_committed=0))
    atomic_json(Path(output)/'result.json',result)
    mark('bias_cpu_complete',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attribution-suite',required=True,type=Path)
    p.add_argument('--child-timeout',type=float,default=900)
    p.add_argument('--stall-timeout',type=float,default=300)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--output',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        if args.output is None: p.error('Worker output required')
        try: execute_cpu(args.attribution_suite,args.output)
        except BaseException as error:
            worker_failure(args.output,error)
            raise
        return 0
    from .bootstrap import check_exchange
    from .diagnostic_watch import Supervisor
    from .return_artifacts import upload_declared
    supervisor=Supervisor(args.child_timeout,args.stall_timeout)
    check_exchange()
    folder=ROOT/'runs'/('attention-bias-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    print('CPU Q/K/V postprocess Suite: '+str(folder),flush=True)
    report=dict(schema=SCHEMA,source_revision=revision(),implementation_sha256=digest(__file__),postprocess_complete=False,
        candidate_admitted=False,formal_training_started=False,training_samples_committed=0)
    error=None
    try:
        _,_,binding=bind(args.attribution_suite)
        report['binding']=binding
        output=folder/'cpu';output.mkdir()
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        code=supervisor.run(dict(target=TARGET,cwd=str(ROOT),argv=[sys.executable,'-B','-m',TARGET,
            '--worker','--attribution-suite',str(args.attribution_suite.resolve()),'--output',str(output)]),folder/'cpu.log',env)
        result=read(output/'result.json') if (output/'result.json').is_file() else {}
        if (code!=0 or result.get('schema')!=SCHEMA or result.get('binding')!=binding
                or result.get('postprocess_complete') is not True or result.get('candidate_admitted') is not False):
            raise ValueError('CPU result incomplete; inspect retained child diagnostics')
        report.update(result)
    except BaseException as caught:
        error=caught;report['error_type']=type(caught).__name__
        worker_failure(folder,caught)
        report['failure_locations']=read(folder/'worker-error.json')
        child=folder/'cpu/worker-error.json'
        if child.is_file(): report['child_failure_locations']=read(child)
    finally:
        report['child_executions']=supervisor.records
        try:
            bind(args.attribution_suite)
            report['input_integrity']=dict(before_after_sha256_equal=True,source_files_written=0)
        except Exception:
            report['input_integrity']=dict(before_after_sha256_equal=False,source_files_written=0)
            report['postprocess_complete']=False
            if error is None: error=ValueError('Pinned evidence changed')
        atomic_json(folder/'attention-bias.json',report)
        try: upload_declared(folder,'attention-bias',['attention-bias.json'])
        except Exception:
            print('Return pending; report and artifact-delivery.json retained: '+str(folder),flush=True)
            if error is None: raise
    if error is not None: raise error
    print('CPU Q/K/V postprocess complete; see attention-bias.json.',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
