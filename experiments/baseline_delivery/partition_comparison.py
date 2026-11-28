"""Bounded rollback-only Graph-versus-Graph timing; no framework imports."""
import math
import statistics
import time


def execution_validation(executor,execution_key):
    """Identify the exact resident capture whose full update was validated."""
    mode=executor.state['mode'];generation=executor.runtime.generation()
    return dict(mode=mode,generation=generation,validation_checks=executor.checks,
        graph_resident=executor.runtime.status().get('graph_resident') is True,
        validated=(mode,execution_key,generation) in executor.validated)


def require_steady(value):
    if (not isinstance(value,dict) or value.get('mode')!='compile_graph'
            or value.get('graph_resident') is not True or value.get('validated') is not True
            or type(value.get('generation')) is not int or value['generation']<1
            or type(value.get('validation_checks')) is not int or value['validation_checks']<1):
        raise ValueError('Graph capture is not resident and validated before timing')


def prepare_route(run,validation,restore,synchronize,attempts=3):
    """A clear may leave an old validated generation until a new capture runs.

    Wait for the *resulting* capture to be validated, outside every timing pair.
    """
    for _ in range(attempts):
        restore();run();synchronize()
        try: require_steady(validation())
        except ValueError: continue
        return
    raise ValueError('Graph route did not reach a validated resident generation')


def validate_comparison(report):
    if (not isinstance(report,dict) or report.get('schema')!='qkan-incumbent-graph-comparison.v2'
            or report.get('measurement')!='balanced_synchronized_incumbent_graph.v2'
            or report.get('pair_orders')!=[['incumbent','candidate'],['candidate','incumbent']]
            or report.get('incumbent_micro_batch')!=16 or report.get('candidate_micro_batch')!=32
            or report.get('samples_committed')!=0 or report.get('optimizer_updates_committed')!=0
            or report.get('simultaneous_resident_routes')!=1):
        raise ValueError('Incumbent Graph comparison identity differs')
    samples=report.get('samples_seconds',{})
    for name in ('incumbent','candidate'):
        values=samples.get(name)
        if (not isinstance(values,list) or len(values)!=2
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values)):
            raise ValueError('Invalid incumbent comparison samples')
        if report.get(name+'_seconds')!=statistics.median(values):
            raise ValueError('Incumbent comparison median differs')
        proof=report.get('steady_validation',{}).get(name)
        if not isinstance(proof,list) or len(proof)!=2:
            raise ValueError('Graph timing validation evidence is missing')
        for row in proof:
            if not isinstance(row,dict) or row.get('before')!=row.get('after'):
                raise ValueError('Graph capture or validation changed inside timing')
            require_steady(row.get('before'))
    if report.get('passed') is not True or report['candidate_seconds']>=report['incumbent_seconds']:
        raise ValueError('Candidate did not improve on the incumbent Graph')
    if report.get('throughput_gain_fraction')!=report['incumbent_seconds']/report['candidate_seconds']-1:
        raise ValueError('Incumbent comparison gain differs')


def compare_routes(routes,restore,synchronize,memory,clock=time.perf_counter):
    """At most one resident route; prepare/validate outside measured intervals."""
    if set(routes)!= {'incumbent','candidate'}: raise ValueError('Expected two Graph routes')
    samples={name:[] for name in routes};peaks={name:[] for name in routes}
    proof={name:[] for name in routes}
    preparation={name:[] for name in routes}
    orders=[('incumbent','candidate'),('candidate','incumbent')]
    active=None
    try:
        for order in orders:
            for name in order:
                if name!=active:
                    if active is not None:
                        synchronize();routes[active]['clear']()
                    restore();routes[name]['reset_peak']();routes[name]['prepare']();synchronize()
                    preparation[name].append(memory());restore()
                    active=name
                synchronize();restore();routes[name]['reset_peak']();synchronize()
                before=routes[name]['validation']();require_steady(before)
                start=clock();routes[name]['run']();synchronize();elapsed=clock()-start
                after=routes[name]['validation']();require_steady(after)
                if before!=after: raise ValueError('Graph capture or validation changed inside timing')
                if not math.isfinite(elapsed) or elapsed<=0: raise ValueError('Invalid Graph timing')
                samples[name].append(elapsed);peaks[name].append(memory())
                proof[name].append(dict(before=before,after=after));restore()
    finally:
        # Always release both, including a partially prepared failing route.
        try:
            routes['incumbent']['clear']()
        finally:
            try: routes['candidate']['clear']()
            finally: restore();synchronize()
    old=statistics.median(samples['incumbent']);new=statistics.median(samples['candidate'])
    return dict(schema='qkan-incumbent-graph-comparison.v2',passed=new<old,
        measurement='balanced_synchronized_incumbent_graph.v2',steady_validation=proof,
        incumbent_micro_batch=16,candidate_micro_batch=32,
        pair_orders=[list(order) for order in orders],samples_seconds=samples,
        incumbent_seconds=old,candidate_seconds=new,throughput_gain_fraction=old/new-1,
        memory=peaks,preparation_memory=preparation,
        preparation_memory_scope='Capture/warmup/numerical validation peak; excluded from timing',
        simultaneous_resident_routes=1,samples_committed=0,optimizer_updates_committed=0,
        interpretation='Matched-micro eager agreement is checked separately; cross-micro trajectories need not be bitwise identical.')
