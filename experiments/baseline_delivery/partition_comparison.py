"""Bounded rollback-only Graph-versus-Graph timing; no framework imports."""
import math
import statistics
import time


def validate_comparison(report):
    if (not isinstance(report,dict) or report.get('schema')!='qkan-incumbent-graph-comparison.v1'
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
    if report.get('passed') is not True or report['candidate_seconds']>=report['incumbent_seconds']:
        raise ValueError('Candidate did not improve on the incumbent Graph')
    if report.get('throughput_gain_fraction')!=report['incumbent_seconds']/report['candidate_seconds']-1:
        raise ValueError('Incumbent comparison gain differs')


def compare_routes(routes,restore,synchronize,memory,clock=time.perf_counter):
    """At most one resident route; prepare/validate outside measured intervals."""
    if set(routes)!= {'incumbent','candidate'}: raise ValueError('Expected two Graph routes')
    samples={name:[] for name in routes};peaks={name:[] for name in routes}
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
                start=clock();routes[name]['run']();synchronize();elapsed=clock()-start
                if not math.isfinite(elapsed) or elapsed<=0: raise ValueError('Invalid Graph timing')
                samples[name].append(elapsed);peaks[name].append(memory());restore()
    finally:
        # Always release both, including a partially prepared failing route.
        try:
            routes['incumbent']['clear']()
        finally:
            try: routes['candidate']['clear']()
            finally: restore();synchronize()
    old=statistics.median(samples['incumbent']);new=statistics.median(samples['candidate'])
    return dict(schema='qkan-incumbent-graph-comparison.v1',passed=new<old,
        measurement='balanced_synchronized_incumbent_graph.v1',
        incumbent_micro_batch=16,candidate_micro_batch=32,
        pair_orders=[list(order) for order in orders],samples_seconds=samples,
        incumbent_seconds=old,candidate_seconds=new,throughput_gain_fraction=old/new-1,
        memory=peaks,preparation_memory=preparation,
        preparation_memory_scope='Capture/warmup/numerical validation peak; excluded from timing',
        simultaneous_resident_routes=1,samples_committed=0,optimizer_updates_committed=0,
        interpretation='Matched-micro eager agreement is checked separately; cross-micro trajectories need not be bitwise identical.')
