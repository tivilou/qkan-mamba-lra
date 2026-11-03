"""Bounded repeated recovery diagnostic; never changes numerical acceptance."""
import argparse
from itertools import combinations
import os
from pathlib import Path
import uuid

from .common import atomic_json, child_spec, digest, read, revision, run_child
from .recovery_audit import FIELDS, check_source, compare


def comparison(left, right):
    result = compare(left, right)
    # Discrete sample rows, metric-selection decisions and scheduler are controls too.
    result['control_state_exact'] = result['control_state_exact'] and all(
        leaf['exact'] for leaf in result['leaves']
        if leaf['kind'] not in ('floating_array', 'floating_scalar')
        or leaf['path'].split('/')[0] == 'scheduler')
    groups = {}
    for field in FIELDS:
        leaves = [leaf for leaf in result['leaves'] if leaf['path'].split('/')[0] == field]
        groups[field] = dict(exact=all(leaf['exact'] for leaf in leaves),
                             differing_leaves=sum(not leaf['exact'] for leaf in leaves),
                             max_abs=max((leaf.get('max_abs', 0.) for leaf in leaves), default=0.))
    result['fields'] = groups
    return result


def summarize(comparisons):
    """Descriptive per-field samples; overlapping pairs are not independent trials."""
    result = {}
    for kind in ('uninterrupted_repeat', 'paired_recovery'):
        selected = [item for item in comparisons.values() if item['kind'] == kind]
        result[kind] = dict(
            comparisons=len(selected),
            all_finite=all(item['all_finite'] for item in selected),
            control_state_exact=all(item['control_state_exact'] for item in selected),
            fields={field: dict(
                max_abs_samples=[item['fields'][field]['max_abs'] for item in selected],
                differing_leaf_samples=[item['fields'][field]['differing_leaves'] for item in selected])
                for field in FIELDS})
    return result


def execute(folder, cache, device, repeats, audit, audit_revision):
    config = read(folder/'config.json')
    if config['epochs'] != 2 or not (folder/'data/synthetic-canary.json').is_file():
        raise ValueError('Requires an existing bounded canary')
    hashes = check_source(config)
    marker = read(folder/'data/synthetic-canary.json')
    if marker.get('bounded') is not True or marker.get('test_included') is not False:
        raise ValueError('Canary must exclude test data')
    env = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONUNBUFFERED': '1'}
    paths = {}
    # Same seed/config/data in every run: replication measures execution noise,
    # not variability across training seeds. Every stage is a fresh process.
    for index in range(repeats):
        for name, extra, expected in (
                ('full', [], 0), ('resumed', ['--stop-after', '2'], 75),
                ('resumed', ['--resume'], 0)):
            key = f'{index:02d}-{name}'
            output = audit/key
            spec = child_spec('--config', folder/'config.json', '--data', folder/'data',
                              '--cache', cache, '--output', output, '--device', device,
                              '--synthetic', *extra)
            stage = key + ('-resume' if extra == ['--resume'] else '')
            atomic_json(audit/'progress.json', dict(pair=index+1, pairs=repeats, stage=stage))
            print(f'Pair {index+1}/{repeats}: {stage}', flush=True)
            if run_child(spec, audit/(stage+'.log'), env) != expected:
                raise ValueError('Diagnostic child failed: '+stage)
            paths[key] = output
    # Import torch only after all training subprocesses have exited.
    from .worker import load_checkpoint
    states, checkpoints = {}, {}
    environment = None
    for key, path in paths.items():
        if not read(path/'trace-validation.json')['passed']:
            raise ValueError('Trace validation failed: '+key)
        current = read(path/'environment.json')
        if environment is None:
            environment = current
        if current != environment:
            raise ValueError('Environment records differ: '+key)
        pointer = read(path/'checkpoints/latest.json')
        state = load_checkpoint(path, pointer['contract'])
        if not state['training_complete'] or state['step'] != 6 or state['epoch'] != 2:
            raise ValueError('Bounded diagnostic did not complete six updates: '+key)
        states[key] = state
        checkpoints[key] = pointer['sha256']
    if len({state['contract'] for state in states.values()}) != 1:
        raise ValueError('Scientific contracts differ')
    comparisons = {}
    pairs = [(f'{i:02d}-full', f'{i:02d}-resumed', 'paired_recovery') for i in range(repeats)]
    pairs += [(a, b, 'uninterrupted_repeat') for a, b in
              combinations([f'{i:02d}-full' for i in range(repeats)], 2)]
    for left, right, kind in pairs:
        comparisons[left+'_vs_'+right] = dict(
            kind=kind, left=left, right=right, **comparison(states[left], states[right]))
    summary = summarize(comparisons)
    report = dict(schema='qkan-recovery-replicates.v1', status='diagnostic_only', formal=False,
                  recovery_gate_changed=False, numerical_acceptance_criterion=None,
                  audit_revision=audit_revision, producer_revision=config['source_revision'],
                  scientific_source_sha256=hashes, config_sha256=digest(folder/'config.json'),
                  model=config['model'], task=config['task'], seed=config['seed'], device=device,
                  repeats=repeats, interruption_after_update=2, updates=6,
                  checkpoint_sha256=checkpoints, environment_records_equal=True,
                  comparisons=comparisons, summary=summary,
                  limitations='Same-seed bounded diagnostic only. Pairwise repeat comparisons share runs '
                  'and are dependent. Observed maxima are not statistical noise bounds. No numerical '
                  'recovery certification, significance claim or formal-training authorization.')
    atomic_json(audit/'report.json', report)
    atomic_json(audit/'summary.json', {k: v for k, v in report.items() if k != 'comparisons'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canary', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--repeats', type=int, default=5, choices=range(3, 11))
    args = parser.parse_args()
    from .bootstrap import check_exchange
    from .return_artifacts import upload_declared
    audit_revision = revision()
    check_exchange()  # Fail before computation if return configuration is missing.
    folder = args.canary.resolve()
    audit = folder/('recovery-replicates-'+uuid.uuid4().hex[:12])
    audit.mkdir()
    print('Audit: '+str(audit), flush=True)
    failed = False
    try:
        execute(folder, args.cache.resolve(), args.device, args.repeats, audit, audit_revision)
    except Exception as error:
        failed = True
        # No credential-bearing exception text in shared reports.
        report = dict(schema='qkan-recovery-replicates.v1', status='failed', formal=False,
                      recovery_gate_changed=False, numerical_acceptance_criterion=None,
                      error_type=type(error).__name__, audit_revision=audit_revision,
                      progress=read(audit/'progress.json') if (audit/'progress.json').exists() else None)
        atomic_json(audit/'report.json', report)
        atomic_json(audit/'summary.json', report)
    try:
        upload_declared(audit, 'recovery-replicates', ['report.json', 'summary.json'])
    except Exception as error:
        raise SystemExit('Upload failed ('+type(error).__name__+'); local reports retained. Retry: '
                         'python -m experiments.baseline_delivery.return_artifacts --directory '
                         +str(audit)+' --experiment recovery-replicates')
    if failed:
        raise SystemExit('Diagnostic failed; failure report uploaded. Inspect local stage logs in '+str(audit))
    print('Diagnostic complete and returned; recovery gate unchanged.', flush=True)


if __name__ == '__main__':
    main()
