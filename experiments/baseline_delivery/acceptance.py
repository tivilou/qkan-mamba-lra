"""One collaborator acceptance handoff; synthetic probes, no formal training."""
import argparse
import copy
import os
from pathlib import Path
import sys
import uuid

from .common import ROOT, atomic_json, child_spec, digest, read, revision, run_child


def fidelity(model_name, cache, device, output):
    import torch
    from torch import nn
    from .models import Classifier, optimizer
    from .runner import implementation_hash
    torch.manual_seed(42)
    config = dict(model=model_name, task='retrieval', vocabulary=32, width=16,
                  layers=2, heads=2, state_size=4, length=32, dropout=0.,
                  lr=.001, weight_decay=.01)
    adapter = Classifier(config, cache).to(device)
    reference = copy.deepcopy(adapter)
    # Parameter deepcopy does not promise to preserve custom optimizer metadata.
    for (name, x), (other, y) in zip(adapter.named_parameters(), reference.named_parameters()):
        if name != other: raise ValueError('Parameter structure mismatch')
        if hasattr(x, '_optim'): y._optim = copy.deepcopy(x._optim)
    # Independent task adapter composition around the pinned upstream modules.
    # This checks the adapter boundary, not paper-level architecture equivalence.
    def encode(x):
        mask = x != 1
        h = reference.embedding(x).masked_fill(~mask[..., None], 0)
        if model_name == 'transformer':
            h = h + reference.position[:x.shape[1]]
            for layer in reference.encoder.layers:
                h = layer(h, src_key_padding_mask=~mask)
            if reference.encoder.norm is not None: h = reference.encoder.norm(h)
        else:
            for block in reference.blocks:
                z = block.mixer(block.norm(h))
                if model_name == 's4d_minimal': z = z[0]
                h = (h + block.dropout(z)).masked_fill(~mask[..., None], 0)
        h = reference.norm(h).masked_fill(~mask[..., None], 0)
        return h.sum(dim=1)/mask.sum(dim=1, keepdim=True)
    batch = {name: torch.randint(2, 32, (2, 32), device=device)
             for name in ('inputs1', 'inputs2')}
    batch['inputs1'][0, -4:] = 1
    batch['inputs2'][1, -8:] = 1
    labels = torch.tensor([0, 1], device=device)
    a = adapter(batch)
    left, right = encode(batch['inputs1']), encode(batch['inputs2'])
    b = reference.head(torch.cat(((left-right).abs(), left*right), dim=-1))
    results = {}
    def check(name, x, y):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
        results[name] = float((x-y).detach().abs().max().cpu())
    check('logits', a, b)
    loss_a, loss_b = nn.functional.cross_entropy(a, labels), nn.functional.cross_entropy(b, labels)
    check('loss', loss_a, loss_b)
    loss_a.backward(); loss_b.backward()
    for (name, x), (other, y) in zip(adapter.named_parameters(), reference.named_parameters()):
        if name != other or (x.grad is None) != (y.grad is None):
            raise ValueError('Gradient structure mismatch')
        if x.grad is not None: check('gradient/'+name, x.grad, y.grad)
    optimizer(adapter, config).step(); optimizer(reference, config).step()
    for name, x in adapter.state_dict().items(): check('updated/'+name, x, reference.state_dict()[name])
    atomic_json(output, dict(schema='qkan-adapter-equivalence.v1', model=model_name,
                status='adapter_equivalence_observed', formal=False, passed=True,
                implementation_sha256=implementation_hash(),
                protocol_sha256=digest(ROOT/'experiments/baseline_delivery/protocol.json'),
                source_revision=revision(), device=device, synthetic=True, test_access=False,
                environment={'torch':torch.__version__, 'cuda':torch.version.cuda,
                             'gpu':torch.cuda.get_device_name() if device.startswith('cuda') else None},
                checks=results, scope='Independent task adapter composition; not a fidelity certificate '
                'or reproduction of upstream benchmark scores. Dropout=0 for paired mechanism isolation.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--probe', choices=('transformer', 's4d_minimal', 'mamba1'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.probe:
        if args.output is None: parser.error('--probe requires --output')
        fidelity(args.probe, args.cache, args.device, args.output)
        return
    if args.suite is None: parser.error('--suite required')
    from .bootstrap import check_exchange
    from .return_artifacts import upload_declared
    from .runner import canary
    source = revision(); check_exchange()
    target = ROOT/'runs'/('acceptance-'+uuid.uuid4().hex[:12]); target.mkdir(parents=True)
    print('Acceptance: '+str(target), flush=True)
    results = {}; files = ['report.json']
    failure = False
    try:
        for model in ('transformer', 's4d_minimal', 'mamba1'):
            name = model+'-adapter.json'
            spec = child_spec()  # Reuse normalized interpreter/root, change module explicitly.
            spec['target'] = 'experiments.baseline_delivery.acceptance'
            spec['argv'] = [sys.executable, '-B', '-m', spec['target'], '--probe', model,
                            '--cache', str(args.cache.resolve()), '--device', args.device,
                            '--output', str(target/name)]
            code = run_child(spec, target/(model+'-adapter.log'),
                             {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
            if (target/name).exists(): files.append(name)
            if code != 0: raise ValueError('Adapter probe failed: '+model)
            results[model] = read(target/name)['status']
        # Target unresolved Mamba recovery and Transformer/Retrieval replay.
        for job in ('mamba1-listops-s42', 'transformer-retrieval-s42'):
            configs = list(args.suite.glob('preflight/*/'+job+'/config.json'))
            if len(configs) != 1: raise ValueError('Expected one original canary config: '+job)
            config = {**read(configs[0]), 'source_revision': source}
            receipt = canary(config, configs[0].parent/'data', args.cache, target/job, args.device)
            name = job+'-canary.json'; atomic_json(target/name, receipt); files.append(name)
            if (target/job/'recovery-validation.json').exists():
                detail = job+'-recovery.json'
                atomic_json(target/detail, read(target/job/'recovery-validation.json')); files.append(detail)
            results[job] = receipt['agreement']
    except Exception as error:
        import traceback
        (target/'acceptance-error.log').write_text(traceback.format_exc(), encoding='utf-8')
        files.append('acceptance-error.log')
        failure = True
        results['failure'] = dict(type=type(error).__name__, stage=list(results))
        print('Acceptance failed; inspect local logs in '+str(target), flush=True)
    for model in ('transformer', 's4d_minimal', 'mamba1'):
        name = model+'-adapter.log'
        if (target/name).is_file(): files.append(name)
    for job in ('mamba1-listops-s42', 'transformer-retrieval-s42'):
        recovery = target/job/'recovery-validation.json'
        detail = job+'-recovery.json'
        if recovery.is_file() and detail not in files:
            atomic_json(target/detail, read(recovery)); files.append(detail)
        for log in ('uninterrupted.log', 'interrupted.log', 'interrupted-resume.log'):
            path = target/job/log
            if path.is_file():
                name = job+'-'+log
                (target/name).write_bytes(path.read_bytes()); files.append(name)
    atomic_json(target/'report.json', dict(schema='qkan-consolidated-acceptance.v1',
                status='failed' if failure else 'engineering_checks_complete',
                source_revision=source, formal=False, formal_release_allowed=False, results=results,
                remaining='Dataset qualification, cohort and reviewed fidelity certificates remain required.'))
    try: upload_declared(target, 'acceptance', files)
    except Exception as error:
        raise SystemExit('Upload failed ('+type(error).__name__+'); retry: python -m '
                         'experiments.baseline_delivery.return_artifacts --directory '+str(target)+
                         ' --experiment acceptance '+ ' '.join('--file '+name for name in files))
    if failure: raise SystemExit(1)


if __name__ == '__main__':
    main()
