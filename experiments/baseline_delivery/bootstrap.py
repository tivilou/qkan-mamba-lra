"""Installation identity checks; no training or environment changes."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path

MAMBA_COMMIT = '95d8aba8a8c75aedcaa6143713b11e745e7cd0d9'


def check_exchange():
    for key in ('PROJECT_EXCHANGE_URL', 'PROJECT_EXCHANGE_TOKEN', 'PROJECT_EXCHANGE_CA_FILE'):
        if not os.environ.get(key):
            raise ValueError(f'Set {key} in this terminal before starting installation')
    if not os.environ['PROJECT_EXCHANGE_URL'].startswith('https://'):
        raise ValueError('PROJECT_EXCHANGE_URL must use HTTPS')
    if not Path(os.environ['PROJECT_EXCHANGE_CA_FILE']).is_file():
        raise ValueError('PROJECT_EXCHANGE_CA_FILE does not point to a certificate file')


def check_core():
    torch = importlib.import_module('torch')
    if torch.__version__ != '2.7.1+cu126' or torch.version.cuda != '12.6':
        raise ValueError('Expected torch 2.7.1+cu126 with CUDA 12.6')
    for name, version in (('numpy', '2.2.6'), ('einops', '0.8.1'), ('triton', '3.3.1'),
                          ('nvidia-cudnn-cu12', '9.5.1.17')):
        if metadata.version(name) != version:
            raise ValueError(f'Expected {name}=={version}')
    runtime = tuple(torch._C._cudnn.getRuntimeVersion())
    if runtime != (9, 5, 1):
        raise ValueError(f'cuDNN library files do not match pinned 9.5.1: loaded {runtime}. Restore core mirror wheels; package metadata alone is insufficient.')


def check_cuda():
    check_core()
    torch = importlib.import_module('torch')
    x = torch.randn(2, 128, 2048, device='cuda:0', requires_grad=True)
    layer = torch.nn.Conv1d(128, 128, 1).to('cuda:0')
    layer(x).square().mean().backward()
    torch.cuda.synchronize()
    print('CUDA/cuDNN Conv1d forward/backward preflight passed', flush=True)


def check_mamba():
    check_core()
    dist = metadata.distribution('mamba-ssm')
    source = json.loads(dist.read_text('direct_url.json') or '{}')
    archive = source.get('archive_info', {}).get('hashes', {}).get('sha256')
    mirrored = False
    if archive:
        manifest = json.loads((Path(__file__).parent / 'install-mirror.json').read_text())
        mirrored = any(item['filename'] == 'mamba-2.2.4.tar.gz' and
                       item['sha256'] == archive and item.get('upstream_commit') == MAMBA_COMMIT
                       for item in manifest['objects'])
    if dist.version != '2.2.4' or (not mirrored and source.get('vcs_info', {}).get('commit_id') != MAMBA_COMMIT):
        raise ValueError('Mamba package does not match the pinned upstream commit')
    url = source.get('url', '')
    if not mirrored and url.removesuffix('.git') != 'https://github.com/state-spaces/mamba':
        raise ValueError('Unexpected Mamba upstream repository')
    importlib.import_module('selective_scan_cuda')
    importlib.import_module('mamba_ssm.modules.mamba_simple').Mamba


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('exchange', 'core', 'mamba', 'cuda'))
    args = parser.parse_args()
    try:
        {'exchange': check_exchange, 'core': check_core, 'mamba': check_mamba, 'cuda': check_cuda}[args.stage]()
    except Exception as error:
        parser.exit(2, f'Bootstrap check failed ({args.stage}): {type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
