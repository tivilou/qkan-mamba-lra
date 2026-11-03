"""Install from authenticated, hash-pinned exchange objects without public indexes."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys

from .bootstrap import check_exchange
from .common import ROOT, digest, read
from .sync import download


def fetch(cache, manifest):
    check_exchange()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('Installation mirror requires Linux x86_64')
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    if tag != manifest['python_tag']:
        raise ValueError(f'Mirror supports {manifest["python_tag"]}; this environment is {tag}. Use the Python version in the collaborator guide.')
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for item in manifest['objects']:
        name = item['filename']
        if Path(name).name != name or name.startswith('.') or '/' in name or '\\' in name:
            raise ValueError('Unsafe mirror filename')
        target = cache / name
        print(f'Dependency: {name} ({item["size_bytes"] / 1048576:.1f} MiB)', flush=True)
        if target.exists():
            if target.is_symlink() or target.stat().st_size != item['size_bytes'] or digest(target) != item['sha256']:
                raise ValueError('Existing dependency differs from manifest')
            continue
        parts = []
        for chunk in item['chunks']:
            part = cache / (name + f'.chunk{len(parts):03d}')
            print(f'  chunk {len(parts)+1}/{len(item["chunks"])}', flush=True)
            download(os.environ['PROJECT_EXCHANGE_URL'].rstrip('/') + '/files/' + chunk['path'],
                     part, chunk['size_bytes'], chunk['sha256'],
                     token=os.environ['PROJECT_EXCHANGE_TOKEN'], ca=os.environ['PROJECT_EXCHANGE_CA_FILE'])
            parts.append(part)
        pending = target.with_name(name + '.assembling')
        if pending.is_symlink():
            raise ValueError('Symlink assembly rejected')
        with pending.open('wb') as out:
            for part in parts:
                with part.open('rb') as src:
                    while block := src.read(1024 * 1024):
                        out.write(block)
        if pending.stat().st_size != item['size_bytes'] or digest(pending) != item['sha256']:
            raise ValueError('Assembled dependency failed verification')
        os.replace(pending, target)
        for part in parts:
            part.unlink()
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--stage', choices=('numpy', 'core', 'mamba'), required=True)
    args = parser.parse_args()
    manifest = read(ROOT / 'experiments/baseline_delivery/install-mirror.json')
    if args.stage == 'numpy':
        manifest = dict(manifest, objects=[item for item in manifest['objects']
                                          if item['filename'].startswith('numpy-')])
    cache = fetch(args.cache, manifest)
    env = dict(os.environ, PIP_NO_INDEX='1', PIP_FIND_LINKS=str(cache.resolve()),
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_BUILD_ISOLATION='1',
               MAMBA_FORCE_BUILD='TRUE')
    packages = {'numpy': ['numpy==2.2.6'],
                'core': ['torch==2.7.1+cu126', 'numpy==2.2.6', 'einops==0.8.1'],
                'mamba': ['packaging', 'ninja', 'setuptools==80.9.0', 'wheel', 'transformers==4.51.3']}[args.stage]
    pip = [sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(cache),
           '-c', str(ROOT / 'requirements-baseline-install.txt')]
    if args.stage == 'core':
        # Old CUDA 13 packages may have overwritten files in the same nvidia/*
        # namespace while CUDA 12 dist-info still reports the expected version.
        # Rewrite all pinned CUDA libraries, not just their metadata dependencies.
        cuda_wheels = [str(cache / item['filename']) for item in manifest['objects']
                       if item['filename'].startswith('nvidia_') and item['filename'].endswith('.whl')]
        if not cuda_wheels:
            raise ValueError('Mirror contains no pinned NVIDIA runtime wheels')
        subprocess.run(pip + ['--force-reinstall', '--no-deps'] + cuda_wheels, env=env, check=True)
    subprocess.run(pip + packages, env=env, check=True)
    if args.stage == 'mamba':
        subprocess.run(pip + ['--no-build-isolation', '--no-deps',
                             str(cache / 'mamba-2.2.4.tar.gz')], env=env, check=True)


if __name__ == '__main__':
    main()
