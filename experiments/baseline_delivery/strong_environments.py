"""Collaborator-only isolated environment setup, with immutable source checks."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from .common import ROOT, atomic_json, digest, read
from .strong_sources import verify_checkout


def run(*argv): subprocess.run(list(map(str,argv)),cwd=ROOT,check=True)


def usable_python(executable, *, prefix=None, ensurepip=False):
    """Probe stdlib/pip only; an interpreter file is not a complete venv."""
    script='import sys, pip; assert sys.version_info[:2]==(3,10)'
    if ensurepip: script+='; import ensurepip'
    if prefix is not None:
        script+='; from pathlib import Path; assert Path(sys.prefix).resolve()==Path(sys.argv[1]).resolve()'
    try:
        result=subprocess.run([str(executable),'-I','-c',script,*([str(prefix)] if prefix else [])],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return result.returncode==0


def python310():
    python=shutil.which('python3.10')
    if python and usable_python(python,ensurepip=True): return python
    conda=shutil.which('conda') or os.environ.get('CONDA_EXE')
    if not conda:
        raise ValueError('Python3.10 lacks ensurepip/pip and no conda is available for isolated bootstrap')
    prefix=ROOT/'.bootstrap/python310';executable=prefix/'bin/python'
    if not executable.exists() or not usable_python(executable,prefix=prefix,ensurepip=True):
        if prefix.exists():
            if prefix.is_symlink() or not prefix.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Refusing to replace an unexpected bootstrap path')
            prefix.rename(prefix.with_name(prefix.name+'.incomplete-'+uuid.uuid4().hex[:8]))
        # Avoid requiring apt/root or changing the active myconda environment.
        run(conda,'create','-y','-p',prefix,'--override-channels','-c','conda-forge',
            'python=3.10.14','pip=23.3.2')
    if not usable_python(executable,prefix=prefix,ensurepip=True):
        raise ValueError('Isolated Python3.10 bootstrap lacks ensurepip/pip')
    return str(executable)


def ensure_environment(python, env):
    executable=env/'bin/python'
    if executable.exists() and usable_python(executable,prefix=env): return executable
    if env.exists():
        # Preserve failed venv creation (which may leave bin/python but no pip).
        # Only rename our exact generated directory, never a symlink or parent.
        if env.is_symlink() or env.resolve().parent!=ROOT.resolve() or env.name not in ('.venv-mega','.venv-s5'):
            raise ValueError('Refusing to replace an unexpected environment path')
        env.rename(env.with_name(env.name+'.incomplete-'+uuid.uuid4().hex[:8]))
    run(python,'-m','venv',env)
    if not usable_python(executable,prefix=env): raise ValueError('New environment has no usable Python3.10/pip')
    return executable


def prepare():
    registry=read(ROOT/'experiments/baseline_delivery/strong-source-registry.json')
    print('QKAN_SETUP_STAGE bootstrap_python',flush=True)
    python=python310()
    for model,record in registry['models'].items():
        print('QKAN_SETUP_STAGE '+model+':source',flush=True)
        source=ROOT/'.bootstrap/strong-upstream'/model
        if not source.exists():
            source.parent.mkdir(parents=True,exist_ok=True)
            run('git','clone','--no-checkout',record['repository'],source)
            run('git','-C',source,'checkout','--detach',record['revision'])
        verify_checkout(model,source)
        print('QKAN_SETUP_STAGE '+model+':environment',flush=True)
        env=ROOT/('.venv-'+model);executable=ensure_environment(python,env)
        receipt=env/'qkan-install.json'
        identity=dict(requirements_sha256=digest(ROOT/('requirements-'+model+'.txt')),
                      source_revision=record['revision'],recipe_sha256=digest(Path(__file__)))
        if receipt.exists() and read(receipt).get('identity')==identity:
            # Probe the installed version consistency on every reuse.
            run(executable,'-m','pip','check')
            freeze=subprocess.check_output([str(executable),'-m','pip','freeze'],text=True).splitlines()
            if freeze!=read(receipt)['freeze']: raise ValueError('Isolated environment changed; preserve it for diagnosis')
            continue
        print('QKAN_SETUP_STAGE '+model+':dependencies',flush=True)
        run(executable,'-m','pip','install','pip==23.3.2','setuptools==68.2.2','wheel==0.41.3')
        args=['--extra-index-url','https://download.pytorch.org/whl/cu117'] if model=='mega' else ['--find-links','https://storage.googleapis.com/jax-releases/jax_cuda_releases.html']
        run(executable,'-m','pip','install',*args,'-r',ROOT/('requirements-'+model+'.txt'))
        if model=='mega':
            print('QKAN_SETUP_STAGE mega:build',flush=True)
            # Cython generates .cpp beside .pyx: build in a disposable copy so
            # the audited checkout stays clean. Keep native extensions only.
            build=ROOT/'.bootstrap/mega-build'/record['revision']
            if not build.exists(): shutil.copytree(source,build,ignore=shutil.ignore_patterns('.git','__pycache__'))
            # libnat_cuda is an unrelated edit-distance extension; upstream
            # enables it solely when CUDA_HOME is exported. Its CUDA 11 build
            # must not accidentally consume the existing Mamba CUDA 12 toolkit.
            build_env={k:v for k,v in os.environ.items() if k!='CUDA_HOME'}
            build_env['MAX_JOBS']='1'
            subprocess.run([str(executable),'-m','pip','install','--no-deps',
                '--no-build-isolation',str(build)],cwd=ROOT,env=build_env,check=True)
            package=Path(subprocess.check_output([str(executable),'-c',
                'import sysconfig; print(sysconfig.get_paths()["purelib"])'],text=True).strip())/'fairseq'
            for extension in package.rglob('*.so'):
                target=source/'fairseq'/extension.relative_to(package)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(extension,target)
        # Build products may be ignored by upstream; source identities still checked.
        print('QKAN_SETUP_STAGE '+model+':verify',flush=True)
        verify_checkout(model,source)
        run(executable,'-m','pip','check')
        freeze=subprocess.check_output([str(executable),'-m','pip','freeze'],text=True)
        atomic_json(receipt,dict(identity=identity,freeze=freeze.splitlines()))


if __name__=='__main__': prepare()
