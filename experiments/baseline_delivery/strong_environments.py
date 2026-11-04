"""Collaborator-only isolated environment setup, with immutable source checks."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from .common import ROOT, atomic_json, digest, read
from .strong_sources import verify_checkout


def run(*argv): subprocess.run(list(map(str,argv)),cwd=ROOT,check=True)


def prepare():
    registry=read(ROOT/'experiments/baseline_delivery/strong-source-registry.json')
    python=shutil.which('python3.10')
    if not python:
        conda=shutil.which('conda') or os.environ.get('CONDA_EXE')
        if not conda: raise ValueError('Need python3.10 or existing conda to isolate official fairseq; no current environment is modified')
        prefix=ROOT/'.bootstrap/python310'
        if not (prefix/'bin/python').exists(): run(conda,'create','-y','-p',prefix,'python=3.10.14')
        python=str(prefix/'bin/python')
    for model,record in registry['models'].items():
        source=ROOT/'.bootstrap/strong-upstream'/model
        if not source.exists():
            source.parent.mkdir(parents=True,exist_ok=True)
            run('git','clone','--no-checkout',record['repository'],source)
            run('git','-C',source,'checkout','--detach',record['revision'])
        verify_checkout(model,source)
        env=ROOT/('.venv-'+model);executable=env/'bin/python'
        if not executable.exists(): run(python,'-m','venv',env)
        receipt=env/'qkan-install.json'
        identity=dict(requirements_sha256=digest(ROOT/('requirements-'+model+'.txt')),
                      source_revision=record['revision'],recipe_sha256=digest(Path(__file__)))
        if receipt.exists() and read(receipt).get('identity')==identity:
            # Probe the installed version consistency on every reuse.
            run(executable,'-m','pip','check')
            freeze=subprocess.check_output([str(executable),'-m','pip','freeze'],text=True).splitlines()
            if freeze!=read(receipt)['freeze']: raise ValueError('Isolated environment changed; preserve it for diagnosis')
            continue
        run(executable,'-m','pip','install','pip==23.3.2','setuptools==68.2.2','wheel==0.41.3')
        args=['--extra-index-url','https://download.pytorch.org/whl/cu117'] if model=='mega' else ['--find-links','https://storage.googleapis.com/jax-releases/jax_cuda_releases.html']
        run(executable,'-m','pip','install',*args,'-r',ROOT/('requirements-'+model+'.txt'))
        if model=='mega':
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
        verify_checkout(model,source)
        run(executable,'-m','pip','check')
        freeze=subprocess.check_output([str(executable),'-m','pip','freeze'],text=True)
        atomic_json(receipt,dict(identity=identity,freeze=freeze.splitlines()))


if __name__=='__main__': prepare()
