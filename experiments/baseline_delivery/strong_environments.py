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


def usable_python(executable, *, prefix=None, require_pip=True):
    """Probe stdlib/pip only; an interpreter file is not a complete venv."""
    script='import sys, venv; assert sys.version_info[:2]==(3,10)'
    if require_pip: script+='; import pip'
    if prefix is not None:
        script+='; from pathlib import Path; assert Path(sys.prefix).resolve()==Path(sys.argv[1]).resolve()'
    try:
        result=subprocess.run([str(executable),'-I','-c',script,*([str(prefix)] if prefix else [])],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return result.returncode==0


def python310():
    # Known collaborator has /usr/bin/python3.10. No Conda, apt or external
    # runtime bootstrap is allowed here. A missing runtime is a prerequisite.
    candidates=[shutil.which('python3.10'),str(ROOT/'.bootstrap/python310/bin/python')]
    for python in candidates:
        if python and usable_python(python,require_pip=False): return python
    raise ValueError('Local Python3.10 with stdlib venv is required; no external fallback is permitted')


def ensure_environment(python, env, wheelhouse):
    executable=env/'bin/python'
    if executable.exists() and usable_python(executable,prefix=env): return executable
    if env.exists():
        # Preserve failed venv creation (which may leave bin/python but no pip).
        # Only rename our exact generated directory, never a symlink or parent.
        if env.is_symlink() or env.resolve().parent!=ROOT.resolve() or env.name not in ('.venv-mega','.venv-s5'):
            raise ValueError('Refusing to replace an unexpected environment path')
        env.rename(env.with_name(env.name+'.incomplete-'+uuid.uuid4().hex[:8]))
    run(python,'-m','venv','--without-pip',env)
    from .strong_offline import pip_environment
    bootstrap=wheelhouse/'pip-23.3.2-py3-none-any.whl'
    if not bootstrap.is_file(): raise ValueError('Offline pip bootstrap wheel is missing')
    install_env=pip_environment(wheelhouse)
    install_env['PYTHONPATH']=str(bootstrap)
    # Execute the verified pip wheel as a zipimport module, then force an actual
    # local installation rather than treating that bootstrap wheel as installed.
    subprocess.run([str(executable),'-B','-m','pip','install','--no-index',
        '--no-deps','--ignore-installed',str(bootstrap)],env=install_env,cwd=ROOT,check=True)
    if not usable_python(executable,prefix=env): raise ValueError('New environment has no usable Python3.10/pip')
    return executable


def prepare():
    registry=read(ROOT/'experiments/baseline_delivery/strong-source-registry.json')
    from .strong_offline import manifest, wheels, source as offline_source, pip_environment
    offline=manifest()
    print('QKAN_SETUP_STAGE bootstrap_python',flush=True)
    python=python310()
    for model,record in registry['models'].items():
        print('QKAN_SETUP_STAGE '+model+':source',flush=True)
        source=offline_source(model,offline['models'][model])
        wheelhouse=wheels(model,offline['models'][model])
        install_env=pip_environment(wheelhouse)
        def pip(*args):
            subprocess.run([str(executable),'-m','pip',*map(str,args)],cwd=ROOT,env=install_env,check=True)
        print('QKAN_SETUP_STAGE '+model+':environment',flush=True)
        env=ROOT/('.venv-'+model);executable=ensure_environment(python,env,wheelhouse)
        receipt=env/'qkan-install.json'
        identity=dict(requirements_sha256=digest(ROOT/('requirements-'+model+'.txt')),
                      source_revision=record['revision'],recipe_sha256=digest(Path(__file__)),
                      mirror_sha256=digest(ROOT/'experiments/baseline_delivery/strong-offline-mirror.json'))
        if receipt.exists() and read(receipt).get('identity')==identity:
            # Probe the installed version consistency on every reuse.
            pip('check')
            freeze=subprocess.check_output([str(executable),'-m','pip','freeze'],text=True).splitlines()
            if freeze!=read(receipt)['freeze']: raise ValueError('Isolated environment changed; preserve it for diagnosis')
            continue
        print('QKAN_SETUP_STAGE '+model+':dependencies',flush=True)
        pip('install','--no-index','--find-links',wheelhouse,'pip==23.3.2','setuptools==68.2.2','wheel==0.41.3')
        pip('install','--no-index','--find-links',wheelhouse,'-r',ROOT/('requirements-'+model+'.txt'))
        if model=='mega':
            print('QKAN_SETUP_STAGE mega:build',flush=True)
            # Cython generates .cpp beside .pyx: build in a disposable copy so
            # the audited checkout stays clean. Keep native extensions only.
            build=ROOT/'.bootstrap/mega-build'/record['revision']
            if not build.exists(): shutil.copytree(source,build,ignore=shutil.ignore_patterns('.git','__pycache__'))
            # libnat_cuda is an unrelated edit-distance extension; upstream
            # enables it solely when CUDA_HOME is exported. Its CUDA 11 build
            # must not accidentally consume the existing Mamba CUDA 12 toolkit.
            build_env={k:v for k,v in install_env.items() if k!='CUDA_HOME'}
            build_env['MAX_JOBS']='1'
            build_env['PYTHONDONTWRITEBYTECODE']='1'
            subprocess.run([str(executable),'-m','pip','install','--no-deps',
                '--no-build-isolation','--no-index','--find-links',str(wheelhouse),str(build)],cwd=ROOT,env=build_env,check=True)
            package=Path(subprocess.check_output([str(executable),'-c',
                'import sysconfig; print(sysconfig.get_paths()["purelib"])'],text=True).strip())/'fairseq'
            for extension in package.rglob('*.so'):
                target=source/'fairseq'/extension.relative_to(package)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(extension,target)
        # Build products may be ignored by upstream; source identities still checked.
        print('QKAN_SETUP_STAGE '+model+':verify',flush=True)
        verify_checkout(model,source)
        pip('check')
        freeze=subprocess.check_output([str(executable),'-m','pip','freeze'],text=True)
        atomic_json(receipt,dict(identity=identity,freeze=freeze.splitlines()))


if __name__=='__main__': prepare()
