"""Fetch only declared authenticated exchange objects; no external fallback."""
import os
from pathlib import Path
import shutil
import stat
import uuid
import zipfile

from .common import ROOT, atomic_json, digest, read
from .install_mirror import fetch


MANIFEST=ROOT/'experiments/baseline_delivery/strong-offline-mirror.json'


def manifest():
    if not MANIFEST.is_file(): raise ValueError('Strong offline assets are not published; collaborator must not fetch external dependencies')
    value=read(MANIFEST)
    if (value.get('schema')!='qkan-strong-offline-mirror.v1' or value.get('status')!='published'
        or value.get('python_tag')!='cp310' or value.get('platform')!='linux-x86_64'
        or set(value.get('models',{}))!={'mega','s5'}):
        raise ValueError('Strong offline asset publication is incomplete')
    for model,entry in value['models'].items():
        if digest(ROOT/('requirements-'+model+'.txt'))!=entry['requirements_sha256']:
            raise ValueError('Offline requirements changed; owner must republish the complete dependency closure')
    return value


def wheels(model,record):
    return fetch(ROOT/'.bootstrap/strong-offline-cache'/model,
        dict(python_tag='cp310',objects=record['wheels']),target_python_tag='cp310')


def source(model,record):
    from .strong_sources import verify_checkout
    destination=ROOT/'.bootstrap/strong-upstream'/model
    if destination.exists(): verify_checkout(model,destination);return destination
    archive_dir=fetch(ROOT/'.bootstrap/strong-offline-cache/sources',
        dict(python_tag='cp310',objects=[record['source']]),target_python_tag='cp310')
    archive_path=archive_dir/record['source']['filename']
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.extracting-'+uuid.uuid4().hex[:8]);temporary.mkdir()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            files=[x for x in archive.infolist() if not x.is_dir()]
            names=[x.filename.split('/',1)[1] for x in files]
            if len(set(names))!=len(names) or set(names)!=set(record['source']['members']):
                raise ValueError('Source archive member inventory differs')
            for info,name in zip(files,names):
                if Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name or stat.S_ISLNK(info.external_attr>>16):
                    raise ValueError('Unsafe source member')
                target=temporary/name;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(info) as src,target.open('xb') as out: shutil.copyfileobj(src,out)
                item=record['source']['members'][name]
                if target.stat().st_size!=item['size_bytes'] or digest(target)!=item['sha256']:
                    raise ValueError('Source member failed identity check')
        atomic_json(temporary/'.qkan-source-snapshot.json',dict(model=model,
            revision=record['source']['revision'],archive_sha256=record['source']['sha256']))
        temporary.rename(destination)
    except BaseException:
        # Preserve failed extraction for inspection; never accept partial source.
        raise
    verify_checkout(model,destination)
    return destination


def pip_environment(wheelhouse):
    env=dict(os.environ)
    env.pop('PYTHONPATH',None)
    for key in tuple(env):
        if key.startswith('PIP_'): env.pop(key)
    env.update(PYTHONDONTWRITEBYTECODE='1',PIP_CONFIG_FILE=os.devnull,PIP_NO_INDEX='1',PIP_FIND_LINKS=str(wheelhouse),
        PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_BUILD_ISOLATION='1')
    return env
