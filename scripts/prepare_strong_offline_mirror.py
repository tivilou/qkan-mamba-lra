"""Owner-side asset preparation only. Download/publish; never install/train.

Run on an internet-connected Linux x86_64 Python3.10 staging host, not on the
collaborator. Credentials are environment variables, outside source/manifests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import stat
import urllib.request
import zipfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def download(url,path,expected=None):
    if path.exists():
        if expected and sha(path)!=expected: raise ValueError('Existing asset hash differs')
        return
    temp=path.with_name(path.name+'.pending')
    with urllib.request.urlopen(url,timeout=90) as src,temp.open('wb') as dst:
        while block:=src.read(1024*1024): dst.write(block)
    if expected and sha(temp)!=expected: raise ValueError('Downloaded asset hash differs')
    temp.replace(path)


def prepare(args):
    if sys.version_info[:2]!=(3,10) or sys.platform!='linux':
        raise ValueError('Resolve actual Linux Python3.10 wheel tags/markers on staging host')
    folder=args.directory.resolve();folder.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://pypi.org/pypi/pip/23.3.2/json',timeout=30) as f: metadata=json.load(f)
    item=next(x for x in metadata['urls'] if x['filename'].endswith('.whl'))
    pip=folder/item['filename'];download(item['url'],pip,item['digests']['sha256'])
    # ANTLR 4.8 predates binary wheels. Build its pure-Python distribution as
    # an owner asset using downloaded build tools on PYTHONPATH, without any
    # package installation or collaborator-side access to an index.
    build_tools=[pip]
    for package,version in [('setuptools','68.2.2'),('wheel','0.41.3')]:
        with urllib.request.urlopen('https://pypi.org/pypi/'+package+'/'+version+'/json',timeout=30) as f: meta=json.load(f)
        artifact=next(x for x in meta['urls'] if x['filename'].endswith('.whl'))
        path=folder/artifact['filename'];download(artifact['url'],path,artifact['digests']['sha256']);build_tools.append(path)
    registry=json.loads((args.inputs/'strong-source-registry.json').read_text())
    for model,record in registry['models'].items():
        if args.supplement_model and model!=args.supplement_model: continue
        wheels=folder/model;wheels.mkdir(exist_ok=True)
        env={k:v for k,v in os.environ.items() if not k.startswith('PIP_')}
        env.update(PYTHONPATH=str(pip),PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1')
        if model=='mega' and not list(wheels.glob('antlr4_python3_runtime-4.8-*.whl')):
            with urllib.request.urlopen('https://pypi.org/pypi/antlr4-python3-runtime/4.8/json',timeout=30) as f: meta=json.load(f)
            artifact=next(x for x in meta['urls'] if x['packagetype']=='sdist')
            sdist=folder/artifact['filename'];download(artifact['url'],sdist,artifact['digests']['sha256'])
            build_env=dict(env,PYTHONPATH=os.pathsep.join(map(str,build_tools)))
            subprocess.run([sys.executable,'-B','-m','pip','wheel','--no-index','--no-deps',
                '--no-build-isolation','--wheel-dir',str(wheels),str(sdist)],env=build_env,check=True)
        req=args.inputs/('requirements-'+model+'.txt')
        command=[sys.executable,'-B','-m','pip','download','--only-binary=:all:',
            '--index-url','https://pypi.org/simple','--find-links',str(wheels),'--dest',str(wheels),'-r',str(req),
            'pip==23.3.2','setuptools==68.2.2','wheel==0.41.3']
        if model=='mega': command+=['--extra-index-url','https://download.pytorch.org/whl/cu117']
        else: command+=['--find-links','https://storage.googleapis.com/jax-releases/jax_cuda_releases.html']
        subprocess.run(command,env=env,check=True)
        repository=record['repository'].removeprefix('https://github.com/')
        source=folder/(model+'-source.zip')
        download('https://codeload.github.com/'+repository+'/zip/'+record['revision'],source)
        members={}
        with zipfile.ZipFile(source) as archive:
            roots={x.filename.split('/')[0] for x in archive.infolist()}
            if len(roots)!=1: raise ValueError('Unexpected source archive root')
            for info in archive.infolist():
                if info.is_dir(): continue
                relative=info.filename.split('/',1)[1]
                if not relative or '..' in Path(relative).parts or Path(relative).is_absolute() or stat.S_ISLNK(info.external_attr>>16): raise ValueError('Unsafe source member')
                data=archive.read(info)
                members[relative]=dict(size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        for name,expected in record['files'].items():
            if members[name]['sha256']!=expected['sha256']: raise ValueError('Official source pin differs: '+name)
        (folder/(model+'-members.json')).write_text(json.dumps(members,indent=2)+'\n')
    if not args.publish: return
    sys.path.insert(0,str(args.project_root))
    from scripts.exchange_artifacts import Transport
    from scripts.exchange_artifacts import safe_path
    prefix=safe_path('qkan-mamba/install-dependencies/'+args.run_id)
    transport=Transport(os.environ['PROJECT_EXCHANGE_URL'],os.environ['PROJECT_EXCHANGE_TOKEN'],os.environ['PROJECT_EXCHANGE_CA_FILE'])
    body=json.dumps({'path':prefix}).encode();transport.request('POST','/api/directories',body,len(body))
    previous=json.loads(args.manifest.read_text()) if args.supplement_model else None
    reusable={x['sha256']:x for e in previous['models'].values() for x in [*e['wheels'],e['source']]} if previous else {}
    def publish(path):
        record=dict(filename=path.name,size_bytes=path.stat().st_size,sha256=sha(path),chunks=[])
        old=reusable.get(record['sha256'])
        if old and old['filename']==path.name and old['size_bytes']==record['size_bytes']:
            return {key:old[key] for key in record}
        with path.open('rb') as source:
            while block:=source.read(64*1024*1024):
                with tempfile.NamedTemporaryFile(dir=folder,delete=False) as f:f.write(block);part=Path(f.name)
                try:
                    chunk_sha=sha(part);remote=prefix+'/'+chunk_sha+'.chunk'
                    _,receipt=transport.request('PUT','/upload/'+remote,part,len(block))
                    if receipt['sha256']!=chunk_sha or receipt['size_bytes']!=len(block): raise ValueError('Upload receipt mismatch')
                    record['chunks'].append(dict(path=remote,size_bytes=len(block),sha256=chunk_sha))
                finally: part.unlink()
        print('Receipt verified: '+path.name,flush=True)
        return record
    manifest=previous or dict(schema='qkan-strong-offline-mirror.v1',status='published',python_tag='cp310',platform='linux-x86_64',models={})
    for model,source in registry['models'].items():
        if args.supplement_model and model!=args.supplement_model: continue
        wheel_records=[publish(p) for p in sorted((folder/model).glob('*.whl'))]
        manifest['models'][model]=dict(requirements_sha256=sha(args.inputs/('requirements-'+model+'.txt')),
            wheels=wheel_records,source=dict(**publish(folder/(model+'-source.zip')),revision=source['revision'],
            members=json.loads((folder/(model+'-members.json')).read_text())))
    args.manifest.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--publish',action='store_true');p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--run-id',default='strong-cp310-20261104-v1');p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--supplement-model',choices=('mega','s5'),help='Extend an existing published manifest; preserve other models and build headers')
    prepare(p.parse_args())
