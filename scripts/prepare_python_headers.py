"""Owner-only: mirror Ubuntu CPython3.10 development headers, without installing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

PACKAGE='libpython3.10-dev_3.10.12-1~22.04.18_amd64.deb'
PACKAGE_SHA='c058675011ec4b64f5ad803d74b5fe7de0c720c3700683040da68d1fb5b44ed9'
URL='https://security.ubuntu.com/ubuntu/pool/main/p/python3.10/'+PACKAGE

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(args):
    args.directory.mkdir(parents=True,exist_ok=True)
    package=args.directory/PACKAGE
    if not package.exists():
        with urllib.request.urlopen(URL,timeout=90) as source: package.write_bytes(source.read())
    if digest(package)!=PACKAGE_SHA: raise ValueError('Ubuntu development package hash differs')
    with tempfile.TemporaryDirectory(dir=args.directory) as temp:
        subprocess.run(['dpkg-deb','-x',str(package),temp],check=True)
        include=Path(temp)/'usr/include'
        archive=args.directory/'python310-ubuntu-amd64-headers.zip'
        members={}
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as out:
            for path in sorted(include.rglob('*')):
                if not path.is_file(): continue
                if path.is_symlink(): raise ValueError('Unexpected header symlink')
                name=path.relative_to(include).as_posix();data=path.read_bytes()
                info=zipfile.ZipInfo(name,date_time=(2026,11,4,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                out.writestr(info,data)
                members[name]=dict(size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    if not {'python3.10/Python.h','python3.10/pyconfig.h','x86_64-linux-gnu/python3.10/pyconfig.h'}.issubset(members):
        raise ValueError('Incomplete CPython header closure')
    sys.path.insert(0,str(args.project_root))
    from scripts.exchange_artifacts import Transport
    client=Transport(os.environ['PROJECT_EXCHANGE_URL'],os.environ['PROJECT_EXCHANGE_TOKEN'],os.environ['PROJECT_EXCHANGE_CA_FILE'])
    sha=digest(archive);remote='qkan-mamba/install-dependencies/strong-cp310-20261104-v1/'+sha+'.chunk'
    _,receipt=client.request('PUT','/upload/'+remote,archive,archive.stat().st_size)
    if receipt['sha256']!=sha or receipt['size_bytes']!=archive.stat().st_size: raise ValueError('Header receipt differs')
    request=urllib.request.Request(os.environ['PROJECT_EXCHANGE_URL']+'/files/'+remote,headers={'Authorization':'Bearer '+os.environ['PROJECT_EXCHANGE_TOKEN']})
    import ssl
    with urllib.request.urlopen(request,context=ssl.create_default_context(cafile=os.environ['PROJECT_EXCHANGE_CA_FILE']),timeout=120) as response:
        if hashlib.sha256(response.read()).hexdigest()!=sha: raise ValueError('Header readback differs')
    value=json.loads(args.manifest.read_text())
    value['build_headers']=dict(filename=archive.name,size_bytes=archive.stat().st_size,sha256=sha,
        chunks=[dict(path=remote,size_bytes=archive.stat().st_size,sha256=sha)],members=members,
        provenance=dict(url=URL,package_sha256=PACKAGE_SHA,scope='Unmodified Ubuntu amd64 CPython3.10 headers only; no package installation'))
    args.manifest.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='published_and_read_back',sha256=sha,bytes=archive.stat().st_size,members=len(members))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','manifest','project-root'): p.add_argument('--'+name,type=Path,required=True)
    prepare(p.parse_args())
