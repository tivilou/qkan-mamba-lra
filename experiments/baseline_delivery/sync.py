"""Hash-checked preparation. Credentials are never written to artifacts."""
from __future__ import annotations

from contextlib import ExitStack
import http.client
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
from urllib.parse import urlsplit
import urllib.request
import zipfile

from .common import ROOT, atomic_json, digest, lock, read


def download(url, target, size, sha256, *, token=None, ca=None, test_http=False):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlsplit(url)
    if parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise ValueError('Download URL may not contain credentials, query or fragment')
    if parsed.scheme != 'https' and not (test_http and parsed.scheme == 'http' and parsed.hostname == '127.0.0.1'):
        raise ValueError('HTTPS required')
    if target.is_symlink() or target.with_name(target.name+'.part').is_symlink():
        raise ValueError('Symlink download target rejected')
    if target.exists():
        if target.stat().st_size != size or digest(target) != sha256:
            raise ValueError('Existing download has wrong size/hash; preserve and inspect it')
        return
    # Exchange v4 does not support byte ranges. Atomic verified objects are reused;
    # an interrupted object is fetched again, without overwriting the final target.
    partial = target.with_name(target.name+'.part')
    context = ssl.create_default_context(cafile=ca) if parsed.scheme == 'https' else None
    connection = (http.client.HTTPSConnection(parsed.hostname, parsed.port, context=context, timeout=120)
                  if context else http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=120))
    try:
        headers = {'Accept-Encoding':'identity'}
        if token:
            if any(c in token for c in '\r\n'):
                raise ValueError('Invalid exchange token')
            headers['Authorization'] = 'Bearer '+token
        connection.request('GET', parsed.path or '/', headers=headers)
        response = connection.getresponse()
        if response.status != 200 or int(response.getheader('Content-Length', '-1')) != size:
            raise ValueError('Download status/length does not match the frozen object')
        count = 0
        with partial.open('wb') as handle:
            for block in iter(lambda: response.read(1024*1024), b''):
                count += len(block)
                if count > size:
                    raise ValueError('Download exceeded expected size')
                handle.write(block)
            handle.flush(); os.fsync(handle.fileno())
        if count != size or digest(partial) != sha256:
            raise ValueError('Download failed content verification')
        os.replace(partial, target)
    finally:
        connection.close()


def asset_sync(cache):
    manifest = read(ROOT/'experiments/baseline_delivery/assets.json')
    origin = os.environ.get('PROJECT_EXCHANGE_URL','').rstrip('/')
    token = os.environ.get('PROJECT_EXCHANGE_TOKEN','')
    if not origin or not token:
        raise ValueError('Set PROJECT_EXCHANGE_URL and PROJECT_EXCHANGE_TOKEN using the private handoff')
    parsed = urlsplit(origin)
    if parsed.scheme != 'https' or parsed.path or parsed.query or parsed.username or parsed.password:
        raise ValueError('Exchange configuration must be a bare HTTPS origin')
    for item in manifest['objects']:
        target = cache/'assets'/item['filename']
        download(origin+'/files/'+item['path'], target, item['size_bytes'], item['sha256'],
                 token=token, ca=os.environ.get('PROJECT_EXCHANGE_CA_FILE'))
        out = cache/'upstream'/item['id']
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            if sorted(archive.namelist()) != sorted(item['members']):
                raise ValueError('Unexpected source archive members')
            for name, record in item['members'].items():
                if '/' in name or '\\' in name or name.startswith('.'):
                    raise ValueError('Unsafe source member')
                destination = out/name
                if destination.is_symlink():
                    raise ValueError('Symlink source destination')
                if not destination.exists():
                    with archive.open(name) as source, destination.open('xb') as sink:
                        shutil.copyfileobj(source, sink, 1024*1024)
                if destination.stat().st_size != record['size_bytes'] or digest(destination) != record['sha256']:
                    raise ValueError('Source member differs from the pinned manifest')
    return manifest


def verify_data(data):
    expected = read(ROOT/'docs/evidence/mega-document-v3-build.json')
    records = []
    for task in expected['tasks']:
        for split, item in task['splits'].items():
            path = data/task['task']/(split+'.npz')
            if not path.is_file() or path.is_symlink() or path.stat().st_size != item['size_bytes'] or digest(path) != item['sha256']:
                raise ValueError('Dataset verification failed: '+task['task']+'/'+split)
            records.append({'task':task['task'],'split':split,'sha256':item['sha256']})
    for task in ('listops','text','retrieval'):
        dictionary = data/'metadata'/(task+'-dictionary.json')
        if dictionary.is_symlink() or not dictionary.is_file():
            raise ValueError('Missing token dictionary: '+task)
    return records


def rebuild_data(cache):
    """Acquire author bytes directly; do not redistribute AAN via our exchange."""
    sys.path.insert(0,str(ROOT/'scripts'))
    raw = cache/'raw/lra.zip'
    if raw.exists():
        from scripts.rebuild_mega_lra import ARCHIVE_SHA256
        if digest(raw) != ARCHIVE_SHA256:
            raise ValueError('Existing author archive hash mismatch')
    else:
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/fetch_mega_lra.py'),
                        '--output',str(raw),'--receipt',str(cache/'raw/download.json')],check=True,cwd=ROOT)
        from scripts.rebuild_mega_lra import ARCHIVE_SHA256
        if digest(raw) != ARCHIVE_SHA256:
            raise ValueError('Author archive changed; refuse rebuilding')
    # Each stage builds into a private .building directory, then commits it.
    # Interrupted builds remain inspectable and are never silently accepted.
    stages = [cache/'data/mega-independent-v1', cache/'data/mega-independent-pairclean-v2',
              cache/'data/mega-document-disjoint-v3']
    for index, final in enumerate(stages):
        if (final/'build-manifest.json').exists():
            continue
        pending = final.with_name(final.name+'.building')
        if pending.exists():
            # Restart only the interrupted stage; retain its bytes for inspection.
            import uuid
            pending.rename(pending.with_name(pending.name+'.interrupted-'+uuid.uuid4().hex[:8]))
        final.parent.mkdir(parents=True, exist_ok=True)
        if index == 0:
            command = [sys.executable,'-B',str(ROOT/'scripts/rebuild_mega_lra.py'),
                       '--archive',str(raw),'--output',str(pending)]
            subprocess.run(command,check=True,cwd=ROOT)
        else:
            sys.path.insert(0,str(ROOT/'scripts'))
            if index == 1:
                from curate_lra_pairs import curate
                curate(stages[0],pending)
            else:
                from curate_lra_documents import curate_documents
                scratch=cache/'scratch'; scratch.mkdir(exist_ok=True)
                curate_documents(stages[1],pending,scratch)
        pending.rename(final)
    return stages[-1]


def prepare(cache, data=None):
    cache = Path(cache).resolve(); cache.mkdir(parents=True,exist_ok=True)
    with lock(cache/'prepare.lock'):
        asset_sync(cache)
        data = Path(data).resolve() if data else rebuild_data(cache)
        records = verify_data(data)
        receipt = {'schema':'qkan-baseline-preparation.v1','data':str(data),
                   'data_profile':'mega-document-disjoint-v3','verified_splits':records,
                   'redistribution':'Author archive acquired directly; no task arrays/text from our exchange.'}
        atomic_json(cache/'prepared.json',receipt)
        return data
