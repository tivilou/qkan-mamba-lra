"""Publish explicitly downloaded dependency artifacts; credentials remain external."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.exchange_artifacts import Transport, digest_file, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--python-tag', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    from scripts.exchange_artifacts import safe_path
    prefix = safe_path('qkan-mamba/install-dependencies/' + args.run_id)
    client = Transport(os.environ['PROJECT_EXCHANGE_URL'], os.environ['PROJECT_EXCHANGE_TOKEN'],
                       os.environ['PROJECT_EXCHANGE_CA_FILE'])
    body = json.dumps({'path': prefix}).encode()
    client.request('POST', '/api/directories', body, len(body))
    records = []
    files = sorted(args.directory.glob('*.whl')) + [args.directory / 'mamba-2.2.4.tar.gz']
    for path in files:
        size, sha = digest_file(path)
        record = dict(filename=path.name, size_bytes=size, sha256=sha, chunks=[])
        if path.name == 'mamba-2.2.4.tar.gz':
            record.update(upstream_commit='95d8aba8a8c75aedcaa6143713b11e745e7cd0d9',
                          source_url='https://codeload.github.com/state-spaces/mamba/tar.gz/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9')
        else:
            record['source_indexes'] = ['https://download.pytorch.org/whl/cu126', 'https://pypi.org/simple']
        with path.open('rb') as src:
            index = 0
            while block := src.read(64 * 1024 * 1024):
                with tempfile.NamedTemporaryFile(dir=args.directory, delete=False) as tmp:
                    tmp.write(block)
                    temporary = Path(tmp.name)
                try:
                    chunk_size, chunk_sha = digest_file(temporary)
                    remote = prefix + '/' + hashlib.sha256(path.name.encode()).hexdigest()[:24] + f'.chunk{index:03d}'
                    for attempt in range(3):
                        try:
                            _, receipt = client.request('PUT', '/upload/' + remote, temporary, chunk_size)
                            break
                        except (OSError, TimeoutError):
                            if attempt == 2:
                                raise
                            time.sleep(2)
                    assert (receipt['path'], receipt['size_bytes'], receipt['sha256']) == (remote, chunk_size, chunk_sha)
                    record['chunks'].append(dict(path=remote, size_bytes=chunk_size, sha256=chunk_sha))
                finally:
                    temporary.unlink()
                index += 1
        records.append(record)
        print('Published and receipt-verified: ' + path.name, flush=True)
    atomic_json(args.manifest, dict(schema='qkan-install-mirror.v1', python_tag=args.python_tag,
                                   platform='linux-x86_64', objects=records))


if __name__ == '__main__':
    main()
