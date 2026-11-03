"""Return explicitly declared artifacts with resumable hash-verified receipts."""
import argparse
import http.client
import os
from pathlib import Path
import time

from .bootstrap import check_exchange
from .common import ROOT, lock, read
from scripts.exchange_artifacts import Transport, atomic_json, prepare, deliver, safe_path


def upload_declared(directory, experiment, files, run_id=None):
    directory=Path(directory).absolute()
    relative=directory.relative_to(ROOT).as_posix()
    safe_path(relative)
    run_id=run_id or directory.name
    target=safe_path('qkan-mamba/'+experiment+'/'+run_id)
    if len(target.split('/'))!=3:
        raise ValueError('Experiment and run ID must be single path segments')
    for name in files:
        if '/' in name or '\\' in name: raise ValueError('Declare artifact basenames explicitly')
        safe_path(name)
    if not files or len(files)!=len(set(files)):
        raise ValueError('Declare a nonempty unique artifact list')
    manifest_path=directory/'artifact-delivery.json'
    with lock(directory/'artifact-delivery.lock'):
        mappings=[relative+'/'+name+'='+name for name in files]
        if not manifest_path.exists():
            atomic_json(manifest_path,prepare(ROOT,experiment,run_id,mappings))
        manifest=read(manifest_path)
        if (manifest['target_dir']!=target or
                {r['local_path']+'='+r['path'] for r in manifest['files']}!=set(mappings)):
            raise ValueError('Existing immutable delivery declares different artifacts or destination')
        # Persist the pending inventory even when configuration is unavailable.
        check_exchange()
        transport=Transport(os.environ['PROJECT_EXCHANGE_URL'],os.environ['PROJECT_EXCHANGE_TOKEN'],
                            os.environ['PROJECT_EXCHANGE_CA_FILE'])
        for attempt in range(3):
            try:
                result=deliver(ROOT,manifest_path,transport)
                print('Artifacts uploaded and SHA-256 receipts verified: '+result['target_dir'],flush=True)
                return result
            except (OSError,http.client.HTTPException):
                if attempt==2: raise
                time.sleep(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--experiment',default='recovery-audit')
    parser.add_argument('--file',action='append',dest='files')
    args=parser.parse_args()
    try:
        upload_declared(args.directory,args.experiment,args.files or ['report.json','summary.json'])
    except Exception as error:
        # Do not print credential-bearing network exception details.
        parser.exit(2,'Artifact return failed ('+type(error).__name__+'); local files and artifact-delivery.json are retained. Retry this same upload command; no training rerun required.\n')


if __name__=='__main__': main()
