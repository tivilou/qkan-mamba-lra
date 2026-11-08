"""Launch an admitted suite using its unchanged executor and recorded configs.

May be fetched to .bootstrap alongside formal_progress.py. No preparation,
backend selection, migration or configuration rewrite belongs in this entry.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


def run(root,suite):
    root=Path(root).resolve();suite=Path(suite).resolve()
    if not suite.is_relative_to(root/'runs'): raise ValueError('Suite must be under executor runs/')
    sys.path.insert(0,str(root))
    from experiments.baseline_delivery.common import read,digest,fingerprint,lock,revision,child_spec,run_child
    from experiments.baseline_delivery.readiness import validate_runtime_release
    from experiments.baseline_delivery.runner import execute,discover_gpus
    from experiments.baseline_delivery.sync import verify_data
    from experiments.baseline_delivery.bootstrap import check_exchange
    from experiments.baseline_delivery.migration import admit,plan_for
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():
        raise ValueError('Executor tracked files changed')
    with lock(suite/'suite.lock'):
        context=read(suite/'context.json');manifest=read(suite/'suite.json');gate=read(suite/'gate-report.json')
        source=revision()
        if (gate.get('status')!='passed' or any(value.get('source_revision')!=source
                for value in (context,manifest,gate))): raise ValueError('Suite not admitted for this executor')
        if gate.get('evidence_sha256')!=digest(suite/'gate-evidence.json'):
            raise ValueError('Admission evidence changed')
        evidence=read(suite/'gate-evidence.json')
        if evidence.get('context')!=context: raise ValueError('Admission context changed')
        os.environ['QKAN_RUNTIME_RELEASE']=str(suite/'runtime-release.json')
        validate_runtime_release(read(suite/'runtime-release.json'),None)
        check_exchange()
        data=Path(manifest['data']);cache=Path(context['cache'])
        verify_data(data)
        gpus,inventory=discover_gpus(context['gpus'])
        if inventory!=manifest['gpu_inventory']: raise ValueError('Admitted GPU identity changed')
        configs={}
        for job in manifest['jobs']:
            config=read(suite/'jobs'/job['id']/'config.json')
            if fingerprint(config)!=job['config_sha256']: raise ValueError('Admitted job config changed')
            configs[job['id']]=config
        # Existing receipts are validated; none are regenerated if present.
        plan=plan_for(suite)
        if plan:
            for name,record in plan['jobs'].items():
                if record.get('checkpoint') and not (suite/'jobs'/name/'migration-validation.json').is_file():
                    raise ValueError('Admitted migration receipt missing')
        admit(suite,configs,data,cache,gpus,run_child,child_spec,inventory)
        print('Using admitted configurations; preserving backend selections and completed results.',flush=True)
        execute(suite,gpus,cache,data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--resume',type=Path,required=True)
    args=parser.parse_args()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',QKAN_MEGA_SOURCE=str(args.root/'.bootstrap/strong-upstream/mega'),
        QKAN_S5_SOURCE=str(args.root/'.bootstrap/strong-upstream/s5'))
    def pause(*_): raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,pause)
    if hasattr(signal,'SIGHUP'): signal.signal(signal.SIGHUP,pause)
    try: run(args.root,args.resume)
    except KeyboardInterrupt: return 75
    except Exception as error:
        # Keep values/logs private; fixed validation errors and type suffice.
        print('Admitted entry stopped: '+(str(error) if isinstance(error,ValueError) else type(error).__name__),flush=True)
        return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
