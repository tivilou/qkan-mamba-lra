"""Transport import compatibility wrapper; no training/config/state changes.

Keep the worker's exact pinned module. Cache only the project's exchange-client
submodule before upstream imports its unrelated top-level `scripts` package.
Never replace that parent package or a trainer/upstream module.
"""
import argparse
import importlib.util
from pathlib import Path
import runpy
import sys

TARGET='experiments.baseline_delivery.mega_formal'


def preload(root):
    name='scripts.exchange_artifacts';path=root/'scripts/exchange_artifacts.py'
    existing=sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve()!=path.resolve():
            raise ValueError('Conflicting artifact transport client')
        return existing
    spec=importlib.util.spec_from_file_location(name,path)
    client=importlib.util.module_from_spec(spec)
    sys.modules[name]=client
    try:spec.loader.exec_module(client)
    except BaseException:
        sys.modules.pop(name,None);raise
    return client


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('worker_args',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv);root=args.root.resolve()
    tail=args.worker_args[1:] if args.worker_args[:1]==['--'] else args.worker_args
    if len(tail)!=3 or tail[:2]!=['--worker','--suite']:parser.error('Bound MEGA worker and Suite required')
    # The unchanged native worker validates the existing record's code revision,
    # implementation hashes, full-state lineage and original certificate.
    sys.path.insert(0,str(root))
    preload(root)
    previous=sys.argv
    try:
        sys.argv=[TARGET,*tail]
        runpy.run_module(TARGET,run_name='__main__',alter_sys=True)
    finally:sys.argv=previous


if __name__=='__main__':main()
