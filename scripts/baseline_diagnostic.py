#!/usr/bin/env python3
"""Scheduler-registered engineering canary; no benchmark score is produced."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--canary',choices=('transformer','s4d_minimal'))
    parser.add_argument('--data-check',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from experiments.baseline_delivery.common import atomic_json, revision
    source=revision();os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',QKAN_TEST_CACHE=str(args.cache))
    if args.data_check:
        from experiments.baseline_delivery.sync import verify_data
        from experiments.baseline_delivery.runner import config_for
        records=verify_data(args.data_check)
        configs=[config_for(t,m,42,args.data_check,source) for t in ('listops','text','retrieval','image','pathfinder')
                 for m in ('transformer','s4d_minimal','mamba1')]
        atomic_json(args.output/'diagnostic.json',{'passed':True,'source_revision':source,'formal':False,
            'kind':'full_split_and_dictionary_identity','verified_splits':records,'configs_constructed':len(configs)})
    elif not args.canary:
        target=args.output/'unit-tests.log'
        with target.open('w') as log:
            env={**os.environ,'QKAN_PYTHON':sys.executable}
            for command,expected in [(['bash','-n',str(ROOT/'scripts/run_baselines.sh')],0),
                    (['bash',str(ROOT/'scripts/run_baselines.sh'),'--help'],0),
                    (['bash',str(ROOT/'scripts/run_baselines.sh'),'--mode','run'],2)]:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode!=expected: raise RuntimeError('Shell entry or release gate failed')
            from experiments.baseline_delivery.common import child_spec
            spec=child_spec('--help')
            result=subprocess.run(spec['argv'],cwd=spec['cwd'],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError('The actual worker module command failed')
            result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_baseline_delivery.py','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError('Baseline diagnostics failed; see '+str(target))
        import resource
        atomic_json(args.output/'diagnostic.json',{'schema':'qkan-baseline-diagnostic.v1','passed':True,'source_revision':source,'formal':False,'kind':'unit_correctness','log':'unit-tests.log',
                   'child_peak_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'entry_and_formal_gate':True})
    else:
        import numpy as np
        from experiments.baseline_delivery.runner import canary
        data=args.output/'generated';(data/'image').mkdir(parents=True)
        rng=np.random.default_rng(23)
        for split,n in [('train',12),('val',2)]:
            np.savez_compressed(data/'image'/(split+'.npz'),inputs=rng.integers(0,256,size=(n,1024),dtype=np.uint8),labels=np.arange(n,dtype=np.int64)%10)
        config=dict(task='image',model=args.canary,width=8,layers=1,heads=2,state_size=4,
                    dropout=.1,lr=.001,weight_decay=.01,vocabulary=None,seed=42,
                    epochs=2,batch_size=4,micro_batch=2,warmup_steps=2,checkpoint_steps=1,
                    clip_norm=1.,source_revision=source,protocol_sha256='0'*64,data_hashes={})
        report=canary(config,data,args.cache,args.output/'recovery')
        atomic_json(args.output/'diagnostic.json',{**report,'source_revision':source,'formal':False,'data':'generated uint8, not LRA performance'})


if __name__=='__main__': main()
