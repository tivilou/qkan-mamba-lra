"""Single-command collaborator preparation, preflight, execution and delivery."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from .common import ROOT, atomic_json, child_spec, digest, fingerprint, lock, read, revision, run_child, spawn, stop_child
from .sync import prepare, verify_data


def check_release(config=None):
    qualification=read(ROOT/'docs/evidence/mega-document-v3-qualification.json')
    release=read(ROOT/'experiments/baseline_delivery/release.json')
    if qualification.get('status')!='qualified' or qualification.get('formal_baseline_allowed') is not True:
        raise ValueError('Formal launch blocked: dataset qualification='+qualification.get('status','missing')+
                         '; see docs/data-use-boundaries.md. Preparation/preflight remain available.')
    if release.get('status')!='released' or release.get('cohort_frozen') is not True:
        raise ValueError('Formal launch blocked: comparison cohort and protocol are not released')
    if config:
        certificate=release.get('certificates',{}).get(config['model'])
        if not certificate:
            raise ValueError('No fidelity certificate for '+config['model'])
        record=read(ROOT/certificate['path'])
        if digest(ROOT/certificate['path'])!=certificate['sha256'] or record.get('status') not in ('exact_upstream','source_faithful_adapted','paper_faithful'):
            raise ValueError('Invalid baseline fidelity certificate')
        if record.get('protocol_sha256')!=config['protocol_sha256'] or record.get('implementation_sha256')!=implementation_hash():
            raise ValueError('Fidelity certificate does not bind this implementation/protocol')
        if config['source_revision']!=revision() or config['protocol_sha256']!=digest(ROOT/'experiments/baseline_delivery/protocol.json'):
            raise ValueError('Source/protocol changed')


def implementation_hash():
    paths=sorted((ROOT/'experiments/baseline_delivery').glob('*.py'))
    paths += [ROOT/'scripts/run_baselines.sh',ROOT/'requirements-baselines.txt',
              ROOT/'experiments/baseline_delivery/assets.json',ROOT/'tools/sample_trace_contract/validate.py']
    return fingerprint({p.relative_to(ROOT).as_posix():digest(p) for p in paths})


def discover_gpus(requested):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total',
                                    '--format=csv,noheader,nounits'],text=True)
    rows={line.split(',')[0].strip():line.strip() for line in output.splitlines() if line.strip()}
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    permitted=set(rows)
    if visible is not None:
        permitted=set()
        for value in visible.split(','):
            value=value.strip()
            for index,row in rows.items():
                if value==index or value==row.split(',')[1].strip(): permitted.add(index)
    chosen=sorted(permitted,key=int) if requested=='auto' else requested.split(',')
    if not chosen or len(chosen)!=len(set(chosen)) or any(not re.fullmatch(r'\d+',v) or v not in permitted for v in chosen):
        raise ValueError('GPU list is empty, malformed, duplicated, unavailable, or excluded by CUDA_VISIBLE_DEVICES')
    return chosen,{gpu:rows[gpu] for gpu in chosen}


def config_for(task,model,seed,data,source):
    protocol=read(ROOT/'experiments/baseline_delivery/protocol.json')
    config={**protocol['training'],**protocol['model_defaults'][model],
            'task':task,'model':model,'seed':seed,'source_revision':source,
            'protocol_sha256':digest(ROOT/'experiments/baseline_delivery/protocol.json')}
    config['vocabulary']=(len(read(Path(data)/'metadata'/(task+'-dictionary.json'))['input_symbols'])
                          if task not in ('image','pathfinder') else None)
    expected=read(ROOT/'docs/evidence/mega-document-v3-build.json')
    item=next(t for t in expected['tasks'] if t['task']==task)
    config['data_hashes']={key:value['sha256'] for key,value in item['splits'].items()}
    return config


def compare_states(left,right):
    import numpy as np
    import torch
    if isinstance(left,torch.Tensor):
        if not torch.equal(left,right): raise ValueError('Recovery tensor differs')
    elif isinstance(left,np.ndarray):
        if not np.array_equal(left,right): raise ValueError('Recovery numpy state differs')
    elif isinstance(left,dict):
        if left.keys()!=right.keys(): raise ValueError('Recovery state keys differ')
        for key in left: compare_states(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right): raise ValueError('Recovery state lengths differ')
        for a,b in zip(left,right): compare_states(a,b)
    elif left!=right:
        raise ValueError('Recovery scalar differs')


def canary(config,data,cache,output,device='cpu'):
    """Real parent -> fresh child -> checkpoint -> different child -> resume."""
    import numpy as np
    # Keep this parent free of torch until all training children have exited.
    # In particular, a CUDA context in the parent must not reserve GPU memory.
    sys.path.insert(0,str(ROOT/'scripts'))
    from audit_lra_data import ArrayStream
    from contextlib import ExitStack
    import zipfile
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    small=folder/'data';task=config['task'];(small/task).mkdir(parents=True)
    count=config['batch_size']*3
    for split in ('train','val'):
        arrays={}
        with ExitStack() as stack:
            archive=stack.enter_context(zipfile.ZipFile(Path(data)/task/(split+'.npz')))
            for name in archive.namelist():
                source=ArrayStream(archive,name,stack)
                arrays[name[:-4]]=source.read(min(source.shape[0],count if split=='train' else 2))
        np.savez_compressed(small/task/(split+'.npz'),**arrays)
    if config['vocabulary']:
        import shutil
        (small/'metadata').mkdir()
        shutil.copyfile(Path(data)/'metadata'/(task+'-dictionary.json'),small/'metadata'/(task+'-dictionary.json'))
    atomic_json(small/'synthetic-canary.json',{'bounded':True,'source_task':task,'test_included':False})
    probe={**config,'epochs':2,'data_hashes':{},'checkpoint_steps':1}
    cfg=folder/'config.json';atomic_json(cfg,probe)
    common=['--config',cfg,'--data',small,'--cache',cache,'--device',device,'--synthetic']
    for name,extra,expected in (('uninterrupted',[],0),('interrupted',['--stop-after','2'],75),('interrupted',['--resume'],0)):
        spec=child_spec(*common,'--output',folder/name,*extra)
        status=run_child(spec,folder/(name+('-resume' if extra==['--resume'] else '')+'.log'))
        if status!=expected: raise ValueError('Canary child failed; inspect '+str(folder))
    # The child owns the fingerprint. Validate the pointer and all checkpoint bytes
    # before deserializing either locally generated state.
    from .worker import load_checkpoint
    states=[]
    for name in ('uninterrupted','interrupted'):
        pointer=read(folder/name/'checkpoints/latest.json')
        states.append(load_checkpoint(folder/name,pointer['contract']))
    for key in ('model','optimizer','scheduler','rng','permutation','cursor','step','epoch',
                'best_weights','best_accuracy','best_epoch','history','update_trace'):
        compare_states(states[0][key],states[1][key])
    for name in ('uninterrupted','interrupted'):
        if not read(folder/name/'trace-validation.json')['passed']:
            raise ValueError('Trace semantic/replay gate failed')
    report={'schema':'qkan-baseline-recovery-canary.v1','passed':True,'device':device,'model':config['model'],
            'task':task,'fresh_process':True,'updates':states[0]['step'],'agreement':'exact',
            'config_sha256':fingerprint(config),'source_revision':config['source_revision'],
            'trace_semantic_and_replay':True,'test_access':False,
            'training_seconds':states[0]['training_seconds']}
    atomic_json(folder/'canary.json',report)
    return report


def return_files(path,remote,part_size=512*1024*1024):
    """Bounded-memory chunks for both selected weights and full state."""
    if path.stat().st_size<=part_size:
        return [path.relative_to(ROOT).as_posix()+'='+remote]
    import hashlib
    parts=path.parent/'return-chunks';parts.mkdir(exist_ok=True);records=[];files=[]
    with path.open('rb') as source:
        index=0
        while source.tell()<path.stat().st_size:
            part=parts/(path.name+'.'+str(index).zfill(4));temp=part.with_suffix(part.suffix+'.pending')
            value=hashlib.sha256();size=0
            with temp.open('wb') as sink:
                while size<part_size:
                    block=source.read(min(1024*1024,part_size-size))
                    if not block: break
                    sink.write(block);value.update(block);size+=len(block)
            if part.exists() and digest(part)!=value.hexdigest():
                raise ValueError('Changed immutable return chunk')
            os.replace(temp,part)
            records.append({'name':part.name,'size_bytes':size,'sha256':value.hexdigest()})
            files.append(part.relative_to(ROOT).as_posix()+'='+remote+'.'+str(index).zfill(4));index+=1
    chunk_manifest=parts/(path.name+'.chunks.json')
    atomic_json(chunk_manifest,{'original_sha256':digest(path),'original_bytes':path.stat().st_size,'parts':records})
    return files+[chunk_manifest.relative_to(ROOT).as_posix()+'='+remote+'.chunks.json']


def deliver(suite):
    """Explicit immutable whitelist. Raw inputs, traces and logs stay local."""
    from scripts.exchange_artifacts import Transport, atomic_json as write, prepare as inventory, deliver as upload
    manifest=suite/'delivery.json'
    if not (suite/'compute-complete.json').exists():
        raise ValueError('Only a fully computed formal suite can be automatically returned')
    if not manifest.exists():
        files=[]
        # This snapshot is immutable even if a later upload retry updates status.
        if not (suite/'return-status.json').exists(): atomic_json(suite/'return-status.json',read(suite/'status.json'))
        for name in ('suite.json','return-status.json','compute-complete.json'):
            files.append(str((suite/name).relative_to(ROOT))+'='+name)
        for job in sorted((suite/'jobs').iterdir()):
            if read(job/'summary.json').get('formal') is not True or not read(job/'trace-validation.json').get('passed'):
                raise ValueError('Automatic formal return excludes diagnostic or invalid jobs')
            # Selected weights plus full latest state, metrics and compact trace
            # readiness. No raw/text-bearing Case Study capture is auto-shared.
            for name in ('config.json','summary.json','environment.json','best-model.pt','trace-validation.json'):
                path=job/name
                if not path.is_file(): raise ValueError('Required return artifact missing: '+str(path))
                files.extend(return_files(path,'jobs/'+job.name+'/'+name))
            pointer=read(job/'checkpoints/latest.json')
            for name in ('latest.json',pointer['file']):
                path=job/'checkpoints'/name
                files.extend(return_files(path,'jobs/'+job.name+'/checkpoints/'+name))
        package=inventory(ROOT,'baseline-suite',suite.name,files)
        package['artifact_sender_revision']=package['source_revision']
        package['source_revision']=read(suite/'suite.json')['source_revision']
        write(manifest,package)
    transport=Transport(os.environ.get('PROJECT_EXCHANGE_URL',''),os.environ.get('PROJECT_EXCHANGE_TOKEN',''),
                        os.environ.get('PROJECT_EXCHANGE_CA_FILE'))
    result=upload(ROOT,manifest,transport)
    if result['status']!='uploaded': raise ValueError('Artifact delivery incomplete')
    atomic_json(suite/'delivery-complete.json',{'state':'delivered','files':len(result['files']),
                                              'delivery_manifest_sha256':digest(manifest)})


def execute(suite,gpus,cache,data):
    manifest=read(suite/'suite.json');jobs=manifest['jobs'];running={};pending=[]
    states=read(suite/'status.json')
    for job in jobs:
        directory=suite/'jobs'/job['id']
        check_release(read(directory/'config.json'))
        if (directory/'summary.json').exists():
            summary=read(directory/'summary.json')
            if summary['config_sha256']!=job['config_sha256'] or not read(directory/'trace-validation.json')['passed']:
                raise ValueError('Existing completion does not match this job')
            states[job['id']]={'state':'computed'}
        else: pending.append(job)
    failed=False;last_display=0.
    try:
        while pending or running:
            for gpu in gpus:
                if gpu in running or not pending or failed: continue
                job=pending.pop(0);directory=suite/'jobs'/job['id']
                args=['--config',directory/'config.json','--data',data,'--cache',cache,
                      '--output',directory,'--device','cuda:0']
                if (directory/'checkpoints/latest.json').exists(): args.append('--resume')
                spec=child_spec(*args);atomic_json(directory/'command.json',spec)
                env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
                log=(directory/'worker.log').open('ab');child=spawn(spec,log,env)
                running[gpu]=(child,log,job)
                states[job['id']]={'state':'running','physical_gpu':gpu,'pid':child.pid,'started':time.time()}
            for gpu,(child,log,job) in list(running.items()):
                code=child.poll()
                if code is None: continue
                log.close();directory=suite/'jobs'/job['id']
                valid=code==0 and (directory/'summary.json').exists() and (directory/'trace-validation.json').exists()
                states[job['id']].update(state='computed' if valid else ('paused' if code==75 else 'failed'),exit_code=code,finished=time.time())
                del running[gpu]
                if not valid: failed=True
            atomic_json(suite/'status.json',states)
            if time.monotonic()-last_display>30:
                print(json.dumps({'running':len(running),'pending':len(pending),'computed':sum(s['state']=='computed' for s in states.values()),'failed':failed}),flush=True)
                last_display=time.monotonic()
            if failed and not running: break
            time.sleep(.5)
    finally:
        for child,log,job in running.values():
            stop_child(child);log.close();states[job['id']]={'state':'paused','exit_code':child.returncode}
        atomic_json(suite/'status.json',states)
    if failed: raise ValueError('Suite paused/failed; fix the cause and resume the same suite; completed jobs are retained')
    atomic_json(suite/'compute-complete.json',{'state':'computed','jobs':len(jobs)})
    deliver(suite)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','preflight','run','upload'),default='run')
    parser.add_argument('--cache',type=Path,default=ROOT/'data/collaborator-cache')
    parser.add_argument('--data',type=Path);parser.add_argument('--gpus',default='auto')
    parser.add_argument('--models',default='transformer,s4d_minimal,mamba1')
    parser.add_argument('--tasks',default='listops,text,retrieval,image,pathfinder')
    parser.add_argument('--seeds',default='42');parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.mode=='upload':
        if not args.resume: parser.error('upload requires --resume SUITE')
        with lock(args.resume/'suite.lock'): deliver(args.resume.resolve())
        return 0
    source=revision()
    models=args.models.split(',');tasks=args.tasks.split(',');seeds=[int(v) for v in args.seeds.split(',')]
    protocol=read(ROOT/'experiments/baseline_delivery/protocol.json')
    if len(models)!=len(set(models)) or any(m not in protocol['model_defaults'] for m in models): parser.error('Unknown/duplicate model')
    if len(tasks)!=len(set(tasks)) or any(t not in protocol['tasks'] for t in tasks): parser.error('Unknown/duplicate task')
    if len(seeds)!=len(set(seeds)) or not seeds or any(v<0 or v>=2**32 for v in seeds): parser.error('Invalid/duplicate seed')
    if args.mode=='run': check_release()
    data=prepare(args.cache,args.data)
    if args.mode=='prepare':
        print('Preparation verified: '+str(data));return 0
    gpus,inventory=discover_gpus(args.gpus)
    if args.resume:
        suite=args.resume.resolve()
        existing=read(suite/'suite.json')
        if existing['source_revision']!=source or existing['protocol_sha256']!=digest(ROOT/'experiments/baseline_delivery/protocol.json'):
            raise ValueError('Resume source/protocol mismatch')
        if existing['models']!=models or existing['tasks']!=tasks or existing['seeds']!=seeds or existing['data']!=str(data):
            raise ValueError('Resume selection/data mismatch')
    else:
        suite=ROOT/'runs'/('baseline-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
        suite.mkdir(parents=True,exist_ok=False);jobs=[]
        for model in models:
            for task in tasks:
                for seed in seeds:
                    name=model+'-'+task+'-s'+str(seed);folder=suite/'jobs'/name;folder.mkdir(parents=True)
                    config=config_for(task,model,seed,data,source);atomic_json(folder/'config.json',config)
                    jobs.append({'id':name,'config_sha256':fingerprint(config)})
        atomic_json(suite/'suite.json',{'schema':'qkan-baseline-suite.v1','source_revision':source,
            'protocol_sha256':digest(ROOT/'experiments/baseline_delivery/protocol.json'),
            'models':models,'tasks':tasks,'seeds':seeds,'data':str(data),'jobs':jobs,'gpu_inventory':inventory})
        atomic_json(suite/'status.json',{job['id']:{'state':'pending'} for job in jobs})
    print('Suite: '+str(suite),flush=True)
    with lock(suite/'suite.lock'):
        # Each task/model passes recovery on real full-length inputs before the
        # suite is admitted, including when training may last more than an hour.
        for gpu in gpus:
          for job in read(suite/'suite.json')['jobs']:
            identity=inventory[gpu]
            folder=suite/'preflight'/fingerprint(identity)[:12]/job['id']
            config=read(suite/'jobs'/job['id']/'config.json')
            if (folder/'canary.json').exists():
                receipt=read(folder/'canary.json')
                if not receipt['passed'] or receipt['config_sha256']!=fingerprint(config) or receipt.get('gpu_identity')!=identity:
                    raise ValueError('Stale preflight')
            else:
                if folder.exists():
                    folder.rename(folder.with_name(folder.name+'.interrupted-'+uuid.uuid4().hex[:8]))
                # Map exactly the requested physical GPU into the canary children.
                previous=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=gpu
                os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                try:
                    receipt=canary(config,data,args.cache,folder,'cuda:0')
                    receipt['gpu_identity']=identity;atomic_json(folder/'canary.json',receipt)
                finally:
                    if previous is None: os.environ.pop('CUDA_VISIBLE_DEVICES',None)
                    else: os.environ['CUDA_VISIBLE_DEVICES']=previous
        if args.mode=='preflight':
            atomic_json(suite/'preflight-complete.json',{'state':'passed','jobs':len(read(suite/'suite.json')['jobs'])})
            return 0
        execute(suite,gpus,args.cache,data)
    return 0


if __name__=='__main__':
    def interrupted(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,interrupted)
    if hasattr(signal,'SIGHUP'): signal.signal(signal.SIGHUP,interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Baseline delivery paused; resume the same suite.',file=sys.stderr)
        raise SystemExit(75)
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        # No credential-bearing transport diagnostics/tracebacks in console logs.
        print('Baseline delivery stopped: '+(str(error) if isinstance(error,ValueError) else type(error).__name__),file=sys.stderr)
        raise SystemExit(2)
