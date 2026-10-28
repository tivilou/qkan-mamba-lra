"""One isolated baseline job. Only collaborator parent launches formal jobs."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import uuid

import numpy as np
import torch
from torch import nn

from .common import atomic_json, digest, fingerprint, lock, read
from .data import Split, vocabulary
from .models import Classifier, optimizer

PAUSE = False


def request_pause(*_):
    global PAUSE
    PAUSE = True


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_copy(state):
    return {key:value.detach().cpu().clone() for key,value in state.items()}


def save_checkpoint(output,payload):
    folder=output/'checkpoints'; folder.mkdir(exist_ok=True)
    generation='step-'+str(payload['step'])+'-'+uuid.uuid4().hex+'.pt'
    temp=folder/(generation+'.pending'); final=folder/generation
    with temp.open('xb') as handle:
        torch.save(payload,handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp,final)
    pointer=folder/'latest.json'; old=read(pointer) if pointer.exists() else None
    previous=([{'file':old['file'],'sha256':old['sha256']}] + old.get('previous',[]))[:1] if old else []
    atomic_json(pointer,{'schema':'qkan-training-checkpoint.v1','file':generation,'sha256':digest(final),
                         'step':payload['step'],'contract':payload['contract'],'previous':previous})
    if os.name != 'nt':
        fd=os.open(folder,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    # Retain previous valid latest as well as current. Best weights live inside
    # each full-state generation, so no cross-file best/latest transaction exists.
    retained={generation,*[p['file'] for p in previous]}
    for path in folder.glob('step-*.pt'):
        if path.name not in retained:
            path.unlink()


def load_checkpoint(output,contract):
    pointer=read(output/'checkpoints/latest.json')
    if pointer.get('schema')!='qkan-training-checkpoint.v1' or pointer['contract']!=contract:
        raise ValueError('Checkpoint scientific contract mismatch')
    filename=pointer['file']
    if Path(filename).name!=filename:
        raise ValueError('Invalid checkpoint generation')
    path=output/'checkpoints'/filename
    if digest(path)!=pointer['sha256']:
        raise ValueError('Corrupt latest checkpoint; prior generation retained for explicit recovery')
    # Only this runner's local, hash-bound checkpoint is accepted. Never load
    # checkpoint files supplied by an untrusted remote source.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('schema')!='qkan-training-state.v1' or state['contract']!=contract:
        raise ValueError('Invalid checkpoint state')
    return state


@torch.no_grad()
def evaluate(model,split,device,microbatch):
    model.eval(); loss=0.;correct=total=0
    for start in range(0,len(split),microbatch):
        batch=split.batch(np.arange(start,min(start+microbatch,len(split))),device)
        logits=model(batch)
        loss+=nn.functional.cross_entropy(logits,batch['labels'],reduction='sum').item()
        correct+=int((logits.argmax(-1)==batch['labels']).sum());total+=len(batch['labels'])
    return {'loss':loss/total,'accuracy':correct/total,'samples':total}


def run(config,data,cache,output,device,resume=False,stop_after=None,synthetic=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(os.environ.get('QKAN_CPU_THREADS','1')))
    seed=config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith('cuda'):
        if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    if device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    train=Split(data,config['task'],'train',cache,config.get('data_hashes',{}).get('train'))
    val=Split(data,config['task'],'val',cache,config.get('data_hashes',{}).get('val'))
    if synthetic:
        marker=read(Path(data)/'synthetic-canary.json')
        if (marker.get('bounded') is not True or marker.get('test_included') is not False or
                config['epochs']!=2 or len(train)>3*config['batch_size'] or len(val)>2):
            raise ValueError('Synthetic mode is restricted to the bounded recovery canary')
    vocab,dict_sha=vocabulary(data,config['task'])
    if config.get('vocabulary')!=vocab:
        raise ValueError('Vocabulary differs from the model contract')
    contract=fingerprint({'config':config,'train':train.identity,'val':val.identity,'dictionary':dict_sha})
    model=Classifier(config,cache).to(device); opt=optimizer(model,config)
    import importlib.metadata
    import platform
    atomic_json(output/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
         'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
         'torch':torch.__version__,'numpy':np.__version__,'cuda_runtime':torch.version.cuda,
         'gpu_name':torch.cuda.get_device_name() if device.startswith('cuda') else None,
         'precision':'float32; no mixed precision','deterministic_algorithms':True})
    steps_per_epoch=math.ceil(len(train)/config['batch_size'])
    total_steps=config['epochs']*steps_per_epoch
    warmup=min(config['warmup_steps'],max(0,total_steps-1))
    def multiplier(step):
        if step<warmup: return (step+1)/max(1,warmup)
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total_steps-warmup))))
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,multiplier)
    state={'schema':'qkan-training-state.v1','contract':contract,'step':0,'epoch':0,'cursor':0,
           'permutation':None,'best_accuracy':-1.,'best_epoch':None,'best_weights':None,
           'history':[],'epoch_loss':0.,'epoch_correct':0,'epoch_samples':0,'update_trace':[],
           'training_seconds':0.,'training_complete':False,'amp':{'enabled':False,'scaler':None},
           'runtime':{'torch':torch.__version__,'numpy':np.__version__,'device_type':device.split(':')[0]}}
    if resume:
        state=load_checkpoint(output,contract)
        if state['runtime']!={'torch':torch.__version__,'numpy':np.__version__,'device_type':device.split(':')[0]}:
            raise ValueError('Runtime changed; exact resume is not certified across runtimes')
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);schedule.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])
    elif (output/'checkpoints/latest.json').exists():
        raise ValueError('Existing checkpoint requires explicit resume')
    atomic_json(output/'config.json',config)
    started=time.monotonic(); last_save=started
    def save():
        nonlocal last_save,started
        now=time.monotonic(); state['training_seconds']+=now-started; started=now
        state.update(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),rng=rng_state())
        save_checkpoint(output,state);last_save=now
    while state['epoch']<config['epochs']:
        if state['permutation'] is None:
            state['permutation']=torch.randperm(len(train))
        model.train()
        while state['cursor']<len(train):
            end=min(len(train),state['cursor']+config['batch_size'])
            ids=state['permutation'][state['cursor']:end].numpy()
            opt.zero_grad(set_to_none=True); update_loss=0.; update_correct=0
            for start in range(0,len(ids),config['micro_batch']):
                sub=ids[start:start+config['micro_batch']]
                batch=train.batch(sub,device); logits=model(batch)
                loss=nn.functional.cross_entropy(logits,batch['labels'],reduction='sum')/len(ids)
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward();update_loss+=loss.item()*len(ids)
                update_correct+=int((logits.argmax(-1)==batch['labels']).sum())
            nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
            opt.step();schedule.step()
            state['cursor']=end;state['step']+=1
            state['epoch_loss']+=update_loss;state['epoch_correct']+=update_correct;state['epoch_samples']+=len(ids)
            if synthetic:
                state['update_trace'].append({'step':state['step'],'rows':ids.tolist(),'loss':update_loss/len(ids)})
            atomic_json(output/'heartbeat.json',{'state':'running','epoch':state['epoch'],
                        'next_row_cursor':state['cursor'],'optimizer_step':state['step'],
                        'loss':update_loss/len(ids),'updated_at':time.time()})
            if PAUSE or (stop_after is not None and state['step']>=stop_after):
                save();atomic_json(output/'status.json',{'state':'paused','step':state['step']});return 75
            if state['step']%config['checkpoint_steps']==0 or time.monotonic()-last_save>=300:
                save()
        metric=evaluate(model,val,device,config['micro_batch'])
        if metric['accuracy']>state['best_accuracy']:
            state['best_accuracy']=metric['accuracy'];state['best_epoch']=state['epoch']
            state['best_weights']=cpu_copy(model.state_dict())
        state['history'].append({'epoch':state['epoch'],'train_loss':state['epoch_loss']/state['epoch_samples'],
                                 'train_accuracy':state['epoch_correct']/state['epoch_samples'],'validation':metric})
        state['epoch']+=1;state['cursor']=0;state['permutation']=None
        state['epoch_loss']=0.;state['epoch_correct']=0;state['epoch_samples']=0
        save()
    if not state['training_complete']:
        state['training_complete']=True;save()
    # Evaluation retry uses the full-state checkpoint's own immutable best
    # weights, without repeating training or testing final-epoch weights.
    model.load_state_dict(state['best_weights'],strict=True)
    selected_val=evaluate(model,val,device,config['micro_batch'])
    if selected_val['accuracy']!=state['best_accuracy']:
        raise ValueError('Selected checkpoint validation does not reproduce')
    if synthetic:
        test_metric=None
    else:
        test=Split(data,config['task'],'test',cache,config['data_hashes']['test'])
        test_metric=evaluate(model,test,device,config['micro_batch'])
    torch.save(state['best_weights'],output/'best-model.pt')
    from .trace import capture
    capture(model,val,output,config,digest(output/'best-model.pt'))
    summary={'schema':'qkan-baseline-result.v1','task':config['task'],'model':config['model'],'seed':seed,
             'source_revision':config['source_revision'],'config_sha256':fingerprint(config),'contract':contract,
             'data_hashes':config.get('data_hashes',{}),'best_epoch':state['best_epoch'],
             'validation':selected_val,'test':test_metric,'history':state['history'],
             'optimizer_steps':state['step'],'parameters':sum(p.numel() for p in model.parameters()),
             'training_seconds':state['training_seconds'],
             'peak_gpu_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
             'synthetic':synthetic,'formal':not synthetic,'best_weights_sha256':digest(output/'best-model.pt')}
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'status.json',{'state':'computed','step':state['step'],'delivery':'pending'})
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--synthetic',action='store_true');parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    if not args.synthetic:
        from .runner import check_release
        check_release(read(args.config))
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,request_pause)
    if hasattr(signal,'SIGHUP'): signal.signal(signal.SIGHUP,request_pause)
    with lock(args.output/'worker.lock'):
        return run(read(args.config),args.data,args.cache,args.output,args.device,args.resume,args.stop_after,args.synthetic)


if __name__=='__main__':
    raise SystemExit(main())
