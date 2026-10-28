from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

from .common import atomic_json, digest, fingerprint, lock, read

TASKS = {'listops':(2048,10), 'text':(4000,2), 'retrieval':(4000,2),
         'image':(1024,10), 'pathfinder':(1024,2)}


class Split:
    """Verified disk arrays, with conversion restricted to the current minibatch."""
    def __init__(self, root, task, split, cache, expected=None):
        source = Path(root)/task/(split+'.npz')
        sha = digest(source)
        if expected and sha != expected:
            raise ValueError('Split content differs from the frozen dataset')
        keys = ['inputs1','inputs2','labels'] if task == 'retrieval' else ['inputs','labels']
        out = Path(cache)/'arrays'/sha
        with lock(out.with_suffix('.lock')):
            out.mkdir(parents=True,exist_ok=True)
            receipt = out/'complete.json'
            if receipt.exists():
                records = read(receipt)
                if set(records) != set(keys) or any(digest(out/(key+'.npy')) != records[key] for key in keys):
                    raise ValueError('Extracted array cache was modified')
            else:
                with zipfile.ZipFile(source) as archive:
                    if sorted(archive.namelist()) != sorted(k+'.npy' for k in keys):
                        raise ValueError('Unexpected split members')
                    for key in keys:
                        temp = out/(key+'.pending')
                        with archive.open(key+'.npy') as src, temp.open('wb') as dst:
                            shutil.copyfileobj(src,dst,1024*1024)
                            dst.flush(); os.fsync(dst.fileno())
                        os.replace(temp,out/(key+'.npy'))
                atomic_json(receipt,{key:digest(out/(key+'.npy')) for key in keys})
        self.arrays={key:np.load(out/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in keys}
        self.task, self.identity = task, sha
        n=len(self.arrays['labels'])
        length,classes=TASKS[task]
        if any(a.shape!=(n,length) for k,a in self.arrays.items() if k!='labels'):
            raise ValueError('Unexpected sequence shape')
        if self.arrays['labels'].shape!=(n,) or self.arrays['labels'].dtype.kind not in 'iu':
            raise ValueError('Invalid labels')
        self.classes=classes

    def __len__(self):
        return len(self.arrays['labels'])

    def batch(self, ids, device):
        values={}
        for key,array in self.arrays.items():
            copy=np.array(array[ids],copy=True)
            if key=='labels':
                values[key]=torch.as_tensor(copy.astype(np.int64),device=device)
            elif self.task in ('image','pathfinder'):
                values[key]=torch.as_tensor(copy.astype(np.float32)/255.-.5,device=device)
            else:
                values[key]=torch.as_tensor(copy.astype(np.int64),device=device)
        return values


def vocabulary(root,task):
    if task in ('image','pathfinder'):
        return None,None
    path=Path(root)/'metadata'/(task+'-dictionary.json')
    return len(read(path)['input_symbols']),digest(path)
