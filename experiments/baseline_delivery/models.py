"""Declared classification adapters; legacy scientific code remains untouched."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types

import torch
from torch import nn

from .common import ROOT, digest, read
from .data import TASKS

MODELS = ('transformer','s4d_minimal','mamba1','mega')


def load_s4d(cache):
    entry=read(ROOT/'experiments/baseline_delivery/assets.json')['objects'][0]
    folder=Path(cache)/'upstream'/entry['id']
    for name,record in entry['members'].items():
        if digest(folder/name)!=record['sha256']:
            raise ValueError('Pinned S4D source hash mismatch')
    dropout_name='_qkan_pinned_s4_dropout'
    spec=importlib.util.spec_from_file_location(dropout_name,folder/'dropout.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sys.modules[dropout_name]=module
    source=(folder/'s4d.py').read_text(encoding='utf8')
    original='from src.models.nn import DropoutNd'
    if source.count(original)!=1:
        raise ValueError('Upstream import boundary changed')
    source=source.replace(original,'from '+dropout_name+' import DropoutNd')
    result=types.ModuleType('_qkan_pinned_s4d')
    exec(compile(source,str(folder/'s4d.py'),'exec'),result.__dict__)
    return result.S4D


class Block(nn.Module):
    def __init__(self, config, cache):
        super().__init__()
        width=config['width']; name=config['model']
        self.name=name
        self.norm=nn.LayerNorm(width)
        if name=='s4d_minimal':
            self.mixer=load_s4d(cache)(width,d_state=config['state_size'],dropout=config['dropout'],
                                      transposed=False,lr=min(config['lr'],1e-3))
        elif name=='mamba1':
            from mamba_ssm.modules.mamba_simple import Mamba
            self.mixer=Mamba(d_model=width,d_state=config['state_size'],d_conv=4,expand=2)
        else:
            raise ValueError('Unsupported mixer')
        self.dropout=nn.Dropout(config['dropout'])

    def forward(self,h,mask):
        value=self.mixer(self.norm(h))
        if self.name=='s4d_minimal':
            value=value[0]
        return (h+self.dropout(value)).masked_fill(~mask.unsqueeze(-1),0)


class Classifier(nn.Module):
    def __init__(self,config,cache):
        super().__init__()
        self.config=config; self.task=config['task']; self.kind=config['model']
        if self.kind not in MODELS:
            raise ValueError('Unknown baseline')
        width=config['width']; self.tokens=self.task not in ('image','pathfinder')
        self.embedding=nn.Embedding(config['vocabulary'],width,padding_idx=1) if self.tokens else nn.Linear(1,width)
        if self.kind=='mega':
            import os
            from .strong_sources import official_source, mega_encoder
            self._source_context=official_source('mega', os.environ['QKAN_MEGA_SOURCE'])
            modules=self._source_context.__enter__()
            self.encoder=mega_encoder(modules, dict(
                padding_idx=1, vocab_size=config['vocabulary'] or 1,
                embedding_type='sparse' if self.tokens else 'linear', embedding_dim=width,
                num_encoder_layers=config['layers'], hidden_dim=width*2,
                ffn_hidden_dim=width*2, z_dim=width, n_dim=config['state_size'],
                norm_type='layernorm', normalize_before=True, chunk_size=-1,
                max_seq_len=config.get('length',TASKS[self.task][0]), sen_rep_type='mp',
                dropout=config['dropout'], attention_dropout=config['dropout'],
                hidden_dropout=config['dropout']))
            del self.embedding  # official encoder owns its input embedding
        elif self.kind=='transformer':
            length=config.get('length',TASKS[self.task][0])
            pe=torch.zeros(length,width)
            angle=torch.arange(length).unsqueeze(1)*torch.exp(torch.arange(0,width,2)*(-math.log(10000.)/width))
            pe[:,0::2]=angle.sin(); pe[:,1::2]=angle[:,:pe[:,1::2].shape[1]].cos()
            self.register_buffer('position',pe)
            layer=nn.TransformerEncoderLayer(width,config['heads'],width*4,config['dropout'],
                                             activation='gelu',batch_first=True,norm_first=True)
            self.encoder=nn.TransformerEncoder(layer,config['layers'],enable_nested_tensor=False)
        else:
            self.blocks=nn.ModuleList([Block(config,cache) for _ in range(config['layers'])])
        self.norm=nn.Identity() if self.kind=='mega' else nn.LayerNorm(width)
        self.head=nn.Linear(width*(2 if self.task=='retrieval' else 1),TASKS[self.task][1])

    def encode(self,x):
        mask=x.ne(1) if self.tokens else torch.ones_like(x,dtype=torch.bool)
        if torch.any(mask.sum(1)==0):
            raise ValueError('Empty model input')
        if self.kind=='mega':
            states,pooled=self.encoder(x, mask.sum(1), last_state_only=True)
            h=states[-1].transpose(0,1).masked_fill(~mask.unsqueeze(-1),0)
            if getattr(self,'capture',False):
                self.captured.append({'input':x.detach().cpu(),'mask':mask.detach().cpu(),
                                      'encoder':h.detach().cpu()})
            return pooled
        h=self.embedding(x if self.tokens else x.unsqueeze(-1))
        h=h.masked_fill(~mask.unsqueeze(-1),0)
        if self.kind=='transformer':
            h=self.encoder(h+self.position[:h.shape[1]],src_key_padding_mask=~mask)
        else:
            for block in self.blocks:
                h=block(h,mask)
        h=self.norm(h).masked_fill(~mask.unsqueeze(-1),0)
        if getattr(self,'capture',False):
            self.captured.append({'input':x.detach().cpu(),'mask':mask.detach().cpu(),
                                  'encoder':h.detach().cpu()})
        return h.sum(1)/mask.sum(1).unsqueeze(-1)

    def forward(self,batch,representations=False):
        if self.task=='retrieval':
            a,b=self.encode(batch['inputs1']),self.encode(batch['inputs2'])
            pooled=torch.cat(((a-b).abs(),a*b),dim=-1)
        else:
            pooled=self.encode(batch['inputs'])
        logits=self.head(pooled)
        return (logits,pooled) if representations else logits

    def load_state_dict(self, state_dict, strict=True):
        result=super().load_state_dict(state_dict,strict=strict)
        if self.kind=='mega':
            # Official EMA caches are derived eval tensors, absent from state_dict.
            # They must be invalidated when loading best or resumed parameters.
            for module in self.encoder.modules():
                if type(module).__name__=='MultiHeadEMA':
                    module._kernel=None
                    module._coeffs=None
        return result


def optimizer(model,config):
    # Preserve S4D optimizer metadata and Mamba's explicit no-decay markers.
    groups={}
    for param in model.parameters():
        options={'lr':config['lr'],'weight_decay':config['weight_decay'],**getattr(param,'_optim',{})}
        if getattr(param,'_no_weight_decay',False):
            options['weight_decay']=0.0
        key=tuple(sorted(options.items()))
        groups.setdefault(key,{'params':[],**options})['params'].append(param)
    return torch.optim.AdamW(list(groups.values()),lr=config['lr'])
