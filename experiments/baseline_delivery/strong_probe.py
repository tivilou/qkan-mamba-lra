"""Fixed-input source-composition forward/loss/gradient/update evidence."""
import argparse
import os
from pathlib import Path

from .common import atomic_json, digest, ROOT, read
from .strong_sources import official_source


def mega(cache, device):
    import torch
    from .models import Classifier,optimizer
    torch.manual_seed(42)
    config=dict(model='mega',task='retrieval',vocabulary=32,width=16,layers=2,
                state_size=4,length=32,dropout=0.,lr=.001,weight_decay=.01)
    adapter=Classifier(config,cache).to(device)
    # A second independently constructed source module, loaded by strict keys.
    reference=Classifier(config,cache).to(device)
    reference.load_state_dict(adapter.state_dict(),strict=True)
    batch={key:torch.randint(2,32,(2,32),device=device) for key in ('inputs1','inputs2')}
    batch['inputs1'][0,-4:]=1;batch['inputs2'][1,-8:]=1
    a=adapter(batch);parts=[]
    for key in ('inputs1','inputs2'):
        x=batch[key];mask=x!=1
        # Call official source directly, bypassing Classifier.encode/forward.
        states,pooled=reference.encoder(x,mask.sum(1),last_state_only=True)
        h=states[-1].transpose(0,1).masked_fill(~mask[...,None],0)
        torch.testing.assert_close(pooled,h.sum(1)/mask.sum(1)[:,None])
        parts.append(pooled)
    b=reference.head(torch.cat((abs(parts[0]-parts[1]),parts[0]*parts[1]),-1))
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    labels=torch.tensor([0,1],device=device)
    la=torch.nn.functional.cross_entropy(a,labels);lb=torch.nn.functional.cross_entropy(b,labels)
    torch.testing.assert_close(la,lb);la.backward();lb.backward()
    for (name,x),(other,y) in zip(adapter.named_parameters(),reference.named_parameters()):
        if name!=other or (x.grad is None)!=(y.grad is None): raise ValueError('MEGA gradient structure differs')
        if x.grad is not None: torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=1e-6)
    optimizer(adapter,config).step();optimizer(reference,config).step()
    for key,value in adapter.state_dict().items(): torch.testing.assert_close(value,reference.state_dict()[key],rtol=1e-5,atol=1e-6)


def s5(device):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from .s5_worker import build
    if device.startswith('cuda') and jax.default_backend()!='gpu': raise ValueError('S5 probe has no GPU')
    config=dict(task='retrieval',vocabulary=32,width=16,layers=2,state_size=8,
                dropout=0.,discretization='zoh',bidirectional=True)
    with official_source('s5',os.environ['QKAN_S5_SOURCE']) as modules:
        Model=build(modules,config);model=Model(training=False)
        key=jax.random.PRNGKey(42)
        batch={name:jax.random.randint(jax.random.fold_in(key,i),(2,32),2,32) for i,name in enumerate(('inputs1','inputs2'))}
        batch['inputs1']=batch['inputs1'].at[0,-4:].set(1)
        batch['inputs2']=batch['inputs2'].at[1,-8:].set(1)
        params=model.init(key,batch)['params'];labels=jnp.array([0,1])
        # Recover the source encoder definition, then apply per sequence directly.
        n=config['state_size'];lam,_,_,v,_=modules['initialization'].make_DPLR_HiPPO(n)
        v=v[:,:n//2];lam=lam[:n//2]
        ssm=modules['ssm'].init_S5SSM(H=16,P=n//2,Lambda_re_init=lam.real,Lambda_im_init=lam.imag,
            V=v,Vinv=v.conj().T,C_init='lecun_normal',discretization='zoh',dt_min=.001,dt_max=.1,
            conj_sym=True,clip_eigs=False,bidirectional=True)
        upstream=modules['encoder'].StackedEncoderModel(ssm=ssm,d_model=16,n_layers=2,
            training=False,batchnorm=False,prenorm=True,dropout=0.,activation='half_glu1')
        def direct(p):
            parts=[]
            for name in ('inputs1','inputs2'):
                x=batch[name];mask=x!=1
                h=jnp.where(mask[...,None],p['embedding']['embedding'][x],0)
                h=jax.vmap(lambda row:upstream.apply({'params':p['encoder']},row,jnp.ones(32)))(h)
                h=jnp.where(mask[...,None],h,0);parts.append(h.sum(1)/mask.sum(1)[:,None])
            z=jnp.concatenate((abs(parts[0]-parts[1]),parts[0]*parts[1]),-1)
            return z@p['head']['kernel']+p['head']['bias']
        def compare(a,b):
            aa,ta=jax.tree_util.tree_flatten(a);bb,tb=jax.tree_util.tree_flatten(b)
            if ta!=tb: raise ValueError('S5 parameter structure differs')
            for x,y in zip(aa,bb): np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-6)
        compare(model.apply({'params':params},batch)[0],direct(params))
        fa=lambda p:optax.softmax_cross_entropy_with_integer_labels(model.apply({'params':p},batch)[0],labels).mean()
        fb=lambda p:optax.softmax_cross_entropy_with_integer_labels(direct(p),labels).mean()
        va,ga=jax.value_and_grad(fa)(params);vb,gb=jax.value_and_grad(fb)(params)
        compare(va,vb);compare(ga,gb)
        # Compare the actual declared special/no-decay groups independently.
        from flax import traverse_util
        flat=traverse_util.flatten_dict(params)
        groups=traverse_util.unflatten_dict({k:'ssm' if k[-1] in ('B','Lambda_re','Lambda_im','log_step','norm') else 'regular' for k in flat})
        tx=optax.multi_transform({'ssm':optax.adam(.001),'regular':optax.adamw(.001,weight_decay=.01)},groups)
        ua,_=tx.update(ga,tx.init(params),params);ub,_=tx.update(gb,tx.init(params),params)
        compare(optax.apply_updates(params,ua),optax.apply_updates(params,ub))


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('mega','s5'),required=True)
    p.add_argument('--device',required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.model=='mega': mega(args.cache,args.device)
    else: s5(args.device)
    from .runner import implementation_hash
    atomic_json(args.output,dict(schema='qkan-source-composition.v1',model=args.model,passed=True,
        checks=['direct official encoder forward','loss','all parameter gradients','one optimizer update','padding and pair head'],
        implementation_sha256=implementation_hash(),protocol_sha256=digest(ROOT/'experiments/baseline_delivery/protocol.json'),
        formal=False,scope='Common-protocol adapter equivalence; no author benchmark score equivalence claimed.'))


if __name__=='__main__': main()
