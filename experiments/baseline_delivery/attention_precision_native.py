"""Native collaborator-only child for the optional BF16 attention probe."""
import argparse
import gc
import hashlib
from pathlib import Path

from .common import ROOT,atomic_json,digest,fingerprint,read,revision,worker_failure


def prepare_qkv(torch,model,tokens):
    """Exact first-layer pre-norm projection; preserve all key PAD positions."""
    valid=tokens.ne(1);batch,length=tokens.shape
    if not bool(valid.any(1).all()): raise ValueError('Empty document input')
    layer=model.encoder.layers[0];attn=layer.self_attn
    if not layer.norm_first or not attn.batch_first or attn.bias_k is not None or attn.bias_v is not None or attn.add_zero_attn:
        raise ValueError('Unexpected original attention adapter')
    with torch.no_grad():
        hidden=model.embedding(tokens).masked_fill(~valid.unsqueeze(-1),0)
        hidden=layer.norm1(hidden+model.position[:length])
        packed=torch.nn.functional.linear(hidden,attn.in_proj_weight,attn.in_proj_bias)
        qkv=tuple(value.reshape(batch,length,attn.num_heads,attn.head_dim).transpose(1,2).contiguous().detach().requires_grad_(True)
            for value in packed.chunk(3,dim=-1))
        mask=torch.zeros((batch,1,1,length),device=tokens.device,dtype=torch.float32)
        mask.masked_fill_(~valid[:,None,None,:],float('-inf'))
    return qkv,mask


def execute(request_path,output):
    from .attention_precision_diagnostic import unchanged,SOURCE_REVISION
    from .optimization_plugins import resolve
    request=read(request_path)
    if (request.get('schema')!='qkan-attention-probe-input.v1' or request.get('source_revision')!=revision()
            or request.get('micro_batch') not in (16,32)):
        raise ValueError('Native probe request/source differs')
    config=request['source_config'];pointer=request['checkpoint'];source=Path(request['source_suite'])
    job=source/'jobs/transformer-retrieval-s42'
    if (source.parent.name!='runs' or config.get('source_revision')!=SOURCE_REVISION
            or pointer['path']!=str(job/'checkpoints'/pointer['file'])
            or read(job/'config.json')!=config or fingerprint(config)!=request['source_config_sha256']
            or Path(pointer['path']).is_symlink()):
        raise ValueError('Native probe requires the original hash-bound checkpoint')
    unchanged(request);selection=resolve(request['optimization_config'],scope='attention_probe')
    if not selection.bf16_amp: raise ValueError('Explicit BF16 required')
    # Imports stay inside the private native child; no installation/network path.
    import torch
    import torch.nn.attention
    import numpy as np
    from .models import Classifier
    from .data import Split
    from .attention_precision import native_probe
    from .diagnostic_watch import mark
    if torch.__version__.split('+')[0]!='2.7.1' or np.__version__!='2.2.6':
        raise ValueError('Expected existing pinned Torch2.7.1/NumPy2.2.6 environment')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise ValueError('Native CUDA BF16 support missing')
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    mark('attention_source_load',flush=True)
    cached=Path(request['cache'])/'arrays'/config['data_hashes']['train']/'complete.json'
    if not cached.is_file(): raise ValueError('Existing train array cache required')
    split=Split(request['data'],'retrieval','train',request['cache'],config['data_hashes']['train'])
    rows=request['micro_batch'];array=np.array(split.arrays['inputs1'][:rows],dtype=np.int64,copy=True)
    if array.shape!=(rows,4000): raise ValueError('Expected fixed full-width train prefix')
    input_sha=hashlib.sha256(array.tobytes(order='C')).hexdigest()
    tokens=torch.as_tensor(array,device='cuda:0');state=torch.load(pointer['path'],map_location='cpu',weights_only=False)
    if (state.get('schema')!='qkan-training-state.v1' or state.get('contract')!=pointer['contract']
            or state.get('step')!=pointer['step'] or state.get('training_complete')):
        raise ValueError('Original full-state checkpoint identity differs')
    model=Classifier(config,request['cache']).to('cuda:0');model.load_state_dict(state['model'],strict=True)
    if any(p.dtype!=torch.float32 for p in model.parameters()): raise ValueError('FP32 master parameters required')
    mark('attention_qkv_projection',flush=True);qkv,mask=prepare_qkv(torch,model,tokens)
    generator=torch.Generator(device='cuda:0').manual_seed(config['seed']+17)
    upstream=torch.randn(qkv[0].shape,device='cuda:0',dtype=torch.float32,generator=generator)
    shape=list(qkv[0].shape);del model,state,tokens,array,split;gc.collect();torch.cuda.empty_cache()
    try:
        result=native_probe(torch,qkv,mask,upstream,selection,config['dropout'],lambda phase:mark(phase,flush=True))
    finally: unchanged(request)
    result.update(schema='qkan-attention-precision-native.v1',request_sha256=digest(request_path),
        checkpoint_sha256=pointer['sha256'],checkpoint_step=pointer['step'],source_config_sha256=request['source_config_sha256'],
        plugin_manifest=selection.manifest(),optimization_config_sha256=request['optimization_config_sha256'],
        dataset=dict(split='train',rows=rows,document='inputs1',selection='fixed first micro_batch source rows',
            train_sha256=config['data_hashes']['train'],input_sha256=input_sha,label_values_used=False,test_access=False),
        qkv_shape=shape,qkv_dtype='torch.float32',mask_dtype='torch.float32',dropout=config['dropout'],
        upstream='Independent fixed Gaussian cotangent; not the full-model loss gradient',
        environment=dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),deterministic_algorithms=True,tf32=False),
        formal_training_started=False,optimizer_updates=0,training_samples_committed=0,
        comparison='Changed precision; kernel mechanism triage only; full-update recovery, trace and convergence pending')
    atomic_json(output/'result.json',result)
    return 0 if result['mechanism_passed'] else 76


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    try: return execute(args.request,args.output)
    except BaseException as error:
        worker_failure(args.output,error);raise


if __name__=='__main__': raise SystemExit(main())
