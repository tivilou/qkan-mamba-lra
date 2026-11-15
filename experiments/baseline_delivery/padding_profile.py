"""Bounded train-only PAD opportunity measurement; no Torch import or training."""
import argparse
import datetime as dt
import math
from pathlib import Path
import time
import uuid

from .common import ROOT,atomic_json,digest,fingerprint,read,revision
from .optimization_plugins import resolve,transform_batch

JOB='transformer-retrieval-s42'
SOURCE_REVISION='b07d98e17ad0f996dbd416b0f38e015df6af4f60'


class NumpyTokenOps:
    """NumPy is supplied by the collaborator entry; no import on the owner."""
    def __init__(self,np): self.np=np

    def inspect(self,value):
        np=self.np
        if value.ndim!=2 or value.dtype.kind not in 'iu' or not all(value.shape):
            raise ValueError('Invalid integer token matrix')
        valid=value!=1
        if not bool(valid.any(axis=1).all()): raise ValueError('Empty model input')
        return value.shape[0],value.shape[1],[int(n) for n in
            value.shape[1]-np.argmax(valid[:,::-1],axis=1)]

    def prefix(self,value,width): return value[:,:width]

    def verify(self,original,trimmed,width):
        if (not self.np.array_equal(original[:,:width],trimmed)
                or not bool((original[:,width:]==1).all())):
            raise ValueError('Token content or position changed')


def summarize_batches(load,rows,micro,selection,ops):
    """No labels, sorting, shuffling, model calls or corpus-wide allocations."""
    if type(rows) is not int or rows<1 or type(micro) is not int or not 1<=micro<=128:
        raise ValueError('Invalid bounded batch scope')
    records={};load_seconds=transform_seconds=0.;batches=0
    for start in range(0,rows,micro):
        count=min(micro,rows-start);stamp=time.perf_counter();tokens=load(start,count)
        load_seconds+=time.perf_counter()-stamp
        if set(tokens)!={'inputs1','inputs2'}: raise ValueError('Expected two Retrieval documents')
        stamp=time.perf_counter();baseline,empty=transform_batch(tokens,resolve(),ops)
        if baseline is not tokens or empty: raise ValueError('Null adapter changed the batch')
        _,observed=transform_batch(tokens,selection,ops)
        if not selection.padding_trim:
            # Statistics inspect the baseline explicitly; the null adapter stays inert.
            observed={}
            for key,value in tokens.items():
                n,width,lengths=ops.inspect(value)
                observed[key]=dict(rows=n,original_width=width,execution_width=width,
                    last_valid_positions=lengths,attention_cells_before=n*width*width,
                    attention_cells_after=n*width*width)
        transform_seconds+=time.perf_counter()-stamp;batches+=1
        for key,item in observed.items():
            if item['rows']!=count: raise ValueError('Batch row coverage differs')
            target=records.setdefault(key,dict(rows=0,span_histogram={},execution_width_histogram={},
                attention_cells_before=0,attention_cells_after=0))
            target['rows']+=count
            for n in item['last_valid_positions']:
                target['span_histogram'][n]=target['span_histogram'].get(n,0)+1
            width=item['execution_width']
            target['execution_width_histogram'][width]=target['execution_width_histogram'].get(width,0)+count
            for field in ('attention_cells_before','attention_cells_after'): target[field]+=item[field]
    for item in records.values():
        def quantile(fraction):
            rank=math.ceil(item['rows']*fraction);seen=0
            for length,n in sorted(item['span_histogram'].items()):
                seen+=n
                if seen>=rank: return length
        item['span_quantiles']={name:quantile(q) for name,q in [('p50',.5),('p90',.9),('p99',.99)]}
        # Fixed small buckets retain aggregate counts rather than individual lengths.
        histogram={}
        for length,n in item.pop('span_histogram').items():
            bucket=((length+127)//128)*128
            histogram[bucket]=histogram.get(bucket,0)+n
        item['span_histogram_128']=histogram
    before=sum(v['attention_cells_before'] for v in records.values())
    after=sum(v['attention_cells_after'] for v in records.values())
    return dict(rows=rows,micro_batch=micro,batches=batches,documents=records,
        attention_cells_before=before,attention_cells_after=after,
        attention_cells_reduction_fraction=1-after/before,load_seconds=load_seconds,
        diagnostic_transform_seconds=transform_seconds,baseline_identity_checked=True,
        valid_tokens_and_coordinates_preserved=True,
        interpretation='CPU opportunity estimate on fixed train prefix. Attention cells are not GPU runtime or measured speedup. Internal PAD positions are retained.')


def collect(args,report):
    source=args.source_suite.resolve()
    suite_path=source/'suite.json';config_path=source/'jobs'/JOB/'config.json'
    initial={str(p):digest(p) for p in (suite_path,config_path)}
    suite=read(suite_path);config=read(config_path)
    if (config.get('source_revision')!=SOURCE_REVISION or config.get('model')!='transformer'
            or config.get('task')!='retrieval'):
        raise ValueError('Expected the admitted b07 Transformer Retrieval source')
    selection=resolve(read(args.optimizations))
    data=Path(suite['data'])
    if not data.is_absolute() or not data.is_dir(): raise ValueError('Original local dataset unavailable')
    if not (args.cache/'arrays'/config['data_hashes']['train']/'complete.json').is_file():
        raise ValueError('Verified source array cache missing; select the existing training cache')
    # Reuse the source-bound verified array cache. Neither validation nor test is opened.
    import numpy as np
    from .data import Split
    split=Split(data,'retrieval','train',args.cache,config['data_hashes']['train'])
    rows=min(args.max_rows,len(split))
    def load(start,count):
        return {key:array[start:start+count] for key,array in split.arrays.items() if key!='labels'}
    measurement=summarize_batches(load,rows,args.micro_batch,selection,NumpyTokenOps(np))
    if initial!={str(p):digest(p) for p in (suite_path,config_path)}:
        raise ValueError('Source metadata changed during observation')
    modules=('optimization_plugins.py','padding_profile.py')
    tree={name:digest(Path(__file__).parent/name) for name in modules}
    report.update(passed=True,status='opportunity_measured',source_revision=revision(),
        source_suite=str(source),source_config_sha256=fingerprint(config),source_metadata_sha256=initial,
        dataset=dict(split='train',identity=split.identity,total_rows=len(split),observed_rows=rows,
            selection='first min(max_rows,train_rows) rows in source order; no reordering',
            max_rows=args.max_rows,full_corpus=rows==len(split),label_values_used=False,test_access=False),
        plugin_manifest=selection.manifest(),optimization_config_sha256=digest(args.optimizations),
        plugin_tree_sha256=fingerprint(tree),plugin_files=tree,
        measurement=measurement,numpy_version=np.__version__,actual_gpu_speedup=None,
        next_action='Review observed PAD opportunity before native numerical/recovery/trace and full-update performance checks.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-suite',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--optimizations',type=Path,required=True)
    parser.add_argument('--micro-batch',type=int,choices=(16,32),default=32)
    parser.add_argument('--max-rows',type=int,default=4096)
    args=parser.parse_args()
    if not 1<=args.max_rows<=16384: parser.error('--max-rows must be between1 and16384')
    resolve(read(args.optimizations))  # Invalid choices fail before any dataset/output work.
    source=revision()
    from .bootstrap import check_exchange
    check_exchange()
    folder=ROOT/'runs'/('padding-profile-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(schema='qkan-padding-opportunity.v1',passed=False,status='failed',
        source_revision=source,formal_training_started=False,source_checkpoint_accessed=False,
        model_executed=False,gpu_accessed=False,native_trace_ready=False,
        architecture_status='approved_pending_native_trace',training_samples_committed=0)
    error=None
    try: collect(args,report)
    except Exception as caught:
        error=caught;report['error_type']=type(caught).__name__
    atomic_json(folder/'padding-profile.json',report)
    from .return_artifacts import upload_declared
    try: upload_declared(folder,'padding-profile',['padding-profile.json'])
    except Exception:
        print('Return pending; compact report and retry manifest retained: '+str(folder),flush=True)
        if error is None: raise
    if error is not None: raise error
    print('Train-only padding opportunity measured; no model or GPU job started.',flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
