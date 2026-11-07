"""Explicit checkpoint forks. Filesystem and policy checks use stdlib only.

Checkpoint deserialization stays in the native worker after source verification.
No original checkpoint/config is rewritten and no release is inherited.
"""
from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess
import uuid

from .common import atomic_json,digest,fingerprint,lock,read

SOURCE_REVISION='31639c014a62504f6596e75cf56a1abd5068bce4'
ADAPTIVE_REVISION='96e542dd527b777d7690f6602fa1d14530957f85'
DYNAMIC_REVISION='9a147cd70cedf1511de89d0381dfbdabac3274f2'


def compatible(old,new):
    if new.get('torch_acceleration'):
        from .torch_acceleration import policy as acceleration_policy
        from .performance import policy as performance_policy
        from .logical_batch import policy as logical_policy
        if (old.get('source_revision')!=DYNAMIC_REVISION or old.get('performance') or old.get('torch_acceleration')
                or old.get('adaptive_memory')!=logical_policy(old['batch_size']) or old['model']=='s5'):
            raise ValueError('Acceleration migration requires the dynamic9a Torch executor')
        expected={**old,'source_revision':new['source_revision'],'performance':performance_policy(),
            'torch_acceleration':acceleration_policy()}
        if expected!=new: raise ValueError('Acceleration migration changed scientific configuration')
        return
    if new.get('performance'):
        from .performance import policy as performance_policy
        from .logical_batch import policy as logical_policy
        if (old.get('source_revision')!=DYNAMIC_REVISION or old.get('performance')
                or old.get('adaptive_memory')!=logical_policy(old['batch_size'])):
            raise ValueError('Performance migration requires dynamic executor9a147cd')
        expected={**old,'source_revision':new['source_revision'],'performance':performance_policy()}
        if expected!=new: raise ValueError('Performance migration changed scientific configuration')
        return
    if (new.get('adaptive_memory') or {}).get('logical_tiers'):
        from .logical_batch import policy
        if old.get('source_revision') not in (SOURCE_REVISION,ADAPTIVE_REVISION):
            raise ValueError('Unsupported source for dynamic logical-batch migration')
        expected={**old,'source_revision':new['source_revision'],'adaptive_memory':policy(old['batch_size'])}
        if expected!=new: raise ValueError('Dynamic migration changed scientific configuration')
        return
    if old.get('source_revision')!=SOURCE_REVISION or old.get('adaptive_memory'):
        raise ValueError('Migration supports the admitted fixed executor31639c0 only')
    expected={**old,'source_revision':new['source_revision'],'adaptive_memory':new.get('adaptive_memory')}
    if not new.get('adaptive_memory') or expected!=new:
        raise ValueError('Migration changed model/data/optimizer/protocol configuration')


def checked_pointer(folder):
    folder=Path(folder);pointer=read(folder/'checkpoints/latest.json')
    name=pointer['file']
    if (Path(name).name!=name or name in ('.','..') or
            pointer.get('schema') not in ('qkan-training-checkpoint.v1','qkan-s5-checkpoint.v1')):
        raise ValueError('Invalid migration checkpoint pointer')
    target=folder/'checkpoints'/name
    if target.is_symlink() or digest(target)!=pointer['sha256']:
        raise ValueError('Migration checkpoint hash mismatch')
    return pointer,target


def completed_files(folder,config):
    """Validate an existing result without deserializing weights or rerunning it."""
    folder=Path(folder)
    if not (folder/'summary.json').is_file() or not (folder/'trace-validation.json').is_file():
        raise ValueError('Completed result lacks summary/trace evidence')
    summary=read(folder/'summary.json');trace=read(folder/'trace-validation.json')
    if (summary.get('formal') is not True or summary.get('synthetic') is not False
            or summary.get('config_sha256')!=fingerprint(config)
            or any(summary.get(k)!=config[k] for k in ('source_revision','model','task','seed'))
            or summary.get('data_hashes')!=config.get('data_hashes',{})
            or trace.get('passed') is not True):
        raise ValueError('Completed result identity/formal trace differs')
    weights='best-model.msgpack' if config['model']=='s5' else 'best-model.pt'
    weight_sha=digest(folder/weights)
    if summary.get('best_weights_sha256')!=weight_sha or trace.get('checkpoint_sha256')!=weight_sha:
        raise ValueError('Completed selected weights/trace hash differs')
    pointer,generation=checked_pointer(folder)
    if pointer['contract']!=summary.get('contract') or pointer['step']!=summary.get('optimizer_steps'):
        raise ValueError('Completed latest checkpoint/summary differs')
    names=['config.json','summary.json','environment.json',weights,'trace-validation.json',
           'checkpoints/latest.json','checkpoints/'+generation.name]
    for previous in pointer.get('previous',[]):
        name=previous['file']
        if Path(name).name!=name or name in ('.','..') or digest(folder/'checkpoints'/name)!=previous['sha256']:
            raise ValueError('Completed retained checkpoint differs')
        names.append('checkpoints/'+name)
    if config.get('adaptive_memory'): names+=['adaptive-memory.json','adaptive-events.jsonl']
    if (folder/'performance-profile.json').exists(): names.append('performance-profile.json')
    if (folder/'torch-acceleration.json').exists(): names.append('torch-acceleration.json')
    hashes={}
    for name in names:
        path=folder/name
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise ValueError('Invalid completed result artifact')
        hashes[name]=digest(path)
    return dict(source_revision=config['source_revision'],config_sha256=fingerprint(config),files=hashes)


def copy_verified(source,target,files):
    """Idempotent immutable copy. Never overwrite an existing differing file."""
    for name,sha in files.items():
        if (Path(name).is_absolute() or '\\' in name or '..' in Path(name).parts):
            raise ValueError('Invalid preserved artifact path')
        src=source/name;dst=target/name
        if src.is_symlink() or src.parent.is_symlink() or digest(src)!=sha:
            raise ValueError('Preserved source artifact changed')
        if dst.exists():
            if dst.is_symlink() or dst.parent.is_symlink() or digest(dst)!=sha:
                raise ValueError('Preserved target artifact changed')
            continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.parent.is_symlink(): raise ValueError('Preserved target directory is a symlink')
        temp=dst.with_name(dst.name+'.pending-'+uuid.uuid4().hex)
        try:
            shutil.copyfile(src,temp)
            if digest(temp)!=sha: raise ValueError('Preserved copied artifact hash differs')
            temp.replace(dst)
        finally:
            if temp.exists(): temp.unlink()


def preserve_completed(suite,name):
    """Materialize the original result, with its original config/source identity."""
    suite=Path(suite);plan=plan_for(suite);entry=plan['jobs'][name]['completion']
    source=suite/'migration-source'/name;target=suite/'jobs'/name
    if completed_files(source,read(source/'config.json'))!=entry:
        raise ValueError('Completed migration snapshot differs')
    copy_verified(source,target,entry['files'])
    job=dict(id=name,config_sha256=entry['config_sha256'],preserved_completion=entry)
    verify_completed(suite,job)
    return job


def verify_completed(suite,job):
    suite=Path(suite);plan=plan_for(suite)
    expected=plan['jobs'][job['id']].get('completion') if plan else None
    if not expected or job.get('preserved_completion')!=expected or job['config_sha256']!=expected['config_sha256']:
        raise ValueError('Missing original completed-result provenance')
    folder=suite/'jobs'/job['id']
    if completed_files(folder,read(folder/'config.json'))!=expected:
        raise ValueError('Preserved completed result changed')
    return expected


def snapshot(source,suite,cache):
    source=Path(source).resolve();suite=Path(suite).resolve()
    if source.parent.name!='runs' or source==suite or source in suite.parents or suite in source.parents:
        raise ValueError('Migration requires a separate source suite under runs/')
    source_root=source.parent.parent
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source_root,text=True).strip()
    if revision not in (SOURCE_REVISION,ADAPTIVE_REVISION,DYNAMIC_REVISION) or dirty: raise ValueError('Original executor revision/files changed')
    destination=suite/'migration-source'
    if destination.exists(): raise ValueError('Migration snapshot already exists; resume the new suite')
    pending=suite/('migration-pending-'+uuid.uuid4().hex)
    with ExitStack() as held:
        for path in (source/'progress-supervisor.lock',source/'suite.lock'):
            held.enter_context(lock(path))
        context=read(source/'context.json');manifest=read(source/'suite.json');gate=read(source/'gate-report.json')
        if (context.get('source_revision')!=revision
                or manifest.get('source_revision')!=revision or gate.get('status')!='passed'
                or gate.get('source_revision')!=revision or context['cache']!=str(Path(cache).resolve())):
            raise ValueError('Original source admission/cache differs')
        records={};pending.mkdir()
        try:
            admission_names=['context.json','suite.json','gate-report.json']
            for leaf in ('gate-evidence.json','runtime-release.json'):
                if (source/leaf).is_file(): admission_names.append(leaf)
            if gate.get('evidence_sha256') and digest(source/'gate-evidence.json')!=gate['evidence_sha256']:
                raise ValueError('Original admission evidence changed')
            admission={name:digest(source/name) for name in admission_names}
            copy_verified(source,pending/'source-admission',admission)
            for job in manifest['jobs']:
                name=job['id']
                if Path(name).name!=name or name in records or name in ('.','..'):
                    raise ValueError('Invalid source job identity')
                folder=source/'jobs'/name
                held.enter_context(lock(folder/'worker.lock'))
                config=read(folder/'config.json')
                if fingerprint(config)!=job['config_sha256'] or config['source_revision']!=revision:
                    raise ValueError('Original job configuration changed')
                target=pending/name;target.mkdir()
                record=dict(config_sha256=fingerprint(config),checkpoint=None)
                if (folder/'summary.json').exists():
                    completion=completed_files(folder,config)
                    copy_verified(folder,target,completion['files'])
                    record['completion']=completion;records[name]=record
                    continue
                atomic_json(target/'config.json',config)
                if read(source/'status.json')[name]['state']=='computed':
                    raise ValueError('Completed source status lacks result; refusing fresh fallback')
                if (folder/'checkpoints/latest.json').exists():
                    pointer,generation=checked_pointer(folder)
                    (target/'checkpoints').mkdir()
                    shutil.copyfile(generation,target/'checkpoints'/generation.name)
                    atomic_json(target/'checkpoints/latest.json',{**pointer,'previous':[]})
                    if digest(target/'checkpoints'/generation.name)!=pointer['sha256']:
                        raise ValueError('Copied checkpoint hash differs')
                    record['checkpoint']=dict(file=pointer['file'],sha256=pointer['sha256'],
                        contract=pointer['contract'],step=pointer['step'])
                else:
                    status=read(source/'status.json')[name]['state']
                    if status!='pending' or any((folder/'checkpoints').glob('*')):
                        raise ValueError('Started job lacks a valid full checkpoint; refusing fresh fallback')
                records[name]=record
            plan=dict(schema='qkan-checkpoint-migration.v1',source_suite=str(source),
                source_revision=revision,source_manifest_sha256=digest(source/'suite.json'),
                source_admission=admission,
                data=manifest['data'],cache=context['cache'],jobs=records,
                trajectory='recorded prefix followed by adaptive continuation; not fixed-policy equivalent')
            atomic_json(pending/'plan.json',plan)
            pending.rename(destination)
        finally:
            if pending.exists(): shutil.rmtree(pending)
    return plan


def plan_for(suite):
    path=Path(suite)/'migration-source/plan.json'
    return read(path) if path.exists() else None


def load_state(output,config,contract,original_contract,loader):
    """Native worker rebinds only contract metadata after exact config validation."""
    output=Path(output);suite=output.parent.parent;plan=plan_for(suite)
    if plan is None: raise ValueError('Missing explicit migration snapshot')
    record=plan['jobs'][output.name];folder=suite/'migration-source'/output.name
    old=read(folder/'config.json');compatible(old,config)
    if fingerprint(old)!=record['config_sha256'] or not record['checkpoint']:
        raise ValueError('Missing migration checkpoint/config identity')
    pointer,_=checked_pointer(folder)
    if any(pointer[k]!=record['checkpoint'][k] for k in ('file','sha256','contract','step')):
        raise ValueError('Migration snapshot changed')
    if pointer['contract']!=original_contract(old):
        raise ValueError('Original checkpoint data/dictionary contract differs')
    state=loader(folder,pointer['contract'])
    dynamic=bool((config.get('adaptive_memory') or {}).get('logical_tiers'))
    if state['step']!=pointer['step'] or (not dynamic and ('adaptive_memory' in state or 'migration' in state)):
        raise ValueError('Original checkpoint step/trajectory differs')
    if state.get('training_complete'): raise ValueError('Cannot migrate a completed training state')
    lineage=dict(source_revision=plan['source_revision'],source_contract=pointer['contract'],
        source_checkpoint_sha256=pointer['sha256'],optimizer_step=state['step'],
        source_config_sha256=record['config_sha256'],target_config_sha256=fingerprint(config),
        trajectory=plan['trajectory'])
    if dynamic:
        if state.get('migration'): lineage['previous_migration']=state['migration']
        if config.get('performance'):
            if not state.get('sample_clock') or not state.get('adaptive_memory'):
                raise ValueError('Dynamic source lacks sample clock/controller; refusing schedule reset')
            lineage['execution_policy']='FP32 lookahead/pinned staging; sample clock and controller preserved'
        else:
            state.pop('adaptive_memory',None)
            state.pop('sample_clock',None)
        lineage['trajectory']='recorded prefix followed by dynamic logical-batch/sample-schedule continuation'
    state['contract']=contract;state['migration']=lineage
    return state


def admit(suite,configs,data,cache,gpus,run,build,inventory):
    """Fresh worker restores and roundtrips every migrated job before scheduling."""
    suite=Path(suite);plan=plan_for(suite)
    if plan is None: return None
    if plan['data']!=str(data) or plan['cache']!=str(Path(cache).resolve()) or set(plan['jobs'])!=set(configs):
        raise ValueError('Migration data/cache/job selection differs')
    for name,sha in plan.get('source_admission',{}).items():
        if digest(suite/'migration-source/source-admission'/name)!=sha:
            raise ValueError('Original source admission snapshot changed')
    receipts={}
    import os
    for name,config in configs.items():
        folder=suite/'jobs'/name;record=plan['jobs'][name]
        if record.get('completion'):
            job=dict(id=name,config_sha256=record['config_sha256'],preserved_completion=record['completion'])
            verify_completed(suite,job)
            continue
        compatible(read(suite/'migration-source'/name/'config.json'),config)
        if record['checkpoint'] is None: continue
        report=folder/'migration-validation.json'
        if not report.exists():
            print('Checkpoint migration: '+name+' | zero-update native restore',flush=True)
            if (folder/'checkpoints/latest.json').exists():
                raise ValueError('Partial migration checkpoint exists without validation; inspect locally')
            spec=build('--config',folder/'config.json','--data',data,'--cache',cache,
                '--output',folder,'--device','cuda:0','--migration-check-only')
            if run(spec,folder/'migration-check.log',{**os.environ,'CUDA_VISIBLE_DEVICES':gpus[0]})!=0:
                raise ValueError('Migration restore check failed: '+name)
        value=read(report)
        if (not value.get('passed') or value['target_config_sha256']!=fingerprint(config)
                or value['source_checkpoint_sha256']!=record['checkpoint']['sha256']):
            raise ValueError('Migration validation identity differs')
        checked_pointer(folder)
        receipts[name]=value
    report=dict(schema='qkan-migration-admission.v1',passed=True,source_revision=plan['source_revision'],
        plan_sha256=digest(suite/'migration-source/plan.json'),restores=receipts,
        preserved_completions={name:entry['completion'] for name,entry in plan['jobs'].items() if entry.get('completion')},
        gpu_identity=inventory[gpus[0]],trajectory=plan['trajectory'])
    atomic_json(suite/'migration-report.json',report)
    return report
