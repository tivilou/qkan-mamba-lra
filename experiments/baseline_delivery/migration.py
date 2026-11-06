"""Explicit fixed-to-adaptive fork. Filesystem and policy checks use stdlib only.

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


def compatible(old,new):
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


def snapshot(source,suite,cache):
    source=Path(source).resolve();suite=Path(suite).resolve()
    if source.parent.name!='runs' or source==suite or source in suite.parents or suite in source.parents:
        raise ValueError('Migration requires a separate source suite under runs/')
    source_root=source.parent.parent
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source_root,text=True).strip()
    if revision!=SOURCE_REVISION or dirty: raise ValueError('Original executor revision/files changed')
    destination=suite/'migration-source'
    if destination.exists(): raise ValueError('Migration snapshot already exists; resume the new suite')
    pending=suite/('migration-pending-'+uuid.uuid4().hex)
    with ExitStack() as held:
        for path in (source/'progress-supervisor.lock',source/'suite.lock'):
            held.enter_context(lock(path))
        context=read(source/'context.json');manifest=read(source/'suite.json');gate=read(source/'gate-report.json')
        if (context.get('source_revision')!=SOURCE_REVISION or context.get('adaptive_memory')
                or manifest.get('source_revision')!=SOURCE_REVISION or gate.get('status')!='passed'
                or gate.get('source_revision')!=SOURCE_REVISION or context['cache']!=str(Path(cache).resolve())):
            raise ValueError('Original source admission/cache differs')
        records={};pending.mkdir()
        try:
            for job in manifest['jobs']:
                name=job['id']
                if Path(name).name!=name or name in records or name in ('.','..'):
                    raise ValueError('Invalid source job identity')
                folder=source/'jobs'/name
                held.enter_context(lock(folder/'worker.lock'))
                if (folder/'summary.json').exists():
                    raise ValueError('Completed fixed jobs cannot be relabeled adaptive; migrate before completion')
                config=read(folder/'config.json')
                if fingerprint(config)!=job['config_sha256'] or config['source_revision']!=SOURCE_REVISION:
                    raise ValueError('Original job configuration changed')
                target=pending/name;target.mkdir();atomic_json(target/'config.json',config)
                record=dict(config_sha256=fingerprint(config),checkpoint=None)
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
                source_revision=SOURCE_REVISION,source_manifest_sha256=digest(source/'suite.json'),
                data=manifest['data'],cache=context['cache'],jobs=records,
                trajectory='fixed prefix followed by adaptive continuation; not fixed-policy equivalent')
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
    if state['step']!=pointer['step'] or 'adaptive_memory' in state or 'migration' in state:
        raise ValueError('Original checkpoint step/trajectory differs')
    if state.get('training_complete'): raise ValueError('Cannot migrate a completed training state')
    lineage=dict(source_revision=SOURCE_REVISION,source_contract=pointer['contract'],
        source_checkpoint_sha256=pointer['sha256'],optimizer_step=state['step'],
        source_config_sha256=record['config_sha256'],target_config_sha256=fingerprint(config),
        trajectory=plan['trajectory'])
    state['contract']=contract;state['migration']=lineage
    return state


def admit(suite,configs,data,cache,gpus,run,build,inventory):
    """Fresh worker restores and roundtrips every migrated job before scheduling."""
    suite=Path(suite);plan=plan_for(suite)
    if plan is None: return None
    if plan['data']!=str(data) or plan['cache']!=str(Path(cache).resolve()) or set(plan['jobs'])!=set(configs):
        raise ValueError('Migration data/cache/job selection differs')
    receipts={}
    import os
    for name,config in configs.items():
        folder=suite/'jobs'/name;record=plan['jobs'][name]
        compatible(read(suite/'migration-source'/name/'config.json'),config)
        if record['checkpoint'] is None: continue
        report=folder/'migration-validation.json'
        if not report.exists():
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
    report=dict(schema='qkan-migration-admission.v1',passed=True,source_revision=SOURCE_REVISION,
        plan_sha256=digest(suite/'migration-source/plan.json'),restores=receipts,
        gpu_identity=inventory[gpus[0]],trajectory=plan['trajectory'])
    atomic_json(suite/'migration-report.json',report)
    return report
