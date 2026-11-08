"""Launch an admitted suite using its unchanged executor and recorded configs.

May be fetched to .bootstrap alongside formal_progress.py. No preparation,
backend selection, migration or configuration rewrite belongs in this entry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


STAGES=('imports','executor','admission','certificate','repair','exchange','data','gpu','configs','migration','scheduler')


def restore_admission_records(root,suite):
    """Restore only certificate-bound admission bytes from sealed gate evidence.

    Authenticate every certificate and inspect every artifact before any repair.
    Configs, certificates, gate evidence and checkpoints are never rewritten.
    Caller holds suite.lock; normal runtime validation must pass again afterward.
    """
    from experiments.baseline_delivery.common import read,digest,fingerprint,atomic_json,revision
    from experiments.baseline_delivery.readiness import policy_check,MODELS
    root=Path(root).resolve();suite=Path(suite).resolve()
    if not suite.is_relative_to(root/'runs'): raise ValueError('Invalid repair directory')
    source=revision();implementation,policy=policy_check()
    release=read(suite/'runtime-release.json');gate=read(suite/'gate-report.json')
    evidence=read(suite/'gate-evidence.json');context=read(suite/'context.json')
    manifest=read(suite/'suite.json')
    if (gate.get('status')!='passed' or gate.get('evidence_sha256')!=digest(suite/'gate-evidence.json')
            or evidence.get('context')!=context or any(item.get('source_revision')!=source
            for item in (release,gate,context,manifest))): raise ValueError('Repair admission identity changed')
    for item in (context,release):
        if item.get('implementation_sha256')!=implementation or item.get('policy_sha256')!=policy:
            raise ValueError('Repair policy or implementation changed')
    if set(release.get('certificates',{}))!=set(MODELS): raise ValueError('Repair certificate cohort incomplete')

    def confined(relative):
        path=(root/relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(suite):
            raise ValueError('Repair artifact outside admitted suite')
        return path

    expected={}
    for name,record in release['certificates'].items():
        path=confined(record['path'])
        if digest(path)!=record['sha256']: raise ValueError('Repair certificate changed')
        certificate=read(path)
        if (certificate!=evidence.get('certificates',{}).get(name) or certificate.get('model')!=name
                or certificate.get('policy_sha256')!=policy): raise ValueError('Repair certificate identity changed')
        for artifact,sha in certificate['evidence'].items():
            target=confined(artifact)
            if target in expected and expected[target]!=sha: raise ValueError('Conflicting certificate artifacts')
            expected[target]=sha

    jobs={job['id']:job for job in manifest['jobs']};candidates={}
    for record in evidence.get('acceleration_admissions',[]):
        name=record['job']
        if name not in jobs or Path(name).name!=name: raise ValueError('Repair admission job invalid')
        target=(suite/'jobs'/name/'acceleration-admission.json').resolve()
        if not target.is_relative_to(suite/'jobs') or target in candidates:
            raise ValueError('Repair admission path invalid')
        candidates[target]=record['report']
    pending=[]
    for target,sha in expected.items():
        before=digest(target)
        if before==sha: continue
        if target not in candidates: raise ValueError('Non-admission runtime evidence changed')
        value=candidates[target];config=read(target.parent/'config.json')
        if (fingerprint(config)!=value.get('config_sha256')
                or fingerprint(config)!=jobs[target.parent.name]['config_sha256']
                or value.get('selected_mode')!=config.get('torch_acceleration',{}).get('mode')):
            raise ValueError('Repair admission config changed')
        payload=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
        if hashlib.sha256(payload).hexdigest()!=sha: raise ValueError('Sealed admission bytes do not match certificate')
        pending.append((target,sha,before,payload))
    if not pending: return None

    folder=suite/('entry-repair-'+uuid.uuid4().hex[:12]);folder.mkdir()
    records=[]
    # Keep all overwritten originals before replacing any file.
    for target,sha,before,payload in pending:
        if digest(target)!=before: raise ValueError('Admission changed during repair')
        backup=folder/(target.parent.name+'.before.json')
        with backup.open('xb') as handle:
            handle.write(target.read_bytes());handle.flush();os.fsync(handle.fileno())
        if digest(backup)!=before: raise ValueError('Repair backup hash mismatch')
        records.append(dict(job=target.parent.name,restored_sha256=sha,overwritten_sha256=before,
            backup=backup.relative_to(suite).as_posix()))
    for target,sha,before,payload in pending:
        if digest(target)!=before: raise ValueError('Admission changed during repair')
        temporary=folder/(target.parent.name+'.restoring')
        with temporary.open('xb') as handle:
            handle.write(payload);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,target)
        if digest(target)!=sha: raise ValueError('Restored admission hash mismatch')
    atomic_json(folder/'entry-repair.json',dict(schema='qkan-admitted-entry-repair.v1',
        source_revision=source,gate_evidence_sha256=gate['evidence_sha256'],records=records,
        scope='Only admission bytes restored to original certificate hashes; full runtime validation follows'))
    print('Restored original admission evidence: '+', '.join(item['job'] for item in records),flush=True)
    return folder


def return_repair(folder):
    from experiments.baseline_delivery.return_artifacts import upload_declared
    try: upload_declared(folder,'admitted-entry',['entry-repair.json'])
    except Exception as error:
        print('Entry repair return pending ('+type(error).__name__+'); local receipt retained.',flush=True)


def failure_report(error,stage):
    """Code locations and fixed stage labels only: no exception values/locals."""
    frames=[];tb=error.__traceback__
    while tb is not None:
        path=Path(tb.tb_frame.f_code.co_filename)
        if path.name=='run_admitted.py': name='scripts/run_admitted.py'
        elif path.parent.name=='baseline_delivery': name='experiments/baseline_delivery/'+path.name
        else: name=None
        if name: frames.append(dict(file=name,line=tb.tb_lineno))
        tb=tb.tb_next
    return dict(schema='qkan-admitted-entry-failure.v1',stage=stage if stage in STAGES else 'imports',
        error_type=type(error).__name__,project_frames=frames[-12:],
        privacy='Exception text, source lines, locals, paths to private assets and tensor values excluded')


def return_failure(root,suite,report):
    from experiments.baseline_delivery.common import atomic_json
    from experiments.baseline_delivery.return_artifacts import upload_declared
    suite=Path(suite).resolve();root=Path(root).resolve()
    if not suite.is_relative_to(root/'runs'): raise ValueError('Invalid failure directory')
    folder=suite/('entry-failure-'+uuid.uuid4().hex[:12])
    atomic_json(folder/'entry-failure.json',report)
    atomic_json(suite/'entry-status.json',dict(report,state='failed',updated_at=time.time(),pid=os.getpid(),
        diagnostic=folder.relative_to(suite).as_posix()+'/entry-failure.json'))
    print('Entry failure: stage='+report['stage']+' type='+report['error_type']+'; '+str(folder/'entry-failure.json'),flush=True)
    try: upload_declared(folder,'admitted-entry',['entry-failure.json'])
    except Exception as error:
        print('Entry failure return pending ('+type(error).__name__+'); local diagnostic retained.',flush=True)


def run(root,suite,progress=lambda _:None):
    root=Path(root).resolve();suite=Path(suite).resolve()
    if not suite.is_relative_to(root/'runs'): raise ValueError('Suite must be under executor runs/')
    sys.path.insert(0,str(root));progress('imports')
    from experiments.baseline_delivery.common import read,digest,fingerprint,lock,revision,child_spec,run_child
    from experiments.baseline_delivery.readiness import validate_runtime_release
    from experiments.baseline_delivery.runner import execute,discover_gpus
    from experiments.baseline_delivery.sync import verify_data
    from experiments.baseline_delivery.bootstrap import check_exchange
    from experiments.baseline_delivery.migration import admit,plan_for
    progress('executor')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():
        raise ValueError('Executor tracked files changed')
    with lock(suite/'suite.lock'):
        progress('admission')
        context=read(suite/'context.json');manifest=read(suite/'suite.json');gate=read(suite/'gate-report.json')
        source=revision()
        if (gate.get('status')!='passed' or any(value.get('source_revision')!=source
                for value in (context,manifest,gate))): raise ValueError('Suite not admitted for this executor')
        if gate.get('evidence_sha256')!=digest(suite/'gate-evidence.json'):
            raise ValueError('Admission evidence changed')
        evidence=read(suite/'gate-evidence.json')
        if evidence.get('context')!=context: raise ValueError('Admission context changed')
        os.environ['QKAN_RUNTIME_RELEASE']=str(suite/'runtime-release.json')
        progress('certificate')
        try: validate_runtime_release(read(suite/'runtime-release.json'),None)
        except (ValueError,FileNotFoundError):
            progress('repair');repair=restore_admission_records(root,suite)
            if repair is None: raise
            progress('certificate');validate_runtime_release(read(suite/'runtime-release.json'),None)
            return_repair(repair)
        progress('exchange')
        check_exchange()
        data=Path(manifest['data']);cache=Path(context['cache'])
        progress('data');verify_data(data)
        progress('gpu')
        gpus,inventory=discover_gpus(context['gpus'])
        if inventory!=manifest['gpu_inventory']: raise ValueError('Admitted GPU identity changed')
        progress('configs');configs={}
        for job in manifest['jobs']:
            config=read(suite/'jobs'/job['id']/'config.json')
            if fingerprint(config)!=job['config_sha256']: raise ValueError('Admitted job config changed')
            configs[job['id']]=config
        # Existing receipts are validated; none are regenerated if present.
        progress('migration');plan=plan_for(suite)
        if plan:
            for name,record in plan['jobs'].items():
                if record.get('checkpoint') and not (suite/'jobs'/name/'migration-validation.json').is_file():
                    raise ValueError('Admitted migration receipt missing')
        admit(suite,configs,data,cache,gpus,run_child,child_spec,inventory)
        print('Using admitted configurations; preserving backend selections and completed results.',flush=True)
        progress('scheduler')
        execute(suite,gpus,cache,data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--resume',type=Path,required=True)
    args=parser.parse_args()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',QKAN_MEGA_SOURCE=str(args.root/'.bootstrap/strong-upstream/mega'),
        QKAN_S5_SOURCE=str(args.root/'.bootstrap/strong-upstream/s5'))
    def pause(*_): raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,pause)
    if hasattr(signal,'SIGHUP'): signal.signal(signal.SIGHUP,pause)
    stage='imports'
    def progress(value):
        nonlocal stage
        stage=value
        from experiments.baseline_delivery.common import atomic_json
        atomic_json(args.resume/'entry-status.json',dict(state='checking' if value!='scheduler' else 'running',
            stage=value,updated_at=time.time(),pid=os.getpid()))
        print('Admitted entry stage: '+value,flush=True)
    try: run(args.root,args.resume,progress)
    except KeyboardInterrupt: return 75
    except Exception as error:
        report=failure_report(error,stage)
        try: return_failure(args.root,args.resume,report)
        except Exception:
            print('Admitted entry stopped: stage='+report['stage']+' type='+report['error_type'],flush=True)
        return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
