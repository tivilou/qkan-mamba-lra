"""Bound formal BF16 Graph continuation; automatic actual-worker recovery gate."""
import argparse
from contextlib import ExitStack
import copy
import datetime as dt
import os
from pathlib import Path
import signal
import sys
import uuid

from .common import ROOT, atomic_json, digest, fingerprint, read, revision, lock, worker_failure
from . import key_bias_formal as original
from . import key_bias_training as strategy
from . import bf16_graph_diagnostic as diagnostic
from .bf16_graph_training import POLICY
from .migration import checked_pointer, completed_files, copy_verified, verify_completed

TARGET = 'experiments.baseline_delivery.bf16_graph_formal'
WORKER = 'experiments.baseline_delivery.bf16_graph_worker'
SCHEMA = 'qkan-bf16-graph-formal-input.v1'
REPORT_SCHEMA = 'qkan-bf16-graph-worker-preflight.v1'
MODES = ('reference', 'continuous', 'interrupt', 'resume')
JOB = diagnostic.JOB
PLAN = ROOT/'docs/baselines/bf16-graph-formal-plan.json'
REVIEW = ROOT/'docs/evidence/bf16-graph-native-review-20261118.json'


def binding():
    paths = [Path(__file__).parent/n for n in ('bf16_graph_formal.py', 'bf16_graph_worker.py',
        'bf16_graph_training.py', 'key_bias_formal.py')]
    paths += [ROOT/'scripts/experiment.py', ROOT/'scripts/formal_progress.py', PLAN, REVIEW]
    return dict(files={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
                inherited_diagnostic=diagnostic.binding(), arithmetic_strategy=strategy.binding())


def bind_diagnostic(folder):
    path = Path(folder).resolve()/'bf16-graph.json'; review = read(REVIEW)
    if (path.is_symlink() or path.stat().st_size != review['source_report']['bytes']
            or digest(path) != review['source_report']['sha256']):
        raise ValueError('Use the already audited immutable BF16 Graph report')
    value = read(path)
    if (not review['native_passed'] or value['passed'] is not True
            or value['source_revision'] != review['source_revision']
            or value['implementation'] != diagnostic.binding()
            or value['source_checkpoint']['sha256'] != review['source_checkpoint']['sha256']
            or not value['original_checkpoint_and_metadata_unchanged']
            or any(value[k] != v for k,v in diagnostic.CLOSED.items())):
        raise ValueError('Reviewed native Graph/source binding differs')
    return dict(suite=str(path.parent), sha256=digest(path), review_sha256=digest(REVIEW),
                gpu_identity=value['gpu_identity'], reviewed_source_checkpoint=review['source_checkpoint'])


def verify(path, ready=False):
    path = original.confined(path, ROOT/'runs'); suite = path.parent; r = read(path)
    if (not suite.name.startswith('bf16-graph-formal-') or r.get('schema') != SCHEMA
            or r['source_revision'] != revision() or r['execution_binding'] != binding()
            or r['execution_policy'] != POLICY or r['implementation'] != strategy.binding()
            or r['diagnostic'] != bind_diagnostic(r['diagnostic']['suite'])
            or r['suite_manifest_sha256'] != digest(suite/'suite.json')
            or r['snapshot_sha256'] != digest(suite/'migration-source/plan.json')):
        raise ValueError('Graph formal source/code/manifest/diagnostic changed')
    manifest = read(suite/'suite.json')
    if (len(manifest['jobs']) != 25 or {j['id'] for j in manifest['jobs']} != original.JOBS
            or manifest['models'] != list(original.MODELS) or manifest['tasks'] != list(original.TASKS)
            or manifest['seeds'] != [42] or manifest['data'] != r['data']):
        raise ValueError('Frozen formal roster changed')
    for job in manifest['jobs']:
        folder = original.confined(suite/'jobs'/job['id'], suite)
        if fingerprint(read(folder/'config.json')) != job['config_sha256']:
            raise ValueError('Graph fork changed a scientific job configuration')
        if job.get('preserved_completion'): verify_completed(suite, job)
    if read(suite/'jobs'/JOB/'config.json') != r['target_config'] or not strategy.select(r['target_config']):
        raise ValueError('Original BF16 arithmetic strategy required')
    if r['expected_strategy'] != strategy.envelope(r['target_config'], r['checkpoint']['sha256'], r['implementation']):
        raise ValueError('Original scientific strategy lineage changed')
    if ready:
        receipt = read(suite/'graph-ready.json'); report_path = suite/'preflight/preflight.json'
        if (receipt.get('passed') is not True or receipt['input_sha256'] != digest(path)
                or receipt['preflight_sha256'] != digest(report_path)
                or receipt['initial_checkpoint_sha256'] != r['source_checkpoint']['sha256']):
            raise ValueError('Actual-worker Graph preparation incomplete')
        validate_preflight(read(report_path), r, digest(path))
    return r


def prepare(args, suite):
    source = args.source_suite.resolve()
    bound = diagnostic.bind_source(source, args.cache)
    inherited = read(source/'formal-input.json'); native = bind_diagnostic(args.diagnostic_from)
    cp = bound['checkpoint']; reviewed = native['reviewed_source_checkpoint']
    if (bound['source_executor_revision'] != read(REVIEW)['source_executor_revision']
            or cp['contract'] != reviewed['contract'] or cp['step'] < reviewed['step']):
        raise ValueError('Use actual latest compatible paused BF16 source, never an older checkpoint')
    original.original_executor(inherited)
    # No arithmetic/configuration/selection transition: copy existing full state.
    source_manifest = read(source/'suite.json'); source_status = read(source/'status.json')
    jobs = []; states = {}; snapshot = {}; hashes = dict(bound['source_metadata_sha256'])
    for entry in source_manifest['jobs']:
        name = entry['id']; src = source/'jobs'/name; dst = suite/'jobs'/name
        job = copy.deepcopy(entry); config = read(src/'config.json')
        if source_status[name]['state'] == 'computed':
            complete = completed_files(src, config); copy_verified(src, dst, complete['files'])
            job.update(preserved_completion=complete, execution_role='preserved_result', execution_root=None)
            snapshot[name] = dict(checkpoint=None, completion=complete)
            states[name] = dict(state='computed', preserved=True)
            hashes.update({str(src/n):h for n,h in complete['files'].items()})
        else:
            if source_status[name]['state'] not in ('pending', 'paused'):
                raise ValueError('Source has an active/failed job; resolve it before a formal fork')
            copy_verified(src, dst, {'config.json':digest(src/'config.json')})
            pointer = None
            if (src/'checkpoints/latest.json').is_file():
                pointer, generation = checked_pointer(src)
                copy_verified(src, dst, {'checkpoints/'+generation.name:pointer['sha256']})
                atomic_json(dst/'checkpoints/latest.json', {**pointer, 'previous':[]})
                hashes[str(src/'checkpoints/latest.json')] = digest(src/'checkpoints/latest.json')
                hashes[str(generation)] = pointer['sha256']
            if name == JOB:
                if pointer is None or any(pointer[k] != cp[k] for k in ('sha256','step','contract')):
                    raise ValueError('Latest candidate source changed while copying')
                job.update(execution_revision=revision(), execution_root=str(ROOT), execution_role='bf16_graph_worker')
            snapshot[name] = dict(checkpoint=pointer, completion=None)
            states[name] = dict(state='paused' if name == JOB else 'pending', started_training=False)
        jobs.append(job)
    manifest = dict(source_manifest, source_revision=revision(), jobs=jobs,
                    execution='BF16 Graph candidate fork; preserved original arithmetic configs and other executors')
    atomic_json(suite/'suite.json', manifest); atomic_json(suite/'status.json', states)
    atomic_json(suite/'migration-source/plan.json', dict(schema='qkan-graph-checkpoint-copy.v1', jobs=snapshot,
        source_suite=str(source), optimizer_updates=0, scientific_configuration_changes=0))
    r = dict(schema=SCHEMA, source_revision=revision(), execution_binding=binding(), execution_policy=POLICY,
        diagnostic=native, origin_suite=str(source), origin_revision=bound['source_executor_revision'],
        origin_metadata_sha256=hashes, source_checkpoint=cp,
        source_suite=inherited['source_suite'], original_root=inherited['original_root'],
        target_config=bound['source_config'], data=bound['data'], cache=bound['cache'],
        checkpoint=inherited['checkpoint'], implementation=inherited['implementation'],
        expected_strategy=strategy.envelope(bound['source_config'], inherited['checkpoint']['sha256'], inherited['implementation']),
        inherited_formal_input_sha256=digest(source/'formal-input.json'),
        snapshot_sha256=digest(suite/'migration-source/plan.json'), suite_manifest_sha256=digest(suite/'suite.json'))
    atomic_json(suite/'graph-input.json', r)
    return r


def unchanged(r):
    if (digest(r['source_checkpoint']['path']) != r['source_checkpoint']['sha256']
            or any(digest(p) != sha for p,sha in r['origin_metadata_sha256'].items())):
        raise ValueError('Paused origin source changed; do not restart it alongside the Graph fork')


def child_spec(suite, mode, output=None):
    if mode not in (*MODES, 'formal'): raise ValueError('Unknown actual-worker mode')
    folder = Path(output) if output else suite/'jobs'/JOB
    return dict(target=WORKER, cwd=str(ROOT), argv=[sys.executable, '-B', '-m', WORKER,
        '--request',str(suite/'graph-input.json'),'--output',str(folder),'--mode',mode])


def accept(result, r, request_sha, mode):
    expected = 2 if mode in ('reference','continuous','resume') else 1
    if (result.get('schema') != REPORT_SCHEMA or result.get('passed') is not True
            or result.get('mode') != mode or result.get('request_sha256') != request_sha
            or result.get('execution_binding') != r['execution_binding']
            or result.get('source_checkpoint_sha256') != r['source_checkpoint']['sha256']
            or result.get('step') != r['source_checkpoint']['step']+expected
            or len(result.get('updates',[])) != expected
            or result.get('formal_training_started') is not False or result.get('training_samples_committed') != 0
            or result.get('worker_exit_status') != 75 or result.get('copy_roundtrip_exact') is not True
            or any(v['micro_batch'] != 16 or v['samples'] != 128 for v in result['updates'])):
        raise ValueError('Actual Graph worker receipt/identity/update budget differs')
    return result


def agreement(results):
    reference, full, part, resumed = (results[n] for n in MODES)
    if (full['scientific_state_sha256'] != resumed['scientific_state_sha256']
            or full['component_sha256'] != resumed['component_sha256'] or full['updates'] != resumed['updates']
            or part['updates'] != resumed['updates'][:1]
            or resumed['loaded_state_sha256'] != part['scientific_state_sha256']
            or len({v['loaded_state_sha256'] for v in (reference, full, part)}) != 1
            or resumed.get('same_worker_parity',{}).get('passed') is not True
            or resumed.get('trace',{}).get('validation',{}).get('passed') is not True):
        raise ValueError('Actual Graph worker reference/recovery/trace gate rejected')
    for name in ('continuous','interrupt','resume'):
        proof = results[name]['execution']
        count = 1 if name in ('interrupt','resume') else 2
        if (proof.get('enabled') is not True or proof['worker_updates'] != count or proof['full_graph_micros'] != 8*count
                or proof['eager_tail_micros'] != 0 or proof['runtime']['graph_builds'] != 1
                or proof['runtime'].get('graph_resident') is not True
                or proof['runtime'].get('forward_replays') != 16*count
                or proof['runtime'].get('backward_replays') != 16*count):
            raise ValueError('Actual worker did not use resident Graph for all finite micros')
    if reference['execution'].get('enabled') is not False or reference['execution']['full_graph_micros'] != 0:
        raise ValueError('Actual same-worker reference must use original BF16 eager execution')
    return dict(passed=True, same_worker_parity=resumed['same_worker_parity'], fresh_process=True,
        continuous_updates=2, interrupted_updates=[1,1], component_sha256=full['component_sha256'],
        trace=resumed['trace'], full_state_sha256=full['scientific_state_sha256'])


def validate_preflight(report, r, request_sha):
    if (report.get('passed') is not True or report.get('schema') != REPORT_SCHEMA
            or report.get('request_sha256') != request_sha or report.get('execution_binding') != r['execution_binding']
            or report.get('original_checkpoint_and_metadata_unchanged') is not True
            or set(report['results']) != set(MODES) or len(report['child_executions']) != 4):
        raise ValueError('Actual-worker preflight not complete')
    for name, child in zip(MODES, report['child_executions']):
        accept(report['results'][name],r,request_sha,name)
        if (child['exit_code'] != 0 or child['outcome'] != 'exited'
                or child['termination'] != dict(parent_reaped=True, group_verified=True, remaining_pids=[])):
            raise ValueError('Preflight child or process group did not finish')
    if agreement(report['results']) != report['recovery']: raise ValueError('Preflight agreement changed')
    if report.get('disposable_checkpoints',{}).get('retained_generations') != 0:
        raise ValueError('Disposable worker generations remain before formal start')


def publish_event(suite, r, event):
    from .return_artifacts import upload_declared
    folder = suite/'events'/(event+'-'+uuid.uuid4().hex[:12]); folder.mkdir(parents=True)
    try: unchanged(r); origin_unchanged=True
    except (OSError,ValueError): origin_unchanged=False
    value = dict(schema='qkan-bf16-graph-formal-status.v1', event=event, suite=str(suite),
        source_revision=r['source_revision'], input_sha256=digest(suite/'graph-input.json'),
        execution_binding=r['execution_binding'], execution_policy=POLICY,
        source_checkpoint={k:r['source_checkpoint'][k] for k in ('sha256','contract','step')},
        jobs=read(suite/'status.json'), original_results_preserved=2,
        formal_training_started=any(v.get('started_training') for v in read(suite/'status.json').values()),
        original_checkpoint_and_metadata_unchanged=origin_unchanged)
    if (suite/'graph-ready.json').exists(): value['readiness'] = read(suite/'graph-ready.json')
    if (suite/'worker-error.json').exists(): value['failure_locations'] = read(suite/'worker-error.json')
    if (suite/'preflight/preflight.json').exists(): value['preflight'] = read(suite/'preflight/preflight.json')
    for job in read(suite/'suite.json')['jobs']:
        folder_job = suite/'jobs'/job['id']
        if (folder_job/'summary.json').exists():
            summary = read(folder_job/'summary.json')
            value.setdefault('results',{})[job['id']] = {k:summary[k] for k in
                ('source_revision','config_sha256','best_epoch','validation','test','optimizer_steps','training_seconds') if k in summary}
    atomic_json(folder/'graph-status.json', value)
    upload_declared(folder,'bf16-graph-formal',['graph-status.json'],run_id=suite.name+'--'+folder.name)


def preflight(suite, r, gpu):
    from .diagnostic_watch import Supervisor
    from .return_artifacts import upload_declared
    folder = suite/'preflight'; folder.mkdir(exist_ok=True); path = folder/'preflight.json'
    if path.exists() and read(path).get('passed') is True:
        report = read(path); validate_preflight(report,r,digest(suite/'graph-input.json'))
        upload_declared(folder,'bf16-graph-worker',['preflight.json'],run_id=suite.name+'--preflight')
    else:
        if path.exists() or any((folder/name/'checkpoints').exists() for name in MODES):
            raise ValueError('Incomplete preflight retained; inspect its compact report before retry')
        supervisor = Supervisor(900,300)
        report = dict(schema=REPORT_SCHEMA, passed=False, request_sha256=digest(suite/'graph-input.json'),
            execution_binding=r['execution_binding'], results={}, child_executions=supervisor.records,
            formal_training_started=False, training_samples_committed=0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',
            PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',QKAN_DIAGNOSTIC_STAGE_FENCES='0')
        error = None
        try:
            for mode in MODES:
                output = folder/mode; print('Actual BF16 Graph worker preflight: '+mode,flush=True)
                atomic_json(folder/'progress.json',dict(mode=mode,completed=len(report['results'])))
                code = supervisor.run(child_spec(suite,mode,output),folder/(mode+'.log'),env)
                if (output/'worker-error.json').exists():
                    report.setdefault('failure_locations',{})[mode] = read(output/'worker-error.json')
                if code != 0 or not (output/'result.json').exists(): raise RuntimeError('Actual Graph worker preflight child failed')
                report['results'][mode] = accept(read(output/'result.json'),r,digest(suite/'graph-input.json'),mode)
            report['recovery'] = agreement(report['results']); report['passed'] = True
        except BaseException as caught: error = caught
        finally:
            removed = []
            for mode in MODES:
                checkpoint = folder/mode/'checkpoints'
                for generation in checkpoint.glob('step-*.pt'):
                    if generation.is_symlink(): raise ValueError('Disposable checkpoint is a link')
                    removed.append(dict(file=str(generation.relative_to(folder)),sha256=digest(generation),bytes=generation.stat().st_size))
                    generation.unlink()
            report['disposable_checkpoints'] = dict(removed=removed,retained_generations=0)
            try: unchanged(r); report['original_checkpoint_and_metadata_unchanged'] = True
            except BaseException as caught:
                report.update(passed=False,original_checkpoint_and_metadata_unchanged=False)
                if error is None: error=caught
            atomic_json(path,report)
            try: upload_declared(folder,'bf16-graph-worker',['preflight.json'],run_id=suite.name+'--preflight')
            except Exception:
                if error is None: raise
                print('Preflight return pending; compact report and retry manifest retained.',flush=True)
        if error is not None: raise error
        validate_preflight(report,r,digest(suite/'graph-input.json'))
    atomic_json(suite/'graph-ready.json',dict(schema='qkan-bf16-graph-formal-ready.v1',passed=True,
        input_sha256=digest(suite/'graph-input.json'),preflight_sha256=digest(path),
        initial_checkpoint_sha256=r['source_checkpoint']['sha256'],formal_updates_before_ready=0))


def job_spec(suite, r, job):
    return child_spec(suite,'formal') if job['id'] == JOB else original.job_spec(suite,r,job)


def validate_result(suite, r, job):
    value = original.validate_job_result(suite,r,job)
    if job['id'] == JOB:
        execution = read(suite/'jobs'/JOB/'bf16-graph-execution.json')
        if not execution['enabled'] or execution['policy'] != POLICY or execution['implementation'] != r['execution_binding']:
            raise ValueError('Completed candidate Graph execution lineage missing')
    return value


def execute(args, suite):
    path = suite/'graph-input.json'
    source = Path(read(path)['origin_suite']) if path.exists() else args.source_suite.resolve()
    if not (source/'formal-input.json').is_file(): raise ValueError('Existing original formal source required')
    with ExitStack() as held:
        # Both original and legacy executors are fenced for the lifetime of this
        # fork. The command never pauses/kills them; active locks reject it.
        inherited = read(source/'formal-input.json')
        for origin in dict.fromkeys((str(source),inherited['source_suite'])):
            for p in (Path(origin)/'progress-supervisor.lock',Path(origin)/'suite.lock',
                      *[Path(origin)/'jobs'/name/'worker.lock' for name in sorted(original.JOBS)]):
                held.enter_context(lock(p))
        r = verify(path) if path.exists() else prepare(args,suite)
        unchanged(r)
        if args.cache and str(args.cache.resolve()) != r['cache']: raise ValueError('Existing source cache required')
        if args.upload_only:
            from .return_artifacts import upload_declared
            for event in sorted((suite/'events').glob('*')):
                if (event/'graph-status.json').is_file():
                    upload_declared(event,'bf16-graph-formal',['graph-status.json'],run_id=suite.name+'--'+event.name)
            if (suite/'preflight/preflight.json').is_file():
                upload_declared(suite/'preflight','bf16-graph-worker',['preflight.json'],run_id=suite.name+'--preflight')
            if (suite/'compute-complete.json').exists():
                from .runner import deliver
                deliver(suite)
            return 0
        original.original_executor(r)
        from .runner import discover_gpus
        from .attention_diagnostic import gpu_preflight
        gpus,inventory = discover_gpus(args.gpu)
        if len(gpus) != 1 or inventory[gpus[0]] != r['diagnostic']['gpu_identity']:
            raise ValueError('Use the reviewed single physical GPU')
        gpu = gpus[0]
        if not gpu_preflight(inventory[gpu])['idle']: raise ValueError('GPU busy; existing jobs preserved')
        if not (suite/'graph-ready.json').exists(): preflight(suite,r,gpu)
        verify(path,ready=True); publish_event(suite,r,'ready')
        if args.prepare_only: return 0
        statuses = read(suite/'status.json')
        for job in read(suite/'suite.json')['jobs']:
            folder = suite/'jobs'/job['id']
            if (folder/'summary.json').exists():
                validate_result(suite,r,job); statuses[job['id']]['state'] = 'computed'; continue
            if statuses[job['id']]['state'] == 'computed': raise ValueError('Computed job lost summary')
            code = original.run_job(suite,r,job,gpu,statuses,spec_builder=job_spec,result_validator=validate_result)
            publish_event(suite,r,'job-complete' if code == 0 else 'job-paused' if code == 75 else 'job-failed')
            if code != 0: return 75 if code == 75 else 2
        atomic_json(suite/'compute-complete.json',dict(state='computed',jobs=25))
        publish_event(suite,r,'computed')
        from .runner import deliver
        deliver(suite)
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__); group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--source-suite',type=Path); group.add_argument('--resume',type=Path)
    p.add_argument('--diagnostic-from',type=Path); p.add_argument('--cache',type=Path)
    p.add_argument('--gpu',default='0'); p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--upload-only',action='store_true'); args = p.parse_args()
    if args.source_suite and (args.diagnostic_from is None or args.cache is None): p.error('New fork requires diagnostic-from/cache')
    if args.upload_only and (args.resume is None or args.prepare_only): p.error('Upload-only requires resume')
    from .bootstrap import check_exchange
    check_exchange(); revision()
    if args.resume and not (args.resume/'graph-input.json').is_file(): raise ValueError('Existing Graph Suite required')
    suite = original.confined(args.resume,ROOT/'runs') if args.resume else ROOT/'runs'/(
        'bf16-graph-formal-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    suite.mkdir(parents=True,exist_ok=bool(args.resume)); print('Formal continuation Suite: '+str(suite),flush=True)
    def interrupted(*_): raise KeyboardInterrupt
    for name in ('SIGTERM','SIGINT','SIGHUP'):
        if hasattr(signal,name): signal.signal(getattr(signal,name),interrupted)
    with lock(suite/'suite.lock'):
        try: return execute(args,suite)
        except BaseException as error:
            worker_failure(suite,error)
            try:
                if (suite/'graph-input.json').is_file(): publish_event(suite,read(suite/'graph-input.json'),'paused' if isinstance(error,KeyboardInterrupt) else 'failed')
                else:
                    from .return_artifacts import upload_declared
                    upload_declared(suite,'bf16-graph-formal',['worker-error.json'],run_id=suite.name+'--early-failure')
            except Exception: print('Return pending; compact report/retry manifest retained: '+str(suite),flush=True)
            return 75 if isinstance(error,KeyboardInterrupt) else 2


if __name__ == '__main__': raise SystemExit(main())
