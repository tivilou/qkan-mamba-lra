"""Actual stdlib child lifecycle and phase/stack tests; no model framework."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

from experiments.baseline_delivery import diagnostic_watch as watch
from experiments.baseline_delivery.common import ROOT,atomic_json,read


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.out=self.root/'worker';self.out.mkdir()
    def tearDown(self): self.temp.cleanup()
    def spec(self,mode):
        target='tests.test_baseline_diagnostic_watch'
        return dict(target=target,cwd=str(ROOT),argv=[sys.executable,'-B','-m',target,
            '--fixture',mode,'--output',str(self.out)])
    def run_child(self,mode,child_timeout=5,stall_timeout=2):
        supervisor=watch.Supervisor(child_timeout,stall_timeout,announce=lambda *a,**k:None)
        terminate=watch.terminate
        with patch.object(watch,'terminate',side_effect=lambda child:terminate(child,grace=.1)):
            code=supervisor.run(self.spec(mode),self.root/'worker.log',
                dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        return supervisor,code

    def test_actual_parent_bootstrap_worker_argv_cwd_and_return(self):
        supervisor,code=self.run_child('success')
        self.assertEqual(code,0);self.assertEqual(len(supervisor.records),1)
        result=read(self.out/'fixture-result.json')
        self.assertTrue(result['instrumented']);self.assertEqual(Path(result['cwd']),ROOT)
        record=supervisor.records[0];self.assertEqual(record['last_progress']['phase'],'fixture_ready')
        command=read(next((self.out/'diagnostics').glob('*/command.json')))
        self.assertEqual(command['worker'],self.spec('success'))
        self.assertEqual(command['launch']['argv'][3],'experiments.baseline_delivery.diagnostic_watch')
        self.assertTrue(record['termination']['parent_reaped'])

    def test_stale_phase_times_out_with_frames_and_cleanup(self):
        with self.assertRaises(watch.DiagnosticFailure) as caught: self.run_child('stall',stall_timeout=1)
        record=caught.exception.diagnostic
        self.assertEqual(record['outcome'],'phase_timeout')
        self.assertEqual(record['last_progress']['phase'],'fixture_ready')
        self.assertTrue(record['termination']['parent_reaped'])
        if sys.platform.startswith('linux'):
            self.assertTrue(record['termination']['group_verified'])
            self.assertTrue(record['signal_stack_frames'])
        self.assertNotIn('secret-local-value',json.dumps(record))

    def test_active_heartbeat_does_not_reset_phase_timeout(self):
        with self.assertRaises(watch.DiagnosticFailure) as caught: self.run_child('heartbeat',stall_timeout=1)
        self.assertEqual(caught.exception.diagnostic['outcome'],'phase_timeout')

    def test_advancing_phases_still_obey_absolute_child_limit(self):
        with self.assertRaises(watch.DiagnosticFailure) as caught:
            self.run_child('advancing',child_timeout=1.5,stall_timeout=1)
        self.assertEqual(caught.exception.diagnostic['outcome'],'child_timeout')

    @unittest.skipUnless(sys.platform.startswith('linux'),'Native Linux group/descendant verification')
    def test_term_ignoring_descendant_is_killed_with_scoped_group(self):
        with self.assertRaises(watch.DiagnosticFailure) as caught: self.run_child('descendant',stall_timeout=1)
        record=caught.exception.diagnostic
        self.assertTrue(record['termination']['escalated'])
        self.assertTrue(record['termination']['group_verified'])
        self.assertEqual(record['termination']['remaining_pids'],[])

    def test_nonzero_exit_is_not_transport_or_scientific_success(self):
        supervisor,code=self.run_child('exit75')
        self.assertEqual(code,75);self.assertEqual(supervisor.records[0]['exit_code'],75)

    def test_group_cleanup_checks_descendants_after_parent_already_exited(self):
        child=Mock(pid=1234);child.poll.return_value=0
        with patch.object(watch.sys,'platform','linux'),patch.object(watch.os,'name','posix'),\
             patch.object(watch.signal,'SIGKILL',9,create=True),\
             patch.object(watch.os,'killpg',create=True) as kill,\
             patch.object(watch,'process_group_members',side_effect=[[5678],[],[]]):
            result=watch.terminate(child,grace=0)
        self.assertEqual([c.args for c in kill.call_args_list],[(1234,signal.SIGTERM),(1234,9)])
        self.assertTrue(result['group_verified']);self.assertTrue(result['escalated'])

    def test_broken_parent_output_cleans_up_before_propagating(self):
        supervisor=watch.Supervisor(5,2,announce=lambda *a,**k:(_ for _ in ()).throw(BrokenPipeError()))
        terminate=watch.terminate
        with patch.object(watch,'terminate',side_effect=lambda child:terminate(child,grace=.1)):
            with self.assertRaises(BrokenPipeError): supervisor.run(self.spec('stall'),self.root/'worker.log')
        self.assertTrue(supervisor.records[0]['termination']['parent_reaped'])

    def test_recorder_ring_and_stack_payload_are_bounded_and_private(self):
        recorder=watch.Recorder(self.root/'progress','a'*32)
        for _ in range(200): recorder.mark('fixture_ready')
        recorder.publish();value=read(recorder.directory/'progress.json')
        self.assertEqual(len(value['events']),64);self.assertEqual(value['progress_sequence'],200)
        raw=self.root/'stack.txt'
        raw.write_text('Thread 0x123:\n  File "/private/user/torch/cuda/graphs.py", line 123 in capture\nsecret-local-value\n')
        frames=watch.signal_stacks(raw)
        self.assertEqual(frames,[dict(file='torch/cuda/graphs.py',line=123,function='capture')])
        self.assertNotIn('/private',json.dumps(frames))

    def test_invalid_limits_or_worker_spec_fail_before_launch(self):
        for value in (0,-1,float('inf'),float('nan'),True):
            with self.assertRaises(ValueError): watch.Supervisor(value,1)
        spec=self.spec('success');spec['target']='other'
        with patch.object(watch,'spawn') as spawn:
            with self.assertRaises(ValueError): watch.Supervisor().run(spec,self.root/'worker.log')
            spawn.assert_not_called()

    def test_failed_cleanup_is_reported_and_never_becomes_success(self):
        supervisor=watch.Supervisor(5,.5,announce=lambda *a,**k:None)
        original=watch.terminate
        def cleanup(child):
            original(child,grace=.1)
            raise OSError('injected cleanup failure')
        with patch.object(watch,'terminate',side_effect=cleanup):
            with self.assertRaises(watch.DiagnosticFailure) as caught:
                supervisor.run(self.spec('stall'),self.root/'worker.log')
        record=caught.exception.diagnostic
        self.assertFalse(record['termination']['group_verified'])
        self.assertEqual(record['termination']['error_type'],'OSError')
        self.assertTrue(list((self.out/'diagnostics').glob('*/child-execution.json')))


def fixture():
    parser=argparse.ArgumentParser();parser.add_argument('--fixture');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    # This module imports the canonical module name from the actual shim. It
    # must see the same recorder, even though the shim executed as __main__.
    assert watch._recorder is not None
    watch.mark('fixture_ready',flush=True)
    if args.fixture=='success':
        atomic_json(args.output/'fixture-result.json',dict(instrumented=True,cwd=os.getcwd()));return 0
    if args.fixture=='exit75': return 75
    if args.fixture=='descendant':
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        subprocess.Popen([sys.executable,'-B','-c',
            'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'])
    secret='secret-local-value'  # Must never appear in structured stack return.
    end=time.monotonic()+30
    while time.monotonic()<end:
        if args.fixture=='advancing': watch.mark('fixture_advancing',flush=True)
        if args.fixture=='heartbeat': watch._recorder.publish()
        time.sleep(.05)
    return len(secret)*0


if __name__=='__main__':
    if '--fixture' in sys.argv: raise SystemExit(fixture())
    unittest.main()
