"""Read-only progress calculations, no model, device or process execution."""
from pathlib import Path
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.baseline_delivery.common import atomic_json
from scripts.formal_progress import Throughput,speed_lines,backend_text,Dashboard,latest_formal,Terminal


class ProgressRateTests(unittest.TestCase):
    def test_repeated_frame_retains_short_speed_and_average_includes_wait(self):
        meter=Throughput();meter.observe(1000,10,10);meter.observe(1120,20,20)
        for now in (21,25,30):
            result=meter.observe(1120,20,now)
            self.assertEqual(result['short'],12)
            self.assertAlmostEqual(result['average'],120/(now-10))
            self.assertNotIn('待测','\n'.join(speed_lines(result,'samples/s',100)))

    def test_rates_depend_on_worker_timestamps_not_redraw_timing(self):
        meters=[Throughput(),Throughput()]
        for stamp,count in ((10,0),(20,100),(50,700)):
            meters[0].observe(count,stamp,stamp)
            meters[1].observe(count,stamp,stamp+3)
        self.assertEqual(meters[0].result(60),meters[1].result(60))
        self.assertEqual(meters[0].short,17.5)

    def test_sixty_second_window_drops_old_slow_history(self):
        meter=Throughput()
        for stamp,count in ((0,0),(30,30),(60,330),(90,630),(120,930)):
            result=meter.observe(count,stamp,stamp)
        self.assertEqual(result['short'],10)
        self.assertEqual(result['average'],930/120)

    def test_process_origin_excludes_checkpoint_prefix(self):
        meter=Throughput()
        result=meter.observe(502000,120,120,origin=(100,500000))
        self.assertEqual(result['average'],100);self.assertEqual(result['scope'],'本次进程')
        self.assertIsNone(result['short'])

    def test_old_worker_uses_observation_average_without_inventing_baseline(self):
        meter=Throughput();meter.observe(500000,100,100)
        result=meter.observe(500100,110,110)
        self.assertEqual(result['average'],10);self.assertEqual(result['scope'],'本次观察')

    def test_stale_measurement_is_retained_eta_suspended(self):
        meter=Throughput();meter.observe(0,1,1);meter.observe(100,11,11)
        result=meter.observe(100,11,140)
        lines='\n'.join(speed_lines(result,'samples/s',1000))
        self.assertIn('10.00 samples/s',lines);self.assertIn('心跳已过期',lines)
        self.assertIn('暂停估计',lines)

    def test_invalid_or_out_of_order_heartbeat_does_not_reset_valid_rate(self):
        meter=Throughput();meter.observe(0,1,1);meter.observe(100,11,11)
        for count,stamp in ((99,10),(999,float('nan')),(999,500),(-1,12),(None,12)):
            result=meter.observe(count,stamp,12)
            self.assertEqual(result['short'],10)

    def test_counter_reset_restarts_scope_and_zero_rate_is_valid(self):
        meter=Throughput();meter.observe(100,1,1);meter.observe(200,11,11)
        result=meter.observe(0,12,12);self.assertIsNone(result['short'])
        result=meter.observe(0,22,22);self.assertEqual(result['short'],0)
        self.assertEqual(result['average'],0)

    def test_fallback_reason_is_taken_from_admission(self):
        text=backend_text(dict(mode='eager',reason='requested'),dict(attempts=[dict(outcome='recovery_mismatch')]))
        self.assertIn('未通过恢复校验',text)
        self.assertIn('benchmark_gain',backend_text(dict(mode='compile',reason='benchmark_gain'),{}))
        graph=backend_text(dict(mode='compile_graph',reason='benchmark_gain'),dict(selected_backend='eager',required='graph'))
        self.assertIn('Dynamo原生autograd＋CUDA Graph',graph);self.assertIn('必须启用Graph',graph)

    def test_dashboard_wait_does_not_reset_or_launch_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);suite=root/'runs/test';folder=suite/'jobs/task';folder.mkdir(parents=True)
            atomic_json(suite/'suite.json',dict(jobs=[dict(id='task')]))
            atomic_json(suite/'status.json',dict(task=dict(state='running',started=1,pid=999999)))
            atomic_json(folder/'config.json',dict(epochs=100))
            heartbeat=dict(epoch=0,optimizer_step=1,samples_seen=100,sample_budget=10000,updated_at=10)
            atomic_json(folder/'heartbeat.json',heartbeat);dashboard=Dashboard(root,suite)
            with patch('scripts.formal_progress.subprocess.run',side_effect=AssertionError('Process execution forbidden')):
                dashboard.render(10)
                atomic_json(folder/'heartbeat.json',dict(heartbeat,samples_seen=200,updated_at=20))
                second=dashboard.render(20);waiting=dashboard.render(25)
            self.assertIn('10.00 samples/s',second);self.assertIn('10.00 samples/s',waiting)
            self.assertIn('6.67 samples/s',waiting);self.assertNotIn('待测',waiting)


class FormalDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.suite=self.make_suite('key-bias-formal-20261117',True)
        folder=self.suite/'jobs/transformer-retrieval-s42'
        atomic_json(folder/'config.json',dict(epochs=100,task='retrieval',training_strategy={'id':'bf16-key-bias-analytic-zero-v1'}))
        atomic_json(folder/'heartbeat.json',dict(epoch=15,optimizer_step=14527,
            samples_seen=1834386,sample_budget=11779000,updated_at=200,run_started_at=100,
            run_samples_start=1831030,loss=.311))

    def make_suite(self,name,running):
        suite=self.root/'runs'/name;suite.mkdir(parents=True)
        atomic_json(suite/'formal-input.json',{})
        atomic_json(suite/'suite.json',dict(jobs=[dict(id='transformer-retrieval-s42')]))
        atomic_json(suite/'status.json',{'transformer-retrieval-s42':dict(
            state='running' if running else 'paused',physical_gpu='0',started_training=True)})
        return suite

    def test_live_v1_without_parent_start_fields_uses_worker_origin(self):
        with patch('scripts.formal_progress.gpu_usage',return_value=({},'query fixture')):
            dashboard=Dashboard(self.root,self.suite)
            first=dashboard.render(200);second=dashboard.render(210)
        self.assertIn('实际 Step 14527',first);self.assertIn('Epoch 16/100',first)
        self.assertIn('33.56 samples/s',first);self.assertIn('30.51 samples/s',second)
        self.assertIn('本次进程耗时 00:01:40',first)
        self.assertIn('BF16 注意力',first);self.assertIn('不含后续验证/测试/回传及排队任务',first)
        self.assertNotIn('等待本次进程的第一个训练心跳',second)

    def test_watch_selection_prefers_one_running_over_newer_paused_suite(self):
        self.make_suite('key-bias-formal-20261118',False)
        self.assertEqual(latest_formal(self.root),self.suite)

    def test_watch_selection_rejects_ambiguous_running_suites(self):
        self.make_suite('key-bias-formal-20261118',True)
        with self.assertRaises(ValueError):latest_formal(self.root)

    def test_watch_selection_latest_paused_and_no_suites(self):
        atomic_json(self.suite/'status.json',{})
        newer=self.make_suite('key-bias-formal-20261118',False)
        self.assertEqual(latest_formal(self.root),newer)
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(ValueError):latest_formal(empty)

    def test_actual_snapshot_cli_never_changes_suite_or_imports_frameworks(self):
        script=Path(__file__).resolve().parents[1]/'scripts/formal_progress.py'
        before={p.relative_to(self.suite):p.read_bytes() for p in self.suite.rglob('*') if p.is_file()}
        # Load a standalone copy as the collaborator does from .bootstrap, while
        # blocking process launch, GPU queries and all training framework imports.
        code='''import importlib.abc,runpy,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('torch','numpy','jax','tensorflow'):raise AssertionError(name)
sys.meta_path.insert(0,Guard())
import subprocess
def blocked(*a,**kw):raise OSError('read-only snapshot fixture')
subprocess.Popen=blocked
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
        copy=self.root/'standalone.py';copy.write_bytes(script.read_bytes())
        result=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(copy),
            '--root',str(self.root),'--latest-formal','--watch-only','--once'],capture_output=True,text=True,encoding='utf8')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('实际 Step 14527',result.stdout);self.assertIn('只读观察',result.stdout)
        after={p.relative_to(self.suite):p.read_bytes() for p in self.suite.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_live_terminal_uses_one_screen_and_restores_cursor(self):
        stream=io.StringIO()
        with patch('scripts.formal_progress.sys.stdout',stream),patch.object(stream,'isatty',return_value=True), \
                patch.dict(os.environ,{'TERM':'xterm'}),patch('scripts.formal_progress.shutil.get_terminal_size',return_value=os.terminal_size((40,12))):
            terminal=Terminal('auto');terminal.show('进度 '*100);terminal.show('下一帧');terminal.close()
        output=stream.getvalue()
        self.assertEqual(output.count('\x1b[?1049h'),1)
        self.assertEqual(output.count('\x1b[H\x1b[J'),2)
        self.assertTrue(output.endswith('\x1b[?25h\x1b[?1049l'))

    def test_owned_dashboard_never_sends_a_liveness_signal(self):
        dashboard=Dashboard(self.root,self.suite)
        dashboard.launch=dict(runner_pid=os.getpid(),started_at=100,updated_at=200,interval=30)
        with patch('scripts.formal_progress.os.kill',side_effect=AssertionError('No signal permitted')), \
                patch('scripts.formal_progress.gpu_usage',return_value=({},'query fixture')):
            text=dashboard.render(200)
        self.assertIn('存活',text);self.assertIn('实际 Step 14527',text)


if __name__=='__main__': unittest.main()
