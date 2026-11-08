"""Read-only progress calculations, no model, device or process execution."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.baseline_delivery.common import atomic_json
from scripts.formal_progress import Throughput,speed_lines,backend_text,Dashboard


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


if __name__=='__main__': unittest.main()
