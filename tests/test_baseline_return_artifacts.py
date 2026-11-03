"""Exercise real delivery manifests/receipts with an isolated transport."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.baseline_delivery import return_artifacts as returns
from scripts.exchange_artifacts import digest_file, DeliveryError


class ReturnTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.folder=self.root/'runs/audit-one';self.folder.mkdir(parents=True)
        for name in ('report.json','summary.json','unlisted.json'):
            (self.folder/name).write_text('{}')
        self.uploads=[];self.bad_receipt=False;self.offline=False
        class Client:
            def request(inner,method,route,body=None,size=0):
                if self.offline: raise TimeoutError('connection unavailable')
                if route=='/healthz': return 200,dict(status='ok',max_bytes=1048576)
                if method=='POST': return 201,{}
                self.uploads.append(route)
                n,sha=digest_file(body)
                return 201,dict(path=route.removeprefix('/upload/'),size_bytes=n,
                                sha256='bad' if self.bad_receipt else sha)
        self.client=Client()
        self.patches=[patch.object(returns,'ROOT',self.root),patch.object(returns,'check_exchange'),
                      patch.object(returns,'Transport',return_value=self.client),
                      patch.object(returns.time,'sleep'),
                      patch.dict(returns.os.environ,PROJECT_EXCHANGE_URL='https://example.invalid',
                                 PROJECT_EXCHANGE_TOKEN='private-token',PROJECT_EXCHANGE_CA_FILE='ca.crt')]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def send(self):return returns.upload_declared(self.folder,'recovery-audit',['report.json','summary.json'])
    def test_declared_only_and_idempotent_retry(self):
        manifest=self.send();self.send()
        self.assertEqual(manifest['status'],'uploaded')
        self.assertEqual(manifest['target_dir'],'qkan-mamba/recovery-audit/audit-one')
        self.assertTrue(all('unlisted' not in route for route in self.uploads))
        raw=(self.folder/'artifact-delivery.json').read_text()
        self.assertNotIn('private-token',raw)
        self.assertTrue(all(r['status']=='verified' for r in manifest['files']))
    def test_wrong_receipt_never_reports_success(self):
        self.bad_receipt=True
        with self.assertRaises(DeliveryError):self.send()
        self.assertEqual(returns.read(self.folder/'artifact-delivery.json')['status'],'pending')
    def test_failed_upload_preserves_inventory_and_can_resume(self):
        self.offline=True
        with self.assertRaises(TimeoutError):self.send()
        self.assertTrue((self.folder/'report.json').is_file())
        self.assertTrue((self.folder/'artifact-delivery.json').is_file())
        self.offline=False;self.assertEqual(self.send()['status'],'uploaded')
    def test_changed_source_rejected(self):
        self.send();(self.folder/'report.json').write_text('{"changed":true}')
        with self.assertRaises(DeliveryError):self.send()
    def test_unsafe_or_changed_declaration_rejected(self):
        with self.assertRaises(ValueError):
            returns.upload_declared(self.folder,'recovery-audit',['../report.json'])
        self.send()
        with self.assertRaises(ValueError):
            returns.upload_declared(self.folder,'other',['report.json'])
