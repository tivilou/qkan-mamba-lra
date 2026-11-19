import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exchange_artifacts as client

spec = importlib.util.spec_from_file_location("exchange_service", ROOT / "tools/exchange/service/qore_upload_server.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.storage = self.root / "storage"
        self.storage.mkdir()
        self.token = "test-only-token-" + "a" * 32
        self.server = service.UploadServer(("127.0.0.1", 0), service.UploadHandler,
            root=self.storage, web_root=ROOT / "tools/exchange/web", token=self.token,
            max_bytes=1024 * 1024, namespace="qkan-mamba")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.transport = client.Transport(self.url, self.token, allow_http=True)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "summary.json").write_text('{"accuracy":0.5}', encoding="utf-8")
        # A real commit is not required for this transport fixture.
        self.manifest = {"schema_version": "qkan-delivery.v1", "project": "qkan-mamba",
            "target_dir": "qkan-mamba/smoke/run-1", "source_revision": "fixture",
            "status": "prepared", "files": [{"local_path": "summary.json", "path": "summary.json",
            "size_bytes": client.digest_file(self.root / "summary.json")[0],
            "sha256": client.digest_file(self.root / "summary.json")[1], "status": "pending"}]}
        self.manifest_path = self.root / "delivery.json"
        client.atomic_json(self.manifest_path, self.manifest)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_upload_readback_and_identical_retry(self):
        result = client.deliver(self.root, self.manifest_path, self.transport)
        self.assertEqual(result["status"], "uploaded")
        self.assertEqual(result["files"][0]["status"], "verified")
        self.assertEqual((self.storage / "qkan-mamba/smoke/run-1/summary.json").read_bytes(),
                         (self.root / "summary.json").read_bytes())
        self.assertEqual(client.deliver(self.root, self.manifest_path, self.transport)["status"], "uploaded")
        self.assertNotIn(self.url, self.manifest_path.read_text())
        self.assertNotIn(self.token, self.manifest_path.read_text())

    def test_conflicting_payload_preserves_original_and_pending(self):
        client.deliver(self.root, self.manifest_path, self.transport)
        original = (self.storage / "qkan-mamba/smoke/run-1/summary.json").read_bytes()
        (self.root / "summary.json").write_text("changed", encoding="utf-8")
        size, digest = client.digest_file(self.root / "summary.json")
        self.manifest["files"][0].update(size_bytes=size, sha256=digest)
        client.atomic_json(self.manifest_path, self.manifest)
        with self.assertRaises(client.DeliveryError):
            client.deliver(self.root, self.manifest_path, self.transport)
        self.assertEqual(json.loads(self.manifest_path.read_text())["status"], "pending")
        self.assertEqual((self.storage / "qkan-mamba/smoke/run-1/summary.json").read_bytes(), original)

    def test_failure_then_retry_without_training(self):
        bad = client.Transport(self.url, "wrong-token", allow_http=True)
        with self.assertRaises(client.DeliveryError):
            client.deliver(self.root, self.manifest_path, bad)
        self.assertEqual(json.loads(self.manifest_path.read_text())["status"], "pending")
        self.assertEqual(client.deliver(self.root, self.manifest_path, self.transport)["status"], "uploaded")

    def test_receipt_mismatch_cannot_complete(self):
        original = self.transport.request
        def altered(method, route, body=None, size=0):
            status, result = original(method, route, body, size)
            if method == "PUT":
                result["sha256"] = "0" * 64
            return status, result
        self.transport.request = altered
        with self.assertRaises(client.DeliveryError):
            client.deliver(self.root, self.manifest_path, self.transport)
        self.assertEqual(json.loads(self.manifest_path.read_text())["status"], "pending")

    def test_namespace_traversal_hidden_and_plain_http_rejected(self):
        for value in ("../a", "/a", "qkan-mamba/.secret", "a//b", "a/./b", "C:/x"):
            with self.assertRaises(client.DeliveryError):
                client.safe_path(value)
        with self.assertRaises(client.DeliveryError):
            client.Transport(self.url, self.token)
        self.manifest["target_dir"] = "other-project/smoke/run-1"
        with self.assertRaises(client.DeliveryError):
            client.validate_manifest(self.root, self.manifest)
        body = b'{"path":"other-project/smoke/run-1"}'
        with self.assertRaises(client.DeliveryError):
            self.transport.request("POST", "/api/directories", body, len(body))

    def test_changed_source_rejected_before_upload(self):
        (self.root / "summary.json").write_text("changed", encoding="utf-8")
        with self.assertRaises(client.DeliveryError):
            client.deliver(self.root, self.manifest_path, self.transport)
        self.assertFalse((self.storage / "qkan-mamba").exists())

    def test_oversize_rejected_before_upload(self):
        (self.root / "summary.json").write_bytes(b"x" * (1024 * 1024 + 1))
        size, digest = client.digest_file(self.root / "summary.json")
        self.manifest["files"][0].update(size_bytes=size, sha256=digest)
        client.atomic_json(self.manifest_path, self.manifest)
        with self.assertRaises(client.DeliveryError):
            client.deliver(self.root, self.manifest_path, self.transport)
        self.assertEqual(json.loads(self.manifest_path.read_text())["status"], "pending")
        self.assertFalse((self.storage / "qkan-mamba").exists())


if __name__ == "__main__":
    unittest.main()
