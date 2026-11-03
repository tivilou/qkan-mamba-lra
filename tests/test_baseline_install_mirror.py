"""Mirror assembly, corruption rejection, and offline pip boundaries."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments.baseline_delivery import install_mirror as mirror


class MirrorTests(unittest.TestCase):
    def test_assembly_reuse_and_corruption(self):
        payload = b'wheel-bytes'
        sha = hashlib.sha256(payload).hexdigest()
        item = dict(filename='test.whl', size_bytes=len(payload), sha256=sha,
                    chunks=[dict(path='qkan-mamba/dependencies/v1/chunk', size_bytes=len(payload), sha256=sha)])
        manifest = dict(python_tag='cp311', objects=[item])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(mirror, 'check_exchange'), \
             patch.object(mirror.platform, 'system', return_value='Linux'), \
             patch.object(mirror.platform, 'machine', return_value='x86_64'), \
             patch.object(mirror.sys, 'version_info', SimpleNamespace(major=3, minor=11)), \
             patch.dict(os.environ, PROJECT_EXCHANGE_URL='https://example.invalid',
                        PROJECT_EXCHANGE_TOKEN='private', PROJECT_EXCHANGE_CA_FILE='ca.crt'), \
             patch.object(mirror, 'download', side_effect=lambda url, path, *a, **kw: Path(path).write_bytes(payload)) as get:
            mirror.fetch(tmp, manifest)
            self.assertEqual((Path(tmp)/'test.whl').read_bytes(), payload)
            mirror.fetch(tmp, manifest)
            self.assertEqual(get.call_count, 1)
            (Path(tmp)/'test.whl').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'differs'):
                mirror.fetch(tmp, manifest)
            (Path(tmp)/'test.whl').unlink()
            item['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'verification'):
                mirror.fetch(tmp, manifest)
            self.assertFalse((Path(tmp)/'test.whl').exists())

    def test_wrong_python_fails_before_download(self):
        with patch.object(mirror, 'check_exchange'), \
             patch.object(mirror.platform, 'system', return_value='Linux'), \
             patch.object(mirror.platform, 'machine', return_value='x86_64'), \
             patch.object(mirror.sys, 'version_info', SimpleNamespace(major=3, minor=10)), \
             patch.object(mirror, 'download') as get:
            with self.assertRaisesRegex(ValueError, 'cp311'):
                mirror.fetch('.', dict(python_tag='cp311'))
            get.assert_not_called()

    def test_mamba_install_has_no_public_index_or_git_fetch(self):
        with patch.object(mirror, 'read', return_value={}), \
             patch.object(mirror, 'fetch', return_value=Path('/cache')), \
             patch.object(mirror.sys, 'argv', ['mirror', '--cache', '/cache', '--stage', 'mamba']), \
             patch.object(mirror.subprocess, 'run') as run:
            mirror.main()
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertIn('--no-index', call.args[0])
                self.assertEqual(call.kwargs['env']['PIP_NO_INDEX'], '1')
                self.assertEqual(call.kwargs['env']['MAMBA_FORCE_BUILD'], 'TRUE')
                self.assertFalse(any('https://' in part or 'git+' in part for part in call.args[0]))
            self.assertIn('--no-deps', run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
