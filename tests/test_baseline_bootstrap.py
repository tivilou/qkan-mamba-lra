"""Installer regressions; no packages installed and no GPU used."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.baseline_delivery import bootstrap as b


class BootstrapTests(unittest.TestCase):
    def test_exchange_configuration(self):
        with tempfile.NamedTemporaryFile() as ca:
            env = dict(PROJECT_EXCHANGE_URL='https://example.invalid', PROJECT_EXCHANGE_TOKEN='secret',
                       PROJECT_EXCHANGE_CA_FILE=ca.name)
            with patch.dict(os.environ, env, clear=True):
                b.check_exchange()
                for key in env:
                    with patch.dict(os.environ, {key: ''}):
                        with self.assertRaisesRegex(ValueError, key):
                            b.check_exchange()
                with patch.dict(os.environ, PROJECT_EXCHANGE_URL='http://example.invalid'):
                    with self.assertRaisesRegex(ValueError, 'HTTPS'):
                        b.check_exchange()
                with patch.dict(os.environ, PROJECT_EXCHANGE_CA_FILE=ca.name+'.absent'):
                    with self.assertRaisesRegex(ValueError, 'certificate'):
                        b.check_exchange()

    def test_core_stack_and_drift(self):
        versions = {'numpy': '2.2.6', 'einops': '0.8.1', 'triton': '3.3.1', 'nvidia-cudnn-cu12':'9.5.1.17'}
        runtime = SimpleNamespace(getRuntimeVersion=lambda:(9,5,1))
        torch = SimpleNamespace(__version__='2.7.1+cu126', version=SimpleNamespace(cuda='12.6'),
                                _C=SimpleNamespace(_cudnn=runtime))
        with patch.object(b.importlib, 'import_module', return_value=torch), \
             patch.object(b.metadata, 'version', side_effect=versions.__getitem__):
            b.check_core()
            for wrong in ('2.15.0+cu132', '2.7.1+cu118', '2.7.1'):
                torch.__version__ = wrong
                with self.assertRaisesRegex(ValueError, 'torch'):
                    b.check_core()
            torch.__version__ = '2.7.1+cu126'
            torch.version.cuda = '13.2'
            with self.assertRaisesRegex(ValueError, 'torch'):
                b.check_core()
            torch.version.cuda = '12.6'
            for name in versions:
                original = versions[name]
                versions[name] = '999'
                with self.assertRaisesRegex(ValueError, name):
                    b.check_core()
                versions[name] = original
            runtime.getRuntimeVersion=lambda:(9,26,0)
            with self.assertRaisesRegex(ValueError, 'library files'):
                b.check_core()

    def test_mamba_identity_and_abi(self):
        source = {'url': 'https://github.com/state-spaces/mamba.git',
                  'vcs_info': {'commit_id': b.MAMBA_COMMIT}}
        dist = SimpleNamespace(version='2.2.4', read_text=lambda name: json.dumps(source))
        with patch.object(b, 'check_core'), patch.object(b.metadata, 'distribution', return_value=dist), \
             patch.object(b.importlib, 'import_module', return_value=SimpleNamespace(Mamba=object)) as imports:
            b.check_mamba()
            self.assertEqual(imports.call_args_list[0].args, ('selective_scan_cuda',))
            source['vcs_info']['commit_id'] = 'old-commit'
            with self.assertRaisesRegex(ValueError, 'commit'):
                b.check_mamba()
            source['vcs_info']['commit_id'] = b.MAMBA_COMMIT
            dist.version = '2.3.2.post1'
            with self.assertRaisesRegex(ValueError, 'commit'):
                b.check_mamba()
            dist.version = '2.2.4'
            source['url'] = 'https://example.invalid/mamba.git'
            with self.assertRaisesRegex(ValueError, 'repository'):
                b.check_mamba()
            source['url'] = 'https://github.com/state-spaces/mamba.git'
            imports.side_effect = ImportError('incompatible torch ABI')
            with self.assertRaisesRegex(ImportError, 'ABI'):
                b.check_mamba()

    def test_mirrored_source_hash_identity(self):
        manifest = json.loads((Path(b.__file__).parent/'install-mirror.json').read_text())
        source_hash = next(x['sha256'] for x in manifest['objects'] if x['filename']=='mamba-2.2.4.tar.gz')
        source = {'url': 'file:///cache/mamba-2.2.4.tar.gz',
                  'archive_info': {'hashes': {'sha256': source_hash}}}
        dist = SimpleNamespace(version='2.2.4', read_text=lambda name: json.dumps(source))
        with patch.object(b, 'check_core'), patch.object(b.metadata, 'distribution', return_value=dist), \
             patch.object(b.importlib, 'import_module', return_value=SimpleNamespace(Mamba=object)):
            b.check_mamba()
            source['archive_info']['hashes']['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'commit'):
                b.check_mamba()

    @unittest.skipUnless(sys.platform == 'linux', 'Linux shell entry')
    def test_missing_exchange_exits_before_install(self):
        root = Path(__file__).resolve().parents[1]
        env = {k: v for k, v in os.environ.items() if not k.startswith('PROJECT_EXCHANGE_')}
        env['QKAN_PYTHON'] = sys.executable
        result = subprocess.run(['bash', 'scripts/run_baselines.sh', '--mode', 'preflight'],
                                cwd=root, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2)
        self.assertIn('PROJECT_EXCHANGE_URL', result.stderr)
        self.assertNotIn('pip install', result.stdout)


if __name__ == '__main__':
    unittest.main()
