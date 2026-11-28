"""Container-independent integrity still rejects every changed NPY byte."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from experiments.baseline_delivery import sync
from experiments.baseline_delivery.common import digest
from experiments.baseline_delivery.runner import config_for


class ContentTests(unittest.TestCase):
    def test_repacked_data_passes_and_records_actual_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'image').mkdir()
            path = root/'image/train.npz'
            arrays = {'inputs.npy': b'npy-header-and-inputs', 'labels.npy': b'npy-header-and-labels'}
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as z:
                for name, raw in arrays.items(): z.writestr(name, raw)
            content = {k: dict(size_bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for k,v in arrays.items()}
            sync.verify_npz_content(path, content)
            expected = dict(archive_sha256='source', profile='v3', tasks=[dict(task='image', splits={'train':dict(sha256='reference')})])
            identities = dict(source_archive_sha256='source', profile='v3', splits={'image/train':dict(reference_npz_sha256='reference', members=content)})
            with patch.object(sync, 'read', side_effect=[expected, identities, dict(dictionaries={})]):
                record = sync.verify_data(root)[0]
            self.assertEqual(record['sha256'], digest(path))
            self.assertEqual(record['reference_npz_sha256'], 'reference')
            protocol = dict(training={}, model_defaults={'transformer':{}}, tasks={})
            with patch('experiments.baseline_delivery.runner.read', side_effect=[protocol,expected]), \
                 patch('experiments.baseline_delivery.runner.digest', side_effect=lambda p: digest(p) if Path(p)==path else 'protocol'):
                config = config_for('image','transformer',42,root,'revision')
            self.assertEqual(config['data_hashes']['train'], digest(path))
            # Same length, one changed label byte: must fail despite intact ZIP CRC.
            arrays['labels.npy'] = b'npy-header-and-labelX'
            with zipfile.ZipFile(path,'w') as z:
                for name, raw in arrays.items(): z.writestr(name,raw)
            with self.assertRaisesRegex(ValueError, 'content differs'):
                sync.verify_npz_content(path,content)

    def test_extra_missing_and_duplicate_members_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.npz'
            content={'inputs.npy':dict(size_bytes=1,sha256=hashlib.sha256(b'x').hexdigest())}
            for names in ([],['other.npy'],['inputs.npy','other.npy'],['inputs.npy','inputs.npy']):
                with zipfile.ZipFile(path,'w') as z:
                    for name in names: z.writestr(name,b'x')
                with self.assertRaisesRegex(ValueError, 'names/count'):
                    sync.verify_npz_content(path,content)

    def test_npy_size_and_broken_zip_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.npz'
            with zipfile.ZipFile(path,'w') as z: z.writestr('inputs.npy',b'xx')
            with self.assertRaisesRegex(ValueError, 'size differs'):
                sync.verify_npz_content(path,{'inputs.npy':dict(size_bytes=1,sha256='unused')})
            path.write_bytes(b'not-a-zip')
            with self.assertRaises(zipfile.BadZipFile): sync.verify_npz_content(path,{})
