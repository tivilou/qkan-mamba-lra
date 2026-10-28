from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_pixel_source import audit_split, transform_audit


class PixelSourceTests(unittest.TestCase):
    def fixture(self, source=None, labels=None, actual=None, actual_labels=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        (root / 'pathfinder').mkdir()
        source = np.tile(np.arange(1024) % 256, (3, 1)) if source is None else source
        labels = np.asarray([0, 1, 0]) if labels is None else labels
        actual = source.astype(np.uint8) if actual is None else actual
        actual_labels = labels if actual_labels is None else actual_labels
        np.savez_compressed(root / 'pathfinder/train.npz', inputs=actual, labels=actual_labels)
        path = root / 'source.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('lra/pathfinder/input/train.src', '\n'.join(' '.join(map(str, row)) for row in source) + '\n')
            archive.writestr('lra/pathfinder/label/train.label', '\n'.join(map(str, labels)) + '\n')
        archive = stack.enter_context(zipfile.ZipFile(path))
        return archive, root

    def test_exact_rows_pass_and_scaling_is_invertible_for_every_pixel(self):
        archive, root = self.fixture()
        result = audit_split(archive, root, 'pathfinder', 'train', 'train')
        self.assertTrue(result['passed'])
        self.assertEqual(result['rows'], 3)
        self.assertTrue(transform_audit()['passed'])

    def test_single_corrupt_pixel_and_label_are_detected(self):
        pixels = np.tile(np.arange(1024) % 256, (3, 1)).astype(np.uint8)
        pixels[1, 97] = 7
        archive, root = self.fixture(actual=pixels, actual_labels=np.asarray([1, 1, 0]))
        result = audit_split(archive, root, 'pathfinder', 'train', 'train')
        self.assertFalse(result['passed'])
        self.assertEqual(result['pixel_row_mismatches'], 1)
        self.assertEqual(result['label_row_mismatches'], 1)

    def test_extra_or_missing_source_rows_fail(self):
        for count in (2, 4):
            with self.subTest(count=count):
                archive, root = self.fixture(source=np.zeros((count, 1024), dtype=np.uint8),
                    labels=np.zeros(count, dtype=np.int64), actual=np.zeros((3, 1024), dtype=np.uint8),
                    actual_labels=np.zeros(3, dtype=np.int64))
                with self.assertRaises(ValueError):
                    audit_split(archive, root, 'pathfinder', 'train', 'train')

    def test_out_of_range_pixels_and_labels_cannot_wrap(self):
        for pixels, labels in ((np.full((3, 1024), 256), np.asarray([0, 1, 0])),
                               (np.zeros((3, 1024)), np.asarray([0, 2, 0]))):
            archive, root = self.fixture(source=pixels, labels=labels)
            with self.assertRaises(ValueError):
                audit_split(archive, root, 'pathfinder', 'train', 'train')


if __name__ == '__main__':
    unittest.main()
