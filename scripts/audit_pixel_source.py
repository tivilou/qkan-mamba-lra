#!/usr/bin/env python3
"""Stream exact pixel/label equivalence to the pinned MEGA archive, without training."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np

from audit_lra_data import ArrayStream, sha256_file
from rebuild_mega_lra import ARCHIVE_SHA256


def audit_split(archive, data, task, source_split, split):
    path = data / task / (split + '.npz')
    prefix = 'lra/' + ('cifar10' if task == 'image' else 'pathfinder')
    with ExitStack() as stack:
        output = stack.enter_context(zipfile.ZipFile(path))
        if sorted(output.namelist()) != ['inputs.npy', 'labels.npy']:
            raise ValueError('Unexpected NPZ members')
        x = ArrayStream(output, 'inputs.npy', stack)
        y = ArrayStream(output, 'labels.npy', stack)
        if len(x.shape) != 2 or x.shape[1] != 1024 or y.shape != (x.shape[0],):
            raise ValueError('Expected matching (N,1024) pixels and (N,) labels')
        if x.dtype != np.dtype('uint8') or y.dtype.kind not in 'iu':
            raise ValueError('Expected uint8 pixels and integer labels')
        source_x = stack.enter_context(archive.open(prefix + '/input/' + source_split + '.src'))
        source_y = stack.enter_context(archive.open(prefix + '/label/' + source_split + '.label'))
        pixel_mismatches = label_mismatches = 0
        canonical_x, canonical_y = hashlib.sha256(), hashlib.sha256()
        classes = 10 if task == 'image' else 2
        for start in range(0, x.shape[0], 128):
            rows = min(128, x.shape[0] - start)
            lines = list(itertools.islice(source_x, rows))
            label_lines = list(itertools.islice(source_y, rows))
            if len(lines) != rows or len(label_lines) != rows:
                raise ValueError('Truncated source rows')
            parsed = []
            for line in lines:
                # The token count plus range check rejects truncated/malformed text.
                tokens = line.split()
                if len(tokens) != 1024:
                    raise ValueError('Expected 1024 source pixels')
                values = np.asarray(tokens, dtype=np.int64)
                if np.any(values < 0) or np.any(values > 255):
                    raise ValueError('Source pixel outside uint8 range')
                parsed.append(values)
            expected_x = np.asarray(parsed, dtype=np.uint8)
            expected_y = np.asarray([int(line) for line in label_lines], dtype='<i8')
            if np.any(expected_y < 0) or np.any(expected_y >= classes):
                raise ValueError('Source label outside class range')
            pixel_mismatches += int(np.count_nonzero(np.any(x.read(rows) != expected_x, axis=1)))
            label_mismatches += int(np.count_nonzero(y.read(rows) != expected_y))
            canonical_x.update(expected_x.tobytes())
            canonical_y.update(expected_y.tobytes())
        if source_x.read(1) or source_y.read(1):
            raise ValueError('Additional source rows after NPZ end')
    return {'task': task, 'split': split, 'source_split': source_split, 'rows': x.shape[0],
            'pixel_row_mismatches': pixel_mismatches, 'label_row_mismatches': label_mismatches,
            'source_pixels_canonical_sha256': canonical_x.hexdigest(),
            'source_labels_canonical_sha256': canonical_y.hexdigest(),
            'npz_sha256': sha256_file(path),
            'passed': pixel_mismatches == label_mismatches == 0}


def transform_audit():
    pixels = np.arange(256, dtype=np.float32)
    common = pixels / np.float32(255.) - np.float32(.5)
    author = (pixels / np.float32(255.) - np.float32(.5)) / np.float32(.5)
    restored = np.rint((common + np.float32(.5)) * np.float32(255.)).astype(np.uint8)
    passed = np.array_equal(restored, np.arange(256, dtype=np.uint8)) and np.array_equal(author, 2 * common)
    return {'values_checked': 256, 'dtype': 'float32',
            'common_transform': 'x/255 - 0.5',
            'mega_pathfinder_transform': '(x/255 - 0.5)/0.5',
            'source_of_formula': 'pinned PixelSequenceDataset.encode_line and README.lra.md',
            'author_equals_two_times_common': bool(np.array_equal(author, 2 * common)),
            'all_uint8_values_recoverable': bool(np.array_equal(restored, np.arange(256, dtype=np.uint8))),
            'passed': bool(passed),
            'boundary': 'Information-preserving scaling only; no claim of equal optimization dynamics or reproduced scores.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refuse to overwrite an audit receipt')
    if sha256_file(args.archive) != ARCHIVE_SHA256:
        parser.error('Archive does not match the pinned source')
    records = []
    with zipfile.ZipFile(args.archive) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate archive members')
        for task in ('image', 'pathfinder'):
            for source_split, split in (('train', 'train'), ('valid', 'val'), ('test', 'test')):
                records.append(audit_split(archive, args.data, task, source_split, split))
                print(json.dumps(records[-1]), flush=True)
    transforms = transform_audit()
    passed = all(item['passed'] for item in records) and transforms['passed']
    report = {'schema': 'qkan-pixel-source-equivalence.v1', 'passed': passed,
              'archive_sha256': ARCHIVE_SHA256, 'code_sha256': sha256_file(Path(__file__)),
              'numpy_version': np.__version__, 'splits': records, 'transform': transforms,
              'purpose': 'Source/label fidelity and information-preservation evidence; not a trained baseline.',
              'test_access': 'Exhaustive equality against frozen source only; no prediction, selection or tuning.',
              'formal_baseline_allowed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
