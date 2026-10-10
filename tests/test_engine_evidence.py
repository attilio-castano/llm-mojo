import gzip
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from llm_mojo.benchmarks.engine_evidence import original_files, restore, verify


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def fixture(root):
    """Original formatting matters: preserve bytes, not just parsed JSON values."""
    originals = {'original.json': b'{\n  "failures": [7, 1], "value": 1.0\n}\n',
                 'nested/replay.json': b'{"actual_exit": 0, "unicode": "\xc3\xa9"}\n'}
    member = dict(name='results/live.json', text=originals['original.json'].decode(),
                  **identity(originals['original.json']))
    raw = (json.dumps(dict(files=[member]), ensure_ascii=False) + '\n').encode()
    packed = gzip.compress(raw, mtime=0)
    manifest = (json.dumps(dict(**identity(packed), uncompressed_bytes=len(raw),
                               uncompressed_sha256=identity(raw)['sha256'])) + '\n').encode()
    (root / 'validation.json.gz').write_bytes(packed)
    (root / 'validation.json').write_bytes(manifest)
    replay = gzip.compress(originals['nested/replay.json'], mtime=0)
    (root / 'replay.json.gz').write_bytes(replay)
    files = {
        'original.json': dict(**identity(originals['original.json']), storage=dict(
            format='archive-member', path='validation.json.gz', **identity(packed),
            member='results/live.json', manifest=dict(path='validation.json', **identity(manifest)))),
        'nested/replay.json': dict(**identity(originals['nested/replay.json']), storage=dict(
            format='gzip', path='replay.json.gz', **identity(replay))),
        'validation.json.gz': dict(**identity(packed), storage=dict(
            format='file', path='validation.json.gz', **identity(packed))),
    }
    catalog = dict(kind='engine-evidence-catalog-v1', basis_commit='historical', files=files)
    (root / 'engine-evidence.json').write_text(json.dumps(catalog))
    return originals, catalog


class EngineEvidenceTests(unittest.TestCase):
    def test_restore_preserves_original_bytes_and_failure_records(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            originals, _ = fixture(root)
            report = verify(root)
            self.assertEqual(report['original_files'], 3)
            self.assertFalse(report['native_execution'])
            destination = root / 'restored'
            restored = restore(root, destination)
            self.assertEqual(restored['original_files'], 3)
            for name, raw in originals.items():
                self.assertEqual((destination / name).read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                restore(root, destination)

    def test_corrupt_storage_never_creates_destination(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            fixture(root)
            (root / 'replay.json.gz').write_bytes(b'changed')
            destination = root / 'restored'
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                restore(root, destination)
            self.assertFalse(destination.exists())

    def test_member_identity_and_original_identity_checked_separately(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, catalog = fixture(root)
            catalog['files']['original.json']['sha256'] = '0' * 64
            (root / 'engine-evidence.json').write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, 'original.json'):
                original_files(root)

    def test_cached_archive_still_checks_each_manifest_reference(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, catalog = fixture(root)
            later = copy.deepcopy(catalog['files']['original.json'])
            later['storage']['manifest']['sha256'] = '0' * 64
            catalog['files']['later.json'] = later
            (root / 'engine-evidence.json').write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, 'validation.json'):
                restore(root, root / 'restored')
            self.assertFalse((root / 'restored').exists())

    def test_unsafe_output_paths_and_storage_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, catalog = fixture(root)
            original = catalog['files'].pop('original.json')
            catalog['files']['../escape.json'] = original
            (root / 'engine-evidence.json').write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, 'relative path'):
                restore(root, root / 'restored')
            self.assertFalse((root / 'restored').exists())
            fixture(root)
            with tempfile.TemporaryDirectory() as other:
                outside = Path(other) / 'replay.json.gz'
                outside.write_bytes((root / 'replay.json.gz').read_bytes())
                (root / 'replay.json.gz').unlink()
                (root / 'replay.json.gz').symlink_to(outside)
                with self.assertRaisesRegex(ValueError, 'escapes'):
                    original_files(root)

    def test_retained_catalog_reconstructs_all_evidence(self):
        root = Path(__file__).resolve().parents[1] / 'studies/model_generation'
        catalog, files = original_files(root)
        changed = [name for name, record in catalog['files'].items()
                   if record['storage']['format'] != 'file']
        self.assertEqual(len(changed), 8)
        self.assertEqual(len(files), 104)
        self.assertEqual(json.loads(files['engine-budget-results.json'])['measured_runs'], 116)
        self.assertEqual(json.loads(files['engine-admission-range-results.json'])['measured_runs'], 144)


if __name__ == '__main__':
    unittest.main()
