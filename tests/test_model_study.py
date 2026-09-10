"""Replay complete and deliberately incomplete retained HF evidence; no model runs."""
from contextlib import redirect_stdout
import gzip
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


STUDY = Path(__file__).resolve().parents[1] / 'studies/model_generation'
spec = importlib.util.spec_from_file_location('model_study', STUDY / 'summarize.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class ModelStudyTests(unittest.TestCase):
    def test_retained_consistency_evidence_is_complete(self):
        output = io.StringIO()
        with patch.object(study, 'table') as table, redirect_stdout(output):
            study.consistency()
        summary = table.call_args_list[0].args[1]
        self.assertEqual(len(summary), 13)
        count = sum(row['checks'] for row in summary)
        self.assertEqual(count, 71250)
        self.assertIn(f'Verified {count:,} HF comparisons', output.getvalue())

    def test_replay_rejects_omitted_schedule_with_matching_archive_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = json.loads((STUDY / 'consistency-study.json').read_text())
            for name in manifest['files']:
                shutil.copyfile(STUDY / name, root / name)

            def save(name, raw):
                encoded = gzip.compress(raw, mtime=0)
                (root / name).write_bytes(encoded)
                manifest['files'][name].update(
                    sha256=study.sha(encoded), uncompressed_sha256=study.sha(raw),
                    uncompressed_bytes=len(raw))

            reference = json.loads(gzip.decompress((root / 'consistency-reference.json.gz').read_bytes()))
            case = next(c for c in reference['cases'] if c['length'] == 4)
            missing = [1, 1, 1, 1]
            case['schedules'] = [s for s in case['schedules'] if s['rows'] != missing]
            observations = gzip.decompress((root / 'consistency-observations.jsonl.gz').read_bytes()).splitlines(keepends=True)
            kept = []
            for line in observations:
                row = json.loads(line)
                if not (row['length'] == 4 and row['seed'] == case['seed'] and row['schedule'] == missing):
                    kept.append(line)
            self.assertEqual(len(observations) - len(kept), 300)
            raw = b''.join(kept)
            save('consistency-observations.jsonl.gz', raw)
            reference['observations_sha256'] = study.sha(raw)
            reference_raw = (json.dumps(reference, indent=2) + '\n').encode()
            save('consistency-reference.json.gz', reference_raw)
            native = json.loads(gzip.decompress((root / 'consistency-native.json.gz').read_bytes()))
            native['accuracy']['qualification_sha256'] = study.sha(reference_raw)
            save('consistency-native.json.gz', (json.dumps(native, indent=2) + '\n').encode())
            (root / 'consistency-study.json').write_text(json.dumps(manifest))

            with patch.object(study, 'ROOT', root), patch.object(study, 'table') as table:
                with self.assertRaisesRegex(ValueError, 'declared.*schedule census'):
                    study.consistency()
                table.assert_not_called()


if __name__ == '__main__':
    unittest.main()
