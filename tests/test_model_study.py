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
    def test_chat_evidence_replays(self):
        with patch.object(study,'table') as table, redirect_stdout(io.StringIO()):
            study.chat_runtime()
        tables={call.args[0]:call.args[1] for call in table.call_args_list}
        self.assertEqual(len(tables['chat-forward.csv']),3)
        self.assertEqual(len(tables['chat-terminal.csv']),7)

    def test_chat_replay_rejects_changed_cache_and_omitted_timing(self):
        for damage in ('cache','timing'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                manifest=json.loads((STUDY/'chat-study.json').read_text())
                report=json.loads(gzip.decompress((STUDY/'chat-study.json.gz').read_bytes()))
                if damage=='cache': report['driver']['storage'][0]['prefix_exact']=False
                else: report['driver']['samples'].pop()
                raw=json.dumps(report).encode();encoded=gzip.compress(raw,mtime=0)
                manifest.update(sha256=study.sha(encoded),raw_sha256=study.sha(raw))
                (root/'chat-study.json.gz').write_bytes(encoded)
                (root/'chat-study.json').write_text(json.dumps(manifest))
                with patch.object(study,'ROOT',root), patch.object(study,'table') as table:
                    with self.assertRaisesRegex(ValueError,'chat (cache invariant|timing census)'):
                        study.chat_runtime()
                    table.assert_not_called()

    def test_completed_runtime_replays_all_evidence(self):
        output = io.StringIO()
        with patch.object(study, 'table') as table, redirect_stdout(output):
            study.runtime()
        tables = {call.args[0]: call.args[1] for call in table.call_args_list}
        self.assertEqual(len(tables['runtime-selection.csv']), 11)
        self.assertEqual(len(tables['runtime-generations.csv']), 6)
        self.assertIn('72,114 runtime diagnostics, 24,816 storage checks', output.getvalue())

    def test_runtime_diagnosis_retains_large_errors_but_rejects_missing_or_corrupt_storage(self):
        report = json.loads(gzip.decompress((STUDY/'runtime-diagnostics.json.gz').read_bytes()))
        row = next(r for r in report['diagnostics'] if r['stage']=='hidden_1')
        row['max_abs'] = row['max_row_relative_l2'] = 1e6
        study.runtime_diagnostic_census(report)
        report['storage'][0]['inactive'] = False
        with self.assertRaisesRegex(ValueError, 'cache invariant'):
            study.runtime_diagnostic_census(report)
        report['storage'][0]['inactive'] = True
        report['diagnostics'].pop()
        with self.assertRaisesRegex(ValueError, 'diagnostic census'):
            study.runtime_diagnostic_census(report)

    def test_runtime_rejects_omitted_evidence_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = json.loads((STUDY/'runtime-study.json').read_text())
            del manifest['files']['runtime-mixed-diagnostics.json.gz']
            (root/'runtime-study.json').write_text(json.dumps(manifest))
            with patch.object(study, 'ROOT', root), patch.object(study, 'table') as table:
                with self.assertRaisesRegex(ValueError, 'evidence file census'):
                    study.runtime()
                table.assert_not_called()

    def test_runtime_selection_requires_complete_pairs_and_gain_above_noise(self):
        report=dict(specification=dict(measurements=[dict(rows=16,total=256,candidates=[21])]),samples=[])
        for block in range(4):
            for arm,config in enumerate([0,0,21]):
                for sample in range(10):
                    report['samples'].append(dict(block=block,arm=arm,prefix=240,rows=16,
                        configuration=config,sample=sample,nanoseconds=[100,101,80][arm]))
        self.assertEqual(study.runtime_ratios(report)[0]['outcome'],'gain')
        noisy=json.loads(json.dumps(report))
        for r in noisy['samples']:
            if r['arm']==1: r['nanoseconds']=130
        self.assertEqual(study.runtime_ratios(noisy)[0]['outcome'],'inconclusive')
        report['samples'].pop()
        with self.assertRaisesRegex(ValueError,'census'): study.runtime_ratios(report)

    def test_fast_reference_stop_replays_from_all_observations(self):
        output = io.StringIO()
        with patch.object(study, 'table') as table, redirect_stdout(output):
            study.fast_reference()
        self.assertEqual(len(table.call_args_list[0].args[1]), 5)
        self.assertEqual(len(table.call_args_list[1].args[1]), 164)
        self.assertEqual(len(table.call_args_list[2].args[1]), 169)
        self.assertIn('12,300 Fast reference checks', output.getvalue())

    def test_fast_replay_rejects_missing_schedule_promotion_and_changed_budget(self):
        for damage, message in [('schedule','schedule census'), ('promotion','promoted'),
                                ('budget','budget derivation')]:
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = json.loads((STUDY/'fast-reference-study.json').read_text())
                for name in manifest['files']:
                    shutil.copyfile(STUDY/name, root/name)
                def load(name):
                    return json.loads(gzip.decompress((root/name).read_bytes()))
                def save(name, raw):
                    encoded = gzip.compress(raw, mtime=0)
                    (root/name).write_bytes(encoded)
                    manifest['files'][name].update(sha256=study.sha(encoded),
                        uncompressed_sha256=study.sha(raw), uncompressed_bytes=len(raw))
                report = load('fast-reference-result.json.gz')
                if damage == 'schedule':
                    rows = [json.loads(line) for line in gzip.decompress(
                        (root/'fast-reference-observations.jsonl.gz').read_bytes()).splitlines()]
                    kept = [r for r in rows if not (r['case'] == 'random-16' and r['schedule'] == [1]*16)]
                    self.assertLess(len(kept), len(rows))
                    raw = ''.join(json.dumps(r)+'\n' for r in kept).encode()
                    save('fast-reference-observations.jsonl.gz', raw)
                    report['observations_sha256'] = study.sha(raw)
                    report['checks'] = len(kept)
                elif damage == 'promotion':
                    report['passed'] = True
                else:
                    frozen = load('fast-reference-budgets.json.gz')
                    for record in (frozen, report):
                        record['gates']['hidden_1']['relative_rms'] = 0.03125
                    raw = (json.dumps(frozen)+'\n').encode()
                    save('fast-reference-budgets.json.gz', raw)
                    report['frozen_budgets_sha256'] = study.sha(raw)
                save('fast-reference-result.json.gz', (json.dumps(report)+'\n').encode())
                (root/'fast-reference-study.json').write_text(json.dumps(manifest))
                with patch.object(study, 'ROOT', root), patch.object(study, 'table') as table:
                    with self.assertRaisesRegex(ValueError, message):
                        study.fast_reference()
                    table.assert_not_called()

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
