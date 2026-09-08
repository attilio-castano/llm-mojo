"""Numerical provenance belongs to the process the evaluator actually launches."""
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from llm_mojo import mlp_validation as validation


class MLPEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        record = json.loads(gzip.decompress(
            (root / 'studies/mlp_sublayer/data/decode_numerics.json.gz').read_bytes()))
        cls.cases = record['fresh_manifest']['cases']
        cls.retained = record['checks']

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.binary = self.root / 'candidate'
        self.output = self.root / 'evaluation'
        self.source = dict(repository=dict(commit='a' * 40, dirty=False), sources={'test': 'source-hash'})
        self.manifest = dict(cases=self.cases, candidate=dict(commit='a' * 40))
        self.addCleanup(patch.stopall)
        patch.object(validation, 'source_identity', return_value=self.source).start()
        patch.object(validation, 'ensure_record_location').start()
        patch.object(validation, 'inputs', side_effect=lambda split: (self.manifest, {'manifest': 'digest'})).start()

    def candidate(self, body):
        self.binary.write_text('#!' + sys.executable + '\n' + body)
        self.binary.chmod(0o755)
        digest = validation.sha(self.binary)
        self.manifest['candidate']['binary_sha256'] = digest
        validation.write(str(self.binary) + '.provenance.json', dict(
            schema=1, kind='mlp_numerical_build', source=self.source, binary_sha256=digest))

    def records(self):
        # These are actual retained GPU results, not reconstructed expectations.
        return copy.deepcopy(self.retained['fresh'])

    def successful_body(self, rows=None):
        data = self.root / 'retained.json'
        data.write_text(json.dumps(self.records() if rows is None else rows))
        return f'''import json, os
from pathlib import Path
assert 'MLP_CASE' not in os.environ
assert 'MLP_CANDIDATE_BINARY' not in os.environ
assert 'MODULAR_DEBUG' not in os.environ
assert os.environ['MLP_SPLIT'] == 'decode_holdout'
assert os.environ['MLP_VARIANTS'] == '0,12'
rows = json.loads(Path({str(data)!r}).read_text())
Path(os.environ['MLP_RECORD_DIR'], 'checks.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))
for case in {list(self.cases)!r}:
    for variant in (0,12):
        print('MLP runtime: Apple Test GPU metal',case,'mapping',variant)
'''

    def evaluate(self, **kwargs):
        return validation.evaluate(self.binary, self.output, 'decode_holdout', [0, 12], **kwargs)

    def test_success_launches_receipted_binary_and_retains_result_identity(self):
        self.candidate(self.successful_body())
        with patch.dict(os.environ, MLP_CASE='wrong', MLP_CANDIDATE_BINARY='/wrong', MODULAR_DEBUG='device-sync-mode'):
            result = self.evaluate()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['command'], [str(self.binary.resolve())])
        self.assertEqual(result['checks'], len(self.records()))
        self.assertEqual(result['build']['binary_sha256'], validation.sha(self.binary))
        self.assertEqual(result['results_sha256']['checks.jsonl'], validation.sha(self.output/'checks/checks.jsonl'))

    def test_declared_other_binary_rejected_before_launch(self):
        self.candidate('raise AssertionError("must not launch")\n')
        self.manifest['candidate']['binary_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'different candidate'):
            self.evaluate()
        self.assertFalse(self.output.exists())

    def test_stale_build_or_replaced_executable_rejected_before_launch(self):
        for field in ('binary', 'source'):
            with self.subTest(field=field):
                self.candidate('raise AssertionError("must not launch")\n')
                receipt = Path(str(self.binary)+'.provenance.json')
                record = json.loads(receipt.read_text())
                if field == 'binary':
                    self.binary.write_text('changed')
                else:
                    record['source']['sources']['test'] = 'old-source'
                    validation.write(receipt, record)
                with self.assertRaisesRegex(ValueError, 'build source'):
                    self.evaluate()
                self.assertFalse(self.output.exists())

    def test_successful_unrelated_executable_cannot_create_acceptance(self):
        self.candidate('pass\n')
        with self.assertRaisesRegex(ValueError, 'runtime coverage'):
            self.evaluate()
        self.assertEqual(json.loads((self.output/'evaluation.json').read_text())['status'], 'failed')

    def test_incomplete_output_rejected_even_with_success_and_runtime_identity(self):
        self.candidate(self.successful_body(self.records()[:-1]))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.evaluate()

    def test_nonzero_exit_rejected_even_with_complete_passing_checks(self):
        self.candidate(self.successful_body() + 'raise SystemExit(1)\n')
        with self.assertRaisesRegex(ValueError, 'suite failed'):
            self.evaluate()
        self.assertEqual(json.loads((self.output/'evaluation.json').read_text())['exit_code'], 1)

    def test_executable_drift_after_launch_rejected(self):
        self.candidate(self.successful_body() + f'Path({str(self.binary)!r}).write_text("changed")\n')
        with self.assertRaisesRegex(ValueError, 'build source'):
            self.evaluate()

    def test_source_and_fixture_drift_after_launch_rejected(self):
        for target in ('source', 'fixtures'):
            with self.subTest(target=target):
                self.candidate(self.successful_body())
                self.output = self.root / target
                if target == 'source':
                    changed = copy.deepcopy(self.source)
                    changed['sources']['test'] = 'changed'
                    context = patch.object(validation, 'source_identity', side_effect=[self.source, changed])
                else:
                    context = patch.object(validation, 'inputs', side_effect=[
                        (self.manifest, {'manifest': 'digest'}), (self.manifest, {'manifest': 'changed'})])
                with context, self.assertRaises(ValueError):
                    self.evaluate()
                self.assertEqual(json.loads((self.output/'evaluation.json').read_text())['status'], 'failed')

    def test_observed_holdout_regression_does_not_claim_original_candidate(self):
        self.candidate(self.successful_body())
        self.manifest['candidate']['commit'] = 'b' * 40
        with self.assertRaisesRegex(ValueError, 'different candidate'):
            self.evaluate()
        result = self.evaluate(regression=True)
        self.assertEqual(result['purpose'], 'observed_holdout_regression')

    def test_build_receipt_binds_compiler_output_to_unchanged_test_source(self):
        def compile_candidate(command, **kwargs):
            self.assertEqual(command[-3:], ['tests/test_mlp.mojo', '-o', str(self.binary)])
            self.binary.write_bytes(b'compiled numerical suite')
        with patch.object(validation.subprocess, 'run', side_effect=compile_candidate):
            validation.build(self.binary)
        receipt = validation.verify_build(self.binary)
        self.assertEqual(receipt['binary_sha256'], validation.sha(self.binary))
        self.assertEqual(receipt['source'], self.source)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            validation.build(self.binary)

    def test_build_cannot_receipt_source_that_changed_during_compilation(self):
        changed = copy.deepcopy(self.source)
        changed['sources']['test'] = 'changed'
        with patch.object(validation, 'source_identity', side_effect=[self.source, changed]), \
             patch.object(validation.subprocess, 'run'):
            with self.assertRaisesRegex(ValueError, 'changed during numerical build'):
                validation.build(self.binary)
        self.assertFalse(Path(str(self.binary)+'.provenance.json').exists())

    def test_retained_results_require_exact_case_mapping_stage_and_reuse_coverage(self):
        checks = self.root / 'checks'
        checks.mkdir()
        original = self.records()
        changes = {'duplicate': lambda rows: rows.append(rows[-1]),
                   'missing': lambda rows: rows.pop(),
                   'mapping': lambda rows: rows[0].update(mapping=8),
                   'elements': lambda rows: rows[0].update(elements=0),
                   'failed': lambda rows: rows[0].update(failed=1),
                   'missing_gate': lambda rows: rows[0].pop('failed')}
        for name, mutate in changes.items():
            rows = copy.deepcopy(original)
            mutate(rows)
            (checks/'checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.subTest(name=name), self.assertRaises(ValueError):
                validation.validate_results(checks, self.cases, [0,12])

    def test_legacy_environment_cannot_attribute_another_executables_checks(self):
        import mlp_support
        with patch.dict(os.environ, MLP_CANDIDATE_BINARY=str(self.binary)):
            with self.assertRaisesRegex(ValueError, 'cannot identify the running executable'):
                mlp_support.holdout_catalog(decode=True)


if __name__ == '__main__':
    unittest.main()
