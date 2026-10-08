"""Acceptance-evidence corruption must fail even with a freshly hashed envelope."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from llm_mojo.benchmarks import model_contract as contract
from llm_mojo.benchmarks.model_profile import (
    engine_run_summary, engine_study_summary, engine_replay,
    engine_specification, engine_trace_tsv, parse_engine_run,
    validate_engine_record_build,
)


def trace_fixture():
    return dict(kind='engine-token-trace-v1', schema_version=1, mode='offline',
                scripted_tokens=[11], requests=[
                    dict(request_id=i, arrival_offset_ns=0, prompt_ids=[42 + i],
                         max_new_tokens=1, stop_ids=[], abort_offset_ns=None,
                         output_script=None) for i in range(2)])


def native_fixture(arm='serial', mode='greedy'):
    device = 'Apple M4 Pro/metal' if mode == 'greedy' else 'simulated/virtual'
    sequence_limit = 1 if arm == 'serial' else 2
    lines = [f'device {device}', f'mode {mode}',
             f'config {arm} 4 {256 if arm == "chunked" else 4096} {sequence_limit}',
             'arrival 0 0 0', 'arrival 1 0 0']
    for i in range(2):
        fields = dict(step_id=i, decode_seqs=0, prefill_seqs=1, prefill_tokens=1,
                      total_tokens=1, attended_positions=1, admitted=1, preempted=0,
                      finished=1, aborted=0, waiting=1-i, blocks_free=4,
                      begin_ns=i*10, schedule_ns=1, build_ns=1, execute_ns=6,
                      postprocess_ns=1, end_ns=(i+1)*10, predicted_ns=0, budget_limited=0)
        lines.append('step ' + ' '.join(str(fields[k]) for k in contract.ENGINE_STEP_FIELDS))
        lines.extend([f'token {i} 11 1 1 0 {(i+1)*10}', f'finish {i} length 1 1 0 {(i+1)*10}'])
    lines.append('drained 2 2 4 20')
    return '\n'.join(lines)


def study_fixture(mode='greedy'):
    trace = trace_fixture()
    runs = []
    for block in range(4):
        for arm, calibration in [('serial', False), ('serial', True), *[(a, False) for a in contract.ENGINE_ARMS[1:]]]:
            stdout = native_fixture(arm, mode)
            conditions = dict(battery={'power_source': 'AC Power'}, power_mode_raw='0',
                              thermal=['No thermal warning level has been recorded',
                                       'No performance warning level has been recorded'])
            runs.append(dict(block=block, arm=arm, calibration=calibration,
                             stdout=stdout, parsed=parse_engine_run(stdout, trace, arm, 4, 2, mode),
                             conditions_before=conditions, conditions_after=conditions))
    document = json.dumps(trace)
    record = dict(kind='qwen-engine-core-v1', declaration=contract.ENGINE_DECLARATION,
                  build=dict(source=dict(repository=dict(commit='a'*40, dirty=False), sources={'uv.lock': 'e'*64}),
                             binaries={'engine': {'sha256': 'b'*64, 'bytes': 1}},
                             assets=dict(prepared_sha256='c'*64, tables_sha256='d'*64),
                             environment=dict(hardware={'chip': 'Apple M4 Pro'},
                                              software={'mojo': '1.0.0', 'max': '26.5.0'}),
                             declaration=contract.ENGINE_DECLARATION),
                  trace=trace, trace_document=document, trace_sha256=hashlib.sha256(document.encode()).hexdigest(),
                  blocks=4, max_sequences=2, mode=mode, warmup_steps=10, runs=runs)
    record['summary'] = engine_study_summary(record)
    return record


def write_archive(root, record):
    raw = json.dumps(record).encode()
    compressed = gzip.compress(raw, mtime=0)
    (root/'engine-core.json.gz').write_bytes(compressed)
    (root/'engine-core.json').write_text(json.dumps(dict(kind='qwen-engine-core-v1', bytes=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(), uncompressed_sha256=hashlib.sha256(raw).hexdigest())))


class EngineTraceTests(unittest.TestCase):
    def test_online_arrivals_are_frozen_by_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = engine_specification(root/'a.json', seed=5, arrival_rate=2)
            second = engine_specification(root/'b.json', seed=5, arrival_rate=2)
            self.assertEqual(first, second)
            self.assertEqual(first['mode'], 'online')
            self.assertGreater(first['requests'][-1]['arrival_offset_ns'], 0)
            self.assertIn('request 0 0 32', engine_trace_tsv(first))

    def test_metrics_use_scheduled_arrival_and_full_drain(self):
        trace = trace_fixture()
        parsed = parse_engine_run(native_fixture(), trace, 'serial', 4, 2, 'greedy')
        result = engine_run_summary(parsed, trace)
        self.assertEqual(result['duration_ns'], 20)
        self.assertEqual(result['ttft']['p50_ns'], 15)
        self.assertIsNone(result['tpot'])
        self.assertEqual(result['delivered_tokens'], 2)

    def test_missing_duplicate_and_post_finish_tokens_fail(self):
        stdout = native_fixture()
        invalid = [stdout.replace('token 0 11 1 1 0 10\n', ''),
                   stdout.replace('token 0 11 1 1 0 10', 'token 0 11 1 2 0 10'),
                   stdout + '\ntoken 0 11 1 2 0 20',
                   stdout.replace('finish 0 length 1 1 0 10', 'finish 0 stop 1 1 0 10'),
                   stdout.replace('step 1 ', 'step 2 ')]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_engine_run(value, trace_fixture(), 'serial', 4, 2, 'greedy')

    def test_device_and_clock_modes_are_distinct(self):
        with self.assertRaises(ValueError):
            parse_engine_run(native_fixture().replace('Apple M4 Pro/metal', 'cpu/host'),
                             trace_fixture(), 'serial', 4, 2, 'greedy')
        summary = engine_study_summary(study_fixture('scripted'))
        self.assertTrue(all(c['outcome'] == 'virtual-clock' for c in summary['comparisons']))
        self.assertIsNone(summary['goodput'])

    def test_drain_receipt_and_empty_output_are_accounted(self):
        with self.assertRaises(ValueError):
            parse_engine_run(native_fixture().replace('drained 2 2 4 20', 'drained 2 1 4 20'),
                             trace_fixture(), 'serial', 4, 2, 'greedy')
        trace = trace_fixture()
        trace['requests'][0]['max_new_tokens'] = 0
        stdout = native_fixture().replace('token 0 11 1 1 0 10\n', '').replace(
            'finish 0 length 1 1 0 10', 'finish 0 length 1 0 0 10')
        parsed = parse_engine_run(stdout, trace, 'serial', 4, 2, 'greedy')
        self.assertIsNone(engine_run_summary(parsed, trace)['requests'][0]['ttft_ns'])

    def test_adaptive_predictions_use_the_frozen_policy(self):
        policy = dict(target_ns=25, fixed_ns=1, per_row_ns=2, per_position_ns=3,
                      per_partition_ns=4, per_logit_ns=5)
        stdout = native_fixture('adaptive').replace('config adaptive 4 4096 2', 'config adaptive 4 256 2')
        stdout = stdout.replace('arrival 0', 'policy 25 1 2 3 4 5\narrival 0')
        stdout = stdout.replace(' 10 0 0\n', ' 10 15 0\n').replace(' 20 0 0\n', ' 20 15 0\n')
        parsed = parse_engine_run(stdout, trace_fixture(), 'adaptive', 4, 2, 'greedy', policy=policy)
        self.assertEqual(parsed['steps'][0]['predicted_ns'], 15)
        with self.assertRaises(ValueError):
            parse_engine_run(stdout.replace(' 10 15 0\n', ' 10 14 0\n'),
                             trace_fixture(), 'adaptive', 4, 2, 'greedy', policy=policy)

    def test_changed_greedy_histories_disable_speed_verdict(self):
        record = study_fixture()
        run = next(r for r in record['runs'] if r['arm'] == 'chunked')
        run['stdout'] = run['stdout'].replace('token 0 11', 'token 0 12')
        run['parsed'] = parse_engine_run(run['stdout'], record['trace'], 'chunked', 4, 2, 'greedy')
        summary = engine_study_summary(record)
        cell = next(c for c in summary['comparisons'] if c['arm'] == 'chunked')
        self.assertEqual(cell['outcome'], 'different-greedy-histories')

    def test_incomplete_grid_and_changed_summary_fail_with_updated_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = study_fixture()
            write_archive(root, record)
            self.assertEqual(engine_replay(root), record['summary'])
            for corrupt in ('grid', 'summary', 'device', 'token'):
                broken = copy.deepcopy(record)
                if corrupt == 'grid': broken['runs'].pop()
                if corrupt == 'summary': broken['summary']['runs'][0]['duration_ns'] = 1
                if corrupt == 'device': broken['runs'][0]['stdout'] = broken['runs'][0]['stdout'].replace('Apple M4 Pro/metal', 'cpu/host')
                if corrupt == 'token': broken['runs'][0]['stdout'] = broken['runs'][0]['stdout'].replace('token 0 11 1 1 0 10\n', '')
                write_archive(root, broken)
                with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                    engine_replay(root)

    def test_retained_build_provenance_fails_with_updated_envelope_hashes(self):
        record = study_fixture()
        self.assertIs(validate_engine_record_build(record), record['build'])
        corruptions = [
            ('source', 'repository', 'commit', 'unknown'),
            ('source', 'repository', 'dirty', True),
            ('source', 'sources', 'uv.lock', 'invalid'),
            ('binaries', 'engine', 'sha256', 'invalid'),
            ('binaries', 'engine', 'bytes', 0),
            ('assets', 'prepared_sha256', ''),
            ('assets', 'tables_sha256', 'invalid'),
            ('environment', 'hardware', {}),
            ('environment', 'software', {}),
            ('declaration', {}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for *path, value in corruptions:
                broken = copy.deepcopy(record)
                node = broken['build']
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
                write_archive(root, broken)
                with self.subTest(path=path), self.assertRaises(ValueError):
                    validate_engine_record_build(broken)
                with self.subTest(path=path), self.assertRaises(ValueError):
                    engine_replay(root)


if __name__ == '__main__':
    unittest.main()
