"""Acceptance-evidence corruption must fail even with a freshly hashed envelope."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from llm_mojo.benchmarks import model_contract as contract
from llm_mojo.benchmarks import model_profile as profile
from llm_mojo.benchmarks.model_profile import (
    engine_run_summary, engine_study_summary, engine_replay,
    engine_collect, engine_specification, engine_trace_tsv, parse_engine_run,
    validate_engine_record_build, execute, checked_execution, NativeCommandError,
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
    stem = {'qwen-engine-admission-v1': 'engine-admission',
            'qwen-engine-admission-range-v1': 'engine-admission-range'}.get(record['kind'], 'engine-core')
    (root/(stem+'.json.gz')).write_bytes(compressed)
    (root/(stem+'.json')).write_text(json.dumps(dict(kind=record['kind'], bytes=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(), uncompressed_sha256=hashlib.sha256(raw).hexdigest())))


def admission_fixture():
    record = study_fixture()
    record.update(kind=contract.ENGINE_ADMISSION_DECLARATION['kind'],
                  declaration=contract.ENGINE_ADMISSION_DECLARATION, max_sequences=8, runs=[])
    record['build']['declaration'] = contract.ENGINE_ADMISSION_DECLARATION
    conditions = dict(battery={'power_source': 'AC Power'}, power_mode_raw='0',
                      thermal=['No thermal warning level has been recorded',
                               'No performance warning level has been recorded'])
    for block in range(4):
        cells = [('incremental', False), ('incremental', True), ('reserved', False)]
        if block in (1, 2):
            cells.reverse()
        for admission, calibration in cells:
            stdout = native_fixture('chunked').replace('config chunked 4 256 2', 'config chunked 4 256 8')
            stdout += '\nadmission ' + admission
            record['runs'].append(dict(block=block, arm='chunked', admission=admission,
                                      calibration=calibration, stdout=stdout,
                                      parsed=parse_engine_run(stdout, record['trace'], 'chunked', 4, 8,
                                                              'greedy', admission=admission),
                                      conditions_before=conditions, conditions_after=conditions))
    record['summary'] = engine_study_summary(record)
    return record


def range_native_fixture(admission='reserved'):
    """Both requests stop naturally after one token despite their three-token caps."""
    stdout = native_fixture('chunked').replace('config chunked 4 256 2', 'config chunked 4 256 8')
    stdout = stdout.replace('finish 0 length', 'finish 0 stop').replace('finish 1 length', 'finish 1 stop')
    stdout = stdout.replace('1 1 0 10', '1 1 0 8').replace('1 1 0 20', '1 1 0 18')
    lines = [stdout, 'admission '+admission, 'telemetry admission-range-v1', 'kv_geometry 24 2 64 2']
    extent = 3 if admission == 'reserved' else 1
    for i in range(2):
        begin = i*10
        lines += [f'kv {i} start {begin} 0 0 0 0 {2-i} 0',
                  f'kv {i} scheduled {begin+1} 1 0 0 {extent} {1-i} 1',
                  f'kv {i} executed {begin+8} 1 1 1 {extent} {1-i} 1',
                  f'kv {i} end {begin+10} 0 0 0 0 {1-i} 0',
                  f'kv_execute {i} {begin+2} {begin+8}',
                  f'admit {i} {i} {begin+1}']
    return '\n'.join(lines)


def range_fixture():
    record = admission_fixture()
    record.update(kind=contract.ENGINE_ADMISSION_RANGE_DECLARATION['kind'],
                  declaration=contract.ENGINE_ADMISSION_RANGE_DECLARATION,
                  native_trace_path='/collection/trace.tsv')
    record['build']['declaration'] = contract.ENGINE_ADMISSION_RANGE_DECLARATION
    record['build']['assets']['prepared'] = '/prepared'
    record['build']['command'] = ['mojo', 'build', '-I', 'src',
                                  'src/llm_mojo/benchmarks/engine_trace.mojo', '-o', '/build/engine']
    for request in record['trace']['requests']:
        request.update(max_new_tokens=3, stop_ids=[11])
    record['trace_document'] = json.dumps(record['trace'])
    record['trace_sha256'] = hashlib.sha256(record['trace_document'].encode()).hexdigest()
    for run in record['runs']:
        run['stdout'] = range_native_fixture(run['admission'])
        run['parsed'] = parse_engine_run(run['stdout'], record['trace'], 'chunked', 4, 8, 'greedy',
                                         admission=run['admission'], observation='admission-range-v1')
        run['execution'] = dict(command=['/build/engine', '/prepared', '/collection/trace.tsv',
                                         'chunked', '4', '256', '8', '10', 'greedy', run['admission'],
                                         'admission-range-v1'], timeout_seconds=180, exit_code=0,
                                wall_elapsed_ns=100)
    record['summary'] = engine_study_summary(record)
    return record


class EngineTraceTests(unittest.TestCase):
    def test_collection_preflight_accepts_zero_output_beyond_pool_capacity(self):
        # Valid context, but the prompt cannot fit in one 32-slot KV block.
        # Stop immediately after preflight so this test launches no model work.
        studies = [(False, False, contract.ENGINE_DECLARATION),
                   (True, False, contract.ENGINE_ADMISSION_DECLARATION),
                   (False, True, contract.ENGINE_ADMISSION_RANGE_DECLARATION)]
        for admission_pair, admission_range, declaration in studies:
            for maximum in (0, 1):
                with self.subTest(study=declaration['kind'], maximum=maximum), \
                        tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    trace = trace_fixture()
                    trace['requests'] = trace['requests'][:1]
                    trace['requests'][0].update(prompt_ids=[42]*33, max_new_tokens=maximum)
                    trace_path = root/'trace.json'
                    trace_path.write_text(json.dumps(trace))
                    receipt = copy.deepcopy(study_fixture()['build'])
                    receipt['declaration'] = declaration
                    with mock.patch.object(profile, 'verify_build', return_value=receipt), \
                            mock.patch.object(profile, 'conditions', side_effect=RuntimeError('after preflight')) as collect, \
                            mock.patch.object(profile, 'execute') as execute:
                        expected = RuntimeError if maximum == 0 else ValueError
                        message = 'after preflight' if maximum == 0 else 'cannot fit the pool alone'
                        with self.assertRaisesRegex(expected, message):
                            engine_collect(root/'build', trace_path, root/'output', blocks=1,
                                           admission_pair=admission_pair, admission_range=admission_range)
                        self.assertEqual(collect.call_count, int(maximum == 0))
                        execute.assert_not_called()

    def test_range_early_stops_admission_delay_and_bounded_byte_time(self):
        record = range_fixture()
        run = next(r for r in record['summary']['runs'] if r['admission'] == 'reserved')
        self.assertEqual(run['delivered_tokens'], 2)
        self.assertEqual(run['total_tokens'], 2)
        self.assertEqual(run['terminal_reasons'], {'stop': 2})
        self.assertEqual(run['admission_delay']['p50_ns'], 6)
        self.assertEqual([r['admissions'] for r in run['requests']], [1, 1])
        occupancy = run['kv_occupancy']
        token_bytes, block_bytes = 24*2*2*64*2, 32*24*2*2*64*2
        self.assertEqual(occupancy['allocated_byte_ns'], 12*block_bytes)
        self.assertEqual(occupancy['unused_block_byte_ns_bounds'], {'lower': 0, 'upper': 12*block_bytes})
        self.assertEqual(occupancy['unused_slot_byte_ns_bounds'],
                         {'lower': 12*31*token_bytes, 'upper': 12*32*token_bytes})
        self.assertEqual(occupancy['unwritten_reserved_extent_byte_ns_bounds'],
                         {'lower': 12*2*token_bytes, 'upper': 12*3*token_bytes})
        self.assertEqual(occupancy['covered_ns'], 12)
        self.assertEqual(occupancy['excluded_ns'], 8)
        self.assertEqual(occupancy['peak_allocated_blocks'], 1)

    def test_range_replay_validates_observer_contract_and_raw_boundaries(self):
        record = range_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_archive(root, record)
            self.assertEqual(engine_replay(root), record['summary'])
            self.assertEqual(engine_replay(root/'engine-admission-range.json.gz'), record['summary'])
            mutations = [
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace('telemetry admission-range-v1', 'telemetry unknown')),
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace('kv_geometry 24 2 64 2', 'kv_geometry 1 1 1 2')),
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace('kv 0 scheduled 1 1 0 0 1 1 1\n', '')),
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace('kv 0 executed 8 1 1 1', 'kv 0 executed 8 2 1 1')),
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace('kv 0 executed 8 1 1 1', 'kv 0 executed 8 1 1 33')),
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace('kv_execute 0 2 8', 'kv_execute 0 2 9')),
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace('kv 0 executed 8', 'kv 0 executed 9')),
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace('admit 0 0 1', 'admit 0 0 2')),
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace('admit 1 1 11', 'admit 1 0 11')),
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace(
                    'admit 0 0 1', 'admit 0 1 1').replace('admit 1 1 11', 'admit 1 0 11')),
                lambda r: r['runs'][0].update(stdout=r['runs'][0]['stdout'].replace('kv 1 start 10 0 0 0 0', 'kv 1 start 10 1 0 0 3')),
                lambda r: r['summary']['runs'][0]['kv_occupancy'].update(allocated_byte_ns=1),
                lambda r: r['runs'][0]['parsed']['admissions'][0].update(timestamp_ns=True),
                lambda r: r['build']['declaration'].update(execution_timeout_seconds=181),
                lambda r: r['runs'][0]['execution'].update(exit_code=True),
                lambda r: r['runs'][0]['execution'].update(wall_elapsed_ns=180_000_000_001),
                lambda r: r['runs'][0]['execution']['command'].__setitem__(9, 'reserved'),
            ]
            for index, mutate in enumerate(mutations):
                broken = copy.deepcopy(record)
                mutate(broken)
                write_archive(root, broken)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    engine_replay(root)

    def test_range_collector_selector_timeout_and_immutable_legacy_declarations(self):
        record = range_fixture()
        receipt = copy.deepcopy(record['build'])
        receipt['assets']['prepared'] = '/prepared'
        observed = []
        def run(command, log, timeout):
            self.assertEqual(command[3:9], ['chunked', 4, 256, 8, 10, 'greedy'])
            self.assertEqual(command[10], 'admission-range-v1')
            self.assertEqual(timeout, 180)
            observed.append(command[9])
            return range_native_fixture(command[9])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt['command'][-1] = str(root/'build'/'engine')
            trace = root/'trace.json'
            trace.write_text(record['trace_document'])
            with mock.patch('llm_mojo.benchmarks.model_profile.verify_build', return_value=receipt), \
                 mock.patch('llm_mojo.benchmarks.model_profile.conditions', return_value=record['runs'][0]['conditions_before']), \
                 mock.patch('llm_mojo.benchmarks.model_profile.execute', side_effect=run):
                result = engine_collect(root/'build', trace, root/'result', blocks=4, admission_range=True)
            self.assertEqual(result, record['summary'])
            self.assertEqual(len(observed), 12)
            with self.assertRaises(ValueError):
                engine_collect(root/'build', trace, root/'bad', admission_pair=True, admission_range=True)
        self.assertNotIn('observation', contract.ENGINE_ADMISSION_DECLARATION)
        self.assertNotIn('execution_timeout_seconds', contract.ENGINE_DECLARATION)

    def test_range_telemetry_must_be_explicit_and_virtual_clock_has_no_speed_claim(self):
        record = range_fixture()
        run = record['runs'][0]
        with self.assertRaises(ValueError):
            parse_engine_run(run['stdout'], record['trace'], 'chunked', 4, 8, 'greedy', admission='incremental')
        record['mode'] = 'scripted'
        for run in record['runs']:
            run['stdout'] = run['stdout'].replace('Apple M4 Pro/metal', 'simulated/virtual').replace(
                'mode greedy', 'mode scripted').replace('kv_geometry 24 2 64 2', 'kv_geometry 1 1 1 2')
            run['execution']['command'][8] = 'scripted'
            # Virtual elapsed time need not fit within actual process wall time.
            run['execution']['wall_elapsed_ns'] = 1
            run['parsed'] = parse_engine_run(run['stdout'], record['trace'], 'chunked', 4, 8, 'scripted',
                                             admission=run['admission'], observation='admission-range-v1')
        summary = engine_study_summary(record)
        self.assertEqual(summary['comparisons'][0]['outcome'], 'virtual-clock')
        self.assertEqual(summary['runs'][0]['kv_occupancy']['token_bytes'], 4)

    def test_range_greedy_execution_wall_covers_the_complete_native_drain(self):
        record = range_fixture()
        execution = record['runs'][0]['execution']
        elapsed = record['runs'][0]['parsed']['drained']['elapsed_ns']
        execution['wall_elapsed_ns'] = elapsed - 1
        with self.assertRaisesRegex(ValueError, 'native execution receipt'):
            engine_study_summary(record)
        execution['wall_elapsed_ns'] = elapsed
        self.assertEqual(engine_study_summary(record), record['summary'])

    def test_range_reservation_must_match_lifetime_extent_even_with_early_stop(self):
        record = range_fixture()
        trace = copy.deepcopy(record['trace'])
        trace['requests'][0]['max_new_tokens'] = 64
        # A natural one-token stop cannot retroactively reduce its admission reservation.
        with self.assertRaisesRegex(ValueError, 'lifetime extents'):
            parse_engine_run(range_native_fixture('reserved'), trace, 'chunked', 4, 8, 'greedy',
                             admission='reserved', observation='admission-range-v1')
        self.assertEqual(parse_engine_run(range_native_fixture('incremental'), trace, 'chunked', 4, 8,
                                          'greedy', admission='incremental', observation='admission-range-v1')['admission'],
                         'incremental')

    def test_timeout_retains_partial_stdout_and_reports_the_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'failed.log'
            failure = subprocess.TimeoutExpired(['engine'], 180, output=b'partial execution\n')
            with mock.patch('llm_mojo.benchmarks.model_profile.subprocess.run', side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, 'partial output retained'):
                    execute(['engine'], log, timeout=180)
            self.assertEqual(log.read_text(), 'partial execution\n')

    def test_completed_native_failure_preserves_numeric_return_code_and_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'failed.log'
            result = subprocess.CompletedProcess(['engine'], 7, stdout='actual failure output\n')
            with mock.patch('llm_mojo.benchmarks.model_profile.subprocess.run', return_value=result):
                with self.assertRaises(NativeCommandError) as caught:
                    execute(['engine'], log, timeout=180)
            self.assertEqual(caught.exception.returncode, 7)
            self.assertEqual(log.read_text(), 'actual failure output\n')

    def test_admission_metadata_is_bound_to_an_explicit_policy(self):
        stdout = native_fixture('chunked') + '\nadmission reserved'
        parsed = parse_engine_run(stdout, trace_fixture(), 'chunked', 4, 2, 'greedy', admission='reserved')
        self.assertEqual(parsed['admission'], 'reserved')
        cases = [(stdout, None), (stdout, 'incremental'), (native_fixture('chunked'), 'reserved'),
                 (stdout+'\nadmission reserved', 'reserved'), (stdout, 'unknown')]
        for value, admission in cases:
            with self.subTest(admission=admission), self.assertRaises(ValueError):
                parse_engine_run(value, trace_fixture(), 'chunked', 4, 2, 'greedy', admission=admission)
        with self.assertRaises(ValueError):
            parse_engine_run(native_fixture('continuous')+'\nadmission reserved', trace_fixture(),
                             'continuous', 4, 2, 'greedy', admission='reserved')

    def test_admission_pair_replay_and_corruption_gates(self):
        record = admission_fixture()
        self.assertEqual(len(record['summary']['runs']), 12)
        comparison = record['summary']['comparisons'][0]
        self.assertEqual(comparison['control_admission'], 'incremental')
        self.assertEqual(comparison['admission'], 'reserved')
        self.assertTrue(comparison['same_generated_histories'])
        self.assertEqual(comparison['block_ratios'], [1.0]*4)
        self.assertIsNone(record['summary']['target'])
        self.assertIsNone(record['summary']['goodput'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_archive(root, record)
            self.assertEqual(engine_replay(root), record['summary'])
            self.assertEqual(engine_replay(root/'engine-admission.json.gz'), record['summary'])
            for corruption in ('grid', 'order', 'cell_type', 'sequences', 'policy', 'build', 'declaration',
                               'declaration_type', 'summary', 'summary_type'):
                broken = copy.deepcopy(record)
                if corruption == 'grid': broken['runs'].pop()
                if corruption == 'order': broken['runs'][0], broken['runs'][1] = broken['runs'][1], broken['runs'][0]
                if corruption == 'cell_type': broken['runs'][1]['calibration'] = 1
                if corruption == 'sequences': broken['max_sequences'] = 7
                if corruption == 'policy': broken['runs'][0]['stdout'] = broken['runs'][0]['stdout'].replace(
                    'admission incremental', 'admission reserved')
                if corruption == 'build': broken['build']['declaration'] = contract.ENGINE_DECLARATION
                if corruption == 'declaration': broken['declaration']['token_budget'] = 128
                if corruption == 'declaration_type': broken['declaration']['asynchronous'] = 0
                if corruption == 'summary': broken['summary']['runs'][0]['duration_ns'] = 1
                if corruption == 'summary_type': broken['summary']['comparisons'][0]['same_generated_histories'] = 1
                write_archive(root, broken)
                with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                    engine_replay(root)

    def test_admission_collector_uses_one_build_and_balanced_fixed_work(self):
        record = admission_fixture()
        receipt = copy.deepcopy(record['build'])
        receipt['assets']['prepared'] = '/prepared'
        observed = []
        def execute(command, log):
            self.assertEqual(command[3:9], ['chunked', 4, 256, 8, 10, 'greedy'])
            observed.append(command[9])
            return native_fixture('chunked').replace('config chunked 4 256 2', 'config chunked 4 256 8') + '\nadmission ' + command[9]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root/'trace.json'
            trace.write_text(json.dumps(record['trace']))
            with mock.patch('llm_mojo.benchmarks.model_profile.verify_build', return_value=receipt), \
                 mock.patch('llm_mojo.benchmarks.model_profile.conditions', return_value=record['runs'][0]['conditions_before']), \
                 mock.patch('llm_mojo.benchmarks.model_profile.execute', side_effect=execute):
                result = engine_collect(root/'build', trace, root/'result', blocks=4, admission_pair=True)
            self.assertEqual(observed, ['incremental', 'incremental', 'reserved',
                                       'reserved', 'incremental', 'incremental',
                                       'reserved', 'incremental', 'incremental',
                                       'incremental', 'incremental', 'reserved'])
            self.assertEqual(result, record['summary'])
            with self.assertRaises(ValueError):
                engine_collect(root/'build', trace, root/'bad', maximum_sequences=7, admission_pair=True)

    def test_admission_greedy_divergence_fails_and_virtual_clock_cannot_claim_speed(self):
        record = admission_fixture()
        run = next(r for r in record['runs'] if r['admission'] == 'reserved')
        run['stdout'] = run['stdout'].replace('token 0 11', 'token 0 12')
        run['parsed'] = parse_engine_run(run['stdout'], record['trace'], 'chunked', 4, 8,
                                         'greedy', admission='reserved')
        with self.assertRaises(ValueError):
            engine_study_summary(record)
        record = admission_fixture()
        run = next(r for r in record['runs'] if r['calibration'])
        run['stdout'] = run['stdout'].replace('token 0 11', 'token 0 12')
        run['parsed'] = parse_engine_run(run['stdout'], record['trace'], 'chunked', 4, 8,
                                         'greedy', admission='incremental')
        with self.assertRaises(ValueError):
            engine_study_summary(record)
        record = admission_fixture()
        record['mode'] = 'scripted'
        for run in record['runs']:
            run['stdout'] = run['stdout'].replace('Apple M4 Pro/metal', 'simulated/virtual').replace('mode greedy', 'mode scripted')
            run['parsed'] = parse_engine_run(run['stdout'], record['trace'], 'chunked', 4, 8,
                                             'scripted', admission=run['admission'])
        self.assertEqual(engine_study_summary(record)['comparisons'][0]['outcome'], 'virtual-clock')

    def test_reserved_preemptions_and_recomputed_rows_fail(self):
        stdout = native_fixture('chunked') + '\nadmission reserved'
        for field, value in [('preempted', 1), ('total_tokens', 2)]:
            lines = stdout.splitlines()
            first = next(i for i, line in enumerate(lines) if line.startswith('step 0 '))
            values = lines[first].split()
            values[contract.ENGINE_STEP_FIELDS.index(field)+1] = str(value)
            if field == 'total_tokens':
                values[contract.ENGINE_STEP_FIELDS.index('prefill_tokens')+1] = '2'
            lines[first] = ' '.join(values)
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_engine_run('\n'.join(lines), trace_fixture(), 'chunked', 4, 2,
                                 'greedy', admission='reserved')
        # Incremental replay can do extra work under pressure; retain its rows.
        parsed = parse_engine_run('\n'.join(lines).replace('admission reserved', 'admission incremental'),
                                  trace_fixture(), 'chunked', 4, 2, 'greedy', admission='incremental')
        self.assertEqual(engine_run_summary(parsed, trace_fixture())['total_tokens'], 3)

    def test_reserved_zero_output_and_no_abort_study_scope(self):
        trace = trace_fixture()
        trace['requests'][0]['max_new_tokens'] = 0
        stdout = (native_fixture('chunked')+'\nadmission reserved').replace(
            'token 0 11 1 1 0 10\n', '').replace('finish 0 length 1 1 0 10', 'finish 0 length 1 0 0 10')
        lines = stdout.splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith('step 0 '))
        values = lines[first].split()
        for field in ('prefill_seqs', 'prefill_tokens', 'total_tokens', 'attended_positions'):
            values[contract.ENGINE_STEP_FIELDS.index(field)+1] = '0'
        lines[first] = ' '.join(values)
        parsed = parse_engine_run('\n'.join(lines), trace, 'chunked', 4, 2, 'greedy', admission='reserved')
        self.assertEqual(engine_run_summary(parsed, trace)['total_tokens'], 1)
        self.assertIsNone(engine_run_summary(parsed, trace)['requests'][0]['ttft_ns'])
        record = admission_fixture()
        record['trace']['requests'][0]['abort_offset_ns'] = 10
        with self.assertRaises(ValueError):
            engine_study_summary(record)
        record = admission_fixture()
        run = next(r for r in record['runs'] if r['admission'] == 'incremental')
        run['stdout'] = run['stdout'].replace('finish 0 length', 'finish 0 error')
        run['parsed'] = parse_engine_run(run['stdout'], record['trace'], 'chunked', 4, 8,
                                         'greedy', admission='incremental')
        with self.assertRaises(ValueError):
            engine_study_summary(record)

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


class EngineExecutionReceiptTests(unittest.TestCase):
    def test_actual_failure_code_and_original_log_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'native.log'
            log.write_text('nonfinite head detected\n')
            with mock.patch.object(profile, 'execute', side_effect=NativeCommandError(17, log)):
                with self.assertRaises(NativeCommandError):
                    checked_execution(['/absolute/native'], log, 180)
            receipt = json.loads(log.with_suffix('.execution.json').read_text())
            self.assertEqual(receipt['exit_code'], 17)
            self.assertEqual(receipt['command'], ['/absolute/native'])
            self.assertGreater(receipt['wall_elapsed_ns'], 0)
            self.assertIn('error', receipt)
            self.assertEqual(log.read_text(), 'nonfinite head detected\n')

    def test_timeout_does_not_claim_a_successful_numeric_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'native.log'
            with mock.patch.object(profile, 'execute', side_effect=RuntimeError('native timeout')):
                with self.assertRaises(RuntimeError):
                    checked_execution(['/absolute/native'], log, 180)
            receipt = json.loads(log.with_suffix('.execution.json').read_text())
            self.assertIsNone(receipt['exit_code'])
            self.assertIn('timeout', receipt['error'])

    def test_success_receipt_records_the_same_executed_command(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'native.log'
            with mock.patch.object(profile, 'execute', return_value='actual native output') as execute:
                stdout, receipt = checked_execution(['/absolute/native', 'prepared'], log, 180)
            execute.assert_called_once_with(['/absolute/native', 'prepared'], log, timeout=180)
            self.assertEqual(stdout, 'actual native output')
            self.assertEqual(receipt, json.loads(log.with_suffix('.execution.json').read_text()))
            self.assertEqual(receipt['exit_code'], 0)
            self.assertNotIn('error', receipt)

    def test_failed_admission_range_cell_preserves_actual_exit_and_partial_log(self):
        record = range_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = record['build']
            receipt['command'][-1] = str(root/'build'/'engine')
            trace = root/'trace.json'
            trace.write_text(record['trace_document'])
            output = root/'failed'

            def failed(command, log, timeout):
                self.assertEqual(timeout, 180)
                log.write_text('partial admission native output\n')
                raise NativeCommandError(23, log)

            with mock.patch.object(profile, 'verify_build', return_value=receipt), \
                    mock.patch.object(profile, 'conditions', return_value=record['runs'][0]['conditions_before']), \
                    mock.patch.object(profile, 'execute', side_effect=failed):
                with self.assertRaises(NativeCommandError):
                    engine_collect(root/'build', trace, output, blocks=4, admission_range=True)
            log = output/'block-0-incremental.log'
            self.assertEqual(log.read_text(), 'partial admission native output\n')
            execution = json.loads(log.with_suffix('.execution.json').read_text())
            self.assertEqual(execution['exit_code'], 23)
            self.assertEqual(execution['command'][0], str(root/'build'/'engine'))
            self.assertEqual(execution['timeout_seconds'], 180)
            self.assertFalse((output/'engine-admission-range.json.gz').exists())


if __name__ == '__main__':
    unittest.main()
