"""Async tickets and delivered/discarded results have independent ownership."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from llm_mojo.benchmarks import engine_async as async_study
from llm_mojo.benchmarks import model_contract as contract
from llm_mojo.benchmarks import model_profile as profile


def trace_fixture():
    return dict(kind='engine-token-trace-v1', schema_version=1, mode='offline', scripted_tokens=[11],
        requests=[dict(request_id=0, arrival_offset_ns=0, prompt_ids=[42], max_new_tokens=2,
                       stop_ids=[11], abort_offset_ns=None, output_script=None)])


def native_fixture(execution='async'):
    lines = ['device Apple M4 Pro/metal', 'mode greedy', 'config chunked 4 256 8',
        'admission reserved', 'study engine-async-v1', 'work_capacity 256 8',
        'execution ' + execution, 'completion_mode ' + contract.ENGINE_ASYNC_DECLARATION['completion_modes'][execution],
        'async_capacity 2', 'arrival 0 0 0',
        'submit 0 0 0 1 1 1 1 1 0 1 1', 'head 0 0 0 1']
    if execution == 'async':
        lines += ['submit 1 1 1 0 0 1 2 1 1 2 2', 'head 1 0 0 2']
    lines += ['complete 0 2 10 ' + ('1' if execution == 'async' else '0'),
        'result 0 0 0 11 1 delivered 10', 'async_token 0 11 1 1 0 10 0',
        'async_finish 0 stop 1 1 0 10 0']
    if execution == 'async':
        lines += ['complete 1 10 18 0', 'result 1 0 0 13 2 discarded-stop 18',
                  'async_drained 1 2 2 1 1 4 0 0 0 0 18']
    else:
        lines += ['async_drained 1 1 1 1 0 4 0 0 0 0 10']
    return '\n'.join(lines)


def three_ticket_fixture():
    trace = trace_fixture()
    trace['requests'][0].update(max_new_tokens=3, stop_ids=[])
    lines = native_fixture().splitlines()[:10]
    lines += ['submit 0 0 0 1 1 1 1 1 0 1 1', 'head 0 0 0 1',
        'submit 1 1 1 0 0 1 2 1 1 2 2', 'head 1 0 0 2',
        'complete 0 2 10 1', 'result 0 0 0 11 1 delivered 10', 'async_token 0 11 1 1 0 10 0',
        'submit 2 0 1 0 0 1 3 1 10 11 2', 'head 2 0 0 3',
        'complete 1 11 18 1', 'result 1 0 0 13 2 delivered 18', 'async_token 0 13 1 2 0 18 1',
        'complete 2 18 25 0', 'result 2 0 0 17 3 delivered 25', 'async_token 0 17 1 3 0 25 2',
        'async_finish 0 length 1 3 0 25 2', 'async_drained 1 3 3 3 0 4 0 0 0 0 25']
    return '\n'.join(lines), trace


def study_fixture():
    build = dict(source=dict(repository=dict(commit='a'*40, dirty=False),
                             sources={'uv.lock': 'b'*64, 'tests/engine_async_metal_driver.mojo': 'c'*64}),
        assets=dict(prepared='/private/tmp/prepared', prepared_sha256='d'*64, tables_sha256='e'*64),
        environment=dict(hardware={'chip': 'Apple M4 Pro'}, software={'mojo': '1.1.0', 'max': '26.6.0'}),
        declaration=contract.ENGINE_ASYNC_DECLARATION,
        command=['/private/tmp/mojo', 'build', '-I', 'src', 'src/llm_mojo/benchmarks/engine_trace.mojo', '-o', '/private/tmp/engine'],
        binaries={'engine': dict(sha256='f'*64, bytes=1)})
    driver = dict(source=build['source'], assets=build['assets'], environment=build['environment'],
        command=['/private/tmp/mojo', 'build', '-I', 'src', '-I', 'tests', 'tests/engine_async_metal_driver.mojo', '-o', '/private/tmp/checkpoint-driver'],
        binaries={'checkpoint-driver': dict(sha256='1'*64, bytes=2)})
    stdout = '\n'.join(['device Apple M4 Pro/metal', 'qualification engine-async-checkpoint-v1',
        *[f'check {name} checked 5 mismatches 0 unexpected-nonfinite 0' for name in contract.ENGINE_ASYNC_CHECKPOINT_CHECKS]])
    qualification = dict(kind='engine-async-qualification-v1', schema_version=1, build=build,
        checkpoint_receipt=dict(build=driver,
            build_execution=dict(command=driver['command'], exit_code=0, timeout_seconds=600, wall_elapsed_ns=100),
            stdout=stdout, stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
            execution=dict(command=['/private/tmp/checkpoint-driver', '/private/tmp/prepared', 'async-qualification'],
                           exit_code=0, timeout_seconds=180, wall_elapsed_ns=100)))
    trace = trace_fixture()
    document, qualification_document = json.dumps(trace), json.dumps(qualification)
    record = dict(kind=contract.ENGINE_ASYNC_DECLARATION['kind'], declaration=contract.ENGINE_ASYNC_DECLARATION,
        build=build, trace=trace, trace_document=document, trace_sha256=hashlib.sha256(document.encode()).hexdigest(),
        blocks=4, max_sequences=8, token_budget=256, admission='reserved', mode='greedy', warmup_steps=10,
        native_trace_path='/private/tmp/trace.tsv', native_trace_sha256=hashlib.sha256(profile.engine_trace_tsv(trace).encode()).hexdigest(),
        qualification=qualification, qualification_document=qualification_document,
        qualification_sha256=hashlib.sha256(qualification_document.encode()).hexdigest(), runs=[])
    conditions = dict(battery={'power_source': 'AC Power'}, power_mode_raw='0',
        thermal=['No thermal warning level has been recorded', 'No performance warning level has been recorded'])
    for block in range(4):
        for execution, calibration in async_study.cells(block):
            stdout = native_fixture(execution)
            run = dict(block=block, execution=execution, calibration=calibration, stdout=stdout,
                parsed=async_study.parse_async_run(stdout, trace, execution, 4),
                conditions_before=conditions, conditions_after=conditions)
            run['native_execution'] = dict(command=async_study._command(record, run), exit_code=0,
                                          timeout_seconds=180, wall_elapsed_ns=100)
            record['runs'].append(run)
    record['summary'] = async_study.summary(record)
    return record


def write_archive(root, record):
    payload = json.dumps(record).encode()
    compressed = gzip.compress(payload, mtime=0)
    (root/'engine-async.json.gz').write_bytes(compressed)
    (root/'engine-async.json').write_text(json.dumps(dict(kind=record['kind'], bytes=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(), uncompressed_sha256=hashlib.sha256(payload).hexdigest())))


class AsyncGrammarTests(unittest.TestCase):
    def parse(self, stdout, execution='async', trace=None):
        return async_study.parse_async_run(stdout, trace or trace_fixture(), execution, 4)

    def test_one_stop_delivered_and_one_speculative_result_discarded(self):
        parsed = self.parse(native_fixture())
        result = async_study.run_summary(parsed, trace_fixture())
        self.assertEqual(result['delivered_tokens'], 1)
        self.assertEqual(result['discarded_tokens'], 1)
        self.assertEqual(result['extra_submitted_rows'], 1)
        self.assertEqual(result['max_pending'], 2)
        self.assertEqual(result['requests'][0]['token_ids'], [11])
        self.assertEqual(result['duration_ns'], 18)

    def test_synchronous_control_uses_the_same_delivery_boundary(self):
        result = self.parse(native_fixture('sync'), 'sync')
        self.assertEqual(result['max_pending'], 1)
        self.assertEqual(result['extra_submitted_rows'], 0)
        self.assertEqual(result['drained']['discarded'], 0)

    def test_unknown_backend_cannot_satisfy_metal(self):
        for marker in ('simulated/virtual', 'Apple M4 Pro/cpu', 'Apple M4 Pro/Metal'):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                self.parse(native_fixture().replace('Apple M4 Pro/metal', marker))

    def test_duplicate_missing_and_unowned_records_fail(self):
        original = native_fixture()
        corruptions = [original.replace('head 1 0 0 2', ''),
            original.replace('complete 1 10 18 0', ''),
            original.replace('result 1 0 0 13 2 discarded-stop 18', ''),
            original.replace('head 1 0 0 2', 'head 1 0 0 2\nhead 1 0 0 2'),
            original.replace('result 1 0 0 13 2 discarded-stop 18', 'result 1 0 0 13 2 discarded-stop 18\nresult 1 0 0 13 2 discarded-stop 18'),
            original.replace('result 1 0 0 13 2 discarded-stop 18', 'result 1 0 9 13 2 discarded-stop 18'),
            original.replace('head 1 0 0 2', 'head 1 0 0 3'),
            original.replace('submit 1 1 ', 'submit 1 0 '),
            original.replace('complete 0 2 10 1', 'complete 0 2 10 0')]
        for broken in corruptions:
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                self.parse(broken)

    def test_cannot_hide_discarded_rows_or_deliver_after_stop(self):
        original = native_fixture()
        for broken in (original.replace('discarded-stop', 'delivered'),
                original.replace('discarded-stop', 'discarded-abort'),
                original.replace('async_drained 1 2 2 1 1', 'async_drained 1 2 2 1 0'),
                original.replace('result 1 0 0 13 2 discarded-stop 18', 'result 1 0 0 13 2 discarded-stop 18\nasync_token 0 13 1 2 0 18 1')):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                self.parse(broken)

    def test_completion_and_terminal_timestamps_bound_delivery(self):
        for source, replacement in (
                ('complete 0 2 10 1', 'complete 0 0 10 1'),
                ('complete 1 10 18 0', 'complete 1 9 18 0'),
                ('discarded-stop 18', 'discarded-stop 9'),
                ('delivered 10', 'delivered 9'),
                ('async_finish 0 stop 1 1 0 10 0', 'async_finish 0 stop 1 1 0 9 0')):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.parse(native_fixture().replace(source, replacement))

    def test_host_chronology_binds_reused_banks_and_prefix_collection(self):
        original, trace = three_ticket_fixture()
        self.assertEqual(self.parse(original, trace=trace)['drained']['delivered'], 3)
        for broken in (
                original.replace('submit 2 0 1 0 0 1 3 1 10 11 2', 'submit 2 0 1 0 0 1 3 1 3 4 2'),
                original.replace('complete 0 2 10 1', 'complete 0 1 10 1')):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                self.parse(broken, trace=trace)

    def test_known_cap_cannot_have_a_speculative_head_past_the_limit(self):
        original, trace = three_ticket_fixture()
        trace['requests'][0]['max_new_tokens'] = 2
        broken = original.replace('result 2 0 0 17 3 delivered 25', 'result 2 0 0 17 3 discarded-length 25')
        broken = broken.replace('async_token 0 17 1 3 0 25 2\n', '')
        broken = broken.replace('async_finish 0 length 1 3 0 25 2', 'async_finish 0 length 1 2 0 18 1')
        broken = broken.replace('async_drained 1 3 3 3 0', 'async_drained 1 3 3 2 1')
        with self.assertRaises(ValueError):
            self.parse(broken, trace=trace)

    def test_natural_finish_ticket_must_own_the_last_delivered_token(self):
        original, trace = three_ticket_fixture()
        for ticket in (-1, 0, 1):
            with self.subTest(ticket=ticket), self.assertRaises(ValueError):
                self.parse(original.replace('async_finish 0 length 1 3 0 25 2',
                                            f'async_finish 0 length 1 3 0 25 {ticket}'), trace=trace)
        with self.assertRaises(ValueError):
            self.parse(native_fixture().replace('async_finish 0 stop 1 1 0 10 0',
                                                'async_finish 0 stop 1 1 0 10 -1'))

    def test_zero_output_can_finish_without_submission(self):
        trace = trace_fixture()
        trace['requests'][0]['max_new_tokens'] = 0
        lines = native_fixture('sync').splitlines()[:10]
        lines += ['async_finish 0 length 1 0 0 1 -1', 'async_drained 1 0 0 0 0 4 0 0 0 0 1']
        result = self.parse('\n'.join(lines), 'sync', trace)
        self.assertEqual(result['submitted_rows'], 0)

    def test_abort_can_discard_result_after_boundary_terminal(self):
        trace = trace_fixture()
        trace['requests'][0]['abort_offset_ns'] = 11
        original = native_fixture().replace('async_finish 0 stop 1 1 0 10 0', 'async_finish 0 abort 1 1 0 11 -1').replace('discarded-stop', 'discarded-abort')
        trace['requests'][0]['stop_ids'] = []
        result = self.parse(original, trace=trace)
        self.assertEqual(result['drained']['discarded'], 1)

    def test_nonzero_final_ownership_cannot_pass(self):
        original = native_fixture()
        for tail in ('3 0 0 0 0 18', '4 1 0 0 0 18', '4 0 1 0 0 18',
                     '4 0 0 1 0 18', '4 0 0 0 1 18'):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                self.parse(original.replace('4 0 0 0 0 18', tail))


class AsyncEvidenceTests(unittest.TestCase):
    def test_checkpoint_compile_receipt_is_required_after_rehashing(self):
        record = study_fixture()
        for corruption in ('missing', 'command', 'failed', 'boolean_exit', 'timeout', 'elapsed'):
            broken = copy.deepcopy(record)
            checkpoint = broken['qualification']['checkpoint_receipt']
            compiled = checkpoint['build_execution']
            if corruption == 'missing': del checkpoint['build_execution']
            elif corruption == 'command': compiled['command'] = ['/private/tmp/wrong-compiler']
            elif corruption == 'failed': compiled['exit_code'] = 17
            elif corruption == 'boolean_exit': compiled['exit_code'] = False
            elif corruption == 'timeout': compiled['timeout_seconds'] = 601
            else: compiled['wall_elapsed_ns'] = 600*10**9+1
            broken['qualification_document'] = json.dumps(broken['qualification'])
            broken['qualification_sha256'] = hashlib.sha256(broken['qualification_document'].encode()).hexdigest()
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                async_study.summary(broken)

    def test_full_grid_replays_without_gpu_or_original_files(self):
        record = study_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_archive(root, record)
            self.assertEqual(profile.engine_replay(root), record['summary'])
            self.assertEqual(profile.engine_replay(root/'engine-async.json.gz'), record['summary'])
        self.assertTrue(record['summary']['performance_eligible'])
        self.assertEqual(record['summary']['speed_verdict'], 'slower')

    def test_rehashed_corruption_does_not_substitute_for_semantic_replay(self):
        record = study_fixture()
        for corruption in ('parsed', 'missing_cell', 'order', 'exit_bool', 'source', 'qualification', 'counter'):
            broken = copy.deepcopy(record)
            if corruption == 'parsed': broken['runs'][0]['parsed']['drained']['discarded'] += 1
            elif corruption == 'missing_cell': broken['runs'].pop()
            elif corruption == 'order': broken['runs'][0], broken['runs'][1] = broken['runs'][1], broken['runs'][0]
            elif corruption == 'exit_bool': broken['runs'][0]['native_execution']['exit_code'] = False
            elif corruption == 'source': broken['build']['source']['repository']['dirty'] = True
            elif corruption == 'counter': broken['runs'][2]['stdout'] = broken['runs'][2]['stdout'].replace('1 1 4 0 0 0 0 18', '1 0 4 0 0 0 0 18')
            else:
                checkpoint = broken['qualification']['checkpoint_receipt']
                checkpoint['stdout'] = checkpoint['stdout'].replace('mismatches 0', 'mismatches 1', 1)
                checkpoint['stdout_sha256'] = hashlib.sha256(checkpoint['stdout'].encode()).hexdigest()
                broken['qualification_document'] = json.dumps(broken['qualification'])
                broken['qualification_sha256'] = hashlib.sha256(broken['qualification_document'].encode()).hexdigest()
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_archive(root, broken)
                with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                    profile.engine_replay(root)

    def test_measured_grid_requires_actual_two_pending_candidate(self):
        record = study_fixture()
        for run in record['runs']:
            if run['execution'] == 'async':
                run['stdout'] = native_fixture('sync').replace('execution sync', 'execution async').replace('completion_mode synchronous-readback', 'completion_mode two-context-prefix-wait')
                run['parsed'] = async_study.parse_async_run(run['stdout'], record['trace'], 'async', 4)
        result = async_study.summary(record)
        self.assertFalse(result['performance_eligible'])
        self.assertIsNone(result['speed_verdict'])


class AsyncExecutionReceiptTests(unittest.TestCase):
    def test_relocated_or_malformed_live_build_rejects_before_output_and_execution(self):
        for operation in ('qualify', 'collect'):
            for corruption in ('relocated', 'entrypoint', 'relative'):
                with self.subTest(operation=operation, corruption=corruption), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    build_dir, output = root/'copied-build', root/'output'
                    build = copy.deepcopy(study_fixture()['build'])
                    build['command'][-1] = str(build_dir/'engine')
                    if corruption == 'relocated': build['command'][-1] = str(root/'original-build'/'engine')
                    elif corruption == 'entrypoint': build['command'][4] = 'tests/wrong_driver.mojo'
                    else: build['command'][-1] = 'copied-build/engine'
                    with mock.patch.object(profile, 'verify_build', return_value=build), \
                            mock.patch.object(profile, 'environment_tool', return_value='/private/tmp/mojo'), \
                            mock.patch.object(profile, 'execute') as execute, self.assertRaises(ValueError):
                        if operation == 'qualify':
                            async_study.qualify(build_dir, output)
                        else:
                            async_study.collect(build_dir, root/'missing-trace', output, root/'missing-qualification')
                    execute.assert_not_called()
                    self.assertFalse(output.exists())

    def test_actual_failure_code_and_log_survive_qualification_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'native.log'
            log.write_text('nonfinite head detected\n')
            with mock.patch.object(profile, 'execute', side_effect=profile.NativeCommandError(17, log)):
                with self.assertRaises(profile.NativeCommandError):
                    async_study._checked_execution(['/absolute/native'], log, 180)
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
                    async_study._checked_execution(['/absolute/native'], log, 180)
            receipt = json.loads(log.with_suffix('.execution.json').read_text())
            self.assertIsNone(receipt['exit_code'])
            self.assertIn('timeout', receipt['error'])

    def test_success_receipt_records_the_same_executed_command(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'native.log'
            with mock.patch.object(profile, 'execute', return_value='actual native output') as execute:
                stdout, receipt = async_study._checked_execution(['/absolute/native', 'prepared'], log, 180)
            execute.assert_called_once_with(['/absolute/native', 'prepared'], log, timeout=180)
            self.assertEqual(stdout, 'actual native output')
            self.assertEqual(receipt, json.loads(log.with_suffix('.execution.json').read_text()))
            self.assertEqual(receipt['exit_code'], 0)
            self.assertNotIn('error', receipt)


if __name__ == '__main__':
    unittest.main()
