"""Collect and replay bounded asynchronous request stepping on verified Metal.

Submission is distinct from host-observed completion and token delivery. The
timestamps are host observations, not GPU kernel timings. Speculative work is
retained even when its selected token is discarded after a terminal boundary.
"""
import gzip
import hashlib
import json
from pathlib import Path
import time

from . import model_contract as contract
from . import model_profile as profile
from .environment import ensure_record_location
from ..validation.evidence import sha, write


def _same(value, expected):
    return profile._engine_same_json(value, expected)


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def parse_async_run(stdout, trace, execution, blocks):
    """Check the complete ticket/result ownership grammar before deriving metrics."""
    profile.validate_engine_trace(trace)
    if execution not in contract.ENGINE_ASYNC_EXECUTIONS or type(blocks) is not int or not 1 <= blocks <= 8192:
        raise ValueError('invalid async engine configuration')
    metadata, arrivals, submissions, completions, heads, results, events = {}, {}, [], [], [], [], []
    pending, slots = set(), {}
    max_pending = 0
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[0]
        if name in ('device', 'mode', 'config', 'admission', 'study', 'work_capacity', 'execution', 'completion_mode', 'async_capacity', 'async_drained'):
            if name in metadata:
                raise ValueError('duplicate async execution identity')
            metadata[name] = fields[1:]
        elif name == 'arrival' and len(fields) == 4:
            identifier, scheduled, actual = map(int, fields[1:])
            if identifier in arrivals:
                raise ValueError('duplicate async arrival')
            arrivals[identifier] = dict(scheduled_ns=scheduled, actual_ns=actual)
        elif name == 'submit' and len(fields) == len(contract.ENGINE_ASYNC_SUBMIT_FIELDS) + 1:
            value = dict(zip(contract.ENGINE_ASYNC_SUBMIT_FIELDS, map(int, fields[1:])))
            ticket, slot = value['ticket'], value['buffer_slot']
            if (ticket != len(submissions) or slot not in (0, 1) or slot in slots
                    or any(v < 0 for v in value.values()) or value['pending'] != len(pending) + 1
                    or not 1 <= value['pending'] <= (2 if execution == 'async' else 1)
                    or value['submitted_ns'] < value['begin_ns']
                    or (submissions and value['begin_ns'] < submissions[-1]['submitted_ns'])
                    or (completions and value['begin_ns'] < completions[-1]['completed_ns'])
                    or value['prefill_seqs'] not in (0, 1)
                    or value['prefill_seqs'] != int(value['prefill_tokens'] > 0)
                    or not 0 < value['total_tokens'] <= 256
                    or value['total_tokens'] != value['decode_seqs'] + value['prefill_tokens']
                    or not 1 <= value['decode_seqs'] + value['prefill_seqs'] <= 8
                    or not value['decode_seqs'] <= value['selected_logits'] <= value['decode_seqs'] + value['prefill_seqs']
                    or value['attended_positions'] < value['total_tokens']):
                raise ValueError('invalid async submission or pending buffer reuse')
            pending.add(ticket)
            slots[slot] = ticket
            max_pending = max(max_pending, len(pending))
            submissions.append(value)
        elif name == 'head' and len(fields) == len(contract.ENGINE_ASYNC_HEAD_FIELDS) + 1:
            value = dict(zip(contract.ENGINE_ASYNC_HEAD_FIELDS, map(int, fields[1:])))
            ticket = value['ticket']
            if (ticket not in pending or any(v < 0 for v in value.values())
                    or value['head'] != sum(h['ticket'] == ticket for h in heads)
                    or value['head'] >= submissions[ticket]['selected_logits']
                    or not value['generated_tokens']
                    or any(h['ticket'] == ticket and h['request_id'] == value['request_id'] for h in heads)):
                raise ValueError('invalid or duplicate async head ownership')
            heads.append(value)
        elif name == 'complete' and len(fields) == len(contract.ENGINE_ASYNC_COMPLETE_FIELDS) + 1:
            value = dict(zip(contract.ENGINE_ASYNC_COMPLETE_FIELDS, map(int, fields[1:])))
            ticket = value['ticket']
            if (ticket != len(completions) or ticket not in pending or any(v < 0 for v in value.values())
                    or value['pending'] != len(pending) - 1
                    or value['begin_ns'] < submissions[ticket]['submitted_ns']
                    or value['begin_ns'] < submissions[-1]['submitted_ns']
                    or value['completed_ns'] < value['begin_ns']
                    or (completions and value['begin_ns'] < completions[-1]['completed_ns'])
                    or sum(h['ticket'] == ticket for h in heads) != submissions[ticket]['selected_logits']):
                raise ValueError('missing or unordered async completion')
            pending.remove(ticket)
            del slots[submissions[ticket]['buffer_slot']]
            completions.append(value)
        elif name == 'result' and len(fields) == len(contract.ENGINE_ASYNC_RESULT_FIELDS) + 1:
            values = [*map(int, fields[1:6]), fields[6], int(fields[7])]
            value = dict(zip(contract.ENGINE_ASYNC_RESULT_FIELDS, values))
            ticket, head = value['ticket'], value['head']
            owner = next((h for h in heads if h['ticket'] == ticket and h['head'] == head), None)
            if (not 0 <= ticket < len(completions) or owner is None
                    or value['disposition'] not in contract.ENGINE_ASYNC_DISPOSITIONS
                    or not 0 <= value['token_id'] < 151936
                    or value['request_id'] != owner['request_id']
                    or value['generated_tokens'] != owner['generated_tokens']
                    or value['observed_ns'] < completions[ticket]['completed_ns']
                    or any(r['ticket'] == ticket and r['head'] == head for r in results)):
                raise ValueError('invalid, duplicate or unowned async result')
            results.append(value)
        elif name in ('async_token', 'async_finish') and len(fields) == 8:
            identifier, ticket = int(fields[1]), int(fields[7])
            events.append(dict(kind='token' if name == 'async_token' else 'finish', request_id=identifier,
                token_id=int(fields[2]) if name == 'async_token' else None,
                reason=fields[2] if name == 'async_finish' else None,
                prompt_tokens=int(fields[3]), generated_tokens=int(fields[4]),
                arrival_ns=int(fields[5]), emitted_ns=int(fields[6]), ticket=ticket))
        else:
            raise ValueError('unknown or malformed async line: ' + line[:120])
    expected_metadata = dict(device=['Apple', 'M4', 'Pro/metal'], mode=['greedy'],
        config=['chunked', str(blocks), '256', '8'], admission=['reserved'],
        study=[contract.ENGINE_ASYNC_STUDY], work_capacity=['256', '8'],
        execution=[execution], completion_mode=[contract.ENGINE_ASYNC_DECLARATION['completion_modes'][execution]],
        async_capacity=['2'])
    if {k: v for k, v in metadata.items() if k != 'async_drained'} != expected_metadata:
        raise ValueError('async runtime device/backend or declared configuration changed')
    expected = {r['request_id']: r for r in trace['requests']}
    if set(arrivals) != set(expected) or pending or slots or len(submissions) != len(completions):
        raise ValueError('async trace omitted arrivals or pending work')
    if len(heads) != len(results) or len(results) != sum(s['selected_logits'] for s in submissions):
        raise ValueError('async trace omitted a completed selected head')
    histories, terminal, last_time = {i: [] for i in expected}, {}, {}
    for event in events:
        identifier = event['request_id']
        if identifier not in expected or identifier in terminal:
            raise ValueError('unknown async request or delivery after terminal event')
        request, arrival = expected[identifier], arrivals[identifier]
        if (arrival['scheduled_ns'] != request['arrival_offset_ns'] or arrival['actual_ns'] < arrival['scheduled_ns']
                or event['arrival_ns'] != arrival['scheduled_ns']
                or event['emitted_ns'] < max(arrival['actual_ns'], last_time.get(identifier, 0))
                or event['prompt_tokens'] != len(request['prompt_ids'])):
            raise ValueError('invalid async event time or prompt accounting')
        tokens = histories[identifier]
        if event['kind'] == 'token':
            matches = [r for r in results if r['ticket'] == event['ticket'] and r['request_id'] == identifier
                       and r['disposition'] == 'delivered']
            if (len(matches) != 1 or matches[0]['token_id'] != event['token_id']
                    or matches[0]['generated_tokens'] != event['generated_tokens']
                    or matches[0]['observed_ns'] != event['emitted_ns']
                    or event['generated_tokens'] != len(tokens) + 1
                    or len(tokens) >= request['max_new_tokens']
                    or (tokens and tokens[-1]['token_id'] in request['stop_ids'])):
                raise ValueError('async token delivery differs from completed result or boundary')
            tokens.append(event)
        else:
            reason = event['reason']
            if (event['generated_tokens'] != len(tokens) or reason not in ('stop', 'length', 'abort', 'error')
                    or not -1 <= event['ticket'] < len(completions)
                    or (event['ticket'] >= 0 and event['emitted_ns'] < completions[event['ticket']]['completed_ns'])
                    or (reason == 'length' and len(tokens) != request['max_new_tokens'])
                    or (reason == 'stop' and (not tokens or tokens[-1]['token_id'] not in request['stop_ids']))
                    or (reason == 'length' and tokens and tokens[-1]['token_id'] in request['stop_ids'])
                    or (reason in ('stop', 'length') and tokens and event['ticket'] != tokens[-1]['ticket'])
                    or (reason == 'abort' and (request['abort_offset_ns'] is None or event['emitted_ns'] < request['abort_offset_ns']))):
                raise ValueError('async terminal accounting differs from delivered history')
            terminal[identifier] = event
        last_time[identifier] = event['emitted_ns']
    if set(terminal) != set(expected):
        raise ValueError('missing async terminal event')
    for identifier, request in expected.items():
        arrival = arrivals[identifier]
        if arrival['scheduled_ns'] != request['arrival_offset_ns'] or arrival['actual_ns'] < arrival['scheduled_ns']:
            raise ValueError('async arrival differs from frozen trace')
        owned = [h for h in heads if h['request_id'] == identifier]
        if (len(owned) > request['max_new_tokens']
                or [h['generated_tokens'] for h in owned] != list(range(1, len(owned) + 1))):
            raise ValueError('async selected-head ordinals are missing or unordered')
    for result in results:
        identifier = result['request_id']
        if identifier not in expected:
            raise ValueError('async result names an unknown request')
        if result['disposition'] == 'delivered':
            if sum(e['kind'] == 'token' and e['ticket'] == result['ticket'] and e['request_id'] == identifier for e in events) != 1:
                raise ValueError('completed async delivery missing its token event')
        elif (execution != 'async' or result['disposition'] != 'discarded-' + terminal[identifier]['reason']
                or result['generated_tokens'] <= len(histories[identifier])
                or result['observed_ns'] < terminal[identifier]['emitted_ns']):
            raise ValueError('async discarded result lacks its earlier terminal boundary')
    drained_fields = metadata.get('async_drained', [])
    if len(drained_fields) != 11:
        raise ValueError('missing async drain receipt')
    names = ('requests', 'submitted', 'completed', 'delivered', 'discarded', 'free_blocks',
             'owned_blocks', 'written_tokens', 'live_requests', 'pending', 'elapsed_ns')
    drained = dict(zip(names, map(int, drained_fields)))
    delivered = sum(len(h) for h in histories.values())
    discarded = sum(r['disposition'] != 'delivered' for r in results)
    if (any(v < 0 for v in drained.values()) or drained['requests'] != len(expected)
            or drained['submitted'] != len(submissions) or drained['completed'] != len(completions)
            or drained['delivered'] != delivered or drained['discarded'] != discarded
            or drained['free_blocks'] != blocks or any(drained[k] for k in ('owned_blocks', 'written_tokens', 'live_requests', 'pending'))
            or drained['elapsed_ns'] < max([c['completed_ns'] for c in completions]
                + [r['observed_ns'] for r in results] + [e['emitted_ns'] for e in events] + [0])):
        raise ValueError('async trace did not drain or its counter census differs')
    necessary = sum(len(expected[i]['prompt_ids']) + len(h) - 1 if h else 0 for i, h in histories.items())
    rows = sum(s['total_tokens'] for s in submissions)
    if rows < necessary or (execution == 'sync' and rows != necessary):
        raise ValueError('async submitted work omitted rows or synchronous control replayed')
    return dict(device='Apple M4 Pro/metal', mode='greedy', execution=execution, blocks=blocks,
        token_budget=256, max_sequences=8, admission='reserved', study=contract.ENGINE_ASYNC_STUDY,
        arrivals={str(k): v for k, v in arrivals.items()}, submissions=submissions, completions=completions,
        heads=heads, results=results, events=events, drained=drained, max_pending=max_pending,
        necessary_rows=necessary, submitted_rows=rows, extra_submitted_rows=rows-necessary)


def run_summary(run, trace):
    # Reuse the existing request metrics after this separate ownership parser.
    metric_input = dict(run, steps=[dict(s, preempted=0) for s in run['submissions']])
    result = profile.engine_run_summary(metric_input, trace)
    result.update(submitted=run['drained']['submitted'], completed=run['drained']['completed'],
        discarded_tokens=run['drained']['discarded'], max_pending=run['max_pending'],
        necessary_rows=run['necessary_rows'], extra_submitted_rows=run['extra_submitted_rows'],
        submit_host_ns=sum(s['submitted_ns']-s['begin_ns'] for s in run['submissions']),
        collect_host_ns=sum(c['completed_ns']-c['begin_ns'] for c in run['completions']),
        completion_timestamp='host-observed readback/fence; not GPU execution duration')
    return result


def cells(block):
    values = [('sync', False), ('sync', True), ('async', False)]
    return values[::-1] if block in (1, 2) else values


def _histories(parsed, trace):
    result = run_summary(parsed, trace)
    return [dict(request_id=r['request_id'], token_ids=r['token_ids'], reason=r['reason']) for r in result['requests']]


def _command(record, run):
    return [record['build']['command'][-1], record['build']['assets']['prepared'], record['native_trace_path'],
        'chunked', str(record['blocks']), '256', '8', '10', 'greedy', 'reserved',
        contract.ENGINE_ASYNC_STUDY, run['execution']]


def _execution(record, run, parsed, receipt):
    if (not _same(receipt.get('command'), _command(record, run))
            or type(receipt.get('exit_code')) is not int or receipt['exit_code'] != 0
            or type(receipt.get('timeout_seconds')) is not int or receipt['timeout_seconds'] != 180
            or type(receipt.get('wall_elapsed_ns')) is not int
            or not 0 < parsed['drained']['elapsed_ns'] <= receipt['wall_elapsed_ns'] <= 180*10**9):
        raise ValueError('async numeric execution receipt changed or exceeded its bound')


def _qualification(record):
    """Replay the separate checkpoint gate without accessing current GPU/assets."""
    qualification = record.get('qualification', {})
    document = record.get('qualification_document', '')
    if (type(document) is not str or hashlib.sha256(document.encode()).hexdigest() != record.get('qualification_sha256')
            or not _same(json.loads(document), qualification)
            or qualification.get('kind') != 'engine-async-qualification-v1'
            or type(qualification.get('schema_version')) is not int or qualification['schema_version'] != 1
            or not _same(qualification.get('build'), record['build'])):
        raise ValueError('async qualification differs from its exact build')
    checkpoint = qualification.get('checkpoint_receipt', {})
    build = checkpoint.get('build', {})
    command = build.get('command', [])
    binary = build.get('binaries', {}).get('checkpoint-driver', {})
    if (not _same(build.get('source'), record['build']['source'])
            or not _same(build.get('assets'), record['build']['assets'])
            or not _same(build.get('environment'), record['build']['environment'])
            or set(build.get('binaries', {})) != {'checkpoint-driver'}
            or not _digest(binary.get('sha256')) or type(binary.get('bytes')) is not int or binary['bytes'] < 1
            or not isinstance(command, list) or len(command) != 9
            or command[:-1] != [record['build']['command'][0], 'build', '-I', 'src', '-I', 'tests',
                                 'tests/engine_async_metal_driver.mojo', '-o']
            or type(command[-1]) is not str or not Path(command[-1]).is_absolute()
            or Path(command[-1]).name != 'checkpoint-driver'
            or not _digest(build.get('source', {}).get('sources', {}).get('tests/engine_async_metal_driver.mojo'))):
        raise ValueError('async checkpoint driver is not the same clean source/assets/environment')
    compiled = checkpoint.get('build_execution', {})
    if (not _same(compiled.get('command'), command)
            or type(compiled.get('exit_code')) is not int or compiled['exit_code'] != 0
            or type(compiled.get('timeout_seconds')) is not int or compiled['timeout_seconds'] != 600
            or type(compiled.get('wall_elapsed_ns')) is not int
            or not 0 < compiled['wall_elapsed_ns'] <= 600*10**9):
        raise ValueError('async checkpoint qualification lacks an actual successful compilation')
    stdout, execution = checkpoint.get('stdout'), checkpoint.get('execution', {})
    if (type(stdout) is not str or hashlib.sha256(stdout.encode()).hexdigest() != checkpoint.get('stdout_sha256')
            or not _same(execution.get('command'), [command[-1], build['assets']['prepared'], 'async-qualification'])
            or type(execution.get('exit_code')) is not int or execution['exit_code'] != 0
            or type(execution.get('timeout_seconds')) is not int or not 1 <= execution['timeout_seconds'] <= 600
            or type(execution.get('wall_elapsed_ns')) is not int
            or not 0 < execution['wall_elapsed_ns'] <= execution['timeout_seconds']*10**9):
        raise ValueError('async checkpoint qualification lacks an actual successful execution')
    metadata, checks = {}, []
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] in ('device', 'qualification') and fields[0] not in metadata:
            metadata[fields[0]] = fields[1:]
        elif len(fields) == 8 and fields[0] == 'check' and fields[2::2] == ['checked', 'mismatches', 'unexpected-nonfinite']:
            checks.append(dict(name=fields[1], checked=int(fields[3]), mismatches=int(fields[5]), nonfinite=int(fields[7])))
        else:
            raise ValueError('unknown async checkpoint qualification line')
    if (metadata != {'device': ['Apple', 'M4', 'Pro/metal'], 'qualification': ['engine-async-checkpoint-v1']}
            or [c['name'] for c in checks] != list(contract.ENGINE_ASYNC_CHECKPOINT_CHECKS)
            or any(c['checked'] < 1 or c['mismatches'] != 0 or c['nonfinite'] != 0 for c in checks)):
        raise ValueError('async checkpoint qualification failed or omitted required checked outputs')
    return dict(numerical_passed=True, checks=checks)


def _checked_execution(command, log, timeout):
    """Retain failures and actual numeric exits as well as successful output."""
    started = time.monotonic_ns()
    receipt = dict(command=list(map(str, command)), timeout_seconds=timeout, exit_code=None)
    try:
        stdout = profile.execute(command, log, timeout=timeout)
        receipt['exit_code'] = 0
        return stdout, receipt
    except Exception as error:
        receipt.update(exit_code=getattr(error, 'returncode', None), error=str(error))
        raise
    finally:
        receipt['wall_elapsed_ns'] = time.monotonic_ns()-started
        write(log.with_suffix('.execution.json'), receipt)


def _live_async_build(directory):
    """Bind the command we execute to the binary verify_build actually hashed."""
    directory = Path(directory).resolve()
    build = profile.verify_build(directory)
    command = build.get('command', [])
    expected = [str(profile.environment_tool('mojo')), 'build', '-I', 'src',
                'src/llm_mojo/benchmarks/engine_trace.mojo', '-o']
    if (not _same(build.get('declaration'), contract.ENGINE_ASYNC_DECLARATION)
            or set(build.get('binaries', {})) != {'engine'}
            or not isinstance(command, list) or len(command) != 7
            or not _same(command[:-1], expected)
            or type(command[-1]) is not str or not Path(command[-1]).is_absolute()
            or Path(command[-1]).resolve() != (directory/'engine').resolve()):
        raise ValueError('async engine command differs from its verified live build binary')
    return build


def qualify(directory, output):
    """Compile and execute the exact-source checkpoint gate before collection."""
    build = _live_async_build(directory)
    profile.ensure_record_location(output)
    output.mkdir(parents=True, exist_ok=False)
    binary = output/'checkpoint-driver'
    command = [profile.environment_tool('mojo'), 'build', '-I', 'src', '-I', 'tests',
               'tests/engine_async_metal_driver.mojo', '-o', binary]
    _, compiled = _checked_execution(command, output/'checkpoint-build.log', 600)
    if (profile.verify_build(directory) != build
            or profile.stable_environment() != build['environment']):
        raise ValueError('async source/assets/environment changed during qualification build')
    checkpoint_build = dict(source=build['source'], assets=build['assets'], environment=build['environment'],
        command=list(map(str, command)), binaries={'checkpoint-driver': dict(sha256=sha(binary), bytes=binary.stat().st_size)})
    stdout, execution = _checked_execution([binary, build['assets']['prepared'], 'async-qualification'],
                                           output/'checkpoint-native.log', 180)
    if (profile.verify_build(directory) != build or sha(binary) != checkpoint_build['binaries']['checkpoint-driver']['sha256']
            or profile.stable_environment() != build['environment']):
        raise ValueError('async source/assets/environment/binary changed during qualification execution')
    qualification = dict(kind='engine-async-qualification-v1', schema_version=1, build=build,
        checkpoint_receipt=dict(build=checkpoint_build, build_execution=compiled, stdout=stdout,
            stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(), execution=execution))
    document = json.dumps(qualification, indent=2, allow_nan=False)+'\n'
    accepted = _qualification(dict(build=build, qualification=qualification, qualification_document=document,
                                   qualification_sha256=hashlib.sha256(document.encode()).hexdigest()))
    (output/'qualification.json').write_text(document)
    write(output/'qualification-summary.json', accepted)
    return accepted


def summary(record):
    profile._engine_declaration(record)
    profile.validate_engine_record_build(record)
    trace = profile.validate_engine_trace(record['trace'])
    if (record.get('mode') != 'greedy' or record.get('token_budget') != 256
            or type(record.get('token_budget')) is not int or record.get('max_sequences') != 8
            or type(record.get('max_sequences')) is not int or record.get('warmup_steps') != 10
            or type(record.get('warmup_steps')) is not int or record.get('admission') != 'reserved'
            or type(record.get('blocks')) is not int or not 1 <= record['blocks'] <= 8192
            or type(record.get('native_trace_path')) is not str or not Path(record['native_trace_path']).is_absolute()
            or Path(record['native_trace_path']).name != 'trace.tsv'
            or record.get('native_trace_sha256') != hashlib.sha256(profile.engine_trace_tsv(trace).encode()).hexdigest()
            or any(r['abort_offset_ns'] is not None for r in trace['requests'])):
        raise ValueError('async collection changed its declared bounded configuration')
    qualification = _qualification(record)
    runs = record.get('runs', [])
    expected = [(b, execution, calibration) for b in range(4) for execution, calibration in cells(b)]
    if [(r.get('block'), r.get('execution'), r.get('calibration')) for r in runs] != expected:
        raise ValueError('incomplete, reordered or duplicate async paired grid')
    results = []
    histories = None
    for run in runs:
        if type(run['block']) is not int or type(run['calibration']) is not bool:
            raise ValueError('async block/calibration field types changed')
        for name in ('conditions_before', 'conditions_after'):
            snapshot = run.get(name, {})
            try:
                profile.require_ac(snapshot)
                profile.require_nominal_thermal_state(snapshot)
                if snapshot['power_mode_raw'] != '0':
                    raise ValueError('non-nominal power mode')
            except (KeyError, RuntimeError) as error:
                raise ValueError('missing or non-nominal async conditions') from error
        parsed = parse_async_run(run['stdout'], trace, run['execution'], record['blocks'])
        if not _same(parsed, run.get('parsed')):
            raise ValueError('async parsed records differ from retained native stdout')
        _execution(record, run, parsed, run.get('native_execution', {}))
        if any(e['reason'] not in ('stop', 'length') for e in parsed['events'] if e['kind'] == 'finish'):
            raise ValueError('async measured trace lacks natural finite completions')
        actual = _histories(parsed, trace)
        if histories is None:
            histories = actual
        if not _same(actual, histories):
            raise ValueError('async delivered greedy histories or terminal reasons differ')
        results.append(dict(block=run['block'], execution=run['execution'], calibration=run['calibration'],
                            **run_summary(parsed, trace)))
    keyed = {(r['block'], r['execution'], r['calibration']): r for r in results}
    calibrations = [keyed[b, 'sync', True]['duration_ns']/keyed[b, 'sync', False]['duration_ns'] for b in range(4)]
    noise = max(.05, max(abs(r-1) for r in calibrations))
    ratios = [keyed[b, 'async', False]['duration_ns']/keyed[b, 'sync', False]['duration_ns'] for b in range(4)]
    overlapped = all(keyed[b, 'async', False]['max_pending'] == 2 for b in range(4))
    return dict(runs=results, qualification=qualification, same_delivered_histories=True,
        paired_duration_ratios=ratios, median_duration_ratio=profile.stats.median(ratios),
        calibration_ratios=calibrations, noise_floor=noise, observed_two_pending=overlapped,
        performance_eligible=overlapped, speed_verdict=(
            'distribution-only' if trace['mode'] == 'online' else profile._outcome(ratios, noise)) if overlapped else None,
        asynchronous=True, promotion=False, goodput=None, target=None, quantile_method='linear')


def collect(build, trace_path, output, qualification_path, blocks=128, maximum_sequences=8, warmup_steps=10):
    if type(blocks) is not int or not 1 <= blocks <= 8192 or type(maximum_sequences) is not int or maximum_sequences != 8 or type(warmup_steps) is not int or warmup_steps != 10:
        raise ValueError('invalid bounded async collection')
    receipt = _live_async_build(build)
    document = trace_path.read_text()
    trace = profile.validate_engine_trace(json.loads(document))
    if any(r['abort_offset_ns'] is not None or (max(len(r['prompt_ids']), len(r['prompt_ids'])+r['max_new_tokens']-1)+31)//32 > blocks for r in trace['requests']):
        raise ValueError('async measured trace has timed aborts or cannot fit each request alone')
    qualification_document = qualification_path.read_text()
    record = dict(kind=contract.ENGINE_ASYNC_DECLARATION['kind'], declaration=contract.ENGINE_ASYNC_DECLARATION,
        build=receipt, trace=trace, trace_document=document, trace_sha256=hashlib.sha256(document.encode()).hexdigest(),
        blocks=blocks, max_sequences=8, token_budget=256, admission='reserved', mode='greedy', warmup_steps=10,
        native_trace_path=str(output/'trace.tsv'),
        native_trace_sha256=hashlib.sha256(profile.engine_trace_tsv(trace).encode()).hexdigest(),
        qualification=json.loads(qualification_document), qualification_document=qualification_document,
        qualification_sha256=hashlib.sha256(qualification_document.encode()).hexdigest(), runs=[])
    _qualification(record)
    numerical = record['qualification']['checkpoint_receipt']['build']
    binary_path = Path(numerical['command'][-1])
    identity = numerical['binaries']['checkpoint-driver']
    if sha(binary_path) != identity['sha256'] or binary_path.stat().st_size != identity['bytes']:
        raise ValueError('async checkpoint qualification binary changed')
    ensure_record_location(output)
    output.mkdir(parents=True, exist_ok=False)
    native = output/'trace.tsv'
    native.write_text(profile.engine_trace_tsv(trace))
    (output/'qualification.json').write_text(qualification_document)
    for block in range(4):
        for execution, calibration in cells(block):
            run = dict(block=block, execution=execution, calibration=calibration)
            if sha(native) != record['native_trace_sha256']:
                raise ValueError('async native input changed before collection')
            before = profile.conditions()
            command = _command(record, run)
            log = output/f'block-{block}-{execution}{"-calibration" if calibration else ""}.log'
            started = profile.time.monotonic_ns()
            native_execution = dict(command=command, timeout_seconds=180)
            try:
                stdout = profile.execute(command, log, timeout=180)
            except Exception as error:
                native_execution.update(wall_elapsed_ns=profile.time.monotonic_ns()-started,
                    exit_code=getattr(error, 'returncode', None), error=str(error))
                write(log.with_suffix('.execution.json'), native_execution)
                raise
            native_execution.update(wall_elapsed_ns=profile.time.monotonic_ns()-started, exit_code=0)
            write(log.with_suffix('.execution.json'), native_execution)
            if sha(native) != record['native_trace_sha256']:
                raise ValueError('async native input changed during collection')
            run.update(stdout=stdout, native_execution=native_execution, conditions_before=before,
                conditions_after=profile.conditions(), parsed=parse_async_run(stdout, trace, execution, blocks))
            record['runs'].append(run)
    if (profile.verify_build(build) != receipt or sha(trace_path) != record['trace_sha256']
            or sha(qualification_path) != record['qualification_sha256']
            or sha(binary_path) != identity['sha256'] or binary_path.stat().st_size != identity['bytes']):
        raise ValueError('async source/assets/qualification changed during collection')
    record['summary'] = summary(record)
    raw = json.dumps(record, separators=(',', ':')).encode()
    compressed = gzip.compress(raw, mtime=0)
    (output/'engine-async.json.gz').write_bytes(compressed)
    write(output/'engine-async.json', dict(kind=record['kind'], bytes=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(), uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
    return profile.engine_replay(output)
