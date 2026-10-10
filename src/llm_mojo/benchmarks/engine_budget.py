"""Fit a small step-cost model, freeze it, and evaluate a separate arrival trace.

The target bounds predicted synchronous step cost. It is a research setting,
not a promised token-latency SLO. Mandatory decodes and one-token progress may
exceed it. All calibration/evaluation observations remain in replayable files.
"""
import argparse
import base64
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from . import model_profile as profile
from .environment import ensure_record_location
from ..validation.evidence import sha, write

COEFFICIENTS = ('fixed_ns', 'per_row_ns', 'per_position_ns', 'per_partition_ns', 'per_logit_ns')


def features(step, events):
    singleton = step['decode_seqs'] > 0 or step['prefill_tokens'] == 1
    partitions = int(singleton) + int(step['prefill_tokens'] > 1)
    logits = sum(e['kind'] == 'token' and e['step_id'] == step['step_id'] for e in events)
    return [1, step['total_tokens'], step['attended_positions'], partitions, logits]


def nonnegative_fit(x, y):
    """Five-dimensional NNLS by enumerating active sets; no extra optimizer dependency."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5 or len(x) < 5 or y.shape != (len(x),):
        raise ValueError('insufficient cost calibration observations')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any() or (y <= 0).any():
        raise ValueError('invalid cost calibration observation')
    scale = np.maximum(np.max(x, axis=0), 1)
    normalized = x / scale
    best, loss = np.zeros(5), float(y @ y)
    for count in range(1, 6):
        for active in itertools.combinations(range(5), count):
            solution = np.linalg.lstsq(normalized[:, active], y, rcond=None)[0]
            if (solution < -1e-6).any():
                continue
            candidate = np.zeros(5)
            candidate[list(active)] = np.maximum(solution, 0)
            error = y - normalized @ candidate
            candidate_loss = float(error @ error)
            if candidate_loss < loss:
                best, loss = candidate, candidate_loss
    # Ceiling gives the native integer arithmetic a conservative rounding rule.
    return np.ceil(best / scale).astype(np.int64).tolist()


def validate_policy(policy):
    if (policy.get('kind'), policy.get('schema_version')) not in (('engine-step-cost-v1', 1), ('engine-step-cost-v2', 2)):
        raise ValueError('invalid step-cost declaration')
    if policy.get('kind') == 'engine-step-cost-v2' and type(policy.get('schema_version')) is not int:
        raise ValueError('invalid step-cost version type')
    cost = policy.get('cost', {})
    if set(cost) != {*COEFFICIENTS, 'target_ns'}:
        raise ValueError('incomplete step-cost coefficients')
    if any(type(v) is not int or v < 0 or v > 1_000_000_000_000 for v in cost.values()) or cost['target_ns'] < 1:
        raise ValueError('invalid bounded step-cost coefficient or target')
    if any(not isinstance(policy.get(k),str) or len(policy[k])!=64
           or any(c not in '0123456789abcdef' for c in policy[k])
           for k in ('calibration_sha256','calibration_trace_sha256','calibration_workload_sha256')):
        raise ValueError('missing calibration identity')
    return cost


def workload_identity(trace):
    # Greedy execution ignores the simulation's global script and all JSON
    # metadata. Compare the request/abort records actually consumed by Metal.
    native='\n'.join(line for line in profile.engine_trace_tsv(trace).splitlines()
                     if not line.startswith('script '))+'\n'
    return hashlib.sha256(native.encode()).hexdigest()


def calibration_samples(record):
    profile.engine_study_summary(record)
    if record['mode']!='greedy':
        raise ValueError('virtual-clock observations cannot fit Metal cost')
    samples=[]
    for run in record['runs']:
        for step in run['parsed']['steps']:
            if step['total_tokens']>0:
                samples.append(dict(block=run['block'],arm=run['arm'],calibration=run['calibration'],
                                    step_id=step['step_id'],features=features(step,run['parsed']['events']),
                                    execute_ns=step['execute_ns']))
    return samples


def fitted_values(samples):
    coefficients=nonnegative_fit([s['features'] for s in samples],[s['execute_ns'] for s in samples])
    errors=[sum(a*b for a,b in zip(coefficients,s['features']))-s['execute_ns'] for s in samples]
    return coefficients,dict(mae_ns=float(np.mean(np.abs(errors))),
                             p95_abs_ns=float(np.percentile(np.abs(errors),95)),
                             max_abs_ns=max(abs(e) for e in errors))


def check_calibration(policy,compressed):
    if hashlib.sha256(compressed).hexdigest()!=policy['calibration_sha256']:
        raise ValueError('frozen calibration archive identity differs')
    record=json.loads(gzip.decompress(compressed))
    profile.validate_engine_record_build(record)
    build=record.get('build',{})
    if (build.get('source',{}).get('repository',{}).get('dirty') is not False
            or not build.get('source',{}).get('sources')
            or not build.get('binaries',{}).get('engine',{}).get('sha256')
            or hashlib.sha256(record['trace_document'].encode()).hexdigest()!=record['trace_sha256']
            or json.loads(record['trace_document'])!=record['trace']):
        raise ValueError('incomplete calibration source or trace provenance')
    samples=calibration_samples(record)
    coefficients,error=fitted_values(samples)
    if (record['build']!=policy['build'] or record['trace_sha256']!=policy['calibration_trace_sha256']
            or workload_identity(record['trace'])!=policy['calibration_workload_sha256']
            or samples!=policy['calibration_samples'] or error!=policy['calibration_error']
            or coefficients!=[policy['cost'][k] for k in COEFFICIENTS]):
        raise ValueError('cost model does not reproduce its retained calibration')
    return record


def fit(calibration, output, target_ns):
    if type(target_ns) is not int or not 1 <= target_ns <= 1_000_000_000:
        raise ValueError('invalid frozen research step target')
    profile.engine_replay(calibration)
    record = json.loads(gzip.decompress((calibration/'engine-core.json.gz').read_bytes()))
    samples=calibration_samples(record)
    coefficients,error=fitted_values(samples)
    policy = dict(kind='engine-step-cost-v1', schema_version=1,
                  cost=dict(zip(COEFFICIENTS,coefficients), target_ns=target_ns),
                  calibration_sha256=sha(calibration/'engine-core.json.gz'),
                  calibration_trace_sha256=record['trace_sha256'], build=record['build'],
                  calibration_workload_sha256=workload_identity(record['trace']),
                  calibration_archive=str((calibration/'engine-core.json.gz').resolve()),
                  fitting='nonnegative least squares; integer-nanosecond coefficients rounded upward',
                  target_scope='predicted synchronous execute cost; mandatory decodes/progress may exceed',
                  calibration_samples=samples,
                  calibration_error=error)
    validate_policy(policy)
    ensure_record_location(output)
    if output.exists():
        raise ValueError('refusing to replace a frozen cost policy')
    output.parent.mkdir(parents=True,exist_ok=True)
    write(output,policy)
    return policy


def summary(record):
    if record.get('kind')!='engine-adaptive-evaluation-v1':
        raise ValueError('invalid adaptive evaluation declaration')
    cost = validate_policy(record['policy'])
    if record['policy'].get('kind') != 'engine-step-cost-v1':
        raise ValueError('legacy adaptive evaluation requires its v1 policy')
    trace = profile.validate_engine_trace(record['trace'])
    if workload_identity(trace)==record['policy']['calibration_workload_sha256']:
        raise ValueError('evaluation must use a separate frozen trace')
    check_calibration(record['policy'],base64.b64decode(record['calibration_archive_base64'],validate=True))
    expected = {(b,a,c) for b in range(4) for a,c in [('chunked',False),('chunked',True),('adaptive',False)]}
    runs = record['runs']
    keys = [(r['block'],r['arm'],r['calibration']) for r in runs]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('incomplete adaptive evaluation grid')
    results = []
    for run in runs:
        for key in ('conditions_before','conditions_after'):
            snapshot=run[key]
            profile.require_ac(snapshot)
            profile.require_nominal_thermal_state(snapshot)
            if snapshot['power_mode_raw']!='0':
                raise ValueError('adaptive evaluation requires normal power mode')
        parsed = profile.parse_engine_run(run['stdout'],trace,run['arm'],record['blocks'],
                                          record['max_sequences'],'greedy',
                                          policy=cost if run['arm']=='adaptive' else None)
        if parsed != run['parsed']:
            raise ValueError('adaptive parsed records differ from native trace')
        steps = [s for s in parsed['steps'] if s['total_tokens']]
        predictions = [sum(a*b for a,b in zip([cost[k] for k in COEFFICIENTS],features(s,parsed['events']))) for s in steps]
        if run['arm']=='adaptive' and predictions != [s['predicted_ns'] for s in steps]:
            raise ValueError('native step prediction differs from declared cost model')
        errors = [p-s['execute_ns'] for p,s in zip(predictions,steps)]
        results.append(dict(block=run['block'],arm=run['arm'],calibration=run['calibration'],
                            **profile.engine_run_summary(parsed,trace),
                            limited_steps=sum(s['budget_limited'] for s in steps),
                            predicted_over_target=sum(p>cost['target_ns'] for p in predictions),
                            measured_over_target=sum(s['execute_ns']>cost['target_ns'] for s in steps),
                            measured_steps=len(steps),
                            prediction_mae_ns=float(np.mean(np.abs(errors))) if errors else None,
                            prediction_p95_abs_ns=float(np.percentile(np.abs(errors),95)) if errors else None))
    keyed = {(r['block'],r['arm'],r['calibration']):r for r in results}
    comparisons = []
    for b in range(4):
        base,candidate = keyed[b,'chunked',False],keyed[b,'adaptive',False]
        comparisons.append(dict(block=b,duration_ratio=candidate['duration_ns']/base['duration_ns'],
                                same_generated_histories=all(a['token_ids']==c['token_ids'] for a,c in zip(base['requests'],candidate['requests'])),
                                calibration_ratio=keyed[b,'chunked',True]['duration_ns']/base['duration_ns']))
    return dict(target_ns=cost['target_ns'],target_scope=record['policy']['target_scope'],
                runs=results,comparisons=comparisons,goodput=None,asynchronous=False)


def evaluate(build, policy_path, trace_path, output, blocks=128, maximum_sequences=8):
    receipt = profile.verify_build(build)
    policy_document=policy_path.read_text()
    policy = json.loads(policy_document)
    cost = validate_policy(policy)
    if policy.get('kind') != 'engine-step-cost-v1':
        raise ValueError('legacy adaptive evaluation requires its v1 policy')
    document = trace_path.read_text()
    trace = profile.validate_engine_trace(json.loads(document))
    trace_hash = hashlib.sha256(document.encode()).hexdigest()
    if receipt != policy['build'] or workload_identity(trace)==policy['calibration_workload_sha256']:
        raise ValueError('evaluation needs the calibrated build and a separate trace')
    calibration_archive=Path(policy['calibration_archive']).read_bytes()
    check_calibration(policy,calibration_archive)
    if not 1 <= blocks <= 8192 or not 1 <= maximum_sequences <= 64:
        raise ValueError('invalid adaptive evaluation bounds')
    ensure_record_location(output)
    output.mkdir(parents=True,exist_ok=False)
    native_trace = output/'trace.tsv'
    native_trace.write_text(profile.engine_trace_tsv(trace))
    native_policy = output/'policy.tsv'
    native_policy.write_text('cost '+' '.join(str(cost[k]) for k in (*COEFFICIENTS,'target_ns'))+'\n')
    runs = []
    for block in range(4):
        order = [('chunked',False),('chunked',True),('adaptive',False)]
        if block in (1,2): order.reverse()
        for arm,calibration in order:
            before = profile.conditions()
            command = [build/'engine',receipt['assets']['prepared'],native_trace,arm,blocks,256,maximum_sequences,10,'greedy']
            if arm=='adaptive': command.append(native_policy)
            log = output/f'block-{block}-{arm}{"-calibration" if calibration else ""}.log'
            stdout = profile.execute(command,log)
            parsed = profile.parse_engine_run(stdout,trace,arm,blocks,maximum_sequences,'greedy',
                                              policy=cost if arm=='adaptive' else None)
            runs.append(dict(block=block,arm=arm,calibration=calibration,stdout=stdout,parsed=parsed,
                             conditions_before=before,conditions_after=profile.conditions()))
    if profile.verify_build(build)!=receipt or sha(trace_path)!=trace_hash or json.loads(policy_path.read_text())!=policy:
        raise ValueError('adaptive source, trace or policy changed during evaluation')
    record = dict(kind='engine-adaptive-evaluation-v1',build=receipt,policy=policy,policy_document=policy_document,
                  policy_sha256=sha(policy_path),calibration_archive_base64=base64.b64encode(calibration_archive).decode(),
                  trace=trace,trace_document=document,trace_sha256=trace_hash,blocks=blocks,
                  max_sequences=maximum_sequences,runs=runs)
    record['summary'] = summary(record)
    raw = json.dumps(record,separators=(',',':')).encode()
    compressed = gzip.compress(raw,mtime=0)
    (output/'engine-adaptive.json.gz').write_bytes(compressed)
    write(output/'engine-adaptive.json',dict(kind=record['kind'],bytes=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
    return replay(output)


def replay(output):
    archive = output if output.is_file() else output/'engine-adaptive.json.gz'
    manifest_path = archive.with_suffix('')
    summary_path = archive.with_name(archive.name.removesuffix('.json.gz')+'-summary.json')
    manifest = json.loads(manifest_path.read_text())
    compressed = archive.read_bytes()
    raw = gzip.decompress(compressed)
    if (manifest['kind']!='engine-adaptive-evaluation-v1' or len(compressed)!=manifest['bytes']
            or hashlib.sha256(compressed).hexdigest()!=manifest['sha256']
            or hashlib.sha256(raw).hexdigest()!=manifest['uncompressed_sha256']):
        raise ValueError('adaptive archive hash mismatch')
    record = json.loads(raw)
    if (hashlib.sha256(record['trace_document'].encode()).hexdigest()!=record['trace_sha256']
            or json.loads(record['trace_document'])!=record['trace']
            or record['build']!=record['policy']['build']
            or hashlib.sha256(record['policy_document'].encode()).hexdigest()!=record['policy_sha256']
            or json.loads(record['policy_document'])!=record['policy']):
        raise ValueError('adaptive trace or calibrated build identity differs')
    result = summary(record)
    if result!=record['summary']:
        raise ValueError('adaptive archived summary differs')
    write(summary_path,result)
    return result



def scheduling_cells(stage, block):
    if stage not in ('calibration', 'evaluation'):
        raise ValueError('invalid scheduling study stage')
    cells = [('fixed-256', 'chunked', 256, False), ('fixed-256', 'chunked', 256, True),
             *[(f'fixed-{n}', 'chunked', n, False) for n in (32, 64, 128)]]
    if stage == 'evaluation':
        cells.append(('adaptive', 'adaptive', 256, False))
    return list(reversed(cells)) if block in (1, 2) else cells


def _budget_policy_cost(policy):
    cost = validate_policy(policy)
    if (policy.get('kind') != 'engine-step-cost-v2'
            or type(policy.get('schema_version')) is not int or policy['schema_version'] != 2
            or policy.get('admission') not in profile.contract.ENGINE_ADMISSION_POLICIES
            or cost['target_ns'] != profile.contract.ENGINE_BUDGET_DECLARATION['research_target_ns']):
        raise ValueError('scheduling study needs its declared same-build fitted policy')
    return cost


def budget_calibration_samples(record):
    if record.get('stage') != 'calibration' or record.get('mode') != 'greedy':
        raise ValueError('cost fit needs fresh greedy fixed-budget calibration')
    scheduling_summary(record)
    return [dict(block=r['block'], budget_arm=r['budget_arm'], arm=r['arm'],
                 token_budget=r['token_budget'], admission=record['admission'],
                 calibration=r['calibration'], step_id=s['step_id'],
                 features=features(s, r['parsed']['events']), execute_ns=s['execute_ns'])
            for r in record['runs'] for s in r['parsed']['steps'] if s['total_tokens']]


def check_budget_calibration(policy, compressed):
    _budget_policy_cost(policy)
    if hashlib.sha256(compressed).hexdigest() != policy['calibration_sha256']:
        raise ValueError('budget calibration archive identity differs')
    record = json.loads(gzip.decompress(compressed))
    profile.validate_engine_record_build(record)
    document = record.get('trace_document', '')
    if (record.get('kind') != profile.contract.ENGINE_BUDGET_DECLARATION['kind']
            or hashlib.sha256(document.encode()).hexdigest() != record.get('trace_sha256')
            or json.loads(document) != record.get('trace')):
        raise ValueError('budget calibration trace provenance differs')
    samples = budget_calibration_samples(record)
    coefficients, error = fitted_values(samples)
    if (not profile._engine_same_json(record.get('summary'), scheduling_summary(record))
            or not profile._engine_same_json(record['build'], policy['build'])
            or record['admission'] != policy['admission']
            or record['trace_sha256'] != policy['calibration_trace_sha256']
            or workload_identity(record['trace']) != policy['calibration_workload_sha256']
            or not profile._engine_same_json(samples, policy['calibration_samples'])
            or not profile._engine_same_json(error, policy['calibration_error'])
            or coefficients != [policy['cost'][k] for k in COEFFICIENTS]):
        raise ValueError('budget policy does not reproduce its retained calibration')
    return record


def scheduling_fit(calibration, output, target_ns=25_000_000):
    if type(target_ns) is not int or target_ns != profile.contract.ENGINE_BUDGET_DECLARATION['research_target_ns']:
        raise ValueError('scheduling target must be frozen before fitting')
    calibration = Path(calibration)
    archive = calibration if calibration.is_file() else calibration/'engine-budget.json.gz'
    profile.engine_replay(archive)
    compressed = archive.read_bytes()
    record = json.loads(gzip.decompress(compressed))
    samples = budget_calibration_samples(record)
    coefficients, error = fitted_values(samples)
    policy = dict(kind='engine-step-cost-v2', schema_version=2,
                  cost=dict(zip(COEFFICIENTS, coefficients), target_ns=target_ns),
                  admission=record['admission'], build=record['build'],
                  calibration_sha256=hashlib.sha256(compressed).hexdigest(),
                  calibration_trace_sha256=record['trace_sha256'],
                  calibration_workload_sha256=workload_identity(record['trace']),
                  calibration_archive=str(archive.resolve()), calibration_samples=samples,
                  calibration_error=error,
                  fitting='nonnegative least squares; integer-nanosecond coefficients rounded upward',
                  target_scope='predicted synchronous execute cost; mandatory decodes/progress may exceed')
    check_budget_calibration(policy, compressed)
    ensure_record_location(output)
    if output.exists():
        raise ValueError('refusing to replace a frozen cost policy')
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, policy)
    return policy


def _budget_command(record, run):
    build = record['build']
    command = build.get('command', [])
    trace_path = record.get('native_trace_path')
    if (not isinstance(command, list) or len(command) < 2
            or any(type(v) is not str for v in command) or command[-2] != '-o'
            or not Path(command[-1]).is_absolute() or Path(command[-1]).name != 'engine'
            or type(trace_path) is not str or not Path(trace_path).is_absolute()
            or Path(trace_path).name != 'trace.tsv' or type(build['assets'].get('prepared')) is not str):
        raise ValueError('missing scheduling native command provenance')
    result = [command[-1], build['assets']['prepared'], trace_path, run['arm'], str(record['blocks']),
              str(run['token_budget']), '8', '10', record['mode']]
    if run['arm'] == 'adaptive':
        policy_path = record.get('native_policy_path')
        if type(policy_path) is not str or not Path(policy_path).is_absolute() or Path(policy_path).name != 'policy.tsv':
            raise ValueError('missing scheduling native policy provenance')
        result.append(policy_path)
    return result + [record['admission'], profile.contract.ENGINE_BUDGET_STUDY]


def scheduling_summary(record):
    profile._engine_declaration(record)
    if record.get('kind') != profile.contract.ENGINE_BUDGET_DECLARATION['kind']:
        raise ValueError('invalid scheduling study declaration')
    stage = record.get('stage')
    mode = record.get('mode')
    admission = record.get('admission')
    trace = profile.validate_engine_trace(record['trace'])
    if (stage not in ('calibration', 'evaluation') or mode not in ('greedy', 'scripted')
            or admission not in profile.contract.ENGINE_ADMISSION_POLICIES
            or type(record.get('blocks')) is not int or not 1 <= record['blocks'] <= 8192
            or type(record.get('max_sequences')) is not int or record['max_sequences'] != 8
            or type(record.get('warmup_steps')) is not int or record['warmup_steps'] != 10
            or (stage == 'evaluation' and mode != 'greedy')
            or any(r['abort_offset_ns'] is not None for r in trace['requests'])):
        raise ValueError('scheduling collection configuration changed')
    if hashlib.sha256(profile.engine_trace_tsv(trace).encode()).hexdigest() != record.get('native_trace_sha256'):
        raise ValueError('scheduling native trace identity changed')
    cost = None
    if stage == 'calibration':
        if any(record.get(k) is not None for k in ('policy', 'policy_document', 'policy_sha256', 'calibration_archive_base64', 'native_policy_path', 'native_policy_sha256')):
            raise ValueError('fixed-budget calibration cannot consume an evaluated policy')
    else:
        policy = record.get('policy', {})
        cost = _budget_policy_cost(policy)
        document = record.get('policy_document', '')
        if (hashlib.sha256(document.encode()).hexdigest() != record.get('policy_sha256')
                or not profile._engine_same_json(json.loads(document), policy)
                or not profile._engine_same_json(policy['build'], record['build'])
                or policy['admission'] != admission
                or workload_identity(trace) == policy['calibration_workload_sha256']):
            raise ValueError('evaluation needs the frozen same-build policy and separate native workload')
        native_policy_document = 'cost '+' '.join(str(cost[k]) for k in (*COEFFICIENTS,'target_ns'))+'\n'
        if hashlib.sha256(native_policy_document.encode()).hexdigest() != record.get('native_policy_sha256'):
            raise ValueError('scheduling native policy identity changed')
        check_budget_calibration(policy, base64.b64decode(record['calibration_archive_base64'], validate=True))
    expected = [list((b, *cell)) for b in range(4) for cell in scheduling_cells(stage, b)]
    runs = record.get('runs', [])
    actual = [[r['block'], r['budget_arm'], r['arm'], r['token_budget'], r['calibration']] for r in runs]
    if not profile._engine_same_json(actual, expected):
        raise ValueError('scheduling budget grid or balanced order changed')
    results = []
    for run in runs:
        for key in ('conditions_before', 'conditions_after'):
            snapshot = run.get(key, {})
            try:
                profile.require_ac(snapshot)
                profile.require_nominal_thermal_state(snapshot)
                if snapshot['power_mode_raw'] != '0':
                    raise ValueError('non-nominal power mode')
            except (KeyError, RuntimeError) as error:
                raise ValueError('incomplete or non-nominal scheduling conditions') from error
        parsed = profile.parse_engine_run(run['stdout'], trace, run['arm'], record['blocks'], 8, mode,
                                          policy=cost if run['arm']=='adaptive' else None,
                                          admission=admission, study=profile.contract.ENGINE_BUDGET_STUDY,
                                          token_budget=run['token_budget'])
        if not profile._engine_same_json(parsed, run.get('parsed')):
            raise ValueError('scheduling parsed records differ from native trace')
        execution = run.get('execution', {})
        if (not profile._engine_same_json(execution.get('command'), _budget_command(record, run))
                or type(execution.get('timeout_seconds')) is not int or execution['timeout_seconds'] != 180
                or type(execution.get('exit_code')) is not int or execution['exit_code'] != 0
                or type(execution.get('wall_elapsed_ns')) is not int
                or not 0 < execution['wall_elapsed_ns'] <= 180*10**9
                or (mode == 'greedy' and execution['wall_elapsed_ns'] < parsed['drained']['elapsed_ns'])):
            raise ValueError('scheduling native execution receipt changed or exceeded its bound')
        terminals = [e for e in parsed['events'] if e['kind']=='finish']
        if any(e['reason'] not in ('stop', 'length') for e in terminals):
            raise ValueError('scheduling study needs completed finite requests')
        necessary = sum(e['prompt_tokens']+e['generated_tokens']-1 if e['generated_tokens'] else 0 for e in terminals)
        result = dict(block=run['block'], budget_arm=run['budget_arm'], arm=run['arm'],
                      token_budget=run['token_budget'], calibration=run['calibration'], admission=admission,
                      **profile.engine_run_summary(parsed, trace))
        if result['duration_ns'] <= 0 or result['total_tokens'] < necessary:
            raise ValueError('scheduling trace lacks positive makespan or necessary rows')
        if admission=='reserved' and (result['preemptions'] or result['total_tokens'] != necessary):
            raise ValueError('reserved scheduling study recomputed or omitted rows')
        if cost is not None:
            steps = [s for s in parsed['steps'] if s['total_tokens']]
            predictions = [sum(cost[k]*v for k,v in zip(COEFFICIENTS, features(s,parsed['events']))) for s in steps]
            errors = [p-s['execute_ns'] for p,s in zip(predictions,steps)]
            result.update(limited_steps=sum(s['budget_limited'] for s in steps),
                          predicted_over_target=sum(p>cost['target_ns'] for p in predictions),
                          measured_over_target=sum(s['execute_ns']>cost['target_ns'] for s in steps),
                          measured_steps=len(steps),
                          prediction_mae_ns=float(np.mean(np.abs(errors))) if errors else None,
                          prediction_p95_abs_ns=float(np.percentile(np.abs(errors),95)) if errors else None,
                          prediction_max_abs_ns=max(map(abs,errors)) if errors else None,
                          prefill_rows=[s['prefill_tokens'] for s in steps if s['prefill_tokens']])
        results.append(result)
    reference = [r['token_ids'] for r in results[0]['requests']]
    if mode == 'greedy' and any([r['token_ids'] for r in r['requests']] != reference for r in results):
        raise ValueError('scheduling greedy histories differ across budgets or repeats')
    keyed = {(r['block'], r['budget_arm'], r['calibration']): r for r in results}
    calibrations = [keyed[b,'fixed-256',True]['duration_ns']/keyed[b,'fixed-256',False]['duration_ns'] for b in range(4)]
    noise = max(.05, max(abs(r-1) for r in calibrations))
    comparisons = []
    for name in ['fixed-32', 'fixed-64', 'fixed-128'] + (['adaptive'] if stage=='evaluation' else []):
        ratios = [keyed[b,name,False]['duration_ns']/keyed[b,'fixed-256',False]['duration_ns'] for b in range(4)]
        comparisons.append(dict(budget_arm=name, control='fixed-256', block_ratios=ratios,
                                median_ratio=profile.stats.median(ratios), same_generated_histories=all(
                                    [r['token_ids'] for r in keyed[b,name,False]['requests']]==
                                    [r['token_ids'] for r in keyed[b,'fixed-256',False]['requests']] for b in range(4)),
                                outcome='virtual-clock' if mode=='scripted' else
                                    'distribution-only' if trace['mode']=='online' else profile._outcome(ratios,noise)))
    return dict(stage=stage, admission=admission, runs=results, comparisons=comparisons,
                calibration_ratios=calibrations, noise_floor=noise, goodput=None,
                target_ns=cost['target_ns'] if cost is not None else None,
                target_scope=record['policy']['target_scope'] if cost is not None else None,
                quantile_method='linear', asynchronous=False)


def scheduling_collect(build, trace_path, output, stage, admission, policy_path=None,
                       blocks=128, maximum_sequences=8, mode='greedy', warmup_steps=10):
    declaration = profile.contract.ENGINE_BUDGET_DECLARATION
    if (stage not in ('calibration', 'evaluation') or admission not in profile.contract.ENGINE_ADMISSION_POLICIES
            or type(blocks) is not int or not 1<=blocks<=8192
            or type(maximum_sequences) is not int or maximum_sequences!=8
            or type(warmup_steps) is not int or warmup_steps!=10 or mode not in ('greedy','scripted')
            or (stage=='evaluation' and mode!='greedy')
            or (stage=='calibration' and policy_path is not None)
            or (stage=='evaluation' and policy_path is None)):
        raise ValueError('invalid bounded scheduling collection')
    receipt = profile.verify_engine_build(build, declaration)
    document = trace_path.read_text()
    trace = profile.validate_engine_trace(json.loads(document))
    if any(r['abort_offset_ns'] is not None or (profile.engine_peak_cached_tokens(r)+31)//32>blocks for r in trace['requests']):
        raise ValueError('scheduling trace has aborts or requests that cannot fit alone')
    policy = policy_document = compressed_calibration = None
    if stage=='evaluation':
        policy_document = policy_path.read_text()
        policy = json.loads(policy_document)
        _budget_policy_cost(policy)
        compressed_calibration = Path(policy['calibration_archive']).read_bytes()
        check_budget_calibration(policy, compressed_calibration)
        if (not profile._engine_same_json(policy['build'],receipt) or policy['admission']!=admission
                or workload_identity(trace)==policy['calibration_workload_sha256']):
            raise ValueError('evaluation requires calibrated build/admission and held-out native workload')
    ensure_record_location(output)
    output.mkdir(parents=True, exist_ok=False)
    native_trace = output/'trace.tsv'
    native_trace.write_text(profile.engine_trace_tsv(trace))
    native_policy = output/'policy.tsv' if policy is not None else None
    if native_policy is not None:
        native_policy.write_text('cost '+' '.join(str(policy['cost'][k]) for k in (*COEFFICIENTS,'target_ns'))+'\n')
    record = dict(kind=declaration['kind'], declaration=declaration, stage=stage, admission=admission,
                  build=receipt, trace=trace, trace_document=document,
                  trace_sha256=hashlib.sha256(document.encode()).hexdigest(), blocks=blocks,
                  max_sequences=8, mode=mode, warmup_steps=10, native_trace_path=str(native_trace),
                  native_trace_sha256=sha(native_trace), runs=[])
    if policy is not None:
        record.update(policy=policy, policy_document=policy_document,
                      policy_sha256=hashlib.sha256(policy_document.encode()).hexdigest(),
                      calibration_archive_base64=base64.b64encode(compressed_calibration).decode(),
                      native_policy_path=str(native_policy), native_policy_sha256=sha(native_policy))
    for block in range(4):
        for name,arm,budget,calibration in scheduling_cells(stage,block):
            run = dict(block=block,budget_arm=name,arm=arm,token_budget=budget,calibration=calibration)
            before = profile.conditions()
            command = _budget_command(record,run)
            if (sha(native_trace) != record['native_trace_sha256']
                    or (native_policy is not None and sha(native_policy) != record['native_policy_sha256'])):
                raise ValueError('scheduling generated input changed before a cell')
            log = output/f'block-{block}-{name}{"-calibration" if calibration else ""}.log'
            stdout, execution = profile.checked_execution(command, log, 180)
            if (sha(native_trace) != record['native_trace_sha256']
                    or (native_policy is not None and sha(native_policy) != record['native_policy_sha256'])):
                raise ValueError('scheduling generated input changed during a cell')
            run.update(stdout=stdout,execution=execution,conditions_before=before,conditions_after=profile.conditions(),
                       parsed=profile.parse_engine_run(stdout,trace,arm,blocks,8,mode,
                            policy=policy['cost'] if arm=='adaptive' else None,admission=admission,
                            study=profile.contract.ENGINE_BUDGET_STUDY,token_budget=budget))
            record['runs'].append(run)
    if (profile.verify_build(build)!=receipt or sha(trace_path)!=record['trace_sha256']
            or (policy_path is not None and sha(policy_path)!=record['policy_sha256'])):
        raise ValueError('scheduling source, trace or policy changed during collection')
    record['summary'] = scheduling_summary(record)
    raw = json.dumps(record,separators=(',',':')).encode()
    compressed = gzip.compress(raw,mtime=0)
    (output/'engine-budget.json.gz').write_bytes(compressed)
    write(output/'engine-budget.json',dict(kind=declaration['kind'],bytes=len(compressed),
          sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
    return profile.engine_replay(output)


def fast_cells(block):
    cells = [('reference', False), ('reference', True), ('fast-decode', False)]
    return list(reversed(cells)) if block in (1, 2) else cells


def _fast_histories(parsed, trace):
    return [dict(request_id=request['request_id'],
                 token_ids=[e['token_id'] for e in parsed['events']
                            if e['kind']=='token' and e['request_id']==request['request_id']],
                 reason=next(e['reason'] for e in parsed['events']
                             if e['kind']=='finish' and e['request_id']==request['request_id']))
            for request in trace['requests']]


def _fast_step_work(parsed):
    """Canonical submitted work; exclude route configuration and measured time.

    Aggregate row/attention totals can hide different batch partitions. Keep
    empty steps too, and bind selected heads to their ordered request support.
    """
    result=[]
    for step in parsed['steps']:
        selected=[event['request_id'] for event in parsed['events']
                  if event['kind']=='token' and event['step_id']==step['step_id']]
        result.append(dict((key,step[key]) for key in
                           ('total_tokens','decode_seqs','prefill_seqs',
                            'prefill_tokens','attended_positions')))
        result[-1].update(selected_logits=len(selected),selected_requests=selected)
    return result


def _fast_qualification(record):
    """Reparse bound numerical-driver checks and untimed own-route engine traces."""
    qualification=record.get('qualification',{})
    document=record.get('qualification_document','')
    if (type(document) is not str or hashlib.sha256(document.encode()).hexdigest()!=record.get('qualification_sha256')
            or not profile._engine_same_json(json.loads(document),qualification)
            or qualification.get('kind')!='engine-fast-qualification-v1'
            or type(qualification.get('schema_version')) is not int or qualification['schema_version']!=1
            or not profile._engine_same_json(qualification.get('build'),record['build'])
            or qualification.get('workload_sha256')!=workload_identity(record['trace'])
            or qualification.get('native_trace_sha256')!=record['native_trace_sha256']
            or qualification.get('admission')!=record['admission']
            or not profile._engine_same_json(qualification.get('blocks'),record['blocks'])
            or not profile._engine_same_json(qualification.get('token_budget'),record['token_budget'])
            or not profile._engine_same_json(qualification.get('work_capacity'),dict(token_rows=256,max_sequences=8))):
        raise ValueError('Fast qualification identity differs from its exact build and collection')
    routes=qualification.get('routes',[])
    if not isinstance(routes,list) or [r.get('runner') for r in routes]!=list(profile.contract.ENGINE_FAST_ROUTES):
        raise ValueError('Fast qualification requires both own-route executions')
    def digest(value):
        return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    qualified={}
    for route in routes:
        runner=route['runner'];receipt=route.get('checkpoint_receipt',{});build=receipt.get('build',{})
        command=build.get('command',[]);binary=build.get('binaries',{}).get('checkpoint-driver',{})
        source=build.get('source',{})
        if (not profile._engine_same_json(source,record['build']['source'])
                or not profile._engine_same_json(build.get('assets'),record['build']['assets'])
                or not profile._engine_same_json(build.get('environment'),record['build']['environment'])
                or not digest(source.get('sources',{}).get('tests/engine_metal_driver.mojo'))
                or set(build.get('binaries',{}))!={'checkpoint-driver'}
                or not digest(binary.get('sha256')) or type(binary.get('bytes')) is not int or binary['bytes']<1
                or not isinstance(command,list) or len(command)!=9 or type(command[-1]) is not str
                or not Path(command[-1]).is_absolute() or Path(command[-1]).name!='checkpoint-driver'
                or command[:-1]!=[record['build']['command'][0],'build','-I','src','-I','tests',
                                  'tests/engine_metal_driver.mojo','-o']):
            raise ValueError('Fast numerical driver is not bound to the same clean source and checkpoint assets')
        log=receipt.get('stdout');execution=receipt.get('execution',{})
        if (type(log) is not str or hashlib.sha256(log.encode()).hexdigest()!=receipt.get('stdout_sha256')
                or not profile._engine_same_json(execution.get('command'),[command[-1],build['assets']['prepared'],'fast-qualification',runner])
                or type(execution.get('exit_code')) is not int or execution['exit_code']!=0
                or type(execution.get('timeout_seconds')) is not int or execution['timeout_seconds']<1
                or type(execution.get('wall_elapsed_ns')) is not int
                or not 0<execution['wall_elapsed_ns']<=execution['timeout_seconds']*10**9):
            raise ValueError('Fast numerical qualification receipt is incomplete or reports a false success')
        metadata={};checks=[]
        for line in log.splitlines():
            fields=line.split()
            if not fields:continue
            if fields[0] in ('device','qualification','runner') and fields[0] not in metadata:
                metadata[fields[0]]=fields[1:]
            elif (fields[0]=='check' and len(fields)==8
                  and fields[2::2]==['checked','mismatches','unexpected-nonfinite']):
                checks.append(dict(name=fields[1],checked=int(fields[3]),mismatches=int(fields[5]),nonfinite=int(fields[7])))
            else:raise ValueError('unknown or duplicate Fast numerical qualification record')
        if (metadata!={'device':['Apple','M4','Pro/metal'],'qualification':['engine-fast-checkpoint-v1'],'runner':[runner]}
                or [c['name'] for c in checks]!=list(profile.contract.ENGINE_FAST_CHECKPOINT_CHECKS)
                or any(c['checked']<1 or c['mismatches']<0 or c['nonfinite']<0
                       or c['mismatches']>c['checked'] or c['nonfinite']>c['checked'] for c in checks)):
            raise ValueError('Fast numerical qualification lacks complete device and checked-output evidence')
        natural=route.get('natural_run',{})
        path=natural.get('native_trace_path')
        if (type(path) is not str or not Path(path).is_absolute() or Path(path).name!='trace.tsv'
                or natural.get('native_trace_sha256')!=record['native_trace_sha256']):
            raise ValueError('Fast natural qualification input differs from the frozen native workload')
        parsed=profile.parse_engine_run(natural.get('stdout',''),record['trace'],'chunked',record['blocks'],8,'greedy',
                    admission=record['admission'],study=profile.contract.ENGINE_FAST_STUDY,
                    token_budget=record['token_budget'],runner=runner)
        if (not profile._engine_same_json(parsed,natural.get('parsed'))
                or any(e['reason'] not in ('stop','length') for e in parsed['events'] if e['kind']=='finish')):
            raise ValueError('Fast natural qualification parsed history or finite completion changed')
        _validate_fast_execution(dict(record,native_trace_path=path),dict(runner=runner),parsed,natural.get('execution',{}))
        qualified[runner]=dict(numerical_passed=all(not c['mismatches'] and not c['nonfinite'] for c in checks),
                               checks=checks,histories=_fast_histories(parsed,record['trace']))
    return qualified


def _verify_fast_qualification_binaries(qualification):
    for route in qualification['routes']:
        build=route['checkpoint_receipt']['build']
        path=Path(build['command'][-1]);binary=build['binaries']['checkpoint-driver']
        if sha(path)!=binary['sha256'] or path.stat().st_size!=binary['bytes']:
            raise ValueError('Fast checkpoint qualification binary changed')


def _fast_command(record, run):
    command = _budget_command(dict(record,mode='greedy'),dict(run,arm='chunked',token_budget=record['token_budget']))
    return command[:-1]+[profile.contract.ENGINE_FAST_STUDY,run['runner']]


def _validate_fast_execution(record,run,parsed,execution):
    if (not profile._engine_same_json(execution.get('command'),_fast_command(record,run))
            or type(execution.get('timeout_seconds')) is not int or execution['timeout_seconds']!=180
            or type(execution.get('exit_code')) is not int or execution['exit_code']!=0
            or type(execution.get('wall_elapsed_ns')) is not int
            or not parsed['drained']['elapsed_ns']<=execution['wall_elapsed_ns']<=180*10**9
            or execution['wall_elapsed_ns']<=0):
        raise ValueError('Fast native execution receipt changed or exceeded its bound')


def fast_summary(record):
    profile._engine_declaration(record)
    if record.get('kind')!=profile.contract.ENGINE_FAST_DECLARATION['kind']:
        raise ValueError('invalid Fast study declaration')
    trace=profile.validate_engine_trace(record['trace'])
    if (record.get('mode')!='greedy' or record.get('admission') not in profile.contract.ENGINE_ADMISSION_POLICIES
            or type(record.get('token_budget')) is not int or record['token_budget'] not in (32,64,128,256)
            or type(record.get('blocks')) is not int or not 1<=record['blocks']<=8192
            or type(record.get('max_sequences')) is not int or record['max_sequences']!=8
            or type(record.get('warmup_steps')) is not int or record['warmup_steps']!=10
            or any(r['abort_offset_ns'] is not None for r in trace['requests'])
            or any(record.get(k) is not None for k in ('policy','policy_document','policy_sha256','calibration_archive_base64'))):
        raise ValueError('Fast collection requires one fixed budget, admission and greedy finite trace')
    if hashlib.sha256(profile.engine_trace_tsv(trace).encode()).hexdigest()!=record.get('native_trace_sha256'):
        raise ValueError('Fast native trace identity changed')
    qualification=_fast_qualification(record)
    runs=record.get('runs', [])
    expected=[[block,*cell] for block in range(4) for cell in fast_cells(block)]
    if not profile._engine_same_json([[r['block'],r['runner'],r['calibration']] for r in runs],expected):
        raise ValueError('Fast paired grid or balanced block order changed')
    results=[]
    diagnostics=[]
    if any(not route['numerical_passed'] for route in qualification.values()):
        diagnostics.append(dict(kind='checkpoint-qualification-failed'))
    for run in runs:
        for key in ('conditions_before','conditions_after'):
            snapshot=run.get(key,{})
            try:
                profile.require_ac(snapshot)
                profile.require_nominal_thermal_state(snapshot)
                if snapshot['power_mode_raw']!='0': raise ValueError('non-nominal power mode')
            except (KeyError,RuntimeError) as error:
                raise ValueError('incomplete or non-nominal Fast conditions') from error
        parsed=profile.parse_engine_run(run['stdout'],trace,'chunked',record['blocks'],8,'greedy',
                    admission=record['admission'],study=profile.contract.ENGINE_FAST_STUDY,
                    token_budget=record['token_budget'],runner=run['runner'])
        if not profile._engine_same_json(parsed,run.get('parsed')):
            raise ValueError('Fast parsed records differ from actual completed trace')
        _validate_fast_execution(record,run,parsed,run.get('execution',{}))
        terminals=[e for e in parsed['events'] if e['kind']=='finish']
        if any(e['reason'] not in ('stop','length') for e in terminals):
            raise ValueError('Fast study needs completed finite requests')
        necessary=sum(e['prompt_tokens']+e['generated_tokens']-1 if e['generated_tokens'] else 0 for e in terminals)
        result=dict(block=run['block'],runner=run['runner'],calibration=run['calibration'],
                    token_budget=record['token_budget'],admission=record['admission'],
                    **profile.engine_run_summary(parsed,trace))
        if result['duration_ns']<=0 or result['total_tokens']<necessary:
            raise ValueError('Fast trace lacks positive makespan or necessary rows')
        if record['admission']=='reserved' and (result['preemptions'] or result['total_tokens']!=necessary):
            raise ValueError('reserved Fast study recomputed or omitted rows')
        result.update(route_steps={str(c):sum(r['configuration']==c for r in parsed['routes']) for c in (26,27)},
                      route_rows={str(c):sum(r['executed_rows'] for r in parsed['routes'] if r['configuration']==c) for c in (26,27)},
                      route_selected_logits={str(c):sum(r['selected_logits'] for r in parsed['routes'] if r['configuration']==c) for c in (26,27)},
                      zero_head_steps=sum(r['selected_logits']==0 for r in parsed['routes']),
                      step_work=_fast_step_work(parsed))
        actual=_fast_histories(parsed,trace)
        result['matches_qualified_history']=profile._engine_same_json(actual,qualification[run['runner']]['histories'])
        if not result['matches_qualified_history']:
            diagnostics.append(dict(kind='own-route-qualified-history-difference',block=run['block'],
                                    runner=run['runner'],calibration=run['calibration'],expected=qualification[run['runner']]['histories'],actual=actual))
        results.append(result)
    keyed={(r['block'],r['runner'],r['calibration']):r for r in results}
    def histories(result):
        return [dict(request_id=r['request_id'],token_ids=r['token_ids'],reason=r['reason']) for r in result['requests']]
    for runner in profile.contract.ENGINE_FAST_ROUTES:
        first=histories(next(r for r in results if r['runner']==runner))
        for result in (r for r in results if r['runner']==runner):
            if not profile._engine_same_json(histories(result),first):
                diagnostics.append(dict(kind='own-route-repeat-history-difference',block=result['block'],
                                        runner=runner,calibration=result['calibration'],expected=first,actual=histories(result)))
    ratios=[]
    matched_history=[]
    matched_work=[]
    matched_step_work=[]
    for block in range(4):
        control=keyed[block,'reference',False]; candidate=keyed[block,'fast-decode',False]
        ratios.append(candidate['duration_ns']/control['duration_ns'])
        same_history=profile._engine_same_json(histories(control),histories(candidate))
        same_aggregate_work=all(candidate[k]==control[k] for k in ('delivered_tokens','total_tokens','attended_positions','preemptions'))
        same_step_work=profile._engine_same_json(candidate['step_work'],control['step_work'])
        same_work=same_aggregate_work and same_step_work
        self_reference=keyed[block,'reference',True]
        if any(self_reference[k]!=control[k] for k in ('delivered_tokens','total_tokens','attended_positions','preemptions')):
            diagnostics.append(dict(kind='self-reference-work-difference',block=block,
                                    reference={k:control[k] for k in ('delivered_tokens','total_tokens','attended_positions','preemptions')},
                                    self_reference={k:self_reference[k] for k in ('delivered_tokens','total_tokens','attended_positions','preemptions')}))
        if not profile._engine_same_json(self_reference['step_work'],control['step_work']):
            diagnostics.append(dict(kind='self-reference-step-work-difference',block=block,
                                    reference=control['step_work'],self_reference=self_reference['step_work']))
        matched_history.append(same_history);matched_work.append(same_work);matched_step_work.append(same_step_work)
        if not same_history:
            diagnostics.append(dict(kind='cross-route-history-difference',block=block,
                                    reference=histories(control),fast_decode=histories(candidate)))
        if not same_aggregate_work:
            diagnostics.append(dict(kind='cross-route-work-difference',block=block,
                                    reference={k:control[k] for k in ('delivered_tokens','total_tokens','attended_positions','preemptions')},
                                    fast_decode={k:candidate[k] for k in ('delivered_tokens','total_tokens','attended_positions','preemptions')}))
        if not same_step_work:
            diagnostics.append(dict(kind='cross-route-step-work-difference',block=block,
                                    reference=control['step_work'],fast_decode=candidate['step_work']))
    calibration_ratios=[keyed[b,'reference',True]['duration_ns']/keyed[b,'reference',False]['duration_ns'] for b in range(4)]
    noise=max(.05,max(abs(r-1) for r in calibration_ratios))
    if not any(r['runner']=='fast-decode' and r['route_steps']['26'] for r in results):
        diagnostics.append(dict(kind='configuration-26-not-exercised'))
    eligible=not diagnostics
    comparison=dict(runner='fast-decode',control='reference',raw_duration_ratios=ratios,
                    median_raw_duration_ratio=profile.stats.median(ratios),matched_histories=matched_history,
                    matched_work=matched_work,matched_step_work=matched_step_work,speed_verdict=(
                        'distribution-only' if trace['mode']=='online' else profile._outcome(ratios,noise)) if eligible else None)
    return dict(runs=results,comparison=comparison,diagnostics=diagnostics,performance_eligible=eligible,
                calibration_ratios=calibration_ratios,noise_floor=noise,promotion=False,goodput=None,
                quantile_method='linear',asynchronous=False)


def fast_collect(build,trace_path,output,qualification_path,token_budget,admission,
                 blocks=128,maximum_sequences=8,warmup_steps=10):
    declaration=profile.contract.ENGINE_FAST_DECLARATION
    if (type(token_budget) is not int or token_budget not in (32,64,128,256)
            or admission not in profile.contract.ENGINE_ADMISSION_POLICIES
            or type(blocks) is not int or not 1<=blocks<=8192
            or type(maximum_sequences) is not int or maximum_sequences!=8
            or type(warmup_steps) is not int or warmup_steps!=10):
        raise ValueError('invalid bounded Fast collection')
    receipt=profile.verify_engine_build(build, declaration)
    document=trace_path.read_text(); trace=profile.validate_engine_trace(json.loads(document))
    if any(r['abort_offset_ns'] is not None or (profile.engine_peak_cached_tokens(r)+31)//32>blocks for r in trace['requests']):
        raise ValueError('Fast trace has aborts or requests that cannot fit alone')
    qualification_document=qualification_path.read_text()
    qualification=json.loads(qualification_document)
    record=dict(kind=declaration['kind'],declaration=declaration,build=receipt,trace=trace,trace_document=document,
                trace_sha256=hashlib.sha256(document.encode()).hexdigest(),blocks=blocks,max_sequences=8,
                mode='greedy',warmup_steps=10,token_budget=token_budget,admission=admission,
                native_trace_path=str(output/'trace.tsv'),native_trace_sha256=hashlib.sha256(profile.engine_trace_tsv(trace).encode()).hexdigest(),
                qualification=qualification,qualification_document=qualification_document,
                qualification_sha256=hashlib.sha256(qualification_document.encode()).hexdigest(),runs=[])
    qualified=_fast_qualification(record)
    ensure_record_location(output);output.mkdir(parents=True,exist_ok=False)
    (output/'qualification.json').write_text(qualification_document)
    if any(not r['numerical_passed'] for r in qualified.values()):
        write(output/'engine-fast-failure.json',dict(kind='checkpoint-qualification-failed',promotion=False))
        raise ValueError('Fast checkpoint qualification failed; retained qualification diagnostics')
    _verify_fast_qualification_binaries(qualification)
    native_trace=output/'trace.tsv';native_trace.write_text(profile.engine_trace_tsv(trace))
    for block in range(4):
        for runner,calibration in fast_cells(block):
            run=dict(block=block,runner=runner,calibration=calibration)
            before=profile.conditions();command=_fast_command(record,run)
            if sha(native_trace)!=record['native_trace_sha256']:
                raise ValueError('Fast generated input changed before a cell')
            log=output/f'block-{block}-{runner}{"-calibration" if calibration else ""}.log'
            stdout, execution = profile.checked_execution(command, log, 180)
            if sha(native_trace)!=record['native_trace_sha256']:
                raise ValueError('Fast generated input changed during a cell')
            run.update(stdout=stdout,execution=execution,conditions_before=before,conditions_after=profile.conditions(),
                       parsed=profile.parse_engine_run(stdout,trace,'chunked',blocks,8,'greedy',admission=admission,
                            study=profile.contract.ENGINE_FAST_STUDY,token_budget=token_budget,runner=runner))
            record['runs'].append(run)
    _verify_fast_qualification_binaries(qualification)
    if (profile.verify_build(build)!=receipt or sha(trace_path)!=record['trace_sha256']
            or sha(qualification_path)!=record['qualification_sha256']):
        raise ValueError('Fast source, trace or qualification changed during collection')
    record['summary']=fast_summary(record)
    raw=json.dumps(record,separators=(',',':')).encode();compressed=gzip.compress(raw,mtime=0)
    (output/'engine-fast.json.gz').write_bytes(compressed)
    write(output/'engine-fast.json',dict(kind=declaration['kind'],bytes=len(compressed),
          sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
    return profile.engine_replay(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['fit','evaluate','replay','scheduling-fit'])
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--build',type=Path)
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--trace',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-ms',type=float,default=25)
    parser.add_argument('--blocks',type=int,default=128)
    parser.add_argument('--max-sequences',type=int,default=8)
    args = parser.parse_args()
    if args.command in ('fit', 'scheduling-fit'):
        if args.calibration is None or not math.isfinite(args.target_ms): parser.error('fit requires calibration and finite target')
        result=(scheduling_fit if args.command=='scheduling-fit' else fit)(args.calibration,args.output,round(args.target_ms*1e6))
        print(json.dumps({'cost':result['cost'],'calibration_error':result['calibration_error']},indent=2))
    elif args.command=='evaluate':
        if None in (args.build,args.policy,args.trace): parser.error('evaluate requires build, policy and trace')
        print(json.dumps(evaluate(args.build,args.policy,args.trace,args.output,args.blocks,args.max_sequences),indent=2))
    else:
        print(json.dumps(replay(args.output),indent=2))


if __name__=='__main__':
    main()
