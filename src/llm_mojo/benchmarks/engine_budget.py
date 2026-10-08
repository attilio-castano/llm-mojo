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
    if policy.get('kind') != 'engine-step-cost-v1' or policy.get('schema_version') != 1:
        raise ValueError('invalid step-cost declaration')
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
    manifest = json.loads((output/'engine-adaptive.json').read_text())
    compressed = (output/'engine-adaptive.json.gz').read_bytes()
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
    write(output/'engine-adaptive-summary.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['fit','evaluate','replay'])
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--build',type=Path)
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--trace',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-ms',type=float,default=25)
    parser.add_argument('--blocks',type=int,default=128)
    parser.add_argument('--max-sequences',type=int,default=8)
    args = parser.parse_args()
    if args.command=='fit':
        if args.calibration is None or not math.isfinite(args.target_ms): parser.error('fit requires calibration and finite target')
        result=fit(args.calibration,args.output,round(args.target_ms*1e6))
        print(json.dumps({'cost':result['cost'],'calibration_error':result['calibration_error']},indent=2))
    elif args.command=='evaluate':
        if None in (args.build,args.policy,args.trace): parser.error('evaluate requires build, policy and trace')
        print(json.dumps(evaluate(args.build,args.policy,args.trace,args.output,args.blocks,args.max_sequences),indent=2))
    else:
        print(json.dumps(replay(args.output),indent=2))


if __name__=='__main__':
    main()
