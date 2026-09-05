"""One bounded protocol and evidence format for the maintained kernel studies."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Stable IDs are local to each operation; they are returned by launch branches.
STUDIES = {
    'rms_norm': dict(operation='rms_norm', control=0, candidates=[0, 1], rows=[1, 16, 128, 512, 4096],
                     names={0: 'shared tree', 1: 'SIMD reduction'},
                     layout='X/O[M,896], weight[896]; contiguous row major'),
    'linear_decode': dict(operation='linear', control=0, candidates=[0, 1, 2], rows=[1],
                         names={0: 'separate QKV', 1: 'packed QKV', 2: 'packed two-output'},
                         layout='X[1,896], W[1152,896], bias[1152], O[1,1152]; contiguous row major'),
    'linear_prefill': dict(operation='linear', control=1, candidates=[1, 3, 4, 5, 6], rows=[1, 8, 16, 64, 256],
                          names={1: 'rowwise', 3: 'direct 8x16', 4: 'shared BK16', 5: 'register 2x2', 6: 'Apple MMA'},
                          layout='X[M,896], W[1152,896], bias[1152], O[M,1152]; contiguous row major'),
    'rope': dict(operation='rope', control=0, candidates=[0], rows=[1, 16, 256, 4096],
                 names={0: 'pair ownership'},
                 layout='X/O[M,14,64], cosine/sine[M,64], start=0; contiguous row major'),
    'gqa_decode': dict(operation='gqa_decode', control=0, candidates=[0, 1, 4, 9], rows=[1, 16, 64, 256, 1024, 4096],
                       names={0: 'materialized', 1: 'fused G1', 4: 'fused G32', 9: 'split64 H4'},
                       layout='Q/O[1,14,64], K/V[T,2,64]; contiguous row major'),
}
BLOCKS, REPETITIONS, WARMUP = 4, 10, 10
FIELDS = ['block', 'rows', 'layers', 'candidate', 'arm', 'variant', 'repetition', 'us']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def encode_samples(samples):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(samples)
    return gzip.compress(stream.getvalue().encode(), mtime=0)


def read_samples(path):
    with gzip.open(path, 'rt', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError('unexpected sample columns')
        return [{k: (v if k == 'arm' else float(v) if k == 'us' else int(v))
                 for k, v in row.items()} for row in reader]


def parse_output(output, control, candidate, first, *, rows, layers, seed, operation):
    lines = output.splitlines()
    expected_headers = [f'shape: {rows} {layers} seed: {seed}',
                        f'variants: {control} {candidate} candidate-first: {int(first)}',
                        'api: metal', 'correctness: passed', 'BENCHMARK_COMPLETE']
    if operation != 'gqa_decode':
        expected_headers.append(f'operation: {operation}')
    if any(lines.count(h) != 1 for h in expected_headers) or not output.rstrip().endswith('BENCHMARK_COMPLETE'):
        raise ValueError('missing or mismatched runtime contract/completion')
    devices = [line.removeprefix('device: ') for line in lines if line.startswith('device: ')]
    if len(devices) != 1 or not devices[0].startswith('Apple '):
        raise ValueError('runtime did not prove an Apple GPU')
    samples = []
    for line in lines:
        if line.startswith('SAMPLE '):
            _, arm, variant, rep, value = line.split()
            us = float(value)
            if not math.isfinite(us) or us <= 0:
                raise ValueError('timings must be finite and positive')
            samples.append(dict(arm=arm, variant=int(variant), repetition=int(rep), us=us))
    arms = [('control', control), ('candidate', candidate)]
    if first:
        arms.reverse()
    expected = [(arm, variant, r) for arm, variant in arms for r in range(REPETITIONS)]
    if [(s['arm'], s['variant'], s['repetition']) for s in samples] != expected:
        raise ValueError('wrong implementation, sample count or execution order')
    return dict(device=devices[0], api='metal', correctness='passed'), samples


def summarize(samples, spec):
    """Reject partial grids; derive noise and every conclusion from raw observations."""
    grouped = defaultdict(list)
    observed = set()
    for s in samples:
        key = tuple(s[k] for k in ('block', 'rows', 'layers', 'candidate', 'arm', 'repetition'))
        if key in observed or not math.isfinite(s['us']) or s['us'] <= 0:
            raise ValueError('duplicate or invalid observation')
        observed.add(key)
        expected_variant = spec['control'] if s['arm'] == 'control' else s['candidate']
        if s['variant'] != expected_variant:
            raise ValueError('sample implementation differs from requested arm')
        grouped[(s['rows'], s['layers'], s['candidate'], s['block'], s['arm'])].append(s['us'])
    expected = {(b, r, l, c, a, n) for b in range(1, BLOCKS + 1)
                for r in spec['rows'] for l in (1, 24) for c in spec['candidates']
                for a in ('control', 'candidate') for n in range(REPETITIONS)}
    if observed != expected:
        raise ValueError('incomplete or unexpected study grid, including self-pair calibration')
    result = []
    for rows in spec['rows']:
        for layers in (1, 24):
            def medians(candidate, arm):
                return [statistics.median(grouped[(rows, layers, candidate, b, arm)]) for b in range(1, BLOCKS + 1)]
            noise_ratios = [a / b for a, b in zip(medians(spec['control'], 'candidate'), medians(spec['control'], 'control'))]
            floor = max(0.05, max(abs(1 - r) for r in noise_ratios))
            for candidate in spec['candidates']:
                a, b = medians(candidate, 'candidate'), medians(candidate, 'control')
                ratios = [x / y for x, y in zip(a, b)]
                ratio = statistics.median(ratios)
                decision = 'calibration' if candidate == spec['control'] else (
                    'faster' if ratio < 1 - floor and all(r < 1 for r in ratios) else
                    'slower' if ratio > 1 + floor and all(r > 1 for r in ratios) else 'inconclusive')
                result.append(dict(rows=rows, layers=layers, candidate=candidate,
                                   control_us=statistics.median(b), candidate_us=statistics.median(a),
                                   ratio=ratio, ratio_min=min(ratios), ratio_max=max(ratios),
                                   noise_floor=floor, decision=decision))
    return result


def load_run(directory):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_text())
    if record.get('schema') != 1 or not record.get('completed_utc'):
        raise ValueError('incomplete or unsupported run')
    if record['samples_sha256'] != sha(directory / 'samples.csv.gz'):
        raise ValueError('raw sample hash mismatch')
    if record['repository']['dirty'] or record['runtime']['api'] != 'metal' or not record['runtime']['device'].startswith('Apple '):
        raise ValueError('unverified source or GPU execution')
    if record['repository'] != record['build']['repository']:
        raise ValueError('run/build source identity mismatch')
    if (record.get('blocks'), record.get('repetitions'), record.get('warmup')) != (BLOCKS, REPETITIONS, WARMUP):
        raise ValueError('unsupported timing/calibration protocol')
    if [c['block'] for c in record['conditions']] != list(range(1, BLOCKS + 1)):
        raise ValueError('missing block conditions')
    samples = read_samples(directory / 'samples.csv.gz')
    # Use the frozen run specification: later matrix changes cannot reinterpret evidence.
    summary = summarize(samples, record['specification'])
    return record, samples, summary


def load_profile(directory):
    directory = Path(directory)
    record = json.loads((directory / 'profiles.json').read_text())
    path = directory / 'profile_samples.csv.gz'
    if record.get('schema') != 1 or record['samples_sha256'] != sha(path):
        raise ValueError('profile sample hash/schema mismatch')
    stages = {0: ['QK', 'softmax', 'PV'], 4: ['fused'], 9: ['decode', 'merge']}
    if len(record['captures']) != 3 or {c['variant'] for c in record['captures']} != set(stages):
        raise ValueError('missing or duplicate profile capture')
    expected = set()
    for capture in record['captures']:
        variant = capture['variant']
        identity = capture['capture']
        if identity['repository'] != record['common']['repository'] or identity['repository']['dirty']:
            raise ValueError('profile source mismatch')
        if identity['runtime']['backend'] != 'metal' or not identity['runtime']['device'].startswith('Apple '):
            raise ValueError('profile runtime mismatch')
        expected.update((variant, iteration, stage)
                        for iteration in range(identity['workload']['profile_iterations'])
                        for stage in stages[variant])
    observed, grouped = set(), defaultdict(list)
    with gzip.open(path, 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (int(row['variant']), int(row['iteration']), row['stage'])
            duration = int(row['duration_ns'])
            if key in observed or duration <= 0:
                raise ValueError('invalid/duplicate profile dispatch')
            observed.add(key)
            grouped[(key[0], key[2])].append(duration / 1000)
    if observed != expected:
        raise ValueError('incomplete profile dispatch sequence')
    return [dict(variant=variant, stage=stage, count=len(values),
                 median_us=statistics.median(values), minimum_us=min(values), maximum_us=max(values))
            for (variant, stage), values in grouped.items()]
