"""Host-only BF16 fixture transport and independent numerical assertions."""
import ctypes
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'fixtures/mlp'))
from numerics import bf16_bits, from_bits, differences, round_bf16
from contract import BUDGETS

ROOT = Path(__file__).resolve().parents[1] / 'build/oracle_data/mlp'


def read(path):
    return np.load(data_root(path) / path, allow_pickle=False)


def data_root(path):
    return ROOT.parent / 'mlp_holdout' if str(path).startswith('holdout_') else ROOT


def holdout_catalog():
    import subprocess
    path = ROOT.parent / 'mlp_holdout/manifest.json'
    data = json.loads(path.read_text())
    expected = data.pop('manifest_payload_sha256')
    assert hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest() == expected
    assert data['status'] == 'complete'
    if os.environ.get('MLP_CANDIDATE_BINARY'):
        binary = Path(os.environ['MLP_CANDIDATE_BINARY'])
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == data['candidate']['binary_sha256']
        assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == data['candidate']['commit']
        assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    return data


def bits_at(address, count):
    return np.ctypeslib.as_array((ctypes.c_uint16 * count).from_address(address))


def load(address, path):
    bits = bf16_bits(read(path)).reshape(-1)
    bits_at(address, bits.size)[:] = bits


def check(address, path, boundary, count):
    expected = read(path).reshape(-1)[:count]
    actual = from_bits(bits_at(address, count).copy())
    result = differences(actual, expected, BUDGETS['operation'][boundary])
    record(dict(probe='silu_sweep', path=path, boundary=boundary, **result))
    print(path, boundary, result, flush=True)
    if result['failed']:
        ab, rb = bf16_bits(actual), bf16_bits(expected)
        idx = np.flatnonzero(ab != rb)[:20]
        print('first differing inputs/outputs', list(zip(idx.tolist(), actual[idx].tolist(), expected[idx].tolist())), flush=True)
        raise AssertionError('MLP operation fails frozen gate')


def check_product(address, count, value):
    a = read('activation/sweep_A.npy').reshape(-1)
    actual = from_bits(bits_at(address, count).copy())
    expected = round_bf16(a.astype(np.float64) * value)
    finite = np.isfinite(expected)
    result = differences(actual[finite], expected[finite], BUDGETS['operation']['S'])
    record(dict(probe='gating_sweep', multiplier=value, overflow=int((~finite).sum()), **result))
    print('gating', value, result, flush=True)
    if result['failed']:
        raise AssertionError('BF16 gating mismatch')

# Each process records all checks, including failures, under ignored build/.
import gzip
import hashlib
import os

RECORDS = []
FULL = {}

def frozen():
    path = Path(__file__).parent / 'fixtures/mlp'
    anchor = json.loads((path / 'checksums.json').read_text())
    payload = (path / 'development.json.gz').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == anchor['evidence_sha256']
    raw = gzip.decompress(payload)
    assert hashlib.sha256(raw).hexdigest() == anchor['uncompressed_sha256']
    result = json.loads(raw)
    for name, digest in result['source_sha256'].items():
        assert hashlib.sha256((ROOT.parents[2] / name).read_bytes()).hexdigest() == digest, name
    return result


def case_specifications():
    data = frozen()
    split = os.environ.get('MLP_SPLIT', 'development')
    if split not in ('development', 'checkpoint', 'holdout'):
        raise ValueError('unsupported MLP case split')
    if split == 'holdout':
        data = holdout_catalog()
        names = list(data['cases'])
    else:
        names = [k for k in data['cases'] if k.startswith('checkpoint_') == (split == 'checkpoint')]
    select = os.environ.get('MLP_CASE')
    if select:
        if select not in names:
            raise ValueError('case not in the requested split')
        names = [select]
    return [(k, data['cases'][k]['rows'], data['cases'][k]['hidden'],
             data['cases'][k]['intermediate']) for k in names]


def verify_case(case):
    FULL.clear()
    data = holdout_catalog() if case.startswith('holdout_') else frozen()
    for name, spec in data['arrays'].items():
        if name.startswith(case + '/'):
            assert hashlib.sha256((data_root(name) / name).read_bytes()).hexdigest() == spec['sha256'], name


def load_slice(address, path, start, count):
    bits = bf16_bits(read(path)).reshape(-1)[start:start+count]
    assert bits.size == count
    bits_at(address, count)[:] = bits


def poison(address, active, capacity):
    bits_at(address, capacity)[:] = 0x42f6  # 123, exactly BF16
    bits_at(address, active)[:] = 0x7fc0


def record(value):
    RECORDS.append(value)
    group = 'probes' if 'probe' in value else 'mlp'
    path = ROOT / ('metal_' + os.environ.get('MLP_SPLIT', 'development') + '_' + group + '_checks.json')
    selected = [r for r in RECORDS if ('probe' in r) == ('probe' in value)]
    path.write_text(json.dumps(selected, indent=2, allow_nan=False) + '\n')


def stage_check(address, case, stage, mode, start, rows, width, capacity):
    count = rows * width
    bits = bits_at(address, capacity).copy()
    if not np.all(bits[count:] == 0x42f6):
        raise AssertionError('MLP changed an inactive workspace region')
    actual = from_bits(bits[:count]).reshape(rows, width)
    expected = read(case + '/' + stage + '.npy')[start:start+rows]
    budget = BUDGETS['operation'][stage] if mode == 'local' else BUDGETS['composition'].get(stage)
    result = differences(actual, expected, budget)
    extra = {}
    if mode == 'full':
        FULL[stage] = actual.copy()
    elif mode == 'chunk':
        extra = differences(actual, FULL[stage][start:start+rows], BUDGETS['composition'].get(stage))
    value = dict(case=case, stage=stage, mode=mode, start=start, rows=rows, **result, full_chunked=extra)
    record(value)
    print('MLP', case, mode, stage, start, rows, 'scaled', result['max_scaled'],
          'failed', result.get('failed', 'diagnostic'), flush=True)
    if result.get('failed', 0) or extra.get('failed', 0):
        raise AssertionError('MLP fails frozen numerical gate; full result retained')


def unchanged(address, path, count):
    assert np.array_equal(bits_at(address, count), bf16_bits(read(path)).reshape(-1)), path


def multiplication_cases(address_a, address_b, count):
    rng = np.random.default_rng(1601)
    a = rng.integers(0, 65536, count, dtype=np.uint16)
    b = rng.integers(0, 65536, count, dtype=np.uint16)
    a[(a & 0x7f80) == 0x7f80] &= 0x807f
    b[(b & 0x7f80) == 0x7f80] &= 0x807f
    # Cover each subnormal paired with significands across the exponent range.
    for j in range(min(count, 65536)):
        a[j] = j % 128 | (j & 0x8000)
    bits_at(address_a, count)[:] = a
    bits_at(address_b, count)[:] = b
    FULL['product'] = round_bf16(from_bits(a).astype(np.float64) * from_bits(b).astype(np.float64))


def product_cases_check(address, count):
    actual = from_bits(bits_at(address, count).copy())
    expected = FULL.pop('product')
    finite = np.isfinite(expected)
    result = differences(actual[finite], expected[finite], BUDGETS['operation']['S'])
    record(dict(probe='arbitrary_finite_product', overflow=int((~finite).sum()), **result))
    assert not result['failed'], result
    assert np.array_equal(bf16_bits(actual[~finite]), bf16_bits(expected[~finite]))


def assert_uniform_bits(address, count, value):
    assert np.all(bits_at(address, count) == value)


def residual_cases(address_a, address_b, count):
    finite = np.arange(65536,dtype=np.uint16)
    finite = finite[(finite & 0x7f80) != 0x7f80]
    a = np.resize(finite, count)
    b = np.roll(a, 333)
    b[:65280] = 0
    b[65280:130560] = a[65280:130560] ^ 0x8000
    bits_at(address_a,count)[:] = a
    bits_at(address_b,count)[:] = b
    FULL['residual'] = round_bf16(from_bits(a).astype(np.float64) + from_bits(b).astype(np.float64))


def residual_subnormal_cases(address_a, address_b):
    finite = np.arange(65536,dtype=np.uint16)
    finite = finite[(finite & 0x7f80) != 0x7f80]
    tiny = np.arange(128,dtype=np.uint16)
    tiny = np.concatenate((tiny,tiny | 0x8000))
    a, b = np.repeat(finite,len(tiny)), np.tile(tiny,len(finite))
    bits_at(address_a,a.size)[:] = a
    bits_at(address_b,b.size)[:] = b
    FULL['residual'] = round_bf16(from_bits(a).astype(np.float64) + from_bits(b).astype(np.float64))


def residual_cases_check(address, count, probe='residual_finite_operands'):
    actual = from_bits(bits_at(address,count).copy())
    expected = FULL.pop('residual')
    finite = np.isfinite(expected)
    result = differences(actual[finite],expected[finite],BUDGETS['operation']['Y'])
    record(dict(probe=probe, overflow=int((~finite).sum()), **result))
    assert not result['failed'], result
