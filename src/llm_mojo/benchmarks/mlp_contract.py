"""Frozen MLP measurement inputs, boundaries and seven-dispatch identity."""
import ctypes
import gzip
import hashlib
import importlib.util
import json
from functools import lru_cache
import numpy as np
from .._repository import repository_root

OPERATION = 'mlp'
PROJECTION_MAPPINGS = {0:(0,0), 1:(1,0), 2:(2,0), 3:(3,0),
                       4:(0,1), 5:(0,2), 6:(0,3), 7:(2,2)}
VARIANTS = set(range(19))
ENTRYPOINTS = {f'mlp_{v}':'enqueue_mlp_apple_gpu' for v in VARIANTS}
TILES = {0:'rowwise',1:'MMA 8x16',2:'MMA 16x16',3:'MMA 8x32'}
NAMES = {v:f'gate/up {TILES[g]}; down {TILES[d]}' for v,(g,d) in PROJECTION_MAPPINGS.items()}
NAMES.update({8:'combined gate/up launch', 9:'separate two-output gate/up', 10:'combined two-output gate/up', 11:'cooperative down G2', 12:'cooperative down G4'})
for v in range(13,19):
    NAMES[v] = NAMES[8+(v-13)//2] + '; ' + NAMES[11+(v-13)%2]

def stages(variant):
    gate = 8+(variant-13)//2 if variant >= 13 else variant
    return [STAGES[0], 'gate/up projections', *STAGES[3:]] if gate in (8,10) else STAGES

STAGES = ['RMSNorm','gate projection','up projection','SiLU','multiply','down projection','residual']
BOUNDARIES = ['N','G','U','A','S','D','Y']
ROWS = [1,7,15,16,17,33,65,257,1024,4096]
PROFILE_ROWS = [1,17,1024,4096]
TARGET_FIELDS = ('profile_workload','dispatches_per_iteration','intermediate_size')
CASE = 'h896_i4864_r4096_s1601'
ARITHMETIC = 'BF16 weights and N/G/U/A/S/D/Y; FP32 reductions and SiLU; explicit BF16 stores between all seven stages.'
INPUTS = 'Prefixes of frozen development seed 1601, R=4096. Ring24 has 24 distinct copies of these nonuniform inputs and weights, sharing workspace. It is not a decoder stack or a claim of cold DRAM.'
TIMING = 'Host monotonic enqueue through completion, microseconds per call. Hot synchronizes each call; ring24 synchronizes once after 24 calls and divides by 24. Allocation, upload, checks and output printing excluded.'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_identity():
    root = repository_root()
    anchor = json.loads((root/'tests/fixtures/mlp/checksums.json').read_text())
    payload = (root/'tests/fixtures/mlp/development.json.gz').read_bytes()
    if hashlib.sha256(payload).hexdigest() != anchor['evidence_sha256']:
        raise ValueError('frozen MLP evidence changed')
    raw = gzip.decompress(payload)
    if hashlib.sha256(raw).hexdigest() != anchor['uncompressed_sha256']:
        raise ValueError('frozen MLP payload changed')
    data = json.loads(raw)
    files = {k:v['sha256'] for k,v in data['arrays'].items() if k.startswith(CASE+'/')}
    for name, digest in files.items():
        if sha(root/'build/oracle_data/mlp'/name) != digest:
            raise ValueError('MLP benchmark fixture changed: '+name)
    for name, digest in data['source_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('MLP reference source changed: '+name)
    return dict(case=CASE, arrays=files, anchor_sha256=sha(root/'tests/fixtures/mlp/checksums.json'),
                arithmetic=ARITHMETIC, inputs=INPUTS)


@lru_cache(maxsize=12)
def array(name):
    return np.load(repository_root()/'build/oracle_data/mlp'/CASE/(name+'.npy'), allow_pickle=False, mmap_mode='r')


def load(address, name, count):
    source = (array(name).reshape(-1)[:count].view(np.uint32) >> 16).astype(np.uint16)
    if source.size != count:
        raise ValueError('MLP benchmark input exceeds fixture')
    np.ctypeslib.as_array((ctypes.c_uint16*count).from_address(address))[:] = source


@lru_cache(maxsize=1)
def numerics():
    path = repository_root()/'tests/fixtures/mlp/numerics.py'
    spec = importlib.util.spec_from_file_location('mlp_benchmark_numerics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(address, name, count, composed):
    bits = np.ctypeslib.as_array((ctypes.c_uint16*count).from_address(address)).copy()
    actual = (bits.astype(np.uint32) << 16).view(np.float32)
    expected = array(name).reshape(-1)[:count]
    budget = json.loads((repository_root()/'tests/fixtures/mlp/checksums.json').read_text())['specification']['budgets']
    rule = budget['composition'][name] if composed else budget['operation'][name]
    result = numerics().differences(actual,expected,rule)
    if result['failed']:
        raise ValueError(f'MLP timed route fails frozen {name} gate: {result}')


def specification(variant, rows):
    if type(variant) is not int or variant not in VARIANTS or type(rows) is not int or not 1 <= rows <= 4096:
        raise ValueError('invalid MLP profile shape/variant')
    if variant >= 8 and rows != 1:
        raise ValueError('MLP decode mapping requires one row')
    return dict(profile_rows=rows,hidden_size=896,intermediate_size=4864,
                profile_workload=f'mlp-r{rows}-v{variant}',dispatches_per_iteration=len(stages(variant)))


def configuration(data):
    implementation = data.get('implementation')
    if implementation not in ENTRYPOINTS or data.get('entrypoint') != ENTRYPOINTS[implementation]:
        raise ValueError('invalid MLP profile entrypoint')
    expected = specification(int(implementation.removeprefix('mlp_')),data.get('profile_rows'))
    if any(data.get(k) != v or (type(v) is int and type(data.get(k)) is not int) for k,v in expected.items()):
        raise ValueError('MLP profile identity mismatch')
    if type(data.get('profile_iterations')) is not int or not 1 <= data['profile_iterations']*expected['dispatches_per_iteration'] <= 5000:
        raise ValueError('MLP profile exceeds dispatch budget')
    return expected
