"""Attention block workload, rounding policy, dispatch order and frozen inputs."""
import hashlib
import json

from .._repository import repository_root

OPERATION = 'attention_sublayer'
VARIANTS = set(range(3,20))
ENTRYPOINTS = {f'attention_sublayer_{v}': 'enqueue_attention_sublayer' for v in VARIANTS}
ENTRYPOINTS.update({f'attention_sublayer_{v}':'enqueue_attention_sublayer_integrated' for v in range(9,20)})
STAGES = ['RMSNorm', 'Q projection', 'K projection', 'V projection',
          'Q RoPE', 'K RoPE', 'KV append', 'QK', 'softmax', 'PV',
          'output projection', 'residual']
STAGES_BY_VARIANT = {3: STAGES, 4: STAGES,
                     5: STAGES[:7]+['GQA G32']+STAGES[-2:],
                     6: STAGES[:7]+['GQA split','GQA merge']+STAGES[-2:],
                     7: STAGES[:7]+['GQA FP32 MMA']+STAGES[-2:],
                     8: STAGES[:7]+['FP32 GQA']+STAGES[-2:],
                     9: ['RMSNorm','packed QKV projection','QKV unpack']+STAGES[4:7]+['FP32 GQA']+STAGES[-2:]}
STAGES_BY_VARIANT.update({v:STAGES_BY_VARIANT[9] for v in (10,11,14,15,16,17,18)})
STAGES_BY_VARIANT.update({v:STAGES_BY_VARIANT[9][:-3]+['FP32 GQA split','FP32 GQA merge']+STAGES[-2:] for v in (12,13,19)})
TARGET_FIELDS = ('profile_workload', 'dispatches_per_iteration', 'key_value_rows',
                 'query_heads', 'key_value_heads')
PROFILE_WORKLOADS = [(1, 4096), (1024, 1024), (4096, 4096), (64, 4096)]
DECODE_PROFILE_WORKLOADS = [(1,64),(1,4096)]
PREFILL_PROFILE_WORKLOADS = [(1024,1024),(4096,4096),(64,4096)]
COMBINED_PROFILE_WORKLOADS = [(256,256),(1024,1024),(4096,4096),(64,4096)]
PARALLELISM_PROFILE_WORKLOADS = [(64,4096),(1024,1024)]
ARITHMETIC = ('BF16 weights/activations/cache/output; FP32 GQA scores, softmax '
              'probabilities and accumulation; GQA output rounded to BF16 before Wo.')
INPUTS = ('Frozen synthetic case 7 (seed 53, T=4096); each workload uses suffix '
          '[T-R,T), upstream cache prefix and causal full-prefix outputs. Ring24 '
          'owns 24 distinct weights/inputs/caches. Odd entries negate X, Wqkv '
          'and Wo, preserving Q/K/V and negating branch/output. Two sign patterns; '
          'not 24 model layers. Python reads arrays only before timing.')
TIMING = ('Host monotonic whole-sublayer enqueue through completion, microseconds '
          'per call. Ring24 synchronizes once and divides by 24. Scratch/output '
          'shared; prefix copies, allocation and numeric checks excluded. Each '
          'enqueue rewinds logical length to T-R and overwrites the same suffix; '
          'cache append and the host length assignment are included.')
FIXTURES = [f'7_{name}.npy' for name in
            ('input', 'weight', 'bias', 'norm_weight', 'output_weight')]
FIXTURES += [f'upstream_7_{name}.npy' for name in
             ('cosine', 'sine', 'cache_key', 'cache_value')]
FIXTURES += [f'fp32_7_{name}.npy' for name in ('projected', 'output', 'attention')]


# Stable IDs retain their historical meaning. Profile collection and smoke use
# this registry; the Mojo launch path independently checks what actually ran.
PROFILE_COMPARISONS = {
    'baseline': ([3], PROFILE_WORKLOADS),
    'wo': ([3,4], PROFILE_WORKLOADS),
    'decode': ([3,5,6], DECODE_PROFILE_WORKLOADS),
    'prefill': ([4,7], PREFILL_PROFILE_WORKLOADS),
    'projections': ([8,9], PROFILE_WORKLOADS),
    'combined': ([9,18], COMBINED_PROFILE_WORKLOADS),
}
PROFILE_CONTROLS = {**{v:3 for v in (3,4,5,6)}, 7:4, 8:8, 9:8,
                    **{v:9 for v in range(10,19)}, 19:13}


def profile_comparison(name, variants=None):
    if name == 'parallelism':
        spec = dict(variants=variants or [], workloads=PARALLELISM_PROFILE_WORKLOADS)
    else:
        if variants is not None:
            raise ValueError('explicit variants require the parallelism comparison')
        selected, grid = PROFILE_COMPARISONS[name]
        spec = dict(variants=list(selected), workloads=list(grid))
    profile_grid(spec)
    return spec


def fixture_identity():
    root = repository_root()
    frozen = json.loads((root / 'tests/fixtures/attention_sublayer/precision_checksums.json').read_text())
    hashes = {}
    for name in FIXTURES:
        actual = hashlib.sha256((root / 'build/oracle_data/attention_sublayer' / name).read_bytes()).hexdigest()
        if actual != frozen['array_sha256'][name]:
            raise RuntimeError(f'attention benchmark fixture changed: {name}')
        hashes[name] = actual
    return dict(case_id=7, case=frozen['cases'][7], array_sha256=hashes,
                recipe=INPUTS, arithmetic=ARITHMETIC)


def specification(variant, query_rows, key_rows):
    if variant not in VARIANTS:
        raise ValueError('unknown attention sublayer profile variant')
    if variant in (5,6) and query_rows != 1:
        raise ValueError('FP32 decode profile requires one query row')
    if variant == 7 and query_rows <= 1:
        raise ValueError('FP32 prefill profile requires multiple query rows')
    return dict(profile_rows=query_rows, hidden_size=896, key_value_rows=key_rows,
                query_heads=14, key_value_heads=2,
                profile_workload=f'sublayer-r{query_rows}-t{key_rows}-v{variant}',
                dispatches_per_iteration=9 if variant in (12,13,19) and query_rows == 1 else len(STAGES_BY_VARIANT[variant]))


def configuration(data):
    implementation = data.get('implementation', '')
    if implementation not in ENTRYPOINTS or data.get('entrypoint') != ENTRYPOINTS[implementation]:
        raise ValueError('invalid attention sublayer implementation identity')
    r, t = data.get('profile_rows'), data.get('key_value_rows')
    if type(r) is not int or type(t) is not int or not 1 <= r <= t <= 4096:
        raise ValueError('invalid attention sublayer query/key length')
    expected = specification(int(implementation.rsplit('_',1)[1]), r, t)
    if any(data.get(k) != v or (type(v) is int and type(data.get(k)) is not int)
           for k, v in expected.items()):
        raise ValueError('attention sublayer shape or dispatch identity mismatch')
    iterations, warmup = data.get('profile_iterations'), data.get('profile_warmup_iterations')
    if type(iterations) is not int or not 1 <= iterations * expected['dispatches_per_iteration'] <= 5000:
        raise ValueError('attention sublayer profile exceeds dispatch budget')
    if type(warmup) is not int or not 0 <= warmup <= 100:
        raise ValueError('attention sublayer profile warmup is outside bounds')
    return expected


def profile_grid(spec):
    grid = [tuple(w) for w in spec['workloads']]
    variants = spec['variants']
    parallelism = (grid == PARALLELISM_PROFILE_WORKLOADS and variants[:1] == [9]
                   and len(variants) in (2,3) and variants == sorted(set(variants))
                   and set(variants[1:]) <= {10,11,12,13}
                   and sum(v in (10,11) for v in variants) <= 1
                   and sum(v in (12,13) for v in variants) <= 1)
    valid = ((spec['variants'] in ([3],[3,4]) and grid == PROFILE_WORKLOADS)
             or (spec['variants'] == [3,5,6] and grid == DECODE_PROFILE_WORKLOADS)
             or (spec['variants'] == [4,7] and grid == PREFILL_PROFILE_WORKLOADS)
             or (spec['variants'] == [8,9] and grid == PROFILE_WORKLOADS)
             or (variants == [9,18] and grid == COMBINED_PROFILE_WORKLOADS)
             or parallelism)
    if not valid:
        raise ValueError('invalid attention sublayer profile grid')
    return {v: STAGES_BY_VARIANT[v] for v in spec['variants']}, {(r,t,v) for r,t in grid for v in spec['variants']}
