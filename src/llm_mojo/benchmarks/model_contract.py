"""Frozen full-model decode trace geometry, using the production Fast route."""
from .decoder_layer_contract import stages as decoder_stages

OPERATION = 'qwen_model'
# Current default builds follow Fast; historical receipts retain their own route.
FAST_DECODE_CONFIGURATION = 26
ENTRYPOINTS = {'qwen_model_fast': 'QwenModel.forward+greedy', 'qwen_model_fused': 'QwenModel.forward+greedy-fused', 'qwen_model_combined':'QwenModel.forward+greedy-combined'}
TARGET_FIELDS = ('profile_workload', 'dispatches_per_iteration', 'key_value_rows')
PREFIXES = (64, 1024, 3968)
DECLARATION = dict(model='Qwen2.5-0.5B-Instruct', policy='fast', batch=1,
                   dtype='BF16 weights, boundaries and KV; existing FP32 reductions',
                   hidden=896, intermediate=4864, vocabulary=151936, layers=24,
                   cache_capacity=4096, cache_layout='per-layer K/V row-major [4096,128]',
                   maximum_prefill_rows=256, prefixes=list(PREFIXES),
                   timing_blocks=4, warmups_per_arm=10, samples_per_arm=10,
                   comparisons=['control/control','observed/control'],
                   timing_boundary='fixed token upload through greedy readback; logical rewind and recording excluded',
                   trace_repeats=2, trace_warmups=10, trace_steps=8,
                   trace_boundary='normal forward+greedy; fixed logical prefix; no layer synchronizations')


def stages(fused=False, combined=False):
    fused = fused or combined
    result = [(-1, 'embedding')]
    for layer in range(24):
        result.extend((layer, ('fused SiLU/multiply' if combined and name=='SiLU' else ('fused QKV/RoPE/cache' if fused and name=='QKV unpack' else name)))
                      for name in decoder_stages(0, 1)
                      if not (fused and name in {'Q RoPE','K RoPE','KV append'})
                      and not (combined and name=='multiply'))
        if layer < 23:
            result.append((layer, 'inter-layer copy'))
    return result + [(-1, 'final RMSNorm'), (-1, 'vocabulary projection')]


def command_stages(fused=False, combined=False):
    """Observed Metal mapping protocol: two token blits, compute, two logit blits.

    These transfers supplement the 410 compute dispatches in the original
    build receipt. Keep them in the coverage check rather than filtering away
    submissions that lack a compute interval.
    """
    return ([(-1, 'token buffer map', 'blit'), (-1, 'token buffer unmap', 'blit')]
            + [(layer, name, 'compute') for layer, name in stages(fused, combined)]
            + [(-1, 'logit buffer map', 'blit'), (-1, 'logit buffer unmap', 'blit')])


def validate_command_sequence(rows, fused=False, combined=False):
    expected = command_stages(fused, combined)
    if not rows or len(rows) % len(expected):
        raise ValueError('incomplete model compute/transfer sequence')
    for index, row in enumerate(rows):
        kind = expected[index % len(expected)][2]
        label = ':Compute Command' if kind == 'compute' else ':Blit Command'
        if label not in row['event-label'][1]:
            raise ValueError('model compute/transfer ordering changed')


def specification(prefix, fused=False, combined=False):
    if type(prefix) is not int or prefix not in PREFIXES:
        raise ValueError('undeclared Qwen profiling context')
    return dict(profile_rows=1, hidden_size=896, key_value_rows=prefix+1,
                profile_workload=f'model-p{prefix}'+('-combined' if combined else ('-fused' if fused else '')), dispatches_per_iteration=len(stages(fused, combined)))


def configuration(data):
    if (data.get('implementation') not in ENTRYPOINTS
            or data.get('entrypoint') != ENTRYPOINTS[data['implementation']]):
        raise ValueError('Qwen profile implementation changed')
    total = data.get('key_value_rows')
    if type(total) is not int:
        raise ValueError('invalid Qwen cache length')
    expected = specification(total-1, data['implementation']=='qwen_model_fused', data['implementation']=='qwen_model_combined')
    if any(type(data.get(k)) is not type(v) or data.get(k) != v for k, v in expected.items()):
        raise ValueError('Qwen trace geometry changed')
    if data.get('profile_iterations') != 8 or data.get('profile_warmup_iterations') != 10:
        raise ValueError('Qwen trace capture budget changed')
    return expected


FUSION_DECLARATION = {**DECLARATION, 'policy':'configuration 25 vs 0',
    'comparisons':['control/control','fused/control'], 'trace_repeats':1,
    'trace_contexts':[1024], 'trace_arms':['control','fused'],
    'candidate':'single-row QKV unpack + Q RoPE + K RoPE + cache append fusion',
    'promotion':'All four ratios below 1 and median reduction exceeds max(5%, largest absolute control self-pair deviation), at every declared context.'}


COMBINED_DECLARATION = {**FUSION_DECLARATION, 'policy':'configuration 26 vs 0; ablation 26 vs 25',
    'comparisons':['control/control','combined/control','combined/qkv-only'],
    'candidate':'single-row QKV/RoPE/cache fusion plus exact SiLU/multiply fusion',
    'ablation':'Four additional paired blocks per context, ten warmups and ten samples per arm; all four combined/QKV-only ratios must be below one at every context.'}
