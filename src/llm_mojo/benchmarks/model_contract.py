"""Frozen full-model decode trace geometry, using the production Fast route."""
from .decoder_layer_contract import stages as decoder_stages

OPERATION = 'qwen_model'
ENTRYPOINTS = {'qwen_model_fast': 'QwenModel.forward+greedy'}
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


def stages():
    result = [(-1, 'embedding')]
    for layer in range(24):
        result.extend((layer, name) for name in decoder_stages(0, 1))
        if layer < 23:
            result.append((layer, 'inter-layer copy'))
    return result + [(-1, 'final RMSNorm'), (-1, 'vocabulary projection')]


def specification(prefix):
    if type(prefix) is not int or prefix not in PREFIXES:
        raise ValueError('undeclared Qwen profiling context')
    return dict(profile_rows=1, hidden_size=896, key_value_rows=prefix+1,
                profile_workload=f'model-p{prefix}', dispatches_per_iteration=len(stages()))


def configuration(data):
    if (data.get('implementation') != 'qwen_model_fast'
            or data.get('entrypoint') != ENTRYPOINTS['qwen_model_fast']):
        raise ValueError('Qwen profile implementation changed')
    total = data.get('key_value_rows')
    if type(total) is not int:
        raise ValueError('invalid Qwen cache length')
    expected = specification(total-1)
    if any(type(data.get(k)) is not type(v) or data.get(k) != v for k, v in expected.items()):
        raise ValueError('Qwen trace geometry changed')
    if data.get('profile_iterations') != 8 or data.get('profile_warmup_iterations') != 10:
        raise ValueError('Qwen trace capture budget changed')
    return expected
