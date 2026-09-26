"""Frozen full-model decode trace geometry, using the production Fast route."""
from .decoder_layer_contract import stages as decoder_stages

OPERATION = 'qwen_model'
# Current default builds follow Fast; historical receipts retain their own route.
FAST_DECODE_CONFIGURATION = 26
FAST_DECODE_VARIANT = 3
ENTRYPOINTS = {'qwen_model_fast': 'QwenModel.forward+greedy', 'qwen_model_fused': 'QwenModel.forward+greedy-fused', 'qwen_model_combined':'QwenModel.forward+greedy-combined'}
ENTRYPOINTS.update(qwen_model_gpu_argmax='QwenModel.forward+greedy-gpu-argmax', qwen_model_fused_head='QwenModel.forward+greedy-fused-head')
ENTRYPOINTS['qwen_model_buffer_swap'] = 'QwenModel.forward+greedy-buffer-swap'
ENTRYPOINTS.update(qwen_model_residual_norm='QwenModel.forward+greedy-residual-norm', qwen_model_swap_argmax='QwenModel.forward+greedy-swap-argmax', qwen_model_all_three='QwenModel.forward+greedy-all-three')
SELECTIONS = {'qwen_model_gpu_argmax':1, 'qwen_model_fused_head':2, 'qwen_model_swap_argmax':1, 'qwen_model_all_three':1}
# The batch-size study: the all-three decode composition for B sequences in one step.
BATCH_IMPLEMENTATION = 'qwen_model_batch'
ENTRYPOINTS[BATCH_IMPLEMENTATION] = 'QwenModel.forward+greedy_tokens'
SELECTIONS[BATCH_IMPLEMENTATION] = 1
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


def stages(fused=False, combined=False, selection=0, copy_free=False, residual_norm=False):
    combined = combined or bool(selection) or copy_free or residual_norm
    fused = fused or combined
    result = [(-1, 'embedding')]
    for layer in range(24):
        for name in decoder_stages(0, 1):
            if (fused and name in {'Q RoPE','K RoPE','KV append'}
                or combined and name=='multiply'
                or residual_norm and (name=='MLP RMSNorm' or name=='attention RMSNorm' and layer>0)):
                continue
            if residual_norm and name=='attention residual': name='fused attention residual/RMSNorm'
            elif residual_norm and name=='MLP residual': name='fused MLP residual/RMSNorm'
            elif combined and name=='SiLU': name='fused SiLU/multiply'
            elif fused and name=='QKV unpack': name='fused QKV/RoPE/cache'
            result.append((layer,name))
        if layer < 23 and not copy_free:
            result.append((layer, 'inter-layer copy'))
    head = [(-1, 'fused vocabulary/local argmax'),(-1, 'argmax finish')] if selection == 2 else [(-1, 'vocabulary projection')]
    if selection == 1:
        head += [(-1,'argmax partials'),(-1,'argmax finish')]
    return result + ([] if residual_norm else [(-1, 'final RMSNorm')]) + head


def command_stages(fused=False, combined=False, selection=0, copy_free=False, residual_norm=False):
    """Observed Metal mapping protocol: two token blits, compute, two logit blits.

    These transfers supplement the 410 compute dispatches in the original
    build receipt. Keep them in the coverage check rather than filtering away
    submissions that lack a compute interval.
    """
    return ([(-1, 'token buffer map', 'blit'), (-1, 'token buffer unmap', 'blit')]
            + [(layer, name, 'compute') for layer, name in stages(fused, combined, selection, copy_free, residual_norm)]
            + [(-1, ('winner' if selection else 'logit')+' buffer map', 'blit'), (-1, ('winner' if selection else 'logit')+' buffer unmap', 'blit')])


def validate_command_sequence(rows, fused=False, combined=False, selection=0, copy_free=False, residual_norm=False):
    expected = command_stages(fused, combined, selection, copy_free, residual_norm)
    if not rows or len(rows) % len(expected):
        raise ValueError('incomplete model compute/transfer sequence')
    for index, row in enumerate(rows):
        kind = expected[index % len(expected)][2]
        label = ':Compute Command' if kind == 'compute' else ':Blit Command'
        if label not in row['event-label'][1]:
            raise ValueError('model compute/transfer ordering changed')


def specification(prefix, fused=False, combined=False, selection=0, copy_free=False, residual_norm=False):
    if type(prefix) is not int or prefix not in PREFIXES:
        raise ValueError('undeclared Qwen profiling context')
    if residual_norm: suffix='-all-three' if copy_free and selection else '-residual-norm'
    elif copy_free: suffix='-swap-argmax' if selection else '-buffer-swap'
    elif selection: suffix='-gpu-argmax' if selection==1 else '-fused-head'
    else: suffix='-combined' if combined else ('-fused' if fused else '')
    return dict(profile_rows=1, hidden_size=896, key_value_rows=prefix+1,
                profile_workload=f'model-p{prefix}'+suffix,
                dispatches_per_iteration=len(stages(fused, combined, selection, copy_free, residual_norm)))


def options(implementation):
    return (implementation=='qwen_model_fused',
            implementation in ('qwen_model_combined','qwen_model_residual_norm','qwen_model_swap_argmax','qwen_model_all_three',BATCH_IMPLEMENTATION),
            SELECTIONS.get(implementation,0),
            implementation in ('qwen_model_buffer_swap','qwen_model_swap_argmax','qwen_model_all_three',BATCH_IMPLEMENTATION),
            implementation in ('qwen_model_residual_norm','qwen_model_all_three',BATCH_IMPLEMENTATION))


def configuration(data):
    if data.get('implementation') == BATCH_IMPLEMENTATION:
        return batch_configuration(data)
    if (data.get('implementation') not in ENTRYPOINTS
            or data.get('entrypoint') != ENTRYPOINTS[data['implementation']]):
        raise ValueError('Qwen profile implementation changed')
    total = data.get('key_value_rows')
    if type(total) is not int:
        raise ValueError('invalid Qwen cache length')
    expected = specification(total-1, *options(data['implementation']))
    if any(type(data.get(k)) is not type(v) or data.get(k) != v for k, v in expected.items()):
        raise ValueError('Qwen trace geometry changed')
    if data.get('profile_iterations') != 8 or data.get('profile_warmup_iterations') != 10:
        raise ValueError('Qwen trace capture budget changed')
    if 'projection_variant' in data:
        variant=data['projection_variant']
        if data['implementation']!='qwen_model_all_three' or type(variant) is not int or not 0<=variant<=5:
            raise ValueError('invalid Qwen projection variant')
        expected['projection_variant']=variant
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


SELECTION_DECLARATION = {**DECLARATION, 'policy':'configuration 26 with CPU, GPU argmax, or fused head selection',
    'comparisons':['cpu/cpu','gpu-argmax/cpu','fused-head/cpu','fused-head/gpu-argmax'],
    'candidate':'1024-logit argmax groups; 64-logit fused projection groups; 128 threads; final reduction; 12-byte result',
    'trace_repeats':1, 'trace_contexts':[1024], 'trace_arms':['cpu','gpu-argmax','fused-head'],
    'promotion':FUSION_DECLARATION['promotion'],
    'choice':'Prefer qualifying GPU argmax unless fused head also clears the same gate against it; otherwise choose the sole qualifier or retain CPU.'}


COPY_FREE_DECLARATION = {**FUSION_DECLARATION, 'policy':'configuration 26 with copy vs owner swap; CPU greedy',
    'candidate':'swap input and MLP-output owners after layers 0..22; no extra allocation or synchronization',
    'comparisons':['copy/copy','swap/copy'],
    'extra_correctness':'all hidden states at history 64; consecutive single/multi-row calls, reset, invalid IDs and owner identities'}


COMPOSITION_ARMS = ['combined','residual-norm','swap-argmax','all-three']
COMPOSITION_IMPLEMENTATIONS = ['qwen_model_combined','qwen_model_residual_norm','qwen_model_swap_argmax','qwen_model_all_three']
COMPOSITION_PAIRS = [(0,0),(0,1),(0,2),(0,3),(2,3),(1,3)]
COMPOSITION_DECLARATION = {**COPY_FREE_DECLARATION,
    'policy':'configuration 26; independent residual RMSNorm and composition with owner swap and separate GPU argmax',
    'candidate':'48 exact residual/RMSNorm fusions; seven BF16 values per thread; same SIMD-group reduction and rounding',
    'arms':COMPOSITION_ARMS, 'comparisons':[list(pair) for pair in COMPOSITION_PAIRS],
    'trace_arms':COMPOSITION_ARMS,
    'extra_correctness':'logits and complete caches at all histories; 195 layer tensors and full ownership/reset/rejection lifecycle per candidate at history 64',
    'choice':'Prefer qualifying all-three if all direct ratios against other qualifiers are below 1; else sole qualifier; unresolved multiple qualifiers retain Fast.'}


PROJECTION_ARMS = ['projection-'+str(v) for v in range(6)]
PROJECTION_LABELS = ['runtime/128','fixed/128','runtime/64','runtime/256','fixed/64','fixed/256']
PROJECTION_DECLARATION = dict(COMPOSITION_DECLARATION,
    policy='all-three Fast; 121 rowwise projections with exact-width and block-size arrangements',
    comparisons=[[0,v] for v in range(6)], arms=PROJECTION_ARMS, trace_arms=PROJECTION_ARMS,
    candidate='one output per SIMD group; runtime or fixed 896/4864 width with four-iteration prefetch; 64/128/256 threads',
    extra_correctness='15 full logits/cache comparisons and 195 extra tensors per candidate at prefix64',
    choice='qualify at all contexts; lowest worst-context median ratio, then mean, then ID; separate confirmation required')


BATCH_CONTEXTS = (64, 1024, 3968)
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
BATCH_MIXED = 32
BATCH_TILES = (4, 8, 16)
BATCH_COMPARISONS = [['tile-4','tile-4'], ['tile-4','tile-8'], ['tile-4','tile-16'], ['tile-4','tile-4 observed']]
BATCH_TRACES = ((1024, 1, 4), (1024, 16, 4), (1024, 64, 4))


def mixed_contexts():
    """The mixed workload (context 0): 32 sequences whose contexts spread evenly from 64 to 3968."""
    return [64 + s*(3968-64)//(BATCH_MIXED-1) for s in range(BATCH_MIXED)]


def batch_workloads():
    """(context, sequences) cells, context 0 being the mixed batch."""
    return [(c, b) for c in BATCH_CONTEXTS for b in BATCH_SIZES] + [(0, BATCH_MIXED)]


BATCH_DECLARATION = dict(DECLARATION, policy='fast; configuration 26 decode composition for B sequences, one token each',
    batch=list(BATCH_SIZES), prefixes=list(BATCH_CONTEXTS),
    mixed=dict(sequences=BATCH_MIXED, contexts=mixed_contexts()), row_tiles=list(BATCH_TILES),
    comparisons=BATCH_COMPARISONS,
    cache_layout='block-major pool of 64 full-context blocks in one 3 GiB allocation; block 0 prefilled with the frozen history and copied to every block',
    timing_boundary='step batch and plan construction through greedy_tokens readback of every sequence; logical rewind and recording excluded',
    trace_repeats=2, trace_workloads=[list(t) for t in BATCH_TRACES],
    trace_boundary='normal forward+greedy_tokens for B sequences; fixed logical contexts; no layer synchronizations',
    decision='Per workload and tile: gain if all four block ratios are below one and the median reduction exceeds '
             'max(5%, largest absolute calibration deviation); regression by the symmetric rule; otherwise inconclusive.')


def batch_specification(context, sequences, tile):
    if (context, sequences, tile) not in BATCH_TRACES:
        raise ValueError('undeclared batch trace workload')
    return dict(profile_rows=sequences, hidden_size=896, key_value_rows=context+1,
                profile_workload=f'model-p{context}-b{sequences}-t{tile}',
                dispatches_per_iteration=len(stages(*options(BATCH_IMPLEMENTATION))))


# 1d, exact batched projections: arrangements 3-6 against arrangement 0 (tile 4) in the batch-size matrix.
BATCH_TILE_ARRANGEMENTS = {4: 0, 8: 1, 16: 2}
BATCH_PROJECTION_ARRANGEMENTS = (3, 4, 5, 6)
BATCH_PROJECTION_TRACES = tuple((1024, 64, a) for a in (0,) + BATCH_PROJECTION_ARRANGEMENTS)
BATCH_PROJECTION_DECLARATION = dict(
    {k: v for k, v in BATCH_DECLARATION.items() if k != 'row_tiles'},
    policy='fast; configuration 26 decode composition for B sequences, one token each; batched projections in one arrangement',
    arrangements={'0': 'tile 4: four rows and one column per SIMD group, runtime width, row guard in the loop',
                  '3': 'four rows and one column, fixed width 896 or 4864 with four iterations of early loads, no guard in the loop',
                  '4': 'four rows and four columns, runtime width, no guard in the loop',
                  '5': 'four rows and four columns, fixed width with early loads, no guard in the loop',
                  '6': 'arrangement 5 with the row tiles of one column block in consecutive SIMD groups'},
    arithmetic='every output keeps the one-row kernel lane-strided FP32 sum, warp.sum, FP32 bias and BF16 rounding',
    comparisons=[['arrangement-0', 'arrangement-0']] + [['arrangement-0', f'arrangement-{a}'] for a in BATCH_PROJECTION_ARRANGEMENTS],
    trace_workloads=[list(t) for t in BATCH_PROJECTION_TRACES],
    decision='Per workload and arrangement: gain if all four block ratios are below one and the median reduction exceeds '
             'max(5%, largest absolute calibration deviation); regression by the symmetric rule; otherwise inconclusive.',
    qualification='no regression in any workload with B >= 2 and a gain in every workload with B >= 4',
    selection='lowest worst-case median ratio over workloads with B >= 4, then lowest mean ratio, then lower ID',
    confirmation='a fresh four-block run of the selected arrangement against arrangement 0 over the same workloads, '
                 'with its own calibration and the same qualifying rule; no other candidate if it fails')


# 1e, reordered batched projections: arrangements 7-10 against arrangement 5, the batched default.
BATCH_REORDERED_CONTROL = 5
BATCH_REORDERED_ARRANGEMENTS = (7, 8, 9, 10)
BATCH_REORDERED_TRACES = tuple((1024, 64, a) for a in (BATCH_REORDERED_CONTROL,) + BATCH_REORDERED_ARRANGEMENTS)
BATCH_ACCURACY_SHAPES = ((8, 1152, 896), (8, 896, 896), (8, 4864, 896), (8, 896, 4864), (2, 151936, 896))
BATCH_REORDERED_DECLARATION = dict(
    {k: v for k, v in BATCH_PROJECTION_DECLARATION.items() if k not in ('arrangements', 'comparisons', 'trace_workloads')},
    policy='fast; configuration 26 decode composition; batched and single-row projections in one arrangement',
    arrangements={'5': 'the batched default: four rows and four columns, fixed width, early loads, today\'s order',
                  '7': 'arrangement 5 with eight rows per SIMD group, today\'s order',
                  '8': 'four rows and four columns; each lane sums four adjacent products in every 128, then warp.sum; '
                       'one row runs the same order',
                  '9': 'matrix-unit 8x32 tiles: 8x8 fragments along K in steps of 8, FP32 accumulators, for any row count',
                  '10': 'matrix-unit 16x16 tiles, as 9'},
    arithmetic='5 and 7 keep the one-row kernel order; 8-10 change it, and a single row follows the same order',
    comparisons=[['arrangement-5', 'arrangement-5']] + [['arrangement-5', f'arrangement-{a}'] for a in BATCH_REORDERED_ARRANGEMENTS],
    trace_workloads=[list(t) for t in BATCH_REORDERED_TRACES],
    accuracy=dict(shapes=[list(s) for s in BATCH_ACCURACY_SHAPES], reference='FP64 sum of the same BF16 operands',
                  unit='absolute error in BF16 units in the last place at the FP64 sum',
                  values='the kernel tests: mixed signs and exponents, every fifth a signed zero, subnormal or neighbour of one',
                  gate='a reordered arrangement\'s worst error per shape does not exceed arrangement 5\'s'),
    qualification='the accuracy gate; no regression in any workload; a gain in every workload with B >= 16',
    selection='lowest worst-case median ratio over workloads with B >= 16, then lowest mean ratio, then lower ID',
    confirmation='a fresh four-block run of the selected arrangement against arrangement 5 over the same workloads, '
                 'with its own calibration and the same qualifying rule; then the model-level diagnostics',
    diagnostics='teacher-forced decode comparison against arrangement 5; HF same-history comparison of 5 and the '
                'selected arrangement; stop if the selected agrees with HF on more than one fewer decode choice or '
                'its largest KL divergence more than doubles',
    adoption='an exact selection becomes the default on 1d\'s terms; a reordered one waits for a separate decision')


def batch_projection_specification(context, sequences, arrangement):
    if (context, sequences, arrangement) not in BATCH_PROJECTION_TRACES + BATCH_REORDERED_TRACES:
        raise ValueError('undeclared projection trace workload')
    return dict(profile_rows=sequences, hidden_size=896, key_value_rows=context+1,
                profile_workload=f'model-p{context}-b{sequences}-a{arrangement}',
                dispatches_per_iteration=len(stages(*options(BATCH_IMPLEMENTATION))))


def batch_configuration(data):
    """A batch trace names 1c's row tile or 1d's arrangement, never both."""
    if data.get('entrypoint') != ENTRYPOINTS[BATCH_IMPLEMENTATION]:
        raise ValueError('Qwen batch profile entrypoint changed')
    if ('row_tile' in data) == ('arrangement' in data):
        raise ValueError('a Qwen batch trace names a row tile or an arrangement')
    arm = 'arrangement' if 'arrangement' in data else 'row_tile'
    fields = (data.get('key_value_rows'), data.get('profile_rows'), data.get(arm))
    if any(type(value) is not int for value in fields):
        raise ValueError('invalid Qwen batch trace geometry')
    specify = batch_projection_specification if arm == 'arrangement' else batch_specification
    expected = specify(fields[0]-1, fields[1], fields[2])
    expected[arm] = fields[2]
    if any(type(data.get(k)) is not type(v) or data.get(k) != v for k, v in expected.items()):
        raise ValueError('Qwen batch trace geometry changed')
    if data.get('profile_iterations') != 8 or data.get('profile_warmup_iterations') != 10:
        raise ValueError('Qwen trace capture budget changed')
    return expected
