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


# 1f, addressing: arrangement 5 with raw-pointer loads (11) and arrangement 8, both against 5.
BATCH_ADDRESSING_CONTROL = 5
BATCH_ADDRESSING_ARRANGEMENTS = (11, 8)
BATCH_ADDRESSING_TRACES = tuple((1024, 64, a) for a in (BATCH_ADDRESSING_CONTROL,) + BATCH_ADDRESSING_ARRANGEMENTS)
BATCH_ADDRESSING_DECLARATION = dict(
    {k: v for k, v in BATCH_REORDERED_DECLARATION.items()
     if k not in ('arrangements', 'arithmetic', 'comparisons', 'trace_workloads', 'accuracy', 'qualification',
                  'selection', 'confirmation', 'diagnostics', 'adoption')},
    question='how much of arrangement 8\'s gain over arrangement 5 comes from addressing loads from raw pointers '
             'rather than from loading four adjacent values at once',
    arrangements={'5': 'four rows and four columns, fixed width, early loads, the one-row kernel\'s order; scalar '
                       'loads through the tensor layout',
                  '11': 'arrangement 5 with each scalar load addressed from a raw pointer offset, in 5\'s order',
                  '8': 'the decode default: arrangement 5\'s tile with four adjacent values per lane in one '
                       'raw-pointer vector load, in another order'},
    arithmetic='5 and 11 keep the one-row kernel order; 8 changes it',
    comparisons=[['arrangement-5', 'arrangement-5']] + [['arrangement-5', f'arrangement-{a}']
                                                        for a in BATCH_ADDRESSING_ARRANGEMENTS],
    trace_workloads=[list(t) for t in BATCH_ADDRESSING_TRACES],
    analysis='diagnostic, no selection: in each workload where arrangement 8 is a gain, the share of its gain that '
             'arrangement 11 reaches, (1 - r11) / (1 - r8) of their median paired ratios against arrangement 5; the '
             'same share of traced projection time at B = 64 and 1,024 cached tokens',
    hypothesis='load width, not addressing, explains most of the gain: arrangement 11 reaches less than a third of '
               'arrangement 8\'s gain from B = 16',
    consequence='only if arrangement 11 is slower in no workload and reaches at least 80% of arrangement 8\'s gain '
                'in every workload from B = 16 does an exact default go back to a decision; otherwise arrangement '
                '8 stays')


def batch_projection_specification(context, sequences, arrangement):
    if (context, sequences, arrangement) not in BATCH_PROJECTION_TRACES + BATCH_REORDERED_TRACES + BATCH_ADDRESSING_TRACES:
        raise ValueError('undeclared projection trace workload')
    return dict(profile_rows=sequences, hidden_size=896, key_value_rows=context+1,
                profile_workload=f'model-p{context}-b{sequences}-a{arrangement}',
                dispatches_per_iteration=len(stages(*options(BATCH_IMPLEMENTATION))))


# 2d, paged KV translation cost (docs/paged-kv-plan.md): blocks of 32, 64 and 128 slots, each
# slot-major and head-major, against layout 0, one block of the whole context per sequence.
PAGED_CONTROL = 0
PAGED_LAYOUTS = {0: dict(block_size=4096, order='slot-major'),
                 1: dict(block_size=32, order='slot-major'), 2: dict(block_size=32, order='head-major'),
                 3: dict(block_size=64, order='slot-major'), 4: dict(block_size=64, order='head-major'),
                 5: dict(block_size=128, order='slot-major'), 6: dict(block_size=128, order='head-major')}
PAGED_CANDIDATES = (1, 2, 3, 4, 5, 6)
# Rows, total positions after the chunk and the configuration Fast's plan runs: the runtime
# study's eleven cells, then 256-row chunks after 256 and 2,816 cached tokens.
PAGED_PREFILL_WORKLOADS = ((16, 1024, 2), (16, 4096, 2), (15, 256, 2), (17, 256, 2), (64, 1024, 3),
                           (64, 4096, 3), (256, 1024, 3), (256, 4096, 3), (65, 4096, 3), (255, 4096, 3),
                           (16, 256, 21), (256, 512, 0), (256, 3072, 0))
PAGED_TRACES = tuple((3968, 64, layout) for layout in (0, 1, 3, 5))
PAGED_DECLARATION = dict(
    {k: v for k, v in BATCH_DECLARATION.items()
     if k not in ('row_tiles', 'comparisons', 'trace_workloads', 'decision', 'cache_layout', 'timing_boundary')},
    question='what address translation costs in decode and prefill at block sizes 32, 64 and 128 against one '
             'block per sequence, and whether head-major order within a block changes it',
    policy='fast; configuration 26 decode composition for B sequences, one token each, decode projection '
           'arrangement 8; prefill chunks under Fast\'s plan',
    layouts={str(k): f"{v['block_size']}-slot blocks, {v['order']}" for k, v in PAGED_LAYOUTS.items()},
    cache_layout='one working pool of 262,144 slots (3 GiB) held in each arm\'s layout: 64 blocks of 4,096, 2,048 of '
                 '128, 4,096 of 64 or 8,192 of 32; every table from a block manager whose free list is a '
                 'permutation seeded with 2026; before every arm, the control\'s included and outside timing, '
                 'each sequence\'s cached blocks are copied from its layout\'s history',
    history='the frozen history prefilled once per layout in 256-row Fast chunks into a one-sequence pool of '
            '48 MiB; every layout\'s K/V rows must equal layout 0\'s byte for byte before measurement, and every '
            'step and chunk must select layout 0\'s tokens',
    prefill=dict(workloads=[list(w) for w in PAGED_PREFILL_WORKLOADS],
                 procedure='one process per block on one-sequence pools, the same comparisons and order'),
    comparisons=[['layout-0', 'layout-0']] + [['layout-0', f'layout-{layout}'] for layout in PAGED_CANDIDATES],
    timing_boundary=dict(
        decode='step batch construction, its tables from the block manager included, through greedy_tokens '
               'readback of every sequence; logical rewind and recording excluded',
        prefill='a resident forward from the token upload to device synchronization; logical rewind, batch '
                'construction and greedy readback excluded'),
    trace_workloads=[list(t) for t in PAGED_TRACES],
    decision='Per workload and layout: gain if all four block ratios are below one and the median reduction exceeds '
             'max(5%, largest absolute calibration deviation); regression by the symmetric rule; otherwise inconclusive.',
    qualification='a regression in none of the 35 workloads, 22 decode and 13 prefill',
    selection='the smallest qualifying block size; at that size slot-major, unless head-major also qualifies and is '
              'a gain in at least one workload; head-major when only it qualifies at that size',
    confirmation='a fresh four-block run of the selected layout against layout 0 over all 35 workloads, with its own '
                 'calibration and the same rule; no other layout if it fails',
    single_sequence='sixteen generation runs in four alternating blocks, 128 tokens after the 1,176-token prompt, '
                    'comparing 6422f84\'s generate executable with one built in the selected layout; slower in all '
                    'four blocks by a median above 5% stops adoption, slower in all four by less returns the decision',
    otherwise='one block per sequence stays the default',
    hypothesis='recorded before measurement in docs/paged-kv-plan.md')


# 2d's rerun after its follow-up: the same matrix, procedure and rule, with decode attention
# walking each SIMD group's keys in one loop from block offsets staged in threadgroup memory.
PAGED_LOOP_DECLARATION = dict(
    PAGED_DECLARATION,
    attention='each SIMD group walks its keys t = g (mod 32) in one loop; the threadgroup first stages the K '
              'offset of every block the row sees in threadgroup memory, one table read per block',
    hypothesis='recorded before the rerun in docs/paged-kv-plan.md')


def paged_specification(context, sequences, layout):
    if (context, sequences, layout) not in PAGED_TRACES:
        raise ValueError('undeclared paged KV trace workload')
    return dict(profile_rows=sequences, hidden_size=896, key_value_rows=context+1,
                profile_workload=f'model-p{context}-b{sequences}-l{layout}',
                dispatches_per_iteration=len(stages(*options(BATCH_IMPLEMENTATION))))


# The single-sequence check of 1a, 1b and 1e: the Fast generator, 128 tokens after a 1,176-token
# prompt, sixteen runs in four alternating blocks.
SINGLE_SEQUENCE_SENTENCE = ('A train travels sixty kilometers in forty-five minutes. Explain how to calculate its '
                            'average speed, keeping track of distance, time, and units.')
SINGLE_SEQUENCE = dict(
    prompt=dict(construction='the sentence repeated 42 times, joined by spaces, then a newline',
                sentence=SINGLE_SEQUENCE_SENTENCE, repeats=42, tokens=1176,
                sha256='7a74b43ca1f6fb3d3ae131776b5dabddb6079bb57428cb4d35d6828b72fad0b2'),
    max_new_tokens=128, chunk_rows=256, policy='fast',
    blocks=[['baseline', 'candidate', 'candidate', 'baseline'], ['candidate', 'baseline', 'baseline', 'candidate']]*2,
    procedure='Each run: generator PREPARED TABLES PROMPT 128 256 fast REPORT. The report must pass the generation '
              'validator, which requires the M4 Pro\'s Metal device, and the run keeps its device and decode events. '
              'Sixteen runs in four blocks ordered baseline candidate candidate baseline, then candidate baseline '
              'baseline candidate, alternating.',
    rule='Block ratio: the candidate\'s mean median decode step over the baseline\'s, per block. All four above 1 '
         'and their median above 1.05: regression. All four above 1 otherwise: consistent slowdown below the floor. '
         'Anything else: no regression. Fixed before measuring.')


def single_sequence_prompt():
    prompt = SINGLE_SEQUENCE['prompt']
    return ' '.join([prompt['sentence']]*prompt['repeats']) + '\n'


def batch_configuration(data):
    """A batch trace names 1c's row tile, an arrangement of 1d-1f or 2d's KV layout, exactly one."""
    if data.get('entrypoint') != ENTRYPOINTS[BATCH_IMPLEMENTATION]:
        raise ValueError('Qwen batch profile entrypoint changed')
    named = [key for key in ('row_tile', 'arrangement', 'layout') if key in data]
    if len(named) != 1:
        raise ValueError('a Qwen batch trace names a row tile, an arrangement or a layout')
    arm = named[0]
    fields = (data.get('key_value_rows'), data.get('profile_rows'), data.get(arm))
    if any(type(value) is not int for value in fields):
        raise ValueError('invalid Qwen batch trace geometry')
    specify = dict(row_tile=batch_specification, arrangement=batch_projection_specification,
                   layout=paged_specification)[arm]
    expected = specify(fields[0]-1, fields[1], fields[2])
    expected[arm] = fields[2]
    if any(type(data.get(k)) is not type(v) or data.get(k) != v for k, v in expected.items()):
        raise ValueError('Qwen batch trace geometry changed')
    if data.get('profile_iterations') != 8 or data.get('profile_warmup_iterations') != 10:
        raise ValueError('Qwen trace capture budget changed')
    return expected


# Phase 3 keeps the token-trace declaration beside the existing model matrices.
ENGINE_ARMS = ('serial', 'static', 'continuous', 'chunked')
ENGINE_STEP_FIELDS = (
    'step_id', 'decode_seqs', 'prefill_seqs', 'prefill_tokens', 'total_tokens',
    'attended_positions', 'admitted', 'preempted', 'finished', 'aborted',
    'waiting', 'blocks_free', 'begin_ns', 'schedule_ns', 'build_ns',
    'execute_ns', 'postprocess_ns', 'end_ns', 'predicted_ns', 'budget_limited',
)
ENGINE_POLICY_FIELDS = ('target_ns', 'fixed_ns', 'per_row_ns', 'per_position_ns',
                        'per_partition_ns', 'per_logit_ns')
ENGINE_DECLARATION = dict(
    kind='qwen-engine-core-v1', schema_version=1, arms=list(ENGINE_ARMS),
    model='Qwen2.5-0.5B-Instruct', policy='reference configuration 27; BF16 storage, FP32 reductions',
    max_context=4096, block_size=32, order='slot-major', paired_blocks=4,
    warmup_steps=10, timing_boundary='scheduled arrival through completed trace drain; '
        'resident weights, initialization and warmup outside measured trace',
    telemetry='schedule/build/execute/postprocess host durations; execute includes '
        'upload, GPU submission, synchronization and selected-token readback',
    numerical_policy='mixed, decode-only and replayed execution use the same reference route; '
        'own-route mixed-versus-solo equality is exact; comparisons against Fast are diagnostics; '
        'record any natural greedy history divergence',
    target=None, fitted_budget=False, asynchronous=False,
)

# Keep the original declaration immutable: retained v1 archives still describe
# their original serial/static/continuous/chunked grid and incremental admission.
ENGINE_ADMISSION_POLICIES = ('incremental', 'reserved')
ENGINE_ADMISSION_DECLARATION = dict(
    ENGINE_DECLARATION, kind='qwen-engine-admission-v1', arms=['chunked'],
    admission_policies=list(ENGINE_ADMISSION_POLICIES), control='incremental',
    token_budget=256, max_sequences=8,
    reservation='zero blocks for zero output budget; otherwise '
        'ceil(max(prompt_length, prompt_length + max_new_tokens - 1) / block_size); '
        'reserved admission commits lifetime capacity before prefill',
    request_scope='bounded traces without aborts; every request terminates by stop or output limit',
    correctness_gates='greedy histories identical across control, self-control and reserved; '
        'reserved preemptions zero and executed rows exactly sum(prompt_length + delivered_tokens - 1) '
        'for positive outputs, zero rows for zero outputs',
    pairing='same binary and frozen trace; incremental control, incremental self-control, '
        'reserved candidate; forward/reverse/reverse/forward block order',
)

# The operating-range study adds observations; the earlier paired study's raw
# records, declaration and summaries remain byte-for-byte replay compatible.
ENGINE_ADMISSION_RANGE_TELEMETRY = 'admission-range-v1'
ENGINE_KV_FIELDS = ('step_id', 'phase', 'timestamp_ns', 'allocated_blocks',
                    'written_blocks', 'written_tokens', 'reserved_tokens',
                    'waiting_requests', 'resident_requests')
ENGINE_KV_PHASES = ('start', 'scheduled', 'executed', 'end')
ENGINE_ADMISSION_RANGE_DECLARATION = dict(
    ENGINE_ADMISSION_DECLARATION, kind='qwen-engine-admission-range-v1',
    observation=ENGINE_ADMISSION_RANGE_TELEMETRY,
    workload_scope='frozen varied output limits and natural greedy stop tokens; no teacher forcing',
    kv_observation='host-only snapshots at start, scheduled, executed and end boundaries; '
        'no GPU readback or additional synchronization',
    byte_time='physical allocated byte-time during execute and between steps; '
        'unused block/slot byte-time bounded by pre/post execute written occupancy; '
        'schedule/build/postprocess excluded from occupancy integration; '
        'the time of writes within execution is unobserved',
    kv_geometry=dict(layers=24, kv_heads=2, head_dim=64, storage_bytes=2),
    admission_delay='scheduled arrival and observed ingress to first positive-output admission; '
        'zero-output requests have no admission and release at a step boundary',
    execution_timeout_seconds=180,
    recommendation='descriptive operating range; no latency SLO, goodput target or default promotion',
)
