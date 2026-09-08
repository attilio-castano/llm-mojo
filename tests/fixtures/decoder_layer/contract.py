"""Decoder-layer v1: declared inputs and gates, before candidate outputs."""

VERSION = 'decoder-layer-v1'
QWEN = (896, 14, 2, 64, 4864)  # H, Nq, Nkv, D, I
TINY = ((8, 2, 1, 4, 12), (16, 4, 2, 4, 37))
STAGES = ('X', 'N_att', 'Q_raw', 'K_raw', 'V_raw', 'Q', 'K_rot', 'O',
          'B_att', 'Z', 'N_mlp', 'G', 'U', 'A', 'S', 'B_mlp', 'Y',
          'cosine', 'sine', 'cache_key', 'cache_value')
WEIGHTS = ('input_norm', 'qkv', 'bias', 'wo', 'post_norm', 'gate', 'up', 'down')
GATES = {name: dict(atol=2**-5, rtol=2**-5) for name in ('B_att', 'Z', 'B_mlp', 'Y')}
MUTATIONS = ('zero_x', 'zero_wo', 'zero_down', 'zero_branches', 'swap_gate_up',
             'swap_norms', 'small_x', 'large_x')
SYSTEM = 'You are a helpful assistant.'
HOLDOUT_PROMPT = ('A box contains three red balls and two blue balls. Explain how the probability '
                  'of drawing a red ball changes after one blue ball is removed.')


def case_spec(geometry, rows, seed, mutation='base'):
    h, nq, nk, d, i = geometry
    if h != nq*d or nq % nk or d % 2 or not 1 <= rows <= 4096:
        raise ValueError('invalid decoder fixture dimensions')
    return dict(h=h, nq=nq, nk=nk, d=d, i=i, rows=rows, seed=seed, mutation=mutation)


def case_id(spec):
    return 'h{h}_i{i}_nq{nq}_nk{nk}_d{d}_t{rows}_s{seed}_{mutation}'.format(**spec)


def schedules(t):
    if not 1 <= t <= 4096:
        raise ValueError('invalid schedule length')
    result = dict(full=[t], chunk=[1]*t if t <= 17 else [t-17, 16, 1])
    if t == 33:
        result['threshold'] = [16, 1, 15, 1]
    if t == 65:
        result['reuse'] = [53]+[1]*12
    return result


DEVELOPMENT = [case_spec(g, t, 4001) for g in TINY for t in (1, 7, 17)]
DEVELOPMENT += [case_spec(QWEN, t, 4001) for t in (1,7,15,16,17,33,65,257,1024,4096)]
DEVELOPMENT += [case_spec(QWEN, t, 4013) for t in (1,17,4096)]
DEVELOPMENT += [case_spec(g, 17, 4001, m) for g in (*TINY, QWEN) for m in MUTATIONS]
HOLDOUT = [case_spec(QWEN, t, seed) for seed in (5003, 5011) for t in (1,17,4096)]


def specification():
    return dict(version=VERSION, stages=STAGES, weights=WEIGHTS,
                development=DEVELOPMENT, holdout=HOLDOUT,
                system=SYSTEM, holdout_prompt=HOLDOUT_PROMPT,
                whole_layer_gates=GATES, schedules={str(t):schedules(t) for t in sorted({s['rows'] for s in DEVELOPMENT})},
                recipe='PCG64 SeedSequence([seed,H,I,Nq,Nkv,D,tag]); FP64 normal -> scale -> FP32 -> BF16',
                policy='CPU math SDPA with explicit FP32 Q/K/V/mask and BF16 output before Wo',
                holdout_outputs_observed=False)
