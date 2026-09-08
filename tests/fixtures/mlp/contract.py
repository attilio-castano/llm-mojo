"""Declared before MLP GPU/holdout evaluation. Only development cases run by default."""

VERSION = 'mlp-reference-v1'
STAGES = ('N', 'G', 'U', 'A', 'S', 'D', 'Y')
WEIGHTS = ('norm', 'gate', 'up', 'down')
DEVELOPMENT = [(h, i, r, 1601) for h, i in ((8, 12), (7, 11)) for r in (1, 7, 17)]
DEVELOPMENT += [(896, 4864, r, 1601) for r in (1, 7, 15, 16, 17, 33, 65, 257, 1024, 4096)]
DEVELOPMENT += [(896, 4864, r, 1613) for r in (1, 17, 4096)]
HOLDOUT = [(896, 4864, r, seed) for seed in (2017, 2027) for r in (1, 17, 4096)]
HOLDOUT_PROMPT = 'Explain why multiplying two negative numbers gives a positive number, using a numerical example.'
MUTATIONS = ('zero_input', 'zero_gate', 'zero_up', 'zero_down', 'swap_gate_up',
             'small_input', 'large_input', 'cancellation')
UP_VALUES = (-16, -1, -1/64, 0, 1/64, 1, 16)
# Initial acceptance hypotheses from the declared development evidence, before
# any Mojo MLP or holdout result. See docs/mlp-sublayer.md for rationale.
BUDGETS = dict(
    operation={
        'N': dict(atol=2**-7, rtol=2**-7),
        'G': dict(atol=2**-7, rtol=2**-7),
        'U': dict(atol=2**-7, rtol=2**-7),
        'A': dict(atol=2**-133, rtol=2**-7, max_bf16_steps=1, exact_reference_zero=True),
        'S': dict(atol=0, rtol=0, exact_bits=True),
        'D': dict(atol=2**-7, rtol=2**-7),
        'Y': dict(atol=0, rtol=0, exact_bits=True),
    },
    composition={'D': dict(atol=2**-6, rtol=2**-6),
                 'Y': dict(atol=2**-5, rtol=2**-5)},
    silu_tail='Match pinned FP32-exp SiLU: G<=-89 yields negative zero; no FP64-tail substitution',
    zero_subnormal='Exact sign when reference is zero; otherwise at most one BF16 step with the stated abs/rel gate',
)


def specification():
    return dict(version=VERSION, development=DEVELOPMENT, holdout=HOLDOUT,
                holdout_prompt=HOLDOUT_PROMPT, mutations=MUTATIONS, up_values=UP_VALUES,
                storage='BF16; arrays stored losslessly as FP32',
                arithmetic='FP32 reductions and SiLU; BF16 N/G/U/A/S/D/Y boundaries',
                reference='Qwen2MLP and Qwen2RMSNorm; Torch 2.4.0 CPU eager; Transformers 4.43.1',
                recipe='PCG64 SeedSequence([seed,H,I,tag]); FP64 normal; scale then FP32 then BF16',
                chunks='single rows for R<=17; [R-18,17,1] otherwise',
                budgets=BUDGETS, holdout_outputs_observed=False)
