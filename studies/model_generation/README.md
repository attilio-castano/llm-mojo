The active [Fast runtime plan](../../docs/fast-generation-plan.md) now treats
full-model numerical differences as diagnostics while retaining exact
implementation invariants. The failed qualifications below remain historical
evidence and do not gate the new implementation effort.

# Full-model reference schedule qualification

The [Fast reference qualification](fast-reference.md) is the latest checkpoint.
It failed its predeclared intermediate-error ceilings during reference-only
calibration; confirmation and native Fast acceptance did not run. This README
retains the earlier schedule-qualification result below.

**Historical reference confirmation failed.** The subsequently authorized
[consistency implementation](consistency.md) qualifies a canonical reference
through 4096 tokens and passes native primitive/layer consistency checks. Its
first native full-model accuracy case fails seven unchanged gates. Full-model
schedule acceptance, generation and performance promotion remain pending.
Final reserved inputs remain unopened. The original eight-failure reference
confirmation below remains historical evidence, with unchanged budgets.

The subsequent [rounding investigation](rounding.md) explains the first
attention difference, its propagation, and the actual failing normalization
coordinate. All 66 declared next-token comparisons and all six bounded cached
greedy trajectories agree with full recomputation. That diagnostic evidence
informs a review of the acceptance criteria; the original failed gate and its
records remain unchanged.

The deeper [HF/PyTorch investigation](backend.md) localizes the first difference
to QK matrix multiplication and demonstrates a stable diagnostic invocation:
one contiguous query and its causal prefix yield 36,225 byte-equal model
boundary comparisons across five declared lengths. This does not replace the
original reference policy or its failed gate.

## Question and contract

Can a full-prefill reference and a cached reference agree within a qualified
numerical budget across all 24 Qwen layers, before comparing Mojo?

The checkpoint is Qwen2.5-0.5B-Instruct at revision
`7ae557604adf67be50417f59c2c2f167def9a775`, with full safetensors SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
The upstream executes on CPU with one thread, Torch 2.4.0, Transformers 4.43.1
and NumPy 1.26.4. Hardware is Apple M4 Pro, macOS 26.6.2. These are numerical
observations, with no latency or GPU-performance claim.

Weights and stored boundaries are BF16. SDPA uses FP32 Q/K/V/masks with the
Torch math backend, then rounds its result to BF16. The real upstream model
executes embeddings, all decoder layers and final norm; its tied head produces
BF16 logits exposed as FP32. Each call checks 75 boundaries: 25 hidden states,
final norm, logits, and keys/values for all 24 layer caches. Hidden tensors are
row-major `[R,896]`, caches are recorded `[T,2,64]`, and next-token logits are
`[1,151936]`. Explicit causal masks and positions account for the cached prefix.

The initial qualification compared full and scheduled upstream execution at
lengths 1, 17, 65 and 257 with `atol=0.0625, rtol=0.03125`. It failed 130 of
1,800 checks. That draft was not an accepted full-model contract; its original
report is retained losslessly. No Mojo model output was examined.

## Bounded calibration and independent confirmation

The [declaration](../../tests/fixtures/model_calibration.json) and
[runner](../../tests/fixtures/model_calibration.py) were committed at `5993f50`
before execution. Calibration uses lengths 1, 17, 65, 257 and 1024, seed
`9103 + length`. Confirmation uses lengths 15, 33, 129, 1025 and 4096, seed
`9133 + length`. IDs come from NumPy PCG64, uniformly in `[0,151643)`.
Lengths at most 17 use one-token calls; others use `[length-17,16,1]`.

Calibration fixes `rtol=1/32` and derives one absolute budget for each boundary
from `max(abs(actual-expected) - rtol*abs(expected))`, with a 1.5 margin and
rounding upward to multiples of 1/32. A second check bounds the relative L2
error **of every token row**, flattening the head dimensions for cache rows.
Those budgets receive the same margin, round upward to 1/256, and may not
exceed 6.25%. Embeddings must be exact. Both checks must pass: low average
error cannot excuse a failing element. Budgets are written and hashed before
any confirmation output is observed.

All 2,025 calibration checks pass the resulting budgets. Independent
confirmation fails eight of 2,025 checks:

| Confirmation length | Checks | Pointwise failures | Row relative-RMS failures | Largest row relative RMS |
| ---: | ---: | ---: | ---: | ---: |
| 15 | 1,125 | 8 | 0 | 3.432% |
| 33 | 225 | 0 | 0 | 2.494% |
| 129 | 225 | 0 | 0 | 2.292% |
| 1025 | 225 | 0 | 0 | 2.228% |
| 4096 | 225 | 0 | 0 | 2.728% |

The failing boundaries are hidden states 22–24 and final norm in the 15-token
case. At its second consumed token, final norm has maximum absolute error 5.0
and requires `atol=3.76953125` after accounting for `rtol`; its frozen budget
is `2.40625`. Its row relative RMS is 3.408%, within its 4.297% budget. This is
localized pointwise drift, not failure of the aggregate error check. Passing
the largest context does not establish the short-context contract.

## Diagnosis of execution shape

A separate [reproducer](../../tests/fixtures/model_reference_diagnosis.py),
committed at `311ce4d`, repeats the previously inspected 17-token input
(`PCG64(9120)`). Each arm executes full prefill twice and cached one-token calls.
All 75 captured boundaries are bitwise equal between repeated full executions.
The table compares the last token and complete active caches across schedules.

| Reference arm | Changed boundaries | Last hidden relative RMS | Logit relative RMS | Maximum logit error |
| --- | ---: | ---: | ---: | ---: |
| Original | 72 | 2.310% | 2.246% | 0.317139 |
| Rowwise linear | 72 | 2.310% | 2.246% | 0.317139 |
| Rowwise RMSNorm | 72 | 2.310% | 2.246% | 0.317139 |
| Query-by-query SDPA over each causal prefix | 0 | 0% | 0% | 0 |

For this input, the first block already differs by up to 0.0078125; the final
hidden state differs by up to 0.2421875. Changing linear/norm row scheduling
does not alter these results. Changing SDPA query/key execution extents removes
the observed difference. This isolates execution shape within SDPA as the
source of variation in this ablation, consistent with FP32 reduction-order
differences crossing BF16 rounding boundaries and propagating through layers.
It does not establish the exact internal rounding mechanism or qualify a
replacement reference. Querywise SDPA has not been adopted as the oracle.

## Consequence and next decision

The [approved stop rule](../../docs/generation-plan.md) applies: independent
confirmation failed, so no native model numerical comparison, reserved
acceptance, model benchmark or automatic configuration promotion follows.
The three additional optimization studies remain unspent. `auto` still uses
decoder configuration 0; existing per-layer choices remain explicit candidates.

The next bounded study should establish the desired full-model numerical
contract: how schedule-dependent BF16 rounding is assessed, which intermediate
outliers are acceptable, and how next-token margin and generation agreement
constrain that policy. The present evidence cannot justify simply increasing
the failed thresholds. Any revised policy needs a new declared confirmation
set and explicit approval before this milestone resumes. Existing component
contracts remain unchanged.

## Evidence and reproduction

`reference-study.json` binds all retained raw records and their original uncompressed bytes.
The initial qualification, 4,050 calibration/confirmation observations and new
diagnosis are compressed losslessly. Frozen budgets and the failed decision are
readable JSON. Together the five evidence files occupy about 86 KB. The source
anchors identify committed bytes; the reports additionally record source,
declaration, checkpoint and pinned upstream implementation hashes. The earlier
temporary diagnostic reports are superseded by this committed-source rerun.

Regenerate both CSV tables without weights or model execution:

```sh
uv run --locked python studies/model_generation/summarize.py
```

The exact model-execution commands are recorded in `reference-study.json`. Run them from
the listed source revisions with the pinned checkpoint available, using fresh
output directories: the reference tools refuse to overwrite evidence. The
initial qualification and calibration commands are expected to exit nonzero
because their numerical gates fail. The diagnosis command records observations
without changing qualification. Arrays, weights, binaries and temporary logs
remain outside Git.
