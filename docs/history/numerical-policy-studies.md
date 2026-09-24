# Numerical-policy studies before Fast

> **Historical record**, kept as written. The [history index](README.md) says what it
> led to and where current guidance lives.

Moved from the [runtime guide](../generation.md) on 2026-09-23.

## Historical numerical-policy studies

Before the diagnostic policy, the [Fast completion effort](fast-generation-plan.md) stopped during
reference-only qualification: the declared intermediate-error ceilings failed
before independent confirmation or native Fast acceptance. The
[Fast reference study](../../studies/model_generation/fast-reference.md) retains
the complete calibration, diagnosis and stop decision. The earlier native
configuration-20 accuracy failure below remains separate historical evidence.

**Historical consistency candidate: promotion paused at native full-model accuracy.**
The authorized [consistency revision](../../studies/model_generation/consistency.md)
passes 71,250 exact canonical HF comparisons through 4096 tokens and native
primitive/layer schedule tests. Its first full-model input fails seven frozen
accuracy gates. All 336 identical-operand operation checks pass; ten projection
elements differ by one BF16 step. Full-model schedule and generation acceptance
remain pending, and configuration 20 is not automatically selected.

The native model and generation call graphs compile. The
embedding/copy and BF16 binary-I/O tests pass on Apple M4 Pro / Metal. The
upstream observation code passes a tiny synthetic 24-layer self-test; that is
not qualification of the pinned checkpoint. The full checkpoint and all 196
prepared tensors have since passed hash/extent verification. Reference-only
calibration completed, but independent confirmation failed 8 of 2,025 checks.
The [retained study](../../studies/model_generation/reference-qualification.md) records the frozen
budgets and diagnosis. That original confirmation remains failed; the approved
consistency revision and native failure are recorded separately.

The authorized [reference-only follow-up](../../studies/model_generation/rounding.md)
traced the discrepancy to FP32 attention differences crossing BF16 rounding
boundaries and propagating through the model. It reproduced the normalization
gate violation while observing identical argmax in 66 comparisons and matching
bounded greedy sequences on three declared prompts. These results motivate
separating corresponding-mode comparisons from cross-schedule diagnostics;
they do not establish Mojo model acceptance or change the frozen thresholds.

The deeper [HF/PyTorch study](../../studies/model_generation/backend.md) locates the
first difference in QK matrix multiplication. Normalizing SDPA query shape and
causal-prefix layout yields 36,225 byte-equal full/cached comparisons at five
declared lengths, while deterministic mode alone leaves the original differences
unchanged. The approved consistency revision now qualifies this canonical
route while preserving the original failed policy and evidence.

The approved scope and stop gates are in [generation-plan.md](generation-plan.md).
The declaration is [model_contract.json](../../tests/fixtures/model_contract.json).
The numerical thresholds are initial reference-only qualification criteria;
they failed, and a bounded reference-only calibration also failed independent
confirmation. Both declarations and results remain intact. The new
[consistency declaration](../../tests/fixtures/model_consistency.json) independently
requires exact schedule agreement and adopts the unchanged frozen budgets as
cross-engine hypotheses. The first native accuracy failure is under that new
declaration, not acceptance under the historical failed qualification.
