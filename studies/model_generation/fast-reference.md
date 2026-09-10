# Fast reference qualification: accuracy stop

**The approved Fast qualification stopped during reference-only calibration.**
The declared intermediate-error ceilings do not cover the measured arithmetic
variation. Independent confirmation, native Fast acceptance, model selection
and generation benchmarks did not run. The original reserved model outputs
remain unopened. No numerical limit or inference arithmetic was changed in
response to the result.

The [approved plan](../../docs/fast-generation-plan.md) and
[declaration](../../tests/fixtures/model_fast.json) were committed at `ce1a8e6`
before collection. This is a new failed qualification, separate from the older
HF schedule confirmation and native configuration-20 failures. It establishes
neither a native Fast defect nor accepted full-model generation.

## What was compared

All arms execute the real pinned Qwen checkpoint through Torch 2.4.0 and
Transformers 4.43.1, NumPy 1.26.4, on one M4 Pro CPU thread. Weights and stored
boundaries are BF16. The control uses the existing wrapper's FP32 math SDPA,
with BF16 output. These are numerical observations, not GPU timing results.

Two arms compare against control on identical input histories and call schedules:

- `hf_canonical`: normalize each attention query and its causal-prefix layout
  using the already implemented reference invocation.
- `fp32_split2`: replace each affine projection with two contiguous K-partition
  FP32 products, add their FP32 partials and promoted bias, then round once to
  BF16. All other HF operations remain. Every call verifies coverage of all 168
  decoder projections and the tied head. This changes reduction association,
  not the affine equation or stored output dtype.

Calibration covers eight synthetic lengths (1,16,17,64,65,256,257,1024) and
three plain-text cases, with full and cached execution. It retains all 12,300
boundary checks, including 164 next-token-logit comparisons. Counts describe
observations, not independent test cases. The declared confirmation cases,
including length 4096, were not executed.

## Why qualification stopped

Budgets pool calibration maxima by boundary role, multiply by the frozen 1.5
margin, then round upward to declared quanta and floors. Every derived row-L2
budget must remain at or below 6.25%; role-specific absolute ceilings also
apply. These were engineering acceptance limits, not theoretical BF16 bounds.
All five nonembedding roles exceeded the row-L2 ceiling after derivation.

| Boundary role | Maximum observed row L2 error | Derived budget | Ceiling |
| --- | ---: | ---: | ---: |
| Hidden states | 14.425% | 21.875% | 6.25% |
| Keys | 5.259% | 8.203% | 6.25% |
| Values | 9.000% | 13.672% | 6.25% |
| Final normalization | 7.182% | 10.938% | 6.25% |
| Logits | 5.266% | 8.203% | 6.25% |

Hidden states also required a derived absolute allowance of 10.5, exceeding
their fixed ceiling of 2.0. The complete values are regenerated in
[the budget table](fast-reference-budgets.csv).

The largest observed row difference is at hidden boundary 22 for the full
64-token synthetic input, in the split-projection arm. The canonical-attention
arm alone also reaches 7.802% somewhere in its intermediate boundaries.
Therefore, the stop is not solely an effect of adding the split-projection arm.

There are zero failures against the *unqualified derived budgets*: the 1.5
calibration margin covers the observed samples by construction. This does not
make calibration pass. The independently declared ceilings reject those
budgets, which is the decisive stop recorded in the result and replay.

## Prediction evidence has a narrower result

All 164 comparisons pass the fixed prediction-distribution ceilings:

- maximum KL(reference || alternate): 0.003324 nats, below 0.015625;
- maximum total variation: 0.033534, below 0.0625;
- 161 of 164 greedy choices agree. All three differences occur where the
  control's top two logits are exactly tied, with zero margin.

These are same-history next-token comparisons. They do not establish matching
free-running generation, model quality, or acceptable behavior on unseen cases.
Passing them does not waive the intermediate gate. The
[prediction table](fast-reference-predictions.csv) retains every comparison.

## Bounded diagnosis of the exposed maximum

The diagnosis reruns only the already exposed 64-token input and reproduces
all 75 recorded numerical boundary metrics. Pass-through observation leaves
the control's 75 boundaries byte-identical to unobserved execution.

Each of the 169 affine operations then receives its original HF operands.
163 operations have changed elements, totaling 3,423 elements. The largest
local row-relative L2 difference is 0.06068%, and the largest absolute
difference is 0.03125. These measurements do not constitute a new operation
acceptance policy; the [operation table](fast-reference-operations.csv) gives
their complete scope.

Exact rational dot sums inspect the first changed coordinate in each of the
first 16 differing affine operations. Eleven favor the HF result and five the
split result. Neither arithmetic route is uniformly closer in this bounded
sample. The small local differences and reproduced full-model amplification
support accumulated projection roundoff as the explanation on this input;
they do not prove the absence of every possible implementation defect.

## Consequence

The current policy cannot be promoted. A further decision must choose whether
to retain strict intermediate fidelity and constrain arithmetic accordingly,
or adopt a different full-model acceptance objective that gives prediction
preservation a larger role while retaining exact semantics/cache invariants
and operation-level accuracy. This run does not authorize either change.
Simply enlarging limits to fit these exposed results would not provide new
independent acceptance evidence.

## Evidence and reproduction

`fast-reference-study.json` binds five lossless compressed files, totaling
216,761 bytes: declaration, frozen budgets, result, all observations, and the
bounded diagnosis. Source and upstream hashes, input IDs, artifact identities
and budget-freeze timing are retained. Arrays, weights and full validation logs
remain under ignored build storage.

Regenerate the tables and independently replay the stop decision, required
schedule census, budget derivation, diagnosis coverage and exact-dot rankings:

```sh
uv run --locked python studies/model_generation/summarize.py
```

The collection command at the frozen source was:

```sh
uv run --locked --script tests/fixtures/model_calibration.py --fast --output build/fast-reference-qualification
```

It records the failed ceiling decision and exits nonzero. To reproduce, first
prepare the hash-pinned checkpoint/tokenizer artifacts in this checkout, use
the frozen source or verify its source bindings, and choose a new output path.
Repeating these now-exposed inputs is regression, not fresh confirmation.

The bounded diagnosis command is:

```sh
uv run --locked --script tests/fixtures/model_reference_diagnosis.py --fast-affine build/fast-reference-qualification/result.json --output build/fast-reference-diagnosis.json
```

Before collection the complete repository validation passed 136 Python tests,
all native suites and benchmark smoke routes on Metal, and both tokenizer parity
runs. Eight reference self-tests passed separately. The new replay regressions
reject omitted schedules even with updated hashes, altered derived budgets,
and a false passing status; the final Python suite passes 138 tests. Repository
validation is distinct from the failed
scientific acceptance gate above.
