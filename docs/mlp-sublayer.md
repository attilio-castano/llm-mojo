# MLP numerical contract and upstream fixture specification

Status: upstream reference package frozen; materialized Mojo MLP implemented and
undergoing final acceptance. The initial budgets were selected before any Mojo
MLP or holdout output. Reference results below retain their original CPU scope.
The [MLP study](../studies/mlp_sublayer/README.md) records implementation and measurement progress.

## Scope and data flow

The target is the second sublayer of a batch-one Qwen2.5-0.5B-Instruct decoder
layer: post-attention RMSNorm, SwiGLU MLP, and residual addition. Follow the
pinned model revision and asset identities in [model.md](model.md).

`X[R,H]` is the output of the attention sublayer **after its residual addition**.
For the target, `H=896`, intermediate width `I=4864`, and `1 <= R <= 4096`.
Each row is independent. There is no KV cache, position offset, mask, or direct
dependence on the number of previously cached tokens inside this sublayer.

All tensors are contiguous row-major. Weights use the upstream
`[output_features,input_features]` orientation:

| Input | Shape | Checkpoint tensor suffix |
| --- | --- | --- |
| Norm weight | `[H]` | `post_attention_layernorm.weight` |
| Gate weight | `[I,H]` | `mlp.gate_proj.weight` |
| Up weight | `[I,H]` | `mlp.up_proj.weight` |
| Down weight | `[H,I]` | `mlp.down_proj.weight` |

Prefix these suffixes with `model.layers.<layer_index>.` for checkpoint lookup.
All three projections are bias-free. Gate and up are distinct weight matrices;
their identities and order must survive any future packing.

## Arithmetic and rounding

Let `B` mean round-to-nearest, ties-to-even BF16 storage and `F` mean promotion
of that stored value to FP32. The baseline materializes each named boundary:

```text
N[R,H] = existing Qwen RMSNorm(X, norm_weight, epsilon=1e-6)
G[R,I] = B(F(N) @ F(Wgate).T)
U[R,I] = B(F(N) @ F(Wup).T)
A[R,I] = B(F(G) / (1.0f32 + exp(-F(G))))
S[R,I] = B(F(A) * F(U))
D[R,H] = B(F(S) @ F(Wdown).T)
Y[R,H] = B(F(X) + F(D))
```

Matrix products accumulate in FP32. This specifies precision, not a particular
parallel reduction order or bitwise agreement with a CPU matrix library.
RMSNorm retains the existing [rounding contract](model.md#rmsnorm-arithmetic):
normalize in FP32, round to BF16 before multiplying by the BF16 norm weight,
and round that product to BF16.

The actual [pinned Qwen MLP](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py#L137-L147)
applies SiLU to the gate projection, multiplies by the up projection, then
applies the down projection. Pinned [Torch 2.4 CPU SiLU](https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/cpu/Activation.cpp#L1187-L1201)
uses float arithmetic internally for BF16 input and returns BF16. There is no
separate BF16 sigmoid boundary. `A` must round before multiplication by `U`;
`D` must round before the residual addition. Keeping either unrounded value
through the following operation defines a different numerical policy.

Compiler contraction or reassociation must not erase these boundaries. A later
single-dispatch SiLU/multiply kernel may retain the rounding without retaining
the intermediate allocation, but requires its own correctness evidence.
The baseline uses separate SiLU and multiplication outputs for inspection.

Finite reference outputs are required for ordinary fixtures. The isolated
activation sweep additionally characterizes signed zero, subnormals, and
exponential overflow/underflow on finite BF16 inputs. Record these cases
separately; do not silently exclude them or assume CPU and Metal handle them
identically. NaN and infinity inputs are outside the initial inference-input
contract. Their exclusion does not permit nonfinite outputs on finite ordinary
fixtures. Any supported-domain restriction needs an explicit contract revision.

## Reference authority and captured arrays

Run the actual `Qwen2MLP` and `Qwen2RMSNorm` from Transformers 4.43.1 with
Torch 2.4.0, NumPy 1.26.4, Python 3.12, CPU eager execution, one Torch thread,
evaluation mode, and gradients disabled. Resolve this environment through the
existing locked `tests/fixtures/generate.py` script environment. Do not upgrade
dependencies for this slice. Set dimensions and `hidden_act="silu"` explicitly.

Capture `X, N, G, U, A, S, D, Y` and the four weight arrays. Observe the actual
module calls: projection hooks capture `G`, `U`, and `D`; a down-projection
pre-hook captures `S`; an observing wrapper calls the original SiLU and captures
`A` without replacing its arithmetic (the implementation uses a forward hook on
the actual upstream SiLU module). Record the norm input/output and residual
operands. Require that the captured tensors have the declared dtypes/shapes and
that observation leaves the module output bitwise unchanged. Reconstructing the
formula in NumPy is not a substitute for running upstream.

Keep two independent diagnostic calculations on the **same stored BF16 inputs**:

- Stage-local FP64 reductions/SiLU with the same materialized BF16 boundaries,
  to locate sensitivity to reduction order or transcendental evaluation.
- An FP64 composed calculation with those same boundaries, to understand how
  discrepancies propagate through gating and the down projection.

Use an explicitly checked BF16 rounding routine for diagnostics, with signed
zero and tie cases. Avoid an unintended FP64-to-FP32-to-BF16 double rounding
when labeling a result as directly rounded from FP64. These calculations help
explain errors; neither defines the CPU backend's exact reduction order or
Qwen's training arithmetic.

For checkpoint fixtures, compute layer-0 attention using the already selected
FP32 attention reference policy, then capture its post-residual output as `X`.
Use the checkpoint's layer-0 post-attention norm and MLP weights. Reuse the
existing pinned prompts for development. This is a reference-policy-specific
layer-0 fixture, not unmodified BF16 eager full-model evidence. The earlier
attention-only checkpoint prefix must not be assumed to contain complete MLP
tensors: verify tensor ranges and source identity before extraction. Downloading
additional assets is a separate explicit action; missing assets must be reported.

## Fixture matrix

Declare recipes, seeds, shapes, and held-out prompts before any Mojo MLP output
is observed. Generate one case at a time to bound memory. All rows in a fixture
use the same weights; chunking slices existing rows without regenerating them.

| Family | Specification | Purpose |
| --- | --- | --- |
| Tiny | `(H,I)=(8,12),(7,11)`; `R=1,7,17`; development seed 1601 | Inspectable intermediates, unequal dimensions, ragged extents |
| Qwen development | `(896,4864)`; `R=1,7,15,16,17,33,65,257,1024,4096`; seed 1601; additionally `R=1,17,4096` with seed 1613 | Real reduction lengths, decode, tile boundaries, larger prefill |
| Qwen holdout | `(896,4864)`; `R=1,17,4096`; seeds 2017 and 2027 | Previously unobserved cases under frozen gates |
| Activation sweep | Every finite BF16 bit pattern, in ascending unsigned-bit order | SiLU rounding, signed zero, subnormals, and extreme finite inputs |
| Structured stress | Tiny shapes and Qwen `R=17`; recipes below | Cancellation, scale, layout and rounding defects |
| Checkpoint development | Layer 0; existing three attention prompts and their retained token IDs | Actual weights and post-attention activation distributions |
| Checkpoint holdout | Layer 0; explicit system message `You are a helpful assistant.`; user prompt `Explain why multiplying two negative numbers gives a positive number, using a numerical example.` | New prompt under the same checkpoint and reference policy |

For the ordinary synthetic families, use NumPy 1.26.4
`Generator(PCG64(SeedSequence([seed, H, I, tag])))`, with separate tags
`0=X, 1=norm, 2=gate, 3=up, 4=down`. Draw FP64 standard normals in row-major
order, then convert through FP32 to BF16 using Torch (this conversion is part
of the **input recipe**, not the FP64 diagnostic rounding rule). Set `X=Z`,
`norm=1+Z/64`, `Wgate=Z/sqrt(H)`, `Wup=Z/sqrt(H)`, and
`Wdown=Z/sqrt(I)` before conversion. Independent streams avoid accidental
gate/up correlation. Reuse a fixed seed's prefix when comparing row counts.

Structured stress includes zero input, zero gate weight, zero up weight, and
zero down weight, each as a separate mutation of development seed 1601.
Also test gate/up weight swaps that must be distinguishable, and input scales
`1/64` and `16`. For projection cancellation, set norm weights to one, input
column pairs to `(1,-1)`, and each gate/up weight column pair to equal values
copied from its first column in the development recipe. Set an odd trailing
input column to zero. Separately test isolated down projection with `S=1` and
each weight row repeating `(1/32,-1/32)`, with an odd trailing coefficient zero.
Retain compact operand-level regressions where removing `B(A)` or `B(D)`
changes the result; those regressions must reject an implementation that skips
the intended boundary even if aggregate output tolerances would permit it.

For gating, combine the finite SiLU sweep with bounded up values
`{-16,-1,-1/64,0,1/64,1,16}` where the reference product is finite. Report
overflow cases explicitly as outside that ordinary finite-output matrix.
This probes amplification of SiLU error without treating an overflowing product
as a valid finite inference fixture.

Do not inspect holdout outputs during characterization or threshold selection.
Reusing old attention prompts provides continuity, not a new MLP prompt holdout.
Freeze rendered token IDs and their hashes for the new prompt before observing
its MLP outputs. Follow the pinned tokenizer/chat-template rules in model.md.

## Acceptance boundaries and numerical budgets

Operation tests must feed each Mojo operation the exact upstream tensors that
operation consumed. For example, down projection consumes captured upstream
`S`, and gating consumes captured upstream `A` and `U`. This isolates local
error from discrepancies inherited from an earlier stage. A combined SiLU/gating
candidate must also be tested from upstream `G` and `U`.

Composition tests feed original `X` through all stages. Require separate passing
gates for `D` and `Y`; a large residual must not conceal an incorrect branch.
Report every intermediate discrepancy. A passing composition gate cannot waive
a failing operation gate. Compare full-row execution with repeated single rows
for `R<=17`; for larger cases, use chunks `[R-18,17,1]`. Characterize this
comparison upstream as well: row independence does not guarantee identical
matrix-library reduction order across different row counts.

Use `abs(actual-reference) <= atol + rtol*abs(reference)` for approximate gates.
Record max absolute error, max scaled error, failing-element count, and the
index/operands of the worst failure. Define scaled error explicitly as
`abs(actual-reference)/(1+abs(reference))`; it is not relative error. Include
BF16 representable-step distances for isolated SiLU to expose errors near zero
that a broad absolute tolerance could hide.

| Boundary | Acceptance rule to establish |
| --- | --- |
| Shape, dtype, weight identity, copies, untouched guards | Exact; no tolerance |
| Isolated BF16 multiply and residual on identical ordinary operands | Exact BF16 result; signed-zero/subnormal behavior additionally checked and recorded |
| RMSNorm | Preserve its existing operation contract and gate |
| Gate/up/down projections | Separate declared absolute/relative budgets at the new reduction lengths |
| Isolated SiLU | Declared absolute/relative and representable-step budgets, including an explicit subnormal/zero rule |
| Composed branch `D` and final `Y` | Separate declared absolute/relative budgets |
| Full/chunked execution | Declared composition budgets; intermediate differences reported |

The initial budgets are recorded in the
[executable contract](../tests/fixtures/mlp/contract.py). They are acceptance
hypotheses for later GPU validation, not measured GPU bounds:

| Boundary | atol | rtol | Additional requirement |
| --- | ---: | ---: | --- |
| Isolated RMSNorm, gate/up/down projections | `2^-7` | `2^-7` | RMSNorm retains its existing gate |
| Isolated SiLU | `2^-133` | `2^-7` | At most one BF16 representable step; reference zero requires identical bits |
| Isolated multiply and residual | 0 | 0 | Identical BF16 bits on identical operands |
| Composed branch `D` | `2^-6` | `2^-6` | Independent from final-output gate |
| Composed final `Y` | `2^-5` | `2^-5` | Both branch and final must pass |

For normal BF16 values, relative spacing is at most `2^-7`; the projection gate couples that
scale with an absolute allowance near cancellation. Development stage-local
max scaled error is 0.00518135 across projections, below `2^-7=0.0078125`.
For composition, the selected branch and final gates allow respectively two
and four such scaled units: the branch accumulates projection and gating error,
and the final output can suffer cancellation with the residual. Development
maxima are 0.00917431 for `D` and 0.01481482 for `Y`. These observations motivate
the initial budgets but do not prove arbitrary-input or future GPU bounds.
Exact boundary regressions are additional requirements because an aggregate
tolerance can admit a skipped rounding step.

SiLU is stricter than a generic absolute `2^-7` gate: that allowance would hide
large errors near zero. The absolute allowance is one minimum BF16 subnormal
(`2^-133`), and the representable-step limit applies across the finite sweep.
Signed zero is exact when upstream returns zero. For finite BF16 `G<=-89`,
the pinned FP32-exponential implementation returns negative zero; the baseline
must preserve this declared behavior. No value range has been removed from the
sweep. FP64 negative-tail results remain diagnostic and are not silently used
as replacement expected outputs.

The reference-only selection sequence was:

1. Execute the development fixtures, actual upstream full/chunked paths, and
   stage-local/composed FP64 diagnostics; retain all discrepancies.
2. Explain the observed reduction, exponential, and BF16 midpoint mechanisms.
   Select and justify the initial budgets and the extreme-value policy without
   observing Mojo MLP outputs or the holdout outputs. A reference comparison
   alone cannot guarantee the future GPU error bound.
3. Freeze the budgets, exact recipes, capture validation, source/array hashes,
   and held-out case list in a versioned manifest before evaluating the baseline.

A failed holdout remains failed. Preserve it, diagnose the mechanism, and seek
an explicit numerical-policy decision if a change is needed. Any revised policy
requires fresh declared holdouts. Do not widen thresholds repeatedly or relabel
compatibility failures as passes. Until budgets are frozen and all required
gates pass, there is no accepted MLP baseline and profiling does not begin.

## Autonomous execution plan: reference readiness

This plan governs the reference-readiness work recorded below. The run ends with a reproducible
upstream reference package and frozen numerical acceptance rules, ready for
Mojo baseline implementation. It follows the request to establish the numerical
contract and fixture specification first.

Work locally on `codex/swiglu-correctness-baseline`. Implement the Python fixture
tooling, its necessary tests, and documentation; run the pinned CPU reference
and diagnostics using the existing locked environment. Routine implementation
and fixture-tooling repairs can proceed without intermediate approval. Existing
Mojo numerical contracts and dependency locks remain fixed. Commits, pushes,
PR creation, new checkpoint downloads, GPU MLP implementation, and profiling
are outside this run's scope.

### 1. Establish reproducible capture

Inspect local checkpoint assets read-only, validate their provenance, and check
whether complete layer-0 MLP tensors are available. Continue synthetic work if
they are absent; record checkpoint validation as pending rather than substituting
synthetic weights. Add an MLP generator to the existing fixture dispatcher.
Execute actual pinned upstream modules and capture the declared boundaries.

Exit evidence: tiny development cases have the expected shapes/dtypes, the
captured data flow is consistent, and adding observation leaves upstream output
unchanged. The manifest identifies sources, dependencies, and fixture inputs.

### 2. Characterize development numerics

Implement and test the independent diagnostic rounding and FP64 calculations.
Run the declared synthetic development matrix, structured stress cases,
activation/gating sweep, and upstream full/chunked comparisons. Add available
checkpoint development cases with post-attention inputs under the selected
attention policy. Record all discrepancies, including signed zero, subnormal,
and overflow cases. Establish small boundary regressions that distinguish the
specified computation from one that skips an intended BF16 rounding step.

Exit evidence: a compact report identifies the worst differences, their
operands, and the supported explanations; the capture and diagnostic tools have
independent checks. Holdout outputs remain unobserved.

### 3. Freeze the contract and acceptance rules

Choose and explain the initial stage budgets from development evidence and
numerical reasoning, preserving the specified BF16 boundaries. Define the
SiLU representable-step and zero/subnormal rules. Initial budget selection may
proceed autonomously; weakening existing gates, changing arithmetic, or narrowing
the supported input domain requires an explicit numerical-policy decision.
Record the evidence as a hypothesis for later GPU validation, not a GPU bound.

Freeze contract version, recipes, development array hashes, the holdout case
list, and available checkpoint token IDs. Verify that the proposed gates reject
the deliberately incorrect rounding/data-flow controls. Preserve boundary
regressions even when aggregate tolerances would accept a changed computation.

Exit evidence: every acceptance rule has a concrete value or exact predicate,
provenance, and rationale. No gate is chosen from Mojo results or holdout outputs.

### 4. Verify and hand off

Regenerate the development arrays once and verify their frozen hashes, capture
invariants, and diagnostic results. Add meaningful tests for the rounding oracle,
capture instrumentation, manifest validation, and refusal to overwrite frozen
anchors silently. Integrate reference preparation with the existing validation
workflow, then run `uv run --locked llm-mojo-validate` to check repository
regressions. Keep holdout execution explicit for the later baseline acceptance;
ordinary reference preparation must not silently consume it.

Exit evidence: reproducible commands, passing relevant checks, compact retained
results, and an updated contract identifying the exact completed matrix. The
handoff names checkpoint evidence still pending and makes no Mojo MLP
correctness or performance claim.

### Stop conditions and progress

Diagnose ordinary tooling defects and continue within the fixed contract.
If evidence requires changing arithmetic, relaxing an exact requirement,
narrowing the supported domain, or selecting budgets without a defensible
explanation, preserve a minimal reproducer and stop the dependent phase for a
decision. Continue independent checks where useful. A missing checkpoint asset
blocks checkpoint acceptance, not the synthetic reference work; acquiring new
assets requires separate authorization. Resource pressure calls for processing
smaller batches of the same declared cases, not dropping coverage silently.

Report progress at each exit criterion and any material failure. Finish with the
changed files, exact validation coverage, unresolved decisions, and the command
sequence for reproduction. Once these criteria are met, end the run; the next
milestone is the Mojo correctness baseline described below.

## Ownership, provenance, and implementation follow-through

The intended Mojo baseline reuses RMSNorm, bias-free rowwise linear, and residual
operations. New SiLU and gating operations have host references and separate
GPU dispatches. No projection-tile selection or fusion experiment is part of
this baseline. The caller owns weights, input, and reusable workspace; allocation
and uploads occur before enqueue. The input must not overlap writable workspace.
Enqueue validates positive dimensions, row capacity, compatible shapes, and
Metal before launching any work; it allocates and synchronizes nothing. All
resources remain alive on the same ordered stream until completion, and output
is consumed before the next invocation overwrites it.

Future tests must verify poisoned outputs are completely written, inactive
workspace/guard regions remain untouched, and rejected calls leave buffers
unchanged. Repeated asynchronous calls require a check of each call's output
(copy to retained test storage before reuse), with a final synchronization.
Also run in normal mode: debug synchronization can hide lifetime/order defects.
Require runtime device name and `ctx.api()=="metal"` in GPU validation evidence.

Extend the existing fixture dispatcher and validation workflow when implementing
the generator. Generated arrays belong in ignored `build/oracle_data/`; add only
the generator, compact frozen manifests, tests, and explanation to Git. Arrays
may use FP32 `.npy` storage for exact BF16 values as the attention fixtures do;
record logical BF16 dtype and preserve signed-zero bits during conversion.
Record array shape, storage dtype, SHA-256, case/split identity, upstream module
source hash, generator hash, dependency versions, CPU/OS/backend/thread settings,
and model revision/tensor identities for checkpoint cases. Record the contract
revision, source commit, dirty state, and source hashes before and after any
validation run. Reject drift rather than assembling a mixed result.

The implementation milestone will run `uv run --locked llm-mojo-validate`,
including the new operation/composition suites, plus the normal-mode reuse
check. Synthetic acceptance and checkpoint acceptance must be reported
separately if assets are missing. Full decoder composition and model-level
logit/generation parity remain subsequent milestones.

This specification applies the attention study's lessons about
[reference authority, rounding defects, and failed holdouts](../studies/attention_sublayer/numerics.md).

## Reference results and reproduction

The development matrix contains 43 synthetic/stress cases and three checkpoint
cases (30, 41, and 4096 rows), plus the 65,280-value finite BF16 activation sweep,
seven gating multipliers, and isolated down-projection cancellation. Holdout
model outputs have not been generated. Checkpoint holdout token IDs are frozen
without running their attention or MLP outputs. The verified historical prefix
does contain all required layer-0 MLP tensors; no assets were downloaded.

| CPU reference comparison | Largest scaled error |
| --- | ---: |
| Isolated norm vs boundary-preserving FP64 | 0.00510204 |
| Isolated gate/up vs boundary-preserving FP64 | 0.00518135 |
| Isolated down vs boundary-preserving FP64 | 0.00492611 |
| Composed branch vs boundary-preserving FP64 | 0.00917431 |
| Composed final vs boundary-preserving FP64 | 0.01481482 |
| Checkpoint composed branch / final vs FP64 | 0.00390625 / 0.005 |
| Upstream full vs chunked, all captured boundaries | 0; identical bits |

On identical captured ordinary operands, SiLU, gating multiplication, and
residual addition agree exactly with the independent diagnostics. In the full
SiLU sweep, 17 inputs (`-89,-89.5,...,-97`) differ from the stable FP64 formula:
FP32 `exp(-G)` overflows, producing negative zero, whereas the FP64 diagnostic
still rounds to a nonzero BF16 result. The largest absolute difference is
`1.98364671701261e-37`. The other 65,263 finite inputs agree bitwise. Exact
BF16 multiplication also passes on every finite product of the gating sweep;
the record counts overflowing products explicitly.

A seven-term projection explains the tiny-shape discrepancy independently of
GPU arithmetic. The pinned ARM
[scalar tail](https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/BlasKernel.cpp#L502-L507)
evaluates its final products using the BF16 operand type before adding them to
a float accumulator. For the retained operands, the exact dot is
`1.2460365295410156`, which rounds to `1.2421875`; reproducing the CPU's rounded
tail yields its observed `1.25`. Explicit FP32 Torch linear agrees with the
FP64-rounded result. This is a named compatibility discrepancy, not evidence
to add BF16 product rounding to the intended Mojo projection contract.

Two regressions make the materialized boundaries visible. At `G=0.015625`
and `U=1.5`, rounding SiLU before the product gives `0.0118408203125`; omitting
that boundary gives `0.01177978515625`. A down accumulator of `1.00390625`
(constructible from BF16 products) with residual `-1` gives zero after the
required down rounding; omitting it gives `0.00390625`. A non-power-of-two
multiplier was deliberately added to the SiLU boundary probe, because the
power-of-two sweep multipliers usually commute with rounding away from
underflow. This probe is separate from the declared seven-value gating sweep.

The [generator](../tests/fixtures/mlp/generate.py) runs only development cases.
Its default mode verifies frozen source, contract, evidence, and array hashes;
it cannot overwrite anchors. `--characterize` emits ignored diagnostic records.
`--freeze` is only for initial anchor creation and refuses existing anchors.
The compressed `tests/fixtures/mlp/development.json.gz` retains every comparison,
array hash/shape, source identity, and checkpoint token list. Its adjacent
`checksums.json` hashes both compressed and uncompressed evidence. Generated
arrays are ignored under `build/oracle_data/mlp/`.

```sh
uv run --locked --script tests/fixtures/generate.py mlp -- --self-test
uv run --locked --script tests/fixtures/generate.py mlp
uv run --locked llm-mojo-validate
```

The full workflow regenerates and verifies the synthetic MLP references and runs
the reference-tooling tests. It never runs a holdout or downloads checkpoint
assets. To additionally reproduce checkpoint evidence, provide a local directory
containing the verified `model.attention-prefix.bin` and companion assets:

```sh
uv run --locked --script tests/fixtures/generate.py mlp -- --checkpoint-dir /absolute/path/to/verified/checkpoint
```

Without that option, checkpoint status is explicitly pending for the current
run while synthetic evidence is still verified. The frozen record retains the
completed checkpoint run. `build/oracle_data/mlp/last_run.json` records the
command, commit before/after, dirty state, and source hashes for the current run.
Reference agreement establishes a target for the next milestone; it does not
establish Mojo MLP or full-model correctness.

Validation on 2026-09-07 completed all four reference-readiness phases on the
local branch based on `7e0cc8e05353b5f804eb39dcc80c0dbc431f449e` (with uncommitted
changes identified by the frozen source hashes). The complete development and
checkpoint regeneration reproduced all 557 array hashes and diagnostics with
no generator-source or commit drift. `uv run --locked llm-mojo-validate` passed:
11 pinned reference-tooling tests, 54 Python tooling tests, 93 existing Mojo
tests on Apple M4 Pro/Metal, and every benchmark-route smoke check. The ordinary
MLP preparation also independently verified its 43-case synthetic subset.
That reference-readiness run generated no MLP holdout outputs, downloaded no
model assets, and introduced no Mojo MLP implementation or optimization.

## Approved implementation and baseline study

Initial fixture acceptance completed on 2026-09-07 at local source `afbe288`.
All 43 synthetic and three checkpoint development cases passed, followed by
the six declared synthetic holdouts and reserved checkpoint prompt on the
same frozen binary. All GPU full/chunked intermediate comparisons were
bit-exact. No numerical budget or supported input domain changed. See the
[MLP study](../studies/mlp_sublayer/README.md) for the retained checks and
the subsequent residual cutoff repair and baseline measurement results.

The next run is authorized for local implementation, validation, local commits,
Metal measurement and trace capture, and study documentation using existing
assets. Preserve this frozen reference package; new acceptance tooling records
its own identity and does not rewrite the reference anchors.

Implement and test materialized SiLU and multiply first, then compose RMSNorm,
rowwise gate/up projections, SiLU, multiply, down projection and residual using
caller-owned buffers. Accept operation and composition gates on all development
cases, followed by the previously unopened holdouts. Include full/chunked and
normal-mode asynchronous reuse, invalid-call, poison and guard tests.

After full validation and benchmark-route checks, commit the source and build
once from that clean identity. Whole-block timing uses R=1,7,15,16,17,33,65,257,
1024,4096 in hot and ring24 modes with the existing four-block, ten-warmup,
ten-sample self-pair protocol. Isolated stages and whole-block traces use
R=1,17,1024,4096. Profile separately, within 5,000 measured dispatches per
capture. Record backend/device, source/binary/fixture hashes, power/thermal
conditions and exact allocation/synchronization boundaries. Preserve valid
noisy observations. Finish with reproducible numerical and performance evidence
and a ranked proposal for the next optimization.

Routine implementation defects may be repaired under the fixed contract. Any
required change to arithmetic, budgets, or supported inputs stops acceptance
for a numerical-policy decision. Holdout failures remain recorded. Invalid
measurement conditions pause collection rather than weaken its requirements.
