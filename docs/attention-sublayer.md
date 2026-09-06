# Attention sublayer contract and study plan

This study composes input RMSNorm, Q/K/V affine projections, RoPE, one layer's
KV cache, causal GQA, bias-free output projection and residual addition. The
target is batch-one BF16 Qwen2.5-0.5B-Instruct: H=896, Nq=14, Nkv=2, D=64,
and at most 4096 live positions. Deterministic synthetic fixtures exercise
numerical sensitivity. A separate checkpoint experiment uses first-layer
weights and token embeddings. Neither is full-model or training-kernel evidence.

## Arithmetic and storage

For R new rows and P cached rows, T=P+R. The caller supplies X[R,H] and
the absolute positions are P through T-1. Surrounding-operation rounding
contracts in model.md apply. Normalization rounds before applying its weight;
projections accumulate FP32 and round BF16; RoPE materializes BF16 products.
The default attention sublayer uses route 3: QK reduction, scaled scores,
softmax probabilities and PV accumulation stay FP32, and GQA output rounds
to BF16 before Wo. Explicit routes 0-2 retain their historical BF16 policies.

The attention result A[R,Nq,D] has row-major storage and is viewed as [R,H]
without a copy. Projection B=BF16(A @ Wo.T) is rounded before
Y=BF16(X+B). A fused final write must preserve both rounding points.

Weights are Wqkv[H+2*Nkv*D,H], bqkv[H+2*Nkv*D], Wo[H,H], and norm[H].
Their source regions retain Q, K, V ordering. A packed row-major projection
uses an explicit bit-preserving unpack for contiguous Q/K/V consumers.
`qkv_mapping=1/2` selects existing packed rowwise/MMA kernels; zero retains
the three separate rowwise projections. The packed row stride is H+2*Nkv*D,
not H or Nkv*D. The layout copy adds no arithmetic or rounding.
K/V caches have fixed [capacity,Nkv,D] storage; only [0,T) is visible.
Rotated K and unchanged V enter [P,T); prefix entries never change.
R must be positive; overflow and invalid routes fail before any enqueue.
Reset changes logical length after outstanding work has completed.

The layer owns reusable device buffers. The caller initializes weights and
BF16 cosine/sine[capacity,D] tables before enqueue. Tables are explicit model
inputs, like weights. The compatibility baseline uses upstream-derived tables;
reproducing a particular CPU trigonometric library is a separate question.
Construction, upload and allocation stay outside enqueue and timing.

Enqueue allocates and synchronizes nothing. Callers retain the layer, input
and context until completion, and submit successive calls on the same stream.
Outputs are overwritten by the next invocation. Input must not overlap writable
workspace/output. Logical cache length records successfully enqueued positions;
an asynchronous device failure invalidates the session. Reset synchronizes.
No Python runs inside the inference enqueue.

## Reference authority and numerical gates

The selected accuracy baseline is the pinned CPU `Qwen2SdpaAttention` with
explicit FP32 Q/K/V and mask at the SDPA boundary, and BF16 output before Wo.
The independent FP64 calculation remains a diagnostic. This is the agreed
inference precision policy; it does not claim to reproduce training kernels.
The FP32 operation gate is atol=rtol=0.0078125, and projected/final composition
gates remain 0.03125. Exact cache and elementwise checks remain strict.

The following BF16 history is retained as compatibility evidence. Ordinary
validation reports BF16 discrepancies, requires finite outputs and exact cache
behavior, and uses the FP32 suite as the attention accuracy gate. Standalone
BF16 kernel tests retain their original requirements. To reproduce the old
strict BF16 eager gates (including the recorded seed-887 failures), run:

```bash
MODULAR_DEBUG=device-sync-mode uv run --locked mojo run -D SUBLAYER_BF16_COMPATIBILITY=1 -D SUBLAYER_HOLDOUT=1 -I src -I build -I tests tests/test_attention_sublayer.mojo
MODULAR_DEBUG=device-sync-mode uv run --locked mojo run -D SUBLAYER_BF16_COMPATIBILITY=1 -D SUBLAYER_HOLDOUT=1 -I src -I build -I tests tests/test_attention_sublayer_operations.mojo
```

Model semantics come from the pinned checkpoint configuration and official
implementation. The language or processor executing a reference does not give
it authority: a handwritten Python formula is an independent diagnostic, and
two Mojo implementations can share a mistake. A comparison must also name its
precision policy, because official backends can round at different boundaries.

The original compatibility target is the actual
[Qwen implementation in Transformers 4.43.1](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py) /
Torch 2.4.0, eager CPU BF16, one Torch thread. Its module source SHA-256 is
`06a9e704e9ec103c47c44fc974519dc93480aa325b9ad7d047239200bcf6fa69`.
Hooks capture its projections, rotated tensors and input to Wo. A wrapper
observes the original rotary function without replacing its arithmetic.
This pins an inference implementation; it does not identify Qwen's training
kernels. Upstream also supports SDPA and Flash Attention with different
intermediate arithmetic. NumPy FP64 remains an independent diagnostic;
disagreement with it does not automatically make Mojo incorrect.

Every primary fixture, including eight 4096-token seeds, runs upstream full
attention and persistent-cache chunks. Original NumPy arrays retain their
original hashes. New upstream arrays and authority metadata have separately
recorded provenance in the same manifest.

The original per-element test is `abs(got-want) <= atol + rtol*abs(want)`.
The current budgets are:

| Boundary | atol | rtol |
| --- | ---: | ---: |
| RMSNorm, raw Q/K/V and rotary tensors | 0.0078125 | 0.0078125 |
| GQA output (approved compatibility calibration) | 0.03125 | 0.03125 |
| Projected branch and final residual output | 0.03125 | 0.03125 |

The approved upstream comparison distinguishes two test boundaries:

- Operation gates give each Mojo operation exactly the tensors upstream
  consumed. RoPE application and residual addition match exactly for these
  BF16 fixtures; the tests enforce that stronger result. Other operations
  retain the stage budgets above.
- Composition gates feed original X through the entire block and require both
  the projected branch and final output to satisfy the original 0.03125
  budgets. All intermediate discrepancies are reported; these also contain
  differences inherited from preceding operations. Checking the branch
  separately prevents a large residual from hiding an error.

Cache append copies actual rotated K and raw V exactly. Prefix snapshots remain
bitwise unchanged and unused capacity retains its poison value. Tests cover
full/chunked execution, nonzero positions, reset, overflow before enqueue and
repeated asynchronous decode. A failed operation gate remains a failure pending
a reviewed numerical decision; final-output agreement alone does not bypass it.
The subsequent FP32 baseline decision above supersedes BF16 compatibility as
the composed attention accuracy gate; the frozen arrays and recorded failures
are preserved.

## Approved work and comparison budget

The following initial budget preceded the separately approved Wo, FP32 decode
and FP32 prefill comparisons specified below. Their bounded protocols and
completed results are retained in the [attention study](../studies/attention_sublayer/README.md).

1. Correct reference composition, bias-free linear and residual operations.
2. Persistent-cache equivalence: full, chunked and single-row execution.
3. Whole-sublayer paired latency and separately captured stage profiles.
4. Screen at most four new configurations: packing/layout, output-projection
   mapping, direct rotary cache writes, and projection/residual fusion.
   Advance at most two to the full matrix; explain negative results too.
5. Curate one study and compact raw evidence using package-owned tooling.

Full matrix (R,T): decode T=1,16,64,256,1024,4096; full prefill
R=T=16,64,256,1024,4096; incremental (4,64),(16,256),(64,1024),(64,4096).
Screen: (1,64),(1,4096),(256,256),(1024,1024),(64,4096).
First isolate existing GQA routes with all surrounding stages fixed, then
compare new candidates against a strong existing-kernel composition.

Hot timing encloses one complete enqueue and synchronization. Ring24 uses
24 distinct weights, inputs and cache prefixes with one synchronization and
divides by 24; it is not a decoder stack. Scratch/output may be shared with
ordered lifetime. Initial prefix setup and resets are outside timing; every
sample overwrites the same suffix starting at P so context cannot drift.
Cache append itself is timed. Allocations, compilation, table construction,
readback and numerical checks remain outside timing. Trace intervals cannot
be added to manufacture whole-operation latency.

Use the existing four-block, ten-warmup, ten-sample paired protocol and
self-pair noise calibration. Record actual Metal device, hardware/software,
source and binary hashes, power and thermal conditions. Capture at most 5000
measured dispatches per trace. Full validation and git diff --check precede
local commits; recorded builds use clean matching sources. The approved scope
includes local implementation, measurement, documentation and commits.
Profiling remains conditional on passing correctness. A change to the numerical
contract requires a separate decision; a failing holdout is retained.

### Contained Wo comparison

The approved first optimization changes only bias-free Wo. The explicit
`wo_mma=True` enqueue argument selects the existing 8x16 Apple MMA mapping;
the default remains rowwise and the GQA route remains 3. Benchmark variant 3
means FP32 attention with rowwise Wo; variant 4 means the same attention with
MMA Wo. These benchmark IDs do not introduce another GQA precision policy.

For `A[R,896] @ Wo[896,896].T`, rowwise assigns one output dot product to
one SIMD group. The candidate assigns an 8x16 output tile to one group, using
two 8x8 matrix fragments per K=8 phase and four FP32 accumulators per lane.
It reuses operands across rows/output features without shared operand storage
or block barriers. The 1,605,632-byte BF16 weight allocation, all buffers,
twelve-dispatch sequence, and BF16 round before residual addition are fixed.
The partial tile at R=1 is deliberately measured, not hidden by a selector.

Before timing, test both mappings on identical upstream attention tensors,
then feed original X through full/chunked composition on all 17 frozen
synthetic and three existing checkpoint cases. Preserve the existing 0.03125
Wo/branch/final gates, 0.0078125 GQA gate, exact cache checks and frozen arrays.
Also exercise repeated asynchronous decode and poison the actual benchmark
buffers for both mappings in hot/ring24 modes. No new tolerance is calibrated.

Use `attention_sublayer_wo_screen` for the five shapes above, comparing variant
4 with 3 and including 3 versus itself in the same run. Advance this one
candidate to `attention_sublayer_wo` only if the screen establishes a gain
under the existing four-block rule in at least one prefill workload/mode;
retain all losses and inconclusive cells. The full run repeats calibration.
If advanced, capture both variants at the existing four profile workloads
with 25, 10, 5 and 10 measured iterations respectively, plus ten warmups.
These 1,200 stage durations explain the comparison; paired latency determines
the speed claim. Counter export is optional and any omission is explicit.
No automatic dispatch crossover is inferred from the older packed-QKV study.

### Contained FP32 decode comparison

After the Wo study, compare the existing G32 and split64-H4 decode ownership
designs with FP32 scaled scores, online softmax state and weighted sums. Keep
BF16 Q/K/V, cache and output, and retain every current numerical gate. The
existing BF16-score specializations remain available with their old defaults.
Rowwise Wo and all surrounding stages are fixed in this experiment.

G32 assigns 32 SIMD groups to each query head, partitions its KV sequence,
then merges FP32 states inside one threadgroup using 8,448 shared bytes and
one barrier. Split64 H4 divides the sequence into 64 independent pieces and
reuses K/V across up to four related query heads. Its 256 groups write
`[14,64,66]` FP32 partial state (236,544 bytes), followed by a separate merge.
These are work/storage counts, not measured DRAM traffic or register usage.
The full block has 10 dispatches with G32 and 11 with split64, versus 12 in
the materialized control. Online weights remain FP32 through accumulation.

Sublayer routes 4/5 explicitly request FP32 G32/split64 decode. For R>1 they
use the existing FP32 materialized route and return actual route 3. Their
single-row calls do not require materialized scratch. Benchmark IDs 5/6 select
these routes with rowwise Wo; benchmark 3 is the fixed materialized control.
The public default remains route 3. No length crossover is introduced.

Before timing, run the complete validation workflow, all synthetic/checkpoint
operation and composition gates, and twelve asynchronous 65-token sequences
per configuration. In addition to each fixture's last token, compare both
decode mappings on fixed prefixes 1,7,16,31,32,33,63,64,65,257,351,668,1024,
4095,4096 when present. The exact upstream Q/K/V and causal output arrays
already contain these comparisons; 351/668 include earlier rounding examples.
Use the existing 0.0078125 GQA and 0.03125 composition gates and exact cache
bits. Also compare the two mappings against materialized FP32 on the existing
standalone edge fixtures (NaN guards, empty/ragged splits, tied/extreme scores,
cancellation and head mapping). That cross-kernel check supplements the pinned
upstream reference. Frozen oracle arrays and tolerances do not change.

Screen the three-way comparison at decode T=64/4096 in hot and ring24 modes.
Advance to all six existing decode lengths if at least one candidate establishes
a gain in one screening cell. Both candidates and all negative/inconclusive
results remain in the full comparison. Use the original four-block paired rule
with fresh self-pair calibration for each run: 960 screen and 2,880 full-run
observations. This compares each candidate with the materialized control; it
does not establish a direct G32-versus-split64 crossover.

If advanced, capture variants 3/5/6 at T=64/4096, with 50 measured iterations
and 20 warmups each. Retain all 3,300 active dispatch durations, validate the
variant-specific sequence, and leave optional counter analysis explicitly
absent if not performed. Extend the current attention study with `decode_`
and `decode_screen_` evidence. Builds require clean matching sources; numerical
failure stops performance work without widening gates.

### Contained FP32 prefill comparison

The decode milestone passed its numerical and performance gates. Adapt one
prefill candidate from the prior rolled-QK study: BQ=BK=32, four SIMD groups
per query-head tile, HEADS=1 and SCHEDULE=2. Keep the rolled QK loop, output
ownership and four barriers per KV tile. QK uses BF16 operands with FP32
accumulation and retains the scaled FP32 score. Online softmax state and tile
weights remain FP32. PV uses FP32 matrix operands/accumulation, widening the
stored BF16 values losslessly; only final attention output rounds to BF16.
Shared K/V remain BF16. Shared score/probability tiles are FP32, increasing
source-declared shared storage from 14 to 16 KiB per block. No global
score/probability scratch is needed by this candidate.

A local 8x8 FP32 matrix identity probe passed on Apple M4 Pro / Metal with
non-BF16-representable operands and zero maximum error. Retain that arithmetic
regression alongside the candidate's tests. This establishes primitive support,
not its performance. The existing BF16 prefill specializations keep their defaults.

Sublayer route 6 explicitly selects FP32 rolled MMA for R>1 and FP32 G32 for
R=1, returning actual route 6 or 4 respectively. This one-dispatch decode
choice completes an explicitly requested family; it is not a measured G32/H4
crossover or the public default. Benchmark 7 selects route 6 with MMA Wo for
R>1. Compare only against benchmark 4 (materialized FP32 plus MMA Wo), so
Wo and all surrounding stages are fixed. Do not multiply this comparison by
the earlier Wo gains. The default remains materialized route 3/rowwise Wo.

Before timing, retain the existing GQA 0.0078125 and composition 0.03125 gates,
exact cache checks, all 510 frozen synthetic arrays and 63 checkpoint arrays.
Compare isolated full/suffix attention on exact upstream Q/K/V, then compose
from X with both Wo mappings. Reuse all 29 standalone prefill edge fixtures
against materialized FP32, including causal future perturbations and full/suffix
agreement; repeat poisoned-output launches. Run twelve asynchronous 65-token
sequences with mixed prefill/decode for route 6 without materialized scratch,
and preserve coverage of the previous seven configurations. A numerical failure
stops performance work; no tolerance is widened.

Screen (R,T)=(256,256),(1024,1024),(64,4096) in hot and ring24 modes using
candidate/control 7/4 plus 4/4 self-pairs: 960 observations. If at least one
cell qualifies under the existing four-block rule, advance to all nine existing
R>1 workloads: full 16,64,256,1024,4096 and chunks (4,64),(16,256),(64,1024),
(64,4096). The full run retains 2,880 observations with fresh calibration.
Keep every loss and inconclusive result. No tile or precision retuning follows
from timing within this experiment.

If advanced, capture variants 4/7 at (1024,1024),(4096,4096),(64,4096), with
25,10,25 measured iterations respectively and ten warmups each. Validate
12/10 dispatches per call and retain all 1,320 active durations and compiler
spill records. Optional counter export may remain explicitly absent. Given
the decode captures' unchanged-stage variation, these separate traces remain
diagnostic; paired latency establishes gains. Extend this same study with
`prefill_` and `prefill_screen_` evidence using clean, matching measured source.

### Integrating the projection studies end to end

The user approved integrating the earlier QKV and Wo studies with the validated
FP32 attention paths. `enqueue_attention_sublayer_integrated` is the explicit
Qwen entrypoint: R<16 uses packed rowwise QKV and rowwise Wo; R>=16 uses the
existing packed 8x16 MMA QKV and bias-free 8x16 MMA Wo. It uses FP32 G32 decode
and rolled-MMA FP32 prefill, returning actual GQA route 4 or 6. Sixteen rows is
a conservative policy chosen before measurement, not a proven crossover for
every unmeasured row count. The original enqueue and all explicit mappings
remain available for comparisons. Split64-H4 remains explicit; the earlier
study did not establish a G32-versus-split crossover.

Packed QKV computes X[R,896] @ Wqkv[1152,896].T + bias with the earlier kernels.
One extra GPU dispatch reads the packed BF16 result and writes contiguous
Q[R,896], K[R,128], V[R,128]. It replaces three projection dispatches with one
projection and one copy. The copy requests 4*R*1152 bytes (read plus write),
including 18 MiB at R=4096; these are source-requested bytes, not measured
DRAM traffic. The integrated block has nine dispatches, versus ten with
separate projections and the same GQA/Wo policy. Wo still rounds to BF16
before the separate residual addition. There is no allocation or synchronization
inside enqueue. Construct `AttentionWorkspace(..., fp32_materialized=False)`
when using the integrated entrypoint alone; it does not need the 896 MiB
materialized FP32 probability workspace at full 4096. Packed and contiguous
diagnostic intermediates remain allocated so the handoff stays inspectable.

The numerical gate precedes timing: all 17 frozen synthetic and three existing
checkpoint cases, isolated packed QKV on exactly upstream's normalized inputs,
full/chunked composition with both packed mappings and MMA Wo, then the public
integrated entrypoint. Keep QKV/GQA 0.0078125 and Wo/final 0.03125 gates, all
frozen arrays, and exact cache-prefix/suffix/unused-capacity checks. A dedicated
ragged multi-row layout test uses signed zeros, subnormals and extreme BF16
bit patterns with guard elements. Twelve asynchronous sequences include
15/16/17-row calls and final decode, with no materialized probability storage.
Poison projection intermediates and check the actual benchmark routes in both
hot and ring24 modes, including the 15/16 boundary. A numerical failure stops
performance work; no tolerance changes or precision retuning are authorized.

Use the existing fifteen-workload matrix and four-block paired protocol for
two bounded comparisons, each including fresh self-pairs and 4,800 retained
observations. `attention_sublayer_projections` compares 9 with 8: integrated
QKV versus separate rowwise QKV, with GQA and Wo policy fixed. Variant 8 uses
rowwise Wo below 16 rows; it is a precise new paired control, not a reuse of
old absolute timings. `attention_sublayer_integrated` compares 9 with the
original variant 3: all selected mappings versus materialized FP32 attention
and rowwise projections. Report these separately; never multiply earlier
stage gains. Keep regressions and inconclusive outcomes and stop after the
declared matrix. The already-studied kernels do not require a new tuning screen.

Profile variants 8/9 at (R,T)=(1,4096),(1024,1024),(4096,4096),(64,4096),
with 50,25,10,25 measured iterations and ten warmups. The eight captures retain
2,090 active dispatch durations, including the explicit copy, and compiler
spill records. Optional counters may remain absent. Retain `projections_`
and `integrated_` results in the existing study directory and regenerate them
with the shared plot command. Freeze validated clean source before recorded
builds, and bind both benchmark/profile records to that source.

## Numerical findings before profiling

The integration exposed a RoPE compatibility defect against the declared eager
path. Ordinary BF16 multiplication/addition lowered with LLVM's `contract`
flag, permitting Metal to fuse across the intended product rounding. For inputs
4.59375 and -0.03466796875 with cosine 0.796875 and sine 0.60546875, upstream
returns 3.671875. The old GPU path returned 3.6875. Contraction-disabled
FMA-with-zero products preserve the eager boundary. The exact regression fails
against the old kernel and passes with the fix. This does not imply that fused
arithmetic is inherently invalid for every supported inference backend.

Table generation is a separate discrepancy. At position 2370, dimension 7,
upstream's sine is 0.73828125 while NumPy yields 0.734375. For the recorded
query pair, upstream and Mojo return 0.09375 while NumPy returns 0.078125.
Another table entry at position 2954 differs between upstream and Mojo; a
higher-precision sine calculation agrees with Mojo. The baseline therefore
consumes explicit upstream tables and tests their application independently.

The original nine full-block cases, including the 4096-token seeds 53 and 103,
passed the composition and cache checks on all applicable routes. Every
operation gate passed on seed 53. Under the original 0.015625 GQA budget,
seed 103 exposed two GQA output failures out of 3,670,016 elements on each
route, even with identical upstream Q/K/V. All other operation gates passed.

One score explains both failures. At query position 350, head 4, key 303:

| Quantity | Value |
| --- | ---: |
| FP64 scaled dot product | 67.25000309944153 |
| Serial FP32 scaled dot product | 67.25 |
| Upstream BF16 score | 67.5 |
| Mojo BF16 score | 67.0 |
| Upstream probability for key 303 | 0.07177734375 |
| Probability with the Mojo score | 0.044677734375 |

A small FP32 reduction difference straddles a BF16 midpoint. Changing just
that score in the Torch calculation reproduces all 896 GPU output components
of this query row exactly. The two dimensions outside the original budget are
head 4 dimensions 4 and 53. This identifies the arithmetic mechanism without
assuming that the CPU's reduction order defines uniquely correct arithmetic.

Four additional seeds (149, 211, 307, 401) were fixed before their GPU results
were observed. These counts use the original 0.015625 operation budget:

| Seed | Materialized GQA failures | Original / rolled MMA failures | Largest scaled error across routes |
| --- | ---: | ---: | ---: |
| 149 | 17 | 15 / 15 | 0.025209572 |
| 211 | 0 | 0 / 0 | 0.00625 |
| 307 | 2 | 0 / 0 | 0.01601281 |
| 401 | 0 | 0 / 0 | 0.00625 |

Each full-prefill comparison has 3,670,016 elements. Decode and 17-row suffix
checks pass on these additional seeds. These are numerical diagnostic samples,
not a statistical bound on arbitrary inputs or performance evidence.

### Approved calibration failed the holdout

For the new GQA-to-upstream comparisons only, atol=rtol=0.03125 was approved,
conditional on two previously unused full-context seeds passing before
profiling resumed. Seeds 509 and 887 were fixed before observing their GPU
results. Existing standalone GQA tolerances, other stage budgets, exact cache
checks and projected/final limits remain unchanged.

| Holdout seed | Materialized GQA failures | Original / rolled MMA failures | Largest scaled error across routes |
| --- | ---: | ---: | ---: |
| 509 | 0 | 0 / 0 | 0.014354067 |
| 887 | 46 | 47 / 47 | 0.15555556 |

The GQA tests feed identical upstream Q/K/V. Each full-prefill comparison has
3,670,016 elements; last-token and last-17-row comparisons pass on both seeds.
Other operation gates pass. Seed 887 also fails both unchanged composition
gates, in full and chunked execution, on all three routes. Other primary cases
pass the composition gates.

| Route | Branch failures / largest scaled error | Final output failures / largest scaled error |
| --- | ---: | ---: |
| Materialized GQA | 51 / 0.044444445 | 39 / 0.05078125 |
| Original MMA | 57 / 0.045333333 | 38 / 0.0546875 |
| Rolled MMA | 57 / 0.045333333 | 38 / 0.0546875 |

These full-prefill counts also occur in the 4078-row first chunk. Subsequent
chunks pass. Cache append, prefix preservation, unused capacity, reset and
asynchronous reuse checks remain exact and separate from numerical agreement.

One score explains the largest GQA discrepancy, at query 667, head 0, key 665:

| Quantity | Value |
| --- | ---: |
| FP64 scaled dot product | 75.74999618530273 |
| Serial FP32 scaled dot product | 75.74999237060547 |
| BF16 rounding midpoint | 75.75 |
| Upstream BF16 score | 76.0 |
| Mojo BF16 score | 75.5 |
| Upstream probability for key 665 | 0.5859375 |
| Probability with the Mojo score | 0.4609375 |

Changing this one score in the Torch calculation reproduces all 896 components
of the materialized GPU output row exactly. Here the FP64 value lies on the
Mojo side of the midpoint: the discrepancy cannot simply be described as Mojo
using less accurate arithmetic. BF16 scores at this magnitude are spaced 0.5
apart. A tiny change before rounding can therefore create a 0.5 logit change;
softmax changes that key's unnormalized weight by exp(0.5), about 1.65 times.

This rejects the proposed empirical bound on these fixtures. The approved
0.03125 gate stays in place and fails; it is not widened again. Both held-out
seeds are retained in the primary suite. Profiling and the optimization screen
remain paused, and this branch has no performance result.

### Official eager and SDPA also differ

The diagnostic runs actual `Qwen2SdpaAttention` from the same pinned module,
with the same synthetic weights and X, forcing `SDPBackend.MATH` on CPU.
Every captured input to attention is exactly equal to eager's, including
normalized X, raw Q/K/V and rotated Q/K. The differences begin inside attention.

| Seed | GQA largest scaled error | Projected branch largest scaled error | Final output largest scaled error |
| --- | ---: | ---: | ---: |
| 103 | 0.37642941 | 0.08818182 | 0.08823530 |
| 887 | 0.34184140 | 0.09798535 | 0.09090909 |

These are official SDPA **versus official eager**, not Mojo results. In pinned
[Torch 2.4.0](https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/native/transformers/attention.cpp),
math SDPA pre-scales Q and K in their input dtype before matmul; eager divides
the BF16 dot-product result by sqrt(D). For D=64, the former applies an
inexact factor sqrt(1/8) to each BF16 input, whereas the latter divides by 8.
Both materialize BF16 intermediates in this pinned version. Do not infer the
old backend's arithmetic from newer Torch documentation, which describes FP32
intermediates. The synthetic recipe produces strongly correlated
inputs/weights and large logits. These are useful sensitivity tests, not an
estimate of checkpoint activation distributions or inference quality.

The official code remains the source for model semantics, but it supports
multiple numerical implementations. A named backend/precision contract is
necessary for compatibility. The approved follow-up experiment tests an
explicit FP32 attention policy while retaining these eager comparisons and
their failed holdout. It does not claim to reproduce Qwen's training kernels.

### Explicit FP32 attention experiment

Route 3 reuses the materialized Mojo GQA algorithm with FP32 score/probability
scratch. Inputs, weights, Q/K/V, cache entries and final attention output remain
BF16. QK reduction, score scaling, softmax and PV accumulation use FP32. The
surrounding normalization, projections, rotary application and residual
addition are identical to the other routes. The experiment initially left the
default unchanged; the subsequent approved baseline decision promotes route 3.

The comparison executes the actual pinned `Qwen2SdpaAttention`. A wrapper
promotes the captured BF16 Q/K/V and mask to FP32 immediately before the
original Torch SDPA call, forces `SDPBackend.MATH`, and rounds its output to
BF16 before Wo. These boundary casts are our explicit experimental inference
policy; this is not the unmodified Torch 2.4 BF16 SDPA path. All captured inputs
to attention are checked exactly against the eager fixtures. An independent
FP64 calculation on those same Q/K/V helps distinguish arithmetic error from
disagreement with a backend's rounding policy.

The operation gate was fixed at atol=rtol=0.0078125 before the experiment.
Composition retains atol=rtol=0.03125 for both the projected branch and final
output, with exact cache checks. The 15 existing cases plus two previously
unused 4096-token seeds, 1009 and 1237, were declared before GPU results.
All four Mojo routes are measured against both references. Only route 3 is
required to satisfy the FP32 gates. The original eager gates remain available
through the explicit BF16 compatibility command above.

On Apple M4 Pro/Metal, the new route passes all 17 synthetic cases, including
full and incremental attention and full/chunked composition. The largest
scaled errors against the declared FP32 reference are:

| Boundary | Largest scaled error | Limit |
| --- | ---: | ---: |
| Isolated GQA | 0.00625 | 0.0078125 |
| Projected branch after the full block | 0.005167959 | 0.03125 |
| Final residual output | 0.0078125 | 0.03125 |

Here scaled error is `abs(got-want)/(1+abs(want))`; it is not relative error
alone. On seed 887, declared CPU FP32 attention versus the independent FP64
calculation has largest scaled error 0.006134969; eager versus FP64 reaches
0.3030043. This supports the rounding-sensitivity explanation on these inputs.
It is not a language ranking or a proof about all model activations.

This materialized route is an accuracy control. At full 4096-token prefill its
`[4096,14,4096]` FP32 scratch uses 939,524,096 bytes (896 MiB), twice the BF16
control's scratch. A later streaming implementation can avoid storing that
matrix while retaining FP32 state, but its arithmetic and performance must be
tested independently. No speedup is inferred from this experiment.

### Checkpoint weights and token embeddings

The three prompts were fixed before their GPU results: an explanation of the
sky, a small Python-function request, and repeated laboratory notes. The pinned
chat template produces 30, 41 and 4096 retained tokens, respectively. Inputs
are the actual token embeddings supplied to layer 0; the weights are that
layer's input normalization, Q/K/V projections and output projection.

Route 3 passes the same isolated and composed gates on all three cases, with
full/chunked execution and exact cache checks. Across them, its largest scaled
errors against the declared FP32 reference are 0.0004568296 for isolated GQA,
0.0014124294 for the projected branch and 0.0013793104 for final output.
Both additional checkpoint test suites pass. All 63 checkpoint arrays
regenerate with identical hashes, and sources remain unchanged during those
checks.

The existing BF16 routes also fall within the eager compatibility limits on
these checkpoint cases. Their largest final-output scaled errors against
eager are 0.001914334 (materialized) and 0.002590674 (either optimized route).
However, the FP32 Mojo route versus eager reaches 0.044159543, while the
optimized BF16 routes versus the FP32 reference reach 0.04359673. These
cross-policy differences exceed the 0.03125 composition limit. A passing
comparison therefore needs the reference's precision policy attached to it.

On identical captured checkpoint Q/K/V, declared CPU FP32 attention versus the
independent FP64 calculation has largest scaled error 0.000456621, while eager
versus FP64 reaches 0.072359845. This supports using the FP32 route as an
accuracy control for subsequent optimization. It does not establish which
kernels ran during training or measure full-model generation quality.

The approved decision adopts explicit FP32 scores/softmax as the accuracy
control and preserves eager as a named compatibility comparison. Route 3 and
FP32 workspace are now the sublayer defaults. The two BF16 synthetic holdout
failures remain recorded. Full validation of this contract precedes baseline
measurement. Profiling will determine which optimization is worth testing;
retaining FP32 state without the full score matrix is a possible later study.

## Reproduction

Generate and validate the selected reference and all standalone operations:

```sh
uv run --locked python -m llm_mojo.validate
```

The historical BF16 calibration and holdout can still be reproduced with
their strict compatibility gates. The holdout commands intentionally fail
on the retained seed-887 discrepancy:

```sh
uv run --script tests/fixtures/attention_sublayer/generate.py --calibrate
uv run --locked mojo run -D SUBLAYER_BF16_COMPATIBILITY=1 -D SUBLAYER_CALIBRATION=1 -I src -I build -I tests tests/test_attention_sublayer_operations.mojo
uv run --script tests/fixtures/attention_sublayer/generate.py --holdout --compare-backends
uv run --locked mojo run -D SUBLAYER_BF16_COMPATIBILITY=1 -D SUBLAYER_HOLDOUT=1 -I src -I build -I tests tests/test_attention_sublayer_operations.mojo
uv run --locked mojo run -D SUBLAYER_BF16_COMPATIBILITY=1 -D SUBLAYER_HOLDOUT=1 -I src -I build -I tests tests/test_attention_sublayer.mojo
```

Generated arrays and manifests remain in ignored `build/oracle_data/attention_sublayer/`.
The manifest includes upstream source identity, exact test boundaries, array
hashes, NumPy/upstream differences, upstream full/chunked checks and the
single-score diagnostics. `--compare-backends` additionally records the actual
official eager/SDPA comparison on available midpoint cases (103 and/or 887).
These commands do not download checkpoint weights.

The explicit precision comparison uses the frozen synthetic inputs and adds
its own checked fixture anchors:

```sh
uv run --script tests/fixtures/attention_sublayer/precision.py
uv run --locked mojo run -I src -I build -I tests tests/test_attention_precision.mojo
```

The checkpoint workflow pins the revision and source asset identities from
`model.md`. Downloading requires the explicit flag; subsequent runs verify
local files. This run uses the first 302,126,368 bytes of `model.safetensors`,
which contain the original header, embeddings and every required layer-0
attention tensor. Its separately verified SHA-256 is
`0d3c86fcaa9573dbac31055974018e4d1a94a07124e5d124747feed78b51f6fa`.
The first-layer attention range [298454048,302126368) additionally matched a
fresh read at the pinned revision, byte for byte. The other six assets have
their complete hashes verified. The full checkpoint hash identifies the source
artifact; it was not verified over a complete local download in this run.

The prefix reader checks the original safetensors header and accepts only
complete BF16 tensors inside that byte range. It extracts embeddings and the
first attention block's weights, applies the checkpoint tokenizer/chat template
to three fixed prompts, and truncates the long prompt to the first 4096 tokens.
Assets and arrays stay in ignored `build/`. Omitting `--attention-prefix` uses
and verifies a complete checkpoint instead.

```sh
uv run --script tests/fixtures/attention_sublayer/checkpoint.py --download --attention-prefix
uv run --script tests/fixtures/attention_sublayer/checkpoint.py --attention-prefix
MODULAR_DEBUG=device-sync-mode uv run --locked mojo run -D PRECISION_CHECKPOINT=1 -I src -I build -I tests tests/test_attention_precision.mojo
```

The earlier [numerical record](../studies/attention_sublayer/numerics.json)
retains the calibration, all full-context GQA and composition gate summaries,
the official-backend comparison, and source/fixture provenance. These are
correctness observations from a dirty local branch, not benchmark evidence.
The documented validation workflow verified all frozen arrays and passed 36
Python checks. Across all Mojo suites, 77 tests passed and the two new
compatibility suites failed on seed 887. The remaining suites were run
explicitly after the workflow stopped at the first failure; existing kernel
tests and benchmark route smoke passed. Source/fixture hashes were unchanged
through that validation. It is a historical snapshot; its hashes identify the
earlier implementation.

The [FP32 experiment record](../studies/attention_sublayer/precision_numerics.json)
retains all 4608 reported attention/branch/final comparisons from the synthetic
and checkpoint matrices, their FP64 diagnostics, precision contracts, fixture
identities and source hashes. That historical validation checked all 510 frozen precision arrays
(including the unchanged 405 original arrays) and passes 36 Python tests.
Across the Mojo suites, 79 tests pass and the same two eager compatibility
tests fail. The remaining suites were run after the first failure; route smoke
passes. Sources remain unchanged through those runs. A separate normal-mode
check passed 12 asynchronous 65-token sequences on each of the four routes,
with per-dispatch debugging synchronization disabled. That snapshot predates
the approved baseline promotion and the whole-sublayer measurement work.

## Baseline promotion validation

The full workflow passes 81 Mojo tests and 38 Python tests with unchanged
source during validation and all frozen arrays intact. Separate checks pass
the three checkpoint cases and 12 asynchronous 65-token sequences per route.
Both explicit strict BF16 holdout commands reproduce the expected failures.

The instrument subsequently gained a stronger pre-timing check: each arm
poisons its cache suffix and attention output, then requires correct branch,
final output, unchanged prefix bits and exact suffix copies. Its route checks
pass hot and ring24 decode/full/chunked cases, including full 4096-token
ring24. The [validation record](../studies/attention_sublayer/validation.json)
distinguishes that instrument-only change from the full numerical run and
retains the final source hashes and exact reproduction commands.

## Completed baseline measurement

The [attention sublayer study](../studies/attention_sublayer/README.md) retains
the complete 2,400-observation latency matrix and four twelve-stage profiles
containing 1,920 measured dispatch durations, all from clean source `8d8c854`.
It separates noisy small calls, whole-block timing and active GPU stage time.
The full-context optional counter export was stopped because of its size;
the raw trace and verified stage durations remain available, and the missing
counter analysis is explicit.

## Completed Wo experiment

The [contained Wo comparison](../studies/attention_sublayer/README.md#contained-wo-results)
tested the existing Apple MMA mapping with FP32 attention and all other stages
fixed. All numerical gates passed without tolerance changes, including all
17 synthetic and three checkpoint cases, exact cache checks and repeated
asynchronous execution. The full workflow passed 81 Mojo and 38 Python tests.

The candidate passed the declared five-shape screen and then completed the
full matrix: 13 workload/mode gains, 15 inconclusive comparisons and two hot
decode regressions. Full-prefill whole-block reductions are about 31% at 256
tokens, 22.5% at 1024 and 10.4% at 4096 in both modes. All 6,400 screen/full
observations and 1,200 profile dispatch durations are retained from clean
source `07984fe`. Optional counter analysis is explicitly absent.

`wo_mma=True` remains explicit and the default remains rowwise. These data do
not establish a general dispatch crossover. The study relates the results to
earlier projection/decode/prefill experiments and recommends a separate FP32
decode ownership comparison next, followed by FP32 prefill tiling and PV.
