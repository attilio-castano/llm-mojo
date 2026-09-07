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

`wo_mma=True` remains explicit and the original enqueue defaults to rowwise.
These data do not establish a general dispatch crossover. The subsequent FP32
decode, FP32 prefill and QKV/Wo integration comparisons are now complete.

## Contained GQA parallelism comparison

The approved follow-up keeps the integrated block as its control (benchmark
9). Packed QKV, Wo, rotary positions, cache handling, FP32 attention arithmetic,
BF16 boundaries and all numerical gates remain fixed. Only GQA ownership
changes. The primary question is the `(R,T)=(64,4096)` cached chunk; its current
32-row query tile exposes 28 threadgroups, although each group streams a long
KV sequence. Group count is a mapping fact, not measured GPU utilization.

| Benchmark | Integrated GQA mapping | Query tile | KV splits | Primary threadgroups at (64,4096) |
| --- | --- | --- | --- | --- |
| 9 | 0, existing control | 32 | 1 | 28 |
| 10 | 1 | 16 | 1 | 56 |
| 11 | 2 | 8 | 1 | 112 |
| 12 | 3 | 32 | 4 | 112 plus merge |
| 13 | 4 | 32 | 8 | 224 plus merge |

All use the existing rolled QK reduction, BK=32, one head per query tile and
FP32 PV. Smaller query tiles retain the same aggregate SIMD-group count at
R=64 (112), but change independently schedulable groups, per-group shared
storage and KV reuse. Shared storage is 16/12/10 KiB for BQ=32/16/8.
The split family retains BQ=32 and divides whole KV tiles into four/eight
disjoint ranges along a third grid dimension. It adds a second dispatch;
the whole integrated block has ten dispatches instead of nine for R>1.
Decode retains the existing G32 path for every mapping.

Split scratch is caller-owned, contiguous FP32 `[R,14,S,66]`, holding 64
unnormalized weighted values, maximum and denominator for each row/head/piece.
Its footprint at R=64 is 0.90234375/1.8046875 MiB for S=4/8; at full 4096 it
is 57.75/115.5 MiB. These are allocation sizes, not measured traffic. Allocate
it through `AttentionWorkspace(..., prefill_splits=4 or 8)` before enqueue and
timing. Both benchmark arms share the same scratch allocation. Missing storage
or invalid selectors must reject before any enqueue or cache mutation.

Empty causal pieces write `(u=0,m=-inf,z=0)`. The merge excludes zero-mass
pieces, computes `M=max(m_s)`, and returns
`sum(exp(m_s-M)*u_s) / sum(exp(m_s-M)*z_s)`, rounding only the final attention
output to BF16. No atomic additions, probability matrix or new precision
policy is introduced. The control specialization retains its existing
arithmetic and synchronization; tile size and splits are compile-time choices.

Before measurement, run the frozen upstream operation cases with identical
Q/K/V, the existing materialized FP32 structural edge fixtures, and full/chunked
attention composition on all 17 synthetic and three checkpoint cases. Preserve
isolated GQA 0.0078125 and projected/final 0.03125 limits and exact cache checks.
Test future perturbations, ragged queries/KV tiles, empty pieces, partial/output
guard elements, and a merge with widely shifted maxima. Poison scratch and
repeat asynchronous 15/16/17-row calls and final decode. Test every actual
benchmark route in hot and ring24 modes. A failed numerical candidate is
investigated or rejected; the gate is never widened to admit it.

The single screen is `attention_sublayer_parallelism_screen`: all four
candidates plus control self-pairs at `(64,1024),(64,4096),(1024,1024)`, both
modes, four paired blocks, ten warmups and ten samples per arm. Retain all
2,400 observations and include partial writes, merge, layout copy and the
complete block through residual in timing. A finalist must satisfy the existing
gain rule at `(64,4096)` in both modes: every block faster and median reduction
greater than both 5% and the matching self-pair noise. Select at most one per
family, minimizing the worse of its two median ratios; exact ties prefer BQ16
or split4. Do not combine families or add head-reuse/barrier/BK ablations.

The full `attention_sublayer_parallelism` matrix uses the existing fifteen
workloads and only screen finalists plus control self-pairs: 4,800 observations
for one finalist, 7,200 for two. Its runner requires the complete screen from
the same clean source/build and records the selection and screen hashes.
No finalist means stop at the screen. Retain every full-run result and use
the new data to bound where a candidate is useful; no universal crossover is
assumed. Do not repeat noisy measurements until a preferred result appears.

Profile control and finalists at `(64,4096)` and `(1024,1024)`, with 25 measured
iterations and ten warmups each. Four or six separate Metal captures diagnose
GQA versus merge time and compiler spills. Optional counters may be absent;
profiles do not establish speed claims or measured occupancy by themselves.
Retain compact `parallelism_screen_`/`parallelism_` evidence and extend the
existing plotter. Freeze clean measured source after validation; preserve raw
samples, source/binary/input identities, device/backend and block conditions.
Finish with a reproducible explanation and local commits, including negative
results. The candidate budget ends with this screen and conditional follow-up.

## Completed projection integration

Validated source `dc77016` composes existing packed QKV, bias-free Wo MMA and
FP32 GQA through `enqueue_attention_sublayer_integrated`. The full workflow
passes 87 Mojo and 41 Python tests, with unchanged numerical gates and all
frozen arrays. Checkpoint-derived cases and twelve asynchronous sequences
per configuration also pass, including 15/16/17-row calls and final decode.

Two fresh 4,800-observation comparisons distinguish incremental QKV value
from the combined gain over the original materialized FP32/rowwise baseline.
QKV integration qualifies in 19 of 30 cells; the combined path qualifies in
26 of 30. The remaining cells are inconclusive, with no qualifying regression.
At full 1024/4096 and the `(64,4096)` chunk, incremental reductions are about
70%/59%/24%, and combined reductions are about 88%/91%/80%. The copy dispatch
is included. Do not multiply earlier results to construct these gains.

Eight validated Metal captures retain 2,090 active dispatch durations. GQA
accounts for 64% at full 4096 and 86% for the long chunk; at full 1024, QKV
and Wo together still account for about 55%. These are diagnostic shares,
not a hardware ceiling. The [integrated report](../studies/attention_sublayer/README.md#integrating-qkv-and-wo-with-fp32-attention)
retains all observations, validation, conditions, profiles and next-step reasoning.

## Completed GQA parallelism comparison

Validated source `5778641` adds the four approved GQA mappings to the integrated
attention entrypoint. All 90 Mojo and 43 Python checks passed before timing,
along with the maintained IR inspector, three checkpoint cases and twelve
asynchronous sequences per configuration. All five mappings satisfy the existing
isolated GQA and projected/final gates, including exact cache checks, ragged
tiles, empty causal pieces, poisoned scratch and stable partial-state merging.
No numerical limit changed.

The 2,400-observation screen rejects both smaller query tiles at the target
chunk. Four and eight KV splits qualify in both modes; eight splits advances
under the declared worst-mode ratio rule. The 4,800-observation full matrix
finds three qualifying gains, one regression and twenty-six inconclusive cells.
For `(64,4096)`, whole-block reductions are 39.69% hot and 47.96% ring24 against
the integrated control, including the merge. Chunk `(64,1024)` gains 31.63%
in ring24. Full 64 is 5.64% slower in ring24; full 1024/4096 have no qualifying
gain. Substantial short-workload calibration variation remains in the evidence.

Four validated Metal captures retain 950 active dispatch durations. At the
long chunk, median GQA including merge falls from 1503.791 to 752.791 µs, with
a 12.334 µs median merge. At full 1024 the merge consumes most of the split
kernel's saving. These stage durations diagnose the result; the paired latency
trials establish the performance claims.

Eight splits remains an explicit `gqa_mapping=4` option, requiring caller-owned
`prefill_splits=8` workspace. The default integrated BQ32 mapping remains the
control, and decode retains G32. The bounded candidate budget is complete.
The [parallelism report](../studies/attention_sublayer/README.md#gqa-parallelism-on-the-integrated-block)
retains all raw observations, source/selection identities, numerical validation,
negative results and the next questions suggested by earlier projection studies.

## Contained projection tiles and split-domain follow-up

The approved next experiment keeps the integrated 8x16 projection kernel and
FP32 GQA as control 9. Two candidates give one SIMD group four Apple 8x8 MMA
fragments: Wo 16x16 (14) and Wo 8x32 (15). Each lane owns eight FP32 accumulator
values instead of four. The former reuses weights across sixteen token rows;
the latter reuses inputs across thirty-two output features. No shared staging,
barriers, reduction splitting or new BF16 boundary is added. The original
8x16 kernel remains intact. The source accumulator count is not a physical
register allocation claim.

Validate exact upstream Wo inputs, ragged/exact matrix tiles, bias-free guards,
full/chunked attention and repeated asynchronous calls across 15/16/17 rows.
Validate both tile implementations with packed QKV bias and its layout handoff
as well, before any conditional performance transfer. Keep all numerical
limits and frozen fixture hashes unchanged, including checkpoint comparisons.
New projection choices are explicit and change one projection at a time;
R<16 retains the existing rowwise policy.

The screen compares 9/14/15 at full 16, 64 and 1024 and chunk (64,4096), in both
hot and ring24 modes. Run the same four-block, ten-warmup, ten-sample protocol
at two boundaries: isolated Wo on frozen BF16 upstream attention values, and
the complete integrated attention block. Each retains 1,920 observations.
An eligible tile must qualify as faster at full 1024 in both modes at both
boundaries under the existing 5%/self-pair-noise rule. Select at most one,
minimizing the worse whole-block ratio; exact ties prefer 16x16. The shared
runner binds both complete screens to the same source/build.

Only an eligible tile advances to the existing fifteen-workload whole-Wo
matrix (4,800 observations). Then test that same tile in packed QKV, keeping
Wo at the 8x16 control: candidate 16 or 17 at full 16/64/256/1024/4096 and
chunk (64,4096), retaining 1,920 observations. No eligible tile ends this family
at its two screens. Keep all negative and inconclusive results; no automatic
projection selector follows from this bounded comparison.

Separately, compare the already implemented unsplit/split8 mappings (9/13)
at R=16/64/256 and T=1024/4096, retaining 1,920 observations. This changes the
workload grid, not the GQA algorithms or projection policy. Before interpreting
a crossover, investigate the earlier short-call variation using control-only
self-pairs at full 16, full 64 and chunk (64,1024). Compare the original
per-sample printing with buffered sample emission after both arms, retaining
480 observations per method. Both preserve enqueue-through-completion timing;
buffering removes printing between samples, not synchronization or work.
The measurement boundary is emitted and checked by the parser. This diagnostic
can identify sensitivity to sample emission; it does not pin GPU clocks or
prove that printing caused historical variation. Keep the domain comparison
on the original protocol and retain noisy outcomes rather than repeat them.

Freeze validated source before measurement. Retain compact `tiles_`,
`tiles_kernel_screen_`, `tiles_screen_`, conditional `tiles_qkv_`, `split_domain_`
and `timing_`/`timing_buffered_` evidence in the existing attention study.
Use the same package runner and offline plotter. Explain measured domain and
remaining uncertainty before proposing a dispatch rule; conclude with local
commits and no remote publication.

## Completed projection tiles and split-domain follow-up

Validated clean source `5c7ca77` implements the two contained projection tiles
and explicit integrated options. All 93 Mojo and 45 Python tests passed before
measurement, including frozen synthetic/checkpoint comparisons and twelve
asynchronous sequences across eighteen configurations. BF16 boundaries,
FP32 accumulation and all gates remain unchanged.

The two 1,920-observation screens select only Wo 16x16: at full 1024, isolated
Wo improves 25.20% hot / 29.45% ring24 and whole attention 6.95% / 7.11%.
The conditional 4,800-observation full matrix repeats about 7% at full 1024
and qualifies five of thirty cells. Isolated full-16 ring24 Wo regresses
33.42%; no universal tile replacement follows. Wo 8x32 does not advance.

The conditional 1,920-observation QKV comparison uses the same 16x16 tile
while keeping Wo at 8x16. Seven of twelve cells qualify, including 7.83% /
8.99% whole-attention reductions at full 1024. The two new projections have
not been combined or assigned an automatic selector. Their explicit options
are `projection_mapping=1` (Wo) and `projection_mapping=3` (QKV), with control
GQA. Calls below sixteen rows retain the existing rowwise implementations.

The two 480-observation control-only diagnostics do not establish deferred
printing as a fix for short hot-call variation. The existing split8 domain
comparison therefore keeps the original timing protocol and its own calibration.
Its 1,920 observations qualify eleven of twelve cells at R=16/64/256 and
T=1024/4096. Context-4096 reductions shrink from about 63–69% at R=16 to
15–16% at R=256, consistent with more unsplit query groups already supplying
parallel work. Hot (16,1024) remains inconclusive. No exact crossover or
physical occupancy claim follows.

The [completed report](../studies/attention_sublayer/README.md#projection-tile-ownership)
retains all 13,440 new observations, numerical validation, source/build/selection
identity, negative cases and next-step reasoning. The existing plotter
reproduces 24 tables and 23 figures offline, preserving prior artifacts.
No new profiler captures, combined mappings or remote publication were added.

## Combining the winning projection tiles

The approved follow-up composes the existing 16x16 QKV and Wo kernels through
`projection_mapping=5` (benchmark 18). Control 9 retains both 8x16 projections.
GQA remains the integrated unsplit FP32 policy; decode and R<16 keep the same
rowwise projections. No new arithmetic, kernel, rounding boundary, allocation,
synchronization, cache policy or automatic selector is introduced.

Validate the combined path against the frozen upstream-derived projected and
final outputs, exact caches, full/chunked synthetic and checkpoint cases, and
nineteen asynchronous configurations with twelve sequences each. Exercise the
actual benchmark route around 15/16/17 rows in both timing modes. All gates and
frozen arrays remain unchanged.

Run `attention_sublayer_combined` once over the existing fifteen workloads,
with 9/18, two timing modes, four paired blocks, ten warmups and ten samples
per arm: 4,800 retained observations. Primary interpretation is full 256/1024;
retain all small-row and long-context outcomes under the existing self-pair
calibration and 5% rule. Do not construct a combined gain from earlier ratios.

Then capture control and combined variant at full 256/1024/4096 and chunk
(64,4096), with ten warmups and 25/25/10/25 measured iterations respectively.
Eight separate Metal captures retain 1,530 dispatch durations. Require clean
source/build/input identity and validated nine-stage dispatch order. Collect
stage times and available compiler-spill events; optional hardware counters
may remain absent and must be recorded as such. Use paired unprofiled timing
for speed claims, and profile shares to identify remaining work. Keep GQA
fixed; the earlier split8 results inform subsequent work rather than changing
this comparison's control.

Retain compact `combined_` evidence and numerical validation in the existing
attention study, extend its offline plots, and finish with local commits.
Stop at this candidate and matrix, retaining negative or inconclusive outcomes.

## Completed combined-projection attention comparison

Measured source `bbdbd4a` combines the existing 16x16 QKV and Wo tiles through
`projection_mapping=5`, keeping integrated FP32 GQA fixed. Before timing,
93 Mojo and 46 Python tests, all frozen fixture checks, three checkpoint
cases and twelve asynchronous sequences across nineteen configurations passed.
All numerical gates and BF16 boundaries are unchanged.

The fresh 4,800-observation matrix finds eight faster, one slower and twenty-one
inconclusive cells. Whole-block reductions are 16.51% hot / 18.82% ring24 at
full 256, 15.45% / 16.10% at full 1024, and 9.19% / 9.11% at full 4096.
Ring24 chunk (16,256) regresses 13.91%; the mapping remains an explicit option.
The long cached chunk has no qualifying projection-combination gain.

Eight validated Metal captures retain 1,530 active dispatch durations from the
same source. In the combined variant, projection/GQA shares are 62.65%/22.06%
at full 256, 45.56%/42.84% at full 1024, 22.88%/70.78% at full 4096, and
8.54%/88.82% at chunk (64,4096). Shares are summed active GPU time within each
capture, not whole-call wall time or a hardware-ceiling claim. All captures
report 48-byte maximum target compiler-spill events; physical traffic and
per-stage attribution are not established. Optional counters were not analyzed.

An obsolete CLI argument list rejected variant 18 before compilation. The
remaining profiles used the already-valid package builder API without changing
measured source or repeating latency. The CLI was fixed afterward and now has
a regression test. Post-measurement 47 Python checks validate retained evidence
and reject profile corruption. All 26 tables and 25 figures regenerate offline.
The [combined report](../studies/attention_sublayer/README.md#combined-qkv-and-wo-complete-attention-timing-and-profiling)
records reproduction, precision, negative results and the remaining stage costs.
