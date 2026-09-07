# Attention experiment plans

These historical plans record the bounded comparisons and stop conditions declared
before measurement. They are completed study history, not pending work. Read the
[results](experiments.md) for outcomes and the [contract](../../docs/attention-sublayer.md)
for current behavior. No new search or automatic route selector is implied.

## Approved work and comparison budget

The following initial budget preceded the separately approved Wo, FP32 decode
and FP32 prefill comparisons specified below. Their bounded protocols and
completed results are retained in the [attention study](README.md).

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

## Baseline promotion validation

The full workflow passes 81 Mojo tests and 38 Python tests with unchanged
source during validation and all frozen arrays intact. Separate checks pass
the three checkpoint cases and 12 asynchronous 65-token sequences per route.
Both explicit strict BF16 holdout commands reproduce the expected failures.

The instrument subsequently gained a stronger pre-timing check: each arm
poisons its cache suffix and attention output, then requires correct branch,
final output, unchanged prefix bits and exact suffix copies. Its route checks
pass hot and ring24 decode/full/chunked cases, including full 4096-token
ring24. The [validation record](data/validation.json)
distinguishes that instrument-only change from the full numerical run and
retains the final source hashes and exact reproduction commands.

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

## Closing the split8 and projection integration gap

Enable only the additional pair `gqa_mapping=4, projection_mapping=5` in the
integrated enqueue: existing split8 FP32 GQA with existing 16x16 packed QKV
and Wo. Benchmark variant 19 names this composition. Variant 13 has split8
with old 8x16 projections; variant 18 has the new projections with unsplit GQA.
R<16 retains rowwise projections; R=1 retains FP32 G32 decode. Multi-row calls
need caller-owned `prefill_splits=8` storage, checked before the first enqueue.
The ten dispatches include the split-state merge. BF16 boundaries, FP32
accumulation, exact cache checks, and final scaled-error gate 0.03125 stay fixed.

Before measuring, run the full frozen validation, three existing checkpoint
cases, actual benchmark smoke routes and twelve asynchronous sequences for
all twenty configurations. Add the composition to full and incremental
upstream-derived FP32 tests, and verify missing split storage leaves cache
and output unchanged. Exercise both comparison controls, self-pairs, decode,
and 15/16/17 row boundaries in the actual measurement instrument.

Freeze a clean source commit, build once, then run exactly two paired studies:
`attention_sublayer_split_combined_projections` (13 versus 19) measures the
projection gain with split8 fixed; `attention_sublayer_split_combined_gqa`
(18 versus 19) measures the split8 gain with new projections fixed. Each has
its own control self-pair calibration. Both use R=16/64/256 crossed with
T=1024/4096 plus (16,256), the previously observed short-chunk projection
regression. Four paired blocks, ten warmups and ten samples per arm, hot and
ring24 yield 2,240 observations per study, 4,480 total. Use the existing
all-blocks and max(5%, self-pair deviation) rule; retain inconclusive and slower
cells without selective reruns. No new tile screen, profiler campaign,
automatic dispatch selector or full-decoder claim is part of this closure.

Retain `split_combined_projections_` and `split_combined_gqa_` samples and
provenance, plus `split_combined_validation.json`, in the existing attention
study. The usual plotter must regenerate all tables and figures, including
unchanged historical artifacts. Interpret each comparison directly; do not
multiply old speed ratios to claim a new combined gain.
