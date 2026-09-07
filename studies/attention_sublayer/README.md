# Qwen attention sublayer

The latest contained experiment divides GQA's KV sequence into eight pieces
while keeping packed QKV, Wo and FP32 attention fixed. On Apple M4 Pro / Metal,
this reduces whole-attention latency by **39.69% hot and 47.96% ring24** for a
64-token chunk at context 4096, compared with the already integrated block.
Smaller query tiles were slower in the screen; splitting did not establish a
general full-prefill gain. The [parallelism results below](#gqa-parallelism-on-the-integrated-block)
retain all 7,200 observations, four diagnostic captures and the unchanged
numerical gates. The split mapping is an explicit integrated-engine option.

The preceding packed-QKV, Wo and FP32 GQA studies run together through one
public Mojo enqueue. On Apple M4 Pro / Metal, the integrated path reduces
whole-attention latency by **about 88% at full 1024-token prefill, 91% at full
4096, 80% for a 64-token chunk at context 4096, and 89–92% for decode at context
4096**, compared directly with the original materialized FP32/rowwise baseline.
A separate comparison with optimized GQA/Wo already fixed finds that QKV
integration alone adds **70%, 59%, and 24% reductions** at the three prefill
workloads respectively. These are two fresh paired comparisons, not products
of earlier speedups. All 9,600 observations and 2,090 profile durations are
retained, including inconclusive cases; neither comparison finds a qualifying
regression.

The measured unit is
`X → RMSNorm → Q/K/V → RoPE → KV append → GQA → Wo → residual X+branch`.
It ends before the decoder MLP. Batch is one, H=896, Nq=14, Nkv=2 and D=64.
The attention output `[R,14,64]` is viewed as `[R,896]` for Wo without a copy.
For R new rows and T visible positions, the cache prefix has length T-R.

## Accuracy comes first

The primary reference is pinned Transformers 4.43.1 `Qwen2SdpaAttention` on
Torch 2.4.0 CPU, with explicitly FP32 Q/K/V and mask at the SDPA boundary,
and BF16 attention output before Wo. Weights, activations, cache and final
output remain BF16. This is our agreed inference precision policy; it does
not identify Qwen's original training kernels.

The default Mojo route retains FP32 scores, probabilities and accumulation.
Initial baseline validation passed 81 Mojo tests and 38 Python tests, all 510 frozen synthetic
arrays, three checkpoint attention cases and repeated asynchronous cache
reuse. The instrument additionally passed full 4096-token ring24 checks
with a poisoned cache suffix. It checks the projected branch separately so
the residual cannot conceal an attention error.

Historical BF16 eager compatibility failures remain reproducible. The
[contract and numerical investigation](../../docs/attention-sublayer.md),
[validation](validation.json), [original numerical record](numerics.json)
and [FP32 comparison](precision_numerics.json) retain the precision decision,
independent gates and provenance. The original baseline's measured engine and
fixtures match its validation hashes; only the runner's per-process timeout
changed afterward.

## GQA parallelism on the integrated block

This experiment changes how the existing fused FP32 GQA assigns work. Its
control already has FlashAttention-style query tiling, online softmax, a
rolled QK reduction, and FP32 PV. Packed QKV and the studied Wo mapping run in
both arms. Benchmark 9 is that integrated control; 10/11 use query tiles of
16/8, and 12/13 keep the 32-query tile with four/eight KV splits. The measured
unit still ends at the residual addition, including the split merge.

### Work ownership and storage

At `(R,T)=(64,4096)`, fourteen query heads and two 32-row query tiles expose
only 28 primary threadgroups. Each group streams the KV sequence. The two
candidate families create more independently schedulable groups in different
ways:

| Mapping | Primary threadgroups | Total primary SIMD groups | Shared storage per group |
| --- | ---: | ---: | ---: |
| BQ32 control | 28 | 112 | 16 KiB |
| BQ16 | 56 | 112 | 12 KiB |
| BQ8 | 112 | 112 | 10 KiB |
| BQ32, split4 | 112 | 448 | 16 KiB |
| BQ32, split8 | 224 | 896 | 16 KiB |

Smaller query tiles keep the aggregate SIMD-group count constant here. They
also share each K/V tile across fewer queries and load it in more threadgroups.
Splitting instead partitions whole 32-position KV tiles along a third grid
dimension while retaining the 32-query reuse. These are source-level ownership
and storage facts; group counts do not measure occupancy or memory bandwidth.

Each split writes an FP32 weighted numerator, maximum and denominator.
A second dispatch rescales the nonempty partials to a common maximum before
combining them. Empty causal pieces contribute zero mass. Only the final
attention output rounds to BF16, at the existing boundary before Wo.
The integrated block therefore has ten dispatches for split prefill, versus
nine for the control. Decode uses the same existing G32 route in every mapping.

Caller-owned scratch is contiguous `[R,14,S,66]` FP32. Eight splits allocate
**1.805 MiB at R=64, 28.875 MiB at R=1024, and 115.5 MiB at R=4096**; four
splits need half as much. These are allocation sizes, not measured DRAM traffic.
Allocation is outside enqueue and timing, and both arms share the same scratch
allocation. Select the measured eight-split path with
`AttentionWorkspace(..., fp32_materialized=False, prefill_splits=8)` and
`enqueue_attention_sublayer_integrated(..., gqa_mapping=4)`.
The default mapping remains the integrated BQ32 control; this bounded study
does not establish an automatic crossover across unmeasured shapes.

### Accuracy before timing

All five mappings passed the frozen upstream-input GQA comparisons and
full/chunked attention composition on 17 synthetic and three checkpoint cases.
The selected reference and every gate above remain fixed. Across the five
mappings, the largest observed scaled errors were:

| Boundary | Synthetic maximum | Checkpoint maximum | Fixed limit |
| --- | ---: | ---: | ---: |
| Isolated GQA | 0.00625000 | 0.00045683 | 0.0078125 |
| Projected branch | 0.00518135 | 0.00141243 | 0.03125 |
| Final residual output | 0.01162791 | 0.00137931 | 0.03125 |

Each mapping reached these same maxima; this does not assert bitwise equality
between mappings. The [validation record](parallelism_validation.json) retains
per-mapping counts, commands and source identities. All 510 frozen synthetic
arrays and 63 checkpoint arrays matched their hashes. The full workflow passed
**90 Mojo and 43 Python tests**, plus the maintained IR inspector, checkpoint
validation and twelve asynchronous 65-token sequences per configuration.

The standalone suite applies all mappings to 29 FP32 structural fixtures.
Coverage includes ragged tiles, future-token perturbations, exact cache/prefix
checks, poisoned partials, output guards, entirely empty causal pieces, and
a stable merge with maxima 1000/999. Missing split scratch rejects before
enqueue or cache mutation. Asynchronous calls cross the 15/16/17-row projection
boundary and finish with decode; every benchmark route also passes its actual
hot and ring24 correctness checks before timing.

### Screen and full workload matrix

The predeclared screen tests all four candidates at `(64,1024)`, `(64,4096)`
and full 1024, in both modes and four paired blocks. A gain requires every
block faster and median reduction above both 5% and the largest matching
control self-pair deviation. At the target `(64,4096)`:

| Candidate | Hot time / paired control | Ring24 time / paired control | Decision in both modes |
| --- | ---: | ---: | --- |
| BQ16 | 1.542x | 1.622x | Slower |
| BQ8 | 2.555x | 2.777x | Slower |
| Split4 | 0.650x | 0.567x | Faster |
| Split8 | 0.604x | 0.520x | Faster |

Across all 24 candidate/mode/workload cells, six qualify as faster, eleven
as slower and seven are inconclusive. Eight splits is the sole finalist:
both split candidates qualify at the target, and eight splits has the lower
worse-mode median ratio under the declared selection rule. There was no
direct paired split4-versus-split8 experiment. The smaller-query family did
not advance. All 2,400 observations remain in the
[screen record](parallelism_screen_run.json), [samples](parallelism_screen_samples.csv.gz)
and [derived table](parallelism_screen_summary.csv).

![Query-tile and KV-split screening ratios](parallelism_screen.png)

The independent full matrix compares eight splits with the integrated control
across all fifteen existing workloads, retaining 4,800 observations. Of thirty
workload/mode cells, **three qualify as faster, one as slower and twenty-six
are inconclusive**:

| Workload R,T | Hot latency reduction | Ring24 per-call reduction |
| --- | ---: | ---: |
| Decode T=1,16,64,256,1024,4096 | All inconclusive | All inconclusive |
| Full 16,16 | Inconclusive | Inconclusive |
| Full 64,64 | Inconclusive | **5.64% slower** |
| Full 256,256 | Inconclusive | Inconclusive |
| Full 1024,1024 | Inconclusive | Inconclusive |
| Full 4096,4096 | Inconclusive | Inconclusive |
| Chunk 4,64 | Inconclusive | Inconclusive |
| Chunk 16,256 | Inconclusive | Inconclusive |
| Chunk 64,1024 | Inconclusive | **31.63%** |
| Chunk 64,4096 | **39.69%** | **47.96%** |

For the target chunk, hot latency is **1.917 to 1.155 ms** and ring24 per-call
latency **1.949 to 1.014 ms**. Reductions use the median of within-block ratios;
the displayed absolute times are medians of block medians. These summaries
need not divide to exactly the same ratio.

Fifteen of thirty matching self-pair noise floors exceed 5%. Hot `(64,1024)`
shows a 23.16% point reduction but its 47.14% calibration floor leaves it
inconclusive. Several short hot cells have much larger variation; the maximum
floor is 255.70% at full 64. Those measurements cannot determine a crossover.
Decode executes the same G32 implementation in both arms and serves as a
control diagnostic. Full 1024 has only 1.29%/0.79% point reductions, while full
4096 has 1.27%/1.22% point slowdowns, all below their 5% decision floors.
No noisy cell was rerun to obtain a preferred outcome.

![Eight KV splits across the complete attention matrix](parallelism.png)

The [full record](parallelism_run.json), [samples](parallelism_samples.csv.gz)
and [derived table](parallelism_summary.csv) preserve every classification.
The run also binds the completed screen's hashes and selected finalist.

### What the split and merge cost

Four separate Metal captures compare control and split8 at the target chunk
and full 1024, with 25 measured iterations after ten warmups each. All 950
measured dispatch durations are retained and validated against the actual
enqueue sequence. Median active GPU durations are:

| R,T | Control GQA µs | Split kernel µs | Merge µs | Split plus merge µs |
| --- | ---: | ---: | ---: | ---: |
| 64,4096 | 1503.791 | 739.667 | 12.334 | 752.791 |
| 1024,1024 | 1612.125 | 1461.042 | 132.625 | 1589.376 |

The last column sums both stages within each iteration before taking the
median; it is not the sum of the two displayed medians. At the long chunk,
GQA including merge takes roughly half the control GQA duration. The merge
is 1.29% of total candidate active time, and GQA's combined share falls from
86.42% to 75.92%. At full 1024, merge consumes most of the split kernel's
saving. QKV and Wo together remain about 55% of active time in both captures.

![Active GPU stages including the split merge](parallelism_profile.png)

The [capture record](parallelism_profiles.json), [raw dispatch durations](parallelism_profile_samples.csv.gz)
and [derived stage table](parallelism_profile_summary.csv) preserve the
instrumented evidence. Compiler spill events report maximum event sizes of
48 bytes for control and 64 for split8 in both workloads, with 35 target events
per capture including warmup. This increase coexists with the chunk latency
gain; compiler event sizes alone do not measure spill traffic or its cost.
Optional hardware counters were not exported or analyzed.

These profiles diagnose where time moves. They are separate instrumented
captures, so paired unprofiled latency establishes the performance claims.
Do not subtract active durations from unprofiled latency to infer host overhead.
The results support sequence parallelism for the measured long cached chunk;
they do not establish measured occupancy, a universal dispatch policy, or a
hardware ceiling.

## Integrating QKV and Wo with FP32 attention

The existing projection work now runs through one public Mojo entrypoint,
`enqueue_attention_sublayer_integrated`. It combines packed QKV, the studied
Wo mapping, and the validated FP32 GQA paths through the residual addition.
This addresses a composition gap: the earlier attention experiments kept
Q/K/V as three separate rowwise projections even though faster projection
kernels were already available.

| New rows R | QKV | GQA | Wo |
| --- | --- | --- | --- |
| 1 | Packed rowwise | FP32 G32 decode | Rowwise |
| 2–15 | Packed rowwise | FP32 rolled MMA prefill | Rowwise |
| 16–4096 | Packed 8×16 MMA | FP32 rolled MMA prefill | Bias-free 8×16 MMA |

Sixteen rows is a conservative policy fixed before measurement, not a proven
optimal crossover across unmeasured sizes. The previous split64-H4 decode
mapping remains explicit; the earlier study did not establish a direct
G32/split crossover. The original enqueue and separate mappings remain useful
controls.

Packed projection produces `[R,1152]` with per-token `[Q | K | V]` ordering.
The new copy dispatch preserves BF16 bits while producing contiguous Q, K,
and V for the existing RoPE and cache consumers. Both its enqueue and execution
are included in timing. The block now has nine dispatches: RMSNorm, packed
QKV, unpack, Q RoPE, K RoPE, cache append, GQA, Wo, residual. GQA output still
rounds to BF16 before Wo; Wo output rounds to BF16 before the residual.

The copy requests 4×R×1152 bytes including reads and writes: 18 MiB at R=4096.
This is source-requested traffic, not measured DRAM traffic. Packed and
contiguous diagnostic buffers remain allocated. The integrated entrypoint can
use `AttentionWorkspace(..., fp32_materialized=False)`, avoiding the original
896 MiB probability allocation at full 4096, apart from a four-byte sentinel.
All allocation remains outside enqueue and timing. The combined 9-versus-3
comparison retains the baseline scratch allocation for both timing arms; the
storage saving is available when the integrated entrypoint is used alone.

### The additional value of the projection work

The first fresh paired comparison fixes GQA and Wo to the policy above in both
arms. Control 8 keeps separate rowwise Q/K/V; candidate 9 calls the integrated
entrypoint. Thus it measures QKV packing/tiling and the layout copy together,
with the previously optimized surrounding block as its control.

All 4,800 observations are retained from clean source `dc77016`. Across thirty
workload/mode comparisons, **19 qualify as faster, 11 are inconclusive, and
none qualify as slower**. All five full-prefill sizes qualify in both modes.
The `(16,256)`, `(64,1024)`, and `(64,4096)` chunks also qualify in both modes;
`(4,64)` is inconclusive.

| Workload | Hot reduction | Ring24 reduction |
| --- | ---: | ---: |
| Full R=T=16 | 37.03% | 25.68% |
| Full R=T=64 | 60.73% | 68.57% |
| Full R=T=256 | 70.98% | 73.59% |
| Full R=T=1024 | 70.20% | 70.67% |
| Full R=T=4096 | 58.54% | 58.53% |
| Chunk R=64, T=4096 | 24.19% | 23.97% |

The full-1024 hot paired control/candidate medians are 15.800 → 4.711 ms;
full-4096 hot is 79.796 → 33.080 ms. Reductions use the median of within-block
ratios; displayed times are medians of block medians. Their ratio need not
match the paired reduction, particularly in noisy short calls.

Decode qualifies only at hot T=1 (11.61%), hot T=256 (12.71%), and ring24 T=64
(10.26%). The remaining nine decode cells are inconclusive. The full-context
decode point estimates improve by 8.67% hot and 9.27% ring24, but their matching
noise floors are 35.63% and 33.05%. They do not support speed claims. This is
consistent with keeping the earlier packing result scoped to its measured
operation, rather than assuming it transfers equally to every complete call.

At fixed R=64, the median of the four paired ring24 latency savings is
619.72 µs at T=64, 623.81 µs at T=1024, and 617.89 µs at T=4096. The nearly
constant absolute saving, despite the shrinking percentage, fits the intended
mechanism: projection dimensions depend on new rows R, while attention also
grows with cached context T.

![Additional whole-block value from integrating QKV](projections.png)

[Complete paired table](projections_summary.csv), [run](projections_run.json),
and [all raw samples](projections_samples.csv.gz) retain calibration, ratios,
absolute timings, runtime identity, and block conditions.

### Combined gain over the original attention baseline

The second fresh comparison pairs integrated entrypoint 9 with original
variant 3: materialized FP32 GQA and separate rowwise Q/K/V and Wo. It measures
all selected mappings together through the residual addition. It retains
4,800 observations from the same clean source and finds **26 faster, four
inconclusive, and zero slower** workload/mode decisions.

| Workload | Original → integrated hot latency | Hot reduction | Ring24 reduction |
| --- | ---: | ---: | ---: |
| Decode R=1, T=4096 | 2.482 → 0.262 ms | 89.43% | 91.57% |
| Full R=T=1024 | 38.505 → 4.736 ms | 87.70% | 88.06% |
| Full R=T=4096 | 349.351 → 33.022 ms | 90.55% | 90.58% |
| Chunk R=64, T=4096 | 9.757 → 1.921 ms | 80.34% | 79.73% |

All full-prefill cases qualify in both modes. The four inconclusive cells are
hot decode T=1 and T=16, plus both `(4,64)` chunk modes. Hot T=16 has a favorable
median ratio but one block reverses direction, so it fails the rule. Short-call
variation is retained rather than hidden. [The complete table](integrated_summary.csv)
contains every ratio, matching calibration threshold and absolute time.

![Combined whole-attention gains](integrated.png)

The [run](integrated_run.json) and [raw observations](integrated_samples.csv.gz)
record hardware/software, source/binary/input hashes, both arm orders, runtime
Metal identity, and conditions before/after each block. Both timing runs use
four blocks, ten warmups and ten samples per arm; AC power, power mode 0 and
no reported thermal warning. The environment is macOS 26.6.2, Xcode 26.6,
Mojo 1.0.0 and MAX 26.5.0. The fixed prefix and ring24 meaning remain as
described in the baseline: two sign patterns across 24 distinct allocations,
not a 24-layer model. Checkpoint inputs establish correctness separately.

### Numerical and data-flow checks

The integrated implementation passed **87 Mojo tests and 41 Python tests**,
all 510 frozen synthetic arrays, and 63 frozen arrays for three existing
checkpoint-derived first-layer cases. Both packed kernels consume exactly
the upstream normalized tensor in isolated tests. Composition then starts
from original X, including candidate-generated cache prefixes, full/chunked
prefill, final decode and reset. All tolerances remain unchanged.

| Check | Maximum synthetic scaled error | Maximum checkpoint scaled error | Fixed limit |
| --- | ---: | ---: | ---: |
| Isolated packed QKV, both mappings | 0.006250 | 0.003226 | 0.0078125 |
| Integrated projected branch | 0.005181 | 0.001412 | 0.03125 |
| Integrated final output | 0.011628 | 0.001379 | 0.03125 |

Scaled error is `abs(got-want)/(1+abs(want))`. The existing isolated GQA gate
remains 0.0078125. The branch is checked separately so the residual cannot
conceal its error. Cache prefix, appended source bits and unused capacity
are exact checks. The new multi-row handoff regression checks signed zeros,
subnormals, extreme BF16 patterns and guard elements. Projection intermediates
are poisoned in composition tests. The actual benchmark routes pass hot and
ring24 checks, including both sides of the 15/16-row boundary.

Twelve asynchronous 65-token sequences pass for each of ten configurations.
The new control and integrated configurations exercise 15,16,17,16,1-row calls
on the same stream with reusable buffers and no materialized probability
storage. [integrated_validation.json](integrated_validation.json) retains the
commands, source hashes, counts and numerical maxima. No kernel, precision
policy or threshold was retuned after the paired measurements started.

### Where time goes after integration

Eight separate captures compare control 8 and integrated entrypoint 9 at four
workloads. All 2,090 measured dispatch durations are retained; 37 segmented
dispatches were coalesced using validated command identities, excluding
preemption gaps before stage assignment. The table reports median active GPU
time in microseconds. QKV totals are summed within each measured iteration
before taking the median, so the layout copy is included correctly.

| R,T | Separate QKV total µs | Packed QKV + unpack µs | Unpack alone µs | Integrated GQA share of active time |
| --- | ---: | ---: | ---: | ---: |
| 1,4096 | 133.52 | 81.04 | 12.69 | 55.85% |
| 1024,1024 | 12576.79 | 1523.92 | 133.83 | 35.63% |
| 4096,4096 | 52539.86 | 6103.48 | 591.50 | 64.21% |
| 64,4096 | 733.79 | 113.88 | 10.13 | 86.39% |

![Stage costs after integrating the projection work](integrated_profile.png)

Unpack accounts for 2.47%, 2.96%, 1.81% and 0.56% of recorded active time in
those four integrated captures respectively. Its cost is visible but small.
At full 1024, packed QKV itself is 30.59% and Wo is 23.97%; projections together
still account for about 55% of active time. At full 4096 their combined share
falls to about 30%, while GQA is 64%. The long cached chunk is much more
strongly dominated by GQA.

These are instrumented diagnostics, not additional paired speed claims.
In particular, decode's instrumented active durations are larger than its
unprofiled whole-call timing; they cannot be subtracted from that timing to
estimate CPU overhead. The unchanged GQA/Wo stages are similar in these
control/candidate captures, but profiling and clock effects remain possible.
Use the latency experiments for the gains and the profiles for work ownership
and remaining stage costs.

Both prefill variants report a maximum compiler spill event of 48 bytes,
with 35,20,35 target events for full 1024, full 4096 and the long chunk. Neither
decode capture reports a target spill event. These event sizes/counts are
compiler evidence, not measured spill traffic or proof of a bottleneck.
Optional counter analysis is absent. [The profile table](integrated_profile_summary.csv),
[record](integrated_profiles.json), and [all dispatch samples](integrated_profile_samples.csv.gz)
retain the evidence and capture conditions.

## Baseline whole-block latency

These are control-arm medians of four block medians, in milliseconds per
sublayer. Both timing arms run the same implementation, providing noise
calibration rather than an optimized comparison.

| Workload | R | T | Hot | Ring24 per call |
| --- | ---: | ---: | ---: | ---: |
| Decode | 1 | 4096 | 2.494 | 2.689 |
| Full prefill | 1024 | 1024 | 39.127 | 39.037 |
| Full prefill | 4096 | 4096 | 355.793 | 355.288 |
| Cached chunk | 64 | 4096 | 9.590 | 9.597 |

![Whole-sublayer latency](latency.png)

The complete 15-shape matrix and calibration ranges are in [summary.csv](summary.csv).
Keep the short cases in perspective: the largest self-pair deviations were
177.3% for hot `(1,64)`, 171.4% for hot `(4,64)` and 54.4% for hot `(16,16)`.
All samples are retained. These points cannot support small optimization
claims under this run's decision rule. Open plot marks identify self-pair
variation above 5%. The four workloads in the table retained the 5% minimum
decision threshold in both modes. Future comparisons need fresh matching
calibration; this run's calibration must not be imported into another run.

## Where the baseline GPU time goes

The following percentages divide a stage's total recorded active duration
by the total active duration in that same capture. They are not percentages
of the separate host-to-completion latency measurement.

| Workload | Main active GPU work | Median stage durations |
| --- | --- | --- |
| Decode `(1,4096)` | Softmax 48.1%, PV 46.5% | 1.110 ms, 1.009 ms |
| Full `(1024,1024)` | All four projections 58.3%; Q 25.6%, Wo 25.4% | Q 9.813 ms, Wo 9.725 ms |
| Full `(4096,4096)` | QK 40.5%, PV 28.6% | 143.409 ms, 101.088 ms |
| Chunk `(64,4096)` | PV 39.9%, QK 31.8%, softmax 14.8% | 3.833 ms, 3.088 ms, 1.429 ms |

![Time spent in each kernel](profile.png)

The [stage table](profile_summary.csv) retains medians and ranges. Each
shape has one separate instrumented capture, with 100, 25, 10 and 25 measured
iterations respectively. Stage names follow verified enqueue order. Instruments
fragmented 54 full-context and 17 chunked dispatches; their execution segments
were joined before assigning stages. Durations exclude preemption and host gaps.
Adding these stage medians would not reconstruct whole-block latency.

The source explains why one optimization will not solve every workload:

- QK assigns one score to a thread; PV assigns one output component to a
  thread. Their serial reduction work grows with the visible key count.
- Softmax assigns an entire row to one thread. Decode has only fourteen
  active softmax threads, each scanning up to 4096 scores. More query rows
  provide more independent softmax work, changing its relative importance.
- Each baseline projection uses one SIMD group per output dot product.
  It does not yet use the existing prefill mapping that reuses an input tile
  across several output columns and rows.

Device-wide counter medians reinforce this as a hypothesis rather than a
per-kernel diagnosis: decode's Kernel Occupancy was 1.46%, while full-1024
prefill was 58.61%. The corresponding Last Level Cache Limiter medians were
2.25% and 100%. These samples cover the enclosing target window, include
other GPU activity and are not measured DRAM bandwidth. No target compiler
spill event was reported in these four captures; this is a capture-bounded observation.

Full-4096 optional counter analysis is explicitly unavailable. Its XML export
was stopped after growing beyond 4 GiB, because the existing analyzer loads a
whole table into memory. The original trace and complete stage-timing exports
are preserved externally. No partial counter XML contributes to the report,
and the absent counter summaries are not recorded as zero.

## Work and storage

The number of visible query-key pairs per head is
`C = R(T-R) + R(R+1)/2`. The reference allocates `4*14*R*T` bytes for FP32
scores, overwritten in place by probabilities. QK plus PV performs roughly
`4*64*14*C` floating-point operations; the four projections perform
`2*R*896*(1152+896)`. Softmax, masking and elementwise operations are additional.

| R | T | FP32 scratch | Projection GFLOPs | QK+PV GFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 224 KiB | 0.00367 | 0.01468 |
| 1024 | 1024 | 56 MiB | 3.758 | 1.881 |
| 4096 | 4096 | 896 MiB | 15.032 | 30.072 |
| 64 | 4096 | 14 MiB | 0.235 | 0.932 |

Weights, norm and biases occupy 3,674,112 bytes per ring entry; both KV caches
together occupy `512*T` bytes. These are source-derived counts and allocated
storage, not measured memory traffic. The materialized FP32 path is an
inspectable accuracy baseline with a substantial memory cost.

## Contained Wo results

Wo is the learned, bias-free matrix that mixes the fourteen heads' outputs
back into the hidden state: `A[R,896] @ Wo[896,896].T`. The candidate changes
only how that matrix multiplication is assigned to the GPU. Both versions
accumulate in FP32 and round to BF16 before the existing residual addition.
The attention precision policy, other projections, cache, buffers and twelve
dispatches remain fixed. `wo_mma=True` selects the candidate; benchmark IDs
3 and 4 distinguish rowwise and MMA Wo while both execute GQA route 3.

The rowwise mapping assigns one output dot product to a 32-lane SIMD group,
with one FP32 accumulator per lane and a final group reduction. The MMA
mapping assigns an 8×16 output tile to that group, using two distributed 8×8
matrix fragments per K=8 phase and four FP32 accumulators per lane. It reuses
operands across rows and output columns without shared operand storage or
block barriers. At R=1024 this changes 917,504 rowwise groups into 7,168 tile
groups. These counts describe work ownership, not measured memory traffic.

### Accuracy and screening

All existing numerical gates passed: GQA `atol=rtol=0.0078125`, Wo/projected
branch/final output `atol=rtol=0.03125`, and exact BF16 cache bits. No tolerance
changed. Each isolated Wo comparison consumes the exact upstream BF16
attention tensor; full and chunked comparisons start from the original X.
The [validation record](wo_validation.json) retains the commands, source
hashes and individual comparisons.

| Dataset | Maximum isolated Wo scaled error | Maximum composed branch error | Maximum residual output error |
| --- | ---: | ---: | ---: |
| 17 synthetic cases | 0.00390625 | 0.00516796 | 0.00781250 |
| 3 checkpoint cases | 0.00023524 | 0.00141243 | 0.00137931 |

Scaled error is `abs(got-want)/(1+abs(want))`. The complete workflow passed
81 Mojo tests and 38 Python tests and verified all 510 frozen synthetic arrays.
The existing checkpoint prefix reproduced its 63 arrays. Normal asynchronous
execution passed twelve 65-token sequences on each of five configurations,
including the new mapping. Both benchmark arms additionally poison the actual
cache suffix and attention/projected/output buffers, then check the projected
branch, final output and exact cache prefix/suffix for every distinct layer
allocation before timing.

The five-shape [screen](wo_screen_summary.csv) retained 1,600 observations.
Its six prefill workload/mode gains met the predeclared advancement rule, so
the candidate proceeded to the full matrix with fresh self-pair calibration.
The screen is selection evidence; the following results come from that full run.

### Whole-block comparison

Across thirty workload/mode comparisons, **13 are faster, 15 inconclusive and
2 slower** under the existing four-block rule. A gain requires all four paired
blocks to be faster and their median reduction to exceed the larger of 5%
and matching self-pair variation. The [complete table](wo_summary.csv) includes
both arms, all ratios, calibration thresholds and decisions.

| Full prefill R=T | Hot latency reduction | Ring24 per-call reduction |
| ---: | ---: | ---: |
| 16 | Inconclusive | 23.66% |
| 64 | 31.99% | 32.44% |
| 256 | 31.17% | 31.24% |
| 1024 | 22.49% | 22.56% |
| 4096 | 10.43% | 10.43% |

For full 1024, paired-control and candidate medians are 38.409 → 29.774 ms
hot and 38.267 → 29.627 ms ring24. For full 4096 they are 348.731 → 312.373 ms
hot and 354.276 → 317.393 ms ring24. These are medians of block medians;
reported reductions use the paired block ratios, not the ratio of those
displayed medians.

Decode establishes no gain. Hot T=16 is 11.91% slower and hot T=1024 is
5.03% slower; the other ten decode comparisons are inconclusive. Some short
hot self-pairs vary substantially, reaching about 190%. All observations are
retained, including hot full-16's inconclusive result despite its lower median.

For chunked prefill, `(16,256)` improves 15.98% in ring24; `(64,1024)` improves
15.37% hot and 13.94% ring24. At `(64,4096)` hot improves 6.02%, but ring24's
4.97% reduction is inconclusive under the fixed 5% floor. That last cell had
barely qualified in the screen, illustrating why screening does not replace
confirmation. Both `(4,64)` modes and hot `(16,256)` are inconclusive.

![Whole-block effect of changing only Wo](wo.png)

### What changed inside the block

The eight separate stage captures compare both mappings at four workloads.
These are median active GPU durations in microseconds, useful for diagnosing
the change. They are not paired latency measurements.

| R | T | Rowwise Wo µs | MMA Wo µs |
| ---: | ---: | ---: | ---: |
| 1 | 4096 | 15.958 | 44.917 |
| 1024 | 1024 | 9724.230 | 1088.854 |
| 4096 | 4096 | 40667.333 | 4303.666 |
| 64 | 4096 | 559.417 | 89.938 |

![Stage comparison with both Wo mappings](wo_profile.png)

Wo becomes roughly nine times faster in the two full-prefill captures, while
the surrounding stage medians remain similar. Its work grows with R; QK/PV
work grows with R and the visible context. This explains why the whole-block
benefit decreases at long context even though Wo improves substantially.
The 896 MiB FP32 scratch allocation at full 4096 is unchanged.

At R=1, Wo itself takes about 2.8 times longer. Only one of the MMA tile's
eight rows is useful, and it exposes 56 groups versus 896 rowwise groups.
The unused row capacity and smaller pool of independent groups are consistent
with the slowdown; this experiment does not isolate their individual costs.
The result argues against an unconditional MMA default. It does not establish
a general dispatch threshold.

In the MMA-Wo captures, stage shares of **total recorded active GPU time** are:

| Workload | Remaining main stages | Wo share |
| --- | --- | ---: |
| Decode `(1,4096)` | Softmax 48.7%, PV 44.6% | 2.0% |
| Full `(1024,1024)` | Q projection 33.1%, QK 28.6%, PV 21.9% | 3.7% |
| Full `(4096,4096)` | QK 45.2%, PV 32.0% | 1.4% |
| Chunk `(64,4096)` | PV 41.2%, QK 34.8%, softmax 14.3% | 1.0% |

These shares use summed raw durations within each capture. They do not divide
by host-to-completion latency, and stage medians are not added to reconstruct
whole-block time. The [profile table](wo_profile_summary.csv) retains ranges.
No target compiler spill event was reported in these captures; that does not
prove every invocation is spill-free. Optional counter tables were not exported
or analyzed for this comparison; missing values are not zero.

## Contained FP32 decode results

The two candidates change GQA ownership while retaining FP32 scaled scores,
online softmax state, weights and accumulation. BF16 Q/K/V, cache and output,
rowwise Wo and all surrounding stages remain fixed. G32 divides a head's
sequence among 32 SIMD groups and merges inside one threadgroup; split64-H4
uses 64 sequence pieces and shares K/V across up to four related query heads,
then launches a separate merge. See the [predeclared design and gates](../../docs/attention-sublayer.md#contained-fp32-decode-comparison).

### Accuracy and discrimination

The full workflow passed **83 Mojo tests and 39 Python tests**, verified all
510 frozen synthetic arrays and reproduced 63 checkpoint arrays. Twelve
asynchronous 65-token sequences passed on each of seven configurations,
including both candidates without materialized scratch. All numerical gates
and exact cache requirements remain unchanged.

There are 408 strict comparisons against the pinned upstream FP32 attention
outputs using exact upstream Q/K/V at predefined cache prefixes. The largest
scaled error is 0.003649635 for synthetic inputs and 0.00012019231 for checkpoint
inputs, within the 0.0078125 GQA gate. Existing standalone edge fixtures add
NaN guards, tied and extreme scores, ragged/empty splits, cancellation and head
mapping checks against materialized FP32 Mojo. Those supplement the primary
upstream comparison. [decode_validation.json](decode_validation.json) retains
all commands, hashes and numerical results.

A diagnostic probe on existing hard seed 887 changes only the isolated decode
comparison back to the older BF16 score boundary. It fails 24 of 30 strict
prefix checks, with maximum scaled error 0.1813063; the FP32 version passes all
30, maximum 0.002020202. This shows the checks distinguish the agreed precision
policy. It does not make the historical BF16 policy invalid for every backend.
Composition gates also pass for all synthetic/checkpoint cases and R>1 fallback.

### Whole-block comparison

Seven of eight candidate screening comparisons qualified, advancing the fixed
three-way matrix. The full run uses fresh self-pair calibration and retains
**13 faster, 11 inconclusive and zero slower** decisions across 24 comparisons.
G32 accounts for eight gains and split64-H4 for five. The candidates are each
paired against materialized FP32; they are not paired directly with one another.
No G32/split crossover or default selector follows from this experiment.

| T | G32 hot reduction | G32 ring24 reduction | Split64-H4 hot reduction | Split64-H4 ring24 reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 13.62% | 14.27% | 17.40% | Inconclusive |
| 16 | Inconclusive | Inconclusive | Inconclusive | Inconclusive |
| 64 | Inconclusive | 16.73% | Inconclusive | Inconclusive |
| 256 | Inconclusive | 24.16% | Inconclusive | Inconclusive |
| 1024 | 66.74% | 65.40% | 68.65% | 61.42% |
| 4096 | 88.79% | 90.67% | 88.97% | 89.47% |

At T=4096, G32 paired-control/candidate medians are 2473.250 → 274.750 µs hot
and 2680.635 → 250.479 µs ring24 per call. Split64-H4 has separate paired
controls: 2492.000 → 274.500 µs hot and 2693.865 → 283.875 µs ring24.
Reported reductions use paired block ratios; the displayed times are medians
of block medians. Their ratio need not equal the paired result. The complete
[table](decode_summary.csv) includes all four ratios, absolute times and rules.

Short hot cases remain noisy: matching self-pair thresholds reach 105.2% at
T=16, 163.8% at T=64 and 46.2% at T=256. All samples are retained. For example,
G32 hot T=256 has a 42.56% median paired reduction but is inconclusive because
it does not exceed its matching calibration threshold.

![FP32 decode whole-block comparison](decode.png)

### What the profiles establish

At T=4096 the materialized control spends 94.5% of its total recorded active
GPU time in serial softmax/PV. Their stage medians are 1108.959 and 1002.812 µs.
The fused candidates distribute that sequence work and retain online FP32
state instead of scanning a materialized score/probability array. Their whole
block executes 10 dispatches (G32) or 11 (split64), versus 12 for the control.
A production single-row call needs no quadratic scratch for these routes;
the comparison instrument still allocates its common control scratch outside
timing, so it does not measure an allocation benefit.

![Decode stages with all three mappings](decode_profile.png)

These separate captures have significant variation in unchanged stages:
Q projection at T=4096 has medians 15.146, 23.604 and 35.145 µs for control,
G32 and split64 respectively. A few large durations also distort summed stage
shares: G32 T=64 K projection has a 35.126 µs median but a 2113.541 µs maximum.
The [profile table](decode_profile_summary.csv) retains every range. All
captures passed provenance and exact dispatch-sequence checks; eight measured
control T=4096 dispatches had segmented execution and were coalesced before
stage assignment. No target spill event was reported; optional counters were
not analyzed. Power/thermal checks passed but do not pin clocks or identify
the cause of cross-capture variation.

Consequently the profiles support a qualitative change in remaining work:
projections and other stages become material once the long serial attention
passes disappear. They do not support a precise cross-capture kernel speedup,
a direct candidate ranking, or summing medians to reconstruct block latency.
The paired measurements establish the speed claims.

## Contained FP32 prefill results

This comparison changes only GQA, with the earlier MMA Wo mapping fixed in
both arms. Benchmark 4 uses materialized FP32 attention; benchmark 7 uses the
FP32 adaptation of the earlier 32×32 rolled-QK MMA design. The measurements
are whole attention calls from RMSNorm through residual, not isolated GQA.
They do not multiply or inherit the separate Wo experiment's speedups.

### What the tile changes

A threadgroup owns one query head and 32 query rows. Its four SIMD groups
each own eight rows, with sixteen FP32 output accumulator variables per lane. It
streams 32 keys/values at a time, reusing the shared K/V tile across its
query rows. Matrix operations calculate QK and PV; online maximum,
denominator and numerator state carry information between KV tiles.
For R=4096 there are 14×128=1,792 threadgroups. That is a work-ownership
count, not measured occupancy.

The candidate preserves the prior rolled QK loop and four barriers per KV
tile. Q/K operands remain BF16 with FP32 accumulation, and the scaled score
stays FP32. Tile weights also remain FP32 through PV. Stored BF16 V is widened
losslessly for the FP32 PV matrix operation; only final attention output
rounds to BF16. This preserves the agreed precision policy while changing
reduction order and online normalization.

Shared K/V storage totals 8 KiB and FP32 scores/weights another 8 KiB, for
16 KiB per block versus the older BF16-weight mapping's 14 KiB. The new path
needs no global score/probability matrix: at full R=T=4096 this removes the
requirement for 896 MiB of attention scratch. The comparison instrument still
owns the common control scratch outside timing; it does not measure allocation
savings. Avoiding that matrix also does not mean every input is read only once.

Sublayer route 6 is explicit: R>1 launches tiled FP32 prefill and R=1 launches
FP32 G32, returning actual route 6 or 4 respectively. Both work without
materialized scratch. Wo is selected independently; the prefill benchmark
requires R>1 and fixes MMA Wo in both arms. Public defaults remain route 3 and
rowwise Wo; no automatic crossover is introduced.

### Numerical results

The full workflow passed **86 Mojo tests and 40 Python tests**, verified all
510 frozen synthetic arrays, and reproduced all 63 checkpoint arrays. The
new isolated operation passed 42 synthetic and nine checkpoint comparisons
on exact upstream Q/K/V, including full and suffix attention. Compositions
from original X pass with both Wo mappings and exact cache checks.

| Dataset | Maximum GQA scaled error | Maximum composed branch error | Maximum residual output error |
| --- | ---: | ---: | ---: |
| Synthetic | 0.00625000 | 0.00516796 | 0.00781250 |
| Checkpoint | 0.00045683 | 0.00141243 | 0.00137931 |

The attention gate remains 0.0078125; branch/final gates remain 0.03125.
Scaled error is `abs(got-want)/(1+abs(want))`. Existing standalone prefill
fixtures add 29 edge cases checked against materialized FP32 Mojo, supplementing
the pinned upstream authority. All pass twelve poisoned-output repetitions in
normal execution, along with causal future perturbation and full/suffix tests.
An 8×8 FP32 identity product with non-BF16-representable operands returns zero
maximum error, guarding against silently narrowing matrix operands.

Twelve asynchronous 65-token sequences pass on eight composed configurations.
The new route uses 33,31,1-row calls without intermediate synchronization or
materialized scratch; the previous seven configurations keep their decode
coverage. [prefill_validation.json](prefill_validation.json) retains commands,
source hashes and individual numerical comparisons. No tolerance or frozen
oracle array changed.

### Screen and full matrix

All six screening cells qualify: about 20% lower whole-attention time at full
256, 47–48% at full 1024 and 72–73% for chunk (64,4096), across hot/ring24.
The screen retains 960 observations and is selection evidence. The full run
uses fresh calibration; it supplies the following final decisions.

The full comparison retains **16 faster, two inconclusive and zero slower**
decisions across eighteen workload/mode cells. A gain requires every paired
block to be faster and the median reduction to exceed the larger of 5% and
matching self-pair variation.

| Workload R,T | Hot latency reduction | Ring24 per-call reduction |
| --- | ---: | ---: |
| Full 16,16 | Inconclusive | 7.84% |
| Full 64,64 | 7.42% | 8.46% |
| Full 256,256 | 20.18% | 21.11% |
| Full 1024,1024 | 47.10% | 47.30% |
| Full 4096,4096 | 74.59% | 74.33% |
| Chunk 4,64 | 9.60% | 21.14% |
| Chunk 16,256 | Inconclusive | 23.30% |
| Chunk 64,1024 | 55.80% | 55.37% |
| Chunk 64,4096 | 72.70% | 72.04% |

At full 1024, paired-control/candidate medians are 30.875 → 16.333 ms hot
and 30.339 → 15.985 ms ring24 per call. At full 4096 they are
323.407 → 82.183 ms hot and 324.106 → 82.921 ms ring24. Chunk (64,4096)
changes from 9.570 → 2.644 ms hot and 9.374 → 2.622 ms ring24.
Displayed times are medians of block medians; percentage reductions use
paired block ratios. [prefill_summary.csv](prefill_summary.csv) retains the
complete times, ratio ranges, calibration thresholds and decisions.

Hot full-16 has a 3.68% median paired reduction and a 60.67% calibration
threshold, so it is inconclusive. Hot chunk (16,256) has a 17.85% median
reduction but one block is 4.59% slower, failing the all-four-block rule.
These observations are retained. A lower absolute median alone does not
establish a gain under this protocol.

![Whole-block effect of FP32 prefill tiling](prefill.png)

### Remaining active GPU work

The six separate stage captures validate 12 dispatches per materialized call
and 10 per tiled call. The table below describes only the candidate captures.
Shares divide each stage's summed recorded active duration by the total active
duration in that same capture; they are not percentages of whole-call latency.

| R,T | Q projection median µs | Fused GQA median µs | Q active share | GQA active share |
| --- | ---: | ---: | ---: | ---: |
| 1024,1024 | 10789.917 | 1721.750 | 63.1% | 10.0% |
| 4096,4096 | 46004.938 | 22529.146 | 52.6% | 25.7% |
| 64,4096 | 645.917 | 1785.500 | 22.6% | 65.3% |

![Remaining stages after FP32 prefill tiling](prefill_profile.png)

In the full-4096 control, QK/PV account for 75.6% of recorded active time.
With tiling, Q projection becomes the largest stage in both full-prefill
captures. Long cached chunks still spend most active time inside GQA. This
is why one optimization target need not serve every workload.

Unchanged stages still vary between separate captures. For example, Q at
full 1024 has a 11987.915 µs control median and 10789.917 µs candidate median;
chunk Q changes from 565.708 to 645.917 µs. Consequently these profiles explain
remaining work but do not supply precise causal kernel-speedup estimates.
The paired latency experiment establishes the gains. The
[profile table](prefill_profile_summary.csv) retains all stage medians/ranges.
There are 352 measured dispatches with segmented execution across these
captures; all segments were coalesced using validated command identities,
excluding preemption gaps before stage assignment.

Each tiled capture reports a maximum compiler spill size of **48 bytes per
event**, with 35,20,35 target events respectively. The materialized captures
report no target spill event. Event sizes/counts are compiler evidence, not
measured spill traffic, and absence of an event does not prove spill-free
execution. The 48-byte maximum is also what the earlier BF16 rolled-QK study
reported, but it does not establish identical spilled values or their cost.
No optional counter tables were exported or analyzed for this comparison.

## What the earlier experiments teach us

The [linear prefill study](../linear_prefill/README.md) already established
that sharing operands across token rows can make Apple MMA effective, while
tiny row counts can lose. This Wo experiment tests that mechanism at N=896,
without bias, and inside the full attention block. It does not inherit the
packed-QKV study's N=1152 crossover or compare its old absolute times with
this run. The block's remaining stages limit how much a faster projection can
improve the whole call.

The [decode study](../gqa_decode/README.md) showed that online softmax alone
was not the end of the optimization: distributing the KV sequence and reusing
K/V across related heads helped further. It also found that reducing
exponential calls did not improve the stronger controls. The completed FP32 decode
comparison reuses those ownership designs and the existing tests while
keeping scores FP32 and checking against the selected FP32 reference. The old
timings establish results for those older
paths, not speed claims for an FP32 adaptation.

The [prefill study](../gqa_prefill/README.md) showed the value of query tiling
and matrix execution. Its follow-up found that rolling the QK reduction
reduced reported compiler spills and improved a subset of workloads; removing
barriers, changing accumulator representation and adding head reuse did not
produce general gains. The completed FP32 prefill comparison retains those
lessons about live state and query ownership. It also preserves FP32 scores
and softmax weights through PV: copying the old MMA path's BF16 tile-weight
cast would change the numerical policy again.

The projection integration completes the next step those profiles identified.
The kernel already existed; the missing work was wiring packed output into
RoPE/cache consumers, retaining the numerical boundaries, and measuring the
result with Wo and FP32 GQA already present. The large incremental gains above
show why composition should precede another round of isolated kernel tuning.

The completed parallelism experiment follows the cost exposed by integration:
GQA occupied 86% of active time for `(64,4096)` but exposed only 28 primary
threadgroups. The two tested ways of creating more groups behaved differently.
Smaller query tiles lost, consistent with reduced reuse and more K/V tile
loads in the source; no hardware counter identifies that as the physical
bottleneck. KV splitting retained the larger tile and improved this chunk
even after partial writes and merge. The earlier decode work supplied a useful
ownership idea, while the new whole-block measurements established its scope.

There are now two useful directions for later contained work. For full 1024,
QKV and Wo together still consume about 55% of active time. Their shared 8x16
linear MMA mapping is a concrete target for operand reuse and tile ownership.
For the long cached chunk, split GQA plus merge still consumes about 76% of
active time; a later GQA study should use split8 as an additional control and
explain which live state or data movement it reduces. Full 4096 already has
many query tiles, and the split8 comparison showed no qualifying gain there.

This study does not justify widening the split count or combining candidates
without a new bounded question. Repeating the older head-reuse or barrier
ablations also needs a fresh reason after their mixed results. Preserve FP32
scores/weights, the existing gates, exact upstream-input checks, and the
integrated whole-attention measurement in any follow-up.

These results show that the original materialized paths were far from the
best mappings tested here. They do not establish a hardware ceiling, a universal
dispatch policy, or full-decoder/token-generation performance.

## Reproduction and evidence

The baseline latency run used clean source `8d8c8540c5f9de2d7d7cf16fc00f502702a3a941`
on 2026-09-06, 16:22:57–16:53:05 UTC. All four profile binaries use that same
source. Hardware was Apple M4 Pro / Metal, Mac16,7 with 24 GiB memory;
Mojo 1.0.0, MAX 26.5.0, macOS 26.6.2 and Xcode 26.6. Every latency block and
capture recorded AC power, Low Power Mode off and no thermal/performance warning.
These checks do not pin GPU clocks or exclude background activity. The Wo
comparison uses the same hardware/software configuration, with independent
source, binary and condition records below.

The workload uses frozen synthetic seed 53 with a CPU-derived cache prefix.
Ring24 owns 24 distinct weights, inputs and caches with two sign patterns;
it shares scratch/output and is not a decoder stack. Baseline timing includes the host
length rewind, twelve enqueues, cache append and completion. Allocation,
fixture reads, uploads, prefix setup and correctness checks are excluded.
Each call overwrites the same suffix, keeping R and T fixed. There are ten
warmups and ten samples per arm in four blocks; blocks two and three reverse
workload and arm order.

[run.json](run.json) and [samples.csv.gz](samples.csv.gz) retain all 2,400
latency observations and their provenance. [profiles.json](profiles.json)
and [profile_samples.csv.gz](profile_samples.csv.gz) retain all 1,920 measured
dispatch durations, capture identities, selected counters and the explicit
counter-analysis omission. The source of the curator is hashed in that record.
The post-measurement curation/plot changes handle absent optional counters and
mark elevated calibration variation; they change no measured engine code.
All 38 Python checks pass with the retained evidence, including duplicate and
missing-dispatch rejection for both attention profile grids.

The Wo screen and full run use clean source
`07984fedafcbe1c1260c37c86ff708c5098cfa18`. They ran on 2026-09-06 at
17:55:02–18:03:41 and 18:04:43–19:04:24 UTC respectively, retaining 1,600 and
4,800 observations in [wo_screen_run.json](wo_screen_run.json),
[wo_screen_samples.csv.gz](wo_screen_samples.csv.gz), [wo_run.json](wo_run.json)
and [wo_samples.csv.gz](wo_samples.csv.gz). All eight profile binaries use that
same clean source. [wo_profiles.json](wo_profiles.json) and
[wo_profile_samples.csv.gz](wo_profile_samples.csv.gz) retain 1,200 measured
dispatch durations: 25, 10, 5 and 10 iterations per variant across the four
workloads, after ten warmups each. All eight captures passed provenance and
dispatch-sequence validation. The default Metal System Trace was used without
optional counter-table export; full traces remain external. Every timing block
and capture recorded AC power, Low Power Mode off and no reported warning.

The same fixed-prefix, hot/ring24 and paired-block protocol applies to Wo.
The experiment retains the original baseline figures separately and uses fresh
controls for its comparisons. Post-measurement changes curate evidence, update
reporting and check retained records; they do not change the measured Mojo
engine or its numerical tests. All five tables and five figures regenerate
byte-for-byte from the retained files.

The decode screen and full run use clean source
`d2a320f0705e0459fd1e8c3c8c50b76a745f1b47`, on 2026-09-06 at
20:22:52–20:23:38 and 20:23:39–20:25:13 UTC. They retain 960 and 2,880
observations in [decode_screen_run.json](decode_screen_run.json),
[decode_screen_samples.csv.gz](decode_screen_samples.csv.gz),
[decode_run.json](decode_run.json) and [decode_samples.csv.gz](decode_samples.csv.gz).
The six captures use that same clean source and retain 3,300 durations in
[decode_profiles.json](decode_profiles.json) and
[decode_profile_samples.csv.gz](decode_profile_samples.csv.gz), with 50 measured
iterations and 20 warmups per capture. Both timing modes and each candidate's
actual route are validated before timing. All conditions checks passed.
Post-measurement reporting changes do not alter the measured engine or gates.
All eight tables and eight figures regenerate from retained raw evidence;
39 Python checks validate the retained records, including missing/duplicate
dispatch rejection for each profile grid.

The prefill screen and full run use clean source
`17720c294ec98a5ba38da004e38ec72eeed6372f`, on 2026-09-06 at
20:56:23–21:02:22 and 21:02:22–21:47:00 UTC. They retain 960 and 2,880
observations in [prefill_screen_run.json](prefill_screen_run.json),
[prefill_screen_samples.csv.gz](prefill_screen_samples.csv.gz),
[prefill_run.json](prefill_run.json) and [prefill_samples.csv.gz](prefill_samples.csv.gz).
The six profile binaries use the same clean source. [prefill_profiles.json](prefill_profiles.json)
and [prefill_profile_samples.csv.gz](prefill_profile_samples.csv.gz) retain
1,320 active durations: 25,10,25 measured iterations per variant at full 1024,
full 4096 and chunk (64,4096), with ten warmups each. Every block/capture
recorded AC power, Low Power Mode off and no reported thermal/performance
warning. Those checks do not pin clocks or exclude background activity.

All 40 Python checks pass against the retained evidence, including duplicate,
missing-dispatch and source-identity checks for the added profile grid. All
11 tables and 11 figures regenerate byte-for-byte. Post-measurement changes
curate evidence and reporting; the measured engine and numerical gates remain
unchanged. Full traces/XML, binaries and oracle arrays remain outside Git.

The projection-only and combined integration comparisons use clean source
`dc77016af29ef0087807b22ed42a0aeca7cde14f`, on 2026-09-06 at
22:32:05–22:46:33 and 22:46:33–23:33:27 UTC respectively. Each retains 4,800
observations in its `projections_` or `integrated_` run/sample files above.
The eight integration captures use that same source and retain 2,090 active
dispatch durations: 50,25,10,25 measured iterations per variant at decode
4096, full 1024, full 4096 and chunk (64,4096), after ten warmups each.
Capture provenance, dispatch identities, and optional-counter absence are
validated. Source hashes match the numerical validation record.

All 41 Python checks passed with the retained integration evidence. At that
stage, the attention study reproduced **14 tables and 14 figures byte-for-byte**
offline, including every prior figure. The final report/evidence changes do
not alter the measured kernels, benchmark or numerical gates. Full traces,
XML, binaries and generated fixtures remain outside Git.

The GQA parallelism screen and full matrix use clean source
`5778641918e256604e093f007cebcc1aa3606523`, on 2026-09-07 at
01:00:15–01:04:03 and 01:04:04–01:11:48 UTC respectively. They retain 2,400
and 4,800 observations in the `parallelism_screen_` and `parallelism_` files
linked above. The full run uses the sole screen finalist and binds both
screen hashes. All four profile binaries use that same clean source and
retain 950 dispatch durations. Capture, build and numerical validation source
hashes agree. Every block and capture recorded AC power, Low Power Mode off,
and no reported thermal/performance warning; the observed calibration variation
still applies. Optional counter analysis is explicitly absent.

The complete workflow passed 90 Mojo and 43 Python tests before measurement.
Post-measurement checks bind the retained screen, selection, full run,
validation and profiles, and reject missing or duplicate dispatches. Reporting
changes do not alter the measured kernels or numerical gates. All **17 tables
and 17 figures** regenerate offline from compact evidence, preserving every
previous table and figure byte-for-byte. Full traces/XML, binaries, checkpoint
assets and generated arrays remain outside Git.

Rebuild tables and figures without a GPU:

```sh
uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.plot studies/attention_sublayer
```

For fresh measurements, use a clean checkout, regenerate and validate fixtures,
then use the [package-owned build/run and trace commands](../../src/llm_mojo/benchmarks/README.md).
Use the recorded commit to reproduce the exact measured source. Raw traces,
XML, binaries, checkpoint assets and oracle arrays stay outside Git.
