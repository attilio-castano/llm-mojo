# Qwen attention sublayer

The completed FP32 attention experiments establish substantial remaining
headroom on Apple M4 Pro / Metal. Decode ownership changes reduce whole-block
time by **61–69% at T=1024** and **89–91% at T=4096**, with rowwise Wo fixed.
Prefill tiling reduces it by **47% at full 1024** and **74–75% at full 4096**,
with MMA Wo fixed. The earlier Wo-only experiment improves full prefill by
about 31% at 256 tokens and 10% at 4096. These are separate paired comparisons
against matching controls, not multipliable speedups. Negative and inconclusive
results are retained; new mappings remain explicit options.

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
independent gates and provenance. The measured engine and fixtures match the
validation hashes; only the runner's per-process timeout changed afterward.

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

Both follow-up milestones are complete. **Q projection is the clearest next
contained experiment for full prefill.** It has the same `[R,896]` by
`[896,896]` matrix dimensions as Wo, with bias, so the existing projection
mapping is a concrete starting point. Unlike Wo, its rounding differences
feed RoPE and attention scores. Validate Q on exact upstream normalized inputs,
then rerun operation and full-block gates before measuring its contribution.
Do not inherit Wo's numerical or performance result merely because shapes match.

Long cached chunks remain a separate GQA question. At `(64,4096)`, the 32-row
tile exposes only 28 threadgroups. Smaller query tiles or splitting the KV
sequence could expose more independent work, informed by the decode study;
they would also change reuse, partial-state traffic and merge costs. The
reported spills are another concrete investigation point, but these captures
do not identify their source values or prove they dominate runtime. Further
head reuse or barrier changes need fresh motivation after their earlier mixed
results. Broader fusion should follow the remaining stage costs.

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

Rebuild tables and figures without a GPU:

```sh
uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.plot studies/attention_sublayer
```

For fresh measurements, use a clean checkout, regenerate and validate fixtures,
then use the [package-owned build/run and trace commands](../../src/llm_mojo/benchmarks/README.md).
Use the recorded commit to reproduce the exact measured source. Raw traces,
XML, binaries, checkpoint assets and oracle arrays stay outside Git.
