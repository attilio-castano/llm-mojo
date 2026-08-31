# EXP-0012: Apple GPU prefill 2x2 register versus public rowwise

Status: **complete**

Evidence level: **operation**

## Question

Does the EXP-0011 `2x2` register mapping beat the public K-parallel rowwise
kernel on the same rotating packed-QKV prefill workload, and where does their
measured crossover begin?

## What was compared

The public control gives one scalar output to one 32-lane SIMD group. Its lanes
split `K=896`, so each lane accumulates 28 products into one FP32 partial sum.
A SIMD-group reduction combines those 32 partial sums, and lane zero adds the
bias and stores the BF16 output. A 128-thread group contains four such SIMD
groups and therefore completes four scalar outputs.

The candidate gives one `8x16` output tile to one 32-lane SIMD group. Each lane
owns a `2x2` microtile, walks all `K=896` values serially, and holds four FP32
accumulators. At each K step it loads two X values and two W values and forms
their four products. It performs no cross-lane reduction.

```text
                         Public rowwise       Register 2x2
Output ownership         1 output / group     128 outputs / group
K ownership              28 values / lane     896 values / lane
FP32 accumulators/lane    1                    4
SIMD reductions/output   1                    0
SIMD groups/threadgroup  4                    1
Shared operand storage   0                    0
Threadgroup barriers     0                    0
```

Both use the same row-major tensors, BF16 storage, FP32 accumulation, 24
rotating `K=896`, `N=1152` projections, enqueue count, and synchronized timing
boundary. Neither uses a BK tile, shared operand staging, an Apple matrix
primitive, or the Neural Engine. The complete frozen protocol is in
[manifest.json](manifest.json).

## Correctness and execution gates

The linear suite reported 27 passing tests, including the existing public
rowwise oracles and the `2x2` exact-tile, ragged-tail, and Qwen-width cases. All
75 Python tests, the import test, 7 RMSNorm tests, and 9 RoPE tests passed. The
rowwise-first, candidate-first, legacy register/direct, and legacy four-BK
benchmark modes compiled. Both timed implementations passed their untimed
exact-BF16 output gate.

`EXP-0012-RUN-001` collected all 640 expected samples at clean commit
`b83bc06`. Every block proved Apple M4 Pro/Metal execution and passed fixed
commit, AC-power, thermal, implementation-order, finite-value, unique-ID, and
sample-count checks. The four blocks used the predeclared ABBA order:
rowwise/candidate, candidate/rowwise, candidate/rowwise, rowwise/candidate. The
M sweep alternated ascending, descending, descending, ascending.

## Result

The workload is `K=896`, `N=1152`, with 24 distinct rotating weight
allocations. Times below are overall medians across 40 samples for orientation.
The percentage is the protocol's median of four within-block
candidate/rowwise median ratios, not a ratio of the two displayed overall
medians.

| M | Public rowwise | Register 2x2 | 2x2 versus rowwise | 2x2 faster blocks | Classification |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.4432 ms | 3.0813 ms | +597.85% | 0/4 | Material regression |
| 4 | 0.6330 ms | 1.4146 ms | +123.25% | 0/4 | Material regression |
| 8 | 1.1587 ms | 1.4153 ms | +22.03% | 0/4 | Material regression |
| 16 | 2.2503 ms | 1.4382 ms | -35.92% | 4/4 | Material improvement |
| 32 | 4.5327 ms | 1.7744 ms | -60.84% | 4/4 | Material improvement |
| 64 | 9.4436 ms | 3.5212 ms | -62.69% | 4/4 | Material improvement |
| 128 | 19.6083 ms | 6.5684 ms | -66.51% | 4/4 | Material improvement |
| 256 | 40.6047 ms | 12.9810 ms | -67.98% | 4/4 | Material improvement |

The candidate satisfied the large-prefill advance rule at `M=64`, `128`, and
`256`. The two preregistered crossover definitions also agree on `M=16`: it is
the first candidate threshold after which no larger tested M regresses, and it
is also the first threshold after which every larger tested M materially
improves. Every classification was unanimous across the four blocks.

For complete row pairs, the `2x2` source requests 49.97% fewer logical operand,
bias, and output bytes than rowwise because a lane reuses two X and two W values
across four outputs. At `M=1`, invalid second rows reduce the source-level
saving to 24.97%. These are source counts, not observed cache, fabric, or DRAM
traffic.

Exact block ratios, implementation medians, source accounting, provenance,
and artifact hashes are retained in [comparison-run.json](comparison-run.json).

## What we learned

The manual `2x2` mapping is not merely better than the scalar direct control.
For this packed-QKV workload it also beats the public K-parallel kernel by a
large margin once `M` reaches 16.

The small-M loss is equally clear. At `M=1`, a candidate tile has only eight of
its 32 lanes doing useful work and launches 72 SIMD groups across N. The public
kernel launches one K-parallel SIMD group for each of the 1,152 scalar outputs.
At `M=4`, only 16 candidate lanes per tile are active. At `M=8`, all lanes are
active but the candidate still has only 72 independent output-tile groups, and
it remains 22.03% slower.

At `M=16`, lane utilization is unchanged from `M=8`, but the candidate grid
doubles to 144 independent output-tile groups and becomes 35.92% faster. That
sharp reversal is consistent with the output-parallel mapping finally exposing
enough independent tiles to repay serial K work, while retaining four-output
reuse and avoiding per-output reductions. It is useful evidence for the
parallelism/reuse model we have been developing, but timing alone does not
prove a saturation threshold or isolate grid size from the other coupled
changes.

The large-M gain grows beyond the roughly 50% reduction in source-requested
bytes. Other plausible contributors are four independent accumulation chains
per lane, less loop and indexing work per output, no SIMD reduction per output,
different scheduling granularity, and different compiler-generated code. The
experiment ranks the two complete arithmetic mappings; it does not attribute
the gain to any one of those effects.

This remains a manual output-register tile. It uses no shared memory, barriers,
BK tiling, or special Apple matrix operation. That makes it the correct manual
control for a later Apple matrix experiment.

## Decision

Advance the `2x2` mapping as the manual prefill candidate for the measured
packed-QKV regime. Keep the public rowwise path for the tested `M<=8` regime.
Do not change public dispatch from this result alone.

The next production-oriented experiment should extend the paired comparison
to the model's other projection widths (`N=128` and `N=896`) and concentrate
rows around the observed boundary before selecting any public dispatch rule.
An Apple matrix-operation experiment may now use this `2x2` kernel—not the
older rowwise baseline—as its strongest manual large-prefill control.
