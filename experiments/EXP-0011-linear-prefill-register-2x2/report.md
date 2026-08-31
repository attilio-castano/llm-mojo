# EXP-0011: Apple GPU prefill 2x2 register ownership

Status: **complete**

Evidence level: **operation**

## Question

With the same direct full-K `8x16` output tile, does assigning each lane a
`2x2` output microtile improve rotating packed-QKV prefill over assigning each
thread one scalar output?

## What was compared

The scalar direct control launches 128 threads for each `8x16` output tile.
Each thread owns one output, holds one FP32 accumulator, and walks all `K=896`
values directly from device memory.

The candidate launches one 32-lane SIMD group for the same `8x16` output tile.
The tile contains exactly 32 `2x2` microtiles, so lane `l` owns one of them and
holds four FP32 accumulators. At each K step the lane loads two X values and two
W values, then forms the four products in their Cartesian product:

```text
               W[n0,k]       W[n1,k]
X[m0,k]     -> Y[m0,n0]      Y[m0,n1]
X[m1,k]     -> Y[m1,n0]      Y[m1,n1]
```

Computing the same four outputs with scalar owners requests four X loads and
four W loads per K step. The `2x2` owner requests two of each. For complete
tiles, source-requested operand loads are therefore halved.

Both implementations use BF16 inputs, weights, bias, and output; accumulate in
FP32; stream the full K dimension; use zero threadgroup operand storage and
zero barriers; and enqueue the same 24 rotating packed-QKV projections. The
candidate changes arithmetic ownership, threadgroup width, live accumulator
count, instruction organization, and generated code. It does not use BK
tiling, shared staging, an Apple matrix primitive, or the Neural Engine. The
frozen protocol is in [manifest.json](manifest.json).

## Correctness and execution gates

The candidate matched the reference on the short oracle, an exact `8x16`
output tile, ragged `M=9 K=129 N=17` tails, and the Qwen-width
`M=8 K=896 N=128` case. The linear suite reported 27 passing tests. All 69
Python tests, the import test, 7 RMSNorm tests, and 9 RoPE tests also passed.
Direct-first, candidate-first, legacy BK16/direct, and legacy four-BK benchmark
modes compiled. Both timed implementations passed their untimed exact-BF16
output gate.

`EXP-0011-RUN-001` collected all 640 expected samples at clean commit
`882eed0`. Every block proved Apple M4 Pro/Metal execution and passed fixed
commit, AC-power, thermal, implementation-order, finite-value, unique-ID, and
sample-count checks. The four blocks used the predeclared ABBA order:
direct/candidate, candidate/direct, candidate/direct, direct/candidate. The M
sweep alternated ascending, descending, descending, ascending.

## Result

The workload is `K=896`, `N=1152`, with 24 distinct rotating weight
allocations. Times below are overall medians across 40 samples for orientation.
The percentage is the protocol's median of four within-block candidate/direct
median ratios, not a ratio of the two displayed overall medians.

| M | Scalar direct | Register 2x2 | 2x2 versus direct | 2x2 faster blocks | Classification |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.4020 ms | 3.1169 ms | +122.73% | 0/4 | Material regression |
| 4 | 1.4186 ms | 1.4315 ms | +1.12% | 0/4 | Inconclusive |
| 8 | 1.4779 ms | 1.4329 ms | -3.66% | 4/4 | Inconclusive |
| 16 | 2.5418 ms | 1.4613 ms | -42.69% | 4/4 | Material improvement |
| 32 | 4.5425 ms | 1.7651 ms | -60.69% | 4/4 | Material improvement |
| 64 | 8.8072 ms | 3.4720 ms | -60.65% | 4/4 | Material improvement |
| 128 | 17.2509 ms | 6.5607 ms | -62.06% | 4/4 | Material improvement |
| 256 | 34.1909 ms | 12.9185 ms | -62.19% | 4/4 | Material improvement |

The candidate satisfied the advance rule: it materially improved all three
decisive rows (`M=64`, `128`, and `256`) and did not materially regress any
tested `M>=8`. It was faster in every block at every `M>=8`.

The source mapping requests 49.97% fewer logical bytes for every tested
`M>=4`; at `M=1`, where the second row in every `2x2` microtile is invalid, the
reduction is only 24.97%. These counts are derived from the source. They do not
measure cache hits, memory-fabric traffic, or DRAM bytes.

Exact block ratios, implementation medians, source accounting, provenance,
and artifact hashes are retained in [comparison-run.json](comparison-run.json).

## What we learned

The concern that multi-output ownership trades away parallelism was correct,
but the trade changes with M.

At `M=1`, only one row exists. Each candidate lane therefore owns only the two
valid outputs in its first row; the W-across-rows reuse disappears, and only
eight lanes of the SIMD group do useful work in each output tile. The candidate
behaves much like a two-output decode mapping and loses badly.

At `M=4`, the candidate still has only half of its 32 lanes active per tile and
is near parity. At `M=8`, every lane and every `2x2` microtile is valid, but the
whole operation launches only 72 single-SIMD-group threadgroups, and the result
is a small, non-material improvement. As M grows, the grid supplies more
independent output tiles while each lane retains reuse inside its own
microtile. The crossover at `M=16` and the gain stabilizing near 61%–62% from
`M=32` onward are consistent with broad parallelism becoming sufficient, but
timing does not prove that mechanism.

Timing alone does not prove why the gain exceeds the roughly 50% reduction in
source-requested bytes. The ownership change also gives each lane four
independent accumulation chains instead of one, reduces repeated loop/index
work per output, changes the number of SIMD groups per threadgroup, and changes
compiler-generated load and arithmetic scheduling. Cache behavior may also
differ. This experiment measures that complete mechanism and cannot attribute
the result to any one factor.

The important conceptual result is that this is useful tiling without shared
memory and without a hardware matrix operation. The output tile is divided
into register microtiles, and reuse occurs inside a lane. BK tiling and shared
operand staging remain separate choices; EXP-0011 did neither.

## Decision

Advance the `2x2` register candidate to a paired comparison against the public
rowwise `enqueue_linear_apple_gpu` path. Do not change public dispatch yet: this
experiment proves a win only against the ownership-matched scalar direct
control.

The public-rowwise comparison should retain the same rotating workload and
four-block order control. If the `2x2` candidate also wins there for the
intended prefill rows, it becomes a credible manual prefill path and a much
stronger control for a later Apple matrix-operation experiment. `M=1` must
remain on the decode path; this experiment gives no reason to route decode to
the candidate.
