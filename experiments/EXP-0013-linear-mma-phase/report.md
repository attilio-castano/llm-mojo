# EXP-0013: Apple 8x8 MMA across decode and prefill

Status: **complete**

Evidence level: **operation**

## Question

Does the Apple 8x8 `simdgroup_matrix` operation improve our linear projection
against the strongest mapping already measured for each row-count regime?

## What was compared

The candidate keeps the same `8x16` output tile and one 32-lane SIMD group as
the manual register-2x2 kernel. It divides that output tile into two adjacent
8x8 hardware fragments. At each `BK=8` K phase the group loads one 8x8 input
fragment and two 8x8 weight fragments, then issues two collective MMA
operations. Each lane retains four FP32 output values across all 112 phases.

```text
                                  Register 2x2       MMA 8x16
Output tile                       8x16               8x16
SIMD groups per tile              1                  1
FP32 accumulators per lane        4                  4
K progress                        scalar K=1         MMA K=8
Cross-lane matrix operation       no                 two 8x8 MMA/phase
Shared operand storage            0                  0
Threadgroup barriers              0                  0
```

The comparison is phase-aware because EXP-0012 already established different
best controls. `M=1,4,8` compare MMA with public rowwise. `M=16..256` compare
MMA with register-2x2. The control is frozen before timing for every row count;
it is not selected after seeing this experiment.

This M4 path calls MAX 26.5's architecture-internal `_mma_apple_8x8` primitive,
which supports Apple M1-M5 float operands. It is distinct from the public
M5-only 16x16 neural-accelerator path. The existing `W[N,K]` layout is retained,
so the logical MMA B fragment is gathered from transposed storage.

## Correctness gate

The candidate passes the committed short-prefill oracle, an exact `8x16` tile,
ragged `M=9 K=129 N=17` tails, Qwen-width `M=8 K=896 N=128`, and batch-1
`M=1 K=896 N=128`. Every lane executes both collective MMA operations; invalid
fragment elements are zero-filled before the call.

## Diagnostic gate

Two unrecorded five-repetition passes reversed both row and implementation
order. MMA improved `M=16` by about 43% and `M=64` by about 52% in both passes.
It lost at `M=1`; `M=8` changed direction with order. Those observations only
earned the formal experiment and are not retained as final performance
evidence.

## Frozen protocol

The formal run sweeps `M=1,4,8,16,32,64,128,256` at `K=896`, `N=1152`, with
24 rotating weight allocations. Four blocks use ascending/control-first,
descending/MMA-first, descending/MMA-first, and ascending/control-first order.
Each specialization has 10 warmups, at most 20 timed iterations, and 10
repetitions per block. The complete protocol and decision rules are in
[manifest.json](manifest.json).

## Correctness and execution gates

The linear suite reported 32 passing tests, including the five MMA cases and
all existing rowwise, direct, register-2x2, and shared-staging cases. All 81
Python tests, the import test, 7 RMSNorm tests, and 9 RoPE tests passed. A
one-block end-to-end smoke run compiled every timed specialization, passed its
untimed exact-BF16 output gate, and produced all 160 expected samples.

`EXP-0013-RUN-001` then collected all 640 expected formal samples at clean
commit `5aabd40`. Every block proved Apple M4 Pro/Metal execution and passed
fixed-commit, AC-power, thermal, implementation-order, finite-value, unique-ID,
and sample-count checks. The emitted build contains one reference to the Apple
8x8 matrix intrinsic for each of the eight compiled M specializations and no
workgroup-barrier intrinsic. That validates the compiled mechanism, not its
individual contribution to elapsed time.

## Result

The workload is `K=896`, `N=1152`, with 24 distinct rotating weight
allocations. Times below are overall medians across 40 samples for orientation.
The percentage is the protocol's median of four within-block MMA/control
median ratios, not a ratio of the two displayed overall medians.

| M | Phase control | Control | MMA 8x16 | MMA versus control | MMA faster blocks | Classification |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Rowwise | 0.4154 ms | 0.7961 ms | +79.84% | 0/4 | Material regression |
| 4 | Rowwise | 0.6591 ms | 0.7919 ms | +20.41% | 0/4 | Material regression |
| 8 | Rowwise | 1.1546 ms | 0.7733 ms | -33.15% | 4/4 | Material improvement |
| 16 | Register 2x2 | 1.4532 ms | 0.7882 ms | -45.85% | 4/4 | Material improvement |
| 32 | Register 2x2 | 1.7903 ms | 0.9654 ms | -46.24% | 4/4 | Material improvement |
| 64 | Register 2x2 | 3.5196 ms | 1.6694 ms | -52.26% | 4/4 | Material improvement |
| 128 | Register 2x2 | 6.5388 ms | 3.0084 ms | -53.96% | 4/4 | Material improvement |
| 256 | Register 2x2 | 12.9648 ms | 5.8667 ms | -54.77% | 4/4 | Material improvement |

The candidate satisfied the preregistered large-prefill rule at `M=64`, `128`,
and `256`. Both crossover definitions selected `M=8`: this was the smallest
tested candidate boundary after which no larger M regressed and the smallest
after which every larger M materially improved. Every classification was
unanimous across the four counterbalanced blocks.

Exact block ratios, implementation medians, source accounting, environment,
provenance, and artifact hashes are retained in
[comparison-run.json](comparison-run.json).

## What we learned

The matrix primitive changes more than the spelling of four scalar
accumulators. A complete `8x8` input or weight fragment is distributed across
the 32 lanes, and the collective operation reuses those fragment values across
the output matrix. For complete `8x16` tiles, the MMA source requests 81.06%
fewer logical operand, bias, and output bytes than register-2x2. It also
replaces each lane's 896-step scalar walk with 112 `BK=8` phases and two
collective matrix operations per phase. These are source-level requests, not
observed cache, fabric, or DRAM traffic.

This is still an ownership-matched comparison at `M>=16`: both implementations
give one `8x16` output tile to one 32-lane SIMD group, retain four FP32 output
values per lane, and use no threadgroup operand scratch or barriers. The result
therefore says that the complete distributed-fragment plus hardware-MMA
mapping is much better than our strongest manual mapping on this shape. Timing
cannot divide the gain among fewer source instructions, cross-lane operand
reuse, matrix-unit throughput, scheduling, or cache behavior.

At `M=1`, the physical tile still computes eight fragment rows, so seven rows
are discarded. It exposes only 72 independent `8x16` tile groups across
`N=1152`, while rowwise exposes one K-parallel SIMD group per scalar output.
The MMA latency is consequently close to its `M=8` latency while producing
one eighth as many useful rows, and it loses by 79.84%. At `M=4`, half the MMA
row fragment is useful and it still loses by 20.41%.

At `M=8`, all fragment rows become useful without increasing the number of tile
groups, and MMA beats rowwise by 33.15%. The hardware primitive therefore moves
the measured crossover from the manual register kernel's `M=16` down to `M=8`
for this packed-QKV width. This is a bounded timing result, not proof that lane
utilization alone caused the transition.

## Decision

Reject this `8x16` MMA mapping for batch-1 decode and retain the public rowwise
kernel there. Advance it as the strongest packed-QKV prefill candidate for the
measured `K=896`, `N=1152`, `M>=8` regime.

Do not change public dispatch yet. The candidate uses an architecture-internal
MAX 26.5 API, and the model's `N=128` and `N=896` projections remain unmeasured.
The next experiment should compare rowwise, register-2x2, and MMA at all three
model output widths, with extra attention to the `M=4` to `M=8` boundary.
