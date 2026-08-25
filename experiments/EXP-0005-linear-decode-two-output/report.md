# EXP-0005: Apple GPU M=1 two-output projection

Status: **complete**

Evidence level: **operation**

## Question

Does computing two adjacent `M=1` projection outputs per SIMD group materially
improve the rotating 24-layer QKV workload without materially regressing Q, KV,
or hot QKV?

## Frozen hypothesis

The baseline assigns one output dot product to a 32-lane SIMD group. The
candidate assigns two adjacent outputs to that group. Each lane loads an input
value once, multiplies it by two weight rows, retains two FP32 accumulators, and
performs two independent `warp.sum` reductions. The candidate introduces no
threadgroup memory or barrier.

This is a narrow M=1 reuse experiment. It does not add vectorized loads,
quantization, weight tiling, a combined-QKV kernel, an N-based selector, or an
M>1/prefill kernel.

## Frozen comparison and decision

The four EXP-0004 workloads are paired in four ABBA blocks. A material effect
requires at least 5% median change and the same direction in at least three
blocks. Promotion additionally requires a material improvement on the rotating
24-layer primary workload and no material regression on Q, KV, or hot QKV.

If that rule passes, only `M=1` uses the candidate; `M>1` retains the baseline.
If it does not pass, the baseline remains the public path. Opposite Q and KV
effects will be documented without adding a crossover selector. The complete
machine-readable protocol is in [`manifest.json`](manifest.json).

## Results

The explicit candidate passed Qwen-width Q and KV correctness against the host
reference, an odd-`N` tail case, and an `M>1` rejection test. The public
one-output entrypoint continued to pass the existing decode and prefill cases.

Clean Metal LLVM from commit `de92750` contains two FP32 loop-accumulator PHIs,
one input load feeding two weight multiply-accumulates for a full pair, and 10
SIMD shuffle calls: five for each reduction. It contains no threadgroup
allocation or workgroup barrier. This confirms the declared structure without
identifying a performance mechanism.

`EXP-0005-RUN-001` then collected all 320 frozen samples at the same clean
commit. Every block proved Apple M4 Pro/Metal execution and passed AC-power,
thermal, ABBA order, and sample-count gates.

| Workload | Baseline median | Candidate median | Paired change | Faster blocks | Classification |
| --- | ---: | ---: | ---: | ---: | --- |
| KV `N=128` | 0.010225 ms | 0.009747 ms | -8.72% | 3/4 | Material improvement |
| Q `N=896` | 0.009410 ms | 0.009139 ms | +0.35% | 2/4 | Inconclusive |
| Hot QKV | 0.029039 ms | 0.037548 ms | +21.28% | 0/4 | Material regression |
| Rotating QKV | 0.993810 ms | 1.409349 ms | +42.46% | 0/4 | Material regression |

The paired percentage is the median of four within-block candidate/baseline
median ratios; it is not derived from the two overall medians shown for
orientation. The primary rotating workload regressed in every block. Hot QKV
also regressed in every block. KV improved materially, but Q was directionally
split and essentially unchanged by the paired statistic.

All block ratios, validation facts, and external artifact checksums are retained
in [`comparison-run.json`](comparison-run.json). Because the candidate did not
win the primary timing rule, it was not profile-qualified. No register,
occupancy, cache, bandwidth, or stall mechanism is claimed.

## Interpretation

Input reuse exists in the lowered loop, but it is not sufficient. The extra
accumulation chain and second reduction accompany substantial regressions once
Q/K/V projections are composed, especially when weights rotate. Timing alone
cannot distinguish register pressure, scheduling, reduced parallelism, or
another cause.

The isolated KV result is useful ladder information, not a dispatch rule. Its
direction failed once, Q showed no repeatable benefit, and both composed
workloads strongly rejected the candidate. Adding an `N` threshold would
optimize the least representative slice while making the public path more
complex.

## Nonclaims

- This is not a general matrix-multiplication result.
- This experiment has no prefill optimization or M>1 performance claim.
- Timing alone cannot prove input-cache reuse or identify a hardware limiter.

## Decision

Retain `enqueue_linear_apple_gpu` as the one-output-per-SIMD-group public path
for every `M`, including `M=1`. Keep
`enqueue_linear_apple_gpu_two_output` as an explicit, decode-only experimental
entrypoint so the negative result remains reproducible. Add no `M` or `N`
runtime selector, and do not profile the rejected candidate.
