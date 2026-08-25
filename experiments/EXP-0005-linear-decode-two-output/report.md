# EXP-0005: Apple GPU M=1 two-output projection

Status: **planned**

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

No candidate has been implemented or measured yet.

## Nonclaims

- This is not a general matrix-multiplication result.
- This experiment has no prefill optimization or M>1 performance claim.
- Timing alone cannot prove input-cache reuse or identify a hardware limiter.

## Decision

Pending implementation, correctness, generated-code inspection, paired timing,
and any timing-qualified profiling.
