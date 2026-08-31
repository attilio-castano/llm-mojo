# EXP-0007: Apple GPU prefill direct 8x16 ownership control

Status: **complete**

Evidence level: **operation**

## Question

Before adding shared-memory tiling, what changes when projection ownership moves
from one 32-lane SIMD group per scalar output to one thread per output in an
`8x16` threadgroup rectangle?

## Frozen control

The rowwise baseline assigns four outputs to each 128-thread group. Thirty-two
lanes cooperate on each output and finish with `warp.sum`. The control assigns
all 128 threads to distinct outputs in an `8x16` rectangle. Each thread walks
the complete `K` dimension serially and retains one FP32 accumulator.

Both implementations read `X` and `W` directly from device memory. Neither has
threadgroup operand storage or a threadgroup barrier. The control therefore
changes output ownership without yet testing explicit input tiling. It is
retained for attribution and was never eligible for production dispatch.

The complete protocol is in [manifest.json](manifest.json).

## Correctness and execution gates

The control matched the committed multi-row Transformers oracle, an exact
`8x16` output tile, ragged `M=9`, `N=17`, `K=33` tails, and the Qwen KV shape
`M=8`, `K=896`, `N=128`. The full projection suite reported 18 passing tests.

`EXP-0007-RUN-001` collected all 2,560 expected samples at clean commit
`0111bf6`. Every block proved Apple M4 Pro/Metal execution and passed fixed
commit, AC-power, thermal, ABBA-order, finite-value, and sample-count gates.

## Rotating packed-QKV result

The 24-layer rotating `N=1152` workload is the useful curve because it owns 24
distinct weight allocations. Hot single-layer timings are retained in
[comparison-run.json](comparison-run.json), but repeated hot weights are an
exceptionally cache-favorable microbenchmark.

| M | Rowwise median | Direct median | Paired change | Faster blocks | Classification |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.417739 ms | 1.378930 ms | +210.82% | 0/4 | Material regression |
| 4 | 0.644611 ms | 1.395178 ms | +115.31% | 0/4 | Material regression |
| 8 | 1.143898 ms | 1.447555 ms | +27.82% | 0/4 | Material regression |
| 16 | 2.219991 ms | 2.537622 ms | +14.39% | 0/4 | Material regression |
| 32 | 4.478567 ms | 4.517279 ms | +0.70% | 0/4 | Inconclusive |
| 64 | 9.349887 ms | 8.741998 ms | -6.95% | 4/4 | Material improvement |
| 128 | 19.381910 ms | 17.174054 ms | -11.36% | 4/4 | Material improvement |
| 256 | 39.959151 ms | 33.928639 ms | -15.42% | 4/4 | Material improvement |

The paired percentage is the median of four within-block direct/rowwise median
ratios; it is not calculated from the two overall medians shown for orientation.
The direction repeated cleanly: short rows rejected direct ownership, `M=32`
was still slower but below materiality, and every `M>=64` point improved in all
four blocks.

## Interpretation

The crossover establishes that output ownership itself is consequential. At
small `M`, incomplete `8`-row tiles and serial per-thread dot products cost far
more than removing the SIMD reduction. At larger `M`, the GPU has enough
independent outputs for the 128-output mapping to become competitive and then
materially faster on rotating weights.

Timing does not identify the mechanism. In particular, it does not prove a
cache, occupancy, scheduling, or bandwidth explanation. The hot workloads also
showed noisy, shape-specific effects and do not justify a selector.

Most importantly, this is not yet evidence for shared-memory tiling. It gives
the next experiment a controlled baseline: keep the same `8x16`
one-thread-per-output mapping and add only `BK=32` staging of `X` and `W`.

## Decision

Keep `enqueue_linear_apple_gpu` as the public path. Retain
`enqueue_linear_prefill_direct_apple_gpu` only as the explicit attribution
control for the next shared-memory experiment. Add no runtime dispatch rule.
