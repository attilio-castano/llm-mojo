# EXP-0009: Apple GPU prefill BK sensitivity

Status: **complete**

Evidence level: **operation**

## Question

With `BM=8`, `BN=16`, and one thread per output held fixed, how does changing
the shared K tile from `BK=32` to `BK=16`, `64`, or `128` affect rotating
packed-QKV prefill timing?

## What changed

The kernel now accepts BK as a compile-time parameter. Every variant uses the
same 128-thread group to own one `8x16` output tile, the same cooperative
zero-fill policy, one persistent FP32 accumulator per valid output, and two
uniform threadgroup barriers per K phase.

At `K=896`, the source-level tradeoff is:

| BK | BF16 operand scratch | K phases | Barriers per dispatch |
| ---: | ---: | ---: | ---: |
| 16 | 768 B | 56 | 112 |
| 32 | 1,536 B | 28 | 56 |
| 64 | 3,072 B | 14 | 28 |
| 128 | 6,144 B | 7 | 14 |

Because all four BKs divide 896, they request the same total source-level
global operand loads for a given M. They also perform the same arithmetic and
the same total number of shared operand reads and writes. What changes is how
that work is grouped into phases, the live scratch capacity, and the generated
loop and coordination structure.

The complete predeclared protocol is in [manifest.json](manifest.json).

## Correctness and execution gates

All four variants matched the host reference at `M=9`, `K=129`, `N=17`. That
case requires partial M and N output tiles and a partial final K phase for every
BK. The complete linear suite reported 23 passing tests. All 57 Python protocol
tests, the import test, 7 RMSNorm tests, and 9 RoPE tests also passed. Both the
default and fourth-sequence benchmark specializations compiled before the run,
and each benchmark variant passed its untimed exact-BF16-output gate.

`EXP-0009-RUN-001` collected all 1,280 expected samples at clean commit
`2e77bc3`. Every block proved Apple M4 Pro/Metal execution and passed fixed
commit, AC-power, thermal, workload-order, BK-order, finite-value, and sample
count gates. Each BK occupied each within-workload execution position exactly
once across the four blocks.

## Result

The workload is `K=896`, `N=1152`, with 24 distinct weight allocations. Times
below are overall medians across 40 samples for orientation. The percentage is
the protocol's median of four within-block candidate/BK32 median ratios, not a
ratio of the displayed overall medians.

| M | BK16 | BK32 | BK64 | BK128 | BK16 vs BK32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.8094 ms | 2.3644 ms | 2.2833 ms | 2.2696 ms | -23.84% |
| 4 | 1.8306 ms | 2.3702 ms | 2.3356 ms | 2.2968 ms | -21.94% |
| 8 | 1.8700 ms | 2.3998 ms | 2.3498 ms | 2.3801 ms | -21.86% |
| 16 | 3.1253 ms | 4.1382 ms | 4.0881 ms | 4.1055 ms | -24.43% |
| 32 | 5.6177 ms | 7.4490 ms | 7.3670 ms | 7.3167 ms | -24.63% |
| 64 | 10.6875 ms | 14.1612 ms | 14.0195 ms | 14.0613 ms | -24.54% |
| 128 | 20.9416 ms | 28.0443 ms | 27.7325 ms | 27.5370 ms | -25.58% |
| 256 | 42.1368 ms | 56.8655 ms | 55.8835 ms | 56.2792 ms | -25.35% |

BK16 materially improved all eight row counts and was faster in 4/4 blocks for
each one. Its paired improvement ranged from 21.86% to 25.58%.

BK64 was 1.05%–2.86% faster by the paired statistic; BK128 was 0.32%–3.76%
faster. None of those 16 comparisons crossed the 5% material threshold, so
every BK64 and BK128 result is classified as inconclusive. Their mostly
consistent direction is retained as a small signal, not promoted to a speedup
claim. Exact block ratios are in [comparison-run.json](comparison-run.json).

## What we learned

The simple prediction that fewer phases and barriers should be faster did not
describe this kernel. BK16 performs 56 phases and 112 barriers—twice BK32's
counts—yet is roughly 22%–26% faster. Increasing BK beyond 32 and reducing the
barrier count further produced only small changes.

That rules out barrier count as a sufficient explanation for this result. It
does not identify the winning mechanism. BK also changes live threadgroup
capacity, cooperative work per phase, shared-arithmetic loop length, bounds and
loop-control frequency, access scheduling, and compiler-generated code. Timing
alone cannot separate those effects, and no occupancy, cache, fabric, DRAM, or
instruction counter was measured.

The shape of the curve is therefore more informative than a generic “smaller
tiles are better” rule: BK16 is a strong favorable discontinuity for this
literal scalar mapping, while BK32, BK64, and BK128 form a much tighter timing
cluster.

## Decision

BK16 is the only variant that satisfies the predeclared advance rule: material
improvement at `M=64`, `128`, and `256`, with no material regression at any
tested `M>=8`. Advance BK16 to a separately paired comparison against the
direct `8x16` no-staging control.

Do not add a runtime selector. This screen proves BK16 is better than BK32; it
does not prove BK16 is better than the direct control or the public rowwise
kernel. `enqueue_linear_apple_gpu` remains the public path.
