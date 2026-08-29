# EXP-0008: Apple GPU prefill 8x16x32 shared staging

Status: **complete**

Evidence level: **operation**

## Question

If we keep one thread per output in an `8x16` output tile, does explicitly
staging `BK=32` slices of `X` and `W` in threadgroup memory make prefill
projection faster?

## What the tiled kernel does

One 128-thread group still owns one `8x16` rectangle of `Y`. For each of the 28
K phases at `K=896`:

1. The group cooperatively loads an `8x32` tile of `X` and a `16x32` tile of
   `W`. Each thread loads two X values and four W values for a complete tile.
2. A block barrier makes the 1,536 bytes of BF16 shared storage visible.
3. Each thread reads one X row and one W row from that storage and adds 32
   products to its persistent FP32 accumulator.
4. A second block barrier prevents the next phase from overwriting storage
   while another thread is still reading it.

Ragged M, N, and K coordinates write zero into shared storage. Every thread
reaches both barriers, including threads that do not own a valid output. Bias
and the one BF16 cast happen only after the full K reduction.

The direct control and tiled candidate therefore have identical output
ownership and arithmetic ownership. The changed mechanism is cooperative
operand staging, including its shared reads and writes, load coordination,
bounds handling, K-phase loop, and 56 barriers per dispatch.

The complete protocol is in [manifest.json](manifest.json).

## Correctness and execution gates

The tiled candidate matched the committed Transformers oracle, an exact
`8x16x32` tile, ragged `M=9`, `N=17`, `K=33` tails across two K phases, and the
Qwen KV shape `M=8`, `K=896`, `N=128`. The full projection suite reported 22
passing tests; the other Mojo suites and all 51 Python protocol tests also
passed before the implementation commit.

`EXP-0008-RUN-001` collected all 2,560 expected samples at clean commit
`0e032f9`. Every block proved Apple M4 Pro/Metal execution and passed fixed
commit, AC-power, thermal, ABBA-order, finite-value, and sample-count gates.

## Rotating packed-QKV result

The 24-layer rotating `N=1152` workload is primary because it uses 24 distinct
weight allocations. The percentage is the median of four within-block
tiled/direct median ratios, not a ratio of the two overall medians shown for
orientation.

| M | Direct median | Tiled median | Paired change | Tiled-faster blocks | Classification |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.349623 ms | 2.354639 ms | +74.53% | 0/4 | Material regression |
| 4 | 1.371118 ms | 2.365290 ms | +72.53% | 0/4 | Material regression |
| 8 | 1.418221 ms | 2.404782 ms | +69.47% | 0/4 | Material regression |
| 16 | 2.489311 ms | 4.154296 ms | +66.79% | 0/4 | Material regression |
| 32 | 4.420501 ms | 7.431684 ms | +68.10% | 0/4 | Material regression |
| 64 | 8.565060 ms | 14.156021 ms | +65.46% | 0/4 | Material regression |
| 128 | 16.722831 ms | 27.838138 ms | +66.41% | 0/4 | Material regression |
| 256 | 33.065155 ms | 55.337333 ms | +67.28% | 0/4 | Material regression |

All eight row counts materially regressed, and the direction repeated in every
block. The hot single-layer shapes were noisy and mixed: 19 were inconclusive,
three improved, and two additional packed-QKV shapes regressed. They do not
override the stable rotating result. Their classifications and the primary
block ratios remain in [comparison-run.json](comparison-run.json).

## What we learned about the cost of tiling

At the source level, staging did accomplish reuse. On rotating packed QKV, its
requested program traffic was 46.82% lower at `M=1`, 84.28% lower at `M=4`,
and 90.52% lower for every tested `M>=8`. Those totals include source-level
input, weight, bias, and output accesses; they are not observed hardware
traffic.

That reduction was not sufficient. The direct kernel issues repeated device
loads, but the hardware may already serve some from cache. The tiled kernel
guarantees additional cooperative stores and many shared reads, plus 56
barriers and loop/control work per dispatch. The paired result says the whole
incremental staging mechanism cost 65–75% on the primary workload.

Timing alone cannot assign that cost specifically to barriers, shared-memory
throughput, occupancy, generated instructions, or another mechanism. It also
does not say tiling is generally bad. It says this literal mapping—one scalar
output per thread, scalar shared accesses, and `BK=32`—is not an optimization
on this workload and device.

## Decision

Keep `enqueue_linear_apple_gpu` as the public path and add no dispatch rule.
Retain `enqueue_linear_prefill_direct_apple_gpu` as the ownership control and
`enqueue_linear_prefill_tiled_apple_gpu` as an explicit, correct learning
candidate.

The clean next experiment is a reduced BK sensitivity screen. Hold `BM=8`,
`BN=16`, and output ownership fixed; compare `BK=16`, `32`, `64`, and `128` on
the rotating workloads. This changes shared capacity and barrier count while
leaving the reuse structure recognizable. Only after that attribution step
should we change the arithmetic mapping—for example, giving each thread
multiple outputs or using hardware matrix operations.
