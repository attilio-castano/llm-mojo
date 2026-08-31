# EXP-0010: Apple GPU prefill BK16 shared staging versus direct control

Status: **complete**

Evidence level: **operation**

## Question

With `BM=8`, `BN=16`, 128 threads, and one scalar output per thread held
fixed, does the `BK=16` shared-staging winner from EXP-0009 beat direct full-K
streaming on rotating packed-QKV prefill?

## What was compared

Both implementations assign the same `8x16` output rectangle to one
128-thread group. Each thread owns one output, keeps one FP32 accumulator live
through `K=896`, adds the same bias, and performs one final BF16 cast.

The direct control reads the required X and W values from device memory as
each output thread walks the full K dimension. It uses no threadgroup operand
storage, K phases, or barriers.

The BK16 candidate divides K into 56 phases. During each phase, the threadgroup
cooperatively stages an `8x16` X tile and a `16x16` W tile into 768 bytes of
threadgroup memory. One barrier publishes those operands, every valid output
owner accumulates 16 products, and a second barrier protects the storage before
the next phase overwrites it. That gives 112 barriers per dispatch.

This comparison therefore holds output ownership constant and changes the
complete shared-staging mechanism: source-requested global loads, shared
writes and reads, phase-loop structure, bounds work, barriers, live scratch,
and generated code. It cannot attribute a result to any one of those costs.
The frozen protocol is in [manifest.json](manifest.json).

## Correctness and execution gates

The BK16 implementation matched the host reference at `M=9`, `K=129`,
`N=17`, including M, N, and final-K tails. The linear suite reported 23 passing
tests. All 63 Python tests, the import test, 7 RMSNorm tests, and 9 RoPE tests
also passed. Direct-first, BK16-first, and legacy four-BK benchmark modes all
compiled, and both timed implementations passed their untimed exact-BF16
output gate.

`EXP-0010-RUN-001` collected all 640 expected samples at clean commit
`e992622`. Every block proved Apple M4 Pro/Metal execution and passed fixed
commit, AC-power, thermal, implementation-order, finite-value, unique-ID, and
sample-count checks. The four blocks used the predeclared ABBA order:
direct/BK16, BK16/direct, BK16/direct, direct/BK16, while the M sweep alternated
ascending, descending, descending, ascending.

## Result

The workload is `K=896`, `N=1152`, with 24 distinct rotating weight
allocations. Times below are overall medians across 40 samples for orientation.
The percentage is the protocol's median of four within-block BK16/direct median
ratios, not a ratio of the two displayed overall medians.

| M | Direct | BK16 | BK16 versus direct | BK16 faster blocks |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.4039 ms | 1.8737 ms | +33.88% | 0/4 |
| 4 | 1.4229 ms | 1.8693 ms | +32.22% | 0/4 |
| 8 | 1.4764 ms | 1.9203 ms | +30.31% | 0/4 |
| 16 | 2.5429 ms | 3.2336 ms | +27.41% | 0/4 |
| 32 | 4.5351 ms | 5.7883 ms | +27.82% | 0/4 |
| 64 | 8.7883 ms | 11.0418 ms | +25.30% | 0/4 |
| 128 | 16.9504 ms | 21.0556 ms | +24.54% | 0/4 |
| 256 | 33.2297 ms | 41.9288 ms | +25.90% | 0/4 |

BK16 materially regressed every tested M and was slower in all four blocks for
each one. It therefore failed the advance rule and met the predeclared reject
rule at all three decisive large-M rows: `M=64`, `128`, and `256`.

This happened despite a 46.82% reduction in source-requested program traffic
at `M=1`, 84.28% at `M=4`, and 90.52% at every `M>=8`. Those counts describe
logical loads requested by the source mapping. They are not observations of
cache hits, fabric traffic, or DRAM bytes.

Exact block ratios, implementation medians, source accounting, provenance,
and artifact hashes are retained in
[comparison-run.json](comparison-run.json).

## What we learned

EXP-0009 established that BK16 is roughly 22%–26% faster than BK32 within the
shared-staging family. EXP-0010 establishes that even that best tested BK is
roughly 25%–34% slower than doing no shared staging under the same output
ownership. The BK sweep found a cheaper version of a mechanism that still
loses to its control; it did not turn the mechanism into an optimization.

This sharpens the tiling model. The direct control is already *output tiled*:
one threadgroup owns an `8x16` output tile. What loses here is the additional
choice to K-block and materialize reusable X and W tiles in threadgroup memory
while leaving one scalar output per thread. Reduced source-requested global
loads were not enough to repay that complete mechanism on this workload and
hardware.

Timing does not prove why. Possible coupled contributors include shared-memory
reads and writes, 112 barriers, phase-loop and bounds instructions, scheduling,
compiler-generated code, or the possibility that the direct loads are served
more cheaply by the memory hierarchy than the source accounting suggests. No
counter in this experiment separates those explanations.

The negative result is narrow. It does not show that prefill tiling is
generally unhelpful. A different arithmetic mapping can let each thread or
SIMD group own several outputs and reuse an input value in registers, or use a
hardware matrix primitive. Those designs change the amount and organization
of useful arithmetic performed between synchronization points; this
experiment did not test them.

## Decision

Reject BK16 shared staging as an optimization for the scalar `8x16`
one-output-per-thread mapping and stop tuning BK for that mapping. Retain the
implementation as a learning and reproduction candidate.

Do not run the planned public-rowwise comparison and do not add a runtime
selector: BK16 failed its ownership-matched direct-control gate. The public
`enqueue_linear_apple_gpu` path remains unchanged.

The next prefill experiment, if pursued, should change arithmetic ownership
rather than try another BK—for example, register-tiled multi-output ownership
or explicit SIMD-group/hardware-matrix cooperation—with its own correctness
oracle and frozen comparison protocol.
