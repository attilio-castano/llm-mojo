# EXP-0004: Apple GPU M=1 projection baseline

Status: **complete**

Evidence level: **operation**

## Question

What synchronized operation latency does the one-output-per-SIMD-group Apple
GPU projection baseline establish for Q, KV, one hot QKV layer, and a rotating
24-layer QKV decode proxy?

## Frozen protocol

Every workload is decode-only `M=1`, with `K=896`, row-major BF16 input and
weights, BF16 output, and FP32 accumulation. The kernel maps one output dot
product to one 32-lane SIMD group. Four SIMD groups share a 128-thread
threadgroup, and no threadgroup memory or workgroup barrier is used.

The operation matrix separates two small single-dispatch shapes from one hot
QKV layer and a 24-layer rotating-weight proxy. The rotating proxy owns 24
distinct Q/K/V weight sets and issues 72 enqueues before one synchronized
completion. It is a controlled cache-pressure proxy, not an end-to-end decoder
block.

Four recorded blocks reverse workload order as ascending, descending,
descending, ascending. Each block retains 10 repetitions after the benchmark's
100 warmup iterations. Compilation, allocation, initialization, correctness,
and host mapping are outside the timed region. The machine-readable protocol is
in [`manifest.json`](manifest.json).

## Timing results

`EXP-0004-RUN-001` collected all four frozen blocks at clean commit `b659b7b`.
Every block identified an Apple M4 Pro through Metal, retained 10 repetitions
per workload with 100 internal iterations, and passed AC-power, thermal,
repository, workload-order, and sample-count gates. The compact record in
[`baseline-run.json`](baseline-run.json) binds all 160 samples to checksums of
the external raw artifacts.

| Workload | Median per workload | Median per dispatch | MAD |
| --- | ---: | ---: | ---: |
| KV `N=128` | 0.010181 ms | 0.010181 ms | 0.001458 ms |
| Q `N=896` | 0.009471 ms | 0.009471 ms | 0.000631 ms |
| Hot QKV, 3 dispatches | 0.032821 ms | 0.010940 ms | 0.001293 ms |
| Rotating QKV, 72 dispatches | 0.951752 ms | 0.013219 ms | 0.007444 ms |

Q and KV are dominated by a noisy single-dispatch floor: the larger Q dot
product is not slower than KV in this operation-level measurement. Hot QKV
amortizes one completion across three enqueues. The rotating proxy is slower per
dispatch than the hot case while using distinct layer weights, which makes it a
useful primary comparison workload but does not by itself prove a cache cause.

## Bounded baseline profiles

Two clean binaries from commit `27995e1` produced receipt-bound Performance
Limiters captures on the same Apple M4 Pro/Metal device. The Q capture retained
100 warmup plus 500 profile dispatches. The rotating capture retained 3,600
warmup plus 7,200 profile dispatches. Both receipts prove the standalone
exact-output correctness gate and complete profile markers. The Q trace does
not include its one correctness dispatch; the rotating trace begins with 71
earlier compute commands, consistent with attachment during its 72-dispatch
correctness iteration. The analyzer therefore assigns the trailing declared
warmup/profile sequence and leaves those 71 commands as an unclassified prelude.

| Workload | Profile dispatches | Instrumented median | Kernel occupancy median | MMU limiter median | LLC limiter median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q `N=896` | 500 | 39.500 us | 25.82% | 12.76% | 14.30% |
| Rotating QKV | 7,200 | 11.959 us/dispatch | 9.83% | 25.92% | 31.95% |

The named counter samples span more than 99.8% of both profile windows. They
are device-wide, not command-buffer-exclusive. The Q and rotating captures also
used different GPU performance-state distributions. The higher median MMU and
last-level-cache limiters in the rotating window are consistent with a
different memory-system envelope, but one baseline capture per workload cannot
establish weight-cache pressure as the cause. Instrumented intervals are
diagnostic and do not replace the ordinary timing table above.

Neither capture reported a target compiler-spill event, and both sampled zero
threadgroup-memory L1 read/write bandwidth. Those observations agree with the
baseline's no-shared-memory structure but remain bounded to these captures. The
compact profile identities, selected counters, capture-boundary details, and
checksums are retained in
[`baseline-profiles.json`](baseline-profiles.json); raw traces and XML exports
remain external.

## Nonclaims

- This experiment has no M>1 or prefill workload.
- Source-derived requested-byte throughput is not observed hardware traffic.
- Operation timing does not establish model-level time per token.

## Decision

Retain this completed characterization as the timing and diagnostic baseline.
Make no production change from the baseline profiles. EXP-0005 separately
applies the frozen paired rule to the two-output candidate and retains this
one-output public path.
