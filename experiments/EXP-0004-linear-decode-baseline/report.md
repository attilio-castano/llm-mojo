# EXP-0004: Apple GPU M=1 projection baseline

Status: **planned; timing complete, profiling pending**

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

Baseline Q and rotating-workload profile attempts remain pending. Timing is
complete and sufficient to begin the separately frozen candidate comparison;
EXP-0004 remains planned until its bounded profiling record is closed.

## Nonclaims

- This experiment has no M>1 or prefill workload.
- Source-derived requested-byte throughput is not observed hardware traffic.
- Operation timing does not establish model-level time per token.

## Decision

Retain this clean run as the candidate comparison baseline. Complete the
receipt-bound profile attempts before marking EXP-0004 complete.
