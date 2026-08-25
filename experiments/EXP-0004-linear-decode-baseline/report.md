# EXP-0004: Apple GPU M=1 projection baseline

Status: **planned**

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

## Results

No recorded result yet.

## Nonclaims

- This experiment has no M>1 or prefill workload.
- Source-derived requested-byte throughput is not observed hardware traffic.
- Operation timing does not establish model-level time per token.

## Decision

Pending the recorded baseline and bounded profile attempts.
