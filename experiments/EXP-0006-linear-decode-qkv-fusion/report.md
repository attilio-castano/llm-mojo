# EXP-0006: Apple GPU M=1 packed-QKV single enqueue

Status: **complete**

Evidence level: **operation**

## Question

Does prepacking Q, K, and V weights into one `M=1` projection and replacing
three enqueues per layer with one materially improve the rotating 24-layer QKV
workload without materially regressing hot QKV?

## Frozen hypothesis

Both implementations assign one scalar output dot product to one 32-lane SIMD
group. They therefore execute the same 1,152 dot products, 1,032,192
multiply-accumulates, 1,152 `warp.sum` reductions, and 288 threadgroups per
layer. The candidate changes the storage and submission composition:

- baseline: Q `[896,896]`, K `[128,896]`, and V `[128,896]` in three buffers,
  producing three output buffers through three enqueues;
- candidate: Q|K|V `[1152,896]` in one prepacked buffer, producing Q|K|V
  `[1,1152]` through one enqueue.

Packing occurs before decode and outside timing. Q, K, and V are direct regions
of the candidate output; no split copy or kernel is permitted. The candidate
does not change the linear kernel, add shared memory, compute multiple outputs
per SIMD group, or reduce weight work.

## Frozen comparison and decision

Hot QKV compares three versus one enqueue. The primary rotating 24-layer proxy
compares 72 versus 24 enqueues over 24 distinct weight sets. Each complete
workload has one completion boundary.

Four paired ABBA blocks use ascending, descending, descending, and ascending
workload order. A material effect requires at least 5% median change and the
same direction in at least three blocks. Promotion requires a material primary
improvement and no material hot-QKV regression.

The primary metric is synchronized milliseconds per complete QKV workload
iteration. Milliseconds per dispatch is not comparable because the dispatches
contain different amounts of work. The full machine-readable protocol is in
[`manifest.json`](manifest.json).

## Correctness gate

Before timing, the packed model-width result must match the host reference for
all 1,152 outputs and exactly match three separate GPU projections in BF16.
That exact comparison covers the Q/K boundary, the K/V boundary, and the final
output. Existing projection tests must remain passing.

## Results

The exact model-width correctness test passed all 1,152 packed outputs against
the host reference and against independently allocated Q, K, and V GPU
projections. That comparison includes both region boundaries and the final
output. The existing projection suite also remained passing.

Clean Metal LLVM generated from commit `f41bd35` contains one FP32
loop-accumulator PHI, one BF16 input load, one BF16 weight load, and five SIMD
shuffle calls for the reduction. It contains no threadgroup address-space
allocation or barrier. The candidate therefore reuses the proven scalar-output
kernel at `N=1152`; only buffer layout and submission composition change.

`EXP-0006-RUN-001` collected all 160 frozen samples at clean commit `c7aa38e`.
Every block proved Apple M4 Pro/Metal execution and passed fixed-commit,
AC-power, thermal, ABBA-order, and sample-count gates.

| Workload | Baseline median | Packed median | Paired change | Faster blocks | Classification |
| --- | ---: | ---: | ---: | ---: | --- |
| Hot QKV | 0.032940 ms | 0.009468 ms | -71.08% | 4/4 | Material improvement |
| Rotating QKV | 0.967034 ms | 0.411833 ms | -57.27% | 4/4 | Material improvement |

The paired percentage is the median of four within-block candidate/baseline
median ratios, not a ratio derived from the two overall medians shown for
orientation. The primary rotating workload and secondary hot workload both
improved in every block, so the candidate passes the preregistered promotion
rule. All block ratios, validation facts, generated-code checks, and raw timing
artifact checksums are retained in
[`comparison-run.json`](comparison-run.json).

## Profiling rule

Receipt-bound Metal profiling occurs only if ordinary timing passes the frozen
promotion rule. A qualifying profile may verify dispatch counts and compare GPU
intervals, but timing alone will not be promoted into a cache, bandwidth,
occupancy, or scheduling explanation.

The timing result qualified. Four clean `f41bd35` profile binaries then covered
separate and packed QKV for hot and ring24 workloads with 50 warmup and 100
profile iterations. The analyzer matched every declared trailing profile
dispatch without a command-buffer collision or duplicate interval group:

| Workload | Separate dispatches/iteration | Packed dispatches/iteration | Separate profiled dispatches | Packed profiled dispatches |
| --- | ---: | ---: | ---: | ---: |
| Hot QKV | 3 | 1 | 300 | 100 |
| Rotating QKV | 72 | 24 | 7,200 | 2,400 |

The packed dispatch lasts longer individually because it computes all 1,152
outputs, while one baseline dispatch computes only Q, K, or V. Nevertheless,
instrumented total windows and target GPU-busy time moved in the same direction
as ordinary timing. Those durations remain diagnostic because Instruments adds
overhead.

The packed captures also reported higher device-wide occupancy, inflight SIMD
groups, read bandwidth, MMU limiter, and last-level-cache limiter medians. They
are single captures with different performance-state histories, so they show a
wider active workload but do not establish why it is faster. Complete
receipt-bound facts and external artifact checksums are retained in
[`comparison-profiles.json`](comparison-profiles.json).

## Interpretation

The useful fusion is at the submission and layout level, not inside the dot
product. One SIMD group still computes one scalar output and performs one
reduction. Packing Q|K|V simply presents all 1,152 independent output rows to
one launch, allowing the GPU scheduler to see the full output width at once and
removing two projection submissions per layer. It does not reduce the number of
weight values or multiply-accumulates.

The result is especially strong for hot QKV, where fixed submission and
fragmentation costs are a larger fraction of the small workload. The rotating
proxy still improves materially even though its unique weight footprint is
unchanged. That supports prepacked single-enqueue composition, but it is not
proof that launch overhead alone caused the gain.

## Nonclaims

- This is not a prefill, `M>1`, general GEMM, or matrix-multiplication result.
- This is not an end-to-end decoder or model-performance result.
- Packing cost is excluded only because weights must be stored packed before
  decode; repacking per token is outside the candidate contract.

## Decision

Adopt prepacked Q|K|V weights and bias for the decode `M=1` composition and call
`enqueue_linear_apple_gpu` once per layer. Consume Q, K, and V as direct regions
of the packed `[1,1152]` output, without a split copy or kernel.

Keep the generic one-output projection kernel and public operation unchanged.
There is no new `M`/`N` kernel selector: the win comes from the model-load
layout and one-enqueue composition. This repository does not yet contain a
decoder/model loader in which to install that layout policy, so adding a QKV
wrapper now would be speculative; the benchmark and correctness test are the
executable contract for that future integration.
