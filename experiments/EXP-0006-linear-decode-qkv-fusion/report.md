# EXP-0006: Apple GPU M=1 packed-QKV single enqueue

Status: **planned**

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

No decisive run has been recorded.

## Profiling rule

Receipt-bound Metal profiling occurs only if ordinary timing passes the frozen
promotion rule. A qualifying profile may verify dispatch counts and compare GPU
intervals, but timing alone will not be promoted into a cache, bandwidth,
occupancy, or scheduling explanation.

## Nonclaims

- This is not a prefill, `M>1`, general GEMM, or matrix-multiplication result.
- This is not an end-to-end decoder or model-performance result.
- Packing cost is excluded only because weights must be stored packed before
  decode; repacking per token is outside the candidate contract.
