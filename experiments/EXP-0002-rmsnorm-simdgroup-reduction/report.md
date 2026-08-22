# EXP-0002: Apple GPU RMSNorm SIMD-group reduction comparison

Status: **planned**

Evidence level: **operation**

## Question

Does one hybrid SIMD-group reduction materially reduce synchronized Apple GPU
RMSNorm dispatch latency versus the shared-memory tree without a relevant
workload regression?

## Frozen hypothesis

The baseline uses 128 FP32 shared partials and nine workgroup barriers. The
variant will retain the same 128-thread mapping and per-thread FP32 accumulation
while reducing within four 32-lane SIMD groups, storing four shared partials,
and using the first SIMD group for the cross-group sum. The expected structure
is four FP32 shared values and two workgroup barriers.

This is one explicit alternative, not a general reduction framework. Generated
code will confirm structure; only ordinary paired timing can establish whether
the variant is useful.

## Frozen comparison

All seven EXP-0001 row counts remain in scope. Four blocks pair the
implementations per workload in ABBA implementation order while also reversing
the workload order. Each implementation retains 40 samples per workload.

For every block and workload, divide the variant median by the baseline median.
A material improvement requires a median paired ratio at or below 0.95 and a
faster variant in at least three blocks. A material regression uses the
symmetric 1.05 threshold and direction rule. Smaller effects are inconclusive.

## Results

No variant or paired measurement exists yet.

## Decision

No production decision yet. The shared-tree implementation remains the public
entrypoint until correctness, generated-code, and paired timing gates pass.
