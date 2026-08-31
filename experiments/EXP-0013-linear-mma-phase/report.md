# EXP-0013: Apple 8x8 MMA across decode and prefill

Status: **planned**

Evidence level: **operation**

## Question

Does the Apple 8x8 `simdgroup_matrix` operation improve our linear projection
against the strongest mapping already measured for each row-count regime?

## What will be compared

The candidate keeps the same `8x16` output tile and one 32-lane SIMD group as
the manual register-2x2 kernel. It divides that output tile into two adjacent
8x8 hardware fragments. At each `BK=8` K phase the group loads one 8x8 input
fragment and two 8x8 weight fragments, then issues two collective MMA
operations. Each lane retains four FP32 output values across all 112 phases.

```text
                                  Register 2x2       MMA 8x16
Output tile                       8x16               8x16
SIMD groups per tile              1                  1
FP32 accumulators per lane        4                  4
K progress                        scalar K=1         MMA K=8
Cross-lane matrix operation       no                 two 8x8 MMA/phase
Shared operand storage            0                  0
Threadgroup barriers              0                  0
```

The comparison is phase-aware because EXP-0012 already established different
best controls. `M=1,4,8` compare MMA with public rowwise. `M=16..256` compare
MMA with register-2x2. The control is frozen before timing for every row count;
it is not selected after seeing this experiment.

This M4 path calls MAX 26.5's architecture-internal `_mma_apple_8x8` primitive,
which supports Apple M1-M5 float operands. It is distinct from the public
M5-only 16x16 neural-accelerator path. The existing `W[N,K]` layout is retained,
so the logical MMA B fragment is gathered from transposed storage.

## Correctness gate

The candidate passes the committed short-prefill oracle, an exact `8x16` tile,
ragged `M=9 K=129 N=17` tails, Qwen-width `M=8 K=896 N=128`, and batch-1
`M=1 K=896 N=128`. Every lane executes both collective MMA operations; invalid
fragment elements are zero-filled before the call.

## Diagnostic gate

Two unrecorded five-repetition passes reversed both row and implementation
order. MMA improved `M=16` by about 43% and `M=64` by about 52% in both passes.
It lost at `M=1`; `M=8` changed direction with order. Those observations only
earned the formal experiment and are not retained as final performance
evidence.

## Frozen protocol

The formal run sweeps `M=1,4,8,16,32,64,128,256` at `K=896`, `N=1152`, with
24 rotating weight allocations. Four blocks use ascending/control-first,
descending/MMA-first, descending/MMA-first, and ascending/control-first order.
Each specialization has 10 warmups, at most 20 timed iterations, and 10
repetitions per block. The complete protocol and decision rules are in
[manifest.json](manifest.json).

No public dispatch change is authorized by this experiment.
