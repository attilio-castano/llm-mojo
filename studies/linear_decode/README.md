# Linear decode: share input and combine QKV

One token produces Q[896], K[128] and V[128] from X[896]. The affine operation
uses weights W[N,K] with contiguous K, FP32 accumulation, promoted BF16 bias,
and one final BF16 output cast. Here K=896 and packed N=1152.
[The source](../../src/llm_mojo/linear.mojo) keeps each implementation explicit.

Packed weights contain 1,032,192 BF16 elements: 2,064,384 bytes per layer,
about 47.25 MiB across 24 layers. The input is only 1,792 bytes. Decode has one
token row, so weights have little reuse across tokens within an operation.

| Mapping | Launches per QKV | Lane ownership |
| --- | --- | --- |
| Separate Q, K, V | 3 | A SIMD group reduces one output's K dimension; each lane handles K/32=28 elements |
| Packed QKV | 1 | Same arithmetic ownership over 1152 outputs; Q/K/V are contiguous output slices |
| Packed two-output | 1 | A SIMD group computes two adjacent outputs and reuses each loaded input element |

Packing combines launches; it does not remove the weight dot products. Two
outputs reuse input loads while doubling accumulator state and reducing the
number of independent groups. Both tradeoffs need measurement. The two-output
entrypoint intentionally supports only M=1.

The maintained comparison uses M=1, with one weight buffer and a rotating set
of 24 weight buffers. All arms use the same packed allocation; the separate
control addresses its Q/K/V slices. Every layer's full output is checked before
timing, with NaN poisoning to expose unwritten elements. The independent
numerical suite separately covers varied inputs, weights, bias and ragged K.

Historical EXP-0004 through EXP-0006 explored these steps. The fresh study
compares both alternatives directly with separate QKV under one common
protocol. It does not change the engine's public projection dispatcher or
measure projection plus RoPE, cache updates or attention.
