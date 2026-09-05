# RoPE: own a dimension pair

Rotary position embedding transforms paired dimensions inside each attention
head using the cosine and sine for an absolute token position. The Qwen
half-split convention pairs d with d+32 for head width 64:

```text
y[d]    = x[d] * cosine[d]    - x[d+32] * sine[d]
y[d+32] = x[d+32] * cosine[d+32] + x[d] * sine[d+32]
```

One GPU thread owns a pair: it reads both input values before writing either
output. Across heads and rows, pairs are independent. This makes ownership
simple and avoids a reduction or threadgroup barrier. The
[implementation](../../src/llm_mojo/rope.mojo) performs the contract's BF16 casts
explicitly; FP32 algebra without matching cast points is a different oracle.

For query heads, X and O each contain M×14×64 BF16 values, or 1,792 bytes per
row each. Cosine/sine tables each supply M×64 BF16 values, reused across heads.
These are unique footprints, not measured cache or DRAM traffic. The operation
has relatively little arithmetic; at small M, dispatch and completion overhead
can dominate the synchronized measurement.

This is baseline characterization. There is one implementation, measured at
M=1,16,256,4096 with hot and 24-input-buffer sweeps. The self-pair describes
noise, not a speedup. Timing inputs use constant cosine/sine for an exact
output gate; oracle tests cover actual pinned rotary tables, query decode,
incremental key positions and the half-split convention.

The caller owns position selection and table storage. A future decoder block
must prove the correct absolute cache positions and composition with Q/K
projection. A fast or correct isolated RoPE kernel cannot establish that.
