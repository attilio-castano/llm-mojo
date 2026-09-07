# Materialized Qwen MLP on Metal

The baseline computes post-attention RMSNorm, gate and up projections, SiLU,
gating multiplication, down projection and residual addition. H=896, I=4864;
R counts new rows and has no KV-cache-length axis. The
[numerical contract](../../docs/mlp-sublayer.md) fixes every BF16 boundary.
The implementation is in `src/llm_mojo/mlp.mojo`; each invocation enqueues seven
kernels using caller-owned storage and one ordered stream.

This study is undergoing acceptance. Timing and holdout results are pending.

## What the primitive checks established

A direct FP32 SiLU implementation on Metal failed 506 of the 65,280 finite BF16
checks because tiny results were flushed. Merely converting an integer result
back to a BF16 scalar before the store still lost subnormals. Loading and storing
UInt16 bit patterns preserves the required rounding: for |G| below twice the
minimum normal FP32 value, the FP32 sigmoid denominator is exactly two and the
result is an integer ties-to-even halving in BF16 subnormal units. With this
path, the complete Metal SiLU sweep matches upstream bit for bit.

Gating has the same input/result-underflow issue. Ordinary products use FP32;
exceptional products use their exact, at-most-sixteen-bit integer significand
and one BF16 rounding. Tests include the frozen seven multipliers, 262,144
additional finite operand pairs, and the host reference. Overflow cases remain
explicitly counted outside the finite-output matrix.

Residual addition also failed 506 checks in a separate 195,840-pair finite-input
probe. The shared residual operation now preserves BF16 subnormal operands and
cancellation results with exact integer addition in the narrow low-exponent
range. Ordinary values retain FP32 addition. This correctness repair is covered
by the full attention regression suites as well as the new MLP checks.

The Mojo host exponential overflowed at -G=88.5 where the pinned upstream still
returns a nonzero SiLU result. The host reference uses FP32 `expf` from libm;
Metal retains its tested FP32 exponential. Neither changes the declared
negative-zero policy for G<=-89. The host and Metal full sweeps pass the frozen
activation gate. Dedicated regressions distinguish rounding SiLU before gating
and down projection before residual addition from skipping those boundaries.

## Measurement plan

Whole MLP: R=1,7,15,16,17,33,65,257,1024,4096, hot and ring24. Isolated stages:
R=1,17,1024,4096, hot only. All use the shared four-block self-pair protocol,
ten warmups and ten samples per arm. Ring24 has distinct input and weight
allocations containing the same nonuniform frozen data and shares workspace.
It changes reuse distance and synchronization amortization; it does not model
24 different layers or establish cold-DRAM behavior.

Measure latency from host enqueue through completion without debug sync.
Profile the seven stages separately from latency at R=1,17,1024,4096, retaining
validated dispatch durations. Isolated host-stage latencies need not sum to
whole-block latency. Source-requested loads are not measured DRAM traffic.

Weights occupy 26,150,656 bytes including the norm; workspace occupies
44,288*R bytes. Each of the three projections performs 2*R*896*4864 operations
when counting multiply and add separately. The rowwise control repeats weight
requests across rows, which motivates studying reuse after measuring this
baseline. No projection mapping or fusion candidate has been selected here.
