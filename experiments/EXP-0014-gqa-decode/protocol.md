# EXP-0014: bounded GQA decode campaign

Status: protocol frozen before candidate measurements. User approved local
implementation, validation, profiling, benchmarks and commits on 2026-09-05.
Starting commit: `92c967d`; branch: `codex/gqa-decode-campaign`.

## Contract

BF16 contiguous row-major Q/O `[1,14,64]`, K/V `[T,2,64]`, `1 <= T <= 4096`.
Query head `h` reads KV head `h // 7`. KV already includes the current token.
An explicit output-only experimental entrypoint keeps the probability-returning
materialized API intact. Caller retains non-overlapping buffers through device
completion. No allocations or synchronization inside the enqueue entrypoint.
Scaled scores retain BF16 rounding; online normalization and output accumulation
use FP32. ATOL=RTOL=0.015625 is frozen, as in the existing attention fixture.
No tolerance adjustment after observing results.

Independent NumPy FP64 materialized calculations on BF16 inputs diagnose online
reduction error; the existing pinned Transformers fixture and BF16 materialized
reference enforce compatibility. Tests include nonuniform data, head mapping,
multiple seeds, extreme finite/tied scores, cancellation, T=1, ragged SIMD and
split boundaries, empty splits, repeated output/workspace reuse and invalid shapes.
Existing attention prefill and all repository validation remain required.

## Candidate budget and sequence

At most 12 distinct parameter configurations; fixes to incorrect code are
recorded and never reported as timing evidence. Initial candidates:

1. `g1-h1-s1`: one SIMD group per query head; lanes own d=l and l+32.
2. `g2-h1-s1`, `g8-h1-s1`, `g32-h1-s1`: partition T within a threadgroup;
   merge (maximum, denominator, 64 unnormalized output values) in shared memory.
3. `g1-h1-s4`, `g1-h1-s16`, `g1-h1-s64`: partition T across threadgroups;
   FP32 partial states and a separate stable merge kernel. Total call has two
   dispatches, including empty-split handling.
4. At the best measured split count, try h=2,4,7 with g=1, reusing K/V loads
   across related query heads, including ragged groups within each group of 7.
5. Up to two additional configurations only when a measured concern motivates
   a specific instruction/tile change. Record that hypothesis before timing.

Each candidate must pass correctness before timing. Compare against the
materialized baseline and strongest candidate so far. A stage-1 slowdown does
not prevent testing the declared parallel mappings. Stop at the cap or when
remaining hypotheses lack supporting evidence; negative results complete work.

## Timing

Training lengths: 1,16,64,256,1024,4096; confirmation lengths: 7,32,128,512,2048,
4095 and a fresh input seed. Both hot and ring24 are measured separately. Ring24
uses 24 distinct KV buffers and measures one sequential sweep, divided by 24;
hot measures one call. These are different synchronization amortizations and
must not be compared as a cache-only experiment or as decoder throughput.

Host monotonic clock brackets enqueue(s) through explicit device completion.
All dispatches, host submission and completion overhead are included. Allocation,
input preparation, compilation, oracle gates, warmup, readback and profiler
overhead are excluded. Per-pair allocation and inputs are identical. Initially
10 warmup samples and 10 measured samples per arm in each of four blocks.
Workload order is ascending/descending/descending/ascending; pair order is
control-first/candidate-first/candidate-first/control-first. No debug sync mode
in timing. Run a materialized/materialized pair to establish noise before tuning.

Retained runs require clean commit, verified runtime Apple GPU/Metal, AC power,
no thermal/performance warnings, power mode, displays, memory, locked software,
binary/source hashes and raw samples. A gain must be >=5%, exceed baseline
self-pair noise for that workload/mode, and have the same direction in all four
blocks. Report >5% regressions. No averaging away bad shapes; shape-specific
choices need confirmation. Freeze finalists before fresh-seed/held-out timing.

## Profiling and retention

Extend existing trace capture/analyzer narrowly for attention identity, shape,
parameters and dispatch counts. Profile baseline and finalists at T=16,256,4096
in separate short captures, <=5000 profiling dispatches. Bind receipt, binary,
commit, runtime device and region markers. Distinguish device-wide counters from
kernel-specific evidence. After two failed capture attempts at a given blocker,
retain the failure and continue valid timing with unresolved causal attribution.
Raw binaries/traces stay outside Git; compact samples, manifests, plots and
accepted/rejected decisions are committed. No model tokens/s claims.

Algorithm references: [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf),
[Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html), and
[MLX Metal SDPA](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h).
