# GQA prefill: query ownership and KV reuse

This study targets full and incremental causal prefill for Qwen2.5-0.5B on
Apple M4 Pro. The bounded comparison is being established before measurement.

## Contract and correctness

Q/O have shape `[R,14,64]`, K/V `[T,2,64]`, with `1 <= R <= T <= 4096`.
All views are contiguous row major, BF16, and non-overlapping. Query row `r`
has absolute position `T-R+r`; only keys through that position are visible.
The caller owns allocation, KV-cache contents and synchronization.

The original materialized operation rounds scaled scores and normalized
probabilities to BF16. Its `[R,14,T]` scratch remains available as probabilities.
The new cooperative softmax preserves that contract. Fused paths expose only
O: they round scores to BF16 and retain FP32 online softmax state. The Apple
MMA path additionally rounds unnormalized tile weights to BF16 before PV.
These paths have separate independent FP64 materialized oracles and must all
satisfy the existing `atol=rtol=0.015625` output gate. Tolerance is frozen.

Tests cover 28 cases, including all full-prefill sizes through 4096, ragged
tiles, incremental positions, tied/large scores and cancellation. Additional
tests perturb future K/V values, compare full and suffix prefill, verify the
cooperative probability scratch and reject unsupported shapes. Generated
NumPy arrays stay in `build/`; Python is used only for the independent oracle
and loading test fixtures, outside the engine and timed operation.

## Bounded comparison

The explicit routes are in
[`attention_prefill_support.mojo`](../src/llm_mojo/benchmarks/attention_prefill_support.mojo):

| IDs | Question |
| --- | --- |
| 0, 1 | Does a SIMD-group softmax help the materialized QK/softmax/PV path? |
| 2 | What changes when one fused dispatch retains each query's online output? |
| 3–6 | Does sharing K/V across 8, 16 or 32 query rows offset barriers and live state? |
| 7, 8 | Can Apple 8x8 matrix instructions improve QK/PV at 16x32 or 32x32 tiles? |
| 9, 10 | Does sharing K/V across two or four related query heads help? |

First screen these eleven routes at `(R,T)=(256,256),(1024,1024),(64,4096)`.
Promote a bounded subset to five full-prefill and six incremental workloads.
Compare final candidates directly with the strongest previous design as well
as the materialized control. Use the existing four-block paired protocol,
hot and ring24 modes, ten warmups and ten samples per arm, and matching
self-pair calibration. No dispatch threshold is promoted from screening alone.

Profile the baseline and finalist separately at `(16,16)`, `(1024,1024)` and
`(64,4096)`. Retain dispatch durations and named diagnostic counters; keep full
traces external. Stop at this candidate budget unless a correctness failure
requires repair. Latency is per operation, not model throughput.

The data-flow ideas follow
[FlashAttention-2](https://crfm.stanford.edu/2023/07/17/flash2.html): disjoint query
ownership, shared K/V tiles, and online output accumulation. The implementation
uses this repository's stable Mojo Apple matrix primitive. Tile sizes and
ownership must be measured on this GPU; a CUDA performance result does not
establish an Apple result.

See [the experimental method](experiments.md) and
[measurement commands](../src/llm_mojo/benchmarks/README.md).
