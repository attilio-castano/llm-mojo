# Combined QKV and SiLU/multiply fusion

Extend the completed QKV experiment with one change: combine the MLP's SiLU
and elementwise multiply. Configuration 26 uses configuration 25's QKV fusion
plus the combined activation; configurations 0 and 25 remain explicit controls.
Fast stays unchanged while collecting this study. The user authorized this
combined experiment after accepting the earlier QKV result as useful.

Scope: BF16 Qwen2.5-0.5B-Instruct, batch one, M4 Pro / Metal, hidden 896,
intermediate 4864, 24 layers, maximum prefill chunk 256 and per-layer K/V
row-major [4096,128]. Only single-row forwards change. Multi-row prefill, projections,
attention reductions, residuals, allocation and CPU token selection stay fixed.

One thread owns one activation product. The new kernel calls the existing
`silu_bits` followed by `multiply_bits`, preserving the intermediate BF16
rounding, signed zeros and subnormal policy. Inputs and output are contiguous,
disjoint row-major views. The decoder's existing ownership preflight remains.
No barriers, new allocations or synchronization. The allocated activation
scratch remains for unfused routes, but the combined route does not read or
write it. Each layer avoids a 4864-element BF16 write and read (19 KiB logical
traffic), or 456 KiB per 24-layer token. This is not measured DRAM traffic.

Expected structure: 410 original, 338 QKV-only, 314 combined compute launches
per token, with the same four mapping blits. The hypothesis is primarily lower
submission overhead; the launch counts do not predict a proportional speedup.

Correctness: compare materialized and fused activation bits over all 65,280
finite BF16 gate values with eight up-input patterns, plus widths 1, 127, 129
and 4864; poison output and old activation and retain protected output tails.
Keep independent existing SiLU/multiply oracle tests. Compare combined versus
original full-model logits and all 48 complete cache buffers at histories 64,
1024,3968, with poisoned activation/gating and cache/logit outputs, protected
cache extents, exact winner and submission accounting. Run full documented
repository validation and paired native streaming token/history/byte checks.

Freeze clean source and binaries. At each of the same three histories, run
four blocks of three paired comparisons: 0/0, 26/0 and 26/25. Ten warmups and
ten retained samples per arm gives 720 retained samples. Reverse arm, context
and comparison order in the two middle blocks. All timings cover fixed token
upload through forward, greedy readback and unmap. Rewind, loading, allocation,
logging and preparation stay outside timing. No profiler, internal observation
clocks or device-sync-mode in measured forwards.

Keep the original promotion gate: at every context all four combined/original
ratios must be below one, and median reduction must exceed the larger of 5%
and that context's largest absolute 0/0 deviation. Additionally require all
four combined/QKV-only ratios below one at every context, to check that the
second fusion actually helps. Retain all samples and stop at this budget;
do not repeat a completed matrix to seek promotion.

Separately capture original and combined traces at 1024 history (ten warmups,
eight measured steps each), require complete 410/314 command attribution and
four blits per step. Run four paired native terminal blocks with the same
three prompts and 128-token cap as the QKV study. Reuse the existing profiling
driver with `build --combined`, `collect`, `fusion-capture`, `fusion-terminal`,
and `fusion-archive --combined`, `fusion-replay --combined`,
`fusion-plot --combined`. Use fresh `/private/tmp/combined-fusion-*` directories.
Retain compact evidence as `combined-fusion.*` alongside the immutable earlier
QKV study. If the gate passes, promote the combined route for the measured
single-row M4 Pro domain and verify the default routing and profiling contract.
