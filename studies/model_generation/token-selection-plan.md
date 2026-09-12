# GPU token selection experiment

The control is the promoted configuration 26 (QKV and SiLU/multiply fusion),
with the existing BF16 vocabulary projection and CPU greedy scan. No decoder
arithmetic changes in this experiment. Target: Apple M4 Pro, Metal, batch one.

Two bounded candidates:

1. Existing projection followed by a two-stage GPU argmax: 1024 logits per
   128-thread group, then one group reduces the partial winners.
2. Projection fused with local selection: 64 logits per 128-thread group,
   preserving the existing lane-strided dot-product accumulation, warp sum,
   addition of zero and BF16 rounding. A second kernel selects the global winner.

Each partial and final result contains an ordered BF16 score, token ID and
nonfinite flag (three UInt32 values). All finite scores, including subnormals,
are ordered by their BF16 bits; signed zeros compare equal. Ties select the
lowest token ID. Any nonfinite score invalidates the result. Timed fused
execution does not materialize the full logits. An untimed specialization
materializes them to compare every BF16 value with the original projection.

Verification covers boundaries, ties, nonfinite rejection, poisoned scratch,
unchanged logits in the nonmaterializing path, and exact full-model cache,
logit and token agreement at histories 64, 1024 and 3968. Run the complete
repository validation before freezing the measured source.

The primary measurement is complete-token wall time, from token upload through
winner readback, with normal asynchronous layer submission and no internal
observation clocks. Four blocks at each history; ten warmups and ten retained
samples per arm. Comparisons: CPU/CPU, GPU argmax/CPU, fused head/CPU and fused
head/GPU argmax. Reverse arm and comparison order in middle blocks. Total:
960 retained samples. Traces, if captured, are separate diagnostics, never
latency samples. Streaming confirmation uses the existing three prompts and
128-token replies, four blocks, all three arms, with exact output and history.

A candidate qualifies only if all four paired block ratios are below one and
its median paired reduction exceeds max(5%, largest absolute CPU self-pair
variation) at every history. If both qualify, prefer the simpler GPU argmax
unless the direct fused-head/argmax comparison clears that same gate at every
history. If only one qualifies, choose it; otherwise retain the CPU default.
No tuning after observing acceptance samples. Preserve raw samples, model and
binary hashes, source identity, hardware/software and power conditions in a
replayable compact archive. Promotion is limited to the measured single-row
Apple M4 Pro Fast/auto policy; explicit historical controls stay available.

## Reproduction

After validation and a clean local source commit, use existing profiling tooling:

```bash
uv run --locked python -m llm_mojo.benchmarks.model_profile build --selection \
  --prepared /absolute/path/to/prepared-model --output /private/tmp/token-selection-build
uv run --locked python -m llm_mojo.benchmarks.model_profile collect \
  --build /private/tmp/token-selection-build --output /private/tmp/token-selection-timings
uv run --locked python -m llm_mojo.benchmarks.model_profile selection-capture \
  --build /private/tmp/token-selection-build --output /private/tmp/token-selection-traces
uv run --locked python -m llm_mojo.benchmarks.model_profile selection-terminal \
  --build /private/tmp/token-selection-build --output /private/tmp/token-selection-terminal
uv run --locked python -m llm_mojo.benchmarks.model_profile selection-archive \
  --timings /private/tmp/token-selection-timings --traces /private/tmp/token-selection-traces \
  --terminal /private/tmp/token-selection-terminal --output studies/model_generation
uv run --locked python -m llm_mojo.benchmarks.model_profile selection-replay \
  --output studies/model_generation
```

The model driver uses explicit selection modes 0/1/2 and configuration 26.
The native terminal uses `combined`, `gpu-argmax` and `fused-head` respectively;
these controls stay fixed if Fast is subsequently promoted. Full-model snapshots
verify both the diagnostic materializing specialization and the actual timed
path. Scratch is allocated once with the model, never inside the timed step.

The paired-ratio figure is regenerated from the archive with:

```bash
uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.model_profile selection-plot --output studies/model_generation
```
