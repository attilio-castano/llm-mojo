# Bounded single-token QKV fusion experiment

Compare explicit decoder configuration 25 against configuration 0 on M4 Pro /
Metal, using BF16 Qwen2.5-0.5B-Instruct. Fast remains unchanged until the gate
below passes. Configuration 25 is restricted to one row with control integrated
GQA/projection mappings. Prefill remains on the existing Fast choices.

The candidate combines QKV unpack, both rotary applications and cache append.
Four groups of 128 threads own 512 rotary pairs (448 Q, 64 K); the first 128
threads also copy one V element each. Packed projection output is already BF16.
Use the original explicit BF16 product roundings and final addition/subtraction;
write Q and the new K/V cache row directly. No scratch reuse, barriers, new
allocation, projection arithmetic, attention arithmetic or MLP changes.

Correctness gates: exact isolated comparison to the existing four-kernel path
at positions 0, 1, 63, 64, 1023, 1024, 3968 and 4095, with diverse BF16 bit
patterns and poisoned skipped scratch; full-model exact logits, all 48 full
cache buffers, protected prefixes/inactive suffixes and token/submission
accounting at cached histories 64, 1024 and 3968. Native streaming chat compares
all token IDs, histories and emitted bytes over four paired repeats of three
128-token replies. The full documented repository validation must pass.

Freeze clean source and binaries before measurement. Reuse the token profiling
matrix: three fixed histories, four blocks, ten warmups and ten samples per
arm, both candidate/control and control/control; 480 samples. Reverse arms and
workload order in the two middle blocks. No profiler or debug synchronization
in timing runs. Loading, allocation and rewind remain outside timing. No host
observation clocks inside either candidate or control forward/greedy.

Promotion requires every context to pass: all four candidate/control ratios
below one, and median reduction greater than both 5% and that context's largest
absolute control self-pair deviation. An inconclusive result is retained with
the control still selected; do not rerun to obtain a preferred result.

Separately capture one control and one candidate trace at 1024 prior tokens,
ten warmups plus eight measured steps each. Verify 410 versus 338 compute
commands, plus four mapping blits per step; retain all active fragments and
complete trailing command coverage. Structural capture failures are reported,
not converted into speed claims. Actual terminal measurements corroborate the
fixed-step result; they do not replace the calibration gate.

Use the existing driver with `uv run --locked python -m
llm_mojo.benchmarks.model_profile`:

```sh
uv run --locked llm-mojo-validate
uv run --locked python -m llm_mojo.benchmarks.model_profile build --fusion --prepared PREPARED --output /private/tmp/qkv-fusion-build
uv run --locked python -m llm_mojo.benchmarks.model_profile collect --build /private/tmp/qkv-fusion-build --output /private/tmp/qkv-fusion-timings
uv run --locked python -m llm_mojo.benchmarks.model_profile fusion-capture --build /private/tmp/qkv-fusion-build --output /private/tmp/qkv-fusion-traces
uv run --locked python -m llm_mojo.benchmarks.model_profile fusion-terminal --build /private/tmp/qkv-fusion-build --output /private/tmp/qkv-fusion-terminal
uv run --locked python -m llm_mojo.benchmarks.model_profile fusion-archive --timings /private/tmp/qkv-fusion-timings --traces /private/tmp/qkv-fusion-traces --terminal /private/tmp/qkv-fusion-terminal --output studies/model_generation
uv run --locked python -m llm_mojo.benchmarks.model_profile fusion-replay --output studies/model_generation
uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.model_profile fusion-plot --output studies/model_generation
```

The study-only native chat build takes an eighth argument selecting `fast` or
`fusion`, leaving the public CLI interface unchanged. Both arms run in the same
binary; receipts bind the source, executable, assets, hardware/software and
before/after machine conditions. Keep raw traces, binaries and snapshots
outside Git; retain compact samples, fragment timelines and numerical records.
