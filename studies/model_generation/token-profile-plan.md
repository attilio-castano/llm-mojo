# Complete token profiling

Approved scope: explain resident Fast BF16 Qwen2.5-0.5B-Instruct decode on
Apple M4 Pro / Metal, then recommend one bounded optimization. Instrumentation,
local validation, collection, evidence curation and local commits are included.
Implementing the recommended optimization is a subsequent experiment.

## Frozen workloads and boundaries

- 24 distinct learned layers; capacity 4096; maximum prefill chunk 256.
- Existing-history lengths 64, 1024 and 3968. Each sample submits one fixed
  next token using production Fast selection. Repeated explanatory train text
  supplies native-tokenized histories, retained in full. This is controlled
  teacher-forced decode, not a natural generation quality or throughput claim.
- Rewind logical lengths only after greedy readback completes; overwrite the
  same suffix. Preparation, initial allocations and printing are outside timing.
- Four paired blocks, reversed arm/workload order in the middle blocks, ten
  warmups and ten samples per arm. Both observation/control and control/control
  comparisons share the same binary and session. Total: 480 retained samples.
- Host timestamps distinguish model preflight, token staging, embedding
  submission, decoder-stack submission, final norm/head submission, map/wait,
  CPU argmax scan and unmap. Compile-time observation flags eliminate clocks
  from the default specialization. No additional per-layer synchronizations.
- Six Metal System Trace captures: two per context, ten warmups and eight
  measured decode steps each, 410 expected compute dispatches per step. Each
  capture has 3280 measured dispatches, below the 5000-dispatch ceiling.
- GPU stage durations are joined by submission identity, including fragmented
  execution intervals. GPU active time, enclosing span and host timings have
  distinct boundaries and cannot be added as independent costs.
- Actual terminal measurements retain prompts, actual lengths, token events,
  first-visible latency, natural stop/reply-limit outcomes and streamed output
  parity with reporting disabled. Four repeats of three prompt lengths plus
  the existing controlling-terminal lifecycle checks.

## Validation and provenance

Require exact instrumentation-on/off logits and all 24 K/V caches, finite
values, preserved cache prefix/inactive suffix, exact token IDs and submission
accounting. Snapshot checks occur outside timing. Tooling tests reject missing,
duplicated or misordered samples and changed trace geometry. Existing trace
tests cover missing submissions and fragmented GPU intervals.

Freeze clean source after documented validation; bind binaries, source hashes,
prepared manifest, tokenizer tables, actual device/backend, software versions
and before/after power/thermal/memory/display conditions. Keep all samples,
including slow samples. Profiler runs are separate from latency collection.
Absent counters remain unavailable, never zero or measured bandwidth.

## Commands

The maintained entrypoint is `python -m llm_mojo.benchmarks.model_profile`.
Use `uv run --locked` for every Python command below. All output directories
must be new. `PREPARED` denotes an existing hash-verified local checkpoint.

```sh
uv run --locked llm-mojo-validate
uv run --locked python -m llm_mojo.benchmarks.model_profile build --prepared PREPARED --output /private/tmp/qwen-token-build
uv run --locked python -m llm_mojo.benchmarks.model_profile collect --build /private/tmp/qwen-token-build --output /private/tmp/qwen-token-timings
uv run --locked python -m llm_mojo.benchmarks.model_profile capture --build /private/tmp/qwen-token-build --output /private/tmp/qwen-token-traces
uv run --locked python -m llm_mojo.benchmarks.model_profile terminal --build /private/tmp/qwen-token-build --output /private/tmp/qwen-token-terminal
uv run --locked python -m llm_mojo.benchmarks.model_profile archive --timings /private/tmp/qwen-token-timings --traces /private/tmp/qwen-token-traces --terminal /private/tmp/qwen-token-terminal --output studies/model_generation
uv run --locked python -m llm_mojo.benchmarks.model_profile replay --output studies/model_generation
```

The result is a complete timing/capture census, a representative GPU timeline,
context-length comparison, ranked mechanisms and bounded benefit estimates.
Raw traces, expanded snapshots and binaries remain outside Git; compact
observations and provenance regenerate the report without a GPU.
