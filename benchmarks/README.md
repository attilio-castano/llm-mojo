# Benchmarks

Benchmark programs are reusable measurement instruments, not experiment
records. The project's [experimental method](../docs/experiments.md) defines how
to freeze a protocol, retain raw samples, separate profiling from timing, and
bound the resulting claims.

## RMSNorm timing instrument

The RMSNorm benchmark measures the supported Apple GPU enqueue path at hidden
size 896 for row counts 1, 4, 16, 128, 512, 2048, and 4096. Ordinary mode
measures the public SIMD-group default. Comparison mode pairs that default with
the retained shared-tree baseline. Those shapes cover batch-one decode,
hypothetical batched-decode-like work, and prefill-like work; they do not prove
that every semantic workload exists in the current engine.

Each workload uses BF16 input, weight, and output buffers filled with ones and
reused within a repetition. Allocation, initialization, compilation, and the
untimed correctness guard are outside the measured region. The primary sample
is milliseconds per dispatch from `bencher_iter_custom`, which measures the GPU
launch through its device-completion boundary. Headline timing runs with
`MODULAR_DEBUG` unset rather than globally forcing every device operation to
synchronize.

Run one exploratory ascending block:

```bash
uv run --locked python benchmarks/run_rms_norm.py
```

The runner proves the runtime device and Metal API, records curated repository,
hardware, software, power, thermal, memory, and display metadata, parses every
reported repetition, and prints median and spread. Give it a new directory to
retain full exploratory output outside the repository:

```bash
uv run --locked python benchmarks/run_rms_norm.py \
  --blocks 2 \
  --experiment-id calibration \
  --run-id calibration-001 \
  --output-dir /absolute/external/path/calibration-001
```

For a decisive single-implementation run, all four blocks use ascending,
descending, descending, ascending workload order. The recorded-run gate rejects
a dirty repository, non-AC power, a reported thermal or performance warning,
an implicit run identity, or an output path inside the repository. Freeze a new
experiment identifier before using this form:

```bash
uv run --locked python benchmarks/run_rms_norm.py \
  --blocks 4 \
  --experiment-id EXP-XXXX \
  --run-id EXP-XXXX-RUN-001 \
  --recorded \
  --output-dir /absolute/external/path/EXP-XXXX-RUN-001
```

The output directory contains block stdout, `metadata.json`, `samples.jsonl`,
and `summary.json`. Review those external artifacts before promoting compact
evidence into `experiments/`; the runner never edits an experiment record.

For the frozen two-implementation comparison, add `--variant-comparison`. The
runner requires all four blocks and registers each row as a baseline/variant
pair. Blocks one and four use baseline then variant; blocks two and three use
variant then baseline. `summary.json` retains each block's ratio of variant
median to baseline median and applies the preregistered 5% and three-of-four
direction rules:

```bash
uv run --locked python benchmarks/run_rms_norm.py \
  --blocks 4 \
  --experiment-id EXP-0002 \
  --run-id EXP-0002-RUN-001 \
  --variant-comparison \
  --recorded \
  --output-dir /absolute/external/path/EXP-0002-RUN-001
```

## RMSNorm profiling instrument

Build a standalone, long-running binary outside the repository so Xcode does
not capture `uv` startup or Mojo compilation:

```bash
uv run --locked python benchmarks/run_rms_norm.py \
  --profile-binary /absolute/external/path/rmsnorm-profile-r1 \
  --profile-rows 1 \
  --profile-warmup-iterations 1000 \
  --profile-iterations 5000 \
  --require-clean
```

The builder defaults to the promoted SIMD-group implementation. Add
`--profile-implementation baseline` to build the retained shared-tree baseline
instead.

The build also writes a provenance JSON file with the commit, environment,
binary size, warmup and profile dispatch counts, and SHA-256 digest. The binary
performs an untimed correctness gate, the declared warmup dispatches, and then
the declared profiling iterations. Warmup defaults to and is bounded at 1,000;
use a shorter explicit warmup when a finite profiler counter window would
otherwise end before the profile region. A Metal trace is diagnostic evidence;
its instrumented duration is not a headline benchmark result. The builder
defaults to and enforces at most 5,000 profiling dispatches. Use multiple short
captures instead of increasing that limit; fewer dispatches may be appropriate
for larger workloads.

Counter streams may have an unflushed tail when a target exits immediately
after its final synchronization. For a counter capture, add a bounded host-only
idle after `PROFILE_REGION_END`; it is outside the GPU profile sequence and
defaults to zero:

Build the launch target outside macOS privacy-protected user folders. A newly
generated ad-hoc executable under `Desktop`, `Documents`, or `Downloads` can be
left suspended when `xctrace` launches it even though direct execution works.
Use an unprotected scratch path such as `/private/tmp`; copy the binary and its
provenance after capture if they need to be retained.

```bash
uv run --locked python benchmarks/run_rms_norm.py \
  --profile-binary /private/tmp/rmsnorm-counter-profile-r512 \
  --profile-rows 512 \
  --profile-warmup-iterations 100 \
  --profile-iterations 500 \
  --profile-post-idle-milliseconds 250 \
  --require-clean
```

Capture a short Metal System Trace outside the repository. Keep the raw trace
external because Instruments may record host identifiers and unrelated system
activity:

```bash
xctrace record \
  --template 'Metal System Trace' \
  --time-limit 1s \
  --output /absolute/external/path/rmsnorm-r1.trace \
  --target-stdout - \
  --launch -- /absolute/external/path/rmsnorm-profile-r1
```

Use `xctrace export --toc` to inspect the capture configuration, then export the
needed tables by schema rather than by a trace-specific table index:

```bash
xctrace export \
  --input /absolute/external/path/rmsnorm-r1.trace \
  --toc \
  --output /absolute/external/path/toc.xml

xctrace export \
  --input /absolute/external/path/rmsnorm-r1.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-application-command-buffer-submissions"]' \
  --output /absolute/external/path/submissions.xml
```

Repeat the schema export for `metal-gpu-intervals`,
`gpu-performance-state-intervals`, and `graphics-compiler-spill-events`. For a
trace configured with a named counter set, also export `gpu-counter-info` and
`gpu-counter-value`.

Reduce those external XML exports to a scrubbed summary:

```bash
uv run --locked python benchmarks/analyze_rms_norm_trace.py \
  --submissions-xml /absolute/external/path/submissions.xml \
  --gpu-intervals-xml /absolute/external/path/gpu-intervals.xml \
  --toc-xml /absolute/external/path/toc.xml \
  --performance-state-xml /absolute/external/path/performance-state.xml \
  --spill-xml /absolute/external/path/spill-events.xml \
  --warmup-iterations 1000 \
  --profile-iterations 5000 \
  --output /absolute/external/path/trace-summary.json
```

For a named-counter capture, use the shorter warmup and profile counts declared
by that binary, then add both counter exports to the analyzer command:

```bash
uv run --locked python benchmarks/analyze_rms_norm_trace.py \
  --submissions-xml /absolute/external/path/submissions.xml \
  --gpu-intervals-xml /absolute/external/path/gpu-intervals.xml \
  --toc-xml /absolute/external/path/toc.xml \
  --counter-info-xml /absolute/external/path/gpu-counter-info.xml \
  --counter-values-xml /absolute/external/path/gpu-counter-values.xml \
  --warmup-iterations 100 \
  --profile-iterations 500 \
  --output /absolute/external/path/counter-summary.json
```

After four valid summaries have been collected in baseline, variant, variant,
baseline order, apply the frozen paired-direction and 5% materiality rule:

```bash
uv run --locked python benchmarks/compare_rms_norm_counters.py \
  --baseline-first /absolute/external/path/capture-01/trace-summary.json \
  --variant-second /absolute/external/path/capture-02/trace-summary.json \
  --variant-third /absolute/external/path/capture-03/trace-summary.json \
  --baseline-fourth /absolute/external/path/capture-04/trace-summary.json \
  --warmup-iterations 100 \
  --profile-iterations 500 \
  --output /absolute/external/path/counter-comparison.json
```

The comparator rechecks the capture validity gates, requires positive medians
in all four captures, and calls a difference repeatable only when both adjacent
pair ratios move in the same direction and their median relative change is at
least 5%.

The analyzer joins GPU intervals to the target command-buffer submissions,
validates the fixed correctness/warmup/profile sequence, strips paths and
identifiers, and reports checksums plus diagnostic interval distributions. If
counter exports are supplied, it rejects a trace whose named counter samples do
not overlap the declared profile window. Those samples remain device-wide,
rather than command-buffer-exclusive. The analyzer does not turn source-level
byte counts into hardware counters.

For generated-code inspection, emit host assembly and the per-Metal-kernel LLVM
sidecar outside the repository:

```bash
uv run --locked mojo build \
  -I src \
  -D RMS_NORM_PROFILE_ROWS=1 \
  -D RMS_NORM_PROFILE_WARMUP_ITERATIONS=1000 \
  -D RMS_NORM_PROFILE_ITERATIONS=1 \
  -D RMS_NORM_PROFILE_POST_IDLE_MILLISECONDS=0 \
  -D RMS_NORM_PROFILE_SIMDGROUP=true \
  --emit asm \
  -o /absolute/external/path/rmsnorm-simdgroup-profile.s \
  benchmarks/rms_norm.mojo
```

Omit `RMS_NORM_PROFILE_SIMDGROUP` to inspect the baseline.

## Memory quantities

The instrument keeps these quantities separate:

- allocated footprint: the live input, weight, and output buffers;
- logical tensor traffic: BF16 input, weight, and output, or 6 bytes per
  row-hidden element;
- program-requested traffic for this implementation: two input reads, one
  weight read, and one output write, or 8 bytes per element;
- observed hardware traffic: only a named counter from a documented profiling
  capture.

The two derived byte rates in `summary.json` are source-level accounting. They
are not measurements of cache, fabric, DRAM traffic, or physical bandwidth.
