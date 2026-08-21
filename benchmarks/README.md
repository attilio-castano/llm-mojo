# Benchmarks

Benchmark programs are reusable measurement instruments, not experiment
records. The project's [experimental method](../docs/experiments.md) defines how
to freeze a protocol, retain raw samples, separate profiling from timing, and
bound the resulting claims.

## RMSNorm timing instrument

The RMSNorm benchmark measures the supported Apple GPU enqueue path at hidden
size 896 for row counts 1, 4, 16, 128, 512, 2048, and 4096. Those shapes cover
batch-one decode, hypothetical batched-decode-like work, and prefill-like work;
they do not prove that every semantic workload exists in the current engine.

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

For a decisive `EXP-0001` run, all four blocks use ascending, descending,
descending, ascending workload order. The recorded-run gate rejects a dirty
repository, non-AC power, a reported thermal or performance warning, an
implicit run identity, or an output path inside the repository:

```bash
uv run --locked python benchmarks/run_rms_norm.py \
  --blocks 4 \
  --experiment-id EXP-0001 \
  --run-id EXP-0001-RUN-001 \
  --recorded \
  --output-dir /absolute/external/path/EXP-0001-RUN-001
```

The output directory contains block stdout, `metadata.json`, `samples.jsonl`,
and `summary.json`. Review those external artifacts before promoting compact
evidence into `experiments/`; the runner never edits an experiment record.

## RMSNorm profiling instrument

Build a standalone, long-running binary outside the repository so Xcode does
not capture `uv` startup or Mojo compilation:

```bash
uv run --locked python benchmarks/run_rms_norm.py \
  --profile-binary /absolute/external/path/rmsnorm-profile-r1 \
  --profile-rows 1 \
  --profile-iterations 100000 \
  --require-clean
```

The build also writes a provenance JSON file with the commit, environment,
binary size, and SHA-256 digest. The binary performs an untimed correctness
gate, 1,000 warmup dispatches, and then the declared profiling iterations. A
Metal trace is diagnostic evidence; its instrumented duration is not a headline
benchmark result.

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
`gpu-performance-state-intervals`, and `graphics-compiler-spill-events`.

Reduce those external XML exports to a scrubbed summary:

```bash
uv run --locked python benchmarks/analyze_rms_norm_trace.py \
  --submissions-xml /absolute/external/path/submissions.xml \
  --gpu-intervals-xml /absolute/external/path/gpu-intervals.xml \
  --toc-xml /absolute/external/path/toc.xml \
  --performance-state-xml /absolute/external/path/performance-state.xml \
  --spill-xml /absolute/external/path/spill-events.xml \
  --profile-iterations 5000 \
  --output /absolute/external/path/trace-summary.json
```

The analyzer joins GPU intervals to the target command-buffer submissions,
validates the fixed correctness/warmup/profile sequence, strips paths and
identifiers, and reports checksums plus diagnostic interval distributions. It
does not turn source-level byte counts into hardware counters.

For generated-code inspection, emit host assembly and the per-Metal-kernel LLVM
sidecar outside the repository:

```bash
uv run --locked mojo build \
  -I src \
  -D RMS_NORM_PROFILE_ROWS=1 \
  -D RMS_NORM_PROFILE_ITERATIONS=1 \
  --emit asm \
  -o /absolute/external/path/rmsnorm-profile.s \
  benchmarks/rms_norm.mojo
```

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
