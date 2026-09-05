# Measure an operation

The Mojo instruments keep allocations, launches, checks, and synchronization
visible. Python builds them once, alternates paired measurements, checks the
runtime identity, and saves a compact record. Kernel implementations live in
`src/llm_mojo`; instruments never substitute Python computation for GPU work.

From a clean commit, after `uv run --locked python tools/test.py` passes:

```bash
uv run --locked python benchmarks/run.py build --build-dir /private/tmp/mojo-study-build
uv run --locked python benchmarks/run.py run --build-dir /private/tmp/mojo-study-build --output /private/tmp/mojo-study-run
```

Both destinations must be new directories outside the checkout. Use
`--studies gqa_decode` (or any combination of the five study names) for a
bounded subset. The matrix and named implementations are in `study.py`.
The runner requires AC power, Low Power Mode off, no reported thermal warning,
a clean matching source commit, unchanged hardware/software, and the exact
built binary. Runtime output must identify an Apple device with the Metal API.

Every study includes its control paired with itself. Four blocks reverse
workload and arm order in blocks 2 and 3; each arm has ten warmups and ten
measured samples. The noise floor comes from that study's own raw self-pair
samples. A gain needs a median improvement exceeding both 5% and the largest
self-pair deviation, with all four blocks faster. Partial grids, wrong routes,
nonfinite samples, changed identity, and missing completion markers fail.

`operations.mojo` covers RMSNorm, linear decode/prefill and RoPE.
`attention_decode.mojo` retains all thirteen GQA routes; the maintained matrix
compares the materialized control, simple fusion, parallel fusion, and split
head reuse. The full numerical suites still test every original GQA candidate.
`tools/smoke.py` exercises the other measurement routes and output gates.

Hot measures one operation through completion. Ring24 measures 24 distinct
input buffers (RMSNorm/RoPE/GQA) or weight buffers (linear), one synchronization,
and divides by 24. Output and scratch are reused. These modes have different
synchronization amortization; their difference is not a pure cache effect.
Unit is microseconds per operation, never end-to-end tokens/second.

Copy only `run.json` and `samples.csv.gz` into the relevant `studies/` folder
after checking completion. Generate the small derived summary and report image:

```bash
uv run --no-project --with matplotlib==3.10.8 python benchmarks/plot.py
```

That command checks the raw hash and complete observation grid before plotting.
No GPU execution or external temporary files are needed. PNG is the single
committed image format; extra exports are disposable. See
[the method](../docs/experiments.md) for interpreting evidence.

## Focused Metal profiling

The capture/analyzer pair retains binary hashes, verified launch receipts,
workload identity, dispatch segmentation and named counters. Its historical
RMSNorm/linear schema support is retained for reading older captures. The
maintained standalone builder currently targets GQA decode:

```bash
uv run --locked python benchmarks/profile.py --build-profile-binary /private/tmp/gqa-profile --profile-variant 9 --profile-rows 4096
uv run --locked python benchmarks/capture_trace.py --profile-binary /private/tmp/gqa-profile --output-trace /private/tmp/gqa-profile.trace --time-limit 2s
```

Use `capture_trace.py --help` and `analyze_trace.py --help` for receipt and XML
export inputs. Default Metal System Trace gives dispatch timing; performance
limiter counters require a separately configured Instruments template. Never
interpret missing counters as zero. Profile separately from latency runs, keep
captures short (at most 5,000 measured dispatches), and retain compact relevant
counter observations only when the report uses them. Raw traces and binaries
stay outside Git. A profile is diagnostic; source-requested bytes are not
measured DRAM traffic and zero observed spills is limited to that capture.

To curate a validated GQA profile set, `profile_summary.py` accepts a source
directory containing variant folders `0/`, `4/`, `9/`, each with `summary.json`,
`capture.json`, `profile.provenance.json`, `conditions.json`, `submissions.xml`
and `gpu-intervals.xml`. It verifies identities/export hashes and preserves all
target dispatch durations plus selected named counter summaries. `plot.py`
then checks those retained samples and regenerates `profile_summary.csv`.
Full traces/XML are needed to redo trace analysis; they are not needed to
rebuild the report's tables or figures.
