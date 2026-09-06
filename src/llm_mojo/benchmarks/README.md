# Measure an operation

The Mojo instruments keep allocations, launches, checks, and synchronization
visible. Python builds them once, alternates paired measurements, checks the
runtime identity, and saves a compact record. Kernel implementations live in
`src/llm_mojo`; instruments never substitute Python computation for GPU work.

Run the commands below as package modules from the source checkout. Building,
validation, and trace capture need that checkout for source identity and the
locked toolchain. Offline analysis and plotting also work from an installed
package when given explicit data directories. Plotting adds pinned Matplotlib
only to the command environment.

From a clean commit, after `uv run --locked python -m llm_mojo.validate` passes:

```bash
uv run --locked python -m llm_mojo.benchmarks.run build --build-dir /private/tmp/mojo-study-build
uv run --locked python -m llm_mojo.benchmarks.run run --build-dir /private/tmp/mojo-study-build --output /private/tmp/mojo-study-run
```

Both destinations must be new directories outside the checkout. Use
`--studies gqa_decode` (or any combination of the maintained study names) for a
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
`attention_prefill.mojo` records both query rows R and KV rows T. Its sixteen
routes cover materialized/cooperative softmax, streaming, query tiling, Apple
MMA, related-head reuse and five isolated compiler/resource ablations. Run `--studies gqa_prefill_screen` explicitly for
the three-workload screen; it is excluded from the default run. The maintained
full/incremental matrix is `--studies gqa_prefill`: it now pairs the original
MMA route 8 with rolled QK route 12. The five-ablation follow-up screen is
`--studies gqa_prefill_resources_screen`, also excluded from default runs.
Historical matrices remain defined by their tagged sources and frozen records.
`src/llm_mojo/benchmarks/smoke.py` exercises the other measurement routes and output gates.

Hot measures one operation through completion. Ring24 measures 24 distinct
input buffers (RMSNorm/RoPE/GQA) or weight buffers (linear), one synchronization,
and divides by 24. Output and scratch are reused. These modes have different
synchronization amortization; their difference is not a pure cache effect.
Unit is microseconds per operation, never end-to-end tokens/second.

Copy only `run.json` and `samples.csv.gz` into the relevant `studies/` folder
after checking completion. Generate the small derived summary and report image:

```bash
uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.plot
```

That command checks the raw hash and complete observation grid before plotting.
The attention study also regenerates its retained `wo_` and `wo_screen_`
comparison tables and figures, including both Wo profile variants.
No GPU execution or external temporary files are needed. PNG is the single
committed image format; extra exports are disposable. See
[the method](../../../docs/experiments.md) for interpreting evidence.

## Focused Metal profiling

The attention sublayer uses the same runner with `--studies attention_sublayer`.
The contained Wo experiment uses `--studies attention_sublayer_wo_screen`,
then `--studies attention_sublayer_wo` only after its declared screen gate
passes. Both include fresh self-pair calibration. Variant 4 changes only Wo
to the existing bias-free 8x16 MMA mapping; variant 3 is the fixed control.
For comparing stage captures use `profile_summary --attention-sublayer
--wo-comparison --prefix wo_`; build/capture both profile variants 3 and 4.
The original baseline matrix measures FP32 route 3 paired with itself over six decode,
five full-prefill and four chunked-prefill workloads, in hot and ring24 modes.
Run full validation first: its frozen synthetic case 7 supplies the instrument's
weights, nonuniform inputs, upstream cache prefix and FP32 output checks.
The builder and runner verify the actual input arrays against their frozen
hashes before and after work. Ring24 owns 24 distinct weight/input/cache
allocations, with two sign patterns; it shares output/scratch and is not a
decoder stack. Every call overwrites the same suffix. Only the host length
rewind, actual sublayer enqueue and completion are timed.

The next contained comparison uses `--studies attention_sublayer_decode_screen`
at T=64/4096, then `--studies attention_sublayer_decode` at all six decode
lengths if the declared gate passes. Benchmark variants 5/6 select FP32
G32/split64-H4 with rowwise Wo; variant 3 is the materialized FP32 control.
Both new variants require R=1 in this instrument. They contain 10/11
dispatches per call. Use `profile_summary --attention-sublayer
--decode-comparison --prefix decode_` to curate variants 3/5/6 at R=1 and
T=64/4096. The existing plot command also regenerates this retained comparison.

For the original twelve-stage Metal baseline, use the existing builder with
`--operation attention_sublayer --profile-variant 3 --profile-query-rows R
--profile-rows T --profile-warmup 10 --profile-iterations N`.
The bounded profile grid is `(1,4096), (1024,1024), (4096,4096), (64,4096)`;
`12*N` must not exceed 5,000 dispatches. Curate folders `rR-tT-v3` with
`profile_summary --attention-sublayer`, then pass the topic directory to
`plot`. Source inputs and every measured cache append are checked before
timing; the existing upstream numerical suites remain the independent gates.
The curator accepts an analysis with optional counters omitted and records
`counter_analysis.status = not_analyzed`, an empty counter list and an explicit
scope statement. A capture-local `counter_analysis_note.json`, when present,
preserves the reason and retention details. Missing counters are never zero
observations. The attention latency plot marks self-pair deviations above 5%.

The capture/analyzer pair retains binary hashes, verified launch receipts,
workload identity, dispatch segmentation and named counters. Its historical
RMSNorm/linear schema support is retained for reading older captures. The
maintained standalone builder supports GQA decode and prefill:

```bash
uv run --locked python -m llm_mojo.benchmarks.profile --build-profile-binary /private/tmp/gqa-profile --profile-variant 9 --profile-rows 4096
uv run --locked python -m llm_mojo.benchmarks.capture_trace --profile-binary /private/tmp/gqa-profile --output-trace /private/tmp/gqa-profile.trace --time-limit 2s
```

For prefill, add `--operation gqa_prefill --profile-query-rows R` to the builder;
`--profile-rows T` remains the KV length. `--profile-warmup` and
`--profile-iterations` bound the capture independently of the latency protocol.
The receipt binds R, T, tile sizes, head sharing and exact dispatch count.
The original prefill comparison profiles variants 0 and 8 at `(R,T)=(16,16)`,
`(1024,1024)` and `(64,4096)`, with ten warmups and respectively 1000, 100 and
100 measured iterations. Each capture stays below 5,000 dispatches.

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

For the prefill set, use folders `rR-tT-vV` for those six captures and pass
`--prefill-variant 8` to `profile_summary.py`. Instruments can split one dispatch
into several active intervals. The analyzer joins non-overlapping segments by
command buffer, encoder and GPU submission, sums active time, and preserves
the final segment end for the counter window. Every trailing submission must
be covered once before stages are assigned. The curator checks that same join
against the analysis and records both analysis and curation source hashes.

In `studies/gqa_prefill/`, `screen_run.json` and `screen_samples.csv.gz` retain
the bounded screen alongside the final `run.json` and `samples.csv.gz`.
The frozen specifications preserve their different controls and source
commits. One plot command rebuilds those original summaries and four figures, plus the
resource follow-up below.

The resource follow-up uses the same six-capture workload grid with variants
8 and 12. Curate it in the existing topic folder with:

```bash
uv run --locked python -m llm_mojo.benchmarks.profile_summary /private/tmp/gqa-resource-captures studies/gqa_prefill --prefill-variants 8 12 --prefix resources_
```

The curator and offline loader require the full declared capture grid and
exact dispatch sequences. No new profile schema is needed. The paired screen
uses `resources_screen_run.json` / `resources_screen_samples.csv.gz`; the
final comparison uses `resources_run.json` / `resources_samples.csv.gz`.
Keep newly collected runs outside Git until their provenance and purpose have
been reviewed. Plotting regenerates both follow-up summaries, the screen and
paired comparison PNGs, and the compact profile summary.

For intermediate Metal LLVM inspection, the small `attention_prefill_ir.mojo`
helper launches only the requested kernel with dump flags. Use schedule 0 for
original route 8, or schedules 1–5 for routes 11–15:

```bash
uv run --locked mojo run -I src -D INSPECT_SCHEDULE=2 src/llm_mojo/benchmarks/attention_prefill_ir.mojo 1024 1024 > /private/tmp/prefill-ir.log
```

This is a compiler diagnostic, not a benchmark. With the pinned toolchain,
both dump flags emit the same module; inspect one copy. The compact IR record
retains counts and hashes. Reproduction normalizes only filename metadata;
IR allocas and line counts are not physical spills or machine-code size.
The rolled reduction is selected through `SCHEDULE=2, MMA=True, BQ=BK=32,
HEADS=1` on the explicit engine entrypoint. The original control stays intact.
