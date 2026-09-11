# Measure an operation

The Mojo instruments keep allocations, launches, checks, and synchronization
visible. Python builds them once, alternates paired measurements, checks the
runtime identity, and saves a compact record. Kernel implementations live in
`src/llm_mojo`; instruments never substitute Python computation for GPU work.

Run the commands below from the source checkout. Building,
validation, and trace capture need that checkout for source identity and the
locked toolchain. Offline analysis and plotting also work from an installed
package when given explicit data directories. Plotting adds pinned Matplotlib
only to the command environment.

From a clean commit, after `uv run --locked llm-mojo-validate` passes:

```bash
uv run --locked llm-mojo-bench build --build-dir /private/tmp/mojo-study-build
uv run --locked llm-mojo-bench run --build-dir /private/tmp/mojo-study-build --output /private/tmp/mojo-study-run
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
The bounded GQA work-distribution study uses
`--studies attention_sublayer_parallelism_screen`, comparing integrated control
9 with BQ16/BQ8 (10/11) and KV split4/split8 (12/13). Run
`--studies attention_sublayer_parallelism --parallelism-screen /path/to/attention_sublayer_parallelism_screen`
only after the screen completes. The runner validates its build identity and
selects at most one qualifying candidate per family using the frozen rule in
[the contract](../../../studies/attention_sublayer/plans.md#contained-gqa-parallelism-comparison).
These two studies are excluded from the default run. Both stages include
control self-pairs, complete attention calls and any partial-state merge.
For profile curation, use `--attention-sublayer --parallelism-variants 9 FINALIST... --prefix parallelism_`
at `(64,4096)` and `(1024,1024)`, with 25 measured iterations and ten warmups.
`src/llm_mojo/benchmarks/smoke.py` exercises the other measurement routes and output gates.

The projection-tile follow-up starts with
`--studies attention_sublayer_tiles_screen attention_sublayer_tiles_kernel_screen`.
Both compare 9/14/15; the first times the complete block and the second only
Wo on frozen upstream attention inputs. After both complete, use
`--studies attention_sublayer_tiles attention_sublayer_tiles_qkv`
with `--tile-screen /path/to/attention_sublayer_tiles_screen` and
`--tile-kernel-screen /path/to/attention_sublayer_tiles_kernel_screen`.
The runner requires a tile to qualify at both boundaries in both modes at
full 1024, binds the screens to this build, and selects at most one. No winner
means stop this family after screening. QKV transfer changes only QKV, leaving
Wo at the control mapping. All these studies are opt-in.

The independent `attention_sublayer_split_domain` study compares existing 9/13
at R=16/64/256 and T=1024/4096. Run control-only `attention_sublayer_timing` and
`attention_sublayer_timing_buffered` beforehand to diagnose sensitivity to
printing between samples. The latter buffers observations until both arms
finish; timing still includes enqueue through completion. The parser verifies
each requested measurement boundary. The [bounded plan](../../../studies/attention_sublayer/plans.md#contained-projection-tiles-and-split-domain-follow-up)
defines the matrices, selection and interpretation. Retain these files under
their matching `tiles_`, `split_domain_` and `timing_` prefixes in the existing
attention study; the common plotter reconstructs all tables and figures.

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

The attention study keeps records and regenerated CSVs in
`studies/attention_sublayer/data/` and generated PNGs in `figures/`.
External run directories remain flat. Numerical `.json` summaries referencing
`.json.gz` records can be read with `study.load_numerical_record()`; it verifies
both compressed and original hashes before returning the full record.

## Focused Metal profiling

The complete resident Qwen decode study uses
`uv run --locked python -m llm_mojo.benchmarks.model_profile` with `build`,
`collect`, `capture`, `terminal`, `archive` and offline `replay` commands.
Its [bounded plan](../../../studies/model_generation/token-profile-plan.md)
defines fixed histories, observation overhead controls and full-model trace
coverage. It reuses the capture/analyzer submission join and requires verified
local model assets; operation-level synthetic fixtures are not a substitute.

Profile curation accepts one `--attention-study` name: `baseline`, `wo`,
`decode`, `prefill`, `projections`, `parallelism`, or `combined`. Parallelism
also requires `--parallelism-variants` with the selected finalists. Historical
`--attention-sublayer --…-comparison` flags remain compatible aliases. The
same fixed grids and dispatch validation apply to both spellings.

The attention sublayer uses the same runner with `--studies attention_sublayer`.
The contained Wo experiment uses `--studies attention_sublayer_wo_screen`,
then `--studies attention_sublayer_wo` only after its declared screen gate
passes. Both include fresh self-pair calibration. Variant 4 changes only Wo
to the existing bias-free 8x16 MMA mapping; variant 3 is the fixed control.
For comparing stage captures use `profile_summary --attention-study wo --prefix wo_`; build/capture both profile variants 3 and 4.
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
dispatches per call. Use `profile_summary --attention-study decode --prefix decode_` to curate variants 3/5/6 at R=1 and
T=64/4096. The existing plot command also regenerates this retained comparison.

The contained FP32 prefill experiment uses
`--studies attention_sublayer_prefill_screen`, then
`--studies attention_sublayer_prefill` if the predeclared screen gate passes.
Benchmark 7 selects FP32 rolled MMA prefill with MMA Wo; its fixed control is
4 (materialized FP32 plus MMA Wo). Both require R>1 in this comparison. Use
`--profile-variant 4` or `7` and curate with
`--attention-study prefill --prefix prefill_` at
(1024,1024),(4096,4096),(64,4096). The dispatch counts are 12/10; the default
attention route remains unchanged. Gates and the complete protocol are in
[the attention contract](../../../studies/attention_sublayer/plans.md#contained-fp32-prefill-comparison).

The integrated QKV/Wo study uses `--studies attention_sublayer_projections
attention_sublayer_integrated`. These are two fresh paired runs on the same
fifteen workloads: variant 9 versus 8 holds the FP32 GQA and Wo policy fixed
and measures QKV packing/tiling plus its layout copy; 9 versus 3 measures all
selected mappings together against the original baseline. Variant 9 calls
`enqueue_attention_sublayer_integrated` directly and executes nine dispatches;
8 keeps separate Q/K/V and executes ten. Use profile variants 8/9 at the four
baseline profile workloads, then curate with `--attention-study projections --prefix integrated_`. The shared plot command
regenerates `projections_`/`integrated_` comparisons and integrated profiles.
See the [declared policy and gates](../../../studies/attention_sublayer/plans.md#integrating-the-projection-studies-end-to-end).

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

## Combined 16x16 projections

After numerical validation, run `--studies attention_sublayer_combined` to
compare integrated control 9 with combined 16x16 QKV/Wo variant 18 across the
existing fifteen workloads. This opt-in comparison keeps GQA fixed and needs
no new screen: both components already qualified independently. Profile 9/18
at (256,256), (1024,1024), (4096,4096), (64,4096), with ten warmups and
25/25/10/25 measured iterations. Curate with `profile_summary SOURCE OUTPUT
--attention-study combined --prefix combined_`. Retain the
run, samples, profiles and validation under that prefix; the normal plotter
regenerates both latency and stage-time figures.

The split8/projection closure uses
`--studies attention_sublayer_split_combined_projections attention_sublayer_split_combined_gqa`.
It compares 13/19 (projection gain with split8 fixed) and 18/19 (split8 gain
with new projections fixed) over the seven predeclared cached-chunk workloads.
Each comparison includes its own control self-pairs; both are opt-in. Keep
results under `split_combined_projections_` and `split_combined_gqa_` in the
existing attention study. No screen or new profile capture is required for
this composition of existing kernels. See the
[closure contract](../../../studies/attention_sublayer/plans.md#closing-the-split8-and-projection-integration-gap).

## MLP baseline and projection campaign

After numerical acceptance and a clean source commit, `--studies mlp` runs the
whole-block self-pair matrix. `mlp_stage_0` through `mlp_stage_6` measure the
seven isolated stages on identical upstream operands, in hot mode only. The
same builder includes `mlp.mojo`; ordinary route smoke verifies both full-block
buffer modes, every stage, and invalid requests. Inputs are verified prefixes
of the frozen 4,096-row development case, seed 1601. Ring24 owns distinct copies
of inputs/weights and shares workspace; all Python transport and checks occur
outside timing. The instrument buffers sample output until both arms finish.

Profile with `--operation mlp --profile-variant 0 --profile-rows R`, where
R is 1,17,1024,4096. Use ten warmups and respectively 500,100,25,10 measured
iterations; each capture stays below 5,000 dispatches. The common capture and
analyzer validate the MLP entrypoint, intermediate width, and seven-dispatch
sequence. Curate `rR-v0` directories with `profile_summary SOURCE OUTPUT --mlp`.
Retain whole-block `run.json`/`samples.csv.gz`, isolated-stage files prefixed
`stage_N_`, and profiles in `studies/mlp_sublayer/data/`. The common plot command
regenerates the measurement tables and figures from those compact records.

The explicit mapping IDs are 0 for all-rowwise, 1/2/3 for gate/up-only
8x16/16x16/8x32, 4/5/6 for down-only with the same tile order, and 7 for
16x16 gate/up/down. Mapping 0 remains the default; there is no row-count
selector. Route smoke covers all configurations in both whole-MLP modes,
the seven stages, isolated projection ring routes, and real non-self output
parsing in both arm orders, including a nonzero control.

The bounded studies extend the same runner. `mlp_gate_screen` and
`mlp_down_screen` screen the mappings at R=1,16,17,1024 in both modes.
`mlp_up_confirmation` checks the selected gate mapping on up's distinct weights.
`mlp_gate_up` compares whole MLP 0 versus 2; `mlp_down_increment` compares 2
versus 7. `mlp_final` directly compares 0 versus 7 over all ten row counts and
both modes, retaining 3,200 observations with matched control self-pairs.
Isolated projection ring24 shares one exact upstream operand across 24
distinct weight copies. Whole-MLP ring24 also uses distinct input copies.

The final profile grid uses variants 0 and 7 at the four row/iteration pairs
above. Create the output directory, then curate the eight `rR-vV` capture
directories with `profile_summary SOURCE OUTPUT --mlp --mlp-variants 0 7
--mlp-rows 1 17 1024 4096 --prefix optimization_final_`. A declared subset of
rows/variants supports intermediate profiles; the reader still requires the
complete declared grid and exact seven-dispatch sequence. Retain final timing
and profile files with `optimization_final_` in the existing study's `data/`.

## Single-token MLP decode

The bounded follow-up is declared in
[decode-plan.md](../../../studies/mlp_sublayer/decode-plan.md). Variants 8/9/10
combine gate/up launches and/or use two outputs per SIMD group; 11/12 change
only down to two/four cooperating groups per output. They reject R != 1
before any enqueue. IDs 13..18 encode possible compositions; the confirmation
runner measures at most one composition of independently qualified families.
Mapping 0 remains the default. Combined launch variants keep separate weight
and G/U buffers: there is no repacking, allocation or output copy.

From a validated clean source, use the common builder and runner:

```sh
uv run --locked llm-mojo-bench build --build-dir /private/tmp/mlp-decode-build
uv run --locked llm-mojo-bench run --build-dir /private/tmp/mlp-decode-build --output /private/tmp/mlp-decode-screen --studies mlp_decode_gate_up mlp_decode_down
uv run --locked llm-mojo-bench run --build-dir /private/tmp/mlp-decode-build --output /private/tmp/mlp-decode-confirmation --studies mlp_decode_final --mlp-decode-screen /private/tmp/mlp-decode-screen
```

Both modes must qualify under the frozen calibrated rule. The confirmation
runner verifies both screen specifications, samples and build identity before
selecting the minimum worst-mode ratio per family. No qualified family leaves
a control self-pair confirmation. Timing is whole MLP in every decode screen.
Isolated stage APIs still enqueue only the named stage; launch combining occurs
only in the complete MLP entrypoint.

Profile control and selected (or explicitly diagnostic) variant at R=1 with
10 warmups and 500 measured iterations. Combined gate/up has six dispatches;
other mappings have seven. The shared receipt, analysis and curation tools
validate that distinction. Curate using `profile_summary SOURCE OUTPUT --mlp
--mlp-variants 0 V --mlp-rows 1 --prefix decode_`. Keep run/sample records under
`decode_gate_up_`, `decode_down_`, `decode_final_` in the existing MLP data
folder. The normal plot command regenerates decode tables and figures.

The explicit `mlp_acceptance.py --decode` path captures only the declared
fresh single-row holdouts, after a clean candidate/binary freeze. It requires
the verified local checkpoint directory and refuses to overwrite its manifest.
The candidate must have a numerical build receipt. Use the
[recorded numerical evaluation workflow](../../../docs/development.md#tests)
to launch the exact candidate and bind complete results to its build and fixture
hashes. Capture completion alone does not establish numerical acceptance, and
`MLP_CANDIDATE_BINARY` is no longer accepted as proof of execution. Older splits
remain regression data. Decode
variants exercise row-one prefixes of all existing fixtures and varying input
rows under asynchronous reuse; original variants still exercise full/chunked
multi-row execution.

## Decoder baseline and configuration selection

The common builder includes the composed decoder. `--studies decoder_layer`
runs its six-workload baseline in hot and ring24 modes. The opt-in
`decoder_selection_*` studies reuse this runner for calibration, screening and
independent confirmation of explicit decoder configurations. `select-decoder`
freezes screen proposals; confirmation runs require `--decoder-screen`, and
`confirm-decoder` records accepted exact shape/mode cells with ID 0 fallback.
Decoder IDs differ from standalone attention and MLP mapping IDs.

See the [selection plan](../../../studies/decoder_layer/selection-plan.md) for
workloads and acceptance gates, and the [completed study](../../../studies/decoder_layer/selection.md)
for retained evidence and offline reproduction. Profile with
`--operation decoder_layer`; curate with `--decoder-layer` and, for selected configurations,
`--decoder-selection FILE --prefix selection_`. Complete captures retain the
actual 15/16/17-dispatch route. Diagnostic profile durations do not determine
latency promotion.

## CPU tokenizer

The tokenizer uses the same four-block paired protocol and sample summaries,
with an explicit CPU backend. From a clean validated commit, after tokenizer
setup:

```sh
uv run --locked llm-mojo-bench build-tokenizer --build-dir /private/tmp/tokenizer-build
uv run --locked llm-mojo-bench run-tokenizer --build-dir /private/tmp/tokenizer-build --output /private/tmp/tokenizer-run
uv run --locked --with matplotlib==3.10.8 llm-mojo-bench report-tokenizer --build-dir /private/tmp/tokenizer-build --output /private/tmp/tokenizer-run
```

The report command checks retained hashes and the complete calibration grid; it
needs no tokenizer artifact or execution. `rows` in this shared sample format is
an opaque case ID for CPU text workloads. `cases` records the actual byte count,
piece distribution, and output token count; `layers=1` means one sequence. Modes
separate pre-split BPE, complete encoding, decoding, streaming, and table loading.
The last two decode modes expose whole-call and incremental API paths over the
same byte-decoder implementation. Table loading includes deserialization and
native structure checks, not preparation or Python's startup SHA-256 checks.

The [QKV fusion experiment](../../../studies/model_generation/qkv-fusion.md)
extends `model_profile` with `build --fusion`, the same calibrated collection
matrix, and `fusion-capture`, `fusion-terminal`, `fusion-archive`,
`fusion-replay` and `fusion-plot`. Configuration 25 remains experimental;
Fast continues selecting the control for decode.
