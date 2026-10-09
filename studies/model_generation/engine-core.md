# Engine core: readiness, correctness and load declaration

Implementation authorized on 2026-10-08 from baseline
`cb2416abf3c9fbb19c460fa99709d93eeabba97e` on `codex/engine-core`.
**Status: synchronous core and bounded load studies validated.** Mixed execution,
request lifecycle, KV-pressure replay, optional fitted budgeting and optional
lifetime-reservation admission are implemented. Asynchronous LLM stepping remains
open on the locked runtime. The original study binds clean implementation
`b18563b`; its final follow-up added acceptance tests and retained evidence.
The [admission successor](#lifetime-reservation-admission-bounded-successor-study)
binds `b81ea6c` and removes replay on both frozen pressure traces.
The [serving plan](../../docs/serving-plan.md) defines the architecture;
[experimental method](../../docs/experiments.md#serving-measurement-contract)
defines retention and paired comparisons.

The native `EngineCore` accepts arrivals and aborts at step boundaries, schedules
decode rows and one prefill tail, and delivers ordered token/finish events. Its
Metal runner completes each step before cache ownership changes. The trace driver
exercises this backend with fixed arrivals; the existing terminal chat still uses
its direct Fast session. Frontend integration follows this core milestone.

```mermaid
flowchart LR
    arrivals[Arrivals and aborts] --> engine[EngineCore]
    engine --> batch[StepBatch]
    batch --> runner[QwenRunner and Metal]
    runner --> engine
    engine --> events[Token and finish events]
    engine <--> cache[BlockManager and KVPool]
```

## Readiness receipt

The [retained readiness receipt](engine-core-readiness.json) passes the baseline
and clean candidate gates. Its [lossless validation archive](engine-core-validation.json.gz)
retains the logs, checkpoint reports, commands, build receipts and small asset
manifests. Baseline source is the exact `cb2416a` archive, whose temporary Git
identity is recorded separately. The baseline suite passed 296 Python tests and
28 native test files; the candidate suite passed 302 Python tests and 31 native
files during development, followed by the final 310-test Python suite. These
are separate recorded scopes, not a combined count from one invocation.

The [final acceptance extension](engine-core-acceptance.json) passes 17 native engine tests, including abort
after partial prefill and actual decode. A full-checkpoint numeric-fault driver
passes ordinary and device-sync-mode runs on Apple M4 Pro/Metal. It submits a
mixed step through all 24 layers, detects a deliberately nonfinite tied-head
weight during greedy readback, emits one error finish for each of three live
requests, returns all blocks and rejects subsequent work. Prepared files remain
unchanged. This tests numeric-fault cleanup, without a device-loss recovery claim.
Production Mojo hashes remain identical to the measured `b18563b` implementation.

Before phase 3 evidence collection, retain one `engine-core-readiness-v1` receipt
with `schema_version`, baseline and candidate source identity, toolchain and
asset identity, and a check list. Every check has a name, exact command, status
(`passed`, `failed`, `unavailable` or `not_run`), exit code where applicable and
the retained result's hash. `ready` is true only when all required checks passed.
Missing assets or a stopped suite cannot produce a ready receipt.

Required baseline checks are:

- clean source/build identity and the stable versions resolved through `uv.lock`;
- `uv run --locked llm-mojo validate`, including frozen oracle anchors, all
  Python/native tests, Unicode tokenizer and benchmark smokes;
- clean full-checkpoint `validation.model lifecycle` and `validation.model batch`
  runs in the default 32-slot slot-major layout;
- a generation report accepted by `validate_generation_events`, proving
  `Apple M4 Pro/metal`, route selection, token budget, cache/submission accounting
  and completion;
- prepared model and tokenizer manifests and their hashes.

The full suite uses device-sync-mode. Repeat checkpoint lifecycle in ordinary
and device-sync modes; timing collection removes device-sync-mode. Record the
checks' actual source and binaries, not only the machine's available backend.
Use new output paths. The October 5 adoption and October 6 device-proof repeat
remain historical evidence; they are not a new phase 3 readiness pass.

## Staged correctness gates

The initial Metal runner uses reference configuration 27 for every arm: BF16
stored tensors with FP32 reductions and the same row arithmetic across solo,
batched and mixed shapes. It establishes a common scheduling baseline, rather
than the existing Fast kernel selection. A faster Fast engine route requires
its own numerical diagnosis and paired comparison.

The chunked implementation uses one ordered Metal stream, one prefill sequence per
step, a fixed 256-token total step budget, and no prefix-cache reuse. Full prefill
comparison arms use up to 4,096 rows in a standalone prefill step. Admission
is bounded. Decode runs before prefill; preemption retains delivered history,
releases blocks and recomputes that history. Queue capacity, maximum sequences,
KV pool capacity and watermark are explicit configuration, never inferred from
available machine memory.

The simulated runner returns scripted tokens and advances a virtual clock. It
uses the same scheduler and block manager as Metal, so it establishes lifecycle
and ownership under deterministic arrivals, stops, limits, aborts, rejection and
memory pressure. Check invariants after every step, including allocation across
block boundaries, abort while waiting/prefilling/decoding, and full drain.
It establishes no GPU performance or native numerical agreement.

Exact gates:

- Existing S = 1 and decode-only Fast gates remain required. Configuration 27's
  solo, batched and mixed rows agree byte for byte with equivalent execution
  using that same reference route.
- Mixed steps use correct absolute positions and slot mappings, select only
  requested logits, preserve existing cache prefixes and inactive storage,
  and change only the scheduled sequences' own write slots.
- Token/sequence budgets and allocator invariants hold after each step.
  Once drained, every request has one terminal event and no block is referenced.
- Rejected adds and invalid steps leave histories, blocks and counters unchanged.
  Delivered tokens are ordered, valid, neither omitted nor duplicated, and never
  delivered after a request finishes. Recomputed history is not delivered again.
- A Metal execution failure has an explicit failed result for every in-flight
  request. Automatic restart and frontend replay are phase 5 work.
- Asynchronous and synchronous execution on an identical frozen step schedule
  deliver identical tokens. Stop/limit/abort tests account separately for any
  extra submitted token that is discarded.

Reference-route mixed steps and recomputed histories have exact own-route gates.
Numerical diagnostics compare the reference with existing Fast execution on the
same supplied token histories.
Retain token agreement, first divergence, logit distances, finite-boundary
checks and route identities. Different Fast chunk/batch shapes may select
different arithmetic; these comparisons have no invented closeness threshold.
Finite outputs, storage ownership and accounting remain required gates.

## Workload declaration

Before collection, freeze a versioned trace containing mode, generator version,
seed and one record per request; its hash is its identity:

```text
request_id, arrival_offset_ns, prompt_ids, max_new_tokens, stop_ids,
abort_offset_ns, output_script
```

The trace's SHA-256 binds every field and its ordering. `abort_offset_ns` is null
when absent. `output_script` is absent in the initial driver. Scripted simulation
uses the trace's global `scripted_tokens` oracle: input token plus absolute
position indexes that script, so interleaving and recomputation preserve its
selections. It is not per-request teacher forcing. Every request's
prompt plus output budget fits 4,096 tokens. Arrival offsets are fixed before
an arm runs; the collector does not change them in response to its speed.

Use the existing native tokenizer/conversation fixtures or the existing declared
synthetic token construction, with pinned provenance. Freeze exact token arrays
and lengths rather than requesting an unspecified category of short prompts.
The bounded first matrix includes an offline trace (all arrivals at zero), an
online short/long prompt mix, and a KV-pressure trace. It must exercise prefill
while other requests decode, uneven completion, replenishment and block growth.
An exploratory capacity probe may set online rates; retain it separately, then
freeze the confirmatory rates, seeds and request counts before collecting them.
The initial synthetic generator freezes eight prompts of 32, 128, 64, 512, 96,
1,024, 256 and 64 tokens, each with a 32-token output budget, using the existing
affine token construction with a request-specific offset. The collector defaults
to 128 blocks and eight sequence slots; both are explicit run fields. An online
rate must be supplied and is recorded with its seed and arrival offsets.
No latency target is selected by this declaration.

The separate adaptive mechanism uses a provisional 25 ms predicted-execution
research target. That is a configuration for its calibration/evaluation, not a
target-capacity or client-SLO claim, and does not change the fixed study's null
target. Its full request-level goodput definition remains undeclared.

The first arms add one mechanism at a time:

| Arm | Scheduled work |
| --- | --- |
| Serial | One request at a time; standalone full prefill |
| Static | A cohort prefills, then decodes; the next cohort joins after drain |
| Continuous | Requests join at boundaries; standalone full prefill |
| Chunked | Continuous decode shares steps with one bounded prefill chunk |

Keep source, model, pool capacity, request limits and trace identical across
arms; record each arm's intentional scheduler differences. Fitted budgeting and
asynchronous stepping are later arms with separate declarations and validation.
Natural greedy runs retain their actual trajectories and work counts. Frozen
same-history replays isolate scheduling and numerical differences; they are
explicitly marked and do not stand in for natural request performance.

## Collection and data schema

Extend `benchmarks/model_contract.py` for the frozen declaration and
`benchmarks/model_profile.py` for build, collect, archive and replay. The native
token-trace driver belongs in `benchmarks/`; reuse `serving/` for the engine and
both runners. Keep records in this existing study topic. No separate experiment
tree or Python inference engine is required. The `model_profile` commands are
`engine-specification`, `engine-build`, `engine-collect` and `engine-replay`.
Collection requires a clean build receipt, pinned prepared assets, actual runtime
device identity and frozen trace JSON. Replay needs only the archive and manifest.
The completed bounded results and their replay commands follow below.

A run manifest contains the schema-versioned declaration, build/source hashes,
binary hashes, prepared-model/tokenizer hashes, exact trace document and its
hash, scheduler configuration,
runner/output mode, timing boundary, warmup count, paired block order and
environment/conditions. The card identifies pinned model/tokenizer assets,
vocabulary/stop IDs, BF16/FP32 policy, 4,096-token limit, layout, block size,
pool capacity and actual runtime device/backend. The initial collector retains
build/asset identity rather than claiming the future API's startup model card.
A simulated run identifies a virtual clock and cannot satisfy a Metal claim.

Each run carries its block, arm, calibration role, mode and unchanged trace.
It retains complete native stdout plus validated parsed records in order. The
initial native line grammar is:

```text
device Apple M4 Pro/metal
mode greedy
config arm blocks token_budget max_sequences
arrival request_id scheduled_ns actual_ns
token request_id token_id prompt_tokens generated_tokens arrival_ns emitted_ns
finish request_id reason prompt_tokens generated_tokens arrival_ns emitted_ns
drained request_count step_count free_blocks elapsed_ns
```

Scripted mode instead names `simulated/virtual`. `generated_tokens` counts
delivered tokens, starting at one. One terminal event closes each arrival,
including a zero-output limit.
Arrival and abort offsets are retained in the input trace. The initial
performance matrix contains valid requests; explicit rejection is an acceptance
test, not an invented collector event.

Each completed step contains:

```text
step step_id decode_seqs prefill_seqs prefill_tokens total_tokens
     attended_positions admitted preempted finished aborted waiting blocks_free
     begin_ns schedule_ns build_ns execute_ns postprocess_ns end_ns
     predicted_ns budget_limited
```

The step occupies one line; `ENGINE_STEP_FIELDS` freezes its order. Fixed-budget
runs require `predicted_ns` and `budget_limited` to be zero. An adaptive run has
a separate frozen policy declaration; its `policy` line names target, fixed,
per-row, per-position, per-attention-partition and per-logit-row nanosecond
coefficients. The initial fixed study has no target or fitted gain claim. Define
`attended_positions` as the sum of causal KV lengths seen by all scheduled query
rows, rather than the maximum context multiplied by batch size. Greedy timings
use a monotonic host clock; scripted timings use the virtual clock. `execute_ns`
covers upload, submission, waiting and readback together. `end_ns - begin_ns`
retains overhead between sections. GPU stage attribution requires separate
captures. Asynchronous telemetry needs a separate schema with submitted and
readback step identities.

### Frozen adaptive policy

`benchmarks/engine_budget.py` implements `fit`, `evaluate` and `replay` for a
separate adaptive study. Calibration consumes the complete checked Metal
fixed-budget archive. It fits nonnegative coefficients for fixed cost, token
rows, attended positions, attention partitions and sampled logit rows, then
rounds them upward to integer nanoseconds. The frozen policy binds the exact
build, calibration archive, workload and every calibration sample. Evaluation
requires a different canonical native token trace; changing only an offline
seed label cannot make the same workload independent.

Evaluation has four paired blocks of fixed chunked, chunked/self-calibration
and adaptive execution. Native `predicted_ns` must reproduce the frozen
coefficients for each executed step. The engine chooses the largest prefill
chunk under its predicted target, while mandatory decode or one-token progress
can exceed it. Replay recomputes the fit, calibration residuals, evaluation
predictions and request metrics from retained data. It reports prediction
errors, constrained steps, measured/predicted target overruns and actual work;
it establishes no goodput or asynchronous claim. The held-out results below
measure its latency/throughput tradeoff; it remains optional.

## Pairing, metrics and replay

Run ten representative warmup steps per arm outside timing before each measured cell.
Reset request/block state and drain the stream between runs while keeping model
weights resident. Use four paired blocks, reversing trace and arm order in
blocks two and three, with control/self-control calibration from the same build
and session. Each online block is a complete trace replay; requests within it
are not independent repeats. Keep all measured records and condition snapshots.

Record TTFT from scheduled arrival to first delivered token,
each delivered-token interval, TPOT from first to last token divided by the
number of intervals, request end-to-end time and terminal reason. TPOT is null
for fewer than two delivered tokens. Report per-block distributions and p50,
p95 and p99 with the sample count and a declared quantile convention. Offline
throughput divides delivered tokens by the makespan from first arrival through
complete drain. Online reports separate the arrival window, measured duration,
drain tail, actual delivered work and accepted/rejected/aborted/completed counts.

Freeze the predicted-execution research target before cost fitting. The initial
adaptive setting is 25 ms; the fixed study has no target and neither study
defines goodput. Freeze client latency targets and the goodput formula before
corresponding latency or target-capacity claims. Fit on calibration records; freeze coefficients and
evaluate predictions on separate traces. No target is fabricated from a result,
and simulation speed is not measured hardware throughput. The paired offline
decision rule can establish a bounded gain/regression/inconclusive result;
online evidence reports the declared tradeoff and cannot infer better user
latency from aggregate tokens/s alone.

Offline replay checks hashes, the entire declared trace/arm/block grid, runtime
device evidence, monotonic step IDs/timestamps, complete request histories,
stop/limit/abort accounting, budget/drain summaries and all raw-derived metrics.
Native acceptance paths check full allocator invariants; an aggregate free-block
counter cannot prove them independently.
Regression tests must show it rejects omitted steps/tokens, duplicate delivery,
incorrect termination, a wrong device and a changed summary even when envelope
hashes are updated. Compact lossless archives, readable manifests and summaries
are retained; weights, binaries, arrays and full traces stay outside Git.

## Completed load studies

The [results card](engine-core-results.json) recomputes the retained matrix's
72 complete natural-greedy trace runs: three fixed-arm
grids of 20 runs and a separate 12-run adaptive evaluation. Each trace has eight
requests. Every run drained, returned its full KV pool and passed event, budget,
device and work-accounting checks. All paired arms delivered identical histories
using reference configuration 27. These are short synthetic token workloads on
one M4 Pro/Metal machine, with BF16 storage, FP32 reductions, 32-slot slot-major
KV, context limit 4,096, MAX 26.5.0 and Mojo 1.0.0. They establish no production
capacity, semantic quality, Fast-route speedup or client goodput result.

Tables report the median of four per-block metrics. Each block's latency
quantiles use linear interpolation: eight request samples for TTFT/TPOT and
248 delivered-token intervals in the fixed traces. Individual blocks, request
histories and quantiles remain in the archives. Requests within a block are not
independent repeats.

### Adequate KV capacity

The [offline archive](engine-core-offline.json.gz) uses all arrivals at zero,
128 blocks and eight sequence slots. Every arm delivers 256 tokens and computes
2,424 token rows. Serial has 256 steps; static/continuous have 39 and chunked has
46. Their control/self-control noise floor is 5.86%.

| Arm | Tokens/s | TTFT p50, ms | TPOT p50, ms | Token gap p95, ms | Token gap p99, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial | 65.0 | 1,602.4 | 10.7 | 12.2 | 13.1 |
| Static | 158.9 | 421.6 | 38.6 | 14.4 | 983.5 |
| Continuous | 158.5 | 428.7 | 38.4 | 14.3 | 983.0 |
| Chunked, 256 rows | 149.3 | 455.1 | 37.8 | 151.8 | 192.4 |

Median paired makespan ratios against serial are 0.41855, 0.41543 and 0.43726,
respectively; all qualify as faster under the declared offline rule. Chunked
makespan falls 56.3% versus serial. Batching decode amortizes per-step work across
requests. Chunking introduces more prefill steps and changes when that work stalls
decodes: its p95 gaps rise, while its p99 gaps fall sharply versus continuous.
Both percentiles matter; aggregate throughput cannot select a token-latency policy.

The [online archive](engine-core-online.json.gz) freezes seed 19 and a Poisson
arrival rate of four requests/s before execution, with the same prompts, outputs
and pool. It reports a finite trace distribution, rather than a capacity claim.

| Arm | Tokens/s | TTFT p50, ms | TPOT p50, ms | Token gap p95, ms | Token gap p99, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial | 67.7 | 500.6 | 10.3 | 12.2 | 13.2 |
| Static | 77.7 | 254.5 | 10.6 | 11.9 | 14.8 |
| Continuous | 82.8 | 109.2 | 12.2 | 12.6 | 371.1 |
| Chunked, 256 rows | 82.1 | 111.5 | 12.3 | 133.9 | 183.7 |

Continuous admission lowers queueing for this trace. Chunking again exchanges
rare large stalls for more frequent smaller stalls. Neither policy dominates
every latency statistic.

### KV pressure

The [pressure archive](engine-core-pressure.json.gz) reuses the offline trace
with 40 blocks. Every request fits alone; their combined working set does not.

| Arm | Tokens/s | Preemptions per run | Computed token rows | Paired makespan / serial |
| --- | ---: | ---: | ---: | ---: |
| Serial | 64.8 | 0 | 2,424 | 1.000 |
| Static | 22.8 | 62 | 18,488 | 2.855 |
| Continuous | 23.2 | 62 | 18,488 | 2.805 |
| Chunked | 27.9 | 57 | 16,432 | 2.378 |

All four blocks have the same preemption and work counts. Concurrent arms are
slower under the 5% noise floor. History retention and replay establish finite
progress and exactly-once delivery, but repeated admission/eviction creates
thrashing. Chunked replay computes 6.78 times the serial token work. This result
motivated the [lifetime-reservation successor](#lifetime-reservation-admission-bounded-successor-study)
below. These historical measurements remain unchanged; the successor uses a
fresh same-build incremental control.

### Fitted execution budget

The [frozen intent](engine-core-intent.json) selects the provisional 25 ms
predicted synchronous execution target before fitting. The
[policy archive](engine-core-policy.json.gz) binds the complete offline calibration
archive and retains every calibration sample. Its fitted cost is

```text
predicted_ns = 9,926,060 + 431,597 * token_rows + 320 * attended_positions
```

The fitted attention-partition and logit-row coefficients are zero. That is a
result of this joint fit, not evidence those operations have zero physical cost.
Calibration MAE is 1.31 ms and p95 absolute error is 3.61 ms; maximum error is
49.23 ms. Upward integer rounding does not turn least squares into a bound.

The [held-out archive](engine-core-adaptive.json.gz) uses seed 29, 16 arrivals/s,
different token IDs and prompt lengths 48, 192, 80, 768, 144, 1,536, 320 and 128,
with 24 output tokens per request and 128 blocks. Its 184 token intervals and
eight request samples per block are separate from calibration. The frozen policy
selects prefill chunks of at most 34 rows, versus the fixed total budget of 256.

| Held-out arm | Tokens/s | TTFT p50, ms | TPOT p50, ms | Token gap p95, ms | Token gap p99, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed chunked | 68.3 | 602.9 | 82.7 | 244.3 | 279.0 |
| Fitted budget | 59.1 | 805.8 | 22.1 | 24.1 | 25.0 |

Every paired history matches. Adaptive/control makespan ratios are 1.18427,
1.31504, 1.03495 and 1.35307 (median 1.24966); self-control variation reaches
13.10%. These online results establish the observed tradeoff, without an offline
gain classification. Prefill is spread over more steps, reducing decode stalls
while increasing overhead and time to finish prompts.

All 648 adaptive step predictions are at most 25 ms. Measured executions exceed
it in **3, 2, 2 and 109 of 162 steps per block**, totaling 116/648. The last
block's token-gap p95 is 31.0 ms, versus 23.8–24.4 ms in the other blocks.
Held-out per-block prediction MAE ranges 2.30–3.14 ms; p95 absolute error ranges
4.50–6.14 ms. The model is useful for a tested scheduling tradeoff, but does not
establish a reliable latency bound. The fixed default stays 256; fitted budgeting
requires an explicit policy. Client SLOs and goodput remain undeclared.

### Numerical comparison with Fast

The [diagnostic archive](engine-core-diagnostics.json.gz) supplies identical
histories to reference 27 and Fast: three prompts of 32, 512 and 1,024 tokens,
256-row prompt chunks and eight supplied decode rows each. All six processes
and 62 captured calls complete on Metal; all 305,225,216 captured BF16 boundary
elements are finite. It retains routes, native output, file identities, FP64
logit distances and the regeneration script. Full arrays remain outside Git.

Greedy choices agree at 26/27 requested prediction points. The first difference
is the third prediction of request 0, at cached length 34: reference chooses
3914 and Fast configuration 26 chooses 50, with maximum logit difference 0.15625
and logit RMS difference 0.036307. The fixed supplied history continues after
that difference. Reference matches all supplied engine-history choices. These
are diagnostics without an invented closeness threshold; they do not promote
configuration 27 as a replacement for Fast.

### Replay and next work

Archives include raw native event/step records, frozen trace documents, complete
source/binary/asset identities and condition snapshots. Adaptive replay embeds
the original calibration archive and refits its coefficients. The readable
envelopes bind compressed and uncompressed hashes. Replay either an external
collection directory or the retained archive file:

```sh
uv run --locked python -m llm_mojo.benchmarks.model_profile engine-replay \
  --output studies/model_generation/engine-core-offline.json.gz
uv run --locked python -m llm_mojo.benchmarks.model_profile engine-replay \
  --output studies/model_generation/engine-core-online.json.gz
uv run --locked python -m llm_mojo.benchmarks.model_profile engine-replay \
  --output studies/model_generation/engine-core-pressure.json.gz
uv run --locked python -m llm_mojo.benchmarks.engine_budget replay \
  --output studies/model_generation/engine-core-adaptive.json.gz
```

Replayed summaries equal the summaries retained inside the archives. Full derived
request-summary files are generated locally and ignored by Git; the compact
results card remains readable evidence. The admission successor below addresses
repeated preemption. Cost-model variability and the cost of reserving maximum
output capacity remain study questions. Multi-prefill steps, a measured Fast runner, prefix
caching and frontend integration remain separate work. The asynchronous gate
below is still open for LLM stepping.

## Lifetime reservation admission: bounded successor study

Implementation authorized on 2026-10-09 from clean baseline `d0e3626`.
The optional `EngineCore(..., reserve_lifetime=True)` policy reserves physical
KV blocks when admitting a request. The existing incremental admission remains
the default and the comparison control. This changes ownership and scheduling,
not model arithmetic, cache layout, GPU kernels or the default 256-row budget.

For a positive output limit, peak cached extent is
`prompt_length + max_new_tokens - 1`: the last emitted token is returned without
being processed into KV. Demand is the ceiling of that extent divided by the
32-slot block size. Reserve the whole demand for the oldest waiting request,
or leave it waiting. Younger requests cannot bypass that FIFO head. Committed
length and written KV still grow only when a synchronous step completes.
Future reserved blocks contain no written tokens. Stop, limit, abort and fatal
execution error release both resident and unused future blocks.

The watermark is an admission margin whenever any request holds KV, including
residents not selected in the current step. Ignore it when no residents remain,
so a request validated to fit alone can run even with a full-pool watermark.
With finite feasible arrivals, resident work is bounded and its growth already
has capacity; oldest-first scheduling eventually releases that capacity and
admits the head waiter. This is a finite-drain argument, not an arrival-rate SLO.
Accepted zero-output requests finish at the next boundary without reservation;
their existing prompt-feasibility validation is preserved.

Execution work is independent of reservation size. Retain the chosen fixed or
fitted prefill row count when constructing a step. Do not infer that count from
the larger reserved extent. The initial physical-reservation implementation
uploads complete block tables, including future blocks; measurements include
that metadata overhead. Logical credit accounting is deferred until evidence
justifies the extra ownership ledger.

The versioned admission declaration uses reference configuration 27, BF16
storage with FP32 reductions, context 4,096, 32-slot slot-major KV, 256 rows,
eight sequence slots, zero watermark and ten warmup steps outside measurement.
Build one clean binary and compare chunked incremental with itself and with
chunked reserved in four balanced blocks. Reuse the exact retained offline and
online trace documents at 128 blocks, then both traces at 40 blocks. These four
collections contain 48 measured finite trace replays. Adaptive fitting, chunk
tuning, Fast execution and asynchronous stepping are outside this milestone.

Correctness gates require exact natural greedy histories across the two modes,
one terminal event per request, complete pool return, and zero conservative
preemptions. On the retained offline pressure trace, the work gate is 2,424
computed rows rather than the incremental control's 16,432 rows and 57
preemptions. Report all four blocks' makespan, throughput, TTFT, end-to-end
latency and p95/p99 token gaps. Apply the existing paired verdict only to
declared offline makespan; retain online distributions without a capacity or
client-SLO claim. Timing gains are hypotheses: reservation can defer admission
and increase TTFT even when it removes replay. The results below retain the
mode as an optional policy.

### Validation and retained results

All four collections use one clean `b81ea6c` binary on Apple M4 Pro/Metal,
Mojo 1.0.0 and MAX 26.5.0. The [result card](engine-admission-results.json)
binds source blobs, locked dependencies, binary and asset hashes, device,
conditions and every block's metrics. All 48 natural-greedy runs finish eight
requests, deliver 256 tokens and return the complete pool. Control, self-control
and reserved histories agree exactly within every collection. All 16 reserved
runs compute the necessary 2,424 rows with zero preemptions. Every offline
pressure control and self-control computes 16,432 rows with 57 preemptions.

The [validation manifest](engine-admission-validation.json) and
[lossless archive](engine-admission-validation.json.gz) retain the completed
`uv run --locked llm-mojo validate` invocation: 316 Python tests, 31 native test
files in 32 invocations, frozen oracle anchors and route smokes. Its 26 native
engine tests cover reservation arithmetic, FIFO blocking, watermarks, rejection
atomicity, unused-block cleanup, all terminal paths and 192 finite-arrival cases.
Source hashes before and after validation match the clean measurement build.
Separate full-checkpoint ordinary and device-sync acceptance runs preserve
natural greedy histories, release future blocks after partial-prefill abort,
and release all requests after an injected nonfinite head fault. Prepared assets
remain unchanged. Omitted and explicit incremental selectors produce identical
records after the explicit selector header is removed.

The tables report medians of four block-level metrics. Each block's TTFT,
end-to-end and TPOT quantiles use eight requests; token-gap quantiles use 248
intervals with linear interpolation. Online results and latency differences
are descriptive. Offline makespan uses the declared paired rule and a 5% noise
floor; adequate-capacity timing is inconclusive, while pressure passes faster
in all four blocks.

| Trace and pool | Incremental tokens/s | Reserved tokens/s | Incremental / reserved preemptions | Incremental / reserved computed rows | Reserved/control makespan, median | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [Offline, 128 blocks](engine-admission-offline.json.gz) | 166.3 | 167.0 | 0 / 0 | 2,424 / 2,424 | 0.99919 | Inconclusive |
| [Online, 128 blocks](engine-admission-online.json.gz) | 83.2 | 83.5 | 0 / 0 | 2,424 / 2,424 | 0.99756 | Distribution only |
| [Offline, 40 blocks](engine-admission-pressure.json.gz) | 31.7 | 123.7 | 57 / 0 | 16,432 / 2,424 | 0.25615 | Faster |
| [Online, 40 blocks](engine-admission-online-pressure.json.gz) | 28.9 | 83.2 | 44 / 0 | 16,246 / 2,424 | 0.34682 | Distribution only |

Online pressure has 44 incremental preemptions in every run. Computed rows
vary from 16,245 to 16,248 across controls and self-controls because arrivals
are applied at measured step boundaries; the table shows the control median.

Offline pressure makespan ratios by block are 0.25165, 0.25614, 0.25616 and
0.25624: a 74.4% median reduction. Removing replay reduces computed rows by
85.2% and raises aggregate throughput about 3.9 times. This comparison uses
the fresh incremental control, rather than the older 27.9 tokens/s measurement.

| Trace and admission | TTFT p50/p95/p99, ms | End-to-end p50/p95/p99, ms | TPOT p50, ms | Token gap p95/p99, ms |
| --- | ---: | ---: | ---: | ---: |
| Offline 128, incremental | 409.4 / 1,198.5 / 1,208.7 | 1,460.2 / 1,536.2 / 1,538.5 | 33.9 | 136.8 / 172.2 |
| Offline 128, reserved | 407.4 / 1,195.4 / 1,205.6 | 1,457.4 / 1,530.2 / 1,532.7 | 33.9 | 136.7 / 172.1 |
| Online 128, incremental | 119.0 / 489.8 / 584.5 | 680.8 / 1,142.3 / 1,144.6 | 10.3 | 117.7 / 163.3 |
| Online 128, reserved | 118.3 / 491.2 / 585.7 | 679.2 / 1,149.0 / 1,153.3 | 10.7 | 117.7 / 163.3 |
| Offline 40, incremental | 407.3 / 7,775.7 / 7,784.8 | 4,102.9 / 8,074.5 / 8,077.3 | 110.1 | 119.3 / 136.0 |
| Offline 40, reserved | 407.3 / 1,772.1 / 1,781.3 | 813.7 / 2,065.2 / 2,068.4 | 13.2 | 32.7 / 127.1 |
| Online 40, incremental | 206.2 / 5,950.3 / 5,965.1 | 4,114.2 / 6,995.4 / 7,308.0 | 33.3 | 136.1 / 136.2 |
| Online 40, reserved | 119.4 / 657.2 / 822.6 | 497.3 / 949.2 / 1,103.9 | 10.2 | 13.1 / 92.6 |

Schedule/build telemetry measures host preparation. Median total preparation
falls from 0.551 to 0.270 ms in offline pressure and from 0.591 to 0.345 ms in
online pressure; the result card retains per-step p95 and every block. Upload
of complete reserved tables remains inside `execute_ns`, together with GPU
submission, synchronization and selected-token readback. These measurements do
not isolate metadata upload or GPU stage time.

Reservation solves replay thrashing for these traces, but it reserves declared
maximum output capacity even when a request later stops early. FIFO also delays
younger requests behind a large waiter. Adequate-capacity online tail TTFT and
end-to-end metrics are slightly higher in this sample, without a declared
latency verdict. Incremental admission therefore remains the default; reservation
is an explicit choice for bounded workloads under pressure. A wider sweep of
output limits, early stops and arrival rates is the next evidence boundary.
No client SLO, goodput, target capacity, Fast-route improvement or asynchronous
stepping claim follows from this study.

### Reproduce the admission evidence

The [benchmark commands](../../src/llm_mojo/benchmarks/README.md#engine-token-traces)
build and collect the optional same-binary pair. Regenerate each retained
summary without weights or a GPU:

```sh
uv run --locked python -m llm_mojo.benchmarks.model_profile engine-replay --output studies/model_generation/engine-admission-offline.json.gz
uv run --locked python -m llm_mojo.benchmarks.model_profile engine-replay --output studies/model_generation/engine-admission-online.json.gz
uv run --locked python -m llm_mojo.benchmarks.model_profile engine-replay --output studies/model_generation/engine-admission-pressure.json.gz
uv run --locked python -m llm_mojo.benchmarks.model_profile engine-replay --output studies/model_generation/engine-admission-online-pressure.json.gz
```

The validation archive's `files` entries retain complete UTF-8 text with byte
counts and SHA256 hashes, including `intent.json`, the closeout script and its
successful command log. Restore those files to an external directory after
checking the compressed/uncompressed hashes in the validation manifest and
each entry's bytes/hash. With the bundle restored to `/private/tmp/admission-evidence`,
regenerate the card and tables independently:

```sh
uv run --locked python /private/tmp/admission-evidence/reproduction/admission_closeout.py \
  --root /private/tmp/admission-replay --retained-dir studies/model_generation \
  --intent /private/tmp/admission-evidence/intent.json \
  --live-receipt studies/model_generation/engine-admission-results.json
```

Retained replay verifies committed source blobs, archive identities, raw records,
all work/history/drain gates and exact derived metrics. It uses the original
live receipt for physical file checks, without requiring the original binary
or checkpoint to remain installed. Binaries, weights and oracle arrays are
excluded from the validation bundle.

## Metadata preparation gate

The [metadata probe receipt](engine-metadata-probe.json) records a bounded public
API test on the pinned MAX 26.5.0 / Mojo 1.0.0 and Apple M4 Pro / Metal. It
populates an idle `HostBuffer` while an earlier dependent integer kernel runs,
then enqueues the next upload and validates both steps after synchronization.
All 4,096 earlier rows re-read their original metadata after the recurrence,
all 2,048 next-step metadata values are checked, and row zero's recurrence has
an independent scalar check. Five balanced mapped/staged pairs and their
warmups, variability, source/binary hashes and exact commands are retained.

Mapping the live metadata buffer waited for the queued work. Preparing the
separate pinned buffer returned before that work finished and moved the wait
to final synchronization. This establishes the tested metadata preparation
gate, not overlapped LLM request stepping or an LLM latency improvement.
Creating a device event was unsupported on the pinned Metal runtime; the
receipt retains that failure separately. Full asynchronous phase 3 stepping
still requires token chaining, buffer reuse discipline and its exact token
and measured-load gates.
