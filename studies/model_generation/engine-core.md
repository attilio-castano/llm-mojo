# Engine core: readiness, correctness and load declaration

Implementation authorized on 2026-10-08 from baseline
`cb2416abf3c9fbb19c460fa99709d93eeabba97e` on `codex/engine-core`.
**Status: in progress.** This is a declaration, not a passing validation receipt
or a measurement result. The [serving plan](../../docs/serving-plan.md)
defines the architecture; [experimental method](../../docs/experiments.md#serving-measurement-contract)
defines retention and paired comparisons.

## Readiness receipt

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
Their existence is not a completed load-study result.

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
it establishes no goodput or asynchronous claim. The existence of this tool is
not a measured improvement.

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
