# Decoder Layer Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to execute this
> plan inline, task by task. Checkboxes track completed evidence gates. Do not
> create separate tasks or delegate unless the user requests it.

**Goal:** Implement and independently validate one Qwen decoder layer, then
explain its costs with a bounded baseline measurement on Apple Silicon/Metal.

**Architecture:** Compose the existing attention and MLP enqueue paths, retaining
their workspaces and numerical contracts. Execute the actual pinned upstream
decoder as the oracle. Extend the existing fixture, validation, and measurement
tools rather than creating a second framework.

**Tech stack:** Locked Mojo/MAX, Apple Metal, BF16 storage with documented FP32
arithmetic, and the pinned Torch 2.4.0 / Transformers 4.43.1 CPU reference.

Specification: [decoder-layer.md](decoder-layer.md). Starting point:
`codex/decoder-layer-baseline`, based on `7f16d6f`, plus the documentation drafts
already in this worktree. The user approved execution on 2026-09-08.

## Execution record

- Task 1 complete: pinned source inspected; local prefix/tensors verified;
  executable contract and ten reference tests pass. Metal probe identifies
  Apple M4 Pro and backend `metal`.
- Task 2 complete: 43 synthetic and three checkpoint references qualified; a fresh
  reproduction matches all case records and array hashes. Reserved outputs
  remain unopened. Existing validation passed, including 79 Python tests, all
  existing Mojo suites, and all benchmark route smoke checks. The run resumed
  after repairing the dispatcher identity issue; already verified fixtures were
  retained and the remaining checks completed successfully.
- Invocation adjustment: the historical MLP hashes the shared dispatcher.
  Preserve that file; `decoder_reference.py` shares `generate.py.lock` by
  symlink instead. Metadata round-trip repair changed no numerical outputs.

- Task 3 numerical/behavior checks pass on Apple M4 Pro / Metal: all 43
  synthetic cases and three checkpoint cases, operation-local and composed
  boundaries, explicit mappings, exact cache and protected storage, invalid
  preflight and asynchronous separate-workspace comparisons. Eight deliberate
  negative controls are rejected. Final full repository validation passed:
  15 Mojo suites / 107 tests, all reference checks and benchmark smoke routes.
  The final checkpoint replay also passed all five decoder tests.
- Task 4 routes pass: all six workloads in hot/ring24 and four adversarial
  ring shapes. Profiles and numerical build/evaluation tools are implemented;
  reserved capture is not yet run. Ten decoder tooling tests pass.
- Preflight evidence combines exact sentinel preservation (including the first
  attention dispatch's normalization output) with inspection of the shared pure
  preflight before any enqueue. No runtime dispatch-counter API was introduced.
- Verified measurement conditions: AC power, power mode 0, no thermal or
  performance warning; 24 GiB unified memory. Instruments lists Metal System
  Trace and the existing `LLM_Mojo_Metal_Limiters` template.

- Reserved run root: `/private/tmp/llm-mojo-decoder-20260908` (confirmed absent
  before allocation). The numerical candidate, benchmark binaries, three profile
  binaries, holdout capture and retained-run originals will live here.

## Execution scope

On instruction to execute this plan, proceed through all tasks without routine
approval pauses. Phase exits below are checks the agent performs, not meetings.
The execution scope includes local code/docs changes, fixture generation,
compilation, numerical tests, sequential local GPU jobs, local Instruments
captures, and small local commits after the applicable checks pass. Clean
commits are needed for the existing build/evaluation receipts.

Use verified model assets already present on this machine. Reading and copying
or linking those immutable assets into this worktree is within scope; modifying
another worktree's files is not. Resolve existing locked dependencies as needed
without changing versions. Keep weights, arrays, binaries and full traces out
of Git. Do not push, open a PR, publish, download additional model assets,
upgrade dependencies, or start remote work under this plan.

Routine implementation errors, API adjustments, fixture plumbing defects and
test failures are work to diagnose and fix locally. Keep the oracle arithmetic,
declared input recipes, numerical gates and held-out outputs unchanged while
fixing the implementation. Show progress at each meaningful result. Do not
claim acceptance from test counts alone.

Stop dependent work and report evidence when:

- A numerical target or reference arithmetic must change. First isolate local
  arithmetic, inherited error, fixture capture, and precision-policy causes;
  present the smallest failing case and proposed contract change.
- A holdout fails. Diagnose it without revising the frozen candidate or
  consuming additional holdouts. A new candidate/declaration is a new decision.
- Required checkpoint assets cannot be located and verified locally. Complete
  independent synthetic work, but leave checkpoint acceptance incomplete.
- Metal or Instruments execution cannot be proved, the declared workload
  cannot fit available memory, or measurement conditions cannot be met.
  Complete unaffected correctness/docs work; do not substitute CPU evidence,
  reduce the declared grid silently, or change machine settings.
- Work requires one of the external/out-of-scope actions above.

No automatic optimization follow-up, full-model implementation, or expansion of
the matrix follows a successful profile. A slow or noisy baseline is a valid
study result; it is not permission for another search.

## File ownership

Paths below are relative to the repository root. Create files only in the task
that needs them. Existing attention/MLP generators and their frozen outputs
must continue to reproduce unchanged.

| Files | Responsibility |
| --- | --- |
| `docs/decoder-layer.md`, `docs/decoder-layer-plan.md` | Contract, decisions, progress and acceptance status |
| `tests/fixtures/decoder_layer/contract.py` | Recipes, stages, schedules, gates and holdout declaration |
| `tests/fixtures/decoder_layer/reference.py` | Actual upstream decoder capture and verified checkpoint input loading |
| `tests/fixtures/decoder_layer/generate.py`, `test_reference.py` | Development generation, observation tests and immutable anchors |
| `tests/fixtures/decoder_reference.py` and shared-lock symlink, `src/llm_mojo/validate.py` | Register the new reference and tests in the locked workflow |
| `src/llm_mojo/decoder_layer.mojo` | Whole-call preflight and ordered attention/MLP composition |
| `src/llm_mojo/attention_sublayer.mojo`, `src/llm_mojo/mlp.mojo` | Minimal validation extraction needed by composition; preserve arithmetic |
| `tests/decoder_layer_support.mojo`, `tests/test_decoder_layer.mojo` | Fixture loading, boundary checks, cache and asynchronous behavior |
| `src/llm_mojo/decoder_validation.py`, `tests/test_decoder_validation.py` | Candidate build/evaluation receipts and coverage verification |
| `tests/fixtures/decoder_acceptance.py` | Explicit, candidate-bound holdout capture through the existing script lock |
| `src/llm_mojo/benchmarks/decoder_layer.mojo`, `decoder_layer_contract.py` | One layer workload and its route/dispatch identity |
| Existing benchmark dispatcher, study, smoke, profile, analysis and plot modules | Integrate this operation into the shared tools |
| `tests/test_decoder_tooling.py`, `studies/decoder_layer/` | Measurement identity/coverage tests and compact final evidence |

## Task 1: Establish the executable reference contract

- [x] Record branch, source identity and existing dirty files. Preserve the
  current documentation edits; stop on unrelated edits whose ownership is
  unclear. Inventory local locked runtimes and existing checkpoint assets.
- [x] Read the actual pinned `Qwen2DecoderLayer.forward` in the resolved oracle
  environment. Confirm argument, cache, mask and hook behavior before writing
  the adapter. Read the Mojo syntax and interop skills before their respective
  implementation work; use the locked project environment throughout.
- [x] Encode exactly the stages, synthetic streams, mutations, schedules and
  gates from the specification in `contract.py`. Declare seeds 5003/5011 and
  the reserved checkpoint prompt without evaluating them.
- [x] Add reference tests that reject changed stage order, missing boundaries,
  mismatched shapes, nonfinite ordinary values, zero-sign corruption in exact
  checks, and incomplete schedules. Verify every schedule sums to T and every
  chunk starts at the previous chunk's end.
- [x] Verify seed-prefix consistency and distinct attention/MLP norm and
  gate/up streams. Keep declared row counts out of the RNG seed.

**Exit:** executable specification matches the draft, holdout execution remains
unavailable through the default development command, and contract tests pass.

## Task 2: Qualify and freeze upstream fixtures

- [x] Implement an adapter that instantiates the actual decoder with
  `Qwen2SdpaAttention`, loads the declared weights, executes the existing FP32
  SDPA boundary wrapper, and returns every named observation. Assert the
  selected class and a positive wrapper call count.
- [x] Test observed versus unobserved runs using separate equivalent caches;
  require identical Y and cache bits. Check hook cleanup after success and
  intentional capture failure. Corrupt a captured residual/gating operand and
  show that reconstruction tests fail.
- [x] Execute tiny development cases first, then Qwen development and structured
  cases, one at a time. Compare upstream full/chunk executions under the
  declared gates. Preserve every intermediate diagnostic and failed attempt.
- [x] Load verified layer-0 checkpoint tensors and existing development token
  IDs, with exact identity checks. Tokenize the reserved new prompt, record its
  IDs and little-endian int64 hash, but do not execute the reserved input.
- [x] Implement explicit initial-freeze and normal verification modes. Initial
  freeze must refuse existing anchors; normal generation must reject any source,
  contract or output hash drift. Store logical BF16 arrays losslessly as FP32,
  matching the current tooling convention, and record both logical/storage dtype.
- [x] Add the decoder entrypoint sharing `generate.py.lock` and run the commands below. These are new CLI
  contracts to implement in this task, not commands available at the plan's
  starting revision. `--checkpoint-assets` verifies a local directory and has
  no download behavior. Default regeneration verifies only synthetic anchors;
  checkpoint verification is explicit with the same local-asset argument.
  Set `DECODER_ASSETS` to the verified directory found in
  Task 1; it is not an inferred or invented path.

```sh
uv run --locked --script tests/fixtures/decoder_reference.py --self-test
uv run --locked --script tests/fixtures/decoder_reference.py --freeze-development --checkpoint-assets "$DECODER_ASSETS"
uv run --locked --script tests/fixtures/decoder_reference.py
uv run --locked llm-mojo-validate
git diff --check
```

**Exit:** existing anchors reproduce unchanged; new development and checkpoint
reference gates pass; token IDs/recipes/budgets/source identities are frozen;
no Mojo decoder or holdout output has influenced the freeze. Record the actual
results in the specification and commit the qualified reference package locally.
This is the first numerical-policy stop point if the targets fail upstream.

## Task 3: Compose the layer and verify development behavior

- [x] Add the tiny fixture test first and demonstrate that the decoder entrypoint
  is absent. Implement `enqueue_decoder_layer` with caller-owned attention
  weights/cache/workspace, MLP weights/workspace, X, and explicit mapping inputs.
  The entrypoint returns the actual attention route; Y remains in MLP output.
- [x] Extract side-effect-free attention/MLP preflight as necessary, call both
  before submitting any work, and keep standalone entrypoints using the same
  checks. Preserve the existing enqueue arithmetic and dispatch sequence.
- [x] For tiny fixtures, use materialized FP32 attention route 3 and MLP 0.
  For Qwen, use integrated attention mappings `(0,0)`. Validate MLP 0 throughout
  and MLP 7 for every multi-row call. One-row calls always use MLP 0 in the
  measured configuration; this plan adds no global automatic mapping selector.
- [x] Enqueue attention, form a contiguous read-only view of its post-residual
  output Z, then enqueue MLP. Keep the two workspaces separate and allocate
  nothing inside the new entrypoint.
- [x] Test operation-local, sublayer-isolated and whole-layer comparisons
  separately. Gate both branches and both residual outputs. Compare each Mojo
  schedule with its matching upstream schedule and with Mojo full execution.
- [x] Add the specification's invalid-call, overlap, poisoned-buffer, exact cache
  append/prefix, capacity-edge and reset cases. A valid attention/invalid MLP
  call must leave cache length, bytes and dispatch count unchanged.
- [x] Exercise the twelve-decode asynchronous schedule, retaining each output
  before overwrite. Verify source X slices, absolute positions and appended
  row counts so prefix recomputation cannot masquerade as cached execution.
- [x] Implement discriminating negative controls for wrong second residual,
  wrong norm/input, missing residual, wrong position/mask and cache corruption.
  Retain the existing BF16 rounding regressions; show each new negative control
  fails the relevant exact or numerical check.
- [x] Register the suite with the existing workflow. Run the focused commands
  after implementation and the complete validator before the local checkpoint.

```sh
uv run --locked mojo run -I src -I build -I tests tests/test_decoder_layer.mojo
env -u MODULAR_DEBUG uv run --locked mojo run -I src -I build -I tests tests/test_decoder_layer.mojo
uv run --locked llm-mojo-validate
git diff --check
```

**Exit:** all development/checkpoint and behavior checks pass on proved Metal
execution, including normal asynchronous mode. No tolerance/fixture changes.
Record route, device and coverage; commit the implementation locally.

## Task 4: Prepare the bounded measurement tools

Implement and validate the tools here; collect retained timings and profiles
only after Task 5 acceptance passes.

- [x] Add one decoder operation to the existing benchmark dispatcher and study
  registry. Reuse environment recording, paired sampling and profile receipts.
  Add route/census tests to the shared smoke and tooling workflow.
- [x] Freeze this workload table. Each row has one explicit configuration, not
  a candidate screen. Integrated attention mappings are `(0,0)` in every row.

| Phase | R | T | MLP mapping |
| --- | ---: | ---: | ---: |
| Full prefill | 256 | 256 | 7 |
| Full prefill | 4096 | 4096 | 7 |
| Cached chunk | 16 | 256 | 7 |
| Cached chunk | 64 | 4096 | 7 |
| Decode | 1 | 256 | 0 |
| Decode | 1 | 4096 | 0 |

- [x] Use synthetic seed 4001 from the qualified fixture recipe, and create
  prefixes by executing the accepted layer before timing. Prepare each repeated
  call at P=T-R. Rewind benchmark-owned logical length only after the preceding
  sample completes; preserve the prefix and overwrite the same suffix. Document
  this as repeated fixed-workload timing, not growing-context generation.
- [x] Hot measures one enqueue through completion. Ring24 uses 24 distinct
  input/weight/cache sets and shared workspace, submits on one stream, and
  synchronizes once per sweep. Verify the full sequence numerically before
  timing. Use distinct nonuniform contents in a separate untimed adversarial
  smoke test to expose accidental buffer reuse; timed replicas all use the same
  seed-4001 fixture. Keep setup and prefix preparation outside the measured window.
- [x] Configure only control self-pairs in both modes: four paired blocks, ten
  warmups and ten samples per arm. Reverse arm/workload order as in the existing method.
  Expected retained census: `6 * 2 * 4 * 2 * 10 = 960` latency observations.
  The two arm labels execute identical configurations; no speedup is claimed.
- [x] Configure three separate diagnostic profiles: `(256,256)` for 25 measured
  iterations, `(64,4096)` for 25, and `(1,4096)` for 100, after ten warmups each.
  Validate the actual dispatch sequence and keep each capture under 5000
  measured dispatches. Join preempted intervals before assigning stage names.
- [x] Implement total latency and within-capture active-time views. Attribute norm,
  projection, attention, MLP, residual and dispatch-gap behavior only to the
  evidence actually captured. Never sum separately captured stage medians to
  construct whole-layer latency or infer achieved DRAM bandwidth from bytes/time.

**Exit:** every numerical and measurement route exists and passes correctness
smoke checks. The sample/profile grids and dispatch identities are fixed. No
retained timing or reserved output has been collected.

## Task 5: Freeze the candidate and run reserved acceptance

- [x] Implement decoder build/evaluation receipts following `mlp_validation.py`.
  Bind source/locks, binary hash, manifest and array hashes, routes, schedules,
  actual device/backend and complete expected check coverage. Reject changed
  binaries, stale manifests, duplicate/missing cases, inherited environment
  filters and truncated output. Test those failure cases before acceptance.
- [x] Keep development/checkpoint evaluation separate from reserved capture.
  Add `decoder_acceptance.py` using the existing locked script environment and
  the same verified-local-asset contract. Refuse overwrite and require a clean,
  verified candidate receipt before any held-out model execution.
- [ ] With Task 4 routes complete, run full validation and commit that source.
  Freeze numerical and measurement binaries at this source so engine changes
  are not needed after opening holdouts. Set the run directory below, then build
  the measurement binary with the shared command shown here before capture.
  Build the three profile executables through the existing profile builder at
  this same source and retain their receipts as well.
- [ ] Set `DECODER_RUN_ROOT` to a new absolute directory under `/private/tmp`,
  record it in progress notes, and refuse existing outputs. Implement and run:

```sh
uv run --locked python -m llm_mojo.decoder_validation build --binary "$DECODER_RUN_ROOT/numerical-candidate"
uv run --locked llm-mojo-bench build --build-dir "$DECODER_RUN_ROOT/bench-build"
uv run --locked --script tests/fixtures/decoder_acceptance.py --candidate-binary "$DECODER_RUN_ROOT/numerical-candidate" --checkpoint-assets "$DECODER_ASSETS" --output "$DECODER_RUN_ROOT/holdout"
uv run --locked python -m llm_mojo.decoder_validation evaluate --binary "$DECODER_RUN_ROOT/numerical-candidate" --fixtures "$DECODER_RUN_ROOT/holdout" --output "$DECODER_RUN_ROOT/acceptance"
```

**Exit:** all six reserved synthetic cases and the reserved checkpoint case
pass for the declared schedules/mappings using the exact frozen binary.
Manifest capture alone does not count. Preserve any failure and stop acceptance
work under the policy above. Do not quietly replace the candidate or holdouts.

## Task 6: Collect the accepted baseline

- [ ] Execute the registered study through the shared CLI:

```sh
uv run --locked llm-mojo-bench run --build-dir "$DECODER_RUN_ROOT/bench-build" --output "$DECODER_RUN_ROOT/timing" --studies decoder_layer
```

Use the frozen binaries from Task 5. Execute the three declared profiles through
the existing capture/analyzer workflow, recording exact successful commands in
the report. Do not rebuild silently during this task.

**Exit:** complete validated timing/profile grids and receipts, with all raw
observations retained. Run GPU jobs sequentially. A malformed capture may be
retried once after a diagnosed tooling repair, retaining the failed attempt;
a source change invalidates affected receipts. Noise alone is not a reason to
repeat a valid measurement. Hardware-condition failures pause collection.

## Task 7: Curate evidence and close the milestone

- [ ] Create `studies/decoder_layer/` when evidence exists. Retain the readable
  report, compact numerical records, run/profile identities, compressed raw
  samples, regenerated summaries and only the figures used by the report.
- [ ] Test the evidence readers against missing/duplicate observations, modified
  raw files, false route/device identity and incomplete dispatch sequences.
- [ ] Extend the common plot command, then regenerate the tables/figures from
  retained evidence without GPU execution. Confirm reproducible numerical
  summaries and complete sample counts.
- [ ] Update the contract, study index and roadmap to the actual achieved state.
  Keep reference qualification, candidate acceptance and performance evidence
  distinct. Report memory ownership/footprint and ring24's limitations.
- [ ] Run relevant tooling/evidence checks and `git diff --check`; rerun full
  numerical validation only if engine/oracle changes since the last full pass
  require it. Commit the final report locally with its measured source identity.

```sh
uv run --locked python -m unittest discover -s tests -p 'test_*.py'
uv run --locked --with matplotlib==3.10.8 python -m llm_mojo.benchmarks.plot studies/decoder_layer
git diff --check
```

**Done means:** independently accepted decoder composition, verified cache and
workspace behavior, six workload baselines in both modes, three diagnostic
profiles, reproducible compact evidence, and a clean local branch. The final
report answers where layer time goes and whether a concrete integration cost
warrants a follow-up. Otherwise recommend full-model forward parity next.
Do not continue into that next milestone automatically.
