# Phase 3 evidence catalog

Start with the [implementation review map](../../docs/serving-plan.md#phase-3-review-map)
and the [engine study](engine-core.md) for mechanisms and results. This catalog
explains where the complete evidence lives and how to recover its original
layout. Curation changes storage, not experimental coverage or conclusions.

## Decisions and coverage

| Question | Retained coverage | Decision and detailed explanation |
| --- | --- | --- |
| Does the mixed engine preserve request ownership? | Mixed numerical checks, lifecycle/fault tests, original load traces | [Validated core](engine-core.md#readiness-receipt); production transport/recovery remain later phases |
| Does lifetime reservation help under pressure? | 48 runs, same-build token histories and full pool release | [Admission](engine-core.md#lifetime-reservation-admission-bounded-successor-study); offline pressure reserved/incremental makespan ratio 0.25615, adequate-capacity comparison inconclusive |
| What does reservation cost with larger output caps? | 144 runs across tight/loose caps, 40/128 blocks, offline and 4/8 requests/s | [Operating range](engine-core.md#admission-operating-range-retained-bounded-results); reservation removes replay but can increase unused capacity and FIFO waiting; remains optional |
| Does fitted budgeting beat fixed 256? | 116 calibration/evaluation runs with identical delivered histories | [Budgeting](engine-core.md#fixed-workspace-budget-retained-bounded-results); adaptive offline/128-block ratio 1.17390 is slower; fixed-256 remains default |
| Is Fast a better engine default? | Four qualification receipts and 12 timed runs | [Fast](engine-core.md#optional-fast-engine-retained-bounded-results); paired ratio 0.987555 is within the 5% noise floor, inconclusive; reference-27 remains default |
| Does engine chat preserve terminal lifecycle? | Independent reference parity: 123 tokens / 455 necessary rows, cancellation and fault boundaries | [Terminal chat](engine-core.md#optional-engine-terminal-chat-accepted-lifecycle); opt-in, complete-history recomputation |
| Does asynchronous stepping improve this workload? | 24 paired runs, exact 97-token histories per run, terminal/reference acceptance | [Async](#asynchronous-stepping); offline ratio 0.95629 is inconclusive at the 5% floor; online distributions are descriptive; remains opt-in |

The admission, range, budget and Fast measurements retain MAX 26.5.0 / Mojo
1.0.0 identities. Async retains MAX 26.6.0 / Mojo 1.1.0. Measurements identify
Apple M4 Pro / Metal, BF16 storage / FP32 accumulation, exact configurations,
source commits and synchronization boundaries in their original records.
No default or performance claim changes during curation.

## Original records and compact storage

The [machine-readable catalog](engine-evidence.json) binds 104 original evidence
files to their byte counts and SHA-256 hashes at `e3cd31f`. Raw measurement and
validation archives, specifications and execution receipts keep their existing
paths and bytes.

The original measured and catalog-basis commits are retained by
[PR #35](https://github.com/attilio-castano/llm-mojo/pull/35). Recover its published
history with `git fetch origin pull/35/head` after squash merge.

Eight expanded reports are reconstructed from these sources:

| Original report | Lossless retained source |
| --- | --- |
| `engine-admission-results.json` | [Admission validation](engine-admission-validation.json.gz), member `closeout/results-card.json` |
| `engine-admission-range-results.json` | [Range validation](engine-admission-range-validation.json.gz), member `closeout/results-card.json` |
| `engine-budget-results.json` | [Budget validation](engine-budget-validation.json.gz), member `results/live-card.json` |
| `engine-budget-publication.json` | [Compressed original](engine-budget-publication.json.gz) |
| `engine-fast-publication.json` | [Compressed original](engine-fast-publication.json.gz) |
| `engine-async-evidence/engine-async-results.json` | [Compressed original](engine-async-evidence/engine-async-results.json.gz) |
| `engine-async-evidence/independent-replay.json` | [Compressed original](engine-async-evidence/independent-replay.json.gz) |
| `engine-async-evidence/canonical-replay.json` | [Compressed original](engine-async-evidence/canonical-replay.json.gz) |

The three large admission/budget reports already existed byte-for-byte in their
validation bundles. The other five reports use deterministic gzip of their
original bytes. Compression preserves whitespace, every observation, historical
failure, command, source identity and distinct replay execution. Historical
seals and publication inventories still bind the original filenames and hashes;
compact summaries must never be substituted for those physical receipts.

## Verify and restore

From the repository root, verify every original file without weights or a GPU:

```sh
uv run --locked python -m llm_mojo.benchmarks.engine_evidence verify
```

Choose a fresh external directory. Restoration verifies all identities before
creating it, then writes the complete original layout, including the eight
expanded reports. It refuses to replace an existing destination.

```sh
uv run --locked python -m llm_mojo.benchmarks.engine_evidence restore \
  --directory /private/tmp/engine-evidence-original
```

Run a measurement replay against that copy; derived output stays outside Git:

```sh
uv run --locked python -m llm_mojo.benchmarks.model_profile engine-replay \
  --output /private/tmp/engine-evidence-original/engine-budget-loose-offline-blocks-128.json.gz
```

For complete acceptance replay, extract the hash-checked helpers from each
validation bundle and follow the corresponding reproduction section in the
engine study. Admission, range and budget also require their measured Git
objects. Budget checks imported source bytes: restore source from `01d8be4`
and use its `src/` as `PYTHONPATH` in a separate Python process. Admission and
range similarly retain `b81ea6c` and `2af0933`. Their original core/admission
baseline archives remain at the canonical repository paths. Fast and async
restore archive-contained source; chat restores its pure Python checkers.
Keep historical helpers unchanged and preserve their native-execution guards.
The [commit retention contract](../../docs/experiments.md#evidence-commits)
explains recovery through the eventual PR.

## Curation verification

The [verification receipt](engine-evidence-verification.json) and
[lossless verification archive](engine-evidence-verification.json.gz) retain
the fresh commands, actual exits, logs, source hashes and extraction checks.
All 104 original files match `e3cd31f` byte-for-byte. Six campaign acceptance
replays passed, covering 344 measured runs plus numerical and terminal gates.
The current parsers also reproduced all 28 measurement archives / 416 runs and
the original fitted policy. The final Python suite passed 400 tests.

Expanded copies were removed only after the acceptance replays passed; a second
fresh restoration then reproduced all original bytes from the curated files.
The initial chat verification invocation supplied the wrong manifest SHA and
was rejected before restoration. Its actual exit 1 is retained alongside the
corrected passing invocation. Historical failure records remain unchanged.
This receipt qualifies storage migration and tooling, not GPU performance.

## Asynchronous stepping

The [acceptance card](engine-async-evidence/engine-async-acceptance.json) retains
full-validation source equivalence, clean-build checkpoint qualification,
terminal lifecycle and independent natural-token reference parity. The complete
[results](engine-async-evidence/engine-async-results.json.gz) retain both
twelve-run grids and request latency distributions. All 24 runs preserve the
frozen 97-token histories and finish reasons. Each async run charges eight
extra submitted rows and discarded selections; all KV blocks return.

Offline median paired async/sync makespan is 0.95629 with a 5% noise floor:
**inconclusive**. The eight-request/s online trace is descriptive. Read the
[mechanism and limits](engine-core.md#asynchronous-stepping-retained-bounded-results)
and [independent CPU replay instructions](engine-async-evidence/README.md#independent-cpu-replay).
Both historical replay reports remain intact in compressed form, with their
separate original execution receipts and timestamps. Local integrity and CPU
restoration establish no fresh GPU execution or remote backup.
