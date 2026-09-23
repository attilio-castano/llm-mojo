# CLI and configuration cleanup

> **Historical record**, kept as written. The [history index](README.md) says what it
> led to and where current guidance lives.

Baseline: `c5f586c` (Fast Qwen decode, configuration 26). This local milestone
unifies access to the existing engine; it does not qualify another model or
change numerical policies. Baseline Python validation: 181 tests passed.

## Ownership

- `cli/` owns Typer commands and native executable entry points.
- `configuration.py` owns typed Hydra composition, precedence and validation.
- `models/qwen2/` owns pinned artifacts, tokenizer, model composition and
  Qwen chat framing. A selectable name is not proof of native support.
- `runtime/` owns launch/build preparation and OS terminal/clock services.
- `layers/` owns decoder, attention and MLP composition.
- `kernels/` owns reusable numerical operations.
- `benchmarks/` retains measurement protocols, qualification and evidence.
- `validation/` owns the repository suite, numerical acceptance and shared
  source/receipt helpers; tests and oracle generators stay under `tests/`.

Dependencies flow from commands through preparation/model composition to
layers and kernels. Native reusable code must not import executable entry points.
Python resolves configuration before native execution; the chat process remains
Mojo-owned, including interruption handling.

## Gates

1. Record baseline tests and retain baseline native executables outside Git.
2. Add Typer and Hydra without upgrading Mojo/MAX; implement commands and typed
   composition, replacing duplicated public defaults.
3. Move modules and update active consumers; verify build dependency invalidation.
4. Run Python and full native validation, local asset preparation, fixed-input
   generation comparison, terminal lifecycle checks and benchmark smoke.
5. Update current usage documentation and review the complete diff.

Defaults resolve before named workload presets, then explicit command options.
Inspection does not compile, prepare assets or create reports. Only Fast is
advertised as an application mode; research policies retain their explicit study
interfaces. Benchmark presets select existing studies, whose workload grids,
pairing, qualification and evidence remain owned by the study registry.

Local edits, dependency setup, validation and incremental commits are approved.
Publication, new checkpoint downloads, toolchain upgrades and numerical-contract
changes require a separate decision.

## Completed validation — 2026-09-13

Implementation commit: `c434b72`. Baseline: `c5f586c`. The installed environment
uses Typer 0.27.2 and Hydra 1.3.6; every previously locked dependency version,
including Mojo 1.0.0 and MAX 26.5.0, was preserved.

- Frozen oracles passed via `uv run --locked llm-mojo validate --prepare-only`.
- The complete Python discovery command passed 194 tests.
- All 24 native test files passed on Apple M4 Pro / Metal, using the documented
  `-I src -I build -I tests` invocation and `MODULAR_DEBUG=device-sync-mode`.
  Execution was split into two batches covering the complete suite; no inherited
  MLP, decoder or prefill filters restricted the tests. The extra Unicode
  tokenizer invocation passed as well.
- `llm_mojo.benchmarks.smoke` passed every maintained route, including the
  complete decoder workload grid and adversarial rings.
- Local model preparation produced and verified 196 BF16 tensors. Repeated
  preparation reused the verified model. No checkpoint download was needed.
- For the fixed raw prompt `The capital of France is`, 16-token generation
  through the new command matched the baseline text and every non-timing event
  exactly with identical chunk size and Fast selection.
- The existing terminal harness exercised the public CLI through an adapter:
  four piped turns, three PTY turns, Ctrl-C, EOF, reset, Unicode, context rejection
  and report-free output parity passed. Baseline and current piped conversation
  output and generated token IDs matched exactly. Interrupted replies were
  checked for lifecycle/accounting correctness, not identical interruption timing.
- Invocation from another working directory resolved explicit relative prompt
  and report paths correctly. Legacy command help and tokenizer encode worked.
- 561 local Markdown file targets resolved; `git diff --check` passed.

The 18 relocated numerical/model/terminal modules retain identical bodies after
normalizing import paths. Shared stop-token and clock helpers were extracted
without changing their implementations. Retained study evidence and frozen
oracle identities were not rewritten. These are correctness and usability
checks; they establish no new speed claim. Local validation outputs and baseline
executables remain outside Git under `build/cleanup/`.

## Validation ownership follow-up

The follow-up moves the repository runner and MLP, decoder and model acceptance
tools into `validation/`. Their shared source hashing and JSON receipt helpers
now live in `validation/evidence.py`, so model, decoder, profiling and chat-study
tooling no longer import those helpers from MLP validation. Function bodies are
unchanged after normalizing the relative import depth.

Maintained Python consumers and current reproduction instructions use canonical
paths. Installed tokenizer and validation aliases point directly at their
implementations. The unused chat and tokenizer module wrappers are removed;
four old validation module commands and the model research-policy command remain
as thin entry points for existing reproduction workflows. The exact retained
surface and replacements are documented in [the CLI guide](../cli.md#validation-and-compatibility).
Historical evidence keeps its original paths, hashes and source commits.

Follow-up validation passed with `uv run --locked llm-mojo validate`: all frozen
oracle anchors, 194 Python tests, all 24 native suites on Apple M4 Pro / Metal,
the additional Unicode tokenizer invocation, and every benchmark smoke route.
The native suite census was checked against `tests/test_*.mojo`. Separate command
checks verified matching legacy/canonical help, direct installed alias targets,
identical tokenizer encode output, and current-source receipt generation and
JSON round-trip. Local Markdown file targets and `git diff --check` also passed.
No native code, lockfiles, frozen arrays or retained raw measurements changed.
The full run log is outside Git at
`/private/tmp/llm-mojo-validation-package-full.log`.
