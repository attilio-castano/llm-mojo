# CLI and configuration cleanup

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
