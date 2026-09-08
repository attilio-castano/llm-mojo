# Development

## Toolchain

The repository is a Python 3.12 `uv` library project. Python provides the
development envelope for reference oracles and tooling; the inference engine is
implemented in Mojo.

Project dependencies track stable Mojo and MAX releases in `pyproject.toml`.
`uv.lock` records the exact reproducible environment. Do not add nightly package
indexes or prerelease flags to the default environment.

## Prerequisites

- `uv`
- a C linker
- Python 3.12, managed automatically by `uv` when necessary

Apple GPU development additionally requires a supported Apple Silicon Mac,
current macOS, Xcode 16 or later, and the Metal toolchain. Verify that full
Xcode is selected and that both public Metal compiler tools resolve before
beginning GPU work:

```bash
xcode-select -p
xcrun -f metal
xcrun -f metallib
```

The selected developer directory should point inside `Xcode.app`, not the
standalone Command Line Tools directory. On Xcode 26 or later, also verify that
the separately managed component reports `"status" : "installed"`:

```bash
xcodebuild -showComponent MetalToolchain -json
```

See the official [Mojo installation guide](https://mojolang.org/install/) and
[system requirements](https://mojolang.org/docs/requirements/).

## Reference machine

The initial Apple Silicon reference machine is a MacBook Pro with the following
stable hardware configuration:

- model identifier: `Mac16,7`;
- SoC: Apple M4 Pro;
- CPU: 14 cores (10 performance and 4 efficiency);
- GPU: 20 cores with Metal 4 support;
- unified memory: 24 GB;
- internal storage: 512 GB Apple SSD;
- published memory bandwidth: 273 GB/s.

The bandwidth figure is an Apple specification, not a measured project result.
See Apple's [MacBook Pro technical specifications](https://support.apple.com/121554).

The software environment observed on 2026-08-20 was:

- macOS 26.5.2 (`25F84`) with Darwin 25.5.0;
- Xcode 26.6 (`17F113`);
- Metal toolchain installed, build `17F109`;
- `uv` 0.12.5;
- Mojo 1.0.0 and MAX 26.5.0 as resolved by `uv.lock`.

This snapshot establishes the development environment; it does not prove that
a Mojo workload executed on the GPU. Every GPU result must report the runtime's
device and backend identity and must satisfy the project's
[evidence requirements](project.md#evidence).

Mutable conditions belong in each benchmark record rather than in this machine
profile. At minimum, record the current software versions, power source and
power mode, thermal state, attached displays, available memory, and repository
commit and dirty state. Do not record serial numbers, hardware UUIDs,
provisioning identifiers, usernames, or volume UUIDs.

## Setup

Create the locked environment and confirm the compiler version:

```bash
uv sync --locked
uv run mojo --version
```

There is no need to activate `.venv`; `uv run` executes commands in the managed
environment. Each worktree has its own `.venv`; uv reuses its package cache when
setting up another checkout.

For everyday commands, use `uv run mojo ...` or `uv run python ...`. Plain
`uv run` uses the existing lockfile and does not upgrade packages just because
new releases exist. Validation and recorded measurements use `--locked` so an
outdated lockfile fails explicitly instead of changing during a run. Their
Python and Mojo subprocesses reuse the selected environment directly.

`--offline` and `--no-sync` are not part of the normal workflow. Offline mode
requires cached dependencies; no-sync assumes the environment is already ready.
If an agent sandbox blocks uv's cache or downloads, grant the required access
for setup rather than adding these flags to project commands.

## Tests

Run the complete validation workflow from a clean checkout:

```bash
uv run --locked llm-mojo-validate
```

This regenerates every independent oracle into ignored `build/oracle_data/`,
checks its SHA-256 against the frozen anchors (the original fixtures at
`a86f4db` and the subsequently added prefill oracle), runs
Python tooling tests, and runs every Mojo correctness suite on Metal with
`MODULAR_DEBUG=device-sync-mode`. The frozen tolerances, diagnostic tensors,
ragged tiles, full and incremental prefill, and all 24 decode cases remain.
Prefill adds 29 Qwen-shape cases and full-versus-suffix, causal-independence and
extreme-score regression checks across sixteen routes, including the five resource ablations. Its generated NumPy
arrays are loaded only by tests; inference and timed paths remain Mojo.
Normal-mode stress for the new schedules uses `-D PREFILL_REPEAT=12` on
`tests/test_attention_prefill.mojo`, with `MODULAR_DEBUG` unset.
The Torch/Transformers oracles share the isolated script environment in
`tests/fixtures/generate.py`, locked by its adjacent `.lock` file. The NumPy
oracles use the project environment. The first run may download dependencies;
no model weights are required. To regenerate one Torch oracle, for example:

```bash
uv run --locked --script tests/fixtures/generate.py rms_norm
```

After deliberately editing dependency declarations, update the corresponding
lock with `uv lock` or `uv lock --script tests/fixtures/generate.py`, then rerun
validation. Add `--upgrade-package NAME` only when intentionally upgrading.

The [attention-sublayer study](attention-sublayer.md) adds 17 synthetic cases
with frozen arrays and a strict FP32 attention accuracy gate. BF16 eager
comparisons report their numerical differences while keeping finite-output
and exact cache checks mandatory. The documented explicit compatibility
command reproduces the retained seed-887 failures. Checkpoint-derived
first-layer checks are an explicit separate workflow and do not make ordinary
validation download a model.

The [MLP reference contract](mlp-sublayer.md) adds upstream development captures,
independent FP64 diagnostics, and a finite BF16 SiLU sweep. Validation runs its
fixture-tooling tests and verifies synthetic frozen evidence. Checkpoint
reproduction uses an explicit local-asset argument; the ordinary workflow does
not capture or evaluate the separate holdouts. The Mojo MLP adds
operation/composition and BF16 boundary tests for all nineteen projection mappings.
Mappings 0 through 7 cover full and chunked rows; decode-only mappings 8 through
18 use each fixture's first row and reject multi-row calls.
The explicit `tests/fixtures/mlp_acceptance.py` entrypoint uses the same pinned
script lock through a symlink and opens holdouts only against a clean candidate.
Normal-mode reuse can be checked with `MODULAR_DEBUG` unset and
`MLP_CASE=h896_i4864_r17_s1601` when running `tests/test_mlp.mojo`.
Set `MLP_SPLIT=checkpoint` to run its three existing checkpoint cases. Holdouts
are explicit with `MLP_SPLIT=holdout` after their initial capture; subsequent
evaluations of observed holdouts are regression checks, not fresh holdouts.
The completed optimization campaign also captured its separately declared
holdouts. Evaluate those existing fixtures with
`MLP_SPLIT=optimization_holdout MLP_VARIANTS=0,7`; `MLP_VARIANTS` can restrict
any regression run to an explicit subset of mappings 0 through 18. The completed
decode campaign's four observed holdouts can be checked with
`MLP_SPLIT=decode_holdout MLP_VARIANTS=0,12`; variant 12 is a diagnostic candidate,
not a promoted route.

During the initial frozen-candidate acceptance, `MLP_CANDIDATE_BINARY` additionally binds
the binary hash and clean commit to the capture manifest. Leave it unset for
later regression runs after source changes. The `--optimization` acceptance
generator refuses to overwrite its existing output directory; replaying the
same declared inputs does not make them independent holdouts again.

Use `--prepare-only` to generate fixtures without running tests. For an individual
Mojo suite, include `-I src -I build -I tests`. Generators and the
small checksum record are versioned; large generated arrays and manifests are
build outputs. A changed checksum requires reviewing the oracle and numerical
contract, never relaxing tolerances to fit a kernel.

## Measurements and studies

See [src/llm_mojo/benchmarks/README.md](../src/llm_mojo/benchmarks/README.md) for the shared build/run
workflow and [studies/README.md](../studies/README.md) for the maintained
explanations and figures. Measurement requires AC power, Low Power Mode off,
and no reported thermal warning. Correctness testing uses device-sync-mode;
timing deliberately removes it and uses the instrument's explicit boundaries.

Before committing, run the full test workflow for numerical/fixture changes.
For tooling-only changes, run the Python tests and relevant route smoke;
repeat numerical suites if an engine, numerical contract, or oracle changes.
Always run `git diff --check`. Stable dependencies remain pinned by `uv.lock`.
