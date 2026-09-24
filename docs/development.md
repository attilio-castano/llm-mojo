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
Minimum hardware requirements have not been established.
See Apple's [MacBook Pro technical specifications](https://support.apple.com/121554).

The software environment observed on 2026-08-20 was:

- macOS 26.5.2 (`25F84`) with Darwin 25.5.0;
- Xcode 26.6 (`17F113`);
- Metal toolchain installed, build `17F109`;
- `uv` 0.12.5;
- Mojo 1.0.0 and MAX 26.5.0 as resolved by `uv.lock`.

Later recorded runs used macOS 26.6.2; each benchmark record carries its own
software versions. This snapshot establishes the development environment; it
does not prove that a Mojo workload executed on the GPU. Every GPU result must
report the runtime's device and backend identity and must satisfy the project's
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

Then run `uv run llm-mojo setup`. It performs the toolchain checks above with
printed remedies, provisions the shared model store and links this checkout to
it, prepares the tokenizer tables and builds chat and generate; see the
[CLI guide](cli.md#prepare-and-run). A new worktree runs the same command and
reuses the store without downloading.

There is no need to activate `.venv`; `uv run` executes commands in the managed
environment. Each worktree has its own `.venv`; uv reuses its package cache when
setting up another checkout.

For everyday commands, use `uv run llm-mojo ...`, `uv run mojo ...` or `uv run python ...`. Plain
`uv run` uses the existing lockfile and does not upgrade packages just because
new releases exist. Validation and recorded measurements use `--locked` so an
outdated lockfile fails explicitly instead of changing during a run. Their
Python and Mojo subprocesses reuse the selected environment directly.

`--offline` and `--no-sync` are not part of the normal workflow. Offline mode
requires cached dependencies; no-sync assumes the environment is already ready.
If an agent sandbox blocks uv's cache or downloads, grant the required access
for setup rather than adding these flags to project commands.

## Tests

Validation needs the pinned tokenizer tables, which `uv run llm-mojo setup`
prepares. For the tokenizer alone, `uv run --locked llm-mojo tokenizer setup`
downloads only the pinned tokenizer artifact when missing, verifies it, and
prepares tables and the native executable. Subsequent tokenizer calls reuse
local artifacts. See [the tokenizer contract](tokenizer.md).

Run the complete validation workflow from a clean checkout:

```bash
uv run --locked llm-mojo validate
```

The runner lives in `src/llm_mojo/validation/suite.py`. The `mlp`, `decoder`,
and `model` modules in that package own numerical qualification, and
`evidence.py` owns their shared source identity and receipt helpers. Tests and
independent oracle generators remain under `tests/`. See the
[CLI compatibility table](cli.md#validation-and-compatibility) for retained
study commands and their canonical replacements.

Validation runs in four stages, stopping at the first failure:

1. **Oracles.** It regenerates every independent oracle into ignored
   `build/oracle_data/` and checks each against its frozen anchor: the original
   fixtures at `a86f4db`, the prefill and attention-sublayer manifests, the MLP,
   decoder and model-calibration self-tests, the tokenizer references (including
   a Unicode run) and the packed chat fixtures. Tokenizer tables must already be
   prepared; validation never downloads.
2. **Python tests.** Setup and the shared store, the CLI and configuration,
   launch boundaries, evidence replays, numerical tooling and documentation links.
3. **Mojo suites.** Every `tests/test_*.mojo` runs on Metal with
   `MODULAR_DEBUG=device-sync-mode`.
4. **Route smoke.** Every maintained benchmark route runs once, and invalid
   selectors must be rejected.

| Level | Mojo suites |
| --- | --- |
| Kernels | `test_rms_norm`, `test_linear`, `test_rope`, `test_swiglu`, `test_residual_norm`, `test_token_selection`, `test_attention_primitives` |
| Attention | `test_attention`, `test_attention_decode`, `test_attention_decode_benchmark`, `test_attention_prefill`, `test_attention_precision`, `test_attention_sublayer`, `test_attention_sublayer_operations`, `test_qkv_fusion` |
| MLP and decoder layer | `test_mlp`, `test_decoder_layer`, `test_decoder_selection`, `test_consistency` |
| Model and runtime | `test_model`, `test_execution_plan`, `test_decode_route`, `test_chat`, `test_tokenizer`, `test_import` |

The attention suites keep their frozen tolerances, diagnostic tensors, ragged
tiles, full and incremental prefill, and all 24 decode cases. Prefill adds 29
Qwen-shape cases and full-versus-suffix, causal-independence and extreme-score
regression checks across sixteen routes, including the five resource ablations.
Its generated NumPy arrays are loaded only by tests; inference and timed paths
remain Mojo.
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

Each numerical contract documents its own fixtures, splits and recorded
evaluation commands: the [attention sublayer](attention-sublayer.md#implementation-and-reproduction),
the [MLP](mlp-sublayer.md#reference-results-and-reproduction) and the
[decoder layer](decoder-layer.md#qualified-reference-package).

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
