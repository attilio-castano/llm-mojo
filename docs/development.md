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
environment.

## Tests

Run the complete validation workflow from a clean checkout:

```bash
uv run --locked python -m llm_mojo.validate
```

This regenerates every independent oracle into ignored `build/oracle_data/`,
checks its SHA-256 against the fixtures at merged revision `a86f4db`, runs
Python tooling tests, and runs every Mojo correctness suite on Metal with
`MODULAR_DEBUG=device-sync-mode`. The frozen tolerances, diagnostic tensors,
ragged tiles, full and incremental prefill, and all 24 decode cases remain.
Generation uses pinned Torch/Transformers script environments and locked NumPy;
the first run may download those dependencies. No model weights are required.

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
