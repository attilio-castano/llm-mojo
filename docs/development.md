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

## Setup

Create the locked environment and confirm the compiler version:

```bash
uv sync --locked
uv run mojo --version
```

There is no need to activate `.venv`; `uv run` executes commands in the managed
environment.

## Tests

Mojo tests use `std.testing.TestSuite` and run as Mojo programs:

```bash
uv run mojo run -I src tests/test_import.mojo
```

The initial smoke test proves that the package resolves through the configured
source path. Future tests should assert numerical behavior rather than merely
exercise execution.

## Dependency updates

Upgrade stable packages intentionally:

```bash
uv lock --upgrade-package mojo --upgrade-package max
uv sync --locked
```

After an upgrade, verify the resolved versions and rerun all correctness and
benchmark checks before accepting the lockfile change.
