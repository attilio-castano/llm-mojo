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

Mojo tests use `std.testing.TestSuite` and run as Mojo programs:

```bash
uv run mojo run -I src tests/test_import.mojo
MODULAR_DEBUG=device-sync-mode \
  uv run mojo run -I src -I tests tests/test_rms_norm.mojo
MODULAR_DEBUG=device-sync-mode \
  uv run mojo run -I src -I tests tests/test_linear.mojo
MODULAR_DEBUG=device-sync-mode \
  uv run mojo run -I src -I tests tests/test_rope.mojo
```

The import smoke test proves that the package resolves through the configured
source path. The RMSNorm test compares both a small diagnostic tensor and the
model's 896-element hidden width through both the host reference path and the
Apple GPU path against a committed Transformers oracle fixture. The GPU tests
require an Apple accelerator and reject a non-Metal device context. Regenerate
the fixture, including its provenance manifest, with:

```bash
uv run --script tests/fixtures/rms_norm/generate.py
```

The affine linear projection test compares a one-token diagnostic case and a
short multi-row case against a committed `torch.nn.Linear` oracle. It also
compares the Apple GPU path with the host reference at the model's query and
key/value projection shapes. The rowwise public kernel, the direct `8x16`
ownership control, and the shared `8x16x32` learning candidate have separate
coverage. Ragged `M=9, K=33, N=17` exercises all three tiled edge policies.
Correctness does not imply that either experimental prefill entrypoint is
faster or suitable for public dispatch.
Regenerate its fixture and provenance manifest with:

```bash
uv run --script tests/fixtures/linear/generate.py
```

The RoPE test covers tiny, query-decode, and incremental-key cases against the
pinned Transformers operation. Regenerate its fixture and provenance manifest
with:

```bash
uv run --script tests/fixtures/rope/generate.py
```

The generator's inline environment pins its oracle dependencies independently
of the Mojo inference environment. Do not change a fixture tolerance after
observing the Mojo result; update the declared numerical contract first and
explain why it changed.

Run the synchronized RMSNorm microbenchmark and its curated environment record
with:

```bash
uv run --locked python benchmarks/run_rms_norm.py
```

The ordinary command measures the current public SIMD-group implementation.
Use the paired mode documented in `benchmarks/README.md` only when an experiment
needs the retained shared-tree baseline.

Run the decode-only `M=1` projection workload matrix with:

```bash
uv run --locked python benchmarks/run_linear.py
```

It measures Q, KV, hot QKV, and a 24-layer rotating-weight QKV proxy through
explicit completion boundaries. The paired mode retains the experimental
two-output input-reuse candidate for reproduction; the public projection path
remains the one-output-per-SIMD-group baseline under the EXP-0005 decision.

Characterize that same public projection path across prefill row counts with:

```bash
uv run --locked python benchmarks/run_linear_prefill.py
```

The prefill instrument sweeps `M=1..256` at the model's `K=896` width for KV,
query, packed-QKV, and rotating 24-layer packed-QKV workloads. It establishes a
rowwise baseline only; it does not imply a tiled implementation or speedup.
Its `--direct-comparison` mode pairs that baseline with the experimental
`8x16` one-thread-per-output control. The control has no shared operand staging
and is never selected by the public enqueue path. Its `--tiled-comparison`
mode instead holds that ownership constant and compares the direct control
with the shared `BM=8, BN=16, BK=32` candidate in four ABBA blocks. The latter
isolates the incremental effect of explicit operand staging and barriers; it
also makes no dispatch decision by itself.

Screen the shared candidate's K-tile sensitivity with:

```bash
uv run --locked python benchmarks/run_linear_prefill_bk_sweep.py
```

This holds output ownership fixed and compares `BK=16`, `32`, `64`, and `128`
only on the rotating packed-QKV workloads. The four-block recorded protocol
counterbalances BK execution position. A qualifying result from this screen
requires a separate direct-control comparison before it can support an
optimization claim.

[EXP-0009](../experiments/EXP-0009-linear-prefill-bk-sweep/report.md) selected
BK16 for that follow-up after a repeatable 21.86%–25.58% improvement over BK32.
It did not compare BK16 with the direct control or alter the public projection
path.

The ownership-matched follow-up uses:

```bash
uv run --locked python benchmarks/run_linear_prefill_bk16_direct.py
```

It compares direct full-K streaming with BK16 shared staging on only the
rotating packed-QKV matrix. [EXP-0010](../experiments/EXP-0010-linear-prefill-bk16-direct/report.md)
found BK16 24.54%–33.88% slower across the tested M range and slower in all
four blocks at every M. That rejects shared staging and ends BK tuning only for
this scalar `8x16` one-output-per-thread mapping. It does not reject tiling with
multi-output, SIMD-group, or hardware-matrix arithmetic ownership. The public
projection path remains unchanged.

[EXP-0011](../experiments/EXP-0011-linear-prefill-register-2x2/report.md)
tested the next manual ownership change before introducing an Apple matrix
primitive. The control gives each of 128 threads one output in the `8x16` tile;
the candidate gives each lane of one 32-lane SIMD group a `2x2` microtile and
four FP32 accumulators. Both stream the full K dimension directly and use no
threadgroup operand storage or barriers. The candidate materially regressed
`M=1`, was inconclusive at `M=4` and `M=8`, and materially improved every tested
row count from `M=16` through `M=256` by 42.69%–62.19%. It advances to a paired
public-rowwise comparison and does not yet select a public path.

This is operation-level evidence only; it is not an end-to-end inference
benchmark. Use the [experimental method](experiments.md) when retaining a run or
using benchmark and profile evidence to support an optimization.

## Dependency updates

Upgrade stable packages intentionally:

```bash
uv lock --upgrade-package mojo --upgrade-package max
uv sync --locked
```

After an upgrade, verify the resolved versions and rerun all correctness and
benchmark checks before accepting the lockfile change.
