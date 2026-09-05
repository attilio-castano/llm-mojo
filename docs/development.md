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
uv run --locked python tools/test.py
```

This regenerates every independent oracle into ignored `build/oracle_data/`,
checks its SHA-256 against the fixtures at merged revision `a86f4db`, runs
Python tooling tests, and runs every Mojo correctness suite on Metal with
`MODULAR_DEBUG=device-sync-mode`. The frozen tolerances, diagnostic tensors,
ragged tiles, full and incremental prefill, and all 24 decode cases remain.
Generation uses pinned Torch/Transformers script environments and locked NumPy;
the first run may download those dependencies. No model weights are required.

Use `--prepare-only` to generate fixtures without running tests. For an individual
Mojo suite, include `-I src -I build -I tests -I benchmarks`. Generators and the
small checksum record are versioned; large generated arrays and manifests are
build outputs. A changed checksum requires reviewing the oracle and numerical
contract, never relaxing tolerances to fit a kernel.

Recorded GQA decode comparisons require `--noise /path/to/noise/summary.json`
and its adjacent `metadata.json`. First record a materialized self-pair
(`--control 0 --candidates 0`) with the same binary, clean commit, seed,
hardware/software, timing settings and all comparison workloads. This baseline
calibration also applies when comparing two optimized candidates. The runner
rejects incompatible or incomplete calibration and records its hashes, identity
and block ratios in the comparison metadata so the noise floors can be reproduced.
This stricter requirement applies to new runs; retained EXP-0014 evidence records
its original calibration relationships in the experiment manifest.

Run the materialized grouped-query attention workload matrix with:

```bash
uv run --locked python benchmarks/run_attention.py
```

The instrument measures the complete three-dispatch Apple GPU baseline across
decode, incremental-prefill, and full-prefill shapes. It keeps allocation,
initialization, and correctness readback outside timing, proves the Metal
runtime identity, and reports synchronized milliseconds per attention call.
It is the control for later attention optimizations, not an end-to-end model
benchmark. See `benchmarks/README.md` for the exact shape sweep and retained-run
protocol.

Use `--stage-attribution` to time the exact QK, softmax, and
probability-times-V kernels independently beside the end-to-end control. The
reported stage fractions use the isolated-stage sum; they are diagnostic and
must not be presented as an exact decomposition of end-to-end latency.

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

Run that paired public-rowwise comparison with:

```bash
uv run --locked python benchmarks/run_linear_prefill_register_rowwise.py
```

[EXP-0012](../experiments/EXP-0012-linear-prefill-register-rowwise/report.md)
holds the rotating `K=896`, `N=1152`, 24-layer workload fixed. Its control is
the public mapping in which one SIMD group owns one scalar output and its lanes
partition K. Its candidate is the frozen EXP-0011 mapping in which one SIMD
group owns an `8x16` output tile and each lane walks all K for a `2x2`
microtile. The candidate materially regressed `M=1`, `4`, and `8`, then
materially improved every tested row count from `M=16` through `M=256` by
35.92%–67.98% in all four blocks. Both crossover definitions selected `M=16`
for this packed-QKV workload. It advances as the manual prefill candidate, but
`N=128` and `N=896` still require paired timing before any public dispatch
threshold is selected.

Run the phase-controlled Apple 8x8 matrix experiment with:

```bash
uv run --locked python benchmarks/run_linear_mma_phase.py
```

[EXP-0013](../experiments/EXP-0013-linear-mma-phase/report.md) compares one
`8x16` MMA tile with the strongest previously measured control for each row
regime: public rowwise at `M=1,4,8`, and register-2x2 at `M=16..256`. The MMA
mapping materially regressed `M=1` by 79.84% and `M=4` by 20.41%, then
materially improved every tested row count from `M=8` through `M=256` by
33.15%–54.77% in all four blocks. It advances as the packed-QKV prefill
candidate and is rejected for batch-1 decode. The implementation uses MAX
26.5's architecture-internal Apple 8x8 primitive, and `N=128` plus `N=896`
remain unmeasured, so no public dispatch rule changes.

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
