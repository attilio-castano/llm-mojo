# llm-mojo

An understandable LLM inference engine built from first principles in Mojo,
initially targeting Apple Silicon.

The repository is both an inference-engine project and an executable study of
how decoder-only transformer inference maps onto runtime machinery, memory,
GPU kernels, and hardware.

Status: RMSNorm and RoPE have Mojo host references and Apple GPU
implementations with provenance-bearing oracle tests. RMSNorm also has a
reproducible microbenchmark. No end-to-end model inference or
model-performance result exists yet.

## Principles

- Correctness before optimization.
- Explicit runtime and memory behavior over hidden framework machinery.
- Measurement before performance claims.
- Every optimization needs a correctness test and a benchmark.
- Readability is a systems requirement.

See [docs/project.md](docs/project.md) for the technical direction and
evidence-gated roadmap, and [docs/model.md](docs/model.md) for the initial model
contract. [docs/layouts.md](docs/layouts.md) defines the concrete language used
to distinguish logical tensors, storage, work partition, and execution order.
[docs/experiments.md](docs/experiments.md) defines how performance experiments
are planned, recorded, and promoted into project decisions.

## Development

This project uses Python 3.12 and `uv` to manage the Mojo and MAX toolchain.

```bash
uv sync --locked
uv run mojo --version
uv run mojo run -I src tests/test_import.mojo
MODULAR_DEBUG=device-sync-mode \
  uv run mojo run -I src -I tests tests/test_rms_norm.mojo
MODULAR_DEBUG=device-sync-mode \
  uv run mojo run -I src -I tests tests/test_rope.mojo
```

See [docs/development.md](docs/development.md) for prerequisites and the
toolchain policy. See [benchmarks/README.md](benchmarks/README.md) for the
operation-level benchmark boundary and runner.

## Layout

```text
docs/   Project direction and development guidance
src/    Mojo engine code and narrowly scoped Python support code
tests/  Correctness and integration tests
benchmarks/  Reproducible operation-level measurements
experiments/ Frozen protocols, raw evidence, and bounded findings
```
