# llm-mojo

An understandable LLM inference engine built from first principles in Mojo,
initially targeting Apple Silicon.

The repository is both an inference-engine project and an executable study of
how decoder-only transformer inference maps onto runtime machinery, memory,
GPU kernels, and hardware.

Status: foundation and reference contracts only. No model inference or
performance result exists yet.

## Principles

- Correctness before optimization.
- Explicit runtime and memory behavior over hidden framework machinery.
- Measurement before performance claims.
- Every optimization needs a correctness test and a benchmark.
- Readability is a systems requirement.

See [docs/project.md](docs/project.md) for the technical direction and
evidence-gated roadmap, and [docs/model.md](docs/model.md) for the initial model
contract.

## Development

This project uses Python 3.12 and `uv` to manage the Mojo and MAX toolchain.

```bash
uv sync --locked
uv run mojo --version
uv run mojo run -I src tests/test_import.mojo
```

See [docs/development.md](docs/development.md) for prerequisites and the
toolchain policy.

## Layout

```text
docs/   Project direction and development guidance
src/    Mojo engine code and narrowly scoped Python support code
tests/  Correctness and integration tests
```
