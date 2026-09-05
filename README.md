# llm-mojo

An understandable LLM inference engine built from first principles in Mojo,
initially targeting Apple Silicon.

The repository is both an inference-engine project and an executable study of
how decoder-only transformer inference maps onto runtime machinery, memory,
GPU kernels, and hardware.

Status: RMSNorm, affine linear projection, RoPE, and grouped-query attention
have Mojo host references and Apple GPU implementations with
provenance-bearing oracle tests. The operations have reproducible microbenchmarks and
[readable studies with graphs](studies/README.md). No end-to-end model inference
or model-performance result exists yet.

## Target and reference platform

The first end-to-end target is
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
in BF16: an instruction-tuned, 0.49-billion-parameter decoder-only transformer.
V0 is batch-one inference with at most 4,096 live session tokens, deterministic
greedy decoding, full first-turn prefill, incremental later-turn prefill, and a
persistent KV cache. The first model-level milestone is numerical parity with
the reference implementation, not performance.

The pinned BF16 weights artifact is 988,097,824 bytes, about 942 MiB. The
theoretical unpadded BF16 KV payload is 48 MiB at the V0 limit. Total end-to-end
runtime memory has not yet been measured, so these figures do not establish a
minimum system-memory requirement. See [docs/model.md](docs/model.md) for the
immutable artifact, architecture, and complete runtime contract.

Apple GPU development and performance evidence are currently anchored to a
MacBook Pro with an Apple M4 Pro, a 14-core CPU, a 20-core GPU, and 24 GB of
unified memory. This is the project's reference platform, not a claimed minimum
requirement. Reproducing the Apple GPU path requires a supported Apple Silicon
Mac, current macOS, full Xcode 16 or later, and the Metal toolchain. See
[docs/development.md](docs/development.md) for the complete environment and
verification procedure.

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

General prerequisites are `uv`, a C linker, and Python 3.12; `uv` can manage
Python and resolves the locked Mojo and MAX toolchain.

```bash
uv sync --locked
uv run --locked mojo --version
uv run --locked python -m llm_mojo.validate
```

See [docs/development.md](docs/development.md) for prerequisites and the
toolchain policy. See [src/llm_mojo/benchmarks/README.md](src/llm_mojo/benchmarks/README.md) for the
operation-level benchmark boundary and runner.

## Layout

```text
src/llm_mojo/             Inference operations and development commands
src/llm_mojo/benchmarks/  Measurement, profiling, and report generation
tests/                   Correctness tests and independent oracle generators
docs/                    Project direction and development guidance
studies/                 Topic explanations, compact measurements, and graphs
```
