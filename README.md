# llm-mojo

An understandable study of LLM inference on a MacBook Pro, implemented from
first principles in Mojo.

We are building an inference engine and explaining how it works: how tensors
are stored, which GPU threads own each computation, where data is reused, and
why an optimization changes the measured result. Code, numerical tests,
profiles, and graphs are all part of the study.

RMSNorm, affine linear projection, RoPE, and grouped-query attention have Mojo
host references, Apple GPU implementations, independent oracle tests, and
reproducible measurements. **The next milestone is composing and numerically
verifying one complete decoder block.** End-to-end model inference and
model-level performance remain future work.

## Explore the studies

Each study connects an implementation choice to measurements and explains the
limits of the result. Start with a question:

| Study | Question |
| --- | --- |
| [RMSNorm](studies/rms_norm/README.md) | How should threads cooperate to reduce a row? |
| [Linear decode](studies/linear_decode/README.md) | What do packing QKV and reusing inputs across outputs buy? |
| [Linear prefill](studies/linear_prefill/README.md) | How does processing more token rows change useful tiling? |
| [RoPE](studies/rope/README.md) | What does rotating dimension pairs cost? |
| [GQA decode](studies/gqa_decode/README.md) | How do fusion, sequence parallelism, and shared KV heads interact? |

![GQA decode latency and paired comparisons on Apple M4 Pro](studies/gqa_decode/latency.png)

GQA decode illustrates how different work partitions behave across context
lengths. Comparisons use this repository's materialized baseline; hot and
ring24 measurements have different synchronization boundaries. The
[GQA study](studies/gqa_decode/README.md) explains the mappings, calibration,
and profile evidence. See the [study index](studies/README.md) for retained
measurements and the command to regenerate every graph.

## Target and reference platform

The first end-to-end target is
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
in BF16, with batch-one greedy decoding, up to 4,096 live session tokens,
full and incremental prefill, and a persistent KV cache. Model-level work
starts with numerical parity against the reference implementation. The
[model contract](docs/model.md) specifies the pinned artifact, architecture,
memory accounting, and complete V0 behavior.

Apple GPU development and performance evidence are currently anchored to a
MacBook Pro with an Apple M4 Pro, a 14-core CPU, a 20-core GPU, and 24 GB of
unified memory, using Metal. This is the reference platform; minimum system
requirements and total model runtime memory have not been established. See
[development guidance](docs/development.md) for the required environment and
device verification procedure.

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
toolchain policy. See the [measurement tools](src/llm_mojo/benchmarks/README.md)
for building benchmarks, collecting profiles, and regenerating reports.

## Layout

```text
src/llm_mojo/             Inference operations and development commands
src/llm_mojo/benchmarks/   Measurement, profiling, and report generation
tests/                   Correctness tests and independent oracle generators
docs/                    Project direction and development guidance
studies/                 Topic explanations, compact measurements, and graphs
```
