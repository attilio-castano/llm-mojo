# llm-mojo

An understandable study of LLM inference on a MacBook Pro, implemented from
first principles in Mojo.

We are building an inference engine and explaining how it works: how tensors
are stored, which GPU threads own each computation, where data is reused, and
why an optimization changes the measured result. Code, numerical tests,
profiles, and graphs are all part of the study.

RMSNorm, affine linear projection, RoPE, and grouped-query attention have Mojo
host references, Apple GPU implementations, independent oracle tests, and
reproducible measurements. The [attention sublayer](studies/attention_sublayer/README.md)
composes these operations with a persistent KV cache, output projection and
residual addition. Its integrated Mojo entrypoint combines the earlier packed
QKV, MMA projection and FP32 GQA studies. The [MLP sublayer](studies/mlp_sublayer/README.md)
composes RMSNorm, SwiGLU and residual under a frozen BF16 contract, with
validated rowwise and tiled projection paths. The [decoder layer](studies/decoder_layer/selection.md)
composes both sublayers and confirms workload-specific kernel choices for
full prefill, cached prefill and decode. The [full-model development candidate](docs/generation.md)
composes 24 layers and native greedy generation, but acceptance is blocked by
[independent reference schedule confirmation](studies/model_generation/README.md).
No full-model speedup or generation parity is established.

The [CPU text tokenizer](docs/tokenizer.md) implements Qwen normalization, splitting,
heap-based BPE, and streaming decoding in Mojo. Its command initializes the pinned
artifact automatically; `uv run --locked llm-mojo-tokenizer encode 'Hello world'`
encodes text without loading model weights.

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
| [GQA prefill](studies/gqa_prefill/README.md) | How do query tiling, online softmax and Apple matrix instructions interact? |
| [Attention sublayer](studies/attention_sublayer/README.md) | Do the individual kernel gains survive composition through Wo and the residual? |
| [MLP sublayer](studies/mlp_sublayer/README.md) | Do tiled projections improve the complete SwiGLU block while preserving its BF16 boundaries? |
| [Decoder layer](studies/decoder_layer/selection.md) | Which kernel combinations improve the complete decoder in each execution mode? |
| [CPU tokenizer](studies/tokenizer/README.md) | When does heap BPE improve complete text encoding? |
| [Full-model reference](studies/model_generation/README.md) | Does BF16 full prefill agree with cached execution across all 24 layers? |

![Full and incremental GQA prefill latency on Apple M4 Pro](studies/gqa_prefill/latency.png)

GQA prefill tracks both query rows and KV context length. It compares fusion,
query tiling, matrix instructions and head reuse, including direct comparisons
against a strong optimized control. Hot and ring24 measurements have different
synchronization boundaries. The [prefill study](studies/gqa_prefill/README.md)
explains the mappings, calibration, compiler spills and profile evidence.
A bounded follow-up isolates accumulator representation, QK loop scheduling,
barriers and score storage; rolled QK reduces reported spills and demonstrates
modest gains on part of the full workload matrix.
See the [study index](studies/README.md) for retained
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
uv run mojo --version
uv run --locked llm-mojo-tokenizer setup
uv run --locked llm-mojo-validate
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
