# llm-mojo

Run Qwen2.5 locally in a terminal with a native Mojo inference engine on Apple
Silicon. Then follow the implementation from conversation tokens and persistent
KV caches down to the GPU kernels that make it fast.

The project is an understandable study of LLM inference from first principles.
The first end-to-end milestone is complete: **Qwen2.5-0.5B-Instruct in BF16,
Fast inference on Metal, and interactive multi-turn chat**. Code, numerical
diagnostics and reproducible measurements explain how it works.

## Run the chat

Use an Apple Silicon Mac with the [development prerequisites](docs/development.md):
`uv`, Python 3.12, a C linker, Xcode and the Metal toolchain. `uv` resolves the
locked Mojo and MAX environment. The measured reference machine is an M4 Pro
with a 20-core GPU and 24 GB of unified memory; minimum hardware requirements
have not been established.

Prepare the pinned tokenizer and checkpoint once, from the repository root:

```sh
uv sync --locked
uv run --locked llm-mojo-tokenizer setup
uv run --locked --script tests/fixtures/generate.py attention_checkpoint -- --download --download-only
uv run --locked --script tests/fixtures/model_reference.py prepare --output build/model-prepared-v1
```

The checkpoint download is approximately 988 MB; preparation writes additional
model tensors locally. Assets are verified against the [pinned model contract](docs/model.md)
and remain outside Git. Preparation requires a new output directory; skip it
if you already have the verified prepared checkpoint.

```sh
uv run --locked python -m llm_mojo.chat --prepared build/model-prepared-v1
```

The first launch compiles the executable. Type a message and the reply streams
to the terminal. `/reset` clears the conversation while keeping weights loaded;
`/exit` or Ctrl-D exits. Ctrl-C cancels input or interrupts a reply. See the
[chat guide](docs/chat.md) for controls, custom system messages and context limits.

The session keeps all 24 layers' KV caches between turns and processes only the
uncached suffix. It supports system/user/assistant messages, greedy decoding,
and up to 4,096 total conversation tokens, including formatting and replies.
Tokenization, chat state, model execution and streaming are native Mojo; Python
handles asset preparation, verification and building before handing off execution.

## What makes it fast

The runtime brings together packed QKV projections, integrated attention,
workload-specific attention and projection kernels, tiled multi-row MLPs, and
persistent caches. **Fast is the default.** Its shared dispatcher chooses measured
kernel combinations by incoming row count and total context length on M4 Pro.
Other shapes use the existing optimized baseline. The [runtime guide](docs/generation.md#workload-policy)
explains the exact selection; Fast does not imply every experimental kernel is
used or that every workload is fully optimized.

Retained Apple M4 Pro / Metal evidence:

| Measurement | Observed result | Scope |
| --- | --- | --- |
| [Full-model optimization](studies/model_generation/runtime.md) | 5.6–51.2% lower latency across 11 selected workloads | 1,360 paired samples against our optimized configuration 0; synchronized model forward, excluding loading and greedy readback |
| [Conversation cache reuse](studies/model_generation/chat.md#performance-observations) | About 24% and 51% less prefill time on two follow-up turns | 120 paired samples against full-history replay; study capacity 512 |
| [Terminal generation](studies/model_generation/chat.md#performance-observations) | About 62–73 output tokens/second after the first token | Six completed instrumented replies; includes readback and streaming, excludes loading and first-token latency |

These measurements have different boundaries and describe the tested workloads.
A matched performance comparison against Hugging Face remains future work.

## Correctness and numerical diagnosis

Pinned assets, causal positions, exact token history, cache preservation and
submission accounting are required invariants. Independent operation tests retain
their numerical contracts. Full-model differences against HF or another prompt
chunk schedule are recorded as diagnostics: tensor errors, output distributions,
next-token choices and generated trajectories.

On the retained generation histories, **191 of 192 greedy token IDs matched HF**;
the remaining reference prediction had an exact top-logit tie. This is a bounded
token-choice observation, not byte-identical logits or a general quality claim.
The [runtime study](studies/model_generation/runtime.md#numerical-diagnosis) retains
both agreements and discrepancies. Fast does not promise identical results when
prompt chunking changes.

## Open research: KV-cache scheduling and determinism

Given the same weights and token sequence, should processing the prompt all at
once, in chunks, or one token at a time produce identical KV caches and logits?
The causal computation is mathematically equivalent, but call shapes can change
kernel selection and floating-point reduction order. Small differences can cross
BF16 rounding boundaries, propagate through layers and change a greedy prediction.
The [HF attention investigation](studies/model_generation/backend.md) traces one
such mechanism in the reference implementation.

Preserving an existing cache byte for byte is already a required invariant.
Producing identical cache values when building it under different schedules is
the additional research question. Here, scheduling means how one sequence is
divided into model calls; multi-request scheduling remains outside current scope.

The [decoder policy study](studies/decoder_layer/policies.md) establishes exact
schedule agreement for the tested single-layer configurations and measures its
cost. The [full-model consistency study](studies/model_generation/consistency.md)
records the remaining native model boundary; full-model schedule invariance has
not been established.

The follow-up asks which arithmetic and dispatch choices preserve that invariant,
how remaining differences affect predictions, and how much determinism costs
relative to Fast. This is a second research track alongside the working chat
engine, with cache identity, numerical closeness and token agreement reported
separately.

## Understand the engine

Read from the working application down to the operations, or start with a kernel:

| Layer | Read |
| --- | --- |
| Terminal and session state | [Chat guide](docs/chat.md) · [cache-reuse and interaction evidence](studies/model_generation/chat.md) |
| Complete Qwen runtime | [Model contract](docs/model.md) · [ownership and dispatch](docs/generation.md) · [Fast measurements](studies/model_generation/runtime.md) |
| Decoder composition | [Kernel selection](studies/decoder_layer/selection.md) · [attention](studies/attention_sublayer/README.md) · [MLP](studies/mlp_sublayer/README.md) |
| GPU operations | [RMSNorm](studies/rms_norm/README.md) · [linear decode](studies/linear_decode/README.md) · [linear prefill](studies/linear_prefill/README.md) · [RoPE](studies/rope/README.md) · [GQA decode](studies/gqa_decode/README.md) · [GQA prefill](studies/gqa_prefill/README.md) |
| Text processing | [Native tokenizer](docs/tokenizer.md) · [CPU measurements](studies/tokenizer/README.md) |

The [study index](studies/README.md) links retained evidence and regeneration
commands. The [project direction](docs/project.md) separates the completed Fast
milestone from follow-ups: schedule determinism, matched HF benchmarking and
further optimization driven by measured application costs. Sampling,
quantization, batching and additional models remain outside the current scope.

## Development

Every optimization needs independent numerical evidence and a reproducible
benchmark. Keep allocation, memory layout, work ownership and synchronization
explicit. See [development guidance](docs/development.md),
[layout notation](docs/layouts.md), the [experimental method](docs/experiments.md)
and [measurement tools](src/llm_mojo/benchmarks/README.md).

```text
src/llm_mojo/             Native inference, chat and development commands
src/llm_mojo/benchmarks/   Measurement, profiling and report generation
tests/                   Correctness tests and independent oracle generators
docs/                    Usage, contracts and project direction
studies/                 Explanations, compact measurements and graphs
```
