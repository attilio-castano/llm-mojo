# llm-mojo

An educational project on LLM inference: how much of a modern inference engine
can I build, understand and measure on a MacBook, from first principles, in
Mojo?

I am a mathematician using this project to learn three things by building them
myself:

- **A systems programming language.** Mojo lets one language cover the
  tokenizer, the model and the GPU kernels, with data flow, memory and
  synchronization written out instead of hidden behind a framework.
- **High-performance GPU kernels.** Why one kernel is faster than another comes
  down to tensor shapes, memory traffic, which thread owns which work, and when
  the GPU has to wait. The studies explain each optimization in those terms.
- **System design.** A chat for one person is only the start. Serving many
  requests at once raises the questions production engines are built around:
  which requests share a GPU step, who owns the cache memory, and what happens
  under memory pressure or failure.

The concrete engine runs Qwen2.5-0.5B-Instruct as an interactive chat on the GPU
of an Apple Silicon Mac, the hardware I have, so a MacBook sets the limits. Every
part, from the token IDs of your message to the GPU kernels, is documented,
tested against independent references and measured. Optimizations that did not
pay off stay in the record alongside the ones that did, with the reason.

## Where things stand

- **Chat works.** One conversation runs on the GPU, with the weights and cache
  kept loaded between turns. On the reference Apple M4 Pro, replies stream at
  107–115 tokens per second after the first token
  ([how this was measured](studies/model_generation/residual-norm.md)).
- **The synchronous serving core works.** Paged KV storage, mixed prefill/decode,
  continuous admission, aborts and recomputation under memory pressure are native
  Mojo. The [engine study](studies/model_generation/engine-core.md) retains
  correctness checks and load measurements. Incremental admission remains the
  default; optional [lifetime reservation](studies/model_generation/engine-core.md#lifetime-reservation-admission-bounded-successor-study)
  removes repeated replay on the frozen pressure trace by waiting until a request's
  declared cache growth fits.
  Asynchronous stepping, a Fast engine route and frontend integration remain in the
  [serving plan](docs/serving-plan.md).
- **Open questions.** For one conversation, submitting GPU work limits speed
  more than arithmetic does. Long prompts are slow to start. And whether building
  the same cache in different chunk sizes can give identical results is not yet
  established for the full model. The [project direction](docs/project.md)
  tracks each question, and the [study index](studies/README.md) holds every
  measurement.

## Run the chat

You need an Apple Silicon Mac with Xcode and its Metal toolchain, and
[uv](https://docs.astral.sh/uv/). The [development guide](docs/development.md)
lists the exact prerequisites. From the repository root:

```sh
uv run llm-mojo setup
uv run llm-mojo chat
```

`setup` checks the toolchain and prints the fix for anything missing. It
downloads the pinned model once per Mac, about 1 GB, into a shared store that
every checkout links to. It verifies every file and builds the chat program.
`uv run llm-mojo setup --check` reports readiness without changing anything.

Type a message and the reply streams back. `/reset` starts a new conversation
and keeps the weights loaded; `/exit` or Ctrl-D quits; Ctrl-C stops a reply or
clears your input. The [chat guide](docs/chat.md) covers system messages and
the 4,096-token context limit.

To continue raw text instead of chatting:

```sh
uv run llm-mojo generate --prompt "The capital of France is" --preset short
```

[Commands and configuration](docs/cli.md) covers everything else, including the
research modes, benchmarks and validation.

## How it works

- **Python prepares, Mojo runs.** Python verifies the model files and then hands
  over to a native Mojo program. Tokenization, the conversation, the model and
  streaming all run in Mojo.
- **Nothing is computed twice.** The model's 24 layers run on the GPU in BF16. A
  key-value cache keeps every processed token, so a new message computes only its
  own tokens, and each reply token after the first costs one model pass.
- **Speed comes from launching less.** For one conversation, generating a token
  is limited mostly by launching GPU work, not by arithmetic or memory
  bandwidth. The fast route launches fewer, fused kernels per token. Its fusions compute exactly the same
  bytes; its decode projections sum in a faster order, so their results can
  differ from the baseline route's in the last bits.

[How a token flows through the engine](docs/walkthrough.md) follows one chat turn
through the code, with every shape and size.

## What is checked

Every optimization has an independent numerical test and a reproducible
measurement. Some properties must match byte for byte: the pinned weights, the
conversation's tokens, cache contents, and routes that claim to compute the same
result. Comparisons with Hugging Face are recorded as diagnostics instead. On the
retained generation histories, 191 of 192 greedy next tokens matched, and the
exception was an exact tie
([numerical diagnosis](studies/model_generation/runtime.md#numerical-diagnosis)).
The [model contract](docs/model.md#correctness-and-diagnostic-policy) states the
rules.

## Where to go next

| To | Read |
| --- | --- |
| Find any guide or contract | [Documentation map](docs/README.md) |
| Follow a token through the code | [Walkthrough](docs/walkthrough.md) |
| See every measurement, and why each optimization was kept or not | [Study index](studies/README.md) |
| Read the goals and open questions, including determinism | [Project direction](docs/project.md) |
| See the plan for serving many requests at once | [Serving plan](docs/serving-plan.md) |
| Build, test and measure | [Development](docs/development.md) |

```text
src/llm_mojo/              Native inference, chat and development commands
src/llm_mojo/benchmarks/   Measurement, profiling and report generation
tests/                     Correctness tests and independent oracle generators
docs/                      Usage, contracts, the walkthrough and project direction
studies/                   Explanations, compact measurements and graphs
```
