# llm-mojo

A small LLM inference engine written in Mojo, built to be read. It runs
Qwen2.5-0.5B-Instruct as an interactive chat on the GPU of an Apple Silicon Mac.
Every part, from the token IDs of your message to the GPU kernels, is
documented, tested against independent references and measured.

On the reference Apple M4 Pro, replies stream at 107–115 tokens per second after
the first token ([how this was measured](studies/model_generation/residual-norm.md)).

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
  own tokens, and each reply token is one pass through the model.
- **Speed comes from launching less.** Generating a token is limited mostly by
  launching GPU work, not by arithmetic or memory bandwidth. The fast route
  launches fewer, fused kernels per token and computes exactly the same bytes.

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
