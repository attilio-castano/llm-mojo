# Documentation map

Each fact has one home; other pages give a sentence and a link. Start with the
[README](../README.md) to set up and chat.

## Use

| Page | What it covers |
| --- | --- |
| [Terminal chat](chat.md) | Controls, system messages, context limits and what the session keeps |
| [Commands and configuration](cli.md) | Every command, setup and the shared model store, presets, modes, bench and validate |
| [Plain-text generation](generation.md#native-plain-text-generation) | `llm-mojo generate`, its modes and its report events |

## Understand

| Page | What it covers |
| --- | --- |
| [How a token flows through the engine](walkthrough.md) | One chat turn from the command line to streamed text, with shapes, sizes and where the time goes |
| [Project direction](project.md) | Goal, what the evidence establishes, determinism and the next research questions |
| [Layout notation](layouts.md) | How logical values, storage, thread ownership and reduction order are written down |
| [Serving plan](serving-plan.md) | The proposed multi-request engine: interfaces, KV blocks, gates and phases |

## Contracts

| Page | What it covers |
| --- | --- |
| [Model contract](model.md) | Pinned weights, operation arithmetic, conversation semantics and the correctness and diagnostic policy |
| [Runtime and workload policy](generation.md) | Model ownership, the Fast/baseline/consistent routes and how a call picks its kernels |
| [Tokenizer](tokenizer.md) | The native byte-level BPE and its exactness against the reference |
| [Attention sublayer](attention-sublayer.md) | Attention arithmetic, reference authority, mappings and reproduction |
| [MLP sublayer](mlp-sublayer.md) | SwiGLU rounding, fixtures, acceptance budgets and reproduction |
| [Decoder layer](decoder-layer.md) | Composition, fixtures, acceptance, configurations and execution policies |

## Develop

| Page | What it covers |
| --- | --- |
| [Development](development.md) | Toolchain, reference machine, setup and what validation runs |
| [Experimental method](experiments.md) | The paired measurement protocol and what evidence belongs in Git |
| [Measurement tools](../src/llm_mojo/benchmarks/README.md) | Building, running, profiling and replaying studies |

## Evidence

| Page | What it covers |
| --- | --- |
| [Study index](../studies/README.md) | Every study grouped as current route, decode experiments, numerical history and operations |
| [Model studies](../studies/model_generation/README.md) | The full-model studies, each with its status |

## History

The [history index](history/README.md) lists completed plans and superseded
narratives, kept as written, with the current page that replaces each.
