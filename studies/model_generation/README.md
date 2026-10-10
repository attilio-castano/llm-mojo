# Qwen runtime, chat and numerical studies

These studies take the complete Qwen2.5-0.5B-Instruct model on Apple M4 Pro /
Metal from its first composition to today's Fast route. They show how it runs,
where a token's time goes, which decode optimizations were promoted and how the
numerical policy developed. The [runtime guide](../../docs/generation.md)
describes current behavior; each study keeps its evidence, limits and
reproduction commands.

**Status** says what each result means for the engine today:

- **Current**: describes the runtime as it runs today.
- **Promoted**: its candidate is part of Fast.
- **Not promoted alone**: exact, but it missed its own gate; it was later
  promoted as part of a composition.
- **Superseded**: a later study replaced the result.
- **Diagnostic**: explains behavior without a promotion decision.
- **Blocked**: stopped at a capability the backend does not provide.
- **Failed gate**: a qualification that did not pass; its evidence is kept unchanged.

## Current routes and engine

| Study | Status | What it establishes |
| --- | --- | --- |
| [Fast runtime](runtime.md) | Current | All 24 layers, native generation, the eleven measured multi-row configuration cells and HF numerical diagnostics |
| [Terminal chat](chat.md) | Current | Session state, exact token history, persistent caches, controls and measured cache reuse |
| [Engine core](engine-core.md) | Current | Mixed reference execution, incremental admission and optional [lifetime reservation](engine-core.md#lifetime-reservation-admission-bounded-successor-study); retained synchronous studies bound pressure and budgeting behavior; optional [async stepping](engine-core.md#asynchronous-stepping-implementation-and-acceptance-contract) is implemented with its own numerical, lifecycle and paired-load gates |
| [Residual/RMSNorm alone and composed](residual-norm.md) | Promoted | Today's single-row decode route: configuration 26 with GPU argmax, buffer swapping and residual/RMSNorm fusion; 17.1–24.5% lower latency and 107–115 tokens/s |
| [Combined QKV and activation fusion](combined-fusion.md) | Promoted | Configuration 26, the decode configuration: 16.9–19.4% lower latency with exact outputs and cache storage |
| [Reordered batched projections](batch-reordered.md) | Promoted | Arrangement 8, four adjacent products per lane: 17–36% shorter batched steps than arrangement 5 from B = 16, about 10% shorter single-sequence steps and 998 tokens/s for 64 sequences at 1,024 cached tokens, with 5's worst-case accuracy and HF agreement; it changes Fast's summation order |
| [Paged KV after the one-loop decode kernel](paged-kv-loop.md) | Promoted | With decode attention in one loop per SIMD group, blocks of 32, 64 and 128 slots cost nothing resolvable in 35 decode and prefill workloads; 32-slot slot-major blocks, selected and confirmed, are the default since 2026-10-05 |

## Decode experiments

| Study | Status | What it establishes |
| --- | --- | --- |
| [Paged KV translation cost](paged-kv.md) | Superseded | 2a's decode kernel made steps up to 2.9, 1.9 and 1.5 times as long with 32-, 64- and 128-slot blocks at 3,968 cached tokens, all of it in its per-block loop; the one-loop kernel replaced it |
| [Exact batched projections](batch-projections.md) | Superseded | Arrangement 5, four rows by four columns per SIMD group: 31–65% shorter batched steps with bit-identical outputs, 705 tokens/s for 64 sequences at 1,024 cached tokens; arrangement 8 replaced it |
| [Batched decode against batch size](batch-size.md) | Diagnostic | With tile 4, throughput levels off at 2.2–3.2× one sequence by B = 64; per-row work in the multi-row projections sets the cost, and tiles 8 and 16 are slower than tile 4 |
| [Inter-layer buffer swapping](buffer-swap.md) | Not promoted alone | 23 fewer compute copies with exact outputs; 5.3–8.3% median paired reductions missed the standalone gate |
| [GPU token selection](token-selection.md) | Not promoted alone | A separate GPU argmax gains 4.5–7% but missed its standalone gate; the fused vocabulary head was slower |
| [QKV fusion](qkv-fusion.md) | Superseded | Configuration 25: exact, with 12–15% lower token latency, but short-context calibration prevented promotion; configuration 26 replaced it |
| [Complete token profile](token-profile.md) | Diagnostic | Where a roughly 17 ms decode step spent its time before the decode fusions |
| [Projection arrangements](projection-arrangements.md) | Failed gate | None of five fixed-width and block-size arrangements met the frozen full-token gate |
| [Projection scheduling](projection-scheduling.md) | Diagnostic | Host submission limits short-context decode; a GPU backlog appears at long context |
| [Runtime enqueue](runtime-enqueue.md) | Diagnostic | Most launch-submission time is inside MAX's enqueue runtime; reusing compiled handles gave no qualifying speedup |
| [Metal batching feasibility](batch-support.md) | Blocked | The pinned MAX Metal backend cannot record a graph, so command batching is unavailable |

## Numerical history

The current [diagnostic policy](../../docs/model.md#correctness-and-diagnostic-policy)
requires exact implementation invariants and preserves independent operation
contracts. Full-model tensor differences, distributions and token choices are
observations to investigate. These studies explain how that policy developed;
their original gates and frozen evidence are unchanged.

| Study | Status | What it establishes |
| --- | --- | --- |
| [Reference schedule qualification](reference-qualification.md) | Failed gate | The original tolerance-gated qualification: independent confirmation failed 8 of 2,025 checks |
| [Fast reference qualification](fast-reference.md) | Failed gate | Reference-only calibration stopped: the intermediate-error ceilings did not cover measured arithmetic variation |
| [Native consistency route](consistency.md) | Failed gate | 71,250 exact canonical HF comparisons pass; the first native full-model input fails seven frozen accuracy gates |
| [Rounding and propagation](rounding.md) | Diagnostic | Schedule-dependent roundoff crosses BF16 rounding boundaries and propagates; no prediction difference was observed |
| [HF/PyTorch execution shape](backend.md) | Diagnostic | The first difference is in PyTorch's QK matrix multiplication; query shape and causal-prefix layout remove it |

Schedule determinism at the full-model level and a matched HF performance
comparison remain follow-ups. Replay the retained model evidence and regenerate
its tables without weights or a GPU:

```sh
uv run --locked python studies/model_generation/summarize.py
```

The engine archives have separate [replay commands](engine-core.md#replay-and-next-work),
including independent reconstruction of the fitted cost model.

The current async implementation deliberately pins Mojo 1.1.0 / MAX 26.6.0.
Historical model and synchronous-engine tables keep their recorded toolchain,
source and device identities; the upgrade does not renew those measurements.
Async stepping overlaps host preparation and submission with ordered GPU work.
It retains two tickets at most, chains token IDs on the device, and drains before
KV reuse. Its [study commands](../../src/llm_mojo/benchmarks/README.md#asynchronous-engine-stepping)
require exact-build checkpoint qualification before paired sync/self-sync/async
collection. No async speed verdict follows from the implementation alone.
