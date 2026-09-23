# Project direction

## Goal and completed milestone

Build a small, understandable LLM inference engine in Mojo, initially optimized
for Apple Silicon and Metal. A reader should be able to run it, inspect the
machinery and connect an implementation choice to numerical and performance
evidence.

The first end-to-end milestone is complete: native Fast Qwen2.5-0.5B-Instruct
terminal chat with batch-one BF16 inference and persistent KV caches. The
[chat guide](chat.md) is the user entry point. The [model contract](model.md)
defines the pinned weights, arithmetic and 4,096-token runtime boundary.

The implementation connects:

```text
terminal messages and exact token history
        ↓
native tokenizer and Qwen chat framing
        ↓
incremental prefill, greedy decode and persistent KV caches
        ↓
24 learned decoder layers, final normalization and tied LM head
        ↓
measured Fast kernel selection
        ↓
explicit layouts, work ownership and synchronization
        ↓
Metal on Apple Silicon
```

Abstractions should be inspectable. The [layout language](layouts.md) separates
logical values, storage, thread ownership and reduction order. Python supplies
asset preparation, independent oracles and development tooling; the inference
engine and interactive session remain Mojo-first.

## What the evidence establishes

The [Fast runtime study](../studies/model_generation/runtime.md) completes model
integration, native generation and workload-specific optimization. Eleven
measured cells reduce synchronized full-model forward time by 5.6–51.2% against
our optimized configuration 0 on M4 Pro / Metal. Other shapes retain that
baseline; the result does not establish a universal best kernel or an HF speedup.
The [composed decode study](../studies/model_generation/residual-norm.md) then
promoted the single-row route that every generated token uses: configuration 26
with GPU argmax, buffer swapping and residual/RMSNorm fusion. It lowers
complete-token latency by 17.1–24.5% against the previous Fast route and
streams 107–115 tokens/s after the first token.

The [chat study](../studies/model_generation/chat.md) adds exact template fixtures,
three-turn cache checks, full-history numerical diagnostics, paired cache-reuse
measurements and actual terminal interaction. Weights and caches stay resident;
subsequent turns submit only the uncached suffix. Reset, interruption, stop
handling and context rejection are part of the implemented lifecycle.

Evidence is specific to its claim:

- Artifact identity, architecture, causal positions, token history, cache storage,
  submission accounting and lifecycle are required implementation invariants.
- Independent operation tests retain their declared numerical contracts.
- Full-model HF and cross-schedule differences are diagnostic observations,
  including intermediate errors, logit distributions and token choices.
- Speed claims require paired measurements with an explicit baseline and timing
  boundary. A faster kernel does not by itself establish faster terminal chat.

Greedy selection has a fixed tie rule, but changing prompt chunk sizes can change
floating-point reductions and predictions. Schedule-invariant full-model execution
is not part of the completed Fast contract. The earlier failed full-model
qualification policies and their records remain [historical evidence](../studies/model_generation/README.md#numerical-history);
they were not converted into passing results. The approved
[Fast plan](history/fast-generation-plan.md) records the move to numerical diagnosis.

## Follow-up direction

The working Fast chat is the baseline for further work. These are separate
research questions, not prerequisites for calling the current milestone complete:

1. **Schedule determinism.** Define the desired invariant across full, chunked
   and one-token execution, then diagnose and measure a dedicated consistent
   route. The existing [decoder policy study](../studies/decoder_layer/policies.md)
   and [full-model investigation](../studies/model_generation/consistency.md)
   establish useful component results and an unresolved full-model boundary.
   Serving adds batch composition, cache history and replay as further sources
   of difference; a batch-invariant serving mode is an optional extension.
2. **Matched HF comparison.** Numerical comparisons already exist. A performance
   study must name the HF backend/device, precision, identical token workload,
   cache behavior and timing boundary before comparing prefill or decode.
3. **Further Fast optimization.** Decode is now limited by host submission.
   About 98% of each token's launch-submission interval is inside MAX's enqueue
   runtime, and reusing compiled kernel handles gave no qualifying speedup
   ([runtime enqueue](../studies/model_generation/runtime-enqueue.md)). The
   pinned Metal backend cannot record command graphs, so batching launches is
   unavailable ([batching feasibility](../studies/model_generation/batch-support.md)).
   What remains is fewer launches per token, or batching below MAX's public API.
   Prefill is now the largest cost a user sees: the first token of a
   3,839-token prompt takes about 2.1 s. Keep current measurements as the
   baseline and retain new numerical diagnostics.
4. **Serving engine.** Serve many concurrent requests with batched decode, a
   paged KV cache, continuous batching and prefix caching, behind a separate
   frontend process that survives engine failures. The [serving plan](serving-plan.md)
   defines its architecture, exact gates and phases. It serves with Fast and
   records numerical differences from batching, cache reuse and replay as diagnostics.

Sampling, quantization, longer contexts, tool-oriented templates and additional
model families can follow when they answer a concrete need. They are not current
functionality or commitments for the next milestone.

## Method

Start with a clear reference operation, measure it, identify a bottleneck,
implement one change, verify its numerical behavior and benchmark the result.
Inspect generated code or profiles when they help explain the outcome through
tensor dimensions, memory traffic, reuse, work ownership and synchronization.
An optimization is incomplete without both numerical evidence and a reproducible
before-and-after measurement.

## Evidence

Record hardware, backend, software versions, source identity, model revision,
dtype, workload, warmups, samples and synchronization boundaries. Keep operation,
composition, model-forward and application timings distinct. The
[experimental method](experiments.md) defines evidence retention and performance
selection; the [study index](../studies/README.md) organizes the results.

## Repository structure

The [CLI guide](cli.md#code-ownership) maps commands, Qwen composition, runtime
services, layers and kernels under `src/llm_mojo/`. Add a subpackage only when
implemented ownership boundaries justify it. Tests live in
`tests/`, with independent oracle generators and frozen identities in
`tests/fixtures/`. Reusable measurement tools belong in
`src/llm_mojo/benchmarks/`; studies own explanations and compact measured evidence.

Usage and current contracts belong in `docs/`, listed by the
[documentation map](README.md). Completed plans and numerical
investigations remain linked as [history](history/README.md), so they do not
obscure the current entry point. A new parameter choice usually belongs in an
existing measurement matrix, not a new experiment hierarchy. Weights, generated
oracle arrays, binaries and full traces remain outside Git.

## Success

The project succeeds when a reader can run the model and explain how tokens
become logits, what the cache saves, which kernels the workload selects and why
a measured optimization helps. Understanding the remaining gap to mature
runtimes is valuable even when closing that entire gap is not the objective.
