# Project direction

## Goal

Build a small, understandable LLM inference engine in Mojo, initially optimized
for Apple Silicon and Metal.

The objective is not merely to generate text. The project should expose and
measure the machinery that turns weights and tokens into next-token logits:

```text
model architecture
      ↓
reference inference
      ↓
explicit runtime machinery
      ↓
custom Mojo operations
      ↓
custom GPU kernels
      ↓
SIMD, layout, and synchronization choices
      ↓
Metal
      ↓
Apple GPU
```

Abstractions should be peelable. A reader should be able to move from a clear
reference operation toward the memory access, execution, and synchronization
decisions that implement it on hardware.

The project uses a small [layout language](layouts.md) to keep logical values,
storage mappings, work partition, and reduction order explicit without
introducing a project-specific tensor or layout algebra.

## Method

The development loop is:

```text
correct reference implementation
        ↓
measurement
        ↓
identify a bottleneck
        ↓
implement one optimization
        ↓
verify numerical correctness
        ↓
benchmark
        ↓
inspect generated code when useful
```

An optimization is incomplete without both correctness evidence and a benchmark
showing what changed. The [experimental method](experiments.md) defines the
shared vocabulary, measurement discipline, and evidence lifecycle used for
performance work.

## Initial model scope

The first model is Qwen2.5-0.5B-Instruct in BF16. Its immutable artifact
revision, runtime boundary, conversation semantics, and correctness criteria are
defined in the [initial model contract](model.md).

The first model-level milestone is numerical parity, not speed:

> Given the same weights and input tokens, reproduce the reference
> implementation's next-token logits within a documented tolerance.

The path includes token embeddings, RMSNorm, Q/K/V projections, RoPE, causal or
grouped-query attention, output projection, SwiGLU, residual connections, final
normalization, the LM head, logits, and sampling.

## Evidence-gated roadmap

The roadmap expresses dependency order, not a delivery schedule. A stage is
complete only when its exit evidence is reproducible. Work may explore a later
stage, but claims may not skip an earlier evidence gate.

### 0. Foundation

Establish the stable Mojo and MAX toolchain, package boundary, test runner, and
project principles.

Exit evidence: the locked environment resolves and the package import smoke
test passes. This is the bootstrap stage.

### 1. Reference contracts

Pin the reference machine, model artifacts, dtype, initial runtime boundary,
conversation semantics, and correctness criteria.

Exit evidence: the machine and model documentation is internally consistent,
the immutable upstream artifacts and checksums resolve, and no inference or
performance claim is made.

### 2. Reference operations

Implement Qwen operations in small, inspectable Mojo modules, beginning with
RMSNorm and progressing through linear projections, RoPE, grouped-query
attention, SwiGLU, residuals, embeddings, and the LM head. Python may generate
small oracle fixtures but is not part of the inference path.

Exit evidence: every implemented operation matches a provenance-bearing oracle
fixture within a tolerance declared before comparison. An operation is not an
optimization and needs no performance claim. This is the current stage.

### 3. Decoder block

Compose the operations into one deterministic Qwen-compatible decoder block
using a deliberately tiny fixture whose intermediate tensors remain easy to
inspect.

Exit evidence: every block boundary and the final block output match the
reference oracle, with shapes, layouts, dtypes, and allocations documented.

### 4. Full-model forward pass

Load the pinned Qwen2.5-0.5B-Instruct weights and compose embeddings, all 24
decoder blocks, final normalization, and the LM head.

Exit evidence: fixed token IDs reproduce reference next-token logits within the
declared tolerance. This proves a forward pass, not generation quality or
performance.

### 5. Stateful generation

Add deterministic greedy generation, separate full prefill from one-token
decode, and introduce the persistent KV cache.

Exit evidence: cached logits match full recomputation at every generated
position, cache accounting is exact, and instrumentation shows that cached
prefixes were not recomputed.

### 6. Multi-turn sessions

Add canonical token history, incremental prefill for appended user turns,
prefix validation, cache invalidation, stop-token handling, and the V0 context
limit.

Exit evidence: the three-turn fixture in the
[initial model contract](model.md#v0-correctness-acceptance) matches
full-transcript recomputation at every turn boundary and generated position.

### 7. Performance baseline

Measure the correct implementation on the reference machine. Separate first
prefill, incremental prefill, and decode workloads; vary prompt, suffix, and
cache lengths; and prove the actual runtime device and backend.

Exit evidence: a reproducible baseline records all required metadata and makes
no causal performance claim beyond the measured implementation.

### 8. Measured optimization

Profile the baseline, choose one demonstrated bottleneck, implement one change,
rerun numerical correctness, and compare against the unchanged workload. Likely
topics include SIMD, weight layout, allocation, fusion, synchronization, and
Apple GPU kernels, but measurement determines their order.

Exit evidence: each optimization has both correctness evidence and a
reproducible before-and-after benchmark. A faster microkernel alone does not
establish faster model decoding.

## Repository growth

The repository should remain shallower than the implementation until real code
creates a stable boundary. Do not create placeholder runtime, kernel, benchmark,
or experiment hierarchies.

Use these placement rules as the project grows:

- Put Mojo inference code directly under `src/llm_mojo/` while the modules are
  few. Introduce a subpackage only when multiple implemented modules share a
  coherent interface or ownership boundary.
- Keep tests directly under `tests/` while they are few, using
  `tests/test_<subject>.mojo`. Split unit and integration test directories only
  when their setup or execution genuinely differs.
- When the first external numerical fixture exists, place its data, manifest,
  and one-purpose generator together under `tests/fixtures/<case>/`. Promote
  Python oracle support into `src/llm_mojo/` only after more than one consumer
  needs the same code.
- Add `benchmarks/` only with the first reproducible benchmark. Every benchmark
  must name the implementation and workload it measures and must retain a
  correctness gate.
- Add `experiments/` only with the first recorded experiment. Keep reusable
  measurement instruments under `benchmarks/`; keep the frozen protocol, raw
  samples, provenance, and bounded conclusion together in the experiment
  record.
- Keep model weights, download caches, compiler output, and generated full-model
  artifacts outside the repository.
- Update documentation when a boundary becomes real. A directory name is not
  evidence that the corresponding capability exists.

The expected near-term evolution is illustrative rather than pre-created:

```text
src/llm_mojo/
    <implemented operation>.mojo
tests/
    test_<implemented behavior>.mojo
    fixtures/<case>/              # after the first external fixture
benchmarks/                       # after the first measured implementation
experiments/                      # after the first recorded experiment
```

## Runtime direction

Once the reference path is correct, the engine can progressively expose:

- autoregressive decoding;
- distinct prefill and decode paths;
- KV caching and its memory layout;
- block-based allocation and batching;
- prefix reuse;
- sampling and quantization;
- speculative decoding;
- unified-memory-aware execution strategies.

These are directions, not claims about current functionality.

## Apple Silicon

Apple Silicon makes memory behavior central because CPU and GPU share unified
memory. The project should measure rather than assume the consequences.

Important questions include:

- what becomes bandwidth-bound during decode;
- how context length changes KV-cache traffic and latency;
- which layouts and synchronization strategies work best;
- which operations benefit from fusion;
- how prefill and decode kernels should differ;
- how close readable Mojo can get to mature runtimes such as MLX and llama.cpp.

Closing the entire performance gap is not required. Explaining it is valuable.

## Evidence

Useful measurements eventually include time to first token, time per output
token, tokens per second, memory use, KV-cache size, memory bandwidth, kernel
latency, arithmetic intensity, and scaling with batch and sequence length.

Results must identify the hardware, operating system, Mojo and MAX versions,
commit, workload, dtype, warmup, repetitions, and synchronization boundaries.
Recorded results follow the [experimental method](experiments.md), which keeps
operation, composition, model-phase, and end-to-end claims distinct.

## Success

The project succeeds when a technically sophisticated reader can understand:

- how a decoder-only transformer performs inference;
- what data lives in memory and how KV caching changes computation;
- why prefill and decode behave differently;
- where time and bandwidth are spent;
- how the critical GPU kernels work;
- how successive optimizations alter correctness and measured performance.

Understanding, correctness, measurement, and engineering depth take priority
over breadth.
