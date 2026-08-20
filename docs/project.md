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
showing what changed.

## Initial model scope

The first model should be a small modern decoder-only transformer, likely in the
Qwen or Llama family and roughly 0.5B–1.5B parameters.

The first model-level milestone is numerical parity, not speed:

> Given the same weights and input tokens, reproduce the reference
> implementation's next-token logits within a documented tolerance.

The path includes token embeddings, RMSNorm, Q/K/V projections, RoPE, causal or
grouped-query attention, output projection, SwiGLU, residual connections, final
normalization, the LM head, logits, and sampling.

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
