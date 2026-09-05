# AGENTS.md

## Purpose

Build an understandable study of LLM inference and its engine from first
principles in Mojo, with Apple Silicon as the initial hardware target.
Code, numerical tests, profiling, and explanations are part of the project.

## Engineering rules

- Keep the inference engine Mojo-first. Use Python only for reference oracles,
  fixtures, interop, and development tooling.
- Establish a correct reference path before optimizing it.
- Require a numerical correctness test and a reproducible benchmark for every
  optimization.
- Explain results through tensor dimensions, memory traffic, work ownership,
  reuse, and synchronization. Graphs should answer a specific study question.
- Record dtype, tensor shape, layout, synchronization boundaries, hardware,
  software versions, and commit identity for performance evidence.
- Never report GPU performance without proving which device and backend ran;
  a silent CPU fallback is not GPU evidence.
- Prefer explicit data flow, allocation, and synchronization over opaque
  framework behavior.
- Use stable Mojo and MAX releases resolved through `uv.lock`. Upgrade them
  deliberately and validate the full repository afterward.

## Project map

- `docs/`: project direction and development guidance.
- `src/llm_mojo/`: Mojo inference operations and Python development commands.
- `src/llm_mojo/benchmarks/`: measurement, profiling, and report generation.
- `tests/`: correctness tests and independent oracle generators.
- `studies/`: topic explanations, compact measurements, and graphs.

Add structure only when real code needs it. Do not create speculative runtime,
kernel, benchmark, or experiment hierarchies in advance.
Extend existing studies and measurement tools; a new parameter choice usually
belongs in the existing matrix rather than a new runner or experiment folder.

## Change discipline

- Inspect current state before editing.
- Keep changes small and explain what they establish.
- Do not add model weights, generated artifacts, or benchmark claims without a
  documented provenance and purpose.
- Retain compact raw samples and provenance sufficient to regenerate published
  tables and graphs. Keep generated oracle arrays, binaries, and full traces
  outside Git; see `docs/experiments.md` for the evidence contract.
- Run the documented validation commands before committing.
