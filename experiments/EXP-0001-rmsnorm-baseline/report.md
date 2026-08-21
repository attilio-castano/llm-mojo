# EXP-0001: Apple GPU RMSNorm baseline characterization

Status: **in progress**

Evidence level: **operation**

## Question

How does the existing Apple GPU RMSNorm implementation scale across decode,
batched-decode-like, and prefill row counts, and what limits representative
regimes?

This is a characterization experiment. It does not compare an optimization to
the baseline.

## Frozen protocol

The machine-readable protocol is in [`manifest.json`](manifest.json). It fixes:

- the shared-memory-tree implementation from commit `148dbaa`;
- BF16 tensors with FP32 accumulation and hidden size 896;
- row counts 1, 4, 16, 128, 512, 2048, and 4096;
- four independent workload blocks in ascending, descending, descending,
  ascending order;
- 1,000 warmup iterations and 10 retained repetitions per workload in each
  block;
- synchronized per-dispatch latency as the primary measurement;
- AC power, no reported thermal warning, a clean commit, and proven Apple GPU
  plus Metal runtime identity;
- separate benchmark and Metal profiling evidence.

## Results

No decisive timing measurements have been collected. The recorded-run gate
correctly rejected battery power while preserving the AC-power requirement.

A rows=1 tooling calibration is retained in
[`profile-calibration.json`](profile-calibration.json). It established that:

- the standalone binary and scrubbed trace-analysis path reproduce the fixed
  one-correctness, 1,000-warmup, 5,000-profile dispatch sequence;
- the emitted Metal LLVM sidecar contains 128 FP32 shared values (512 bytes),
  seven shared-tree reduction stages, and nine workgroup barriers;
- the stock Metal System Trace selected no counter set and disabled the shader
  timeline;
- a direct `Metal GPU Counters` capture selected a counter profile that this
  target reported as unsupported;
- this single trace reported no target compiler-spill event, which is bounded
  to the capture and is not a universal spill-free claim.

The calibration was battery-powered and system load was uncontrolled. Its
instrumented intervals support no latency, bandwidth, occupancy, or stall-time
claim.

## Interpretation

The shared-memory tree is now a concrete optimization hypothesis, not yet a
causal bottleneck finding. Compiler output proves that the synchronization and
storage exist; it does not prove that removing them will materially improve
ordinary execution. That requires the clean AC baseline, representative
transition and saturation profiles, and a paired variant comparison.

## Nonclaims

This experiment will not by itself establish decoder-block, model-phase, or
end-to-end inference performance. Source-derived bytes will not be described as
observed hardware traffic, and timing alone will not be used as a causal
bottleneck claim.

## Decision

The baseline implementation remains unchanged. Collect the four-block AC
baseline next, then profile the measured transition and saturation workloads
before implementing one reduction variant.
