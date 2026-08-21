# EXP-0001: Apple GPU RMSNorm baseline characterization

Status: **planned**

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

No decisive measurements have been collected.

## Interpretation

No finding yet.

## Nonclaims

This experiment will not by itself establish decoder-block, model-phase, or
end-to-end inference performance. Source-derived bytes will not be described as
observed hardware traffic, and timing alone will not be used as a causal
bottleneck claim.

## Decision

None. The baseline implementation remains unchanged until this experiment is
complete.
