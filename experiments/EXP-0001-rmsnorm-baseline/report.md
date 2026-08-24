# EXP-0001: Apple GPU RMSNorm baseline characterization

Status: **complete**

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

`EXP-0001-RUN-001` collected all four frozen blocks on AC power at clean commit
`b59717d`. The runtime identified the Apple M4 Pro through Metal in every block,
and the power and thermal gates remained valid at both run boundaries. The
retained evidence contains 40 samples per workload, 280 samples total, with
1,000 internal iterations per sample. Compact results and external-artifact
checksums are retained in [`baseline-run.json`](baseline-run.json).

| Workload | Median (ms/dispatch) | MAD (ms) | Interpretation |
| --- | ---: | ---: | --- |
| r1-h896 | 0.010433 | 0.001067 | batch-one decode; launch-floor regime |
| r4-h896 | 0.009572 | 0.001183 | hypothetical batched decode; launch-floor regime |
| r16-h896 | 0.009703 | 0.000679 | hypothetical batched decode; launch-floor regime |
| r128-h896 | 0.009912 | 0.000765 | short prefill-like; launch-floor regime |
| r512-h896 | 0.011752 | 0.000248 | measured transition workload |
| r2048-h896 | 0.033058 | 0.000077 | long prefill-like scaling regime |
| r4096-h896 | 0.057332 | 0.000254 | largest measured workload |

The first four workloads remain within a roughly 9.6–10.4 microsecond median
dispatch floor and have substantial small-signal variation. Rows=512 is the
first frozen workload above that floor: its median latency exceeds rows=128 in
all four same-order blocks. It is therefore the measured transition workload.

Saturation was not observed. The logical work rate continued to increase by
about 15.3% from rows=2048 to rows=4096. Rows=4096 is used below as the
largest-measured profiling proxy, not described as a saturation point or as a
measurement of physical memory bandwidth.

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

Representative AC-powered target-exit captures for rows=1, rows=512, and
rows=4096 are retained in compact form in
[`baseline-profiles.json`](baseline-profiles.json). Every capture contains the
fixed one-correctness, 1,000-warmup, and 5,000-profile sequence. Rows=4096 also
contains 22 earlier target-process compute commands from large-buffer setup;
the analyzer reports that prefix and segments the trailing fixed sequence.

| Workload | Setup compute commands | Instrumented median (ns) | Instrumented p95 (ns) | Reported spills |
| --- | ---: | ---: | ---: | ---: |
| r1-h896 | 0 | 6,208 | 8,208 | 0 |
| r512-h896 | 0 | 11,250 | 12,834 | 0 |
| r4096-h896 | 22 | 59,625 | 60,709 | 0 |

These intervals are diagnostic Instruments output, not ordinary benchmark
latency. All three traces selected a null counter set, disabled the shader
timeline, and exposed no named occupancy, bandwidth, or barrier-stall counter.
No sequence-scoped compiler spill event was reported; that bounded observation
does not prove that the kernel is universally spill-free.

## Interpretation

The clean timing baseline establishes where work begins to rise above fixed
dispatch cost, but the available profiler does not identify why. The
shared-memory tree remains a concrete optimization hypothesis, not a causal
bottleneck finding. Compiler output proves that the synchronization and storage
exist; the representative traces prove the workloads were captured but expose
no counter that attributes their cost. A paired variant comparison is therefore
the next valid empirical test.

## Nonclaims

This experiment will not by itself establish decoder-block, model-phase, or
end-to-end inference performance. Source-derived bytes will not be described as
observed hardware traffic, and timing alone will not be used as a causal
bottleneck claim.

## Decision

EXP-0001 is complete and the baseline implementation remains unchanged. Proceed
to one separately frozen reduction variant and paired comparison. Do not
introduce a saturation-specific or multi-path dispatch rule from this
characterization.
