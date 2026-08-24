# EXP-0003: Apple GPU RMSNorm limiter-counter comparison

Status: **complete**

Evidence level: **operation**

## Question

Do matched Performance Limiters traces show a repeatable device-counter
difference between the shared-tree and SIMD-group RMSNorm implementations at
rows=512 that can refine, but not replace, the paired timing result?

## Frozen scope

EXP-0002 already established a 5.38% paired ordinary-timing improvement at
`r512-h896` and selected the SIMD-group implementation. This follow-up does not
reopen that production decision. It tests whether named Apple GPU counters show
a repeatable difference consistent with the structural change from 128 shared
FP32 partials and nine workgroup barriers to four shared partials and two
barriers.

The counter set does not expose a named workgroup-barrier stall counter. A
change in occupancy, compute launch, instruction, L1, or bandwidth counters may
identify correlated pressure, but cannot by itself prove that barrier removal
caused the timing improvement.

## Frozen capture protocol

- Hardware and backend: Apple M4 Pro through Metal, on AC power with no reported
  thermal or performance warning.
- Workload: BF16 `r512-h896`, one row per 128-thread threadgroup.
- Sequence: one correctness dispatch, 100 warmup dispatches, then 500 profile
  dispatches followed by device synchronization, `PROFILE_REGION_END`, and a
  250 ms host-only idle with no later GPU work.
- Implementations: retained shared-tree baseline and public SIMD-group default,
  built from the same clean commit with provenance.
- Launch location: an unprotected scratch directory outside `Desktop`,
  `Documents`, and `Downloads`; retained artifacts may be copied after capture.
- Instruments template: `LLM_Mojo_Metal_Limiters`, Counter Set
  `Performance Limiters`, Performance State `Default`, Shader Timeline off.
- Capture order: baseline, SIMD-group, SIMD-group, baseline. Each target must
  exit normally under a 10-second guard.
- Capture identity: the helper generates an opaque ID, uses it for the
  Instruments run and ephemeral executable, verifies runtime identity against
  binary provenance, and requires the analyzer to match that ID in the TOC and
  command-buffer submissions. The comparator derives roles from verified
  implementation identity rather than filenames or argument positions.
- Raw traces and exports remain outside Git; only scrubbed summaries, checksums,
  compact comparisons, and bounded findings may be retained here.

Every trace must contain the exact declared target sequence, a named counter
set, counter samples overlapping the enclosing 500-dispatch profile window, at
least 10 distinct counter timestamps in that window, and a counter-sample span
covering at least 80% of the profile window. A failed gate excludes that capture
rather than weakening the rule.

## Calibration amendment

The first baseline calibration was excluded because its device-counter stream
ended 20,147,138 ns before the declared profile window. The target had exited
immediately after its final synchronization, so shortening the warmup alone did
not retain the counter tail.

A second calibration target launched from `Documents` remained in system
interface initialization until the 10-second guard killed it. That trace had
one main thread, zero CPU samples, and zero Metal submissions. The identical
binary (same SHA-256) completed normally under the same template when launched
from `/private/tmp`, isolating the failure to the protected launch location
rather than the RMSNorm workload.

The accepted calibration therefore freezes both corrections above. With the
250 ms post-region idle, the analyzer found 881 distinct counter timestamps
inside the 500-dispatch profile window and a 99.84% sample-span fraction. These
calibrations establish the collection method only and are not part of the
four-capture comparison.

## Identity amendment and revalidation

The original `EXP-0003-RUN-002` summaries predated capture receipts. Their
filenames and positional comparison order described baseline versus variant,
but the analyzer could not prove which provenance-bearing binary produced each
trace. Those observations remain historical and are not retrospectively given
stronger identity.

`EXP-0003-RUN-003` repeated the frozen ABBA protocol through the receipt-bound
path. Each summary now carries a unique capture receipt, binary and provenance
hash, clean commit, runtime device/backend, exact workload, and an ID verified
against the Instruments run and launched target process. The comparator
rejected caller-assigned roles and derived all four roles from the verified
implementation and entrypoint. The in-repository comparison artifact now comes
from this revalidation run.

## Frozen comparison

For each capture, summarize the named counter samples whose timestamps fall
inside the enclosing target profile window. The values are device-wide and are
not command-buffer-exclusive. Retain the observed target GPU busy fraction and
GPU performance-state intervals to expose contamination or frequency drift.

Pair capture 1 with capture 2 and capture 4 with capture 3, giving one
baseline-then-variant and one variant-then-baseline comparison. For counters
with positive medians in all four captures, retain both within-pair
variant/baseline ratios and their median. A counter difference is called
repeatable only when both pairs move in the same direction and the median
relative change is at least 5%. Zero-valued or directionally inconsistent
counters remain descriptive only.

The primary counter families are:

- Kernel Occupancy and Compute SIMD Groups Inflight;
- Compute Shader Launch and Instruction Throughput limiters;
- ALU, F32, and F16 limiter/utilization counters;
- L1 cache limiter/utilization and threadgroup-memory L1 bandwidth;
- GPU and last-level-cache bandwidth and limiter counters.

Instrumented command durations remain diagnostic and are not substituted for
the ordinary EXP-0002 benchmark.

## Result

`EXP-0003-RUN-003` completed the frozen baseline, SIMD-group, SIMD-group,
baseline order from clean commit `8a0eeeb`. Every target exited normally on AC
power, with no thermal or performance warning before or after its capture. All
four summaries contain the exact 1/100/500 compute sequence, all 85 named
counters, unique receipt-bound identities, no reported target spill event, and
valid profile-window coverage:

| Capture | Role | Counter timestamps | Sample span | Target GPU busy |
| --- | --- | ---: | ---: | ---: |
| 01 | Baseline | 913 | 99.80% | 98.53% |
| 02 | SIMD-group | 727 | 99.95% | 74.98% |
| 03 | SIMD-group | 603 | 99.86% | 95.96% |
| 04 | Baseline | 766 | 99.71% | 89.01% |

Of the 85 counters, 58 had positive medians in every capture and 41 passed the
frozen same-direction and 5% materiality rule. Many are correlated or derived,
and the target-busy fraction varied across traces, so the bounded interpretation
uses the prespecified structural signals rather than the total count:

| Named device-wide counter | Profiler unit | Pair 1 V/B | Pair 2 V/B | Median change |
| --- | --- | ---: | ---: | ---: |
| Threadgroup Memory L1 Write Bandwidth | GiB/s | 0.0409 | 0.0420 | -95.86% |
| Threadgroup Memory L1 Read Bandwidth | GiB/s | 0.0480 | 0.0494 | -95.13% |
| L1 Threadgroup Residency | percent | 0.0870 | 0.0880 | -91.25% |
| L1 Write Bandwidth | GiB/s | 0.6869 | 0.6951 | -30.90% |
| F32 Utilization | percent | 1.3267 | 1.3193 | +32.30% |
| F32 Limiter | percent | 1.2800 | 1.2675 | +27.38% |
| ALU Utilization | percent | 1.1471 | 1.1387 | +14.29% |
| Kernel Occupancy | percent | 0.8634 | 0.8797 | -12.84% |

The roughly 95% reduction in the sampled Threadgroup Memory L1 Read and Write
Bandwidth counters, whose profiler unit is GiB/s, is consistent with replacing
128 shared FP32 reduction partials with four. These remain device-wide samples
inside the enclosing target profile window, not command-buffer-exclusive
measurements. The lower threadgroup residency and total L1 write bandwidth
reinforce the bounded mechanism: the SIMD-group kernel puts materially less
pressure on the threadgroup-memory path. The higher F32 and ALU utilization
show a repeatable shift in the sampled execution mix, but are not a standalone
throughput claim.

Global GPU and last-level-cache bandwidth moved in one direction in RUN-003,
but were directionally inconsistent in RUN-002. The samples are device-wide,
the target-busy fraction varied, and neither run isolates unified-memory
traffic. F16 limiter and utilization medians were zero. Therefore this
experiment still makes no global or unified-memory traffic claim. It also does
not use the RUN-003 instrumented duration medians, which ranged from 43.23 to
48.71 microseconds across captures under the default minimum GPU performance
state.

The exact scrubbed capture gates, input hashes, medians, pair ratios, and all 85
classifications are retained in [`counter-comparison.json`](counter-comparison.json).

## Bounded conclusion

The valid matched comparison supports a repeatable reduction in device-wide
threadgroup-memory pressure inside the target profile window. It is consistent
with the SIMD-group reduction structure and refines the EXP-0002 timing result;
it does not isolate barrier stalls or reopen the production decision. No direct
physical-traffic, energy, or end-to-end inference claim follows.
