# EXP-0002: Apple GPU RMSNorm SIMD-group reduction comparison

Status: **complete**

Evidence level: **operation**

## Question

Does one hybrid SIMD-group reduction materially reduce synchronized Apple GPU
RMSNorm dispatch latency versus the shared-memory tree without a relevant
workload regression?

## Frozen hypothesis

The baseline uses 128 FP32 shared partials and nine workgroup barriers. The
variant will retain the same 128-thread mapping and per-thread FP32 accumulation
while reducing within four 32-lane SIMD groups, storing four shared partials,
and using the first SIMD group for the cross-group sum. The expected structure
is four FP32 shared values and two workgroup barriers.

This is one explicit alternative, not a general reduction framework. Generated
code will confirm structure; only ordinary paired timing can establish whether
the variant is useful.

## Frozen comparison

All seven EXP-0001 row counts remain in scope. Four blocks pair the
implementations per workload in ABBA implementation order while also reversing
the workload order. Each implementation retains 40 samples per workload.

For every block and workload, divide the variant median by the baseline median.
A material improvement requires a median paired ratio at or below 0.95 and a
faster variant in at least three blocks. A material regression uses the
symmetric 1.05 threshold and direction rule. Smaller effects are inconclusive.

## Results

The SIMD-group candidate passed the existing small and Qwen-width BF16 oracle
fixtures at implementation commit `e947397`. Clean generated code from
measurement commit `476fc29` contains one four-FP32 shared array, two
workgroup-barrier calls, and ten SIMD shuffle calls: five for each of the two
32-lane reductions. This validates the intended structure but does not prove
which structural change causes a timing effect.

`EXP-0002-RUN-001` then collected all four frozen paired blocks on AC power at
clean commit `476fc29`. Every block identified an Apple M4 Pro through Metal,
the repository commit remained fixed, and no thermal or performance warning
was reported at any gate. The retained record contains 560 unique valid
samples: 10 repetitions for each implementation, workload, and block, with
1,000 iterations per sample.

| Workload | Baseline median (ms) | Variant median (ms) | Paired change | Faster blocks | Classification |
| --- | ---: | ---: | ---: | ---: | --- |
| r1-h896 | 0.009893 | 0.009695 | -4.92% | 4/4 | Inconclusive |
| r4-h896 | 0.009584 | 0.009212 | -3.42% | 3/4 | Inconclusive |
| r16-h896 | 0.009582 | 0.009582 | +1.98% | 1/4 | Inconclusive |
| r128-h896 | 0.009612 | 0.009417 | -2.01% | 3/4 | Inconclusive |
| r512-h896 | 0.012808 | 0.012192 | -5.38% | 4/4 | Material improvement |
| r2048-h896 | 0.036056 | 0.034458 | -4.53% | 4/4 | Inconclusive |
| r4096-h896 | 0.063883 | 0.062695 | -2.04% | 4/4 | Inconclusive |

The paired percentage is computed from the median of four within-block
variant/baseline median ratios, not from the two overall medians shown for
orientation. Rows=512 crosses the frozen 5% threshold and favors the variant
in every block. No relevant workload meets the symmetric regression rule.
Rows=1 and rows=2048 favor the variant consistently but remain inside the
predeclared inconclusive band.

The material rows=512 result permitted one diagnostic variant profile. A clean
500-dispatch profile binary completed normally in 0.30 seconds and printed the
expected Apple M4 Pro/Metal identity outside Instruments. Two `xctrace`
attempts instead held the target suspended before its first instruction. The
auditable retry ended after 73.996850 seconds with `SIGKILL`, “User pressed
Stop,” and zero Metal command-buffer submissions. It is excluded, so no
variant trace, counter, spill, occupancy, bandwidth, or stall claim exists.
The first excluded bundle allocated about 5.7 GiB while collecting no valid
target sequence and remains outside the repository.

Compact values, all four ratios, validation gates, and external-artifact
checksums are retained in [`comparison-run.json`](comparison-run.json). The
profiler boundary is retained separately in
[`profile-attempts.json`](profile-attempts.json).

## Interpretation

The useful result is regime-specific without requiring regime-specific code.
The measured transition workload gains 5.38%, while no measured production
proxy materially regresses. The opposite effects needed to justify a row-based
dispatch threshold never appear. One simpler default therefore follows the
frozen rule; a multi-path selector does not.

Timing alone does not establish that fewer barriers caused the gain. It shows
that the explicitly declared structural candidate is better under the frozen
operation benchmark. Hardware mechanism remains unresolved because the Metal
trace path failed before target launch and previously available traces expose
no named counters.

## Nonclaims

- This is not an end-to-end inference, decoder-block, model-phase,
  time-to-first-token, or time-per-output-token result.
- Source-derived byte rates are not observed cache, fabric, or DRAM traffic.
- No occupancy, hardware bandwidth, barrier-stall, or register-pressure
  measurement is available.
- The result applies to BF16 RMSNorm with hidden size 896 on the measured Apple
  M4 Pro; it does not establish an optimum for other hidden sizes or GPUs.

## Decision

The frozen rule selects the SIMD-group implementation: rows=512 is a relevant
material improvement and no relevant workload is a material regression.
Promote it as the single public Apple GPU RMSNorm default, retain the
shared-tree implementation only as an explicit benchmark baseline, and add no
runtime dispatch rule. Commit `9bf73f6` implements that decision:
`enqueue_rms_norm_apple_gpu` now selects the SIMD-group kernel, and
`enqueue_rms_norm_apple_gpu_shared_tree` names the retained comparison path.
