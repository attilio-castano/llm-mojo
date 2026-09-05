# Adaptive decisions within the frozen budget

## 2026-09-05: advance grouped-head reuse at 64 splits

Source `1b03ddc` passed 65 Mojo tests and 98 Python tests. Every initial
candidate passed 24 generated cases plus the pinned Qwen decode fixture, with
two poisoned-buffer repetitions per case. No tolerance changed.

The four-block `noise`, `fused`, `split-baseline`, and `split-best-fused` runs
completed on Apple M4 Pro/Metal and AC, without thermal/performance warnings.
The largest baseline self-pair deviation was 18.47% at T=64/ring24; all
candidate comparisons use the per-workload self-pair noise floor.

At T=4096, g32-h1-s1 improved against the original by 91.1% hot and 97.2%
ring24. Comparing g1-h1-s64 directly with g32-h1-s1 improved 11.8% hot and
8.2% ring24, in all four blocks. Split candidates materially regressed shorter
ring24 workloads against g32. Four and sixteen splits lost to 64 at long T.

Therefore configurations 8,9,10 freeze g=1 and s=64, varying h=2,4,7 only.
Their primary ownership control is configuration 7; they also run paired
against the original. g32-h1-s1 remains the short-context control. This is
parameter selection for the next experiment, not a public dispatch rule.

Raw local run directories are under `/private/tmp/llm-mojo-gqa-exp0014`.
Compact retained evidence and final decisions follow after the campaign.

## 2026-09-05: final two configurations

The first grouped-head timing conclusion was invalidated below. Kernel
correctness passed; those timing runs cannot decide whether head reuse helps.

Receipt-bound profiles of source `f1f23af` at T=4096 measured instrumented
median intervals of 47.417us for split decode and 10.250us for its merge;
g32's single dispatch measured 70.083us. These are diagnostic GPU intervals,
not synchronized headline timings. Named device-wide counters in the split
profile showed a 58.7% instruction-throughput limiter and 53.0% integer/complex
limiter median. They motivate an instruction experiment without establishing
an exclusive causal bottleneck.

Configurations 11 and 12 freeze (g,h,s)=(32,1,1) and (1,1,64), respectively,
and change only the online-softmax update. Each score computes one exponential.
When the running maximum stays fixed, add the weighted V directly; rescale the
accumulated state only when the maximum increases. The branch is uniform within
each SIMD group. Compare each with its matching control (4 or 7) and the original.
This exhausts the 12-configuration budget. No further tuning follows.

Validation correction before these trials: the explicit cancellation fixture
now alternates V's sign across KV positions, rather than only across dimensions.
Regenerate its independent oracle and rerun every candidate. This strengthens
coverage; it changes neither benchmark inputs (kind=0) nor tolerance.

## 2026-09-05: invalidate misrouted head-reuse timings and repair harness

The first configuration-11 benchmark failed its workspace-shape gate before
timing. Investigation found that the harness's variant-7 branch used `>= 7`,
shadowing variants 8-12. Consequently `reuse-split-control` and `reuse-baseline`
actually timed variant 7 under incorrect labels. Invalidate both runs and the
earlier head-reuse rejection; retain their artifacts only to explain the error.
The baseline, configurations 1-7, and profiles of 0/4/7 were correctly routed.
The separate numerical suite called every kernel directly, so its passes remain
valid. `conditional-fused-control` contains no accepted timing samples.

Repair the branch to `== 7`. Each launch branch now returns its own constant ID;
the untimed benchmark gate verifies that ID against the request. An executable
Mojo test traverses all 13 routes and would catch the shadowing. Rerun head reuse
and both conditional candidates from the corrected clean commit. This is a
harness repair, not an additional candidate configuration. The budget remains 12.

## 2026-09-05: freeze finalists and finish fresh-input confirmation

Corrected source `6902c94` completed all `verified-*` head-reuse and conditional
trials. At T=4096, four-head reuse (9) improves against the one-head 64-split
control (7) by 13.8% hot and 17.8% ring24 in four of four blocks. Two-head reuse
has a ring24 gain but an inconclusive hot result; seven-head reuse improves
both modes with lower median gains than four heads in these trials. This
replaces the invalid head-reuse rejection, not the underlying numerical passes.

Conditional rescaling fails to improve either strong control consistently:
configuration 11 regresses ring24 at T=4096 by 11.7%; configuration 12 regresses
hot by 6.0%. Reject both instruction changes. All 12 configurations are now
tested, and the tuning budget is exhausted.

Finalists 4 and 9 were selected before the seed-101 confirmation commands.
Confirmation repeats training lengths and adds 7,32,128,512,2048,4095, with a
fresh baseline self-pair noise run. The local `finalists.json` is a retrospective
summary written after those commands; it is not a pre-run timestamped record.

Direct confirmation accepts finalist 9 over 4 at T=4095 and 4096 in both modes,
by approximately 24–25%. At T=2048, hot improves 15.3%, while ring24's 10.1%
median gain has only three of four faster blocks and remains inconclusive.
Short-context ring24 strongly favors the one-kernel finalist 4. Retain both
explicit experimental choices without an automatic shape selector.

Finalist profiles completed at T=16,256,4096 with 85 named device-wide counters
and verified 500/1000 profile dispatches. Baseline profiles have 1500 dispatches.
Their instrumented stage durations support diagnosis only; no causal DRAM or
full-decoder performance claim is inferred. Complete findings, retained samples,
provenance and reproduction steps are in [report.md](report.md).
