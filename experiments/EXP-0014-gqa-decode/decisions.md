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
