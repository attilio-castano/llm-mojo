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

All three grouped-head reuse candidates passed correctness but failed to
establish a consistent >=5% gain over g1-h1-s64 at T=4096. Ring24 medians were
essentially unchanged. Keep h=1 and reject head reuse as a performance change.

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
