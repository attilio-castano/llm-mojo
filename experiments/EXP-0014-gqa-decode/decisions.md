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
