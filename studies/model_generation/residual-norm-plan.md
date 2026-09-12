# Residual addition plus RMSNorm, independently and composed

Control is current Fast: configuration 26, BF16 boundaries, CPU greedy,
inter-layer copies. New single-row Metal kernel stores the exact BF16 residual
sum, retains its seven per-thread values, and applies the existing 128-thread
SIMD-group RMSNorm reduction and BF16 normalization/scale boundaries.
Fuse all 48 residual/norm pairs. Preserve distinct residual and normalized
storage, ordered-stream lifetime, cache invariants and multi-row behavior.

Four arms: control; residual/norm fusion alone; buffer swapping plus separate
GPU argmax; all three. No fused vocabulary projection. Fixed-token comparisons
are control self-pairs, each of the three candidates against control, all-three
against the previous-two combination, and all-three against norm alone.
Four paired blocks at histories 64/1024/3968, ten warmups and ten samples per
arm: 1,440 retained samples. Reverse arm and workload order in blocks 2 and 3.
The independent norm comparison precedes the composed comparisons in the first
block. No tuning or replacement run after acceptance samples.

First validate the kernel against separate residual and RMSNorm operations:
ordinary nonuniform data, cancellation, signed zero, subnormal/rounding edges,
poisoned outputs, protected tails, unchanged inputs and overlap/shape rejection.
Then compare complete logits and caches at all three histories for each arm,
all layer captures at history 64, and multi-row/single-row/reset/rejection
lifecycle sequences. Full repository validation precedes a clean measured
source commit. Compile once and verify hardware/backend, binary/source/assets,
power and thermal conditions. Measure token upload through greedy completion;
exclude preparation, rewind, allocation and recording.

Capture four separate Metal traces at history 1024, ten warmups and eight steps
each. Expected compute commands: 314 control, 266 norm-only, 293 previous-two,
245 all-three, each plus four mapping blits. Four streaming blocks across all
four arms and three fixed 128-token replies: 48 replies, all outputs/history
must agree exactly. Retain every sample, trace command and provenance field.
Reuse the existing model profiling runner and evidence contract.

A candidate qualifies against control only if all four ratios are below one
and median reduction exceeds max(5%, largest absolute control self-pair
deviation), at all three histories. Prefer all-three if it qualifies and all
its direct paired ratios against each other qualifying candidate are below one.
Otherwise retain the sole qualifier, or report the unresolved choice and keep
Fast unchanged. Promotion is restricted to single-row Apple M4 Pro / Metal.
Do not add measured gains from independent experiments. Any promoted route
must pass public policy/lifecycle checks and preserve historical explicit arms.
