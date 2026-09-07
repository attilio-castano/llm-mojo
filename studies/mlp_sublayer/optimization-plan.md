# MLP optimization campaign

Approved scope: local implementation, commits, numerical validation, Metal
timing/profiling and study writeback. Begin from `ee3b99f` on
`codex/mlp-optimization-study`. The seven-stage baseline remains variant 0.
The metric is complete post-attention RMSNorm/MLP/residual latency. Existing
reference sources, dependency locks, arithmetic, input domain and budgets stay
fixed. There is no decoder, KV-cache or model-generation change in this campaign.

## Frozen comparison budget

1. Screen existing bias-free rowwise, MMA 8x16, 16x16 and 8x32 mappings for gate
   and down independently at R=1,16,17,1024, hot and ring24. Gate/up have K=896,
   N=4864; down has K=4864,N=896. Both use BF16 operands/output and FP32
   accumulators. These are eight shape/mode cells per projection, four
   comparisons including the control self-pair: 5,120 observations total.
2. Advance at most one tiled mapping per projection geometry. Qualification
   requires an isolated R=1024 gain in both modes. Among qualifying mappings,
   minimize the worse of the two paired median ratios; ties prefer 8x16, then
   16x16, then 8x32. This selects a finalist, not proof that it beats every
   other tile. Preserve all short-row regressions and keep mapping selection
   explicit; no inferred universal crossover becomes a dispatcher default.
3. Confirm the gate finalist on up's distinct weights. Measure gate/up added
   to the original MLP, then down added with gate/up fixed, at
   R=1,16,17,1024 in both modes. Each comparison includes its matching control
   self-pair (1,280 observations per comparison). A projection with no finalist
   retains rowwise. Measure the combined implementation directly.
4. Profile the resulting projection configuration at R=1 and R=1024, using
   respectively 500 and 25 measured iterations after ten warmups. Consider at
   most two further configurations: fused SiLU/multiply on separate G/U, then
   packed gate/up with that fused consumer. Preserve the BF16 A result before
   multiplication and the BF16 D result before residual. Packing must include
   the real multi-row consumer layout, without silently adding an unmeasured
   unpack. Each admitted follow-up gets a bounded R=1,17,1024 hot/ring24
   comparison against its immediate control and, for the combined result,
   the projection-only control. No wider tile, split-K or staging search.
   A follow-up may be skipped when the updated profile supplies no useful
   remaining-cost hypothesis; record that decision before coding it.
5. Freeze the final implementation and evaluate the fresh holdouts once before
   its final campaign. Compare original versus final complete MLP on the full
   R=1,7,15,16,17,33,65,257,1024,4096 matrix, hot/ring24, with matching control
   self-pairs (3,200 observations). Capture both implementations at
   R=1,17,1024,4096 with 500,100,25,10 measured iterations respectively.
   Dispatch identities/counts must describe the actual variant; joining
   preempted intervals must precede stage assignment.

Every latency comparison uses four paired blocks, ten warmups and ten samples
per arm; blocks two and three reverse workload and arm order. Both arms use the
same frozen seed-1601 input prefixes and source build. Hot includes one enqueue
through completion; ring24 uses 24 distinct weight/input copies with shared
workspace and one synchronization per sweep. Isolated projection ring24 shares
one exact upstream operand across distinct weight copies. Allocation, uploads,
checks and printing remain outside timing. Run GPU work sequentially.

A gain requires all four ratios below one and a median reduction exceeding
both 5% and the maximum matching self-pair deviation. Regressions use the
symmetric rule. Retain noisy/incomplete/failed runs; do not selectively repeat
cells or change the rule after observing results. Estimate no DRAM bandwidth
from source-requested bytes. Profiles are diagnostics, separate from latency.

## Correctness and ownership

Check every candidate against exact upstream operation inputs, then original
X through composed D/Y under their separate frozen gates. Keep all intermediate
diagnostics, full/chunk comparisons, ragged tiny dimensions, poisoned outputs,
inactive guards, invalid-call preflight and asynchronous workspace reuse.
Each measured route performs its own numerical checks before timing.

The existing 53 cases are regressions. New final holdouts use seeds 3037 and
3041 at R=1,17,4096 plus one reserved checkpoint prompt, under the same pinned
CPU reference and layer-0 FP32 attention policy. The declaration and token IDs
are in `tests/fixtures/mlp_optimization_holdout.json`. Tokenization precedes any
candidate output; held-out model arrays are captured only against the frozen
final binary. Existing holdouts and manifests are never overwritten.

Primitive checks retain every finite BF16 SiLU input, exact multiply/residual
requirements, subnormal and signed-zero handling, and the regressions that
reject omission of the A/D rounding boundaries. Enqueue allocates and
synchronizes nothing; the caller owns buffers and consumes outputs before
workspace reuse. New packing/fusion must preserve these responsibilities.

Ordinary implementation defects may be repaired under the fixed contract.
Failing or inconclusive candidates remain unselected. A necessary arithmetic,
domain or tolerance change requires a separate decision; preserve the failure
and finish independent work. A failed holdout remains recorded as failed.

## Completion

Run required validation before freezing measurement source; run the final full
repository workflow and normal-mode checkpoint/reuse checks for the final
implementation. Retain compact numerical records, complete latency samples,
validated profile samples and source/binary/fixture/environment identities.
Update the existing study with ownership, bytes, gains, regressions and limits,
then commit locally. A negative or inconclusive result also closes the bounded
campaign. Pushes, PR publication and new asset downloads are separate actions.

## Progress

- Baseline inspected; branch created from `ee3b99f`.
- Fresh recipes and reserved checkpoint tokens declared before candidate output.
- Projection implementation and validation in progress.
