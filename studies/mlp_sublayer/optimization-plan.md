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
- Seven projection configurations passed 35,294 stage/reuse checks over the
  53 existing cases, plus thirteen primitive-check records. Full validation:
  102 Mojo, 62 tooling and eleven pinned reference tests; all route smokes.
- The first screen stopped when the strict reader rejected reversed
  candidate/control IDs in its header. The incomplete attempt remains in
  `data/optimization_screen_attempt.json`. The header repair changes no
  arithmetic; expanded smoke checks validate real non-self outputs in both
  arm orders. Both complete screens restart from the repaired clean source.
- The corrected screens completed at `a8b62cd`: 5,120 observations. Gate chose
  variant 2 and down chose variant 5, both 16x16. At R=1024 the gate paired
  ratios are 0.0811 hot / 0.0776 ring24; down 0.1116 / 0.1068. All qualify
  under the fixed rule. Gate R=1 ring24 regresses by 45.8%; hot is inconclusive.
  Full selection and all observations are retained in `data/optimization_*`.
- Variant 7 composes these choices explicitly, with the same seven stores and
  dispatches. The up confirmation, gate/up whole-block comparison and down
  increment each contain 1,280 observations. A change advances into the
  projection configuration only if its R=1024 comparisons are faster in both
  modes; otherwise its previous control remains. This decision is fixed
  before any of these follow-up timings. Small-row outcomes remain reported.
- The combined configuration passed the 43 synthetic cases and asynchronous
  reuse with the fixed gates. The full workflow passed 102 Mojo, 63 tooling
  and eleven pinned-reference tests, primitive sweeps and all route smokes.
  Normal-mode checkpoint and observed-holdout checks precede follow-up timing.
- All eight configurations passed the 53 existing cases; checkpoint and
  observed-holdout regressions also ran in normal mode. There are 40,336
  stage/reuse checks and thirteen primitive
  records. The complete 3,840-observation follow-up campaign at `2760400`
  confirmed up (R=1024 ratios 0.0811/0.0776), whole gate/up (0.3312/0.3304),
  and down added to that block (0.2825/0.2806). All advance under the frozen
  rule. Variant 7 is the selected projection configuration; the two whole
  comparison ratios are not multiplied to construct a final gain.
- Two validated profiles retain 3,675 measured dispatches. At R=1024,
  projections occupy 91.44% of active GPU time; SiLU/multiply 7.70%. At R=1,
  projections occupy 94.72%, down alone 59.61%, and SiLU/multiply 1.78%.
  `data/optimization_followup_decision.json` records the decision before any
  optional kernel implementation: skip fusion and packing in this campaign.
  The traffic-scaled fusion estimate is about 3.08% of total active GPU time,
  below the 5% decision floor, and is explicitly not a measured gain or bound.
  Packing the existing tile preserves matrix work and source request counts;
  these captures give no strong launch-cost hypothesis. This is a bounded
  allocation of experiments, not proof that either technique never helps.
- The final comparison is now declared as `mlp_final`: original 0 versus 7,
  all ten row counts and both modes, 3,200 observations with self-pairs. The
  candidate is fixed before fresh holdout output access. Final validation,
  fresh holdouts, the full latency campaign and eight final profiles remain.
