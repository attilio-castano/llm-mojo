The active revision is [Fast implementation with numerical diagnosis](fast-generation-plan.md).
The qualification stop gates below describe historical plans.

# Qwen forward and generation milestone

## Approved Fast completion revision — 2026-09-10

The active [Fast completion plan](fast-generation-plan.md) consolidates existing
kernel selections, independently qualifies a new full-model numerical contract,
then completes native generation acceptance and model-level selection. It does
not require native schedule identity. The consistency investigation and its
failed gates below remain historical evidence. Qualification failure stops
dependent native acceptance; no historical failure is retroactively waived.

## Approved consistency revision — 2026-09-09

The user approved autonomous implementation of schedule consistency after the
HF/ATen diagnosis. This revision replaces the old cross-schedule tolerance
prerequisite for new acceptance; it does not reinterpret the old failed result.
The executable declaration is `tests/fixtures/model_consistency.json`.

1. Qualify canonical upstream execution through capacity 4096. Stream growing
   cache comparisons rather than retaining every historical snapshot. Full,
   repeated full, tokenwise short cases and ragged/mixed chunk schedules must
   agree in all 75 recorded boundaries. Logits are computed at call endpoints
   with a one-row upstream head; final norm is checked for every processed row.
2. Establish one schedule-consistent Mojo arithmetic path, retaining BF16
   storage and FP32 reductions. Audit normalization, all linear projections,
   attention, elementwise boundaries and the LM head. Preserve query parallelism
   inside GPU launches. Validate operations, a layer, the full model and greedy
   generation progressively, with exact native schedule comparisons and separate
   cross-engine numerical checks.
3. Optimize only candidates that preserve the stored baseline results exactly.
   The existing budget remains at most three studies with two challengers and
   one independent confirmation each. Require complete-model timing evidence
   before automatic selection; old mapping IDs are historical candidates.

The historical frozen budgets are copied unchanged into the new declaration
as cross-engine hypotheses before observing native outputs. Their reuse does
not imply that the historical confirmation passed, and they cannot be enlarged
after candidate exposure. Existing operation-level accuracy gates also remain.
All original reserved inputs remain untouched until final acceptance. A required
precision/tolerance change or reserved failure stops dependent work. Exhausting
the optimization budget retains the consistent baseline; a speedup is optional.
The deliverable is a checked consistent generator, receipts, compact evidence,
reproducible measurements and local commits. No push, publication or agents.

Execution checkpoint: canonical HF qualification passes 71,250 exact checks
through 4096 tokens. Native configuration 20 passes primitive and one-layer
accuracy/schedule tests, including 13,165 retained decoder records. Its first
full-model development input fails seven of 75 numerical gates; all 48 cache
storage checks pass. All 336 subsequent identical-operand operation checks
pass. Ten projection elements differ by one BF16 step; exact rational sums
favor Mojo in five and HF in five. No local gate defect was identified on this
input. Promotion is paused at the numerical-policy decision, with no enlarged
gate, new precision, optimization study or final reserved exposure. The
[consistency study](../studies/model_generation/consistency.md) contains the
evidence and reproduction commands. Stage 2 full-model schedule/generation
acceptance and stage 3 remain incomplete.

Final validation passes the frozen fixture anchors, 119 Python tests, every
native regression file and all Metal benchmark smoke routes. The stale invalid
GQA mapping test was corrected and rerun before completing the remaining suite.
Retained evidence verification and table regeneration also pass.

Approved for autonomous local execution on 2026-09-09. Starting checkout:
`478cdc1`, updated to merged tokenizer head `d9aaad5` before implementation;
branch `codex/qwen-generation-engine`. Local implementation,
integration of `codex/qwen-tokenizer`, pinned artifact downloads, fixture
generation, sequential Metal benchmarks/profiles, documentation and local
commits are authorized. No push or PR publication. Work inline without agents.

## Scope and gates

Build native Mojo batch-one BF16 Qwen2.5-0.5B-Instruct inference: verified
prepared weights, embeddings, all 24 decoder layers, final norm, tied LM head,
cached greedy generation and streaming text decoding. Capacity is at most 4096
prompt plus generated tokens. Python is preparation/oracle tooling only.
Chat rendering, multi-turn sessions, sampling, quantization and batching remain
outside this milestone; this does not close the complete multi-turn V0 contract.

1. Integrate and validate the existing native tokenizer.
2. Qualify an independently executed pinned full-model reference; freeze
   intermediate/logit gates, development fixtures, reserved inputs, and greedy
   semantics before comparing candidate outputs. Preserve existing component
   arithmetic. No tolerance can be fitted to Mojo results.
3. Build fixed-ID-0 model forward with explicit persistent weights and per-layer
   caches, shared nonaliasing scratch and ordered submission. Validate every
   block boundary and final logits. Preflight all layers before dispatch.
4. Add baseline/auto/explicit configuration policies, full and explicit chunked
   prefill, cached one-token decode and streaming output. Validate mixed routes,
   cache preservation, resets, failure invalidation and capacity accounting.
5. Measure actual model loading, prefill, cached suffix and growing decode.
   Existing layer lookup entries are candidates until model-level confirmation.
   Unknown cells use ID 0. No guessed crossover or claimed chunking speedup.
6. At most three further bottleneck studies, each with at most two challengers
   and one independent confirmation. Freeze mechanism, comparison matrix and
   numerical acceptance before execution. Use the paired four-block procedure
   in experiments.md and retain all valid observations. No winner is required.
7. Final reserved acceptance launches the exact receipted executable; retain
   compact results, raw samples and reproduction commands. Run documented
   validation and commit validated changes throughout.

## Ownership and evidence

Weights and KV caches persist per layer. Shared temporary storage must preserve
the decoder's input/output nonaliasing rules and all asynchronous consumers.
Normal next-token inference needs the final row's logits, while diagnostic
execution captures selected earlier rows and every layer boundary. Device
failure invalidates the execution; no partial cache rollback is promised.

Cache length counts consumed/enqueued tokens. A newly selected terminal token
can be in output history without being cached. Define ties, nonfinite logits,
empty prompts, stop IDs and output limits explicitly before implementation.
Greedy reference equality applies to adequate-margin cases; logits remain
authoritative for numerically ambiguous argmax.

Qualify numerical thresholds using reference-only evidence before candidate
comparison. Bind artifacts, declarations, source and launched binaries in
acceptance records. Reserved failures stop dependent promotion; do not widen
tolerances, tune on the failure or silently spend additional reserved cases.
Unavailable required hardware or necessary numerical-contract changes require
returning with evidence. Routine development failures are fixed locally.

Arrays, weights, binaries and traces stay outside Git. Extend existing tooling;
retain a compact model study only when real execution produces evidence.

## Progress

- Branch attached and fast-forwarded to `d9aaad5` (tokenizer PR #18).
- The temporary tokenizer merge was aborted in favor of the identical merged
  tree. Its validation was interrupted for the update; baseline validation
  restarts on the merged head.

### Development checkpoint

- Existing complete validation passed on the updated base, including all Metal
  benchmark smoke routes. The expanded Python suite passes 116 tests.
- Two new native primitive tests pass on M4 Pro / Metal: repeated-ID embedding
  gather with copy guards, and exact BF16 binary I/O (including signed zero and
  subnormal bits). Model and generation call graphs compile.
- Tiny synthetic upstream capture self-test passes 225 boundaries. Initial
  pinned-checkpoint schedule qualification failed; model comparisons, greedy acceptance,
  reserved acceptance, performance measurements and promotion remain pending.
- The full checkpoint SHA-256 matches the pinned artifact. Preparation completed
  with 196 BF16 tensors. Redundant download fragments remain outside Git.
- Reference-only diagnosis identified SDPA execution shape as the source of the
  observed full-versus-cached divergence on the 17-token diagnostic: rowwise
  linear/norm execution did not change it; rowwise SDPA removed it. This is an
  ablation result, not authorization to change the reference arithmetic.
- Before any Mojo model comparison, a bounded reference calibration is declared
  in `tests/fixtures/model_calibration.json`. It derives per-boundary pointwise
  and per-token relative-RMS budgets from five development lengths, with a
  fixed 1.5 safety margin and a 6.25% relative-RMS ceiling. Five independent
  lengths/seed combinations then confirm frozen budgets. Failed initial
  thresholds remain in the contract and evidence; failure of this bounded
  calibration/confirmation stops dependent acceptance.
- The declared calibration passed its fixed ceiling. Independent confirmation
  failed 8 of 2,025 checks (hidden 22–24 and final norm at length 15), while all
  row relative-RMS checks passed. All declared cases, including 4096, completed.
  The budgets are frozen and dependent acceptance is stopped. No native model
  outputs or final reserved outputs were observed; optimization budget is unspent.
- The committed-source SDPA ablation reproduces the 17-token diagnosis across
  all 75 boundaries. Compact raw records, source hashes and regeneration commands
  are retained in `studies/model_generation/`. A new numerical-policy decision
  and explicit authorization are needed before dependent work resumes.
- Final tooling validation passes all 116 Python tests, the two calibration
  metric self-tests, and lossless evidence verification/table regeneration.
  The complete existing native validation and two new Metal primitive checks
  passed earlier; native source has not changed since that checkpoint. No
  additional performance study or full-model acceptance is claimed.

### Authorized reference diagnosis follow-up

The user authorized a focused reference-only investigation after reviewing the
failed gate. Trace the first SDPA difference before/after BF16 rounding on the
already observed 17-token and 15-token cases; follow its propagation and final
RMSNorm scaling; assess logits, top-token margins and bounded greedy sequences.
Use three declared text inputs and eight output steps per input. Compare full
recomputation with cached execution on identical histories, and report any
free-running divergence separately. Keep all existing gates and reserved inputs
unchanged. This investigation does not authorize numerical-policy promotion or
resumption of the native-model optimization campaign.

The follow-up is complete. Identical layer-1 Q/K/V operands produce FP32 SDPA
differences up to 1.889e-6; some cross BF16 midpoints. Full/cached observation is
bitwise equal to unobserved execution. Propagation and the learned final-norm
scale explain the recorded pointwise violation (11.375 versus 7.25 at the worst
coordinate, distinct from the largest absolute-error coordinate). All 66
next-token comparisons agree; 48 satisfy the sufficient margin bound. All six
cached trajectories on the three declared text prompts match full recomputation
through stop or the eight-token budget. The report, complete compact evidence,
regenerated tables and figure are in `studies/model_generation/rounding.md`.
Existing gates and reserved cases remain unchanged; revising acceptance is the
next decision, and no native model execution or optimization followed this study.

### Authorized HF/PyTorch implementation follow-up

The user requested deeper localization and more reliable reference results.
The declared ATen investigation observes actual scaled-Q/K multiplication,
scores, softmax and weighted values beneath HF's SDPA call. Upstream source is
matched to the installed HF file and exact PyTorch build revision. Identical
matrix operands are probed across the CPU bmm dispatch threshold. Deterministic
mode is checked separately from a diagnostic that fixes SDPA query/prefix shape
and layout. The latter is tested through the actual model at lengths 15, 17,
65, 129 and 257. This is bounded reference diagnosis, with unchanged thresholds,
reserved inputs and dtypes; no new numerical policy is automatically promoted.

This follow-up is complete. Actual ATen observation places the first difference
in QK multiplication with identical scaled inputs. Duplicate-query probes show
both a CPU dispatch threshold effect and additional dimension/row-position
effects within the larger path. Fixed-score softmax has smaller extent effects;
fixed-probability PV is exact in these probes. Deterministic mode changes none
of the observed operations. The canonical contiguous-query route passes all
36,225 byte comparisons over five lengths, with exact full-call repeats. For
the two localized inputs, its cached boundary hashes also match the original
cached route. The report and reproducible evidence are in
`studies/model_generation/backend.md`. This establishes a diagnostic control,
not a new acceptance policy or native-model result.
