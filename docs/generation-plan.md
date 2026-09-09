# Qwen forward and generation milestone

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
