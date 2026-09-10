# Fast full-model implementation

## Active revision: implementation with numerical diagnosis

Approved on 2026-09-10 after the failed reference qualification. Complete the
native Fast runtime using the existing kernels. The historical numerical
ceilings below remain recorded but no longer block integration or measurement.

Required checks establish pinned assets, architecture, causal positions, exact
cache preservation/appends/guards, submission accounting, invalid-input handling,
reset, greedy ties/nonfinite rejection, stop/context limits and UTF-8 streaming.
Existing independent operation tests remain required. HF intermediate errors,
same-history logits/distributions and independent trajectories are diagnostic
observations, not automatic pass/fail criteria. Investigate suspicious differences
on identical operands; retain discrepancies and their scope without retuning
historical limits. No general quality or exact-HF-equivalence claim is implied.

Finish the existing public generator, expose candidates 0/2/3/21, exercise the
real 24-layer model and capture all 75 boundaries on declared development
schedules. Measure the complete model with allocation and diagnostic capture
outside timing. Only measured workload-specific gains enter Fast/auto dispatch;
other workloads retain configuration 0. Test the final selected path and retain
compact observations, provenance, generation examples and reproducible timings.
Local edits, artifact verification, execution, tests and commits remain approved;
no push, publication, agents or new kernel search. Historical reserved inputs
are not needed for this implementation milestone and remain untouched.

## Completed implementation checkpoint

The full-model paired run retained 1,360 samples in four complete blocks.
All eleven workload cells have a qualifying candidate; 16/256 chooses 21,
while its configuration-2 comparison is inconclusive. The other cells use
2 or 3 as listed below. The shared Fast/auto dispatcher now installs those
choices on M4 Pro, with baseline 0 elsewhere. Final automatic-dispatch capture,
two mixed-configuration histories and the public default-Fast launcher passed.
The retained replay verifies 72,114 numerical observations, 24,816 exact cache
checks and all 1,360 timing samples. See the [completed runtime study](../studies/model_generation/runtime.md)
for results, validation scope and reproduction commands.

The primary and actual-generation-history diagnostic runs passed all 16,992
cache-storage observations. Five of six independent 32-token continuations
match HF exactly. Same-history comparisons agree on 191 of 192 choices; the
exception is an exact HF top-logit tie. A bounded HF replay reproduces the
large early-layer amplification when given recorded native inputs.

## Historical execution checkpoint

The reference-only calibration at `ce1a8e6` failed its frozen budget ceilings.
All 12,300 declared calibration observations were retained. The largest hidden
row-relative L2 difference was 14.425%, above the 6.25% ceiling; the 1.5 margin
required a 21.875% hidden budget. Prediction checks passed, with all three
changed greedy choices occurring at exact reference ties. A bounded diagnosis
reproduced all 75 metrics on the exposed worst case and inspected 169 affine
operations on identical operands. See the
[retained result](../studies/model_generation/fast-reference.md).

At that historical checkpoint, the original stop gate left confirmation, native
Fast acceptance, integration promotion and model timings unexecuted. The active
diagnostic revision above supersedes that workflow; the original failed
qualification and its limits remain unchanged.

## Validation at the initial freeze

Pre-qualification validation passed: `uv run --locked llm-mojo-validate`
completed the frozen fixture anchors, 136 Python regressions, every native
suite, both tokenizer parity runs and all benchmark smoke routes on Metal.
The new reference self-test separately passed eight tests; it is now registered
in the validation command. Numerical qualification had not run at that freeze.

Approved for autonomous local execution on 2026-09-10, starting at `ac09d16`.
Scope: native batch-one BF16 Qwen2.5-0.5B-Instruct plain-text greedy generation,
Metal on M4 Pro, at most 4096 prompt plus generated tokens. Local edits,
verified pinned artifact preparation/downloads, tests, sequential measurements,
documentation and commits are authorized. No agents, push or publication.

## Historical sequence and stop gates

1. Consolidate the existing decoder selections below. They nominate candidates,
   not automatic model-level winners.
2. Qualify Fast numerical budgets once using reference-only arithmetic and
   schedule variation. Freeze the derivation, dataset and ceilings before
   collection, then freeze derived budgets before independent confirmation.
   Preserve existing operation gates and historical failed model evidence.
3. Connect the qualified Fast candidates to the full model and public launcher.
   Keep configuration 0 as the baseline and unmeasured-workload fallback.
4. Validate all 24 layers, intermediate boundaries, final norm/logits, exact
   cache ownership and lifecycle, mixed configurations and growing context.
5. Accept actual generation, same-history predictions, independent trajectories,
   stop/limit handling, resets, invalid input and UTF-8 streaming.
6. Screen existing numerically qualified configurations on the actual model,
   then confirm once independently using the four-block paired protocol.
   Confirmed gains alone enter automatic dispatch. No new kernel search.
7. Freeze source, dispatch and executable; run reserved acceptance, documented
   repository validation and evidence replay; document and commit the result.

Reference qualification failure stops dependent native acceptance and timing.
Native accuracy failures receive bounded identical-operand diagnosis; ordinary
implementation defects may be fixed. No failed candidate or reserved result
authorizes expanding numerical limits, changing precision or selecting new
reserved inputs. Failed/inconclusive performance proposals retain the baseline.

## Existing candidates

R denotes incoming rows, T total cached rows after the call. Configuration IDs
are decoder IDs, not individual kernel mappings. Preserve their combinations
and BF16 materialization boundaries.

| Workloads R/T | Candidates | Existing basis |
| --- | --- | --- |
| Full prefill and decode | 0 | Integrated attention; tiled MLP 7 for multiple rows, rowwise MLP 0 for decode |
| 16/1024, 16/4096, 15/256, 17/256 | 0, 2 | Earlier shared selection: split8 attention |
| 64/1024, 64/4096, 256/1024, 256/4096, 65/4096, 255/4096 | 0, 3 | Earlier shared selection: split8 plus larger projections |
| 16/256 | 0, 2, 21 | Earlier hot result and newer hot fixed-MMA result; ring results did not promote either |

At the initial freeze, the latest decoder Fast policy selected only 21 at hot
16/256 and 3 at 64/4096. The completed model selection above combines the
measured winners. Treat the earlier hot/ring24 results as nomination evidence:
neither is a functional 24-layer workload. Existing standalone kernels with a
different precision contract are not interchangeable candidates. Rejected MLP
decode candidates and deterministic row-reuse studies do not reopen here.

## Reference-only qualification decision

The executable declaration is `tests/fixtures/model_fast.json`. Use actual
pinned HF Qwen with BF16 stored boundaries, FP32 attention and accumulation.
Compare the existing HF arithmetic with explicit FP32 affine projections using
two contiguous K partitions, summed in FP32 before bias and BF16 rounding.
Both evaluate the same affine equation; this independently changes reduction
association without consuming Mojo results. Include full and cached execution
and the previously qualified canonical attention invocation as a second arm.
HF remains the compatibility target, not an assertion of exact real arithmetic.

Use shared budgets by boundary role (hidden, key, value, final norm, logits),
rather than fitting each layer separately. Derivation retains the historical
rtol 1/32, 1.5 margin, rounding quanta and floors. New declaration fixes absolute
ceilings and retains the 1/16 maximum row-relative L2 budget. Embeddings remain
exact. Prediction distribution checks impose separate KL and total-variation
ceilings; large reference margins require the same greedy choice. These are
explicit engineering acceptance limits, not a universal BF16 error theorem.
Qualification tests the declared limits; observing valid reference variation
outside them does not automatically enlarge them.

Historical native outputs have already been inspected in prior work. The new
calibration reads no native results; its inputs and confirmation cases are new.
Reusing an exposed case later establishes regression only, not independence.

## Completion evidence

Freeze development lengths 1,15,16,17,63,64,65,257,1024,1025,4096, plus schedules
reaching every candidate cell. Cover synthetic IDs and plain-text/Unicode/
repetition inputs. Compare all 75 model boundaries, exact cache preservation,
appended storage and inactive guards. Include cached generation and native full
recomputation on common histories. Preserve the original reserved model inputs.
Use controlled fixtures for stop IDs, ties, nonfinite logits and fault handling.

Measure actual loading, prefill, cached suffixes, growing decode, time to first
token and complete requests, with allocation/compilation and diagnostic capture
boundaries explicit. Record device/backend, dimensions, dtype, versions, source
and executable identity. Keep weights and arrays under ignored build storage;
retain compact raw checks, timings and replayable decisions in the model study.
