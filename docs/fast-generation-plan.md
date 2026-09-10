# Fast full-model acceptance

Pre-qualification validation passed: `uv run --locked llm-mojo-validate`
completed the frozen fixture anchors, 136 Python regressions, every native
suite, both tokenizer parity runs and all benchmark smoke routes on Metal.
The new reference self-test separately passed eight tests; it is now registered
in the validation command. Numerical qualification has not run at this freeze.

Approved for autonomous local execution on 2026-09-10, starting at `ac09d16`.
Scope: native batch-one BF16 Qwen2.5-0.5B-Instruct plain-text greedy generation,
Metal on M4 Pro, at most 4096 prompt plus generated tokens. Local edits,
verified pinned artifact preparation/downloads, tests, sequential measurements,
documentation and commits are authorized. No agents, push or publication.

## Sequence and stop gates

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

Latest Fast only selects 21 at hot 16/256 and 3 at 64/4096; it is not a
superset of the older selection. Treat hot/ring24 as nomination evidence:
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
