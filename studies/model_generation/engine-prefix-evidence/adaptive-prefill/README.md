# Adaptive prefill evidence

The adaptive policy passed the frozen primary gate and all twelve workload
cost/worst-gap guards. At full 1,024-token prefix reuse and 40 blocks, it kept
the fixed32 streaming benefit while the three 384-block workloads retained
the current policy's exact logical schedules. This is a bounded offline result;
the engine option remains disabled by default (`adaptive_prefill=False`).

The contract was frozen before implementation in `e16f175`. Collection completed
on 2026-10-22 from clean source `7dd3864dcd34e04170215876d304a54808abb856`,
using one complete 192-call campaign after all 36 untimed preflights. No policy,
workload, threshold or sample was changed in response to timing results.

## Primary result

Values are medians of four **per-run** metrics, in milliseconds. Frozen gates
use medians of paired candidate/current ratios, which can differ from ratios
of these descriptive medians. Gaps are never pooled across repetitions.

| Policy | Completion | TTFT p95 | Token gap p95 | Worst gap | Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 954.86 | 798.71 | 53.51 | 53.83 | 65 |
| Fixed32 | 982.83 | 824.09 | 30.94 | 31.23 | 69 |
| Adaptive | 973.04 | 817.99 | 30.91 | 31.12 | 69 |

The adaptive/current p95-gap paired ratios are 0.577370, 0.578205, 0.577546 and
0.577116; their median is 0.577458, a **42.3% reduction**. All four are below one,
the median is below 0.75, and the reduction exceeds the 5% self-control noise
buffer. Completion and p95-TTFT paired ratios are 1.017585 and 1.020965;
adding their 5% buffers gives 1.067585 and 1.070965, below the 1.10 limit.
Worst-gap ratio 0.577810 passes too. These results support the primary
hypothesis and bounded cost guards; the small completion/TTFT changes are
within noise and do not establish improvements in those metrics. Fixed32
measurements are descriptive, without an adaptive/fixed32 timing-equivalence
claim or a separate fixed32 self-control.

## Complete matrix

All prompts have a 1,024-token prefix; reuse is the number of intentionally
matching requests out of eight. Each metric triple is **current / fixed32 /
adaptive**, in milliseconds. All adaptive completion, p95-TTFT and worst-gap
guards pass. The per-cell p95-gap outcome tests the 25% improvement hypothesis;
it is distinct from the cost/worst-gap guards.

| Reuse | Blocks | Suffix | Outputs | Order | Completion | TTFT p95 | Token gap p95 | Adaptive gap hypothesis |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 4 | 384 | 64 | 16 | first | 3047.6 / 3374.2 / 3048.7 | 2665.9 / 2973.2 / 2666.7 | 179.0 / 31.9 / 178.9 | inconclusive |
| 8 | 384 | 64 | 16 | first | 630.1 / 694.2 / 634.5 | 426.7 / 498.5 / 426.9 | 59.0 / 36.6 / 59.1 | inconclusive |
| 4 | 40 | 64 | 16 | first | 3625.9 / 3651.3 / 3654.1 | 3205.5 / 3225.5 / 3237.0 | 13.5 / 14.9 / 14.8 | fail |
| 8 | 40 | 64 | 16 | first | 954.9 / 982.8 / 973.0 | 798.7 / 824.1 / 818.0 | 53.5 / 30.9 / 30.9 | pass |
| 0 | 384 | 64 | 16 | first | 5431.9 / 6005.3 / 5429.6 | 5047.2 / 5585.2 / 5047.4 | 179.1 / 24.8 / 179.1 | inconclusive |
| 8 | 41 | 64 | 16 | first | 819.1 / 854.7 / 836.9 | 666.9 / 692.9 / 671.7 | 54.3 / 31.6 / 54.3 | inconclusive |
| 8 | 56 | 64 | 16 | first | 631.8 / 689.1 / 631.4 | 426.7 / 497.8 / 426.8 | 59.0 / 36.5 / 59.1 | inconclusive |
| 8 | 36 | 33 | 16 | first | 765.4 / 805.1 / 791.9 | 622.8 / 659.5 / 646.0 | 31.0 / 30.5 / 30.5 | inconclusive |
| 8 | 40 | 63 | 16 | first | 936.5 / 974.1 / 972.1 | 788.2 / 819.5 / 812.9 | 52.1 / 30.6 / 30.6 | pass |
| 8 | 40 | 65 | 16 | first | 962.7 / 1017.3 / 960.1 | 805.1 / 855.2 / 808.2 | 54.0 / 30.9 / 54.0 | inconclusive |
| 8 | 40 | 64 | 2 | first | 443.0 / 483.8 / 443.1 | 410.5 / 449.6 / 410.5 | 54.1 / 30.9 / 54.0 | inconclusive |
| 4 | 40 | 64 | 16 | even | 5698.9 / 5710.3 / 5704.1 | 5277.7 / 5287.5 / 5284.6 | 13.0 / 13.1 / 13.4 | inconclusive |

The closest adaptive cost guard is the 63-token tail's completion bound:
1.037985 + 0.05 = **1.087985**, below 1.10. The 33-token tail's p95-TTFT bound
is 1.085393. Its gap reduction is only 1.65%, below the noise buffer, while
completion and TTFT paired medians increase 2.82% and 3.54%. It passes the
cost guards without establishing a streaming benefit. The rule was not
retuned around that result.

Half reuse at 40 blocks has **worse p95 gaps**: 13.47 to 14.78 ms, paired ratio
1.089335, so its gap hypothesis fails. Worst gap improves from 53.79 to
31.19 ms. Three long mixed gaps become six shorter mixed gaps out of 120:
changing interruption frequency moves more mass into the p95 boundary. Thus
all cost/worst-gap guards passing does not imply universal p95 improvement.

The two-output cell delivers sixteen tokens and has only eight gaps per run,
compared with 128 tokens and 120 gaps in the other cells. Current and adaptive
both take nine steps; no streaming benefit is established there. In the
alternating-order cell, only 1,024 prefix tokens are actually reused per run:
intervening cold work evicts reusable cache state before later nominal matches.
All policies agree on those actual hits and compute 7,800 rows; the study does
not substitute the intended four-match count for observed reuse.

## Mechanism and implementation

The engine computes the rule from the existing FIFO match and admission-fit
values, before changing ownership. It caps only the first chunk of a newly
admitted request, when exactly one decoder is selected with at least two
declared outputs remaining, the uncomputed tail is 33–64 rows, another
sequence slot remains, and available
capacity after this reservation cannot fit another comparable request.
All other work follows the current scheduler. Held prefills need no policy
state: a capped first chunk leaves at most 32 rows. There is no additional
cache lookup, speculative eviction, or positive-hit requirement. Unsupported
configuration and async combinations reject before mutation. The precise
predicate and claim limits are in the
[frozen contract](../../engine-core.md#adaptive-prefill-frozen-comparison-contract).

At full reuse/40 blocks, all policies compute exactly 632 continuation rows
and 672,448 causal attended positions, reuse 8,192 tokens and deliver the same
128 tokens. Current versus adaptive uses 65 versus 69 steps and seven versus
fourteen mixed steps. Gaps over 50 ms fall from seven to zero; gaps over 20 ms
rise from seven to fourteen. Splitting an interruption changes both length
and frequency, without reducing model work.

The raw audit confirms exact ordered selections and cache-event schedules:
adaptive equals fixed32 at 40 blocks, 69/115 steps for full/half reuse, and
equals current at 384 blocks, 23/39/55 steps for full/half/zero reuse. For those
three large-pool cells, adaptive completion paired ratios are 1.006816,
0.999957 and 0.999639, all within the 5% noise buffer. Identical logical fallback
schedules do not establish zero elapsed overhead. The 41/56-block, 65-token,
two-output and alternating-order holdouts also retain current schedules.
GPU traffic and occupancy were not directly measured.

## Qualification and provenance

Full documented validation passed **444 Python tests, 270 native tests across
32 TestSuites, all 38 native invocations and all benchmark smoke routes**.
Fixture self-tests also passed 11, 10 and 8 tests. The full gate took
2,135.15 seconds; its unchanged 271-file source map is bound to the clean
measured commit by [the source receipt](source-binding.json).

The exact-build Apple M4 Pro/Metal checkpoint passed **189,770,889 checks**,
with zero mismatches or unexpected nonfinite values: 115,325,670 mixed
KV/logit/greedy cold-reference checks, 7,532,572 zero-head/partial-prefill
checks, 66,912,487 actual adaptive-engine KV/logit/greedy checks, and 160
terminal/fault/cache-clear checks. Actual-engine numerical probes exercise
adaptive and fallback paths, R=33/63/64/65, capacity equality and protected
idle cache. Broader predicate boundaries have host tests; this is not an
arbitrary-input backend proof. Ordered receipts establish logical batch
inputs and selections, rather than direct reads of physical device buffers.

All 36 preflights passed before the 192 timed calls. All calls exited zero,
matched frozen natural histories, actual per-request hits, rows, attention
work and schedules, drained without preemption, and cleared to Free. Four
Williams blocks compare current, self-current, fixed32 and adaptive. Every
invocation has identical current-policy warmup and seed; only continuation
selects the policy. Native calls have a 180-second bound and collection a
two-hour bound. No samples were selectively repeated. Earlier development
compile, sandbox, test-environment and fixture failures are retained with
their actual known statuses; transcript-only evidence is labelled as such.

The pinned Qwen2.5-0.5B-Instruct route has 24 layers, width 896, fourteen query
heads, two KV heads and dimension 64. It uses synchronous reference 27 /
projection 8, BF16 storage / FP32 accumulation, 32-slot slot-major KV, fixed
256-row/eight-sequence workspaces, lifetime reservation and zero watermark.
The device is Apple M4 Pro/Metal (`metal:4-metal4`), 24 GiB; software is macOS
26.6.2 / 25G83, Xcode 27.0 / 27A266a, Mojo 1.1.0 and MAX 26.6.0. Conditions
record AC power, normal power mode and no thermal warning. Source, compiler,
assets, tokenizer, binary, dtype, layout and device identities are retained.

## Retained evidence and restoration

- [Compact results](engine-prefix-adaptive-results.json) retain every arm's
  four samples, paired/calibration ratios, noise floors, decisions and logical
  schedule matches. [The extractor](retention/extract_results.py) checks the
  measurement manifest. [The derivation receipt](derivation-check.json) confirms
  canonical regeneration, apart from original input-path fields.
- [Original measurements](measurements/engine-prefix-adaptive.json.gz) and
  [manifest](measurements/engine-prefix-adaptive.json) retain all raw native
  logs/events, commands, exits, conditions, source/binary/asset identities,
  campaign literals, checkpoint and all 36 preflights. Compressed SHA-256 is
  `22092f6bc69297df4538ca6bbff0ba33f23b1dbfadf0d0760c50693750744fcc`.
- [Source/validation archive](engine-prefix-adaptive-validation.json.gz) and
  [manifest](engine-prefix-adaptive-validation.json) retain 331 files: measured
  source, frozen protocol, full validation log, checkpoint/preliminary receipts,
  development-attempt records and independent auditors. Twenty-two generated
  source fixtures unused by CPU replay are hash-only; decoder policy JSON is
  retained in full. Binaries, weights and generated oracle arrays are excluded.
- [Acceptance](engine-prefix-adaptive-acceptance.json),
  [canonical-copy receipt](canonical-copy.json),
  [sealed inventory](retention-inventory.json) and
  [final inventory](final-inventory.json) bind the stages and retained bytes.
  First and canonical-copy guarded restores both exited zero. The second
  [CPU replay result](cpu-replay.json), [actual exit](cpu-replay-execution.json)
  and [log](cpu-replay.log) confirm archived-source imports, exact recomputed
  summary, no native/GPU/model-asset/network access and no prohibited attempts.
- [Independent raw audit](independent-raw-audit.json) and
  [actual exit](independent-raw-audit-execution.json) check 230 embedded zero
  exits, 13,416 timed ownership snapshots, 26,160 ordered selections, 1,168 hit
  receipts, 20,864 stored/removed key pairs and 192 clears, plus all preflights.
  Its [standalone helper](retention/independent_raw_audit.py) imports no
  production parser. Optional live checks matched all 271 source hashes and
  both original binaries. [Independent decision audit](independent-decision-audit.json)
  and [actual exit](independent-decision-audit-execution.json) recompute all
  ratios, buffers, decisions, medians and arm census from compact samples using
  [a separate standard-library helper](retention/independent_decision_audit.py).

Custody is local, without a remote backup or fresh GPU rerun. Earlier prefix-v1
and active-prefill artifacts remain byte-for-byte unchanged. Restore from the
repository root with locked Python and a fresh `/private/tmp` directory:

```sh
uv run --locked python -B \
  studies/model_generation/engine-prefix-evidence/adaptive-prefill/retention/prefix_retention.py restore \
  --archive studies/model_generation/engine-prefix-evidence/adaptive-prefill/engine-prefix-adaptive-validation.json.gz \
  --manifest studies/model_generation/engine-prefix-evidence/adaptive-prefill/engine-prefix-adaptive-validation.json \
  --expected-sha256 f7bce811055054a4dc7c9b9759a985b9b16f8b65f4d3103083b462b3970ee882 \
  --directory /private/tmp/engine-prefix-adaptive-replay
```

The raw auditor can run with only the retained measurement archive:

```sh
uv run --locked python -B \
  studies/model_generation/engine-prefix-evidence/adaptive-prefill/retention/independent_raw_audit.py \
  studies/model_generation/engine-prefix-evidence/adaptive-prefill/measurements/engine-prefix-adaptive.json.gz
```

For fresh native collection, check out measured commit `7dd3864` and follow
[the build/qualification/collection commands](../../../../src/llm_mojo/benchmarks/README.md#adaptive-short-tail-prefill-campaign)
using the recorded device and verified asset identities, with fresh outputs.

Completion is earliest scheduled arrival through drain; TTFT includes queueing
and prefill; inter-token gaps exclude TTFT. Seed-to-drain episode and final clear
are separate. Timings include instrumentation and stdout/cache serialization
between steps; these are engine-availability observations, not isolated kernel
timings or client rendering. The maximum-based engineering noise buffer is not
a confidence interval. This finite offline grid establishes no online capacity,
goodput, client SLO, async support, universal tail improvement or default promotion.
