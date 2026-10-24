# Frequency-aware prefill: qualified, measured, acceptance inconclusive

Executed 2026-10-22 from clean source `df042c20d0bdd4fe51f5fc32f9bbad663a1769c3` ([PR #41](https://github.com/attilio-castano/llm-mojo/pull/41)), after the contract was frozen in `900bf1b`. The policy is implemented and qualified. The single 320-call campaign completed; its overall performance acceptance remains **inconclusive**. Defaults remain current.

The primary full-reuse/40 aggregate p95-gap hypothesis passed: **-41.89%**, with completion **+2.36%** and p95 TTFT **+2.67%**. All four gap ratios were below one, with a paired median of 0.581116930809 and a 5% noise floor.

Cost/worst-gap guard sets passed in **19/20 cells**; p95-no-harm passed in **18/20**. All remaining statuses are inconclusive, with **no frequency guard FAIL**. Half reuse/384 has unresolved completion, TTFT and p95-no-harm guards; half reuse/40 has unresolved p95-no-harm. Thus `broad_guard_pass=false`, `broad_p95_guard_pass=false`, `study_success=false`. No resampling or policy revision followed these results.

## Primary and secondary observations

The following are medians of four per-run metrics. Percentage effects elsewhere are medians of paired ratios, which need not equal ratios of these medians. Times are milliseconds.

| Cell | Arm | Completion | p95 TTFT | Aggregate p95 gap | Worst gap | Steps / mixed | >20 / >50 ms gaps | Largest request p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full/40 | current | 1051.997 | 872.645 | 54.239 | 54.894 | 65 / 7 | 7 / 7 | 25.882 |
| full/40 | adaptive | 1074.228 | 896.895 | 31.632 | 32.064 | 69 / 14 | 14 / 0 | 31.795 |
| full/40 | frequency | 1078.778 | 892.870 | 31.556 | 31.829 | 69 / 14 | 14 / 0 | 31.678 |
| half/40 | current | 3797.674 | 3335.153 | 13.588 | 54.415 | 113 / 3 | 3 / 3 | 25.529 |
| half/40 | adaptive | 3792.488 | 3339.028 | 15.032 | 31.724 | 115 / 6 | 6 / 0 | 31.530 |
| half/40 | frequency | 3790.604 | 3335.672 | 13.788 | 54.538 | 113 / 3 | 3 / 3 | 26.001 |

Full reuse preserves adaptive’s exact 69-step/14-mixed-step schedule. Seven long interruptions become fourteen shorter interruptions. Aggregate p95 and maximum gap improve, but the largest per-request p95 increases from 25.882 to 31.678 ms. With fifteen gaps per request, one long gap lies above most of that request’s p95 interpolation; two medium gaps enter both interpolation endpoints. The aggregate has 120 gaps and a different quantile population. This policy does not establish per-request p95 or client-SLO improvement.

Half reuse restores current’s exact 113-step/3-mixed-step schedule. Its paired aggregate p95 effect is +0.4255%, versus old adaptive’s descriptive +8.6852%. Frequency retains three >50 ms stalls instead of adaptive’s six medium stalls. The new no-harm guard remains inconclusive because its self-control noise is 13.2463%; matching work and a near-one median are not timing-equivalence evidence.

## Complete frozen matrix

Effects below are frequency/current paired median percentages. Cost status combines completion, p95 TTFT and worst-gap guards. I means inconclusive. All rows have four balanced blocks; no samples are pooled across repetitions. `M/T` describes only the valid leading-run forecast; a dash means the geometry/tail/output premise was unsupported.

| Matching / queued | Pool | Suffix | Output limit | Order | M/T; caps | Completion % | p95 TTFT % | p95 gap % | Worst % | Costs | p95 no harm |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 4/8 | 384 | 64 | 16 | first | —; 0 | +2.14 | +1.82 | +3.54 | +3.62 | I | I |
| 8/8 | 384 | 64 | 16 | first | —; 0 | +0.19 | -0.07 | +0.04 | +0.04 | PASS | PASS |
| 4/8 | 40 | 64 | 16 | first | 3/7; 0 | -0.30 | +0.02 | +0.43 | +0.50 | PASS | I |
| 8/8 | 40 | 64 | 16 | first | 7/7; 7 | +2.36 | +2.67 | -41.89 | -42.04 | PASS | PASS |
| 0/8 | 384 | 64 | 16 | first | —; 0 | -0.96 | -1.06 | -1.98 | -2.49 | PASS | PASS |
| 8/8 | 41 | 64 | 16 | first | —; 0 | +0.51 | +0.43 | +0.20 | +1.71 | PASS | PASS |
| 8/8 | 56 | 64 | 16 | first | —; 0 | +0.22 | +0.05 | -0.13 | -0.04 | PASS | PASS |
| 8/8 | 36 | 33 | 16 | first | 7/7; 7 | +3.61 | +3.58 | -1.60 | -1.45 | PASS | PASS |
| 8/8 | 40 | 63 | 16 | first | 7/7; 7 | +1.49 | +1.61 | -41.24 | -40.52 | PASS | PASS |
| 8/8 | 40 | 65 | 16 | first | —; 0 | +0.26 | +0.91 | -0.10 | -0.02 | PASS | PASS |
| 8/8 | 40 | 64 | 2 | first | —; 0 | +0.02 | +0.05 | -0.20 | -0.25 | PASS | PASS |
| 4/8 | 40 | 64 | 16 | even | 0/7; 0 | +0.17 | +0.39 | +0.72 | +1.87 | PASS | PASS |
| 5/8 | 40 | 64 | 16 | first | 4/7; 0 | +0.27 | +0.30 | -3.20 | -0.21 | PASS | PASS |
| 6/8 | 40 | 64 | 16 | first | 5/7; 0 | +1.75 | +1.63 | +0.39 | +0.03 | PASS | PASS |
| 7/8 | 40 | 64 | 16 | first | 6/7; 0 | +0.52 | +0.82 | +0.12 | +0.21 | PASS | PASS |
| 7/7 | 40 | 64 | 16 | first | 6/7; 0 | -0.54 | -0.03 | -0.02 | +0.37 | PASS | PASS |
| 9/9 | 40 | 64 | 16 | first | 8/8; 8 | +3.35 | +3.41 | -41.72 | -41.78 | PASS | PASS |
| 4/8 | 40 | 64 | 3 | first | 3/2; 3 | +0.36 | +0.35 | -41.78 | -41.68 | PASS | PASS |
| 8/8 | 40 | 64 | 8 | first | 7/4; 7 | +2.09 | +2.21 | -41.77 | -41.89 | PASS | PASS |
| 8/8 | 40 | 64 | 32 | first | 7/14; 0 | +0.84 | +0.45 | -1.57 | +0.16 | PASS | PASS |

The six permitted cells were exactly the predeclared full/40, suffix-33/pool-36, suffix-63/pool-40, nine-request, G=3 half-reuse and G=8 full-reuse cells. Every valid forecast’s declared gap count and predicted leading-run interruptions matched the same-block current observation; measured invalidations were zero. Staged arrival/EOS/abort/fault invalidation remains separately tested in host and Metal qualification.

The conservative seven-request rule deliberately forgoes a measured opportunity: old adaptive’s p95 ratio was 0.683807, while frequency selected baseline (0.999805). Conversely, G=32 had only seven interruptions among 248 gaps: old adaptive’s p95 ratio was 2.493220, while frequency fell back (0.984282). These are descriptive comparisons within the frozen campaign, not equivalence or new acceptance claims.

The alternating FIFO case had four intended matches but only one actual hit after cache pressure crossed the cold barrier. The forecast counted that leading run of one, performed one extra lookup and applied no caps. Primary and secondary forecast processing, including later admission checks, had median total costs of 272.913 and 146.748 microseconds per measured run; controls recorded zero forecast work. These costs are included in the reported times.

## Why broad acceptance remains open

| Cell / metric | Paired median | Self-control noise | Median + noise | Limit | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| half/384 / duration_ns | 1.021413755 | 0.098138979 | 1.119552734 | 1.10 | I |
| half/384 / p95_ttft_ns | 1.018212972 | 0.102073393 | 1.120286364 | 1.10 | I |
| half/384 / p95_gap_ns | 1.035392068 | 0.124684952 | 1.160077020 | 1.10 | I |
| half/40 / p95_gap_ns | 1.004254704 | 0.132462556 | 1.136717260 | 1.10 | I |

For half/40 p95, self/current ratios were [1.132462556, 0.869850133, 0.965430246, 1.009123018]; frequency/current ratios were [1.030908903, 0.899628101, 0.977600506, 1.031736935]. Half/384 p95 self ratios were [0.875315048, 0.999667887, 1.002263516, 1.082956404]. Their buffers exceed the predeclared upper limit. The physical cause of this variation was not identified; it is not evidence of a frequency-policy regression, and it is insufficient evidence to close no-harm.

## What changed and why

`EngineCore` has a separate opt-in `frequency_prefill` mode. At the first
successful admission it freezes the already-arrived positive-output FIFO cohort
by immutable tickets. Normal completion never shrinks that cohort. It scans the
leading compatible prefix run once, without changing cache ownership, reference
counts, LRU or KV data. Distinct private suffix keys and the two-reader geometry
are required for a valid forecast. New arrivals, cancellation, early termination
or an unexpected counted-member geometry disable further caps until full drain.
An already-issued partial tail finishes through the ordinary held-prefill path.

The gate retains every adaptive-v1 condition and adds a frequency permission:

```
N = sum(max(declared_output_limit - 1, 0)) over the frozen cohort
M = max(leading_compatible_run_length - 1, 0)
T = N - floor(19 * (N - 1) / 20)
permission = valid_forecast and M >= T and member_of_counted_run
```

The implementation evaluates the cutoff with bounded integer arithmetic. It
never uses future arrivals, actual future EOS, measured gap durations or the
current-control result to choose the candidate schedule. Prefix lookups, copies,
membership checks, invalidation and logging remain inside measured scheduling
and continuation time. Current and old-adaptive controls emit disabled receipts
without running the forecast. Defaults remain current.

For N gaps, linear p95 lies at index `0.95*(N-1)`. If all mixed gaps are slower,
M must reach `N-floor(0.95*(N-1))` to put both interpolation endpoints inside the
mixed population. With 120 gaps that cutoff is seven. Full reuse has seven
unsplit interruptions: splitting can shorten a tail that p95 already samples.
Half reuse has three: splitting to six can move slower work into the p95
interpolation boundary. The new rule preserves the full-reuse schedule and
chooses baseline for half reuse. That deliberately accepts baseline's rarer,
longer worst stalls. It does not promise improvement in every latency metric.

The seven-request case is intentionally stricter than an approximate five-percent
fraction: N=105, M=6, T=7. Nine matching requests give N=135, M=8, T=8 and permit
caps. G=3 half reuse has N=16/M=3/T=2; G=8 full reuse N=56/M=7/T=4; G=32 full reuse
N=248/M=7/T=14. These prospective cases test the finite quantile calculation
rather than a hardcoded full/half reuse label.

## Measurement and acceptance

The frozen contract is in [engine-core.md](../../engine-core.md#frequency-aware-prefill-frozen-comparison-contract).
Twenty cells each have four balanced Williams blocks of current, self-current,
old adaptive and frequency: 320 timed native invocations. Sixty unique-policy
preflights precede timing. Natural greedy histories, terminal reasons, actual
per-request cache hits, necessary model rows and causal attention agree across
policies. Each timed run must reproduce its own policy's preflight schedule.
Seed and ten warmup steps use current scheduling in every invocation.

Completion is earliest scheduled arrival to drain. TTFT uses the first engine
availability event; gaps use successive engine availability events and exclude
TTFT. Seed, episode and cache clear are separate boundaries. Tables report
medians of four per-run metrics; effects use the median of four paired ratios.
Gaps are never pooled across repetitions. Actual-hit/miss and intended-match/miss
groups, individual request p95/max/counts, >20/50 ms counts and decode/mixed
attribution remain inspectable in the derived report and raw archive.

For each metric, noise is `max(0.05, max(abs(self-current/current - 1)))`.
The primary requires four p95 ratios below one, paired median at most 0.75 and
an effect larger than noise. Completion and p95-TTFT pass when median+noise is
at most 1.10; worst gap passes when median is at most 1+noise. The separate
p95-no-harm guard requires both median<=1.05 and median+noise<=1.10. Overall
study success requires the primary and both broad guard sets across all twenty
cells. Missing/noisy evidence stays inconclusive. No resampling, cell replacement,
threshold tuning or default promotion follows from this campaign.

## Claim boundaries

This is the synchronous reference-27/projection-8 route on Apple M4 Pro/Metal,
BF16 storage/FP32 accumulation, slot-major KV with 32-token blocks, lifetime
reservation, 256 model rows, eight workspace sequence slots and zero watermark.
The 7/8/9-request cells concern queued cohorts, not a larger active workspace.
The independent KV/logit/greedy checkpoint uses H=32/pool=7/context=128 with
actual EngineCore policy execution. Full H=1,024 cells add complete natural
histories, ownership/work replay and exact logical schedules; that is a separate
claim layer from an arbitrary-size full-KV numerical proof. Staged arrivals are
functional/numerical tests outside the all-arrivals-zero performance grid.

N uses declared upper bounds. EOS, changed overlap, cache churn and heterogeneous
work can invalidate the forecast. Observed counts are compared post-run with the
same-block current control; those counts are never scheduler input. The literal
predicate also permits an empty matched prefix in short cold cohorts; cold-64
performance is not measured here. The measured no-hit cell has a long uncached
prompt and falls back. These data do not establish online capacity, goodput,
client SLOs, async support, universal p95 improvement or default promotion.
Custody is local: archived-source CPU restoration verifies replayability, not a
remote backup or fresh GPU rerun.

## Verification and retained evidence

Measured hardware/software: Mac16,7, Apple M4 Pro, 14 logical CPUs, 24 GiB, Metal target `metal:4-metal4`; macOS 26.6.2/25G83, Xcode 27.0/27A266a, Metal 32023.921, Mojo 1.1.0, MAX 26.6.0 and uv 0.12.5. AC, normal power mode and nominal thermal-state checks are retained before and after each timed call. No CPU fallback is accepted.

Full documented validation exited 0 in 2423.187 s: 455 Python tests, 280 native tests across 32 suites, 38 native Mojo invocations, tokenizer parity including Unicode, and all benchmark smoke routes. The 271 source-file hashes match the subsequent clean measured commit. The exact-build Metal checkpoint checked 64,510,413 cohort/KV/logit/greedy checks and 8,035,159 invalidation/terminal/clear checks, with zero declared mismatches/nonfinite values. All 60 preflights and 320 timed calls exited 0. The collection and final built-in replay exited 0.

Independent raw replay checked all 382 embedded zero-exit receipts, raw first-arrival/cohort inputs, literal quantiles, ownership/LRU/cache events, ordered selections, work, histories, schedules, all 271 current source hashes and both original binary hashes. The separate decision audit recomputed 200 metric comparisons, 40 policy decisions and 40 no-harm guards using 34,038 recursive checks. Both exited 0. Two fresh archived-source CPU restorations also exited 0 with no prohibited attempts; no model assets, native subprocesses, GPU frameworks or network access were used by those guarded replays.

Development failures remain inside the validation archive: compile attempts, an observed-event fixture setting, zero-based ticket expectations, a known-schedule tooling scope error, uv sandbox access and the independent raw auditor’s omitted fourth header literal. The last repair added the already-declared numerical-scope identity to that auditor’s expectation. No runtime policy, acceptance rule or measured bytes changed after timing; no timed call was replaced. Intermediate native compile source hashes/raw diagnostics were not captured per attempt; the retained index labels its command/exit summaries honestly.

The expanded results and two independent audit reports are preserved
byte-for-byte in the [supplementary package](supplementary.tar.gz), together
with the original campaign README. The [packaging manifest](../packaging-manifest.json)
describes the current layout; the final inventory below remains an
original-layout receipt. [Restore that layout](../README.md#verify-and-restore-the-evidence-layout)
to recover every original filename and README. Packaging does not rerun Metal
or change the inconclusive overall performance verdict.

| Artifact | Purpose |
| --- | --- |
| [Raw measurement archive](measurements/engine-prefix-frequency.json.gz) and [manifest](measurements/engine-prefix-frequency.json) | All timed records and embedded preflights/qualification, original stdout and exits |
| [Packaged derived results](supplementary.tar.gz), member `engine-prefix-frequency-results.json` | Full matrix, paired ratios, per-request/group observations and forecasts |
| [Validation/source archive](engine-prefix-frequency-validation.json.gz) and [manifest](engine-prefix-frequency-validation.json) | 406 retained source/document/acceptance files, plus 22 hash-only generated source fixtures unused by CPU replay |
| [Source binding](source-binding.json) | Validated source bytes equal clean measured source |
| [Packaged audits](supplementary.tar.gz), members `independent-raw-audit.json` and `independent-decision-audit.json` | Independent logical and arithmetic checks |
| [First CPU replay](first-cpu-replay.json) and [canonical CPU replay](cpu-replay.json) | Two fresh guarded restorations |
| [Acceptance](engine-prefix-frequency-acceptance.json), [final checks](final-checks.json), [inventory](final-inventory.json) | Exact execution/verdict/custody bindings |

Measurement SHA-256: `19a57868535d5682a0a9adafcd2da32d3592df0380cf9f7bd7cdc4c0b630bb77` (15,987,536 bytes). Validation archive SHA-256: `8215d5719703ad936049d2c0b44a70192f6be8cc2c392e100d8f60ed012f8c5f` (1,975,251 bytes). Source identity is the measured commit above; the frozen contract precedes it. Old prefix/cap/adaptive archives remain byte-for-byte unchanged.

Generate a new derived report under ignored `build/`, with a fresh output path:

```sh
mkdir -p build/engine-prefix-reports
uv run --locked python -B \
  studies/model_generation/engine-prefix-evidence/frequency-prefill/retention/extract_results.py \
  studies/model_generation/engine-prefix-evidence/frequency-prefill/measurements/engine-prefix-frequency.json.gz \
  build/engine-prefix-reports/engine-prefix-frequency-results.json
```

From the repository root with its existing locked environment, choose a fresh restoration directory:

```sh
.venv/bin/python -B studies/model_generation/engine-prefix-evidence/frequency-prefill/retention/prefix_retention.py restore \
  --archive studies/model_generation/engine-prefix-evidence/frequency-prefill/engine-prefix-frequency-validation.json.gz \
  --manifest studies/model_generation/engine-prefix-evidence/frequency-prefill/engine-prefix-frequency-validation.json \
  --expected-sha256 8215d5719703ad936049d2c0b44a70192f6be8cc2c392e100d8f60ed012f8c5f \
  --directory /private/tmp/frequency-prefill-fresh-restore \
  --python "$PWD/.venv/bin/python"
```

The helper refuses to replace a directory. The SHA is pinned independently here; checking an untrusted archive against only its adjacent manifest is insufficient. Source and receipts are restored from measured Git bytes. The guard reproduces the retained summary, including its inconclusive overall verdict; it does not rerun Metal or close the performance acceptance gate.
