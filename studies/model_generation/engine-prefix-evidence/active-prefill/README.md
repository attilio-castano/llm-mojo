# Active-stream prefill cap evidence

The 32-token cap passed the frozen primary gate for full 1,024-token prefix
reuse at 40 blocks. It reduced p95 engine token gaps while completion and p95
TTFT stayed within the declared 10% regression budget. Broader workload guards
did not all pass. The cap remains disabled by default (`active_prefill_cap=0`);
16 tokens remains exploratory. Collection completed on 2026-10-22 from clean
source `75dbfdebf1aba467d3d3eeba516f740d7eb8641c`.

## Results

These are medians of four per-run measurements, in milliseconds, for the
40-block/full-reuse primary cell. Paired ratios below, rather than ratios of
these descriptive medians, determine the frozen gates.

| Active cap | Completion | TTFT p95 | Token gap p95 | Worst gap | Steps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (current) | 985.49 | 825.36 | 54.42 | 55.18 | 65 |
| 32 | 974.63 | 821.44 | 31.60 | 32.07 | 69 |
| 16 | 1035.03 | 877.17 | 19.99 | 20.56 | 77 |

Cap 32 has a median paired p95-gap ratio of 0.580916 (41.9% reduction); all
four ratios are below one and the effect exceeds its 5.106% self-control noise
buffer. Completion and p95-TTFT ratios are 0.975622 and 0.991328, each with a
5% buffer; their upper bounds 1.025622 and 1.041328 pass 1.10. Worst-gap ratio
0.580345 passes as well. The completion/TTFT changes are within noise, so this
does not establish a speedup for either metric. Cap 16 reduces p95 gaps further,
but completion/TTFT upper bounds 1.100961/1.109004 straddle the 1.10 boundary:
its primary gate is inconclusive. The buffer is an engineering rule, not a
confidence interval, and is not rounded down to make a decision pass.

Across the five cells, each triple is current / cap 32 / cap 16. All prefixes
are 1,024 tokens; matching requests are out of eight.

| Matching requests | Blocks | Completion, ms | TTFT p95, ms | Token gap p95, ms | Cap-32 cost/worst-gap guards |
| ---: | ---: | --- | --- | --- | --- |
| 4 | 384 | 3095.9 / 3445.1 / 3507.2 | 2718.2 / 3037.0 / 3101.0 | 182.5 / 32.4 / 21.2 | inconclusive |
| 8 | 384 | 641.9 / 701.2 / 785.3 | 435.7 / 508.7 / 612.2 | 60.4 / 37.3 / 21.7 | inconclusive |
| 4 | 40 | 3798.1 / 3808.7 / 3848.7 | 3364.7 / 3358.9 / 3401.1 | 12.8 / 14.5 / 20.2 | pass |
| 8 | 40 | 985.5 / 974.6 / 1035.0 | 825.4 / 821.4 / 877.2 | 54.4 / 31.6 / 20.0 | pass |
| 0 | 384 | 5564.4 / 6267.6 / 6232.3 | 5170.2 / 5836.7 / 5818.1 | 183.4 / 26.1 / 16.0 | inconclusive |

The large-pool half-reuse and zero-reuse cells show roughly 11% paired
completion overhead for cap 32; their noise-buffered 10% cost guards are
inconclusive. Full reuse at 384 blocks also has inconclusive cost guards, with
larger self-control noise (12.39% completion and 19.98% TTFT). At half reuse
and 40 blocks, cap 32 passes cost/worst-gap guards but does not establish a
p95-gap benefit. Cap 16 worsens p95 gaps in that cell and fails the full-reuse
384-block TTFT guard. The matrix supports the bounded primary result, without
establishing a single best cap across workloads.

Half reuse at 40 blocks also illustrates why worst gap and p95 can move in
opposite directions. Its mixed gaps increase from 3 to 6 to 12 out of 120.
The current policy's three long gaps occupy only 2.5% of the sample, beyond
p95; the 16-token cap's twelve shorter mixed gaps occupy 10% and move into
p95. Worst gap improves from 55.41 to 21.45 ms while p95 worsens from 12.79
to 20.16 ms. Chunking changes both interruption length and frequency.

## Mechanism and qualification

At full reuse, every policy computes exactly 632 continuation rows, reuses
8,192 prefix tokens, delivers the same 128 tokens and records 120 inter-token
gaps per run. Full prefix sharing needs 32 shared blocks plus three private
blocks per request: two readers fit in 38 blocks, while three need 41. At
40 blocks, current / 32 / 16 produce 65 / 69 / 77 total steps and 7 / 14 / 28
mixed steps. Mixed gaps are also 7 / 14 / 28; gaps above 50 ms are 7 / 0 / 0.
Cap 32 trades seven longer interruptions for fourteen shorter interruptions.
At 384 blocks, steps are 23 / 30 / 44 and mixed gaps are 28 / 56 / 84, because
more active streams share a step. Logical model work is unchanged. More step
executions and changed decode concurrency explain the scheduling tradeoff;
GPU traffic and occupancy were not measured directly.

The exact-build Apple M4 Pro/Metal checkpoint passed 127,138,680 comparison and
lifecycle checks, with zero mismatches or unexpected nonfinite values:
115,325,670 mixed KV/logit/greedy cold-reference checks, 7,532,572 zero-head /
partial-prefill checks, 4,280,120 engine cap/lifecycle checks and 318 fault/cache
clear checks. The complete repository validation passed 434 Python tests,
261 native tests across 32 TestSuites, all 38 native invocations and benchmark
smoke routes. Validation ran before the source commit; the retained receipt
binds its identical 271-file source map to the clean measured commit.

All fifteen untimed preflights passed before the single 80-call campaign.
All native exits were zero, natural token histories and per-request reuse/work
matched the frozen preflights, requests drained without preemption, and final
clear returned the pool to Free. Four balanced blocks compare current,
repeated current, cap 32 and cap 16 without selective resampling.

The pinned Qwen2.5-0.5B-Instruct model uses 24 layers, width 896, fourteen query
heads, two KV heads and head dimension 64. The recorded route is reference 27 /
projection 8, synchronous Metal (`metal:4-metal4`), BF16 storage / FP32
accumulation, 32-slot slot-major KV, fixed 256-row/eight-sequence workspaces,
lifetime reservation and no fitted cost budget. Hardware has 24 GiB memory;
software is macOS 26.6.2 / 25G83, Xcode 27.0 / 27A266a, Mojo 1.1.0 and MAX
26.6.0. Runs recorded AC power, normal power mode and no thermal warning.

## Retained evidence

- [Compact results](engine-prefix-active-results.json) include every arm's four
  samples, paired/calibration ratios, noise floors and exact decisions.
  [Extraction code](retention/extract_results.py) verifies the archive manifest
  before deriving the table. Original input paths record provenance; a new
  location changes those path fields only. The [derivation check](derivation-check.json)
  verifies exact reproduction from the canonical measurement archive.
- [Original measurements](measurements/engine-prefix-active.json.gz) and their
  [manifest](measurements/engine-prefix-active.json) retain all 80 native logs,
  commands, actual exits, conditions, complete token events, source/binary/asset
  identities, campaign literals and the exact-build qualification with all
  fifteen preflights. Compressed SHA-256 is
  `a2829701c982681d1b12118b704685abd0faba3e680dfef5b4ba62e2006b3c12`.
- [Validation/source archive](engine-prefix-active-validation.json.gz) and its
  [manifest](engine-prefix-active-validation.json) retain 338 files, including
  exact source from the measured commit, frozen study protocol, full validation
  log, source-map binding, numerical checkpoint and preliminary acceptance logs.
  Twenty-two generated source fixtures unused by CPU replay are hash-only;
  decoder policy JSON is retained in full. Compiled binaries, weights and
  generated numerical arrays are excluded.
- [Acceptance](engine-prefix-active-acceptance.json),
  [canonical-copy receipt](canonical-copy.json) and
  [retention inventory](retention-inventory.json) bind the stages and hashes.
  [Guarded CPU replay](cpu-replay.json), its [actual exit](cpu-replay-execution.json)
  and [log](cpu-replay.log) record a second fresh-directory restore from the
  canonical archives, exit 0, with no prohibited attempts. It imported the
  archived parser and recomputed the full summary without native execution,
  GPU frameworks, prepared/tensor assets or network access.
- [Independent raw audit](independent-raw-audit.json) checks all 80 measured
  histories/work formulas, 6,564 physical ownership snapshots, 384 hits,
  12,032 stored/removed exact-key pairs and 80 clear events. Its
  [helper](retention/independent_raw_audit.py) also matched all 271 live source
  hashes and the original binary hash at audit time. These optional live-file
  checks require those files; the raw-record audit alone is portable.

Custody is local; this is not a remote backup or a fresh GPU run. Earlier
prefix-v1 archives and the 104-file historical catalog are unchanged.

## Reproduce from the archived source

From the repository root, use the locked Python environment and a fresh
`/private/tmp` directory. The independent archive SHA is fixed below; the
[restore helper](retention/prefix_retention.py) is also sealed inside the archive.

```sh
uv run --locked python -B \
  studies/model_generation/engine-prefix-evidence/active-prefill/retention/prefix_retention.py restore \
  --archive studies/model_generation/engine-prefix-evidence/active-prefill/engine-prefix-active-validation.json.gz \
  --manifest studies/model_generation/engine-prefix-evidence/active-prefill/engine-prefix-active-validation.json \
  --expected-sha256 b20d2a061eed81328b59796323e5fef31941acde8b0efae821b1dd85178cf261 \
  --directory /private/tmp/engine-prefix-active-replay
```

The independent raw audit can also run without model assets or a native binary:

```sh
uv run --locked python -B \
  studies/model_generation/engine-prefix-evidence/active-prefill/retention/independent_raw_audit.py \
  studies/model_generation/engine-prefix-evidence/active-prefill/measurements/engine-prefix-active.json.gz
```

The restored directory contains `cpu-replay-result.json`,
`cpu-replay-execution.json` and `cpu-replay.log`. To regenerate native evidence,
check out the measured commit and follow the
[specification/build/qualification/collection commands](../../../../src/llm_mojo/benchmarks/README.md#conditional-prefill-caps-during-active-streams)
with the qualified device and recorded asset identities. New collection uses
fresh output directories and never replaces this grid.

Completion runs from earliest scheduled arrival through trace drain; TTFT
includes queueing and prefill; token gaps exclude TTFT. The seed-to-drain
episode and final clear are reported separately. Timings include retained
stdout/cache-event serialization between steps. Engine token availability does
not measure network/client rendering. Each run has only 120 gaps, so p95,
worst gap and exceedance counts describe finite observations. Native numerical
qualification and logical CPU replay establish distinct parts of the contract.
These offline traces do not establish online capacity, goodput, a client SLO,
arbitrary-input backend correctness, async support or default promotion.
