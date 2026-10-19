# Prefix-cache evidence

The [study](../engine-core.md#prefix-caching-retained-bounded-results) records
the first opt-in synchronous cache and its finite offline comparison. Collection
completed on 2026-10-19 UTC (October 18 locally) from clean source commit
`dcd09b5d57eb4cb20ed7f7bd806d6e6c5c23cd91`, on Apple M4 Pro / Metal with Mojo
1.1.0, MAX 26.6.0 and the pinned Qwen2.5-0.5B-Instruct assets. Numerical policy is
reference 27, projection 8, BF16 storage / FP32 accumulation and 32-slot
slot-major KV. The cache remains disabled by default.

All sixteen preflights passed before the single 96-call campaign. All delivered
histories match. Six reuse cells are faster under the frozen paired rule;
the two zero-reuse cells are inconclusive. Full 1,024-token reuse at 384 blocks
has continuation/episode ratios 0.108679 / 0.191188. The 40-block full-reuse
case has faster completion but larger p95 token gaps, 13.16 → 53.58 ms.
These are instrumented host timings for finite reserved traces. The episode
includes seed and handoff and excludes final clear; cleanup is reported
separately. There is no default-promotion, online capacity or SLO conclusion.

## Retained files

- [Acceptance](engine-prefix-acceptance.json) binds the measured commit,
  qualification, actual exits and successful drain/clear. Full repository
  validation ran on `c959179`; subsequent changes affected three trace/parser/
  test files, covered by targeted tests, the final 424-test Python run and
  exact-build qualification. Production engine/model/cache sources stayed
  identical to the complete native validation.
- [Compact results](engine-prefix-results.json) retain all eight cells, exact
  paired/self-control ratios, individual noise floors and the latency/work/cache
  statistics. [Extraction code](retention/extract_prefix_results.py) computes
  these from raw records. Its recorded input path is provenance; regenerating
  from another path changes that path field.
- [Measurements](measurements/engine-prefix.json.gz) retain every native stdout,
  actual execution receipt, literal workload and history, conditions, build
  receipt, and the final qualification with sixteen preflights. The
  [manifest](measurements/engine-prefix.json) seals compressed and uncompressed
  bytes. Compressed SHA-256 is
  `1cb0dc9a848adefccb25d46d0816fa9db771f751221b2925a0f03c0b462032f4`.
- [Validation/source archive](engine-prefix-validation.json.gz) and its
  [manifest](engine-prefix-validation.json) retain exact committed source,
  validation logs, preliminary numerical/chat/fault evidence and both earlier
  failed untimed qualification attempts. The initial NumPy interop transport
  failure is preserved separately from numerical results. The two final
  harness failures concern seed arrival bookkeeping and zero-duration no-op
  handoff. Their original exits remain failed; no timed cell was replaced.
- [Canonical copy](canonical-copy.json) verifies byte-exact promotion from the
  sealed originals. [CPU replay](cpu-replay.json), its
  [execution receipt](cpu-replay-execution.json) and [log](cpu-replay.log) record
  a second fresh-directory restore from these canonical files: actual exit 0,
  96 measurements, 16 preflights and the seven-case/123-token preliminary chat
  acceptance, with no prohibited execution, asset or network attempts.
- [Independent raw audit](independent-audit/audit.json) checks all work formulas,
  6,500 ownership snapshots, 144 hits and logical cache events. Its
  [helper](independent-audit/reproduce.py) also checked the original live source
  bytes and native binary hash at audit time. Repeating that particular check
  requires those recorded inputs and an explicit `--source-root`; portable
  source/record restoration uses the command below.

Weights, compiled binaries and generated numerical arrays are excluded.
Generated JSON/gzip source fixtures unused by CPU replay are retained by hash;
the required decoder policy JSON is retained in full. Existing historical
Phase 3 catalog bytes are unchanged. Custody is local; restoration does not
constitute a new GPU run or remote backup.

## Restore and replay on CPU

Run from the repository root with the locked Python environment. The restore
directory must be fresh and under `/private/tmp`. The independent archive SHA
is fixed here; the helper checks it before restoring source and records, then
runs the archived helper with native execution, GPU frameworks, prepared-asset
access and networking blocked. The shipped helper is also retained inside the
sealed archive.

```sh
uv run --locked python -B \
  studies/model_generation/engine-prefix-evidence/retention/prefix_retention.py restore \
  --archive studies/model_generation/engine-prefix-evidence/engine-prefix-validation.json.gz \
  --manifest studies/model_generation/engine-prefix-evidence/engine-prefix-validation.json \
  --expected-sha256 73c10a73c0c84422177db8e9cea4a523b1e557baf90c9c11be5e95c4c464bd3f \
  --directory /private/tmp/engine-prefix-replay
```

The result and actual child exit appear in `cpu-replay-result.json` and
`cpu-replay-execution.json` inside that directory. It verifies all retained
histories and recomputes the complete paired summary from the archived parser.
Logical identity and total-work replay is distinct from the native tests that
establish per-request completed-step publication.

For a quick measurement-only check with the current source:

```sh
uv run --locked python -m llm_mojo.benchmarks.model_profile engine-replay \
  --output studies/model_generation/engine-prefix-evidence/measurements/engine-prefix.json.gz
```

Native regeneration requires the qualified hardware and recorded assets.
Check out the measured source commit and use the
[specification/build/qualification/collection commands](../../../src/llm_mojo/benchmarks/README.md#synchronous-reference-prefix-cache-campaign).
Regeneration produces new evidence and never replaces this grid.
