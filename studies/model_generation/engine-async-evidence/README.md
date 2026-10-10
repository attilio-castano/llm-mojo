# Asynchronous engine evidence

Accepted optional stepping is described in the
[engine study](../engine-core.md#asynchronous-stepping-retained-bounded-results).
The clean implementation commit is
`adae54c095a747f8f69e283177420f71059421de`; collection completed on
2026-10-10 UTC with stable MAX 26.6.0 / Mojo 1.1.0, Qwen2.5-0.5B-Instruct
BF16 storage / FP32 accumulation and Apple M4 Pro / Metal. Both execution arms
used configuration 27 and the same binary, checkpoint, fixed workspace and
reserved admission policy.

The [acceptance card](engine-async-acceptance.json) records successful full
repository validation on the same source bytes, exact-build numerical
qualification, terminal acceptance and independent natural-token parity.
The [results](engine-async-results.json) retain both complete twelve-run grids
and their request latency distributions. Offline median paired async/sync
makespan is 0.95629 with a 5% noise floor: **inconclusive**. The online
eight-request/s trace is descriptive. All 24 runs match the frozen 97-token
histories and finish reasons; each async run charges eight extra rows and
discarded selections. Async remains opt-in.

The [validation archive](engine-async-validation.json.gz) retains exact source,
raw terminal records, reference comparisons, commands, logs, actual numeric
execution receipts and original source snapshots. It preserves the failed
public stream-event probe, the successful context-prefix probe, and the initial
full-validation compiler failure alongside its narrow fixed-array repair and
successful rerun. Historical failed receipts keep their original exits.
The two measurement archives remain separate under `measurements/`.
Weights, compiled binaries and generated oracle arrays are excluded. The
natural chat fixture is regenerated from retained terminal token histories.

The [copy record](canonical-copy.json) binds the original eleven sealed files
to their canonical bytes. The [canonical retrieval card](canonical-retrieval.json)
binds a second fresh CPU restore and a comparison of all 24 restored histories
with the retained frozen oracle. Both actual exits were zero. The full replay
result and its receipt are [canonical-replay.json](canonical-replay.json) and
[canonical-replay-execution.json](canonical-replay-execution.json); the frozen
history proof and receipt are
[frozen-history-comparison.json](frozen-history-comparison.json) and
[frozen-history-execution.json](frozen-history-execution.json).
Checksums establish integrity and CPU replay reconstructs retained evidence.
Custody is local; this is not a fresh GPU execution or remote backup.

## Independent CPU replay

Run from the repository root with the locked Python environment. Choose fresh
`/private/tmp` paths if these example names already exist. This extracts the
hash-checked helper from the archive itself, then restores and reparses retained
source and raw records. Replay disables native execution and asset access; no
checkpoint download or GPU is needed.

```sh
uv run --locked python - <<'PY'
from pathlib import Path
import gzip, hashlib, json
root = Path('studies/model_generation/engine-async-evidence')
manifest = json.loads((root/'engine-async-validation.json').read_text())
compressed = (root/'engine-async-validation.json.gz').read_bytes()
assert len(compressed) == manifest['bytes']
assert hashlib.sha256(compressed).hexdigest() == manifest['sha256']
raw = gzip.decompress(compressed)
assert hashlib.sha256(raw).hexdigest() == manifest['uncompressed_sha256']
record = json.loads(raw)
helper = next(e for e in record['files'] if e['name'] == 'retention/retain_async.py')
body = helper['text'].encode()
assert hashlib.sha256(body).hexdigest() == helper['sha256']
destination = Path('/private/tmp/engine-async-replay-helper.py')
assert not destination.exists()
destination.write_bytes(body)
PY

uv run --locked python -B /private/tmp/engine-async-replay-helper.py restore \
  --archive studies/model_generation/engine-async-evidence/engine-async-validation.json.gz \
  --manifest studies/model_generation/engine-async-evidence/engine-async-validation.json \
  --directory /private/tmp/engine-async-replay \
  --result /private/tmp/engine-async-replay-result.json

uv run --locked python -B /private/tmp/engine-async-replay/frozen-workload/compare_restored.py \
  /private/tmp/engine-async-replay-result.json \
  /private/tmp/engine-async-replay/frozen-workload/expected-histories.json \
  /private/tmp/engine-async-replay-history-proof.json
```

The measurement-only parser is also available through
`uv run --locked python -m llm_mojo.benchmarks.model_profile engine-replay --output studies/model_generation/engine-async-evidence/measurements/engine-async-offline.json.gz`
and the corresponding online archive. Native regeneration requires the recorded
checkpoint assets and hardware, plus the
[build/qualification/collection commands](../../../src/llm_mojo/benchmarks/README.md#asynchronous-engine-stepping).
The original two grids each ran once; rerunning them produces new evidence and
does not replace their timing cells.
