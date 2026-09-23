# Decoder layer implementation record

> **Historical record**, kept as written. The [history index](README.md) says what it
> led to and where current guidance lives.

Moved from the [decoder layer contract](../decoder-layer.md) on 2026-09-23.

## Original implementation handoff

This section preserves the reference-package handoff before Mojo composition.
The completed implementation and configuration study are recorded below.

Keep the new contract beside existing docs. When implementing the fixture
package, add `tests/fixtures/decoder_layer/{contract,reference,generate}.py`
and its reference tests, using the established package pattern. Use `tests/fixtures/decoder_reference.py` with a symlink to the existing
script lock; the shared dispatcher is itself a frozen MLP source. Reuse or
extract common upstream capture helpers without changing old oracle outputs.
Do not create a benchmark runner or experiment hierarchy for this contract.

The ordered reference-package tasks are:

- [x] Encode these recipes, schedules, capture names and gates in the executable
  contract; keep holdout execution outside the default development command.
- [x] Implement actual decoder observation and tests for transparent hooks,
  policy-wrapper execution, residual/gating reconstruction, and chunk slicing.
- [x] Tokenize the reserved prompt without evaluating it. Qualify development
  captures and preserve all diagnostics, including failed attempts.
- [x] Freeze compact manifest/checksum records containing reference versions,
  source/lock hashes, model/tensor identities, dtypes, shapes/layouts, schedules,
  gates, per-array hashes, and the holdout declaration. Generated arrays remain
  under ignored `build/oracle_data/decoder_layer/`.
- [x] Review the qualified reference package before implementing Mojo layer
  composition. Keep source/binary/fixture receipts separate from acceptance.

The handoff required the Mojo implementation to extend the existing
test/validation workflow and run `uv run --locked llm-mojo-validate`, explicit
local checkpoint checks, and normal-mode asynchronous tests. At that checkpoint,
the reference generator and its self-tests were integrated in ordinary
validation; Mojo composition followed under the execution plan.

After layer acceptance, use the bounded six-workload profiling proposal:
`(R,T)=(256,256),(4096,4096),(16,256),(64,4096),(1,256),(1,4096)`.
Freeze explicit mappings before that study, starting correctness with integrated
attention mappings `(0,0)` and MLP 0; validate MLP 7 for multi-row composition
before measuring it. Additional attention mappings need the same layer gates.
Allocation/uploads stay outside timing; latency and diagnostic profiles remain
separate. Close the milestone once the layer is correct and its measured costs
are explained, following [experiments.md](../experiments.md).

The existing repository validation also passed: 79 Python tests, every existing
Mojo suite, and all benchmark route smoke checks. This was the reference-package
checkpoint; it does not yet qualify the newly added Mojo decoder wrapper.

## Decoder implementation checkpoint

`src/llm_mojo/layers/decoder_layer.mojo` composes the existing attention and MLP
entrypoints. A shared, side-effect-free attention preflight plus MLP preflight
runs before the first dispatch. The wrapper rejects invalid geometry, layout,
capacity, mapping, short buffers and overlapping writable storage. Attention's
arithmetic and launch order are unchanged. The MLP reads Z directly from the
attention workspace; Y remains in the MLP workspace.

Development checks passed for all 43 synthetic and three checkpoint cases on
Apple M4 Pro / Metal. Both MLP mappings are explicit, with mapping 0 for one
row. Tests distinguish operation-local error, isolated MLP error and composed
error, then check the declared full/chunk schedules. The twelve-decode test
retains all four boundaries before overwrite and compares with a separate
workspace/cache execution. Eight negative controls establish sensitivity to
wrong residuals, norm inputs/weights, absolute position, mask and cache prefix.
Reserved acceptance subsequently passed all seven declared cases with the exact
frozen binary: 2,468 core checks plus preservation and behavior records. Its
largest whole-layer Y scaled error is 0.015504 against the 0.03125 gate. See the
[numerical evidence](../../studies/decoder_layer/numerics.json) for complete coverage,
candidate/fixture identity and original checks.

The registered `decoder_layer` benchmark uses one fixed policy (ID 0), the
six declared shapes and control self-pairs. Prefix preparation executes the
Mojo layer outside timing. Its shared workspaces have max_rows=T so they also
serve prefix preparation; only R rows are written in measured calls. Ring24
uses distinct allocations with identical timed contents. The untimed
adversarial check changes 24 hidden-coordinate sign patterns, absorbing each
sign into the corresponding input/weight axes so upstream expected outputs
transform by the same sign. This preserves arithmetic while detecting wrong
allocation selection. It does not represent 24 learned model layers.

## Completed baseline and next step

The [study](../../studies/decoder_layer/README.md) reports the full six-workload grid
on Apple M4 Pro / Metal. At full R=T=256, MLP contributes 78.8% of captured
active GPU time; for R=64,T=4096, attention contributes 66.8%. Decode has 16.1%
gaps in the enclosing diagnostic window and substantial latency self-pair noise.
These gaps do not isolate host overhead. No kernel optimization was selected.

The one malformed decode capture was preserved and retried once after fixing
the capture parser's rejection of valid MLP mapping 0 (`05e1def`). The retry
used the same `d67fd94` binary; no engine or numerical policy changed. Full
traces remain external, with compact samples and receipts retained in Git.

A later milestone is full-model forward parity: compose embeddings, all 24
layers, final normalization and LM head under a separately declared logits
contract. This baseline does not yet establish model logits or generation.
