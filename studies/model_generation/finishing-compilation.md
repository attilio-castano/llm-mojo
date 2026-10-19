# Finishing-proof compilation

Named host admission functions let callers reuse a checked specialization.
Their bodies enforce compile-time assertions using the existing independent
proof checkers. A merely repeated `comptime` expression was not enough: the
earlier canonical-expression version added 24.6% compiler
instructions for a second caller despite nearly unchanged wall time.

## Retained observations

Apple M4 Pro, Mojo 1.1.0 (8189361e), two compiler workers,
the committed lockfile, and a new compiler cache for every build. These are
serialized observations with one sample per cell, not a statistical guarantee.

| Workload | Original / full checks | Named admissions | Observed reduction |
| --- | ---: | ---: | ---: |
| Real operation benchmark build | 592.00 s | 384.09 s | 35.1% |
| Two host callers, sequential/balanced local trees | 403.89 s | 363.11 s | 10.1% |

| Shared path | Build wall time | Retired compiler instructions |
| --- | ---: | ---: |
| One caller | 369.94 s | 6,087,165,133,210 |
| Two callers with different local trees | 363.11 s | 6,104,674,706,200 |

The additional caller changed retired instructions by 0.29%. Separate
cold-build wall times include noise and should not be subtracted to claim a
precise marginal cost. First admission remains expensive. The host diagnostic
holds a 64-thread, three-item loop program fixed; production shapes remain
parameters, and each local tree and executable descriptor is checked.

Operation offload elaboration fell from 375.8372 to 0.1010
seconds. The device compilation no longer re-enters proof admission. Peak
process footprint changed from 11.03
to 6.57 GiB. First operation
executable runs took 0.328 and
0.283 seconds and passed on Apple M4 Pro / Metal.
These startup timings are not GPU kernel benchmarks. Offload pass times are
already included in host elaboration and must not be added again.

## What is reused

`require_finishing_execution()` is an ordinary host call whose body checks the
fixed general theorem at compile time. `require_finishing_traversal[program]()`
specializes only on the complete loop program; its body checks coverage,
headroom, base-linearity and termination. Layouts and reduction trees can reuse
that specialization. A genuinely different program still needs admission;
coverage and termination are not yet one theorem quantified over all program
fields. Runtime counts remain quantified within their declared contract.

The launcher independently checks each small executable descriptor before
enqueuing its nested GPU closure. No proof AST, runtime witness or acceptance
flag is passed to the device. Arbitrary submitted proofs retain their original
full-input checks, including unused syntax, roots, declarations, closed expected
types, resource budgets and certificate/request equality. Base-linearity and
termination share the existing multi-root inference state; no inference rule
or trusted persistent receipt was added.

## Qualification and provenance

The final export test passed compile-time positives, 19 changed-input controls
(one invalid canonical program, 18 native controls including one resource
preflight), and exact host-plan and ordinary-call compiler rejections. The
ordinary-call control requires the checker diagnostic and no executable;
optimization must not erase an unchecked compile-time obligation.

All 452 host Python tests outside the two heavy compiler modules passed again,
including real Matplotlib 3.10.8 rendering. The complete existing token-selection
matrix passed on Apple M4 Pro / Metal with all geometries, layouts, runtime
folds, sequential/balanced/compact plans, tails, ties, guards and rejection cases.
The final operation binary passed every maintained linear/RMSNorm/RoPE route,
all 24 valid token-selection routes and both invalid selectors under device-sync
diagnostics. These establish correctness for the exercised cases.

The existing native proof suite and two exact malformed-induction compiler
rejections were qualified at `cf4e8612ae9c516db5f9bbab805a30395505e28e`; all their test and proof-component
inputs are byte-identical at the final code commit. The dataset preserves the
temporary harness's earlier output-line correction and unchanged-binary rerun.
A fresh complete repository `validate` run remains pending; these focused checks
do not relabel the immutable historical proof receipts as fresh runs.

Original operation sources: `b211354ea9979dfa59b64750c6828123a9560e06`. Final code: `a49506e92524730d990838d6698b90091024be30`. All
retained build source inventories were checked against Git. The original
operation measurement used the identical profiling-driver bytes in the earlier
commit context; the dataset retains both identities. The final numerical
qualification preceded the commit, and every input byte is verified against
that commit afterward. Earlier expression diagnostics are explicitly tied to
their own source/probe commit. Uncommitted exploratory runs are excluded.

The generic kernel and admission specification, compiler/evaluator and immutable
parameter identity, function specialization and closure transport, allocation
and storage obligations, native-word/address interpretation, source-to-model
correspondence, GPU lowering, memory visibility and progress remain trusted.
There is no formal soundness proof for the kernel. These results do not establish
full-model floating-point determinism or GPU performance. Historical receipts,
source archives and generated explorer remain unchanged.

## Reproduce

From the final code commit, use new output paths:

```sh
uv run --locked python -B tests/profile_finishing_compilation.py --target operations --source-revision a49506e92524730d990838d6698b90091024be30 --output /private/tmp/finishing-operations
uv run --locked python -B tests/profile_finishing_compilation.py --copies 2 --reuse 0 --source-revision a49506e92524730d990838d6698b90091024be30 --output /private/tmp/finishing-full
uv run --locked python -B tests/profile_finishing_compilation.py --copies 2 --reuse 1 --source-revision a49506e92524730d990838d6698b90091024be30 --output /private/tmp/finishing-shared-two
uv run --locked python -B tests/profile_finishing_compilation.py --copies 1 --reuse 1 --source-revision a49506e92524730d990838d6698b90091024be30 --output /private/tmp/finishing-shared-one
```

For the original operation comparison, materialize its committed `src/` outside
the checkout and pass `--source-root /absolute/path/to/src --source-revision b211354ea9979dfa59b64750c6828123a9560e06`
to the same operation driver. It verifies the complete Mojo inventory and bytes.
Full caches and binaries remain outside Git.

Regenerate the main table from retained observations without a compiler or GPU:

```sh
python3 -B - <<'PY'
import gzip, json
from pathlib import Path
data = json.loads(gzip.decompress(Path('studies/model_generation/finishing-compilation.json.gz').read_bytes()))
for title, left, right in [
    ('Operations', 'retained-baseline-operations', 'named-retained-operations'),
    ('Two mixed-tree callers', 'named-retained-unshared-callers2', 'named-retained-shared-callers2'),
]:
    a, b = [data['runs'][name]['record']['compile_wall_ns'] / 1e9 for name in (left, right)]
    print(title, format(a, '.2f'), format(b, '.2f'), format(100 * (1-b/a), '.1f') + '%')
PY
```

[Compact raw observations](finishing-compilation.json.gz) retain source/driver
hashes, commands, compiler passes/resources, native outputs and qualification
records. Dataset SHA-256: `036c3f1a709cda7ec0ee7188faf96de81b458893e73e016d49854d9e4aaad401`.
