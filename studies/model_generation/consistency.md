# Schedule consistency and native accuracy

The canonical HF reference passes all **71,250 exact boundary comparisons**
through 4096 tokens. The new native consistency route passes primitive and
single-layer checks. Its first full-model input fails seven frozen accuracy
gates, so full-model schedule acceptance, generation and optimization remain
pending. No final reserved input was opened and no tolerance was enlarged.

The declaration was frozen at `b15872f` before native exposure. It separates
two requirements: native outputs must be byte-identical across schedules on the
same hardware/build, and native outputs must satisfy independent numerical
accuracy gates against canonical HF. The old failed HF confirmation remains
historical evidence. Its unchanged budgets were explicitly adopted as
cross-engine hypotheses, not treated as successfully confirmed budgets.

## What now agrees exactly

The actual pinned HF Qwen executes each SDPA query contiguously with its causal
K/V prefix. The original math SDPA and BF16 storage boundaries remain. The
upstream LM head receives one final row per requested call endpoint; all hidden
and final-normalization rows are checked. Streaming comparison avoids keeping
every historical copy of the growing KV cache.

Thirteen declared lengths cover 1, 4, 15, 16, 17, 63, 64, 65, 129, 257, 1024,
1025 and 4096. Schedules include repeated full calls, every partition through
length four, tokenwise calls through 257, and ragged/mixed chunks. Each call
checks 75 boundaries, including all active KV prefixes. Every declared check
passes; [the table](consistency-reference.csv) gives the complete census.

Native configuration 20 uses the existing FP32 G32 decode arithmetic for every
query, including prefill queries. A query's causal prefix determines its key
traversal and reduction order. GPU blocks still process different queries in
parallel. Rowwise Q/K/V, output and MLP projections retain their arithmetic
across row counts. Stored values remain BF16 and reductions FP32.

The receipted Metal tests at `e09957a` pass all three tests, with 13,165 decoder
records and zero failed elements. Primitive comparisons cover full, chunked and
individual decode queries through 257; decoder development cases cover rows
through 65, including branch perturbations. All captured decoder intermediate
outputs and active caches agree byte for byte across their tested schedules.
These are component results, not full-model or capacity-4096 native acceptance.

## The separate accuracy gate fails

The first native full-model case is length 1, seed 9104. The exact executable
built at `d91511c` runs on Apple M4 Pro / Metal. All 48 cache-storage checks pass;
seven of the 75 numerical boundaries fail. Execution stops before later cases.

| Boundary | Failed criterion |
| --- | --- |
| Keys, layers 5, 7, 9, 22 | Per-row relative L2 error |
| Keys, layer 23 | Pointwise and per-row relative L2 error |
| Hidden state after 22 layers | Pointwise error |
| Hidden state after 23 layers | Pointwise and per-row relative L2 error |

Final normalization and logits pass their budgets on this input. For example,
logits have maximum absolute error 0.48046875 and relative L2 error 3.043%;
their combined absolute/relative pointwise gate passes. Passing logits alone
does not waive the intermediate gates. All 123 checks are retained in
[consistency-accuracy.csv](consistency-accuracy.csv).

A one-token input has no alternative prefill schedule. This failure therefore
cannot be explained by prefill versus decode scheduling.

## Identical-operand diagnosis

Actual HF hooks export 360 BF16 intermediate boundaries for that already
exposed input. Observation leaves all 75 model boundaries byte-identical to
unobserved execution. Each native operation then receives its corresponding
HF input, preventing upstream native error from contaminating the local test.

All **336 operation checks pass** existing decoder operation gates. Across
645,120 compared elements, ten differ, each by one BF16 step. Normalization,
attention, residuals, SiLU and gated products are exact on these inputs; the
ten differences occur in projections. Both diagnostic executions are retained.
This rules out a local gate violation on this input, not every possible defect
on other inputs. [Operation totals](consistency-operations.csv) give the scope.

The first difference is layer 0's MLP down projection at coordinate 511:

| Quantity | Value |
| --- | ---: |
| Exact dot product of stored BF16 operands | -0.05065918676700676 |
| Midpoint between the two stored outputs | -0.0506591796875 |
| Mojo output | -0.05078125 |
| HF output | -0.050537109375 |

The exact sum is only about 7.08e-9 below the midpoint. Mojo rounds to the
nearer BF16 value here. Exact rational sums for all ten differing dot products
favor Mojo in five and HF in five; neither execution is universally closer.
Some cancellation-sensitive outputs place both results farther from the exact
sum. These sums are offline diagnostics, never inference inputs or replacement
acceptance gates. [The rounding table](consistency-rounding.csv) retains exact
numerators and denominators as well as both outputs.

The evidence supports ordinary projection rounding followed by error
propagation, rather than an attention schedule defect. It does not prove that
no arithmetic change could satisfy the current full-model gates. Choosing such
a change or revising cross-engine acceptance requires a prospective numerical
policy decision; tuning to this exposed case would not establish generality.
The agreed accuracy stop is preserved. Configuration 20 remains explicit;
automatic selection and all three optimization studies remain unspent.

## Reproduction and provenance

`consistency-study.json` binds four compressed files containing every reference
comparison, qualification metadata, native accuracy checks, both operation
diagnoses, decoder records and clean-source executable receipts. The study is
numerical evidence only. Test elapsed times are not performance measurements.
Weights, activation arrays, binaries and full validation traces stay outside Git.

The checkpoint revision is `7ae557604adf67be50417f59c2c2f167def9a775`; its complete
SHA is in every reference receipt. Upstream uses Torch 2.4.0, Transformers
4.43.1 and NumPy 1.26.4 on one CPU thread. Native execution uses locked Mojo
1.0.0 / MAX 26.5.0 on M4 Pro / Metal. HF qualification was produced at `b15872f`,
native accuracy at `d91511c`, operation capture and initial diagnosis at
`64a8468`, and the exact-sum diagnosis and receipted layer tests at `e09957a`.
Receipts also bind source files, compiler commands and exact binary hashes.

Regenerate retained tables without model execution:

```sh
uv run --locked python studies/model_generation/summarize.py
```

With pinned assets prepared as documented in `docs/generation.md`, reproduce
the new reference, native accuracy and diagnosis in fresh output directories:

```sh
uv run --locked --script tests/fixtures/model_consistency.py --self-test
uv run --locked --script tests/fixtures/model_consistency.py --output build/reproduce-consistency-reference
uv run --locked python -m llm_mojo.model_validation build --binary build/reproduce-consistency-model
uv run --locked python -m llm_mojo.model_validation consistency --binary build/reproduce-consistency-model --reference build/reproduce-consistency-reference --output build/reproduce-consistency-accuracy --length 1
# The preceding command records seven failures and exits nonzero.
uv run --locked --script tests/fixtures/model_reference_diagnosis.py --native-operations --output build/reproduce-consistency-operations-reference
uv run --locked python -m llm_mojo.model_validation operations --binary build/reproduce-consistency-model --reference build/reproduce-consistency-operations-reference --output build/reproduce-consistency-operations
DECODER_RECORDS=build/reproduce-consistency-layer.jsonl uv run --locked mojo run -I src -I build -I tests tests/test_consistency.mojo
```

Build and execution require a clean checkout with unchanged source between
them. Use the recorded historical commits for exact historical reproduction;
fresh builds receive their own receipts. The complete repository validation
command remains `uv run --locked llm-mojo-validate`.

Validation completed: regenerated fixtures match frozen anchors, all 119 Python
tests pass, all native regression files pass, and all benchmark smoke routes
pass on Metal. The first full validation stopped at an outdated negative test
that still rejected newly valid GQA mapping 5. That test now rejects mapping 6;
the corrected sublayer file passed and validation resumed with the remaining
files and smoke checks. The numerical full-model failure above remains an
acceptance failure, independent of these passing repository regressions.
