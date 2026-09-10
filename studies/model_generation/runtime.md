# Native Fast Qwen runtime

Completed locally on 2026-09-10: native batch-one BF16 Qwen2.5-0.5B-Instruct
plain-text greedy generation on Apple M4 Pro / Metal, with at most 4096 prompt
plus generated tokens. The runtime composes the existing Mojo tokenizer, all
24 learned decoder layers and independent caches, final RMSNorm, tied head and
streaming UTF-8 decoder. Python verifies and prepares assets before inference.
No kernel arithmetic was changed for this milestone.

`fast` is the public default; `auto` uses the same measured model selection.
Existing configuration 0 remains the fallback for full prefill, ordinary
single-row decode, unmeasured shapes and other device names. Configuration 0
already includes integrated attention and optimized multi-row MLP 7, with
rowwise MLP 0 for decode. This completes the declared integration scope; it is
not a claim that every possible kernel or workload has been optimized.

## Measured selection

Eleven workload cells showed repeatable gains. R is incoming rows and T is total
cached rows after the call. The [complete selection](runtime-selection.csv) and
[paired measurements](runtime-measurements.csv) are regenerated from raw samples.

| R / T | Selected configuration | Baseline ms | Selected ms |
| --- | --- | ---: | ---: |
| 16 / 256 | 21 | 20.668 | 19.271 |
| 16 / 1024 | 2 | 28.550 | 20.827 |
| 16 / 4096 | 2 | 58.869 | 28.707 |
| 64 / 4096 | 3 | 72.102 | 48.841 |
| 256 / 1024 | 3 | 122.994 | 116.114 |

Configuration 2 also wins at 15/256 and 17/256. Configuration 3 also wins at
64/1024, 256/4096, 65/4096 and 255/4096. The configuration-2 comparison at
16/256 is inconclusive; configuration 21 wins there. The selected cells reduce
median paired time by 5.6–51.2%, relative to configuration 0 on these workloads.

The study retains 1,360 samples in four paired blocks, reversing workload and
arm order in blocks two and three. Each arm has ten warmups and ten samples.
Configuration 0 is paired with itself to estimate control variation. Selection
requires a lower median in all four blocks and median improvement exceeding both
5% and the largest control self-pair deviation. These are performance selection
rules; numerical distances are diagnostic observations.

Timing covers the actual 24-layer model with distinct weights and persistent
caches: token upload, intermediate copies and tied head, ending at device
synchronization. Allocation, prefix preparation, compilation, greedy readback
and diagnostic capture are outside the samples. Cached suffixes overwrite the
same suffix after restoring logical cache lengths. These are complete model
forward timings, not complete application request latencies or HF comparisons.

## Numerical diagnosis

The approved policy keeps pinned assets, architecture, causal positions, exact
embedding lookup, cache preservation/appends/guards, finite boundaries and
lifecycle requirements. Existing independent operation tests remain required.
HF and native schedule differences are recorded without a global closeness
threshold. The historical failed qualifications remain unchanged in this study.

The base declaration covers lengths 1, 15, 16, 17, 63, 64, 65, 256, 257, 1024,
1025 and 4096, candidate-specific schedules and three plain-text prompts.
Captures retain all 75 boundaries: embeddings and 24 layer outputs, 48 K/V
boundaries, final norm and logits. References use pinned Torch 2.4.0 /
Transformers 4.43.1, CPU single-thread execution, BF16 stored boundaries and the
existing FP32 math-attention wrapper. Reference arithmetic and backend details
are recorded in the archives.

On the explicit base cases, 149 of 151 same-history next-token choices agree
with HF. Maximum KL divergence is 0.00689 nats and maximum total variation is
0.0473 (rounded upward). Intermediate discrepancies can be much larger than
logit discrepancies; treating their size alone as failure obscures propagation.

Six native generation runs cover three prompts, with whole-prompt and four-row
chunked prefill, followed by 32 greedy tokens. Five of six independent token
trajectories match HF exactly. Comparing HF on the actual native histories gives
191 matching choices out of 192. At the remaining choice, HF assigns exactly
equal top logits to tokens 2086 and 4843; native selects 4843. KL is 0.00228 nats
and total variation is 0.0322 at that step. All 192 native choices reproduce in
the diagnostic capture. The [prediction table](runtime-predictions.csv) preserves
both agreement and differences; [generation examples](runtime-generations.csv)
include the complete bounded outputs. These small raw-prompt examples do not
establish general model quality; both implementations can hallucinate or repeat.

A bounded [propagation experiment](runtime-propagation.csv) investigates the
exposed length-17 case. It first reproduces all 75 original HF boundaries
byte-for-byte, then gives each of the first four HF layers its recorded native
input. HF itself reproduces the amplification to approximately 20.6% and 58.3%
maximum row-relative L2 output differences at layers 2 and 3. Comparing native
and HF on identical layer inputs leaves residuals below 0.36% across those four
layers. This supports accumulated perturbation and model sensitivity as the
cause of that amplification; it is bounded evidence, not proof about every case.

## Execution and validation

The complete replay verifies 72,114 numerical observations, 24,816 exact cache
storage checks and all 1,360 timing samples. This includes explicit candidates,
actual generation histories, the final automatic Fast dispatch across the base
matrix and two mixed-configuration histories. The mixed histories exercise
0 → 21 → 0 → 2 → 0 and 0 → 21 → 0 → 3 → 0 in the same persistent caches.

The public launcher also completed a 1024-token prompt in 16-row chunks plus
eight generated tokens with the default Fast policy, selecting 21 at total 256
and 2 at total 1024. Generation with and without reporting emits identical text
on all six study runs. Tests cover invalid and empty input, overflow, late-cache
preflight, reset/replay, greedy ties, nonfinite rejection and recovery, stop and
context budgets, and tokenizer/UTF-8 streaming. Native source, device identity,
actual selected configurations and submission/cache counters are retained.

Validation passed 146 Python tests, native tests for the selected model routes,
all other native suites with the MLP sweep explicitly limited to Fast mappings
0 and 7, both tokenizer parity runs, and all benchmark smoke routes. The initial
unfiltered validation command was intentionally stopped during the redundant
historical MLP sweep; the remaining suites and smokes then passed with that
explicit scope. This is not an unfiltered full-command pass. Final offline replay
also passes; regressions verify that large finite diagnostic errors remain
reportable while missing evidence or corrupt cache invariants are rejected.

Instrumented generation observed median decode-forward durations of roughly
13–15 ms across the six runs. They end at device synchronization and exclude
greedy readback; reports separately record native initialization, TTFT and whole
request duration. Python asset verification and compiler startup are excluded.
There is no HF latency comparison in this study.

## Retained evidence and reproduction

Twelve lossless JSON archives total approximately 3.1 MB. The
[runtime manifest](runtime-study.json) binds every archive and uncompressed
payload by hash. Five CSV tables regenerate from those complete observations.
Checkpoint weights, arrays, binaries and full traces stay in ignored `build/`.
Measurements and six generation runs use clean source `3834fd8`; the final
selection, mixed-history and public-launch checks use `c4a06ba`. Exact executable,
source, hardware, backend, software and condition records accompany each run.
The public launcher uses `mojo run` and retains source/command evidence rather
than a receipt for an ephemeral executable; its timings support no comparison.

Replay retained evidence without a GPU or model download:

```sh
uv run --locked python studies/model_generation/summarize.py
uv run --locked python -m unittest discover -s tests -p 'test_*.py'
```

For fresh collection, use a clean checkout and verified prepared checkpoint as
in [generation.md](../../docs/generation.md). Each build has a receipt verified
against the exact current clean source. Use new output paths; collection refuses
to replace evidence. The following names assume they do not already exist:

```sh
uv run --locked python -m llm_mojo.model_validation build --binary build/repeat-model
uv run --locked python -m llm_mojo.model_validation build --generation --binary build/repeat-generator
uv run --locked python -m llm_mojo.model_validation specification --output build/repeat-specification.json
uv run --locked python -m llm_mojo.model_validation lifecycle --binary build/repeat-model --prepared build/model-prepared-v1 --output build/repeat-lifecycle.json
uv run --locked python -m llm_mojo.model_validation generate --binary build/repeat-generator --prepared build/model-prepared-v1 --output build/repeat-generation
uv run --locked --script tests/fixtures/model_reference.py diagnose --specification build/repeat-specification.json --output build/repeat-reference
uv run --locked python -m llm_mojo.model_validation diagnose --binary build/repeat-model --prepared build/model-prepared-v1 --reference build/repeat-reference --output build/repeat-diagnostics
uv run --locked python -m llm_mojo.model_validation benchmark --binary build/repeat-model --prepared build/model-prepared-v1 --specification build/repeat-specification.json --output build/repeat-measurements
uv run --locked python -m llm_mojo.model_validation diagnose --binary build/repeat-model --prepared build/model-prepared-v1 --reference build/repeat-reference --policy fast --output build/repeat-selected
uv run --locked python -m llm_mojo.model_validation specification --generations build/repeat-generation/result.json --output build/repeat-history-specification.json
uv run --locked --script tests/fixtures/model_reference.py diagnose --specification build/repeat-history-specification.json --output build/repeat-history-reference
uv run --locked python -m llm_mojo.model_validation diagnose --binary build/repeat-model --prepared build/model-prepared-v1 --reference build/repeat-history-reference --output build/repeat-history
uv run --locked python -m llm_mojo.model_validation specification --mixed-only --output build/repeat-mixed-specification.json
uv run --locked --script tests/fixtures/model_reference.py diagnose --specification build/repeat-mixed-specification.json --output build/repeat-mixed-reference
uv run --locked python -m llm_mojo.model_validation diagnose --binary build/repeat-model --prepared build/model-prepared-v1 --reference build/repeat-mixed-reference --policy fast --output build/repeat-mixed
uv run --locked --script tests/fixtures/model_reference_diagnosis.py --runtime-propagation build/repeat-diagnostics/result.json --runtime-reference build/repeat-reference --output build/repeat-propagation.json
```

Fresh collection produces new observations; replay of the committed study uses
the retained archives and does not silently replace them. The declaration is
`tests/fixtures/model_runtime.json`. All cases are development observations;
historical reserved inputs remain unopened.
