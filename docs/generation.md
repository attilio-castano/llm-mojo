# Qwen model composition and generation

For interactive multi-turn use with persistent KV caches, see [terminal chat](chat.md).

The active [Fast implementation revision](fast-generation-plan.md) uses numerical
comparisons as diagnostics. Required checks cover assets, data flow, cache and
generation semantics; historical full-model distance ceilings no longer block
integration. `fast` is the public default, with `auto` as an alias. On Apple M4 Pro they select the eleven measured workload cells described in
the [runtime study](../studies/model_generation/runtime.md), with configuration
0 elsewhere.
Explicit configurations 0/2/3/21 and the historical `consistent` path remain
available. The full-checkpoint generator, lifecycle checks, numerical diagnostics and
paired model measurements have executed. Numerical differences are retained
and explained separately from exact implementation invariants.

## Historical numerical-policy studies

The subsequent [Fast completion effort](fast-generation-plan.md) stopped during
reference-only qualification: the declared intermediate-error ceilings failed
before independent confirmation or native Fast acceptance. The
[Fast reference study](../studies/model_generation/fast-reference.md) retains
the complete calibration, diagnosis and stop decision. The earlier native
configuration-20 accuracy failure below remains separate historical evidence.

**Development candidate: promotion is paused at native full-model accuracy.**
The authorized [consistency revision](../studies/model_generation/consistency.md)
passes 71,250 exact canonical HF comparisons through 4096 tokens and native
primitive/layer schedule tests. Its first full-model input fails seven frozen
accuracy gates. All 336 identical-operand operation checks pass; ten projection
elements differ by one BF16 step. Full-model schedule and generation acceptance
remain pending, and configuration 20 is not automatically selected.

The native model and generation call graphs compile. The
embedding/copy and BF16 binary-I/O tests pass on Apple M4 Pro / Metal. The
upstream observation code passes a tiny synthetic 24-layer self-test; that is
not qualification of the pinned checkpoint. The full checkpoint and all 196
prepared tensors have since passed hash/extent verification. Reference-only
calibration completed, but independent confirmation failed 8 of 2,025 checks.
The [retained study](../studies/model_generation/README.md) records the frozen
budgets and diagnosis. That original confirmation remains failed; the approved
consistency revision and native failure are recorded separately.

The authorized [reference-only follow-up](../studies/model_generation/rounding.md)
traced the discrepancy to FP32 attention differences crossing BF16 rounding
boundaries and propagating through the model. It reproduced the normalization
gate violation while observing identical argmax in 66 comparisons and matching
bounded greedy sequences on three declared prompts. These results motivate
separating corresponding-mode comparisons from cross-schedule diagnostics;
they do not establish Mojo model acceptance or change the frozen thresholds.

The deeper [HF/PyTorch study](../studies/model_generation/backend.md) locates the
first difference in QK matrix multiplication. Normalizing SDPA query shape and
causal-prefix layout yields 36,225 byte-equal full/cached comparisons at five
declared lengths, while deterministic mode alone leaves the original differences
unchanged. The approved consistency revision now qualifies this canonical
route while preserving the original failed policy and evidence.

The approved scope and stop gates are in [generation-plan.md](generation-plan.md).
The declaration is [model_contract.json](../tests/fixtures/model_contract.json).
The numerical thresholds are initial reference-only qualification criteria;
they failed, and a bounded reference-only calibration also failed independent
confirmation. Both declarations and results remain intact. The new
[consistency declaration](../tests/fixtures/model_consistency.json) independently
requires exact schedule agreement and adopts the unchanged frozen budgets as
cross-engine hypotheses. The first native accuracy failure is under that new
declaration, not acceptance under the historical failed qualification.

## Native ownership

`QwenModel` owns one BF16 embedding allocation, also used as the tied LM head,
the final RMSNorm weights, 24 distinct learned decoder weight sets, and 24
independent persistent KV caches. It owns one attention workspace, one MLP
workspace, input staging, a final normalized row and next-token logits. Clients
must preserve this ownership; replacing internal allocations is unsupported.

The first implementation copies each intermediate decoder output into distinct
input storage before the next layer. This satisfies the existing decoder's
input/output nonaliasing rule. There is no host synchronization between layers
in normal execution. A later measured buffer-rotation experiment may eliminate
these copies without changing arithmetic; no speedup is claimed now.

Every call preflights all layers, token IDs, shapes, allocation extents, row
capacity, cache lengths and configuration requirements before the first model
dispatch. The host token upload synchronizes. Layer work then uses one ordered
Metal stream. Cache lengths count submitted tokens; greedy readback waits for
completion. Submission/readback failure invalidates the model. Reset marks the
model invalid before waiting and restores validity only after synchronization.

The runtime processes only new token rows. `submitted_rows` counts submitted
layer rows, and the diagnostic driver reports it with logical cache length.
The completed runtime study verifies these counters during diagnostic captures,
mixed-configuration calls and native generation.

## Workload policy

`select_configuration` is shared by model clients. `fast` (the public default)
and `auto` use the measured M4 Pro lookup: split8 configuration 2 at
16/1024, 16/4096, 15/256 and 17/256; configuration 3, combining split8 and
larger projections, at 64/1024, 64/4096, 256/1024, 256/4096, 65/4096 and
255/4096; configuration 21 at 16/256. Pairs denote incoming rows / total
cached rows. Every other shape and device name falls back to configuration 0.
Baseline 0 already includes integrated attention and optimized multi-row MLP
7, with rowwise MLP 0 for decode. Full prefill and ordinary single-row decode
therefore retain these existing optimized routes.

`baseline` always selects 0. `candidate` retains the prior split8 lookup, and
explicit IDs 0/2/3/21 remain available for comparison. The historical
`consistent` / 20 route uses FP32 G32 attention and rowwise projections at all
row counts; it remains an explicit study mode. The same selected configuration
applies to all 24 layers for a call.

Scratch includes split8 capacity before execution, so selection allocates
nothing within the layer loop. The diagnostic suite checks exact cache
preservation, append storage and inactive capacity across mixed configurations.

## Prepared checkpoint and reference

Use the pinned assets and hashes from [model.md](model.md). The existing explicit
download command is:

```sh
uv run --locked --script tests/fixtures/generate.py attention_checkpoint -- --download --download-only
```

Preparation runs the pinned Torch/Transformers environment, verifies the full
checkpoint/configuration hashes, rejects incomplete model loading, and writes
196 BF16 tensors with shapes, source names, byte counts and hashes. Q/K/V are
concatenated in source output-row order. Rotary tables come from the pinned
upstream implementation. Python is used for preparation and reference execution;
no Python interop runs in the native model or text generator.

```sh
uv run --locked --script tests/fixtures/model_reference.py self-test
uv run --locked --script tests/fixtures/model_reference.py prepare --output build/checkpoints/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775/model-prepared-v1
uv run --locked --script tests/fixtures/model_reference.py qualify --output build/oracle_data/model-qualification
```

Under the historical qualification workflow, failure prevents dependent comparison.
The new `model_reference.py diagnose` / `model_validation diagnose` workflow
captures corresponding histories without requiring qualification; it retains
numerical distances separately from required exact storage checks. The historical
`model_reference.py qualify` command above is expected to reproduce its original
failure. Use the [consistency study commands](../studies/model_generation/consistency.md)
for the historical canonical route. That consistency workflow requires
an explicit passing qualification from the same reference source and contract;
the active diagnostic workflow does not.
The upstream executes the real `Qwen2Model` and tied LM head with BF16 stored
boundaries and math SDPA with FP32 Q/K/V/masks, rounding the attention output to
BF16. It records every decoder boundary, final normalization, logits and caches.
The LM-head wrapper projects only the final row when capturing next-token logits.

Native diagnostic files contain raw little-endian BF16 bytes. `write_bytes` is
required; generic file `write` formats lists as text. The regression includes
negative zero and a BF16 subnormal. Short diagnostic cases allocate their active
context plus three guard positions, bounded by 4096, avoiding enormous inactive
cache dumps while retaining exact guard checks. Full-capacity cases remain.

The development evaluator builds only from clean source, records the compiled
executable hash, and launches that exact executable. It checks reference array
hashes, complete boundary coverage, actual M4 Pro / Metal identity, cache
accounting, numerical error, exact preserved prefixes, exact appended K/V and
untouched inactive capacity. The completed [runtime study](../studies/model_generation/runtime.md)
adds actual generation, same-history diagnosis, paired full-model measurements
and final Fast dispatch verification. Historical reserved inputs remain unopened.

## Native plain-text generation

The verified development launcher is:

```sh
uv run --locked python -m llm_mojo.model_assets --prepared "$MODEL_PREPARED" --prompt "$PROMPT_FILE" --max-new-tokens 16 --policy fast --report build/generation-events.tsv
```

`MODEL_PREPARED` names the prepared model directory and `PROMPT_FILE` contains
raw prompt bytes. The launcher verifies the checkpoint identity, manifest,
all 196 tensor extents and hashes, and pinned tokenizer tables before starting
the native driver. Missing model artifacts fail; tokenizer preparation uses
local assets without downloads. Keep the prepared files unchanged during
execution. Python performs initialization only, with no interop in inference.

`generate_cli.mojo` is the internal driver: direct invocation assumes that its
caller has already performed those checks. It composes the native tokenizer,
model and streaming decoder.
It accepts raw prompt bytes, optional fixed-size prompt chunks and a maximum
output count. It performs raw-logit greedy selection with lowest-ID ties,
rejects nonfinite logits, stops at IDs 151645/151643 or the requested/context
limit, and emits complete UTF-8 fragments. It applies no sampling or repetition
penalty. A selected final token is in history but need not have been consumed
into the KV cache. Chat-template rendering is outside this milestone.

The optional TSV report records prompt/generated IDs, selected prefill
configurations, native initialization, prefill and decode durations, and cache
accounting. Native durations exclude Python asset verification and compilation.
TTFT includes native initialization/tokenization; decode timings end at device
synchronization. Output remains generated UTF-8 text on stdout.

Full-checkpoint execution, lifecycle checks, six 32-token generation runs and
a public default-Fast smoke with a 1024-token prompt passed. Same-history
predictions agree with HF on 191 of 192 generated choices; the exception is an
exact HF top-logit tie. These are bounded development observations, not a
general model-quality or exact trajectory-equivalence claim.
