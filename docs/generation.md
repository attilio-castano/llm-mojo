# Qwen model composition and generation

**Development candidate: acceptance is blocked by independent reference
schedule confirmation.** The native model and generation call graphs compile. The
embedding/copy and BF16 binary-I/O tests pass on Apple M4 Pro / Metal. The
upstream observation code passes a tiny synthetic 24-layer self-test; that is
not qualification of the pinned checkpoint. The full checkpoint and all 196
prepared tensors have since passed hash/extent verification. Reference-only
calibration completed, but independent confirmation failed 8 of 2,025 checks.
The [retained study](../studies/model_generation/README.md) records the frozen
budgets and diagnosis. Native full-model comparison, generation acceptance and
performance promotion have not run.

The authorized [reference-only follow-up](../studies/model_generation/rounding.md)
traced the discrepancy to FP32 attention differences crossing BF16 rounding
boundaries and propagating through the model. It reproduced the normalization
gate violation while observing identical argmax in 66 comparisons and matching
bounded greedy sequences on three declared prompts. These results motivate
separating corresponding-mode comparisons from cross-schedule diagnostics;
they do not establish Mojo model acceptance or change the frozen thresholds.

The approved scope and stop gates are in [generation-plan.md](generation-plan.md).
The declaration is [model_contract.json](../tests/fixtures/model_contract.json).
The numerical thresholds are initial reference-only qualification criteria;
they failed, and a bounded reference-only calibration also failed independent
confirmation. Both declarations and results remain intact. No full-model
candidate outputs have been compared against them.

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
Full-model acceptance must still confirm these counters and actual execution.

## Workload policy

`select_configuration` is shared by model clients. `baseline` always selects ID
0. `candidate` uses the prior shared decoder lookup at its exact confirmed
shapes on Apple M4 Pro, falling back to 0 elsewhere. Explicit IDs 0/2/3 remain
available for comparison. `auto` currently selects 0 everywhere: historical
one-layer choices require full-model confirmation before automatic promotion.
The same choice applies to all 24 layers for that call.

Scratch includes split8 capacity before execution, so selecting an existing
configuration does not allocate within the layer loop. Cached-prefix numerical
compatibility must be validated when a later call changes configuration.

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

A qualification failure prevents dependent comparison. Reference capture requires
an explicit passing qualification from the same reference source and contract.
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
untouched inactive capacity. It is a development evaluator; reserved acceptance,
generation acceptance and performance promotion are not yet implemented.

## Plain-text generation candidate

`generate_cli.mojo` composes the native tokenizer, model and streaming decoder.
It accepts raw prompt bytes, optional fixed-size prompt chunks and a maximum
output count. It performs raw-logit greedy selection with lowest-ID ties,
rejects nonfinite logits, stops at IDs 151645/151643 or the requested/context
limit, and emits complete UTF-8 fragments. It applies no sampling or repetition
penalty. A selected final token is in history but need not have been consumed
into the KV cache. Chat-template rendering is outside this milestone.

This candidate has compiled but has not yet executed with the full checkpoint.
Do not treat compilation, the primitive tests or the tiny upstream self-test as
model/generation acceptance.
