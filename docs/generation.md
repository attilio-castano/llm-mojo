# Qwen model composition and generation

For interactive multi-turn use with persistent KV caches, see [terminal chat](chat.md).

The completed [Fast implementation revision](history/fast-generation-plan.md) uses numerical
comparisons as diagnostics. Required checks cover assets, data flow, cache and
generation semantics; historical full-model distance ceilings no longer block
integration. `fast` is the public default. On Apple M4 Pro it selects the eleven
prefill workload cells described in the
[runtime study](../studies/model_generation/runtime.md) and configuration 26 for
single-row calls, plus residual/RMSNorm fusion, buffer swapping and GPU argmax
following the [composed study](../studies/model_generation/residual-norm.md).
Other workloads and devices retain configuration 0. The `baseline` and
historical `consistent` routes, and explicit retained configurations for
diagnostics, remain available. The full-checkpoint generator, lifecycle checks,
numerical diagnostics and paired model measurements have executed. Numerical
differences are retained and explained separately from exact implementation
invariants.

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

`ExecutionPlan` in `models/qwen2/plan.mojo` fixes how one model call runs: a
decoder configuration plus the single-row decode features. Every model client
builds its plans there. `fast` (the public default) uses the measured M4 Pro
lookup: split8 configuration 2 at
16/1024, 16/4096, 15/256 and 17/256; configuration 3, combining split8 and
larger projections, at 64/1024, 64/4096, 256/1024, 256/4096, 65/4096 and
255/4096; configuration 21 at 16/256. Pairs denote incoming rows / total
cached rows. Single-row M4 Pro calls use configuration 26: exact QKV/RoPE/cache
fusion plus SiLU/multiply fusion, always together with residual/RMSNorm fusion,
inter-layer buffer swapping and separate GPU argmax. The composed route
passed paired whole-token gates at histories 64, 1024 and 3968. Every other shape and device name falls
back to configuration 0.
Baseline 0 already includes integrated attention and optimized multi-row MLP
7, with rowwise MLP projections for decode. Configuration 26 preserves those
projection and attention reductions and the intermediate BF16 activation rounding.

`baseline` always selects 0. The historical `consistent` / 20 route uses FP32
G32 attention and rowwise projections at all row counts; it remains an explicit
research mode. Diagnostic drivers also accept an explicit retained configuration
(0, 2, 3, 20, 21, 22 or 26). The same selected configuration applies to all 24
layers for a call.

A plan cannot express the unpromoted compositions from the decode studies:
configuration 26 always carries all three decode features and exactly one row,
and no other configuration carries any. Those study arms, configuration 25 and
the `auto`, `candidate`, `unfused`, `fusion`, `combined` and per-arm policies
exist through `edb610a`. Default profiling follows current Fast; historical
trace replay uses the route recorded in each capture's provenance.

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
uv run --locked --script tests/fixtures/model_reference.py prepare --output build/model-prepared-v1
```

Preparation requires a new output directory. It does not run the historical
qualification workflow. For the complete interactive setup, see the
[README quickstart](../README.md#run-the-chat).

Under the historical qualification workflow, failure prevents dependent comparison.
The new `model_reference.py diagnose` / `model_validation diagnose` workflow
captures corresponding histories without requiring qualification; it retains
numerical distances separately from required exact storage checks. The historical
`model_reference.py qualify` command is expected to reproduce its original
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

Run it through the public command:

```sh
uv run --locked llm-mojo generate --mode fast --prompt-file "$PROMPT_FILE" --max-new-tokens 16 --report build/generation-events.tsv
```

`PROMPT_FILE` contains UTF-8 prompt text; `--prepared` selects a prepared model
directory other than the checkout's default. `--mode baseline` and
`--mode consistent` run the reference routes. The launcher verifies the checkpoint identity, manifest,
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
into the KV cache. This plain-text driver does not apply a chat template; [terminal chat](chat.md)
provides native framing and persistent multi-turn state.

The optional TSV report records the mode, prompt/generated IDs, selected prefill
configurations, one route record per model call, native initialization, prefill
and decode durations, and cache accounting. A route record is what the model
enqueued (`ForwardRoute`): the configuration, the fused residual/RMSNorm steps,
buffer swaps versus copies, whether the final RMSNorm ran separately, and GPU
argmax. In Fast mode on M4 Pro every decode call must report configuration 26
with 23 swaps and GPU argmax, so a silent fallback to the baseline route cannot
pass `validation.model generate`. Native durations exclude Python asset verification and compilation.
TTFT includes native initialization/tokenization; decode timings end at device
synchronization. Output remains generated UTF-8 text on stdout.

Full-checkpoint execution, lifecycle checks, six 32-token generation runs and
a public default-Fast smoke with a 1024-token prompt passed. Same-history
predictions agree with HF on 191 of 192 generated choices; the exception is an
exact HF top-logit tie. These are bounded development observations, not a
general model-quality or exact trajectory-equivalence claim.

`decode-parity` checks the Fast decode route against baseline on the real
24-layer model:

```sh
uv run --locked python -m llm_mojo.validation.model build --binary build/parity-model
uv run --locked python -m llm_mojo.validation.model decode-parity --binary build/parity-model --output build/decode-parity.json
```

Both runs prefill the same 53 fixed tokens with configuration 0 and then decode
32 fixed tokens one row at a time, so every call sees the same input. Every
call's hidden states, final norm, logits and K/V must be byte-identical. The
receipt keeps hashes only. `tests/test_decode_route.mojo` runs the same
comparison on three synthetic layers in default validation.

## History

The numerical-policy studies that preceded the diagnostic policy, from the
stopped Fast qualification to the consistency candidate and the HF/PyTorch
backend study, are recorded in [history](history/numerical-policy-studies.md).
