# Attention sublayer contract

This study composes input RMSNorm, Q/K/V affine projections, RoPE, one layer's
KV cache, causal GQA, bias-free output projection and residual addition. The
target is batch-one BF16 Qwen2.5-0.5B-Instruct: H=896, Nq=14, Nkv=2, D=64,
and at most 4096 live positions. Deterministic synthetic fixtures exercise
numerical sensitivity. A separate checkpoint experiment uses first-layer
weights and token embeddings. Neither is full-model or training-kernel evidence.

## Arithmetic and storage

For R new rows and P cached rows, T=P+R. The caller supplies X[R,H] and
the absolute positions are P through T-1. Surrounding-operation rounding
contracts in model.md apply. Normalization rounds before applying its weight;
projections accumulate FP32 and round BF16; RoPE materializes BF16 products.
The materialized reference enqueue defaults to route 3. Both that route and
the integrated enqueue keep QK reduction, scaled scores, softmax probabilities
and PV accumulation FP32, then round GQA output to BF16 before Wo. Explicit
routes 0-2 retain their historical BF16 policies.

The attention result A[R,Nq,D] has row-major storage and is viewed as [R,H]
without a copy. Projection B=BF16(A @ Wo.T) is rounded before
Y=BF16(X+B). A fused final write must preserve both rounding points.

Weights are Wqkv[H+2*Nkv*D,H], bqkv[H+2*Nkv*D], Wo[H,H], and norm[H].
Their source regions retain Q, K, V ordering. A packed row-major projection
uses an explicit bit-preserving unpack for contiguous Q/K/V consumers.
`qkv_mapping=1/2` selects existing packed rowwise/MMA kernels; zero retains
the three separate rowwise projections. The packed row stride is H+2*Nkv*D,
not H or Nkv*D. The layout copy adds no arithmetic or rounding.
K/V caches have fixed [capacity,Nkv,D] storage; only [0,T) is visible.
Rotated K and unchanged V enter [P,T); prefix entries never change.
R must be positive; overflow and invalid routes fail before any enqueue.
Reset changes logical length after outstanding work has completed.

The layer owns reusable device buffers. The caller initializes weights and
BF16 cosine/sine[capacity,D] tables before enqueue. Tables are explicit model
inputs, like weights. The compatibility baseline uses upstream-derived tables;
reproducing a particular CPU trigonometric library is a separate question.
Construction, upload and allocation stay outside enqueue and timing.

Enqueue allocates and synchronizes nothing. Callers retain the layer, input
and context until completion, and submit successive calls on the same stream.
Outputs are overwritten by the next invocation. Input must not overlap writable
workspace/output. Logical cache length records successfully enqueued positions;
an asynchronous device failure invalidates the session. Reset synchronizes.
No Python runs inside the inference enqueue.

## Reference authority and numerical gates

The selected accuracy baseline is the pinned CPU `Qwen2SdpaAttention` with
explicit FP32 Q/K/V and mask at the SDPA boundary, and BF16 output before Wo.
The independent FP64 calculation remains a diagnostic. This is the agreed
inference precision policy; it does not claim to reproduce training kernels.
The FP32 operation gate is atol=rtol=0.0078125, and projected/final composition
gates remain 0.03125. Exact cache and elementwise checks remain strict.

Operation gates give each Mojo operation the tensors consumed by upstream.
Composition gates feed original X through the whole block and check both the
projected branch and final output; a large residual cannot hide a branch error.
RoPE application and residual addition match exactly for these BF16 fixtures.

Cache append copies rotated K and raw V exactly. Tests require an unchanged
prefix, untouched poisoned capacity, full/chunked execution, nonzero positions,
reset, overflow rejection before enqueue, and repeated asynchronous decode.

Ordinary validation reports historical BF16 eager discrepancies while requiring
finite outputs and exact cache behavior; FP32 comparisons gate attention
accuracy. Standalone BF16 kernel tests keep their original requirements. See
[the numerical history](../studies/attention_sublayer/numerics.md) to reproduce
the strict BF16 compatibility failures and understand the reference decision.

## Implementation and reproduction

The public API lives in `src/llm_mojo/attention_sublayer.mojo`:
`AttentionWeights`, `AttentionCache`, `AttentionWorkspace`, and
`enqueue_attention_sublayer_integrated`. The caller owns buffers and chooses
mappings explicitly. The integrated default uses packed 8×16 QKV/Wo for R≥16
and rowwise projections below that threshold. GQA mapping 4 selects split8;
projection mapping 5 selects both 16×16 projections. Their composition is the
only additional nonzero mapping pair enabled by the latest experiment.

```sh
uv run --locked llm-mojo-validate
uv run --locked --script tests/fixtures/generate.py attention_sublayer
uv run --locked --script tests/fixtures/generate.py attention_precision
```

The shared script lock pins NumPy 1.26.4, Torch 2.4.0 and Transformers 4.43.1.
Generated arrays, binaries and checkpoint assets remain in ignored `build/`.
Checkpoint validation is explicit and reuses verified local assets:

```sh
uv run --locked --script tests/fixtures/generate.py attention_checkpoint -- --attention-prefix
MODULAR_DEBUG=device-sync-mode uv run --locked mojo run -D PRECISION_CHECKPOINT=1 -I src -I build -I tests tests/test_attention_precision.mojo
```

See the [study overview](../studies/attention_sublayer/README.md) for current
results, [numerics](../studies/attention_sublayer/numerics.md) for historical
compatibility failures and checkpoint provenance, and
[experiment plans](../studies/attention_sublayer/plans.md) for the predeclared
budgets. The [complete results](../studies/attention_sublayer/experiments.md)
retain all comparisons, including unsuccessful candidates.
