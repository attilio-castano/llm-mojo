# Qwen model contract

## Target

The native Fast runtime and terminal chat use
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
at immutable Hugging Face revision
`7ae557604adf67be50417f59c2c2f167def9a775`.

The model is an Apache-2.0-licensed, instruction-tuned, decoder-only
transformer. Its source weights are BF16. The initial engine must preserve that
dtype; quantization is a later optimization and is not part of the reference
path.

The relevant architecture is:

- 0.49 billion parameters, including embeddings;
- 24 decoder layers;
- hidden size 896 and SwiGLU intermediate size 4,864;
- 14 query heads and 2 key/value heads, with head dimension 64;
- RoPE with theta 1,000,000;
- RMSNorm with epsilon `1e-6`;
- tied token embeddings and LM head;
- vocabulary size 151,936;
- model context limit 32,768 tokens with sliding-window attention disabled.

## Artifact provenance

Weights and tokenizer assets remain external to this repository. Download them
from the pinned revision above and verify these SHA-256 digests before producing
fixtures or running the engine:

```text
fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe  model.safetensors
18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45  config.json
e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6  generation_config.json
c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539  tokenizer.json
5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583  tokenizer_config.json
599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3  merges.txt
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json
```

The safetensors artifact is 988,097,824 bytes. Do not commit model weights,
download caches, or generated full-model artifacts. Small test fixtures must
record the source revision, source tensor names, extraction procedure, oracle
versions, dtype, shapes, and checksums.

The bounded first-layer attention workflow also supports a separately hashed
prefix containing the original header, embeddings and complete required
tensors. That workflow verifies the prefix identity and explicitly records
that the full-file digest was not verified. See the
[attention fixture provenance](../studies/attention_sublayer/numerics.md#reproduction).

## Current runtime boundary

Qwen's configuration, weights and official implementation define model
semantics. Numerical compatibility also requires a named precision policy and
an independently executed upstream comparison on a recorded backend, device
and dtype. The independent NumPy implementations
are mathematical and rounding diagnostics; disagreement with them is not, by
itself, proof of a Mojo defect. The upstream eager, SDPA and Flash Attention
paths can also differ numerically. A pinned eager CPU reference is reproducible
compatibility evidence, not a claim to reproduce the original training kernels.
Agreement among Mojo variants is useful regression evidence but cannot replace
an independent comparison. The attention sublayer's selected CPU baseline
explicitly casts SDPA inputs to FP32 and its output to BF16. The Mojo sublayer
defaults to matching that policy, with FP32 scores/probabilities/accumulation
and BF16 operands/cache/output. The standalone BF16 GQA paths below retain
their own contracts and remain named compatibility comparisons.
The composed attention study describes its
[reference hierarchy and validation boundaries](attention-sublayer.md).
The [decoder-layer contract and fixture specification](decoder-layer.md)
defines the composition gate for attention followed by MLP. The layer has
passed its numerical acceptance and workload selection study. The completed
[Fast runtime](../studies/model_generation/runtime.md) composes all 24 layers;
the [terminal chat](chat.md) adds persistent multi-turn sessions. Full-model
numerical comparisons are diagnostic under the policy below. The historical
[consistency investigation](../studies/model_generation/consistency.md) remains
incomplete and is a separate follow-up.

The implemented runtime is deliberately narrower than the model's complete
advertised capability:

- BF16 weights and activations, with wider accumulation only where explicitly
  documented;
- batch size 1;
- at most 4,096 live session tokens, including all prompt and generated tokens;
- system, user, and assistant chat roles without tool calls;
- greedy decoding with lowest-token-ID tie breaking;
- first-turn prefill, optionally split into chunks;
- incremental prefill for later user turns;
- one-token autoregressive decode with a persistent KV cache;
- native Mojo tokenization, Qwen chat framing and streaming UTF-8 decoding.

Fast dispatch is the default. It selects measured M4 Pro configurations by
workload and retains the optimized baseline elsewhere; see the
[runtime policy](generation.md#workload-policy). Fixed greedy tie breaking does
not guarantee identical predictions across different prefill chunk schedules.

The 4,096-token limit is a V0 engineering boundary, not a statement about the
model's 32,768-token context capability. At batch size 1, the unpadded BF16 KV
payload is 12,288 bytes per token and 48 MiB at the V0 limit:

```text
2 tensors × 24 layers × 2 KV heads × 64 values × 2 bytes = 12,288 bytes/token
```

Allocator overhead, alignment, padding, and temporary buffers must be measured
separately rather than folded into that theoretical payload.

## Operation arithmetic

The V0 arithmetic below defines the inspectable operation references. Optimized
compositions retain their own documented rounding boundaries and independent
tests; the standalone materialized GQA reference is not the Fast attention path.

### RMSNorm arithmetic

For each hidden row, V0 follows the
[Qwen2 reference operation from Transformers 4.43.1](https://github.com/huggingface/transformers/blob/v4.43.1/src/transformers/models/qwen2/modeling_qwen2.py)
in this exact order:

1. Promote the BF16 activation row to FP32.
2. Compute the mean of the squared FP32 values over the hidden axis.
3. Add epsilon `1e-6` and compute the reciprocal square root in FP32.
4. Multiply the FP32 row by that reciprocal root.
5. Cast the normalized row to BF16.
6. Multiply by the BF16 RMSNorm weight, producing BF16 output.

The cast before the weight multiplication is part of the reference contract.
An implementation with a different cast point or reduction order is a distinct
numerical path and must be compared under a predeclared tolerance. Tensor and
execution mappings use the project's [layout language](layouts.md).

### Attention projection arithmetic

Qwen applies separate query, key, and value affine projections after the
attention-input RMSNorm. For `R` token rows and hidden width `H = 896`, the
pinned source operations have these shapes:

```text
Q_states = linear(X[R, 896], Wq[896, 896], Bq[896]) -> [R, 896]
K_states = linear(X[R, 896], Wk[128, 896], Bk[128]) -> [R, 128]
V_states = linear(X[R, 896], Wv[128, 896], Bv[128]) -> [R, 128]
```

The weight orientation is `(output_features, input_features)`, matching the
pinned `torch.nn.Linear` tensors without a runtime transpose. Query, key, and
value each include a BF16 bias. The V0 reference operation computes each output
element by accumulating BF16 input and weight products in FP32 over the input
axis, promotes and adds the BF16 bias in FP32, then casts the result once to
BF16:

```text
acc = 0.0f32
for k in 0 .. input_features:
    acc += f32(X[row, k]) * f32(W[output_feature, k])
Y[row, output_feature] = bf16(acc + f32(B[output_feature]))
```

The host reference uses the displayed serial reduction order. GPU reductions
may use a different FP32 association and must match the pinned oracle under the
predeclared tolerance. The reference keeps Q, K, and V as three source-compatible
operations. The Fast runtime uses the validated packed projections described in
the [attention study](../studies/attention_sublayer/README.md). Bias-free
projections, including the attention output projection, have explicit paths
without an implicit zero-bias allocation.
Tensor and execution mappings use the project's
[layout language](layouts.md#affine-linear-projection-v0).

### RoPE arithmetic

V0 applies rotary position embeddings to query and key heads after their
linear projections and before rotated keys enter the KV cache. Values do not
receive RoPE. The operation itself owns neither the KV cache nor position
history; its caller supplies the absolute position of the first input row.

For Qwen head dimension `D = 64`, each dimension `i` in the first half pairs
with `j = i + D / 2` in the second half. This is the half-split permutation in
the pinned Transformers 4.43.1 `rotate_half` operation, not adjacent even/odd
pairing. All 64 dimensions participate.

The rotary table uses theta `1,000,000`. Frequencies, cosine, and sine are
formed in FP32, the full duplicated length-`D` cosine and sine rows are cast to
BF16, and table application follows the pinned eager BF16 operation. For input
row `r`, absolute position `p = start_position + r`, and paired dimensions
`i` and `j`, V0 materializes both BF16 products before the final BF16
subtraction or addition:

```text
Y[r, n, i] = bf16(
    bf16(X[r, n, i] * C[p, i]) - bf16(X[r, n, j] * S[p, i])
)
Y[r, n, j] = bf16(
    bf16(X[r, n, j] * C[p, j]) + bf16(X[r, n, i] * S[p, j])
)
```

The baseline interface specializes the V0 batch-one contract to contiguous
positions through `start_position`; it does not yet accept an arbitrary
position-ID tensor or generate the cosine/sine table. Query and key tensors
use the same operation despite having 14 and 2 heads respectively. Tensor and
execution mappings are recorded in the project's
[layout language](layouts.md#rope-v0).

### Grouped-query attention arithmetic

The initial attention operation begins after RoPE. Its caller supplies rotated
queries and the full active key/value prefix, including the `R` positions being
processed now:

```text
Q[R, Nq, D]  BF16
K[T, Nkv, D] BF16
V[T, Nkv, D] BF16
O[R, Nq, D]  BF16
```

Here `T` is the active key/value length and `R` is the suffix of query rows, so
`1 <= R <= T` and `past = T - R`. Full prefill has `R = T`, incremental
prefill has `1 < R < T`, and one-token decode has `R = 1`. For Qwen,
`Nq = 14`, `Nkv = 2`, and `D = 64`. Seven consecutive query heads share each
key/value head without physically repeating K or V:

```text
group_size = Nq / Nkv = 7
kv_head(query_head) = query_head / group_size
```

Query row `r` represents active position `past + r`, so its inclusive causal
key range is `0 .. past + r`. Within that range, V0 performs:

```text
score[r, qh, k] = bf16(
    sum_d(f32(Q[r, qh, d]) * f32(K[k, kv_head(qh), d]))
    / sqrt(D)
)
probability[r, qh, :] = bf16(
    stable_softmax_f32(score[r, qh, 0 .. past + r])
)
O[r, qh, d] = bf16(
    sum_k(
        f32(probability[r, qh, k])
        * f32(V[k, kv_head(qh), d])
    )
)
```

For `D = 64`, the scale is `1 / 8`. Stable softmax subtracts the maximum
visible BF16 score before exponentiation, accumulates its denominator in FP32,
and writes exact zero to causally masked positions. The initial implementation
materializes scores and then probabilities in the same caller-owned BF16
scratch tensor `[R, Nq, T]`; the cast points on either side of softmax are part
of this baseline's numerical contract.

This operation owns no projections, RoPE application, KV-cache mutation,
attention output projection, padding mask, dropout, or sliding-window policy.
The caller owns Q, K, V, scratch, and output storage. In particular, an
asynchronous GPU caller must retain those allocations until the enqueued work
has completed, and the five tensor views must not overlap. Storage and
execution mappings are recorded in the project's
[layout language](layouts.md#grouped-query-attention-v0).

Explicit optimized Qwen implementations and their intermediate rounding and
workspace contracts are documented in the [GQA decode](../studies/gqa_decode/README.md)
and [GQA prefill](../studies/gqa_prefill/README.md) studies. The materialized
baseline above remains the inspectable reference path.

V0 defines no separate reasoning channel or thinking-mode protocol. Any
rationale the model emits is ordinary assistant-token output and follows the
same autoregressive path as any other response.

### MLP sublayer specification

The [MLP numerical contract and upstream fixture specification](mlp-sublayer.md)
defines post-attention RMSNorm, SwiGLU, and the second residual boundary.
The upstream fixtures and numerical budgets are frozen. The materialized Mojo
baseline, tiled prefill projections, and bounded decode experiments have passed
their numerical checks and are documented in the [MLP study](../studies/mlp_sublayer/README.md).
The standalone MLP defaults to rowwise mapping 0. The Fast model uses tiled
mapping 7 for multi-row calls and mapping 0's projections for decode. Its
single-row M4 Pro route now fuses SiLU and multiply while retaining the exact
intermediate BF16 rounding; see the
[combined fusion study](../studies/model_generation/combined-fusion.md).
The earlier MLP projection candidates did not qualify for promotion. The accepted
[decoder composition](../studies/decoder_layer/selection.md) combines attention
and MLP with workload-specific configurations, now integrated into the
[complete model](generation.md).

## Conversation semantics

The model is stateless. A session consists of the canonical token history,
generation state, and a KV cache with an explicit logical `cache_length`.
`cache_length` is the number of leading history tokens whose key/value entries
have been materialized, so the invariant is:

```text
0 <= cache_length <= len(token_history)
```

Greedy selection appends the selected token to canonical history before it is
used as the next model input. At a generation boundary, the history may
therefore be one token longer than the cache, including when the sampled token
is a stop ID or generation ends at a token limit. V0 derives reusable prefixes
from `cache_length`; it does not assume that all canonical history is cached.

Native `ChatHistory` owns system/user/assistant framing and token history.
Independent HF fixtures verify seven prompt prefixes against the pinned
`tokenizer_config.json` template, including its generation prompt, default
system instruction and explicit custom/empty system messages. Generated token
IDs remain authoritative; displayed assistant text is never re-tokenized.

The tokenizer's chat end token is `<|im_end|>` (`151645`). The official
generation configuration treats both `151645` and `<|endoftext|>` (`151643`) as
stop IDs. Stop tokens remain part of the canonical token history and become
cached only after they have been processed as model input.

For each later user turn:

1. Construct the Qwen-framed user suffix and assistant generation prompt with
   the native tokenizer. Check the complete request and reply allowance before
   mutating history or submitting model work.
2. Preserve the invariant that the cache represents the leading `cache_length`
   tokens of canonical history.
3. Submit the uncached suffix, including any pending final assistant token,
   end marker and newline from the preceding turn, followed by the new prompt.
4. Decode one token at a time, retaining each selected token ID and tracking
   separately whether it has been submitted to the model.

Ending a reply appends an end marker if needed and the template newline. Those
closure tokens can remain uncached until the next turn. Context rejection leaves
history and caches unchanged; execution failure invalidates the session until
reset. `/reset` synchronizes and clears the conversation and logical cache lengths
while retaining loaded weights and the system instruction. See
[chat ownership and turn boundaries](chat.md#ownership-and-turn-boundaries).

## Correctness and diagnostic policy

The completed Fast milestone requires reproducible evidence for:

1. **Artifact validation:** pinned identities and every loaded tensor's name,
   dtype, shape and byte count.
2. **Architecture and execution:** embeddings, all 24 learned layers, final
   normalization, tied LM head, causal positions and finite outputs.
3. **Local numerical contracts:** independent operation and composition tests
   under their declared arithmetic policies and tolerances.
4. **Cache and lifecycle invariants:** exact preserved prefixes, appended storage,
   inactive guards, logical lengths, submission accounting, reset and invalid-input
   handling. Previously cached tokens must not be recomputed during normal turns.
5. **Generation and chat semantics:** greedy tie handling, nonfinite rejection,
   stop/context limits, exact template fixtures, authoritative token history,
   UTF-8 streaming and terminal interruption/continuation behavior.
6. **Evidence identity:** source, executable, model revision, device/backend,
   software, dtype, shapes and measurement boundaries.

HF comparisons and comparisons between native chunk schedules record numerical
distances, output distributions, same-history token choices and independent
trajectories. They diagnose discrepancies rather than apply a global full-model
closeness threshold. An exact token-ID match is distinct from approximate logit
agreement; neither establishes byte-equal stored tensors. Suspicious differences
are investigated on identical operands without retuning historical tolerances.

The [runtime study](../studies/model_generation/runtime.md) and
[chat study](../studies/model_generation/chat.md) define the completed evidence
scope and observed differences. Full-model schedule determinism and a matched HF
performance comparison remain follow-ups. The original tolerance-gated V0
qualification and consistency plans are retained in
[generation-plan.md](generation-plan.md), with their failures in the
[numerical history](../studies/model_generation/README.md). Those failed results
remain unchanged; this policy does not claim that their gates passed.


### GPU selection experiment

The [token selection study](../studies/model_generation/token-selection.md)
compares configuration 26 with CPU greedy, a separate GPU argmax, and a fused
vocabulary projection/local argmax. Both GPU routes preserve rounded BF16
scores, lowest-ID ties and rejection of any nonfinite score. The fused route
leaves logits untouched except during explicit diagnostic materialization.
Neither candidate passed the full promotion rule, so Fast/auto retain CPU
selection. Native study policies `gpu-argmax` and `fused-head` enable these
experiments only for single-row Apple M4 Pro calls.
