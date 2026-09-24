# How a token flows through the engine

This page follows one chat turn from the command line to text on the screen.
Each step names the code that does the work; links point to the file, and the
name tells you what to search for. Shapes are written `[rows, columns]`. The
model is Qwen2.5-0.5B-Instruct in BF16, running on Metal with batch size one.

```text
llm-mojo chat ──► Python checks the assets ──► exec the native chat
                                                      │
 your message ──► token IDs ──► prefill: every new token through 24 layers
                                                      │
          text ◄── bytes ◄── next token ◄── logits ◄──┘
                                  │
                                  └─► decode: that token through 24 layers, repeat
```

## Words used here

| Word | Meaning here |
| --- | --- |
| Token | An integer ID for a piece of text. The vocabulary has 151,936 IDs. |
| Hidden state | The 896 numbers that represent one token between layers; one *row*. |
| Prefill | Computing all new prompt tokens in one or a few model calls. |
| Decode | A model call that processes one token, the one just chosen, to score the next; everything earlier comes from the cache. |
| KV cache | Each layer's keys and values for every token seen so far, so earlier tokens are never recomputed. |
| BF16 | A 16-bit floating-point format: the range of FP32 with 8 bits of precision. Weights, activations and caches are stored in it. |
| Kernel, command | A GPU function, and one launch of it. |
| Enqueue | Hand a command to the GPU's ordered queue without waiting for it to finish. |
| Configuration | A numbered choice of kernels for a decoder layer (0, 2, 3, 20, 21, 22 or 26). |
| Plan | An [`ExecutionPlan`](../src/llm_mojo/models/qwen2/plan.mojo): a configuration plus the single-row decode features, fixed for one call. |

## 1. From the command to the native process

[`chat`](../src/llm_mojo/cli/app.py) resolves the options with
[`resolve_run`](../src/llm_mojo/configuration.py), then
[`launch_chat`](../src/llm_mojo/runtime/launch.py) prepares the launch:

- [`verify_prepared`](../src/llm_mojo/models/qwen2/assets.py) checks the size and
  SHA-256 of all 196 prepared tensors against their manifest;
- [`ensure_prepared`](../src/llm_mojo/models/qwen2/tokenizer_assets.py) checks the
  tokenizer tables;
- [`ensure_binary`](../src/llm_mojo/runtime/build.py) reuses the compiled chat
  program, or rebuilds it if any source it imports has changed.

The launcher then calls `os.execv`, which replaces the Python process with the
native program. From here on no Python runs: the chat loop in
[`main`](../src/llm_mojo/cli/chat_cli.mojo) owns the model, the tokenizer and the
terminal.

## 2. Loading the model

[`ChatSession`](../src/llm_mojo/models/qwen2/chat.mojo) creates a
[`QwenModel`](../src/llm_mojo/models/qwen2/model.mojo), which allocates GPU
buffers and fills them with [`load_bf16`](../src/llm_mojo/models/qwen2/model.mojo),
one layer at a time through [`ModelLayer.load`](../src/llm_mojo/models/qwen2/model.mojo).

| Tensor | Shape | Size |
| --- | --- | ---: |
| Embedding, also used as the output head | [151936, 896] | 272.3 MB |
| Each layer: QKV projection and bias | [1152, 896], [1152] | 2.07 MB |
| Each layer: output projection Wo | [896, 896] | 1.61 MB |
| Each layer: MLP gate, up and down | [4864, 896] ×2, [896, 4864] | 26.1 MB |
| Each layer: two RMSNorm weights | [896] ×2 | 3.6 kB |
| All 24 layers | | 715.8 MB |
| KV cache, each layer | K and V, [4096, 128] each | 2 MiB |
| KV cache, all 24 layers | | 48 MiB |

The whole prepared model, with the final RMSNorm weights and the rotary tables,
is 989,114,112 bytes. One cached token costs 12,288 bytes of KV: 2 tensors × 24
layers × 2 KV heads × 64 values × 2 bytes. The cache holds up to 4,096 tokens.

## 3. From text to token IDs

When you press Enter, [`ChatSession.begin`](../src/llm_mojo/models/qwen2/chat.mojo)
calls [`ChatHistory.begin`](../src/llm_mojo/models/qwen2/chat.mojo). It wraps the
message in Qwen's chat markers, `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`,
and turns it into IDs with the native byte-level BPE
[`Tokenizer.encode`](../src/llm_mojo/models/qwen2/tokenizer.mojo). The IDs are
appended to the history. If the message plus the whole reply budget would not
fit in 4,096 tokens, the turn is rejected before anything changes.

The history is the source of truth. `model.length` counts how many of its tokens
are already in the KV cache; everything after that still has to be computed.

## 4. Prefill: computing the new tokens

The chat loop calls [`ChatSession.submit_next`](../src/llm_mojo/models/qwen2/chat.mojo)
until the whole history is cached. Each call takes up to 256 uncached tokens
(the chunk size) and runs [`QwenModel.forward`](../src/llm_mojo/models/qwen2/model.mojo)
with a plan from [`fast_plan`](../src/llm_mojo/models/qwen2/plan.mojo). On Apple
M4 Pro, a multi-row call uses configuration 2, 3 or 21 for the eleven measured
row and cache sizes in
[`fast_prefill_configuration`](../src/llm_mojo/models/qwen2/plan.mojo) and
configuration 0 otherwise. Other devices always get configuration 0.

For a call with R new tokens, the model:

1. **Checks everything first.** [`QwenModel.preflight`](../src/llm_mojo/models/qwen2/model.mojo)
   validates the plan, the IDs, the shapes and the cache capacity before any GPU
   work, so a bad call cannot leave half-written state.
2. **Uploads the IDs** and looks up their embedding rows with
   [`_embedding`](../src/llm_mojo/models/qwen2/model.mojo): [R] IDs become [R, 896]
   hidden states.
3. **Runs 24 decoder layers**, each through
   [`enqueue_decoder_layer_configuration`](../src/llm_mojo/layers/decoder_layer.mojo).
   Every layer does the same two steps with its own weights:

   | Attention ([`enqueue_attention_sublayer_integrated`](../src/llm_mojo/layers/attention_sublayer.mojo)) | Shape |
   | --- | --- |
   | RMSNorm of each row | [R, 896] |
   | QKV projection: 14 query heads and 2 key and 2 value heads of 64 values | [R, 896] → [R, 1152] |
   | RoPE: rotate queries and keys by their position ([`enqueue_rope_apple_gpu`](../src/llm_mojo/kernels/rope.mojo)) | [R, 14, 64], [R, 2, 64] |
   | Append the new keys and values to this layer's cache | K, V: [R, 128] |
   | Attention: each query head scores the cached keys of its KV head up to its own position (7 query heads share each KV head), with FP32 scores and softmax | [R, 896] |
   | Output projection Wo, plus the layer input (the residual) | [R, 896] |

   | MLP ([`enqueue_mlp_apple_gpu`](../src/llm_mojo/layers/mlp.mojo)) | Shape |
   | --- | --- |
   | RMSNorm | [R, 896] |
   | Gate and up projections | [R, 896] → [R, 4864] each |
   | SiLU(gate) × up | [R, 4864] |
   | Down projection, plus the attention output (the residual) | [R, 4864] → [R, 896] |

   A layer's output is the next layer's input. The decoder must not read and
   write the same buffer, so multi-row calls copy the output into the input
   buffer with [`_copy_rows`](../src/llm_mojo/models/qwen2/model.mojo).
4. **Computes the logits for the last token only.** The final RMSNorm
   ([`enqueue_rms_norm_apple_gpu`](../src/llm_mojo/kernels/rms_norm.mojo)) and the
   output head ([`enqueue_linear_apple_gpu`](../src/llm_mojo/kernels/linear.mojo),
   reusing the embedding as its weights) turn one [1, 896] row into [1, 151936]
   logits: one score per vocabulary entry.

All of this is enqueued on one ordered GPU stream; the host does not wait between
layers. Each layer's cache now holds R more tokens, and `model.length` grows by R.

## 5. Choosing the next token

[`ChatSession.sample`](../src/llm_mojo/models/qwen2/chat.mojo) calls
[`QwenModel.greedy`](../src/llm_mojo/models/qwen2/model.mojo), which waits for the
GPU and takes the highest-scoring ID from the last model call's scores. Ties go
to the lowest ID, and any non-finite score is an error rather than a guess. How
it reads the scores follows that call's plan. After a multi-row call, as at the
end of most prefills, it scans all 151,936 scores on the host. After a one-row
call on M4 Pro, the GPU has already picked the winner (section 6).
[`ChatHistory.accept`](../src/llm_mojo/models/qwen2/chat.mojo) appends the token.
A stop token ([`is_stop`](../src/llm_mojo/models/qwen2/tokens.mojo): IDs 151643 and
151645) or the reply limit ends the turn.

## 6. Decode: one token at a time

The chosen token is in the history but not in the cache, so the loop calls
`submit_next` again, now with one row. That call processes the token just chosen
and produces the scores for the next one. A reply of n tokens therefore takes
n − 1 decode calls. Its first token is chosen from the prefill's scores, and its
last one, a stop token or the token that reaches the limit, is chosen but only
processed by the next turn's prefill.

For one row on M4 Pro, `fast_plan` returns configuration 26, which fuses two pairs
of steps, together with its three decode features: residual/RMSNorm fusion,
buffer swapping and GPU argmax. Every decode call takes this route:

- **Fused kernels.** One kernel unpacks the QKV projection, applies RoPE and
  appends to the cache ([`_enqueue_fused_decode_qkv`](../src/llm_mojo/layers/attention_sublayer.mojo)),
  and one kernel computes SiLU(gate) × up.
- **Residual and RMSNorm together.** [`enqueue_residual_norm`](../src/llm_mojo/kernels/residual_norm.mojo)
  adds each residual and computes the RMSNorm that follows it in one kernel:
  48 per token, two in each layer, the last one producing the final norm.
- **Buffer swapping.** Instead of copying 896 values between layers,
  [`swap_hidden_buffers`](../src/llm_mojo/models/qwen2/model.mojo) exchanges which
  buffer is the input and which is the MLP output: 23 swaps, no copies.
- **GPU argmax.** [`enqueue_argmax`](../src/llm_mojo/kernels/token_selection.mojo)
  finds the winner on the GPU in two kernels (149 groups of 1,024 logits, then
  the group winners), with the same rule: highest BF16 score, lowest ID on ties,
  non-finite rejected. The host reads three numbers instead of 151,936 scores.

These change how work is launched, not the arithmetic: the fused route produces
the same bytes as configuration 0. `tests/test_decode_route.mojo` checks this in
default validation, and `validation.model decode-parity` checks it on the real
model. The model records what it enqueued in a
[`ForwardRoute`](../src/llm_mojo/models/qwen2/model.mojo); generate reports print
it and validation rejects a Fast decode that did not take this route.

## 7. Text out, the end of a turn, and `/reset`

[`TokenizerDecoder.push`](../src/llm_mojo/models/qwen2/tokenizer.mojo) turns each
token into bytes and holds back an incomplete UTF-8 character until the rest
arrives, so the terminal only ever prints whole characters. At the end of a turn,
[`ChatHistory.finish`](../src/llm_mojo/models/qwen2/chat.mojo) makes sure the
history ends with `<|im_end|>` and a newline, exactly as Qwen's template
expects. Your next message prefills only what is not cached yet: the reply's
last token and end markers, then the new message. The cache already holds the
rest of the conversation.

`/reset` ([`ChatSession.reset`](../src/llm_mojo/models/qwen2/chat.mojo)) waits for
the GPU, marks every cache empty and restores the system prompt. The weights stay
loaded.

## 8. Where the time goes

A calculation, not a measurement: each decode call reads every weight once,
715.8 MB of layers plus the 272.3 MB head, about 988 MB. At the 273 GB/s Apple
publishes for this chip's memory, reading it takes at least 3.6 ms. The KV cache
adds 12,288 bytes per cached token, about 50 MB at 4,096 tokens.

Measured on the reference M4 Pro:

- The Fast route streams 107–115 tokens per second after the first token, about
  9 ms for each decode call ([composed decode study](../studies/model_generation/residual-norm.md)).
- A decode call issues 245 compute commands. Submitting them takes about 7 ms,
  and about 98% of that time is inside MAX's enqueue runtime, measured with call
  recording enabled ([runtime enqueue study](../studies/model_generation/runtime-enqueue.md)).

So decode is limited by launching work, not by memory bandwidth, and the fusions
above help mostly by issuing fewer commands. Prefill is a separate cost: for a
3,839-token prompt the first token arrives after about 2.1 s, now the largest
wait a user sees.

## 9. What is exact and what is compared

Some properties must hold byte for byte: the pinned weights, the token history,
cache contents that are preserved or appended, and routes that claim to compute
the same result. Comparisons with Hugging Face, or between different prompt chunk
sizes, are diagnostic: small floating-point differences are recorded and
explained, not hidden behind a tolerance. The
[correctness and diagnostic policy](model.md#correctness-and-diagnostic-policy)
states the rules; the [study index](../studies/README.md) holds the evidence.
