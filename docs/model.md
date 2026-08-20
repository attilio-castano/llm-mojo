# Initial model contract

## Target

The first end-to-end model target is
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

## V0 runtime boundary

V0 is deliberately narrower than the model's complete advertised capability:

- BF16 weights and activations, with wider accumulation only where explicitly
  documented;
- batch size 1;
- at most 4,096 live session tokens, including all prompt and generated tokens;
- system, user, and assistant chat roles without tool calls;
- deterministic greedy decoding before probabilistic sampling;
- full prefill for the first turn;
- incremental prefill for later user turns;
- one-token autoregressive decode with a persistent KV cache;
- token IDs supplied by reference tooling rather than a Mojo tokenizer.

The 4,096-token limit is a V0 engineering boundary, not a statement about the
model's 32,768-token context capability. At batch size 1, the unpadded BF16 KV
payload is 12,288 bytes per token and 48 MiB at the V0 limit:

```text
2 tensors × 24 layers × 2 KV heads × 64 values × 2 bytes = 12,288 bytes/token
```

Allocator overhead, alignment, padding, and temporary buffers must be measured
separately rather than folded into that theoretical payload.

V0 defines no separate reasoning channel or thinking-mode protocol. Any
rationale the model emits is ordinary assistant-token output and follows the
same autoregressive path as any other response.

## Conversation semantics

The model is stateless. A session consists of the canonical token history,
generation state, and the KV entries derived from that history.

Reference tooling owns chat-template rendering and tokenization in V0. Fixtures
must use an explicit system message, the pinned `tokenizer_config.json` chat
template, and `add_generation_prompt=true`. The default template behavior must
not be allowed to introduce an implicit system message unnoticed.

The tokenizer's chat end token is `<|im_end|>` (`151645`). The official
generation configuration treats both `151645` and `<|endoftext|>` (`151643`) as
stop IDs. Stop tokens remain part of the canonical token history.

For each later user turn:

1. Render and tokenize the complete updated transcript with the reference
   tooling.
2. Verify that the cached token history is an exact prefix of that transcript.
3. Prefill only the new suffix while attending to the existing KV cache.
4. Decode the assistant response one token at a time and extend both token
   history and cache.

If the prefix check fails, the engine must invalidate and rebuild the cache. It
must never assume that independently tokenizing only the new text preserves the
same token boundary.

## V0 correctness acceptance

V0 is complete only when all of the following are reproducible from documented
commands and fixtures:

1. **Artifact validation:** all required artifacts match the pinned revision and
   checksums, and every loaded tensor has the expected name, dtype, shape, and
   byte count.
2. **Reference forward pass:** embeddings, every decoder block boundary, final
   normalization, and output logits match an independently executed reference
   oracle within predeclared absolute and relative tolerances. Tolerances and
   accumulation dtypes belong in the fixture manifest and may not be selected
   after observing the Mojo result.
3. **Uncached generation:** for fixed prompts with an adequate top-logit margin,
   greedy token selection matches the reference oracle. Logit parity remains
   the authoritative result when an argmax is numerically ambiguous.
4. **Cached generation:** prefill plus one-token cached decode matches full
   uncached recomputation at every generated position within the declared
   tolerance. Instrumentation must demonstrate that cached prefixes were not
   recomputed.
5. **Multi-turn generation:** a fixture with an explicit system message and at
   least three user turns produces the same per-position logits and token
   history through incremental prefill as full-transcript recomputation.
6. **Cache accounting:** cache shapes, positions, logical length, allocated
   bytes, and reset behavior agree with the documented layout and context
   limit.
7. **Execution identity:** correctness and performance records identify the
   commit, model revision, hardware, software, dtype, tensor shapes, and actual
   runtime device and backend.

V0 has no performance threshold. Optimization begins only after this reference
contract passes. Later work can add sampling, longer contexts, quantization,
batching, tool-oriented templates, and additional model families without
changing what V0 established.
