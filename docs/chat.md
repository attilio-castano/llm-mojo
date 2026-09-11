# Native terminal chat

The terminal uses Qwen2.5-0.5B-Instruct, batch-one BF16 on Metal, greedy decoding
and the existing Fast dispatcher. Model weights, tokenizer and all 24 KV caches
stay resident between turns. The native session submits only tokens after the
cached prefix. Python verifies assets and builds the executable when needed,
then replaces itself with the Mojo process; there is no Python inference loop.

## Start

From the repository, using the verified prepared checkpoint from
[generation.md](generation.md#prepared-checkpoint-and-reference):

```sh
uv run --locked python -m llm_mojo.chat --prepared build/model-prepared-v1
```

The first launch compiles the chat executable. Subsequent launches reuse it when
source, lockfile and executable hashes match. Each launch verifies weights and
tokenizer tables locally. Missing assets produce an error; the launcher does not
download them. Compilation, verification and loading happen once per session.

Type one message per line. `/reset` starts a new conversation without reloading
weights; `/exit` or Ctrl-D at an empty prompt exits. Ctrl-C while typing cancels
the line. During generation, it stops at a model-call boundary and retains the
partial reply, closing the assistant turn before the next user message.
Terminal line editing and echo use the OS's normal canonical mode.

Defaults are 256 generated tokens per reply and prompt chunks of at most 256
rows. These are bounded operating defaults, not a claim of optimal chunk size.
Use `--max-new-tokens` and `--chunk-rows` to change them. A UTF-8 `--system-file`
replaces Qwen's pinned default system instruction. The interface intentionally
supports plain system/user/assistant messages, not tool calls or chat-template
extensions. Greedy generation can still produce errors or repetition.

The entire history, formatting, requested reply allowance, and turn closure must
fit within 4096 tokens. A rejected message leaves the existing history and cache
unchanged. Use `/reset`, a shorter message, or a smaller reply limit. Old messages
are not silently removed. Input lines are limited to 65536 bytes; a terminal may
have a smaller canonical input-line limit. Invalid UTF-8 input is rejected.
An execution failure invalidates the session; `/reset` is required before reuse.

## Ownership and turn boundaries

`ChatHistory` owns exact token IDs. `QwenModel.length` identifies the prefix
already submitted to all 24 caches. Every model call receives a suffix beginning
at that position and selects Fast using the actual row count and cumulative
context length. Normal generation never resets the cache between turns.

Generated assistant tokens stay as token IDs; they are never decoded and
re-tokenized to rebuild history. The displayed text uses the existing incremental
UTF-8 decoder. A stop token can remain uncached. A reply limit can leave the final
ordinary token uncached; Ctrl-C may arrive before or after it is consumed. Each
case preserves the same prefix invariant. Ending the assistant turn adds a Qwen
end marker if needed, followed by the template newline. Those tokens enter the
cache before the next user message. An interruption before prefill completes
retains the whole accepted user message and an empty/partial assistant turn.

`ChatHistory.begin` constructs and checks the complete suffix before mutation.
It reserves two closure tokens beyond the configured generation allowance.
`/reset` synchronizes and resets model lengths before restoring the system-only
history. Normal generation uses no signal handler: the Darwin process blocks
SIGINT before creating GPU threads and consumes it synchronously between calls.

## Verification and measurements

The pinned HF oracle covers seven prompt prefixes, including multiple turns,
custom and empty system messages, Unicode normalization, newlines and literal
special-token strings. Exact parity follows the pinned HF template, including
its treatment of those strings. The ordinary test workflow packs committed
fixtures without downloading weights or importing HF:

```sh
uv run --locked python tests/fixtures/chat_reference.py --pack
uv run --locked mojo run -I src -I tests tests/test_chat.mojo
```

Regenerate and compare the independent oracle with existing local pinned assets:

```sh
uv run --locked --script tests/fixtures/chat_reference.py
```

The explicit checkpoint study requires clean source and new output paths:

```sh
uv run --locked python tests/chat_study.py --prepared build/model-prepared-v1 --output build/chat-study-repeat
```

It compiles and records exact native executables, checks preserved prefixes and
inactive capacity across all caches, compares logits with full-history replay,
and exercises actual terminal Ctrl-C, continuation, reset and EOF through a
controlling pseudo-terminal. Numerical differences remain diagnostic; token
accounting, finite outputs, cache preservation and reset replay are exact checks.

For a session report, add `--report build/my-chat.tsv` with a new output path.
The optional report contains conversation token IDs, cache counters and timing
observations. First visible text is timestamped after terminal output is flushed,
starting before formatting/tokenizing the user message. Native loading is
separate; compilation and Python asset verification are excluded. Per-token
observations include greedy readback and stream decoding, not only GPU forward
time. Instrumented observations are distinct from the paired study's synchronized
forward-only measurements. Reporting is optional and adds collection overhead.

The [completed chat study](../studies/model_generation/chat.md) retains numerical
diagnostics, exact cache checks, terminal execution and timing observations.
