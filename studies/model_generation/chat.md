# Native terminal chat with persistent KV caches

Completed locally on 2026-09-11. The terminal supports consecutive
Qwen2.5-0.5B-Instruct messages with one resident Mojo model, Fast dispatch and persistent KV
caches. `/reset`, `/exit`, Ctrl-C, EOF, context rejection and UTF-8 streaming are
implemented. See [the launch instructions](../../docs/chat.md).

## What was established

Seven prompt prefixes match the pinned HF chat template exactly. Fixtures cover
multiple turns, default/custom/empty system messages, Unicode, newlines and
special-token strings. Generation history retains actual token IDs; it does not
round-trip assistant text through tokenization. The native session tracks the
submitted prefix and consumes any pending final token plus turn closure before
processing the next user message.

The full-checkpoint driver exercises three turns on Apple M4 Pro / Metal and
retains 144 observations across all 24 layers' K/V caches. Every previously
cached prefix and inactive suffix is preserved byte-for-byte; every observed
cache value is finite. Reset reproduces the first prompt's logits byte-for-byte.
The driver also checks context rejection, invalid-session rejection, reset
recovery, and interruption before any prompt submission.

At each turn's first prediction, cached-suffix logits are compared with native
full-history replay on identical token history. All three selected token IDs
agree. The first prompt's logits are byte-identical; later maximum absolute
logit differences are 0.265625 and 0.3125, with KL divergences 0.000348 and
0.001534 nats. These are numerical diagnostics, not global tolerance gates, and
this comparison is against native replay rather than HF. See
[all three numerical observations](chat-diagnostics.csv).

Four piped turns and three controlling-terminal turns check actual interaction:
name retention, repeated reply after reset, invalid UTF-8, context overflow,
Ctrl-C while typing and during generation, continuation after interruption,
reset and Ctrl-D. Reporting on/off gives identical visible output on the piped
conversation. A separate public-launch smoke verifies asset checks, compilation,
native process replacement, and another two-turn conversation. All observed
terminal history and submission counters satisfy the cached-prefix invariant.

## Performance observations

These paired measurements compare cached-suffix prefill with recomputing the
whole conversation. Both arms use the existing Fast selector, BF16 storage,
the actual 24 learned layers, capacity 512 and maximum chunk size 256. The
public terminal instead allocates capacity 4096. Four blocks reverse arm order
in the middle two blocks; each arm receives three warmups and five retained
samples per block. The total is 120 retained timings.

| Turn | Already cached / prompt tokens | Cached suffix | Full-history replay |
| --- | --- | ---: | ---: |
| First | 0 / 37 | 26.46 ms | 25.62 ms |
| Second | 46 / 61 | 24.30 ms | 32.08 ms |
| Third | 66 / 87 | 21.84 ms | 44.68 ms |

Each time is the median of the four block medians. The median paired cached/full
ratios are 1.021, 0.756 and 0.491. The first turn has no prior cache to reuse;
the two follow-up observations reduce prefill time by approximately 24% and 51%.
These are bounded examples, not a universal speedup or chunk-size optimum.
[Raw-derived paired table](chat-forward.csv).

The timer covers token upload, model forward and final device synchronization.
Weight loading, allocation, logical reset, prior-prefix preparation, captures,
and greedy readback are excluded. Cached samples restore the logical prefix
and overwrite its suffix; full-history samples begin with empty logical caches.
Diagnostic captures are outside timing. Source, raw samples, machine/software
identity and before/after conditions are retained.

Actual terminal observations include formatting, tokenization, greedy readback
and output streaming. First visible text took 48–58 ms on the first turns after
native loading, and 22–27 ms on the subsequent or reset turns. The six completed
replies generated approximately 62–73 output tokens/second after the first token.
This rate is computed between the first and last non-stop token timestamps;
it excludes initial response latency and the stop token. Native loading was
about 1.6 seconds in the piped run, separate from Python verification and
compilation. These small instrumented observations describe the tested machine
and prompts; they are not an HF speed comparison. See
[terminal timing observations](chat-terminal.csv).

## Evidence and validation

Collection used clean source `cd14983`. The [archive manifest](chat-study.json)
binds the complete [compressed observations](chat-study.json.gz), approximately
41 KB, including both executable receipts, all 120 timing samples, numerical
metrics, all 144 cache observations, raw-capture hashes, terminal token histories,
stream events and public-launch evidence. Checkpoints, binaries and full cache
arrays remain in ignored `build/`. The three CSV tables regenerate offline:

```sh
uv run --locked python studies/model_generation/summarize.py
```

Validation passed 153 Python tests, native chat framing/state tests, model
primitive tests and both complete tokenizer parity suites (ordinary and Unicode,
including streaming). The full-checkpoint chat study and actual public command
also passed. Replay regressions reject altered cache flags and omitted timing
samples even when archive hashes are updated to match the damaged records.

The broader validation command was intentionally stopped during the unchanged
decoder-policy sweep after fixture preparation, Python tests and preceding
attention/chat suites. It is not reported as an unfiltered full-suite pass. The
focused suites and explicit checkpoint collection above define this milestone's
validation scope; no kernel arithmetic, precision or context capacity changed.

Fresh collection commands and terminal controls are in
[docs/chat.md](../../docs/chat.md#verification-and-measurements). Reproduction
requires a clean checkout, pinned local assets and a new output directory.
