# Native Fast Qwen runtime

The implementation milestone composes the existing Mojo tokenizer, 24 distinct
Qwen decoder layers and caches, final RMSNorm, tied head and greedy generation
on Metal. Batch size is one, stored tensors are BF16, and context is bounded by
4096 prompt plus generated tokens. Plain-text input is used without chat
rendering, sampling or repetition penalties.

The user-approved revision treats full-model numerical comparisons as diagnosis.
Pinned assets, exact embedding lookup, preserved cache prefixes, exact appended
storage, inactive guards, finite outputs, submission accounting and generation
lifecycle remain required invariants. Existing independent kernel tests remain
required. Numerical distance from HF or another native schedule is recorded
without a global pass/fail threshold. Historical failed qualifications remain
in the adjacent studies.

## Reproduction

Use the locked checkout and verified prepared checkpoint described in
[the generation documentation](../../docs/generation.md). Build from clean
source; collection verifies the exact executable and source identity. Choose
new output paths when repeating collection.

```sh
uv run --locked python -m llm_mojo.model_validation build --binary build/runtime-model
uv run --locked python -m llm_mojo.model_validation build --generation --binary build/runtime-generator
uv run --locked python -m llm_mojo.model_validation specification --output build/runtime-specification.json
uv run --locked python -m llm_mojo.model_validation generate --binary build/runtime-generator --prepared build/model-prepared-v1 --output build/runtime-generation
uv run --locked --script tests/fixtures/model_reference.py diagnose --specification build/runtime-specification.json --output build/runtime-reference
uv run --locked python -m llm_mojo.model_validation diagnose --binary build/runtime-model --prepared build/model-prepared-v1 --reference build/runtime-reference --output build/runtime-diagnostics
uv run --locked python -m llm_mojo.model_validation benchmark --binary build/runtime-model --prepared build/model-prepared-v1 --specification build/runtime-specification.json --output build/runtime-measurements
uv run --locked python studies/model_generation/summarize.py
```

The declaration is `tests/fixtures/model_runtime.json`. Runtime generation
reports can additionally supply common native token histories to diagnostic
capture; those are explicitly labeled development observations.

## Measurement boundary

Model measurements use four paired blocks, reversing workload and arm order
in blocks two and three. Each arm has ten warmups and ten retained samples.
Configuration 0 is paired with itself to estimate control variation. A selected
candidate must be faster in all four block medians with a median improvement
exceeding both 5% and the largest control self-pair deviation. Inconclusive
workloads retain 0. The comparison uses an actual 24-layer model with distinct
weights and persistent caches, including token upload, intermediate copies and
the tied head, ending at device synchronization. Allocation, prefix preparation,
compilation, greedy readback and diagnostic capture are outside these samples.
Cached suffixes overwrite the same suffix after restoring logical cache lengths;
independent generation reports separately measure growing-context decode.

Generation reports measure native initialization, prefill, individual decode
forwards and whole requests. They exclude Python asset verification and compiler
startup. These few request samples describe observed behavior and latency;
they are not a comparative performance study against Hugging Face.

## Evidence status

Native collection follows the source freeze. HF reference capture is bound to
its source hashes and pinned artifacts. The replay verifies complete diagnostic,
storage and timing censuses, regenerates numerical/prediction tables and applies
the declared performance rule. Arrays, checkpoints and binaries remain under
ignored build storage. Compact complete observations and provenance accompany
this report when collection finishes.
