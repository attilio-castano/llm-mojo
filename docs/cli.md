# Commands and configuration

Use `uv run llm-mojo --help` from the source checkout. After `uv sync --locked`,
activating `.venv` also makes `llm-mojo` available directly. Installed commands
still require the checkout, native sources and its toolchain; a standalone wheel
runtime is outside the current packaging contract.

## Prepare and run

```sh
uv run llm-mojo setup
uv run llm-mojo setup --check
uv run llm-mojo models list
uv run llm-mojo chat
uv run llm-mojo generate --prompt "The capital of France is" --preset short
uv run llm-mojo generate --prompt-file prompt.txt --max-new-tokens 64
```

`setup` checks the toolchain, provisions the shared model store, links this
checkout to it, prepares the tokenizer tables and builds chat and generate. A
missing pinned file is imported from a verified copy when one exists (this
checkout, another Git worktree, or `--import-from PATH`, cloned copy-on-write) and
downloaded otherwise; `--offline` forbids downloads and `--no-build` skips
compilation. `--check` changes nothing and exits nonzero until everything is ready.
If `uv` is missing or the store's volume lacks room for what the store still
needs, setup and `models prepare` stop before changing anything.

The store lives outside every checkout: `--store`, else `LLM_MOJO_CACHE_DIR`, else
`$XDG_CACHE_HOME/llm-mojo`, else `~/.cache/llm-mojo`. Files enter it only after
their pinned size and SHA-256 match, by atomic rename, and are read-only
afterwards. The prepared model is identified by a pinned digest of its 196 tensor
hashes. Each checkout keeps its `build/` paths: the seven downloads become
per-file links and `build/model-prepared-v1` a directory link. Tokenizer tables
and native binaries stay per checkout because they depend on its sources; a real
file or directory is never replaced by a link.

`models prepare` is the asset-only part of setup and downloads only with
`--download`; `--output PATH` writes a separate verified copy. Chat and generation
never download assets. `models list` reports capabilities and store, link, table,
binary and device state; it does not prepare or compile anything. Checksum
verification reads the model weights.

The supported application model is `qwen2.5-0.5b-instruct`, with BF16 storage,
Metal, batch one and 4096-token capacity. These are implementation constraints.
A model name or preset cannot grant another architecture, dtype, backend or
context capacity. Generation consumes raw text; chat applies Qwen's plain
system/user/assistant template. Generation requires exactly one prompt source; an
empty prompt is rejected.

Chat runs the Fast route. `generate --mode` also accepts two reference routes:
`baseline` runs decoder configuration 0 for every call, and `consistent` runs the
deterministic research route. Both are slower than `fast`.

## Presets and precedence

Hydra composes typed Python dataclasses registered in `configuration.py`.
Precedence is schema defaults, then the named workload preset, then explicitly
supplied command options. There is no second YAML copy of these defaults.

| Preset | Maximum new tokens | Prompt chunk rows | Commands |
| --- | ---: | ---: | --- |
| `interactive` (default) | 256 | 256 | chat, generate |
| `short` | 32 | 64 | chat, generate |
| `whole-prompt` | 256 | 0 (entire prompt) | generate |

```sh
uv run llm-mojo chat --preset short --max-new-tokens 64 --show-config
uv run llm-mojo generate --preset whole-prompt --show-config
```

`--show-config` resolves and validates options without reading prompt files,
loading weights, compiling, writing reports or changing the working directory.
It does not assert that a requested file exists or that assets are ready.
Unknown presets, models, modes and conflicting prompt inputs are rejected.
Numeric bounds are checked before asset verification; native tokenization still
checks token-dependent context limits. Existing chat rejection and generation
budget behavior are unchanged.

The default prepared path belongs to the checkout. Explicit relative paths are
resolved from the caller's working directory. Prompt text and paths are literal
values, including `${...}`; they never enter Hydra's interpolation or override
language. Typer owns CLI parsing. Hydra's Compose API does not introduce multirun,
automatic logging, output directories or working-directory changes.

With `--report new-events.tsv`, an adjacent `new-events.tsv.config.json` records
the command and resolved configuration. Both destinations must be new. Reports
may contain prompt text, paths and token IDs. A sidecar records attempted launch
configuration; native completion events establish whether execution completed.

## Studies and validation

```sh
uv run llm-mojo bench list
uv run llm-mojo bench run --preset core --show-config
uv run --locked llm-mojo bench build --build-dir /private/tmp/my-study-binaries
uv run --locked llm-mojo bench run --preset core \
  --build-dir /private/tmp/my-study-binaries --output /private/tmp/my-study-run
uv run --locked llm-mojo validate
uv run --locked llm-mojo validate --regenerate-fixtures
uv run llm-mojo fixtures list
uv run llm-mojo fixtures prune
uv run llm-mojo fixtures detach mlp
```

Build and run directories must be new and outside the checkout; recorded runs
refuse paths inside it. The `core` preset selects `rms_norm` and `linear_decode`;
`attention` selects `gqa_decode` and `gqa_prefill`. Repeated `--study NAME`
options replace the preset's selection. They select complete existing grids, not arbitrary shapes or kernel
combinations. Run options expose the existing screen/confirmation inputs. The
study registry still owns workload dimensions, candidate ordering, paired blocks,
calibration and acceptance gates. Build/run still require the same clean commit,
verified binaries, fixtures, environment and suitable measurement conditions.
Resolved selection and paths are saved in the run's `configuration.json` and
in each study's `run.json` beside existing provenance. Historical evidence and its specifications are not
rewritten by configuration resolution.

`validate` links the large attention-sublayer, MLP and decoder-layer oracles to
a shared, verified copy. `--regenerate-fixtures` regenerates them and requires a
byte-for-byte match with that copy; `--no-fixture-cache` generates them in the
checkout instead. `fixtures list` shows the cached entries and the worktrees
using them. `fixtures prune` lists entries nothing uses, and `--yes` removes
them. `fixtures detach FAMILY` swaps a link for a writable copy before a
generator command run by hand. See
[shared oracle fixtures](development.md#shared-oracle-fixtures).

`bench tokenizer` exposes the existing tokenizer workflows. `tokenizer` exposes
setup, encode and decode tooling. Every group has help.

`bench list` shows replay-only studies separately: the decoder selection screens
and confirmations and the policy round 2 studies. Their arms include decoder
configurations the engine no longer implements. Their retained runs still
replay. Rerunning one requires the commit recorded in its `run.json`; the
configurations exist through `edb610a`.

## Validation and compatibility

Use `llm-mojo validate` for the full repository suite. Numerical study tools live
under `llm_mojo.validation`: `mlp`, `decoder`, and `model` each retain their
existing build/evaluation arguments and support `--help`. Their shared
`evidence.py` owns source hashes and JSON receipt writing; test cases and oracle
generators remain under `tests/`.

The supported compatibility surface is deliberately limited:

| Retained entry point | Recommended interface | Reason retained |
| --- | --- | --- |
| `llm-mojo-validate` and `python -m llm_mojo.validate` | `llm-mojo validate` | Existing validation and study instructions |
| `python -m llm_mojo.mlp_validation` | `python -m llm_mojo.validation.mlp` | Existing numerical reproduction commands |
| `python -m llm_mojo.decoder_validation` | `python -m llm_mojo.validation.decoder` | Existing numerical reproduction commands |
| `python -m llm_mojo.model_validation` | `python -m llm_mojo.validation.model` | Existing numerical reproduction commands |
| `llm-mojo-tokenizer` | `llm-mojo tokenizer` | Existing installed command and study instructions |
| `llm-mojo-bench` | `llm-mojo bench` | Existing measurement command and study instructions |

The installed aliases call their implementations directly. Root-level Python
wrappers support command execution only; maintained imports use the owning
packages. Remove an alias when its reproduction workflow has been retired or
explicitly migrated, rather than extending it with new behavior.

The unused `llm_mojo.chat` and `llm_mojo.tokenizer_assets` module wrappers were
removed. Use `llm-mojo chat` and `llm-mojo tokenizer`, respectively. Historical
receipts retain their original source paths and hashes; reproducing an exact
historical source identity requires checking out the recorded commit.

## Code ownership

| Package | Responsibility |
| --- | --- |
| `cli/` | Typer commands and native executable entry points |
| `configuration.py` | Typed composition, literal overrides and validation |
| `models/qwen2/` | Pinned asset preparation, tokenizer, Qwen model and chat semantics |
| `runtime/` | Native builds, launch preparation, terminal and clock services |
| `layers/` | Decoder, attention and MLP composition |
| `kernels/` | Reusable numerical operations |
| `validation/` | Numerical acceptance, source receipts and repository validation |
| `benchmarks/` | Measurement protocols, qualification and retained evidence readers |

Python finishes preparation before native inference. Chat replaces the launcher
process so terminal interruption still reaches the native session directly.
Generation uses a native child and a temporary UTF-8 prompt snapshot. The build
cache follows absolute local Mojo imports and package initializers plus `uv.lock`;
benchmark-only edits do not rebuild chat. New native project imports must use
absolute `llm_mojo` module paths so dependency tracking remains explicit.

To add a model, implement and validate its assets, tokenizer/template behavior,
model composition and native capabilities first. Then register its selectable
configuration and launch dispatch. Shared operations belong in layers/kernels
when their contracts already support that model. Experimental policy names are
not automatically promoted to application modes.
