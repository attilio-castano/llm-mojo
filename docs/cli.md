# Commands and configuration

Use `uv run llm-mojo --help` from the source checkout. After `uv sync --locked`,
activating `.venv` also makes `llm-mojo` available directly. Installed commands
still require the checkout, native sources and its toolchain; a standalone wheel
runtime is outside the current packaging contract.

## Prepare and run

```sh
uv run llm-mojo models list
uv run llm-mojo models prepare qwen2.5-0.5b-instruct --download
uv run llm-mojo chat
uv run llm-mojo generate --prompt "The capital of France is" --preset short
uv run llm-mojo generate --prompt-file prompt.txt --max-new-tokens 64
```

Preparation downloads missing pinned assets only with `--download`. Without that
flag it requires local assets. It verifies/reuses an existing prepared model;
a new destination must not exist. Chat and generation never download assets.
`models list` verifies local preparation and reports capabilities; it does not
prepare or compile anything. Checksum verification reads the model weights.

The supported application model is `qwen2.5-0.5b-instruct`, with `--mode fast`,
BF16 storage, Metal, batch one and 4096-token capacity. These are implementation
constraints. A model name or preset cannot grant another architecture, dtype,
backend or context capacity. Generation consumes raw text; chat applies Qwen's
plain system/user/assistant template. Generation requires exactly one prompt
source; an empty prompt is rejected.

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
uv run --locked llm-mojo bench build --build-dir build/my-study-binaries
uv run --locked llm-mojo bench run --preset core \
  --build-dir build/my-study-binaries --output build/my-study-run
uv run --locked llm-mojo validate
```

The `core` preset selects `rms_norm` and `linear_decode`; `attention` selects
`gqa_decode` and `gqa_prefill`. Repeated `--study NAME` options replace the preset's
selection. They select complete existing grids, not arbitrary shapes or kernel
combinations. Run options expose the existing screen/confirmation inputs. The
study registry still owns workload dimensions, candidate ordering, paired blocks,
calibration and acceptance gates. Build/run still require the same clean commit,
verified binaries, fixtures, environment and suitable measurement conditions.
Resolved selection and paths are saved in the run's `configuration.json` and
in each study's `run.json` beside existing provenance. Historical evidence and its specifications are not
rewritten by configuration resolution.

`bench select-decoder`, `bench confirm-decoder` and `bench tokenizer` expose the
existing qualification and tokenizer workflows. `tokenizer` exposes setup,
encode and decode tooling. Every group has help.

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
| `python -m llm_mojo.model_assets` | `python -m llm_mojo.models.qwen2.assets` | Explicit research policies beyond the public Fast mode |
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
