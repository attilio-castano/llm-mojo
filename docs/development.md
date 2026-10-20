# Development

## Toolchain

The repository is a Python 3.12 `uv` library project. Python provides the
development envelope for reference oracles and tooling; the inference engine is
implemented in Mojo.

Project dependencies track stable Mojo and MAX releases in `pyproject.toml`.
`uv.lock` records the exact reproducible environment. Do not add nightly package
indexes or prerelease flags to the default environment.

## Prerequisites

- `uv`
- a C linker
- Python 3.12, managed automatically by `uv` when necessary

Apple GPU development additionally requires a supported Apple Silicon Mac,
current macOS, Xcode 16 or later, and the Metal toolchain. Verify that full
Xcode is selected and that both public Metal compiler tools resolve before
beginning GPU work:

```bash
xcode-select -p
xcrun -f metal
xcrun -f metallib
```

The selected developer directory should point inside `Xcode.app`, not the
standalone Command Line Tools directory. On Xcode 26 or later, also verify that
the separately managed component reports `"status" : "installed"`:

```bash
xcodebuild -showComponent MetalToolchain -json
```

See the official [Mojo installation guide](https://mojolang.org/install/) and
[system requirements](https://mojolang.org/docs/requirements/).

## Reference machine

The initial Apple Silicon reference machine is a MacBook Pro with the following
stable hardware configuration:

- model identifier: `Mac16,7`;
- SoC: Apple M4 Pro;
- CPU: 14 cores (10 performance and 4 efficiency);
- GPU: 20 cores with Metal 4 support;
- unified memory: 24 GB;
- internal storage: 512 GB Apple SSD;
- published memory bandwidth: 273 GB/s.

The bandwidth figure is an Apple specification, not a measured project result.
Minimum hardware requirements have not been established.
See Apple's [MacBook Pro technical specifications](https://support.apple.com/121554).

The software environment observed on 2026-08-20 was:

- macOS 26.5.2 (`25F84`) with Darwin 25.5.0;
- Xcode 26.6 (`17F113`);
- Metal toolchain installed, build `17F109`;
- `uv` 0.12.5;
- Mojo 1.0.0 and MAX 26.5.0 as resolved by `uv.lock`.

Later recorded runs used macOS 26.6.2; each benchmark record carries its own
software versions. This snapshot establishes the development environment; it
does not prove that a Mojo workload executed on the GPU. Every GPU result must
report the runtime's device and backend identity and must satisfy the project's
[evidence requirements](project.md#evidence).

Mutable conditions belong in each benchmark record rather than in this machine
profile. At minimum, record the current software versions, power source and
power mode, thermal state, attached displays, available memory, and repository
commit and dirty state. Do not record serial numbers, hardware UUIDs,
provisioning identifiers, usernames, or volume UUIDs.

## Setup

Create the locked environment and confirm the compiler version:

```bash
uv sync --locked
uv run mojo --version
```

Then run `uv run llm-mojo setup`. It performs the toolchain checks above with
printed remedies, provisions the shared model store and links this checkout to
it, prepares the tokenizer tables and builds chat and generate; see the
[CLI guide](cli.md#prepare-and-run). A new worktree runs the same command and
reuses the store without downloading.

There is no need to activate `.venv`; `uv run` executes commands in the managed
environment. Each worktree has its own `.venv`; uv reuses its package cache when
setting up another checkout.

For everyday commands, use `uv run llm-mojo ...`, `uv run mojo ...` or `uv run python ...`. Plain
`uv run` uses the existing lockfile and does not upgrade packages just because
new releases exist. Validation and recorded measurements use `--locked` so an
outdated lockfile fails explicitly instead of changing during a run. Their
Python and Mojo subprocesses reuse the selected environment directly.

`--offline` and `--no-sync` are not part of the normal workflow. Offline mode
requires cached dependencies; no-sync assumes the environment is already ready.
If an agent sandbox blocks uv's cache or downloads, grant the required access
for setup rather than adding these flags to project commands.

## Tests

Validation needs the pinned tokenizer tables, which `uv run llm-mojo setup`
prepares. For the tokenizer alone, `uv run --locked llm-mojo tokenizer setup`
downloads only the pinned tokenizer artifact when missing, verifies it, and
prepares tables and the native executable. Subsequent tokenizer calls reuse
local artifacts. See [the tokenizer contract](tokenizer.md).

Run the complete validation workflow from a clean checkout:

```bash
uv run --locked llm-mojo validate
```

The runner lives in `src/llm_mojo/validation/suite.py`. The `mlp`, `decoder`,
and `model` modules in that package own numerical qualification, and
`evidence.py` owns their shared source identity and receipt helpers. Tests and
independent oracle generators remain under `tests/`. See the
[CLI compatibility table](cli.md#validation-and-compatibility) for retained
study commands and their canonical replacements.

Validation runs in four stages, stopping at the first failure:

1. **Oracles.** It regenerates the small independent oracles into ignored
   `build/oracle_data/` and checks each against its frozen anchor: the original
   fixtures at `a86f4db`, the prefill manifest, the MLP, decoder and
   model-calibration self-tests, the tokenizer references (including a Unicode
   run) and the packed chat fixtures. The attention-sublayer, MLP and
   decoder-layer references come from a
   [shared, verified copy](#shared-oracle-fixtures), generated and checked
   against their anchors once per generator version. Tokenizer tables must
   already be prepared; validation never downloads.
2. **Python tests.** Setup and the shared store, the CLI and configuration,
   launch boundaries, evidence replays, numerical tooling and documentation links.
   This stage also checks successful Mojo compilation and deliberate rejection
   of malformed finishing-loop certificates, typed proof terms, inductive
   definitions (including Nat), their derived eliminators, and typed coverage
   with explicit inverse ownership and headroom. Checked Nat-recursive arithmetic
   definitions, Base's derived zero/successor equations, and order/division proofs
   also have compiler controls for altered bodies, exports, circular references,
   retired driver axioms and namespace interference. Checked export
   instantiation has independent contracts for all sixteen exports, mixed
   parameters and large symbolic specializations, with compiler controls for
   wrong factors/divisors, namespaces, closure and capture. Compact operators
   also have checked raw representations, with compiler controls
   for wrong meanings, projections, parameter routing and circular bootstrap use.
   Its positive controls cover symbolic operands and large parameter values;
   resugaring must pass an independent raw administrative comparison. Each
   compact numeral also has a checked binary Nat denotation and constructor-view
   conversion derivation. Its raw bootstrap forbids compact numeral syntax;
   controls reject altered definitions, conversion rules, endpoints and binding.
   Large literal and power-of-two carry controls check bounded workspace growth
   for typing and one-layer recursion/elimination, including `succ(Int.MAX)`.
   Word interpretation additionally needs independent structural checks of
   the actual binary Nat spines: fixed lengths, bit positions and the raw zero
   endpoint, at compile time and natively. Matching producer/checker results
   alone cannot establish that interpretation. A pinned compiler/evaluator
   defect truncated a signed `Int` division loop at compile time, decoding
   `Int.MAX` into 31 digits instead of 63 despite preserving its original word
   value. Digit extraction uses explicit `UInt64` masks and logical shifts
   consistently in producers and kernel endpoint reconstruction. These word
   controls qualify known boundaries; unsigned conversion and bit decoding
   remain trusted primitives, without a checked machine-word theory. Word
   width and proof workspace do not select a computation geometry.
   Optional literal arithmetic uses raw Nat reflexivity/conversion proofs in an
   isolated environment with every native arithmetic shortcut disabled. Its
   controls bind operations, operands, results, definitions and divisor routing;
   reject unused compact/reflection syntax; and preserve typed expressions when
   overflow or finite conversion resources prevent compaction. Symbolic
   induction lemmas do not become definitional equality shortcuts.
   Coverage keeps nine roots in its shared bundle and checks headroom through a
   separate raw Nat certificate. Its nine canonical data definitions and
   thirteen ordinary helper definitions derive the binary addition/carry and
   order steps; two addition proofs bind the sum and proposed slack to the
   requested count cap, stride and integer limit. The fresh checker forbids
   native, export and reflection syntax, including unused entries. Its required
   theorem still quantifies over arbitrary Nat counts and keeps computation
   shape parameterized. Mandatory headroom checking rejects exhausted resources
   without a reflection fallback. Focused controls cover tight bounds, long
   carries, insufficient and overflowing bounds, altered proofs and definitions,
   unused forbidden syntax, and generic alpha/beta/congruence conversion.
   Affine reasoning in other coverage roots, literal-order reflection outside
   headroom, integer decoding, descriptor arithmetic, generic kernel/compiler
   correctness and runtime/GPU correspondence remain explicit assumptions; the
   [headroom contract](token-selection-contract.md#headroom-from-ordinary-nat-proofs)
   describes that boundary. Test results qualify the implementation separately
   from its proof construction. The superseded `headroom-nat-check-v3` attempt
   had focused positive proofs but failed to qualify word meaning; retain that
   evidence without treating those positives as end-to-end headroom validation.
   Each compiler-negative suite builds and executes its positive control first.
   Nat arithmetic certification adds substantial compilation work. Native typed proof
   compiler controls have separate budgets of 50 minutes, or 60 minutes for
   coverage/fold gates; native positive-control execution retains its three-minute
   bound. A timeout is a validation error, never evidence of proof rejection.
3. **Mojo suites.** Every `tests/test_*.mojo` runs with
   `MODULAR_DEBUG=device-sync-mode`. Device suites require Metal; the
   [BF16 selection contract](token-selection-contract.md) also has host-only
   encoding and reduction-plan suites whose assertions execute during
   compilation. Python tests check
   deliberate compile failures outside this glob.
4. **Route smoke.** Every maintained benchmark route runs once, and invalid
   selectors must be rejected.

| Level | Mojo suites |
| --- | --- |
| Kernels | `test_rms_norm`, `test_linear`, `test_rope`, `test_swiglu`, `test_residual_norm`, `test_token_selection`, `test_attention_primitives` |
| Attention | `test_attention`, `test_attention_decode`, `test_attention_decode_benchmark`, `test_attention_prefill`, `test_attention_precision`, `test_attention_sublayer`, `test_attention_sublayer_operations`, `test_qkv_fusion` |
| MLP and decoder layer | `test_mlp`, `test_decoder_layer`, `test_decoder_selection`, `test_consistency` |
| Model and runtime | `test_model`, `test_execution_plan`, `test_decode_route`, `test_chat`, `test_tokenizer`, `test_import` |

For the standalone private record-population model and checked commutation
requests, run:

```bash
uv run --locked mojo run -I src -I tests tests/test_finishing_phase.mojo
uv run --locked python tests/test_proof_term_compile.py ProofTermCompileTests.test_private_finishing_phase_commutation_is_checked
```

The focused Python command compiles and executes the positive host suite before
requiring eight malformed certificates to fail compilation with the exact
phase-admission diagnostic.

The [phase contract](token-selection-contract.md#private-record-population-commutation)
defines the fixed-row/base scope, immutable input prerequisite, selected-key
proof binding and remaining model/compiler/device assumptions. This phase ends
before local reduction and adds no runtime or all-thread barrier. Its host
checks are separate from GPU/backend qualification and do not establish full
schedule determinism. Record actual compilation and native results separately;
the command and mathematical proof design alone are not validation evidence.

The attention suites keep their frozen tolerances, diagnostic tensors, ragged
tiles, full and incremental prefill, and all 24 decode cases. Prefill adds 29
Qwen-shape cases and full-versus-suffix, causal-independence and extreme-score
regression checks across sixteen routes, including the five resource ablations.
Its generated NumPy arrays are loaded only by tests; inference and timed paths
remain Mojo.
Normal-mode stress for the new schedules uses `-D PREFILL_REPEAT=12` on
`tests/test_attention_prefill.mojo`, with `MODULAR_DEBUG` unset.
The Torch/Transformers oracles share the isolated script environment in
`tests/fixtures/generate.py`, locked by its adjacent `.lock` file. The NumPy
oracles use the project environment. The first run may download dependencies;
no model weights are required. To regenerate one Torch oracle, for example:

```bash
uv run --locked --script tests/fixtures/generate.py rms_norm
```

After deliberately editing dependency declarations, update the corresponding
lock with `uv lock` or `uv lock --script tests/fixtures/generate.py`, then rerun
validation. Add `--upgrade-package NAME` only when intentionally upgrading.

Each numerical contract documents its own fixtures, splits and recorded
evaluation commands: the [attention sublayer](attention-sublayer.md#implementation-and-reproduction),
the [MLP](mlp-sublayer.md#reference-results-and-reproduction) and the
[decoder layer](decoder-layer.md#qualified-reference-package).

Use `--prepare-only` to generate fixtures without running tests. For an individual
Mojo suite, include `-I src -I build -I tests`. Generators and the
small checksum record are versioned; large generated arrays and manifests are
build outputs. A changed checksum requires reviewing the oracle and numerical
contract, never relaxing tolerances to fit a kernel.

### Shared oracle fixtures

The attention-sublayer, MLP and decoder-layer references hold about 8.4 GB and
take about ten minutes to generate. They are deterministic and pinned by frozen
anchors, so validation keeps one read-only copy of each per generator version in
the shared store, under `fixtures/` in `~/.cache/llm-mojo` or
`LLM_MOJO_CACHE_DIR`, and links `build/oracle_data/<family>` to it.
[`fixtures.py`](../src/llm_mojo/validation/fixtures.py) implements it.

- **Key.** SHA-256 of the family's generator commands and the content of every
  tracked or untracked file the generators read: `tests/fixtures/generate.py`,
  `tests/fixtures/decoder_reference.py`, their script lock, and the
  `attention_sublayer/`, `mlp/` and `decoder_layer/` directories under
  `tests/fixtures/`. Editing any of them, committed or not, selects a new entry.
  Other edits under `tests/fixtures/` do not. If any of them changes while
  validation runs, validation fails at the end: its oracles belong to the
  earlier inputs.
- **Hit.** Validation re-hashes every file of the entry against its record,
  which takes seconds, and links the checkout. It writes nothing to the store.
  If the entry was generated on another macOS or Python version, validation
  says so and continues: the entry still matches its frozen anchors.
- **Miss.** Validation generates the family as before, including its anchor
  checks, and publishes the tree read-only by rename. Another worktree
  validating the same key waits, then reuses it. A damaged entry is set aside as
  `<key>.invalid-<time>` and regenerated. `mlp/last_run.json` names the
  worktree that ran the generator, so it moves into the entry's record,
  `fixtures/<family>/<key>.json`.
- **Checkout copies.** A real `build/oracle_data/<family>` directory gives way
  to the link only when validation wrote everything in it. Anything else is
  kept beside it as `<family>.local-<time>`. The Mojo MLP suites record their
  checks in `build/oracle_records/mlp/`, outside the linked directory.

`uv run --locked llm-mojo validate --regenerate-fixtures` regenerates the three
families even when cached, and fails unless they match the cached copies byte
for byte; a differing tree is kept as `<key>.regenerated-<time>` for comparison.
`--no-fixture-cache` generates them inside the checkout as before and leaves the
store untouched.

`uv run llm-mojo fixtures list` shows each entry, its size and the worktrees
that link it or whose inputs currently select it. Entries accumulate as the
generators change. `uv run llm-mojo fixtures prune` lists the set-aside trees,
interrupted staging and entries that no worktree of this repository links or
selects; `--yes` removes them. It skips an entry that a validation is generating.
Run it when no validation is using the store.

Generator commands run by hand, such as the checkpoint, calibration and holdout
captures, write into the family directory and fail on the read-only link. Run
`uv run llm-mojo fixtures detach <family>` first. It replaces the link with a
writable copy-on-write clone, and the next validation links the family again.

## Measurements and studies

See [src/llm_mojo/benchmarks/README.md](../src/llm_mojo/benchmarks/README.md) for the shared build/run
workflow and [studies/README.md](../studies/README.md) for the maintained
explanations and figures. Measurement requires AC power, Low Power Mode off,
and no reported thermal warning. Correctness testing uses device-sync-mode;
timing deliberately removes it and uses the instrument's explicit boundaries.

Before committing, run the full test workflow for numerical/fixture changes.
For tooling-only changes, run the Python tests and relevant route smoke;
repeat numerical suites if an engine, numerical contract, or oracle changes.
Always run `git diff --check`. Stable dependencies remain pinned by `uv.lock`.
