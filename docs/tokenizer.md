# Qwen text tokenizer contract

This study implements text-to-ID and ID-to-text processing on the CPU in Mojo,
using only the pinned `tokenizer.json` from `docs/model.md`. It excludes chat
rendering, `tokenizer_config.json`, model execution, batching, padding,
truncation, token offsets, GPU work, and training. Tokenization has no 4,096-token
limit; the caller owns model context limits.

## Reference and exact behavior

Authority: `tokenizers==0.19.1`, `Tokenizer.from_file`, from the existing locked
fixture environment. Encode uses `add_special_tokens=False`; decode explicitly
sets `skip_special_tokens=False` or `True`. The Python binding executes Rust.
The heap design follows [the pinned Rust BPE source](https://github.com/huggingface/tokenizers/blob/v0.19.1/tokenizers/src/models/bpe/word.rs).
The Mojo scan and heap paths must match this authority exactly; mutual agreement
alone is not acceptance. All vocabulary IDs are valid for decoding; negative or
out-of-range IDs raise. Encoding accepts valid UTF-8 and rejects malformed input.

Added tokens match before ordinary-span normalization, using leftmost-longest
literal matching and the pinned flags. All pinned tokens have `normalized`,
`single_word`, `lstrip`, and `rstrip` false. The `special` flag controls skipping
on decode; added tokens without that flag remain visible. Unsupported tokenizer
configurations fail preparation instead of silently changing behavior.

Ordinary spans use NFC, the exact Qwen regex split, byte-to-symbol mapping, then
BPE. Decoding applies the inverse byte mapping and Rust-compatible UTF-8 lossy
replacement. A streaming decoder emits only complete valid UTF-8, retains up to
three incomplete bytes, and replaces an incomplete final sequence on `finish`.
Concatenated streaming output after finish equals whole-sequence decoding. NFC
means byte-for-byte round trips to the original unnormalized input are not a
universal invariant.

## Artifact initialization

Run `uv run --locked llm-mojo-tokenizer setup` to prepare ahead of time. The
`encode` and `decode` entrypoints perform the same initialization automatically:

```sh
uv run --locked llm-mojo-tokenizer encode 'Hello world'
uv run --locked llm-mojo-tokenizer decode '9707,1879'
```

The asset root is `build/checkpoints/qwen2.5-0.5b-instruct/` followed by immutable
revision `7ae557604adf67be50417f59c2c2f167def9a775`. The original JSON is 7,031,645
bytes, with SHA-256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
`prepared-v1/` holds `tables.bin` and its provenance manifest. Everything in
`build/` is ignored by Git. `--asset-dir` overrides the source directory;
`--offline` prohibits model-artifact downloading, but does not alter uv's own
dependency resolution behavior. A fully prepared environment works offline.

Setup serializes concurrent initializations, publishes downloads atomically only
after checksum verification, and publishes the prepared manifest last. Existing
source corruption is an error, with removal/re-download instructions. Missing or
outdated prepared tables are regenerated. The executable cache is bound to source and
lockfile hashes and rebuilt atomically when stale. Startup verifies source and table
checksums; the native loader additionally checks structural bounds and IDs.
Native direct callers must supply verified prepared tables.

Python performs preparation, not runtime text processing. It parses JSON,
resolves IDs, builds reproducible binary tables, and records source/generator
hashes. A native Mojo executable loads tables and handles all text computation
without Python, Rust, regex-engine, or Unicode-library calls.

## Unicode authority

Python 3.12 provides Unicode 15.0 canonical data as preparation input. Its behavior
is not assumed to match Rust. Preparation queries Rust over every Unicode scalar
for NFD and the splitter's letter, number, whitespace, and contraction case
classes. Canonical composition candidates are filtered by Rust NFC. Combining
classes use Unicode 15 values checked with Rust NFD reorder/barrier probes.
The manifest records every deviation. The pinned Rust normalizer lacks the
U+11938 decomposition and treats a number of newer combining marks as starters.
Mojo follows that frozen behavior, including algorithmic Hangul decomposition
and composition. No normalization correctness claim rests on Python alone.

The initial stable insertion ordering can take quadratic time on pathological
combining-mark runs. This is separate from BPE's complexity and is not described
as an O(n log n) bound on the complete tokenizer.

## Ownership and BPE algorithms

Immutable tokenizer state holds byte IDs, contiguous token bytes and offsets,
merge pair lookups, added-token metadata, and Unicode data. A pair key packs two
32-bit IDs losslessly into UInt64; dictionary equality compares the full key.
SHA-256 artifact checks and dictionary hashing serve separate purposes.

A caller-owned `TokenizerWorkspace` reuses symbol, neighbor, and heap buffers.
Normalization, piece-byte storage, and returned results currently allocate per
call/span; zero-allocation encoding is not claimed. Shared tables are read-only
and workspaces must not be shared by concurrent callers.

Variant 0 scans live adjacent pairs and shifts the remaining array. Variant 1
stores symbols in a stable array, links neighbors by index, and uses a min-heap
ordered by `(rank, original_position)`. A merge marks its right symbol removed,
updates links, and inserts at most two neighbor candidates. Stale entries must
match the current adjacent IDs and indices before execution.

For one piece with n input bytes, scan is O(n²) worst case, heap is O(n log n)
under expected constant-time hash lookup, and scratch is O(n). Piece boundaries
are fixed before merging. For a prompt, sum over its piece lengths; byte count
alone does not describe the workload.

## Correctness and held-out acceptance

Generate development fixtures after setup:

```sh
uv run --locked --script tests/fixtures/tokenizer_reference.py
uv run --locked mojo run -I src tests/test_tokenizer.mojo
uv run --locked llm-mojo-validate
```

Ordinary validation requires the local artifact and never downloads model data.
It verifies frozen fixture checksums. Tests compare text IDs under both BPE paths,
NFC output, split boundaries, complete decoding, and per-token streaming. Every
vocabulary ID is decoded independently. A separate supplemental suite checks
all 2,061 canonical decompositions in their decomposed spelling. Cases cover multilingual text, emoji,
whitespace, code, added tokens, canonical decompositions, combining marks, byte
fragments, malformed UTF-8, and reuse. Setup tests cover first-run download,
cached reuse, interruption, stale partial files, failed publication, and checksum
and generator mismatch. Bad prepared tables must fail native loading.

The holdout recipes are fixed in `tests/fixtures/tokenizer/generate.py`, with
independent seed 871019, unseen lengths, and additional scripts. Generate/run them
only against a clean candidate after development passes. A failed holdout is a
regression to fix and disclose; do not change its expected output to accept code.

## Bounded CPU study

The study adds one CPU route to existing benchmark tooling. Four families
(prose, multilingual, code, whitespace) use target byte sizes 64, 256, 4096, and
16384, with UTF-8-safe cuts. Long single-piece stress cases use 128, 512, 2048,
and 4096 bytes. Record actual input bytes, piece counts/maxima, output tokens,
source and binary identity, table footprint and scratch capacities.

Measure loading, BPE-only scan/heap, full encoding scan/heap, complete decoding,
and streaming decoding separately. Setup, compilation, and oracle work are
outside steady-state samples. Reuse buffers where declared. Follow the existing
four-block paired protocol, ten warmups and ten samples, reversed arm order in
blocks 2/3, control self-pairs, and the 5%/calibration gates. CPU execution is
explicit; Metal availability is not relevant to these timings. No best-of-run
selection, unbounded tuning, automatic dispatch, or GPU speed claim is allowed.

Completion requires exact development and holdout acceptance, documented full
repository validation, reproducible retained measurements (including negative
results), and local milestone commits. Pushing or opening a PR is separate.

### Benchmark timing details

The `load` mode measures warm-filesystem file reading, table construction,
structural validation, and destruction of the temporary tokenizer. It excludes
setup's source/table SHA-256 checks and is not a cold-start measurement. Reported
prepared-table bytes describe serialized storage; they are not resident memory.
Recorded symbol/neighbor/heap capacities describe reusable BPE scratch, excluding
hash-table allocator overhead and temporary Unicode/result allocations.
