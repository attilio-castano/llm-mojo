# Inside the inference engine

Open the [hosted explorer](https://attilio-castano.github.io/llm-mojo/) or
[index.html](index.html) in a browser. The explorer embeds its styles,
scripts, recorded data, source excerpts, and third-party notices. It works
offline without installing Mojo, loading weights, or running a server.

The shared component map connects three guided views:

1. **Prompt to next token:** one recorded Fast conversation, followed by a
   46-step reference-engine replay with mixed prefill/decode and KV accounting.
2. **Make one token faster:** the promoted residual/RMSNorm, buffer-swap and
   GPU-argmax composition, alongside the projection-arrangement candidate that
   missed its complete-token gate.
3. **Check before execution:** fixture-derived reduction plans, illustrative
   values, and a dated receipt from actual host compiler checks.

Choose **Direct chat** or **Serving core** to inspect the two execution paths.
Select any component for its purpose, ownership, implementation and evidence.
The address bar preserves the selected guide, route, component and example.

## Evidence and scope

The browser replays records and illustrates source-defined mechanisms. It does
not execute Mojo, compile plans, simulate the inference engine, or measure speed.
The internal model flow is an explanation; no per-kernel recording is implied.

The conversation is the first turn of selection 3 in the first retained terminal
block of `studies/model_generation/residual-norm.json.gz`. Its 44 prompt IDs and
128 generated IDs are decoded with a pinned subset of the original tokenizer.
The final cache contains 171 processed positions. The retained history also
contains the pending generated token and two turn-closing IDs.

The serving view preserves all 46 steps from the first non-calibration
`chunked` run in `studies/model_generation/engine-core-offline.json.gz`.
It uses synchronous reference configuration 27, a fixed 256-row budget,
128 blocks, and frozen synthetic token requests on actual Metal hardware.
Lanes show recorded delivered-token progress. Free-block counts are retained;
the archive does not record physical block identities.

Performance tables are recomputed from the original four paired blocks and
their raw samples, with the original calibration thresholds. They retain
Mojo 1.0 / MAX 26.5 identities. Current source snapshots and historical measured
source links are distinct.

The compile-time view uses the exact graph constructors from
`tests/reduction_plan_cases.mojo` and the missing-input fixture.
Its small numeric scores are illustrative; the compiler tests exercise separate
representative records. The result concerns selection from already rounded
BF16 scores and checked host evaluation. GPU schedule correspondence and
full-model schedule invariance remain open.

## Rebuild and verify

The generator uses only the Python standard library:

```sh
python3 -B docs/explorer/build.py
python3 -B docs/explorer/build.py --check
python3 -B -m unittest discover -s tests -p test_explorer.py -v
```

`recordings.json` freezes archive hashes, exact selections, source excerpts,
the tokenizer byte subset, and vendored runtime identities. A normal build
rejects changed inputs. `--check` additionally rejects stale generated HTML
without overwriting it. Historical archives are never rewritten.

When changing presentation, edit `explorer.template.html` and rebuild. When
changing source or choosing another recorded run, review and update the relevant
manifest hashes and excerpt ranges deliberately. Keep the original measurements
and historical compiler receipts unchanged; a changed contract needs a fresh
active receipt.

Commit the contract sources, verifier, and its regression tests before running
the verifier. It captures `HEAD` before testing and requires every declared input
to match that commit, including the complete rejection-fixture set. Staged edits
alone are insufficient. It rechecks `HEAD`, the input set and the input hashes
after execution; a change aborts without replacing the active receipt. Unrelated
presentation edits and generated outputs do not prevent verification. These
endpoint checks detect persistent changes, not a transient edit restored during
compilation; they do not isolate the compiler from concurrent writers.

To rerun the existing host compiler harness with the installed locked toolchain:

```sh
uv run --locked python docs/explorer/verify_contract.py
```

The verifier runs the unchanged positive/negative harness, stores detailed logs
under ignored `build/explorer/`, and writes the compact dated
`contract-receipt.json`. It binds source hashes, toolchain versions, commands,
exit codes and intended rejection diagnostics. The retained receipt records an
explicit compatible installed environment and a per-process Command Line Tools
selection used on this machine. Reproduction on another machine may use its own
installed compiler; `--compiler /absolute/path/to/mojo` selects one explicitly.

After a new receipt, rebuild the page and run the integrity checks above.
Commit the receipt and generated page separately from the source changes they
describe. Since the verifier and its regression tests are themselves receipt
inputs, the earlier receipt is temporarily stale between those two commits.
Refreshing the page does not rerun the compiler tests.

The [original October 16 receipt](contract-receipt-2026-10-16-pre-provenance-fix.json)
is retained byte-for-byte (SHA-256
`9baa3e181fb9123ca6a0bc318cfb880f0384b4311b3032fccf25fb9760f9234e`).
Its `source_commit` names the checkout's earlier `HEAD`, which did not contain
the verifier and compiler-version test listed in its source hashes. The Mojo
contract and fixture hashes match that commit. This historical provenance defect
does not supply evidence against its recorded host outcomes; the active receipt
comes from a fresh run against committed inputs.

For a local HTTP preview, run from the repository root:

```sh
python3 -B -m http.server 8765 --bind 127.0.0.1 --directory docs/explorer
```

Open `http://127.0.0.1:8765/`.

## Publication

`.github/workflows/explorer-pages.yml` validates the generated page, retained
evidence and documentation links on pull requests. Its checks use Python's
standard library and the retained host compiler receipt; CI does not install
Mojo or execute a model.

On `main`, the workflow uploads only `index.html` and deploys it with GitHub
Pages. The repository's Pages source is GitHub Actions. Publication neither
refreshes measurements nor reruns the compiler harness.

The vendored Pretext module supplies responsive title measurement. Its license
and upstream notices are in `vendor/PRETEXT-LICENSE.txt` and are embedded in the
standalone page. System fonts keep the page independent of network requests.
