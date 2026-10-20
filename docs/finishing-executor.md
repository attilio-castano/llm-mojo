# Proof-carrying finishing execution

The finishing pass takes partial selection records and returns one record per
row. Its traversal was already checked, but traversal alone does not establish
the value returned after local trees, repeated accumulation and final grouping.
This work connects those stages through a closed execution description, a
native event interpreter and ordinary checked proof terms.

## The logical result

The first checkpoint concerns an abstract join. The mathematical carrier is
Nat, with a binary function `Join` and an identity `Empty`. Associativity,
commutativity, both identity equations and idempotence are explicit quantified
premises. The kernel checks a proof function accepting that evidence; it grants
no special join, whole-trace or coverage axiom.

For arbitrary input function `X`, runtime count `N`, support map and event
streams, the result theorem has the shape:

```text
algebra laws
  + prescribed input/Empty initialization
  + legal structural trace of length n
  + root ready after n steps
  + root support contains every input below N
  -> value at root = FlatFold(Join, Empty, X, N)
```

Both `N` and `n` are Nat binders. Neither is a compile-time unrolling bound.
Two complete traces satisfying these premises have the same result, even if
their register names, trees, grouping and event orders differ. Finite syntax,
conversion fuel and interpreter storage bound checking resources, rather than
fixing a thread count or runtime extent.

This is not yet an unconditional theorem about the three native UInt32 words.
The scalar Nat carrier is an abstract record interpretation; no native packing
or word-conversion theorem is implemented here. Interpreting the fixed native
record operation as this algebra, including its max-rank/min-token/invalid-OR
specification, is a remaining refinement obligation. The record algebra is
supplied for mathematical bounded-token Nat records by the
[correlated record execution layer](finishing-record-execution.md). That layer
also proves equality of all three fields with an independent input fold;
native word and Mojo-source correspondence remain separate obligations.

## Why the trace predicate is not the conclusion

[`finishing_execution_terms.mojo`](../src/llm_mojo/kernels/finishing_execution_terms.mojo)
defines the actual logical value and readiness stores by Nat recursion. Each
instruction reads two ready registers, joins their values, writes a fresh
register, and marks that register ready.

[`finishing_execution_trace.mojo`](../src/llm_mojo/kernels/finishing_execution_trace.mojo)
defines structural legality independently of the desired output. A step must
read ready sources, write a pending destination, and have destination support
equal to the pointwise union of its source supports. These are structural
conditions on provenance and readiness, rather than equations asserting that
the machine already computed its final answer.

The value connection is derived in separate checked definitions:

- A masked fold joins the input values selected by a support function.
- The fold of a support union equals the join of the two folds, under the
  explicitly supplied algebra laws.
- Singleton support below `N` folds to its actual input. Empty support folds
  to `Empty`.
- The prescribed initial stores therefore satisfy the reference invariant.
- Nat induction on trace length preserves that invariant for every legal step.
- Complete root support reduces its masked reference to the independent flat
  fold, closing the final equation.

The algebra and singleton proofs live in
[`finishing_execution_algebra.mojo`](../src/llm_mojo/kernels/finishing_execution_algebra.mojo)
and
[`finishing_execution_leaves.mojo`](../src/llm_mojo/kernels/finishing_execution_leaves.mojo).
There is no assumed initial-value invariant or final root-value equation in the
closed result export. Those were premises of an intermediate generic invariant
and are discharged by the later proofs.

[`finishing_execution_check.mojo`](../src/llm_mojo/kernels/finishing_execution_check.mojo)
independently reconstructs the required data definitions and theorem types.
It checks the actual data bodies by conversion, admits all ordinary proof
definitions, checks unused submitted syntax, and requires closed roots. Only
the previously certified numeral abbreviations are enabled. Arithmetic
exports and affine/literal-order reflection are outside this proof profile.
Workspace metadata is bounded before copying a submission into checker state.

## Native execution and provenance

The [derived join algebra](finishing-join-algebra.md) provides a concrete Nat
score/Boolean-flag instance of these laws and a specialized canonical theorem.
The [record layer](finishing-record-execution.md) derives the arbitrary-Nat
rank/token encoding and whole-record trace theorem. Its native UInt32/source
interpretation remains to be proved, so the production certificate retains
the conditional abstract-join profile here.

[`finishing_execution.mojo`](../src/llm_mojo/kernels/finishing_execution.mojo)
defines a descriptor parameterized by thread count, items per thread, each
tree's node count and logical subgroup width. A subgroup width must divide the
thread count. Accepted local, subgroup and final DAGs retain their exact nodes
and roots, including well-formed unreachable nodes and legitimate repeated
leaves.

Each thread independently performs:

```text
Fill all own slots -> local DAG -> accumulate -> advance -> repeat or publish
```

A subgroup can evaluate after its own threads publish. The final DAG waits
for every subgroup DAG node, including unused nodes, then one output event
writes the result. There is no
global barrier between logical loop tiles. The model uses actual native
records and immutable input arrays; padding writes the literal identity and
also becomes ready.

Canonical events retain owner, node/slot index and generation. An event cannot
supply a callback, free payload, source override or alternative merge. Invalid
events and insufficient storage must fail before changing state. Source load
visits are counted separately from reduction leaves: a repeated DAG leaf can
be valid for an idempotent join, while a duplicated input load violates exact
traversal even when the final winner happens to agree.

The state is inspectable and mutable. Structural state validation does not
authenticate its history. Claims about complete execution start at the
prescribed fresh state and follow accepted events; an arbitrarily forged
state is not an execution certificate.

[`finishing_execution_replay.mojo`](../src/llm_mojo/kernels/finishing_execution_replay.mojo)
reconstructs that history into immutable fresh-register join instructions.
Input registers `0..N-1` and the identity register `N` are initially ready.
Every value-producing event gets a fresh destination. Copies lower to a join
with `Empty`; padding lowers to an empty join. Per-thread generations and cell
IDs determine parent edges, so commutation alone cannot validate a wrong
source. The replay retains the exact descriptor, event history, values, edges
and input support. Candidate checking rebuilds these from the requested
descriptor and events instead of accepting caller-supplied provenance.

## Connection to the device implementation

The production connection uses the same closed program and local/final DAG
description as native interpretation. The Apple backend's subgroup width is
32, reflecting its existing warp contract; logical thread/item counts and
tree shapes remain parameters. The backend's warp max/min and leader/shared
publication implement logical subgroup aggregation under an explicit
refinement assumption. The logical final readiness boundary does not prove
Metal's physical barrier and visibility behavior.

The prior checked coverage and termination certificates remain required by
the finishing launcher. The new result theorem is conditional on a complete
legal logical trace. It does not prove fairness, physical GPU progress, or
that every arbitrary scheduling policy produces such a trace. A native
remaining-event rank and bounded state exploration provide separate concrete
progress checks.

## Reusing checked exports during compilation

`finishing_checked_exports.mojo` exposes immutable compiler-associated
admissions. Each Boolean is evaluated by the existing independent checker;
there is no caller-provided acceptance flag, serialized proof receipt, new
inference rule, or state imported into the proof kernel.

Production coverage and termination use `FinishingTraversalLibrary`, keyed by
the complete `FinishingProgram`. It constructs the canonical proof bundles as
associated data and independently checks all roots and headroom evidence.
Callers with different tensor layouts or local reduction trees reuse that
admission. Each caller still checks its own tree and exact program binding.
A genuinely different program requires another admission. This does not yet
replace the shape-specific
coverage and termination proofs with theorems universally quantified over
all program fields.

The canonical general execution proof already quantifies over its inputs,
join evidence, registers and schedules. A module-level constant admits its
actual syntax once with the existing checker and budget. Canonical requests
validate their descriptor and use that fixed theorem; they accept no submitted
proof, claimed theorem type, or acceptance flag.

Arbitrary submitted proofs retain the full-input APIs, with no memoized
acceptance for caller syntax. Every submission checks its roots, unused syntax,
declarations and resource budget. Certificate/request equality remains
mandatory. A canonical admission never authorizes a changed submitted proof
or mismatched certificate.

The production host launcher uses these checked canonical libraries. The
original mutable-value checker APIs remain available and
independently check submitted proofs. The base-linearity and termination roots
now use the kernel's existing multi-root API to share one inference state;
both independently required closed types and every syntax entry are checked.

Production makes ordinary host calls to `require_finishing_traversal[program]`
and `require_finishing_execution`. Their bodies contain the compile-time
assertions. The traversal body specializes only on the complete program;
the general execution body has no shape parameter. Named function
specializations provide the reuse boundary. Merely reading a `comptime`
expression in each caller can repeat substantial elaboration even when the
mathematical inputs match. The admission functions pass no proof handle or
acceptance flag and perform no runtime proof checking.
The ordinary-call rejection fixture requires an actual compiler diagnostic
and no executable, including under optimization: a no-op runtime body must
not allow the compiler to erase an unchecked compile-time obligation.

`_finish` takes a small local plan and is a checked host launcher. It constructs
the canonical executable descriptor and performs these compile-time checks
before enqueuing a nested, value-taking GPU closure. The closure owns only the
tensor views and runtime count; its executable descriptor remains the exact
compile-time value checked by the launcher. Its body contains the same fixed
loop and local/final DAG checks, but no proof construction or admission.
An explicit host-target assertion catches accidental re-entry of the launcher
during GPU offload. There is no unchecked kernel entrypoint, caller-provided
acceptance flag, or proof handle whose constructor must be trusted.
Proof bundles are associated library data rather than closure type parameters;
their full syntax would otherwise exceed the Apple linker's symbol-name limit.
The existing compiler, closure argument transport and source-to-model boundary
still require trust and actual device qualification.

`tests/test_finishing_checked_exports.mojo` forces canonical admissions during
compilation, rejects a changed canonical program at compile time, then checks
altered submitted proofs, unused syntax, trees, descriptors and budgets in
native code. These inputs always use the original full checkers. The
compiler's correct evaluation and identity of immutable program parameters
remain within the existing compiler trust assumption.
The host-launcher and ordinary-call rejection fixtures require the actual
fold-admission diagnostic and absence of an executable. An unrelated parser/device failure,
crash or timeout is not a passing rejection.

For a reproducible host compilation profile on macOS:

```sh
uv run --locked python -B tests/profile_finishing_compilation.py --stage 4 --copies 4 --reuse 0 --output /private/tmp/finishing-unshared
uv run --locked python -B tests/profile_finishing_compilation.py --stage 4 --copies 4 --reuse 1 --output /private/tmp/finishing-shared
```

Use a clean source commit and new output paths for each invocation. The tool
uses a fresh compiler cache, records compiler/host identity and complete input
hashes, retains MLIR pass and process-resource reports, checks unchanged inputs,
and executes the resulting probe. `--copies` varies distinct callers of one
loop program, alternating sequential and balanced local reduction trees.
`--stage` isolates arithmetic
bootstrap (0), coverage (1), termination (2), the general execution theorem (3),
or the complete fold/request (4). Compiler `--threads` controls compilation
concurrency and does not constrain GPU computation shapes. These host costs
are separate from device correctness and GPU performance evidence.

`--target operations` profiles the real operation benchmark and checks a paired
token-selection route on Metal. `--target numerical` profiles and runs the
entire existing token-selection numerical matrix. Those targets use the
selected source's production verification path; `--reuse` only switches the
host proof probe. `--source-root` and `--probe` can select a pinned source
snapshot for a before/after comparison. The complete input hashes identify
that snapshot even when it differs from the tooling checkout.

The [retained compilation study](../studies/model_generation/finishing-compilation.md)
compares cold builds and mixed-tree caller scaling, including compiler
instructions, memory, exact source identities and actual Metal qualification.

## Qualification and remaining work

Run the focused native/proof suite with:

```bash
uv run --locked mojo run -I src -I tests tests/test_finishing_execution.mojo
```

Check immutable admissions, changed inputs and host-launcher rejection with:

```sh
uv run --locked python -B tests/test_proof_term_compile.py ProofTermCompileTests.test_finishing_checked_exports_are_bound
```

Run its positive compiler/native gate followed by the exact negative fixtures:

```bash
uv run --locked python tests/test_proof_term_compile.py ProofTermCompileTests.test_finishing_execution_is_checked
```

The native oracle determines the maximum rank, then the minimum eligible token,
and separately combines invalid words. It does not call the production merge.
Small-state exploration deduplicates full states and explores every outgoing
legal event, with an explicit failure if its budget is exhausted. Larger
seeded executions check every intermediate effect and the final record.
Concrete examples support interpreter correspondence; they do not replace
the quantified logical proof.

Qualification results must be recorded separately from this contract. Source,
compiler, logs and native executable must be pinned, and the positive build
and execution must pass before exact compiler rejection controls. A timeout,
crash or exhausted proof/exploration budget is not a successful rejection.
The earlier private-fill evidence remains preserved under
`build/finishing-phase-check/`; the larger project's records belong under
`build/finishing-execution-check/`.

Remaining trust includes the generic kernel's typing, induction, substitution
and conversion implementations; the independent admission specification;
native integer/word interpretation and address arithmetic; event-to-register
lowering and source/model correspondence; and input extents, immutable inputs,
injective layouts, nonaliasing storage and buffer lifetime. Compiler lowering,
warp participation, shared-memory visibility, synchronization and device
progress require their own execution evidence and assumptions. Neither the
kernel implementation nor this native refinement has a formal soundness proof.

The larger project's remaining stages are the checked native record algebra,
more explicit collective-memory refinement and actual device qualification.
An abstract finishing theorem and host agreement do not establish full-model
schedule invariance for floating-point inference, cache contents or logits.
