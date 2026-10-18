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

## Qualification and remaining work

Run the focused native/proof suite with:

```bash
uv run --locked mojo run -I src -I tests tests/test_finishing_execution.mojo
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
