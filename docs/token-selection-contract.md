# Compile-time token-selection contract

Token selection is a small first step toward compositional verification in
Mojo. The production kernel and its compile-time checks share the scalar
definitions in `src/llm_mojo/kernels/token_selection_contract.mojo`:
`bf16_rank`, `bf16_nonfinite`, `ArgmaxRecord`, and `merge_argmax`. The checks
establish bounded integer contracts before device execution. They are not a
Lean-checked proof of the GPU kernel or compiler.

## BF16 ordering

Assume the IEEE BF16 encoding: one sign bit, eight exponent bits, seven
fraction bits, and exponent bias 127. For a finite magnitude, let `e` be its
exponent field and `f` its fraction. Scaling its mathematical value by
`2^133` gives `f` when `e = 0`, and `(128 + f) * 2^(e - 1)` otherwise.
Within an exponent bin these increase with `f`; its largest value is smaller
than the next bin's smallest value because `255 < 256`. Thus the unsigned
encoding magnitude orders positive finite values exactly. Negation reverses
this order, and the two zero encodings have equal numerical value.

`check_bf16_ordering` visits magnitudes `0..0x7f7f` with both signs: all
65,280 finite encodings. It checks the finite rank chain against this encoding
order, preserves subnormals, and checks equal ranks for signed zero. The
nonfinite classification check covers all 65,536 encodings; exponent 255
marks both infinities and every NaN payload invalid. This is exhaustive
encoding coverage, without evaluating floating-point comparisons.

The result concerns logits already rounded to BF16. It does not establish
the accuracy of the calculation producing them. When any logit is nonfinite,
the invalid flag poisons the result; its token ID has no valid interpretation.

## Combining records

A record contains an unsigned rank, unsigned token ID, and invalid flag.
`merge_argmax` selects the larger rank, breaking equal-rank ties with the
smaller token ID, and ORs both invalid flags. The identity is
`(0, UInt32.MAX, 0)`; every finite BF16 rank is greater than zero.

`check_argmax_laws` enumerates 18 records from rank and ID alphabets
`{0, 1, UInt32.MAX}` and flags `{0, 1}`. It checks a direct pair-selection
oracle, commutativity, idempotence, identity, and associativity across all
5,832 triples. These are exhaustive checks over that representative domain.

The general argument is explicit and mathematical: rank ascending and token
ID descending define a total order on winner pairs. Selecting its maximum is
associative, commutative, and idempotent, with the identity pair as its least
element. Boolean OR has the same laws, so their product does too. Every weak
ordering of three ranks or IDs has a representative in the three-value
alphabet. Relabelling preserves comparison-and-selection behavior because
the merge only compares fields and copies a selected winner. This argument
lifts the comparison patterns to arbitrary UInt32 values; the checker does
not machine-check that source restriction or prove an arbitrary Mojo body.

These laws support regrouping selection over a complete set of logits. They
do not require floating-point addition to be associative: selection compares
integer ranks and performs no floating-point reduction.

## Closed reduction descriptions

`src/llm_mojo/kernels/argmax_reduction.mojo` describes a merge expression over
exactly eight symbolic input records. `ArgmaxPlan` holds an array of
`ReductionNode` values and a root index. Its vocabulary is closed: an `Input`
node selects one of those records, an `Empty` node supplies the identity
record, and a `Merge` node combines two earlier nodes with the fixed
`merge_argmax` primitive. There are no arbitrary callbacks, arithmetic
expressions, buffer accesses, or GPU operations in this
description. The helpers `input_node`, `empty_node`, and `merge_node` construct
these node kinds.

`check_argmax_plan` validates every node's kind, input indices, backward
references, canonical unused operands, and the root index. Unreachable nodes
must also be valid; the root may precede the final node. Backward references
make the graph acyclic and ensure each operand exists before it is used. Each
node receives a support mask using eight bits:
`Input` sets its input's bit, `Empty` has no bits, and `Merge` takes the union of
its operands' masks. The root must have mask `0xff`, containing every input
bit. This checks the set of inputs contributing to the root, rather than only
counting leaves.
Repeated uses of an input are permitted because merging a record with itself
is idempotent; inserted identity nodes are permitted by the identity law.

The symbolic argument is structural. Every node denotes the merge of the
inputs in its support set. Associativity and commutativity permit regrouping
and reordering; idempotence removes duplicate inputs; the identity law removes
empty nodes. `check_argmax_equivalence` accepts two descriptions when both
pass the checker: their roots then denote the same merge of all eight inputs
for any fixed tuple of records with UInt32 rank and token fields and Boolean
invalid flags. Symbolic input indices are independent of the records' token
IDs, which need not be distinct. This conclusion is
conditional on those merge laws and the correctness of the
checker and interpreter. It does not enumerate all possible record values or
machine-check the implementation of `merge_argmax`.

`evaluate_argmax_plan` asserts acceptance of its compile-time plan parameter
and unrolls that exact description, using the same fixed scalar primitive.
This connects the checked expression to its host evaluation without reflecting
a Mojo function body. The production scalar primitive remains unchanged. The
description is not yet used to generate the production GPU schedule: it does
not check which logits
reach each thread, warp reduction behavior, memory ownership, synchronization,
or correspondence with the complete Metal execution route.

## Execution boundary and reproduction

The local, finishing, and shared-memory stages use the shared scalar merge.
Warp reductions still use backend `max`/`min`; `max` agrees with OR only
because invalid flags are Boolean. Integer semantics, compile-time evaluation,
compiler lowering, warp operations, synchronization, and memory operations
remain trusted. The host test forces the exhaustive scans during compilation
and repeats them in compiled host code. Ordinary production builds do not run
the exhaustive scans; running the validation suite is the acceptance gate.

Callers must provide valid extents and representable launch/index counts,
injective layouts over accessed cells, nonaliasing input/scratch/output
storage, correct copy/submission order, and buffer lifetime through device
completion. The execution route requires Metal, 128 threads per block, and
32-thread warps. Scalar laws do not verify these obligations, indexing
coverage, shared-memory communication, or end-to-end inference determinism.

From the repository root:

```sh
uv run --locked mojo run -I src tests/test_token_selection_contract.mojo
uv run --locked mojo run -I src -I tests tests/test_reduction_plan.mojo
uv run --locked python -m unittest discover -s tests -p test_token_selection_compile.py
MODULAR_DEBUG=device-sync-mode uv run --locked mojo run -I src tests/test_token_selection.mojo
```

The two host Mojo commands force positive compile-time checks and exercise
compiled host implementations. The reduction-plan suite checks sequential,
balanced, reordered/duplicated, and earlier-root descriptions against an
independent two-pass selection oracle. Its native malformed-plan checks
include dead invalid nodes and noncanonical operands. The Python suite checks
successful compilation and rejection of deliberately incorrect
variants. The last command separately checks the actual GPU selection route;
its device/backend evidence is required for a claim about Metal execution.
