# Compile-time token-selection contract

Token selection is a small first step toward compositional verification in
Mojo. The production kernel and its compile-time checks share the scalar
definitions in `src/llm_mojo/kernels/token_selection_contract.mojo`:
`bf16_rank`, `bf16_nonfinite`, `ArgmaxRecord`, and `merge_argmax`. The checks
establish bounded integer contracts before device execution. A Mojo-native
proof kernel also checks a symbolic derivation of finishing-loop coverage and
termination during compilation. Its rule semantics, fixed execution interpreter,
and GPU compiler/runtime remain trusted.

The [correlated record execution proof](finishing-record-execution.md) derives
rank/token/Boolean-invalidity merge laws and the three-field result of every
complete legal mathematical trace, compared with an independent input fold.
Counts, token bounds and event streams remain symbolic. Native UInt32/source
correspondence and physical GPU execution remain separate obligations.

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
a compile-time number of symbolic input records. `ArgmaxPlan[inputs, size]`
holds an array of `size` `ReductionNode` values and a root index. Its vocabulary
is closed: an `Input` node selects one of those records, an `Empty` node supplies
the identity record, and a `Merge` node combines two earlier nodes with the fixed
`merge_argmax` primitive. There are no arbitrary callbacks, arithmetic
expressions, buffer accesses, or GPU operations in this
description. The helpers `input_node`, `empty_node`, and `merge_node` construct
these node kinds.

`check_argmax_plan` validates every node's kind, input indices, backward
references, canonical unused operands, and the root index. Unreachable nodes
must also be valid; the root may precede the final node. Backward references
make the graph acyclic and ensure each operand exists before it is used. Each
node receives a support set represented by a Boolean for each symbolic input:
`Input` contains its input, `Empty` contains none, and `Merge` takes the union of
its operands' sets. The root must contain all `inputs` inputs. This checks the
set of inputs contributing to the root, rather than only counting leaves.
There is no eight-input or machine-word mask limit. Checker storage and work
scale with `size * inputs`; compilation and device resource feasibility still
bound practical specializations.
Repeated uses of an input are permitted because merging a record with itself
is idempotent; inserted identity nodes are permitted by the identity law.

The symbolic argument is structural. Every node denotes the merge of the
inputs in its support set. Associativity and commutativity permit regrouping
and reordering; idempotence removes duplicate inputs; the identity law removes
empty nodes. `check_argmax_equivalence` accepts two descriptions when both
pass the checker for the same input count: their roots then denote the same
merge of all inputs for any fixed tuple of records with UInt32 rank and token fields and Boolean
invalid flags. Symbolic input indices are independent of the records' token
IDs, which need not be distinct. This conclusion is
conditional on those merge laws and the correctness of the
checker and interpreter. It does not enumerate all possible record values or
machine-check the implementation of `merge_argmax`.

`evaluate_argmax_plan` asserts acceptance of its compile-time plan parameter
and unrolls that exact description, using the same fixed scalar primitive.
This connects the checked expression to its execution without reflecting a
Mojo function body. `sequential_argmax_plan[inputs]` and
`balanced_argmax_plan[inputs]` construct two shapes; callers may supply other
accepted descriptions with different node counts. The production scalar
primitive remains unchanged.

## Parameterized production execution

`enqueue_argmax` has compile-time `threads`, `items_per_thread`, `plan_nodes`,
and `plan` parameters for the first pass. The defaults are 128 threads, eight items per thread,
and a sequential plan. These are one launch configuration, not limits of the
reduction language. For example, with scratch sized for the selected tile:

```mojo
from llm_mojo.kernels.argmax_reduction import balanced_argmax_plan
enqueue_argmax[threads=64, items_per_thread=3,
               plan=balanced_argmax_plan[3](),
               finish_threads=32, finish_items_per_thread=5,
               finish_plan=balanced_argmax_plan[5]()](ctx, logits, partials, result)
```

Each first-pass thread loads `items_per_thread` records, using `Empty` for
positions beyond the row's extent, and evaluates the exact accepted `plan` on
the GPU. After warp reduction, thread zero also evaluates a checked sequential
plan over the `threads / 32` shared warp records. Shared storage and the final
pass's shared-record count derive from their respective thread counts.

`check_argmax_schedule` runs at compile time for the chosen instance. It
requires positive item count, a positive multiple of 32 threads, and a tile
size representable within the route's Int32 count domain. The tile map shared
by checking and production execution is
`offset = thread + threads * item`. For every offset in the finite tile, the
checker verifies its inverse `thread = offset % threads`,
`item = offset // threads` and reconstruction. These inverses establish unique
logical ownership and complete tile coverage. The first pass adds the block's
tile offset. The launcher computes the required number of groups, uses the
selected first and finishing thread counts, and rejects unrepresentable row/count/record
extents before enqueueing. Scratch must have at least `rows * groups` records.

For a fixed tuple of loaded records, an accepted local description computes
their join, conditional on the merge laws, checker, interpreter, and compiler
semantics. Changing the accepted tree, thread count, or item count preserves
the mathematical selection result when every row's inputs reach the reduction
and the remaining execution obligations hold. This is an instance check over
logical tile ownership, not a machine-checked theorem about arbitrary GPU code.

## Checked finishing fold over a runtime count

The number of first-pass partial records is known at runtime. The finishing
pass describes their traversal with `ArgmaxFoldPlan[threads, items_per_thread,
plan_nodes]`: a stride and a local `ArgmaxPlan`. Its vocabulary is fixed
striped repetition with identity padding and the shared join, rather than an
arbitrary callback or Mojo loop body. `make_argmax_fold_plan` sets the stride
to `threads * items_per_thread` and retains the exact local description.

The public `finish_threads`, `finish_items_per_thread`, `finish_plan_nodes`,
and `finish_plan` parameters are independent of the first pass. Finishing
threads default to the first pass's thread count, items default to one, and
the tree defaults to sequential. The defaults preserve the previous per-thread
record order. Odd item counts, balanced trees, and trees with different node
counts are supported. These choices do not change first-pass scratch sizing.

`check_argmax_fold_plan` runs during compilation. It checks the symbolic
traversal derivation and local tree, requires stride to equal the tile size, and
requires `Int32.MAX + stride <= Int.MAX`. A zero/backward stride, overlapping
tiles, gaps, or an omitted local input are rejected. The finishing interpreter
consumes this checked stride and tree directly:

1. Initialize the thread accumulator to `Empty` and tile base to zero.
2. While the base is below the partial count, load records at
   `base + thread + threads * item`; out-of-range slots receive `Empty`.
3. Evaluate the accepted local tree and join its result into the accumulator.
4. Advance the base by the descriptor's stride.
5. Combine thread accumulators through the existing warp/shared-record stages.

The runtime count remains capped at `Int32.MAX`, but bases and padded indices
use `Int`: the last padded index and the post-loop base can exceed Int32.
The guard runs before computing/accessing the partial record's row address.
Public preflight bounds real row/record offsets; allocation extent, injective
layouts and nonaliasing remain caller obligations.
These count and row/scratch guarantees come from the public launcher; direct
calls to private `_finish` must establish positive count and valid row/scratch
extents themselves.

### What is checked, and where

| Obligation | Evidence and scope |
| --- | --- |
| Local tree syntax and complete input support | Compile-time checker on the selected description |
| Unique ownership and complete coverage within one tile | First-pass finite checker; finishing proof derives the inverse map symbolically |
| Contiguous tile progression and arithmetic headroom | Compile-time stride equality and integer-bound checks |
| Execution uses the checked tree and stride | Fixed interpreter specializes directly from the accepted descriptor; interpreter and compiler correctness remain trusted |
| Coverage and termination for every valid runtime count | Mojo proof kernel checks arithmetic and induction derivations over symbolic runtime variables during compilation |
| Correct device communication and addressing | Metal tests plus explicit warp, barrier, ordering, lifetime and physical-layout assumptions |

Let `S = threads * items_per_thread` and runtime count `N > 0`. For any partial
index `0 <= p < N`, Euclidean division gives exactly one owner:

```text
round = p // S
offset = p % S
thread = offset % threads
item = offset // threads
base = round * S
```

The finishing proof derives thread/item bounds and offset reconstruction
symbolically. The fixed interpreter visits bases `0, S, 2*S, ...`
below `N`, so that owner is visited
once and its load is in bounds. Tiles neither overlap nor leave gaps. Padding
contributes only identities. Inductively, each thread accumulator is the join
of its records in the tiles already visited; combining the thread accumulators
therefore gives the join of all `N` partials, conditional on the shared merge
laws and execution assumptions.

The root explicitly proves `N + S <= Int.MAX`. For the prescribed interpreter,
`base < N` and a tile offset below `S` put every padded index and the next base
below `N + S`; positive stride puts the exit base in `[N, N + S)`. These are
consequences used in the interpreter correspondence argument, rather than
separate quantified formulas in the certificate's root. Together they make
progress and padded indexing representable for every admitted `N`, including
the Int32 endpoint. This argument depends on the prescribed integer/control
semantics; the checker does not reflect an arbitrary while-loop implementation.

### Historical first-order coverage comparison

[`finishing_proof.mojo`](../src/llm_mojo/kernels/finishing_proof.mojo) exposes the
program, propositions and proof steps as compile-time Mojo values. The reusable
checker lives in
[`finishing_proof_kernel.mojo`](../src/llm_mojo/kernels/finishing_proof_kernel.mojo);
the untrusted producer lives separately in
[`finishing_derivation.mojo`](../src/llm_mojo/kernels/finishing_derivation.mojo).
`FinishingProgram` records the thread/item dimensions, start and stride,
affine index coefficients, strict loop/load guards, and count/integer limits.
`FinishingProof[size]` stores acyclic term/formula DAGs and inference steps
with references to earlier premises. Formulas contain equality, natural order,
conjunction, implication, universal/existential quantification, and predicates
for visited rounds. The coverage-only comparison rejects the legacy termination
predicate and constructors. The full historical profile is retained for
comparison tests. The checker validates each formula against
its rule's actual operands and propagates exact sets of open assumptions.
Implication introduction discharges a specific assumption; universal
introduction checks that its variable is absent from every remaining
assumption. Substitution checks prevent variable capture. Every declared node
must be valid, including nodes unused by the root; storage beyond the declared
prefix is capacity, not a proof step. The literal root is:

```text
forall N : Nat, (1 <= N and N <= count_limit) =>
  (forall p : Nat, p < N => exists q, t, i,
     Event(q,t,i,p,N) and
     forall q',t',i', Event(q',t',i',p,N) => (q'=q and t'=t and i'=i))
  and N + S <= integer_limit

Event(q,t,i,p,N) = Visited(q,N) and t < threads and i < items
                  and actual_index(Base(q),t,i) = p
```

Constructing that target formula supplies no evidence. The root must match it
and have an empty assumption set. The historical coverage-only comparison uses
`make_finishing_coverage_proof` and `check_finishing_coverage_proof`; these omit
termination and reject every legacy `FORM_DONE` formula and `DONE_STOP` or
`DONE_STEP` inference step, including unused declarations. Production coverage
and termination use the proof-term kernel below. The retained
`make_finishing_proof` / `check_finishing_proof` comparison profile also includes
`Done(0,N)` in its root.

Runtime indices, counts, quotient/remainder coordinates and loop bases remain
symbolic. Term DAGs carry variables and scalar coefficients; the kernel
constructs affine normal forms and checks coefficient equality plus
capture-avoiding substitution.
It does not enumerate runtime counts or execute a billion-iteration loop.
The canonical producer constructs a finite derivation independent of the
selected computation dimensions. The producer is not trusted to prove the
claim: the checker revalidates its output, and other well-formed derivations
may use a different order or number of nodes.

The producer constructs these derived lemmas and combines them:

- Positive dimensions and `stride = threads * items` establish positive stride.
- Euclidean decomposition supplies bounded, unique quotient/remainder pairs
  for `p / stride` and `(p % stride) / threads`.
- Substitution and arithmetic connect those pairs to the actual affine load
  index; the bounds establish valid thread and item coordinates.
- Natural induction derives `forall q, Base(q) = q * S` from
  `Base(0) = start` and `Base(q+1) = Base(q) + S`, using arithmetic to require
  the configured start to be zero.
- A second natural induction derives
  `forall q, q * S < N => Visited(q,N)`. Its step derives the predecessor's
  guard, applies the induction hypothesis, and then applies the driver's
  guarded successor rule. There is no trusted whole-loop visitation rule.
- The owner coordinates instantiate these lemmas. Two applications of the
  general Euclidean uniqueness rule establish that every other bounded owner
  has the same round, thread, and item. Existential introduction supplies the
  owner, then implication/universal introduction closes coverage.
- In the retained full comparison profile, a third natural induction derives
  finite execution. Its predicate is
  `M(k) = forall q, N <= S*(q+k) => Done(q,N)`. At `k=0`, base linearity
  turns the assumed bound into the inactive guard, yielding a zero-body run.
  At `k+1`, generic natural-order cases split the actual guard. The inactive
  branch stops; the active branch rewrites the bound to
  `N <= S*((q+1)+k)`, applies the induction hypothesis at the next round,
  and prepends one returning body execution to that finite suffix.
- In that comparison proof, since `S >= 1`, nonnegativity of `(S-1)*N`, addition, and affine
  normalization derive `N <= S*N`. Instantiating `M(k)` at `k=N` and `q=0`
  therefore yields `Done(0,N)`. This symbolic budget guides the proof;
  it does not set the execution stride or require the loop to run `N` times.
- The count limit and constant arithmetic derive the root's integer headroom.

This is the retained comparison calculus for the closed striped traversal language.
Its trusted rules are reusable equality/affine normalization, natural
order (including generic comparison cases), logical introduction/elimination,
natural induction, three Euclidean division laws, and the driver's
base/visited zero, successor and guard semantics. Finite execution has only
two operational constructors in the retained full comparison profile: `N <= Base(q)` yields `Done(q,N)` with zero
body executions; `Base(q) < N` together with `Done(q+1,N)` yields `Done(q,N)`
by prepending one returning body. The kernel checks the actual round, count,
guard and successor operands. `Done` means existence of a finite execution
tree generated by those constructors; it is not an unchecked Boolean flag.
Coverage, inverse reconstruction, owner uniqueness, base linearity, whole-loop
visitation and positive-stride termination are derived formulas. The previous
whole-loop termination opcode `140` and unary termination formula tag `8` are
unsupported, along with the older whole-lemma rules. The canonical derivation
in the retained full profile has 159 steps, independent of thread/item dimensions; the earlier
certificates had 125 steps with a trusted termination schema and 33 larger
schemas respectively.
The semantic assumptions shrink even though the generic calculus has more
opcodes and a longer derivation. Term/formula/variable capacities bound proof
complexity; they do not enumerate the computation shape. Logical variables
currently use a 32-bit mask, independent of thread or input counts.
This is not a general dependent-type prover. The Mojo implementation of the
rulebook remains trusted; its soundness has not itself been formally proved.
The two finite-execution constructors still rely on the fixed driver's
control/transition semantics and body return. The retained full-profile root proves finite execution,
without encoding an exact iteration count or proving GPU liveness.

Each chosen thread/item specialization checks its derivation during ordinary
Mojo compilation, for all runtime counts satisfying its stated admission
premises. Threads and items remain parameters, including odd item counts;
the local reduction tree is independent of the traversal proof. Pure logical
proofs also allow positive thread counts that are not valid GPU launches.
The production checker separately requires whole 32-thread warps and the
public route's exact `Int32.MAX`/`Int.MAX` limits.
The larger derivation adds compilation work, including build-on-first-use
paths. Proof checking adds no operations inside the runtime finishing loop.

`ArgmaxFoldPlan` retains the program and separate typed coverage/headroom and
termination bundles. Both public admission
and the private finishing specialization check the certificate with
`comptime assert`. Execution consumes the checked program's start, stride,
index coefficients and guards through shared helpers. The fixed interpreter,
Mojo integer semantics, compile-time evaluator and compiler lowering remain
trusted. Row addressing, physical layout, synchronization and device liveness
are outside the logical traversal proof. No external proof toolchain is used.

### Typed proof terms in Mojo

The proof-term kernel supplies production coverage, headroom and termination.
Coverage and termination use separate inductive environments in the same
language. The small
[`proof_term_kernel.mojo`](../src/llm_mojo/kernels/proof_term_kernel.mojo)
checks an embedded dependent term language; Mojo's compile-time evaluator runs
that checker. Mojo's own type system does not acquire dependent types for
arbitrary Mojo functions. The producer in
[`finishing_typed_derivation.mojo`](../src/llm_mojo/kernels/finishing_typed_derivation.mojo)
constructs untrusted proof terms. Its admission helper separately constructs
the required types below; that helper is part of the trusted specification,
not a type annotation supplied by the proof producer:

```text
base_linearity : forall q : Nat, Base(q) = S*q
termination   : forall N : Nat, Done(0,N)
```

The natural-number termination theorem includes zero and is stronger than the
positive, machine-bounded domain of the production traversal certificate.
It does not give unbounded machine integers to the executable driver. The
typed coverage certificate establishes inverse ownership, uniqueness and
integer headroom under the prescribed count bound.

The typed core has natural-number expressions, propositions, dependent
function types, lambda abstraction and application. A proof of an implication
is a function from evidence of its premise; a universally quantified proof is
a function whose result proposition depends on its natural-number argument.
Application checks the argument's type and substitutes it into the dependent
result type. Natural induction uses the eliminator derived from the checked
Nat definition: ordinary application checks a proposition-valued motive,
evidence at zero, and a step accepting the predecessor's evidence. Equality substitution
transports evidence between propositions using a checked equality. Binding,
scope and conversion are checked by the kernel; the root must be closed and
have the independently constructed expected type.

The language has `Prop` and a terminal `Type` sort: `Type` has no type and
cannot be used as a binder domain. It supports a checked zero-index Nat family
in `Type` and one checked inductive proposition family with natural-number
indices and direct positive recursive premises. It omits universe
polymorphism, mutual/nested inductives and general dependent elimination into
`Type`. A derived eliminator supports structural induction into `Prop`. Checked
structural Nat recursion also returns a fixed carrier whose sort is `Type`,
including Nat, proposition codes and ordinary function types. It never takes
`Type` itself as its carrier and adds no `Type : Type`. Conversion checks alpha/beta equality, constructor iota reduction,
compact Nat numeral/constructor conversion and literal/zero arithmetic;
it does not equate expressions merely because
they express the same mathematical function. Capture and invalid dependent
abstraction are rejected. Resource limits may reject a valid large term,
without supplying evidence for an invalid one.

The first port is the existing base-linearity induction. The second port uses
the same symbolic budget argument as above, now as proof functions and a
derived Nat eliminator application. The producer does not execute the runtime loop or
enumerate natural counts. Thread/item choices remain parameters, and proof
storage limits the complexity of the certificate rather than the computation
shape. The 32-thread, 2-item certificate uses 203 syntax nodes with capacity
256. Variables have distinct syntax-node identities, and context storage follows proof
capacity rather than a fixed number of threads or natural values. Every
declared entry is checked, including unused entries; intermediate open terms
are allowed, while both the root and its expected type must be closed.

`Done` is now declaration data rather than a trusted constant signature:

```text
family Done : Nat -> Nat -> Prop
stop : forall q,N, N <= Base(q) -> Done(q,N)
step : forall q,N, Base(q) < N -> Done(q+1,N) -> Done(q,N)
```

The generic checker first admits the family and every submitted constructor.
A constructor must have a closed, well-sorted telescope of natural-number
binders and proposition premises, ending in a saturated application of the
family. A recursive premise may only be a direct saturated family application.
Recursive occurrences in function domains, nested premises, variable-type
dependencies or natural-number indices are rejected. Constructor evidence is
unavailable during declaration validation. Once the environment is admitted,
`PT_FAMILY` and `PT_CONSTRUCTOR` have types obtained from that environment;
ordinary application checks their actual arguments. An empty family is valid
but supplies no evidence. Arity and constructor counts are bounded by proof
storage, independently of computation shape.

General admissibility does not identify the intended driver. For example,
`forall q,N, Done(q,N)` would be a valid constructor for a different positive
predicate, and would make its termination theorem trivial. The finishing
admission helper therefore independently rebuilds the intended definition
with the two signatures above, checks both environments, and compares their
constructor types by alpha/beta conversion. Definition and constructor order
are part of this specification. The tests include an unconditional alternate
predicate with a closed theorem accepted by generic type checking but rejected
by finishing admission.

The old trusted constructor IDs `5` and `6`, and the special `PT_DONE` atom
`14`, are rejected. All `PT_CONST` declarations are rejected. The two driver
equation names now select proofs derived from the checked definition of Base:

| Declaration | Signature |
| --- | --- |
| `BASE_ZERO` | `Base(0) = 0` |
| `BASE_STEP` | `forall q, Base(q+1) = Base(q)+S` |

Together with fourteen arithmetic declaration IDs (`2`, `3`, `4`, `7` through
`17`), these names select independently type-checked proof definitions through
`PT_ARITHMETIC`; they supply no arithmetic or driver-equation axiom. The shared
base-linearity/termination producer uses these checked
exports too. Equality symmetry, transitivity and congruence are expressed
through equality substitution. Compact numeral conversions have checked
derivations from Nat, as described below. Machine integer decoding and the
isolated native arithmetic and affine/literal reflection operations remain
explicit trust boundaries.
Generic sorting, closure, positivity, induction, structural recursion and
reduction remain kernel code whose soundness has not been formally proved.

The intended definition denotes a finite constructor tree: a false guard
stops immediately; a true guard prepends one returning body to a finite
suffix. Interpreting this definition as execution of the actual driver is
still a specification/correspondence assumption. Body return, compiler and
compile-time evaluator correctness, integer interpretation, physical memory,
synchronization and device liveness also remain outside this proof. The new
definition establishes no exact iteration count or GPU liveness theorem.

The production admission gate requires the typed coverage/headroom certificate,
the checked Visited and Done definitions matching their driver specifications,
and closed typed base-linearity/termination terms. Both legacy profiles remain
only for comparison tests. These checks run during compilation; none
adds proof evaluation to the runtime finishing loop. The typed checker adds
substantial compilation work when many specializations are built. Compilation
cost remains a qualification limit of this initial implementation.

### Typed coverage, visitation and headroom

[`finishing_typed_coverage.mojo`](../src/llm_mojo/kernels/finishing_typed_coverage.mojo)
constructs a separate proof-term bundle with the checked definition:

```text
Visited : Nat -> Nat -> Prop
zero : forall N, Base(0) < N -> Visited(0,N)
step : forall q,N, Visited(q,N) -> Base(q+1) < N -> Visited(q+1,N)
```

Its declaration and positivity checks use the same machinery as Done.
Admission independently rebuilds and matches these constructor signatures;
an unconditional or altered visitation predicate cannot certify the driver.
The two environments each use the existing single indexed Prop-family slot.
No additional family profile or trusted visitation signature is needed.

An untrusted producer applies Nat's derived eliminator to the motive
`forall N, S*q < N -> Visited(q,N)`. At zero it transports base linearity
into the actual guard and constructs zero visitation. In the successor step,
order and nonnegative stride give the predecessor guard; the induction
hypothesis supplies its visit, and the checked successor constructor adds the
next active round. The count and round remain symbolic.

Coverage supplies explicit inverse witnesses rather than an existential
introduction rule:

```text
q(p) = p / S    r(p) = p % S
t(p) = r(p) % T    i(p) = r(p) / T
index(q,t,i) = Base(q) + t + T*i
```

The gate independently constructs and checks ten closed theorem types: base
linearity, active-round visitation, and the following eight coverage obligations:

```text
forall N,p, p < N -> Visited(q(p),N)
forall p, t(p) < T
forall p, i(p) < I
forall p, index(q(p),t(p),i(p)) = p
forall p,q,t,i, t<T -> i<I -> index(q,t,i)=p -> q=q(p)
forall p,q,t,i, t<T -> i<I -> index(q,t,i)=p -> t=t(p)
forall p,q,t,i, t<T -> i<I -> index(q,t,i)=p -> i=i(p)
forall N, N<=count_limit -> N+S<=integer_limit
```

Together these give a bounded visited owner and its uniqueness for every
`p<N`, including all counts admitted by the former existential theorem.
The reconstruction and uniqueness results are stronger: they need no count
domain or candidate-visitation premise. Each root must have its independently
constructed type and be closed. Nine roots share the coverage bundle; headroom
has a separate raw `HeadroomProof` and an isolated check. A generic multi-root
checker shares one typing state among the nine coverage roots in each attempt;
it adds no coverage axiom.
It starts with a smaller generic scratch budget and retries with the original
bound only after storage or fuel exhaustion. A semantic rejection is final.
This changes workspace allocation, not inference or conversion rules; the
budgets depend on certificate capacity, never on thread or runtime counts.
Headroom has its own finite syntax and checking budgets. These bound accepted
certificate complexity, not runtime counts or geometry. Resource exhaustion in
its mandatory check rejects the certificate; it cannot fall back to reflection.

Natural quotient and remainder have a positive literal divisor in the
coverage language. Their mathematical definitions and fourteen arithmetic
laws are now checked from Nat, as described below. Literal scalars instantiate
coefficient-generic or divisor-predecessor-generic proofs; the library does
not generate one proof step per thread, item or runtime count. Optional closed
literal conversion requires checked raw Nat computation; affine reflection
remains a trusted interpretation used by other coverage roots. Symbolic
quotients/remainders remain opaque affine atoms. Headroom instead uses the
ordinary Nat proofs described below, with affine equality and literal-order
reflection forbidden throughout its separate certificate.

Production `ArgmaxFoldPlan` now contains this typed coverage bundle and the
separate raw headroom proof, and calls only the proof-term gate for coverage.
It does not consume a first-order
certificate or `RULE_NAT_INDUCTION` (`131`). The legacy rule remains in the
explicit comparison APIs. Both production inductions—base linearity and
visitation—use Nat's generated eliminator, as does termination.

The remaining trust includes the independent driver/theorem specification,
generic kernel soundness, structural induction and reduction, the generic
numeral-conversion checker and machine integer decoding,
immutable export import, checked operator elaboration, affine reflection in
other coverage roots, literal-order reflection available to other consumers,
and integer/compiler/evaluator semantics. Interpreting finite Visited trees as visits by the actual sealed
loop still assumes body return and driver correspondence. Physical memory,
synchronization, backend operations and GPU liveness remain outside the proof.
This port establishes no exact iteration count or full-inference determinism.

### Arithmetic derived from checked Nat definitions

[`nat_arithmetic.mojo`](../src/llm_mojo/kernels/nat_arithmetic.mojo) and
[`nat_arithmetic_division.mojo`](../src/llm_mojo/kernels/nat_arithmetic_division.mojo)
produce ordinary lambda, application, equality-substitution and Nat-induction
terms. The library contains six data definitions, thirty proof definitions and
eight representation definitions. Its capacity bounds proof storage, not geometry.
Sixteen independently required theorem types are exported to coverage and
termination; helper theorems remain checked definitions in the same library.
Construction uses a private append-only syntax index; the returned library
contains ordinary proof data. Before the Base extension, native and compile-time
comparisons with the original linear producer checked every syntax slot,
constructor, definition and export for exact equality. This producer cache
grants no typing judgment.
It is discarded before checking. The kernel independently validates the
returned definitions, terms and exports; its own syntax index remains part
of the trusted checker implementation.

The data definitions use checked structural recursion over the admitted
canonical Nat definition:

```text
NatRec(n, base, step) : A
  base : A, step : Nat -> A -> A, A : Type
NatRec(0, base, step) = base
NatRec(succ(n), base, step) = step(n, NatRec(n, base, step))

Add(a,0) = a
Add(a,succ(b)) = succ(Add(a,b))
Mul(k,0) = 0
Mul(k,succ(n)) = Add(k,Mul(k,n))
Le(0,b) = True
Le(succ(a),0) = False
Le(succ(a),succ(b)) = Le(a,b)
Lt(a,b) = Le(succ(a),b)
Pred(0) = 0
Pred(succ(n)) = n
Base(s,0) = 0
Base(s,succ(q)) = Add(Base(s,q),s)
```

`A` is inferred from the base term and independently sorted. Every major,
base and step is checked before iota reduction. `True`, `False`, conjunctions
and the counter's data tuple are lambda/Pi encodings; they add no primitive
arithmetic rule. Addition laws, order totality/transitivity, strict
irreflexivity, scaling and cancellation are closed proof functions built with
Nat's derived induction and equality transport. Coefficients are symbolic Nat
arguments. The strict scaling cancellation proof even covers coefficient
zero through its impossible premise; the production adapter retains its
positive-literal requirement.

Base is a sixth data definition, `lambda s,q. NatRec(q,0,lambda n,acc.
Add(acc,s))`. Its two exported proofs quantify over `s : Nat` (and over
`q : Nat` for the successor equation). The zero proof is `lambda s. Refl(0)`;
the step proof is `lambda s,q. Refl(Add(Base(s,q),s))`. The checker unfolds the
admitted definition and applies beta/iota conversion, so the reflexivity term's
inferred equality matches the independently required equation. The proofs use
no driver constants, arithmetic exports, affine reflection or geometry-specific
enumeration. Checking that an arbitrary body has type `Nat -> Nat -> Nat`
would not establish the intended meaning: independent body matching also
rejects well-typed constant-zero, wrong-successor and wrong-capture definitions.

The generic Base export receives `program.stride` through an ordinary checked
application. Its actual imported Pi type determines the argument domain and
the dependent substitution; no separate native theorem signature supplies the
result. The native `Base(q)` notation still interprets that selected stride
through the compact operator bridge, while affine reasoning keeps Base as an
opaque atom. Its representation definition is checked against the actual Base
definition; generic import and conversion implementations remain trusted code.
Unfolding uses raw recursion within the checker's storage and fuel bounds;
direct conversion of a successor equation at a large literal stride may exceed
those bounds by unfolding addition. Specializing its checked symbolic export
avoids enumerating that stride, including at `Int.MAX`. This is a conversion
strategy limit, separate from the computation shape.
Interpreting the checked Nat recursion as the runtime loop's zero initialization
and `base += stride` update also remains a correspondence assumption. The
definition proves its own equations; it does not reflect or verify arbitrary
Mojo statements. Positive stride, machine headroom, body return and backend
execution retain their separate admission requirements and assumptions.

For divisor `d=succ(k)`, division uses a quotient/remainder/countdown triple:

```text
state(0) = (0,0,k)
advance(q,r,0) = (succ(q),0,k)
advance(q,r,succ(c)) = (q,succ(r),c)
state(succ(n)) = advance(state(n))
Div_d(n) = quotient(state(n))
Mod_d(n) = remainder(state(n))
```

Induction on the symbolic dividend proves both invariants:

```text
n = d*quotient(state(n)) + remainder(state(n))
remainder(state(n)) + countdown(state(n)) = k
```

The second gives the remainder bound. Quotient uniqueness follows by comparing
candidate and computed quotients: either strict ordering would put one bounded
decomposition below the other, contradicting equality to the same dividend.
Rewriting the equal quotient and canceling the shared additive prefix gives
remainder uniqueness. No division theorem schema or Boolean decision oracle
is assumed.

A generic closed-definition table checks each declared type and actual body,
exact closure, and earlier-only references. Every unused definition and syntax
entry is checked. The arithmetic bootstrap rejects trusted constants,
arithmetic exports, native arithmetic syntax and arithmetic reflection. It
independently reconstructs and pins all six recursive data bodies, then checks
the actual proof definition against each required theorem type. It cannot
consult its own certification result.

An immutable compile-time certificate binds these exact library bodies and
exports. Consumer arithmetic names select references to the actual checked
definitions. The kernel lazily imports the actual declared type, preserving
its binder dependencies through a source-to-destination syntax map, and checks
its sort and closure in the consumer environment. Ordinary `PT_APP` inference
checks each argument against that Pi domain and substitutes it into the actual
codomain. Base receives `program.stride`, scaling receives its factor, and
division receives the predecessor `d-1` of its positive divisor. There is no
handwritten native theorem-signature table. Different parameter values retain
different application nodes and typing judgments.

Compact operators have eight closed raw representation definitions: Add, Mul,
Scale, Le, Lt, Base, quotient and remainder. The untrusted producer supplies
ordinary lambdas and declared types. The bootstrap checks their types, exact
closure and bodies against independently written compositions of the certified
Nat definitions, with compact syntax, compaction and arithmetic reflection
disabled. For example, strict order expands to `Le(Succ(a),b)`, and quotient
expands to the first projection of `DivMod(k,n)`. The divisor is `Succ(k)`.
Well-typed but incorrect operand order, successor placement or projection cannot
be admitted merely because the representation has the expected type.

Native operator type inference and expansion instantiate the actual checked
representation through ordinary application. A recognizer may propose a compact
operator, but the kernel accepts the conversion only after comparing the raw
source with that operator's instantiated meaning. This administrative comparison
uses alpha equality, beta substitution and unfolding of representation aliases,
plus checked canonical successor/literal conversions. A conversion certificate
can identify `Succ(0)` with literal `1`, or `Succ(k)` with the next encodable
literal, without expanding a numeral into unary syntax. The six underlying data
definitions stay opaque. The comparison does not evaluate arbitrary NatRec,
Add, Mul, division, general literal arithmetic or affine reflection. Its numeral
subchecker may replay constructor NatRec iota steps from a conversion certificate.
Thus the generic symbolic check
does not enumerate a factor, divisor, stride or runtime count, including large
literal specializations. Failure to establish this equality declines compaction.
The recognizer itself grants no conversion judgment.

When structural matching is insufficient, a separate raw unfolding fallback
disables compaction to prevent native/raw rewrite cycles. Both paths remain
bounded by generic storage and fuel limits. Bootstrap checking enables neither
immutable import nor compaction and continues to check the raw definitions and
proof bodies independently. It cannot use its own certification result.

Consumer checking keeps a smaller first workspace and retries only after actual
storage or fuel exhaustion. The full workspace includes the immutable library's
syntax-node count as well as the input proof's capacity, accounting for imported
representation definitions and their applications. This budget depends on proof
and library complexity, not thread counts, strides or runtime extent. A larger
workspace uses exactly the same typing, closure and conversion rules; it cannot
turn a semantic rejection into an accepted proof through a resource retry.

All imported library references use an internal immutable namespace separate
from caller-supplied definition IDs; callers cannot submit that namespace.
The importer, immutable certificate binding, application/argument-selection
implementation and generic administrative equality checker remain trusted kernel
code. The operator-specific meanings and each proposed resugaring are checked;
they are not separately assumed conversion rules. Export specialization still
uses the actual exported proof type rather than a reconstructed theorem signature.
Positivity inspection unfolds admitted aliases so a name cannot conceal a
negative recursive constructor field. Substitution leaves an already checked
subtree unchanged when its exact context contains no target variable; actual
capture remains rejected. Context bitsets use heap storage so larger native
certificates do not require a raised stack limit. Their maximum capacity and
fixed row width stay unchanged, but rows are zero-initialized only when syntax
nodes become live, including bulk declaration imports. Every live node still
has a complete context row; future rows have zero context-prefix metadata and
are initialized before use. This changes storage initialization, not context
membership, free-variable checking, proof budgets or inference rules. The
generic context implementation remains trusted kernel code. A checked nonzero-prefix
index skips empty context words; equality comparisons store only active binder
pairs. An exact five-field syntax index preserves the previous first matching
node, while every variable keeps a fresh identity. Native controls compare the
context prefix with the full matrix and syntax lookup with the previous linear
rule, including duplicate nodes and full-capacity behavior. These are generic
checker representations, independent of thread, tile and runtime counts.

The production proof-term path's local arithmetic/driver axiom catalog now has
zero accepted entries: all sixteen equations and laws are checked proof exports.
The separate legacy comparison calculus remains unchanged. The generic calculus,
definition/recursor and alias
machinery, immutable export/representation import and binding, administrative
equality, the generic numeral-conversion checker and machine integer decoding,
affine/literal-order reflection remain trusted implementations. Optional closed
literal arithmetic now requires the raw Nat conversion proof described below;
symbolic zero/one shortcuts that require induction are no longer definitional
conversions. Checking an operator's representation does not prove Mojo machine arithmetic.
The mathematical-to-native integer bridge and compiler/interpreter/backend
correspondence remain outside these arithmetic proofs. This proves no physical
GPU progress or full-model determinism.

### Checked compact numerals

[`nat_numerals.mojo`](../src/llm_mojo/kernels/nat_numerals.mojo) produces
compact literals' binary Nat denotations and their conversion certificates.
Compact literals and successor syntax are abbreviations for checked Nat terms.
An independent bootstrap admits closed definitions for zero, successor and
doubling from the actual canonical Nat constructors. Doubling is structural
recursion:

```text
Double(n) = NatRec(n, zero, fun predecessor previous => succ(succ(previous)))
```

This bootstrap forbids compact literals, compact successor syntax, native
arithmetic, arithmetic exports and reflection, including unused occurrences.
It cannot consult its own certificate. Raw Nat recursion must therefore reduce
actual constructor applications before the compact bridge is available.
The numeral certificate precedes arithmetic certification, whose proof terms
already use compact zero and successor. The two stages have separate immutable
bindings and import maps. Numeral definitions contain no definition references;
the consumer imports their actual bodies with fresh binder identities, so
caller-supplied definition IDs cannot select or replace them.

The untrusted producer builds a binary denotation `E(n)`:

```text
E(0)      = zero
E(2*k)    = Double(E(k))          when k > 0
E(2*k+1)  = succ(Double(E(k)))
```

Qualification reproduced a pinned compiler/evaluator defect in signed `Int`
digit-extraction loops: `Int.MAX` still had the value `9223372036854775807`,
but repeated `% 2` and `//= 2` produced only 31 digits at compile time,
rather than the expected 63. Native execution retained the upper bits. The
numeral, literal-arithmetic and headroom producers, and the kernel's independent
endpoint decoder, therefore use an explicit `UInt64` word, bit masking and
logical right shifts. Numeral caches retain nonnegative `Int` keys; each
unsigned prefix is at most the original supported signed word before conversion
back to its cache key.

Producer/checker agreement alone cannot detect a shared decoding error.
Qualification also requires independent inspection of the actual raw
`Double`/successor spines and their zero endpoint against fixed bit positions
and lengths. For example, `4294967296` requires 33 digit layers with only bit
32 set; `4294967297` additionally has bit 0 set; and `Int.MAX` requires 63
layers with every bit set. These controls execute at compile time and natively,
and check the generated Nat syntax rather than reconstructing an expected
value with the same decoder. This qualifies the known word boundaries; it is
not a checked theory of machine integers or proof of the compiler's bitwise
operations. Integer parsing, unsigned conversion and bit decoding remain part
of the stated trust boundary. The word width here is a host representation
constraint, independent of computation shape or runtime Nat counts.

Literal typing checks this actual Nat expression. A separate untrusted
conversion derivation supplies a constructor view of the same denotation:
`E(0)` converts to `zero`, and positive `E(n)` converts to
`succ(E(n-1))`. The kernel binds the requested literal, the independently
reconstructed denotation, and the exact constructor/predecessor endpoints.
A well-typed expression with an incorrect value or predecessor cannot pass
on its type alone.

The conversion language has small generic rules: reflexivity, beta reduction,
unfolding an admitted definition, constructor NatRec iota, application
congruence, symmetry and transitivity. Every rule checks its actual operands
and endpoints. Unfolding uses the actual namespace-bound body; beta uses
capture-safe substitution; iota checks the recursor and its actual constructor
argument. Congruence and transitivity check their intermediate terms. All
declared derivation entries are checked, including unused ones. This language
certifies definitional reduction steps; it does not turn arbitrary
propositional equality proofs into conversion rules.

For an even positive value, the producer obtains the half-value's
constructor view, applies congruence under `Double`, and uses the ordinary
equation `Double(succ(k)) = succ(succ(Double(k)))`. That equation follows from
beta, unfolding and constructor iota. Odd values expose their leading
successor directly; the base case explicitly checks `Double(zero) = zero`.
There is no trusted whole-numeral or carry theorem. Verification of these
derivations does not call the compact numeral bridge to justify itself.

Literal typing, zero/successor normalization, the operator administrative
comparison, and literal cases of Nat recursion and the derived Nat eliminator
consume the same checked numeral machinery. Successor folding to another
literal checks the corresponding conversion derivation. `succ(Int.MAX)`
remains a valid raw Nat expression and is preserved when no compact literal
can encode it.

Constructor exposure does not rely on native predecessor arithmetic. The producer
extracts predecessor syntax from its constructor derivation; the checker
independently reconstructs it by borrowing across binary digits. Native
predecessor and successor arithmetic only proposes optional compactions, which
must match the actual checked Nat meaning. A failed proposal leaves a raw Nat
term. This matters for the pinned toolchain: retained controls reproduced
incorrect constant-folded word arithmetic around `2^32`. The bridge does not
assume that arithmetic result is correct, and this change does not qualify
native arithmetic or reflection elsewhere in the kernel.

Binary denotations and one-layer constructor views scale with bit length,
rather than the literal's numerical value. A large literal does not produce
a unary constructor tree. This does not bound arbitrary recursive evaluation:
an observer that actually consumes every recursive hypothesis may still need
value-sized work and exhaust fuel. The reducer is intentionally incomplete:
opaque raw terms can also prevent further recursive reduction. Certificate
storage and conversion fuel
remain finite; their bounds limit accepted proofs, not thread counts, strides,
items per thread or runtime extent.

The generic conversion-rule checker, ordinary typing/reduction/substitution,
definition admission, immutable binding/import and overall kernel soundness
remain trusted implementations. Mojo integer parsing, range checks, bit
extraction and the interpretation of a machine integer as a bit sequence remain
trusted; there is no checked machine-word theory here. Affine/literal-order reflection,
compiler/evaluator correctness, driver correspondence, body return and physical
GPU execution assumptions also remain outside this numeral derivation.

### Checked literal arithmetic conversion

[`nat_literal_arithmetic.mojo`](../src/llm_mojo/kernels/nat_literal_arithmetic.mojo)
supplies ordinary proof terms for optional closed addition, multiplication,
scaling, Base, quotient and remainder computations. For example, the candidate
`3 + 2 = 5` supplies `Refl(Add(E(3), E(2)))`. The kernel checks this term against
the independently constructed type `Eq(Add(E(3), E(2)), E(5))`. Reflexivity has
that type only when ordinary definitional conversion checks both endpoints.
The result is a checked proof, rather than a native arithmetic result asserted
as an equation.

The check uses a fresh environment containing the three raw numeral definitions
and the six actual arithmetic data definitions. It independently checks their
canonical recursive meanings without consulting its own consumer's arithmetic
certificate, and binds
their types and bodies, the exact requested operation, operand order, positive
divisor predecessor, quotient/remainder projection and result denotation. It
checks the predecessor's successor against the decoded positive divisor by raw
conversion, so binary borrow itself grants no arithmetic equation. It
checks canonical Nat, every syntax entry including unused entries, closed
definitions and the proof's closure. Compact literals, successor shorthand,
native operators, arithmetic exports, private references and affine/order
reflection are forbidden throughout this environment. Its numeral, arithmetic
and fast-normalization flags remain disabled. The existing typechecker therefore
uses alpha/beta/delta and structural constructor/NatRec computation; no new
whole-arithmetic rule or equality-reflection principle is admitted.

Native `+`, `*`, `//` and `%` propose compact payloads only. A wrong proposal,
including a wrapped result or a compiler constant-folding error, cannot authorize
a conversion. Failed or exhausted optional checking preserves the original typed
expression and uses no consumer normalization fuel. Both production divisor
adapters extract the predecessor from the checked positive numeral view, rather
than computing the divisor minus one in a native word.

This is bounded definitional computation, not a binary arithmetic algorithm.
Raw evaluation can unfold a binary denotation into unary work and exhaust proof
storage or fuel even for a valid equation involving a large literal. Such an
equation may remain opaque. Symbolic `Add(x,0)` is definitional in the admitted
right-recursive Add definition; `Add(0,x)`, zero/one scaling and division/modulo
by one generally need induction and are no longer automatic conversions.
Propositional arithmetic theorems remain available through explicit proof terms
and substitution. No runtime extent, number of threads or items per thread is
enumerated by these checks; finite workspace limits accepted reductions only.

Remaining trust includes generic typing, definition admission, substitution and
conversion implementation; native integer parsing and bit decoding; affine
equality and normalized literal-order reflection; driver descriptor arithmetic
and correspondence to runtime code; compiler/evaluator correctness, body return
and GPU execution assumptions. Neither this kernel nor these implementation
bridges have a formal soundness proof. Host proof checks establish no GPU/backend
or full-model determinism claim.

### Headroom from ordinary Nat proofs

[`nat_headroom.mojo`](../src/llm_mojo/kernels/nat_headroom.mojo) produces the
separate certificate required by coverage. Its raw environment contains the
three numeral definitions and six arithmetic data definitions above, followed
by thirteen ordinary helper definitions. Nat's derived induction checks
addition with a successor on the left, distribution of doubling over addition,
eight digit/carry cases, and `Le(x, Add(x, gap))`. The existing ordinary
`LeAdd` and `LeTrans` proof producers supply the last two helpers after their
syntax is translated to this local raw namespace. Every helper's declared type,
body and closure must pass the generic checker; their signatures grant no rule.

Binary addition chooses applications of the eight checked carry lemmas. The
first addition supplies an ordinary equality proof
`Add(E(count_limit), E(stride)) = sum`. Binary borrow proposes a Nat `slack`;
a second addition must prove `Add(sum, slack) = E(integer_limit)`. Neither
digit arithmetic nor borrow asserts an equation or comparison. In particular,
an overflowing or insufficient bound cannot be certified by a wrapped native
sum. Equality substitution and the self-add lemma give the closed bound
`Le(Add(E(count_limit), E(stride)), E(integer_limit))`. Ordinary application of
`LeAdd` and `LeTrans` then supplies the full theorem:

```text
forall n : Nat, Le(n, E(count_limit)) ->
  Le(Add(n, E(stride)), E(integer_limit))
```

The headroom checker independently decodes the requested cap, stride and limit,
reconstructs this theorem, pins canonical Nat and all nine data definitions,
and checks the sum and slack equalities, bound, final proof and every unused
syntax entry. It starts in a fresh state with native numeral, arithmetic and
fast-normalization shortcuts disabled. Compact literals, successor shorthand,
native operators, arithmetic exports, private references, affine equality and
literal-order reflection are forbidden throughout this environment. No
reflected judgment or cached type from another coverage root enters it.
There is no trusted sum, carry, comparison or whole-headroom rule, and no
reflection fallback on a semantic or resource failure.

The retained `headroom-nat-check-v3` attempt is superseded for word-meaning
qualification. Its focused proof positives checked Nat derivations, but did
not expose the signed-loop decoder defect; fold admission revealed that gap.
A Nat proof about an incorrectly decoded limit does not establish the intended
machine-word headroom statement. The independent structural word controls
above are therefore part of the qualification boundary for this bridge.

Structural alpha comparison, administrative beta reduction and application
congruence let matching shared binary expressions be compared before raw
recursors are unfolded. These are generic conversion rules, rather than
arithmetic shortcuts. Identical syntax nodes can use immediate reflexivity
when no binder renaming is active, or when their checked free-variable context
is empty. Open terms under active renamings still require the actual binder
comparison; identical node indices alone do not identify their bound meaning.
The exact context invariant already used by substitution remains part of the
generic trusted implementation. This optimization changes neither conversion
rules nor the storage and fuel bounds. The checker may
reject a valid derivation when those resources are insufficient. The proof
quantifies over arbitrary Nat counts, and its construction does not enumerate
threads, items or runtime counts.

Affine reflection remains in other coverage roots, and the generic
literal-order reflection operation remains available outside this raw checker.
Remaining headroom trust includes the independent theorem specification,
integer parsing and bit decoding, generic definition admission, induction,
typing, substitution and conversion implementations, and the compiler's
evaluation of that code. The descriptor's machine arithmetic and its
correspondence with the runtime loop remain separate assumptions. Body return,
memory, synchronization, backend correctness and GPU progress remain outside
this Nat proof. The implementation has no formal kernel soundness proof;
qualification results must be recorded separately from these design claims.

### Private record-population commutation

[`finishing_phase.mojo`](../src/llm_mojo/kernels/finishing_phase.mojo) models
one logical tile's private record population for a fixed row, base and count.
This ends before `evaluate_argmax_plan` in `_finish`. A `Fill(thread, item)`
reads an immutable partial record, or chooses `Empty` for padding, and writes
one private three-word record and its readiness cell. The destination key is
`thread + threads * item`; the source additionally includes the tile base.
The input array denotes the fixed row's partial records; production's row
offset is outside this model. Input addresses and private destinations therefore
have separate roles.

The phase and state checks require positive thread/item counts, representable
tile and padded indices, and enough private storage. Extra state capacity is
outside the tile and is preserved. An inactive phase enables no fills. In an
active phase, a fill requires valid owner coordinates, the prescribed source
and destination, and an unready destination. Execution additionally requires
the immutable input array to cover the declared count. Padding still writes
`Empty` and marks its slot ready. A thread is ready for its local reduction
only when all its own slots, including padding, are ready.

`FinishingPhaseCommutation[C]` retains the selected phase, fills and checked
Nat proof material. `make_finishing_phase_commutation(phase, left, right)`
constructs the certificate;
`check_finishing_phase_commutation_request(cert, expected_phase, left, right)`
binds it to the caller's actual request. Acceptance requires checked evidence
that the selected private destination keys are apart, rather than trusting a
native Boolean declaration of independence.

[`finishing_phase_check.mojo`](../src/llm_mojo/kernels/finishing_phase_check.mojo)
independently specifies the eight logical/operational data definitions and the
required universal and selected theorem types. It checks the actual definition
bodies by conversion, admits ordinary helpers with checked bodies, checks unused
syntax as well as selected roots, and requires closed proofs. Only the existing
certified numeral abbreviations are enabled. Native arithmetic operators,
arithmetic exports and affine/literal-order reflection are forbidden. The
selected literal keys require a checked proof of `EqB(left, right) = 1`, where
zero denotes equality and one denotes inequality. Its proved discrimination
lemma then supplies their apartness witness.

The mathematical state is a Nat-indexed function. Ordinary checked recursive
definitions implement selection and point update; Nat's derived induction
checks the read/frame and pointwise commutation proofs. There is no trusted
whole-commutation rule or enumeration of record values. The same scalar-cell
theorem applies separately to rank, token, invalid and readiness; no assumed
record-packing conversion is needed. Pointwise equality avoids assuming
function extensionality. For two distinct enabled fills with valid immutable
input storage, each preserves the other's readiness prerequisite, and either
order gives the same four modeled cell values while preserving other cells.

Finite syntax storage and conversion fuel bound proof checking. In particular,
the recursively defined equality selector can require work proportional to a
large common predecessor prefix when checking a selected pair of literal keys;
binary numeral notation does not make that recursion a binary algorithm.
Resource exhaustion rejects the request and grants no independence judgment.
These are limits on accepted certificates, not a fixed thread/item geometry.

This is a standalone execution model and proof checker. Production `_finish`
is unchanged, and the model introduces no runtime barrier. Threads may finish
their own loads, reduce and advance independently; grouping events by a common
logical tile base is not an all-thread synchronization boundary. The readiness
predicate describes the prerequisite for local reduction, not a reduction
transition. Accumulator updates, final warp/shared-memory stages, coverage of
all execution traces, deadlock freedom and hardware progress are outside this
commutation statement.

Remaining trust includes the generic proof kernel and its recursion,
substitution and conversion implementation; the independent theorem/request
specification and its binding to the native model; native integer decoding and
address arithmetic; and correspondence between the model and private arrays
in the Mojo source. Immutable inputs, valid extents, injective layouts,
nonaliasing storage, launch ordering and visibility, compiler lowering and GPU
memory/barrier behavior remain execution assumptions. This does not supply a
GPU weak-memory model or establish end-to-end determinism. Implementation
qualification must be recorded separately from these proof and scope claims.

### Complete logical finishing execution

The larger [finishing executor](finishing-executor.md) connects private fills,
local trees, repeated accumulation and logical final aggregation. A checked
Nat-induction proof relates complete structurally legal register traces to a
canonical flat join, under explicitly quantified algebra laws. Input/Empty
initialization and the final root equation are derived from structural support
premises rather than assumed as value assertions. A native staged interpreter
and fresh-register provenance replay connect this theorem to concrete event
histories and the production descriptor.

This is a conditional abstract-value result. Native record-algebra refinement,
event/source correspondence and physical warp/shared-memory behavior remain
explicit boundaries; actual device qualification and full-model determinism
are separate claims. The executor document defines the exact theorem,
reproduction commands and qualification policy.

The [derived join algebra](finishing-join-algebra.md) supplies checked laws for
a total Nat score/Boolean-flag join and applies that instance to the canonical
trace theorem. The native UInt32 rank/token interpretation remains a separate
proof obligation; this extension does not silently discharge it.

### Checked Nat and derived induction

Nat is also declaration data, admitted by the same family and constructor
checker used for Done:

```text
family Nat : Type
zero : Nat
succ : Nat -> Nat
```

The kernel checks closed, sorted constructor telescopes and direct positive
recursion. For this zero-index Type profile, fields must be direct Nat
arguments; nested recursion, other data carriers and proposition guards are
unsupported. Nat constructor arguments are recursive; Nat parameters in a
Done constructor remain ordinary data arguments. Declaration checking makes
only the carrier's sort available, so constructor evidence or elimination
cannot justify its own declaration.

The shared generator inserts an induction hypothesis after each recursive
argument and derives:

```text
Nat.elim : forall P : (Nat -> Prop),
  P(zero) ->
  (forall n : Nat, ih : P(n), P(succ(n))) ->
  forall n : Nat, P(n)
```

This generated signature must itself be closed and sorted. Ordinary
application checks its motive, base, step and target. Constructor computation
uses the same telescope replay and recursive-hypothesis machinery as Done.
The specialized `PT_NAT_IND` expression (`16`) is rejected, including unused
instances. Base linearity and the symbolic budget induction for termination
now use `PT_ELIMINATOR` for Nat, with no separate whole-Nat induction checker.
The positive test independently constructs the expected signature and checks
both symbolic constructor computation and compact literal computation.

Generic declaration admissibility alone does not establish the intended Nat.
An empty or binary-recursive Type definition may satisfy the restricted
positivity check. Before native proof admission, the kernel independently
requires exactly the zero and successor signatures in their declared order.
Only this canonical match permits the arithmetic carrier and the separate
checked numeral bootstrap. Family `1` identifies Nat; family `0`
retains the indexed Prop profile used by Done.

The checked compact numeral derivations described above expose one literal
constructor layer at a time through the shared recursion/elimination machinery.
Arithmetic operations and order laws are defined and checked through structural
Nat recursion and induction. Closed literal arithmetic additionally requires
the raw conversion proof above. Machine integer decoding and affine/literal-order
reflection remain trusted. Symbolic
logical counts remain arbitrary naturals; native integer storage, conversion
fuel and proof capacity can limit accepted certificates. They do not fix the
GPU shape or require counts to be enumerated at compile time.

This removes the Nat-specific induction schema from the dependent proof-term
kernel. Production coverage also uses this derived rule; the separate
first-order comparison calculus retains `RULE_NAT_INDUCTION` (`131`) only
for its historical tests. The generic structural-induction principle,
signature generator, reduction implementation and overall kernel soundness
also remain trusted. Neither checker has a formal soundness proof. Nat's
interpretation as mathematical naturals, the generic numeral-conversion checker
and machine integer decoding, native
export/representation import and generic conversion, arithmetic reflection,
driver correspondence and body return,
compiler/evaluator correctness and GPU execution assumptions remain explicit
boundaries. Both families retain their dependent eliminators into `Prop`;
Nat also supports the fixed-carrier structural recursor described above.
General Type-valued dependent elimination, mutual/nested definitions and
reflection of arbitrary recursive Mojo code remain absent.

### Deriving the Done eliminator

`PT_ELIMINATOR` identifies the admitted family. It contains no user-supplied
signature or branch types. After declaration checking, the generic kernel
derives a dependent eliminator from that family's constructor telescopes.
For `Done`, the generated type is:

```text
Done.elim :
  forall P : (forall q,N : Nat, h : Done(q,N), Prop),
  (forall q,N : Nat, g : N <= Base(q), P(q,N,stop(q,N,g))) ->
  (forall q,N : Nat, g : Base(q) < N, h : Done(q+1,N),
     ih : P(q+1,N,h), P(q,N,step(q,N,g,h))) ->
  forall q,N : Nat, h : Done(q,N), P(q,N,h)
```

The generator preserves each original constructor argument. After each direct
recursive premise it inserts an induction hypothesis at that premise's actual
indices and witness. The branch result applies the motive to the original
constructor's result indices and constructed witness. Fresh binder identities
and telescope substitution preserve dependencies. The kernel checks that the
generated type is closed and sorted in `Prop`; ordinary function application
then checks motives, branches, indices and the major proof. There is no fixed
`DONE_ELIM` declaration in the arithmetic/driver catalog.

Conversion implements the corresponding constructor computation, called iota
reduction. A saturated elimination of STOP reduces to the STOP branch applied
to its original arguments. Eliminating STEP applies its branch to those
arguments and recursively eliminates the suffix to obtain its induction
hypothesis. The same implementation handles other admitted families, any
supported number of indices or constructors, and multiple direct recursive
premises. It checks all operand types, major indices, and the resulting branch
type before reducing. A neutral major proof stays unreduced. Fuel and scratch
limits cause rejection; they do not bound the runtime count or launch shape.

The finishing helper independently constructs the required `Done.elim` type
and checks the generated eliminator against it. Another untrusted producer
uses it to prove the independently specified theorem:

```text
forall q,N : Nat, h : Done(q,N), forall M : Nat, M <= N -> Done(q,M)
```

In STOP, order transitivity gives `M <= Base(q)`. In STEP, the generated
induction hypothesis supplies `Done(q+1,M)`; natural-order cases decide the
actual smaller-count guard and either stop or prepend one body. The original
suffix only proves `Done(q+1,N)`, so this branch needs the induction hypothesis.
All counts and rounds remain symbolic. An eliminator consumes existing `Done`
evidence; declaration admission alone does not establish termination. Nat's
derived induction rule still constructs the production theorem `forall N, Done(0,N)`.

These standalone compile-time checks leave the production certificate and
runtime finishing loop unchanged. Tests cover evidence-dependent motives,
generic index arities zero/one/three, an empty family, multiple recursive
premises, constructor computation, malformed applications, and flexible
finishing dimensions. Compiler-negative controls require the actual
admission diagnostic and absence of an executable.

The structural-induction principle, signature generator and iota implementation
are part of the trusted generic kernel; their soundness is not itself proved.
The generated signature is checked, rather than accepted as a driver-specific
axiom. Positivity and typing checks do not prove their own implementation or
the correspondence between finite constructor trees and device execution.
Immutable export/representation import, generic conversion, arithmetic reflection, driver interpretation, body
return, integer/compiler/evaluator semantics and GPU execution assumptions
remain. Done elimination stays inside `Prop`; the separate Nat recursor returns
fixed data carriers. This does not add general recursive Mojo reflection,
dependent elimination into data/universes, mutual/nested definitions,
exact iteration counts, GPU liveness or end-to-end inference determinism.

## Execution boundary and reproduction

The first-pass, finishing-tile, accumulator and shared-memory stages use the
shared scalar merge. The finishing pass has a checked descriptor and a
machine-checked generic traversal model; the compile-time checks establish its
finite shape and stride constraints. The proof kernel and fixed execution
interpreter remain trusted, as do the scalar merge laws and tree interpreter.
The traversal proof does not prove reduction-value semantics.
Warp reductions still use backend `max`/`min`; `max` agrees with OR only
because invalid flags are Boolean. Integer semantics, compile-time evaluation,
compiler lowering, warp operations, synchronization, and memory operations
remain trusted. The host test forces the exhaustive scans during compilation
and repeats them in compiled host code. Ordinary production builds do not run
the exhaustive scans; running the validation suite is the acceptance gate.

Callers must provide valid extents and device-feasible launch parameters,
injective layouts over accessed cells, nonaliasing input/scratch/output
storage, correct copy/submission order, and buffer lifetime through device
completion. The execution route requires Metal and 32-thread warps; the runtime
must support the chosen thread count and shared-memory footprint. The checks
do not verify layout injectivity, physical memory ownership, arbitrary runtime
loop bodies, shared-memory visibility, barriers, backend warp semantics, or
end-to-end inference determinism. Deterministic selection of fixed BF16 logits
does not establish that different model schedules produce identical logits.

From the repository root:

```sh
uv run --locked mojo run -I src tests/test_token_selection_contract.mojo
uv run --locked mojo run -I src -I tests tests/test_reduction_plan.mojo
uv run --locked mojo run -I src tests/test_argmax_schedule.mojo
uv run --locked python -m unittest discover -s tests -p test_token_selection_compile.py
uv run --locked mojo run -I src -I tests tests/test_finishing_proof.mojo
uv run --locked mojo run -I src -I tests tests/test_proof_terms.mojo
uv run --locked mojo run -I src -I tests tests/test_inductive_proofs.mojo
uv run --locked mojo run -I src -I tests tests/test_inductive_eliminator.mojo
uv run --locked mojo run -I src -I tests tests/test_nat_inductive.mojo
uv run --locked mojo run -I src -I tests tests/test_nat_arithmetic.mojo
uv run --locked mojo run -I src -I tests tests/test_typed_coverage.mojo
uv run --locked mojo run -I src -I tests tests/test_finishing_phase.mojo
uv run --locked python -m unittest discover -s tests -p test_proof_term_compile.py
MODULAR_DEBUG=device-sync-mode uv run --locked mojo run -I src tests/test_token_selection.mojo
```

The host Mojo commands force positive compile-time checks and exercise
compiled host implementations. The reduction-plan suite checks sequential,
balanced, reordered/duplicated, and earlier-root descriptions against an
independent two-pass selection oracle at 1, 3, 8, 33, and 65 inputs, including
trees with different node counts. Its native malformed-plan checks
include dead invalid nodes and noncanonical operands. The Python suite checks
successful compilation and rejection of deliberately incorrect
variants, including omissions above a machine-word boundary and invalid launch
geometry and fold strides. Host folding tests simulate per-thread accumulators
over multiple runtime tiles against an independent flat oracle, and check the
Int32 endpoint without allocating or enumerating that many records. The last
command checks the actual GPU selection route across eight
thread/item configurations, both sequential and balanced plans, and row-major
and column-major layouts. It retains the exhaustive BF16/default vocabulary
cases, checks per-group and final records against an independent oracle, and
guards inactive storage. Its device/backend evidence is required for a claim
about Metal execution.

The native proof suite forces positive derivations during compilation and
checks malformed certificates in host code. The Python compiler suite requires
the valid proof to build and execute before checking intended proof-rejection
diagnostics for malformed derivations and programs. Existing validation runs
both suites with the locked Mojo toolchain.

Direct finishing tests supply canonical partial records at runtime counts
around thread/tile boundaries and through three full tiles plus a tail. They
cover sequential, balanced and compact trees, distinct first/final geometries,
row/column layouts, tail winners, lowest-ID ties, poisoned tail records, and
unchanged input/output guards. The public dispatcher also exercises a compact
five-input finishing plan with a different node count.

The existing operation benchmark includes a bounded `token_selection` route:

```sh
uv run --locked mojo build -I src src/llm_mojo/benchmarks/operations.mojo -o build/operations
build/operations token_selection 1 1 1 0 0 53 bench 10 10
```

It uses BF16 logits with 151,936 columns, default launch geometry, and resident
input/scratch/output buffers. Variant 0 is sequential; variant 1 changes the
first-pass tree to balanced. Both finish with 128 threads and one item. Variant
2 keeps the default first pass and finishes with 64 threads, three items and
a balanced tree. Substitute candidate `2` in the command to compare that shape.
Allocation and numerical/guard checks precede timing. Each sample measures
host enqueue through synchronization, with one call or 24 repeated calls to
the same buffers according to the third argument. This is an operation
regression diagnostic, not full-token performance evidence. Use the paired
protocol and source/device provenance in [experiments.md](experiments.md) for
a retained performance conclusion.
