# A derived join for complete finishing traces

The [whole-execution theorem](finishing-executor.md) says that complete legal
register traces produce a canonical flat join, assuming five algebra laws.
This layer supplies ordinary checked proofs of those laws for a concrete
mathematical operation: maximum score paired with Boolean invalidity. Its
specialized trace theorem has no algebra-law premises.

The values here are Nat. Relating native UInt32 rank/token records to these
values is a further refinement obligation. The production descriptor retains
its previous conditional certificate until that connection is established.
This layer does not prove the native `merge_argmax` body or GPU execution.

## Define the operation on every Nat

Use a Nat-recursive clock whose states alternate as follows:

```text
(q, 0) -> (q, 1) -> (succ(q), 0)
```

Its projections define `Half` and `Parity`. `Double` is ordinary Nat recursion,
and `Pack(q,b)` chooses `Double(q)` for zero or `succ(Double(q))` otherwise.
For Boolean `b`, this represents `2*q+b`. The proof derives:

```text
forall n, IsBit(Parity(n))
forall n, Pack(Half(n), Parity(n)) = n
forall q b, IsBit(b) -> Half(Pack(q,b)) = q
forall q b, IsBit(b) -> Parity(Pack(q,b)) = b
```

`IsBit(b)` is the checked proposition `b=0 or b=1`, encoded by ordinary
proposition functions. Its elimination transports the zero and one cases
through equality. It is not a special Boolean axiom or native reflection rule.

Define structural maximum by its three constructor equations:

```text
Max(0,b) = b
Max(succ(a),0) = succ(a)
Max(succ(a),succ(b)) = succ(Max(a,b))
```

Nat induction derives its two zero identities, commutativity, associativity
and idempotence. `Or(a,b)` chooses `b` when `a=0` and one otherwise. Its laws
retain the required `IsBit` hypotheses; they are proved by ordinary checked
zero/one cases, equality transport and reflexivity.

The total join is then:

```text
Join(n,m) = Pack(Max(Half(n),Half(m)), Or(Parity(n),Parity(m)))
Empty = 0
```

Every Nat has the required decomposition, so this operation satisfies all five
laws on **all** Nat. It does not merely canonicalize values in a bounded native
encoding. That distinction is essential: decoding a bounded encoding and
discarding out-of-range bits would generally violate `Join(n,n)=n` outside its
image.

## Check and use the proof terms

The untrusted producers are
[`nat_join_max.mojo`](../src/llm_mojo/kernels/nat_join_max.mojo),
[`nat_join_bits.mojo`](../src/llm_mojo/kernels/nat_join_bits.mojo),
[`nat_join_bool.mojo`](../src/llm_mojo/kernels/nat_join_bool.mojo), and
[`finishing_join_terms.mojo`](../src/llm_mojo/kernels/finishing_join_terms.mojo).
They append ordinary definitions and proofs to the existing execution theory:

| Definitions | Meaning |
| --- | --- |
| 55–60 | Structural maximum and its laws |
| 61–79 | Doubling, pair clock, half/parity, Boolean evidence and packing inverses |
| 80–85 | Boolean OR closure and algebra laws |
| 86–93 | Total join, its projections and all five laws |
| 94 | Canonical trace theorem with the algebra arguments supplied |

Definition 94 is an ordinary application of the prior canonical theorem to
`Join`, zero, and the derived law proofs. It retains symbolic input count and
trace length. Its premises concern prescribed input/Empty support, structural
readiness and support union, and complete root coverage. There is no assumed
initial-value invariant, root-value equation, or algebra-law binder.

[`finishing_join_check.mojo`](../src/llm_mojo/kernels/finishing_join_check.mojo)
independently reconstructs the raw definitions and export types. It admits all
ordinary definitions and submitted syntax, including unused nodes, and checks
closed exports. The final root is checked by its type; equivalent proof terms
can be accepted. Metadata bounds are checked before allocating checker state.

No primitive kernel rule is added. This profile installs no general arithmetic
export namespace and enables no native arithmetic or affine/order reflection.
Previously certified Nat
numeral abbreviations remain available through the existing conversion path.

## Native interpretation still to prove

For native records with Boolean invalidity, a candidate encoding is:

```text
B = 2^32; M = B-1
score(rank,token) = rank*B + (M-token)
encode(rank,token,bad) = Pack(score(rank,token),bad)
```

The [correlated record layer](finishing-record-execution.md) derives bounded
complement, score order versus rank/token order, injectivity and merge
preservation for arbitrary Nat bounds and Boolean flags. Native UInt32
interpretation and the actual Mojo merge body's correspondence remain separate
obligations. None of those equations is supplied as an axiom by this layer.
The mathematical encoded value can require 65 bits; computing
it in native Int or UInt64 would not be a valid general implementation.

Production warp aggregation requires invalidity in `{0,1}`. The broader native
execution interpreter also tests multibit invalid words. Refining that whole
domain additionally needs a width-parametric word-OR and packing proof; the
Boolean theorem does not cover that broader interpretation automatically.

## Qualification and trust

Run the focused suite with the toolchain's default optimization:

```bash
uv run --locked mojo run -I src -I tests tests/test_finishing_join.mojo
uv run --locked python tests/test_proof_term_compile.py ProofTermCompileTests.test_finishing_join_is_checked
```

Qualification results belong under `build/finishing-record-algebra-check/` and
must bind exact source snapshots, compiler/SDK identity, logs and executable
bytes. A positive optimized build and native execution must pass before exact
negative controls. Stack failure, timeout or exhausted checker workspace is a
resource failure, not evidence that a mathematical statement is false. Native
O0 constructor temporaries exceeded the ordinary stack during development;
that sealed checkpoint retained the original initializer and ordinary process
limits in its default-optimization qualification path. The later
[record extension](finishing-record-execution.md) refactors one canonical state
constructor for in-place initialization, retaining its normal-return values and
all typing, conversion and induction rules, and reruns affected proof gates.

The sealed whole-executor evidence remains preserved under
`build/finishing-execution-check/`. Its source snapshot is the baseline for this
extension; later source changes do not rewrite that historical result.

Remaining trust includes the kernel's typing, induction, substitution and
conversion implementation, the compile-time evaluator, and the independent
admission specification. Native word interpretation, branch/source
correspondence, event-to-register refinement, valid immutable inputs, layout
and lifetime assumptions remain explicit. Compiler lowering, warp
participation, shared-memory visibility, synchronization and device progress
still require their own qualification. This is not a full-model floating-point
determinism or performance result.
