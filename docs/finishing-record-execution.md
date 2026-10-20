# Correlated record execution in the Mojo proof kernel

The finishing proof now carries rank, token and invalid flag together through
one mathematical execution. Every complete legal finite trace has the same
three output fields as an independent input fold. The count, token bound,
register stores, event streams and trace length remain symbolic.

This extends the [complete finishing executor](finishing-executor.md) and
[total Nat join](finishing-join-algebra.md). It derives the record algebra and
encoding bridge rather than asking callers to assume them. The production
certificate keeps its existing conditional profile until native word and
source-to-trace correspondence are established separately.

## The mathematical result

This layer models records as three correlated natural-number fields:
`(rank, token, bad)`. The token bound `M` is symbolic, ranks are arbitrary Nat,
and valid records require `token <= M` and `bad = 0 or bad = 1`.

For input functions `Ranks`, `Tokens`, `Bads`, input count `N`, support map
`Support`, destination/left/right streams `D/L/R`, trace length `n`, and root
register `r`, the final exported theorem has this shape:

```text
ValidInputs(M, Ranks, Tokens, Bads, N)
  + Support(N) is empty
  + each input i < N has singleton input support
  + a legal structural support/readiness trace of length n
  + root r is ready after n steps
  + Support(r) contains every input below N
  -> rank(Run(..., n)[r]) = rank(Fold(M, Ranks, Tokens, Bads, N))
   & token(Run(..., n)[r]) = token(Fold(M, Ranks, Tokens, Bads, N))
   & bad(Run(..., n)[r]) = bad(Fold(M, Ranks, Tokens, Bads, N))
```

`ValidInputs` quantifies only over indices with evidence of `Before(i,N)`.
It makes no validity assertion about unread input locations. The support
premises describe provenance and readiness. Their pointwise equations concern
input membership and support union, rather than the desired output value.

The final statement has no assumed initial value invariant, encoded-value
agreement, record-algebra law, score-order law, or final root equation.
Those connections are derived intermediate theorems, checked by the kernel.
The conclusion is three field equalities; it does not assume function
extensionality for the function-valued record representation.

Both `N` and `n` remain Nat binders. Zero counts are allowed by this mathematical
statement; existing native/production entry points retain their own extent and
positive-count contracts. Proof storage capacities do not unroll either count.

## Why the record encoding preserves selection

The mathematical token complement is defined by Nat recursion. Under
`token <= M`, its conservation, boundedness, injectivity and reversed order
are derived using ordinary arithmetic proofs. Define:

```text
B = succ(M)
Score(M, rank, token) = rank * B + Complement(M, token)
Encode(M, rank, token, bad) = Pack(Score(M, rank, token), bad)
```

A strict rank increase dominates the entire token-complement range. Equal
ranks compare by reversed token order. Together with the checked Half/Parity
projection and repacking proofs, this derives encoding injectivity on valid
records and exact preservation of the mathematical merge by the existing
total Nat join. None of these equations is supplied through a trusted
arithmetic or record-reflection rule.

The closed selector is an actual Nat-valued definition. B wins when A's rank
is before B's rank, or when ranks are equal and B's token is before A's token.
The mathematical rank and token merge components use that same selector on
the identical operand tuple. Boolean OR combines both invalid flags, including
the losing record's flag. The empty record is `(0,M,0)`; its encoding equals the
Nat identity zero by a derived theorem.

This is an arbitrary-bound Nat result. Substituting `M = 2^32-1` describes a
candidate interpretation of native UInt32 fields, but the source-level and
machine-integer refinement must still be established separately.

## Correlation is retained throughout execution

A record is represented by a Nat-to-Nat field function. Field zero is rank,
field one is token, and fields two and above return the invalidity flag. A
whole register state is a field-to-slot function.

`Run` uses one Nat recursor over that whole state. A step reads both operand
rows from the same old state, computes their correlated merged record, and
returns all destination-field updates together. It does not update rank and
then use that new rank to choose a token. Rank, token and bad therefore refer
to the same pair of logical records at every step. This is a statement about
the mathematical old-state definition, not yet a physical memory-aliasing or
GPU visibility theorem.

The independent reference `Fold` has only `M`, the three input functions and
`N` as arguments. It starts at `(0,M,0)` and folds inputs `0..N-1`. It takes no
destination stream, source stream, schedule, support map or execution state.
Its definition is independent of `Run`. A derived fold theorem relates its
encoding to the already checked canonical flat Nat join.

The invariant is deliberately asymmetric:

```text
ReadyValid: for every slot, Ready(slot)=1 -> Valid(record at slot)
EncodedEqual: for every slot, Encode(record at slot)=Values(slot)
Invariant: ReadyValid and EncodedEqual
```

Validity is required only at ready locations. Encoding agreement is pointwise
over all slots, including unread ones. This distinction keeps input functions
outside the requested prefix arbitrary without weakening value correspondence.

Initialization overwrites slot `N` with `(0,M,0)` in the three field stores.
The scalar store overwrites that same slot with zero. Update-hit proofs derive
the sentinel's validity and encoding; update-miss proofs preserve other input
locations. Away from the sentinel, a ready flag gives `Before(slot,N)`, so only
the declared prefix validity premise is used. The initial invariant is proved,
not requested from callers.

Step preservation combines readiness, valid-merge closure and encoding
preservation. Nat induction yields prefix refinement. At a complete ready root,
the old closed canonical join theorem and the independent record fold theorem
give equal encodings; derived injectivity gives the three field equations.

## Shape remains a parameter

The theorem quantifies over register names and all D/L/R streams, rather than
fixing eight threads, one reduction tree, a subgroup count or a trace length.
Different complete legal traces can therefore be compared through the same
canonical input fold. The native descriptor still supplies the chosen thread,
item, subgroup and tree parameters, subject to its existing validity checks.

The proof does not assert that every scheduler terminates or eventually visits
every enabled event. Existing coverage/termination certificates and native
remaining-event checks are separate obligations. Nor does the theorem's
idempotence allowance for repeated reduction operands replace exact input-load
visit counting in the native execution tests.

## Independent checking and resource outcomes

The common untrusted maker returns one heap-held `FinishingRecordProof` with a
bundle and untrusted layout metadata. The independent checker imports no
producer. It reconstructs raw data bodies and required theorem types, admits
all 183 closed ordinary definitions, and infers every submitted node, including
unused syntax. Raw binder nodes can have open contexts; exported proofs and
definition bodies must be closed. Both the named final theorem and the supplied
root must have the exact expected type, allowing equivalent beta-wrapped proofs.

The new public diagnostic result distinguishes:

| Outcome | Meaning for qualification |
| --- | --- |
| `ACCEPTED` | Exact profile accepted within positive budgets |
| `INVALID_PROOF` | Submission violates the declared syntax, metadata, data or theorem profile |
| `RESOURCE_PREFLIGHT` | Requested storage budget cannot hold submitted metadata/source |
| `RESOURCE_EXHAUSTED` | Checking exhausted nodes or fuel; no invalidity conclusion |

Preflight checks run before allocating checker state. The canonical
in-place constructor and ordinary process limits are used. Its field/default
equivalence is independently reviewed; heap ownership and no-inline boundaries
change construction and temporary placement, while proof rules remain unchanged. A crash or
timeout is also a failed qualification attempt, never a successful rejection.
Native semantic controls must return `INVALID_PROOF`; quota controls must return
`RESOURCE_PREFLIGHT`. A positive that exhausts resources blocks the negative gate.

## Native UInt32 evidence and remaining trust

The finite host oracle compares the actual maintained `merge_argmax` against
independent Python arbitrary-precision arithmetic on
25,145 cases, checking 36,937 result records. It includes the six UInt32 edge
words `0`, `1`, `0x7fffffff`, `0x80000000`, `0xfffffffe`, `0xffffffff`, adjacent-rank
carry cases, seeded pairs/triples, both triple bracketings, empty and idempotence.
Flags are restricted to `{0,1}`.

Only Python computes the score and packed mathematical encoding. The maximum
encoded case needs 65 bits. The Mojo probe calls the actual scalar merge and
prints word records; it performs no native Int/UInt64 score calculation. The Mojo probe uses explicit LF splitting for its strict generated TSV input.

That finite agreement is useful execution evidence. It is not a machine-checked
proof for every UInt32 tuple and does not connect an arbitrary Mojo/SSA body to
the Nat definition. The native interpreter's broader multibit invalid words
remain outside this Boolean-record theorem. The production certificate keeps
its existing conditional profile until its separate connection is established.

Remaining trust includes the generic kernel's typing, induction, substitution
and conversion code; its compile-time evaluator; the independently written
admission specification; native word comparisons/OR and source correspondence;
input extents, immutability, nonaliasing and layout/lifetime assumptions; and
compiler lowering, warp participation, barriers, visibility and device progress.
Neither the kernel implementation nor that physical refinement has a formal
metatheory proof here. No full-model floating-point determinism or performance
result follows from this record theorem or from the finite host oracle.

## Sources and reproducible checks

The producer is [finishing_record_terms.mojo](../src/llm_mojo/kernels/finishing_record_terms.mojo).
Its 183 ordinary definitions culminate in definition182; export metadata is
untrusted. [nat_record_execution_check.mojo](../src/llm_mojo/kernels/nat_record_execution_check.mojo)
builds the intended data and theorem specification independently. The earlier
148-definition encoding profile is also checked, including the relocated Pred
conversion. No general arithmetic export namespace or native arithmetic,
affine or literal-order reflection is enabled.

The independent native profile accepts the named theorem and a distinct beta
wrapper at C16384/S65536, using 50,439 and 50,443 nodes respectively. The full
record controls require 44 `INVALID_PROOF` results, three allocation preflights,
three complete changed-final-type typing positives and four explicitly narrower
wrong-scope declaration-prefix typing positives. The earlier encoding controls
retain 30 semantic checks, three resource checks and a welltyped assumed-result
control. Prefix checks are not claims that all later changed declarations type.

[The maintained test](../tests/test_record_execution.mojo) adds one full183
compile-time positive. Its focused Python gate first builds and executes that
positive, then requires the exact constraint diagnostic from seven semantic
fixtures and one resource fixture. Resource exhaustion cannot earn a semantic
rejection diagnostic. Run the focused gates with the pinned toolchain:

```sh
uv run python -m unittest discover -s tests -p 'test_proof_term_compile.py' -k finishing_record_execution
uv run python -m unittest discover -s tests -p 'test_token_selection_compile.py' -k native_record_merge
```

The constructor refactor also uses
[proof_state_storage_cases.mojo](../tests/proof_state_storage_cases.mojo) through
`test_proof_terms.mojo`: source/copy isolation, binder identity and independent
cache authority must hold. The constructor regression gate includes existing
typed-proof, inductive-eliminator and compact-numeral positives, followed by
selected exact rejection fixtures. The refactor changes one canonical state
initializer; all typing, substitution, conversion and induction code remains
unchanged. Its normal-return initialization values and cache seeding match the
previous constructor. Allocation failure and compiler/allocator implementation
remain trusted. No process stack limit is raised.

Exact sources, compiler/SDK files, ordinary stack limits, retained binary bytes,
logs and classifications belong to each local qualification receipt. Earlier
sealed sources and failed/resource-limited attempts remain historical evidence;
a source snapshot or failed build is not a proof acceptance result. Backend
lowering/execution, physical memory visibility, scheduler progress and full-model
floating-point determinism remain separate gates.
