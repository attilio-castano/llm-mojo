"""A restricted dependent proof-term calculus, implemented and checked in Mojo.

This is an embedded language: Mojo's own types do not acquire dependent typing.
The trusted checker admits Prop, a non-self-typed Type sort, dependent
functions and equality substitution into Prop. Nat is a checked zero-index
Type-family declaration; one Nat-indexed Prop family may also be declared.
Both use the same closure, sorting, result-family and conservative direct
strict-positivity checks. Their eliminators into Prop and constructor iota
reductions are generated from admitted telescopes, without a specialized Nat
induction schema. Canonical zero/successor signatures are independently pinned
before checking a constructor-only zero/successor/double library. Compact
literal typing elaborates its binary Nat denotation. Each constructor view is
an actual checked derivation of alpha, beta, delta, structural NatRec iota,
application congruence, symmetry and transitivity; no whole-numeral rule is
admitted. The numeral bootstrap has no literal/successor shorthand or access
to its own certificate. Native integer parsing and bit decoding remain trusted.
Nondependent data recursion is a separately checked structural rule over that
canonical Nat: the zero value fixes a carrier of sort Type, the successor
branch has type Nat -> carrier -> carrier, and iota recurses on the predecessor.
Its rule and computation are explicit kernel code, whereas the Prop eliminator
signature is generated from the admitted constructor telescope.
Arithmetic facts and the two Base driver equations are exported from a
separately bootstrap-checked, immutable library of closed
Nat-recursive definitions and ordinary proof terms. The bootstrap cannot use
those exports or the native arithmetic reflection bridge. Division has a
positive literal divisor in consumer syntax, with symbolic positive divisors
in the checked library. Affine equality and normalized literal order reflection
remain separately visible trusted checks. Native literal arithmetic only
proposes payloads; raw Nat conversion must check a reflexivity proof before
compaction.
Consumers import the actual certified proof type and instantiate it through
ordinary Pi application. Compact operators elaborate by ordinary application
of eight checked closed raw abbreviations. Their bootstrap checks actual type,
closure and meaning against the certified Nat definitions with no compaction
or reflection available. Heuristic resugaring is admitted only when generic
alpha/beta/abbreviation equality checks the original and candidate meanings;
data recursion stays opaque to that check. Its reduction, binding and import
implementation remains unproved kernel code. Base stays opaque to affine
normalization; raw unfolding remains resource bounded.
Generic positivity, eliminator
generation and reduction implementation remain part of the unproved kernel.
Variables have unique syntax-node identities and carry their declared type.
Exact free-variable contexts include dependencies of variable types. A closed
root is checked against a separately supplied expected type. Multiple roots may
share one checked environment without granting any additional inference rule.
Substitution never
captures a variable; exact contexts skip independent typed subtrees, and the
implementation rejects a remaining capture instead of silently renaming a
binder. Definitional equality uses alpha/beta/delta and constructor/NatRec
computation, with checked numeral and operator abbreviations. It does not
reflect arbitrary propositional theorems into conversion or evaluate without
bounds.
"""
from std.collections import Array
from llm_mojo.kernels.finishing_proof_kernel import FinishingProgram, FINISH_LT
from llm_mojo.kernels.nat_arithmetic_terms import (
    NatArithmeticLibrary,
    NAT_REP_ADD, NAT_REP_MUL, NAT_REP_LE, NAT_REP_LT, NAT_REP_SCALE,
    NAT_REP_BASE, NAT_REP_DIV, NAT_REP_MOD, NAT_REPRESENTATIONS,
)
from llm_mojo.kernels.nat_arithmetic import make_nat_arithmetic_library
from llm_mojo.kernels.nat_literal_arithmetic import (
    LiteralArithmeticProof, make_literal_arithmetic_proof,
)
from llm_mojo.kernels.nat_headroom import HeadroomProof
from llm_mojo.kernels.nat_numerals import (
    NumeralLibrary, NumeralView, make_numeral_library, make_numeral_view,
    NUMERAL_LIBRARY_CAPACITY, NUMERAL_VIEW_CAPACITY, NUMERAL_CONVERSION_CAPACITY,
    NC_REFL, NC_BETA, NC_DELTA, NC_NAT_IOTA, NC_CONG_LEFT, NC_CONG_RIGHT,
    NC_SYMM, NC_TRANS,
)

# Immutable proof data and a bootstrap judgment. The bootstrap state does not
# receive this flag and rejects arithmetic export syntax, so it cannot certify
# itself by referring to its own result.
# Generated only inside this checker; never admitted from a caller bundle.
comptime _PT_IMMUTABLE_REF = 31
comptime _NAT_NUMERAL_LIBRARY = make_numeral_library()
comptime _NUMERAL_NODE_COUNT = _NAT_NUMERAL_LIBRARY.bundle.count
comptime _NAT_NUMERAL_CERTIFIED = check_native_numeral_library(_NAT_NUMERAL_LIBRARY)
comptime _NAT_ARITHMETIC_LIBRARY = make_nat_arithmetic_library()
comptime _IMMUTABLE_NODE_COUNT = _NAT_ARITHMETIC_LIBRARY.bundle.count
comptime _IMMUTABLE_DEFINITION_COUNT = _NAT_ARITHMETIC_LIBRARY.bundle.definition_count
comptime _NAT_ARITHMETIC_CERTIFIED = check_native_arithmetic_library(_NAT_ARITHMETIC_LIBRARY)

from llm_mojo.kernels.proof_term_syntax import (
    PT_TYPE,
    PT_PROP,
    PT_NAT,
    PT_VAR,
    PT_PI,
    PT_LAM,
    PT_APP,
    PT_LITERAL,
    PT_ADD,
    PT_SCALE,
    PT_BASE,
    PT_EQ,
    PT_LE,
    PT_LT,
    PT_DONE,
    PT_CONST,
    PT_NAT_IND,
    PT_SUBST,
    PT_REFL,
    PT_AFFINE_EQ,
    PT_FAMILY,
    PT_CONSTRUCTOR,
    PT_ELIMINATOR,
    PT_DIV,
    PT_MOD,
    PT_LITERAL_LE,
    PT_DEFINED_REF,
    PT_SUCC,
    PT_NAT_REC,
    PT_MUL,
    PT_ARITHMETIC,
    PROP_FAMILY_ID,
    NAT_FAMILY_ID,
    DEC_BASE_ZERO,
    DEC_BASE_STEP,
    DEC_NAT_NONNEG,
    DEC_LE_ADD,
    DEC_LE_TRANS,
    DEC_DONE_STOP,
    DEC_DONE_STEP,
    DEC_ORDER_CASES,
    DEC_LT_ADD,
    DEC_LE_LT_TRANS,
    DEC_LT_LE_TRANS,
    DEC_LE_SCALE,
    DEC_LT_SCALE_CANCEL,
    DEC_LT_TO_LE_SUCC,
    DEC_DIV_DECOMP,
    DEC_MOD_LT,
    DEC_DIV_QUOTIENT_UNIQUE,
    DEC_DIV_REMAINDER_UNIQUE,
    PROOF_TERM_DECLARATIONS,
    ProofTerm,
    ProofTermJudgment,
    ProofTermBundle,
    empty_proof_terms,
    pt_add,
    pt_declare_family,
    pt_declare_constructor,
    pt_declare_definition,
)


struct _TypedState[capacity: Int](Copyable, Movable):
    var syntax: ProofTermBundle[Self.capacity]
    var types: Array[Int, Self.capacity]
    # The exact context matrix is quadratic in the generic workspace capacity.
    # Heap storage prevents native checking from depending on process stack size.
    var contexts: List[UInt64]
    # Exact last nonzero word + 1 per row: every word at or above high is zero.
    # This skips empty storage without dropping any free-variable dependency.
    var context_high: Array[Int, Self.capacity]
    # Append-only syntax index, keyed by all five fields. First duplicate wins;
    # variables are excluded because their node index is a fresh binder identity.
    var interned: Dict[Tuple[Int, Int, Int, Int, Int], Int]
    var program: FinishingProgram
    var fuel: Int
    var family_type: Int
    var environment_checked: Bool
    var eliminator_type: Int
    var nat_eliminator_type: Int
    var nat_environment_checked: Bool
    var nat_canonical: Bool
    var admitted_definitions: Int
    var arithmetic_verified: Bool
    var fast_arithmetic: Bool
    var numerals_verified: Bool
    var numeral_nodes: Array[Int, _NUMERAL_NODE_COUNT]
    var numeral_denotations: Dict[Int, Int]
    var numeral_values: Dict[Int, Int]
    var numeral_views: Dict[Int, Int]
    # Checked binary denotations stay opaque during eager normalization. A
    # predecessor need not have a compact native-word payload.
    var numeral_opaque: Dict[Int, Bool]
    # Optional literal conversion checks are isolated from this state's budget.
    # Both successes and failures are memoized; a failure grants no judgment.
    var literal_conversions: Dict[Tuple[Int, Int, Int, Int], Bool]
    # One immutable namespace for actual certified types and bodies. The map
    # freshens each source variable once, retaining sharing and binder identity.
    var immutable_nodes: Array[Int, _IMMUTABLE_NODE_COUNT]
    var immutable_types: Array[Int, _IMMUTABLE_DEFINITION_COUNT]
    var immutable_bodies: Array[Int, _IMMUTABLE_DEFINITION_COUNT]

    def __init__[source_capacity: Int](out self, program: FinishingProgram,
        proof: ProofTermBundle[source_capacity]):
        # One canonical initializer. Direct field initialization lets callers
        # construct the workspace at its final destination without a state copy.
        self.syntax = empty_proof_terms[Self.capacity]()
        self.types = Array[Int, Self.capacity](fill=-3)
        self.contexts = List[UInt64](capacity=Self.capacity * ((Self.capacity + 63) // 64))
        self.context_high = Array[Int, Self.capacity](fill=0)
        self.interned = Dict[Tuple[Int, Int, Int, Int, Int], Int]()
        self.program = program
        self.fuel = Self.capacity * Self.capacity * 16
        self.family_type = -1
        self.environment_checked = False
        self.eliminator_type = -1
        self.nat_eliminator_type = -1
        self.nat_environment_checked = False
        self.nat_canonical = False
        self.admitted_definitions = 0
        self.arithmetic_verified = False
        self.fast_arithmetic = False
        self.numerals_verified = False
        self.numeral_nodes = Array[Int, _NUMERAL_NODE_COUNT](fill=-1)
        self.numeral_denotations = Dict[Int, Int]()
        self.numeral_values = Dict[Int, Int]()
        self.numeral_views = Dict[Int, Int]()
        self.numeral_opaque = Dict[Int, Bool]()
        self.literal_conversions = Dict[Tuple[Int, Int, Int, Int], Bool]()
        self.immutable_nodes = Array[Int, _IMMUTABLE_NODE_COUNT](fill=-1)
        self.immutable_types = Array[Int, _IMMUTABLE_DEFINITION_COUNT](fill=-3)
        self.immutable_bodies = Array[Int, _IMMUTABLE_DEFINITION_COUNT](fill=-3)
        for j in range(proof.count):
            self.syntax.nodes[j] = proof.nodes[j]
        self.syntax.count = proof.count
        self.syntax.root = proof.root
        self.syntax.family_arity = proof.family_arity
        self.syntax.constructor_count = proof.constructor_count
        for c in range(proof.constructor_count):
            self.syntax.constructor_types[c] = proof.constructor_types[c]
        self.syntax.definition_count = proof.definition_count
        for d in range(proof.definition_count):
            self.syntax.definition_types[d] = proof.definition_types[d]
            self.syntax.definition_bodies[d] = proof.definition_bodies[d]
        self.syntax.nat_constructor_count = proof.nat_constructor_count
        for c in range(proof.nat_constructor_count):
            self.syntax.nat_constructor_types[c] = proof.nat_constructor_types[c]
        self.contexts.resize(proof.count * ((Self.capacity + 63) // 64), 0)
        _seed_interned(self, 0, proof.count)


def _valid_syntax[capacity: Int](p: ProofTermBundle[capacity]) -> Bool:
    if (p.count < 0 or p.count > capacity or p.family_arity < -1
        or p.family_arity > capacity or p.constructor_count < 0
        or p.constructor_count > capacity
        or (p.family_arity == -1 and p.constructor_count != 0)
        or p.nat_constructor_count < 0 or p.nat_constructor_count > capacity
        or p.definition_count < 0 or p.definition_count > capacity):
        return False
    for c in range(p.constructor_count):
        if p.constructor_types[c] < 0 or p.constructor_types[c] >= p.count:
            return False
    for c in range(p.nat_constructor_count):
        if p.nat_constructor_types[c] < 0 or p.nat_constructor_types[c] >= p.count:
            return False
    for d in range(p.definition_count):
        if (p.definition_types[d] < 0 or p.definition_types[d] >= p.count
            or p.definition_bodies[d] < 0 or p.definition_bodies[d] >= p.count):
            return False
    for j in range(p.count):
        var n = p.nodes[j]
        if (n.kind < PT_TYPE or n.kind > PT_ARITHMETIC
            or n.kind == PT_DONE or n.kind == PT_NAT_IND or n.kind == PT_CONST):
            return False
        var arity = 0
        if (n.kind == PT_VAR or n.kind == PT_SCALE or n.kind == PT_BASE or n.kind == PT_REFL
            or n.kind == PT_DIV or n.kind == PT_MOD or n.kind == PT_SUCC):
            arity = 1
        elif (n.kind == PT_PI or n.kind == PT_LAM or n.kind == PT_APP or n.kind == PT_ADD
            or (n.kind >= PT_EQ and n.kind <= PT_LT) or n.kind == PT_AFFINE_EQ
            or n.kind == PT_LITERAL_LE or n.kind == PT_MUL):
            arity = 2
        elif n.kind == PT_SUBST or n.kind == PT_NAT_REC:
            arity = 3
        if arity >= 1 and (n.left < 0 or n.left >= j):
            return False
        if arity >= 2 and (n.right < 0 or n.right >= j):
            return False
        if arity >= 3 and (n.extra < 0 or n.extra >= j):
            return False
        if arity == 0 and n.left != -1:
            return False
        if arity < 2 and n.right != -1:
            return False
        if (arity < 3 and n.extra != -1
            and not (n.kind == PT_CONSTRUCTOR and n.extra == 1)
            and n.kind != PT_ARITHMETIC):
            return False
        if n.kind == PT_LITERAL or n.kind == PT_SCALE:
            if n.value < 0:
                return False
        elif n.kind == PT_DIV or n.kind == PT_MOD:
            if n.value <= 0:
                return False
        elif n.kind == PT_ARITHMETIC:
            if (n.value < DEC_BASE_ZERO or n.value >= PROOF_TERM_DECLARATIONS
                or n.value == DEC_DONE_STOP or n.value == DEC_DONE_STEP):
                return False
            if n.value == DEC_LE_SCALE:
                if n.extra < 0:
                    return False
            elif (n.value == DEC_LT_SCALE_CANCEL or n.value == DEC_DIV_DECOMP
                or n.value == DEC_MOD_LT or n.value == DEC_DIV_QUOTIENT_UNIQUE
                or n.value == DEC_DIV_REMAINDER_UNIQUE):
                if n.extra <= 0:
                    return False
            elif n.extra != -1:
                return False
        elif n.kind == PT_DEFINED_REF:
            if n.value < 0 or n.value >= p.definition_count:
                return False
        elif n.kind == PT_FAMILY:
            if n.value != 1 and not (n.value == 0 and p.family_arity >= 0):
                return False
        elif n.kind == PT_CONSTRUCTOR:
            var count = p.nat_constructor_count if n.extra == 1 else p.constructor_count
            if n.value < 0 or n.value >= count:
                return False
        elif n.kind == PT_ELIMINATOR:
            if n.value != 1 and not (n.value == 0 and p.family_arity >= 0):
                return False
        elif n.value != -1:
            return False
        if n.kind == PT_PI or n.kind == PT_LAM:
            if p.nodes[n.left].kind != PT_VAR:
                return False
    return True


def _consume[capacity: Int](mut s: _TypedState[capacity]) -> Bool:
    if s.fuel <= 0:
        return False
    s.fuel -= 1
    return True


def _has[capacity: Int](s: _TypedState[capacity], node: Int, variable: Int) -> Bool:
    comptime W = (capacity + 63) // 64
    if node < 0 or node >= s.syntax.count or variable < 0 or variable >= s.syntax.count:
        return False
    if variable // 64 >= s.context_high[node]:
        return False
    return (s.contexts[node * W + variable // 64] & (UInt64(1) << UInt64(variable % 64))) != 0


def _union[capacity: Int](mut s: _TypedState[capacity], dst: Int, source: Int):
    comptime W = (capacity + 63) // 64
    for w in range(s.context_high[source]):
        s.contexts[dst * W + w] |= s.contexts[source * W + w]
    if s.context_high[source] > s.context_high[dst]:
        s.context_high[dst] = s.context_high[source]


def _remove[capacity: Int](mut s: _TypedState[capacity], node: Int, variable: Int):
    comptime W = (capacity + 63) // 64
    s.contexts[node * W + variable // 64] &= ~(UInt64(1) << UInt64(variable % 64))
    var high = s.context_high[node]
    while high > 0 and s.contexts[node * W + high - 1] == 0:
        high -= 1
    s.context_high[node] = high


def _can_abstract[capacity: Int](s: _TypedState[capacity], body: Int, binder: Int) -> Bool:
    # A surviving free variable's declared type cannot depend on this binder.
    # Read exactly the same context bits, visiting only populated words/slots.
    comptime W = (capacity + 63) // 64
    for w in range(s.context_high[body]):
        var bits = s.contexts[body * W + w]
        if bits == 0:
            continue
        for bit in range(64):
            if (bits & (UInt64(1) << UInt64(bit))) == 0:
                continue
            var j = w * 64 + bit
            if j < s.syntax.count and j != binder and s.syntax.nodes[j].kind == PT_VAR:
                if _has(s, s.syntax.nodes[j].left, binder):
                    return False
    return True


def _put[capacity: Int](mut s: _TypedState[capacity], kind: Int,
    left: Int = -1, right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    if s.syntax.count < 0 or s.syntax.count > capacity:
        return -1
    var key = (kind, left, right, extra, value)
    # Match pt_add's first-node rule even at full capacity. Only a fresh node
    # needs storage; an existing nonvariable is still available at the bound.
    if kind != PT_VAR and key in s.interned:
        return s.interned.get(key, -1)
    if s.syntax.count == capacity:
        return -1
    var node = s.syntax.count
    # Initialize a full fixed-width row before exposing each live node. The
    # maximum capacity, exact context bits and proof budget stay unchanged.
    comptime W = (capacity + 63) // 64
    s.contexts.resize((node + 1) * W, 0)
    s.syntax.nodes[node] = ProofTerm(kind, left, right, extra, value)
    s.syntax.count += 1
    if kind != PT_VAR:
        s.interned[key] = node
    return node


def _seed_interned[C: Int](mut s: _TypedState[C], start: Int, end: Int):
    for j in range(start, end):
        var n = s.syntax.nodes[j]
        if n.kind == PT_VAR:
            continue
        var key = (n.kind, n.left, n.right, n.extra, n.value)
        if key not in s.interned:
            s.interned[key] = j


def _substitute[capacity: Int](mut s: _TypedState[capacity], node: Int,
    variable: Int, replacement: Int) -> Int:
    if not _consume(s) or node < 0 or node >= s.syntax.count:
        return -1
    # Fresh binders created when a dependent domain changes must carry their
    # complete typed context before a capture check, just like input arguments.
    if _infer(s, replacement) < 0:
        return -1
    if node == variable:
        return replacement
    # Exact contexts permit skipping a fully checked subtree that contains no
    # free occurrence of the substituted variable. In particular, a recursive
    # recursor's closed step must not acquire a capture merely because its local
    # binder identity also occurs in an unrelated replacement argument.
    if s.types[node] >= 0 and not _has(s, node, variable):
        return node
    var n = s.syntax.nodes[node]
    if n.kind == PT_VAR:
        # Variable identities are not reconstructed when their declared types
        # change: the surrounding binder supplies its original typed context.
        return node
    if (n.kind <= PT_NAT or n.kind == PT_LITERAL
        or n.kind == PT_FAMILY or n.kind == PT_CONSTRUCTOR or n.kind == PT_ELIMINATOR
        or n.kind == PT_DEFINED_REF or n.kind == PT_ARITHMETIC or n.kind == _PT_IMMUTABLE_REF):
        return node
    if n.kind == PT_PI or n.kind == PT_LAM:
        if n.left == variable:
            return node
        if _has(s, replacement, n.left):
            # Substitution would capture this free variable under the binder.
            # Reject conservatively, even when the substituted variable is absent.
            return -1
        var domain = s.syntax.nodes[n.left].left
        var changed_domain = _substitute(s, domain, variable, replacement)
        if changed_domain < 0:
            return -1
        var binder = n.left
        var body = n.right
        if changed_domain != domain:
            binder = _put(s, PT_VAR, changed_domain)
            if binder < 0:
                return -1
            body = _substitute(s, body, n.left, binder)
            if body < 0:
                return -1
        body = _substitute(s, body, variable, replacement)
        if body < 0:
            return -1
        return _put(s, n.kind, binder, body)
    var a = _substitute(s, n.left, variable, replacement)
    if a < 0:
        return -1
    var b = n.right
    if b >= 0:
        b = _substitute(s, b, variable, replacement)
        if b < 0:
            return -1
    var c = n.extra
    if c >= 0:
        c = _substitute(s, c, variable, replacement)
        if c < 0:
            return -1
    var v = n.value
    return _put(s, n.kind, a, b, c, v)


def _nat_rec_carrier[capacity: Int](mut s: _TypedState[capacity], node: Int) -> Int:
    """Check a nondependent data recursor over the admitted native Nat.

    Its result carrier is inferred from the zero branch and must itself have
    Type as its sort. No universe is passed as an object-language argument.
    """
    if not s.nat_canonical or node < 0 or node >= s.syntax.count:
        return -1
    var n = s.syntax.nodes[node]
    if n.kind != PT_NAT_REC:
        return -1
    var nat = _put(s, PT_NAT)
    var universe = _put(s, PT_TYPE)
    var carrier = _infer(s, n.right)
    if (carrier < 0 or _infer(s, carrier) != universe
        or not _convert(s, _infer(s, n.left), nat)):
        return -1
    var predecessor = _put(s, PT_VAR, nat)
    var hypothesis = _put(s, PT_VAR, carrier)
    var expected_step = _put(s, PT_PI, predecessor,
        _put(s, PT_PI, hypothesis, carrier))
    if not _convert(s, _infer(s, n.extra), expected_step):
        return -1
    return carrier


def _reduce_nat_rec[capacity: Int](mut s: _TypedState[capacity], node: Int,
    compact: Bool = True) -> Int:
    """Apply the zero/successor equations after checking every operand."""
    # Ordinary inference caches the complete operand check and its context.
    # Reconstructing fresh step-type binders on every conversion would consume
    # storage even when the same recursor has already been fully checked.
    var carrier = _infer(s, node)
    if carrier < 0:
        return -1
    var n = s.syntax.nodes[node]
    var major = _whnf(s, n.left, compact)
    if major < 0:
        return -1
    var m = s.syntax.nodes[major]
    if m.kind == PT_LITERAL or m.kind == PT_SUCC:
        major = _checked_numeral_view(s, major)
        if major < 0:
            return -1
    var layer = _raw_nat_layer(s, major)
    if layer[0] == 0:
        return _whnf(s, n.right, compact)
    var predecessor = -1
    if layer[0] == 1:
        predecessor = layer[1]
    if predecessor < 0:
        return _put(s, PT_NAT_REC, major, n.right, n.extra)
    var hypothesis = _put(s, PT_NAT_REC, predecessor, n.right, n.extra)
    var applied = _put(s, PT_APP, _put(s, PT_APP, n.extra, predecessor), hypothesis)
    return _whnf(s, applied, compact)


def _check_definitions[capacity: Int](mut s: _TypedState[capacity]) -> Bool:
    """Admit closed definitions in order, with no self or forward reference."""
    s.admitted_definitions = 0
    var prop = _put(s, PT_PROP)
    var universe = _put(s, PT_TYPE)
    comptime W = (capacity + 63) // 64
    for d in range(s.syntax.definition_count):
        var declared = s.syntax.definition_types[d]
        var body = s.syntax.definition_bodies[d]
        var sort = _infer(s, declared)
        if ((sort != prop and sort != universe)
            or not _convert(s, _infer(s, body), declared)):
            return False
        if s.context_high[declared] != 0 or s.context_high[body] != 0:
            return False
        s.admitted_definitions += 1
    return True


def _raw_app2[C: Int](mut s: _TypedState[C], f: Int, a: Int, b: Int) -> Int:
    return _put(s, PT_APP, _put(s, PT_APP, f, a), b)


def _raw_numeral_alpha_under[C: Int](mut s: _TypedState[C], a: Int, b: Int,
    mut left_names: List[Int], mut right_names: List[Int], depth: Int) -> Bool:
    """Structural alpha equality only: no delta, beta, iota or numeral codec."""
    if (not _consume(s) or depth >= C or a < 0 or b < 0
        or a >= s.syntax.count or b >= s.syntax.count):
        return False
    # Reflexivity is independent of inactive maps. Under active renamings,
    # exact checked closure also excludes any reinterpreted free variable.
    # Keep inspecting open terms: identical IDs can name different binders.
    if a == b and (depth == 0 or (s.types[a] >= 0 and s.context_high[a] == 0)):
        return True
    var x = s.syntax.nodes[a]
    var y = s.syntax.nodes[b]
    if x.kind != y.kind:
        return False
    if x.kind == PT_VAR:
        var ax = -1
        var by = -1
        for j in range(depth):
            if left_names[j] == a:
                ax = j
            if right_names[j] == b:
                by = j
        return ax == by if ax >= 0 or by >= 0 else a == b
    if x.kind == PT_PI or x.kind == PT_LAM:
        if not _raw_numeral_alpha_under(s, s.syntax.nodes[x.left].left,
            s.syntax.nodes[y.left].left, left_names, right_names, depth):
            return False
        if len(left_names) == depth and len(right_names) == depth:
            left_names.append(x.left)
            right_names.append(y.left)
        elif len(left_names) > depth and len(right_names) > depth:
            left_names[depth] = x.left
            right_names[depth] = y.left
        else:
            return False
        return _raw_numeral_alpha_under(s, x.right, y.right,
            left_names, right_names, depth + 1)
    if x.value != y.value:
        return False
    var third_child = x.kind == PT_NAT_REC or x.kind == PT_SUBST
    if not third_child and x.extra != y.extra:
        return False
    if ((x.left >= 0) != (y.left >= 0)
        or (x.right >= 0) != (y.right >= 0)
        or (third_child and ((x.extra >= 0) != (y.extra >= 0)))):
        return False
    if x.left >= 0 and not _raw_numeral_alpha_under(s, x.left, y.left,
        left_names, right_names, depth):
        return False
    if x.right >= 0 and not _raw_numeral_alpha_under(s, x.right, y.right,
        left_names, right_names, depth):
        return False
    return (not third_child or _raw_numeral_alpha_under(s, x.extra, y.extra,
        left_names, right_names, depth))


def _raw_numeral_alpha[C: Int](mut s: _TypedState[C], a: Int, b: Int) -> Bool:
    var left_names = List[Int]()
    var right_names = List[Int]()
    return _raw_numeral_alpha_under(s, a, b, left_names, right_names, 0)


def _raw_nat_layer[C: Int](s: _TypedState[C], node: Int) -> Tuple[Int, Int]:
    if not s.nat_canonical or node < 0 or node >= s.syntax.count:
        return (-1, -1)
    var n = s.syntax.nodes[node]
    if n.kind == PT_CONSTRUCTOR and n.extra == NAT_FAMILY_ID and n.value == 0:
        return (0, -1)
    if n.kind == PT_APP and n.left >= 0 and n.left < s.syntax.count:
        var head = s.syntax.nodes[n.left]
        if head.kind == PT_CONSTRUCTOR and head.extra == NAT_FAMILY_ID and head.value == 1:
            return (1, n.right)
    return (-1, -1)


def _raw_numeral_syntax[C: Int](proof: ProofTermBundle[C]) -> Bool:
    if (not _valid_syntax(proof) or proof.definition_count != 3
        or proof.family_arity != -1 or proof.constructor_count != 0):
        return False
    for j in range(proof.count):
        var n = proof.nodes[j]
        if (n.kind != PT_TYPE and n.kind != PT_NAT and n.kind != PT_VAR
            and n.kind != PT_PI and n.kind != PT_LAM and n.kind != PT_APP
            and n.kind != PT_DEFINED_REF and n.kind != PT_NAT_REC
            and n.kind != PT_CONSTRUCTOR and n.kind != PT_FAMILY):
            return False
        if ((n.kind == PT_CONSTRUCTOR and n.extra != NAT_FAMILY_ID)
            or (n.kind == PT_FAMILY and n.value != NAT_FAMILY_ID)):
            return False
    return True


def _canonical_numeral_body[C: Int](mut s: _TypedState[C], definition: Int) -> Int:
    var zero = _put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=0)
    if definition == 0:
        return zero
    var nat = _put(s, PT_NAT)
    var successor = _put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=1)
    var n = _put(s, PT_VAR, nat)
    if definition == 1:
        return _put(s, PT_LAM, n, _put(s, PT_APP, successor, n))
    if definition == 2:
        var p = _put(s, PT_VAR, nat)
        var acc = _put(s, PT_VAR, nat)
        var step = _put(s, PT_LAM, p, _put(s, PT_LAM, acc,
            _put(s, PT_APP, successor, _put(s, PT_APP, successor, acc))))
        return _put(s, PT_LAM, n, _put(s, PT_NAT_REC, n, zero, step))
    return -1


def _check_raw_numeral_library[C: Int](mut s: _TypedState[C]) -> Bool:
    if s.numerals_verified or s.arithmetic_verified or not _check_inductive_environment(s):
        return False
    for definition in range(3):
        var intended = _canonical_numeral_body(s, definition)
        if (intended < 0 or _infer(s, intended) < 0
            or not _raw_numeral_alpha(s, s.syntax.definition_bodies[definition], intended)):
            return False
    return True


def check_native_numeral_library[C: Int](library: NumeralLibrary[C]) -> Bool:
    """Independent constructor-only bootstrap; its own certificate is unavailable."""
    if C <= 0 or not _raw_numeral_syntax(library.bundle):
        return False
    var program = FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        FINISH_LT, FINISH_LT, 2147483647, Int.MAX)
    comptime S = C * 3 + 128
    var state = _make_state[C, S](program, library.bundle)
    if not _check_raw_numeral_library(state):
        return False
    for j in range(library.bundle.count):
        var inferred = _infer(state, j)
        if inferred == -1 or inferred == -4:
            return False
    return True


def _raw_literal_denotation[C: Int](mut s: _TypedState[C], value: Int, double: Int) -> Int:
    """Native bit parsing proposes an ordinary raw Nat term, never a judgment."""
    if value < 0:
        return -1
    var bits = List[Int]()
    # The pinned compile-time evaluator truncates large signed Int division
    # loops. Parse the full nonnegative host word with explicit unsigned ops.
    var remaining = UInt64(value)
    while remaining > UInt64(0):
        bits.append(Int(remaining & UInt64(1)))
        remaining = remaining >> UInt64(1)
    var result = _put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=0)
    var successor = _put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=1)
    for j in range(len(bits)):
        result = _put(s, PT_APP, double, result)
        if bits[len(bits) - j - 1] == 1:
            result = _put(s, PT_APP, successor, result)
    return result


def _raw_literal_predecessor_denotation[C: Int](mut s: _TypedState[C], value: Int,
    double: Int) -> Int:
    """Decode the predecessor by binary borrow, without a whole-word minus one.

    Native bit parsing remains trusted. Only the small digits zero and one are
    changed here; the ordinary raw Nat definitions give those digits meaning.
    """
    if value <= 0:
        return -1
    var bits = List[Int]()
    var remaining = UInt64(value)
    while remaining > UInt64(0):
        bits.append(Int(remaining & UInt64(1)))
        remaining = remaining >> UInt64(1)
    var borrow = True
    for j in range(len(bits)):
        if borrow:
            if bits[j] == 0:
                bits[j] = 1
            else:
                bits[j] = 0
                borrow = False
    if borrow:
        return -1
    var length = len(bits)
    while length > 0 and bits[length - 1] == 0:
        length -= 1
    var result = _put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=0)
    var successor = _put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=1)
    for j in range(length):
        result = _put(s, PT_APP, double, result)
        if bits[length - j - 1] == 1:
            result = _put(s, PT_APP, successor, result)
    return result


def _check_numeral_conversion[C: Int, V: Int, D: Int](mut s: _TypedState[C],
    proof: NumeralView[V, D], position: Int) -> Bool:
    var step = proof.conversions[position]
    if (step.kind < NC_REFL or step.kind > NC_TRANS or step.left < 0
        or step.right < 0 or step.left >= proof.bundle.count or step.right >= proof.bundle.count):
        return False
    var a_type = _infer(s, step.left)
    var b_type = _infer(s, step.right)
    if a_type < 0 or b_type < 0 or not _raw_numeral_alpha(s, a_type, b_type):
        return False
    var first = step.first
    var second = step.second
    if step.kind == NC_SYMM or step.kind == NC_CONG_LEFT or step.kind == NC_CONG_RIGHT:
        if first < 0 or first >= position or second != -1:
            return False
    elif step.kind == NC_TRANS:
        if first < 0 or first >= position or second < 0 or second >= position:
            return False
    elif first != -1 or second != -1:
        return False
    var lhs = s.syntax.nodes[step.left]
    if step.kind == NC_REFL:
        return _raw_numeral_alpha(s, step.left, step.right)
    if step.kind == NC_DELTA:
        return (lhs.kind == PT_DEFINED_REF and lhs.value >= 0 and lhs.value < s.admitted_definitions
            and _raw_numeral_alpha(s, s.syntax.definition_bodies[lhs.value], step.right))
    if step.kind == NC_BETA:
        if lhs.kind != PT_APP or s.syntax.nodes[lhs.left].kind != PT_LAM:
            return False
        var function = s.syntax.nodes[lhs.left]
        return _raw_numeral_alpha(s, _substitute(s, function.right, function.left, lhs.right), step.right)
    if step.kind == NC_NAT_IOTA:
        if lhs.kind != PT_NAT_REC or _nat_rec_carrier(s, step.left) < 0:
            return False
        var layer = _raw_nat_layer(s, lhs.left)
        if layer[0] == 0:
            return _raw_numeral_alpha(s, lhs.right, step.right)
        if layer[0] == 1:
            var hypothesis = _put(s, PT_NAT_REC, layer[1], lhs.right, lhs.extra)
            return _raw_numeral_alpha(s, _raw_app2(s, lhs.extra, layer[1], hypothesis), step.right)
        return False
    var premise = proof.conversions[first]
    if step.kind == NC_SYMM:
        return (_raw_numeral_alpha(s, step.left, premise.right)
            and _raw_numeral_alpha(s, step.right, premise.left))
    if step.kind == NC_TRANS:
        var next = proof.conversions[second]
        return (_raw_numeral_alpha(s, step.left, premise.left)
            and _raw_numeral_alpha(s, premise.right, next.left)
            and _raw_numeral_alpha(s, step.right, next.right))
    var rhs = s.syntax.nodes[step.right]
    if lhs.kind != PT_APP or rhs.kind != PT_APP:
        return False
    if step.kind == NC_CONG_LEFT:
        return (_raw_numeral_alpha(s, lhs.left, premise.left)
            and _raw_numeral_alpha(s, rhs.left, premise.right)
            and _raw_numeral_alpha(s, lhs.right, rhs.right))
    return (_raw_numeral_alpha(s, lhs.right, premise.left)
        and _raw_numeral_alpha(s, rhs.right, premise.right)
        and _raw_numeral_alpha(s, lhs.left, rhs.left))


def check_native_numeral_view[C: Int, D: Int](value: Int, proof: NumeralView[C, D]) -> Bool:
    """Check every raw entry and primitive conversion, then bind actual endpoints."""
    if (value < 0 or proof.value != value or C <= 0 or D <= 0
        or proof.count <= 0 or proof.count > D or proof.root < 0 or proof.root >= proof.count
        or proof.denotation < 0 or proof.denotation >= proof.bundle.count
        or proof.constructor < 0 or proof.constructor >= proof.bundle.count
        or (value == 0 and proof.predecessor != -1)
        or (value > 0 and (proof.predecessor < 0 or proof.predecessor >= proof.bundle.count))
        or not _raw_numeral_syntax(proof.bundle)):
        return False
    var program = FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        FINISH_LT, FINISH_LT, 2147483647, Int.MAX)
    comptime S = C * 2 + 128
    var state = _make_state[C, S](program, proof.bundle)
    if not _check_raw_numeral_library(state):
        return False
    for j in range(proof.bundle.count):
        var inferred = _infer(state, j)
        if inferred == -1 or inferred == -4:
            return False
    for j in range(proof.count):
        if not _check_numeral_conversion(state, proof, j):
            return False
    var expected = _raw_literal_denotation(state, value, _put(state, PT_DEFINED_REF, value=2))
    var expected_constructor = _put(state, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=0)
    if value == 0:
        if proof.predecessor != -1:
            return False
    else:
        var predecessor = _raw_literal_predecessor_denotation(state, value,
            _put(state, PT_DEFINED_REF, value=2))
        if not _raw_numeral_alpha(state, proof.predecessor, predecessor):
            return False
        expected_constructor = _put(state, PT_APP,
            _put(state, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=1), predecessor)
    var final = proof.conversions[proof.root]
    return (expected >= 0 and _infer(state, expected) == _put(state, PT_NAT)
        and _infer(state, expected_constructor) == _put(state, PT_NAT)
        and _raw_numeral_alpha(state, proof.denotation, expected)
        and _raw_numeral_alpha(state, proof.constructor, expected_constructor)
        and _raw_numeral_alpha(state, final.left, proof.denotation)
        and _raw_numeral_alpha(state, final.right, proof.constructor)
        and state.context_high[final.left] == 0 and state.context_high[final.right] == 0)


def _import_numeral_definition_node[C: Int](mut s: _TypedState[C], node: Int) -> Int:
    if (not s.numerals_verified or not s.nat_canonical or not _consume(s)
        or node < 0 or node >= _NUMERAL_NODE_COUNT):
        return -1
    if s.numeral_nodes[node] >= 0:
        return s.numeral_nodes[node]
    var source = materialize[_NAT_NUMERAL_LIBRARY.bundle]()
    var n = source.nodes[node]
    var left = n.left
    var right = n.right
    var extra = n.extra
    if left >= 0:
        left = _import_numeral_definition_node(s, left)
        if left < 0:
            return -1
    if right >= 0:
        right = _import_numeral_definition_node(s, right)
        if right < 0:
            return -1
    if n.kind == PT_NAT_REC:
        extra = _import_numeral_definition_node(s, extra)
        if extra < 0:
            return -1
    if n.kind == PT_DEFINED_REF:
        return -1  # The pinned three bodies are constructor-only and independent.
    var imported = _put(s, n.kind, left, right, extra, n.value)
    if imported >= 0:
        s.numeral_nodes[node] = imported
    return imported


def _numeral_definition_body[C: Int](mut s: _TypedState[C], definition: Int) -> Int:
    if definition < 0 or definition >= 3:
        return -1
    var source = materialize[_NAT_NUMERAL_LIBRARY.bundle]()
    var body = _import_numeral_definition_node(s, source.definition_bodies[definition])
    return body if _infer(s, body) >= 0 else -1


def _numeral_denotation[C: Int](mut s: _TypedState[C], value: Int) -> Int:
    if not s.numerals_verified or not s.nat_canonical or value < 0:
        return -1
    if value in s.numeral_denotations:
        return s.numeral_denotations.get(value, -1)
    var double = _numeral_definition_body(s, 2)
    if double < 0:
        return -1
    var result = _raw_literal_denotation(s, value, double)
    if result < 0 or _infer(s, result) != _put(s, PT_NAT):
        return -1
    s.numeral_denotations[value] = result
    s.numeral_values[result] = value
    s.numeral_opaque[result] = True
    return result


def _numeral_predecessor_denotation[C: Int](mut s: _TypedState[C], value: Int) -> Int:
    if not s.numerals_verified or not s.nat_canonical or value <= 0:
        return -1
    var double = _numeral_definition_body(s, 2)
    if double < 0:
        return -1
    var result = _raw_literal_predecessor_denotation(s, value, double)
    if result < 0 or _infer(s, result) != _put(s, PT_NAT):
        return -1
    s.numeral_opaque[result] = True
    return result


def _checked_numeral_view[C: Int](mut s: _TypedState[C], node: Int) -> Int:
    """Expose one constructor using an actual checked conversion derivation."""
    if not s.numerals_verified or not s.nat_canonical or node < 0 or node >= s.syntax.count:
        return -1
    var n = s.syntax.nodes[node]
    if n.kind == PT_SUCC:
        var successor = _numeral_definition_body(s, 1)
        var application = _put(s, PT_APP, successor, n.left)
        if _infer(s, application) != _put(s, PT_NAT):
            return -1
        var body = s.syntax.nodes[successor]
        if body.kind != PT_LAM:
            return -1
        var result = _substitute(s, body.right, body.left, n.left)
        var layer = _raw_nat_layer(s, result)
        return result if (layer[0] == 1 and _infer(s, result) == _put(s, PT_NAT)
            and _raw_numeral_alpha(s, layer[1], n.left)) else -1
    if n.kind != PT_LITERAL or n.value < 0:
        return node if (_raw_nat_layer(s, node)[0] >= 0
            and _infer(s, node) == _put(s, PT_NAT)) else -1
    if n.value in s.numeral_views:
        return s.numeral_views.get(n.value, -1)
    var proof = make_numeral_view(n.value)
    if not check_native_numeral_view(n.value, proof):
        return -1
    if _numeral_denotation(s, n.value) < 0:
        return -1
    var constructor = _numeral_definition_body(s, 0)
    if _raw_nat_layer(s, constructor)[0] != 0 or _infer(s, constructor) != _put(s, PT_NAT):
        return -1
    if n.value > 0:
        var predecessor = _numeral_predecessor_denotation(s, n.value)
        if predecessor < 0:
            return -1
        # A machine-word neighbor is only an optional compaction proposal.
        # The pinned compiler can fold a positive all-ones word to a negative
        # Int. The checked binary predecessor remains valid in that case.
        var proposed_value = n.value - 1
        if proposed_value >= 0:
            var proposed = _put(s, PT_LITERAL, value=proposed_value)
            var meaning = _numeral_denotation(s, proposed_value)
            if meaning >= 0 and _raw_numeral_alpha(s, predecessor, meaning):
                predecessor = proposed
        constructor = _checked_numeral_view(s, _put(s, PT_SUCC, predecessor))
        if constructor < 0:
            return -1
    s.numeral_views[n.value] = constructor
    return constructor


def _checked_numeral_successor[C: Int](mut s: _TypedState[C], predecessor: Int) -> Int:
    if (not s.numerals_verified or not s.nat_canonical
        or predecessor < 0 or predecessor >= s.syntax.count
        or _infer(s, predecessor) != _put(s, PT_NAT)):
        return -1
    var p = s.syntax.nodes[predecessor]
    var shorthand = _put(s, PT_SUCC, predecessor)
    if _checked_numeral_view(s, shorthand) < 0:
        return -1
    if p.kind != PT_LITERAL or p.value < 0 or p.value == Int.MAX:
        return shorthand
    var proposed_value = p.value + 1
    if proposed_value < 0:
        return shorthand
    var proposed = _put(s, PT_LITERAL, value=proposed_value)
    var view = _checked_numeral_view(s, proposed)
    var layer = _raw_nat_layer(s, view)
    return proposed if layer[0] == 1 and _numeral_meaning_equal(s, layer[1], predecessor) else shorthand


def _numeral_raw_meaning[C: Int](mut s: _TypedState[C], node: Int) -> Int:
    """Erase compact numeral syntax without evaluating binary Nat recursion."""
    if (not s.numerals_verified or not s.nat_canonical or not _consume(s)
        or node < 0 or node >= s.syntax.count):
        return -1
    if node in s.numeral_opaque:
        return node
    var n = s.syntax.nodes[node]
    if n.kind == PT_LITERAL:
        return _numeral_denotation(s, n.value)
    var layer = _raw_nat_layer(s, node)
    if n.kind == PT_SUCC:
        layer = (1, n.left)
    if layer[0] == 0:
        return node if _infer(s, node) == _put(s, PT_NAT) else -1
    if layer[0] != 1:
        return -1
    var predecessor = _numeral_raw_meaning(s, layer[1])
    if predecessor < 0:
        return -1
    return _checked_numeral_view(s, _put(s, PT_SUCC, predecessor))


def _numeral_meaning_equal[C: Int](mut s: _TypedState[C], a: Int, b: Int) -> Bool:
    """Compare checked raw meanings and one constructor layer, before WHNF.

    This is restricted numeral conversion, not equality reflection. Raw
    endpoints are compared structurally; data definitions are never evaluated.
    """
    var left = _numeral_raw_meaning(s, a)
    var right = _numeral_raw_meaning(s, b)
    if left < 0 or right < 0:
        return False
    if _raw_numeral_alpha(s, left, right):
        return True
    var left_view = _checked_numeral_view(s, a)
    var right_view = _checked_numeral_view(s, b)
    if left_view < 0 or right_view < 0:
        return False
    left = _numeral_raw_meaning(s, left_view)
    right = _numeral_raw_meaning(s, right_view)
    return left >= 0 and right >= 0 and _raw_numeral_alpha(s, left, right)


def _raw_named[C: Int](mut s: _TypedState[C], definition: Int) -> Int:
    return _put(s, PT_DEFINED_REF, value=definition)


def _raw_plus[C: Int](mut s: _TypedState[C], a: Int, b: Int) -> Int:
    return _raw_app2(s, _raw_named(s, 0), a, b)


def _raw_base[C: Int](mut s: _TypedState[C], stride: Int, round: Int) -> Int:
    return _raw_app2(s, _raw_named(s, 5), stride, round)


def _raw_mul[C: Int](mut s: _TypedState[C], a: Int, b: Int) -> Int:
    return _raw_app2(s, _raw_named(s, 1), a, b)


def _raw_le[C: Int](mut s: _TypedState[C], a: Int, b: Int) -> Int:
    return _raw_app2(s, _raw_named(s, 2), a, b)


def _raw_lt[C: Int](mut s: _TypedState[C], a: Int, b: Int) -> Int:
    return _raw_le(s, _put(s, PT_SUCC, a), b)


def _raw_true[C: Int](mut s: _TypedState[C]) -> Int:
    var prop = _put(s, PT_PROP)
    var result = _put(s, PT_VAR, prop)
    var evidence = _put(s, PT_VAR, result)
    return _put(s, PT_PI, result, _put(s, PT_PI, evidence, result))


def _raw_false[C: Int](mut s: _TypedState[C]) -> Int:
    var result = _put(s, PT_VAR, _put(s, PT_PROP))
    return _put(s, PT_PI, result, result)


def _raw_triple_type[C: Int](mut s: _TypedState[C]) -> Int:
    var nat = _put(s, PT_NAT)
    var q = _put(s, PT_VAR, nat)
    var r = _put(s, PT_VAR, nat)
    var c = _put(s, PT_VAR, nat)
    var selector = _put(s, PT_VAR, _put(s, PT_PI, q,
        _put(s, PT_PI, r, _put(s, PT_PI, c, nat))))
    return _put(s, PT_PI, selector, nat)


def _raw_triple[C: Int](mut s: _TypedState[C], q: Int, r: Int, c: Int) -> Int:
    var nat = _put(s, PT_NAT)
    var aq = _put(s, PT_VAR, nat)
    var ar = _put(s, PT_VAR, nat)
    var ac = _put(s, PT_VAR, nat)
    var selector = _put(s, PT_VAR, _put(s, PT_PI, aq,
        _put(s, PT_PI, ar, _put(s, PT_PI, ac, nat))))
    var selected = _put(s, PT_APP, _raw_app2(s, selector, q, r), c)
    return _put(s, PT_LAM, selector, selected)


def _raw_project[C: Int](mut s: _TypedState[C], state: Int, component: Int) -> Int:
    var nat = _put(s, PT_NAT)
    var q = _put(s, PT_VAR, nat)
    var r = _put(s, PT_VAR, nat)
    var c = _put(s, PT_VAR, nat)
    var chosen = q if component == 0 else r if component == 1 else c
    var selector = _put(s, PT_LAM, q, _put(s, PT_LAM, r, _put(s, PT_LAM, c, chosen)))
    return _put(s, PT_APP, state, selector)


def _canonical_successor[C: Int](mut s: _TypedState[C], node: Int, raw: Bool) -> Int:
    return (_put(s, PT_APP, _put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=1), node)
        if raw else _put(s, PT_SUCC, node))


def _canonical_arithmetic_body[C: Int](mut s: _TypedState[C], definition: Int,
    raw_constructors: Bool = False, definition_offset: Int = 0) -> Int:
    """Independent raw recursive specifications for the native arithmetic bridge."""
    var nat = _put(s, PT_NAT)
    var zero = (_put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=0)
        if raw_constructors else _put(s, PT_LITERAL, value=0))
    var a = _put(s, PT_VAR, nat)
    var b = _put(s, PT_VAR, nat)
    var n = _put(s, PT_VAR, nat)
    var acc = _put(s, PT_VAR, nat)
    if definition == 0:
        var step = _put(s, PT_LAM, n, _put(s, PT_LAM, acc, _canonical_successor(s, acc, raw_constructors)))
        return _put(s, PT_LAM, a, _put(s, PT_LAM, b,
            _put(s, PT_NAT_REC, b, a, step)))
    if definition == 1:
        var step = _put(s, PT_LAM, n, _put(s, PT_LAM, acc, _raw_app2(s, _raw_named(s, definition_offset), a, acc)))
        return _put(s, PT_LAM, a, _put(s, PT_LAM, b,
            _put(s, PT_NAT_REC, b, zero, step)))
    if definition == 2:
        var prop = _put(s, PT_PROP)
        var index = _put(s, PT_VAR, nat)
        var previous = _put(s, PT_VAR, _put(s, PT_PI, index, prop))
        var rhs = _put(s, PT_VAR, nat)
        var pred = _put(s, PT_VAR, nat)
        var ignored = _put(s, PT_VAR, prop)
        var right_step = _put(s, PT_LAM, pred, _put(s, PT_LAM, ignored,
            _put(s, PT_APP, previous, pred)))
        var left_step = _put(s, PT_LAM, n, _put(s, PT_LAM, previous,
            _put(s, PT_LAM, rhs, _put(s, PT_NAT_REC, rhs, _raw_false(s), right_step))))
        var initial = _put(s, PT_LAM, index, _raw_true(s))
        var comparison = _put(s, PT_APP, _put(s, PT_NAT_REC, a, initial, left_step), b)
        return _put(s, PT_LAM, a, _put(s, PT_LAM, b, comparison))
    if definition == 3:
        var previous_n = _put(s, PT_VAR, nat)
        var state = _put(s, PT_VAR, _raw_triple_type(s))
        var pred = _put(s, PT_VAR, nat)
        var ignored = _put(s, PT_VAR, _raw_triple_type(s))
        var q = _raw_project(s, state, 0)
        var r = _raw_project(s, state, 1)
        var countdown = _raw_project(s, state, 2)
        var rollover = _raw_triple(s, _canonical_successor(s, q, raw_constructors), zero, a)
        var next = _raw_triple(s, q, _canonical_successor(s, r, raw_constructors), pred)
        var advance = _put(s, PT_NAT_REC, countdown, rollover,
            _put(s, PT_LAM, pred, _put(s, PT_LAM, ignored, next)))
        var step = _put(s, PT_LAM, previous_n, _put(s, PT_LAM, state, advance))
        var initial = _raw_triple(s, zero, zero, a)
        return _put(s, PT_LAM, a, _put(s, PT_LAM, b,
            _put(s, PT_NAT_REC, b, initial, step)))
    if definition == 4:
        var step = _put(s, PT_LAM, n, _put(s, PT_LAM, acc, n))
        return _put(s, PT_LAM, a, _put(s, PT_NAT_REC, a, zero, step))
    if definition == 5:
        var step = _put(s, PT_LAM, n, _put(s, PT_LAM, acc, _raw_app2(s, _raw_named(s, definition_offset), acc, a)))
        return _put(s, PT_LAM, a, _put(s, PT_LAM, b,
            _put(s, PT_NAT_REC, b, zero, step)))
    return -1


def _required_arithmetic_export[C: Int](mut s: _TypedState[C], declaration: Int) -> Int:
    """Spell each required generic theorem independently of its producer."""
    var nat = _put(s, PT_NAT)
    var prop = _put(s, PT_PROP)
    var zero = _put(s, PT_LITERAL, value=0)
    var one = _put(s, PT_LITERAL, value=1)
    var a = _put(s, PT_VAR, nat)
    var b = _put(s, PT_VAR, nat)
    var c = _put(s, PT_VAR, nat)
    if declaration == DEC_BASE_ZERO:
        return _put(s, PT_PI, a, _put(s, PT_EQ, _raw_base(s, a, zero), zero))
    if declaration == DEC_BASE_STEP:
        var lhs = _raw_base(s, a, _put(s, PT_SUCC, b))
        var rhs = _raw_plus(s, _raw_base(s, a, b), a)
        return _put(s, PT_PI, a, _put(s, PT_PI, b, _put(s, PT_EQ, lhs, rhs)))
    if declaration == DEC_NAT_NONNEG:
        return _put(s, PT_PI, a, _raw_le(s, zero, a))
    if declaration == DEC_ORDER_CASES:
        var goal = _put(s, PT_VAR, prop)
        var less = _put(s, PT_VAR, _raw_lt(s, a, b))
        var greater = _put(s, PT_VAR, _raw_le(s, b, a))
        var left = _put(s, PT_VAR, _put(s, PT_PI, less, goal))
        var right = _put(s, PT_VAR, _put(s, PT_PI, greater, goal))
        var result = _put(s, PT_PI, right, goal)
        result = _put(s, PT_PI, left, result)
        result = _put(s, PT_PI, goal, result)
        return _put(s, PT_PI, a, _put(s, PT_PI, b, result))
    if declaration == DEC_LE_ADD or declaration == DEC_LT_ADD:
        var strict = declaration == DEC_LT_ADD
        var domain = _raw_lt(s, a, b) if strict else _raw_le(s, a, b)
        var premise = _put(s, PT_VAR, domain)
        var lhs = _raw_plus(s, a, c)
        var rhs = _raw_plus(s, b, c)
        var conclusion = _raw_lt(s, lhs, rhs) if strict else _raw_le(s, lhs, rhs)
        var result = _put(s, PT_PI, premise, conclusion)
        return _put(s, PT_PI, a, _put(s, PT_PI, b, _put(s, PT_PI, c, result)))
    if (declaration == DEC_LE_TRANS or declaration == DEC_LE_LT_TRANS
        or declaration == DEC_LT_LE_TRANS):
        var first = (_raw_lt(s, a, b) if declaration == DEC_LT_LE_TRANS else _raw_le(s, a, b))
        var second = (_raw_lt(s, b, c) if declaration == DEC_LE_LT_TRANS else _raw_le(s, b, c))
        var conclusion = (_raw_le(s, a, c) if declaration == DEC_LE_TRANS else _raw_lt(s, a, c))
        var h = _put(s, PT_VAR, first)
        var g = _put(s, PT_VAR, second)
        var result = _put(s, PT_PI, h, _put(s, PT_PI, g, conclusion))
        return _put(s, PT_PI, a, _put(s, PT_PI, b, _put(s, PT_PI, c, result)))
    if declaration == DEC_LT_TO_LE_SUCC:
        var h = _put(s, PT_VAR, _raw_lt(s, a, b))
        return _put(s, PT_PI, a, _put(s, PT_PI, b,
            _put(s, PT_PI, h, _raw_le(s, _raw_plus(s, a, one), b))))
    var factor = _put(s, PT_VAR, nat)
    if declaration == DEC_LE_SCALE or declaration == DEC_LT_SCALE_CANCEL:
        var scaled_a = _raw_mul(s, factor, a)
        var scaled_b = _raw_mul(s, factor, b)
        var premise = (_raw_le(s, a, b) if declaration == DEC_LE_SCALE
            else _raw_lt(s, scaled_a, scaled_b))
        var conclusion = (_raw_le(s, scaled_a, scaled_b) if declaration == DEC_LE_SCALE
            else _raw_lt(s, a, b))
        var h = _put(s, PT_VAR, premise)
        var result = _put(s, PT_PI, h, conclusion)
        return _put(s, PT_PI, factor, _put(s, PT_PI, a, _put(s, PT_PI, b, result)))
    var divisor = _put(s, PT_SUCC, factor)
    var state = _raw_app2(s, _raw_named(s, 3), factor, a)
    var quotient = _raw_project(s, state, 0)
    var remainder = _raw_project(s, state, 1)
    if declaration == DEC_DIV_DECOMP:
        var decomposition = _put(s, PT_EQ, a,
            _raw_plus(s, _raw_mul(s, divisor, quotient), remainder))
        return _put(s, PT_PI, factor, _put(s, PT_PI, a, decomposition))
    if declaration == DEC_MOD_LT:
        return _put(s, PT_PI, factor, _put(s, PT_PI, a, _raw_lt(s, remainder, divisor)))
    if declaration == DEC_DIV_QUOTIENT_UNIQUE or declaration == DEC_DIV_REMAINDER_UNIQUE:
        var bound = _put(s, PT_VAR, _raw_lt(s, c, divisor))
        var decomposition = _put(s, PT_EQ, a, _raw_plus(s, _raw_mul(s, divisor, b), c))
        var equality = _put(s, PT_VAR, decomposition)
        var selected = quotient if declaration == DEC_DIV_QUOTIENT_UNIQUE else remainder
        var expected = b if declaration == DEC_DIV_QUOTIENT_UNIQUE else c
        var result = _put(s, PT_PI, bound, _put(s, PT_PI, equality, _put(s, PT_EQ, selected, expected)))
        result = _put(s, PT_PI, a, _put(s, PT_PI, b, _put(s, PT_PI, c, result)))
        return _put(s, PT_PI, factor, result)
    return -1


def _required_operator_representation[C: Int](mut s: _TypedState[C],
    representation: Int) -> Int:
    """Independent raw meanings, over symbolic arguments, of compact syntax."""
    if representation < 0 or representation >= NAT_REPRESENTATIONS:
        return -1
    var nat = _put(s, PT_NAT)
    var a = _put(s, PT_VAR, nat)
    var b = _put(s, PT_VAR, nat)
    var body: Int
    if representation == NAT_REP_ADD:
        body = _raw_plus(s, a, b)
    elif representation == NAT_REP_MUL or representation == NAT_REP_SCALE:
        body = _raw_mul(s, a, b)
    elif representation == NAT_REP_LE:
        body = _raw_le(s, a, b)
    elif representation == NAT_REP_LT:
        body = _raw_lt(s, a, b)
    elif representation == NAT_REP_BASE:
        body = _raw_base(s, a, b)
    else:
        body = _raw_project(s, _raw_app2(s, _raw_named(s, 3), a, b),
            0 if representation == NAT_REP_DIV else 1)
    return _put(s, PT_LAM, a, _put(s, PT_LAM, b, body))


def _check_operator_representations[C: Int, source_capacity: Int](
    mut s: _TypedState[C], library: NatArithmeticLibrary[source_capacity]) -> Bool:
    """Check closed raw recipes without importing or using compact conversions."""
    if s.arithmetic_verified or s.fast_arithmetic:
        return False
    for representation in range(NAT_REPRESENTATIONS):
        var definition = library.representations[representation]
        if definition < 6 or definition >= library.bundle.definition_count:
            return False
        for previous in range(representation):
            if library.representations[previous] == definition:
                return False
        for declaration in range(PROOF_TERM_DECLARATIONS):
            if library.exports[declaration] == definition:
                return False
        var required = _required_operator_representation(s, representation)
        var inferred = _infer(s, required)
        if (required < 0 or inferred < 0 or s.context_high[required] != 0
            or not _convert(s, library.bundle.definition_types[definition], inferred)
            or not _convert(s, library.bundle.definition_bodies[definition], required)):
            return False
    return True


def _import_immutable_node[C: Int, source_capacity: Int](mut s: _TypedState[C],
    source: ProofTermBundle[source_capacity], node: Int) -> Int:
    """Rename actual certified syntax; this creates no typing judgment."""
    if (not s.arithmetic_verified or not s.nat_canonical or not _consume(s)
        or source.count != _IMMUTABLE_NODE_COUNT or node < 0 or node >= source.count):
        return -1
    if s.immutable_nodes[node] >= 0:
        return s.immutable_nodes[node]
    var n = source.nodes[node]
    var left = n.left
    var right = n.right
    var extra = n.extra
    if left >= 0:
        left = _import_immutable_node(s, source, left)
        if left < 0:
            return -1
    if right >= 0:
        right = _import_immutable_node(s, source, right)
        if right < 0:
            return -1
    if n.kind == PT_SUBST or n.kind == PT_NAT_REC:
        extra = _import_immutable_node(s, source, extra)
        if extra < 0:
            return -1
    var kind = n.kind
    if kind == PT_DEFINED_REF:
        if n.value < 0 or n.value >= _IMMUTABLE_DEFINITION_COUNT:
            return -1
        kind = _PT_IMMUTABLE_REF
    var imported = _put(s, kind, left, right, extra, n.value)
    if imported >= 0:
        s.immutable_nodes[node] = imported
    return imported


def _immutable_definition_type[C: Int](mut s: _TypedState[C], definition: Int) -> Int:
    """Use the actual type bound to the actual body by the immutable certificate."""
    if (not s.arithmetic_verified or not s.nat_canonical
        or definition < 0 or definition >= _IMMUTABLE_DEFINITION_COUNT):
        return -1
    if s.immutable_types[definition] >= 0:
        return s.immutable_types[definition]
    if s.immutable_types[definition] != -3:
        return -1
    s.immutable_types[definition] = -4
    var source = materialize[_NAT_ARITHMETIC_LIBRARY.bundle]()
    var declared = _import_immutable_node(s, source, source.definition_types[definition])
    var sort = _infer(s, declared)
    var prop = _put(s, PT_PROP)
    var universe = _put(s, PT_TYPE)
    if (declared < 0 or (sort != prop and sort != universe)
        or s.context_high[declared] != 0):
        s.immutable_types[definition] = -1
        return -1
    s.immutable_types[definition] = declared
    return declared


def _immutable_definition_body[C: Int](mut s: _TypedState[C], definition: Int) -> Int:
    """Load an actual body only when conversion requests delta reduction."""
    var declared = _immutable_definition_type(s, definition)
    if declared < 0:
        return -1
    if s.immutable_bodies[definition] >= 0:
        return s.immutable_bodies[definition]
    if s.immutable_bodies[definition] != -3:
        return -1
    s.immutable_bodies[definition] = -4
    var source = materialize[_NAT_ARITHMETIC_LIBRARY.bundle]()
    var body = _import_immutable_node(s, source, source.definition_bodies[definition])
    var inferred = _infer(s, body)
    if (body < 0 or not _convert(s, inferred, declared) or s.context_high[body] != 0):
        s.immutable_bodies[definition] = -1
        return -1
    s.immutable_bodies[definition] = body
    return body


def _instantiate_arithmetic_export[C: Int](mut s: _TypedState[C], declaration: Int,
    parameter: Int) -> Int:
    """Build ordinary applications of the actual certified proof definition."""
    if (not s.arithmetic_verified or declaration < 0
        or declaration >= PROOF_TERM_DECLARATIONS
        or declaration == DEC_DONE_STOP or declaration == DEC_DONE_STEP):
        return -1
    var scaling = declaration == DEC_LE_SCALE or declaration == DEC_LT_SCALE_CANCEL
    var division = (declaration == DEC_DIV_DECOMP or declaration == DEC_MOD_LT
        or declaration == DEC_DIV_QUOTIENT_UNIQUE or declaration == DEC_DIV_REMAINDER_UNIQUE)
    if ((scaling and (parameter < 0 or (parameter == 0 and declaration != DEC_LE_SCALE)))
        or (division and parameter <= 0) or (not scaling and not division and parameter != -1)):
        return -1
    var exports = materialize[_NAT_ARITHMETIC_LIBRARY.exports]()
    var definition = exports[declaration]
    if definition < 6 or definition >= _IMMUTABLE_DEFINITION_COUNT:
        return -1
    var evidence = _put(s, _PT_IMMUTABLE_REF, value=definition)
    if declaration == DEC_BASE_ZERO or declaration == DEC_BASE_STEP:
        evidence = _put(s, PT_APP, evidence, _put(s, PT_LITERAL, value=s.program.stride))
    elif scaling:
        evidence = _put(s, PT_APP, evidence, _put(s, PT_LITERAL, value=parameter))
    elif division:
        var predecessor = _positive_literal_predecessor(s, parameter)
        if predecessor < 0:
            return -1
        evidence = _put(s, PT_APP, evidence, predecessor)
    # No theorem signature is reconstructed. Reference inference imports its
    # actual declared type; ordinary APP checks the Pi domain and substitutes.
    var result = _infer(s, evidence)
    if result < 0 or _infer(s, result) != _put(s, PT_PROP) or s.context_high[result] != 0:
        return -1
    return result


def _immutable_binary_arguments[C: Int](s: _TypedState[C], node: Int,
    definition: Int) -> Tuple[Int, Int]:
    if node < 0 or node >= s.syntax.count:
        return (-1, -1)
    var outer = s.syntax.nodes[node]
    if outer.kind != PT_APP or outer.left < 0 or outer.left >= s.syntax.count:
        return (-1, -1)
    var inner = s.syntax.nodes[outer.left]
    if inner.kind != PT_APP or inner.left < 0 or inner.left >= s.syntax.count:
        return (-1, -1)
    if (inner.right < 0 or inner.right >= s.syntax.count
        or outer.right < 0 or outer.right >= s.syntax.count):
        return (-1, -1)
    var head = s.syntax.nodes[inner.left]
    if head.kind != _PT_IMMUTABLE_REF or head.value != definition:
        return (-1, -1)
    return (inner.right, outer.right)


def _projection_component[C: Int](mut s: _TypedState[C], selector: Int) -> Int:
    """Recognize precisely the first two typed Church-triple projections."""
    var binders = Array[Int, 3](fill=-1)
    var cursor = selector
    var nat = _put(s, PT_NAT)
    for j in range(3):
        if cursor < 0 or cursor >= s.syntax.count:
            return -1
        var n = s.syntax.nodes[cursor]
        if n.kind != PT_LAM or n.left < 0 or n.left >= s.syntax.count:
            return -1
        if s.syntax.nodes[n.left].kind != PT_VAR or _infer(s, n.left) != nat:
            return -1
        for earlier in range(j):
            if binders[earlier] == n.left:
                return -1
        binders[j] = n.left
        cursor = n.right
    if cursor == binders[0]:
        return 0
    if cursor == binders[1]:
        return 1
    return -1


def _instantiate_operator_representation[C: Int](mut s: _TypedState[C],
    representation: Int, a: Int, b: Int) -> Int:
    """Infer an ordinary application of an actual checked raw abbreviation."""
    if (not s.arithmetic_verified or not s.nat_canonical
        or representation < 0 or representation >= NAT_REPRESENTATIONS):
        return -1
    var definitions = materialize[_NAT_ARITHMETIC_LIBRARY.representations]()
    var definition = definitions[representation]
    if definition < 6 or definition >= _IMMUTABLE_DEFINITION_COUNT:
        return -1
    var application = _raw_app2(s, _put(s, _PT_IMMUTABLE_REF, value=definition), a, b)
    return application if _infer(s, application) >= 0 else -1


def _is_operator_representation_definition(definition: Int) -> Bool:
    var definitions = materialize[_NAT_ARITHMETIC_LIBRARY.representations]()
    for representation in range(NAT_REPRESENTATIONS):
        if definitions[representation] == definition:
            return True
    return False


def _operator_administrative_whnf[C: Int](mut s: _TypedState[C], node: Int) -> Int:
    """Erase abbreviations and beta-reduce; data definitions stay opaque.

    Numeral normalization consumes the same checked constructor view used by
    ordinary reduction and induction. There is no full NatRec, native data
    arithmetic, affine or order evaluation here. Checked universal recipes
    instantiate at large literals without building unary successor chains.
    """
    if not _consume(s) or node < 0 or node >= s.syntax.count:
        return -1
    if s.numerals_verified and node in s.numeral_values:
        var value = s.numeral_values.get(node, -1)
        var literal = _put(s, PT_LITERAL, value=value)
        return literal if _checked_numeral_view(s, literal) >= 0 else -1
    if s.numerals_verified and node in s.numeral_opaque:
        return node
    var erased = _expand_native_arithmetic(s, node)
    if erased < 0:
        return -1
    if erased != node:
        return _operator_administrative_whnf(s, erased)
    var n = s.syntax.nodes[node]
    if n.kind == _PT_IMMUTABLE_REF and _is_operator_representation_definition(n.value):
        return _operator_administrative_whnf(s, _immutable_definition_body(s, n.value))
    if n.kind == PT_SUCC and s.nat_canonical:
        var predecessor = _operator_administrative_whnf(s, n.left)
        if predecessor < 0:
            return -1
        return _checked_numeral_successor(s, predecessor)
    if (n.kind == PT_CONSTRUCTOR and n.extra == NAT_FAMILY_ID and n.value == 0
        and s.numerals_verified):
        var zero = _put(s, PT_LITERAL, value=0)
        return zero if _checked_numeral_view(s, zero) >= 0 else -1
    if n.kind == PT_APP:
        var f = _operator_administrative_whnf(s, n.left)
        if f < 0:
            return -1
        var head = s.syntax.nodes[f]
        if (head.kind == PT_CONSTRUCTOR and head.extra == NAT_FAMILY_ID and head.value == 1
            and s.numerals_verified):
            return _checked_numeral_successor(s, _operator_administrative_whnf(s, n.right))
        if head.kind == PT_LAM:
            return _operator_administrative_whnf(s,
                _substitute(s, head.right, head.left, n.right))
        return _put(s, PT_APP, f, n.right) if f != n.left else node
    return node


def _operator_representation_equal_under[C: Int](mut s: _TypedState[C],
    var a: Int, var b: Int, mut left_names: List[Int],
    mut right_names: List[Int], depth: Int) -> Bool:
    if not _consume(s) or depth >= C:
        return False
    if _numeral_meaning_equal(s, a, b):
        return True
    a = _operator_administrative_whnf(s, a)
    b = _operator_administrative_whnf(s, b)
    if a < 0 or b < 0:
        return False
    var x = s.syntax.nodes[a]
    var y = s.syntax.nodes[b]
    if x.kind != y.kind:
        return False
    if x.kind == PT_VAR:
        var ax = -1
        var by = -1
        for j in range(depth):
            if left_names[j] == a:
                ax = j
            if right_names[j] == b:
                by = j
        return ax == by if ax >= 0 or by >= 0 else a == b
    if x.kind == PT_PI or x.kind == PT_LAM:
        if not _operator_representation_equal_under(s,
            s.syntax.nodes[x.left].left, s.syntax.nodes[y.left].left,
            left_names, right_names, depth):
            return False
        if len(left_names) == depth and len(right_names) == depth:
            left_names.append(x.left)
            right_names.append(y.left)
        elif len(left_names) > depth and len(right_names) > depth:
            left_names[depth] = x.left
            right_names[depth] = y.left
        else:
            return False
        return _operator_representation_equal_under(s, x.right, y.right,
            left_names, right_names, depth + 1)
    if (x.value != y.value
        or ((x.kind == PT_CONSTRUCTOR or x.kind == PT_ARITHMETIC) and x.extra != y.extra)):
        return False
    if x.left >= 0 and not _operator_representation_equal_under(s, x.left, y.left,
        left_names, right_names, depth):
        return False
    if x.right >= 0 and not _operator_representation_equal_under(s, x.right, y.right,
        left_names, right_names, depth):
        return False
    if (x.kind == PT_SUBST or x.kind == PT_NAT_REC) and not _operator_representation_equal_under(s,
        x.extra, y.extra, left_names, right_names, depth):
        return False
    return True


def _operator_representation_equal[C: Int](mut s: _TypedState[C], a: Int, b: Int) -> Bool:
    var left_names = List[Int]()
    var right_names = List[Int]()
    return _operator_representation_equal_under(s, a, b, left_names, right_names, 0)


def _checked_operator_compaction[C: Int](mut s: _TypedState[C],
    original: Int, candidate: Int) -> Int:
    """A heuristic has no authority until its actual erased meanings agree."""
    var original_type = _infer(s, original)
    var candidate_type = _infer(s, candidate)
    if original_type < 0:
        return -1
    if candidate_type < 0 or not _convert(s, original_type, candidate_type):
        return original
    return candidate if _operator_representation_equal(s, original, candidate) else original


def _compact_arithmetic[C: Int](mut s: _TypedState[C], node: Int) -> Int:
    """Suggest compact syntax, then check it against the certified raw meaning."""
    if (not s.arithmetic_verified or not s.fast_arithmetic or not s.nat_canonical
        or node < 0 or node >= s.syntax.count):
        return node
    # Expose only actual checked alias applications for the heuristic. Its
    # candidate is still compared with the untouched original node below.
    var source = node
    if s.syntax.nodes[node].kind == PT_APP:
        var definitions = materialize[_NAT_ARITHMETIC_LIBRARY.representations]()
        for representation in range(NAT_REPRESENTATIONS):
            var arguments = _immutable_binary_arguments(s, node, definitions[representation])
            if arguments[0] >= 0:
                source = _operator_administrative_whnf(s, node)
                if source < 0:
                    return -1
                break
    var n = s.syntax.nodes[source]
    if n.kind == PT_LT:
        return _checked_operator_compaction(s, node,
            _put(s, PT_LE, _put(s, PT_SUCC, n.left), n.right))
    if n.kind == PT_MUL:
        var coefficient = _operator_administrative_whnf(s, n.left)
        if coefficient < 0:
            return node
        var factor = s.syntax.nodes[coefficient]
        if factor.kind == PT_LITERAL:
            return _checked_operator_compaction(s, node,
                _put(s, PT_SCALE, n.right, value=factor.value))
        return node
    if n.kind != PT_APP:
        return node
    for definition in range(6):
        if definition == 3 or definition == 4:
            continue
        var arguments = _immutable_binary_arguments(s, source, definition)
        if arguments[0] < 0:
            continue
        if definition == 0:
            return _checked_operator_compaction(s, node,
                _put(s, PT_ADD, arguments[0], arguments[1]))
        if definition == 1:
            var coefficient = _operator_administrative_whnf(s, arguments[0])
            if coefficient < 0:
                return node
            var factor = s.syntax.nodes[coefficient]
            if factor.kind == PT_LITERAL:
                return _checked_operator_compaction(s, node,
                    _put(s, PT_SCALE, arguments[1], value=factor.value))
            return _checked_operator_compaction(s, node,
                _put(s, PT_MUL, arguments[0], arguments[1]))
        if definition == 2:
            return _checked_operator_compaction(s, node,
                _put(s, PT_LE, arguments[0], arguments[1]))
        var stride = _operator_administrative_whnf(s, arguments[0])
        if stride < 0:
            return node
        var stride_node = s.syntax.nodes[stride]
        if stride_node.kind == PT_LITERAL and stride_node.value == s.program.stride:
            return _checked_operator_compaction(s, node,
                _put(s, PT_BASE, arguments[1]))
        return node
    var state = _immutable_binary_arguments(s, n.left, 3)
    if state[0] < 0:
        return node
    var component = _projection_component(s, n.right)
    if component < 0:
        return node
    var predecessor = _operator_administrative_whnf(s, state[0])
    if predecessor < 0:
        return node
    var k = s.syntax.nodes[predecessor]
    if k.kind != PT_LITERAL or k.value < 0 or k.value == Int.MAX:
        return node
    return _checked_operator_compaction(s, node,
        _put(s, PT_DIV if component == 0 else PT_MOD, state[1], value=k.value + 1))


def _expand_native_arithmetic[C: Int](mut s: _TypedState[C], node: Int) -> Int:
    """Elaborate compact syntax by ordinary application of actual checked recipes."""
    if node < 0 or node >= s.syntax.count:
        return -1
    var n = s.syntax.nodes[node]
    if (n.kind != PT_ADD and n.kind != PT_MUL and n.kind != PT_SCALE
        and n.kind != PT_LE and n.kind != PT_LT
        and n.kind != PT_DIV and n.kind != PT_MOD and n.kind != PT_BASE):
        return node
    if not s.arithmetic_verified or not s.nat_canonical:
        return -1
    if n.kind == PT_BASE:
        return _instantiate_operator_representation(s, NAT_REP_BASE,
            _put(s, PT_LITERAL, value=s.program.stride), n.left)
    if n.kind == PT_ADD:
        return _instantiate_operator_representation(s, NAT_REP_ADD, n.left, n.right)
    if n.kind == PT_MUL:
        return _instantiate_operator_representation(s, NAT_REP_MUL, n.left, n.right)
    if n.kind == PT_SCALE:
        if n.value < 0:
            return -1
        return _instantiate_operator_representation(s, NAT_REP_SCALE,
            _put(s, PT_LITERAL, value=n.value), n.left)
    if n.kind == PT_LE:
        return _instantiate_operator_representation(s, NAT_REP_LE, n.left, n.right)
    if n.kind == PT_LT:
        return _instantiate_operator_representation(s, NAT_REP_LT, n.left, n.right)
    if n.value <= 0:
        return -1
    var predecessor = _positive_literal_predecessor(s, n.value)
    if predecessor < 0:
        return -1
    return _instantiate_operator_representation(s,
        NAT_REP_DIV if n.kind == PT_DIV else NAT_REP_MOD, predecessor, n.left)


def _positive_literal_predecessor[C: Int](mut s: _TypedState[C], value: Int) -> Int:
    """Extract the predecessor proved by the checked positive numeral view.

    In particular, neither divisor adapter interprets native `value - 1`.
    The predecessor may be a raw binary Nat without a native-word payload.
    """
    if value <= 0:
        return -1
    var view = _checked_numeral_view(s, _put(s, PT_LITERAL, value=value))
    var layer = _raw_nat_layer(s, view)
    return layer[1] if layer[0] == 1 else -1


def _literal_arithmetic_syntax[C: Int](p: ProofTermBundle[C]) -> Bool:
    if (not _valid_syntax(p) or p.definition_count != 9
        or p.family_arity != -1 or p.constructor_count != 0
        or p.nat_constructor_count != 2):
        return False
    for j in range(p.count):
        var n = p.nodes[j]
        if (n.kind != PT_TYPE and n.kind != PT_PROP and n.kind != PT_NAT
            and n.kind != PT_VAR and n.kind != PT_PI and n.kind != PT_LAM
            and n.kind != PT_APP and n.kind != PT_NAT_REC
            and n.kind != PT_DEFINED_REF and n.kind != PT_CONSTRUCTOR
            and n.kind != PT_EQ and n.kind != PT_REFL):
            return False
        if n.kind == PT_CONSTRUCTOR and n.extra != NAT_FAMILY_ID:
            return False
    return True


def _literal_data_node[C: Int](mut s: _TypedState[C], node: Int,
    mut imported: Array[Int, _IMMUTABLE_NODE_COUNT]) -> Int:
    """Elaborate actual immutable data syntax into the raw local namespace.

    Only its zero literal and successor shorthand are admitted here. There is
    no native constant calculation, theorem export, or private reference.
    """
    if node < 0 or node >= _IMMUTABLE_NODE_COUNT or not _consume(s):
        return -1
    if imported[node] >= 0:
        return imported[node]
    var source = materialize[_NAT_ARITHMETIC_LIBRARY.bundle]()
    var n = source.nodes[node]
    var result = -1
    if n.kind == PT_LITERAL:
        if n.value == 0:
            result = _put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=0)
    elif n.kind == PT_DEFINED_REF:
        if n.value >= 0 and n.value < 6:
            result = _raw_named(s, n.value + 3)
    else:
        if (n.kind != PT_TYPE and n.kind != PT_PROP and n.kind != PT_NAT
            and n.kind != PT_VAR and n.kind != PT_PI and n.kind != PT_LAM
            and n.kind != PT_APP and n.kind != PT_NAT_REC and n.kind != PT_SUCC):
            return -1
        var left = n.left
        var right = n.right
        var extra = n.extra
        if left >= 0:
            left = _literal_data_node(s, left, imported)
            if left < 0:
                return -1
        if right >= 0:
            right = _literal_data_node(s, right, imported)
            if right < 0:
                return -1
        if n.kind == PT_NAT_REC:
            extra = _literal_data_node(s, extra, imported)
            if extra < 0:
                return -1
        if n.kind == PT_SUCC:
            result = _put(s, PT_APP,
                _put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=1), left)
        else:
            result = _put(s, n.kind, left, right, extra, n.value)
    imported[node] = result
    return result


def _literal_arithmetic_source[C: Int](mut s: _TypedState[C], operation: Int,
    left: Int, right: Int) -> Int:
    var double = _raw_named(s, 2)
    var a = _raw_literal_denotation(s, left, double)
    var b = _raw_literal_denotation(s, right, double)
    if a < 0 or b < 0:
        return -1
    if operation == PT_ADD or operation == PT_MUL or operation == PT_SCALE or operation == PT_BASE:
        var definition = 3 if operation == PT_ADD else 8 if operation == PT_BASE else 4
        return _raw_app2(s, _raw_named(s, definition), a, b)
    if operation == PT_DIV or operation == PT_MOD:
        if right <= 0:
            return -1
        var predecessor = _raw_literal_predecessor_denotation(s, right, double)
        if predecessor < 0:
            return -1
        var successor = _put(s, PT_APP,
            _put(s, PT_CONSTRUCTOR, extra=NAT_FAMILY_ID, value=1), predecessor)
        if not _convert(s, b, successor):
            return -1
        return _raw_project(s, _raw_app2(s, _raw_named(s, 6), predecessor, a),
            0 if operation == PT_DIV else 1)
    return -1


def _check_raw_nat_data_prefix[C: Int](mut state: _TypedState[C]) -> Bool:
    """Pin Num's three and arithmetic's six actual raw data definitions.

    This grants no proof schema. Further definitions must still pass ordinary
    ordered definition admission. Neither global arithmetic certification nor
    native arithmetic/numeral/reflection shortcuts are used here.
    """
    if state.syntax.definition_count < 9:
        return False
    var immutable = materialize[_NAT_ARITHMETIC_LIBRARY.bundle]()
    var imported = Array[Int, _IMMUTABLE_NODE_COUNT](fill=-1)
    for definition in range(9):
        var expected_body: Int
        if definition < 3:
            expected_body = _canonical_numeral_body(state, definition)
        else:
            var data = definition - 3
            var expected_type = _literal_data_node(state, immutable.definition_types[data], imported)
            expected_body = _literal_data_node(state, immutable.definition_bodies[data], imported)
            if not _raw_numeral_alpha(state, state.syntax.definition_types[definition], expected_type):
                return False
            var canonical = _canonical_arithmetic_body(state, data, True, 3)
            if not _raw_numeral_alpha(state, expected_body, canonical):
                return False
        if not _raw_numeral_alpha(state, state.syntax.definition_bodies[definition], expected_body):
            return False
    return True


def check_native_literal_arithmetic_request[C: Int, scratch: Int = 4096](
    proof: LiteralArithmeticProof[C], operation: Int, left: Int, right: Int,
    candidate: Int) -> Bool:
    """Check Refl(raw operation) against Eq(raw operation, E(candidate)).

    This is ordinary definitional conversion, not propositional equality
    reflection. All arithmetic/numeral/reflection shortcuts are disabled.
    Source syntax, actual definitions, endpoints, closure and every unused
    node are checked independently. The scratch bound affects completeness,
    never the meaning of a successful judgment or the computation shape.
    """
    if (scratch <= 0 or proof.bundle.count > scratch or left < 0 or right < 0
        or candidate < 0 or proof.operation != operation
        or proof.left_value != left or proof.right_value != right
        or proof.candidate_value != candidate or not _literal_arithmetic_syntax(proof.bundle)
        or proof.source < 0 or proof.source >= proof.bundle.count
        or proof.result < 0 or proof.result >= proof.bundle.count
        or proof.proof < 0 or proof.proof >= proof.bundle.count
        or proof.bundle.root != proof.proof):
        return False
    var program = FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        FINISH_LT, FINISH_LT, 2147483647, Int.MAX)
    var state = _make_state[C, scratch](program, proof.bundle)
    # Independent, finite conversion workspace; no consumer fuel is spent.
    state.fuel = scratch * 32
    if not _check_inductive_environment(state):
        return False
    var nat = _put(state, PT_NAT)
    if not _check_raw_nat_data_prefix(state):
        return False
    for j in range(proof.bundle.count):
        var inferred = _infer(state, j)
        if inferred == -1 or inferred == -4:
            return False
    var source = _literal_arithmetic_source(state, operation, left, right)
    var result = _raw_literal_denotation(state, candidate, _raw_named(state, 2))
    if (source < 0 or result < 0
        or not _raw_numeral_alpha(state, proof.source, source)
        or not _raw_numeral_alpha(state, proof.result, result)
        or not _convert(state, _infer(state, source), nat)
        or not _convert(state, _infer(state, result), nat)
        or state.context_high[source] != 0 or state.context_high[result] != 0
        or state.context_high[proof.proof] != 0):
        return False
    var expected = _put(state, PT_EQ, source, result)
    return _convert(state, _infer(state, proof.proof), expected)


def check_native_literal_arithmetic[C: Int, scratch: Int = 4096](
    proof: LiteralArithmeticProof[C]) -> Bool:
    return check_native_literal_arithmetic_request[C, scratch](proof,
        proof.operation, proof.left_value, proof.right_value, proof.candidate_value)


def _checked_literal_arithmetic[C: Int](mut s: _TypedState[C], operation: Int,
    left: Int, right: Int, candidate: Int) -> Bool:
    if (not s.arithmetic_verified or not s.numerals_verified or not s.nat_canonical
        or left < 0 or right < 0 or candidate < 0):
        return False
    var key = (operation, left, right, candidate)
    if key in s.literal_conversions:
        return s.literal_conversions.get(key, False)
    var proof = make_literal_arithmetic_proof(operation, left, right, candidate)
    var accepted = check_native_literal_arithmetic(proof)
    s.literal_conversions[key] = accepted
    return accepted


def _headroom_syntax[C: Int](p: ProofTermBundle[C]) -> Bool:
    if (not _valid_syntax(p) or p.definition_count < 9
        or p.family_arity != -1 or p.constructor_count != 0
        or p.nat_constructor_count != 2):
        return False
    for j in range(p.count):
        var n = p.nodes[j]
        if (n.kind != PT_TYPE and n.kind != PT_PROP and n.kind != PT_NAT
            and n.kind != PT_VAR and n.kind != PT_PI and n.kind != PT_LAM
            and n.kind != PT_APP and n.kind != PT_NAT_REC
            and n.kind != PT_DEFINED_REF and n.kind != PT_CONSTRUCTOR
            and n.kind != PT_ELIMINATOR and n.kind != PT_EQ
            and n.kind != PT_REFL and n.kind != PT_SUBST):
            return False
        if n.kind == PT_CONSTRUCTOR and n.extra != NAT_FAMILY_ID:
            return False
        if n.kind == PT_ELIMINATOR and n.value != NAT_FAMILY_ID:
            return False
    return True


def check_native_headroom_request[C: Int, scratch: Int = 16384](
    proof: HeadroomProof[C], count_limit: Int, stride: Int, integer_limit: Int) -> Bool:
    """Check an ordinary raw Nat theorem, with no affine/order reflection.

    The independent expected theorem is forall n, n<=E(cap) implies
    Add(n,E(stride))<=E(limit). All certificate syntax, including unused terms,
    is checked in a fresh state. Extra helper lemmas are ordinary closed
    definitions; there is no trusted headroom, sum, carry or comparison rule.
    Resource failure rejects the mandatory certificate without a fallback.
    """
    if (scratch <= 0 or count_limit < 0 or stride < 0 or integer_limit < 0
        or proof.count_limit != count_limit or proof.stride != stride
        or proof.integer_limit != integer_limit or proof.bundle.count > scratch
        or not _headroom_syntax(proof.bundle) or proof.bundle.root != proof.proof):
        return False
    var endpoints = [proof.source, proof.limit, proof.sum, proof.slack,
        proof.sum_equality, proof.slack_equality, proof.bound, proof.proof]
    for node in endpoints:
        if node < 0 or node >= proof.bundle.count:
            return False
    var program = FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        FINISH_LT, FINISH_LT, 2147483647, Int.MAX)
    var state = _make_state[C, scratch](program, proof.bundle)
    state.fuel = scratch * 256
    # _make_state leaves numeral, arithmetic and fast-arithmetic flags false.
    # No previously inferred reflected proof can enter this fresh state.
    if not _check_inductive_environment(state) or not _check_raw_nat_data_prefix(state):
        return False
    for j in range(proof.bundle.count):
        var inferred = _infer(state, j)
        if inferred == -1 or inferred == -4:
            return False
    var nat = _put(state, PT_NAT)
    var double = _raw_named(state, 2)
    var cap = _raw_literal_denotation(state, count_limit, double)
    var step = _raw_literal_denotation(state, stride, double)
    var limit = _raw_literal_denotation(state, integer_limit, double)
    var add = _raw_named(state, 3)
    var le = _raw_named(state, 5)
    var source = _raw_app2(state, add, cap, step)
    if (not _raw_numeral_alpha(state, proof.source, source)
        or not _raw_numeral_alpha(state, proof.limit, limit)):
        return False
    for node in endpoints:
        if _infer(state, node) < 0 or state.context_high[node] != 0:
            return False
    if (not _convert(state, _infer(state, proof.sum), nat)
        or not _convert(state, _infer(state, proof.slack), nat)):
        return False
    var sum_type = _put(state, PT_EQ, source, proof.sum)
    var slack_type = _put(state, PT_EQ,
        _raw_app2(state, add, proof.sum, proof.slack), limit)
    var bound_type = _raw_app2(state, le, source, limit)
    var n = _put(state, PT_VAR, nat)
    var premise = _put(state, PT_VAR, _raw_app2(state, le, n, cap))
    var conclusion = _raw_app2(state, le, _raw_app2(state, add, n, step), limit)
    var expected = _put(state, PT_PI, n, _put(state, PT_PI, premise, conclusion))
    return (_infer(state, expected) == _put(state, PT_PROP)
        and state.context_high[expected] == 0
        and _convert(state, _infer(state, proof.sum_equality), sum_type)
        and _convert(state, _infer(state, proof.slack_equality), slack_type)
        and _convert(state, _infer(state, proof.bound), bound_type)
        and _convert(state, _infer(state, proof.proof), expected))


def check_native_headroom[C: Int, scratch: Int = 16384](proof: HeadroomProof[C]) -> Bool:
    return check_native_headroom_request[C, scratch](proof,
        proof.count_limit, proof.stride, proof.integer_limit)


def _whnf[capacity: Int](mut s: _TypedState[capacity], node: Int,
    compact: Bool = True) -> Int:
    if not _consume(s) or node < 0 or node >= s.syntax.count:
        return -1
    if s.numerals_verified and node in s.numeral_values:
        var value = s.numeral_values.get(node, -1)
        var literal = _put(s, PT_LITERAL, value=value)
        return literal if _checked_numeral_view(s, literal) >= 0 else -1
    if s.numerals_verified and node in s.numeral_opaque:
        return node
    if compact:
        var compacted = _compact_arithmetic(s, node)
        if compacted < 0:
            return -1
        if compacted != node:
            return _whnf(s, compacted, compact)
    var n = s.syntax.nodes[node]
    if n.kind == _PT_IMMUTABLE_REF:
        var body = _immutable_definition_body(s, n.value)
        return _whnf(s, body, compact)
    if n.kind == PT_DEFINED_REF:
        if n.value < 0 or n.value >= s.admitted_definitions:
            return -1
        return _whnf(s, s.syntax.definition_bodies[n.value], compact)
    if n.kind == PT_SUCC:
        var predecessor = _whnf(s, n.left, compact)
        if predecessor < 0:
            return -1
        return _checked_numeral_successor(s, predecessor)
    if n.kind == PT_NAT_REC:
        return _reduce_nat_rec(s, node, compact)
    if n.kind == PT_FAMILY and n.value == NAT_FAMILY_ID:
        return _put(s, PT_NAT)
    if (n.kind == PT_CONSTRUCTOR and n.extra == NAT_FAMILY_ID and s.nat_canonical
        and s.numerals_verified and n.value == 0):
        var zero = _put(s, PT_LITERAL, value=0)
        return zero if _checked_numeral_view(s, zero) >= 0 else -1
    if n.kind == PT_APP:
        var f = _whnf(s, n.left, compact)
        if f < 0:
            return -1
        var head = s.syntax.nodes[f]
        if (head.kind == PT_CONSTRUCTOR and head.extra == NAT_FAMILY_ID
            and s.nat_canonical and s.numerals_verified and head.value == 1):
            return _checked_numeral_successor(s, _whnf(s, n.right, compact))
        if head.kind == PT_LAM:
            var result = _substitute(s, head.right, head.left, n.right)
            return _whnf(s, result, compact)
        var application = node
        if f != n.left:
            application = _put(s, PT_APP, f, n.right)
            if application < 0:
                return -1
        var reduced = _reduce_eliminator(s, application, compact)
        if reduced < 0:
            return -1
        if reduced != application:
            return _whnf(s, reduced, compact)
        return application
    elif n.kind == PT_ADD:
        var a = _whnf(s, n.left, compact)
        var b = _whnf(s, n.right, compact)
        if a < 0 or b < 0:
            return -1
        var x = s.syntax.nodes[a]
        var y = s.syntax.nodes[b]
        if x.kind == PT_LITERAL and y.kind == PT_LITERAL:
            # Native arithmetic proposes a payload; raw Nat conversion alone
            # authorizes replacing the expression. Overflow or compiler
            # constant-folding defects therefore cannot create an equation.
            var candidate = x.value + y.value
            if _checked_literal_arithmetic(s, PT_ADD, x.value, y.value, candidate):
                return _put(s, PT_LITERAL, value=candidate)
        if a != n.left or b != n.right:
            return _put(s, PT_ADD, a, b)
    elif n.kind == PT_DIV or n.kind == PT_MOD:
        # Symbolic quotients and remainders remain opaque natural expressions.
        # The arithmetic bridge is enabled only by the canonical Nat pin.
        if not s.nat_canonical or n.value <= 0:
            return -1
        var a = _whnf(s, n.left, compact)
        if a < 0:
            return -1
        var x = s.syntax.nodes[a]
        if x.kind == PT_LITERAL:
            var candidate = x.value // n.value if n.kind == PT_DIV else x.value % n.value
            if _checked_literal_arithmetic(s, n.kind, x.value, n.value, candidate):
                return _put(s, PT_LITERAL, value=candidate)
        if a != n.left:
            return _put(s, n.kind, a, value=n.value)
    elif n.kind == PT_SCALE:
        var a = _whnf(s, n.left, compact)
        if a < 0:
            return -1
        var x = s.syntax.nodes[a]
        if x.kind == PT_LITERAL:
            var candidate = x.value * n.value
            if _checked_literal_arithmetic(s, PT_SCALE, n.value, x.value, candidate):
                return _put(s, PT_LITERAL, value=candidate)
        if a != n.left:
            return _put(s, PT_SCALE, a, value=n.value)
    return node


def _beta_head[C: Int](mut s: _TypedState[C], node: Int) -> Int:
    """Remove administrative beta spines without unfolding data definitions.

    Equality substitution returns an application of its motive. Comparing that
    type with an already substituted type should not eagerly evaluate the
    latter's closed Nat recursors. Every step here is ordinary beta reduction.
    """
    if not _consume(s) or node < 0 or node >= s.syntax.count:
        return -1
    var n = s.syntax.nodes[node]
    if n.kind != PT_APP:
        return node
    var function = _beta_head(s, n.left)
    if function < 0:
        return -1
    var head = s.syntax.nodes[function]
    if head.kind == PT_LAM:
        return _beta_head(s, _substitute(s, head.right, head.left, n.right))
    return node if function == n.left else _put(s, PT_APP, function, n.right)


def _equal_under[capacity: Int](mut s: _TypedState[capacity], var a: Int, var b: Int,
    mut left_names: List[Int], mut right_names: List[Int], depth: Int,
    compact: Bool = True) -> Bool:
    if not _consume(s) or depth >= capacity:
        return False
    # Structural alpha equality is reflexivity modulo binder names. Trying it
    # before reduction preserves shared binary Nat terms without unfolding
    # their recursors. Reuse the active maps: identical IDs alone are unsound
    # under different binder renamings. Failed attempts only alter inactive
    # map slots and cannot grant equality.
    if _raw_numeral_alpha_under(s, a, b, left_names, right_names, depth):
        return True
    var beta_a = _beta_head(s, a)
    var beta_b = _beta_head(s, b)
    if beta_a < 0 or beta_b < 0:
        return False
    if beta_a != a or beta_b != b:
        return _equal_under(s, beta_a, beta_b, left_names, right_names, depth, compact)
    # Congruence can match a shared binary argument before delta/iota would
    # unfold its value. This is an ordinary conversion rule, not arithmetic.
    var before_a = s.syntax.nodes[a]
    var before_b = s.syntax.nodes[b]
    if (not s.arithmetic_verified and before_a.kind == PT_APP and before_b.kind == PT_APP
        and _equal_under(s, before_a.left, before_b.left, left_names, right_names, depth, compact)
        and _equal_under(s, before_a.right, before_b.right, left_names, right_names, depth, compact)):
        return True
    if _numeral_meaning_equal(s, a, b):
        return True
    a = _whnf(s, a, compact)
    b = _whnf(s, b, compact)
    if a < 0 or b < 0:
        return False
    var x = s.syntax.nodes[a]
    var y = s.syntax.nodes[b]
    if x.kind != y.kind:
        if s.arithmetic_verified:
            var expanded_a = _expand_native_arithmetic(s, a)
            var expanded_b = _expand_native_arithmetic(s, b)
            if expanded_a < 0 or expanded_b < 0:
                return False
            if expanded_a != a or expanded_b != b:
                # Raw fallback disables compaction all the way through recursive
                # equality and value reduction: never native -> raw -> native.
                return _equal_under(s, expanded_a, expanded_b, left_names, right_names, depth, False)
        return False
    if x.kind == PT_VAR:
        var ax = -1
        var by = -1
        for j in range(depth):
            if left_names[j] == a:
                ax = j
            if right_names[j] == b:
                by = j
        return ax == by if ax >= 0 or by >= 0 else a == b
    if x.kind == PT_PI or x.kind == PT_LAM:
        if not _equal_under(s, s.syntax.nodes[x.left].left, s.syntax.nodes[y.left].left,
            left_names, right_names, depth, compact):
            return False
        # Only 0..<depth is active. Domain comparison may leave deeper slots;
        # overwrite the current pair before comparing these binders' bodies.
        if len(left_names) == depth and len(right_names) == depth:
            left_names.append(x.left)
            right_names.append(y.left)
        elif len(left_names) > depth and len(right_names) > depth:
            left_names[depth] = x.left
            right_names[depth] = y.left
        else:
            return False
        return _equal_under(s, x.right, y.right, left_names, right_names, depth + 1, compact)
    if (x.value != y.value
        or ((x.kind == PT_CONSTRUCTOR or x.kind == PT_ARITHMETIC) and x.extra != y.extra)):
        return False
    if x.left >= 0 and not _equal_under(s, x.left, y.left, left_names, right_names, depth, compact):
        return False
    if x.right >= 0 and not _equal_under(s, x.right, y.right, left_names, right_names, depth, compact):
        return False
    if (x.kind == PT_SUBST or x.kind == PT_NAT_REC) and not _equal_under(s, x.extra, y.extra, left_names, right_names, depth, compact):
        return False
    return True


def _convert[capacity: Int](mut s: _TypedState[capacity], a: Int, b: Int) -> Bool:
    if a >= 0 and a < s.syntax.count and a == b:
        return True
    var left_names = List[Int]()
    var right_names = List[Int]()
    return _equal_under(s, a, b, left_names, right_names, 0)


def _affine[capacity: Int](mut s: _TypedState[capacity], var node: Int, factor: Int,
    mut coefficients: Array[Int, capacity + 1]) -> Bool:
    if not _consume(s) or factor < 0:
        return False
    node = _whnf(s, node)
    if node < 0:
        return False
    var n = s.syntax.nodes[node]
    if n.kind == PT_ADD:
        return _affine(s, n.left, factor, coefficients) and _affine(s, n.right, factor, coefficients)
    if n.kind == PT_SCALE:
        if n.value != 0 and factor > Int.MAX // n.value:
            return False
        return _affine(s, n.left, factor * n.value, coefficients)
    var slot = node + 1
    var amount = factor
    if n.kind == PT_LITERAL:
        slot = 0
        if n.value != 0 and factor > Int.MAX // n.value:
            return False
        amount = factor * n.value
    elif n.kind == PT_BASE or n.kind == PT_DIV or n.kind == PT_MOD:
        # Canonicalize opaque Base/Div/Mod atoms by checked definitional equality.
        for j in range(node):
            if s.syntax.nodes[j].kind == n.kind and _convert(s, node, j):
                slot = j + 1
                break
    elif n.kind != PT_VAR:
        return False
    if coefficients[slot] > Int.MAX - amount:
        return False
    coefficients[slot] += amount
    return True


def _affine_equal[capacity: Int](mut s: _TypedState[capacity], a: Int, b: Int) -> Bool:
    var left_coefficients = Array[Int, capacity + 1](fill=0)
    var right_coefficients = Array[Int, capacity + 1](fill=0)
    if not _affine(s, a, 1, left_coefficients) or not _affine(s, b, 1, right_coefficients):
        return False
    for j in range(capacity + 1):
        if left_coefficients[j] != right_coefficients[j]:
            return False
    return True


def _family_arity[capacity: Int](s: _TypedState[capacity], family: Int) -> Int:
    return 0 if family == 1 else s.syntax.family_arity


def _constructor_count[capacity: Int](s: _TypedState[capacity], family: Int) -> Int:
    return s.syntax.nat_constructor_count if family == 1 else s.syntax.constructor_count


def _constructor_type[capacity: Int](s: _TypedState[capacity], family: Int, constructor: Int) -> Int:
    return (s.syntax.nat_constructor_types[constructor] if family == 1
        else s.syntax.constructor_types[constructor])


def _constructor_family(n: ProofTerm) -> Int:
    return 1 if n.extra == 1 else 0


def _family_type[capacity: Int](mut s: _TypedState[capacity], family: Int) -> Int:
    if family == 1:
        return _put(s, PT_TYPE)
    if s.family_type >= 0:
        return s.family_type
    if s.syntax.family_arity < 0 or s.syntax.family_arity > capacity:
        return -1
    var nat = _put(s, PT_NAT)
    var type = _put(s, PT_PROP)
    for _ in range(s.syntax.family_arity):
        var binder = _put(s, PT_VAR, nat)
        type = _put(s, PT_PI, binder, type)
        if binder < 0 or type < 0:
            return -1
    s.family_type = type
    return type


def _contains_family[capacity: Int](mut s: _TypedState[capacity], node: Int, family: Int) -> Bool:
    # Target-specific syntax traversal includes variable domains. Nat is an
    # ordinary index carrier for family 0, but a recursive occurrence for 1.
    if not _consume(s) or node < 0 or node >= s.syntax.count:
        return True
    var n = s.syntax.nodes[node]
    # A definition alias is transparent to strict positivity. Treating it as
    # an atom would allow a negative occurrence F -> False to be hidden in an
    # otherwise well-typed closed definition and then used as a constructor
    # field. Earlier-only admission and fuel bound this traversal.
    if n.kind == PT_DEFINED_REF:
        if n.value < 0 or n.value >= s.admitted_definitions:
            return True
        return _contains_family(s, s.syntax.definition_bodies[n.value], family)
    if n.kind == _PT_IMMUTABLE_REF:
        var body = _immutable_definition_body(s, n.value)
        if body < 0:
            return True
        return _contains_family(s, body, family)
    if n.kind == PT_NAT:
        return family == 1
    if n.kind == PT_FAMILY or n.kind == PT_ELIMINATOR:
        return n.value == family
    if n.kind == PT_CONSTRUCTOR:
        return _constructor_family(n) == family
    if n.left >= 0 and _contains_family(s, n.left, family):
        return True
    if n.right >= 0 and _contains_family(s, n.right, family):
        return True
    if (n.kind == PT_SUBST or n.kind == PT_NAT_REC) and _contains_family(s, n.extra, family):
        return True
    return False


def _saturated_family[capacity: Int](mut s: _TypedState[capacity], var node: Int, family: Int) -> Bool:
    # Only a direct saturated self occurrence is recursive. General nested
    # positivity, parameters and mutually recursive declarations are omitted.
    var arguments = 0
    var nat = _put(s, PT_NAT)
    while node >= 0 and node < s.syntax.count and s.syntax.nodes[node].kind == PT_APP:
        if not _consume(s):
            return False
        var n = s.syntax.nodes[node]
        if (_contains_family(s, n.right, family) or not _convert(s, _infer(s, n.right), nat)):
            return False
        arguments += 1
        node = n.left
    if node < 0 or node >= s.syntax.count or arguments != _family_arity(s, family):
        return False
    var head = s.syntax.nodes[node]
    return (head.kind == PT_NAT and family == 1) or (head.kind == PT_FAMILY and head.value == family)


def _check_family_definition[capacity: Int](mut s: _TypedState[capacity], family: Int) -> Bool:
    """Check one family using the shared closure/sort/positivity rulebook."""
    var sort = _put(s, PT_TYPE if family == 1 else PT_PROP)
    var prop = _put(s, PT_PROP)
    var nat = _put(s, PT_NAT)
    for c in range(_constructor_count(s, family)):
        var type = _constructor_type(s, family, c)
        if type < 0 or type >= s.syntax.count or _infer(s, type) != sort:
            return False
        comptime W = (capacity + 63) // 64
        if s.context_high[type] != 0:
            return False
        var result = type
        while s.syntax.nodes[result].kind == PT_PI:
            if not _consume(s):
                return False
            var pi = s.syntax.nodes[result]
            var domain = s.syntax.nodes[pi.left].left
            if _contains_family(s, domain, family):
                if not _saturated_family(s, domain, family):
                    return False
            elif family == 0:
                if not _convert(s, domain, nat) and _infer(s, domain) != prop:
                    return False
            else:
                # This bounded Type-family profile has direct recursive fields
                # only. It introduces no additional carrier or mutual family.
                return False
            result = pi.right
        if not _saturated_family(s, result, family):
            return False
    return True


def _canonical_nat_definition[capacity: Int](mut s: _TypedState[capacity]) -> Bool:
    """Pin the native numeral carrier to independent zero/successor signatures.

    Generic inductive admissibility is not an interpretation of Mojo integers.
    This independent operational specification enables the compact numeral and
    arithmetic bridge only after checked zero:Nat and succ:Nat->Nat match.
    """
    if s.syntax.nat_constructor_count != 2:
        return False
    var nat = _put(s, PT_NAT)
    var predecessor = _put(s, PT_VAR, nat)
    var successor = _put(s, PT_PI, predecessor, nat)
    return (nat >= 0 and predecessor >= 0 and successor >= 0
        and _convert(s, s.syntax.nat_constructor_types[0], nat)
        and _convert(s, s.syntax.nat_constructor_types[1], successor))


def _check_inductive_environment[capacity: Int](mut s: _TypedState[capacity], require_canonical_nat: Bool = True) -> Bool:
    """Admit checked Nat plus an optional Nat-indexed proposition family.

    Both families use the same constructor telescope checker. Nat is checked
    first because proposition indices and arithmetic guards use its carrier.
    No evidence constructor is available while its environment is checked.
    """
    s.environment_checked = False
    s.nat_environment_checked = False
    s.nat_canonical = False
    s.eliminator_type = -1
    s.nat_eliminator_type = -1
    if not _check_family_definition(s, 1):
        return False
    s.nat_environment_checked = True
    s.nat_canonical = _canonical_nat_definition(s)
    if require_canonical_nat and not s.nat_canonical:
        return False
    if not _check_definitions(s):
        return False
    if s.syntax.family_arity == -1:
        s.environment_checked = s.syntax.constructor_count == 0
        return s.environment_checked
    if (s.syntax.family_arity < 0 or s.syntax.family_arity > capacity
        or not _check_family_definition(s, 0)):
        return False
    s.environment_checked = True
    return True


def _family_indices[capacity: Int](mut s: _TypedState[capacity], var node: Int,
    mut indices: Array[Int, capacity], family: Int) -> Bool:
    """Extract the ordered indices of an admitted direct family application."""
    if not _saturated_family(s, node, family):
        return False
    var count = _family_arity(s, family)
    for j in range(count):
        var n = s.syntax.nodes[node]
        indices[count - j - 1] = n.right
        node = n.left
    return True


def _apply_indices[capacity: Int](mut s: _TypedState[capacity], var function: Int,
    indices: Array[Int, capacity], family: Int) -> Int:
    if function < 0:
        return -1
    for j in range(_family_arity(s, family)):
        if indices[j] < 0:
            return -1
        function = _put(s, PT_APP, function, indices[j])
        if function < 0:
            return -1
    return function


def _eliminator_type[capacity: Int](mut s: _TypedState[capacity], family: Int) -> Int:
    """Derive an evidence-dependent Prop eliminator from admitted telescopes.

    Every original constructor argument is retained. A direct recursive proof
    argument is followed by its induction hypothesis; hypotheses never become
    constructor arguments. Re-instantiating the telescope at each fresh binder
    preserves dependencies in all later domains and in the result indices.
    """
    var admitted = s.nat_environment_checked if family == NAT_FAMILY_ID else s.environment_checked
    if not admitted or _family_arity(s, family) < 0:
        return -1
    var cached = s.nat_eliminator_type if family == 1 else s.eliminator_type
    if cached >= 0:
        return cached
    var arity = _family_arity(s, family)
    var constructor_count = _constructor_count(s, family)
    var nat = _put(s, PT_NAT)
    var prop = _put(s, PT_PROP)
    var family_term = _put(s, PT_NAT) if family == 1 else _put(s, PT_FAMILY, value=family)
    var indices = Array[Int, capacity](fill=-1)
    var branch_binders = Array[Int, capacity](fill=-1)
    if nat < 0 or prop < 0 or family_term < 0:
        return -1

    # P : forall i_0 ... i_n, F(i_0,...,i_n) -> Prop.
    for j in range(arity):
        indices[j] = _put(s, PT_VAR, nat)
        if indices[j] < 0:
            return -1
    var witness = _put(s, PT_VAR, _apply_indices(s, family_term, indices, family))
    if witness < 0:
        return -1
    var motive_type = _put(s, PT_PI, witness, prop)
    for j in range(arity):
        if motive_type < 0:
            return -1
        motive_type = _put(s, PT_PI, indices[arity - j - 1], motive_type)
    var motive = _put(s, PT_VAR, motive_type)
    if witness < 0 or motive_type < 0 or motive < 0:
        return -1

    for c in range(constructor_count):
        if not _consume(s):
            return -1
        var telescope = _constructor_type(s, family, c)
        var constructor = _put(s, PT_CONSTRUCTOR, extra=1 if family == 1 else -1, value=c)
        var arguments = Array[Int, capacity](fill=-1)
        var argument_count = 0
        if constructor < 0:
            return -1
        while telescope >= 0 and s.syntax.nodes[telescope].kind == PT_PI:
            if not _consume(s) or argument_count >= capacity:
                return -1
            var pi = s.syntax.nodes[telescope]
            var domain = s.syntax.nodes[pi.left].left
            var binder = _put(s, PT_VAR, domain)
            if binder < 0:
                return -1
            arguments[argument_count] = binder
            argument_count += 1
            constructor = _put(s, PT_APP, constructor, binder)
            if constructor < 0:
                return -1
            if _contains_family(s, domain, family):
                if argument_count >= capacity or not _family_indices(s, domain, indices, family):
                    return -1
                var hypothesis_type = _put(s, PT_APP, _apply_indices(s, motive, indices, family), binder)
                var hypothesis = _put(s, PT_VAR, hypothesis_type)
                if hypothesis_type < 0 or hypothesis < 0:
                    return -1
                arguments[argument_count] = hypothesis
                argument_count += 1
            telescope = _substitute(s, pi.right, pi.left, binder)
            if telescope < 0:
                return -1
        if telescope < 0 or not _family_indices(s, telescope, indices, family):
            return -1
        var branch_type = _put(s, PT_APP, _apply_indices(s, motive, indices, family), constructor)
        for j in range(argument_count):
            if branch_type < 0:
                return -1
            branch_type = _put(s, PT_PI, arguments[argument_count - j - 1], branch_type)
        branch_binders[c] = _put(s, PT_VAR, branch_type)
        if branch_type < 0 or branch_binders[c] < 0:
            return -1

    # forall indices, h:F(indices), P(indices,h).
    for j in range(arity):
        indices[j] = _put(s, PT_VAR, nat)
        if indices[j] < 0:
            return -1
    witness = _put(s, PT_VAR, _apply_indices(s, family_term, indices, family))
    if witness < 0:
        return -1
    var result = _put(s, PT_APP, _apply_indices(s, motive, indices, family), witness)
    result = _put(s, PT_PI, witness, result)
    for j in range(arity):
        if result < 0:
            return -1
        result = _put(s, PT_PI, indices[arity - j - 1], result)
    for c in range(constructor_count):
        if result < 0:
            return -1
        result = _put(s, PT_PI, branch_binders[constructor_count - c - 1], result)
    result = _put(s, PT_PI, motive, result)
    if witness < 0 or result < 0 or _infer(s, result) != prop:
        return -1
    comptime W = (capacity + 63) // 64
    for w in range(W):
        if s.contexts[result * W + w] != 0:
            return -1
    if family == 1:
        s.nat_eliminator_type = result
    else:
        s.eliminator_type = result
    return result


def _reduce_eliminator[capacity: Int](mut s: _TypedState[capacity], node: Int,
    compact: Bool = True) -> Int:
    """Bounded constructor iota reduction; neutral majors keep their spine.

    Operand types are checked by the same Pi/application rules as ordinary
    inference, without re-inferring the potentially busy whole redex. This
    checks the major's indices as well as all motive and branch arguments.
    A recursive call is generated only for a direct recursive proof argument.
    """
    # Most applications in existing certificates have no eliminator head.
    # Find the head before allocating a capacity-sized argument buffer.
    var head = node
    while head >= 0 and head < s.syntax.count and s.syntax.nodes[head].kind == PT_APP:
        if not _consume(s):
            return -1
        head = s.syntax.nodes[head].left
    if head < 0 or head >= s.syntax.count:
        return -1
    if s.syntax.nodes[head].kind != PT_ELIMINATOR:
        return node
    var family = s.syntax.nodes[head].value
    var admitted = s.nat_environment_checked if family == NAT_FAMILY_ID else s.environment_checked
    if not admitted:
        return node
    var arity = _family_arity(s, family)
    var constructor_count = _constructor_count(s, family)
    if arity < 0:
        return -1
    var arguments = Array[Int, capacity](fill=-1)
    var argument_count = 0
    var cursor = node
    while s.syntax.nodes[cursor].kind == PT_APP:
        if not _consume(s) or argument_count >= capacity:
            return -1
        var application = s.syntax.nodes[cursor]
        arguments[argument_count] = application.right
        argument_count += 1
        cursor = application.left
    var required = 1 + constructor_count + arity + 1
    if argument_count != required:
        return node

    var major = _whnf(s, arguments[0], compact)
    if major < 0:
        return -1
    if family == NAT_FAMILY_ID and s.nat_canonical and s.numerals_verified:
        var major_node = s.syntax.nodes[major]
        if major_node.kind == PT_ADD:
            var successor = s.syntax.nodes[major_node.right]
            if successor.kind == PT_LITERAL and successor.value == 1:
                # Derive Add(q,1)'s constructor shape through the actual checked
                # Add definition and generic NatRec iota at checked literal 1.
                major = _whnf(s, _expand_native_arithmetic(s, major), False)
                if major < 0:
                    return -1
                major_node = s.syntax.nodes[major]
        if major_node.kind == PT_LITERAL or major_node.kind == PT_SUCC:
            major = _checked_numeral_view(s, major)
            if major < 0:
                return -1
    var constructor_arguments = Array[Int, capacity](fill=-1)
    var constructor_argument_count = 0
    var constructor_head = major
    while s.syntax.nodes[constructor_head].kind == PT_APP:
        if not _consume(s) or constructor_argument_count >= capacity:
            return -1
        var application = s.syntax.nodes[constructor_head]
        constructor_arguments[constructor_argument_count] = application.right
        constructor_argument_count += 1
        constructor_head = application.left
        if constructor_head < 0 or constructor_head >= s.syntax.count:
            return -1
    var constructor = s.syntax.nodes[constructor_head]
    if constructor.kind != PT_CONSTRUCTOR or _constructor_family(constructor) != family:
        return node
    if constructor.value < 0 or constructor.value >= constructor_count:
        return -1

    var result_type = _infer(s, head)
    var prefix = head
    for j in range(argument_count):
        if not _consume(s):
            return -1
        var type_head = _whnf(s, result_type, compact)
        var argument = arguments[argument_count - j - 1]
        var argument_type = _infer(s, argument)
        if type_head < 0 or argument_type < 0:
            return -1
        var pi = s.syntax.nodes[type_head]
        if pi.kind != PT_PI or not _convert(s, argument_type, s.syntax.nodes[pi.left].left):
            return -1
        result_type = _substitute(s, pi.right, pi.left, argument)
        if result_type < 0:
            return -1
        if j <= constructor_count:
            prefix = _put(s, PT_APP, prefix, argument)
            if prefix < 0:
                return -1

    # Select the declaration-indexed branch, then replay the constructor's
    # original telescope. The replay substitutes actual arguments into later
    # domains, so IH indices follow the declaration rather than user payloads.
    var branch = arguments[argument_count - constructor.value - 2]
    var telescope = _constructor_type(s, family, constructor.value)
    var consumed = 0
    var indices = Array[Int, capacity](fill=-1)
    while telescope >= 0 and s.syntax.nodes[telescope].kind == PT_PI:
        if not _consume(s) or consumed >= constructor_argument_count:
            return -1
        var pi = s.syntax.nodes[telescope]
        var domain = s.syntax.nodes[pi.left].left
        var argument = constructor_arguments[constructor_argument_count - consumed - 1]
        branch = _put(s, PT_APP, branch, argument)
        if branch < 0:
            return -1
        if _contains_family(s, domain, family):
            if not _family_indices(s, domain, indices, family):
                return -1
            var hypothesis = _put(s, PT_APP, _apply_indices(s, prefix, indices, family), argument)
            branch = _put(s, PT_APP, branch, hypothesis)
            if hypothesis < 0 or branch < 0:
                return -1
        telescope = _substitute(s, pi.right, pi.left, argument)
        if telescope < 0:
            return -1
        consumed += 1
    if consumed != constructor_argument_count or not _saturated_family(s, telescope, family):
        return -1
    var branch_type = _infer(s, branch)
    if branch_type < 0 or not _convert(s, branch_type, result_type):
        return -1
    return branch


def _infer[capacity: Int](mut s: _TypedState[capacity], node: Int) -> Int:
    if not _consume(s) or node < 0 or node >= s.syntax.count:
        return -1
    if s.types[node] == -4:
        return -1
    if s.types[node] != -3:
        return s.types[node]
    s.types[node] = -4
    var n = s.syntax.nodes[node]
    var result = -1
    var nat = _put(s, PT_NAT)
    var prop = _put(s, PT_PROP)
    var universe = _put(s, PT_TYPE)
    if nat < 0 or prop < 0 or universe < 0:
        return -1
    if n.kind == PT_TYPE:
        result = -2  # Type has no type; it cannot be used as a domain.
    elif n.kind == PT_PROP or n.kind == PT_NAT:
        result = universe
    elif n.kind == PT_LITERAL:
        var meaning = _numeral_denotation(s, n.value)
        if meaning >= 0:
            result = _infer(s, meaning)
    elif n.kind == PT_VAR:
        var sort = _infer(s, n.left)
        if sort == prop or sort == universe:
            result = n.left
            _union(s, node, n.left)
            comptime W = (capacity + 63) // 64
            s.contexts[node * W + node // 64] |= UInt64(1) << UInt64(node % 64)
            if node // 64 + 1 > s.context_high[node]:
                s.context_high[node] = node // 64 + 1
    elif n.kind == PT_PI or n.kind == PT_LAM:
        var domain = s.syntax.nodes[n.left].left
        var domain_sort = _infer(s, domain)
        var binder_type = _infer(s, n.left)
        var body_type = _infer(s, n.right)
        if (domain_sort == prop or domain_sort == universe) and binder_type >= 0 and _can_abstract(s, n.right, n.left):
            if n.kind == PT_PI:
                if body_type == prop or body_type == universe:
                    result = body_type
            elif body_type >= 0 and _infer(s, body_type) >= 0:
                # A lambda returning the universe itself would require a
                # larger universe for its Pi type, which this core omits.
                result = _put(s, PT_PI, n.left, body_type)
            if result >= 0:
                _union(s, node, n.left)
                _union(s, node, n.right)
                _remove(s, node, n.left)
    elif n.kind == PT_APP:
        var function_type = _infer(s, n.left)
        var argument_type = _infer(s, n.right)
        var head = _whnf(s, function_type)
        if head >= 0 and argument_type >= 0:
            var f = s.syntax.nodes[head]
            if f.kind == PT_PI and _convert(s, argument_type, s.syntax.nodes[f.left].left):
                result = _substitute(s, f.right, f.left, n.right)
                if result >= 0:
                    _union(s, node, n.left)
                    _union(s, node, n.right)
    elif n.kind == _PT_IMMUTABLE_REF:
        result = _immutable_definition_type(s, n.value)
    elif n.kind == PT_DEFINED_REF:
        if n.value >= 0 and n.value < s.admitted_definitions:
            result = s.syntax.definition_types[n.value]
    elif n.kind == PT_SUCC:
        var successor = _numeral_definition_body(s, 1)
        var meaning = _put(s, PT_APP, successor, n.left)
        if successor >= 0 and meaning >= 0:
            result = _infer(s, meaning)
            if result >= 0:
                _union(s, node, meaning)
    elif n.kind == PT_NAT_REC:
        result = _nat_rec_carrier(s, node)
        if result >= 0:
            _union(s, node, n.left)
            _union(s, node, n.right)
            _union(s, node, n.extra)
    elif (n.kind == PT_ADD or n.kind == PT_SCALE or n.kind == PT_BASE
        or n.kind == PT_DIV or n.kind == PT_MOD or n.kind == PT_MUL
        or n.kind == PT_LE or n.kind == PT_LT):
        # Compact operators have no separate native signature rule. Their
        # actual raw recipes are instantiated by ordinary Pi application.
        var meaning = _expand_native_arithmetic(s, node)
        if meaning >= 0:
            result = _infer(s, meaning)
            if result >= 0:
                _union(s, node, meaning)
    elif (n.kind == PT_EQ or n.kind == PT_REFL
        or n.kind == PT_AFFINE_EQ or n.kind == PT_LITERAL_LE):
        var a = _infer(s, n.left)
        var b = nat
        if n.right >= 0:
            b = _infer(s, n.right)
        var primitive = (n.kind == PT_EQ or n.kind == PT_REFL)
        if (s.nat_canonical and (primitive or s.arithmetic_verified)
            and a >= 0 and b >= 0 and _convert(s, a, nat) and _convert(s, b, nat)):
            result = nat
            if n.kind == PT_EQ:
                result = prop
            elif n.kind == PT_REFL:
                result = _put(s, PT_EQ, n.left, n.left)
            elif n.kind == PT_AFFINE_EQ:
                result = _put(s, PT_EQ, n.left, n.right) if _affine_equal(s, n.left, n.right) else -1
            elif n.kind == PT_LITERAL_LE:
                var lhs = _whnf(s, n.left)
                var rhs = _whnf(s, n.right)
                result = -1
                if lhs >= 0 and rhs >= 0:
                    var x = s.syntax.nodes[lhs]
                    var y = s.syntax.nodes[rhs]
                    if x.kind == PT_LITERAL and y.kind == PT_LITERAL and x.value <= y.value:
                        result = _put(s, PT_LE, n.left, n.right)
            if result >= 0:
                _union(s, node, n.left)
                if n.right >= 0:
                    _union(s, node, n.right)
    elif n.kind == PT_FAMILY:
        if n.value == NAT_FAMILY_ID or (n.value == PROP_FAMILY_ID and s.syntax.family_arity >= 0):
            result = _family_type(s, n.value)
    elif n.kind == PT_CONSTRUCTOR:
        var family = _constructor_family(n)
        var admitted = (s.nat_environment_checked and s.nat_canonical) if family == NAT_FAMILY_ID else s.environment_checked
        if admitted and n.value >= 0 and n.value < _constructor_count(s, family):
            result = _constructor_type(s, family, n.value)
    elif n.kind == PT_ELIMINATOR:
        if ((n.value == NAT_FAMILY_ID and s.nat_environment_checked)
            or (n.value == PROP_FAMILY_ID and s.environment_checked
                and s.syntax.family_arity >= 0)):
            result = _eliminator_type(s, n.value)
    elif n.kind == PT_ARITHMETIC:
        result = _instantiate_arithmetic_export(s, n.value, n.extra)
    elif n.kind == PT_SUBST:
        var motive_type = _infer(s, n.left)
        var binder = _put(s, PT_VAR, nat)
        var motive_expected = _put(s, PT_PI, binder, prop)
        if motive_type >= 0 and motive_expected >= 0 and _convert(s, motive_type, motive_expected):
            var equality = _whnf(s, _infer(s, n.right))
            var evidence = _infer(s, n.extra)
            if equality >= 0 and evidence >= 0:
                var e = s.syntax.nodes[equality]
                if e.kind == PT_EQ and _convert(s, evidence, _put(s, PT_APP, n.left, e.left)):
                    result = _put(s, PT_APP, n.left, e.right)
            if result >= 0:
                _union(s, node, n.left)
                _union(s, node, n.right)
                _union(s, node, n.extra)
    s.types[node] = result
    return result


def _valid_program(program: FinishingProgram) -> Bool:
    if (program.threads <= 0 or program.items <= 0 or program.stride <= 0
        or program.threads > Int.MAX // program.items
        or program.stride != program.threads * program.items or program.start != 0
        or program.thread_scale != 1 or program.item_scale != program.threads
        or program.index_bias != 0 or program.loop_guard != FINISH_LT
        or program.load_guard != FINISH_LT or program.count_limit <= 0
        or program.integer_limit <= 0):
        return False
    return True


def _make_state[capacity: Int, scratch: Int](program: FinishingProgram,
    proof: ProofTermBundle[capacity]) -> _TypedState[scratch]:
    return _TypedState[scratch](program, proof)


def check_inductive_definition[capacity: Int](program: FinishingProgram,
    definition: ProofTermBundle[capacity]) -> Bool:
    """Check generic declaration data, including noncanonical Nat definitions.

    This does not activate an alternate definition as the arithmetic carrier
    or check an unrelated evidence root.

    Syntax entries receive structural validation. Constructor types, their
    dependencies and their exact closure receive full type checking.
    """
    if capacity <= 0 or not _valid_syntax(definition) or not _valid_program(program):
        return False
    comptime C = capacity * 4 + 128
    var state = _make_state[capacity, C](program, definition)
    _enable_native_arithmetic(state)
    return _check_inductive_environment(state, False)


def check_native_definitions[capacity: Int](program: FinishingProgram,
    definition: ProofTermBundle[capacity]) -> Bool:
    """Check closed acyclic definitions without arithmetic certificates.

    The bootstrap accepts only raw syntax, generic Nat machinery and local
    earlier-definition references. It checks every unused syntax entry too.
    """
    if (capacity <= 0 or not _valid_syntax(definition)
        or not _valid_program(program)):
        return False
    for j in range(definition.count):
        var kind = definition.nodes[j].kind
        if (kind == PT_CONST or kind == PT_ARITHMETIC
            or kind == PT_AFFINE_EQ or kind == PT_LITERAL_LE
            or kind == PT_ADD or kind == PT_SCALE or kind == PT_BASE
            or kind == PT_LE or kind == PT_LT or kind == PT_MUL
            or kind == PT_DIV or kind == PT_MOD):
            return False
    comptime C = capacity * 4 + 128
    var state = _make_state[capacity, C](program, definition)
    _enable_native_numerals(state)
    if not _check_inductive_environment(state):
        return False
    for j in range(definition.count):
        var inferred = _infer(state, j)
        if inferred == -1 or inferred == -4:
            return False
    return True


def _enable_native_arithmetic[C: Int](mut state: _TypedState[C]):
    """Grant only the private judgment for the exact immutable module library."""
    _enable_native_numerals(state)
    state.arithmetic_verified = _NAT_ARITHMETIC_CERTIFIED
    state.fast_arithmetic = _NAT_ARITHMETIC_CERTIFIED


def _enable_native_numerals[C: Int](mut state: _TypedState[C]):
    """Permit checked abbreviation elaboration, not native numeral equations."""
    state.numerals_verified = _NAT_NUMERAL_CERTIFIED


def check_native_arithmetic_library[capacity: Int](
    library: NatArithmeticLibrary[capacity]) -> Bool:
    """Bootstrap-check immutable arithmetic data and all exported proof bodies.

    This function never reads the global certification result. Its state has
    no arithmetic certificates, reflection rules or native operator bridge.
    Required theorem types, the six recursive data bodies and eight raw
    abbreviation meanings are independent specifications written in this kernel.
    """
    var proof = library.bundle.copy()
    if (capacity <= 0 or not _valid_syntax(proof) or proof.definition_count < 6
        or proof.family_arity != -1 or proof.constructor_count != 0):
        return False
    for j in range(proof.count):
        var kind = proof.nodes[j].kind
        if (kind == PT_CONST or kind == PT_ARITHMETIC
            or kind == PT_AFFINE_EQ or kind == PT_LITERAL_LE
            or kind == PT_ADD or kind == PT_SCALE or kind == PT_BASE
            or kind == PT_LE or kind == PT_LT or kind == PT_MUL
            or kind == PT_DIV or kind == PT_MOD):
            return False
    comptime Initial = capacity * 3 + 128
    var initial = _native_arithmetic_attempt[capacity, Initial](library)
    if initial.accepted or not initial.resource_exhausted:
        return initial.accepted
    comptime Full = capacity * 4 + 128
    return _native_arithmetic_attempt[capacity, Full](library).accepted


def _native_arithmetic_attempt[capacity: Int, scratch: Int](
    library: NatArithmeticLibrary[capacity]) -> _ProofCheckResult:
    var proof = library.bundle.copy()
    var program = FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        FINISH_LT, FINISH_LT, 2147483647, Int.MAX)
    var state = _make_state[capacity, scratch](program, proof)
    _enable_native_numerals(state)
    if not _check_inductive_environment(state):
        return _attempt_result(state, False)
    for j in range(proof.count):
        var inferred = _infer(state, j)
        if inferred == -1 or inferred == -4:
            return _attempt_result(state, False)
    for d in range(6):
        var intended = _canonical_arithmetic_body(state, d)
        if (intended < 0 or _infer(state, intended) < 0
            or not _convert(state, state.syntax.definition_bodies[d], intended)):
            return _attempt_result(state, False)
    if not _check_operator_representations(state, library):
        return _attempt_result(state, False)
    var prop = _put(state, PT_PROP)
    comptime W = (scratch + 63) // 64
    for declaration in range(PROOF_TERM_DECLARATIONS):
        var exported = library.exports[declaration]
        if declaration == DEC_DONE_STOP or declaration == DEC_DONE_STEP:
            if exported != -1:
                return _attempt_result(state, False)
            continue
        if exported < 6 or exported >= proof.definition_count:
            return _attempt_result(state, False)
        var required = _required_arithmetic_export(state, declaration)
        if required < 0 or _infer(state, required) != prop:
            return _attempt_result(state, False)
        if state.context_high[required] != 0:
            return _attempt_result(state, False)
        if not _convert(state, state.syntax.definition_types[exported], required):
            return _attempt_result(state, False)
    return _ProofCheckResult(True, False)


@fieldwise_init
struct _ProofCheckResult(ImplicitlyCopyable):
    var accepted: Bool
    var resource_exhausted: Bool


def _attempt_result[scratch: Int](state: _TypedState[scratch], accepted: Bool) -> _ProofCheckResult:
    return _ProofCheckResult(accepted,
        not accepted and (state.syntax.count >= scratch or state.fuel <= 0))


def _match_definitions_attempt[capacity: Int, expected_capacity: Int, scratch: Int](
    program: FinishingProgram, proof: ProofTermBundle[capacity],
    expected_definition: ProofTermBundle[expected_capacity]) -> _ProofCheckResult:
    if (proof.count > scratch or proof.constructor_count > scratch
        or proof.nat_constructor_count > scratch):
        return _ProofCheckResult(False, True)
    var state = _make_state[capacity, scratch](program, proof)
    _enable_native_arithmetic(state)
    if not _check_inductive_environment(state):
        return _attempt_result(state, False)
    # Append without hash-consing: binder identities in independently supplied
    # declarations must remain distinct until alpha equality compares them.
    var offset = state.syntax.count
    if (offset > scratch - expected_definition.count
        or expected_definition.constructor_count > scratch
        or expected_definition.nat_constructor_count > scratch
        or proof.definition_count > scratch - expected_definition.definition_count):
        # A bulk append can exhaust storage before the array itself is full.
        return _ProofCheckResult(False, True)
    # Bulk insertion bypasses _put, so initialize its rows before publishing
    # the enlarged syntax. Future rows have not yet acquired any context.
    comptime ContextWords = (scratch + 63) // 64
    state.contexts.resize((offset + expected_definition.count) * ContextWords, 0)
    for j in range(expected_definition.count):
        var n = expected_definition.nodes[j]
        if n.left >= 0:
            n.left += offset
        if n.right >= 0:
            n.right += offset
        if n.kind == PT_SUBST or n.kind == PT_NAT_REC:
            n.extra += offset
        if n.kind == PT_DEFINED_REF:
            n.value += proof.definition_count
        state.syntax.nodes[offset + j] = n
    state.syntax.count += expected_definition.count
    _seed_interned(state, offset, state.syntax.count)
    for d in range(expected_definition.definition_count):
        var index = proof.definition_count + d
        state.syntax.definition_types[index] = expected_definition.definition_types[d] + offset
        state.syntax.definition_bodies[index] = expected_definition.definition_bodies[d] + offset
    state.syntax.definition_count += expected_definition.definition_count
    for c in range(expected_definition.constructor_count):
        state.syntax.constructor_types[c] = expected_definition.constructor_types[c] + offset
    for c in range(expected_definition.nat_constructor_count):
        state.syntax.nat_constructor_types[c] = expected_definition.nat_constructor_types[c] + offset
    if not _check_inductive_environment(state):
        return _attempt_result(state, False)
    for c in range(proof.nat_constructor_count):
        if not _convert(state, proof.nat_constructor_types[c], state.syntax.nat_constructor_types[c]):
            return _attempt_result(state, False)
    for c in range(proof.constructor_count):
        if not _convert(state, proof.constructor_types[c], state.syntax.constructor_types[c]):
            return _attempt_result(state, False)
    return _ProofCheckResult(True, False)


def check_inductive_definition_matches[capacity: Int, expected_capacity: Int](
    program: FinishingProgram, proof: ProofTermBundle[capacity],
    expected_definition: ProofTermBundle[expected_capacity]) -> Bool:
    """Check both environments and compare their constructor types by conversion.

    General positivity does not identify the intended operational relation.
    A runtime admission gate supplies its independent intended definition here.
    Constructor order is part of that contract, not an inferred name mapping.
    Scratch attempts differ only in storage/fuel. Retry with the original larger
    budget only after actual resource exhaustion, never after a semantic error.
    """
    if (capacity <= 0 or expected_capacity <= 0 or not _valid_program(program)
        or not _valid_syntax(proof) or not _valid_syntax(expected_definition)
        or proof.family_arity != expected_definition.family_arity
        or proof.constructor_count != expected_definition.constructor_count
        or proof.nat_constructor_count != expected_definition.nat_constructor_count):
        return False
    comptime Initial = capacity + expected_capacity + 128
    var initial = _match_definitions_attempt[capacity, expected_capacity, Initial](
        program, proof, expected_definition)
    if initial.accepted or not initial.resource_exhausted:
        return initial.accepted
    comptime Full = (capacity + expected_capacity) * 4 + 128
    return _match_definitions_attempt[capacity, expected_capacity, Full](
        program, proof, expected_definition).accepted


def _check_many_attempt[capacity: Int, root_count: Int, scratch: Int](program: FinishingProgram,
    proof: ProofTermBundle[capacity], judgments: Array[ProofTermJudgment, root_count]) -> _ProofCheckResult:
    if (proof.count > scratch or proof.constructor_count > scratch
        or proof.nat_constructor_count > scratch):
        return _ProofCheckResult(False, True)
    var state = _make_state[capacity, scratch](program, proof)
    _enable_native_arithmetic(state)
    if not _check_inductive_environment(state):
        return _attempt_result(state, False)
    for j in range(proof.count):
        var checked = _infer(state, j)
        if checked == -1 or checked == -4:
            return _attempt_result(state, False)
    var prop = _put(state, PT_PROP)
    var universe = _put(state, PT_TYPE)
    comptime W = (scratch + 63) // 64
    for j in range(root_count):
        var judgment = judgments[j]
        var expected_sort = _infer(state, judgment.expected)
        if expected_sort != prop and expected_sort != universe:
            return _attempt_result(state, False)
        if (state.context_high[judgment.root] != 0
            or state.context_high[judgment.expected] != 0):
            return _attempt_result(state, False)
        if not _convert(state, _infer(state, judgment.root), judgment.expected):
            return _attempt_result(state, False)
    return _ProofCheckResult(True, False)


def check_proof_terms_many[capacity: Int, root_count: Int](program: FinishingProgram,
    proof: ProofTermBundle[capacity], judgments: Array[ProofTermJudgment, root_count]) -> Bool:
    """Check closed roots against independently supplied closed types, in one state.

    Every declared syntax entry is validated and typed, including unused entries.
    Intermediate open terms are allowed, but every root and expected type is closed.
    Constructor environments must pass admission before evidence is checked.
    Scratch accounts for caller syntax and possible immutable-library imports,
    independently of executable geometry. A smaller first attempt uses
    identical rules; resource exhaustion alone retries the larger budget.
    Either budget can reject a valid large derivation; more storage grants no
    additional inference rule.
    """
    if (capacity <= 0 or root_count <= 0 or not _valid_syntax(proof)
        or not _valid_program(program)):
        return False
    for j in range(root_count):
        var judgment = judgments[j]
        if (judgment.root < 0 or judgment.root >= proof.count
            or judgment.expected < 0 or judgment.expected >= proof.count):
            return False
    comptime Full = capacity * 4 + 128 + _IMMUTABLE_NODE_COUNT
    comptime Initial = capacity * 2 + 256 if capacity >= 64 else capacity * 4 + 128
    var initial = _check_many_attempt[capacity, root_count, Initial](program, proof, judgments)
    comptime if Initial == Full:
        return initial.accepted
    if initial.accepted or not initial.resource_exhausted:
        return initial.accepted
    return _check_many_attempt[capacity, root_count, Full](program, proof, judgments).accepted


def check_proof_terms[capacity: Int](program: FinishingProgram,
    proof: ProofTermBundle[capacity], expected: Int) -> Bool:
    """Single-root compatibility wrapper around the shared multi-root checker."""
    var judgments = Array[ProofTermJudgment, 1](fill=ProofTermJudgment(proof.root, expected))
    return check_proof_terms_many(program, proof, judgments)
