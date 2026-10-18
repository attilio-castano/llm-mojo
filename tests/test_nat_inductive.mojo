"""Checked Nat declarations and definition-derived induction in Mojo.

The expected induction signature is written independently of the kernel's
constructor telescope generator. Compact literals remain a separately checked
arithmetic bridge: generic positive declarations need not be the intended Nat.
"""
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program

comptime NAT_INDUCTIVE_CAPACITY = 128
comptime NEGATIVE_RECURSION = 0
comptime NESTED_RECURSION = 1
comptime WRONG_RESULT = 2
comptime ILL_TYPED_CONSTRUCTOR = 3
comptime OPEN_CONSTRUCTOR = 4
comptime CONSTRUCTOR_IN_DECLARATION = 5
comptime ELIMINATOR_IN_DECLARATION = 6
comptime NEGATIVE_CONSTRUCTOR_COUNT = 7
comptime EXCESS_CONSTRUCTOR_COUNT = 8
comptime BAD_CONSTRUCTOR_REFERENCE = 9
comptime FORWARD_CONSTRUCTOR_REFERENCE = 10
comptime MISSING_CONSTRUCTOR = 11
comptime WRONG_MOTIVE = 12
comptime WRONG_BASE = 13
comptime WRONG_STEP = 14
comptime WRONG_IH = 15
comptime WRONG_TARGET = 16
comptime OMITTED_IH = 17
comptime UNKNOWN_CONSTRUCTOR = 18
comptime UNKNOWN_FAMILY = 19
comptime UNKNOWN_ELIMINATOR = 20
comptime CONSTRUCTOR_PAYLOAD = 21
comptime ZERO_SELECTOR = 22
comptime UNKNOWN_SELECTOR = 23
comptime RETIRED_INDUCTION = 24
comptime UNUSED_MALFORMED = 25
comptime OPEN_ROOT = 26
comptime BAD_REDUCIBLE_TARGET = 27
comptime WRONG_SIGNATURE_IH = 28
comptime WRONG_IOTA = 29
comptime ALTERNATE_BINARY = 30
comptime ALTERNATE_EMPTY = 31
comptime ALTERNATE_REORDERED = 32
comptime LAST_NAT_MUTANT = ALTERNATE_REORDERED


@fieldwise_init
struct NatCase(Copyable, Movable):
    var bundle: tk.ProofTermBundle[NAT_INDUCTIVE_CAPACITY]
    var expected: Int


def _node[capacity: Int](mut p: tk.ProofTermBundle[capacity], kind: Int,
    left: Int = -1, right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _app[capacity: Int](mut p: tk.ProofTermBundle[capacity], f: Int, x: Int) -> Int:
    return _node(p, tk.PT_APP, f, x)


def _var[capacity: Int](mut p: tk.ProofTermBundle[capacity], type: Int) -> Int:
    return _node(p, tk.PT_VAR, type)


def _pi[capacity: Int](mut p: tk.ProofTermBundle[capacity], binder: Int, body: Int) -> Int:
    return _node(p, tk.PT_PI, binder, body)


def _lam[capacity: Int](mut p: tk.ProofTermBundle[capacity], binder: Int, body: Int) -> Int:
    return _node(p, tk.PT_LAM, binder, body)


def _nat[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Int:
    return _node(p, tk.PT_NAT)


def _zero[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Int:
    return _node(p, tk.PT_CONSTRUCTOR, extra=1, value=0)


def _succ[capacity: Int](mut p: tk.ProofTermBundle[capacity], n: Int) -> Int:
    return _app(p, _node(p, tk.PT_CONSTRUCTOR, extra=1, value=1), n)


def _literal[capacity: Int](mut p: tk.ProofTermBundle[capacity], n: Int) -> Int:
    return _node(p, tk.PT_LITERAL, value=n)


def _eq[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_EQ, a, b)


def _induct[capacity: Int](mut p: tk.ProofTermBundle[capacity], motive: Int,
    base: Int, step: Int, target: Int) -> Int:
    var result = _node(p, tk.PT_ELIMINATOR, value=1)
    result = _app(p, result, motive)
    result = _app(p, result, base)
    result = _app(p, result, step)
    return _app(p, result, target)


def independent_nat_induction_type[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    wrong_ih: Bool = False) -> Int:
    # forall P:Nat->Prop, P(zero) ->
    # (forall n:Nat, P(n) -> P(succ(n))) -> forall n:Nat, P(n).
    var motive_index = _var(p, _nat(p))
    var motive = _var(p, _pi(p, motive_index, _node(p, tk.PT_PROP)))
    var base = _var(p, _app(p, motive, _zero(p)))
    var n = _var(p, _nat(p))
    var ih_index = n
    if wrong_ih:
        ih_index = _succ(p, n)
    var ih = _var(p, _app(p, motive, ih_index))
    var step = _var(p, _pi(p, n, _pi(p, ih, _app(p, motive, _succ(p, n)))))
    var target = _var(p, _nat(p))
    return _pi(p, motive, _pi(p, base, _pi(p, step,
        _pi(p, target, _app(p, motive, target)))))


def nat_signature_case(wrong_ih: Bool = False) -> NatCase:
    var p = tk.empty_proof_terms[NAT_INDUCTIVE_CAPACITY]()
    p.root = _node(p, tk.PT_ELIMINATOR, value=1)
    var expected = independent_nat_induction_type(p, wrong_ih)
    return NatCase(p^, expected)


def constructor_case() -> NatCase:
    var p = tk.empty_proof_terms[NAT_INDUCTIVE_CAPACITY]()
    # Both the carrier alias and constructors are checked against declaration
    # data. Numeric selector 1 is not a syntax-node reference.
    var two = _succ(p, _succ(p, _zero(p)))
    p.root = _node(p, tk.PT_REFL, two)
    var expected = _eq(p, two, _literal(p, 2))
    return NatCase(p^, expected)


def first_node_constructor_case() -> NatCase:
    var p = tk.empty_proof_terms[NAT_INDUCTIVE_CAPACITY]()
    # Reorder the declaration syntax so zero sits at node 0 and its selector 1
    # is numerically a forward reference. Selectors are tags, never references.
    p.count = 0
    var zero = _zero(p)
    var nat = _nat(p)
    var predecessor = _var(p, nat)
    var successor_type = _pi(p, predecessor, nat)
    p.nat_constructor_types[0] = nat
    p.nat_constructor_types[1] = successor_type
    p.root = zero
    var expected = _node(p, tk.PT_FAMILY, value=1)
    return NatCase(p^, expected)


def induction_case(mode: Int = -1, endpoint: Int = 37) -> NatCase:
    var p = tk.empty_proof_terms[NAT_INDUCTIVE_CAPACITY]()
    var nat = _nat(p)
    var zero = _zero(p)
    var variable = _var(p, nat)
    var proposition = _eq(p, variable, variable)
    if mode == WRONG_MOTIVE:
        proposition = nat
    var motive = _lam(p, variable, proposition)
    var base = _node(p, tk.PT_REFL, zero)
    if mode == WRONG_BASE:
        base = zero
    var n = _var(p, nat)
    var ih_index = n
    if mode == WRONG_IH:
        ih_index = _succ(p, n)
    var ih = _var(p, _eq(p, ih_index, ih_index))
    var next = _succ(p, n)
    var body = _node(p, tk.PT_REFL, next)
    if mode == WRONG_STEP:
        body = ih
    var step = body
    if mode != OMITTED_IH:
        step = _lam(p, ih, step)
    step = _lam(p, n, step)
    var target = _literal(p, endpoint)
    if mode == WRONG_TARGET:
        target = _node(p, tk.PT_PROP)
    elif mode == OPEN_ROOT:
        target = _var(p, nat)
    elif mode == BAD_REDUCIBLE_TARGET:
        target = _app(p, zero, zero)
    p.root = _induct(p, motive, base, step, target)
    var expected = _eq(p, target, target)
    if mode == UNKNOWN_CONSTRUCTOR:
        p.nodes[zero].value = 2
    elif mode == UNKNOWN_FAMILY:
        _ = _node(p, tk.PT_FAMILY, value=2)
    elif mode == UNKNOWN_ELIMINATOR:
        _ = _node(p, tk.PT_ELIMINATOR, value=2)
    elif mode == CONSTRUCTOR_PAYLOAD:
        p.nodes[zero].left = _literal(p, 0)
    elif mode == ZERO_SELECTOR:
        p.nodes[zero].extra = 0
    elif mode == UNKNOWN_SELECTOR:
        p.nodes[zero].extra = 2
    elif mode == RETIRED_INDUCTION:
        _ = _node(p, 16, motive, base, step, target)
    elif mode == UNUSED_MALFORMED:
        _ = _app(p, nat, zero)
    return NatCase(p^, expected)


def iota_case(target_kind: Int = 0, wrong_result: Bool = False) -> NatCase:
    var p = tk.empty_proof_terms[NAT_INDUCTIVE_CAPACITY]()
    var zero = _zero(p)
    var zero_literal = _literal(p, 0)
    var motive_index = _var(p, _nat(p))
    var proposition = _eq(p, zero, zero)
    var motive = _lam(p, motive_index, proposition)
    var base = _node(p, tk.PT_REFL, zero)
    var n = _var(p, _nat(p))
    var ih = _var(p, proposition)
    # The observer distinguishes two proofs of the same proposition. The
    # successor branch ignores the IH, allowing a huge literal to reduce in
    # bounded space without constructing its complete unary representation.
    var step_result = _node(p, tk.PT_AFFINE_EQ, zero_literal, zero_literal)
    var step = _lam(p, n, _lam(p, ih, step_result))
    var target = zero
    var symbolic_target = -1
    if target_kind == 1:
        target = _succ(p, zero)
    elif target_kind == 2:
        target = _literal(p, 3)
    elif target_kind == 3:
        target = _literal(p, 2147483647)
    elif target_kind == 4:
        symbolic_target = _var(p, _nat(p))
        target = _succ(p, symbolic_target)
    var folded = _induct(p, motive, base, step, target)
    var proof_argument = _var(p, proposition)
    var observer = _var(p, _pi(p, proof_argument, _nat(p)))
    var observed = _app(p, observer, folded)
    p.root = _lam(p, observer, _node(p, tk.PT_REFL, observed))
    var expected_result = base
    if target_kind != 0:
        expected_result = step_result
    if wrong_result:
        expected_result = base
    var expected = _pi(p, observer, _eq(p, observed, _app(p, observer, expected_result)))
    if symbolic_target >= 0:
        p.root = _lam(p, symbolic_target, p.root)
        expected = _pi(p, symbolic_target, expected)
    return NatCase(p^, expected)


def malformed_definition_case(mode: Int) -> NatCase:
    var result = induction_case()
    var nat = _nat(result.bundle)
    var zero = _zero(result.bundle)
    var binder = _var(result.bundle, nat)
    var constructor = nat
    if mode == NEGATIVE_RECURSION:
        # A Nat occurrence underneath a function domain is forbidden.
        var function = _var(result.bundle, _pi(result.bundle, binder, nat))
        constructor = _pi(result.bundle, function, nat)
    elif mode == NESTED_RECURSION:
        var h = _var(result.bundle, _eq(result.bundle, zero, zero))
        var function = _var(result.bundle, _pi(result.bundle, h, nat))
        constructor = _pi(result.bundle, function, nat)
    elif mode == WRONG_RESULT:
        constructor = _node(result.bundle, tk.PT_PROP)
    elif mode == ILL_TYPED_CONSTRUCTOR:
        constructor = _literal(result.bundle, 0)
    elif mode == OPEN_CONSTRUCTOR:
        var h = _var(result.bundle, _eq(result.bundle, binder, binder))
        constructor = _pi(result.bundle, h, nat)
    elif mode == CONSTRUCTOR_IN_DECLARATION:
        constructor = zero
    elif mode == ELIMINATOR_IN_DECLARATION:
        constructor = _node(result.bundle, tk.PT_ELIMINATOR, value=1)
    result.bundle.nat_constructor_types[0] = constructor
    if mode == NEGATIVE_CONSTRUCTOR_COUNT:
        result.bundle.nat_constructor_count = -1
    elif mode == EXCESS_CONSTRUCTOR_COUNT:
        result.bundle.nat_constructor_count = NAT_INDUCTIVE_CAPACITY + 1
    elif mode == BAD_CONSTRUCTOR_REFERENCE:
        result.bundle.nat_constructor_types[0] = -1
    elif mode == FORWARD_CONSTRUCTOR_REFERENCE:
        result.bundle.nat_constructor_types[0] = result.bundle.count
    elif mode == MISSING_CONSTRUCTOR:
        result.bundle.nat_constructor_count = 1
    elif mode == ALTERNATE_BINARY:
        var other = _var(result.bundle, nat)
        result.bundle.nat_constructor_types[1] = _pi(result.bundle, binder,
            _pi(result.bundle, other, nat))
        result.bundle.nat_constructor_types[0] = nat
    elif mode == ALTERNATE_EMPTY:
        # Keep declaration syntax only; an absent constructor cannot appear even
        # in dead proof syntax. The carrier itself is a closed Type expression.
        result.bundle.count = 3
        result.bundle.nat_constructor_count = 0
        result.bundle.root = nat
        result.expected = tk.pt_add(result.bundle, tk.PT_TYPE)
    elif mode == ALTERNATE_REORDERED:
        result.bundle.nat_constructor_types[0] = result.bundle.nat_constructor_types[1]
        result.bundle.nat_constructor_types[1] = nat
    return result^


def malformed_nat_case(mode: Int) -> NatCase:
    if mode <= MISSING_CONSTRUCTOR or mode >= ALTERNATE_BINARY:
        return malformed_definition_case(mode)
    if mode == WRONG_SIGNATURE_IH:
        return nat_signature_case(True)
    if mode == WRONG_IOTA:
        return iota_case(1, True)
    return induction_case(mode)


def check_small_buffer[capacity: Int]() raises:
    # A certificate buffer too small for zero/successor is a rejected witness,
    # rather than a compiler array-index failure during declaration seeding.
    comptime program = make_finishing_program(3, 5, 15)
    comptime p = tk.empty_proof_terms[capacity]()
    comptime assert p.nat_constructor_count == 0, "undersized Nat declaration unexpectedly seeded"
    comptime assert not tk.check_proof_terms(program, p, -1), "undersized Nat evidence accepted"


def check_nat_inductive() raises:
    comptime program = make_finishing_program(3, 5, 15)
    comptime signature = nat_signature_case()
    comptime assert tk.check_inductive_definition(program, signature.bundle), "Nat declaration rejected"
    comptime assert tk.check_proof_terms(program, signature.bundle, signature.expected), "Nat induction signature rejected"
    comptime constructors = constructor_case()
    comptime assert tk.check_proof_terms(program, constructors.bundle, constructors.expected), "Nat constructors rejected"
    comptime first_constructor = first_node_constructor_case()
    comptime assert first_constructor.bundle.root == 0, "Nat selector control misplaced"
    comptime assert tk.check_proof_terms(program, first_constructor.bundle, first_constructor.expected), "Nat selector mistaken for node reference"
    comptime induction = induction_case()
    comptime assert tk.check_proof_terms(program, induction.bundle, induction.expected), "Nat induction rejected"
    comptime huge = induction_case(endpoint=2147483647)
    comptime assert tk.check_proof_terms(program, huge.bundle, huge.expected), "compact Nat endpoint rejected"
    comptime assert huge.bundle.count < 64, "Nat literal expanded into unary syntax"
    comptime zero_iota = iota_case()
    comptime assert tk.check_proof_terms(program, zero_iota.bundle, zero_iota.expected), "Nat zero reduction rejected"
    comptime successor_iota = iota_case(1)
    comptime assert tk.check_proof_terms(program, successor_iota.bundle, successor_iota.expected), "Nat successor reduction rejected"
    comptime symbolic_iota = iota_case(4)
    comptime assert tk.check_proof_terms(program, symbolic_iota.bundle, symbolic_iota.expected), "Nat symbolic successor reduction rejected"
    comptime literal_iota = iota_case(2)
    comptime assert tk.check_proof_terms(program, literal_iota.bundle, literal_iota.expected), "Nat literal reduction rejected"
    comptime huge_iota = iota_case(3)
    comptime assert tk.check_proof_terms(program, huge_iota.bundle, huge_iota.expected), "compact Nat literal reduction rejected"
    comptime assert huge_iota.bundle.count < 64, "Nat reduction expanded into unary syntax"
    for mode in range(LAST_NAT_MUTANT + 1):
        var malformed = malformed_nat_case(mode)
        assert_equal(tk.check_proof_terms(program, malformed.bundle, malformed.expected), False)
        if mode >= ALTERNATE_BINARY:
            # Generic admissibility is distinct from canonical Nat admission.
            assert_equal(tk.check_inductive_definition(program, malformed.bundle), True)


def main() raises:
    check_small_buffer[0]()
    check_small_buffer[1]()
    check_small_buffer[2]()
    check_nat_inductive()
    print("Nat declaration and generated induction checked; malformed terms rejected")
