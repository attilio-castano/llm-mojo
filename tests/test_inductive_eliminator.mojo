"""Independent signatures, dependent motives and constructor computation.

The tests submit ordinary proof terms. The expected Done eliminator signature
below is written directly from the loop specification; it does not call the
kernel's telescope elaborator. An observer distinguishes proof syntax when
checking beta/iota computation, so choosing the wrong recursive IH is visible.
"""
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_proof import make_finishing_program
from llm_mojo.kernels.finishing_typed_derivation import (
    declare_finishing_done, make_finishing_done_eliminator,
    check_finishing_done_eliminator, make_finishing_done_elimination_proof,
    check_finishing_done_elimination_proof, FINISHING_DONE_ELIMINATION_CAPACITY,
)

comptime ELIMINATOR_CAPACITY = 192
comptime WRONG_MOTIVE_SORT = 0
comptime OMITTED_BRANCH = 1
comptime REORDERED_BRANCHES = 2
comptime OMITTED_IH = 3
comptime WRONG_IH_INDEX = 4
comptime WRONG_MAJOR_INDEX = 5
comptime WRONG_MAJOR_TYPE = 6
comptime OMITTED_MAJOR = 7
comptime EXTRA_MAJOR = 8
comptime OPEN_ROOT = 9
comptime UNKNOWN_ELIMINATOR = 10
comptime ELIMINATOR_OPERAND = 11
comptime UNKNOWN_FAMILY = 12
comptime WRONG_CONSTRUCTOR_COUNT = 13
comptime UNUSED_MALFORMED_ELIMINATOR = 14
comptime ELIMINATOR_IN_DECLARATION = 15
comptime UNSUPPORTED_ELIMINATOR_KIND = 16
comptime UNBOUND_DEPENDENT_VARIABLE = 17
comptime DONE_EXPECTED_MISSING_IH = 18
comptime DONE_EXPECTED_WRONG_IH_INDEX = 19
comptime DONE_EXPECTED_WRONG_RESULT_WITNESS = 20
comptime FORK_EXPECTED_WRONG_IH_WITNESS = 21
comptime IOTA_WRONG_SECOND_IH = 22
comptime MALFORMED_REDUCIBLE_MAJOR = 23
comptime NO_FAMILY_ELIMINATOR = 24
comptime WRONG_BRANCH_RESULT = 25
comptime LAST_ELIMINATOR_MUTANT = WRONG_BRANCH_RESULT


@fieldwise_init
struct EliminatorCase(Copyable, Movable):
    var bundle: tk.ProofTermBundle[ELIMINATOR_CAPACITY]
    var expected: Int


def _node[capacity: Int](mut p: tk.ProofTermBundle[capacity], kind: Int,
    left: Int = -1, right: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, value=value)


def _app[capacity: Int](mut p: tk.ProofTermBundle[capacity], f: Int, x: Int) -> Int:
    return _node(p, tk.PT_APP, f, x)


def _pi[capacity: Int](mut p: tk.ProofTermBundle[capacity], x: Int, body: Int) -> Int:
    return _node(p, tk.PT_PI, x, body)


def _lam[capacity: Int](mut p: tk.ProofTermBundle[capacity], x: Int, body: Int) -> Int:
    return _node(p, tk.PT_LAM, x, body)


def _var[capacity: Int](mut p: tk.ProofTermBundle[capacity], type: Int) -> Int:
    return _node(p, tk.PT_VAR, type)


def _nat[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Int:
    return _node(p, tk.PT_NAT)


def _literal[capacity: Int](mut p: tk.ProofTermBundle[capacity], value: Int) -> Int:
    return _node(p, tk.PT_LITERAL, value=value)


def _family[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Int:
    return _node(p, tk.PT_FAMILY, value=0)


def _ctor[capacity: Int](mut p: tk.ProofTermBundle[capacity], c: Int) -> Int:
    return _node(p, tk.PT_CONSTRUCTOR, value=c)


def _done[capacity: Int](mut p: tk.ProofTermBundle[capacity], q: Int, n: Int) -> Int:
    return _app(p, _app(p, _family(p), q), n)


def _motive_done[capacity: Int](mut p: tk.ProofTermBundle[capacity], motive: Int,
    q: Int, n: Int, evidence: Int) -> Int:
    return _app(p, _app(p, _app(p, motive, q), n), evidence)


def _stop[capacity: Int](mut p: tk.ProofTermBundle[capacity], q: Int,
    n: Int, guard: Int) -> Int:
    return _app(p, _app(p, _app(p, _ctor(p, 0), q), n), guard)


def _step[capacity: Int](mut p: tk.ProofTermBundle[capacity], q: Int,
    n: Int, guard: Int, suffix: Int) -> Int:
    return _app(p, _app(p, _app(p, _app(p, _ctor(p, 1), q), n), guard), suffix)


def independent_done_eliminator_type[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    mode: Int = -1) -> Int:
    # P : forall q,N,h:Done(q,N), Prop.
    var mq = _var(p, _nat(p))
    var mn = _var(p, _nat(p))
    var mh = _var(p, _done(p, mq, mn))
    var motive = _var(p, _pi(p, mq, _pi(p, mn,
        _pi(p, mh, _node(p, tk.PT_PROP)))))

    var q0 = _var(p, _nat(p))
    var n0 = _var(p, _nat(p))
    var g0 = _var(p, _node(p, tk.PT_LE, n0, _node(p, tk.PT_BASE, q0)))
    var stop_case = _pi(p, q0, _pi(p, n0, _pi(p, g0,
        _motive_done(p, motive, q0, n0, _stop(p, q0, n0, g0)))))
    var stop_branch = _var(p, stop_case)

    var q1 = _var(p, _nat(p))
    var n1 = _var(p, _nat(p))
    var g1 = _var(p, _node(p, tk.PT_LT, _node(p, tk.PT_BASE, q1), n1))
    var next_q = _node(p, tk.PT_ADD, q1, _literal(p, 1))
    var suffix = _var(p, _done(p, next_q, n1))
    var ih_q = next_q
    if mode == DONE_EXPECTED_WRONG_IH_INDEX:
        ih_q = q1
    var ih = _var(p, _motive_done(p, motive, ih_q, n1, suffix))
    var witness = _step(p, q1, n1, g1, suffix)
    if mode == DONE_EXPECTED_WRONG_RESULT_WITNESS:
        witness = suffix
    var step_case = _motive_done(p, motive, q1, n1, witness)
    if mode != DONE_EXPECTED_MISSING_IH:
        step_case = _pi(p, ih, step_case)
    step_case = _pi(p, q1, _pi(p, n1, _pi(p, g1, _pi(p, suffix, step_case))))
    var step_branch = _var(p, step_case)

    var q = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var h = _var(p, _done(p, q, n))
    var result = _pi(p, q, _pi(p, n, _pi(p, h,
        _motive_done(p, motive, q, n, h))))
    return _pi(p, motive, _pi(p, stop_branch, _pi(p, step_branch, result)))


def done_signature_case(mode: Int = -1) -> EliminatorCase:
    var p = tk.empty_proof_terms[ELIMINATOR_CAPACITY]()
    _ = declare_finishing_done(p)
    p.root = _node(p, tk.PT_ELIMINATOR, value=0)
    var expected = independent_done_eliminator_type(p, mode)
    return EliminatorCase(p^, expected)


def indexed_fold_case(mode: Int = -1, shift: Int = 2) -> EliminatorCase:
    # F(0); forall n,F(n),F(n+2). The fold rebuilds its witness using the IH.
    var p = tk.empty_proof_terms[ELIMINATOR_CAPACITY]()
    _ = tk.pt_declare_family(p, 1)
    var zero = _literal(p, 0)
    var two = _literal(p, shift)
    var family = _family(p)
    _ = tk.pt_declare_constructor(p, _app(p, family, zero))
    var n = _var(p, _nat(p))
    var h = _var(p, _app(p, family, n))
    _ = tk.pt_declare_constructor(p, _pi(p, n, _pi(p, h,
        _app(p, family, _node(p, tk.PT_ADD, n, two)))))

    var mn = _var(p, _nat(p))
    var mh = _var(p, _app(p, family, mn))
    var motive_body = _app(p, family, mn)
    if mode == WRONG_MOTIVE_SORT:
        # Nat is well-sorted in Type, but this family only eliminates into Prop.
        motive_body = _node(p, tk.PT_NAT)
    var motive = _lam(p, mn, _lam(p, mh, motive_body))
    var bn = _var(p, _nat(p))
    var bh = _var(p, _app(p, family, bn))
    var ih_n = bn
    if mode == WRONG_IH_INDEX:
        ih_n = _node(p, tk.PT_ADD, bn, _literal(p, 1))
    var ih = _var(p, _app(p, family, ih_n))
    var branch = _app(p, _app(p, _ctor(p, 1), bn), ih)
    if mode == WRONG_BRANCH_RESULT:
        branch = bh
    if mode == OMITTED_IH:
        branch = _app(p, _app(p, _ctor(p, 1), bn), bh)
    else:
        branch = _lam(p, ih, branch)
    branch = _lam(p, bn, _lam(p, bh, branch))
    var leaf = _ctor(p, 0)
    var major = _app(p, _app(p, _ctor(p, 1), zero), leaf)
    var elim = _node(p, tk.PT_ELIMINATOR, value=0)
    var root = _app(p, elim, motive)
    if mode == REORDERED_BRANCHES:
        root = _app(p, _app(p, root, branch), leaf)
    else:
        root = _app(p, root, leaf)
        if mode != OMITTED_BRANCH:
            root = _app(p, root, branch)
    var target = two
    if mode == WRONG_MAJOR_INDEX:
        target = zero
    root = _app(p, root, target)
    if mode == WRONG_MAJOR_TYPE:
        major = zero
    if mode != OMITTED_MAJOR:
        root = _app(p, root, major)
    if mode == EXTRA_MAJOR:
        root = _app(p, root, major)
    if mode == OPEN_ROOT:
        root = _var(p, _app(p, family, two))
    if mode == UNBOUND_DEPENDENT_VARIABLE:
        var outside_n = _var(p, _nat(p))
        var outside_h = _var(p, _app(p, family, outside_n))
        root = _lam(p, outside_n, outside_h)
    var expected = _app(p, family, two)
    p.root = root
    if mode == UNKNOWN_ELIMINATOR:
        p.nodes[elim].value = 2
    elif mode == ELIMINATOR_OPERAND:
        p.nodes[elim].left = zero
    elif mode == UNKNOWN_FAMILY:
        p.nodes[family].value = 2
    elif mode == WRONG_CONSTRUCTOR_COUNT:
        p.constructor_count = 1
    elif mode == UNUSED_MALFORMED_ELIMINATOR:
        _ = _node(p, tk.PT_ELIMINATOR, value=2)
    elif mode == ELIMINATOR_IN_DECLARATION:
        p.constructor_types[0] = _app(p, elim, motive)
    elif mode == UNSUPPORTED_ELIMINATOR_KIND:
        p.nodes[elim].kind = tk.PT_ELIMINATOR + 1
    elif mode == MALFORMED_REDUCIBLE_MAJOR:
        # A saturated reducer must not excuse the ill-typed extra ctor argument.
        var malformed = _app(p, major, zero)
        p.root = _app(p, p.nodes[root].left, malformed)
    elif mode == NO_FAMILY_ELIMINATOR:
        p.family_arity = -1
        p.constructor_count = 0
    return EliminatorCase(p^, expected)


def nullary_signature_case(empty: Bool = False, mode: Int = -1) -> EliminatorCase:
    # F has two leaves and a constructor with TWO direct recursive premises.
    var p = tk.empty_proof_terms[ELIMINATOR_CAPACITY]()
    _ = tk.pt_declare_family(p, 0)
    var family = _family(p)
    if not empty:
        _ = tk.pt_declare_constructor(p, family)
        _ = tk.pt_declare_constructor(p, family)
        var a = _var(p, family)
        var b = _var(p, family)
        _ = tk.pt_declare_constructor(p, _pi(p, a, _pi(p, b, family)))
    p.root = _node(p, tk.PT_ELIMINATOR, value=0)
    var mh = _var(p, family)
    var motive = _var(p, _pi(p, mh, _node(p, tk.PT_PROP)))
    var h = _var(p, family)
    var result = _pi(p, h, _app(p, motive, h))
    if not empty:
        var first = _var(p, _app(p, motive, _ctor(p, 0)))
        var second = _var(p, _app(p, motive, _ctor(p, 1)))
        var a = _var(p, family)
        var ia = _var(p, _app(p, motive, a))
        var b = _var(p, family)
        var witness = b
        if mode == FORK_EXPECTED_WRONG_IH_WITNESS:
            witness = a
        var ib = _var(p, _app(p, motive, witness))
        var fork = _app(p, _app(p, _ctor(p, 2), a), b)
        var branch = _var(p, _pi(p, a, _pi(p, ia, _pi(p, b,
            _pi(p, ib, _app(p, motive, fork))))))
        result = _pi(p, first, _pi(p, second, _pi(p, branch, result)))
    var expected = _pi(p, motive, result)
    return EliminatorCase(p^, expected)


def triple_signature_case() -> EliminatorCase:
    var p = tk.empty_proof_terms[ELIMINATOR_CAPACITY]()
    _ = tk.pt_declare_family(p, 3)
    var family = _family(p)
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var c = _var(p, _nat(p))
    var result = _app(p, _app(p, _app(p, family, a), b), c)
    _ = tk.pt_declare_constructor(p, _pi(p, a, _pi(p, b, _pi(p, c, result))))
    var ma = _var(p, _nat(p))
    var mb = _var(p, _nat(p))
    var mc = _var(p, _nat(p))
    var mh = _var(p, _app(p, _app(p, _app(p, family, ma), mb), mc))
    var motive = _var(p, _pi(p, ma, _pi(p, mb, _pi(p, mc,
        _pi(p, mh, _node(p, tk.PT_PROP))))))
    var ca = _var(p, _nat(p))
    var cb = _var(p, _nat(p))
    var cc = _var(p, _nat(p))
    var witness = _app(p, _app(p, _app(p, _ctor(p, 0), ca), cb), cc)
    var branch_result = _app(p, _app(p, _app(p, _app(p, motive, ca), cb), cc), witness)
    var branch = _var(p, _pi(p, ca, _pi(p, cb, _pi(p, cc, branch_result))))
    var qa = _var(p, _nat(p))
    var qb = _var(p, _nat(p))
    var qc = _var(p, _nat(p))
    var major = _var(p, _app(p, _app(p, _app(p, family, qa), qb), qc))
    var conclusion = _app(p, _app(p, _app(p, _app(p, motive, qa), qb), qc), major)
    var expected = _pi(p, motive, _pi(p, branch, _pi(p, qa,
        _pi(p, qb, _pi(p, qc, _pi(p, major, conclusion))))))
    p.root = _node(p, tk.PT_ELIMINATOR, value=0)
    return EliminatorCase(p^, expected)


def fork_iota_case(mode: Int = -1, trailing_argument: Bool = False) -> EliminatorCase:
    # The branch returns its SECOND IH. Different leaf proof syntax makes both
    # recursion and the recursive argument/IH positions observable.
    var p = tk.empty_proof_terms[ELIMINATOR_CAPACITY]()
    _ = tk.pt_declare_family(p, 0)
    var family = _family(p)
    _ = tk.pt_declare_constructor(p, family)
    _ = tk.pt_declare_constructor(p, family)
    var ca = _var(p, family)
    var cb = _var(p, family)
    _ = tk.pt_declare_constructor(p, _pi(p, ca, _pi(p, cb, family)))
    var zero = _literal(p, 0)
    var proposition = _node(p, tk.PT_EQ, zero, zero)
    var computation_argument = _var(p, _nat(p))
    if trailing_argument:
        proposition = _pi(p, computation_argument,
            _node(p, tk.PT_EQ, computation_argument, computation_argument))
    var mh = _var(p, family)
    var motive = _lam(p, mh, proposition)
    var leaf_a = _node(p, tk.PT_REFL, zero)
    var leaf_b = _node(p, tk.PT_AFFINE_EQ, zero, zero)
    if trailing_argument:
        leaf_a = _lam(p, computation_argument,
            _node(p, tk.PT_REFL, computation_argument))
        leaf_b = _lam(p, computation_argument,
            _node(p, tk.PT_AFFINE_EQ, computation_argument, computation_argument))
    var a = _var(p, family)
    var ia = _var(p, proposition)
    var b = _var(p, family)
    var ib = _var(p, proposition)
    var chosen = ib
    if mode == IOTA_WRONG_SECOND_IH:
        chosen = ia
    var fork_branch = _lam(p, a, _lam(p, ia, _lam(p, b, _lam(p, ib, chosen))))
    var major = _app(p, _app(p, _ctor(p, 2), _ctor(p, 0)), _ctor(p, 1))
    var folded = _app(p, _app(p, _app(p, _app(p,
        _node(p, tk.PT_ELIMINATOR, value=0), motive), leaf_a), leaf_b), fork_branch)
    folded = _app(p, folded, major)
    var expected_result = leaf_b
    if trailing_argument:
        folded = _app(p, folded, zero)
        expected_result = _app(p, leaf_b, zero)
        proposition = _node(p, tk.PT_EQ, zero, zero)
    var argument = _var(p, proposition)
    var observer = _var(p, _pi(p, argument, _nat(p)))
    var observed = _app(p, observer, folded)
    p.root = _lam(p, observer, _node(p, tk.PT_REFL, observed))
    var expected = _pi(p, observer, _node(p, tk.PT_EQ, observed,
        _app(p, observer, expected_result)))
    return EliminatorCase(p^, expected)


def done_dependent_motive_case() -> EliminatorCase:
    var p = tk.empty_proof_terms[ELIMINATOR_CAPACITY]()
    _ = declare_finishing_done(p)
    # The motive really mentions h through a free observer, closed outside the
    # whole theorem. Constructor arguments must be kept in generated results.
    var oq = _var(p, _nat(p))
    var on = _var(p, _nat(p))
    var oh = _var(p, _done(p, oq, on))
    var observer = _var(p, _pi(p, oq, _pi(p, on, _pi(p, oh, _nat(p)))))
    var mq = _var(p, _nat(p))
    var mn = _var(p, _nat(p))
    var mh = _var(p, _done(p, mq, mn))
    var observed_motive = _motive_done(p, observer, mq, mn, mh)
    var motive = _lam(p, mq, _lam(p, mn, _lam(p, mh,
        _node(p, tk.PT_EQ, observed_motive, observed_motive))))
    var sq = _var(p, _nat(p))
    var sn = _var(p, _nat(p))
    var sg = _var(p, _node(p, tk.PT_LE, sn, _node(p, tk.PT_BASE, sq)))
    var stop_observed = _motive_done(p, observer, sq, sn, _stop(p, sq, sn, sg))
    var stop_branch = _lam(p, sq, _lam(p, sn, _lam(p, sg,
        _node(p, tk.PT_REFL, stop_observed))))
    var tq = _var(p, _nat(p))
    var tn = _var(p, _nat(p))
    var tg = _var(p, _node(p, tk.PT_LT, _node(p, tk.PT_BASE, tq), tn))
    var next_q = _node(p, tk.PT_ADD, tq, _literal(p, 1))
    var suffix = _var(p, _done(p, next_q, tn))
    var ih = _var(p, _motive_done(p, motive, next_q, tn, suffix))
    var step_observed = _motive_done(p, observer, tq, tn, _step(p, tq, tn, tg, suffix))
    var step_branch = _lam(p, tq, _lam(p, tn, _lam(p, tg, _lam(p, suffix,
        _lam(p, ih, _node(p, tk.PT_REFL, step_observed))))))
    var q = _var(p, _nat(p))
    var n = _var(p, _nat(p))
    var h = _var(p, _done(p, q, n))
    var eliminated = _app(p, _app(p, _app(p, _node(p, tk.PT_ELIMINATOR, value=0),
        motive), stop_branch), step_branch)
    eliminated = _app(p, _app(p, _app(p, eliminated, q), n), h)
    p.root = _lam(p, observer, _lam(p, q, _lam(p, n, _lam(p, h, eliminated))))
    var observed = _motive_done(p, observer, q, n, h)
    var expected = _pi(p, observer, _pi(p, q, _pi(p, n, _pi(p, h,
        _node(p, tk.PT_EQ, observed, observed)))))
    return EliminatorCase(p^, expected)


def malformed_eliminator_case(mode: Int) -> EliminatorCase:
    if (mode == DONE_EXPECTED_MISSING_IH or mode == DONE_EXPECTED_WRONG_IH_INDEX
        or mode == DONE_EXPECTED_WRONG_RESULT_WITNESS):
        return done_signature_case(mode)
    if mode == FORK_EXPECTED_WRONG_IH_WITNESS:
        return nullary_signature_case(mode=mode)
    if mode == IOTA_WRONG_SECOND_IH:
        return fork_iota_case(mode)
    return indexed_fold_case(mode)


def malformed_monotonicity_proof(mode: Int) -> tk.ProofTermBundle[FINISHING_DONE_ELIMINATION_CAPACITY]:
    var program = make_finishing_program(3, 5, 15)
    var p = make_finishing_done_elimination_proof(program)
    if mode == 0:
        # Replace the smaller-count suffix derived from IH with the original
        # suffix at N. These are different indexed propositions for symbolic M.
        var suffix = -1
        for j in range(p.count):
            if p.nodes[j].kind == tk.PT_VAR:
                var type = p.nodes[j].left
                if p.nodes[type].kind == tk.PT_APP:
                    var indexed_family = p.nodes[type].left
                    if p.nodes[indexed_family].kind == tk.PT_APP:
                        var index = p.nodes[indexed_family].right
                        if (p.nodes[p.nodes[indexed_family].left].kind == tk.PT_FAMILY
                            and p.nodes[index].kind == tk.PT_ADD):
                            suffix = j
        if suffix >= 0:
            for j in range(p.count):
                if p.nodes[j].kind == tk.PT_APP:
                    var head = p.nodes[j].left
                    var arguments = 0
                    while p.nodes[head].kind == tk.PT_APP:
                        arguments += 1
                        head = p.nodes[head].left
                    if (arguments == 3 and p.nodes[head].kind == tk.PT_CONSTRUCTOR
                        and p.nodes[head].value == 1):
                        p.nodes[j].right = suffix
                        break
    else:
        # A generic admissible unconditional constructor changes the definition;
        # the separate driver-definition pin must reject it.
        var q = _var(p, _nat(p))
        var n = _var(p, _nat(p))
        p.constructor_types[0] = _pi(p, q, _pi(p, n, _done(p, q, n)))
    return p^


def check_eliminators() raises:
    comptime program = make_finishing_program(3, 5, 15)
    comptime done_case = done_signature_case()
    comptime assert tk.check_proof_terms(program, done_case.bundle, done_case.expected), "eliminator admission rejected"
    comptime indexed = indexed_fold_case()
    comptime assert tk.check_proof_terms(program, indexed.bundle, indexed.expected), "eliminator admission rejected"
    comptime changed_definition = indexed_fold_case(shift=3)
    comptime assert tk.check_proof_terms(program, changed_definition.bundle, changed_definition.expected), "eliminator admission rejected"
    comptime fork = nullary_signature_case()
    comptime assert tk.check_proof_terms(program, fork.bundle, fork.expected), "eliminator admission rejected"
    comptime empty = nullary_signature_case(empty=True)
    comptime assert tk.check_proof_terms(program, empty.bundle, empty.expected), "eliminator admission rejected"
    comptime triple = triple_signature_case()
    comptime assert tk.check_proof_terms(program, triple.bundle, triple.expected), "eliminator admission rejected"
    comptime computation = fork_iota_case()
    comptime assert tk.check_proof_terms(program, computation.bundle, computation.expected), "eliminator admission rejected"
    comptime trailing = fork_iota_case(trailing_argument=True)
    comptime assert tk.check_proof_terms(program, trailing.bundle, trailing.expected), "eliminator admission rejected"
    comptime dependent = done_dependent_motive_case()
    comptime assert tk.check_proof_terms(program, dependent.bundle, dependent.expected), "eliminator admission rejected"
    comptime done_eliminator = make_finishing_done_eliminator(program)
    comptime assert check_finishing_done_eliminator(program, done_eliminator), "eliminator admission rejected"
    comptime monotone = make_finishing_done_elimination_proof(program)
    comptime assert check_finishing_done_elimination_proof(program, monotone), "eliminator admission rejected"
    comptime other_program = make_finishing_program(8, 3, 24)
    comptime other_eliminator = make_finishing_done_eliminator(other_program)
    comptime assert check_finishing_done_eliminator(other_program, other_eliminator), "eliminator admission rejected"
    comptime other_monotone = make_finishing_done_elimination_proof(other_program)
    comptime assert check_finishing_done_elimination_proof(other_program, other_monotone), "eliminator admission rejected"
    comptime metal_program = make_finishing_program(32, 2, 64)
    comptime metal_eliminator = make_finishing_done_eliminator(metal_program)
    comptime assert check_finishing_done_eliminator(metal_program, metal_eliminator), "eliminator admission rejected"
    comptime metal_monotone = make_finishing_done_elimination_proof(metal_program)
    comptime assert check_finishing_done_elimination_proof(metal_program, metal_monotone), "eliminator admission rejected"
    comptime wider_program = make_finishing_program(64, 3, 192)
    comptime wider_eliminator = make_finishing_done_eliminator(wider_program)
    comptime assert check_finishing_done_eliminator(wider_program, wider_eliminator), "eliminator admission rejected"
    comptime wider_monotone = make_finishing_done_elimination_proof(wider_program)
    comptime assert check_finishing_done_elimination_proof(wider_program, wider_monotone), "eliminator admission rejected"
    for mode in range(LAST_ELIMINATOR_MUTANT + 1):
        var malformed = malformed_eliminator_case(mode)
        assert_equal(tk.check_proof_terms(program, malformed.bundle, malformed.expected), False)
    for mode in range(2):
        var malformed_corollary = malformed_monotonicity_proof(mode)
        assert_equal(check_finishing_done_elimination_proof(program, malformed_corollary), False)


def main() raises:
    check_eliminators()
