"""Independent coverage contracts and focused malformed proof terms.

Canonical quotient/remainder witnesses remain symbolic. The independent
required types below quantify over counts and indices; the geometry matrix
does not enumerate the execution or natural-number domain.
"""
from std.collections import Array
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
import llm_mojo.kernels.finishing_typed_coverage as coverage
from llm_mojo.kernels.finishing_proof import FinishingProgram, make_finishing_program

comptime CORE_CAPACITY = 128
comptime WRONG_VISIT_BASE = 0
comptime WRONG_VISIT_STEP = 1
comptime WRONG_VISIT_IH = 2
comptime WRONG_INVERSE_WITNESS = 3
comptime WRONG_THREAD_BOUND = 4
comptime WRONG_ITEM_BOUND = 5
comptime WRONG_INDEX = 6
comptime WEAK_ROUND_UNIQUENESS = 7
comptime SWAPPED_UNIQUENESS = 8
comptime MISSING_ITEM_UNIQUENESS = 9
comptime WEAK_HEADROOM = 10
comptime SWAPPED_HEADROOM = 11
comptime MISSING_HEADROOM = 12
comptime FALSE_LITERAL_ORDER = 13
comptime SYMBOLIC_LITERAL_ORDER = 14
comptime UNCONDITIONAL_VISIT_ZERO = 15
comptime WRONG_VISIT_GUARD = 16
comptime WRONG_VISIT_COUNT = 17
comptime WRONG_VISIT_ROUND = 18
comptime RETIRED_NAT_INDUCTION = 19
comptime LEGACY_NAT_INDUCTION = 20
comptime DIV_ZERO = 21
comptime DIV_NEGATIVE = 22
comptime MOD_ZERO = 23
comptime MOD_NEGATIVE = 24
comptime ZERO_DIVISION_PARAMETER = 25
comptime NEGATIVE_SCALE_PARAMETER = 26
comptime NONPARAMETER_EXTRA = 27
comptime UNKNOWN_KIND = 28
comptime UNKNOWN_DECLARATION = 29
comptime UNUSED_MALFORMED = 30
comptime OPEN_ROOT = 31
comptime INSUFFICIENT_MACHINE_HEADROOM = 32
comptime LAST_TYPED_COVERAGE_MUTANT = INSUFFICIENT_MACHINE_HEADROOM


@fieldwise_init
struct CoreCase(Copyable, Movable):
    var bundle: tk.ProofTermBundle[CORE_CAPACITY]
    var expected: Int


def _node[capacity: Int](mut p: tk.ProofTermBundle[capacity], kind: Int,
    left: Int = -1, right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _nat[capacity: Int](mut p: tk.ProofTermBundle[capacity]) -> Int:
    return _node(p, tk.PT_NAT)


def _number[capacity: Int](mut p: tk.ProofTermBundle[capacity], value: Int) -> Int:
    return _node(p, tk.PT_LITERAL, value=value)


def _var[capacity: Int](mut p: tk.ProofTermBundle[capacity], type: Int) -> Int:
    return _node(p, tk.PT_VAR, type)


def _app[capacity: Int](mut p: tk.ProofTermBundle[capacity], f: Int, x: Int) -> Int:
    return _node(p, tk.PT_APP, f, x)


def _pi[capacity: Int](mut p: tk.ProofTermBundle[capacity], binder: Int, body: Int) -> Int:
    return _node(p, tk.PT_PI, binder, body)


def _lam[capacity: Int](mut p: tk.ProofTermBundle[capacity], binder: Int, body: Int) -> Int:
    return _node(p, tk.PT_LAM, binder, body)


def _div[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, divisor: Int) -> Int:
    return _node(p, tk.PT_DIV, a, value=divisor)


def _mod[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, divisor: Int) -> Int:
    return _node(p, tk.PT_MOD, a, value=divisor)


def _scale[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, factor: Int) -> Int:
    return _node(p, tk.PT_SCALE, a, value=factor)


def _add[capacity: Int](mut p: tk.ProofTermBundle[capacity], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_ADD, a, b)


def _visited[capacity: Int](mut p: tk.ProofTermBundle[capacity], q: Int, n: Int) -> Int:
    return _app(p, _app(p, _node(p, tk.PT_FAMILY, value=0), q), n)


def _index[capacity: Int](mut p: tk.ProofTermBundle[capacity], program: FinishingProgram,
    q: Int, t: Int, i: Int) -> Int:
    return _add(p, _add(p, _node(p, tk.PT_BASE, q), t), _scale(p, i, program.threads))


def independent_coverage_types[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Array[Int, 10]:
    """Required types written without the producer's expected-type helpers."""
    var types = Array[Int, 10](fill=-1)
    var nat = _nat(p)
    var q = _var(p, nat)
    types[0] = _pi(p, q, _node(p, tk.PT_EQ, _node(p, tk.PT_BASE, q),
        _scale(p, q, program.stride)))

    q = _var(p, nat)
    var n = _var(p, nat)
    var guard = _var(p, _node(p, tk.PT_LT, _scale(p, q, program.stride), n))
    types[1] = _pi(p, q, _pi(p, n, _pi(p, guard, _visited(p, q, n))))

    n = _var(p, nat)
    var position = _var(p, nat)
    guard = _var(p, _node(p, tk.PT_LT, position, n))
    types[2] = _pi(p, n, _pi(p, position, _pi(p, guard,
        _visited(p, _div(p, position, program.stride), n))))

    position = _var(p, nat)
    var remainder = _mod(p, position, program.stride)
    types[3] = _pi(p, position, _node(p, tk.PT_LT,
        _mod(p, remainder, program.threads), _number(p, program.threads)))

    position = _var(p, nat)
    remainder = _mod(p, position, program.stride)
    types[4] = _pi(p, position, _node(p, tk.PT_LT,
        _div(p, remainder, program.threads), _number(p, program.items)))

    position = _var(p, nat)
    remainder = _mod(p, position, program.stride)
    types[5] = _pi(p, position, _node(p, tk.PT_EQ,
        _index(p, program, _div(p, position, program.stride),
            _mod(p, remainder, program.threads), _div(p, remainder, program.threads)),
        position))

    for coordinate in range(3):
        position = _var(p, nat)
        q = _var(p, nat)
        var t = _var(p, nat)
        var i = _var(p, nat)
        var thread_guard = _var(p, _node(p, tk.PT_LT, t, _number(p, program.threads)))
        var item_guard = _var(p, _node(p, tk.PT_LT, i, _number(p, program.items)))
        var equation = _var(p, _node(p, tk.PT_EQ, _index(p, program, q, t, i), position))
        var candidate = q
        var inverse = _div(p, position, program.stride)
        remainder = _mod(p, position, program.stride)
        if coordinate == 1:
            candidate = t
            inverse = _mod(p, remainder, program.threads)
        elif coordinate == 2:
            candidate = i
            inverse = _div(p, remainder, program.threads)
        var body = _node(p, tk.PT_EQ, candidate, inverse)
        types[6 + coordinate] = _pi(p, position, _pi(p, q, _pi(p, t, _pi(p, i,
            _pi(p, thread_guard, _pi(p, item_guard, _pi(p, equation, body)))))))

    n = _var(p, nat)
    guard = _var(p, _node(p, tk.PT_LE, n, _number(p, program.count_limit)))
    types[9] = _pi(p, n, _pi(p, guard, _node(p, tk.PT_LE,
        _add(p, n, _number(p, program.stride)), _number(p, program.integer_limit))))
    return types^


def independent_coverage_accepts[capacity: Int](program: FinishingProgram,
    proofs: coverage.FinishingTypedCoverageProofs[capacity]) -> Bool:
    var p = proofs.bundle.copy()
    var types = independent_coverage_types(p, program)
    var checks = Array[tk.ProofTermJudgment, 9](fill=tk.ProofTermJudgment(-1, -1))
    checks[0] = tk.ProofTermJudgment(proofs.base_root, types[0])
    checks[1] = tk.ProofTermJudgment(proofs.visitation_root, types[1])
    checks[2] = tk.ProofTermJudgment(proofs.owner_visitation_root, types[2])
    checks[3] = tk.ProofTermJudgment(proofs.thread_bound_root, types[3])
    checks[4] = tk.ProofTermJudgment(proofs.item_bound_root, types[4])
    checks[5] = tk.ProofTermJudgment(proofs.index_root, types[5])
    checks[6] = tk.ProofTermJudgment(proofs.round_unique_root, types[6])
    checks[7] = tk.ProofTermJudgment(proofs.thread_unique_root, types[7])
    checks[8] = tk.ProofTermJudgment(proofs.item_unique_root, types[8])
    # Headroom is a separate ordinary Nat theorem with independently bound
    # cap, stride and limit. It cannot use the other roots' affine judgments.
    return (tk.check_proof_terms_many(program, p, checks)
        and proofs.headroom_root == proofs.headroom.proof
        and tk.check_native_headroom_request(proofs.headroom,
            program.count_limit, program.stride, program.integer_limit))


def _weak_uniqueness[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    # A closed, well-typed theorem q=q under the full candidate telescope is
    # insufficient to establish that q is the canonical owner round.
    var nat = _nat(p)
    var position = _var(p, nat)
    var q = _var(p, nat)
    var t = _var(p, nat)
    var i = _var(p, nat)
    var thread_guard = _var(p, _node(p, tk.PT_LT, t, _number(p, program.threads)))
    var item_guard = _var(p, _node(p, tk.PT_LT, i, _number(p, program.items)))
    var equation = _var(p, _node(p, tk.PT_EQ, _index(p, program, q, t, i), position))
    return _lam(p, position, _lam(p, q, _lam(p, t, _lam(p, i,
        _lam(p, thread_guard, _lam(p, item_guard,
            _lam(p, equation, _node(p, tk.PT_REFL, q))))))))


def _weak_headroom[capacity: Int](mut p: tk.ProofTermBundle[capacity],
    program: FinishingProgram) -> Int:
    var n = _var(p, _nat(p))
    # A well-typed raw theorem with an insufficient conclusion.
    from llm_mojo.kernels.nat_literal_arithmetic import _denotation
    var le = _node(p, tk.PT_DEFINED_REF, value=5)
    var cap = _denotation(p, program.count_limit)
    var bound = _var(p, _app(p, _app(p, le, n), cap))
    return _lam(p, n, _lam(p, bound, bound))


def _is_nat_induction[capacity: Int](p: tk.ProofTermBundle[capacity], node: Int) -> Bool:
    var head = node
    for _ in range(4):
        if head < 0 or head >= p.count or p.nodes[head].kind != tk.PT_APP:
            return False
        head = p.nodes[head].left
    return (head >= 0 and p.nodes[head].kind == tk.PT_ELIMINATOR
        and p.nodes[head].value == tk.NAT_FAMILY_ID)


def coverage_nat_induction_count[capacity: Int](proofs: coverage.FinishingTypedCoverageProofs[capacity]) -> Int:
    var count = 0
    for node in range(proofs.bundle.count):
        if _is_nat_induction(proofs.bundle, node):
            count += 1
    return count


def core_case(mode: Int = -1) -> CoreCase:
    var p = tk.empty_proof_terms[CORE_CAPACITY]()
    var zero = _number(p, 0)
    p.root = _node(p, tk.PT_REFL, zero)
    var expected = _node(p, tk.PT_EQ, zero, zero)
    if mode == FALSE_LITERAL_ORDER:
        var two = _number(p, 2)
        var one = _number(p, 1)
        p.root = _node(p, tk.PT_LITERAL_LE, two, one)
        expected = _node(p, tk.PT_LE, two, one)
    elif mode == SYMBOLIC_LITERAL_ORDER:
        var n = _var(p, _nat(p))
        p.root = _lam(p, n, _node(p, tk.PT_LITERAL_LE, n, n))
        expected = _pi(p, n, _node(p, tk.PT_LE, n, n))
    elif mode == DIV_ZERO or mode == DIV_NEGATIVE or mode == MOD_ZERO or mode == MOD_NEGATIVE:
        var divisor = 0 if mode == DIV_ZERO or mode == MOD_ZERO else -1
        var operation = tk.PT_DIV if mode == DIV_ZERO or mode == DIV_NEGATIVE else tk.PT_MOD
        _ = _node(p, operation, zero, value=divisor)
    elif mode == ZERO_DIVISION_PARAMETER:
        _ = _node(p, tk.PT_ARITHMETIC, extra=0, value=tk.DEC_MOD_LT)
    elif mode == NEGATIVE_SCALE_PARAMETER:
        _ = _node(p, tk.PT_ARITHMETIC, extra=-2, value=tk.DEC_LE_SCALE)
    elif mode == NONPARAMETER_EXTRA:
        _ = _node(p, tk.PT_ARITHMETIC, extra=1, value=tk.DEC_LT_ADD)
    elif mode == RETIRED_NAT_INDUCTION:
        _ = _node(p, tk.PT_NAT_IND)
    elif mode == LEGACY_NAT_INDUCTION:
        # A first-order rule number has no meaning as a proof-term opcode.
        _ = _node(p, 131)
    elif mode == UNKNOWN_KIND:
        _ = _node(p, 99999)
    elif mode == UNKNOWN_DECLARATION:
        _ = _node(p, tk.PT_CONST, value=99999)
    elif mode == UNUSED_MALFORMED:
        _ = _app(p, _nat(p), zero)
    elif mode == OPEN_ROOT:
        p.root = _var(p, _nat(p))
        expected = _nat(p)
    return CoreCase(p^, expected)


def parameter_literal_case() -> CoreCase:
    var p = tk.empty_proof_terms[CORE_CAPACITY]()
    var n = _var(p, _nat(p))
    # 65537 is much larger than the node count: extra is a literal divisor,
    # not a node reference or a fixed list of supported geometries.
    var declaration = _node(p, tk.PT_ARITHMETIC, extra=65537, value=tk.DEC_MOD_LT)
    p.root = _lam(p, n, _app(p, declaration, n))
    var expected = _pi(p, n, _node(p, tk.PT_LT,
        _mod(p, n, 65537), _number(p, 65537)))
    return CoreCase(p^, expected)


def arithmetic_case() -> CoreCase:
    var p = tk.empty_proof_terms[CORE_CAPACITY]()
    var twenty = _number(p, 20)
    var quotient = _div(p, twenty, 7)
    var remainder = _mod(p, twenty, 7)
    var reconstructed = _add(p, _scale(p, quotient, 7), remainder)
    p.root = _node(p, tk.PT_REFL, reconstructed)
    var expected = _node(p, tk.PT_EQ, reconstructed, twenty)
    return CoreCase(p^, expected)


def literal_order_case() -> CoreCase:
    var p = tk.empty_proof_terms[CORE_CAPACITY]()
    var three = _add(p, _number(p, 1), _number(p, 2))
    var five = _number(p, 5)
    p.root = _node(p, tk.PT_LITERAL_LE, three, five)
    var expected = _node(p, tk.PT_LE, three, five)
    return CoreCase(p^, expected)


def zero_scale_case() -> CoreCase:
    var p = tk.empty_proof_terms[CORE_CAPACITY]()
    var zero = _number(p, 0)
    var premise = _node(p, tk.PT_LITERAL_LE, zero, zero)
    var declaration = _node(p, tk.PT_ARITHMETIC, extra=0, value=tk.DEC_LE_SCALE)
    p.root = _app(p, _app(p, _app(p, declaration, zero), zero), premise)
    var expected = _node(p, tk.PT_LE, zero, zero)
    return CoreCase(p^, expected)


def _alter_visited[capacity: Int](mut p: tk.ProofTermBundle[capacity], mode: Int):
    var zero = _number(p, 0)
    var nat = _nat(p)
    if mode == UNCONDITIONAL_VISIT_ZERO:
        var n = _var(p, nat)
        p.constructor_types[0] = _pi(p, n, _visited(p, zero, n))
        return
    var q = _var(p, nat)
    var n = _var(p, nat)
    var next = _add(p, q, _number(p, 1))
    var prior = _var(p, _visited(p, q, n))
    var guard_type = _node(p, tk.PT_LT, _node(p, tk.PT_BASE, next), n)
    if mode == WRONG_VISIT_GUARD:
        guard_type = _node(p, tk.PT_LT, n, _node(p, tk.PT_BASE, next))
    var active = _var(p, guard_type)
    var result_q = q if mode == WRONG_VISIT_ROUND else next
    var result_n = _add(p, n, _number(p, 1)) if mode == WRONG_VISIT_COUNT else n
    p.constructor_types[1] = _pi(p, q, _pi(p, n, _pi(p, prior,
        _pi(p, active, _visited(p, result_q, result_n)))))


def typed_coverage_mutant_accepts(program: FinishingProgram, mode: Int) -> Bool:
    if mode == INSUFFICIENT_MACHINE_HEADROOM:
        var insufficient = program
        insufficient.integer_limit = program.count_limit + program.stride - 1
        var proofs = coverage.make_finishing_typed_coverage[768](insufficient)
        return coverage.check_finishing_typed_coverage(insufficient, proofs)
    if (mode == FALSE_LITERAL_ORDER or mode == SYMBOLIC_LITERAL_ORDER
        or mode >= RETIRED_NAT_INDUCTION):
        var proof_case = core_case(mode)
        return tk.check_proof_terms(program, proof_case.bundle, proof_case.expected)

    var proofs = coverage.make_finishing_typed_coverage[768](program)
    if mode == WRONG_VISIT_BASE or mode == WRONG_VISIT_STEP or mode == WRONG_VISIT_IH:
        var ordinal = 0
        var changed = False
        var initial_count = proofs.bundle.count
        for node in range(initial_count):
            if _is_nat_induction(proofs.bundle, node):
                if ordinal == 1:
                    var step_application = proofs.bundle.nodes[node].left
                    var base_application = proofs.bundle.nodes[step_application].left
                    if mode == WRONG_VISIT_BASE:
                        proofs.bundle.nodes[base_application].right = _number(proofs.bundle, 0)
                    elif mode == WRONG_VISIT_STEP:
                        proofs.bundle.nodes[step_application].right = proofs.base_root
                    else:
                        var step = proofs.bundle.nodes[step_application].right
                        var inner_lambda = proofs.bundle.nodes[step].right
                        var ih = proofs.bundle.nodes[inner_lambda].left
                        proofs.bundle.nodes[ih].left = _nat(proofs.bundle)
                    changed = True
                ordinal += 1
        # Fail open as a test construction: a missing mutation anchor causes
        # the negative fixture to compile, which the Python harness rejects.
        if not changed or ordinal != 2:
            return True
    elif mode == WRONG_INVERSE_WITNESS:
        var changed = False
        for node in range(proofs.bundle.count):
            if (not changed and proofs.bundle.nodes[node].kind == tk.PT_DIV
                and proofs.bundle.nodes[node].value == program.stride):
                proofs.bundle.nodes[node].value = program.stride + 1
                changed = True
        if not changed:
            return True
    elif mode == WRONG_THREAD_BOUND:
        proofs.thread_bound_root = proofs.item_bound_root
    elif mode == WRONG_ITEM_BOUND:
        proofs.item_bound_root = proofs.thread_bound_root
    elif mode == WRONG_INDEX:
        proofs.index_root = proofs.base_root
    elif mode == WEAK_ROUND_UNIQUENESS:
        proofs.round_unique_root = _weak_uniqueness(proofs.bundle, program)
    elif mode == SWAPPED_UNIQUENESS:
        var original = proofs.round_unique_root
        proofs.round_unique_root = proofs.thread_unique_root
        proofs.thread_unique_root = original
    elif mode == MISSING_ITEM_UNIQUENESS:
        proofs.item_unique_root = -1
    elif mode == WEAK_HEADROOM:
        proofs.headroom.proof = _weak_headroom(proofs.headroom.bundle, program)
        proofs.headroom.bundle.root = proofs.headroom.proof
        proofs.headroom_root = proofs.headroom.proof
    elif mode == SWAPPED_HEADROOM:
        proofs.headroom.proof = proofs.headroom.bound
        proofs.headroom.bundle.root = proofs.headroom.proof
        proofs.headroom_root = proofs.headroom.proof
    elif mode == MISSING_HEADROOM:
        proofs.headroom_root = -1
    elif mode >= UNCONDITIONAL_VISIT_ZERO and mode <= WRONG_VISIT_ROUND:
        _alter_visited(proofs.bundle, mode)
    return coverage.check_finishing_typed_coverage(program, proofs)


def check_many_parameterized_declarations() raises:
    # Distinct valid declarations require fresh telescope elaboration even
    # though the required theorem is small. Their count measures proof syntax
    # resources and has no relation to threads, items or runtime counts.
    var program = make_finishing_program(3, 5, 15)
    var p = tk.empty_proof_terms[256]()
    var zero = _number(p, 0)
    for factor in range(80):
        _ = _node(p, tk.PT_ARITHMETIC, extra=factor, value=tk.DEC_LE_SCALE)
    p.root = _node(p, tk.PT_REFL, zero)
    var expected = _node(p, tk.PT_EQ, zero, zero)
    assert_equal(tk.check_proof_terms(program, p, expected), True)
    # Every unused declaration/expression is still checked after elaborating a
    # larger environment; a small valid root cannot hide malformed input.
    _ = _app(p, _nat(p), zero)
    assert_equal(tk.check_proof_terms(program, p, expected), False)


def check_core() raises:
    comptime program = make_finishing_program(3, 5, 15)
    comptime payload = parameter_literal_case()
    comptime assert tk.check_proof_terms(program, payload.bundle, payload.expected), "typed coverage rejected"
    comptime arithmetic = arithmetic_case()
    comptime assert tk.check_proof_terms(program, arithmetic.bundle, arithmetic.expected), "typed coverage rejected"
    comptime order = literal_order_case()
    comptime assert tk.check_proof_terms(program, order.bundle, order.expected), "typed coverage rejected"
    comptime zero_scale = zero_scale_case()
    comptime assert tk.check_proof_terms(program, zero_scale.bundle, zero_scale.expected), "typed coverage rejected"
    var tight_program = make_finishing_program(3, 5, 15, count_limit=37, integer_limit=52)
    var tight_proofs = coverage.make_finishing_typed_coverage[768](tight_program)
    assert_equal(coverage.check_finishing_typed_coverage(tight_program, tight_proofs), True)
    check_many_parameterized_declarations()
    for mode in range(LAST_TYPED_COVERAGE_MUTANT + 1):
        assert_equal(typed_coverage_mutant_accepts(program, mode), False)


def check_shape[threads: Int, items: Int, verify_at_compile_time: Bool = False]() raises:
    comptime program = make_finishing_program(threads, items, threads * items)
    comptime if verify_at_compile_time:
        comptime proofs = coverage.make_finishing_typed_coverage(program)
        comptime assert independent_coverage_accepts(program, proofs), "typed coverage rejected"
        comptime assert coverage_nat_induction_count(proofs) == 2, "visitation mutation target missing"
    var native_proofs = coverage.make_finishing_typed_coverage(program)
    assert_equal(coverage.check_finishing_typed_coverage(program, native_proofs), True)
    assert_equal(independent_coverage_accepts(program, native_proofs), True)
    assert_equal(coverage_nat_induction_count(native_proofs), 2)
    # Concrete endpoint checks independently exercise inverse ownership,
    # without enumerating runtime counts or selecting a fixed thread shape.
    for position in range(33):
        var round = position // program.stride
        var remainder = position % program.stride
        var thread = remainder % program.threads
        var item = remainder // program.threads
        assert_equal(round * program.stride + thread + program.threads * item, position)
        assert_equal(thread < program.threads and item < program.items, True)
    var count = program.count_limit
    var last = (count - 1) // program.stride
    var exit_base = (last + 1) * program.stride
    assert_equal(exit_base >= count and exit_base < count + program.stride, True)
    assert_equal(count + program.stride <= program.integer_limit, True)


def main() raises:
    check_core()
    check_shape[1, 1]()
    check_shape[3, 5, True]()
    check_shape[32, 2]()
    check_shape[64, 3, True]()
    check_shape[128, 1]()
    check_shape[3, 65]()
    check_shape[2147483647, 1]()
    check_shape[1, 2147483647]()
    print("Typed coverage accepted for eight geometries; malformed proofs rejected")
