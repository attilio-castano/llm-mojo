"""Independent public contracts for immutable checked export instantiation.

Expected propositions use caller-side native syntax only. They never consult
kernel signature builders, imported syntax, or the library's required types.
All natural operands remain bound variables; parameter probes are literals.
"""
from std.collections import Array
import llm_mojo.kernels.proof_term_kernel as tk

comptime EXPORT_CASE_CAPACITY = 2048
comptime WRONG_FACTOR = 0
comptime WRONG_DIVISOR = 1
comptime SWAPPED_DIVISION_RESULT = 2
comptime NONPARAMETER_PAYLOAD = 3
comptime WRONG_BASE_STRIDE = 4
comptime SWAPPED_ORDER_EXPORT = 5
comptime PRIVATE_REFERENCE = 6
comptime EXPORT_CONTRACT_MUTANTS = 7


def _n[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _v[C: Int](mut p: tk.ProofTermBundle[C], type: Int) -> Int:
    return _n(p, tk.PT_VAR, type)


def _lit[C: Int](mut p: tk.ProofTermBundle[C], number: Int) -> Int:
    return _n(p, tk.PT_LITERAL, value=number)


def _pi[C: Int](mut p: tk.ProofTermBundle[C], binder: Int, body: Int) -> Int:
    return _n(p, tk.PT_PI, binder, body)


def _app[C: Int](mut p: tk.ProofTermBundle[C], f: Int, argument: Int) -> Int:
    return _n(p, tk.PT_APP, f, argument)


def _lam[C: Int](mut p: tk.ProofTermBundle[C], binder: Int, body: Int) -> Int:
    return _n(p, tk.PT_LAM, binder, body)


def _parameterized(declaration: Int) -> Bool:
    return (declaration == tk.DEC_LE_SCALE or declaration == tk.DEC_LT_SCALE_CANCEL
        or declaration == tk.DEC_DIV_DECOMP or declaration == tk.DEC_MOD_LT
        or declaration == tk.DEC_DIV_QUOTIENT_UNIQUE or declaration == tk.DEC_DIV_REMAINDER_UNIQUE)


def _declaration_at(slot: Int) -> Int:
    return slot if slot < 5 else slot + 2


def independent_export_contract[C: Int](mut p: tk.ProofTermBundle[C],
    declaration: Int, parameter: Int, stride: Int) -> Int:
    """Pin the proposition exposed to callers, independently of the importer."""
    var nat = _n(p, tk.PT_NAT)
    var zero = _lit(p, 0)
    var a = _v(p, nat)
    var b = _v(p, nat)
    var c = _v(p, nat)
    if declaration == tk.DEC_BASE_ZERO:
        return _n(p, tk.PT_EQ, _n(p, tk.PT_BASE, zero), zero)
    if declaration == tk.DEC_BASE_STEP:
        var lhs = _n(p, tk.PT_BASE, _n(p, tk.PT_ADD, a, _lit(p, 1)))
        var rhs = _n(p, tk.PT_ADD, _n(p, tk.PT_BASE, a), _lit(p, stride))
        return _pi(p, a, _n(p, tk.PT_EQ, lhs, rhs))
    if declaration == tk.DEC_NAT_NONNEG:
        return _pi(p, a, _n(p, tk.PT_LE, zero, a))
    if declaration == tk.DEC_ORDER_CASES:
        var proposition = _v(p, _n(p, tk.PT_PROP))
        var less = _v(p, _n(p, tk.PT_LT, a, b))
        var greater = _v(p, _n(p, tk.PT_LE, b, a))
        var first = _v(p, _pi(p, less, proposition))
        var second = _v(p, _pi(p, greater, proposition))
        return _pi(p, a, _pi(p, b, _pi(p, proposition,
            _pi(p, first, _pi(p, second, proposition)))))
    if declaration == tk.DEC_LE_ADD or declaration == tk.DEC_LT_ADD:
        var relation = tk.PT_LE if declaration == tk.DEC_LE_ADD else tk.PT_LT
        var premise = _v(p, _n(p, relation, a, b))
        var goal = _n(p, relation, _n(p, tk.PT_ADD, a, c), _n(p, tk.PT_ADD, b, c))
        return _pi(p, a, _pi(p, b, _pi(p, c, _pi(p, premise, goal))))
    if (declaration == tk.DEC_LE_TRANS or declaration == tk.DEC_LE_LT_TRANS
        or declaration == tk.DEC_LT_LE_TRANS):
        var first_relation = tk.PT_LT if declaration == tk.DEC_LT_LE_TRANS else tk.PT_LE
        var second_relation = tk.PT_LT if declaration == tk.DEC_LE_LT_TRANS else tk.PT_LE
        var conclusion_relation = tk.PT_LE if declaration == tk.DEC_LE_TRANS else tk.PT_LT
        var first = _v(p, _n(p, first_relation, a, b))
        var second = _v(p, _n(p, second_relation, b, c))
        var goal = _n(p, conclusion_relation, a, c)
        return _pi(p, a, _pi(p, b, _pi(p, c, _pi(p, first, _pi(p, second, goal)))))
    if declaration == tk.DEC_LE_SCALE or declaration == tk.DEC_LT_SCALE_CANCEL:
        var scaled_a = _n(p, tk.PT_SCALE, a, value=parameter)
        var scaled_b = _n(p, tk.PT_SCALE, b, value=parameter)
        var premise_type = (_n(p, tk.PT_LE, a, b) if declaration == tk.DEC_LE_SCALE
            else _n(p, tk.PT_LT, scaled_a, scaled_b))
        var conclusion = (_n(p, tk.PT_LE, scaled_a, scaled_b) if declaration == tk.DEC_LE_SCALE
            else _n(p, tk.PT_LT, a, b))
        var premise = _v(p, premise_type)
        return _pi(p, a, _pi(p, b, _pi(p, premise, conclusion)))
    if declaration == tk.DEC_LT_TO_LE_SUCC:
        var premise = _v(p, _n(p, tk.PT_LT, a, b))
        var goal = _n(p, tk.PT_LE, _n(p, tk.PT_ADD, a, _lit(p, 1)), b)
        return _pi(p, a, _pi(p, b, _pi(p, premise, goal)))
    if declaration == tk.DEC_DIV_DECOMP:
        var quotient = _n(p, tk.PT_DIV, a, value=parameter)
        var remainder = _n(p, tk.PT_MOD, a, value=parameter)
        var reconstruction = _n(p, tk.PT_ADD,
            _n(p, tk.PT_SCALE, quotient, value=parameter), remainder)
        return _pi(p, a, _n(p, tk.PT_EQ, a, reconstruction))
    if declaration == tk.DEC_MOD_LT:
        return _pi(p, a, _n(p, tk.PT_LT,
            _n(p, tk.PT_MOD, a, value=parameter), _lit(p, parameter)))
    if declaration == tk.DEC_DIV_QUOTIENT_UNIQUE or declaration == tk.DEC_DIV_REMAINDER_UNIQUE:
        var bound = _v(p, _n(p, tk.PT_LT, c, _lit(p, parameter)))
        var decomposition = _n(p, tk.PT_EQ, a,
            _n(p, tk.PT_ADD, _n(p, tk.PT_SCALE, b, value=parameter), c))
        var equality = _v(p, decomposition)
        var operation = tk.PT_DIV if declaration == tk.DEC_DIV_QUOTIENT_UNIQUE else tk.PT_MOD
        var witness = b if declaration == tk.DEC_DIV_QUOTIENT_UNIQUE else c
        var goal = _n(p, tk.PT_EQ, _n(p, operation, a, value=parameter), witness)
        return _pi(p, a, _pi(p, b, _pi(p, c, _pi(p, bound, _pi(p, equality, goal)))))
    return -1


def _caller_shadows[C: Int](mut p: tk.ProofTermBundle[C]):
    # Data IDs 0..5 compute other operations. IDs 6..35 are ordinary Nat
    # definitions, overlapping the immutable library's actual theorem IDs.
    var nat = _n(p, tk.PT_NAT)
    var zero = _lit(p, 0)
    for definition in range(3):
        var a = _v(p, nat)
        var b = _v(p, nat)
        var result_type = _n(p, tk.PT_PROP) if definition == 2 else nat
        var body = zero
        if definition == 2:
            var proposition = _v(p, _n(p, tk.PT_PROP))
            body = _pi(p, proposition, proposition)
        _ = tk.pt_declare_definition(p, _pi(p, a, _pi(p, b, result_type)),
            _lam(p, a, _lam(p, b, body)))
    var k = _v(p, nat)
    var dividend = _v(p, nat)
    var q = _v(p, nat)
    var r = _v(p, nat)
    var countdown = _v(p, nat)
    var selector = _v(p, _pi(p, q, _pi(p, r, _pi(p, countdown, nat))))
    var triple_type = _pi(p, selector, nat)
    var triple = _lam(p, selector, _app(p, _app(p, _app(p, selector, zero), zero), zero))
    _ = tk.pt_declare_definition(p, _pi(p, k, _pi(p, dividend, triple_type)),
        _lam(p, k, _lam(p, dividend, triple)))
    var predecessor = _v(p, nat)
    _ = tk.pt_declare_definition(p, _pi(p, predecessor, nat), _lam(p, predecessor, predecessor))
    var stride = _v(p, nat)
    var round = _v(p, nat)
    _ = tk.pt_declare_definition(p, _pi(p, stride, _pi(p, round, nat)),
        _lam(p, stride, _lam(p, round, zero)))
    for _ in range(30):
        _ = tk.pt_declare_definition(p, nat, zero)
    # These unused binders exercise import freshness independently of closure.
    for _ in range(40):
        _ = _v(p, nat)


def export_contracts_accept(parameter: Int = 17, stride: Int = 17,
    mode: Int = -1, shadows_and_aliases: Bool = False, zero_scale: Bool = False) -> Bool:
    var p = tk.empty_proof_terms[EXPORT_CASE_CAPACITY]()
    if shadows_and_aliases:
        _caller_shadows(p)
    var checks = Array[tk.ProofTermJudgment, 16](fill=tk.ProofTermJudgment(-1, -1))
    for slot in range(16):
        var declaration = _declaration_at(slot)
        var argument = parameter if _parameterized(declaration) else -1
        if zero_scale and declaration == tk.DEC_LE_SCALE:
            argument = 0
        var wanted_declaration = declaration
        var wanted_parameter = argument
        var wanted_stride = stride
        var actual_declaration = declaration
        if mode == WRONG_FACTOR and declaration == tk.DEC_LE_SCALE:
            wanted_parameter = parameter - 1
        elif mode == WRONG_DIVISOR and declaration == tk.DEC_DIV_DECOMP:
            wanted_parameter = parameter + 1
        elif mode == SWAPPED_DIVISION_RESULT and declaration == tk.DEC_DIV_QUOTIENT_UNIQUE:
            wanted_declaration = tk.DEC_DIV_REMAINDER_UNIQUE
        elif mode == NONPARAMETER_PAYLOAD and declaration == tk.DEC_BASE_STEP:
            argument = 0
        elif mode == WRONG_BASE_STRIDE and declaration == tk.DEC_BASE_STEP:
            wanted_stride = stride + 1
        elif mode == SWAPPED_ORDER_EXPORT and declaration == tk.DEC_LE_TRANS:
            actual_declaration = tk.DEC_LE_LT_TRANS
        var expected = independent_export_contract(p, wanted_declaration, wanted_parameter, wanted_stride)
        if shadows_and_aliases:
            var alias = tk.pt_declare_definition(p, _n(p, tk.PT_PROP), expected)
            expected = _n(p, tk.PT_DEFINED_REF, value=alias)
        var theorem = _n(p, tk.PT_ARITHMETIC, extra=argument, value=actual_declaration)
        checks[slot] = tk.ProofTermJudgment(theorem, expected)
    if mode == PRIVATE_REFERENCE:
        _ = _n(p, tk.PT_ARITHMETIC + 1, value=7)
    var program = tk.FinishingProgram(stride, 1, stride, 0, 1, stride, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    return tk.check_proof_terms_many(program, p, checks)


def mixed_export_parameters_accept(reverse: Bool = False, mode: Int = -1) -> Bool:
    var p = tk.empty_proof_terms[EXPORT_CASE_CAPACITY]()
    var checks = Array[tk.ProofTermJudgment, 12](fill=tk.ProofTermJudgment(-1, -1))
    for family in range(6):
        var declaration = tk.DEC_LE_SCALE + family if family < 2 else tk.DEC_DIV_DECOMP + family - 2
        for instance in range(2):
            var parameter = 17 if (instance == 0) != reverse else Int.MAX
            var wanted = parameter
            if mode == 0 and family == 0 and instance == 1:
                wanted = 17 if parameter == Int.MAX else Int.MAX
            elif mode == 1 and family == 2 and instance == 1:
                wanted = 17 if parameter == Int.MAX else Int.MAX
            var theorem = _n(p, tk.PT_ARITHMETIC, extra=parameter, value=declaration)
            var expected = independent_export_contract(p, declaration, wanted, 17)
            checks[family * 2 + instance] = tk.ProofTermJudgment(theorem, expected)
    var program = tk.FinishingProgram(17, 1, 17, 0, 1, 17, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    return tk.check_proof_terms_many(program, p, checks)


def export_capture_accepts(capture: Bool = False) -> Bool:
    var p = tk.empty_proof_terms[128]()
    var nat = _n(p, tk.PT_NAT)
    var zero = _lit(p, 0)
    var outer = _v(p, nat)
    var argument = _v(p, nat)
    var inner_binder = outer if capture else _v(p, nat)
    var theorem = _n(p, tk.PT_ARITHMETIC, value=tk.DEC_NAT_NONNEG)
    var inner = _lam(p, inner_binder, _app(p, theorem, argument))
    var function = _lam(p, argument, inner)
    p.root = _lam(p, outer, _app(p, function, outer))
    var a = _v(p, nat)
    var b = _v(p, nat)
    var expected = _pi(p, a, _pi(p, b, _n(p, tk.PT_LE, zero, a)))
    var program = tk.FinishingProgram(17, 1, 17, 0, 1, 17, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    return tk.check_proof_terms(program, p, expected)


def caller_export_spoof_accepts() -> Bool:
    var p = tk.empty_proof_terms[256]()
    _caller_shadows(p)
    var theorem = _n(p, tk.PT_DEFINED_REF, value=7)
    p.root = theorem
    var expected = independent_export_contract(p, tk.DEC_BASE_STEP, -1, 17)
    var program = tk.FinishingProgram(17, 1, 17, 0, 1, 17, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    return tk.check_proof_terms(program, p, expected)


def export_open_root_accepts() -> Bool:
    var p = tk.empty_proof_terms[128]()
    var a = _v(p, _n(p, tk.PT_NAT))
    var theorem = _n(p, tk.PT_ARITHMETIC, value=tk.DEC_NAT_NONNEG)
    p.root = _app(p, theorem, a)
    var expected = _n(p, tk.PT_LE, _lit(p, 0), a)
    var program = tk.FinishingProgram(17, 1, 17, 0, 1, 17, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    return tk.check_proof_terms(program, p, expected)
