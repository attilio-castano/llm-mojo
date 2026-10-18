"""Untrusted Boolean law terms checked through ordinary constructor cases.

Native choices select syntax to generate; every leaf must typecheck by
conversion and reflexivity. These laws support a total Nat score/Boolean-flag
join, without proving native UInt32 record or GPU execution correspondence.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.nat_join_bits as bits


@fieldwise_init
struct RecordBoolExports(ImplicitlyCopyable):
    var closed: Int
    var commutative: Int
    var associative: Int
    var left_unit: Int
    var right_unit: Int
    var idempotent: Int


def _intro[C: Int](mut p: ph._Builder[C], one: Bool) -> Int:
    var proposition = ph._var(p, ph._prop(p))
    var zero_case = ph._var(p, ph._pi(p,
        ph._var(p, ph._eq(p, ph._one(p) if one else ph._zero(p), ph._zero(p))),
        proposition))
    var one_case = ph._var(p, ph._pi(p,
        ph._var(p, ph._eq(p, ph._one(p) if one else ph._zero(p), ph._one(p))),
        proposition))
    var result = ph._app(p, one_case if one else zero_case,
        ph._refl(p, ph._one(p) if one else ph._zero(p)))
    return ph._lams(p, [proposition, zero_case, one_case], result)


def _lhs[C: Int](mut p: ph._Builder[C], law: Int, values: List[Int]) -> Int:
    if law == 1 or law == 0:
        return bits.record_bool_or(p, values[0], values[1])
    if law == 2:
        return bits.record_bool_or(p,
            bits.record_bool_or(p, values[0], values[1]), values[2])
    if law == 3:
        return bits.record_bool_or(p, ph._zero(p), values[0])
    if law == 4:
        return bits.record_bool_or(p, values[0], ph._zero(p))
    return bits.record_bool_or(p, values[0], values[0])


def _goal[C: Int](mut p: ph._Builder[C], law: Int, values: List[Int]) -> Int:
    var left = _lhs(p, law, values)
    if law == 0:
        return bits.record_is_bit(p, left)
    var right = values[0]
    if law == 1:
        right = bits.record_bool_or(p, values[1], values[0])
    elif law == 2:
        right = bits.record_bool_or(p, values[0],
            bits.record_bool_or(p, values[1], values[2]))
    return ph._eq(p, left, right)


def _cases[C: Int](mut p: ph._Builder[C], law: Int, values: List[Int],
    evidence: List[Int], position: Int) -> Int:
    if position == len(values):
        if law == 0:
            return _intro(p, values[0] == ph._one(p) or values[1] == ph._one(p))
        return ph._refl(p, _lhs(p, law, values))
    var value = values[position]
    var goal = _goal(p, law, values)
    var x = ph._var(p, ph._nat(p))
    var motive_values = values.copy()
    motive_values[position] = x
    var motive = ph._lam(p, x, _goal(p, law, motive_values))
    var branches = List[Int]()
    for choice in range(2):
        var literal = ph._zero(p) if choice == 0 else ph._one(p)
        var equality = ph._var(p, ph._eq(p, value, literal))
        var next_values = values.copy()
        next_values[position] = literal
        var proof = _cases(p, law, next_values, evidence, position + 1)
        branches.append(ph._lam(p, equality, ph._subst(p, motive,
            ph._sym(p, value, literal, equality), proof)))
    return ph._apps(p, evidence[position], [goal, branches[0], branches[1]])


def _declare_law[C: Int](mut p: ph._Builder[C], law: Int) -> Int:
    var arity = 1
    if law == 0 or law == 1:
        arity = 2
    elif law == 2:
        arity = 3
    var values = List[Int]()
    var evidence = List[Int]()
    for _ in range(arity):
        values.append(ph._var(p, ph._nat(p)))
    for value in values:
        evidence.append(ph._var(p, bits.record_is_bit(p, value)))
    var binders = values.copy()
    for proof in evidence:
        binders.append(proof)
    var index = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, _goal(p, law, values)),
        ph._lams(p, binders, _cases(p, law, values, evidence, 0)), index)
    return index


def append_record_bool_terms[C: Int](mut p: ph._Builder[C]) -> RecordBoolExports:
    var closed = _declare_law(p, 0)
    var commutative = _declare_law(p, 1)
    var associative = _declare_law(p, 2)
    var left_unit = _declare_law(p, 3)
    var right_unit = _declare_law(p, 4)
    var idempotent = _declare_law(p, 5)
    return RecordBoolExports(closed, commutative, associative,
        left_unit, right_unit, idempotent)
