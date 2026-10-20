"""Untrusted binary Nat proof terms for a finishing driver's headroom bound.

Native integer decoding supplies digits, not arithmetic or order judgments.
Ordinary checked Nat induction derives the generic digit/carry lemmas; an
untrusted bit adder and subtractor merely choose proof terms. Their final
endpoints must match the requested cap, stride and integer limit independently.
No shape enumeration, affine rule, order reflection or compact arithmetic node
occurs in this certificate. Storage bounds checking, not computation shape.
"""
from std.collections import Array
import llm_mojo.kernels.proof_term_syntax as tk
from llm_mojo.kernels.nat_numerals import make_numeral_library
from llm_mojo.kernels.nat_literal_arithmetic import _copy_arithmetic_data
from llm_mojo.kernels.nat_arithmetic import (
    declare_nat_arithmetic_definitions, _le_add as _source_le_add,
    _le_trans as _source_le_trans,
)
from llm_mojo.kernels.nat_arithmetic_terms import _empty_builder

comptime HEADROOM_PROOF_CAPACITY = 8192
comptime HEADROOM_DATA_DEFINITIONS = 9
comptime HEADROOM_ADD_SUCC_LEFT = 9
comptime HEADROOM_DOUBLE_ADD = 10
comptime HEADROOM_DIGIT_START = 11
comptime HEADROOM_SELF_ADD = 19
comptime HEADROOM_LE_ADD = 20
comptime HEADROOM_LE_TRANS = 21
comptime HEADROOM_DEFINITIONS = 22


@fieldwise_init
struct HeadroomProof[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var count_limit: Int
    var stride: Int
    var integer_limit: Int
    var source: Int
    var limit: Int
    var sum: Int
    var slack: Int
    var sum_equality: Int
    var slack_equality: Int
    # Closed proof of Le(Add(E(count_limit), E(stride)), E(integer_limit)).
    var bound: Int
    # Full forall n, Le(n,E(count_limit)) -> Le(Add(n,E(stride)),E(limit)).
    var proof: Int


@fieldwise_init
struct _Builder[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var nodes: Dict[Tuple[Int, Int, Int, Int, Int], Int]


def _node[C: Int](mut p: _Builder[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    if p.bundle.count < 0 or p.bundle.count >= C:
        p.bundle.count = -1
        return -1
    var key = (kind, left, right, extra, value)
    if kind != tk.PT_VAR and key in p.nodes:
        return p.nodes.get(key, -1)
    var result = p.bundle.count
    p.bundle.nodes[result] = tk.ProofTerm(kind, left, right, extra, value)
    p.bundle.count += 1
    if kind != tk.PT_VAR:
        p.nodes[key] = result
    return result


def _nat[C: Int](mut p: _Builder[C]) -> Int:
    return _node(p, tk.PT_NAT)


def _prop[C: Int](mut p: _Builder[C]) -> Int:
    return _node(p, tk.PT_PROP)


def _var[C: Int](mut p: _Builder[C], declared: Int) -> Int:
    return _node(p, tk.PT_VAR, declared)


def _zero[C: Int](mut p: _Builder[C]) -> Int:
    return _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)


def _named[C: Int](mut p: _Builder[C], definition: Int) -> Int:
    return _node(p, tk.PT_DEFINED_REF, value=definition)


def _app[C: Int](mut p: _Builder[C], function: Int, argument: Int) -> Int:
    if function < 0 or argument < 0:
        p.bundle.count = -1
        return -1
    return _node(p, tk.PT_APP, function, argument)


def _app2[C: Int](mut p: _Builder[C], function: Int, a: Int, b: Int) -> Int:
    return _app(p, _app(p, function, a), b)


def _app3[C: Int](mut p: _Builder[C], function: Int, a: Int, b: Int, c: Int) -> Int:
    return _app(p, _app2(p, function, a, b), c)


def _succ[C: Int](mut p: _Builder[C], n: Int) -> Int:
    return _app(p, _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1), n)


def _double[C: Int](mut p: _Builder[C], n: Int) -> Int:
    return _app(p, _named(p, 2), n)


def _plus[C: Int](mut p: _Builder[C], a: Int, b: Int) -> Int:
    return _app2(p, _named(p, 3), a, b)


def _le[C: Int](mut p: _Builder[C], a: Int, b: Int) -> Int:
    return _app2(p, _named(p, 5), a, b)


def _pi[C: Int](mut p: _Builder[C], x: Int, body: Int) -> Int:
    return _node(p, tk.PT_PI, x, body)


def _lam[C: Int](mut p: _Builder[C], x: Int, body: Int) -> Int:
    return _node(p, tk.PT_LAM, x, body)


def _l2[C: Int](mut p: _Builder[C], a: Int, b: Int, body: Int) -> Int:
    return _lam(p, a, _lam(p, b, body))


def _p2[C: Int](mut p: _Builder[C], a: Int, b: Int, body: Int) -> Int:
    return _pi(p, a, _pi(p, b, body))


def _eq[C: Int](mut p: _Builder[C], a: Int, b: Int) -> Int:
    return _node(p, tk.PT_EQ, a, b)


def _refl[C: Int](mut p: _Builder[C], a: Int) -> Int:
    return _node(p, tk.PT_REFL, a)


def _subst[C: Int](mut p: _Builder[C], motive: Int, equality: Int, evidence: Int) -> Int:
    return _node(p, tk.PT_SUBST, motive, equality, evidence)


def _sym[C: Int](mut p: _Builder[C], a: Int, b: Int, equality: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _eq(p, x, a)), equality, _refl(p, a))


def _trans[C: Int](mut p: _Builder[C], a: Int, b: Int, c: Int,
    first: Int, second: Int) -> Int:
    var x = _var(p, _nat(p))
    return _subst(p, _lam(p, x, _eq(p, a, x)), second, first)


def _cong[C: Int](mut p: _Builder[C], function: Int, a: Int, b: Int,
    equality: Int) -> Int:
    var x = _var(p, _nat(p))
    var start = _app(p, function, a)
    return _subst(p, _lam(p, x, _eq(p, start, _app(p, function, x))),
        equality, _refl(p, start))


def _succ_cong[C: Int](mut p: _Builder[C], a: Int, b: Int, equality: Int) -> Int:
    var x = _var(p, _nat(p))
    return _cong(p, _lam(p, x, _succ(p, x)), a, b, equality)


def _induct[C: Int](mut p: _Builder[C], motive: Int, base: Int,
    step: Int, target: Int) -> Int:
    return _app(p, _app3(p, _node(p, tk.PT_ELIMINATOR, value=tk.NAT_FAMILY_ID),
        motive, base, step), target)


def _declare[C: Int](mut p: _Builder[C], type: Int, body: Int) -> Int:
    return tk.pt_declare_definition(p.bundle, type, body)


def _add_succ_left[C: Int](mut p: _Builder[C]) -> Int:
    var a = _var(p, _nat(p))
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var motive = _lam(p, q, _eq(p, _plus(p, _succ(p, a), q),
        _succ(p, _plus(p, a, q))))
    var n = _var(p, _nat(p))
    var lhs = _plus(p, _succ(p, a), n)
    var rhs = _succ(p, _plus(p, a, n))
    var ih = _var(p, _eq(p, lhs, rhs))
    var step = _l2(p, n, ih, _succ_cong(p, lhs, rhs, ih))
    return _declare(p, _p2(p, a, target, _eq(p,
        _plus(p, _succ(p, a), target), _succ(p, _plus(p, a, target)))),
        _l2(p, a, target, _induct(p, motive, _refl(p, _succ(p, a)), step, target)))


def _double_add[C: Int](mut p: _Builder[C]) -> Int:
    # Add recurses on its right operand; both sides acquire exactly two
    # successors in the induction step. No arithmetic theorem is assumed.
    var a = _var(p, _nat(p))
    var target = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var motive = _lam(p, q, _eq(p, _plus(p, _double(p, a), _double(p, q)),
        _double(p, _plus(p, a, q))))
    var n = _var(p, _nat(p))
    var lhs = _plus(p, _double(p, a), _double(p, n))
    var rhs = _double(p, _plus(p, a, n))
    var ih = _var(p, _eq(p, lhs, rhs))
    var one = _succ_cong(p, lhs, rhs, ih)
    var step = _l2(p, n, ih, _succ_cong(p, _succ(p, lhs), _succ(p, rhs), one))
    return _declare(p, _p2(p, a, target, _eq(p,
        _plus(p, _double(p, a), _double(p, target)),
        _double(p, _plus(p, a, target)))),
        _l2(p, a, target, _induct(p, motive, _refl(p, _double(p, a)), step, target)))


def _bit[C: Int](mut p: _Builder[C], digit: Int, high: Int) -> Int:
    var result = _double(p, high)
    return _succ(p, result) if digit == 1 else result


def _with_carry[C: Int](mut p: _Builder[C], carry: Int, value: Int) -> Int:
    return _succ(p, value) if carry == 1 else value


def _digit_lemma[C: Int](mut p: _Builder[C], left_digit: Int,
    right_digit: Int, carry: Int) -> Int:
    var a = _var(p, _nat(p))
    var b = _var(p, _nat(p))
    var da = _double(p, a)
    var db = _double(p, b)
    var sum = _plus(p, a, b)
    var dsum = _double(p, sum)
    var doubled_add = _app2(p, _named(p, HEADROOM_DOUBLE_ADD), a, b)
    var lhs = _plus(p, _bit(p, left_digit, a), _bit(p, right_digit, b))
    var rhs = dsum
    var proof = doubled_add
    if left_digit == 1:
        var edge = _app2(p, _named(p, HEADROOM_ADD_SUCC_LEFT), da, db)
        var start = _plus(p, _succ(p, da), db)
        var middle = _succ(p, _plus(p, da, db))
        rhs = _succ(p, dsum)
        proof = _trans(p, start, middle, rhs, edge,
            _succ_cong(p, _plus(p, da, db), dsum, doubled_add))
    if right_digit == 1:
        # The right successor is a definitional Add equation.
        proof = _succ_cong(p,
            _plus(p, _bit(p, left_digit, a), db), rhs, proof)
        rhs = _succ(p, rhs)
    if carry == 1:
        proof = _succ_cong(p, lhs, rhs, proof)
    var total = left_digit + right_digit + carry
    var expected = _bit(p, total % 2, _with_carry(p, total // 2, sum))
    return _declare(p, _p2(p, a, b, _eq(p, _with_carry(p, carry, lhs), expected)),
        _l2(p, a, b, proof))


def _self_add[C: Int](mut p: _Builder[C]) -> Int:
    # Induct on the shared prefix. The successor case uses add_succ_left;
    # Le(S a,S b) computes to Le(a,b), including with symbolic b.
    var target = _var(p, _nat(p))
    var gap = _var(p, _nat(p))
    var q = _var(p, _nat(p))
    var motive = _lam(p, q, _le(p, q, _plus(p, q, gap)))
    var proposition = _var(p, _prop(p))
    var witness = _var(p, proposition)
    var unit = _l2(p, proposition, witness, witness)
    var n = _var(p, _nat(p))
    var ih = _var(p, _le(p, n, _plus(p, n, gap)))
    var left = _plus(p, _succ(p, n), gap)
    var right = _succ(p, _plus(p, n, gap))
    var edge = _app2(p, _named(p, HEADROOM_ADD_SUCC_LEFT), n, gap)
    var x = _var(p, _nat(p))
    var transported = _subst(p, _lam(p, x, _le(p, _succ(p, n), x)),
        _sym(p, left, right, edge), ih)
    return _declare(p, _p2(p, target, gap, _le(p, target, _plus(p, target, gap))),
        _l2(p, target, gap, _induct(p, motive, unit, _l2(p, n, ih, transported), target)))


def _copy_order_helpers[C: Int](mut p: _Builder[C]) -> Bool:
    """Translate existing ordinary LeAdd/LeTrans proof producers to raw syntax.

    They are copied syntax, never a trusted signature. All zero/successor
    abbreviations become actual constructors; source definition references are
    remapped to the pinned data prefix and these two newly checked definitions.
    """
    var source = _empty_builder[C]()
    if not declare_nat_arithmetic_definitions(source):
        return False
    var add = _source_le_add(source)
    var trans = _source_le_trans(source)
    if add != 3 or trans != 4 or source.bundle.count < 0:
        return False
    var copied = Array[Int, C](fill=-1)
    for j in range(source.bundle.count):
        var n = source.bundle.nodes[j]
        var left = -1
        var right = -1
        var extra = n.extra
        if n.left >= 0:
            if n.left >= j or copied[n.left] < 0:
                return False
            left = copied[n.left]
        if n.right >= 0:
            if n.right >= j or copied[n.right] < 0:
                return False
            right = copied[n.right]
        if n.kind == tk.PT_NAT_REC or n.kind == tk.PT_SUBST:
            if n.extra < 0 or n.extra >= j or copied[n.extra] < 0:
                return False
            extra = copied[n.extra]
        var result = -1
        if n.kind == tk.PT_LITERAL:
            if n.value != 0:
                return False
            result = _zero(p)
        elif n.kind == tk.PT_SUCC:
            result = _succ(p, left)
        elif n.kind == tk.PT_DEFINED_REF:
            var definition = -1
            if n.value >= 0 and n.value < 3:
                definition = n.value + 3
            elif n.value == 3:
                definition = HEADROOM_LE_ADD
            elif n.value == 4:
                definition = HEADROOM_LE_TRANS
            if definition < 0:
                return False
            result = _named(p, definition)
        elif (n.kind == tk.PT_TYPE or n.kind == tk.PT_PROP or n.kind == tk.PT_NAT
            or n.kind == tk.PT_VAR or n.kind == tk.PT_PI or n.kind == tk.PT_LAM
            or n.kind == tk.PT_APP or n.kind == tk.PT_NAT_REC
            or n.kind == tk.PT_EQ or n.kind == tk.PT_REFL or n.kind == tk.PT_SUBST
            or n.kind == tk.PT_ELIMINATOR or n.kind == tk.PT_CONSTRUCTOR):
            result = _node(p, n.kind, left, right, extra, n.value)
        if result < 0:
            return False
        copied[j] = result
    for definition in range(3, 5):
        var declared = source.bundle.definition_types[definition]
        var body = source.bundle.definition_bodies[definition]
        if _declare(p, copied[declared], copied[body]) != definition + 17:
            return False
    return True


def _bits(value: Int) -> List[Int]:
    var result = List[Int]()
    if value <= 0:
        return result^
    # Int division loops truncate high bits in the pinned compile-time
    # evaluator. Explicit unsigned shifts preserve the full host word.
    var remaining = UInt64(value)
    while remaining > UInt64(0):
        result.append(Int(remaining & UInt64(1)))
        remaining = remaining >> UInt64(1)
    return result^


def _bits_node[C: Int](mut p: _Builder[C], bits: List[Int], start: Int = 0) -> Int:
    var length = len(bits)
    while length > start and bits[length - 1] == 0:
        length -= 1
    var result = _zero(p)
    for j in range(length - start):
        result = _bit(p, bits[length - j - 1], result)
    return result


@fieldwise_init
struct _Sum(Copyable, Movable):
    var bits: List[Int]
    var value: Int
    var proof: Int


def _sum[C: Int](mut p: _Builder[C], a: List[Int], b: List[Int],
    start: Int = 0, carry: Int = 0) -> _Sum:
    if start >= len(a) and start >= len(b):
        var bits = List[Int]()
        if carry == 1:
            bits.append(1)
        var value = _bits_node(p, bits)
        return _Sum(bits^, value, _refl(p, _with_carry(p, carry, _zero(p))))
    var left_digit = a[start] if start < len(a) else 0
    var right_digit = b[start] if start < len(b) else 0
    var total = left_digit + right_digit + carry
    var digit = total % 2
    var next_carry = total // 2
    var high = _sum(p, a, b, start + 1, next_carry)
    var ah = _bits_node(p, a, start + 1)
    var bh = _bits_node(p, b, start + 1)
    var inner = _with_carry(p, next_carry, _plus(p, ah, bh))
    var lemma = HEADROOM_DIGIT_START + left_digit * 4 + right_digit * 2 + carry
    var edge = _app2(p, _named(p, lemma), ah, bh)
    var x = _var(p, _nat(p))
    var function = _lam(p, x, _bit(p, digit, x))
    var continued = _cong(p, function, inner, high.value, high.proof)
    var source = _with_carry(p, carry,
        _plus(p, _bits_node(p, a, start), _bits_node(p, b, start)))
    var value = _bit(p, digit, high.value)
    var proof = _trans(p, source, _bit(p, digit, inner), value, edge, continued)
    var bits = List[Int]()
    bits.append(digit)
    for bit in high.bits:
        bits.append(bit)
    while len(bits) > 0 and bits[len(bits) - 1] == 0:
        _ = bits.pop()
    var canonical = _bits_node(p, bits)
    return _Sum(bits^, canonical, proof)


def _difference(a: List[Int], b: List[Int]) -> List[Int]:
    """Untrusted binary-borrow proposal, independently verified by addition."""
    var result = List[Int]()
    var length = len(a) if len(a) > len(b) else len(b)
    var borrow = 0
    for j in range(length):
        var left = a[j] if j < len(a) else 0
        var right = b[j] if j < len(b) else 0
        var proposed = left - right - borrow
        if proposed < 0:
            result.append(proposed + 2)
            borrow = 1
        else:
            result.append(proposed)
            borrow = 0
    # Underflow only changes the proposed syntax. The final Eq endpoint is
    # always the independently decoded limit, so no comparison grants proof.
    if borrow != 0:
        result.append(1)
    while len(result) > 0 and result[len(result) - 1] == 0:
        _ = result.pop()
    return result^


def make_headroom_proof[capacity: Int = HEADROOM_PROOF_CAPACITY](
    cap: Int, stride: Int, limit: Int) -> HeadroomProof[capacity]:
    var numerals = make_numeral_library[capacity]()
    var bundle = numerals.bundle.copy()
    var raw_data = _copy_arithmetic_data(bundle)
    var p = _Builder[capacity](bundle^,
        Dict[Tuple[Int, Int, Int, Int, Int], Int]())
    for j in range(p.bundle.count):
        var n = p.bundle.nodes[j]
        if n.kind != tk.PT_VAR:
            var key = (n.kind, n.left, n.right, n.extra, n.value)
            if key not in p.nodes:
                p.nodes[key] = j
    var valid = (raw_data and cap >= 0 and stride >= 0 and limit >= 0
        and _add_succ_left(p) == HEADROOM_ADD_SUCC_LEFT
        and _double_add(p) == HEADROOM_DOUBLE_ADD)
    for left_digit in range(2):
        for right_digit in range(2):
            for carry in range(2):
                var expected = HEADROOM_DIGIT_START + left_digit * 4 + right_digit * 2 + carry
                if _digit_lemma(p, left_digit, right_digit, carry) != expected:
                    valid = False
    if _self_add(p) != HEADROOM_SELF_ADD or not _copy_order_helpers(p):
        valid = False
    var source = -1
    var limit_node = -1
    var sum_node = -1
    var slack = -1
    var sum_equality = -1
    var slack_equality = -1
    var bound = -1
    var proof = -1
    if valid:
        var cap_bits = _bits(cap)
        var stride_bits = _bits(stride)
        var limit_bits = _bits(limit)
        var first = _sum(p, cap_bits, stride_bits)
        var slack_bits = _difference(limit_bits, first.bits)
        var second = _sum(p, first.bits, slack_bits)
        var cap_node = _bits_node(p, cap_bits)
        var stride_node = _bits_node(p, stride_bits)
        limit_node = _bits_node(p, limit_bits)
        sum_node = first.value
        slack = _bits_node(p, slack_bits)
        sum_equality = first.proof
        slack_equality = second.proof
        source = _plus(p, cap_node, stride_node)
        var x = _var(p, _nat(p))
        var add_slack = _lam(p, x, _plus(p, x, slack))
        var first_lift = _cong(p, add_slack, source, first.value, first.proof)
        var total_eq = _trans(p, _plus(p, source, slack),
            _plus(p, first.value, slack), limit_node, first_lift, second.proof)
        var self_add = _app2(p, _named(p, HEADROOM_SELF_ADD), source, slack)
        var rhs = _var(p, _nat(p))
        bound = _subst(p, _lam(p, rhs, _le(p, source, rhs)), total_eq, self_add)
        var n = _var(p, _nat(p))
        var capped = _var(p, _le(p, n, cap_node))
        var monotone = _app(p, _app3(p, _named(p, HEADROOM_LE_ADD),
            n, cap_node, stride_node), capped)
        var combined = _app2(p, _app3(p, _named(p, HEADROOM_LE_TRANS),
            _plus(p, n, stride_node), source, limit_node), monotone, bound)
        proof = _l2(p, n, capped, combined)
        p.bundle.root = proof
    if not valid or bound < 0 or proof < 0:
        p.bundle.count = -1
    return HeadroomProof[capacity](p.bundle.copy(), cap, stride, limit,
        source, limit_node, sum_node, slack, sum_equality, slack_equality, bound, proof)
