"""Untrusted binary Nat syntax and explicit definitional-conversion witnesses.

The three closed definitions use actual Nat constructors and structural data
recursion. No literal, successor shorthand, arithmetic operator or reflection
rule occurs in their syntax. A numeral's binary digits select a shared syntax
DAG; each constructor view carries an ordinary beta/delta/iota derivation.
The kernel checks definitions, endpoints, types and every conversion premise.
Host integer parsing and digit extraction remain a separate interpretation.
"""
from std.collections import Array
import llm_mojo.kernels.proof_term_syntax as tk

comptime NUMERAL_LIBRARY_CAPACITY = 64
comptime NUMERAL_VIEW_CAPACITY = 2048
comptime NUMERAL_CONVERSION_CAPACITY = 2048

comptime NC_REFL = 0
comptime NC_BETA = 1
comptime NC_DELTA = 2
comptime NC_NAT_IOTA = 3
comptime NC_CONG_LEFT = 4
comptime NC_CONG_RIGHT = 5
comptime NC_SYMM = 6
comptime NC_TRANS = 7


@fieldwise_init
struct NumeralConversion(ImplicitlyCopyable):
    var kind: Int
    var left: Int
    var right: Int
    var first: Int
    var second: Int


@fieldwise_init
struct NumeralLibrary[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]


@fieldwise_init
struct NumeralView[capacity: Int, derivation_capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var conversions: Array[NumeralConversion, Self.derivation_capacity]
    var count: Int
    var root: Int
    var value: Int
    var denotation: Int
    var constructor: Int
    var predecessor: Int


def _node[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _zero[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)


def _successor[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1)


def _app[C: Int](mut p: tk.ProofTermBundle[C], function: Int, argument: Int) -> Int:
    return _node(p, tk.PT_APP, function, argument)


def _succ[C: Int](mut p: tk.ProofTermBundle[C], argument: Int) -> Int:
    return _app(p, _successor(p), argument)


def _named[C: Int](mut p: tk.ProofTermBundle[C], definition: Int) -> Int:
    return _node(p, tk.PT_DEFINED_REF, value=definition)


def make_numeral_library[capacity: Int = NUMERAL_LIBRARY_CAPACITY]() -> NumeralLibrary[capacity]:
    """Candidate zero, successor and double definitions, in that order."""
    var p = tk.empty_proof_terms[capacity]()
    var nat = _node(p, tk.PT_NAT)
    var zero = _zero(p)
    if tk.pt_declare_definition(p, nat, zero) != 0:
        p.count = -1
        return NumeralLibrary[capacity](p^)
    var argument = _node(p, tk.PT_VAR, nat)
    var successor_type = _node(p, tk.PT_PI, argument, nat)
    var successor_body = _node(p, tk.PT_LAM, argument, _succ(p, argument))
    if tk.pt_declare_definition(p, successor_type, successor_body) != 1:
        p.count = -1
        return NumeralLibrary[capacity](p^)
    var major = _node(p, tk.PT_VAR, nat)
    var predecessor = _node(p, tk.PT_VAR, nat)
    var previous = _node(p, tk.PT_VAR, nat)
    var step = _node(p, tk.PT_LAM, predecessor,
        _node(p, tk.PT_LAM, previous, _succ(p, _succ(p, previous))))
    var body = _node(p, tk.PT_LAM, major,
        _node(p, tk.PT_NAT_REC, major, zero, step))
    var declared = _node(p, tk.PT_PI, major, nat)
    if tk.pt_declare_definition(p, declared, body) != 2:
        p.count = -1
    return NumeralLibrary[capacity](p^)


@fieldwise_init
struct _NumeralBuilder[capacity: Int, derivation_capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var conversions: Array[NumeralConversion, Self.derivation_capacity]
    var count: Int
    var denotations: Dict[Int, Int]
    var views: Dict[Int, Int]
    var double_delta: Int


def _failed[C: Int, D: Int](mut b: _NumeralBuilder[C, D]) -> Int:
    b.count = -1
    return -1


def _valid_node[C: Int, D: Int](b: _NumeralBuilder[C, D], node: Int) -> Bool:
    return node >= 0 and node < b.bundle.count


def _valid_conversion[C: Int, D: Int](b: _NumeralBuilder[C, D], step: Int) -> Bool:
    return step >= 0 and step < b.count


def _constructor_predecessor[C: Int](p: tk.ProofTermBundle[C], node: Int) -> Int:
    """Read candidate predecessor syntax, without a host-word subtraction."""
    if node < 0 or node >= p.count:
        return -1
    var application = p.nodes[node]
    if (application.kind != tk.PT_APP or application.left < 0
        or application.left >= p.count or application.right < 0
        or application.right >= p.count):
        return -1
    var head = p.nodes[application.left]
    if (head.kind != tk.PT_CONSTRUCTOR or head.extra != tk.NAT_FAMILY_ID
        or head.value != 1):
        return -1
    return application.right


def _conversion[C: Int, D: Int](mut b: _NumeralBuilder[C, D], kind: Int,
    left: Int, right: Int, first: Int = -1, second: Int = -1) -> Int:
    if (b.count < 0 or b.count >= D or not _valid_node(b, left)
        or not _valid_node(b, right)):
        return _failed(b)
    var result = b.count
    b.conversions[result] = NumeralConversion(kind, left, right, first, second)
    b.count += 1
    return result


def _substitute[C: Int](mut p: tk.ProofTermBundle[C], node: Int,
    variable: Int, replacement: Int) -> Int:
    """Untrusted substitution for these Nat-only, closed-argument beta steps.

    The kernel recomputes substitution independently. This producer deliberately
    does not implement dependent-domain renaming: its binder domains are Nat.
    """
    if node < 0 or node >= p.count or replacement < 0 or replacement >= p.count:
        return -1
    if node == variable:
        return replacement
    var n = p.nodes[node]
    if n.kind == tk.PT_VAR or n.kind == tk.PT_DEFINED_REF or n.kind == tk.PT_CONSTRUCTOR or n.kind == tk.PT_NAT:
        return node
    if n.kind == tk.PT_LAM or n.kind == tk.PT_PI:
        if n.left == variable:
            return node
        var body = _substitute(p, n.right, variable, replacement)
        return _node(p, n.kind, n.left, body) if body >= 0 else -1
    var left = n.left
    var right = n.right
    var extra = n.extra
    if left >= 0:
        left = _substitute(p, left, variable, replacement)
        if left < 0:
            return -1
    if right >= 0:
        right = _substitute(p, right, variable, replacement)
        if right < 0:
            return -1
    if n.kind == tk.PT_NAT_REC:
        extra = _substitute(p, extra, variable, replacement)
        if extra < 0:
            return -1
    return _node(p, n.kind, left, right, extra, n.value)


def _beta[C: Int, D: Int](mut b: _NumeralBuilder[C, D], source: Int) -> Int:
    if not _valid_node(b, source):
        return _failed(b)
    var application = b.bundle.nodes[source]
    if application.kind != tk.PT_APP or not _valid_node(b, application.left):
        return _failed(b)
    var function = b.bundle.nodes[application.left]
    if function.kind != tk.PT_LAM:
        return _failed(b)
    var target = _substitute(b.bundle, function.right, function.left, application.right)
    return _conversion(b, NC_BETA, source, target)


def _symm[C: Int, D: Int](mut b: _NumeralBuilder[C, D], proof: Int) -> Int:
    if not _valid_conversion(b, proof):
        return _failed(b)
    var c = b.conversions[proof]
    return _conversion(b, NC_SYMM, c.right, c.left, proof)


def _trans[C: Int, D: Int](mut b: _NumeralBuilder[C, D], first: Int, second: Int) -> Int:
    if not _valid_conversion(b, first) or not _valid_conversion(b, second):
        return _failed(b)
    var a = b.conversions[first]
    var z = b.conversions[second]
    if a.right != z.left:
        return _failed(b)
    return _conversion(b, NC_TRANS, a.left, z.right, first, second)


def _cong_left[C: Int, D: Int](mut b: _NumeralBuilder[C, D], proof: Int,
    argument: Int) -> Int:
    if not _valid_conversion(b, proof):
        return _failed(b)
    var c = b.conversions[proof]
    var left = _app(b.bundle, c.left, argument)
    var right = _app(b.bundle, c.right, argument)
    return _conversion(b, NC_CONG_LEFT, left, right, proof)


def _cong_right[C: Int, D: Int](mut b: _NumeralBuilder[C, D], function: Int,
    proof: Int) -> Int:
    if not _valid_conversion(b, proof):
        return _failed(b)
    var c = b.conversions[proof]
    var left = _app(b.bundle, function, c.left)
    var right = _app(b.bundle, function, c.right)
    return _conversion(b, NC_CONG_RIGHT, left, right, proof)


def _double_to_rec[C: Int, D: Int](mut b: _NumeralBuilder[C, D], argument: Int) -> Int:
    if b.double_delta < 0:
        var head = _named(b.bundle, 2)
        b.double_delta = _conversion(b, NC_DELTA, head, b.bundle.definition_bodies[2])
    var exposed = _cong_left(b, b.double_delta, argument)
    if not _valid_conversion(b, exposed):
        return _failed(b)
    var reduced = _beta(b, b.conversions[exposed].right)
    return _trans(b, exposed, reduced)


def _iota[C: Int, D: Int](mut b: _NumeralBuilder[C, D], source: Int) -> Int:
    if not _valid_node(b, source):
        return _failed(b)
    var n = b.bundle.nodes[source]
    if n.kind != tk.PT_NAT_REC or not _valid_node(b, n.left):
        return _failed(b)
    var major = b.bundle.nodes[n.left]
    var target = -1
    if major.kind == tk.PT_CONSTRUCTOR and major.extra == tk.NAT_FAMILY_ID and major.value == 0:
        target = n.right
    elif major.kind == tk.PT_APP and _valid_node(b, major.left):
        var constructor = b.bundle.nodes[major.left]
        if (constructor.kind == tk.PT_CONSTRUCTOR and constructor.extra == tk.NAT_FAMILY_ID
            and constructor.value == 1):
            var hypothesis = _node(b.bundle, tk.PT_NAT_REC, major.right, n.right, n.extra)
            target = _app(b.bundle, _app(b.bundle, n.extra, major.right), hypothesis)
    return _conversion(b, NC_NAT_IOTA, source, target)


def _denotation[C: Int, D: Int](mut b: _NumeralBuilder[C, D], value: Int) -> Int:
    if b.count < 0 or value < 0:
        return _failed(b)
    if value in b.denotations:
        return b.denotations.get(value, -1)
    # Record the word before mutating syntax, then build its binary prefixes
    # bottom-up. This avoids recursive construction with a borrowed word.
    var quotients = List[Int]()
    var bits = List[Int]()
    # Use explicit unsigned word operations: the pinned compile-time Int
    # division loop can truncate high bits. Cache keys remain nonnegative Int.
    var remaining = UInt64(value)
    while remaining > UInt64(0):
        quotients.append(Int(remaining))
        bits.append(Int(remaining & UInt64(1)))
        remaining = remaining >> UInt64(1)
    var result = b.denotations.get(0, -1)
    if result < 0:
        result = _zero(b.bundle)
        if result < 0:
            return _failed(b)
        b.denotations[0] = result
    for j in range(len(quotients)):
        var position = len(quotients) - j - 1
        var current = quotients[position]
        if current in b.denotations:
            result = b.denotations.get(current, -1)
        else:
            var doubled = _app(b.bundle, _named(b.bundle, 2), result)
            if doubled < 0:
                return _failed(b)
            result = _succ(b.bundle, doubled) if bits[position] == 1 else doubled
            if result < 0:
                return _failed(b)
            b.denotations[current] = result
    if result < 0:
        return _failed(b)
    return result


def _view[C: Int, D: Int](mut b: _NumeralBuilder[C, D], value: Int) -> Int:
    if b.count < 0 or value < 0:
        return _failed(b)
    if value in b.views:
        return b.views.get(value, -1)
    # An even view needs only the half's view. Build this carry chain from its
    # zero or odd leaf, without recursive mutation of the shared builder.
    var pending = List[Int]()
    var remaining = UInt64(value)
    while remaining > UInt64(0) and (remaining & UInt64(1)) == UInt64(0):
        pending.append(Int(remaining))
        remaining = remaining >> UInt64(1)
    pending.append(Int(remaining))
    var result = -1
    for j in range(len(pending)):
        var current = pending[len(pending) - j - 1]
        if current in b.views:
            result = b.views.get(current, -1)
            if not _valid_conversion(b, result):
                return _failed(b)
            continue
        var denotation = _denotation(b, current)
        if current == 0:
            result = _conversion(b, NC_REFL, denotation, _zero(b.bundle))
        elif (UInt64(current) & UInt64(1)) == UInt64(1):
            var previous = (_zero(b.bundle) if current == 1
                else _constructor_predecessor(b.bundle, denotation))
            if not _valid_node(b, previous):
                return _failed(b)
            var constructor = _succ(b.bundle, previous)
            if current == 1:
                var double_zero = _double_to_rec(b, previous)
                if not _valid_conversion(b, double_zero):
                    return _failed(b)
                var iota_zero = _iota(b, b.conversions[double_zero].right)
                var collapsed = _trans(b, double_zero, iota_zero)
                result = _cong_right(b, _successor(b.bundle), collapsed)
            else:
                result = _conversion(b, NC_REFL, denotation, constructor)
        else:
            var child = result
            if not _valid_conversion(b, child):
                return _failed(b)
            var predecessor = _constructor_predecessor(b.bundle, b.conversions[child].right)
            if not _valid_node(b, predecessor):
                return _failed(b)
            var child_lift = _cong_right(b, _named(b.bundle, 2), child)
            var constructor = _succ(b.bundle, predecessor)
            var double_successor = _double_to_rec(b, constructor)
            if not _valid_conversion(b, double_successor):
                return _failed(b)
            var rec_step = _iota(b, b.conversions[double_successor].right)
            if not _valid_conversion(b, rec_step):
                return _failed(b)
            var applied = b.bundle.nodes[b.conversions[rec_step].right]
            var step_predecessor = _beta(b, applied.left)
            var step_lift = _cong_left(b, step_predecessor, applied.right)
            if not _valid_conversion(b, step_lift):
                return _failed(b)
            var step_value = _beta(b, b.conversions[step_lift].right)
            var double_previous = _double_to_rec(b, predecessor)
            var folded_previous = _symm(b, double_previous)
            var folded_once = _cong_right(b, _successor(b.bundle), folded_previous)
            var folded_twice = _cong_right(b, _successor(b.bundle), folded_once)
            result = _trans(b, child_lift, double_successor)
            result = _trans(b, result, rec_step)
            result = _trans(b, result, step_lift)
            result = _trans(b, result, step_value)
            result = _trans(b, result, folded_twice)
        if not _valid_conversion(b, result):
            return _failed(b)
        b.views[current] = result
    return result


def make_numeral_view[capacity: Int = NUMERAL_VIEW_CAPACITY,
    derivation_capacity: Int = NUMERAL_CONVERSION_CAPACITY](value: Int) -> NumeralView[capacity, derivation_capacity]:
    """Build an untrusted one-constructor view with logarithmic binary carry.

    The budgets limit syntax and conversion data, never the literal domain or
    a runtime computation shape. The checker independently reconstructs binary
    denotations and validates all supplied definitions and conversion nodes.
    """
    var library = make_numeral_library[capacity]()
    var b = _NumeralBuilder[capacity, derivation_capacity](library.bundle.copy(),
        Array[NumeralConversion, derivation_capacity](fill=NumeralConversion(-1, -1, -1, -1, -1)),
        0, Dict[Int, Int](), Dict[Int, Int](), -1)
    var root = -1
    var denotation = -1
    var constructor = -1
    var predecessor = -1
    if b.bundle.count >= 0 and b.bundle.definition_count == 3 and value >= 0:
        root = _view(b, value)
        denotation = _denotation(b, value)
        if value == 0:
            constructor = _zero(b.bundle)
        elif _valid_conversion(b, root):
            constructor = b.conversions[root].right
            predecessor = _constructor_predecessor(b.bundle, constructor)
            if not _valid_node(b, predecessor):
                b.count = -1
                root = -1
        if (not _valid_conversion(b, root) or b.conversions[root].left != denotation
            or b.conversions[root].right != constructor):
            b.count = -1
            root = -1
    else:
        b.count = -1
    return NumeralView[capacity, derivation_capacity](b.bundle.copy(), b.conversions.copy(),
        b.count, root, value, denotation, constructor, predecessor)
