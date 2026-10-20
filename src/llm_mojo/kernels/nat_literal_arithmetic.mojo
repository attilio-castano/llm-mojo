"""Untrusted raw Nat certificates for optional literal arithmetic conversion.

Native arithmetic may suggest a candidate value, but this producer supplies no
arithmetic judgment. Its certificate is an ordinary reflexivity term over the
actual recursive operation. The kernel must bind both endpoints independently
and check that reflexivity has the requested equality type by raw definitional
conversion. No compact numeral, arithmetic opcode or arithmetic theorem occurs
in the resulting bundle. Storage bounds certificate construction, never a
runtime computation shape or the mathematical Nat domain.
"""
from std.collections import Array
import llm_mojo.kernels.proof_term_syntax as tk
from llm_mojo.kernels.nat_numerals import make_numeral_library
from llm_mojo.kernels.nat_arithmetic import (
    declare_nat_arithmetic_definitions, declare_nat_arithmetic_pred,
    declare_nat_arithmetic_base,
)
from llm_mojo.kernels.nat_arithmetic_division import declare_nat_arithmetic_division
from llm_mojo.kernels.nat_arithmetic_terms import _empty_builder

comptime LITERAL_ARITHMETIC_CAPACITY = 1024


@fieldwise_init
struct LiteralArithmeticProof[capacity: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.capacity]
    var operation: Int
    var left_value: Int
    var right_value: Int
    var candidate_value: Int
    var source: Int
    var result: Int
    var proof: Int


def _node[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _zero[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)


def _successor[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1)


def _app[C: Int](mut p: tk.ProofTermBundle[C], function: Int, argument: Int) -> Int:
    if function < 0 or argument < 0:
        return -1
    return _node(p, tk.PT_APP, function, argument)


def _named[C: Int](mut p: tk.ProofTermBundle[C], definition: Int) -> Int:
    return _node(p, tk.PT_DEFINED_REF, value=definition)


def _app2[C: Int](mut p: tk.ProofTermBundle[C], function: Int,
    left: Int, right: Int) -> Int:
    return _app(p, _app(p, function, left), right)


def _copy_arithmetic_data[C: Int](mut p: tk.ProofTermBundle[C]) -> Bool:
    """Copy actual data producers, remapping their six definition identities.

    The numeral definitions already occupy indices zero through two. This
    translator is untrusted; the checker pins the copied definitions afresh.
    Only the source library's literal zero and successor abbreviations are
    elaborated, to actual constructors. All other arithmetic/proof opcodes fail.
    """
    if p.count < 0 or p.definition_count != 3:
        return False
    var source = _empty_builder[C]()
    if not declare_nat_arithmetic_definitions(source):
        return False
    if declare_nat_arithmetic_division(source) != 3:
        return False
    if not declare_nat_arithmetic_pred(source):
        return False
    if declare_nat_arithmetic_base(source) != 5:
        return False
    if source.bundle.count < 0 or source.bundle.definition_count != 6:
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
        if n.kind == tk.PT_NAT_REC:
            if n.extra < 0 or n.extra >= j or copied[n.extra] < 0:
                return False
            extra = copied[n.extra]
        var result: Int
        if n.kind == tk.PT_LITERAL:
            if n.value != 0:
                return False
            result = _zero(p)
        elif n.kind == tk.PT_SUCC:
            result = _app(p, _successor(p), left)
        elif n.kind == tk.PT_DEFINED_REF:
            if n.value < 0 or n.value >= 6:
                return False
            result = _named(p, n.value + 3)
        elif (n.kind == tk.PT_TYPE or n.kind == tk.PT_PROP or n.kind == tk.PT_NAT
            or n.kind == tk.PT_VAR or n.kind == tk.PT_PI or n.kind == tk.PT_LAM
            or n.kind == tk.PT_APP or n.kind == tk.PT_NAT_REC):
            result = _node(p, n.kind, left, right, extra, n.value)
        else:
            return False
        if result < 0:
            return False
        copied[j] = result
    for definition in range(6):
        var declared = source.bundle.definition_types[definition]
        var body = source.bundle.definition_bodies[definition]
        if (declared < 0 or declared >= source.bundle.count
            or body < 0 or body >= source.bundle.count):
            return False
        if tk.pt_declare_definition(p, copied[declared], copied[body]) != definition + 3:
            return False
    return True


def _from_bits[C: Int](mut p: tk.ProofTermBundle[C], bits: List[Int], length: Int) -> Int:
    var result = _zero(p)
    var double = _named(p, 2)
    var successor = _successor(p)
    for j in range(length):
        result = _app(p, double, result)
        if bits[length - j - 1] == 1:
            result = _app(p, successor, result)
        if result < 0:
            return -1
    return result


def _denotation[C: Int](mut p: tk.ProofTermBundle[C], value: Int) -> Int:
    if value < 0:
        return -1
    var bits = List[Int]()
    # Keep machine-word decoding explicit; Int division loops truncate high
    # bits in the pinned compile-time evaluator. Digits alone become Int.
    var remaining = UInt64(value)
    while remaining > UInt64(0):
        bits.append(Int(remaining & UInt64(1)))
        remaining = remaining >> UInt64(1)
    return _from_bits(p, bits, len(bits))


def _predecessor_denotation[C: Int](mut p: tk.ProofTermBundle[C], value: Int) -> Int:
    """Propose E(value - 1) by binary borrow, without host-word subtraction."""
    if value <= 0:
        return -1
    var bits = List[Int]()
    # Keep machine-word decoding explicit; Int division loops truncate high
    # bits in the pinned compile-time evaluator. Digits alone become Int.
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
    return _from_bits(p, bits, length)


def _project[C: Int](mut p: tk.ProofTermBundle[C], state: Int, component: Int) -> Int:
    var nat = _node(p, tk.PT_NAT)
    var q = _node(p, tk.PT_VAR, nat)
    var r = _node(p, tk.PT_VAR, nat)
    var countdown = _node(p, tk.PT_VAR, nat)
    var selected = q if component == 0 else r
    var selector = _node(p, tk.PT_LAM, q,
        _node(p, tk.PT_LAM, r, _node(p, tk.PT_LAM, countdown, selected)))
    return _app(p, state, selector)


def make_literal_arithmetic_proof[capacity: Int = LITERAL_ARITHMETIC_CAPACITY](
    operation: Int, left: Int, right: Int, candidate: Int,
) -> LiteralArithmeticProof[capacity]:
    """Suggest a raw reflexivity certificate for a closed arithmetic result.

    Add/Mul/Scale use left and right in that order; Scale's left value is its
    coefficient. Base uses left=stride and right=round. Div/Mod use left as the
    dividend and right as the positive divisor. Candidate generation belongs
    to the caller and grants no authority. Every requested field is retained
    for the independent checker to bind to the actual operation and endpoints.
    """
    var numerals = make_numeral_library[capacity]()
    var p = numerals.bundle.copy()
    var source = -1
    var result = -1
    var proof = -1
    var supported = (operation == tk.PT_ADD or operation == tk.PT_MUL
        or operation == tk.PT_SCALE or operation == tk.PT_BASE
        or operation == tk.PT_DIV or operation == tk.PT_MOD)
    if (supported and left >= 0 and right >= 0 and candidate >= 0
        and ((operation != tk.PT_DIV and operation != tk.PT_MOD) or right > 0)
        and _copy_arithmetic_data(p)):
        var a = _denotation(p, left)
        var b = _denotation(p, right)
        if operation == tk.PT_ADD:
            source = _app2(p, _named(p, 3), a, b)
        elif operation == tk.PT_MUL or operation == tk.PT_SCALE:
            source = _app2(p, _named(p, 4), a, b)
        elif operation == tk.PT_BASE:
            source = _app2(p, _named(p, 8), a, b)
        else:
            var predecessor = _predecessor_denotation(p, right)
            var state = _app2(p, _named(p, 6), predecessor, a)
            source = _project(p, state, 0 if operation == tk.PT_DIV else 1)
        result = _denotation(p, candidate)
        if source >= 0 and result >= 0:
            proof = _node(p, tk.PT_REFL, source)
            p.root = proof
    if source < 0 or result < 0 or proof < 0:
        p.count = -1
    return LiteralArithmeticProof[capacity](p.copy(), operation, left, right,
        candidate, source, result, proof)
