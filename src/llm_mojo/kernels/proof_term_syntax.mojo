"""Untrusted syntax data for the restricted Mojo-native proof-term kernel.

Constructors here build syntax only. Definition and proof admission belongs to
proof_term_kernel; no helper in this module grants a typing judgment.
"""
from std.collections import Array

comptime PT_TYPE = 0
comptime PT_PROP = 1
comptime PT_NAT = 2
comptime PT_VAR = 3
comptime PT_PI = 4
comptime PT_LAM = 5
comptime PT_APP = 6
comptime PT_LITERAL = 7
comptime PT_ADD = 8
comptime PT_SCALE = 9
comptime PT_BASE = 10
comptime PT_EQ = 11
comptime PT_LE = 12
comptime PT_LT = 13
comptime PT_DONE = 14  # Retired: rejected by the syntax checker.
comptime PT_CONST = 15  # Retired: every theorem export requires a checked proof body.
comptime PT_NAT_IND = 16  # Retired: use the definition-derived Nat eliminator.
comptime PT_SUBST = 17
comptime PT_REFL = 18
comptime PT_AFFINE_EQ = 19
comptime PT_FAMILY = 20
comptime PT_CONSTRUCTOR = 21
comptime PT_ELIMINATOR = 22
comptime PT_DIV = 23
comptime PT_MOD = 24
comptime PT_LITERAL_LE = 25
comptime PT_DEFINED_REF = 26
comptime PT_SUCC = 27
comptime PT_NAT_REC = 28
comptime PT_MUL = 29
comptime PT_ARITHMETIC = 30
comptime PROP_FAMILY_ID = 0
comptime NAT_FAMILY_ID = 1

comptime DEC_BASE_ZERO = 0
comptime DEC_BASE_STEP = 1
comptime DEC_NAT_NONNEG = 2
comptime DEC_LE_ADD = 3
comptime DEC_LE_TRANS = 4
comptime DEC_DONE_STOP = 5  # Retired: no trusted constructor signature.
comptime DEC_DONE_STEP = 6  # Retired: no trusted constructor signature.
comptime DEC_ORDER_CASES = 7
comptime DEC_LT_ADD = 8
comptime DEC_LE_LT_TRANS = 9
comptime DEC_LT_LE_TRANS = 10
comptime DEC_LE_SCALE = 11
comptime DEC_LT_SCALE_CANCEL = 12
comptime DEC_LT_TO_LE_SUCC = 13
comptime DEC_DIV_DECOMP = 14
comptime DEC_MOD_LT = 15
comptime DEC_DIV_QUOTIENT_UNIQUE = 16
comptime DEC_DIV_REMAINDER_UNIQUE = 17
# Sparse export slots: retired IDs 5/6 stay vacant. Parameterized signatures
# are checked per export node, not cached under an unsound ID key.
comptime PROOF_TERM_DECLARATIONS = 18


@fieldwise_init
struct ProofTerm(ImplicitlyCopyable):
    var kind: Int
    var left: Int
    var right: Int
    var extra: Int
    var value: Int


@fieldwise_init
struct ProofTermJudgment(ImplicitlyCopyable):
    var root: Int
    var expected: Int


@fieldwise_init
struct ProofTermBundle[capacity: Int](Copyable, Movable):
    var nodes: Array[ProofTerm, Self.capacity]
    var count: Int
    var root: Int
    # Family 0: -1 means absent; otherwise the number of Nat indices.
    var family_arity: Int
    var constructor_count: Int
    var constructor_types: Array[Int, Self.capacity]
    # Family 1 is the zero-index Type family used as the native Nat carrier.
    var nat_constructor_count: Int
    var nat_constructor_types: Array[Int, Self.capacity]
    # Closed, acyclic ordinary definitions. References name earlier entries.
    var definition_count: Int
    var definition_types: Array[Int, Self.capacity]
    var definition_bodies: Array[Int, Self.capacity]


def empty_proof_terms[capacity: Int]() -> ProofTermBundle[capacity]:
    var nodes = Array[ProofTerm, capacity](fill=ProofTerm(-1, -1, -1, -1, -1))
    var result = ProofTermBundle[capacity](nodes^, 0, -1, -1, 0,
        Array[Int, capacity](fill=-1), 0, Array[Int, capacity](fill=-1),
        0, Array[Int, capacity](fill=-1), Array[Int, capacity](fill=-1))
    # Untrusted declaration data, checked afresh before every admission. These
    # nodes encode zero : Nat and succ : Nat -> Nat; they supply no proof rule.
    comptime if capacity >= 3:
        var nat = pt_add(result, PT_NAT)
        var predecessor = pt_add(result, PT_VAR, nat)
        var successor_type = pt_add(result, PT_PI, predecessor, nat)
        if nat >= 0 and predecessor >= 0 and successor_type >= 0:
            result.nat_constructor_count = 2
            result.nat_constructor_types[0] = nat
            result.nat_constructor_types[1] = successor_type
    return result^


def pt_add[capacity: Int](mut p: ProofTermBundle[capacity], kind: Int,
    left: Int = -1, right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    if p.count < 0 or p.count > capacity:
        return -1
    # A variable is a binder identity, not a hash-consed name/type pair.
    if kind != PT_VAR:
        for j in range(p.count):
            var n = p.nodes[j]
            if (n.kind == kind and n.left == left and n.right == right
                and n.extra == extra and n.value == value):
                return j
    if p.count == capacity:
        return -1
    var result = p.count
    p.nodes[result] = ProofTerm(kind, left, right, extra, value)
    p.count += 1
    return result


def pt_declare_family[capacity: Int](mut p: ProofTermBundle[capacity], arity: Int) -> Bool:
    """Declare family 0 of type Nat -> ... -> Nat -> Prop.

    This records untrusted declaration data; it does not admit the definition.
    The checker validates the complete constructor environment before use.
    """
    if p.family_arity != -1 or p.constructor_count != 0 or arity < 0 or arity > capacity:
        return False
    p.family_arity = arity
    return True


def pt_declare_constructor[capacity: Int](mut p: ProofTermBundle[capacity], type: Int) -> Int:
    if (p.family_arity < 0 or p.constructor_count < 0 or p.constructor_count >= capacity
        or type < 0 or type >= p.count):
        return -1
    var index = p.constructor_count
    p.constructor_types[index] = type
    p.constructor_count += 1
    return index


def pt_declare_definition[capacity: Int](mut p: ProofTermBundle[capacity],
    type: Int, body: Int) -> Int:
    """Record untrusted closed definition data; the kernel checks it later."""
    if (p.definition_count < 0 or p.definition_count >= capacity
        or type < 0 or type >= p.count or body < 0 or body >= p.count):
        return -1
    var index = p.definition_count
    p.definition_types[index] = type
    p.definition_bodies[index] = body
    p.definition_count += 1
    return index
