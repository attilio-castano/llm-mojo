"""Read raw binary Nat spines against independent fixed word patterns.

The expected widths and digits below are explicit mathematical boundary facts.
They use neither the production word codec nor unary Nat evaluation. These
controls run at compile time and natively so a shared decoder defect cannot
make a producer and its independently implemented checker agree on a wrong Nat.
"""
import llm_mojo.kernels.proof_term_kernel as tk
import llm_mojo.kernels.nat_headroom as headroom
import llm_mojo.kernels.nat_literal_arithmetic as literal
from llm_mojo.kernels.nat_numerals import make_numeral_view
from llm_mojo.kernels.finishing_proof import make_finishing_program

comptime WORD_ALL_ONES = 0
comptime WORD_LOW_ZERO = 1
comptime WORD_HIGH_ONLY = 2
comptime WORD_HIGH_AND_LOW = 3
comptime WORD_BIT_ONE_ZERO = 4


def _known_digit(position: Int, width: Int, pattern: Int) -> Int:
    if pattern == WORD_ALL_ONES:
        return 1
    if pattern == WORD_LOW_ZERO:
        return 0 if position == 0 else 1
    if pattern == WORD_HIGH_ONLY:
        return 1 if position == width - 1 else 0
    if pattern == WORD_HIGH_AND_LOW:
        return 1 if position == 0 or position == width - 1 else 0
    if pattern == WORD_BIT_ONE_ZERO:
        return 0 if position == 1 else 1
    return -1


def _node_exists[C: Int](p: tk.ProofTermBundle[C], node: Int) -> Bool:
    return node >= 0 and node < p.count and p.count <= C


def _fixed_raw_digits[C: Int](p: tk.ProofTermBundle[C], source: Int,
    width: Int, pattern: Int) -> Bool:
    # Outside-in traversal reads the least-significant digit first: the
    # optional actual successor surrounds exactly one application of Double.
    var current = source
    for position in range(width):
        if not _node_exists(p, current):
            return False
        var outer = p.nodes[current]
        var digit = 0
        if outer.kind == tk.PT_APP and _node_exists(p, outer.left):
            var head = p.nodes[outer.left]
            if (head.kind == tk.PT_CONSTRUCTOR and head.extra == tk.NAT_FAMILY_ID
                and head.value == 1):
                digit = 1
                current = outer.right
        if digit != _known_digit(position, width, pattern) or not _node_exists(p, current):
            return False
        var doubled = p.nodes[current]
        if doubled.kind != tk.PT_APP or not _node_exists(p, doubled.left):
            return False
        var double = p.nodes[doubled.left]
        if double.kind != tk.PT_DEFINED_REF or double.value != 2:
            return False
        current = doubled.right
    if not _node_exists(p, current):
        return False
    var zero = p.nodes[current]
    return (zero.kind == tk.PT_CONSTRUCTOR and zero.extra == tk.NAT_FAMILY_ID
        and zero.value == 0)


def _addition_left[C: Int](p: tk.ProofTermBundle[C], source: Int) -> Int:
    if not _node_exists(p, source):
        return -1
    var applied = p.nodes[source]
    if (applied.kind != tk.PT_APP or not _node_exists(p, applied.left)
        or not _fixed_raw_digits(p, applied.right, 0, WORD_ALL_ONES)):
        return -1
    var first = p.nodes[applied.left]
    if first.kind != tk.PT_APP or not _node_exists(p, first.left):
        return -1
    var operation = p.nodes[first.left]
    if operation.kind != tk.PT_DEFINED_REF or operation.value != 3:
        return -1
    return first.right


def _word_case(value: Int, width: Int, pattern: Int,
    predecessor_width: Int, predecessor_pattern: Int) -> Bool:
    var bits = headroom._bits(value)
    if len(bits) != width:
        return False
    for position in range(width):
        if bits[position] != _known_digit(position, width, pattern):
            return False
    var proof = headroom.make_headroom_proof(value, 0, value)
    if (not _fixed_raw_digits(proof.bundle, proof.limit, width, pattern)
        or not _fixed_raw_digits(proof.bundle,
            _addition_left(proof.bundle, proof.source), width, pattern)
        or not _fixed_raw_digits(proof.bundle, proof.sum, width, pattern)
        or not _fixed_raw_digits(proof.bundle, proof.slack, 0, WORD_ALL_ONES)):
        return False
    var view = make_numeral_view(value)
    if (view.root < 0 or not _fixed_raw_digits(view.bundle, view.denotation, width, pattern)
        or not _fixed_raw_digits(view.bundle, view.predecessor,
            predecessor_width, predecessor_pattern)):
        return False
    var arithmetic = literal.make_literal_arithmetic_proof[2048](tk.PT_ADD, value, 0, value)
    var borrowed = literal._predecessor_denotation(arithmetic.bundle, value)
    if (not _fixed_raw_digits(arithmetic.bundle,
            _addition_left(arithmetic.bundle, arithmetic.source), width, pattern)
        or not _fixed_raw_digits(arithmetic.bundle, arithmetic.result, width, pattern)
        or not _fixed_raw_digits(arithmetic.bundle, borrowed,
            predecessor_width, predecessor_pattern)):
        return False
    var empty = tk.empty_proof_terms[16]()
    var program = make_finishing_program(1, 1, 1)
    var state = tk._make_state[16, 1024](program, empty)
    var double = tk._put(state, tk.PT_DEFINED_REF, value=2)
    var expected = tk._raw_literal_denotation(state, value, double)
    var previous = tk._raw_literal_predecessor_denotation(state, value, double)
    return (_fixed_raw_digits(state.syntax, expected, width, pattern)
        and _fixed_raw_digits(state.syntax, previous,
            predecessor_width, predecessor_pattern))


def headroom_word_denotations_accepts() -> Bool:
    # Int.MAX, Int.MAX-1, 2^32, 2^32+1 and 2^62 are written directly.
    # Their predecessors' widths/patterns are also specified independently.
    return (_word_case(Int(9223372036854775807), 63, WORD_ALL_ONES, 63, WORD_LOW_ZERO)
        and _word_case(Int(9223372036854775806), 63, WORD_LOW_ZERO, 63, WORD_BIT_ONE_ZERO)
        and _word_case(Int(4294967296), 33, WORD_HIGH_ONLY, 32, WORD_ALL_ONES)
        and _word_case(Int(4294967297), 33, WORD_HIGH_AND_LOW, 33, WORD_HIGH_ONLY)
        and _word_case(Int(4611686018427387904), 63, WORD_HIGH_ONLY, 62, WORD_ALL_ONES))
