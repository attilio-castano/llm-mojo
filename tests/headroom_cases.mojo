"""Independent boundaries and malformed raw Nat headroom certificates.

Decimal boundary expectations do not reuse native addition or subtraction.
Mutation construction fails open: a missing anchor returns True so that the
rejection harness cannot mistake a failed test setup for a rejected proof.
"""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.nat_headroom import (
    HeadroomProof, make_headroom_proof, HEADROOM_DIGIT_START, HEADROOM_DOUBLE_ADD,
)

comptime HEAD_FALSE_TIGHT = 0
comptime HEAD_FALSE_CARRY = 1
comptime HEAD_FALSE_LONG_CARRY = 2
comptime HEAD_FALSE_MAX_OVERFLOW = 3
comptime HEAD_REQUEST_CAP = 4
comptime HEAD_REQUEST_STRIDE = 5
comptime HEAD_REQUEST_LIMIT = 6
comptime HEAD_STORED_CAP = 7
comptime HEAD_STORED_STRIDE = 8
comptime HEAD_STORED_LIMIT = 9
comptime HEAD_SOURCE_ENDPOINT = 10
comptime HEAD_LIMIT_ENDPOINT = 11
comptime HEAD_SUM_ENDPOINT = 12
comptime HEAD_SLACK_ENDPOINT = 13
comptime HEAD_SUM_EQUALITY = 14
comptime HEAD_SLACK_EQUALITY = 15
comptime HEAD_BOUND_ENDPOINT = 16
comptime HEAD_PROOF_ENDPOINT = 17
comptime HEAD_ROOT_BINDING = 18
comptime HEAD_NEGATIVE_COUNT = 19
comptime HEAD_OVERSIZED_COUNT = 20
comptime HEAD_MISSING_DEFINITION = 21
comptime HEAD_EXTRA_MALFORMED_DEFINITION = 22
comptime HEAD_WRONG_ADD_DEFINITION = 23
comptime HEAD_ALTERED_NAT = 24
comptime HEAD_CYCLIC_DEFINITION = 25
comptime HEAD_FORWARD_DEFINITION = 26
comptime HEAD_OPEN_DEFINITION = 27
comptime HEAD_WRONG_DECLARED_TYPE = 28
comptime HEAD_UNUSED_MALFORMED = 29
comptime HEAD_BOOTSTRAP_AFFINE = 30
comptime HEAD_BOOTSTRAP_ORDER = 31
comptime HEAD_BOOTSTRAP_LITERAL = 32
comptime HEAD_BOOTSTRAP_SUCCESSOR = 33
comptime HEAD_BOOTSTRAP_OPERATOR = 34
comptime HEAD_BOOTSTRAP_EXPORT = 35
comptime HEAD_BOOTSTRAP_PRIVATE = 36
comptime HEAD_UNKNOWN_SYNTAX = 37
comptime HEAD_NEGATIVE_CAP = 38
comptime HEAD_NEGATIVE_STRIDE = 39
comptime HEAD_NEGATIVE_LIMIT = 40
comptime HEAD_SOURCE_BOUNDARY = 41
comptime HEAD_PROOF_BOUNDARY = 42
comptime HEAD_BOUND_BOUNDARY = 43
comptime HEAD_SUM_EQUALITY_BOUNDARY = 44
comptime HEAD_SLACK_EQUALITY_BOUNDARY = 45
comptime HEAD_UNUSED_FALSE_REFLEXIVITY = 46
comptime HEAD_NEGATIVE_DEFINITIONS = 47
comptime HEAD_OVERSIZED_DEFINITIONS = 48
comptime HEAD_WRONG_CARRY_EQUALITY = 49
comptime HEAD_WRONG_CARRY_LEMMA = 50
comptime HEAD_WRONG_DOUBLE_LEMMA = 51
comptime HEADROOM_MUTANTS = 52


def _raw_zero[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return tk.pt_add(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)


def headroom_has_raw_syntax[C: Int](proof: HeadroomProof[C]) -> Bool:
    if proof.bundle.count < 0 or proof.bundle.count > C:
        return False
    for index in range(proof.bundle.count):
        var kind = proof.bundle.nodes[index].kind
        if (kind == tk.PT_LITERAL or kind == tk.PT_SUCC
            or kind == tk.PT_ADD or kind == tk.PT_MUL or kind == tk.PT_SCALE
            or kind == tk.PT_BASE or kind == tk.PT_DIV or kind == tk.PT_MOD
            or kind == tk.PT_LE or kind == tk.PT_LT
            or kind == tk.PT_AFFINE_EQ or kind == tk.PT_LITERAL_LE
            or kind == tk.PT_ARITHMETIC or kind == tk.PT_CONST
            or kind == 31 or kind == 32):
            return False
    return proof.bundle.root == proof.proof


def headroom_accepts(cap: Int, stride: Int, limit: Int) -> Bool:
    var proof = make_headroom_proof(cap, stride, limit)
    return (headroom_has_raw_syntax(proof)
        and tk.check_native_headroom_request(proof, cap, stride, limit))


def headroom_small_cases_accepts() -> Bool:
    return (headroom_accepts(0, 0, 0)
        and headroom_accepts(0, 1, 1)
        and headroom_accepts(1, 0, 1)
        and headroom_accepts(1, 1, 2)
        and headroom_accepts(1, 1, 3)
        and headroom_accepts(2, 3, 5)
        and headroom_accepts(3, 2, 5)
        and headroom_accepts(7, 8, 15)
        and headroom_accepts(8, 7, 15)
        and headroom_accepts(15, 1, 16)
        and headroom_accepts(16, 15, 31)
        and headroom_accepts(37, 15, 52)
        and headroom_accepts(37, 15, 53))


def headroom_word_boundaries_accepts() -> Bool:
    # These are exact mathematical expectations, including carries where
    # native Mojo constant-folded word arithmetic previously gave bad hints.
    return (headroom_accepts(2147483647, 2147483647, Int(4294967294))
        and headroom_accepts(Int(4294967295), 1, Int(4294967296))
        and headroom_accepts(Int(4294967295), Int(4294967295), Int(8589934590))
        and headroom_accepts(Int(9223372036854775806), 1, Int(9223372036854775807))
        and headroom_accepts(Int(9223372036854775807), 0, Int(9223372036854775807))
        and headroom_accepts(2147483647, 195, Int(9223372036854775807)))


def _beta_wrap[C: Int](mut p: tk.ProofTermBundle[C], term: Int) -> Int:
    var nat = tk.pt_add(p, tk.PT_NAT)
    var ignored = tk.pt_add(p, tk.PT_VAR, nat)
    var function = tk.pt_add(p, tk.PT_LAM, ignored, term)
    return tk.pt_add(p, tk.PT_APP, function, _raw_zero(p))


def headroom_alternate_proofs_accepts() -> Bool:
    # Administrative beta expansion gives distinct valid proof terms; the
    # checker must validate their types instead of matching a producer trace.
    var full = make_headroom_proof(37, 15, 52)
    full.proof = _beta_wrap(full.bundle, full.proof)
    full.bundle.root = full.proof
    if full.proof < 0 or not tk.check_native_headroom(full):
        return False
    var closed = make_headroom_proof(37, 15, 52)
    closed.bound = _beta_wrap(closed.bundle, closed.bound)
    return closed.bound >= 0 and tk.check_native_headroom(closed)


def _replace_with_zero_function[C: Int](mut proof: HeadroomProof[C], definition: Int) -> Bool:
    if definition < 0 or definition >= proof.bundle.definition_count:
        return False
    var declared = proof.bundle.definition_types[definition]
    if declared < 0 or declared >= proof.bundle.count:
        return False
    var outer = proof.bundle.nodes[declared]
    if outer.kind != tk.PT_PI or outer.right < 0 or outer.right >= proof.bundle.count:
        return False
    var inner = proof.bundle.nodes[outer.right]
    if inner.kind != tk.PT_PI:
        return False
    var zero = _raw_zero(proof.bundle)
    var body = tk.pt_add(proof.bundle, tk.PT_LAM, inner.left, zero)
    body = tk.pt_add(proof.bundle, tk.PT_LAM, outer.left, body)
    if body < 0:
        return False
    proof.bundle.definition_bodies[definition] = body
    return True


def headroom_mutant_accepts(mode: Int) -> Bool:
    if mode == HEAD_FALSE_TIGHT:
        return tk.check_native_headroom(make_headroom_proof(37, 15, 51))
    if mode == HEAD_FALSE_CARRY:
        return tk.check_native_headroom(make_headroom_proof(Int(4294967295), 1, Int(4294967295)))
    if mode == HEAD_FALSE_LONG_CARRY:
        return tk.check_native_headroom(make_headroom_proof(
            Int(4294967295), Int(4294967295), Int(8589934589)))
    if mode == HEAD_FALSE_MAX_OVERFLOW:
        # Mathematical sum MAX+1 exceeds this representable limit; no wrapped
        # native proposal may turn the false statement into a true theorem.
        return tk.check_native_headroom(make_headroom_proof(
            Int(9223372036854775807), 1, Int(9223372036854775807)))
    if mode == HEAD_NEGATIVE_CAP:
        return tk.check_native_headroom(make_headroom_proof(-1, 15, 52))
    if mode == HEAD_NEGATIVE_STRIDE:
        return tk.check_native_headroom(make_headroom_proof(37, -1, 52))
    if mode == HEAD_NEGATIVE_LIMIT:
        return tk.check_native_headroom(make_headroom_proof(37, 15, -1))
    var baseline = make_headroom_proof(37, 15, 52)
    return _mutated_headroom_accepts(baseline, mode)


def _mutated_headroom_accepts(baseline: HeadroomProof[8192], mode: Int) -> Bool:
    var proof = baseline.copy()
    if (proof.bundle.count <= 0 or proof.bundle.definition_count < 9
        or proof.proof < 0 or proof.bound < 0 or proof.source < 0
        or proof.limit < 0 or proof.sum < 0 or proof.slack < 0
        or proof.sum_equality < 0 or proof.slack_equality < 0):
        return True
    if mode == HEAD_REQUEST_CAP:
        return tk.check_native_headroom_request(proof, 38, 15, 52)
    if mode == HEAD_REQUEST_STRIDE:
        return tk.check_native_headroom_request(proof, 37, 16, 52)
    if mode == HEAD_REQUEST_LIMIT:
        return tk.check_native_headroom_request(proof, 37, 15, 51)
    var zero = _raw_zero(proof.bundle)
    if zero < 0:
        return True
    if mode == HEAD_STORED_CAP:
        proof.count_limit = 38
    elif mode == HEAD_STORED_STRIDE:
        proof.stride = 16
    elif mode == HEAD_STORED_LIMIT:
        proof.integer_limit = 51
    elif mode == HEAD_SOURCE_ENDPOINT:
        proof.source = zero
    elif mode == HEAD_LIMIT_ENDPOINT:
        proof.limit = zero
    elif mode == HEAD_SUM_ENDPOINT:
        proof.sum = zero
    elif mode == HEAD_SLACK_ENDPOINT:
        # This tight example has slack zero; use source so the mutation changes it.
        proof.slack = proof.source
    elif mode == HEAD_SUM_EQUALITY:
        proof.sum_equality = tk.pt_add(proof.bundle, tk.PT_REFL, zero)
    elif mode == HEAD_SLACK_EQUALITY:
        proof.slack_equality = tk.pt_add(proof.bundle, tk.PT_REFL, zero)
    elif mode == HEAD_BOUND_ENDPOINT:
        proof.bound = proof.sum_equality
    elif mode == HEAD_PROOF_ENDPOINT:
        proof.proof = proof.bound
        proof.bundle.root = proof.proof
    elif mode == HEAD_ROOT_BINDING:
        proof.bundle.root = proof.bound
    elif mode == HEAD_NEGATIVE_COUNT:
        proof.bundle.count = -1
    elif mode == HEAD_OVERSIZED_COUNT:
        proof.bundle.count = 8193
    elif mode == HEAD_MISSING_DEFINITION:
        proof.bundle.definition_count = 8
    elif mode == HEAD_EXTRA_MALFORMED_DEFINITION:
        var nat = tk.pt_add(proof.bundle, tk.PT_NAT)
        var bad_body = tk.pt_add(proof.bundle, tk.PT_APP, nat, zero)
        if tk.pt_declare_definition(proof.bundle, nat, bad_body) < 0:
            return True
    elif mode == HEAD_WRONG_ADD_DEFINITION:
        # Num zero/succ/double occupy0..2; actual Add is definition3.
        if not _replace_with_zero_function(proof, 3):
            return True
    elif mode == HEAD_ALTERED_NAT:
        proof.bundle.nat_constructor_types[1] = proof.bundle.nat_constructor_types[0]
    elif mode == HEAD_CYCLIC_DEFINITION:
        proof.bundle.definition_bodies[0] = tk.pt_add(proof.bundle,
            tk.PT_DEFINED_REF, value=0)
    elif mode == HEAD_FORWARD_DEFINITION:
        proof.bundle.definition_bodies[0] = tk.pt_add(proof.bundle,
            tk.PT_DEFINED_REF, value=2)
    elif mode == HEAD_OPEN_DEFINITION:
        proof.bundle.definition_bodies[0] = tk.pt_add(proof.bundle,
            tk.PT_VAR, tk.pt_add(proof.bundle, tk.PT_NAT))
    elif mode == HEAD_WRONG_DECLARED_TYPE:
        proof.bundle.definition_types[1] = tk.pt_add(proof.bundle, tk.PT_NAT)
    elif mode == HEAD_UNUSED_MALFORMED:
        _ = tk.pt_add(proof.bundle, tk.PT_APP, tk.pt_add(proof.bundle, tk.PT_NAT), zero)
    elif mode == HEAD_BOOTSTRAP_AFFINE:
        _ = tk.pt_add(proof.bundle, tk.PT_AFFINE_EQ, zero, zero)
    elif mode == HEAD_BOOTSTRAP_ORDER:
        _ = tk.pt_add(proof.bundle, tk.PT_LITERAL_LE, zero, zero)
    elif mode == HEAD_BOOTSTRAP_LITERAL:
        _ = tk.pt_add(proof.bundle, tk.PT_LITERAL, value=0)
    elif mode == HEAD_BOOTSTRAP_SUCCESSOR:
        _ = tk.pt_add(proof.bundle, tk.PT_SUCC, zero)
    elif mode == HEAD_BOOTSTRAP_OPERATOR:
        _ = tk.pt_add(proof.bundle, tk.PT_ADD, zero, zero)
    elif mode == HEAD_BOOTSTRAP_EXPORT:
        _ = tk.pt_add(proof.bundle, tk.PT_ARITHMETIC, value=tk.DEC_LE_TRANS)
    elif mode == HEAD_BOOTSTRAP_PRIVATE:
        _ = tk.pt_add(proof.bundle, 31, value=0)
    elif mode == HEAD_UNKNOWN_SYNTAX:
        _ = tk.pt_add(proof.bundle, 99999)
    elif mode == HEAD_SOURCE_BOUNDARY:
        proof.source = proof.bundle.count
    elif mode == HEAD_PROOF_BOUNDARY:
        proof.proof = proof.bundle.count
        proof.bundle.root = proof.proof
    elif mode == HEAD_BOUND_BOUNDARY:
        proof.bound = proof.bundle.count
    elif mode == HEAD_SUM_EQUALITY_BOUNDARY:
        proof.sum_equality = proof.bundle.count
    elif mode == HEAD_SLACK_EQUALITY_BOUNDARY:
        proof.slack_equality = proof.bundle.count
    elif mode == HEAD_UNUSED_FALSE_REFLEXIVITY:
        # A well-typed unused Nat equality cannot conceal a malformed term.
        var false_refl = tk.pt_add(proof.bundle, tk.PT_REFL, zero)
        _ = tk.pt_add(proof.bundle, tk.PT_APP, false_refl, zero)
    elif mode == HEAD_NEGATIVE_DEFINITIONS:
        proof.bundle.definition_count = -1
    elif mode == HEAD_OVERSIZED_DEFINITIONS:
        proof.bundle.definition_count = 8193
    elif mode == HEAD_WRONG_CARRY_EQUALITY:
        var carry = make_headroom_proof(Int(4294967295), 1, Int(4294967296))
        if carry.sum_equality < 0 or carry.slack_equality < 0:
            return True
        var carry_zero = _raw_zero(carry.bundle)
        if carry_zero < 0:
            return True
        carry.sum_equality = tk.pt_add(carry.bundle, tk.PT_REFL, carry_zero)
        return tk.check_native_headroom(carry)
    elif mode == HEAD_WRONG_CARRY_LEMMA:
        if proof.bundle.definition_count <= HEADROOM_DIGIT_START + 1:
            return True
        # Same two input digits, but carry-in one must prove a different sum.
        # Every lemma body is checked even if this concrete request never uses it.
        proof.bundle.definition_bodies[HEADROOM_DIGIT_START + 1] = (
            proof.bundle.definition_bodies[HEADROOM_DIGIT_START])
    elif mode == HEAD_WRONG_DOUBLE_LEMMA:
        if proof.bundle.definition_count <= HEADROOM_DOUBLE_ADD:
            return True
        # Closed reflexivity is a valid proof, but does not prove the declared
        # universally quantified doubling theorem.
        proof.bundle.definition_bodies[HEADROOM_DOUBLE_ADD] = tk.pt_add(
            proof.bundle, tk.PT_REFL, zero)
    else:
        return True
    return tk.check_native_headroom(proof)


def headroom_rejection_controls_accepts() -> Bool:
    # Reuse the same untrusted data/proof library in the native mutation matrix;
    # each checker call still makes its own fresh isolated typing environment.
    var baseline = make_headroom_proof(37, 15, 52)
    for mode in range(HEADROOM_MUTANTS):
        var accepted: Bool
        if (mode <= HEAD_FALSE_MAX_OVERFLOW
            or (mode >= HEAD_NEGATIVE_CAP and mode <= HEAD_NEGATIVE_LIMIT)):
            accepted = headroom_mutant_accepts(mode)
        else:
            accepted = _mutated_headroom_accepts(baseline, mode)
        if accepted:
            print("Headroom rejection control unexpectedly accepted:", mode)
            return False
    return True


def headroom_resource_controls_accepts() -> Bool:
    var tiny = make_headroom_proof[24](37, 15, 52)
    var full = make_headroom_proof(37, 15, 52)
    return (not tk.check_native_headroom(tiny)
        and not tk.check_native_headroom_request[8192, 64](full, 37, 15, 52)
        and tk.check_native_headroom_request(full, 37, 15, 52))
