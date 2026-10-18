"""Independent total Nat join controls; resource preflights are separate."""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_join_terms import make_finishing_join_proof
from llm_mojo.kernels.finishing_join_check import check_finishing_join_proof

comptime FINISHING_JOIN_MUTANTS = 25
comptime FINISHING_JOIN_RESOURCE_CONTROLS = 3
comptime JOIN_BAD_MAX = 0
comptime JOIN_BAD_PARITY = 4
comptime JOIN_BAD_PACK = 6
comptime JOIN_BAD_BOOL_OR = 7
comptime JOIN_BAD_JOIN_FLAG = 12
comptime JOIN_UNDISCHARGED_LAWS = 17
comptime JOIN_UNUSED_MALFORMED = 18
comptime JOIN_METADATA_BUDGET = 0


def _proof_holder() -> List[tk.ProofTermBundle[8192]]:
    var proof = make_finishing_join_proof()
    var holders = List[tk.ProofTermBundle[8192]](capacity=1)
    holders.append(proof^)
    return holders^


def _n[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _named[C: Int](mut p: tk.ProofTermBundle[C], definition: Int) -> Int:
    return _n(p, tk.PT_DEFINED_REF, value=definition)


def _apps[C: Int](mut p: tk.ProofTermBundle[C], var head: Int,
    arguments: List[Int]) -> Int:
    for argument in arguments:
        head = _n(p, tk.PT_APP, head, argument)
    return head


def _binders[C: Int](p: tk.ProofTermBundle[C], definition: Int) -> List[Int]:
    var binders = List[Int]()
    var current = p.definition_types[definition]
    while p.nodes[current].kind == tk.PT_PI:
        binders.append(p.nodes[current].left)
        current = p.nodes[current].right
    return binders^


def _replace[C: Int](mut p: tk.ProofTermBundle[C], definition: Int,
    kind: Int, selected: Int = 0):
    var binders = _binders(p, definition)
    var zero = _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    var body = zero
    if kind == 0:
        body = binders[selected]
    elif kind == 1:
        body = _n(p, tk.PT_EQ, zero, zero)
    elif kind == 2:
        body = _n(p, tk.PT_REFL, zero)
    for index in range(len(binders)):
        body = _n(p, tk.PT_LAM, binders[len(binders) - index - 1], body)
    p.definition_bodies[definition] = body


def _change_join[C: Int](mut p: tk.ProofTermBundle[C], lose_flag: Bool):
    var binders = _binders(p, 86)
    var left = binders[0]
    var right = binders[1]
    var half_left = _apps(p, _named(p, 64), [left])
    var half_right = _apps(p, _named(p, 64), [right])
    var score = _apps(p, _named(p, 55), [half_left, half_right])
    var flag = _apps(p, _named(p, 68), [
        _apps(p, _named(p, 65), [left]), _apps(p, _named(p, 65), [right])])
    if lose_flag:
        flag = _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    else:
        score = half_left
    var body = _apps(p, _named(p, 67), [score, flag])
    for index in range(len(binders)):
        body = _n(p, tk.PT_LAM, binders[len(binders) - index - 1], body)
    p.definition_bodies[86] = body


def finishing_join_accepts() -> Bool:
    var proofs = _proof_holder()
    return check_finishing_join_proof(proofs[0])


def finishing_join_alternate_accepts() -> Bool:
    var proofs = _proof_holder()
    var nat = _n(proofs[0], tk.PT_NAT)
    var dummy = _n(proofs[0], tk.PT_VAR, nat)
    var zero = _n(proofs[0], tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    var wrapped = _n(proofs[0], tk.PT_APP,
        _n(proofs[0], tk.PT_LAM, dummy, proofs[0].root), zero)
    proofs[0].root = wrapped
    return wrapped >= 0 and check_finishing_join_proof(proofs[0])


def finishing_join_mutant_accepts(mode: Int) -> Bool:
    var proofs = _proof_holder()
    if proofs[0].count < 1:
        return True
    var zero = _n(proofs[0], tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    if mode == 0:
        _replace(proofs[0], 55, 0)
    elif mode == 1:
        _replace(proofs[0], 61, 0)
    elif mode == 2:
        _replace(proofs[0], 63, 3)
    elif mode == 3:
        _replace(proofs[0], 64, 3)
    elif mode == 4:
        _replace(proofs[0], 65, 3)
    elif mode == 5:
        _replace(proofs[0], 66, 1)
    elif mode == 6:
        _replace(proofs[0], 67, 0)
    elif mode == 7:
        _replace(proofs[0], 68, 0, 1)
    elif mode == 8:
        proofs[0].definition_bodies[70] = proofs[0].definition_bodies[71]
    elif mode == 9:
        _replace(proofs[0], 71, 2)
    elif mode == 10:
        proofs[0].definition_bodies[77] = proofs[0].definition_bodies[78]
    elif mode == 11:
        proofs[0].definition_bodies[78] = proofs[0].definition_bodies[77]
    elif mode == 12:
        _change_join(proofs[0], True)
    elif mode == 13:
        _change_join(proofs[0], False)
    elif mode == 14:
        proofs[0].definition_bodies[91] = proofs[0].definition_bodies[90]
    elif mode == 15:
        proofs[0].definition_bodies[92] = proofs[0].definition_bodies[91]
    elif mode == 16:
        _replace(proofs[0], 93, 2)
    elif mode == 17:
        # A well-typed old generic theorem still carries the algebra premises.
        # Admission must require the new specialized, discharged result type.
        proofs[0].definition_types[94] = proofs[0].definition_types[54]
        proofs[0].definition_bodies[94] = _named(proofs[0], 54)
    elif mode == 18:
        _ = _n(proofs[0], tk.PT_APP, _n(proofs[0], tk.PT_NAT), zero)
    elif mode == 19:
        proofs[0].root = _n(proofs[0], tk.PT_VAR, _n(proofs[0], tk.PT_NAT))
    elif mode == 20:
        proofs[0].definition_bodies[94] = _named(proofs[0], 94)
    elif mode == 21:
        # Acyclic and unused, but declaration 95 refers forward to 96.
        var nat = _n(proofs[0], tk.PT_NAT)
        proofs[0].definition_types[95] = nat
        proofs[0].definition_bodies[95] = _named(proofs[0], 96)
        proofs[0].definition_types[96] = nat
        proofs[0].definition_bodies[96] = zero
        proofs[0].definition_count = 97
    elif mode == 22:
        _ = _n(proofs[0], tk.PT_ARITHMETIC, value=tk.DEC_LE_TRANS)
    elif mode == 23:
        _ = _n(proofs[0], tk.PT_AFFINE_EQ, zero, zero)
    elif mode == 24:
        var binders = _binders(proofs[0], 62)
        var body = _apps(proofs[0], _named(proofs[0], 3), [binders[2], binders[1], binders[0]])
        for index in range(len(binders)):
            body = _n(proofs[0], tk.PT_LAM, binders[len(binders) - index - 1], body)
        proofs[0].definition_bodies[62] = body
    else:
        return True
    return check_finishing_join_proof(proofs[0])


def finishing_join_resource_mutant_accepts(mode: Int) -> Bool:
    """Resource preflights, kept separate from semantic rejection controls.

    Returning True on a bad test setup prevents a source-node overflow from
    vacuously passing the metadata-only guard tests.
    """
    var proofs = _proof_holder()
    comptime Scratch = 7168
    if proofs[0].count > Scratch or proofs[0].count <= 64:
        return True
    if mode == 0:
        for index in range(proofs[0].definition_count, Scratch + 1):
            proofs[0].definition_types[index] = proofs[0].definition_types[0]
            proofs[0].definition_bodies[index] = proofs[0].definition_bodies[0]
        proofs[0].definition_count = Scratch + 1
    elif mode == 1:
        for index in range(Scratch + 1):
            proofs[0].nat_constructor_types[index] = proofs[0].nat_constructor_types[0]
        proofs[0].nat_constructor_count = Scratch + 1
    elif mode == 2:
        return check_finishing_join_proof[8192, 64](proofs[0])
    else:
        return True
    return check_finishing_join_proof[8192, Scratch](proofs[0])
