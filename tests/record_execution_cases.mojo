"""Adversarial cases for the complete183 correlated-record profile.

Semantics, wrong public contracts, and allocation preflights are classified
separately. Full positives must establish the actual resource budget before
negative results count. Three final-statement controls have separate FULL
raw-environment typing positives. Four wrong-scope data controls have limited
declaration-prefix typing diagnostics, not full-environment admission claims.
"""
import llm_mojo.kernels.proof_term_kernel as tk
import llm_mojo.kernels.finishing_phase_check as phase
import llm_mojo.kernels.nat_score_bridge_check as score_profile
import llm_mojo.kernels.nat_record_execution_check as checked
from llm_mojo.kernels.finishing_record_terms import make_finishing_record_proof_holder
from llm_mojo.kernels.nat_record_execution_check import RecordExecutionLayout, check_nat_record_execution_proof

# Measured independent-profile budget; this bounds checker storage only.
comptime RECORD_EXECUTION_WORKSPACE = 65536
comptime RECORD_EXECUTION_CAPACITY = 16384
comptime RECORD_EXECUTION_SEMANTIC_CONTROLS = 44
comptime RECORD_EXECUTION_RESOURCE_CONTROLS = 3
comptime RECORD_EXECUTION_ASSUMED_FINAL = 14
comptime RECORD_EXECUTION_GENERIC_OLD_FINAL = 15
comptime RECORD_EXECUTION_RANK_ONLY_FINAL = 16


def _n[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _zero[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)


def _one[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _n(p, tk.PT_APP, _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1), _zero(p))


def _nat_var[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    return _n(p, tk.PT_VAR, _n(p, tk.PT_NAT))


def _named[C: Int](mut p: tk.ProofTermBundle[C], definition: Int) -> Int:
    return _n(p, tk.PT_DEFINED_REF, value=definition)


def _apps[C: Int](mut p: tk.ProofTermBundle[C], var result: Int,
    arguments: List[Int]) -> Int:
    for argument in arguments:
        result = _n(p, tk.PT_APP, result, argument)
    return result


def _call[C: Int](mut p: tk.ProofTermBundle[C], definition: Int, args: List[Int]) -> Int:
    return _apps(p, _named(p, definition), args)


def _lams[C: Int](mut p: tk.ProofTermBundle[C], binders: List[Int], var result: Int) -> Int:
    for index in range(len(binders)):
        result = _n(p, tk.PT_LAM, binders[len(binders) - index - 1], result)
    return result


def _pis[C: Int](mut p: tk.ProofTermBundle[C], binders: List[Int], var result: Int) -> Int:
    for index in range(len(binders)):
        result = _n(p, tk.PT_PI, binders[len(binders) - index - 1], result)
    return result


def _binders[C: Int](p: tk.ProofTermBundle[C], definition: Int, limit: Int) -> List[Int]:
    var result = List[Int]()
    var current = p.definition_types[definition]
    while p.nodes[current].kind == tk.PT_PI and len(result) < limit:
        result.append(p.nodes[current].left)
        current = p.nodes[current].right
    return result^


def _conclusion[C: Int](p: tk.ProofTermBundle[C], definition: Int, arguments: Int) -> Int:
    var current = p.definition_types[definition]
    for _ in range(arguments):
        current = p.nodes[current].right
    return current


def _field[C: Int](mut p: tk.ProofTermBundle[C], record: Int, position: Int) -> Int:
    var index = _zero(p)
    for _ in range(position):
        index = _n(p, tk.PT_APP,
            _n(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1), index)
    return _n(p, tk.PT_APP, record, index)


def _row[C: Int](mut p: tk.ProofTermBundle[C], state: Int, slot: Int) -> Int:
    var component = _nat_var(p)
    return _lams(p, [component], _apps(p, state, [component, slot]))


def _change_run[C: Int](mut p: tk.ProofTermBundle[C]):
    var args = _binders(p, 164, 5)
    var k = _nat_var(p)
    var old = _n(p, tk.PT_VAR, p.nodes[args[0]].left)
    var component = _nat_var(p)
    var destination = _n(p, tk.PT_APP, args[1], k)
    var left = _row(p, old, _n(p, tk.PT_APP, args[2], k))
    # Right rank/bad use R(k), while its token comes from D(k): a correlated
    # operand tuple violation that cannot be hidden by merge idempotence.
    var right = _row(p, old, _n(p, tk.PT_APP, args[3], k))
    var stale = _row(p, old, destination)
    var wrong_right = _call(p, 160, [_field(p, right, 0), _field(p, stale, 1), _field(p, right, 2)])
    var merged = _call(p, 163, [left, wrong_right])
    var update = _call(p, 5, [_n(p, tk.PT_APP, old, component), destination,
        _n(p, tk.PT_APP, merged, component)])
    var step = _lams(p, [k, old, component], update)
    p.definition_bodies[164] = _lams(p, args, _n(p, tk.PT_NAT_REC, args[4], args[0], step))


def _change_fold[C: Int](mut p: tk.ProofTermBundle[C]):
    var args = _binders(p, 165, 5)
    var k = _nat_var(p)
    var ignored = _n(p, tk.PT_VAR, p.nodes[_binders(p, 163, 1)[0]].left)
    var input = _call(p, 160, [_n(p, tk.PT_APP, args[1], k),
        _n(p, tk.PT_APP, args[2], k), _n(p, tk.PT_APP, args[3], k)])
    var empty = _call(p, 160, [_zero(p), args[0], _zero(p)])
    # The requested fold must accumulate all inputs, rather than keep last.
    var step = _lams(p, [k, ignored], input)
    p.definition_bodies[165] = _lams(p, args, _n(p, tk.PT_NAT_REC, args[4], empty, step))


def _change_scope[C: Int](mut p: tk.ProofTermBundle[C], mode: Int):
    if mode == 9:
        var args = _binders(p, 173, 5)
        var slot = _nat_var(p)
        var row = _call(p, 160, [_n(p, tk.PT_APP, args[1], slot),
            _n(p, tk.PT_APP, args[2], slot), _n(p, tk.PT_APP, args[3], slot)])
        # Stronger but incorrect public input contract: all slots, not only
        # Before(slot,count). Unread input functions must remain arbitrary.
        p.definition_bodies[173] = _lams(p, args,
            _pis(p, [slot], _call(p, 168, [args[0], row])))
    elif mode == 11:
        var args = _binders(p, 176, 3)
        var slot = _nat_var(p)
        p.definition_bodies[176] = _lams(p, args,
            _pis(p, [slot], _call(p, 168, [args[0], _row(p, args[1], slot)])))
    elif mode == 12:
        var args = _binders(p, 177, 3)
        var slot = _nat_var(p)
        var value = _n(p, tk.PT_APP, args[2], slot)
        var guarded = _n(p, tk.PT_VAR, _n(p, tk.PT_EQ, value, _one(p)))
        var row = _row(p, args[1], slot)
        var encoded = _call(p, 142, [args[0], _field(p, row, 0), _field(p, row, 1), _field(p, row, 2)])
        # Lost all-slot agreement: encoding is compared only when the value
        # happens to equal1, incorrectly treating values as readiness flags.
        p.definition_bodies[177] = _lams(p, args,
            _pis(p, [slot, guarded], _n(p, tk.PT_EQ, encoded, value)))
    else:
        var args = _binders(p, 178, 4)
        p.definition_bodies[178] = _lams(p, args, _call(p, 177, [args[0], args[1], args[3]]))


def _change_final[C: Int](mut p: tk.ProofTermBundle[C], mode: Int):
    if mode == 15:
        p.definition_types[182] = p.definition_types[94]
        p.definition_bodies[182] = _named(p, 94)
        return
    # Stop at the17 public arguments: the conclusion itself is a Church And
    # with two internal Pi binders, which must not become assumed premises.
    var args = _binders(p, 182, 17)
    var goal = _conclusion(p, 182, 17)
    if mode == 14:
        var assumed = _n(p, tk.PT_VAR, goal)
        args.append(assumed)
        p.definition_types[182] = _pis(p, args, goal)
        p.definition_bodies[182] = _lams(p, args, assumed)
    else:
        var old_body = p.definition_bodies[182]
        var consume = p.nodes[p.nodes[goal].right].left
        var consumer_type = p.nodes[consume].left
        var evidence = List[Int]()
        for _ in range(3):
            evidence.append(p.nodes[consumer_type].left)
            consumer_type = p.nodes[consumer_type].right
        var rank_equal = p.nodes[evidence[0]].left
        var selected = _apps(p, _apps(p, old_body, args),
            [rank_equal, _lams(p, evidence, evidence[0])])
        p.definition_types[182] = _pis(p, args, rank_equal)
        p.definition_bodies[182] = _lams(p, args, selected)


def _mutate[C: Int](mut p: tk.ProofTermBundle[C], mut layout: RecordExecutionLayout,
    mode: Int) -> Bool:
    var zero = _zero(p)
    if mode == 0:
        var args = _binders(p, 148, 4)
        p.definition_bodies[148] = _lams(p, args, args[0])
    elif mode == 1:
        var args = _binders(p, 149, 4)
        p.definition_bodies[149] = _lams(p, args, args[2])
    elif mode == 2:
        var args = _binders(p, 160, 4)
        var predecessor = _call(p, 2, [args[3]])
        var body = _call(p, 3, [args[3], args[0], _call(p, 3, [predecessor, args[2], args[1]])])
        p.definition_bodies[160] = _lams(p, args, body)
    elif mode == 3:
        var args = _binders(p, 161, 5)
        var row = _call(p, 160, [_n(p, tk.PT_APP, args[1], args[4]),
            _n(p, tk.PT_APP, args[0], args[4]), _n(p, tk.PT_APP, args[2], args[4])])
        p.definition_bodies[161] = _lams(p, args, _n(p, tk.PT_APP, row, args[3]))
    elif mode == 4:
        var args = _binders(p, 162, 6)
        var stores = _call(p, 161, [args[2], args[3], args[4]])
        var wrong_empty = _call(p, 160, [zero, zero, zero])
        var body = _call(p, 5, [_n(p, tk.PT_APP, stores, args[5]), args[1],
            _n(p, tk.PT_APP, wrong_empty, args[5])])
        p.definition_bodies[162] = _lams(p, args, body)
    elif mode == 5:
        var args = _binders(p, 163, 3)
        var ra = _field(p, args[0], 0)
        var rb = _field(p, args[1], 0)
        var ta = _field(p, args[0], 1)
        var tb = _field(p, args[1], 1)
        var flag = _call(p, 139, [ra, rb, ta, tb])
        var rank = _call(p, 148, [ra, rb, ta, tb])
        var wrong_token = _call(p, 3, [flag, tb, ta])
        var bad = _call(p, 68, [_field(p, args[0], 2), _field(p, args[1], 2)])
        var row = _call(p, 160, [rank, wrong_token, bad])
        p.definition_bodies[163] = _lams(p, args, _n(p, tk.PT_APP, row, args[2]))
    elif mode == 6:
        _change_run(p)
    elif mode == 7:
        _change_fold(p)
    elif mode == 8:
        var args = _binders(p, 168, 2)
        p.definition_bodies[168] = _lams(p, args, _call(p, 66, [_field(p, args[1], 2)]))
    elif mode == 9:
        _change_scope(p, mode)
    elif mode == 10:
        var args = _binders(p, 174, 5)
        var body = _call(p, 142, [args[0], _n(p, tk.PT_APP, args[2], args[4]),
            _n(p, tk.PT_APP, args[1], args[4]), _n(p, tk.PT_APP, args[3], args[4])])
        p.definition_bodies[174] = _lams(p, args, body)
    elif mode == 11:
        _change_scope(p, mode)
    elif mode == 12:
        _change_scope(p, mode)
    elif mode == 13:
        _change_scope(p, mode)
    elif mode == 14:
        _change_final(p, mode)
    elif mode == 15:
        _change_final(p, mode)
    elif mode == 16:
        _change_final(p, mode)
    elif mode == 17:
        p.definition_bodies[156] = p.definition_bodies[155]
    elif mode == 18:
        p.definition_bodies[166] = p.definition_bodies[167]
    elif mode == 19:
        p.definition_bodies[169] = p.definition_bodies[170]
    elif mode == 20:
        p.definition_bodies[175] = p.definition_bodies[181]
    elif mode == 21:
        p.definition_bodies[179] = p.definition_bodies[180]
    elif mode == 22:
        p.definition_bodies[181] = p.definition_bodies[175]
    elif mode == 23:
        _ = _n(p, tk.PT_APP, _n(p, tk.PT_NAT), zero)
    elif mode == 24:
        p.root = _nat_var(p)
    elif mode == 25:
        p.definition_bodies[182] = _named(p, 182)
    elif mode == 26:
        var nat = _n(p, tk.PT_NAT)
        p.definition_types[155] = nat
        p.definition_bodies[155] = _named(p, 156)
        p.definition_types[156] = nat
        p.definition_bodies[156] = zero
    elif mode == 27:
        _ = _n(p, tk.PT_ARITHMETIC, value=tk.DEC_LE_TRANS)
    elif mode == 28:
        _ = _n(p, tk.PT_AFFINE_EQ, zero, zero)
    elif mode == 29:
        _ = _n(p, tk.PT_LITERAL_LE, zero, zero)
    elif mode == 30:
        layout.bridge.predecessor_conversion_root = p.definition_bodies[159]
    elif mode == 31:
        layout.trace_first += 1
    elif mode == 32:
        layout.basics_first += 1
    elif mode == 33:
        layout.fold_first += 1
    elif mode == 34:
        layout.refinement_first += 1
    elif mode == 35:
        layout.initialization -= 1
    elif mode == 36:
        layout.canonical -= 1
    elif mode == 37:
        layout.prefix_first += 1
    elif mode == 38:
        layout.laws_first += 1
    elif mode == 39:
        layout.bridge.selector_first += 1
    elif mode == 40:
        p.definition_count = 182
    elif mode == 41:
        p.root = -1
    elif mode == 42:
        p.root = _named(p, 147)
    elif mode == 43:
        var args = _binders(p, 163, 3)
        var ra = _field(p, args[0], 0)
        var rb = _field(p, args[1], 0)
        var ta = _field(p, args[0], 1)
        var tb = _field(p, args[1], 1)
        var row = _call(p, 160, [_call(p, 148, [ra, rb, ta, tb]),
            _call(p, 149, [ra, rb, ta, tb]), _field(p, args[0], 2)])
        p.definition_bodies[163] = _lams(p, args, _n(p, tk.PT_APP, row, args[2]))
    else:
        return False
    return True


@no_inline
def record_execution_accepts[scratch: Int = RECORD_EXECUTION_WORKSPACE]() -> Bool:
    var proofs = make_finishing_record_proof_holder[RECORD_EXECUTION_CAPACITY]()
    return check_nat_record_execution_proof[RECORD_EXECUTION_CAPACITY, scratch](proofs[0].bundle, proofs[0].layout)


@no_inline
def record_execution_alternate_accepts[scratch: Int = RECORD_EXECUTION_WORKSPACE]() -> Bool:
    var proofs = make_finishing_record_proof_holder[RECORD_EXECUTION_CAPACITY]()
    if proofs[0].bundle.count <= 0:
        return False
    var dummy = _nat_var(proofs[0].bundle)
    var function = _lams(proofs[0].bundle, [dummy], proofs[0].bundle.root)
    proofs[0].bundle.root = _n(proofs[0].bundle, tk.PT_APP, function, _zero(proofs[0].bundle))
    return check_nat_record_execution_proof[RECORD_EXECUTION_CAPACITY, scratch](proofs[0].bundle, proofs[0].layout)


@no_inline
def record_execution_mutant_result[scratch: Int = RECORD_EXECUTION_WORKSPACE](mode: Int) -> checked.RecordExecutionCheckResult:
    var proofs = make_finishing_record_proof_holder[RECORD_EXECUTION_CAPACITY]()
    if proofs[0].bundle.count <= 0 or not _mutate(proofs[0].bundle, proofs[0].layout, mode):
        return checked.RecordExecutionCheckResult(True, checked.RECORD_CHECK_ACCEPTED, -1, -1)
    return checked.check_nat_record_execution_result[RECORD_EXECUTION_CAPACITY, scratch](proofs[0].bundle, proofs[0].layout)


@no_inline
def record_execution_mutant_accepts[scratch: Int = RECORD_EXECUTION_WORKSPACE](mode: Int) -> Bool:
    return record_execution_mutant_result[scratch](mode).accepted


@no_inline
def record_execution_resource_mutant_result(mode: Int) -> checked.RecordExecutionCheckResult:
    var proofs = make_finishing_record_proof_holder[RECORD_EXECUTION_CAPACITY]()
    comptime Scratch = RECORD_EXECUTION_CAPACITY - 1
    if proofs[0].bundle.count <= 64 or proofs[0].bundle.count > Scratch:
        return checked.RecordExecutionCheckResult(True, checked.RECORD_CHECK_ACCEPTED, -1, -1)
    if mode == 0:
        for index in range(proofs[0].bundle.definition_count, Scratch + 1):
            proofs[0].bundle.definition_types[index] = proofs[0].bundle.definition_types[0]
            proofs[0].bundle.definition_bodies[index] = proofs[0].bundle.definition_bodies[0]
        proofs[0].bundle.definition_count = Scratch + 1
    elif mode == 1:
        for index in range(Scratch + 1):
            proofs[0].bundle.nat_constructor_types[index] = proofs[0].bundle.nat_constructor_types[0]
        proofs[0].bundle.nat_constructor_count = Scratch + 1
    elif mode == 2:
        return checked.check_nat_record_execution_result[RECORD_EXECUTION_CAPACITY, 64](proofs[0].bundle, proofs[0].layout)
    else:
        return checked.RecordExecutionCheckResult(True, checked.RECORD_CHECK_ACCEPTED, -1, -1)
    return checked.check_nat_record_execution_result[RECORD_EXECUTION_CAPACITY, Scratch](proofs[0].bundle, proofs[0].layout)


@no_inline
def record_execution_resource_mutant_accepts(mode: Int) -> Bool:
    return record_execution_resource_mutant_result(mode).accepted


@no_inline
def record_execution_final_control_is_welltyped[scratch: Int = RECORD_EXECUTION_WORKSPACE](mode: Int) -> Bool:
    """FULL ordinary183 admission for changed final types14/15/16 only."""
    if mode != 14 and mode != 15 and mode != 16:
        return False
    var proofs = make_finishing_record_proof_holder[RECORD_EXECUTION_CAPACITY]()
    if proofs[0].bundle.count <= 0 or not _mutate(proofs[0].bundle, proofs[0].layout, mode):
        return False
    if proofs[0].bundle.count > scratch or not phase._phase_syntax(proofs[0].bundle):
        return False
    var states = score_profile._state_holder[RECORD_EXECUTION_CAPACITY, scratch](proofs[0].bundle)
    states[0].fuel = scratch * 2048
    tk._enable_native_numerals(states[0])
    if not tk._check_inductive_environment(states[0]):
        return False
    for index in range(proofs[0].bundle.count):
        var actual = tk._infer(states[0], index)
        if actual == -1 or actual == -4:
            return False
    var root = proofs[0].bundle.root
    var actual = tk._infer(states[0], root)
    return (actual >= 0 and tk._infer(states[0], actual) == tk._put(states[0], tk.PT_PROP)
        and states[0].context_high[root] == 0 and states[0].context_high[actual] == 0
        and states[0].fuel > 0 and states[0].syntax.count < scratch
        and not states[0].arithmetic_verified and not states[0].fast_arithmetic)


@no_inline
def record_execution_scope_data_is_welltyped[scratch: Int = RECORD_EXECUTION_WORKSPACE](mode: Int) -> Bool:
    """Limited declaration-prefix typing; NOT full183 candidate admission.

    This establishes that the changed input-validity/ReadyValid/encoding/Inv
    predicate is itself a welltyped closed Prop definition. Later proofs may
    reject that changed contract; the profile must reject the full candidate.
    """
    var definition: Int
    if mode == 9:
        definition = 173
    elif mode == 11:
        definition = 176
    elif mode == 12:
        definition = 177
    elif mode == 13:
        definition = 178
    else:
        return False
    var proofs = make_finishing_record_proof_holder[RECORD_EXECUTION_CAPACITY]()
    if proofs[0].bundle.count <= 0 or not _mutate(proofs[0].bundle, proofs[0].layout, mode):
        return False
    if proofs[0].bundle.count > scratch or not phase._phase_syntax(proofs[0].bundle):
        return False
    var states = score_profile._state_holder[RECORD_EXECUTION_CAPACITY, scratch](proofs[0].bundle)
    states[0].syntax.definition_count = definition + 1
    states[0].fuel = scratch * 2048
    tk._enable_native_numerals(states[0])
    if not tk._check_inductive_environment(states[0]):
        return False
    var declared = states[0].syntax.definition_types[definition]
    var body = states[0].syntax.definition_bodies[definition]
    var actual = tk._infer(states[0], body)
    return (tk._infer(states[0], declared) == tk._put(states[0], tk.PT_TYPE)
        and actual >= 0 and tk._convert(states[0], actual, declared)
        and states[0].context_high[declared] == 0 and states[0].context_high[body] == 0
        and states[0].fuel > 0 and states[0].syntax.count < scratch
        and not states[0].arithmetic_verified and not states[0].fast_arithmetic)
