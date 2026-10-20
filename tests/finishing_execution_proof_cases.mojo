"""Independent statements and malformed controls for execution proofs."""
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.finishing_execution_terms import (
    EXECUTION_PROOF_CAPACITY, make_finishing_execution_proof,
)
import llm_mojo.kernels.finishing_execution_trace as trace
import llm_mojo.kernels.finishing_execution_check as checked
import llm_mojo.kernels.finishing_execution as execution
from llm_mojo.kernels.argmax_reduction import sequential_argmax_plan, merge_node


def _node[C: Int](mut p: tk.ProofTermBundle[C], kind: Int, left: Int = -1,
    right: Int = -1, extra: Int = -1, value: Int = -1) -> Int:
    return tk.pt_add(p, kind, left, right, extra, value)


def _variable[C: Int](mut p: tk.ProofTermBundle[C], type_id: Int) -> Int:
    return _node(p, tk.PT_VAR, type_id)


def _application[C: Int](mut p: tk.ProofTermBundle[C], f: Int, x: Int) -> Int:
    return _node(p, tk.PT_APP, f, x)


def _applications[C: Int](mut p: tk.ProofTermBundle[C], var f: Int,
    arguments: List[Int]) -> Int:
    for argument in arguments:
        f = _application(p, f, argument)
    return f


def _products[C: Int](mut p: tk.ProofTermBundle[C], binders: List[Int],
    var body: Int) -> Int:
    for index in range(len(binders)):
        body = _node(p, tk.PT_PI, binders[len(binders) - 1 - index], body)
    return body


def _store_type[C: Int](mut p: tk.ProofTermBundle[C], nat: Int) -> Int:
    return _node(p, tk.PT_PI, _variable(p, nat), nat)


def independent_execution_completion_type[C: Int](
    mut p: tk.ProofTermBundle[C],
) -> Int:
    """Independently assemble the complete generic trace theorem statement.

    Definition 23 executes values; 24 executes readiness; 26 is operational
    legality; 27 is the independent left input fold. The checker additionally
    pins their bodies. This does not import producer-side statement helpers.
    """
    var nat = _node(p, tk.PT_NAT)
    var j_a = _variable(p, nat)
    var j_b = _variable(p, nat)
    var join = _variable(p, _products(p, [j_a, j_b], nat))
    var store = _store_type(p, nat)
    var initial_values = _variable(p, store)
    var initial_ready = _variable(p, store)
    var reference = _variable(p, store)
    var destinations = _variable(p, store)
    var lefts = _variable(p, store)
    var rights = _variable(p, store)
    var zero = _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    var one = _application(p,
        _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1), zero)
    var x = _variable(p, nat)
    var initially_ready = _variable(p,
        _node(p, tk.PT_EQ, _application(p, initial_ready, x), one))
    var initial_invariant = _variable(p,
        _products(p, [x, initially_ready], _node(p, tk.PT_EQ,
            _application(p, initial_values, x), _application(p, reference, x))))
    var steps = _variable(p, nat)
    var legal = _variable(p, _applications(p,
        _node(p, tk.PT_DEFINED_REF, value=26),
        [join, reference, initial_ready, destinations, lefts, rights, steps]))
    var root = _variable(p, nat)
    var ready_store = _applications(p, _node(p, tk.PT_DEFINED_REF, value=24),
        [initial_ready, destinations, steps])
    var root_ready = _variable(p,
        _node(p, tk.PT_EQ, _application(p, ready_store, root), one))
    var empty = _variable(p, nat)
    var inputs = _variable(p, store)
    var count = _variable(p, nat)
    var expected = _applications(p, _node(p, tk.PT_DEFINED_REF, value=27),
        [join, empty, inputs, count])
    var root_equation = _variable(p,
        _node(p, tk.PT_EQ, _application(p, reference, root), expected))
    var actual_store = _applications(p, _node(p, tk.PT_DEFINED_REF, value=23),
        [join, initial_values, destinations, lefts, rights, steps])
    var actual = _application(p, actual_store, root)
    return _products(p, [join, initial_values, initial_ready, reference,
        destinations, lefts, rights, initial_invariant, steps, legal, root,
        root_ready, empty, inputs, count, root_equation],
        _node(p, tk.PT_EQ, actual, expected))


def independent_execution_completion_accepts() -> Bool:
    var proof = make_finishing_execution_proof()
    var expected = independent_execution_completion_type(proof.bundle)
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    var state = tk._make_state[EXECUTION_PROOF_CAPACITY, 16384](program, proof.bundle)
    state.fuel = 16384 * 512
    tk._enable_native_numerals(state)
    if not tk._check_inductive_environment(state):
        return False
    var actual = tk._infer(state, proof.completion_root)
    return (actual >= 0 and tk._infer(state, expected) == tk._put(state, tk.PT_PROP)
        and state.context_high[expected] == 0
        and state.context_high[proof.completion_root] == 0
        and tk._convert(state, actual, expected)
        and not state.arithmetic_verified and not state.fast_arithmetic)


def _named[C: Int](mut p: tk.ProofTermBundle[C], definition: Int) -> Int:
    return _node(p, tk.PT_DEFINED_REF, value=definition)


def _equality[C: Int](mut p: tk.ProofTermBundle[C], left: Int, right: Int) -> Int:
    return _node(p, tk.PT_EQ, left, right)


def _join[C: Int](mut p: tk.ProofTermBundle[C], join: Int, left: Int, right: Int) -> Int:
    return _applications(p, join, [left, right])


def _independent_algebra[C: Int](mut p: tk.ProofTermBundle[C], nat: Int) -> List[Int]:
    var first = _variable(p, nat)
    var second = _variable(p, nat)
    var join = _variable(p, _products(p, [first, second], nat))
    var empty = _variable(p, nat)
    var a = _variable(p, nat)
    var b = _variable(p, nat)
    var c = _variable(p, nat)
    var associative = _variable(p, _products(p, [a, b, c], _equality(p,
        _join(p, join, _join(p, join, a, b), c), _join(p, join, a, _join(p, join, b, c)))))
    a = _variable(p, nat)
    b = _variable(p, nat)
    var commutative = _variable(p, _products(p, [a, b], _equality(p,
        _join(p, join, a, b), _join(p, join, b, a))))
    a = _variable(p, nat)
    var left_unit = _variable(p, _products(p, [a], _equality(p, _join(p, join, empty, a), a)))
    a = _variable(p, nat)
    var right_unit = _variable(p, _products(p, [a], _equality(p, _join(p, join, a, empty), a)))
    a = _variable(p, nat)
    var idempotent = _variable(p, _products(p, [a], _equality(p, _join(p, join, a, a), a)))
    return [join, empty, associative, commutative, left_unit, right_unit, idempotent]


def independent_execution_canonical_type[C: Int](mut p: tk.ProofTermBundle[C]) -> Int:
    """Closed theorem for arbitrary input count, trace length and schedules.

    Premises are algebra laws and structural readiness/support facts. No
    value-preservation equation or assumed final-result equality occurs here.
    """
    var nat = _node(p, tk.PT_NAT)
    var algebra = _independent_algebra(p, nat)
    var join = algebra[0]
    var empty = algebra[1]
    var store = _store_type(p, nat)
    var inputs = _variable(p, store)
    var count = _variable(p, nat)
    var support_slot = _variable(p, nat)
    var support = _variable(p, _products(p, [support_slot], store))
    var destinations = _variable(p, store)
    var lefts = _variable(p, store)
    var rights = _variable(p, store)
    var zero = _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    var one = _application(p, _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=1), zero)
    var observation = _variable(p, nat)
    var empty_mask = _application(p, support, count)
    var empty_support = _variable(p, _products(p, [observation], _equality(p,
        _application(p, empty_mask, observation), zero)))
    var source = _variable(p, nat)
    var before = _variable(p, _applications(p, _named(p, 34), [source, count]))
    observation = _variable(p, nat)
    var unit_mask = _application(p, _named(p, 33), source)
    var input_support = _variable(p, _products(p, [source, before, observation], _equality(p,
        _application(p, _application(p, support, source), observation),
        _application(p, unit_mask, observation))))
    var initial_values = _applications(p, _named(p, 46), [empty, inputs, count])
    var initial_ready = _application(p, _named(p, 47), count)
    var steps = _variable(p, nat)
    var structural = _variable(p, _applications(p, _named(p, 51),
        [initial_ready, support, destinations, lefts, rights, steps]))
    var root = _variable(p, nat)
    var ready_store = _applications(p, _named(p, 24), [initial_ready, destinations, steps])
    var ready = _variable(p, _equality(p, _application(p, ready_store, root), one))
    var complete = _variable(p, _applications(p, _named(p, 35),
        [_application(p, support, root), count]))
    var values = _applications(p, _named(p, 23), [join, initial_values, destinations, lefts, rights, steps])
    var expected = _applications(p, _named(p, 27), [join, empty, inputs, count])
    var binders = algebra.copy()
    for binder in [inputs, count, support, destinations, lefts, rights, empty_support,
        input_support, steps, structural, root, ready, complete]:
        binders.append(binder)
    return _products(p, binders, _equality(p, _application(p, values, root), expected))


def independent_execution_canonical_accepts() -> Bool:
    var proof = trace.make_finishing_execution_proof()
    var expected = independent_execution_canonical_type(proof.bundle)
    var program = tk.FinishingProgram(1, 1, 1, 0, 1, 1, 0,
        tk.FINISH_LT, tk.FINISH_LT, 2147483647, Int.MAX)
    var state = tk._make_state[8192, 16384](program, proof.bundle)
    state.fuel = 16384 * 512
    tk._enable_native_numerals(state)
    if not tk._check_inductive_environment(state):
        return False
    var actual = tk._infer(state, proof.canonical_root)
    return (actual >= 0 and tk._infer(state, expected) == tk._put(state, tk.PT_PROP)
        and state.context_high[expected] == 0 and state.context_high[proof.canonical_root] == 0
        and tk._convert(state, actual, expected)
        and not state.arithmetic_verified and not state.fast_arithmetic)


def execution_certificate_accepts() -> Bool:
    var local = sequential_argmax_plan[3]()
    var subgroup = sequential_argmax_plan[2]()
    var final = sequential_argmax_plan[2]()
    var descriptor = execution.make_finishing_execution_descriptor[4, 3, 7, 2, 5, 5](local, subgroup, final)
    var certificate = checked.make_finishing_execution_certificate(descriptor)
    return checked.check_finishing_execution_certificate_request(certificate, descriptor)


def execution_alternate_proof_accepts() -> Bool:
    var proof = trace.make_finishing_execution_proof()
    var nat = _node(proof.bundle, tk.PT_NAT)
    var dummy = _variable(proof.bundle, nat)
    var zero = _node(proof.bundle, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    var wrapped = _application(proof.bundle,
        _node(proof.bundle, tk.PT_LAM, dummy, proof.canonical_root), zero)
    proof.canonical_root = wrapped
    proof.bundle.root = wrapped
    return wrapped >= 0 and checked.check_finishing_execution_proof(proof)


comptime EXECUTION_WRONG_THEOREM = 0
comptime EXECUTION_MISSING_INITIALIZATION = 4
comptime EXECUTION_FALSE_SUPPORT_STEP = 11
comptime EXECUTION_FALSE_FULL_SUPPORT = 29
comptime EXECUTION_UNUSED_MALFORMED = 19
comptime EXECUTION_RESOURCE_METADATA = 24
comptime EXECUTION_STALE_DESCRIPTOR = 32
comptime EXECUTION_WRONG_SOURCE = 33
comptime FINISHING_EXECUTION_PROOF_MUTANTS = 35


def _replace_definition[C: Int](mut p: tk.ProofTermBundle[C], definition: Int,
    body_binder: Int, proposition: Bool = False, constant: Bool = False,
    store_zero: Bool = False, application: Bool = False):
    var binders = List[Int]()
    var current = p.definition_types[definition]
    while p.nodes[current].kind == tk.PT_PI:
        binders.append(p.nodes[current].left)
        current = p.nodes[current].right
    var zero = _node(p, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    var body: Int
    if proposition:
        body = _equality(p, zero, zero)
    elif store_zero:
        var at = _variable(p, _node(p, tk.PT_NAT))
        body = _node(p, tk.PT_LAM, at, zero)
    elif constant:
        body = zero
    elif application:
        body = _application(p, binders[body_binder], binders[len(binders) - 1])
    else:
        body = binders[body_binder]
    for index in range(len(binders)):
        body = _node(p, tk.PT_LAM, binders[len(binders) - 1 - index], body)
    p.definition_bodies[definition] = body


def finishing_execution_proof_mutant_accepts(mode: Int) -> Bool:
    var proof = trace.make_finishing_execution_proof()
    if proof.bundle.count < 1:
        return True
    if mode == 24 or mode == 25:
        comptime Scratch = 6144
        if proof.bundle.count > Scratch:
            return True
        if mode == 24:
            for index in range(proof.bundle.definition_count, Scratch + 1):
                proof.bundle.definition_types[index] = proof.bundle.definition_types[0]
                proof.bundle.definition_bodies[index] = proof.bundle.definition_bodies[0]
            proof.bundle.definition_count = Scratch + 1
        else:
            for index in range(Scratch + 1):
                proof.bundle.nat_constructor_types[index] = proof.bundle.nat_constructor_types[0]
            proof.bundle.nat_constructor_count = Scratch + 1
        return checked.check_finishing_execution_proof[8192, Scratch](proof)
    if mode == 26:
        return checked.check_finishing_execution_proof[8192, 64](proof)
    if mode >= 32:
        var pair = sequential_argmax_plan[2]()
        var single = sequential_argmax_plan[1]()
        var descriptor = execution.make_finishing_execution_descriptor[2, 2, 5, 2, 5, 3](pair, pair, single)
        var certificate = checked.make_finishing_execution_certificate(descriptor)
        if mode == 32:
            certificate.descriptor.program.count_limit -= 1
        elif mode == 33:
            certificate.descriptor.program.item_scale = 1
        elif mode == 34:
            certificate.descriptor.local.nodes[3] = merge_node(0, 2)
        else:
            return True
        return checked.check_finishing_execution_certificate_request(certificate, descriptor)
    var zero = _node(proof.bundle, tk.PT_CONSTRUCTOR, extra=tk.NAT_FAMILY_ID, value=0)
    if mode == 0:
        proof.canonical_root = proof.prefix_refinement_root
        proof.bundle.root = proof.canonical_root
    elif mode == 1:
        proof.canonical_root = -1
    elif mode == 2:
        proof.bundle.root = proof.prefix_refinement_root
    elif mode == 3:
        proof.prefix_refinement_root = -1
    elif mode == 4:
        proof.initialization_root = -1
    elif mode == 5:
        proof.fold_union_root = proof.full_fold_root
    elif mode == 6:
        proof.full_fold_root = proof.leaf_root
    elif mode == 7:
        proof.leaf_root = proof.initialization_root
    elif mode == 8:
        _replace_definition(proof.bundle, 23, 1, application=True)
    elif mode == 9:
        _replace_definition(proof.bundle, 24, 0, application=True)
    elif mode == 10:
        _replace_definition(proof.bundle, 25, 0, proposition=True)
    elif mode == 11:
        _replace_definition(proof.bundle, 50, 0, proposition=True)
    elif mode == 12:
        _replace_definition(proof.bundle, 46, 1, application=True)
    elif mode == 13:
        _replace_definition(proof.bundle, 47, 0, constant=True)
    elif mode == 14:
        _replace_definition(proof.bundle, 31, 0, application=True)
    elif mode == 15:
        _replace_definition(proof.bundle, 27, 1)
    elif mode == 16:
        _replace_definition(proof.bundle, 45, 0, constant=True)
    elif mode == 17:
        proof.bundle.nat_constructor_types[1] = proof.bundle.nat_constructor_types[0]
    elif mode == 18:
        proof.bundle.definition_bodies[0] = _named(proof.bundle, 0)
    elif mode == 19:
        _ = _application(proof.bundle, _node(proof.bundle, tk.PT_NAT), zero)
    elif mode == 20:
        _ = _node(proof.bundle, tk.PT_AFFINE_EQ, zero, zero)
    elif mode == 21:
        _ = _node(proof.bundle, tk.PT_ARITHMETIC, value=tk.DEC_LE_TRANS)
    elif mode == 22:
        proof.bundle.definition_count -= 1
    elif mode == 23:
        proof.bundle.count = -1
    elif mode == 27:
        proof.canonical_root = proof.bundle.count
        proof.bundle.root = proof.bundle.count
    elif mode == 28:
        proof.bundle.definition_bodies[0] = _variable(proof.bundle, _node(proof.bundle, tk.PT_PROP))
    elif mode == 29:
        _replace_definition(proof.bundle, 35, 0, proposition=True)
    elif mode == 30:
        _ = _named(proof.bundle, 55)
    elif mode == 31:
        var index = _node(proof.bundle, tk.PT_REFL, zero)
        proof.bundle.nodes[index].value = 17
    else:
        return True
    return checked.check_finishing_execution_proof(proof)
