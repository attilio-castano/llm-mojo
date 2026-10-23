"""Ordinary refinement proofs for one correlated mathematical record run.

The invariant combines validity at ready slots with encoding agreement at all
slots. Step preservation uses ready children and derives the output encoding;
it assumes neither destination freshness nor a desired value equation.
Nat induction yields prefix refinement for universal bounds, states and events.
No function extensionality or new primitive rule is used.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as algebra
import llm_mojo.kernels.finishing_execution_terms as execution
import llm_mojo.kernels.finishing_execution_trace as trace
import llm_mojo.kernels.finishing_join_terms as joined
import llm_mojo.kernels.nat_record_encoding as encoding
import llm_mojo.kernels.nat_record_trace_data as data
import llm_mojo.kernels.nat_record_basics as basics

comptime RECORD_TRACE_REFINEMENT_DEFINITIONS = 5


@fieldwise_init
struct RecordTraceRefinementExports(ImplicitlyCopyable):
    var ready_valid: Int
    var encoded_equal: Int
    var invariant: Int
    var step_preservation: Int
    var prefix_refinement: Int


def _call[C: Int](mut p: ph._Builder[C], definition: Int, arguments: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), arguments)


def _declare[C: Int](mut p: ph._Builder[C], type: Int, body: Int) -> Int:
    var index = p.bundle.definition_count
    ph._declare(p, type, body, index)
    return index


def _field_index[C: Int](mut p: ph._Builder[C], position: Int) -> Int:
    var result = ph._zero(p)
    for _ in range(position):
        result = ph._succ(p, result)
    return result


def record_ready_valid[C: Int](mut p: ph._Builder[C], exports: RecordTraceRefinementExports,
    bound: Int, state: Int, ready: Int) -> Int:
    return _call(p, exports.ready_valid, [bound, state, ready])


def record_encoded_equal[C: Int](mut p: ph._Builder[C], exports: RecordTraceRefinementExports,
    bound: Int, state: Int, values: Int) -> Int:
    return _call(p, exports.encoded_equal, [bound, state, values])


def record_invariant[C: Int](mut p: ph._Builder[C], exports: RecordTraceRefinementExports,
    bound: Int, state: Int, ready: Int, values: Int) -> Int:
    return _call(p, exports.invariant, [bound, state, ready, values])


def record_invariant_parts[C: Int](mut p: ph._Builder[C], exports: RecordTraceRefinementExports,
    bound: Int, state: Int, ready: Int, values: Int) -> List[Int]:
    return [record_ready_valid(p, exports, bound, state, ready),
        record_encoded_equal(p, exports, bound, state, values)]


def record_step_state[C: Int](mut p: ph._Builder[C], records: data.RecordTraceDataExports,
    state: Int, destination: Int, left: Int, right: Int) -> Int:
    var merged = data.record_merge(p, records,
        data.record_row(p, state, left), data.record_row(p, state, right))
    var field = ph._var(p, ph._nat(p))
    return ph._lam(p, field, ph._update(p, ph._app(p, state, field), destination,
        ph._app(p, merged, field)))


def _ready_valid_data[C: Int](mut p: ph._Builder[C], valid: basics.RecordBasicsExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var state = ph._var(p, data.record_state_type(p))
    var ready = ph._var(p, ph._store(p))
    var slot = ph._var(p, ph._nat(p))
    var present = ph._var(p, ph._eq(p, ph._app(p, ready, slot), ph._one(p)))
    var body = ph._pis(p, [slot, present],
        basics.valid_record(p, valid, bound, data.record_row(p, state, slot)))
    return _declare(p, ph._pis(p, [bound, state, ready], ph._prop(p)),
        ph._lams(p, [bound, state, ready], body))


def _encoded_equal_data[C: Int](mut p: ph._Builder[C], enc: encoding.EncodedRecordExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var state = ph._var(p, data.record_state_type(p))
    var values = ph._var(p, ph._store(p))
    var slot = ph._var(p, ph._nat(p))
    var body = ph._pi(p, slot, ph._eq(p,
        basics.encoded(p, enc, bound, data.record_row(p, state, slot)),
        ph._app(p, values, slot)))
    return _declare(p, ph._pis(p, [bound, state, values], ph._prop(p)),
        ph._lams(p, [bound, state, values], body))


def _invariant_data[C: Int](mut p: ph._Builder[C], exports: RecordTraceRefinementExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var state = ph._var(p, data.record_state_type(p))
    var ready = ph._var(p, ph._store(p))
    var values = ph._var(p, ph._store(p))
    var binders: List[Int] = [bound, state, ready, values]
    return _declare(p, ph._pis(p, binders, ph._prop(p)), ph._lams(p, binders,
        algebra._and_type(p, record_invariant_parts(p, exports, bound, state, ready, values))))


def _field_equations[C: Int](mut p: ph._Builder[C], state: Int, merged: Int,
    destination: Int, observation: Int, apart: Int, hit: Bool) -> List[Int]:
    var result = List[Int]()
    for position in range(3):
        var field = _field_index(p, position)
        var row = ph._app(p, state, field)
        var value = ph._app(p, merged, field)
        if hit:
            result.append(_call(p, ph.PHASE_UPDATE_HIT, [row, destination, value]))
        else:
            result.append(_call(p, ph.PHASE_UPDATE_MISS,
                [row, destination, value, observation, apart]))
    return result^


def _encoding_congruence[C: Int](mut p: ph._Builder[C], valid: basics.RecordBasicsExports,
    bound: Int, left: Int, right: Int, equalities: List[Int]) -> Int:
    return _call(p, valid.encode_congruent,
        [bound, left, right, equalities[0], equalities[1], equalities[2]])


def _validity_back[C: Int](mut p: ph._Builder[C], valid: basics.RecordBasicsExports,
    bound: Int, changed: Int, old: Int, old_valid: Int, equalities: List[Int]) -> Int:
    # Each equation is changed(field)=old(field); validity moves old->changed.
    var token_equal = ph._sym(p, basics.field(p, changed, 1), basics.field(p, old, 1), equalities[1])
    var bad_equal = ph._sym(p, basics.field(p, changed, 2), basics.field(p, old, 2), equalities[2])
    return _call(p, valid.validity_transport,
        [bound, old, changed, old_valid, token_equal, bad_equal])


def _step_preservation[C: Int](mut p: ph._Builder[C], exports: RecordTraceRefinementExports,
    enc: encoding.EncodedRecordExports, valid: basics.RecordBasicsExports,
    records: data.RecordTraceDataExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var state = ph._var(p, data.record_state_type(p))
    var ready = ph._var(p, ph._store(p))
    var values = ph._var(p, ph._store(p))
    var destination = ph._var(p, ph._nat(p))
    var left = ph._var(p, ph._nat(p))
    var right = ph._var(p, ph._nat(p))
    var invariant = ph._var(p, record_invariant(p, exports, bound, state, ready, values))
    var left_ready = ph._var(p, ph._eq(p, ph._app(p, ready, left), ph._one(p)))
    var right_ready = ph._var(p, ph._eq(p, ph._app(p, ready, right), ph._one(p)))
    var old_parts = record_invariant_parts(p, exports, bound, state, ready, values)
    var old_valid = algebra._project(p, invariant, old_parts, 0)
    var old_encoded = algebra._project(p, invariant, old_parts, 1)
    var left_record = data.record_row(p, state, left)
    var right_record = data.record_row(p, state, right)
    var left_valid = ph._apps(p, old_valid, [left, left_ready])
    var right_valid = ph._apps(p, old_valid, [right, right_ready])
    var merged = data.record_merge(p, records, left_record, right_record)
    var merged_valid = _call(p, valid.merge_valid,
        [bound, left_record, right_record, left_valid, right_valid])
    var merged_encoded = _call(p, valid.merge_encoding,
        [bound, left_record, right_record, left_valid, right_valid])
    var new_state = record_step_state(p, records, state, destination, left, right)
    var new_ready = ph._update(p, ready, destination, ph._one(p))
    var joined_values = joined.record_join(p, ph._app(p, values, left), ph._app(p, values, right))
    var new_values = ph._update(p, values, destination, joined_values)
    var destination_record = data.record_row(p, new_state, destination)
    var hits = _field_equations(p, state, merged, destination, destination, -1, True)
    var destination_valid = _validity_back(p, valid, bound,
        destination_record, merged, merged_valid, hits)

    # Ready-validity: the destination gets derived merge validity. Elsewhere
    # the ready update and both relevant record fields are checked misses.
    var observation = ph._var(p, ph._nat(p))
    var present = ph._var(p, ph._eq(p, ph._app(p, new_ready, observation), ph._one(p)))
    var observed_record = data.record_row(p, new_state, observation)
    var ready_goal = basics.valid_record(p, valid, bound, observed_record)
    var equal = ph._var(p, ph._eq(p, destination, observation))
    var slot = ph._var(p, ph._nat(p))
    var valid_at = ph._lam(p, slot,
        basics.valid_record(p, valid, bound, data.record_row(p, new_state, slot)))
    var ready_yes = ph._lam(p, equal, ph._subst(p, valid_at, equal, destination_valid))
    var apart = ph._var(p, ph._apart(p, destination, observation))
    var ready_miss = _call(p, ph.PHASE_UPDATE_MISS,
        [ready, destination, ph._one(p), observation, apart])
    var old_present = ph._trans(p, ph._app(p, ready, observation),
        ph._app(p, new_ready, observation), ph._one(p),
        ph._sym(p, ph._app(p, new_ready, observation), ph._app(p, ready, observation), ready_miss),
        present)
    var valid_before = ph._apps(p, old_valid, [observation, old_present])
    var old_record = data.record_row(p, state, observation)
    var misses = _field_equations(p, state, merged, destination, observation, apart, False)
    var ready_no = ph._lam(p, apart,
        _validity_back(p, valid, bound, observed_record, old_record, valid_before, misses))
    var ready_proof = ph._lams(p, [observation, present], _call(p, ph.PHASE_DECIDE,
        [destination, observation, ready_goal, ready_yes, ready_no]))

    # Encoding at the destination uses the merge homomorphism and the two
    # global encoding equations. No validity is required at unready misses.
    var encoded_destination = basics.encoded(p, enc, bound, destination_record)
    var encoded_merged = basics.encoded(p, enc, bound, merged)
    var encoded_left = basics.encoded(p, enc, bound, left_record)
    var encoded_right = basics.encoded(p, enc, bound, right_record)
    var source_join = joined.record_join(p, encoded_left, encoded_right)
    var middle_join = joined.record_join(p, ph._app(p, values, left), encoded_right)
    var left_equation = ph._app(p, old_encoded, left)
    var right_equation = ph._app(p, old_encoded, right)
    var join_left = algebra._left(p, ph._named(p, joined.RECORD_JOIN), encoded_left,
        ph._app(p, values, left), left_equation, encoded_right)
    var join_right = algebra._right(p, ph._named(p, joined.RECORD_JOIN),
        ph._app(p, values, left), encoded_right, ph._app(p, values, right), right_equation)
    var join_equation = ph._trans(p, source_join, middle_join, joined_values, join_left, join_right)
    var destination_encoding = ph._trans(p, encoded_destination, encoded_merged, source_join,
        _encoding_congruence(p, valid, bound, destination_record, merged, hits), merged_encoded)
    destination_encoding = ph._trans(p, encoded_destination, source_join, joined_values,
        destination_encoding, join_equation)
    var value_hit = _call(p, ph.PHASE_UPDATE_HIT, [values, destination, joined_values])
    destination_encoding = ph._trans(p, encoded_destination, joined_values,
        ph._app(p, new_values, destination), destination_encoding,
        ph._sym(p, ph._app(p, new_values, destination), joined_values, value_hit))

    observation = ph._var(p, ph._nat(p))
    observed_record = data.record_row(p, new_state, observation)
    var observed_encoded = basics.encoded(p, enc, bound, observed_record)
    var encoding_goal = ph._eq(p, observed_encoded, ph._app(p, new_values, observation))
    equal = ph._var(p, ph._eq(p, destination, observation))
    slot = ph._var(p, ph._nat(p))
    var encoding_at = ph._lam(p, slot, ph._eq(p,
        basics.encoded(p, enc, bound, data.record_row(p, new_state, slot)),
        ph._app(p, new_values, slot)))
    var encoding_yes = ph._lam(p, equal, ph._subst(p, encoding_at, equal, destination_encoding))
    apart = ph._var(p, ph._apart(p, destination, observation))
    old_record = data.record_row(p, state, observation)
    misses = _field_equations(p, state, merged, destination, observation, apart, False)
    var miss_encoding = _encoding_congruence(p, valid, bound, observed_record, old_record, misses)
    var encoding_miss = ph._trans(p, observed_encoded, basics.encoded(p, enc, bound, old_record),
        ph._app(p, values, observation), miss_encoding, ph._app(p, old_encoded, observation))
    var value_miss = _call(p, ph.PHASE_UPDATE_MISS,
        [values, destination, joined_values, observation, apart])
    encoding_miss = ph._trans(p, observed_encoded, ph._app(p, values, observation),
        ph._app(p, new_values, observation), encoding_miss,
        ph._sym(p, ph._app(p, new_values, observation), ph._app(p, values, observation), value_miss))
    var encoding_no = ph._lam(p, apart, encoding_miss)
    var encoding_proof = ph._lam(p, observation, _call(p, ph.PHASE_DECIDE,
        [destination, observation, encoding_goal, encoding_yes, encoding_no]))
    var result = record_invariant(p, exports, bound, new_state, new_ready, new_values)
    var body = algebra._and_intro(p,
        record_invariant_parts(p, exports, bound, new_state, new_ready, new_values),
        [ready_proof, encoding_proof])
    var binders: List[Int] = [bound, state, ready, values, destination, left, right,
        invariant, left_ready, right_ready]
    return _declare(p, ph._pis(p, binders, result), ph._lams(p, binders, body))


def _prefix_result[C: Int](mut p: ph._Builder[C], exports: RecordTraceRefinementExports,
    records: data.RecordTraceDataExports, bound: Int, initial_state: Int, initial_ready: Int,
    initial_values: Int, destinations: Int, lefts: Int, rights: Int, steps: Int) -> Int:
    return record_invariant(p, exports, bound,
        data.record_run(p, records, initial_state, destinations, lefts, rights, steps),
        execution.execution_readiness(p, initial_ready, destinations, steps),
        execution.execution_values(p, ph._named(p, joined.RECORD_JOIN), initial_values,
            destinations, lefts, rights, steps))


def _prefix_refinement[C: Int](mut p: ph._Builder[C], exports: RecordTraceRefinementExports,
    records: data.RecordTraceDataExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var initial_state = ph._var(p, data.record_state_type(p))
    var initial_ready = ph._var(p, ph._store(p))
    var initial_values = ph._var(p, ph._store(p))
    var support = ph._var(p, ph._pi(p, ph._var(p, ph._nat(p)), ph._store(p)))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
    var steps = ph._var(p, ph._nat(p))
    var initial_invariant = ph._var(p,
        record_invariant(p, exports, bound, initial_state, initial_ready, initial_values))
    var k = ph._var(p, ph._nat(p))
    var structural = ph._var(p, trace.support_prefix(p,
        initial_ready, support, destinations, lefts, rights, k))
    var motive = ph._lam(p, k, ph._pi(p, structural,
        _prefix_result(p, exports, records, bound, initial_state, initial_ready, initial_values,
            destinations, lefts, rights, k)))
    var zero_guard = ph._var(p, trace.support_prefix(p,
        initial_ready, support, destinations, lefts, rights, ph._zero(p)))
    var base = ph._lam(p, zero_guard, initial_invariant)
    k = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, k))
    var next_structural = ph._var(p, trace.support_prefix(p,
        initial_ready, support, destinations, lefts, rights, ph._succ(p, k)))
    var prefix_parts: List[Int] = [trace.support_prefix(p,
        initial_ready, support, destinations, lefts, rights, k),
        trace.support_step(p, initial_ready, support, destinations, lefts, rights, k)]
    var previous = algebra._project(p, next_structural, prefix_parts, 0)
    var current_step = algebra._project(p, next_structural, prefix_parts, 1)
    var step_parts = trace.support_step_propositions(p,
        initial_ready, support, destinations, lefts, rights, k)
    var left_ready = algebra._project(p, current_step, step_parts, 0)
    var right_ready = algebra._project(p, current_step, step_parts, 1)
    var old_state = data.record_run(p, records, initial_state, destinations, lefts, rights, k)
    var old_ready = execution.execution_readiness(p, initial_ready, destinations, k)
    var old_values = execution.execution_values(p, ph._named(p, joined.RECORD_JOIN),
        initial_values, destinations, lefts, rights, k)
    var destination = ph._app(p, destinations, k)
    var left = ph._app(p, lefts, k)
    var right = ph._app(p, rights, k)
    var next_invariant = _call(p, exports.step_preservation,
        [bound, old_state, old_ready, old_values, destination, left, right,
            ph._app(p, ih, previous), left_ready, right_ready])
    var step = ph._lams(p, [k, ih, next_structural], next_invariant)
    structural = ph._var(p, trace.support_prefix(p,
        initial_ready, support, destinations, lefts, rights, steps))
    var body = ph._app(p, ph._induct(p, motive, base, step, steps), structural)
    var result = _prefix_result(p, exports, records, bound, initial_state, initial_ready,
        initial_values, destinations, lefts, rights, steps)
    var binders: List[Int] = [bound, initial_state, initial_ready, initial_values,
        support, destinations, lefts, rights, steps, initial_invariant, structural]
    return _declare(p, ph._pis(p, binders, result), ph._lams(p, binders, body))


def append_nat_record_trace_refinement[C: Int](mut p: ph._Builder[C],
    enc: encoding.EncodedRecordExports, valid: basics.RecordBasicsExports,
    records: data.RecordTraceDataExports) -> RecordTraceRefinementExports:
    var first = p.bundle.definition_count
    var original_root = p.bundle.root
    var ready_valid = _ready_valid_data(p, valid)
    var encoded_equal = _encoded_equal_data(p, enc)
    var exports = RecordTraceRefinementExports(ready_valid, encoded_equal, -1, -1, -1)
    exports.invariant = _invariant_data(p, exports)
    exports.step_preservation = _step_preservation(p, exports, enc, valid, records)
    exports.prefix_refinement = _prefix_refinement(p, exports, records)
    if (p.bundle.count < 0 or p.bundle.definition_count != first + RECORD_TRACE_REFINEMENT_DEFINITIONS
        or p.bundle.root != original_root):
        p.bundle.count = -1
        return RecordTraceRefinementExports(-1, -1, -1, -1, -1)
    return exports
