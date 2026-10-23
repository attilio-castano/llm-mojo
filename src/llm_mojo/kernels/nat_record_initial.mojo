"""Ordinary initialization proof for correlated record execution.

Only ready inputs require bounded tokens and Boolean flags. Encoding agreement
is pointwise over every slot, including arbitrary unread input functions.
The sentinel's validity and empty encoding are derived rather than assumed.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.finishing_execution_leaves as leaf
import llm_mojo.kernels.nat_join_bool as bo
import llm_mojo.kernels.nat_record_encoding as encoding
import llm_mojo.kernels.nat_record_laws as laws
import llm_mojo.kernels.nat_record_trace_data as data
import llm_mojo.kernels.nat_record_basics as basics
import llm_mojo.kernels.nat_record_fold as fold
import llm_mojo.kernels.nat_record_trace_refinement as refinement


def _call[C: Int](mut p: ph._Builder[C], definition: Int, args: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), args)


def _apart_reverse[C: Int](mut p: ph._Builder[C], left: Int,
    right: Int, apart: Int) -> Int:
    var equal = ph._var(p, ph._eq(p, right, left))
    return ph._lam(p, equal, ph._app(p, apart, ph._sym(p, right, left, equal)))


def _goal[C: Int](mut p: ph._Builder[C], valid: basics.RecordBasicsExports,
    enc: encoding.EncodedRecordExports, bound: Int, state: Int,
    ready: Int, values: Int, slot: Int, validity: Bool) -> Int:
    var row = data.record_row(p, state, slot)
    if not validity:
        return ph._eq(p, basics.encoded(p, enc, bound, row), ph._app(p, values, slot))
    var guard = ph._var(p, ph._eq(p, ph._app(p, ready, slot), ph._one(p)))
    return ph._pi(p, guard, basics.valid_record(p, valid, bound, row))


def _pointwise[C: Int](mut p: ph._Builder[C], enc: encoding.EncodedRecordExports,
    operations: laws.RecordLawsExports, records: data.RecordTraceDataExports,
    valid: basics.RecordBasicsExports, inputs: fold.RecordFoldExports,
    bound: Int, count: Int, ranks: Int, tokens: Int, bads: Int,
    input_valid: Int, validity: Bool) -> Int:
    var state = data.record_initial(p, records, bound, count, ranks, tokens, bads)
    var ready = leaf.execution_initial_ready(p, count)
    var encoded_inputs = fold.record_encoded_inputs(p, inputs, bound, ranks, tokens, bads)
    var values = leaf.execution_initial_values(p, ph._zero(p), encoded_inputs, count)
    var slot = ph._var(p, ph._nat(p))
    var goal = _goal(p, valid, enc, bound, state, ready, values, slot, validity)
    var equal = ph._var(p, ph._eq(p, slot, count))
    var row_at_count = data.record_row(p, state, count)
    var empty = data.record_fields(p, records, ph._zero(p), bound, ph._zero(p))
    var stores: List[Int] = [ranks, tokens, bads]
    var empty_fields: List[Int] = [ph._zero(p), bound, ph._zero(p)]
    var hits = List[Int]()
    for position in range(3):
        hits.append(_call(p, ph.PHASE_UPDATE_HIT, [stores[position], count, empty_fields[position]]))
    var at_count: Int
    if validity:
        var empty_valid = a._and_intro(p, basics.valid_parts(p, bound, empty),
            [_call(p, operations.le_refl, [bound]), bo._intro(p, False)])
        var valid_count = _call(p, valid.validity_transport, [bound, empty, row_at_count,
            empty_valid, ph._sym(p, basics.field(p, row_at_count, 1), bound, hits[1]),
            ph._sym(p, basics.field(p, row_at_count, 2), ph._zero(p), hits[2])])
        var ignored = ph._var(p, ph._eq(p, ph._app(p, ready, count), ph._one(p)))
        at_count = ph._lam(p, ignored, valid_count)
    else:
        var first = _call(p, valid.encode_congruent,
            [bound, row_at_count, empty, hits[0], hits[1], hits[2]])
        var second = _call(p, enc.identity, [bound])
        var actual = basics.encoded(p, enc, bound, row_at_count)
        var middle = basics.encoded(p, enc, bound, empty)
        var combined = ph._trans(p, actual, middle, ph._zero(p), first, second)
        var value_hit = _call(p, ph.PHASE_UPDATE_HIT, [encoded_inputs, count, ph._zero(p)])
        at_count = ph._trans(p, actual, ph._zero(p), ph._app(p, values, count), combined,
            ph._sym(p, ph._app(p, values, count), ph._zero(p), value_hit))
    var index = ph._var(p, ph._nat(p))
    var observation_goal = ph._lam(p, index,
        _goal(p, valid, enc, bound, state, ready, values, index, validity))
    var yes = ph._lam(p, equal, ph._subst(p, observation_goal,
        ph._sym(p, slot, count, equal), at_count))

    var apart = ph._var(p, ph._apart(p, slot, count))
    var reversed = _apart_reverse(p, slot, count, apart)
    var row = data.record_row(p, state, slot)
    var input = data.record_fields(p, records, ph._app(p, ranks, slot),
        ph._app(p, tokens, slot), ph._app(p, bads, slot))
    var misses = List[Int]()
    for position in range(3):
        misses.append(_call(p, ph.PHASE_UPDATE_MISS,
            [stores[position], count, empty_fields[position], slot, reversed]))
    var other: Int
    if validity:
        var guard = ph._var(p, ph._eq(p, ph._app(p, ready, slot), ph._one(p)))
        var at = ph._var(p, ph._nat(p))
        var flags = ph._lam(p, at, leaf.execution_before_flag(p, at, count))
        var ready_miss = _call(p, ph.PHASE_UPDATE_MISS, [flags, count, ph._one(p), slot, reversed])
        var before_flag = leaf.execution_before_flag(p, slot, count)
        var flag_one = ph._trans(p, before_flag, ph._app(p, ready, slot), ph._one(p),
            ph._sym(p, ph._app(p, ready, slot), before_flag, ready_miss), guard)
        var before = _call(p, leaf.EXEC_FLAG_BOUND_SOUND, [slot, count, flag_one])
        var input_proof = ph._apps(p, input_valid, [slot, before])
        var result = _call(p, valid.validity_transport, [bound, input, row, input_proof,
            ph._sym(p, basics.field(p, row, 1), ph._app(p, tokens, slot), misses[1]),
            ph._sym(p, basics.field(p, row, 2), ph._app(p, bads, slot), misses[2])])
        other = ph._lam(p, guard, result)
    else:
        var first = _call(p, valid.encode_congruent, [bound, row, input, misses[0], misses[1], misses[2]])
        var value_miss = _call(p, ph.PHASE_UPDATE_MISS,
            [encoded_inputs, count, ph._zero(p), slot, reversed])
        other = ph._trans(p, basics.encoded(p, enc, bound, row), ph._app(p, encoded_inputs, slot),
            ph._app(p, values, slot), first,
            ph._sym(p, ph._app(p, values, slot), ph._app(p, encoded_inputs, slot), value_miss))
    var no = ph._lam(p, apart, other)
    return ph._lam(p, slot, _call(p, ph.PHASE_DECIDE, [slot, count, goal, yes, no]))


def append_nat_record_initial[C: Int](mut p: ph._Builder[C],
    enc: encoding.EncodedRecordExports, operations: laws.RecordLawsExports,
    records: data.RecordTraceDataExports, valid: basics.RecordBasicsExports,
    inputs: fold.RecordFoldExports, proofs: refinement.RecordTraceRefinementExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ranks = ph._var(p, ph._store(p))
    var tokens = ph._var(p, ph._store(p))
    var bads = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var input_valid = ph._var(p, fold.record_valid_inputs(p, inputs, bound, ranks, tokens, bads, count))
    var state = data.record_initial(p, records, bound, count, ranks, tokens, bads)
    var ready = leaf.execution_initial_ready(p, count)
    var encoded_inputs = fold.record_encoded_inputs(p, inputs, bound, ranks, tokens, bads)
    var values = leaf.execution_initial_values(p, ph._zero(p), encoded_inputs, count)
    var result = refinement.record_invariant(p, proofs, bound, state, ready, values)
    var parts = refinement.record_invariant_parts(p, proofs, bound, state, ready, values)
    var body = a._and_intro(p, parts, [
        _pointwise(p, enc, operations, records, valid, inputs, bound, count, ranks, tokens, bads, input_valid, True),
        _pointwise(p, enc, operations, records, valid, inputs, bound, count, ranks, tokens, bads, input_valid, False)])
    var binders: List[Int] = [bound, ranks, tokens, bads, count, input_valid]
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, result), ph._lams(p, binders, body), definition)
    return definition
