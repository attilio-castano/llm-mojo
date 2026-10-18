"""A complete legal record trace equals its independent input fold.

Count, stores, trace length and token bound are universally bound. Premises
state input validity and structural readiness/support. The desired output
value, record algebra and encoding bridge are derived, never assumed.
This untrusted producer adds no kernel rules or native correspondence claim.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.finishing_execution_terms as execution
import llm_mojo.kernels.finishing_execution_leaves as leaf
import llm_mojo.kernels.finishing_execution_trace as trace
import llm_mojo.kernels.finishing_join_terms as join
import llm_mojo.kernels.nat_record_encoding as encoding
import llm_mojo.kernels.nat_record_trace_data as data
import llm_mojo.kernels.nat_record_basics as basics
import llm_mojo.kernels.nat_record_fold as fold
import llm_mojo.kernels.nat_record_trace_refinement as refinement


def _call[C: Int](mut p: ph._Builder[C], definition: Int, args: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), args)


def append_nat_record_canonical[C: Int](mut p: ph._Builder[C],
    enc: encoding.EncodedRecordExports, records: data.RecordTraceDataExports,
    valid: basics.RecordBasicsExports, inputs: fold.RecordFoldExports,
    proofs: refinement.RecordTraceRefinementExports, initialization: Int) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ranks = ph._var(p, ph._store(p))
    var tokens = ph._var(p, ph._store(p))
    var bads = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var support = ph._var(p, leaf.execution_support_type(p))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
    var input_valid = ph._var(p, fold.record_valid_inputs(p, inputs, bound, ranks, tokens, bads, count))
    var at = ph._var(p, ph._nat(p))
    var empty_support = ph._var(p, ph._pi(p, at, ph._eq(p,
        ph._app(p, ph._app(p, support, count), at), ph._zero(p))))
    var source = ph._var(p, ph._nat(p))
    var before = ph._var(p, a._before(p, source, count))
    at = ph._var(p, ph._nat(p))
    var input_support = ph._var(p, ph._pis(p, [source, before, at], ph._eq(p,
        ph._app(p, ph._app(p, support, source), at), ph._app(p, a._unit(p, source), at))))
    var steps = ph._var(p, ph._nat(p))
    var initial_ready = leaf.execution_initial_ready(p, count)
    var structural = ph._var(p, trace.support_prefix(p, initial_ready,
        support, destinations, lefts, rights, steps))
    var root = ph._var(p, ph._nat(p))
    var ready = ph._var(p, ph._eq(p, ph._app(p,
        execution.execution_readiness(p, initial_ready, destinations, steps), root), ph._one(p)))
    var complete = ph._var(p, a._prefix(p, ph._app(p, support, root), count))

    var initial_records = data.record_initial(p, records, bound, count, ranks, tokens, bads)
    var encoded_inputs = fold.record_encoded_inputs(p, inputs, bound, ranks, tokens, bads)
    var initial_values = leaf.execution_initial_values(p, ph._zero(p), encoded_inputs, count)
    var initial_invariant = _call(p, initialization, [bound, ranks, tokens, bads, count, input_valid])
    var combined = _call(p, proofs.prefix_refinement, [bound, initial_records, initial_ready,
        initial_values, support, destinations, lefts, rights, steps, initial_invariant, structural])
    var final_state = data.record_run(p, records, initial_records, destinations, lefts, rights, steps)
    var final_ready = execution.execution_readiness(p, initial_ready, destinations, steps)
    var final_values = execution.execution_values(p, ph._named(p, join.RECORD_JOIN), initial_values,
        destinations, lefts, rights, steps)
    var invariant_parts = refinement.record_invariant_parts(p, proofs, bound,
        final_state, final_ready, final_values)
    var ready_valid = a._project(p, combined, invariant_parts, 0)
    var encoded_equal = a._project(p, combined, invariant_parts, 1)
    var actual = data.record_row(p, final_state, root)
    var actual_valid = ph._apps(p, ready_valid, [root, ready])
    var first = ph._app(p, encoded_equal, root)
    var second = _call(p, join.RECORD_CANONICAL_TRACE, [encoded_inputs, count, support,
        destinations, lefts, rights, empty_support, input_support, steps, structural, root, ready, complete])
    var flat = execution.execution_flat_fold(p, ph._named(p, join.RECORD_JOIN),
        ph._zero(p), encoded_inputs, count)
    var folded = data.record_fold(p, records, bound, ranks, tokens, bads, count)
    var fold_proof = _call(p, inputs.correct, [bound, ranks, tokens, bads, count, input_valid])
    var fold_parts: List[Int] = [basics.valid_record(p, valid, bound, folded),
        ph._eq(p, basics.encoded(p, enc, bound, folded), flat)]
    var folded_valid = a._project(p, fold_proof, fold_parts, 0)
    var fold_equal = a._project(p, fold_proof, fold_parts, 1)
    var actual_encoded = basics.encoded(p, enc, bound, actual)
    var folded_encoded = basics.encoded(p, enc, bound, folded)
    var encoded_result = ph._trans(p, actual_encoded, ph._app(p, final_values, root), flat, first, second)
    encoded_result = ph._trans(p, actual_encoded, flat, folded_encoded, encoded_result,
        ph._sym(p, folded_encoded, flat, fold_equal))
    var body = _call(p, enc.injective, [bound,
        basics.field(p, actual, 0), basics.field(p, folded, 0),
        basics.field(p, actual, 1), basics.field(p, folded, 1),
        basics.field(p, actual, 2), basics.field(p, folded, 2),
        basics.valid_projection(p, bound, actual, actual_valid, 0),
        basics.valid_projection(p, bound, folded, folded_valid, 0),
        basics.valid_projection(p, bound, actual, actual_valid, 1),
        basics.valid_projection(p, bound, folded, folded_valid, 1), encoded_result])
    var field_equations = List[Int]()
    for position in range(3):
        field_equations.append(ph._eq(p, basics.field(p, actual, position), basics.field(p, folded, position)))
    var result = a._and_type(p, field_equations)
    var binders: List[Int] = [bound, ranks, tokens, bads, count, support, destinations, lefts, rights,
        input_valid, empty_support, input_support, steps, structural, root, ready, complete]
    var definition = p.bundle.definition_count
    ph._declare(p, ph._pis(p, binders, result), ph._lams(p, binders, body), definition)
    p.bundle.root = ph._named(p, definition)
    return definition
