"""Validity and encoding proofs for the independent record input fold.

Only inputs Before(i,N) must be valid. Nat induction quantifies that prefix
premise inside its motive; unread slots remain arbitrary. These three ordinary
definitions propose syntax, preserving the supplied proof root. Their public
types require independent checking; native word/source/GPU refinement is separate.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.finishing_execution_terms as execution
import llm_mojo.kernels.finishing_join_terms as join
import llm_mojo.kernels.nat_join_bool as boolean
import llm_mojo.kernels.nat_record_encoding as encoding
import llm_mojo.kernels.nat_record_laws as laws
import llm_mojo.kernels.nat_record_trace_data as data
import llm_mojo.kernels.nat_record_prefix as prefix
import llm_mojo.kernels.nat_record_basics as basics

comptime RECORD_FOLD_DEFINITIONS = 3


@fieldwise_init
struct RecordFoldExports(ImplicitlyCopyable):
    # valid_inputs: (M,R,T,B,N) -> Prop, with R/T/B : Nat->Nat.
    # Body: forall i, Before(i,N) -> Valid(M,Record(Ri,Ti,Bi)).
    var valid_inputs: Int
    # encoded_inputs: (M,R,T,B) -> Nat->Nat; no count or validity premise.
    var encoded_inputs: Int
    # correct: (M,R,T,B,N,ValidInputs(M,R,T,B,N)) -> ChurchAnd(
    #   Valid(M,Fold(M,R,T,B,N)),
    #   Eq(Encode(M,Fold), FlatFold27(Join86,0,EncodedInputs(M,R,T,B),N))).
    var correct: Int


def _call[C: Int](mut p: ph._Builder[C], definition: Int, arguments: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), arguments)


def _declare[C: Int](mut p: ph._Builder[C], type: Int, body: Int) -> Int:
    var definition = p.bundle.definition_count
    ph._declare(p, type, body, definition)
    return definition


def record_valid_inputs[C: Int](mut p: ph._Builder[C], exports: RecordFoldExports,
    bound: Int, ranks: Int, tokens: Int, bads: Int, count: Int) -> Int:
    return _call(p, exports.valid_inputs, [bound, ranks, tokens, bads, count])


def record_encoded_inputs[C: Int](mut p: ph._Builder[C], exports: RecordFoldExports,
    bound: Int, ranks: Int, tokens: Int, bads: Int) -> Int:
    return _call(p, exports.encoded_inputs, [bound, ranks, tokens, bads])


def _input[C: Int](mut p: ph._Builder[C], records: data.RecordTraceDataExports,
    ranks: Int, tokens: Int, bads: Int, slot: Int) -> Int:
    return data.record_fields(p, records, ph._app(p, ranks, slot),
        ph._app(p, tokens, slot), ph._app(p, bads, slot))


def _valid_inputs[C: Int](mut p: ph._Builder[C], records: data.RecordTraceDataExports,
    valid: basics.RecordBasicsExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ranks = ph._var(p, ph._store(p))
    var tokens = ph._var(p, ph._store(p))
    var bads = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var slot = ph._var(p, ph._nat(p))
    var inside = ph._var(p, a._before(p, slot, count))
    var row = _input(p, records, ranks, tokens, bads, slot)
    var body = ph._pis(p, [slot, inside], basics.valid_record(p, valid, bound, row))
    var binders: List[Int] = [bound, ranks, tokens, bads, count]
    return _declare(p, ph._pis(p, binders, ph._prop(p)), ph._lams(p, binders, body))


def _encoded_inputs[C: Int](mut p: ph._Builder[C],
    enc: encoding.EncodedRecordExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ranks = ph._var(p, ph._store(p))
    var tokens = ph._var(p, ph._store(p))
    var bads = ph._var(p, ph._store(p))
    var slot = ph._var(p, ph._nat(p))
    var body = encoding.encoded_record(p, enc, bound, ph._app(p, ranks, slot),
        ph._app(p, tokens, slot), ph._app(p, bads, slot))
    var binders: List[Int] = [bound, ranks, tokens, bads]
    return _declare(p, ph._pis(p, binders, ph._store(p)),
        ph._lams(p, [bound, ranks, tokens, bads, slot], body))


def _flat[C: Int](mut p: ph._Builder[C], encoded_inputs: Int,
    bound: Int, ranks: Int, tokens: Int, bads: Int, count: Int) -> Int:
    var inputs = _call(p, encoded_inputs, [bound, ranks, tokens, bads])
    return execution.execution_flat_fold(p, ph._named(p, join.RECORD_JOIN),
        ph._zero(p), inputs, count)


def _parts[C: Int](mut p: ph._Builder[C], enc: encoding.EncodedRecordExports,
    records: data.RecordTraceDataExports, valid: basics.RecordBasicsExports,
    encoded_inputs: Int, bound: Int, ranks: Int, tokens: Int, bads: Int, count: Int) -> List[Int]:
    var fold = data.record_fold(p, records, bound, ranks, tokens, bads, count)
    return [basics.valid_record(p, valid, bound, fold), ph._eq(p,
        basics.encoded(p, enc, bound, fold),
        _flat(p, encoded_inputs, bound, ranks, tokens, bads, count))]


def _correct[C: Int](mut p: ph._Builder[C], enc: encoding.EncodedRecordExports,
    operations: laws.RecordLawsExports, records: data.RecordTraceDataExports,
    prefixes: prefix.RecordPrefixExports, valid: basics.RecordBasicsExports,
    valid_inputs: Int, encoded_inputs: Int) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ranks = ph._var(p, ph._store(p))
    var tokens = ph._var(p, ph._store(p))
    var bads = ph._var(p, ph._store(p))
    var n = ph._var(p, ph._nat(p))
    var all = ph._var(p, _call(p, valid_inputs, [bound, ranks, tokens, bads, n]))
    var motive = ph._lam(p, n, ph._pi(p, all, a._and_type(p,
        _parts(p, enc, records, valid, encoded_inputs, bound, ranks, tokens, bads, n))))

    # Empty validity is proved rather than requested from the input stores.
    # Fold(0) is Record(0,M,0); FlatFold(0) is the mathematical Join identity0.
    var zero = ph._zero(p)
    var empty = data.record_fields(p, records, zero, bound, zero)
    var empty_valid = a._and_intro(p, basics.valid_parts(p, bound, empty),
        [_call(p, operations.le_refl, [bound]), boolean._intro(p, False)])
    var empty_encoding = _call(p, enc.identity, [bound])
    var empty_inputs = ph._var(p, _call(p, valid_inputs, [bound, ranks, tokens, bads, zero]))
    var base = ph._lam(p, empty_inputs, a._and_intro(p,
        _parts(p, enc, records, valid, encoded_inputs, bound, ranks, tokens, bads, zero),
        [empty_valid, empty_encoding]))

    var predecessor = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, predecessor))
    var successor = ph._succ(p, predecessor)
    var successor_inputs = ph._var(p,
        _call(p, valid_inputs, [bound, ranks, tokens, bads, successor]))
    # Restrict the successor premise to Before(i,n). No validity of any slot
    # beyond the requested prefix is asserted or used.
    var slot = ph._var(p, ph._nat(p))
    var inside = ph._var(p, a._before(p, slot, predecessor))
    var lifted = _call(p, prefixes.weakening, [slot, predecessor, inside])
    var smaller_inputs = ph._lams(p, [slot, inside], ph._apps(p, successor_inputs, [slot, lifted]))
    var previous_result = ph._app(p, ih, smaller_inputs)
    var previous_parts = _parts(p, enc, records, valid, encoded_inputs,
        bound, ranks, tokens, bads, predecessor)
    var previous_valid = a._project(p, previous_result, previous_parts, 0)
    var previous_encoding = a._project(p, previous_result, previous_parts, 1)
    var previous = data.record_fold(p, records, bound, ranks, tokens, bads, predecessor)

    var current = _input(p, records, ranks, tokens, bads, predecessor)
    var current_inside = _call(p, prefixes.diagonal, [predecessor])
    var current_valid = ph._apps(p, successor_inputs, [predecessor, current_inside])
    var next_valid = _call(p, valid.merge_valid,
        [bound, previous, current, previous_valid, current_valid])
    var merged_encoding = _call(p, valid.merge_encoding,
        [bound, previous, current, previous_valid, current_valid])
    var encoded_previous = basics.encoded(p, enc, bound, previous)
    var encoded_current = basics.encoded(p, enc, bound, current)
    var flat_previous = _flat(p, encoded_inputs, bound, ranks, tokens, bads, predecessor)
    var changed = ph._var(p, ph._nat(p))
    var join_context = ph._lam(p, changed, join.record_join(p, changed, encoded_current))
    var advance_encoding = ph._cong(p, join_context,
        encoded_previous, flat_previous, previous_encoding)
    var next = data.record_fold(p, records, bound, ranks, tokens, bads, successor)
    var middle = join.record_join(p, encoded_previous, encoded_current)
    var flat_next = _flat(p, encoded_inputs, bound, ranks, tokens, bads, successor)
    var next_encoding = ph._trans(p, basics.encoded(p, enc, bound, next),
        middle, flat_next, merged_encoding, advance_encoding)
    var next_parts = _parts(p, enc, records, valid, encoded_inputs,
        bound, ranks, tokens, bads, successor)
    var step = ph._lams(p, [predecessor, ih, successor_inputs],
        a._and_intro(p, next_parts, [next_valid, next_encoding]))

    var count = ph._var(p, ph._nat(p))
    var binders: List[Int] = [bound, ranks, tokens, bads, count]
    return _declare(p, ph._pis(p, binders, ph._app(p, motive, count)),
        ph._lams(p, binders, ph._induct(p, motive, base, step, count)))


def append_nat_record_fold[C: Int](mut p: ph._Builder[C],
    enc: encoding.EncodedRecordExports, operations: laws.RecordLawsExports,
    records: data.RecordTraceDataExports, prefixes: prefix.RecordPrefixExports,
    valid: basics.RecordBasicsExports) -> RecordFoldExports:
    var first = p.bundle.definition_count
    var original_root = p.bundle.root
    var valid_inputs = _valid_inputs(p, records, valid)
    var encoded_inputs = _encoded_inputs(p, enc)
    var correct = _correct(p, enc, operations, records, prefixes, valid, valid_inputs, encoded_inputs)
    if (p.bundle.count < 0 or p.bundle.definition_count != first + RECORD_FOLD_DEFINITIONS
        or p.bundle.root != original_root):
        p.bundle.count = -1
        return RecordFoldExports(-1, -1, -1)
    return RecordFoldExports(valid_inputs, encoded_inputs, correct)
