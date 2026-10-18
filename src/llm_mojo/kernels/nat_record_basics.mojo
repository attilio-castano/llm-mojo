"""Ordinary record validity, transport and homomorphism proof terms.

A record has Nat fields0/1/2, with a bounded token and Boolean invalidity flag.
This producer grants no judgment; the independent profile checks its exports.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.nat_join_bits as bits
import llm_mojo.kernels.finishing_join_terms as joined
import llm_mojo.kernels.nat_score_link as ar
import llm_mojo.kernels.nat_record_encoding as enc
import llm_mojo.kernels.nat_record_laws as laws
import llm_mojo.kernels.nat_record_trace_data as data


@fieldwise_init
struct RecordBasicsExports(ImplicitlyCopyable):
    var valid: Int
    var encode_congruent: Int
    var validity_transport: Int
    var merge_valid: Int
    var merge_encoding: Int


def _call[C: Int](mut p: ph._Builder[C], definition: Int, args: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), args)


def _declare[C: Int](mut p: ph._Builder[C], type: Int, body: Int) -> Int:
    var definition = p.bundle.definition_count
    ph._declare(p, type, body, definition)
    return definition


def field[C: Int](mut p: ph._Builder[C], record: Int, position: Int) -> Int:
    var index = ph._zero(p)
    for _ in range(position):
        index = ph._succ(p, index)
    return ph._app(p, record, index)


def valid_record[C: Int](mut p: ph._Builder[C], exports: RecordBasicsExports,
    bound: Int, record: Int) -> Int:
    return _call(p, exports.valid, [bound, record])


def encoded[C: Int](mut p: ph._Builder[C], encoding: enc.EncodedRecordExports,
    bound: Int, record: Int) -> Int:
    return enc.encoded_record(p, encoding, bound, field(p, record, 0),
        field(p, record, 1), field(p, record, 2))


def valid_parts[C: Int](mut p: ph._Builder[C], bound: Int, record: Int) -> List[Int]:
    return [ar.score_le(p, field(p, record, 1), bound), bits.record_is_bit(p, field(p, record, 2))]


def valid_projection[C: Int](mut p: ph._Builder[C], bound: Int,
    record: Int, evidence: Int, position: Int) -> Int:
    return a._project(p, evidence, valid_parts(p, bound, record), position)


def _valid_data[C: Int](mut p: ph._Builder[C]) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var record = ph._var(p, data.record_type(p))
    return _declare(p, ph._pis(p, [bound, record], ph._prop(p)),
        ph._lams(p, [bound, record], a._and_type(p, valid_parts(p, bound, record))))


def _encode_congruent[C: Int](mut p: ph._Builder[C], encoding: enc.EncodedRecordExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var left = ph._var(p, data.record_type(p))
    var right = ph._var(p, data.record_type(p))
    var equalities = List[Int]()
    for position in range(3):
        equalities.append(ph._var(p, ph._eq(p, field(p, left, position), field(p, right, position))))
    var old_fields: List[Int] = [field(p, left, 0), field(p, left, 1), field(p, left, 2)]
    var actual = encoded(p, encoding, bound, left)
    var body = ph._refl(p, actual)
    var previous = actual
    for position in range(3):
        var slot = ph._var(p, ph._nat(p))
        var next_fields = old_fields.copy()
        next_fields[position] = slot
        var function = ph._lam(p, slot, enc.encoded_record(p, encoding, bound,
            next_fields[0], next_fields[1], next_fields[2]))
        var edge = ph._cong(p, function, old_fields[position], field(p, right, position), equalities[position])
        old_fields[position] = field(p, right, position)
        var next = enc.encoded_record(p, encoding, bound, old_fields[0], old_fields[1], old_fields[2])
        body = ph._trans(p, actual, previous, next, body, edge)
        previous = next
    var binders: List[Int] = [bound, left, right]
    for equality in equalities:
        binders.append(equality)
    return _declare(p, ph._pis(p, binders, ph._eq(p, actual, encoded(p, encoding, bound, right))),
        ph._lams(p, binders, body))


def _validity_transport[C: Int](mut p: ph._Builder[C], exports: RecordBasicsExports) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var left = ph._var(p, data.record_type(p))
    var right = ph._var(p, data.record_type(p))
    var valid = ph._var(p, valid_record(p, exports, bound, left))
    var token_equal = ph._var(p, ph._eq(p, field(p, left, 1), field(p, right, 1)))
    var bad_equal = ph._var(p, ph._eq(p, field(p, left, 2), field(p, right, 2)))
    var slot = ph._var(p, ph._nat(p))
    var token_bound = ph._subst(p, ph._lam(p, slot, ar.score_le(p, slot, bound)),
        token_equal, valid_projection(p, bound, left, valid, 0))
    slot = ph._var(p, ph._nat(p))
    var bit = ph._subst(p, ph._lam(p, slot, bits.record_is_bit(p, slot)),
        bad_equal, valid_projection(p, bound, left, valid, 1))
    var result = valid_record(p, exports, bound, right)
    var body = a._and_intro(p, valid_parts(p, bound, right), [token_bound, bit])
    var binders: List[Int] = [bound, left, right, valid, token_equal, bad_equal]
    return _declare(p, ph._pis(p, binders, result), ph._lams(p, binders, body))


def _merge[C: Int](mut p: ph._Builder[C], exports: RecordBasicsExports,
    encoding: enc.EncodedRecordExports, operations: laws.RecordLawsExports,
    records: data.RecordTraceDataExports, preserve_encoding: Bool) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var left = ph._var(p, data.record_type(p))
    var right = ph._var(p, data.record_type(p))
    var valid_left = ph._var(p, valid_record(p, exports, bound, left))
    var valid_right = ph._var(p, valid_record(p, exports, bound, right))
    var ra = field(p, left, 0)
    var rb = field(p, right, 0)
    var ta = field(p, left, 1)
    var tb = field(p, right, 1)
    var ba = field(p, left, 2)
    var bb = field(p, right, 2)
    var token_a = valid_projection(p, bound, left, valid_left, 0)
    var token_b = valid_projection(p, bound, right, valid_right, 0)
    var bit_a = valid_projection(p, bound, left, valid_left, 1)
    var bit_b = valid_projection(p, bound, right, valid_right, 1)
    var merged = data.record_merge(p, records, left, right)
    var result: Int
    var body: Int
    if preserve_encoding:
        result = ph._eq(p, encoded(p, encoding, bound, merged),
            joined.record_join(p, encoded(p, encoding, bound, left), encoded(p, encoding, bound, right)))
        body = _call(p, operations.homomorphism,
            [bound, ra, rb, ta, tb, ba, bb, token_a, token_b, bit_a, bit_b])
    else:
        var token_bound = _call(p, operations.token_bounded, [bound, ra, rb, ta, tb, token_a, token_b])
        var bit = _call(p, operations.bit_closed, [ba, bb, bit_a, bit_b])
        result = valid_record(p, exports, bound, merged)
        body = a._and_intro(p, valid_parts(p, bound, merged), [token_bound, bit])
    var binders: List[Int] = [bound, left, right, valid_left, valid_right]
    return _declare(p, ph._pis(p, binders, result), ph._lams(p, binders, body))


def append_nat_record_basics[C: Int](mut p: ph._Builder[C],
    encoding: enc.EncodedRecordExports, operations: laws.RecordLawsExports,
    records: data.RecordTraceDataExports) -> RecordBasicsExports:
    var exports = RecordBasicsExports(_valid_data(p), -1, -1, -1, -1)
    exports.encode_congruent = _encode_congruent(p, encoding)
    exports.validity_transport = _validity_transport(p, exports)
    exports.merge_valid = _merge(p, exports, encoding, operations, records, False)
    exports.merge_encoding = _merge(p, exports, encoding, operations, records, True)
    return exports
