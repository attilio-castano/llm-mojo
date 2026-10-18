"""Total Nat score/Boolean-flag join and checked trace instantiation terms.

Every Nat has a score/parity decomposition. The operation is total: a bounded
decoder must not silently canonicalize arbitrary Nat inputs. This library
proves the algebra used by the logical trace theorem; native UInt32 rank/token
merge refinement and GPU execution remain separate obligations.
"""
import llm_mojo.kernels.proof_term_syntax as tk
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_terms as ex
import llm_mojo.kernels.finishing_execution_algebra as a
import llm_mojo.kernels.finishing_execution_leaves as leaf
import llm_mojo.kernels.finishing_execution_trace as trace
import llm_mojo.kernels.nat_join_max as mx
import llm_mojo.kernels.nat_join_bits as bits
import llm_mojo.kernels.nat_join_bool as bo

comptime RECORD_JOIN = 86
comptime RECORD_JOIN_HALF = 87
comptime RECORD_JOIN_PARITY = 88
comptime RECORD_JOIN_LEFT_UNIT = 89
comptime RECORD_JOIN_RIGHT_UNIT = 90
comptime RECORD_JOIN_COMMUTATIVE = 91
comptime RECORD_JOIN_ASSOCIATIVE = 92
comptime RECORD_JOIN_IDEMPOTENT = 93
comptime RECORD_CANONICAL_TRACE = 94
comptime RECORD_JOIN_DEFINITIONS = 95


def _call[C: Int](mut p: ph._Builder[C], definition: Int, arguments: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), arguments)


def record_join[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return _call(p, RECORD_JOIN, [left, right])


def _score[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return mx.record_max(p, bits.record_half(p, left), bits.record_half(p, right))


def _flag[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return bits.record_bool_or(p, bits.record_parity(p, left), bits.record_parity(p, right))


def _bit[C: Int](mut p: ph._Builder[C], value: Int) -> Int:
    return _call(p, bits.RECORD_PARITY_BIT, [value])


def _pack_congruence[C: Int](mut p: ph._Builder[C], left_score: Int,
    right_score: Int, left_flag: Int, right_flag: Int, score_eq: Int, flag_eq: Int) -> Int:
    var score = ph._var(p, ph._nat(p))
    var first = ph._cong(p, ph._lam(p, score, bits.record_pack(p, score, left_flag)),
        left_score, right_score, score_eq)
    var flag = ph._var(p, ph._nat(p))
    var second = ph._cong(p, ph._lam(p, flag, bits.record_pack(p, right_score, flag)),
        left_flag, right_flag, flag_eq)
    return ph._trans(p, bits.record_pack(p, left_score, left_flag),
        bits.record_pack(p, right_score, left_flag),
        bits.record_pack(p, right_score, right_flag), first, second)


def _data[C: Int](mut p: ph._Builder[C]):
    var left = ph._var(p, ph._nat(p))
    var right = ph._var(p, ph._nat(p))
    ph._declare(p, ph._pis(p, [left, right], ph._nat(p)),
        ph._lams(p, [left, right], bits.record_pack(p,
            _score(p, left, right), _flag(p, left, right))), RECORD_JOIN)


def _projections[C: Int](mut p: ph._Builder[C], exports: bo.RecordBoolExports):
    for projection in range(2):
        var left = ph._var(p, ph._nat(p))
        var right = ph._var(p, ph._nat(p))
        var score = _score(p, left, right)
        var flag = _flag(p, left, right)
        var flag_bit = _call(p, exports.closed,
            [bits.record_parity(p, left), bits.record_parity(p, right), _bit(p, left), _bit(p, right)])
        var lemma = bits.RECORD_PACK_HALF if projection == 0 else bits.RECORD_PACK_PARITY
        var body = _call(p, lemma, [score, flag, flag_bit])
        var actual = (bits.record_half(p, record_join(p, left, right)) if projection == 0
            else bits.record_parity(p, record_join(p, left, right)))
        var expected = score if projection == 0 else flag
        ph._declare(p, ph._pis(p, [left, right], ph._eq(p, actual, expected)),
            ph._lams(p, [left, right], body), RECORD_JOIN_HALF + projection)


def _simple[C: Int](mut p: ph._Builder[C], exports: bo.RecordBoolExports, kind: Int):
    # kind 0 left identity, 1 right identity, 2 commutativity, 3 idempotence.
    var n = ph._var(p, ph._nat(p))
    var m = ph._var(p, ph._nat(p)) if kind == 2 else n
    var binders: List[Int] = [n]
    if kind == 2:
        binders.append(m)
    var left = n
    var right = m
    if kind == 0:
        left = ph._zero(p)
        right = n
    elif kind == 1:
        right = ph._zero(p)
    var left_score = _score(p, left, right)
    var left_flag = _flag(p, left, right)
    var hn = bits.record_half(p, n)
    var bn = bits.record_parity(p, n)
    var right_score = hn
    var right_flag = bn
    var score_eq: Int
    var flag_eq: Int
    var definition = RECORD_JOIN_LEFT_UNIT
    if kind == 0:
        score_eq = _call(p, mx.RECORD_MAX_LEFT_ZERO, [hn])
        flag_eq = _call(p, exports.left_unit, [bn, _bit(p, n)])
    elif kind == 1:
        score_eq = _call(p, mx.RECORD_MAX_RIGHT_ZERO, [hn])
        flag_eq = _call(p, exports.right_unit, [bn, _bit(p, n)])
        definition = RECORD_JOIN_RIGHT_UNIT
    elif kind == 2:
        var hm = bits.record_half(p, m)
        var bm = bits.record_parity(p, m)
        right_score = _score(p, m, n)
        right_flag = _flag(p, m, n)
        score_eq = _call(p, mx.RECORD_MAX_COMMUTATIVE, [hn, hm])
        flag_eq = _call(p, exports.commutative, [bn, bm, _bit(p, n), _bit(p, m)])
        definition = RECORD_JOIN_COMMUTATIVE
    else:
        score_eq = _call(p, mx.RECORD_MAX_IDEMPOTENT, [hn])
        flag_eq = _call(p, exports.idempotent, [bn, _bit(p, n)])
        definition = RECORD_JOIN_IDEMPOTENT
    var body = _pack_congruence(p, left_score, right_score, left_flag, right_flag, score_eq, flag_eq)
    var actual = record_join(p, left, right)
    var expected = n
    if kind == 2:
        expected = record_join(p, m, n)
    else:
        body = ph._trans(p, actual, bits.record_pack(p, hn, bn), n,
            body, _call(p, bits.RECORD_REPACK, [n]))
    ph._declare(p, ph._pis(p, binders, ph._eq(p, actual, expected)),
        ph._lams(p, binders, body), definition)


def _associative[C: Int](mut p: ph._Builder[C], exports: bo.RecordBoolExports):
    var n = ph._var(p, ph._nat(p))
    var m = ph._var(p, ph._nat(p))
    var k = ph._var(p, ph._nat(p))
    var hn = bits.record_half(p, n)
    var hm = bits.record_half(p, m)
    var hk = bits.record_half(p, k)
    var bn = bits.record_parity(p, n)
    var bm = bits.record_parity(p, m)
    var bk = bits.record_parity(p, k)
    var nm = record_join(p, n, m)
    var mk = record_join(p, m, k)
    var score_left = mx.record_max(p, bits.record_half(p, nm), hk)
    var score_first = mx.record_max(p, mx.record_max(p, hn, hm), hk)
    var score_second = mx.record_max(p, hn, mx.record_max(p, hm, hk))
    var score_right = mx.record_max(p, hn, bits.record_half(p, mk))
    var slot = ph._var(p, ph._nat(p))
    var first = ph._cong(p, ph._lam(p, slot, mx.record_max(p, slot, hk)),
        bits.record_half(p, nm), mx.record_max(p, hn, hm),
        _call(p, RECORD_JOIN_HALF, [n, m]))
    var middle = _call(p, mx.RECORD_MAX_ASSOCIATIVE, [hn, hm, hk])
    slot = ph._var(p, ph._nat(p))
    var last = ph._cong(p, ph._lam(p, slot, mx.record_max(p, hn, slot)),
        bits.record_half(p, mk), mx.record_max(p, hm, hk),
        _call(p, RECORD_JOIN_HALF, [m, k]))
    var score_eq = ph._trans(p, score_left, score_first, score_second, first, middle)
    score_eq = ph._trans(p, score_left, score_second, score_right,
        score_eq, ph._sym(p, score_right, score_second, last))
    var flag_left = bits.record_bool_or(p, bits.record_parity(p, nm), bk)
    var flag_first = bits.record_bool_or(p, bits.record_bool_or(p, bn, bm), bk)
    var flag_second = bits.record_bool_or(p, bn, bits.record_bool_or(p, bm, bk))
    var flag_right = bits.record_bool_or(p, bn, bits.record_parity(p, mk))
    slot = ph._var(p, ph._nat(p))
    first = ph._cong(p, ph._lam(p, slot, bits.record_bool_or(p, slot, bk)),
        bits.record_parity(p, nm), bits.record_bool_or(p, bn, bm),
        _call(p, RECORD_JOIN_PARITY, [n, m]))
    middle = _call(p, exports.associative, [bn, bm, bk, _bit(p, n), _bit(p, m), _bit(p, k)])
    slot = ph._var(p, ph._nat(p))
    last = ph._cong(p, ph._lam(p, slot, bits.record_bool_or(p, bn, slot)),
        bits.record_parity(p, mk), bits.record_bool_or(p, bm, bk),
        _call(p, RECORD_JOIN_PARITY, [m, k]))
    var flag_eq = ph._trans(p, flag_left, flag_first, flag_second, first, middle)
    flag_eq = ph._trans(p, flag_left, flag_second, flag_right,
        flag_eq, ph._sym(p, flag_right, flag_second, last))
    var body = _pack_congruence(p, score_left, score_right, flag_left, flag_right, score_eq, flag_eq)
    ph._declare(p, ph._pis(p, [n, m, k], ph._eq(p,
        record_join(p, nm, k), record_join(p, n, mk))),
        ph._lams(p, [n, m, k], body), RECORD_JOIN_ASSOCIATIVE)


def record_canonical_type[C: Int](mut p: ph._Builder[C]) -> Int:
    var inputs = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var support = ph._var(p, leaf.execution_support_type(p))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
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
        ex.execution_readiness(p, initial_ready, destinations, steps), root), ph._one(p)))
    var complete = ph._var(p, a._prefix(p, ph._app(p, support, root), count))
    var join = ph._named(p, RECORD_JOIN)
    var actual = ph._app(p, ex.execution_values(p, join,
        leaf.execution_initial_values(p, ph._zero(p), inputs, count),
        destinations, lefts, rights, steps), root)
    var expected = ex.execution_flat_fold(p, join, ph._zero(p), inputs, count)
    return ph._pis(p, [inputs, count, support, destinations, lefts, rights,
        empty_support, input_support, steps, structural, root, ready, complete],
        ph._eq(p, actual, expected))


def append_record_join_terms[C: Int](mut p: ph._Builder[C], exports: bo.RecordBoolExports):
    if p.bundle.definition_count != RECORD_JOIN:
        p.bundle.count = -1
        return
    _data(p)
    _projections(p, exports)
    _simple(p, exports, 0)
    _simple(p, exports, 1)
    _simple(p, exports, 2)
    _associative(p, exports)
    _simple(p, exports, 3)
    var body = _call(p, trace.EXEC_CANONICAL_TRACE,
        [ph._named(p, RECORD_JOIN), ph._zero(p), ph._named(p, RECORD_JOIN_ASSOCIATIVE),
            ph._named(p, RECORD_JOIN_COMMUTATIVE), ph._named(p, RECORD_JOIN_LEFT_UNIT),
            ph._named(p, RECORD_JOIN_RIGHT_UNIT), ph._named(p, RECORD_JOIN_IDEMPOTENT)])
    ph._declare(p, record_canonical_type(p), body, RECORD_CANONICAL_TRACE)
    p.bundle.root = ph._named(p, RECORD_CANONICAL_TRACE)


def make_finishing_join_proof[
    capacity: Int = 8192,
]() -> tk.ProofTermBundle[capacity]:
    """Build the concrete total-join trace proof in one append-only environment."""
    var p = trace.make_finishing_execution_trace_builder[capacity]()
    mx.append_record_max_terms(p)
    bits.append_record_parity_terms(p)
    var exports = bo.append_record_bool_terms(p)
    append_record_join_terms(p, exports)
    return p.bundle.copy()
