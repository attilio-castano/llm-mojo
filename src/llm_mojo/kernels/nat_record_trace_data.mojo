"""Correlated record/state execution definitions, expressed as ordinary terms.

A record is Nat->Nat: fields0/1 are rank/token, fields>=2 are bad. State is
field->slot->Nat. One NatRec updates the whole state, reading every operand
component from the same old state. The independent fold uses only inputs,
count and token bound, with no Run or D/L/R stream dependency.

Bounds, Boolean closure, readiness, coverage and trace invariants are proved in
other modules. Native source/backend correspondence remains a separate claim.
"""
import llm_mojo.kernels.proof_term_syntax as tk
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.nat_join_bits as bits
import llm_mojo.kernels.nat_record_laws as laws

comptime RECORD_TRACE_DATA_DEFINITIONS = 6
comptime RECORD_RANK_FIELD = 0
comptime RECORD_TOKEN_FIELD = 1
comptime RECORD_BAD_FIELD = 2


@fieldwise_init
struct RecordTraceDataExports(ImplicitlyCopyable):
    # record: (rank,token,bad) -> (field:Nat -> Nat).
    var record: Int
    # state: (ranks,tokens,bads : Nat->Nat) -> (field -> slot -> Nat).
    var state: Int
    # initial: (M,N,ranks,tokens,bads) -> combined state, sentinel N=Empty.
    var initial: Int
    # merge: (left,right : Nat->Nat) -> record, using component merge148/149.
    var merge: Int
    # run: (S0 : combined state,D,L,R : Nat->Nat,n:Nat) -> combined state.
    var run: Int
    # fold: (M,ranks,tokens,bads : input stores,N:Nat) -> record.
    var fold: Int


def _call[C: Int](mut p: ph._Builder[C], definition: Int, arguments: List[Int]) -> Int:
    return ph._apps(p, ph._named(p, definition), arguments)


def _declare[C: Int](mut p: ph._Builder[C], type: Int, body: Int) -> Int:
    var definition = p.bundle.definition_count
    ph._declare(p, type, body, definition)
    return definition


def record_type[C: Int](mut p: ph._Builder[C]) -> Int:
    return ph._store(p)


def record_state_type[C: Int](mut p: ph._Builder[C]) -> Int:
    var field = ph._var(p, ph._nat(p))
    return ph._pi(p, field, ph._store(p))


def record_at[C: Int](mut p: ph._Builder[C], record: Int, field: Int) -> Int:
    return ph._app(p, record, field)


def state_at[C: Int](mut p: ph._Builder[C], state: Int, field: Int, slot: Int) -> Int:
    return ph._apps(p, state, [field, slot])


def record_row[C: Int](mut p: ph._Builder[C], state: Int, slot: Int) -> Int:
    """A record-valued view of one slot in the supplied whole state."""
    var field = ph._var(p, ph._nat(p))
    return ph._lam(p, field, state_at(p, state, field, slot))


def record_fields[C: Int](mut p: ph._Builder[C], exports: RecordTraceDataExports,
    rank: Int, token: Int, bad: Int) -> Int:
    return _call(p, exports.record, [rank, token, bad])


def record_state[C: Int](mut p: ph._Builder[C], exports: RecordTraceDataExports,
    ranks: Int, tokens: Int, bads: Int) -> Int:
    return _call(p, exports.state, [ranks, tokens, bads])


def record_initial[C: Int](mut p: ph._Builder[C], exports: RecordTraceDataExports,
    bound: Int, count: Int, ranks: Int, tokens: Int, bads: Int) -> Int:
    return _call(p, exports.initial, [bound, count, ranks, tokens, bads])


def record_merge[C: Int](mut p: ph._Builder[C], exports: RecordTraceDataExports,
    left: Int, right: Int) -> Int:
    return _call(p, exports.merge, [left, right])


def record_run[C: Int](mut p: ph._Builder[C], exports: RecordTraceDataExports,
    initial: Int, destinations: Int, lefts: Int, rights: Int, steps: Int) -> Int:
    return _call(p, exports.run, [initial, destinations, lefts, rights, steps])


def record_fold[C: Int](mut p: ph._Builder[C], exports: RecordTraceDataExports,
    bound: Int, ranks: Int, tokens: Int, bads: Int, count: Int) -> Int:
    return _call(p, exports.fold, [bound, ranks, tokens, bads, count])


def _record[C: Int](mut p: ph._Builder[C]) -> Int:
    var rank = ph._var(p, ph._nat(p))
    var token = ph._var(p, ph._nat(p))
    var bad = ph._var(p, ph._nat(p))
    var field = ph._var(p, ph._nat(p))
    var predecessor = ph._app(p, ph._named(p, ph.PHASE_PREDECESSOR), field)
    var body = ph._select(p, field, rank, ph._select(p, predecessor, token, bad))
    return _declare(p, ph._pis(p, [rank, token, bad], record_type(p)),
        ph._lams(p, [rank, token, bad, field], body))


def _state[C: Int](mut p: ph._Builder[C], record: Int) -> Int:
    var ranks = ph._var(p, ph._store(p))
    var tokens = ph._var(p, ph._store(p))
    var bads = ph._var(p, ph._store(p))
    var field = ph._var(p, ph._nat(p))
    var slot = ph._var(p, ph._nat(p))
    var row = _call(p, record, [ph._app(p, ranks, slot),
        ph._app(p, tokens, slot), ph._app(p, bads, slot)])
    return _declare(p, ph._pis(p, [ranks, tokens, bads], record_state_type(p)),
        ph._lams(p, [ranks, tokens, bads, field, slot], record_at(p, row, field)))


def _initial[C: Int](mut p: ph._Builder[C], record: Int, state: Int) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var count = ph._var(p, ph._nat(p))
    var ranks = ph._var(p, ph._store(p))
    var tokens = ph._var(p, ph._store(p))
    var bads = ph._var(p, ph._store(p))
    var field = ph._var(p, ph._nat(p))
    var inputs = _call(p, state, [ranks, tokens, bads])
    var empty = _call(p, record, [ph._zero(p), bound, ph._zero(p)])
    var updated = ph._update(p, ph._app(p, inputs, field), count, record_at(p, empty, field))
    var binders: List[Int] = [bound, count, ranks, tokens, bads]
    return _declare(p, ph._pis(p, binders, record_state_type(p)),
        ph._lams(p, [bound, count, ranks, tokens, bads, field], updated))


def _merge[C: Int](mut p: ph._Builder[C], record: Int, components: laws.RecordLawsExports) -> Int:
    var left = ph._var(p, record_type(p))
    var right = ph._var(p, record_type(p))
    var zero = ph._zero(p)
    var one = ph._one(p)
    var two = ph._succ(p, one)
    var ra = record_at(p, left, zero)
    var rb = record_at(p, right, zero)
    var ta = record_at(p, left, one)
    var tb = record_at(p, right, one)
    var ba = record_at(p, left, two)
    var bb = record_at(p, right, two)
    # Both closed components compute the same selector on this exact old
    # (ra,rb,ta,tb) tuple. No component is read after any row update.
    var rank = laws.record_merge_rank(p, components, ra, rb, ta, tb)
    var token = laws.record_merge_token(p, components, ra, rb, ta, tb)
    var bad = bits.record_bool_or(p, ba, bb)
    var body = _call(p, record, [rank, token, bad])
    return _declare(p, ph._pis(p, [left, right], record_type(p)), ph._lams(p, [left, right], body))


def _run[C: Int](mut p: ph._Builder[C], merge: Int) -> Int:
    var initial = ph._var(p, record_state_type(p))
    var destinations = ph._var(p, ph._store(p))
    var lefts = ph._var(p, ph._store(p))
    var rights = ph._var(p, ph._store(p))
    var steps = ph._var(p, ph._nat(p))
    var k = ph._var(p, ph._nat(p))
    var old = ph._var(p, record_state_type(p))
    var field = ph._var(p, ph._nat(p))
    var destination = ph._app(p, destinations, k)
    var left = record_row(p, old, ph._app(p, lefts, k))
    var right = record_row(p, old, ph._app(p, rights, k))
    var result = _call(p, merge, [left, right])
    var updated_row = ph._update(p, ph._app(p, old, field), destination, record_at(p, result, field))
    # The recursor's carrier is the whole field->slot state. Its step closes
    # over ONE old state and returns all row updates as a single function.
    var step = ph._lams(p, [k, old, field], updated_row)
    var body = ph._node(p, tk.PT_NAT_REC, steps, initial, step)
    var binders: List[Int] = [initial, destinations, lefts, rights, steps]
    return _declare(p, ph._pis(p, binders, record_state_type(p)), ph._lams(p, binders, body))


def _fold[C: Int](mut p: ph._Builder[C], record: Int, merge: Int) -> Int:
    var bound = ph._var(p, ph._nat(p))
    var ranks = ph._var(p, ph._store(p))
    var tokens = ph._var(p, ph._store(p))
    var bads = ph._var(p, ph._store(p))
    var count = ph._var(p, ph._nat(p))
    var k = ph._var(p, ph._nat(p))
    var previous = ph._var(p, record_type(p))
    var input = _call(p, record, [ph._app(p, ranks, k),
        ph._app(p, tokens, k), ph._app(p, bads, k)])
    var step = ph._lams(p, [k, previous], _call(p, merge, [previous, input]))
    var empty = _call(p, record, [ph._zero(p), bound, ph._zero(p)])
    var body = ph._node(p, tk.PT_NAT_REC, count, empty, step)
    var binders: List[Int] = [bound, ranks, tokens, bads, count]
    return _declare(p, ph._pis(p, binders, record_type(p)), ph._lams(p, binders, body))


def append_nat_record_trace_data[C: Int](mut p: ph._Builder[C],
    components: laws.RecordLawsExports) -> RecordTraceDataExports:
    var first = p.bundle.definition_count
    var record = _record(p)
    var state = _state(p, record)
    var initial = _initial(p, record, state)
    var merge = _merge(p, record, components)
    var run = _run(p, merge)
    var fold = _fold(p, record, merge)
    if p.bundle.count < 0 or p.bundle.definition_count != first + RECORD_TRACE_DATA_DEFINITIONS:
        p.bundle.count = -1
        return RecordTraceDataExports(-1, -1, -1, -1, -1, -1)
    return RecordTraceDataExports(record, state, initial, merge, run, fold)
