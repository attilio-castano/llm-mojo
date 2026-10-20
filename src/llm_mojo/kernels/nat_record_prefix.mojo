"""Ordinary prefix proofs for the checked Before definition.

Both statements are universal Nat-induction terms. No native comparison,
assumed order law, arithmetic reflection or new primitive participates.
The independent profile checks each proposed closed theorem type.
"""
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_algebra as algebra

comptime RECORD_PREFIX_DEFINITIONS = 2


@fieldwise_init
struct RecordPrefixExports(ImplicitlyCopyable):
    var weakening: Int
    var diagonal: Int


def _declare[C: Int](mut p: ph._Builder[C], type: Int, body: Int) -> Int:
    var index = p.bundle.definition_count
    ph._declare(p, type, body, index)
    return index


def _weakening_at[C: Int](mut p: ph._Builder[C], source: Int, count: Int) -> Int:
    var before = ph._var(p, algebra._before(p, source, count))
    return ph._pi(p, before, algebra._before(p, source, ph._succ(p, count)))


def _weakening[C: Int](mut p: ph._Builder[C]) -> Int:
    var source = ph._var(p, ph._nat(p))
    var count = ph._var(p, ph._nat(p))
    var outer_motive = ph._lam(p, source,
        ph._pi(p, count, _weakening_at(p, source, count)))

    # Before(0,Succ n) reduces to Eq(0,0), irrespective of the old premise.
    count = ph._var(p, ph._nat(p))
    var before = ph._var(p, algebra._before(p, ph._zero(p), count))
    var outer_base = ph._lams(p, [count, before], ph._refl(p, ph._zero(p)))

    var source_predecessor = ph._var(p, ph._nat(p))
    var outer_ih = ph._var(p, ph._app(p, outer_motive, source_predecessor))
    var source_successor = ph._succ(p, source_predecessor)
    count = ph._var(p, ph._nat(p))
    var inner_motive = ph._lam(p, count, _weakening_at(p, source_successor, count))
    var impossible = ph._var(p, algebra._before(p, source_successor, ph._zero(p)))
    var zero_goal = algebra._before(p, source_successor, ph._succ(p, ph._zero(p)))
    var inner_base = ph._lam(p, impossible, ph._app(p, impossible, zero_goal))
    var count_predecessor = ph._var(p, ph._nat(p))
    var inner_ih = ph._var(p, ph._app(p, inner_motive, count_predecessor))
    before = ph._var(p, algebra._before(p, source_successor, ph._succ(p, count_predecessor)))
    # Both successor arguments cancel by the certified Before definition:
    # the premise becomes Before(i,n), and the goal Before(i,Succ n).
    var inner_step = ph._lams(p, [count_predecessor, inner_ih, before],
        ph._apps(p, outer_ih, [count_predecessor, before]))
    count = ph._var(p, ph._nat(p))
    var outer_step = ph._lams(p, [source_predecessor, outer_ih, count],
        ph._induct(p, inner_motive, inner_base, inner_step, count))
    source = ph._var(p, ph._nat(p))
    return _declare(p, ph._pi(p, source, ph._app(p, outer_motive, source)),
        ph._lam(p, source, ph._induct(p, outer_motive, outer_base, outer_step, source)))


def _diagonal[C: Int](mut p: ph._Builder[C]) -> Int:
    var count = ph._var(p, ph._nat(p))
    var motive = ph._lam(p, count, algebra._before(p, count, ph._succ(p, count)))
    var predecessor = ph._var(p, ph._nat(p))
    var ih = ph._var(p, ph._app(p, motive, predecessor))
    var step = ph._lams(p, [predecessor, ih], ih)
    count = ph._var(p, ph._nat(p))
    return _declare(p, ph._pi(p, count, ph._app(p, motive, count)),
        ph._lam(p, count, ph._induct(p, motive, ph._refl(p, ph._zero(p)), step, count)))


def append_nat_record_prefix[C: Int](mut p: ph._Builder[C]) -> RecordPrefixExports:
    if p.bundle.count <= 0 or p.bundle.definition_count <= algebra.EXEC_BEFORE:
        p.bundle.count = -1
        return RecordPrefixExports(-1, -1)
    var first = p.bundle.definition_count
    var original_root = p.bundle.root
    var weakening = _weakening(p)
    var diagonal = _diagonal(p)
    if (p.bundle.count < 0 or p.bundle.definition_count != first + RECORD_PREFIX_DEFINITIONS
        or p.bundle.root != original_root):
        p.bundle.count = -1
        return RecordPrefixExports(-1, -1)
    return RecordPrefixExports(weakening, diagonal)
