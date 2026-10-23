"""Untrusted producer for the complete 183-definition record execution proof.

Layout IDs are ordinary metadata. Only independent kernel admission of the data,
public theorem types and supplied root grants a judgment. A one-element heap
holder keeps fixed proof storage separate from computation shape: counts,
bounds, stores and trace length remain universally quantified.
"""
import llm_mojo.kernels.proof_term_kernel as tk
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_execution_trace as trace
import llm_mojo.kernels.nat_join_max as maximum
import llm_mojo.kernels.nat_join_bits as bits
import llm_mojo.kernels.nat_join_bool as boolean
import llm_mojo.kernels.finishing_join_terms as join
import llm_mojo.kernels.nat_score_link as arithmetic
import llm_mojo.kernels.nat_score_complement as complement
import llm_mojo.kernels.nat_score_order as order
import llm_mojo.kernels.nat_comparison_bridge as comparison
import llm_mojo.kernels.nat_score_embedding as embedding
import llm_mojo.kernels.nat_record_selector as selector
import llm_mojo.kernels.nat_record_encoding as encoding
import llm_mojo.kernels.nat_record_laws as laws
import llm_mojo.kernels.nat_record_trace_data as data
import llm_mojo.kernels.nat_record_prefix as prefix
import llm_mojo.kernels.nat_record_basics as basics
import llm_mojo.kernels.nat_record_fold as fold
import llm_mojo.kernels.nat_record_trace_refinement as refinement
import llm_mojo.kernels.nat_record_initial as initial
import llm_mojo.kernels.nat_record_canonical as canonical
from llm_mojo.kernels.nat_score_bridge_check import NatScoreBridgeLayout
from llm_mojo.kernels.nat_record_execution_check import RecordExecutionLayout

comptime FINISHING_RECORD_DEFINITIONS = 183


@fieldwise_init
struct FinishingRecordProof[C: Int](Copyable, Movable):
    var bundle: tk.ProofTermBundle[Self.C]
    var layout: RecordExecutionLayout


def _invalid_layout() -> RecordExecutionLayout:
    return RecordExecutionLayout(NatScoreBridgeLayout(-1, -1, -1, -1, -1, -1, -1),
        -1, -1, -1, -1, -1, -1, -1, -1)


@no_inline
def _hold_proof[C: Int](mut p: ph._Builder[C],
    layout: RecordExecutionLayout) -> List[FinishingRecordProof[C]]:
    # Keep the fixed-capacity bundle copy out of the generator's append frame.
    var holders = List[FinishingRecordProof[C]](capacity=1)
    holders.append(FinishingRecordProof[C](p.bundle.copy(), layout))
    return holders^


def _invalid_proof[C: Int](mut p: ph._Builder[C]) -> List[FinishingRecordProof[C]]:
    p.bundle.count = -1
    p.bundle.root = -1
    return _hold_proof(p, _invalid_layout())


def _block_ready[C: Int](p: ph._Builder[C], definitions: Int) -> Bool:
    return (p.bundle.count > 0 and p.bundle.count <= C
        and p.bundle.definition_count == definitions)


@no_inline
def _builder_holder[C: Int]() -> List[ph._Builder[C]]:
    var builders = List[ph._Builder[C]](capacity=1)
    builders.append(trace.make_finishing_execution_trace_builder[C]())
    return builders^


def _append_proof[C: Int](mut p: ph._Builder[C]) -> List[FinishingRecordProof[C]]:
    if not _block_ready(p, 55):
        return _invalid_proof(p)
    maximum.append_record_max_terms(p)
    if not _block_ready(p, 61):
        return _invalid_proof(p)
    bits.append_record_parity_terms(p)
    if not _block_ready(p, 80):
        return _invalid_proof(p)
    var booleans = boolean.append_record_bool_terms(p)
    if not _block_ready(p, 86):
        return _invalid_proof(p)
    join.append_record_join_terms(p, booleans)
    if not _block_ready(p, 95):
        return _invalid_proof(p)
    var original_root = p.bundle.root
    if original_root < 0 or original_root >= p.bundle.count:
        return _invalid_proof(p)
    if (p.bundle.nodes[original_root].kind != tk.PT_DEFINED_REF
        or p.bundle.nodes[original_root].value != 94):
        return _invalid_proof(p)
    var report = arithmetic.append_nat_score_arithmetic(p)
    if (not report.valid or not _block_ready(p, 111)
        or report.predecessor_conversion_root < 0
        or report.predecessor_conversion_root >= p.bundle.count):
        return _invalid_proof(p)
    var complements = complement.append_nat_score_complement(p)
    if not _block_ready(p, 117):
        return _invalid_proof(p)
    var orders = order.append_nat_score_order_terms(p)
    if not _block_ready(p, 125):
        return _invalid_proof(p)
    var comparisons = comparison.append_nat_comparison_terms(p)
    if not _block_ready(p, 133):
        return _invalid_proof(p)
    var embeddings = embedding.append_nat_score_embedding(p, complements, orders)
    if not _block_ready(p, 139):
        return _invalid_proof(p)
    var selectors = selector.append_nat_record_selector(p, comparisons, orders, embeddings)
    if not _block_ready(p, 142):
        return _invalid_proof(p)
    var encodings = encoding.append_nat_record_encoding(p, complements, embeddings, selectors)
    if not _block_ready(p, 148):
        return _invalid_proof(p)
    var operations = laws.append_nat_record_laws(p, encodings, selectors)
    if not _block_ready(p, 160):
        return _invalid_proof(p)
    var records = data.append_nat_record_trace_data(p, operations)
    if not _block_ready(p, 166):
        return _invalid_proof(p)
    var prefixes = prefix.append_nat_record_prefix(p)
    if not _block_ready(p, 168):
        return _invalid_proof(p)
    var valid = basics.append_nat_record_basics(p, encodings, operations, records)
    if not _block_ready(p, 173):
        return _invalid_proof(p)
    var inputs = fold.append_nat_record_fold(p, encodings, operations, records, prefixes, valid)
    if not _block_ready(p, 176):
        return _invalid_proof(p)
    var proofs = refinement.append_nat_record_trace_refinement(p, encodings, valid, records)
    if not _block_ready(p, 181):
        return _invalid_proof(p)
    var initialization = initial.append_nat_record_initial(p, encodings, operations,
        records, valid, inputs, proofs)
    if not _block_ready(p, 182) or p.bundle.root != original_root:
        return _invalid_proof(p)
    var result = canonical.append_nat_record_canonical(p, encodings, records,
        valid, inputs, proofs, initialization)
    if (not _block_ready(p, FINISHING_RECORD_DEFINITIONS)
        or p.bundle.root < 0 or p.bundle.root >= p.bundle.count):
        return _invalid_proof(p)
    if (p.bundle.nodes[p.bundle.root].kind != tk.PT_DEFINED_REF
        or p.bundle.nodes[p.bundle.root].value != result):
        return _invalid_proof(p)
    var bridge = NatScoreBridgeLayout(complements.comp, orders.prefix_le,
        comparisons.before_to_lt, embeddings.score, selectors.flag, encodings.encode,
        report.predecessor_conversion_root)
    var layout = RecordExecutionLayout(bridge, operations.rank, records.record,
        prefixes.weakening, valid.valid, inputs.valid_inputs, proofs.ready_valid,
        initialization, result)
    return _hold_proof(p, layout)


@no_inline
def make_finishing_record_proof_holder[C: Int = 16384]() -> List[FinishingRecordProof[C]]:
    """Generate one proposed proof, or one explicitly invalid bounded result.

    Generation does not admit its terms or establish a theorem. The independent
    checker must validate the complete bundle and its untrusted layout.
    """
    comptime if C < FINISHING_RECORD_DEFINITIONS:
        # There cannot be 183 declarations in fewer than 183 metadata entries.
        # Avoid the full generator for such capacities, including zero.
        var bundle = tk.empty_proof_terms[C]()
        bundle.count = -1
        bundle.root = -1
        var holders = List[FinishingRecordProof[C]](capacity=1)
        holders.append(FinishingRecordProof[C](bundle^, _invalid_layout()))
        return holders^
    else:
        var builders = _builder_holder[C]()
        return _append_proof(builders[0])
