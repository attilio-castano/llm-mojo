"""Untrusted relocation of ordinary arithmetic definitions into the record proof.

Sixteen definitions are copied through their recomputed declaration/type/body
dependency closure. No theorem signature is trusted through an export index.
The independent profile pins data and theorem types and checks the returned
Pred conversion judgment. This installs no arithmetic export namespace and
enables no native arithmetic, affine or order reflection.
"""
import llm_mojo.kernels.proof_term_syntax as tk
import llm_mojo.kernels.finishing_phase_terms as ph
import llm_mojo.kernels.finishing_join_terms as join
import llm_mojo.kernels.nat_arithmetic as arithmetic
from std.collections import Array

comptime NAT_SCORE_SOURCE_CAPACITY = arithmetic.NAT_ARITHMETIC_CAPACITY
comptime NAT_SCORE_SOURCE_DECLARATIONS = 44
comptime NAT_SCORE_SELECTED_DECLARATIONS = 17
comptime NAT_SCORE_APPENDED_DECLARATIONS = 16

comptime NAT_SCORE_ADD = 95
comptime NAT_SCORE_MUL = 96
comptime NAT_SCORE_LE = 97
comptime NAT_SCORE_ADD_ZERO_LEFT = 98
comptime NAT_SCORE_ADD_SUCC_LEFT = 99
comptime NAT_SCORE_ADD_ASSOC = 100
comptime NAT_SCORE_ADD_COMM = 101
comptime NAT_SCORE_ADD_CANCEL_SUFFIX = 102
comptime NAT_SCORE_MUL_SUCC_LEFT = 103
comptime NAT_SCORE_MUL_ZERO_LEFT = 104
comptime NAT_SCORE_NONNEG = 105
comptime NAT_SCORE_LE_TRANS = 106
comptime NAT_SCORE_LE_ADD = 107
comptime NAT_SCORE_LT_IRREFL = 108
comptime NAT_SCORE_ORDER_CASES = 109
comptime NAT_SCORE_LE_ANTISYM = 110
comptime NAT_SCORE_DEFINITIONS = 111

# Stable API consumed by the forthcoming complement/score proof producers.
comptime SCORE_ADD = NAT_SCORE_ADD
comptime SCORE_MUL = NAT_SCORE_MUL
comptime SCORE_LE = NAT_SCORE_LE
comptime SCORE_ADD_ZERO_LEFT = NAT_SCORE_ADD_ZERO_LEFT
comptime SCORE_ADD_SUCC_LEFT = NAT_SCORE_ADD_SUCC_LEFT
comptime SCORE_ADD_ASSOC = NAT_SCORE_ADD_ASSOC
comptime SCORE_ADD_COMM = NAT_SCORE_ADD_COMM
comptime SCORE_ADD_CANCEL = NAT_SCORE_ADD_CANCEL_SUFFIX
comptime SCORE_MUL_SUCC_LEFT = NAT_SCORE_MUL_SUCC_LEFT
comptime SCORE_MUL_ZERO_LEFT = NAT_SCORE_MUL_ZERO_LEFT
comptime SCORE_NONNEG = NAT_SCORE_NONNEG
comptime SCORE_LE_TRANS = NAT_SCORE_LE_TRANS
comptime SCORE_LE_ADD = NAT_SCORE_LE_ADD
comptime SCORE_LT_IRREFL = NAT_SCORE_LT_IRREFL
comptime SCORE_ORDER_CASES = NAT_SCORE_ORDER_CASES
comptime SCORE_LE_ANTISYM = NAT_SCORE_LE_ANTISYM

# Producer diagnostics only. They are not proof-kernel admission results.
comptime NAT_SCORE_LINK_OK = 0
comptime NAT_SCORE_LINK_BAD_DESTINATION = 1
comptime NAT_SCORE_LINK_BAD_SOURCE = 2
comptime NAT_SCORE_LINK_BAD_REACHABLE_NODE = 3
comptime NAT_SCORE_LINK_BAD_SOURCE_REFERENCE = 4
comptime NAT_SCORE_LINK_CLOSURE_MISMATCH = 5
comptime NAT_SCORE_LINK_BAD_PREDECESSOR = 6
comptime NAT_SCORE_LINK_CAPACITY = 7


@fieldwise_init
struct NatScoreLinkReport(ImplicitlyCopyable):
    var valid: Bool
    var failure: Int
    var source_nodes: Int
    var reachable_nodes: Int
    var copied_source_nodes: Int
    var appended_nodes: Int
    var source_variables: Int
    var expanded_successors: Int
    var expanded_literals: Int
    var source_definitions: Int
    var appended_definitions: Int
    var observed_definition_mask: Int
    # The relocated raw source Pred declaration is retained, rather than
    # replacing its semantics with a supposedly equivalent export signature.
    var predecessor_type: Int
    var predecessor_body: Int
    var predecessor_conversion_root: Int
    var predecessor_conversion_type: Int


def _report() -> NatScoreLinkReport:
    return NatScoreLinkReport(False, NAT_SCORE_LINK_OK, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, -1, -1, -1, -1)


def _reject[C: Int](mut p: ph._Builder[C], var report: NatScoreLinkReport,
    failure: Int) -> NatScoreLinkReport:
    p.bundle.count = -1
    report.valid = False
    report.failure = failure
    return report


def nat_score_add[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return ph._apps(p, ph._named(p, NAT_SCORE_ADD), [left, right])


def nat_score_mul[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return ph._apps(p, ph._named(p, NAT_SCORE_MUL), [left, right])


def nat_score_le[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return ph._apps(p, ph._named(p, NAT_SCORE_LE), [left, right])


def nat_score_lt[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return nat_score_le(p, ph._succ(p, left), right)


def score_add[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return nat_score_add(p, left, right)


def score_mul[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return nat_score_mul(p, left, right)


def score_le[C: Int](mut p: ph._Builder[C], left: Int, right: Int) -> Int:
    return nat_score_le(p, left, right)


def _selected_source(definition: Int) -> Bool:
    return (definition == 0 or definition == 1 or definition == 2
        or definition == 4 or definition == 8 or definition == 9
        or definition == 10 or definition == 11 or definition == 12
        or definition == 15 or definition == 16 or definition == 17
        or definition == 20 or definition == 21 or definition == 26
        or definition == 27 or definition == 30)


def _requested_source(definition: Int) -> Bool:
    # These nine requested laws discover Add/Mul/Le/Pred and the eight helper
    # declarations transitively. Seeding all seventeen expected declarations
    # would conceal an accidentally missing dependency from this observation.
    return (definition == 11 or definition == 12 or definition == 15
        or definition == 16 or definition == 20 or definition == 21
        or definition == 26 or definition == 27 or definition == 30)


def _target_definition(source: Int) -> Int:
    if source == 0:
        return NAT_SCORE_ADD
    if source == 1:
        return NAT_SCORE_MUL
    if source == 2:
        return NAT_SCORE_LE
    if source == 4:
        return ph.PHASE_PREDECESSOR
    if source == 8:
        return NAT_SCORE_ADD_ZERO_LEFT
    if source == 9:
        return NAT_SCORE_ADD_SUCC_LEFT
    if source == 10:
        return NAT_SCORE_ADD_ASSOC
    if source == 11:
        return NAT_SCORE_ADD_COMM
    if source == 12:
        return NAT_SCORE_ADD_CANCEL_SUFFIX
    if source == 15:
        return NAT_SCORE_MUL_SUCC_LEFT
    if source == 16:
        return NAT_SCORE_MUL_ZERO_LEFT
    if source == 17:
        return NAT_SCORE_NONNEG
    if source == 20:
        return NAT_SCORE_LE_TRANS
    if source == 21:
        return NAT_SCORE_LE_ADD
    if source == 26:
        return NAT_SCORE_LT_IRREFL
    if source == 27:
        return NAT_SCORE_ORDER_CASES
    if source == 30:
        return NAT_SCORE_LE_ANTISYM
    return -1


def _arity(kind: Int) -> Int:
    # Strict allowlist. Every arithmetic opcode, reflection rule, private
    # family, retired theorem constructor and unknown opcode is rejected.
    if (kind == tk.PT_TYPE or kind == tk.PT_PROP or kind == tk.PT_NAT
        or kind == tk.PT_LITERAL or kind == tk.PT_DEFINED_REF
        or kind == tk.PT_CONSTRUCTOR or kind == tk.PT_ELIMINATOR):
        return 0
    if kind == tk.PT_VAR or kind == tk.PT_REFL or kind == tk.PT_SUCC:
        return 1
    if (kind == tk.PT_PI or kind == tk.PT_LAM or kind == tk.PT_APP
        or kind == tk.PT_EQ):
        return 2
    if kind == tk.PT_SUBST or kind == tk.PT_NAT_REC:
        return 3
    return -1


def _ordinary_node[S: Int](source: tk.ProofTermBundle[S], index: Int) -> Bool:
    if index < 0 or index >= source.count:
        return False
    var node = source.nodes[index]
    var arity = _arity(node.kind)
    if arity < 0:
        return False
    if arity >= 1 and (node.left < 0 or node.left >= index):
        return False
    if arity >= 2 and (node.right < 0 or node.right >= index):
        return False
    if arity == 3 and (node.extra < 0 or node.extra >= index):
        return False
    if arity == 0 and node.left != -1:
        return False
    if arity < 2 and node.right != -1:
        return False
    if arity < 3 and node.extra != -1:
        if node.kind != tk.PT_CONSTRUCTOR or node.extra != tk.NAT_FAMILY_ID:
            return False
    if node.kind == tk.PT_LITERAL:
        # The selected proof graph uses only zero/one. Expand both to ordinary
        # constructors. Refuse arbitrary native-sized unary normalization.
        if node.value != 0 and node.value != 1:
            return False
    elif node.kind == tk.PT_DEFINED_REF:
        if node.value < 0 or node.value >= source.definition_count:
            return False
    elif node.kind == tk.PT_CONSTRUCTOR:
        if node.extra != tk.NAT_FAMILY_ID or (node.value != 0 and node.value != 1):
            return False
    elif node.kind == tk.PT_ELIMINATOR:
        if node.value != tk.NAT_FAMILY_ID:
            return False
    elif node.value != -1:
        return False
    if node.kind == tk.PT_PI or node.kind == tk.PT_LAM:
        if source.nodes[node.left].kind != tk.PT_VAR:
            return False
    return True


def _is_nat[S: Int](source: tk.ProofTermBundle[S], index: Int) -> Bool:
    return _ordinary_node(source, index) and source.nodes[index].kind == tk.PT_NAT


def _is_nat_variable[S: Int](source: tk.ProofTermBundle[S], index: Int) -> Bool:
    return (_ordinary_node(source, index) and source.nodes[index].kind == tk.PT_VAR
        and _is_nat(source, source.nodes[index].left))


def _is_zero[S: Int](source: tk.ProofTermBundle[S], index: Int) -> Bool:
    if not _ordinary_node(source, index):
        return False
    var node = source.nodes[index]
    return ((node.kind == tk.PT_LITERAL and node.value == 0)
        or (node.kind == tk.PT_CONSTRUCTOR and node.extra == tk.NAT_FAMILY_ID
            and node.value == 0))


def _raw_predecessor[S: Int](source: tk.ProofTermBundle[S], type: Int, body: Int) -> Bool:
    # This is a producer preflight of raw data, not a trusted admission rule.
    # The returned ordinary conversion judgment supplies the checkable reuse.
    if not _ordinary_node(source, type) or not _ordinary_node(source, body):
        return False
    var declared = source.nodes[type]
    var function = source.nodes[body]
    if (declared.kind != tk.PT_PI or function.kind != tk.PT_LAM
        or declared.left != function.left or not _is_nat_variable(source, declared.left)
        or not _is_nat(source, declared.right)):
        return False
    if not _ordinary_node(source, function.right):
        return False
    var recursion = source.nodes[function.right]
    if (recursion.kind != tk.PT_NAT_REC or recursion.left != function.left
        or not _is_zero(source, recursion.right) or not _ordinary_node(source, recursion.extra)):
        return False
    var step = source.nodes[recursion.extra]
    if step.kind != tk.PT_LAM or not _is_nat_variable(source, step.left):
        return False
    if not _ordinary_node(source, step.right):
        return False
    var ignored = source.nodes[step.right]
    return (ignored.kind == tk.PT_LAM and _is_nat_variable(source, ignored.left)
        and ignored.right == step.left and ignored.left != step.left
        and step.left != function.left and ignored.left != function.left)


def _source_environment[S: Int](source: tk.ProofTermBundle[S]) -> Bool:
    if (source.count <= 0 or source.count > S or source.definition_count > S
        or source.definition_count != NAT_SCORE_SOURCE_DECLARATIONS
        or source.family_arity != -1 or source.constructor_count != 0
        or source.nat_constructor_count != 2):
        return False
    if not _is_nat(source, source.nat_constructor_types[0]):
        return False
    var successor_type = source.nat_constructor_types[1]
    if not _ordinary_node(source, successor_type):
        return False
    var node = source.nodes[successor_type]
    return (node.kind == tk.PT_PI and _is_nat_variable(source, node.left)
        and _is_nat(source, node.right))


def _observe_closure[S: Int](source: tk.ProofTermBundle[S],
    mut selected: Array[Bool, NAT_SCORE_SOURCE_DECLARATIONS],
    mut reachable: Array[Bool, S]) -> Int:
    var processed = Array[Bool, NAT_SCORE_SOURCE_DECLARATIONS](fill=False)
    var local = Array[Bool, S](fill=False)
    for definition in range(NAT_SCORE_SOURCE_DECLARATIONS):
        selected[definition] = _requested_source(definition)
    # The finite worklist contains declarations, not recursion on syntax.
    # Children precede parents, so each declaration's two syntax roots can be
    # traversed in one descending scan. A source reference discovers another
    # declaration and its raw roots; no export signature participates.
    var progress = True
    while progress:
        progress = False
        for definition in range(NAT_SCORE_SOURCE_DECLARATIONS):
            if not selected[definition] or processed[definition]:
                continue
            var type = source.definition_types[definition]
            var body = source.definition_bodies[definition]
            if type < 0 or type >= source.count or body < 0 or body >= source.count:
                return NAT_SCORE_LINK_BAD_SOURCE
            for index in range(source.count):
                local[index] = False
            local[type] = True
            local[body] = True
            for offset in range(source.count):
                var index = source.count - offset - 1
                if not local[index]:
                    continue
                if not _ordinary_node(source, index):
                    return NAT_SCORE_LINK_BAD_REACHABLE_NODE
                reachable[index] = True
                var node = source.nodes[index]
                var arity = _arity(node.kind)
                if arity >= 1:
                    local[node.left] = True
                if arity >= 2:
                    local[node.right] = True
                if arity == 3:
                    local[node.extra] = True
                if node.kind == tk.PT_DEFINED_REF:
                    # Closed ordinary definitions may name only earlier
                    # declarations. This also prevents a cyclic worklist.
                    if node.value >= definition:
                        return NAT_SCORE_LINK_BAD_SOURCE_REFERENCE
                    selected[node.value] = True
            processed[definition] = True
            progress = True
    for definition in range(NAT_SCORE_SOURCE_DECLARATIONS):
        if selected[definition] != _selected_source(definition):
            return NAT_SCORE_LINK_CLOSURE_MISMATCH
    return NAT_SCORE_LINK_OK


def append_nat_score_arithmetic_from_source[C: Int, S: Int](mut p: ph._Builder[C],
    source: tk.ProofTermBundle[S]) -> NatScoreLinkReport:
    """Copy a proposed raw library after computing its actual dependency closure.

    This entry point permits adversarial producer diagnostics. A successful
    report means relocation completed; it is never a proof admission result.
    """
    var report = _report()
    if (p.bundle.count <= 0 or p.bundle.count > C or p.bundle.definition_count > C
        or p.bundle.definition_count != join.RECORD_JOIN_DEFINITIONS):
        return _reject(p, report, NAT_SCORE_LINK_BAD_DESTINATION)
    if not _source_environment(source):
        return _reject(p, report, NAT_SCORE_LINK_BAD_SOURCE)
    report.source_nodes = source.count
    var selected = Array[Bool, NAT_SCORE_SOURCE_DECLARATIONS](fill=False)
    var reachable = Array[Bool, S](fill=False)
    var failure = _observe_closure(source, selected, reachable)
    if failure != NAT_SCORE_LINK_OK:
        return _reject(p, report, failure)
    for definition in range(NAT_SCORE_SOURCE_DECLARATIONS):
        if selected[definition]:
            report.source_definitions += 1
            report.observed_definition_mask |= 1 << definition
    if report.source_definitions != NAT_SCORE_SELECTED_DECLARATIONS:
        return _reject(p, report, NAT_SCORE_LINK_CLOSURE_MISMATCH)
    if (not _raw_predecessor(source, source.definition_types[4], source.definition_bodies[4])
        or not _raw_predecessor(p.bundle,
            p.bundle.definition_types[ph.PHASE_PREDECESSOR],
            p.bundle.definition_bodies[ph.PHASE_PREDECESSOR])):
        return _reject(p, report, NAT_SCORE_LINK_BAD_PREDECESSOR)
    var first_node = p.bundle.count
    var relocated = Array[Int, S](fill=-1)
    # Visit every reachable source node once in topological order. In
    # particular, each source PT_VAR gets exactly one fresh destination binder
    # identity, shared by its type/body occurrences and all declaration roots.
    for index in range(source.count):
        if not reachable[index]:
            continue
        report.reachable_nodes += 1
        var node = source.nodes[index]
        var arity = _arity(node.kind)
        var left = relocated[node.left] if arity >= 1 else -1
        var right = relocated[node.right] if arity >= 2 else -1
        var extra = relocated[node.extra] if arity == 3 else node.extra
        if ((arity >= 1 and left < 0) or (arity >= 2 and right < 0)
            or (arity == 3 and extra < 0)):
            return _reject(p, report, NAT_SCORE_LINK_BAD_REACHABLE_NODE)
        var copied: Int
        if node.kind == tk.PT_SUCC:
            copied = ph._succ(p, left)
            report.expanded_successors += 1
        elif node.kind == tk.PT_LITERAL:
            copied = ph._zero(p)
            if node.value == 1:
                copied = ph._succ(p, copied)
            report.expanded_literals += 1
        elif node.kind == tk.PT_DEFINED_REF:
            var target = _target_definition(node.value)
            if target < 0:
                return _reject(p, report, NAT_SCORE_LINK_CLOSURE_MISMATCH)
            copied = ph._named(p, target)
        else:
            copied = ph._node(p, node.kind, left, right, extra, node.value)
        if node.kind == tk.PT_VAR:
            report.source_variables += 1
        if copied < 0 or p.bundle.count < 0:
            return _reject(p, report, NAT_SCORE_LINK_CAPACITY)
        relocated[index] = copied
        report.copied_source_nodes += 1
    report.predecessor_type = relocated[source.definition_types[4]]
    report.predecessor_body = relocated[source.definition_bodies[4]]
    for definition in range(NAT_SCORE_SOURCE_DECLARATIONS):
        if not selected[definition] or definition == 4:
            continue
        var type = relocated[source.definition_types[definition]]
        var body = relocated[source.definition_bodies[definition]]
        if type < 0 or body < 0:
            return _reject(p, report, NAT_SCORE_LINK_BAD_REACHABLE_NODE)
        ph._declare(p, type, body, _target_definition(definition))
        if p.bundle.count < 0:
            return _reject(p, report, NAT_SCORE_LINK_CAPACITY)
        report.appended_definitions += 1
    if (report.appended_definitions != NAT_SCORE_APPENDED_DECLARATIONS
        or p.bundle.definition_count != NAT_SCORE_DEFINITIONS):
        return _reject(p, report, NAT_SCORE_LINK_CLOSURE_MISMATCH)
    # Reuse of source Pred4 as destination Pred2 is itself presented as a
    # closed ordinary equality judgment, to be independently typechecked by
    # conversion. No extra definition or signature adapter is inserted.
    var n = ph._var(p, ph._nat(p))
    var existing = ph._app(p, ph._named(p, ph.PHASE_PREDECESSOR), n)
    var proposed = ph._app(p, report.predecessor_body, n)
    report.predecessor_conversion_type = ph._pi(p, n, ph._eq(p, existing, proposed))
    report.predecessor_conversion_root = ph._lam(p, n, ph._refl(p, existing))
    if p.bundle.count < 0 or report.predecessor_conversion_root < 0:
        return _reject(p, report, NAT_SCORE_LINK_CAPACITY)
    report.appended_nodes = p.bundle.count - first_node
    report.valid = True
    return report


def append_nat_score_arithmetic[C: Int](mut p: ph._Builder[C]) -> NatScoreLinkReport:
    # Keep the source library in this caller-owned frame. The copier borrows
    # its raw bundle and returns only a small report, avoiding a second large
    # builder/bundle return ladder around the existing library producer.
    var source = arithmetic.make_nat_arithmetic_library[NAT_SCORE_SOURCE_CAPACITY]()
    return append_nat_score_arithmetic_from_source(p, source.bundle)
