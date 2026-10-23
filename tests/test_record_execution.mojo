"""Correlated record proof and independent adversarial controls.

One full183 compile-time positive covers the earlier148 specification. Native
bridge controls remain separately labelled Boolean checks at their previously
measured budget; full183 controls require exact diagnostic classifications.
"""
from std.testing import assert_equal
import llm_mojo.kernels.nat_record_execution_check as checked
from record_execution_cases import (
    RECORD_EXECUTION_WORKSPACE, RECORD_EXECUTION_SEMANTIC_CONTROLS,
    RECORD_EXECUTION_RESOURCE_CONTROLS, record_execution_accepts,
    record_execution_alternate_accepts, record_execution_final_control_is_welltyped,
    record_execution_scope_data_is_welltyped, record_execution_mutant_result,
    record_execution_resource_mutant_result,
)
from finishing_record_bridge_cases import (
    NATIVE_SCORE_SEMANTIC_CONTROLS, NATIVE_SCORE_RESOURCE_CONTROLS,
    native_score_accepts, native_score_alternate_accepts,
    native_score_assumed_bridge_is_welltyped, native_score_mutant_accepts,
    native_score_resource_mutant_accepts,
)

# Proof workspace capacity is separate from computation shape.
comptime WORKSPACE = RECORD_EXECUTION_WORKSPACE
comptime BRIDGE_WORKSPACE = 40960


@no_inline
def _bridge_positive() -> Bool:
    return native_score_accepts[BRIDGE_WORKSPACE]()


@no_inline
def _bridge_alternate() -> Bool:
    return native_score_alternate_accepts[BRIDGE_WORKSPACE]()


@no_inline
def _bridge_assumed_typing() -> Bool:
    return native_score_assumed_bridge_is_welltyped[BRIDGE_WORKSPACE]()


@no_inline
def _bridge_semantic(mode: Int) -> Bool:
    return native_score_mutant_accepts[BRIDGE_WORKSPACE](mode)


@no_inline
def _bridge_resource(mode: Int) -> Bool:
    return native_score_resource_mutant_accepts(mode)


def main() raises:
    # No separate148 CT positive: this exact profile includes all its exports.
    comptime assert record_execution_accepts[WORKSPACE](), "checked correlated record execution positive failed"
    print("Candidate183 native-control workspace", WORKSPACE)
    print("Earlier148 native-control workspace", BRIDGE_WORKSPACE)
    assert_equal(record_execution_accepts[WORKSPACE](), True)
    assert_equal(record_execution_alternate_accepts[WORKSPACE](), True)
    assert_equal(_bridge_positive(), True)
    assert_equal(_bridge_alternate(), True)
    assert_equal(_bridge_assumed_typing(), True)
    # These complete mutated183 environments must pass ordinary typing before
    # profile rejection distinguishes the intended public statement.
    for mode in [14, 15, 16]:
        if not record_execution_final_control_is_welltyped[WORKSPACE](mode):
            raise Error("Final-statement raw typing positive failed: " + String(mode))
    # These establish only the changed data's declaration prefix. They do not
    # assert that the candidate's later183 proofs retain the changed contract.
    for mode in [9, 11, 12, 13]:
        if not record_execution_scope_data_is_welltyped[WORKSPACE](mode):
            raise Error("Scope-data prefix typing positive failed: " + String(mode))
    for mode in range(NATIVE_SCORE_SEMANTIC_CONTROLS):
        if _bridge_semantic(mode):
            raise Error("Earlier148 Boolean semantic control accepted: " + String(mode))
    for mode in range(NATIVE_SCORE_RESOURCE_CONTROLS):
        if _bridge_resource(mode):
            raise Error("Earlier148 Boolean resource preflight accepted: " + String(mode))
    for mode in range(RECORD_EXECUTION_SEMANTIC_CONTROLS):
        var result = record_execution_mutant_result[WORKSPACE](mode)
        if result.accepted or result.failure_kind != checked.RECORD_CHECK_INVALID_PROOF:
            print("Semantic control status", mode, result.accepted, result.failure_kind,
                result.used_nodes, result.fuel)
            raise Error("Record execution semantic control did not return INVALID_PROOF: " + String(mode))
    for mode in range(RECORD_EXECUTION_RESOURCE_CONTROLS):
        var result = record_execution_resource_mutant_result(mode)
        if (result.accepted or result.failure_kind != checked.RECORD_CHECK_RESOURCE_PREFLIGHT
            or result.used_nodes != -1 or result.fuel != -1):
            print("Resource control status", mode, result.accepted, result.failure_kind,
                result.used_nodes, result.fuel)
            raise Error("Record execution control did not return RESOURCE_PREFLIGHT: " + String(mode))
    print("Checked earlier148 Boolean semantic controls:", NATIVE_SCORE_SEMANTIC_CONTROLS)
    print("Checked earlier148 Boolean resource preflights:", NATIVE_SCORE_RESOURCE_CONTROLS)
    print("Checked earlier148 assumed-bridge full typing positives:", 1)
    print("Checked record execution semantic rejection controls:", RECORD_EXECUTION_SEMANTIC_CONTROLS)
    print("Checked record execution resource preflight controls:", RECORD_EXECUTION_RESOURCE_CONTROLS)
    print("Checked final-statement full-environment typing positives:", 3)
    print("Checked wrong-scope data declaration-prefix typing positives:", 4)
    print("Checked correlated record execution and independent controls passed")
