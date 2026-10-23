"""Only the intended semantic rejection earns this fixture's diagnostic."""
import llm_mojo.kernels.nat_record_execution_check as checked
from record_execution_cases import (
    RECORD_EXECUTION_WORKSPACE, record_execution_mutant_result,
)


def _escapes_target_rejection() -> Bool:
    var result = record_execution_mutant_result[RECORD_EXECUTION_WORKSPACE](6)
    return result.accepted or result.failure_kind != checked.RECORD_CHECK_INVALID_PROOF


def main():
    # Exhaustion, a crash or a timeout cannot earn INVALID_PROOF credit.
    comptime assert _escapes_target_rejection(), "checked correlated record execution rejected"
