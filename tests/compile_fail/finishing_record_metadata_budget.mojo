"""Only the intended resource preflight earns this fixture's diagnostic."""
import llm_mojo.kernels.nat_record_execution_check as checked
from record_execution_cases import record_execution_resource_mutant_result


def _escapes_target_rejection() -> Bool:
    var result = record_execution_resource_mutant_result(0)
    return (result.accepted or result.failure_kind != checked.RECORD_CHECK_RESOURCE_PREFLIGHT
        or result.used_nodes != -1 or result.fuel != -1)


def main():
    comptime assert _escapes_target_rejection(), "checked correlated record execution rejected"
