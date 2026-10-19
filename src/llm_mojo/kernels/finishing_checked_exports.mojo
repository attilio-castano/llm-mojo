"""Checked canonical proof libraries, reused across specializations.

These are compiler-associated constants, not caller-supplied receipts or a new
proof rule. Each value is computed by the existing independent checker. The
complete program identifies its canonical traversal library. Arbitrary
submitted proofs continue through the original independent full checkers;
they never borrow a canonical library's acceptance.
"""
from llm_mojo.kernels.finishing_proof_kernel import FinishingProgram
from llm_mojo.kernels.finishing_typed_coverage import (
    check_finishing_typed_coverage, make_finishing_typed_coverage,
)
from llm_mojo.kernels.finishing_typed_derivation import (
    check_finishing_typed_proofs, make_finishing_typed_proofs,
)
from llm_mojo.kernels.finishing_execution_trace import (
    make_finishing_execution_proof,
)
from llm_mojo.kernels.finishing_execution_check import (
    check_finishing_execution_proof,
)
from llm_mojo.kernels.finishing_execution import (
    FinishingExecutionDescriptor, check_finishing_execution_descriptor,
)


# This proof is already universal over counts, schedules, registers and join
# evidence. Its actual syntax is independently checked once, not per caller.
comptime _CANONICAL_EXECUTION = make_finishing_execution_proof()
comptime _CANONICAL_EXECUTION_ADMITTED = check_finishing_execution_proof(_CANONICAL_EXECUTION)


struct FinishingTraversalLibrary[program: FinishingProgram](Movable):
    """Canonical producers grant no authority until both checkers accept.

    Only the complete small program is a type parameter. Proof bundles remain
    associated data, so their payloads cannot bloat GPU closure symbol names.
    """
    comptime coverage = make_finishing_typed_coverage(Self.program)
    comptime termination = make_finishing_typed_proofs(Self.program)
    comptime coverage_accepted = check_finishing_typed_coverage(Self.program, Self.coverage)
    comptime termination_accepted = check_finishing_typed_proofs(Self.program, Self.termination)
    comptime accepted = Self.coverage_accepted and Self.termination_accepted


def canonical_finishing_execution_admitted() -> Bool:
    return _CANONICAL_EXECUTION_ADMITTED


def check_canonical_finishing_execution[
    T: Int, I: Int, L: Int, W: Int, S: Int, G: Int,
](descriptor: FinishingExecutionDescriptor[T, I, L, W, S, G]) -> Bool:
    """Bind admitted general semantics to checked executable descriptor data.

    This API takes no submitted proof or acceptance flag. Arbitrary submitted
    certificates still use their original independent full-input checker.
    """
    return (check_finishing_execution_descriptor(descriptor)
        and _CANONICAL_EXECUTION_ADMITTED)
