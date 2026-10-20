"""Prime canonical libraries, then independently reject changed submitted proofs."""
from std.testing import assert_equal
import llm_mojo.kernels.proof_term_kernel as tk
from llm_mojo.kernels.argmax_reduction import ArgmaxPlan, sequential_argmax_plan
from llm_mojo.kernels.token_selection import (
    ArgmaxFoldPlan, make_argmax_fold_plan, check_argmax_fold_plan,
    make_argmax_finishing_execution, _argmax_execution_descriptor,
)
from llm_mojo.kernels.finishing_execution_check import (
    FinishingExecutionCertificate, make_finishing_execution_certificate,
    check_finishing_execution_certificate_request, check_finishing_execution_proof,
)
from llm_mojo.kernels.finishing_proof import FinishingProgram, make_finishing_program
from llm_mojo.kernels.finishing_typed_derivation import check_finishing_typed_proofs
from llm_mojo.kernels.finishing_execution import check_finishing_execution_descriptor
from llm_mojo.kernels.finishing_checked_exports import (
    FinishingTraversalLibrary,
    check_canonical_finishing_execution, require_finishing_traversal,
    require_finishing_execution,
)


def changed_fold(mode: Int) -> ArgmaxFoldPlan[64, 3, 7]:
    var fold = make_argmax_fold_plan[64, 3, 7, sequential_argmax_plan[3]()]()
    if mode == 0:
        fold.typed_proof.base_root = fold.typed_proof.termination_root
    elif mode == 1:
        fold.proof.owner_visitation_root = fold.proof.thread_bound_root
    elif mode == 2:
        _ = tk.pt_add(fold.typed_proof.bundle, 99999)
    elif mode == 3:
        _ = tk.pt_add(fold.proof.bundle, 99999)
    elif mode == 4:
        fold.proof.headroom_root = -1
    elif mode == 5:
        fold.program.stride += 1
        fold.stride += 1
    elif mode == 6:
        fold.reduction.root = 0
    elif mode == 7:
        fold.program.item_scale += 1
    elif mode == 8:
        fold.program.count_limit -= 1
    elif mode == 9:
        fold.program.integer_limit -= 1
    return fold^


def changed_certificate[T: Int, I: Int, L: Int, plan: ArgmaxPlan[I, L]](
    mode: Int,
) -> FinishingExecutionCertificate[T, I, L, 32, 65, 2 * (T // 32) + 1, 8192]:
    var descriptor = _argmax_execution_descriptor[T, I, L](materialize[plan]())
    var certificate = make_finishing_execution_certificate(descriptor)
    if mode == 0:
        certificate.proof.canonical_root = certificate.proof.prefix_refinement_root
        certificate.proof.bundle.root = certificate.proof.canonical_root
    elif mode == 1:
        _ = tk.pt_add(certificate.proof.bundle, 99999)
    elif mode == 2:
        certificate.descriptor.program.count_limit -= 1
    elif mode == 3:
        certificate.proof.bundle.definition_count -= 1
    elif mode == 4:
        certificate.proof.leaf_root = certificate.proof.full_fold_root
    elif mode == 5:
        certificate.descriptor.program.index_bias = 1
    return certificate^


def raw_fold_controls() raises:
    # Keep large native proof bundles in separate frames from execution data.
    var fold = make_argmax_fold_plan[64, 3, 7, sequential_argmax_plan[3]()]()
    assert_equal(check_argmax_fold_plan(fold), True)
    for mode in range(10):
        var bad = changed_fold(mode)
        assert_equal(check_argmax_fold_plan(bad), False)
    var bad_program = make_finishing_program(64, 3, 192)
    bad_program.item_scale += 1
    assert_equal(check_finishing_typed_proofs(bad_program, fold.typed_proof), False)


def raw_execution_controls() raises:
    comptime plan = sequential_argmax_plan[3]()
    var descriptor = _argmax_execution_descriptor[64, 3, 7](materialize[plan]())
    var certificate = make_finishing_execution_certificate(descriptor)
    assert_equal(check_canonical_finishing_execution(descriptor), True)
    assert_equal(check_finishing_execution_certificate_request(certificate, descriptor), True)
    for mode in range(5):
        var bad = changed_certificate[64, 3, 7, plan](mode)
        assert_equal(check_finishing_execution_certificate_request(bad, descriptor), False)
    assert_equal(check_finishing_execution_proof[8192, 64](certificate.proof), False)
    var bad = changed_certificate[64, 3, 7, plan](5)
    assert_equal(check_canonical_finishing_execution(bad.descriptor), False)


def wrong_program() -> FinishingProgram:
    var program = make_finishing_program(64, 3, 192)
    program.item_scale += 1
    return program


def main() raises:
    comptime program = make_finishing_program(64, 3, 192)
    comptime plan = sequential_argmax_plan[3]()
    comptime descriptor = _argmax_execution_descriptor[64, 3, 7](materialize[plan]())
    require_finishing_traversal[program]()
    require_finishing_execution()
    comptime assert check_finishing_execution_descriptor(descriptor), "canonical execution request rejected"
    comptime bad_program = wrong_program()
    comptime assert not FinishingTraversalLibrary[bad_program].accepted, "invalid canonical traversal accepted"
    # Arbitrary inputs never use canonical acceptance. Check mutations through
    # the original checker after compile-time library admission, in native code.
    # Separate compiler fixtures enforce rejection diagnostics, not timeouts.
    raw_fold_controls()
    raw_execution_controls()
    print("Checked immutable finishing exports and 19 changed-input controls passed")
