"""Host descriptor/certificate binding; this does not compile a GPU body."""
from std.testing import assert_equal
from llm_mojo.kernels.argmax_reduction import (
    ArgmaxPlan, sequential_argmax_plan, balanced_argmax_plan, check_argmax_plan,
)
from llm_mojo.kernels.token_selection import (
    make_argmax_fold_plan, make_argmax_finishing_execution, check_argmax_finishing_execution,
)
from llm_mojo.kernels.finishing_execution import (
    FinishingExecutionDescriptor, check_finishing_execution_descriptor, finishing_execution_index,
    finishing_execution_loop_active, finishing_execution_load_active,
)
from llm_mojo.kernels.finishing_execution_check import (
    make_finishing_execution_certificate, check_finishing_execution_certificate_request,
)


def check_fold_binding[T: Int, I: Int, L: Int, plan: ArgmaxPlan[I, L]]() raises -> FinishingExecutionDescriptor[T,I,L,32,65,2*(T//32)+1]:
    var fold = make_argmax_fold_plan[T, I, L, plan]()
    var descriptor = make_argmax_finishing_execution(fold)
    assert_equal(check_finishing_execution_descriptor(descriptor), True)
    assert_equal(check_argmax_finishing_execution(fold, descriptor), True)
    assert_equal(descriptor.program.stride, T * I)
    assert_equal(finishing_execution_index(descriptor, T * I, T - 1, I - 1), 2 * T * I - 1)
    assert_equal(finishing_execution_loop_active(descriptor, T * I, T * I + 1), True)
    assert_equal(finishing_execution_load_active(descriptor, 2 * T * I - 1, T * I + 1), I == 1 and T == 1)
    var changed = fold.copy()
    changed.program.count_limit -= 1
    assert_equal(check_argmax_finishing_execution(changed, descriptor), False)
    var changed_descriptor = make_argmax_finishing_execution(changed)
    assert_equal(check_finishing_execution_descriptor(changed_descriptor), True)
    changed = fold.copy()
    var root = changed.reduction.root
    var left = changed.reduction.nodes[root].left
    changed.reduction.nodes[root].left = changed.reduction.nodes[root].right
    changed.reduction.nodes[root].right = left
    assert_equal(check_argmax_plan(changed.reduction), True)
    assert_equal(check_argmax_finishing_execution(changed, descriptor), False)
    changed_descriptor = make_argmax_finishing_execution(changed)
    changed = fold.copy()
    changed.stride += 1
    changed_descriptor = make_argmax_finishing_execution(changed)
    assert_equal(check_finishing_execution_descriptor(changed_descriptor), False)
    changed = fold.copy()
    changed.program.item_scale += 1
    assert_equal(check_argmax_finishing_execution(changed, descriptor), False)
    changed_descriptor = make_argmax_finishing_execution(changed)
    assert_equal(changed_descriptor.program.item_scale, changed.program.item_scale)
    assert_equal(check_finishing_execution_descriptor(changed_descriptor), False)
    return descriptor^


def check_shape[T: Int, I: Int, L: Int, plan: ArgmaxPlan[I, L]]() raises:
    # The old fold carries large historical proof bundles. Finish its native
    # binding controls in another frame before creating/admitting the new
    # proof, keeping this diagnostic within the normal process stack.
    var descriptor = check_fold_binding[T, I, L, plan]()
    var certificate = make_finishing_execution_certificate(descriptor)
    assert_equal(check_finishing_execution_certificate_request(certificate, descriptor), True)
    var changed_descriptor = descriptor.copy()
    changed_descriptor.program.count_limit -= 1
    assert_equal(check_finishing_execution_descriptor(changed_descriptor), True)
    assert_equal(check_finishing_execution_certificate_request(certificate, changed_descriptor), False)
    changed_descriptor = descriptor.copy()
    var root = changed_descriptor.local.root
    var left = changed_descriptor.local.nodes[root].left
    changed_descriptor.local.nodes[root].left = changed_descriptor.local.nodes[root].right
    changed_descriptor.local.nodes[root].right = left
    assert_equal(check_argmax_plan(changed_descriptor.local), True)
    assert_equal(check_finishing_execution_certificate_request(certificate, changed_descriptor), False)


def main() raises:
    comptime single = sequential_argmax_plan[1]()
    comptime triple = balanced_argmax_plan[3]()
    check_shape[32, 1, 3, single]()
    check_shape[64, 3, 7, triple]()
    print("Production finishing descriptor and certificate host bindings passed")
