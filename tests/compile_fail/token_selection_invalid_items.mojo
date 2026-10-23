from std.collections import Array
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from llm_mojo.kernels.token_selection import enqueue_argmax
from llm_mojo.kernels.argmax_reduction import ArgmaxPlan, ReductionNode, empty_node


def zero_input_plan() -> ArgmaxPlan[0, 1]:
    var nodes = Array[ReductionNode, 1](uninitialized=True)
    nodes[0] = empty_node()
    return ArgmaxPlan[0, 1](nodes^, 0)


def main() raises:
    var ctx = DeviceContext()
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](1)
    var partials = ctx.enqueue_create_buffer[DType.uint32](3)
    var result = ctx.enqueue_create_buffer[DType.uint32](3)
    comptime plan = zero_input_plan()
    enqueue_argmax[items_per_thread=0, plan_nodes=1, plan=plan](ctx, TileTensor(logits, row_major(1, 1)),
        TileTensor(partials, row_major(1, 3)), TileTensor(result, row_major(1, 3)))
