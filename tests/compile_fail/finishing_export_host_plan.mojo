"""The checked host launcher rejects an incomplete plan before offload."""
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from llm_mojo.kernels.argmax_reduction import ArgmaxPlan, sequential_argmax_plan
from llm_mojo.kernels.token_selection import _finish


def incomplete_plan() -> ArgmaxPlan[3, 7]:
    var plan = sequential_argmax_plan[3]()
    plan.root = 0
    return plan^


def main() raises:
    comptime layout = row_major[1, 3]()
    comptime plan = incomplete_plan()
    var ctx = DeviceContext()
    var partials = ctx.enqueue_create_buffer[DType.uint32](3)
    var output = ctx.enqueue_create_buffer[DType.uint32](3)
    _finish[type_of(layout), type_of(layout), 64, 3, 7, plan](
        ctx, TileTensor(partials, layout), TileTensor(output, layout), Int32(1))
