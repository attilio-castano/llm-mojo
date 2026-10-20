from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from llm_mojo.kernels.token_selection import enqueue_argmax


def main() raises:
    var ctx = DeviceContext()
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](1)
    var partials = ctx.enqueue_create_buffer[DType.uint32](3)
    var result = ctx.enqueue_create_buffer[DType.uint32](3)
    enqueue_argmax[threads=33](ctx, TileTensor(logits, row_major(1, 1)),
        TileTensor(partials, row_major(1, 3)), TileTensor(result, row_major(1, 3)))
