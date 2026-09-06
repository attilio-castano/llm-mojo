# Inspect intermediate device lowering; this is not a timing instrument.
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from llm_mojo.attention_prefill import _mma, _mma_tuned
from std.sys import argv, get_defined_int


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("expected query rows R and KV rows T")
    var r = Int(String(args[1]))
    var t = Int(String(args[2]))
    if not 1 <= r <= t <= 4096:
        raise Error("expected 1 <= R <= T <= 4096")
    var ctx = DeviceContext()
    print(ctx.name(), ctx.api())
    var ql = row_major(r, 14, 64)
    var kl = row_major(t, 2, 64)
    var qb = ctx.enqueue_create_buffer[DType.bfloat16](r * 896)
    var kb = ctx.enqueue_create_buffer[DType.bfloat16](t * 128)
    var vb = ctx.enqueue_create_buffer[DType.bfloat16](t * 128)
    var ob = ctx.enqueue_create_buffer[DType.bfloat16](r * 896)
    qb.enqueue_fill(0)
    kb.enqueue_fill(0)
    vb.enqueue_fill(0)
    comptime schedule = get_defined_int["INSPECT_SCHEDULE", default=0]()
    comptime assert 0 <= schedule <= 5
    comptime kernel = _mma[
        32, 32, 1, type_of(ql), type_of(kl)
    ] if schedule == 0 else _mma_tuned[schedule, type_of(ql), type_of(kl)]
    ctx.enqueue_function[kernel, dump_llvm=True, dump_asm=True](
        TileTensor(qb, ql),
        TileTensor(kb, kl),
        TileTensor(vb, kl),
        TileTensor(ob, ql),
        Int32(r),
        Int32(t),
        grid_dim=(14, (r + 31) // 32),
        block_dim=128,
    )
    ctx.synchronize()
