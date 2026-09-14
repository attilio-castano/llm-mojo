"""Exact greedy selection of rounded BF16 logits on Metal, without atomics.

Callers own nonoverlapping input, logits, partial and result storage and retain
it through device completion. Each record is (ordered score, lowest token ID,
any-nonfinite flag). No interpretation of the ID is valid when the flag is set.
The fused nonmaterializing specialization leaves logits untouched; its view
still declares vocabulary extent. Tensor indexing respects the supplied layouts.
"""
from layout import TensorLayout, TileTensor, row_major, stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.gpu.primitives import warp
from std.memory import bitcast
from std.sys.info import is_apple_gpu


@always_inline
def bf16_rank(bits: UInt16) -> UInt32:
    # Canonicalize signed zero. Integer comparison also preserves subnormals.
    var b = UInt32(bits)
    if (b & 0x7FFF) == 0:
        b = 0
    return ((~b) & 0xFFFF) if (b & 0x8000) != 0 else (b ^ 0x8000)


@always_inline
def _group_winner[OL: TensorLayout](
    output: TileTensor[DType.uint32, OL, MutAnyOrigin],
    rank: UInt32, token: UInt32, invalid: UInt32,
):
    comptime assert output.flat_rank == 2
    var top = warp.max(rank)
    var id = warp.min(token if rank == top else UInt32(0xFFFFFFFF))
    var bad = warp.max(invalid)
    var shared = stack_allocation[DType.uint32, address_space=AddressSpace.SHARED](row_major[4,3]())
    var group = thread_idx.x // WARP_SIZE
    if lane_id() == 0:
        shared[group,0] = top
        shared[group,1] = id
        shared[group,2] = bad
    barrier()
    if thread_idx.x == 0:
        var best: UInt32 = 0
        var winner = UInt32(0xFFFFFFFF)
        var any_bad: UInt32 = 0
        for i in range(4):
            var r = rebind[UInt32](shared[i,0])
            var t = rebind[UInt32](shared[i,1])
            any_bad |= rebind[UInt32](shared[i,2])
            if r > best or (r == best and t < winner):
                best = r
                winner = t
        output[block_idx.x,0] = best
        output[block_idx.x,1] = winner
        output[block_idx.x,2] = any_bad


def _argmax[IL: TensorLayout, OL: TensorLayout](
    logits: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, OL, MutAnyOrigin], count: Int32,
):
    comptime assert is_apple_gpu() and WARP_SIZE == 32
    comptime assert logits.flat_rank == 2
    var best: UInt32 = 0
    var winner = UInt32(0xFFFFFFFF)
    var bad: UInt32 = 0
    for j in range(8):
        var i = block_idx.x*1024 + thread_idx.x + j*128
        if i < Int(count):
            var bits = bitcast[DType.uint16](rebind[Scalar[DType.bfloat16]](logits[0,i]))
            bad |= UInt32((bits & 0x7F80) == 0x7F80)
            var rank = bf16_rank(bits)
            if rank > best or (rank == best and UInt32(i) < winner):
                best = rank
                winner = UInt32(i)
    _group_winner(partials,best,winner,bad)


def _finish[IL: TensorLayout, OL: TensorLayout](
    partials: TileTensor[DType.uint32, IL, MutAnyOrigin],
    result: TileTensor[DType.uint32, OL, MutAnyOrigin], count: Int32,
):
    comptime assert is_apple_gpu() and WARP_SIZE == 32
    comptime assert partials.flat_rank == 2
    var best: UInt32 = 0
    var winner = UInt32(0xFFFFFFFF)
    var bad: UInt32 = 0
    var i = thread_idx.x
    while i < Int(count):
        var rank = rebind[UInt32](partials[i,0])
        var token = rebind[UInt32](partials[i,1])
        bad |= rebind[UInt32](partials[i,2])
        if rank > best or (rank == best and token < winner):
            best = rank
            winner = token
        i += 128
    _group_winner(result,best,winner,bad)


def _head[WRITE_LOGITS: Bool, IL: TensorLayout, WL: TensorLayout, LL: TensorLayout, PL: TensorLayout](
    input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    logits: TileTensor[DType.bfloat16, LL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, PL, MutAnyOrigin], width: Int32, count: Int32,
):
    comptime assert is_apple_gpu() and WARP_SIZE == 32
    comptime assert input.flat_rank == 2 and weight.flat_rank == 2 and logits.flat_rank == 2
    var lane = lane_id()
    var group = thread_idx.x // WARP_SIZE
    var best: UInt32 = 0
    var winner = UInt32(0xFFFFFFFF)
    var bad: UInt32 = 0
    for j in range(16):
        var token = block_idx.x*64 + j*4 + group
        if token < Int(count):
            var accumulator: Float32 = 0
            var k = lane
            while k < Int(width):
                var x = rebind[Scalar[DType.bfloat16]](input[0,k])
                var w = rebind[Scalar[DType.bfloat16]](weight[token,k])
                accumulator += x.cast[DType.float32]() * w.cast[DType.float32]()
                k += WARP_SIZE
            var total = warp.sum(accumulator)
            if lane == 0:
                # Exactly the bias-free linear kernel's materialization boundary.
                var rounded = (total + Float32(0)).cast[DType.bfloat16]()
                comptime if WRITE_LOGITS:
                    logits[0,token] = rebind[logits.ElementType](rounded)
                var bits = bitcast[DType.uint16](rounded)
                bad |= UInt32((bits & 0x7F80) == 0x7F80)
                var rank = bf16_rank(bits)
                if rank > best or (rank == best and UInt32(token) < winner):
                    best = rank
                    winner = UInt32(token)
    _group_winner(partials,best,winner,bad)


def _validate[LL: TensorLayout, PL: TensorLayout, RL: TensorLayout](
    ctx: DeviceContext, logits: TileTensor[DType.bfloat16, LL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, PL, MutAnyOrigin],
    result: TileTensor[DType.uint32, RL, MutAnyOrigin], groups: Int,
) raises:
    comptime assert logits.flat_rank == 2 and partials.flat_rank == 2 and result.flat_rank == 2
    if ctx.api() != "metal" or Int(logits.dim[0]()) != 1 or Int(logits.dim[1]()) < 1:
        raise Error("token selection requires Metal and one positive logit row")
    if Int(partials.dim[0]()) < groups or Int(partials.dim[1]()) != 3 or Int(result.dim[0]()) != 1 or Int(result.dim[1]()) != 3:
        raise Error("invalid token selection scratch geometry")


def enqueue_argmax[LL: TensorLayout, PL: TensorLayout, RL: TensorLayout](
    ctx: DeviceContext, logits: TileTensor[DType.bfloat16, LL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, PL, MutAnyOrigin],
    result: TileTensor[DType.uint32, RL, MutAnyOrigin],
) raises:
    var count = Int(logits.dim[1]())
    var groups = (count+1023)//1024
    _validate(ctx,logits,partials,result,groups)
    comptime first = _argmax[LL,PL]
    comptime last = _finish[PL,RL]
    ctx.enqueue_function[first](logits,partials,Int32(count),grid_dim=groups,block_dim=128)
    ctx.enqueue_function[last](partials,result,Int32(groups),grid_dim=1,block_dim=128)


def enqueue_head_argmax[WRITE_LOGITS: Bool, IL: TensorLayout, WL: TensorLayout, LL: TensorLayout, PL: TensorLayout, RL: TensorLayout](
    ctx: DeviceContext, input: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    weight: TileTensor[DType.bfloat16, WL, MutAnyOrigin],
    logits: TileTensor[DType.bfloat16, LL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, PL, MutAnyOrigin],
    result: TileTensor[DType.uint32, RL, MutAnyOrigin],
) raises:
    comptime assert input.flat_rank == 2 and weight.flat_rank == 2
    var count = Int(logits.dim[1]())
    var groups = (count+63)//64
    _validate(ctx,logits,partials,result,groups)
    var width = Int(input.dim[1]())
    if Int(input.dim[0]()) != 1 or width < 1 or Int(weight.dim[0]()) != count or Int(weight.dim[1]()) != width:
        raise Error("invalid fused head shape")
    comptime first = _head[WRITE_LOGITS,IL,WL,LL,PL]
    comptime last = _finish[PL,RL]
    ctx.enqueue_function[first](input,weight,logits,partials,Int32(width),Int32(count),grid_dim=groups,block_dim=128)
    ctx.enqueue_function[last](partials,result,Int32(groups),grid_dim=1,block_dim=128)
