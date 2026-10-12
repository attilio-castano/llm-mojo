"""Exact greedy selection of rounded BF16 logits on Metal, without atomics.

Callers own nonoverlapping logits, partial and result storage and retain it
through device completion. Each record is (ordered score, lowest token ID,
any-nonfinite flag). No interpretation of the ID is valid when the flag is set.
Each logit row is selected independently: one group of partial records and one
result record per row, with the same group partition whatever the row count.
Tensor indexing respects the supplied layouts. The fused vocabulary-projection
variant measured in studies/model_generation/token-selection.md was not promoted.
"""
from layout import TensorLayout, TileTensor, row_major, stack_allocation
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.primitives import warp
from std.memory import bitcast
from std.sys.info import is_apple_gpu
from llm_mojo.kernels.token_selection_contract import (
    ArgmaxRecord, bf16_rank, bf16_nonfinite, empty_argmax, merge_argmax,
)


@always_inline
def _group_winner[OL: TensorLayout](
    output: TileTensor[DType.uint32, OL, MutAnyOrigin],
    rank: UInt32, token: UInt32, invalid: UInt32, record: Int,
):
    comptime assert output.flat_rank == 2
    var top = warp.max(rank)
    var id = warp.min(token if rank == top else UInt32(0xFFFFFFFF))
    var bad = warp.max(invalid)
    var shared = stack_allocation[DType.uint32, address_space=AddressSpace.SHARED](row_major[4,3]())
    comptime assert shared.flat_rank == 2
    var group = thread_idx.x // WARP_SIZE
    if lane_id() == 0:
        shared[group,0] = top
        shared[group,1] = id
        shared[group,2] = bad
    barrier()
    if thread_idx.x == 0:
        var best = empty_argmax()
        for i in range(4):
            best = merge_argmax(best, ArgmaxRecord(
                rebind[UInt32](shared[i,0]),
                rebind[UInt32](shared[i,1]),
                rebind[UInt32](shared[i,2]),
            ))
        output[record,0] = best.rank
        output[record,1] = best.token
        output[record,2] = best.invalid


def _argmax[IL: TensorLayout, OL: TensorLayout](
    logits: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, OL, MutAnyOrigin], count: Int32,
):
    comptime assert is_apple_gpu() and WARP_SIZE == 32
    comptime assert logits.flat_rank == 2
    var row = block_idx.y
    var best = empty_argmax()
    for j in range(8):
        var i = block_idx.x*1024 + thread_idx.x + j*128
        if i < Int(count):
            var bits = bitcast[DType.uint16](rebind[Scalar[DType.bfloat16]](logits[row,i]))
            best = merge_argmax(best, ArgmaxRecord(bf16_rank(bits), UInt32(i), bf16_nonfinite(bits)))
    _group_winner(partials,best.rank,best.token,best.invalid,row*((Int(count)+1023)//1024)+block_idx.x)


def _finish[IL: TensorLayout, OL: TensorLayout](
    partials: TileTensor[DType.uint32, IL, MutAnyOrigin],
    result: TileTensor[DType.uint32, OL, MutAnyOrigin], count: Int32,
):
    comptime assert is_apple_gpu() and WARP_SIZE == 32
    comptime assert partials.flat_rank == 2
    var row = block_idx.x
    var first = row*Int(count)
    var best = empty_argmax()
    var i = thread_idx.x
    while i < Int(count):
        best = merge_argmax(best, ArgmaxRecord(
            rebind[UInt32](partials[first+i,0]),
            rebind[UInt32](partials[first+i,1]),
            rebind[UInt32](partials[first+i,2]),
        ))
        i += 128
    _group_winner(result,best.rank,best.token,best.invalid,row)


def _validate[LL: TensorLayout, PL: TensorLayout, RL: TensorLayout](
    ctx: DeviceContext, logits: TileTensor[DType.bfloat16, LL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, PL, MutAnyOrigin],
    result: TileTensor[DType.uint32, RL, MutAnyOrigin], groups: Int,
) raises:
    comptime assert logits.flat_rank == 2 and partials.flat_rank == 2 and result.flat_rank == 2
    var rows = Int(logits.dim[0]())
    if ctx.api() != "metal" or rows < 1 or Int(logits.dim[1]()) < 1:
        raise Error("token selection requires Metal and positive logit rows")
    if (Int(partials.dim[0]()) < rows*groups or Int(partials.dim[1]()) != 3
            or Int(result.dim[0]()) != rows or Int(result.dim[1]()) != 3):
        raise Error("invalid token selection scratch geometry")


def enqueue_argmax[LL: TensorLayout, PL: TensorLayout, RL: TensorLayout](
    ctx: DeviceContext, logits: TileTensor[DType.bfloat16, LL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, PL, MutAnyOrigin],
    result: TileTensor[DType.uint32, RL, MutAnyOrigin],
) raises:
    var rows = Int(logits.dim[0]())
    var count = Int(logits.dim[1]())
    var groups = (count+1023)//1024
    _validate(ctx,logits,partials,result,groups)
    comptime first = _argmax[LL,PL]
    comptime last = _finish[PL,RL]
    ctx.enqueue_function[first](logits,partials,Int32(count),grid_dim=(groups,rows),block_dim=128)
    ctx.enqueue_function[last](partials,result,Int32(groups),grid_dim=rows,block_dim=128)
