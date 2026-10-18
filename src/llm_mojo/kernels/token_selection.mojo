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
from std.collections import Array
from std.memory import bitcast
from std.sys.info import is_apple_gpu
from llm_mojo.kernels.token_selection_contract import (
    ArgmaxRecord, bf16_rank, bf16_nonfinite, empty_argmax, merge_argmax,
)
from llm_mojo.kernels.argmax_reduction import (
    ArgmaxPlan, check_argmax_plan, evaluate_argmax_plan, sequential_argmax_plan,
)
from llm_mojo.kernels.finishing_proof import (
    FinishingProgram, make_finishing_program,
)
from llm_mojo.kernels.finishing_execution import (
    FinishingExecutionDescriptor, make_finishing_execution_descriptor,
    check_finishing_execution_descriptor, finishing_execution_index,
    finishing_execution_loop_active, finishing_execution_load_active,
)
from llm_mojo.kernels.finishing_execution_check import (
    make_finishing_execution_certificate, check_finishing_execution_certificate_request,
    same_finishing_execution_descriptor,
)
from llm_mojo.kernels.finishing_typed_derivation import (
    FinishingTypedProofs, FINISHING_TYPED_PROOF_CAPACITY,
    make_finishing_typed_proofs, check_finishing_typed_proofs,
)
from llm_mojo.kernels.finishing_typed_coverage import (
    FinishingTypedCoverageProofs, FINISHING_TYPED_COVERAGE_CAPACITY,
    make_finishing_typed_coverage, check_finishing_typed_coverage,
)


@always_inline
def argmax_tile_offset[threads: Int](thread: Int, item: Int) -> Int:
    """The production ownership map, shared with its compile-time check."""
    return thread + item * threads


def check_argmax_schedule[threads: Int, items_per_thread: Int]() -> Bool:
    """Check a representable tile and its bijection with (thread, item).

    This checks logical coverage, not device resource limits or warp behavior.
    The Metal route launches whole 32-thread warps; item count is independent.
    """
    if threads < 1 or threads % 32 != 0 or items_per_thread < 1:
        return False
    if threads > 2147483647 // items_per_thread:
        return False
    for offset in range(threads * items_per_thread):
        var thread = offset % threads
        var item = offset // threads
        if (thread < 0 or thread >= threads or item < 0 or item >= items_per_thread
                or argmax_tile_offset[threads](thread, item) != offset):
            return False
    return True


@fieldwise_init
struct ArgmaxFoldPlan[threads: Int, items_per_thread: Int, plan_nodes: Int](Copyable, Movable):
    """Closed striped repetition of a checked local join over a runtime extent.

    The fixed interpreter starts at zero, pads with Empty, joins each tile,
    and advances by stride. It accepts no arbitrary loop body or callback.
    """
    var stride: Int
    var reduction: ArgmaxPlan[Self.items_per_thread, Self.plan_nodes]
    var program: FinishingProgram
    var proof: FinishingTypedCoverageProofs[FINISHING_TYPED_COVERAGE_CAPACITY]
    var typed_proof: FinishingTypedProofs[FINISHING_TYPED_PROOF_CAPACITY]


def make_argmax_fold_plan[
    threads: Int, items_per_thread: Int, plan_nodes: Int,
    plan: ArgmaxPlan[items_per_thread, plan_nodes],
]() -> ArgmaxFoldPlan[threads, items_per_thread, plan_nodes]:
    var program = make_finishing_program(threads, items_per_thread, threads * items_per_thread)
    var proof = make_finishing_typed_coverage(program)
    var typed_proof = make_finishing_typed_proofs(program)
    return ArgmaxFoldPlan[threads, items_per_thread, plan_nodes](
        threads * items_per_thread, materialize[plan](), program, proof^, typed_proof^,
    )


def check_argmax_fold_plan[threads: Int, items_per_thread: Int, plan_nodes: Int](
    fold: ArgmaxFoldPlan[threads, items_per_thread, plan_nodes],
) -> Bool:
    """Check a symbolic traversal derivation and the local reduction tree.

    Runtime count is at most Int32.MAX. Padded indices and the exit base may
    exceed it; Int must hold count + stride. The proof-term kernel checks
    explicit inverse ownership, uniqueness, visitation and headroom, plus
    closed base-linearity and termination terms. Nat induction is derived
    from checked declarations. It does not reflect an arbitrary Mojo body.
    """
    if threads < 1 or threads % 32 != 0 or items_per_thread < 1:
        return False
    if (fold.program.threads != threads or fold.program.items != items_per_thread
            or fold.stride != fold.program.stride
            or fold.program.count_limit != 2147483647 or fold.program.integer_limit != Int.MAX):
        return False
    if not check_finishing_typed_coverage(fold.program, fold.proof):
        return False
    if not check_finishing_typed_proofs(fold.program, fold.typed_proof):
        return False
    return check_argmax_plan(fold.reduction)


@always_inline
def argmax_fold_index[threads: Int](base: Int, thread: Int, item: Int) -> Int:
    return base + argmax_tile_offset[threads](thread, item)


def make_argmax_finishing_execution[threads: Int, items_per_thread: Int, plan_nodes: Int](
    fold: ArgmaxFoldPlan[threads, items_per_thread, plan_nodes],
) -> FinishingExecutionDescriptor[
    threads, items_per_thread, plan_nodes, 32, 65, 2 * (threads // 32) + 1,
]:
    """Bind the source finisher to the same closed data the model consumes.

    The logical subgroup is a checked join DAG. Its implementation by the
    rank-max/token-min/invalid-max warp operations remains a named backend
    refinement, including the boundary condition invalid in {0,1}.
    """
    var subgroup = sequential_argmax_plan[32]()
    var final = sequential_argmax_plan[threads // 32]()
    var descriptor = make_finishing_execution_descriptor[
        threads, items_per_thread, plan_nodes, 32, 65, 2 * (threads // 32) + 1,
    ](fold.reduction, subgroup, final)
    # Keep every program field, including limits, rather than silently
    # replacing a caller-supplied program with a canonical one.
    descriptor.program = fold.program
    if fold.stride != fold.program.stride:
        descriptor.operation_profile = -1
    return descriptor^


def check_argmax_finishing_execution[threads: Int, items_per_thread: Int, plan_nodes: Int](
    fold: ArgmaxFoldPlan[threads, items_per_thread, plan_nodes],
    descriptor: FinishingExecutionDescriptor[
        threads, items_per_thread, plan_nodes, 32, 65, 2 * (threads // 32) + 1,
    ],
) -> Bool:
    """Exact descriptor binding; traversal proof checks remain independent."""
    if (not check_finishing_execution_descriptor(descriptor)
        or fold.stride != descriptor.program.stride):
        return False
    var expected = make_argmax_finishing_execution(fold)
    return same_finishing_execution_descriptor(descriptor, expected)


@always_inline
def _group_winner[
    OL: TensorLayout, threads: Int,
    plan_nodes: Int = 2 * (threads // 32) + 1,
    plan: ArgmaxPlan[threads // 32, plan_nodes] = sequential_argmax_plan[threads // 32](),
](
    output: TileTensor[DType.uint32, OL, MutAnyOrigin],
    rank: UInt32, token: UInt32, invalid: UInt32, record: Int,
):
    comptime assert output.flat_rank == 2
    comptime assert WARP_SIZE == 32, "argmax subgroup requires 32-lane Metal warps"
    comptime warps = threads // 32
    comptime assert check_argmax_plan(plan), "argmax reduction plan rejected"
    var top = warp.max(rank)
    var id = warp.min(token if rank == top else UInt32(0xFFFFFFFF))
    var bad = warp.max(invalid)
    var shared = stack_allocation[DType.uint32, address_space=AddressSpace.SHARED](row_major[warps,3]())
    comptime assert shared.flat_rank == 2
    var group = thread_idx.x // WARP_SIZE
    if lane_id() == 0:
        shared[group,0] = top
        shared[group,1] = id
        shared[group,2] = bad
    barrier()
    if thread_idx.x == 0:
        var records = Array[ArgmaxRecord, warps](uninitialized=True)
        comptime for i in range(warps):
            records[i] = ArgmaxRecord(
                rebind[UInt32](shared[i,0]),
                rebind[UInt32](shared[i,1]),
                rebind[UInt32](shared[i,2]),
            )
        var best = evaluate_argmax_plan[warps, plan_nodes, plan](records)
        output[record,0] = best.rank
        output[record,1] = best.token
        output[record,2] = best.invalid


def _argmax[
    IL: TensorLayout, OL: TensorLayout, threads: Int, items_per_thread: Int,
    plan_nodes: Int, plan: ArgmaxPlan[items_per_thread, plan_nodes],
](
    logits: TileTensor[DType.bfloat16, IL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, OL, MutAnyOrigin], count: Int32,
):
    comptime assert is_apple_gpu() and WARP_SIZE == 32
    comptime assert logits.flat_rank == 2
    comptime tile = threads * items_per_thread
    var row = block_idx.y
    var records = Array[ArgmaxRecord, items_per_thread](uninitialized=True)
    comptime for j in range(items_per_thread):
        var i = Int(block_idx.x) * tile + argmax_tile_offset[threads](Int(thread_idx.x), j)
        records[j] = empty_argmax()
        if i < Int(count):
            var bits = bitcast[DType.uint16](rebind[Scalar[DType.bfloat16]](logits[row,i]))
            records[j] = ArgmaxRecord(bf16_rank(bits), UInt32(i), bf16_nonfinite(bits))
    var best = evaluate_argmax_plan[items_per_thread, plan_nodes, plan](records)
    var groups = 1 + (Int(count) - 1) // tile
    _group_winner[OL, threads](partials,best.rank,best.token,best.invalid,Int(row)*groups+Int(block_idx.x))


def _finish[
    IL: TensorLayout, OL: TensorLayout, threads: Int, items_per_thread: Int,
    plan_nodes: Int, fold: ArgmaxFoldPlan[threads, items_per_thread, plan_nodes],
](
    partials: TileTensor[DType.uint32, IL, MutAnyOrigin],
    result: TileTensor[DType.uint32, OL, MutAnyOrigin], count: Int32,
):
    comptime assert is_apple_gpu() and WARP_SIZE == 32
    comptime assert partials.flat_rank == 2
    comptime assert check_argmax_fold_plan(fold), "argmax fold plan rejected"
    comptime descriptor = make_argmax_finishing_execution(fold)
    comptime assert check_argmax_finishing_execution(fold, descriptor), "checked finishing execution rejected"
    comptime certificate = make_finishing_execution_certificate(descriptor)
    comptime assert check_finishing_execution_certificate_request(certificate, descriptor), "checked finishing execution rejected"
    comptime plan = descriptor.local
    comptime program = descriptor.program
    var row = block_idx.x
    var first = Int(row)*Int(count)
    var best = empty_argmax()
    var base: Int = program.start
    while finishing_execution_loop_active(descriptor, base, Int(count)):
        var records = Array[ArgmaxRecord, items_per_thread](uninitialized=True)
        comptime for item in range(items_per_thread):
            var i = finishing_execution_index(descriptor, base, Int(thread_idx.x), item)
            records[item] = empty_argmax()
            if finishing_execution_load_active(descriptor, i, Int(count)):
                records[item] = ArgmaxRecord(
                    rebind[UInt32](partials[first+i,0]),
                    rebind[UInt32](partials[first+i,1]),
                    rebind[UInt32](partials[first+i,2]),
                )
        var tile_best = evaluate_argmax_plan[items_per_thread, plan_nodes, plan](records)
        best = merge_argmax(best, tile_best)
        base += program.stride
    _group_winner[OL, threads, 2 * (threads // 32) + 1, descriptor.final](
        result,best.rank,best.token,best.invalid,Int(row))


def _validate[LL: TensorLayout, PL: TensorLayout, RL: TensorLayout](
    ctx: DeviceContext, logits: TileTensor[DType.bfloat16, LL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, PL, MutAnyOrigin],
    result: TileTensor[DType.uint32, RL, MutAnyOrigin], groups: Int,
) raises:
    comptime assert logits.flat_rank == 2 and partials.flat_rank == 2 and result.flat_rank == 2
    var rows = Int(logits.dim[0]())
    var count = Int(logits.dim[1]())
    if ctx.api() != "metal" or rows < 1 or count < 1:
        raise Error("token selection requires Metal and positive logit rows")
    if (count > 2147483647 or rows > 2147483647 or groups < 1 or groups > 2147483647
            or rows > Int.MAX // count or rows > (Int.MAX // 3) // groups):
        raise Error("token selection extents exceed representable counts")
    if (Int(partials.dim[0]()) < rows*groups or Int(partials.dim[1]()) != 3
            or Int(result.dim[0]()) != rows or Int(result.dim[1]()) != 3):
        raise Error("invalid token selection scratch geometry")


def enqueue_argmax[
    LL: TensorLayout, PL: TensorLayout, RL: TensorLayout,
    threads: Int = 128, items_per_thread: Int = 8,
    plan_nodes: Int = 2 * items_per_thread + 1,
    plan: ArgmaxPlan[items_per_thread, plan_nodes] = sequential_argmax_plan[items_per_thread](),
    finish_threads: Int = threads, finish_items_per_thread: Int = 1,
    finish_plan_nodes: Int = 2 * finish_items_per_thread + 1,
    finish_plan: ArgmaxPlan[finish_items_per_thread, finish_plan_nodes] = sequential_argmax_plan[finish_items_per_thread](),
](
    ctx: DeviceContext, logits: TileTensor[DType.bfloat16, LL, MutAnyOrigin],
    partials: TileTensor[DType.uint32, PL, MutAnyOrigin],
    result: TileTensor[DType.uint32, RL, MutAnyOrigin],
) raises:
    comptime assert check_argmax_schedule[threads, items_per_thread](), "argmax schedule rejected"
    comptime assert check_argmax_plan(plan), "argmax reduction plan rejected"
    comptime fold = make_argmax_fold_plan[finish_threads,finish_items_per_thread,finish_plan_nodes,finish_plan]()
    comptime assert check_argmax_fold_plan(fold), "argmax fold plan rejected"
    var rows = Int(logits.dim[0]())
    var count = Int(logits.dim[1]())
    comptime tile = threads * items_per_thread
    var groups = 1 + (count - 1) // tile if count > 0 else 0
    _validate(ctx,logits,partials,result,groups)
    comptime first = _argmax[LL,PL,threads,items_per_thread,plan_nodes,plan]
    comptime last = _finish[PL,RL,finish_threads,finish_items_per_thread,finish_plan_nodes,fold]
    ctx.enqueue_function[first](logits,partials,Int32(count),grid_dim=(groups,rows),block_dim=threads)
    ctx.enqueue_function[last](partials,result,Int32(groups),grid_dim=rows,block_dim=finish_threads)
