"""BF16 ordering, reduction boundaries and rounded ties for GPU argmax."""
from std.memory import bitcast
from std.collections import Array
from std.testing import assert_equal, assert_raises
from layout import TensorLayout, TileTensor, row_major, col_major
from max.gpu.host import DeviceContext
from llm_mojo.kernels.linear import enqueue_linear_apple_gpu
from llm_mojo.kernels.token_selection import (
    enqueue_argmax, bf16_rank, _finish, make_argmax_fold_plan,
)
from llm_mojo.kernels.argmax_reduction import (
    ArgmaxPlan, ReductionNode, input_node, merge_node,
    sequential_argmax_plan, balanced_argmax_plan,
)


def reduction(ctx: DeviceContext, count: Int, pattern: Int) raises:
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](count)
    var groups = (count+1023)//1024
    var partials = ctx.enqueue_create_buffer[DType.uint32]((groups+1)*3)
    var result = ctx.enqueue_create_buffer[DType.uint32](6)
    partials.enqueue_fill(0xDEADBEEF)
    result.enqueue_fill(0xDEADBEEF)
    var winner = 0
    var best = Float32(-3.402823466e38)
    var invalid: UInt32 = 0
    with logits.map_to_host() as mapped:
        for i in range(count):
            var bits = UInt16(i % 65536)
            if pattern == 1:
                bits = UInt16(0xBF80)  # all negative, last wins
                if i == count-1:
                    bits = UInt16(0x8001)  # negative subnormal
            elif pattern == 2:
                bits = UInt16(0x8000 if i%2 == 0 else 0)
            elif pattern == 3:
                bits = UInt16(0xBF80)
                if i == 1 or i == count-1:
                    bits = UInt16(1)  # tied positive subnormals across groups
            elif pattern == 7:
                bits = UInt16(1 if i == count-1 else 0)
            elif pattern == 8:
                bits = UInt16(0x8000 if i == count-1 else 0x8001)
            elif pattern >= 4:
                bits = UInt16(0x3F80)
                if i == count-1:
                    bits = UInt16(0x7FC0 if pattern == 4 else (0x7F80 if pattern == 5 else 0xFF80))
            # Pattern zero enumerates every finite encoding.
            elif (bits & 0x7F80) == 0x7F80:
                bits = UInt16(0)
            var value = bitcast[DType.bfloat16](bits)
            mapped.unsafe_ptr()[unsafe_offset=i] = value
            var f = value.cast[DType.float32]()
            if f != f or f > Float32(3.402823466e38) or f < Float32(-3.402823466e38):
                invalid = 1
            elif f > best:
                best = f
                winner = i
    enqueue_argmax(ctx,TileTensor(logits,row_major(1,count)),
        TileTensor(partials,row_major(groups,3)),TileTensor(result,row_major(1,3)))
    with result.map_to_host() as mapped:
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=2],invalid)
        if invalid == 0:
            assert_equal(Int(mapped.unsafe_ptr()[unsafe_offset=1]),winner)
        for i in range(3,6):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i],UInt32(0xDEADBEEF))
    with partials.map_to_host() as mapped:
        for i in range(groups*3,(groups+1)*3):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i],UInt32(0xDEADBEEF))


def rounded_tie(ctx: DeviceContext) raises:
    # Token 64 is larger in FP32 but ties token 0 after BF16 materialization.
    var input = ctx.enqueue_create_buffer[DType.bfloat16](2)
    var weight = ctx.enqueue_create_buffer[DType.bfloat16](65*2)
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](65)
    var partials = ctx.enqueue_create_buffer[DType.uint32](6)
    var result = ctx.enqueue_create_buffer[DType.uint32](3)
    input.enqueue_fill(1)
    weight.enqueue_fill(-1)
    with weight.map_to_host() as mapped:
        mapped.unsafe_ptr()[unsafe_offset=0] = 1
        mapped.unsafe_ptr()[unsafe_offset=1] = 0
        mapped.unsafe_ptr()[unsafe_offset=128] = 1
        mapped.unsafe_ptr()[unsafe_offset=129] = Float32(1.0/1024).cast[DType.bfloat16]()
    # The production route: materialize BF16 logits, then select on the rounded values.
    enqueue_linear_apple_gpu(ctx,TileTensor(input,row_major(1,2)),TileTensor(weight,row_major(65,2)),
        TileTensor(logits,row_major(1,65)))
    enqueue_argmax(ctx,TileTensor(logits,row_major(1,65)),TileTensor(partials,row_major(2,3)),
        TileTensor(result,row_major(1,3)))
    with result.map_to_host() as mapped:
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=1],UInt32(0))
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=2],UInt32(0))
    with logits.map_to_host() as mapped:
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=0].cast[DType.float32](),Float32(1))
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=64].cast[DType.float32](),Float32(1))


def independent_rows(ctx: DeviceContext) raises:
    # A ragged second group, ties across groups and a poisoned row exercise
    # the scalar contract's connection to the actual multi-row device route.
    var count = 1025
    var rows = 3
    var groups = 2
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](rows * count)
    var partials = ctx.enqueue_create_buffer[DType.uint32](rows * groups * 3 + 3)
    var result = ctx.enqueue_create_buffer[DType.uint32](rows * 3 + 3)
    partials.enqueue_fill(0xDEADBEEF)
    result.enqueue_fill(0xDEADBEEF)
    with logits.map_to_host() as mapped:
        for row in range(rows):
            for i in range(count):
                var bits = UInt16(0x8000 if row == 0 else 0xBF80)
                if row == 0 and i == count - 1:
                    bits = UInt16(0)
                elif row == 1 and i == count - 1:
                    bits = UInt16(0x7F81)
                elif row == 2 and (i == 127 or i == count - 1):
                    bits = UInt16(0x0001)
                mapped.unsafe_ptr()[unsafe_offset=row * count + i] = bitcast[DType.bfloat16](bits)
    enqueue_argmax(ctx, TileTensor(logits, row_major(rows, count)),
                  TileTensor(partials, row_major(rows * groups, 3)),
                  TileTensor(result, row_major(rows, 3)))
    with result.map_to_host() as mapped:
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=0], bf16_rank(UInt16(0)))
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=1], UInt32(0))
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=2], UInt32(0))
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=5], UInt32(1))
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=6], bf16_rank(UInt16(1)))
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=7], UInt32(127))
        assert_equal(mapped.unsafe_ptr()[unsafe_offset=8], UInt32(0))
        for i in range(rows * 3, rows * 3 + 3):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i], UInt32(0xDEADBEEF))
    with partials.map_to_host() as mapped:
        for i in range(rows * groups * 3, rows * groups * 3 + 3):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i], UInt32(0xDEADBEEF))


def _offset(row: Int, column: Int, rows: Int, columns: Int, column_major: Bool) -> Int:
    return column * rows + row if column_major else row * columns + column


def _expected_rank(bits: UInt16) -> UInt32:
    # Choose with FP32 comparisons, then encode the chosen BF16 score without
    # calling the production rank/merge helpers. Signed zeros have one rank.
    if (bits & 0x7FFF) == 0:
        return UInt32(0x8000)
    return UInt32(0xFFFF - bits) if (bits & 0x8000) != 0 else UInt32(bits) + 0x8000


def _schedule_case[threads: Int, items: Int, balanced: Bool,
                   LL: TensorLayout, PL: TensorLayout, RL: TensorLayout,
                   finish_threads: Int = threads, finish_items: Int = 1,
                   finish_balanced: Bool = False, finish_compact: Bool = False](
    ctx: DeviceContext, count: Int, column_major: Bool,
    logits_layout: LL, partial_layout: PL, result_layout: RL,
) raises:
    # Invalid selections must stay isolated to their row. Check the partial
    # records as well as the result, so a broken first pass cannot hide behind
    # a correct winner elsewhere in the second pass.
    var rows = 8
    var tile = threads * items
    var groups = (count + tile - 1) // tile
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](rows * count + 8)
    var partials = ctx.enqueue_create_buffer[DType.uint32](rows * groups * 3 + 9)
    var result = ctx.enqueue_create_buffer[DType.uint32](rows * 3 + 9)
    partials.enqueue_fill(0xDEADBEEF)
    result.enqueue_fill(0xDEADBEEF)
    var expected_rank = List[UInt32](capacity=rows * groups)
    var expected_token = List[UInt32](capacity=rows * groups)
    var expected_invalid = List[UInt32](capacity=rows * groups)
    var row_rank = List[UInt32](capacity=rows)
    var row_token = List[UInt32](capacity=rows)
    var row_invalid = List[UInt32](capacity=rows)
    with logits.map_to_host() as mapped:
        for row in range(rows):
            var best = Float32(-3.402823466e38)
            var winner: UInt32 = 0xFFFFFFFF
            var best_bits: UInt16 = 0
            var invalid: UInt32 = 0
            for group in range(groups):
                var group_best = Float32(-3.402823466e38)
                var group_winner: UInt32 = 0xFFFFFFFF
                var group_bits: UInt16 = 0
                var group_invalid: UInt32 = 0
                for i in range(group * tile, min((group + 1) * tile, count)):
                    var bits = UInt16(0xBF80)
                    if row == 0 and i == count - 1:
                        bits = UInt16(0x3F80)  # unique winner in the ragged tail
                    elif row == 1 and (i == min(31, count - 1) or i == count - 1):
                        bits = UInt16(1)  # tied subnormals across warps and tiles
                    elif row == 2:
                        bits = UInt16(0x8000 if i % 2 == 0 else 0)
                    elif row == 3 and i == count - 1:
                        bits = UInt16(0x8001)  # negative subnormal beats -1
                    elif row == 4 and i == count - 1:
                        bits = UInt16(0x7FC1)  # quiet NaN after finite values
                    elif row == 5 and i == min(threads - 1, count - 1):
                        bits = UInt16(0x7F81)  # signaling NaN in a later warp lane
                    elif row == 6 and i == min(tile - 1, count - 1):
                        bits = UInt16(0x7F80)
                    elif row == 7 and i == 0:
                        bits = UInt16(0xFF80)  # poison need not be the maximum
                    var value = bitcast[DType.bfloat16](bits)
                    mapped.unsafe_ptr()[unsafe_offset=_offset(row, i, rows, count, column_major)] = value
                    var scalar = value.cast[DType.float32]()
                    if scalar != scalar or scalar > Float32(3.402823466e38) or scalar < Float32(-3.402823466e38):
                        invalid = 1
                        group_invalid = 1
                    else:
                        if scalar > best:
                            best = scalar
                            winner = UInt32(i)
                            best_bits = bits
                        if scalar > group_best:
                            group_best = scalar
                            group_winner = UInt32(i)
                            group_bits = bits
                expected_rank.append(_expected_rank(group_bits))
                expected_token.append(group_winner)
                expected_invalid.append(group_invalid)
            row_rank.append(_expected_rank(best_bits))
            row_token.append(winner)
            row_invalid.append(invalid)
        for i in range(rows * count, rows * count + 8):
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](UInt16(0x7FC1))
    comptime first_plan = (balanced_argmax_plan[items]() if balanced
                           else sequential_argmax_plan[items]())
    comptime if finish_compact:
        enqueue_argmax[threads=threads, items_per_thread=items, plan=first_plan,
                       finish_threads=finish_threads, finish_items_per_thread=finish_items,
                       finish_plan_nodes=2 * finish_items - 1, finish_plan=_compact_plan[finish_items]()](
            ctx, TileTensor(logits, logits_layout), TileTensor(partials, partial_layout), TileTensor(result, result_layout))
    else:
        comptime finish_plan = (balanced_argmax_plan[finish_items]() if finish_balanced
                                else sequential_argmax_plan[finish_items]())
        enqueue_argmax[threads=threads, items_per_thread=items, plan=first_plan,
                       finish_threads=finish_threads, finish_items_per_thread=finish_items,
                       finish_plan=finish_plan](
            ctx, TileTensor(logits, logits_layout), TileTensor(partials, partial_layout), TileTensor(result, result_layout))
    with result.map_to_host() as mapped:
        for row in range(rows):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=_offset(row, 2, rows, 3, column_major)], row_invalid[row])
            if row_invalid[row] == 0:
                assert_equal(mapped.unsafe_ptr()[unsafe_offset=_offset(row, 0, rows, 3, column_major)], row_rank[row])
                assert_equal(mapped.unsafe_ptr()[unsafe_offset=_offset(row, 1, rows, 3, column_major)], row_token[row])
        for i in range(rows * 3, rows * 3 + 9):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i], UInt32(0xDEADBEEF))
    with partials.map_to_host() as mapped:
        for record in range(rows * groups):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=_offset(record, 2, rows * groups, 3, column_major)], expected_invalid[record])
            if expected_invalid[record] == 0:
                assert_equal(mapped.unsafe_ptr()[unsafe_offset=_offset(record, 0, rows * groups, 3, column_major)], expected_rank[record])
                assert_equal(mapped.unsafe_ptr()[unsafe_offset=_offset(record, 1, rows * groups, 3, column_major)], expected_token[record])
        for i in range(rows * groups * 3, rows * groups * 3 + 9):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i], UInt32(0xDEADBEEF))
    with logits.map_to_host() as mapped:
        for i in range(rows * count, rows * count + 8):
            assert_equal(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]), UInt16(0x7FC1))


def _schedule_layout[threads: Int, items: Int, balanced: Bool, column_major: Bool](
    ctx: DeviceContext,
) raises:
    var rows = 8
    var tile = threads * items
    var counts: List[Int] = [1, threads - 1, threads, threads + 1, tile - 1, tile, tile + 1,
                            2 * tile - 1, 2 * tile + 1]
    comptime if threads == 32 and items == 1:
        counts.append(1057)  # 34 partial records: finish needs a second iteration
    for count in counts:
        var groups = (count + tile - 1) // tile
        comptime if column_major:
            _schedule_case[threads, items, balanced](ctx, count, True, col_major(rows, count),
                col_major(rows * groups, 3), col_major(rows, 3))
        else:
            _schedule_case[threads, items, balanced](ctx, count, False, row_major(rows, count),
                row_major(rows * groups, 3), row_major(rows, 3))


def _schedule_pair[threads: Int, items: Int](ctx: DeviceContext) raises:
    _schedule_layout[threads, items, False, False](ctx)
    _schedule_layout[threads, items, True, False](ctx)
    _schedule_layout[threads, items, False, True](ctx)
    _schedule_layout[threads, items, True, True](ctx)


def checked_schedules(ctx: DeviceContext) raises:
    # Eight pairs cover every requested thread count and item count with both
    # trees and both layouts; the existing exhaustive/default tests stay above.
    _schedule_pair[32, 1](ctx)
    _schedule_pair[32, 3](ctx)
    _schedule_pair[64, 8](ctx)
    _schedule_pair[64, 16](ctx)
    _schedule_pair[128, 1](ctx)
    _schedule_pair[128, 3](ctx)
    _schedule_pair[256, 8](ctx)
    _schedule_pair[256, 16](ctx)


def _compact_plan[items: Int]() -> ArgmaxPlan[items, 2 * items - 1]:
    # A caller-supplied plan without identity padding checks that execution
    # uses the declared node count, rather than either factory's shape.
    var nodes = Array[ReductionNode, 2 * items - 1](uninitialized=True)
    for i in range(items):
        nodes[i] = input_node(i)
    var root = 0
    for i in range(1, items):
        nodes[items + i - 1] = merge_node(root, i)
        root = items + i - 1
    return ArgmaxPlan[items, 2 * items - 1](nodes^, root)


def _finish_case[threads: Int, items: Int, nodes: Int, plan: ArgmaxPlan[items, nodes],
                 PL: TensorLayout, RL: TensorLayout](
    ctx: DeviceContext, count: Int, column_major: Bool, partial_layout: PL, result_layout: RL,
) raises:
    # Fabricated canonical records isolate the runtime finisher without making
    # a large vocabulary just to produce hundreds of partials. IDs deliberately
    # differ from record offsets; the scalar oracle uses no production merge.
    var rows = 6
    var partials = ctx.enqueue_create_buffer[DType.uint32](rows * count * 3 + 9)
    var result = ctx.enqueue_create_buffer[DType.uint32](rows * 3 + 9)
    partials.enqueue_fill(0xDEADBEEF)
    result.enqueue_fill(0xDEADBEEF)
    var original = List[UInt32](capacity=rows * count * 3)
    var expected = List[UInt32](capacity=rows * 3)
    with partials.map_to_host() as mapped:
        for row in range(rows):
            var best_rank: UInt32 = 0
            var best_token: UInt32 = 0xFFFFFFFF
            var invalid: UInt32 = 0
            for i in range(count):
                var rank: UInt32 = 0xBF80  # finite BF16 1.0
                var token = UInt32(5000 - i)
                var bad: UInt32 = 0
                if row == 0 and i == count - 1:
                    rank = 0xC080  # unique winner in the final runtime tile
                elif row == 1:
                    rank = 0x8000
                    token = UInt32(5000 + i)
                    if i == 0 or i == count - 1:
                        rank = 0xC000
                        token = UInt32(3 if i == count - 1 else 1000)
                elif row == 2:
                    rank = 0
                    token = 0xFFFFFFFF  # a complete identity-only row
                elif row == 3 and i == count - 1:
                    rank = 0
                    token = 0xFFFFFFFF
                    bad = 1  # poison has a lower rank than every finite winner
                elif row == 4:
                    rank = 0x8000
                    token = UInt32(count - i + 7)
                elif row == 5:
                    rank = 0
                    token = 0xFFFFFFFF
                    if i == count - 1:
                        rank = 0x0080  # finite most-negative BF16 beats identity
                        token = 17
                var record = row * count + i
                mapped.unsafe_ptr()[unsafe_offset=_offset(record, 0, rows * count, 3, column_major)] = rank
                mapped.unsafe_ptr()[unsafe_offset=_offset(record, 1, rows * count, 3, column_major)] = token
                mapped.unsafe_ptr()[unsafe_offset=_offset(record, 2, rows * count, 3, column_major)] = bad
                original.append(rank)
                original.append(token)
                original.append(bad)
                if rank > best_rank or (rank == best_rank and token < best_token):
                    best_rank = rank
                    best_token = token
                invalid |= bad
            expected.append(best_rank)
            expected.append(best_token)
            expected.append(invalid)
    comptime fold = make_argmax_fold_plan[threads, items, nodes, plan]()
    comptime kernel = _finish[PL, RL, threads, items, nodes, fold]
    ctx.enqueue_function[kernel](TileTensor(partials, partial_layout), TileTensor(result, result_layout),
                                 Int32(count), grid_dim=rows, block_dim=threads)
    with result.map_to_host() as mapped:
        for row in range(rows):
            for field in range(3):
                # All fields are well-defined for these canonical records,
                # including the poisoned row's surviving finite maximum.
                assert_equal(mapped.unsafe_ptr()[unsafe_offset=_offset(row, field, rows, 3, column_major)],
                             expected[row * 3 + field])
        for i in range(rows * 3, rows * 3 + 9):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i], UInt32(0xDEADBEEF))
    with partials.map_to_host() as mapped:
        for record in range(rows * count):
            for field in range(3):
                assert_equal(mapped.unsafe_ptr()[unsafe_offset=_offset(record, field, rows * count, 3, column_major)],
                             original[record * 3 + field])
        for i in range(rows * count * 3, rows * count * 3 + 9):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i], UInt32(0xDEADBEEF))


def _finish_layout[threads: Int, items: Int, nodes: Int, plan: ArgmaxPlan[items, nodes],
                   column_major: Bool](ctx: DeviceContext) raises:
    var rows = 6
    var tile = threads * items
    for count in [1, threads - 1, threads, threads + 1, tile - 1, tile, tile + 1,
                  3 * tile, 3 * tile + 1]:
        comptime if column_major:
            _finish_case[threads, items, nodes, plan](ctx, count, True,
                col_major(rows * count, 3), col_major(rows, 3))
        else:
            _finish_case[threads, items, nodes, plan](ctx, count, False,
                row_major(rows * count, 3), row_major(rows, 3))


def _finish_pair[threads: Int, items: Int, nodes: Int, plan: ArgmaxPlan[items, nodes]](
    ctx: DeviceContext,
) raises:
    _finish_layout[threads, items, nodes, plan, False](ctx)
    _finish_layout[threads, items, nodes, plan, True](ctx)


def checked_runtime_finish(ctx: DeviceContext) raises:
    _finish_pair[32, 1, 3, sequential_argmax_plan[1]()](ctx)
    _finish_pair[32, 1, 3, balanced_argmax_plan[1]()](ctx)
    _finish_pair[32, 3, 7, sequential_argmax_plan[3]()](ctx)
    _finish_pair[32, 3, 7, balanced_argmax_plan[3]()](ctx)
    _finish_pair[32, 3, 5, _compact_plan[3]()](ctx)
    _finish_pair[64, 5, 11, sequential_argmax_plan[5]()](ctx)
    _finish_pair[64, 5, 11, balanced_argmax_plan[5]()](ctx)
    _finish_pair[64, 5, 9, _compact_plan[5]()](ctx)
    _finish_pair[128, 3, 7, balanced_argmax_plan[3]()](ctx)
    _finish_pair[256, 1, 3, sequential_argmax_plan[1]()](ctx)
    # First and final passes have independent geometry and plans. The longer
    # case creates 322 partials for a 320-record finisher tile.
    for count in [97, 96 * 321 + 1]:
        var groups = (count + 95) // 96
        _schedule_case[32, 3, True, finish_threads=64, finish_items=5, finish_balanced=True](
            ctx, count, False, row_major(8, count), row_major(8 * groups, 3), row_major(8, 3))
        _schedule_case[32, 3, True, finish_threads=64, finish_items=5, finish_balanced=True](
            ctx, count, True, col_major(8, count), col_major(8 * groups, 3), col_major(8, 3))
    # Public dispatch must also pass through a caller's nondefault node count.
    _schedule_case[32, 3, True, finish_threads=64, finish_items=5, finish_compact=True](
        ctx, 97, False, row_major(8, 97), row_major(16, 3), row_major(8, 3))
    _schedule_case[32, 3, True, finish_threads=64, finish_items=5, finish_compact=True](
        ctx, 97, True, col_major(8, 97), col_major(16, 3), col_major(8, 3))


def rejected_extents_preserve_storage(ctx: DeviceContext) raises:
    # Layout declarations exercise host preflight using tiny backing buffers.
    # A rejected extent must never reach a device launch or change scratch.
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](65)
    var partials = ctx.enqueue_create_buffer[DType.uint32](6)
    var result = ctx.enqueue_create_buffer[DType.uint32](3)
    partials.enqueue_fill(0xDEADBEEF)
    result.enqueue_fill(0xDEADBEEF)
    with assert_raises(contains="extents exceed representable counts"):
        enqueue_argmax(ctx, TileTensor(logits, row_major(1, 2147483648)),
            TileTensor(partials, row_major(2, 3)), TileTensor(result, row_major(1, 3)))
    with assert_raises(contains="extents exceed representable counts"):
        enqueue_argmax(ctx, TileTensor(logits, row_major(2147483648, 1)),
            TileTensor(partials, row_major(2, 3)), TileTensor(result, row_major(2147483648, 3)))
    # Two records suffice for the old/default tile, but this schedule needs
    # three. Scratch validation must use the selected launch geometry.
    with assert_raises(contains="invalid token selection scratch geometry"):
        enqueue_argmax[threads=32, items_per_thread=1](ctx, TileTensor(logits, row_major(1, 65)),
            TileTensor(partials, row_major(2, 3)), TileTensor(result, row_major(1, 3)))
    with partials.map_to_host() as mapped:
        for i in range(6):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i], UInt32(0xDEADBEEF))
    with result.map_to_host() as mapped:
        for i in range(3):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i], UInt32(0xDEADBEEF))


def main() raises:
    var ctx = DeviceContext()
    if ctx.api() != "metal":
        raise Error("requires Metal")
    print("device:",ctx.name(),"api:",ctx.api())
    assert_equal(bf16_rank(UInt16(0)),bf16_rank(UInt16(0x8000)))
    for count in [1,31,64,65,1023,1024,1025,65536,151936]:
        for pattern in range(9):
            reduction(ctx,count,pattern)
    rounded_tie(ctx)
    independent_rows(ctx)
    checked_schedules(ctx)
    checked_runtime_finish(ctx)
    rejected_extents_preserve_storage(ctx)
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](65)
    var partials = ctx.enqueue_create_buffer[DType.uint32](6)
    var result = ctx.enqueue_create_buffer[DType.uint32](3)
    with assert_raises():
        enqueue_argmax(ctx,TileTensor(logits,row_major(1,65)),TileTensor(partials,row_major(0,3)),TileTensor(result,row_major(1,3)))
    with assert_raises():
        enqueue_argmax(ctx,TileTensor(logits,row_major(1,0)),TileTensor(partials,row_major(2,3)),TileTensor(result,row_major(1,3)))
    print("Token selection ordering, rounded ties, checked schedules, runtime folds and rejection tests passed")
