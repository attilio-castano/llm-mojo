"""Mixed-step address and layer checks on identical operation inputs.

Token-wise QKV scatter is compared with separate unpack and per-row RoPE,
using an independent pool-offset formula and poison in every unwritten slot.
A mixed layer is compared with the existing decode layer on each identical
input row, including several contiguous rows of one prefill sequence. This
checks composition and causal addressing, not full-model schedule invariance.
"""
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises
from llm_mojo.kernels.rope import enqueue_rope_apple_gpu
from llm_mojo.layers.attention_sublayer import (
    AttentionWeights, AttentionWorkspace, _unpack_qkv, enqueue_step_qkv_paged,
)
from llm_mojo.layers.decoder_layer import enqueue_decode_batch_layer, enqueue_mixed_layer, validate_mixed_layer
from llm_mojo.layers.mlp import MLPWeights, MLPWorkspace

comptime POISON = UInt16(0x7FC1)


def _fill(mut buffer: DeviceBuffer[DType.bfloat16], seed: Int) raises:
    with buffer.map_to_host() as mapped:
        for i in range(len(buffer)):
            var raw = UInt32((i * 1664525 + seed * 1013904223) & 0xffffffff)
            var bits = UInt16((raw >> 16) & 0x807f) | UInt16((119 + Int((raw >> 7) % 4)) << 7)
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bits)


def _ints(ctx: DeviceContext, values: List[Int]) raises -> DeviceBuffer[DType.int32]:
    var buffer = ctx.enqueue_create_buffer[DType.int32](len(values))
    with buffer.map_to_host() as mapped:
        for i in range(len(values)):
            mapped.unsafe_ptr()[unsafe_offset=i] = Int32(values[i])
    return buffer^


def _same(expected: DeviceBuffer[DType.bfloat16], actual: DeviceBuffer[DType.bfloat16], count: Int,
          label: String) raises:
    with expected.map_to_host() as a:
        with actual.map_to_host() as b:
            for i in range(count):
                if (bitcast[DType.uint16](a.unsafe_ptr()[unsafe_offset=i])
                        != bitcast[DType.uint16](b.unsafe_ptr()[unsafe_offset=i])):
                    raise Error(label + " differs at element " + String(i))


def _offset(block: Int, kv: Int, slot: Int, head: Int, size: Int, head_major: Bool) -> Int:
    # Two layers, writing only layer 1. Independent of kernels/paged_kv.mojo.
    var region = ((block * 2 + 1) * 2 + kv) * size * 2 * 64
    return region + (head * size + slot) * 64 if head_major else region + (slot * 2 + head) * 64


def _scatter_equals_unfused[HEAD_MAJOR: Bool](ctx: DeviceContext, size: Int) raises:
    comptime ROWS = 7
    comptime QUERY = 4 * 64
    comptime WIDTH = 2 * 64
    var work = AttentionWorkspace(ctx, ROWS + 1, 129, 4, 2, 64, False, False)
    _fill(work.packed, 31)
    _fill(work.cosine, 37)
    _fill(work.sine, 41)
    work.query.enqueue_fill(bitcast[DType.bfloat16](POISON))
    var positions = _ints(ctx, [0, 31, 32, 61, 62, 63, 64])
    var slots: List[Int] = [3 * size, 5 * size + 31, size, 2 * size + 29, 2 * size + 30, 2 * size + 31, 4 * size]
    var slot_mapping = _ints(ctx, slots)
    var pool = ctx.enqueue_create_buffer[DType.bfloat16](6 * 2 * 2 * size * WIDTH)
    var expected_pool = ctx.enqueue_create_buffer[DType.bfloat16](len(pool))
    pool.enqueue_fill(bitcast[DType.bfloat16](POISON))
    expected_pool.enqueue_fill(bitcast[DType.bfloat16](POISON))
    var expected_query = ctx.enqueue_create_buffer[DType.bfloat16]((ROWS + 1) * QUERY)
    expected_query.enqueue_fill(bitcast[DType.bfloat16](POISON))
    var raw_query = ctx.enqueue_create_buffer[DType.bfloat16](ROWS * QUERY)
    var raw_key = ctx.enqueue_create_buffer[DType.bfloat16](ROWS * WIDTH)
    var raw_value = ctx.enqueue_create_buffer[DType.bfloat16](ROWS * WIDTH)
    var rotated_key = ctx.enqueue_create_buffer[DType.bfloat16](ROWS * WIDTH)
    var packed = TileTensor(work.packed, row_major(ROWS, QUERY + 2 * WIDTH))
    var q = TileTensor(raw_query, row_major(ROWS, QUERY))
    var k = TileTensor(raw_key, row_major(ROWS, WIDTH))
    var v = TileTensor(raw_value, row_major(ROWS, WIDTH))
    comptime unpack = _unpack_qkv[type_of(packed.layout), type_of(q.layout), type_of(k.layout)]
    ctx.enqueue_function[unpack](packed, q, k, v, Int32(ROWS), Int32(QUERY), Int32(WIDTH),
        grid_dim=ceildiv(ROWS * (QUERY + 2 * WIDTH), 128), block_dim=128)
    var host_positions: List[Int] = [0, 31, 32, 61, 62, 63, 64]
    for r in range(ROWS):
        enqueue_rope_apple_gpu(ctx,
            TileTensor(raw_query.unsafe_ptr().unsafe_offset(r * QUERY), row_major(1, 4, 64)),
            TileTensor(work.cosine, row_major(129, 64)), TileTensor(work.sine, row_major(129, 64)),
            TileTensor(expected_query.unsafe_ptr().unsafe_offset(r * QUERY), row_major(1, 4, 64)), host_positions[r])
        enqueue_rope_apple_gpu(ctx,
            TileTensor(raw_key.unsafe_ptr().unsafe_offset(r * WIDTH), row_major(1, 2, 64)),
            TileTensor(work.cosine, row_major(129, 64)), TileTensor(work.sine, row_major(129, 64)),
            TileTensor(rotated_key.unsafe_ptr().unsafe_offset(r * WIDTH), row_major(1, 2, 64)), host_positions[r])
    with expected_pool.map_to_host() as dst:
        with rotated_key.map_to_host() as key:
            with raw_value.map_to_host() as value:
                for r in range(ROWS):
                    for head in range(2):
                        var ko = _offset(slots[r] // size, 0, slots[r] % size, head, size, HEAD_MAJOR)
                        var vo = _offset(slots[r] // size, 1, slots[r] % size, head, size, HEAD_MAJOR)
                        for d in range(64):
                            dst.unsafe_ptr()[unsafe_offset=ko + d] = key.unsafe_ptr()[unsafe_offset=r * WIDTH + head * 64 + d]
                            dst.unsafe_ptr()[unsafe_offset=vo + d] = value.unsafe_ptr()[unsafe_offset=r * WIDTH + head * 64 + d]
    enqueue_step_qkv_paged[4, 2, 64, HEAD_MAJOR](ctx, work, pool,
        TileTensor(positions, row_major(ROWS)), TileTensor(slot_mapping, row_major(ROWS)), 1, 2, size)
    _same(expected_query, work.query, len(expected_query), "step rotary query including guard")
    _same(expected_pool, pool, len(pool), "step scatter including all unwritten slots")


def test_step_qkv_matches_unfused_rows_and_writes_only_slots() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    for size in [32, 64, 128]:
        _scatter_equals_unfused[False](ctx, size)
        _scatter_equals_unfused[True](ctx, size)


def _mixed_equals_decode_rows[HEAD_MAJOR: Bool](ctx: DeviceContext) raises:
    comptime ROWS = 5
    comptime HIDDEN = 896
    comptime SIZE = 32
    var aw = AttentionWeights(ctx)
    var mw = MLPWeights(ctx)
    _fill(aw.norm, 11)
    _fill(aw.qkv, 13)
    _fill(aw.bias, 17)
    _fill(aw.output, 19)
    _fill(mw.norm, 23)
    _fill(mw.gate, 29)
    _fill(mw.up, 31)
    _fill(mw.down, 37)
    var work = AttentionWorkspace(ctx, ROWS, 96, materialized=False, fp32_materialized=False)
    var mlp = MLPWorkspace(ctx, ROWS)
    var solo_work = AttentionWorkspace(ctx, 1, 96, materialized=False, fp32_materialized=False)
    var solo_mlp = MLPWorkspace(ctx, 1)
    _fill(work.cosine, 41)
    _fill(work.sine, 43)
    ctx.enqueue_copy(solo_work.cosine, work.cosine)
    ctx.enqueue_copy(solo_work.sine, work.sine)
    var x = ctx.enqueue_create_buffer[DType.bfloat16](ROWS * HIDDEN)
    _fill(x, 47)
    var solo_x = ctx.enqueue_create_buffer[DType.bfloat16](HIDDEN)
    var pool = ctx.enqueue_create_buffer[DType.bfloat16](8 * 2 * 2 * SIZE * 128)
    var solo_pool = ctx.enqueue_create_buffer[DType.bfloat16](len(pool))
    _fill(pool, 53)
    ctx.enqueue_copy(solo_pool, pool)
    var positions = _ints(ctx, [31, 63, 30, 31, 32])
    var tables = _ints(ctx, [5, 1, 0, 3, 6, 0, 2, 4, 7])
    var slots = _ints(ctx, [5 * SIZE + 31, 6 * SIZE + 31, 2 * SIZE + 30, 2 * SIZE + 31, 4 * SIZE])
    var expected_residual = ctx.enqueue_create_buffer[DType.bfloat16](ROWS * HIDDEN)
    var expected_down = ctx.enqueue_create_buffer[DType.bfloat16](ROWS * HIDDEN)
    var launches = enqueue_mixed_layer[14, 2, 64, 8, HEAD_MAJOR](ctx, aw, work, mw, mlp, x, pool,
        TileTensor(positions, row_major(ROWS)), TileTensor(slots, row_major(ROWS)), TileTensor(tables, row_major(3, 3)),
        2, 1, 2, SIZE, False)
    assert_equal(launches, 11)
    for r in range(ROWS):
        var source = x.create_sub_buffer[DType.bfloat16](r * HIDDEN, HIDDEN)
        ctx.enqueue_copy(solo_x, source)
        var seq = r if r < 2 else 2
        _ = enqueue_decode_batch_layer[14, 2, 64, 8, HEAD_MAJOR](ctx, aw, solo_work, mw, solo_mlp, solo_x, solo_pool,
            TileTensor(positions.unsafe_ptr().unsafe_offset(r), row_major(1)),
            TileTensor(tables.unsafe_ptr().unsafe_offset(seq * 3), row_major(1, 3)), 1, 2, SIZE, False)
        var residual = expected_residual.create_sub_buffer[DType.bfloat16](r * HIDDEN, HIDDEN)
        var down = expected_down.create_sub_buffer[DType.bfloat16](r * HIDDEN, HIDDEN)
        ctx.enqueue_copy(residual, solo_work.output)
        ctx.enqueue_copy(down, solo_mlp.down)
    _same(expected_residual, work.output, ROWS * HIDDEN, "mixed attention residual")
    _same(expected_down, mlp.down, ROWS * HIDDEN, "mixed MLP down")
    _same(solo_pool, pool, len(pool), "mixed and per-row layer KV")
    # Two multi-row sequences and a trailing singleton shape are not admitted.
    with assert_raises():
        validate_mixed_layer[14, 2, 64](ctx, aw, work, mw, mlp, x, pool,
            TileTensor(positions, row_major(ROWS)), TileTensor(slots, row_major(ROWS)),
            TileTensor(tables, row_major(2, 3)), 0, 1, 2, SIZE)
    with assert_raises():
        validate_mixed_layer[14, 2, 64](ctx, aw, work, mw, mlp, x, pool,
            TileTensor(positions, row_major(ROWS)), TileTensor(slots, row_major(ROWS)),
            TileTensor(tables, row_major(5, 1)), 4, 1, 2, SIZE)


def test_mixed_layer_equals_existing_decode_on_identical_rows() raises:
    var ctx = DeviceContext()
    _mixed_equals_decode_rows[False](ctx)
    _mixed_equals_decode_rows[True](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
