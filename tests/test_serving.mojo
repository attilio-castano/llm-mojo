"""Step format rules and KV pool layout, isolation and logical lengths."""
from std.gpu import global_idx
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceContext
from llm_mojo.layers.attention_sublayer import AttentionCache
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVGeometry, KVPool

comptime VOCABULARY = 151936


def _assert_list(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(expected)):
        assert_equal(actual[i], expected[i])


def _mixed() -> StepBatch:
    """Two decode sequences and one three-token chunk; block size 4, six blocks."""
    return StepBatch([10, 11, 12, 13, 14], [2, 4, 3, 4, 5], [0, 1, 2, 5], 2,
        [3, 5, 6], 2, [0, 5, 1, 2, 3, 4], [2, 8, 15, 16, 17], [0, 1, 4])


def test_sequence_describes_one_block() raises:
    var chunk = StepBatch.sequence([7, 8, 9], 5, 2, 16)
    _assert_list(chunk.token_ids, [7, 8, 9])
    _assert_list(chunk.positions, [5, 6, 7])
    _assert_list(chunk.query_start, [0, 3])
    assert_equal(chunk.decode_count, 0)
    _assert_list(chunk.seq_lens, [8])
    assert_equal(chunk.max_blocks, 1)
    _assert_list(chunk.block_table, [2])
    _assert_list(chunk.slot_mapping, [37, 38, 39])
    _assert_list(chunk.logits_rows, [2])
    assert_equal(chunk.rows(), 3)
    assert_equal(chunk.sequences(), 1)
    chunk.validate(3, 16, VOCABULARY)
    var decode = StepBatch.sequence([4], 0, 0, 8)
    assert_equal(decode.decode_count, 1)
    decode.validate(1, 8, VOCABULARY)


def test_sequence_rejects_steps_outside_one_block() raises:
    with assert_raises():
        _ = StepBatch.sequence(List[Int](), 0, 0, 8)
    with assert_raises():
        _ = StepBatch.sequence([1], -1, 0, 8)
    with assert_raises():
        _ = StepBatch.sequence([1], 0, -1, 8)
    with assert_raises():
        _ = StepBatch.sequence([1, 2], 7, 0, 8)
    with assert_raises():
        StepBatch.sequence([1], 0, 3, 8).validate(3, 8, VOCABULARY)


def test_mixed_batch_validates() raises:
    _mixed().validate(6, 4, VOCABULARY)


def test_validate_rejects_each_rule() raises:
    var batch = _mixed()
    batch.query_start[0] = 1
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.query_start[2] = 1
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.positions.append(6)
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.positions[3] = 5
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.slot_mapping[4] = 18
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.block_table[1] = 6
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.token_ids[0] = VOCABULARY
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.token_ids[0] = -1
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.decode_count = 1
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.logits_rows[2] = 3
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.logits_rows = [1, 0, 4]
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    batch = _mixed()
    batch.seq_lens[0] = 9
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    # The chunk's second block is the block the second decode sequence writes.
    batch = _mixed()
    batch.block_table[5] = 2
    batch.slot_mapping[3] = 8
    batch.slot_mapping[4] = 9
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    # One sequence may not map two of its logical blocks to one physical block.
    batch = _mixed()
    batch.block_table[2] = 2
    with assert_raises():
        batch.validate(6, 4, VOCABULARY)
    with assert_raises():
        _mixed().validate(6, 0, VOCABULARY)


def _mark[L: TensorLayout](view: TileTensor[DType.bfloat16, L, MutAnyOrigin]):
    comptime assert view.flat_rank == 2
    var i = global_idx.x
    if i < 16:
        view[i // 4, i % 4] = Scalar[DType.bfloat16](Float32(100 + i))


def test_pool_views_follow_block_major_layout() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    var pool = KVPool(ctx, 2, 4, KVGeometry(3, 2, 2))
    assert_true(pool.geometry == KVGeometry(3, 2, 2))
    assert_equal(len(pool.storage), 2 * 3 * 2 * 16)
    assert_equal(len(pool.caches), 6)
    for block in range(2):
        for layer in range(3):
            var view = pool.index(block, layer)
            assert_equal(view, block * 3 + layer)
            assert_equal(pool.key_offset(block, layer), 2 * view * 16)
            assert_equal(pool.caches[view].capacity, 4)
            assert_equal(len(pool.caches[view].key), 16)
            pool.caches[view].key.enqueue_fill(Scalar[DType.bfloat16](Float32(2 * view + 1)))
            pool.caches[view].value.enqueue_fill(Scalar[DType.bfloat16](Float32(2 * view + 2)))
    ctx.synchronize()
    with pool.storage.map_to_host() as mapped:
        for element in range(len(pool.storage)):
            var region = element // 16
            assert_equal(Int(mapped.unsafe_ptr()[unsafe_offset=element].cast[DType.float32]()), region + 1)
    # A kernel writing through one view changes exactly that view's range.
    var target = pool.index(1, 2)
    var tensor = TileTensor(pool.caches[target].value, row_major(4, 4))
    ctx.enqueue_function[_mark[type_of(tensor.layout)]](tensor, grid_dim=1, block_dim=32)
    ctx.synchronize()
    with pool.storage.map_to_host() as mapped:
        for element in range(len(pool.storage)):
            var region = element // 16
            var expected = region + 1
            if region == 2 * target + 1:
                expected = 100 + element % 16
            assert_equal(Int(mapped.unsafe_ptr()[unsafe_offset=element].cast[DType.float32]()), expected)


def test_pool_lengths_truncate_and_reset() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 4, KVGeometry(3, 2, 2))
    for layer in range(3):
        pool.caches[pool.index(1, layer)].length = 3
    assert_equal(pool.length(1), 3)
    assert_equal(pool.length(0), 0)
    pool.truncate(1, 2)
    for layer in range(3):
        assert_equal(pool.caches[pool.index(1, layer)].length, 2)
    with assert_raises():
        pool.truncate(1, 3)
    with assert_raises():
        pool.truncate(1, -1)
    # Rejection is atomic: one short layer blocks the whole truncation.
    pool.caches[pool.index(1, 2)].length = 1
    with assert_raises():
        pool.truncate(1, 2)
    assert_equal(pool.caches[pool.index(1, 0)].length, 2)
    assert_equal(pool.caches[pool.index(1, 2)].length, 1)
    pool.reset(ctx)
    for view in range(len(pool.caches)):
        assert_equal(pool.caches[view].length, 0)


def test_pool_rejects_invalid_geometry_and_views() raises:
    var ctx = DeviceContext()
    var unit = KVGeometry(1, 1, 1)
    with assert_raises():
        _ = KVPool(ctx, 0, 4, unit)
    with assert_raises():
        _ = KVPool(ctx, 1, 0, unit)
    with assert_raises():
        _ = KVPool(ctx, 1, 4097, unit)
    with assert_raises():
        _ = KVPool(ctx, 1, 4, KVGeometry(0, 1, 1))
    with assert_raises():
        _ = KVPool(ctx, 1, 4, KVGeometry(1, 0, 1))
    with assert_raises():
        _ = KVPool(ctx, 1, 4, KVGeometry(1, 1, 0))
    assert_true(KVGeometry(2, 1, 2) != KVGeometry(2, 2, 1))
    var pool = KVPool(ctx, 1, 2, KVGeometry(2, 1, 2))
    with assert_raises():
        _ = pool.index(1, 0)
    with assert_raises():
        _ = pool.index(0, 2)
    with assert_raises():
        _ = pool.length(-1)
    var storage = ctx.enqueue_create_buffer[DType.bfloat16](10)
    with assert_raises():
        _ = AttentionCache(storage.create_sub_buffer[DType.bfloat16](0, 5),
            storage.create_sub_buffer[DType.bfloat16](5, 5), 4, 1, 2)
    var adopted = AttentionCache(storage.create_sub_buffer[DType.bfloat16](0, 4),
        storage.create_sub_buffer[DType.bfloat16](4, 4), 2, 1, 2)
    assert_true(adopted.length == 0 and adopted.capacity == 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
