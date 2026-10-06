"""Step format rules and KV pool layout, isolation and logical lengths."""
from std.gpu import global_idx
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceContext
from llm_mojo.layers.attention_sublayer import AttentionCache
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.blocks import BlockManager
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
    var chunk = StepBatch.sequence([7, 8, 9], 5, [2], 16)
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
    var decode = StepBatch.sequence([4], 0, [0], 8)
    assert_equal(decode.decode_count, 1)
    decode.validate(1, 8, VOCABULARY)


def test_sequence_rejects_steps_outside_one_block() raises:
    with assert_raises():
        _ = StepBatch.sequence(List[Int](), 0, [0], 8)
    with assert_raises():
        _ = StepBatch.sequence([1], -1, [0], 8)
    with assert_raises():
        _ = StepBatch.sequence([1], 0, [-1], 8)
    with assert_raises():
        _ = StepBatch.sequence([1, 2], 7, [0], 8)
    with assert_raises():
        StepBatch.sequence([1], 0, [3], 8).validate(3, 8, VOCABULARY)


def test_sequence_spans_its_table() raises:
    # Positions 6..9 in blocks of four: slots 2 and 3 of block 2, then 0 and 1 of block 9.
    var chunk = StepBatch.sequence([7, 8, 9, 10], 6, [5, 2, 9], 4)
    _assert_list(chunk.positions, [6, 7, 8, 9])
    _assert_list(chunk.slot_mapping, [10, 11, 36, 37])
    _assert_list(chunk.block_table, [5, 2, 9])
    _assert_list(chunk.seq_lens, [10])
    assert_equal(chunk.max_blocks, 3)
    chunk.validate(10, 4, VOCABULARY)
    with assert_raises():
        _ = StepBatch.sequence([1, 2], 11, [5, 2, 9], 4)
    with assert_raises():
        _ = StepBatch.sequence([1], 0, List[Int](), 4)
    # A table naming one block twice is rejected by validation.
    with assert_raises():
        StepBatch.sequence([1], 4, [3, 3], 4).validate(10, 4, VOCABULARY)


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
    assert_equal(pool.region(), 16)
    assert_equal(len(pool.storage), 2 * 3 * 2 * 16)
    for block in range(2):
        for layer in range(3):
            var index = block * 3 + layer
            assert_equal(pool.key_offset(block, layer), 2 * index * 16)
            var key = pool.view(block, layer, 0)
            var value = pool.view(block, layer, 1)
            assert_equal(len(key), 16)
            assert_equal(len(value), 16)
            key.enqueue_fill(Scalar[DType.bfloat16](Float32(2 * index + 1)))
            value.enqueue_fill(Scalar[DType.bfloat16](Float32(2 * index + 2)))
    ctx.synchronize()
    with pool.storage.map_to_host() as mapped:
        for element in range(len(pool.storage)):
            var region = element // 16
            assert_equal(Int(mapped.unsafe_ptr()[unsafe_offset=element].cast[DType.float32]()), region + 1)
    # A kernel writing through one view changes exactly that view's range.
    var target = pool.view(1, 2, 1)
    var tensor = TileTensor(target, row_major(4, 4))
    ctx.enqueue_function[_mark[type_of(tensor.layout)]](tensor, grid_dim=1, block_dim=32)
    ctx.synchronize()
    with pool.storage.map_to_host() as mapped:
        for element in range(len(pool.storage)):
            var region = element // 16
            var expected = region + 1
            if region == 2 * (1 * 3 + 2) + 1:
                expected = 100 + element % 16
            assert_equal(Int(mapped.unsafe_ptr()[unsafe_offset=element].cast[DType.float32]()), expected)
    # Head-major order rearranges rows inside a region, not the regions.
    var heads = KVPool(ctx, 2, 4, KVGeometry(3, 2, 2), True)
    assert_true(heads.head_major and not pool.head_major)
    assert_equal(len(heads.storage), len(pool.storage))
    assert_equal(heads.key_offset(1, 2), pool.key_offset(1, 2))


def test_pool_written_slots_truncate_and_reset() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 4, KVGeometry(3, 2, 2))
    assert_equal(pool.length(0), 0)
    assert_equal(pool.length(1), 0)
    pool.written[1] = 3
    assert_equal(pool.length(1), 3)
    pool.truncate(1, 2)
    assert_equal(pool.length(1), 2)
    # Truncation cannot extend a block, go below zero or name a block outside the pool.
    with assert_raises():
        pool.truncate(1, 3)
    with assert_raises():
        pool.truncate(1, -1)
    with assert_raises():
        pool.truncate(2, 0)
    assert_equal(pool.length(1), 2)
    pool.reset(ctx)
    assert_equal(pool.length(0), 0)
    assert_equal(pool.length(1), 0)


def test_pool_relayout_keeps_storage_and_empties_blocks() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 2, 64, KVGeometry(3, 2, 2))
    var storage = len(pool.storage)
    pool.written[1] = 5
    pool.relayout(ctx, 32, True)
    assert_equal(pool.blocks, 4)
    assert_equal(pool.block_size, 32)
    assert_true(pool.head_major)
    assert_equal(len(pool.storage), storage)
    for block in range(4):
        assert_equal(pool.length(block), 0)
    # Offsets follow the new blocks.
    assert_equal(pool.region(), 32 * 2 * 2)
    assert_equal(pool.key_offset(3, 1), 2 * (3 * 3 + 1) * pool.region())
    pool.written[2] = 7
    pool.relayout(ctx, 128, False)
    assert_equal(pool.blocks, 1)
    assert_equal(pool.length(0), 0)
    assert_true(not pool.head_major)
    # The slots must split into whole blocks of a valid size; rejection changes nothing.
    pool.written[0] = 9
    for size in [0, 48, 256, 4097]:
        with assert_raises():
            pool.relayout(ctx, size, True)
    assert_equal(pool.blocks, 1)
    assert_equal(pool.block_size, 128)
    assert_true(not pool.head_major)
    assert_equal(pool.length(0), 9)


def test_pool_follows_the_block_manager() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 6, 4, KVGeometry(2, 2, 2))
    var blocks = BlockManager(6, 4, 24)
    var a = blocks.add()
    var b = blocks.add()
    # a holds six positions in blocks 0 and 1, and b three in block 2; the model advances the pool.
    blocks.reserve(a, 6)
    blocks.reserve(b, 3)
    pool.written[0] = 4
    pool.written[1] = 2
    pool.written[2] = 3
    blocks.commit(a, 6)
    blocks.commit(b, 3)
    _assert_list(blocks.table(a), [0, 1])
    _assert_list(blocks.table(b), [2])
    blocks.check_pool(pool)
    # A pool that disagrees with a committed length fails the check.
    pool.written[1] = 1
    with assert_raises():
        blocks.check_pool(pool)
    pool.written[1] = 2
    # Truncation empties the blocks past the new length in the pool, then frees them.
    pool.truncate_table(blocks.table(a), 3)
    blocks.truncate(a, 3)
    assert_equal(pool.length(0), 3)
    assert_equal(pool.length(1), 0)
    blocks.check_pool(pool)
    # It cannot extend a sequence, and a rejected truncation changes nothing.
    with assert_raises():
        pool.truncate_table(blocks.table(a), 5)
    assert_equal(pool.length(0), 3)
    # Releasing a sequence empties its blocks.
    pool.truncate_table(blocks.table(b), 0)
    blocks.release(b)
    blocks.check_pool(pool)
    # A pool of another geometry fails the check.
    with assert_raises():
        blocks.check_pool(KVPool(ctx, 6, 8, KVGeometry(2, 2, 2)))


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
        _ = pool.key_offset(1, 0)
    with assert_raises():
        _ = pool.view(0, 2, 0)
    with assert_raises():
        _ = pool.view(0, 0, 2)
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
