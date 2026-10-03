"""Paged KV: every paged launch equals its contiguous form byte for byte.

Each check lays sequences' K and V rows into a poisoned block-major pool through
scattered tables, using an address formula written out here independently of
kernels/paged_kv.mojo, and compares the paged launch with its contiguous form
on the same logical rows:
- G32 attention (route 4's arithmetic) against route 4 for batched decode rows,
  and against route 11 for one sequence's query rows;
- the FP32 rolled-MMA prefill against routes 6 and 10;
- fused decode QKV/RoPE/append against the unfused unpack, RoPE and copy, and
  the prefill append against the contiguous append, with every other pool
  element keeping its poison.
Block sizes 32, 64 and 128 and one block per sequence, in slot-major and
head-major order. Unwritten slots and unused blocks hold a NaN poison, so a
stray read changes an output and a stray write changes the pool.
"""
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises
from llm_mojo.kernels.attention_decode import (
    enqueue_grouped_query_attention_consistent_apple_gpu,
    enqueue_grouped_query_attention_decode_apple_gpu,
    enqueue_paged_attention_g32_apple_gpu,
)
from llm_mojo.kernels.attention_prefill import (
    enqueue_grouped_query_attention_prefill_apple_gpu,
    enqueue_grouped_query_attention_prefill_split_apple_gpu,
    enqueue_paged_attention_prefill_apple_gpu,
    enqueue_paged_attention_prefill_split_apple_gpu,
)
from llm_mojo.kernels.paged_kv import kv_row
from llm_mojo.kernels.rope import enqueue_rope_apple_gpu
from llm_mojo.layers.attention_sublayer import (
    AttentionWorkspace, _append, _unpack_qkv, enqueue_append_paged, enqueue_fused_decode_qkv_paged,
)
from llm_mojo.serving.kv_pool import KVGeometry, KVPool

comptime POISON = UInt16(0x7FC1)
# Rows live in layer 1 of 2, so every address crosses a layer.
comptime LAYERS = 2
comptime LAYER = 1
comptime WIDTH = 128
comptime ONE_BLOCK = 4096


def _fill(mut buffer: DeviceBuffer[DType.bfloat16], seed: Int, low: Int = 119, span: Int = 8) raises:
    # Mixed signs, mantissas and exponents 2^(low-127) .. 2^(low+span-128).
    with buffer.map_to_host() as mapped:
        for i in range(len(buffer)):
            var raw = UInt32((i*1664525+seed*1013904223) & 0xffffffff)
            var bits = UInt16((raw >> 16) & 0x807f) | UInt16((low+Int((raw >> 7)%UInt32(span))) << 7)
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bits)


def _poison(mut buffer: DeviceBuffer[DType.bfloat16]) raises:
    buffer.enqueue_fill(bitcast[DType.bfloat16](POISON))


def _same(mut expected: DeviceBuffer[DType.bfloat16], mut actual: DeviceBuffer[DType.bfloat16],
          label: String) raises:
    with expected.map_to_host() as a:
        with actual.map_to_host() as b:
            for i in range(len(expected)):
                if (bitcast[DType.uint16](a.unsafe_ptr()[unsafe_offset=i])
                        != bitcast[DType.uint16](b.unsafe_ptr()[unsafe_offset=i])):
                    raise Error(label + " differs at element " + String(i))


def _offset(block: Int, kv: Int, slot: Int, head: Int, size: Int, head_major: Bool) -> Int:
    """Pool[block, 1, kv, slot, head, 0] slot-major, or Pool[block, 1, kv, head, slot, 0] head-major, of 2 layers."""
    var region = ((block * LAYERS + LAYER) * 2 + kv) * size * WIDTH
    if head_major:
        return region + head * size * 64 + slot * 64
    return region + slot * WIDTH + head * 64


def _label(size: Int, head_major: Bool) -> String:
    return String(size) + ("-slot head-major" if head_major else "-slot slot-major")


struct Pages(Movable):
    """Scattered tables: row s lists sequence s's blocks in position order, then a poisoned spare block."""
    var size: Int
    var width: Int
    var blocks: Int
    var tables: List[Int]

    def __init__(out self, lengths: List[Int], size: Int, seed: Int):
        self.size = size
        self.width = 1
        var needed = 0
        for length in lengths:
            needed += ceildiv(length, size)
            self.width = max(self.width, ceildiv(length, size))
        # A quarter more blocks than the sequences hold; the extra ones are never written.
        self.blocks = needed + needed // 4 + 2
        var order = List[Int](capacity=self.blocks)
        for b in range(self.blocks):
            order.append(b)
        var state = UInt64(seed) * 2654435761 + 1
        for i in range(self.blocks - 1, 0, -1):
            state = state * 6364136223846793005 + 1442695040888963407
            var j = Int((state >> 33) % UInt64(i + 1))
            var swap = order[i]
            order[i] = order[j]
            order[j] = swap
        var spare = order[self.blocks - 1]
        self.tables = List[Int](capacity=len(lengths) * self.width)
        var next = 0
        for length in lengths:
            for b in range(self.width):
                if b < ceildiv(length, size):
                    self.tables.append(order[next])
                    next += 1
                else:
                    self.tables.append(spare)

    def elements(self) -> Int:
        return self.blocks * LAYERS * 2 * self.size * WIDTH

    def block(self, sequence: Int, position: Int) -> Int:
        return self.tables[sequence * self.width + position // self.size]


def _tables(ctx: DeviceContext, pages: Pages, sequences: Int) raises -> DeviceBuffer[DType.int32]:
    var buffer = ctx.enqueue_create_buffer[DType.int32](sequences * pages.width)
    with buffer.map_to_host() as mapped:
        for i in range(sequences * pages.width):
            mapped.unsafe_ptr()[unsafe_offset=i] = Int32(pages.tables[i])
    return buffer^


def _positions(ctx: DeviceContext, values: List[Int]) raises -> DeviceBuffer[DType.int32]:
    var buffer = ctx.enqueue_create_buffer[DType.int32](len(values))
    with buffer.map_to_host() as mapped:
        for i in range(len(values)):
            mapped.unsafe_ptr()[unsafe_offset=i] = Int32(values[i])
    return buffer^


def _lay(mut storage: DeviceBuffer[DType.bfloat16], mut keys: DeviceBuffer[DType.bfloat16],
         mut values: DeviceBuffer[DType.bfloat16], starts: List[Int], lengths: List[Int], pages: Pages,
         head_major: Bool) raises:
    """Copy each sequence's contiguous rows [length, 2, 64], from starts[s], into its blocks."""
    with storage.map_to_host() as pool:
        with keys.map_to_host() as k:
            with values.map_to_host() as v:
                for s in range(len(lengths)):
                    for t in range(lengths[s]):
                        for head in range(2):
                            var key = _offset(pages.block(s, t), 0, t % pages.size, head, pages.size, head_major)
                            var value = _offset(pages.block(s, t), 1, t % pages.size, head, pages.size, head_major)
                            var source = (starts[s] + t) * WIDTH + head * 64
                            for d in range(64):
                                pool.unsafe_ptr()[unsafe_offset=key + d] = k.unsafe_ptr()[unsafe_offset=source + d]
                                pool.unsafe_ptr()[unsafe_offset=value + d] = v.unsafe_ptr()[unsafe_offset=source + d]


def _written(mut storage: DeviceBuffer[DType.bfloat16]) raises -> Int:
    """Pool elements that no longer hold the poison."""
    var count = 0
    with storage.map_to_host() as pool:
        for i in range(len(storage)):
            if bitcast[DType.uint16](pool.unsafe_ptr()[unsafe_offset=i]) != POISON:
                count += 1
    return count


def _decode_lengths() -> List[Int]:
    """Keys per sequence: around one, two and four SIMD-group rounds and block edges, and a full context."""
    var lengths: List[Int] = [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 1024, 4095, 4096]
    for s in range(len(lengths), 32):
        lengths.append((s * 977) % 4096 + 1)
    return lengths^


def _chunks() -> List[Int]:
    """Prefill chunk rows: one, around the 16-row policy boundary, and whole query tiles."""
    return [1, 15, 16, 17, 64, 255, 256]


def _prefixes(size: Int, rows: Int) -> List[Int]:
    """Cached rows before a chunk: ending at a block edge, inside a block, and filling the context."""
    if size == ONE_BLOCK:
        return [0, 1000, 4096 - rows]
    return [2 * size, 2 * size + 5, 4096 - rows]


def test_address_function_follows_both_documented_layouts() raises:
    for size in [1, 32, 64, 4096]:
        for block in [0, 1, 7]:
            for layer in range(3):
                for kv in range(2):
                    for slot in [0, size // 2, size - 1]:
                        for head in range(2):
                            var region = ((block * 3 + layer) * 2 + kv) * size * 128
                            assert_equal(kv_row[2, 64, False](block, layer, 3, kv, slot, head, size),
                                         region + slot * 128 + head * 64)
                            assert_equal(kv_row[2, 64, True](block, layer, 3, kv, slot, head, size),
                                         region + head * size * 64 + slot * 64)
    # Slot-major K rows start where serving phase 1's per-layer views do.
    var ctx = DeviceContext()
    var pool = KVPool(ctx, 3, 8, KVGeometry(2, 2, 64))
    for block in range(3):
        for layer in range(2):
            assert_equal(kv_row[2, 64, False](block, layer, 2, 0, 0, 0, 8), pool.key_offset(block, layer))


def _g32_decode[HEAD_MAJOR: Bool](ctx: DeviceContext, size: Int) raises:
    var lengths = _decode_lengths()
    var starts = List[Int](capacity=len(lengths))
    var total = 0
    for length in lengths:
        starts.append(total)
        total += length
    var keys = ctx.enqueue_create_buffer[DType.bfloat16](total * WIDTH)
    var values = ctx.enqueue_create_buffer[DType.bfloat16](total * WIDTH)
    _fill(keys, 31)
    _fill(values, 32)
    var pages = Pages(lengths, size, 7)
    var storage = ctx.enqueue_create_buffer[DType.bfloat16](pages.elements())
    _poison(storage)
    _lay(storage, keys, values, starts, lengths, pages, HEAD_MAJOR)
    var pool = TileTensor(storage, row_major(len(storage)))
    var query = ctx.enqueue_create_buffer[DType.bfloat16](32 * 896)
    _fill(query, 33)
    var split = ctx.enqueue_create_buffer[DType.float32](14 * 66)
    for sequences in [1, 2, 3, 8, 16, 32]:
        var ends = List[Int](capacity=sequences)
        for s in range(sequences):
            ends.append(lengths[s] - 1)
        var positions = _positions(ctx, ends)
        var tables = _tables(ctx, pages, sequences)
        var paged = ctx.enqueue_create_buffer[DType.bfloat16]((sequences + 2) * 896)
        var solo = ctx.enqueue_create_buffer[DType.bfloat16]((sequences + 2) * 896)
        _poison(paged)
        _poison(solo)
        enqueue_paged_attention_g32_apple_gpu[14, 2, 64, HEAD_MAJOR](ctx,
            TileTensor(query, row_major(sequences, 14, 64)), pool,
            TileTensor(paged.unsafe_ptr().unsafe_offset(896), row_major(sequences, 14, 64)),
            TileTensor(positions, row_major(sequences)),
            TileTensor(tables, row_major(sequences, pages.width)), 1, LAYER, LAYERS, size)
        for s in range(sequences):
            enqueue_grouped_query_attention_decode_apple_gpu[32, 1, 1, fp32_scores=True](ctx,
                TileTensor(query.unsafe_ptr().unsafe_offset(s * 896), row_major(1, 14, 64)),
                TileTensor(keys.unsafe_ptr().unsafe_offset(starts[s] * WIDTH), row_major(lengths[s], 2, 64)),
                TileTensor(values.unsafe_ptr().unsafe_offset(starts[s] * WIDTH), row_major(lengths[s], 2, 64)),
                TileTensor(solo.unsafe_ptr().unsafe_offset((s + 1) * 896), row_major(1, 14, 64)),
                TileTensor(split, row_major(14, 1, 66)))
        _same(solo, paged, "G32 decode, " + _label(size, HEAD_MAJOR) + ", sequences " + String(sequences))


def test_g32_decode_rows_equal_route_4() raises:
    var ctx = DeviceContext()
    for size in [32, 64, 128, ONE_BLOCK]:
        _g32_decode[False](ctx, size)
        _g32_decode[True](ctx, size)


def _prefill[HEAD_MAJOR: Bool](ctx: DeviceContext, size: Int) raises:
    var keys = ctx.enqueue_create_buffer[DType.bfloat16](4096 * WIDTH)
    var values = ctx.enqueue_create_buffer[DType.bfloat16](4096 * WIDTH)
    _fill(keys, 41)
    _fill(values, 42)
    var pages = Pages([4096], size, 11)
    var storage = ctx.enqueue_create_buffer[DType.bfloat16](pages.elements())
    var pool = TileTensor(storage, row_major(len(storage)))
    var table = _tables(ctx, pages, 1)
    var partial = ctx.enqueue_create_buffer[DType.float32](256 * 14 * 8 * 66)
    var workspace = ctx.enqueue_create_buffer[DType.float32](14 * 66)
    for rows in _chunks():
        for past in _prefixes(size, rows):
            var tokens = past + rows
            var label = _label(size, HEAD_MAJOR) + ", " + String(rows) + " rows after " + String(past)
            # Only the sequence's first `tokens` rows are written; the rest of its blocks keep their poison.
            _poison(storage)
            _lay(storage, keys, values, [0], [tokens], pages, HEAD_MAJOR)
            var q = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
            _fill(q, rows * 31 + past)
            var query = TileTensor(q, row_major(rows, 14, 64))
            var k = TileTensor(keys, row_major(tokens, 2, 64))
            var v = TileTensor(values, row_major(tokens, 2, 64))
            var expected = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
            var actual = ctx.enqueue_create_buffer[DType.bfloat16](rows * 896)
            var ends = List[Int](capacity=rows)
            for r in range(rows):
                ends.append(past + r)
            var positions = _positions(ctx, ends)
            _poison(expected)
            _poison(actual)
            enqueue_grouped_query_attention_consistent_apple_gpu(ctx, query, k, v,
                TileTensor(expected, row_major(rows, 14, 64)), TileTensor(workspace, row_major(14, 1, 66)))
            enqueue_paged_attention_g32_apple_gpu[14, 2, 64, HEAD_MAJOR](ctx, query, pool,
                TileTensor(actual, row_major(rows, 14, 64)), TileTensor(positions, row_major(rows)),
                TileTensor(table, row_major(1, pages.width)), rows, LAYER, LAYERS, size)
            _same(expected, actual, "route 11, " + label)
            _poison(expected)
            _poison(actual)
            enqueue_grouped_query_attention_prefill_apple_gpu[32, 32, MMA=True, SCHEDULE=2, FP32=True](ctx,
                query, k, v, TileTensor(expected, row_major(rows, 14, 64)))
            enqueue_paged_attention_prefill_apple_gpu[HEAD_MAJOR](ctx, query, pool,
                TileTensor(table, row_major(pages.width)), TileTensor(actual, row_major(rows, 14, 64)),
                tokens, LAYER, LAYERS, size)
            _same(expected, actual, "route 6, " + label)
            _poison(expected)
            _poison(actual)
            enqueue_grouped_query_attention_prefill_split_apple_gpu[8](ctx, query, k, v,
                TileTensor(partial, row_major(rows, 14, 8, 66)), TileTensor(expected, row_major(rows, 14, 64)))
            enqueue_paged_attention_prefill_split_apple_gpu[8, HEAD_MAJOR](ctx, query, pool,
                TileTensor(table, row_major(pages.width)), TileTensor(partial, row_major(rows, 14, 8, 66)),
                TileTensor(actual, row_major(rows, 14, 64)), tokens, LAYER, LAYERS, size)
            _same(expected, actual, "route 10, " + label)


def test_prefill_attention_equals_routes_6_10_and_11() raises:
    var ctx = DeviceContext()
    for size in [32, 64, 128, ONE_BLOCK]:
        _prefill[False](ctx, size)
        _prefill[True](ctx, size)


def _fused[HEAD_MAJOR: Bool](ctx: DeviceContext, size: Int) raises:
    var lengths = _decode_lengths()
    var pages = Pages(lengths, size, 17)
    var storage = ctx.enqueue_create_buffer[DType.bfloat16](pages.elements())
    var work = AttentionWorkspace(ctx, 32, ONE_BLOCK, 14, 2, 64, False, False)
    var solo = AttentionWorkspace(ctx, 1, ONE_BLOCK, 14, 2, 64, False, False)
    _fill(work.cosine, 11, 121, 6)
    _fill(work.sine, 12, 121, 6)
    _fill(solo.cosine, 11, 121, 6)
    _fill(solo.sine, 12, 121, 6)
    with work.packed.map_to_host() as mapped:
        for i in range(32 * 1152):
            # Diverse exact BF16 bits, signs, subnormals and rounding cases.
            var bits = UInt16((i*1667 + (i//1152)*61) % 0x4300)
            if i % 2:
                bits |= 0x8000
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bits)
    # The unfused path, one sequence at a time: unpack, then RoPE for Q and K; V is copied.
    var expected = ctx.enqueue_create_buffer[DType.bfloat16](32 * 896)
    var rows = List[UInt16](capacity=32 * 2 * WIDTH)
    _poison(expected)
    var c = TileTensor(solo.cosine, row_major(ONE_BLOCK, 64))
    var t = TileTensor(solo.sine, row_major(ONE_BLOCK, 64))
    for s in range(32):
        var position = lengths[s] - 1
        with work.packed.map_to_host() as source:
            with solo.packed.map_to_host() as target:
                for i in range(1152):
                    target.unsafe_ptr()[unsafe_offset=i] = source.unsafe_ptr()[unsafe_offset=s * 1152 + i]
        var packed = TileTensor(solo.packed, row_major(1, 1152))
        var raw_q = TileTensor(solo.raw_query, row_major(1, 896))
        var raw_k = TileTensor(solo.raw_key, row_major(1, 128))
        var raw_v = TileTensor(solo.raw_value, row_major(1, 128))
        ctx.enqueue_function[_unpack_qkv[type_of(packed.layout), type_of(raw_q.layout), type_of(raw_k.layout)]](
            packed, raw_q, raw_k, raw_v, Int32(1), Int32(896), Int32(128), grid_dim=9, block_dim=128)
        enqueue_rope_apple_gpu(ctx, TileTensor(solo.raw_query, row_major(1, 14, 64)), c, t,
                               TileTensor(expected.unsafe_ptr().unsafe_offset(s * 896), row_major(1, 14, 64)), position)
        enqueue_rope_apple_gpu(ctx, TileTensor(solo.raw_key, row_major(1, 2, 64)), c, t,
                               TileTensor(solo.rotated_key, row_major(1, 2, 64)), position)
        with solo.rotated_key.map_to_host() as key:
            for i in range(WIDTH):
                rows.append(bitcast[DType.uint16](key.unsafe_ptr()[unsafe_offset=i]))
        with solo.raw_value.map_to_host() as value:
            for i in range(WIDTH):
                rows.append(bitcast[DType.uint16](value.unsafe_ptr()[unsafe_offset=i]))
    var query = ctx.enqueue_create_buffer[DType.bfloat16](32 * 896)
    for sequences in [1, 2, 3, 8, 16, 32]:
        var ends = List[Int](capacity=sequences)
        for s in range(sequences):
            ends.append(lengths[s] - 1)
        var positions = _positions(ctx, ends)
        var tables = _tables(ctx, pages, sequences)
        _poison(storage)
        _poison(work.query)
        enqueue_fused_decode_qkv_paged[14, 2, 64, HEAD_MAJOR](ctx, work, storage,
            TileTensor(positions, row_major(sequences)), TileTensor(tables, row_major(sequences, pages.width)),
            LAYER, LAYERS, size)
        var label = "fused decode, " + _label(size, HEAD_MAJOR) + ", sequences " + String(sequences)
        # The first `sequences` query rows are the unfused path's; the rest keep their poison.
        _poison(query)
        ctx.enqueue_copy(dst_buf=query.create_sub_buffer[DType.bfloat16](0, sequences * 896),
                         src_buf=expected.create_sub_buffer[DType.bfloat16](0, sequences * 896))
        _same(query, work.query, label + ", query")
        with storage.map_to_host() as b:
            for s in range(sequences):
                var p = ends[s]
                for kv in range(2):
                    for head in range(2):
                        var paged = _offset(pages.block(s, p), kv, p % size, head, size, HEAD_MAJOR)
                        for d in range(64):
                            if (rows[(s * 2 + kv) * WIDTH + head * 64 + d]
                                    != bitcast[DType.uint16](b.unsafe_ptr()[unsafe_offset=paged + d])):
                                raise Error(label + ": sequence " + String(s) + " row differs")
        assert_equal(_written(storage), sequences * 2 * WIDTH)


def test_fused_decode_writes_equal_the_unfused_path() raises:
    var ctx = DeviceContext()
    for size in [32, 64, 128, ONE_BLOCK]:
        _fused[False](ctx, size)
        _fused[True](ctx, size)


def _appended[HEAD_MAJOR: Bool](ctx: DeviceContext, size: Int) raises:
    var pages = Pages([4096], size, 23)
    var storage = ctx.enqueue_create_buffer[DType.bfloat16](pages.elements())
    var table = _tables(ctx, pages, 1)
    var cache_key = ctx.enqueue_create_buffer[DType.bfloat16](4096 * WIDTH)
    var cache_value = ctx.enqueue_create_buffer[DType.bfloat16](4096 * WIDTH)
    for rows in _chunks():
        for past in _prefixes(size, rows):
            var key = ctx.enqueue_create_buffer[DType.bfloat16](rows * WIDTH)
            var value = ctx.enqueue_create_buffer[DType.bfloat16](rows * WIDTH)
            _fill(key, rows + past)
            _fill(value, rows + past + 1)
            _poison(storage)
            _poison(cache_key)
            _poison(cache_value)
            var k = TileTensor(key, row_major(rows, WIDTH))
            var v = TileTensor(value, row_major(rows, WIDTH))
            var ck = TileTensor(cache_key, row_major(4096, WIDTH))
            var cv = TileTensor(cache_value, row_major(4096, WIDTH))
            ctx.enqueue_function[_append[type_of(k.layout), type_of(v.layout), type_of(ck.layout)]](
                k, v, ck, cv, Int32(rows), Int32(WIDTH), Int32(past),
                grid_dim=ceildiv(rows * WIDTH, 128), block_dim=128)
            enqueue_append_paged[2, 64, HEAD_MAJOR](ctx, k, v, storage, TileTensor(table, row_major(pages.width)),
                past, LAYER, LAYERS, size)
            var label = "append, " + _label(size, HEAD_MAJOR) + ", " + String(rows) + " rows after " + String(past)
            with cache_key.map_to_host() as ks:
                with cache_value.map_to_host() as vs:
                    with storage.map_to_host() as pool:
                        for t in range(past, past + rows):
                            for head in range(2):
                                var key_row = _offset(pages.block(0, t), 0, t % size, head, size, HEAD_MAJOR)
                                var value_row = _offset(pages.block(0, t), 1, t % size, head, size, HEAD_MAJOR)
                                for d in range(64):
                                    var source = t * WIDTH + head * 64 + d
                                    if (bitcast[DType.uint16](ks.unsafe_ptr()[unsafe_offset=source])
                                            != bitcast[DType.uint16](pool.unsafe_ptr()[unsafe_offset=key_row + d])
                                            or bitcast[DType.uint16](vs.unsafe_ptr()[unsafe_offset=source])
                                            != bitcast[DType.uint16](pool.unsafe_ptr()[unsafe_offset=value_row + d])):
                                        raise Error(label + ": position " + String(t) + " differs")
            assert_equal(_written(storage), rows * 2 * WIDTH)


def test_append_writes_its_rows_and_nothing_else() raises:
    var ctx = DeviceContext()
    for size in [32, 64, 128, ONE_BLOCK]:
        _appended[False](ctx, size)
        _appended[True](ctx, size)


def test_paged_launches_reject_invalid_geometry() raises:
    var ctx = DeviceContext()
    var pages = Pages([100], 32, 3)
    var storage = ctx.enqueue_create_buffer[DType.bfloat16](pages.elements())
    var pool = TileTensor(storage, row_major(len(storage)))
    var table = _tables(ctx, pages, 1)
    var tables = TileTensor(table, row_major(1, pages.width))
    var q = ctx.enqueue_create_buffer[DType.bfloat16](4 * 896)
    var out = ctx.enqueue_create_buffer[DType.bfloat16](4 * 896)
    _fill(q, 5)
    _poison(out)
    var positions = _positions(ctx, [96, 97, 98, 99])
    var query = TileTensor(q, row_major(4, 14, 64))
    var output = TileTensor(out, row_major(4, 14, 64))
    # The valid call these rejections break.
    enqueue_paged_attention_g32_apple_gpu[14, 2, 64, False](ctx, query, pool, output,
        TileTensor(positions, row_major(4)), tables, 4, LAYER, LAYERS, 32)
    _poison(out)
    # Rows that are not a whole number of sequences, and too few positions.
    with assert_raises():
        enqueue_paged_attention_g32_apple_gpu[14, 2, 64, False](ctx, query, pool, output,
            TileTensor(positions, row_major(4)), tables, 3, LAYER, LAYERS, 32)
    with assert_raises():
        enqueue_paged_attention_g32_apple_gpu[14, 2, 64, False](ctx, query, pool, output,
            TileTensor(positions, row_major(3)), tables, 4, LAYER, LAYERS, 32)
    # A layer outside the pool, and a pool that is not a whole number of blocks.
    with assert_raises():
        enqueue_paged_attention_g32_apple_gpu[14, 2, 64, False](ctx, query, pool, output,
            TileTensor(positions, row_major(4)), tables, 4, LAYERS, LAYERS, 32)
    with assert_raises():
        enqueue_paged_attention_g32_apple_gpu[14, 2, 64, False](ctx, query,
            TileTensor(storage, row_major(len(storage) - 64)), output,
            TileTensor(positions, row_major(4)), tables, 4, LAYER, LAYERS, 32)
    # Several blocks of a size that is not a multiple of 32, whose 32-row tiles would straddle blocks.
    var uneven = ctx.enqueue_create_buffer[DType.bfloat16](8 * LAYERS * 2 * 48 * WIDTH)
    with assert_raises():
        enqueue_paged_attention_g32_apple_gpu[14, 2, 64, False](ctx, query,
            TileTensor(uneven, row_major(len(uneven))), output,
            TileTensor(positions, row_major(4)), tables, 4, LAYER, LAYERS, 48)
    # A table that does not cover the sequence, and more rows than tokens.
    var k = ctx.enqueue_create_buffer[DType.bfloat16](4 * WIDTH)
    with assert_raises():
        enqueue_paged_attention_prefill_apple_gpu[False](ctx, query, pool, TileTensor(table, row_major(3)),
            output, 100, LAYER, LAYERS, 32)
    with assert_raises():
        enqueue_paged_attention_prefill_apple_gpu[False](ctx, query, pool, TileTensor(table, row_major(pages.width)),
            output, 3, LAYER, LAYERS, 32)
    # An append past the table's last block.
    with assert_raises():
        enqueue_append_paged[2, 64, False](ctx, TileTensor(k, row_major(4, WIDTH)), TileTensor(k, row_major(4, WIDTH)),
            storage, TileTensor(table, row_major(pages.width)), pages.width * 32 - 3, LAYER, LAYERS, 32)
    # Fused decode with more sequences than table rows.
    var work = AttentionWorkspace(ctx, 4, ONE_BLOCK, 14, 2, 64, False, False)
    with assert_raises():
        enqueue_fused_decode_qkv_paged[14, 2, 64, False](ctx, work, storage,
            TileTensor(positions, row_major(4)), tables, LAYER, LAYERS, 32)
    assert_equal(_written(out), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
