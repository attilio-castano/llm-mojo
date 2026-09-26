"""Batched decode: every row of a batched step equals its sequence decoded alone.

Each batched kernel is compared bit for bit with the same work done one row or
one sequence at a time, with guard rows and unwritten pool rows poisoned:
- every decode projection arrangement against one-row launches at every decode
  projection width, with signed zeros and subnormals among the values;
- residual RMSNorm and argmax against single-row launches;
- fused QKV/RoPE/append against the unfused path for each sequence;
- decode attention against route 4 on each sequence's own cache view;
- whole batched decode steps against each sequence decoded alone, on three
  layers of the verified decoder fixture, with invalid batches rejected before
  any state changes.
"""
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from llm_mojo.kernels.attention_decode import (
    enqueue_grouped_query_attention_decode_apple_gpu,
    enqueue_grouped_query_attention_decode_sequences_apple_gpu,
)
from llm_mojo.kernels.linear import DECODE_ARRANGEMENTS, enqueue_linear_apple_gpu, enqueue_linear_decode_rows_apple_gpu
from llm_mojo.kernels.residual_norm import enqueue_residual_norm
from llm_mojo.kernels.rope import enqueue_rope_apple_gpu
from llm_mojo.kernels.token_selection import enqueue_argmax
from llm_mojo.layers.attention_sublayer import AttentionWorkspace, _append, _unpack_qkv, enqueue_fused_decode_qkv_batch
from llm_mojo.layers.decoder_layer import DECODER_FUSED_DECODE
from llm_mojo.models.qwen2.model import CaptureRequest, QwenModel
from llm_mojo.models.qwen2.plan import baseline_plan, configured_plan
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVGeometry, KVPool
from decoder_layer_support import decoder_support, load_decoder

comptime POISON = UInt16(0x7FC1)
# Pool for the sequence kernels: scattered blocks of the full context, in layer 1 of 2.
comptime BLOCKS = 33
comptime BLOCK_SIZE = 4096
comptime LAYER = 1
# Whole steps: three layers of the verified decoder fixture, whose 65 positions bound the context.
comptime CASE = "h896_i4864_nq14_nk2_d64_t65_s4001_base"
comptime FIXTURE_LAYERS = 3
comptime CONTEXT = 65
comptime MAX_PREFIX = 40
comptime DECODE_STEPS = 12
comptime VOCABULARY = 151936


def _fill(mut buffer: DeviceBuffer[DType.bfloat16], seed: Int, low: Int = 119, span: Int = 16) raises:
    # Mixed signs, mantissas and exponents 2^(low-127) .. 2^(low+span-128).
    with buffer.map_to_host() as mapped:
        for i in range(len(buffer)):
            var raw = UInt32((i*1664525+seed*1013904223) & 0xffffffff)
            var bits = UInt16((raw >> 16) & 0x807f) | UInt16((low+Int((raw >> 7)%UInt32(span))) << 7)
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bits)


def _poison(mut buffer: DeviceBuffer[DType.bfloat16]) raises:
    buffer.enqueue_fill(bitcast[DType.bfloat16](POISON))


def _length(s: Int) -> Int:
    """Keys attended by sequence s: short, around one and two SIMD-group rounds, and full."""
    var lengths: List[Int] = [1, 2, 31, 32, 33, 64, 65, 257, 1024, 4096]
    return lengths[s] if s < len(lengths) else (s*977) % BLOCK_SIZE + 1


def _block(s: Int) -> Int:
    return (s*7+3) % BLOCKS


def _steps(ctx: DeviceContext, sequences: Int) raises -> DeviceBuffer[DType.int32]:
    """Positions, then blocks, for a decode step of `sequences` sequences."""
    var steps = ctx.enqueue_create_buffer[DType.int32](2*sequences)
    with steps.map_to_host() as mapped:
        for s in range(sequences):
            mapped.unsafe_ptr()[unsafe_offset=s] = Int32(_length(s)-1)
            mapped.unsafe_ptr()[unsafe_offset=sequences+s] = Int32(_block(s))
    return steps^


def _same(mut expected: DeviceBuffer[DType.bfloat16], mut actual: DeviceBuffer[DType.bfloat16],
          label: String) raises:
    with expected.map_to_host() as a:
        with actual.map_to_host() as b:
            for i in range(len(expected)):
                if (bitcast[DType.uint16](a.unsafe_ptr()[unsafe_offset=i])
                        != bitcast[DType.uint16](b.unsafe_ptr()[unsafe_offset=i])):
                    raise Error(label + " differs at element " + String(i))


def _edges(mut buffer: DeviceBuffer[DType.bfloat16], seed: Int) raises:
    """Every fifth value becomes a signed zero, a subnormal or a neighbour of one."""
    var edges: List[UInt16] = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x3F80, 0x3F81, 0xBF7F]
    with buffer.map_to_host() as mapped:
        for i in range(seed % 5, len(buffer), 5):
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](edges[(i // 5) % len(edges)])


def _arranged[ARRANGEMENT: Int, HAS_BIAS: Bool](ctx: DeviceContext, mut x: DeviceBuffer[DType.bfloat16],
                                                mut w: DeviceBuffer[DType.bfloat16], mut b: DeviceBuffer[DType.bfloat16],
                                                mut solo: DeviceBuffer[DType.bfloat16], rows: Int, n: Int, k: Int) raises:
    # One guard row on each side stays poisoned.
    var batch = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*n)
    batch.enqueue_fill(-123)
    var input = TileTensor(x,row_major(rows,k))
    var weight = TileTensor(w,row_major(n,k))
    var output = TileTensor(batch.unsafe_ptr().unsafe_offset(n),row_major(rows,n))
    comptime if HAS_BIAS:
        enqueue_linear_decode_rows_apple_gpu[ARRANGEMENT](ctx,input,weight,TileTensor(b,row_major(n)),output)
    else:
        enqueue_linear_decode_rows_apple_gpu[ARRANGEMENT](ctx,input,weight,output)
    _same(solo,batch,"arrangement "+String(ARRANGEMENT)+" rows "+String(rows)+" outputs "+String(n)+" inputs "+String(k))


def _widths[HAS_BIAS: Bool, FIRST: Int = 0, LAST: Int = DECODE_ARRANGEMENTS](ctx: DeviceContext, n: Int, k: Int) raises:
    var w = ctx.enqueue_create_buffer[DType.bfloat16](n*k)
    var b = ctx.enqueue_create_buffer[DType.bfloat16](n)
    _fill(w,n)
    _fill(b,k)
    _edges(w,n)
    _edges(b,k)
    for rows in [1, 2, 3, 5, 8, 13, 16, 31, 33, 64]:
        var x = ctx.enqueue_create_buffer[DType.bfloat16](rows*k)
        var solo = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*n)
        _fill(x,rows)
        _edges(x,rows)
        solo.enqueue_fill(-123)
        var weight = TileTensor(w,row_major(n,k))
        for row in range(rows):
            var input = TileTensor(x.unsafe_ptr().unsafe_offset(row*k),row_major(1,k))
            var output = TileTensor(solo.unsafe_ptr().unsafe_offset((row+1)*n),row_major(1,n))
            comptime if HAS_BIAS:
                enqueue_linear_apple_gpu(ctx,input,weight,TileTensor(b,row_major(n)),output)
            else:
                enqueue_linear_apple_gpu(ctx,input,weight,output)
        comptime for arrangement in range(FIRST, LAST):
            _arranged[arrangement,HAS_BIAS](ctx,x,w,b,solo,rows,n,k)


def test_decode_arrangements_equal_one_row_launches_at_decode_widths() raises:
    var ctx = DeviceContext()
    assert_equal(ctx.api(),"metal")
    _widths[True](ctx,1152,896)
    _widths[False](ctx,896,896)
    _widths[False](ctx,4864,896)
    _widths[False](ctx,896,4864)
    _widths[False](ctx,151936,896)


def test_decode_arrangements_check_shapes_before_launch() raises:
    """Other widths use the runtime loop or one column; unsupported shapes change nothing."""
    var ctx = DeviceContext()
    _widths[False, 4, 5](ctx,1152,512)
    _widths[False, 3, 4](ctx,1150,896)
    var x = ctx.enqueue_create_buffer[DType.bfloat16](2*512)
    var w = ctx.enqueue_create_buffer[DType.bfloat16](1150*512)
    var y = ctx.enqueue_create_buffer[DType.bfloat16](2*1150)
    _fill(x,1)
    _fill(w,2)
    y.enqueue_fill(-123)
    var input = TileTensor(x,row_major(2,512))
    var weight = TileTensor(w,row_major(1150,512))
    var output = TileTensor(y,row_major(2,1150))
    # Arrangement 3 fixes the width; 4-6 compute four columns at a time.
    with assert_raises():
        enqueue_linear_decode_rows_apple_gpu[3](ctx,input,weight,output)
    with assert_raises():
        enqueue_linear_decode_rows_apple_gpu[4](ctx,input,weight,output)
    with assert_raises():
        enqueue_linear_decode_rows_apple_gpu[5](ctx,input,weight,output)
    with assert_raises():
        enqueue_linear_decode_rows_apple_gpu[6](ctx,input,weight,output)
    with y.map_to_host() as mapped:
        for i in range(len(y)):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i].cast[DType.float32](), Float32(-123))


def test_residual_norm_rows_equal_single_rows() raises:
    var ctx = DeviceContext()
    var weight = ctx.enqueue_create_buffer[DType.bfloat16](896)
    _fill(weight,5)
    var w = TileTensor(weight,row_major(896))
    for rows in [1, 2, 3, 8, 16, 32]:
        var x = ctx.enqueue_create_buffer[DType.bfloat16](rows*896)
        var branch = ctx.enqueue_create_buffer[DType.bfloat16](rows*896)
        _fill(x,rows)
        _fill(branch,rows+100)
        var solo_residual = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*896)
        var solo_normal = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*896)
        var batch_residual = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*896)
        var batch_normal = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*896)
        _poison(solo_residual)
        _poison(solo_normal)
        _poison(batch_residual)
        _poison(batch_normal)
        enqueue_residual_norm[896](ctx,TileTensor(x,row_major(rows,896)),TileTensor(branch,row_major(rows,896)),w,
            TileTensor(batch_residual.unsafe_ptr().unsafe_offset(896),row_major(rows,896)),
            TileTensor(batch_normal.unsafe_ptr().unsafe_offset(896),row_major(rows,896)))
        for r in range(rows):
            enqueue_residual_norm[896](ctx,TileTensor(x.unsafe_ptr().unsafe_offset(r*896),row_major(1,896)),
                TileTensor(branch.unsafe_ptr().unsafe_offset(r*896),row_major(1,896)),w,
                TileTensor(solo_residual.unsafe_ptr().unsafe_offset((r+1)*896),row_major(1,896)),
                TileTensor(solo_normal.unsafe_ptr().unsafe_offset((r+1)*896),row_major(1,896)))
        _same(solo_residual,batch_residual,"residual rows "+String(rows))
        _same(solo_normal,batch_normal,"normal rows "+String(rows))


def test_argmax_rows_equal_single_rows() raises:
    var ctx = DeviceContext()
    for count in [65, 1025, 151936]:
        var groups = (count+1023)//1024
        for rows in [1, 2, 3, 8, 16, 32]:
            var logits = ctx.enqueue_create_buffer[DType.bfloat16](rows*count)
            with logits.map_to_host() as mapped:
                for r in range(rows):
                    for i in range(count):
                        # Ninety-seven distinct values force ties; row 2 holds one NaN.
                        var bits = UInt16(0x3F80 + (i*7919 + r*104729) % 97)
                        if r == 2 and i == count // 2:
                            bits = 0x7FC0
                        mapped.unsafe_ptr()[unsafe_offset=r*count+i] = bitcast[DType.bfloat16](bits)
            var partials = ctx.enqueue_create_buffer[DType.uint32](rows*groups*3)
            var batch = ctx.enqueue_create_buffer[DType.uint32](rows*3)
            var scratch = ctx.enqueue_create_buffer[DType.uint32](groups*3)
            var solo = ctx.enqueue_create_buffer[DType.uint32](rows*3)
            enqueue_argmax(ctx,TileTensor(logits,row_major(rows,count)),TileTensor(partials,row_major(rows*groups,3)),
                           TileTensor(batch,row_major(rows,3)))
            for r in range(rows):
                enqueue_argmax(ctx,TileTensor(logits.unsafe_ptr().unsafe_offset(r*count),row_major(1,count)),
                               TileTensor(scratch,row_major(groups,3)),TileTensor(solo.unsafe_ptr().unsafe_offset(r*3),row_major(1,3)))
            with batch.map_to_host() as actual:
                with solo.map_to_host() as expected:
                    for i in range(rows*3):
                        assert_equal(actual.unsafe_ptr()[unsafe_offset=i],expected.unsafe_ptr()[unsafe_offset=i])
                    for r in range(rows):
                        assert_equal(actual.unsafe_ptr()[unsafe_offset=r*3+2],UInt32(1 if r == 2 else 0))


def _fused_qkv(ctx: DeviceContext, sequences: Int) raises:
    var geometry = KVGeometry(2,2,64)
    var candidate = KVPool(ctx,BLOCKS,BLOCK_SIZE,geometry)
    var reference = KVPool(ctx,BLOCKS,BLOCK_SIZE,geometry)
    _poison(candidate.storage)
    _poison(reference.storage)
    var work = AttentionWorkspace(ctx,sequences,BLOCK_SIZE,14,2,64,False,False)
    var solo = AttentionWorkspace(ctx,1,BLOCK_SIZE,14,2,64,False,False)
    _fill(work.cosine,11,121,6)
    _fill(work.sine,12,121,6)
    _fill(solo.cosine,11,121,6)
    _fill(solo.sine,12,121,6)
    with work.packed.map_to_host() as mapped:
        for i in range(sequences*1152):
            # Diverse exact BF16 bits, signs, subnormals and rounding cases.
            var bits = UInt16((i*1667 + (i//1152)*61) % 0x4300)
            if i % 2:
                bits |= 0x8000
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bits)
    _poison(work.query)
    var expected = ctx.enqueue_create_buffer[DType.bfloat16](sequences*896)
    _poison(expected)
    var c = TileTensor(solo.cosine,row_major(BLOCK_SIZE,64))
    var t = TileTensor(solo.sine,row_major(BLOCK_SIZE,64))
    for s in range(sequences):
        var position = _length(s)-1
        with work.packed.map_to_host() as source:
            with solo.packed.map_to_host() as target:
                for i in range(1152):
                    target.unsafe_ptr()[unsafe_offset=i] = source.unsafe_ptr()[unsafe_offset=s*1152+i]
        var packed = TileTensor(solo.packed,row_major(1,1152))
        var raw_q = TileTensor(solo.raw_query,row_major(1,896))
        var raw_k = TileTensor(solo.raw_key,row_major(1,128))
        var raw_v = TileTensor(solo.raw_value,row_major(1,128))
        ctx.enqueue_function[_unpack_qkv[type_of(packed.layout),type_of(raw_q.layout),type_of(raw_k.layout)]](
            packed,raw_q,raw_k,raw_v,Int32(1),Int32(896),Int32(128),grid_dim=9,block_dim=128)
        enqueue_rope_apple_gpu(ctx,TileTensor(solo.raw_query,row_major(1,14,64)),c,t,
                               TileTensor(expected.unsafe_ptr().unsafe_offset(s*896),row_major(1,14,64)),position)
        enqueue_rope_apple_gpu(ctx,TileTensor(solo.raw_key,row_major(1,2,64)),c,t,
                               TileTensor(solo.rotated_key,row_major(1,2,64)),position)
        var view = reference.index(_block(s),LAYER)
        var rotated = TileTensor(solo.rotated_key,row_major(1,128))
        var keys = TileTensor(reference.caches[view].key,row_major(BLOCK_SIZE,128))
        var values = TileTensor(reference.caches[view].value,row_major(BLOCK_SIZE,128))
        ctx.enqueue_function[_append[type_of(rotated.layout),type_of(raw_v.layout),type_of(keys.layout)]](
            rotated,raw_v,keys,values,Int32(1),Int32(128),Int32(position),grid_dim=1,block_dim=128)
    var steps = _steps(ctx,sequences)
    enqueue_fused_decode_qkv_batch[14,2,64](ctx,work,candidate.storage,
        TileTensor(steps.unsafe_ptr(),row_major(sequences)),
        TileTensor(steps.unsafe_ptr().unsafe_offset(sequences),row_major(sequences)),LAYER,2,BLOCK_SIZE)
    _same(expected,work.query,"fused query, sequences "+String(sequences))
    _same(reference.storage,candidate.storage,"fused pool, sequences "+String(sequences))


def test_fused_qkv_batch_equals_unfused_path_per_sequence() raises:
    var ctx = DeviceContext()
    for sequences in [1, 2, 3, 8, 16, 32]:
        _fused_qkv(ctx,sequences)


def test_decode_attention_sequences_equal_route_4() raises:
    var ctx = DeviceContext()
    var pool = KVPool(ctx,BLOCKS,BLOCK_SIZE,KVGeometry(2,2,64))
    _fill(pool.storage,21,119,8)
    var split = ctx.enqueue_create_buffer[DType.float32](14*66)
    var storage = TileTensor(pool.storage,row_major(len(pool.storage)//128,2,64))
    for sequences in [1, 2, 3, 8, 16, 32]:
        var query = ctx.enqueue_create_buffer[DType.bfloat16](sequences*896)
        _fill(query,sequences,119,8)
        var batch = ctx.enqueue_create_buffer[DType.bfloat16]((sequences+2)*896)
        var solo = ctx.enqueue_create_buffer[DType.bfloat16]((sequences+2)*896)
        _poison(batch)
        _poison(solo)
        var steps = _steps(ctx,sequences)
        enqueue_grouped_query_attention_decode_sequences_apple_gpu[14,2,64](ctx,
            TileTensor(query,row_major(sequences,14,64)),storage,
            TileTensor(batch.unsafe_ptr().unsafe_offset(896),row_major(sequences,14,64)),
            TileTensor(steps.unsafe_ptr(),row_major(sequences)),
            TileTensor(steps.unsafe_ptr().unsafe_offset(sequences),row_major(sequences)),LAYER,2,BLOCK_SIZE)
        for s in range(sequences):
            var view = pool.index(_block(s),LAYER)
            var keys = _length(s)
            enqueue_grouped_query_attention_decode_apple_gpu[32,1,1,fp32_scores=True](ctx,
                TileTensor(query.unsafe_ptr().unsafe_offset(s*896),row_major(1,14,64)),
                TileTensor(pool.caches[view].key,row_major(keys,2,64)),
                TileTensor(pool.caches[view].value,row_major(keys,2,64)),
                TileTensor(solo.unsafe_ptr().unsafe_offset((s+1)*896),row_major(1,14,64)),
                TileTensor(split,row_major(14,1,66)))
        _same(solo,batch,"attention, sequences "+String(sequences))
    assert_true(_length(9) == BLOCK_SIZE)


def _fixture_model(ctx: DeviceContext, max_sequences: Int) raises -> QwenModel:
    var model = QwenModel.allocate(ctx, FIXTURE_LAYERS, CONTEXT, MAX_PREFIX, max_sequences)
    model.embedding.enqueue_fill(0)
    model.attention.cosine.enqueue_fill(0)
    model.attention.sine.enqueue_fill(0)
    # Token i embeds fixture row i, for the 65 IDs the tests use.
    load_decoder(model.embedding, CASE, "input_X", 0, CONTEXT * 896)
    load_decoder(model.norm, CASE, "input_post_norm", 0, 896)
    load_decoder(model.attention.cosine, CASE, "full_cosine", 0, CONTEXT * 64)
    load_decoder(model.attention.sine, CASE, "full_sine", 0, CONTEXT * 64)
    for i in range(FIXTURE_LAYERS):
        load_decoder(model.layers[i].attention.norm, CASE, "input_input_norm", 0, 896)
        load_decoder(model.layers[i].attention.qkv, CASE, "input_qkv", 0, 1152 * 896)
        load_decoder(model.layers[i].attention.bias, CASE, "input_bias", 0, 1152)
        load_decoder(model.layers[i].attention.output, CASE, "input_wo", 0, 896 * 896)
        load_decoder(model.layers[i].mlp.norm, CASE, "input_post_norm", 0, 896)
        load_decoder(model.layers[i].mlp.gate, CASE, "input_gate", 0, 4864 * 896)
        load_decoder(model.layers[i].mlp.up, CASE, "input_up", 0, 4864 * 896)
        load_decoder(model.layers[i].mlp.down, CASE, "input_down", 0, 896 * 4864)
    return model^


def _token(s: Int, position: Int) -> Int:
    return (position + 13 * s) % CONTEXT


def _prefix(s: Int) -> Int:
    return 1 + (s * 7) % MAX_PREFIX


def _prefill(ctx: DeviceContext, mut model: QwenModel, mut pool: KVPool, s: Int, length: Int) raises:
    var ids = List[Int]()
    for p in range(length):
        ids.append(_token(s, p))
    model.forward(ctx, StepBatch.sequence(ids, 0, _block(s), CONTEXT), pool, baseline_plan(length, length))


def _decode(sequences: Int, step: Int, lengths: List[Int]) raises -> StepBatch:
    """One decode token for each sequence, at the end of its block."""
    var ids = List[Int]()
    var positions = List[Int]()
    var starts = List[Int]()
    var seq_lens = List[Int]()
    var blocks = List[Int]()
    var slots = List[Int]()
    var rows = List[Int]()
    for s in range(sequences):
        var p = lengths[s] + step
        ids.append(_token(s, p))
        positions.append(p)
        starts.append(s)
        seq_lens.append(p + 1)
        blocks.append(_block(s))
        slots.append(_block(s) * CONTEXT + p)
        rows.append(s)
    starts.append(sequences)
    return StepBatch(ids^, positions^, starts^, sequences, seq_lens^, 1, blocks^, slots^, rows^)


def _same_rows(mut batched: DeviceBuffer[DType.bfloat16], start: Int, mut solo: DeviceBuffer[DType.bfloat16],
               count: Int, label: String) raises:
    with batched.map_to_host() as a:
        with solo.map_to_host() as b:
            for i in range(count):
                if (bitcast[DType.uint16](a.unsafe_ptr()[unsafe_offset=start+i])
                        != bitcast[DType.uint16](b.unsafe_ptr()[unsafe_offset=i])):
                    raise Error(label + " differs at element " + String(i))


def _steps_equal_solo(ctx: DeviceContext, sequences: Int) raises:
    var batched = _fixture_model(ctx, sequences)
    var solo = _fixture_model(ctx, 1)
    var batched_pool = KVPool(ctx, BLOCKS, CONTEXT, batched.kv_geometry())
    var solo_pool = KVPool(ctx, BLOCKS, CONTEXT, solo.kv_geometry())
    _poison(batched_pool.storage)
    _poison(solo_pool.storage)
    var lengths = List[Int]()
    var longest = 0
    for s in range(sequences):
        lengths.append(_prefix(s))
        longest = max(longest, _prefix(s))
        _prefill(ctx, batched, batched_pool, s, _prefix(s))
        _prefill(ctx, solo, solo_pool, s, _prefix(s))
    for step in range(DECODE_STEPS):
        batched.forward(ctx, _decode(sequences, step, lengths), batched_pool,
                        configured_plan(DECODER_FUSED_DECODE, sequences, longest + step + 1, sequences))
        var tokens = batched.greedy_tokens(ctx)
        assert_equal(len(tokens), sequences)
        assert_equal(batched.last_route.sequences, sequences)
        # Launches do not depend on the number of sequences.
        assert_equal(batched.last_route.decode_launches, 1 + 10 + 9 * (FIXTURE_LAYERS - 1) + FIXTURE_LAYERS + 3)
        for s in range(sequences):
            var p = lengths[s] + step
            solo.forward(ctx, StepBatch.sequence([_token(s, p)], p, _block(s), CONTEXT), solo_pool,
                         configured_plan(DECODER_FUSED_DECODE, 1, p + 1))
            var label = "sequences " + String(sequences) + " step " + String(step) + " sequence " + String(s)
            assert_equal(tokens[s], solo.greedy(ctx))
            _same_rows(batched.logits, s * VOCABULARY, solo.logits, VOCABULARY, label + " logits")
            _same_rows(batched.normalized, s * 896, solo.normalized, 896, label + " final norm")
            assert_equal(batched_pool.length(_block(s)), p + 1)
            assert_equal(solo_pool.length(_block(s)), p + 1)
        # Every block, with the rows no sequence wrote still poisoned.
        _same(solo_pool.storage, batched_pool.storage, "pool, sequences " + String(sequences))
        assert_equal(batched.submitted_rows, solo.submitted_rows)


def test_batched_steps_equal_each_sequence_decoded_alone() raises:
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    for sequences in [2, 3, 8, 16, 32]:
        _steps_equal_solo(ctx, sequences)


def _tile_step[TILE: Int](ctx: DeviceContext, mut model: QwenModel, mut pool: KVPool, sequences: Int,
                          lengths: List[Int], longest: Int) raises -> List[Int]:
    for s in range(sequences):
        pool.truncate(_block(s), lengths[s])
    model.forward[False, TILE](ctx, _decode(sequences, 0, lengths), pool,
                               configured_plan(DECODER_FUSED_DECODE, sequences, longest + 1, sequences))
    assert_equal(model.last_route.decode_launches, 1 + 10 + 9 * (FIXTURE_LAYERS - 1) + FIXTURE_LAYERS + 3)
    return model.greedy_tokens[True](ctx)


def test_row_tiles_give_identical_batched_steps() raises:
    """The row tile changes weight reuse only: tiles 8 and 16 reproduce tile 4's bytes."""
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    var sequences = 16
    var model = _fixture_model(ctx, sequences)
    var pool = KVPool(ctx, BLOCKS, CONTEXT, model.kv_geometry())
    _poison(pool.storage)
    var lengths = List[Int]()
    var longest = 0
    for s in range(sequences):
        lengths.append(_prefix(s))
        longest = max(longest, _prefix(s))
        _prefill(ctx, model, pool, s, _prefix(s))
    var expected = _tile_step[4](ctx, model, pool, sequences, lengths, longest)
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](len(model.logits))
    var storage = ctx.enqueue_create_buffer[DType.bfloat16](len(pool.storage))
    ctx.enqueue_copy(dst_buf=logits, src_buf=model.logits)
    ctx.enqueue_copy(dst_buf=storage, src_buf=pool.storage)
    var eight = _tile_step[8](ctx, model, pool, sequences, lengths, longest)
    _same(logits, model.logits, "tile 8 logits")
    _same(storage, pool.storage, "tile 8 pool")
    var sixteen = _tile_step[16](ctx, model, pool, sequences, lengths, longest)
    _same(logits, model.logits, "tile 16 logits")
    _same(storage, pool.storage, "tile 16 pool")
    for s in range(sequences):
        assert_equal(eight[s], expected[s])
        assert_equal(sixteen[s], expected[s])
    # Observed selection records every host mark in order.
    for i in range(1, 10):
        assert_true(model.observation[i] >= model.observation[i - 1] or i == 6)


def _unchanged(model: QwenModel, pool: KVPool, submitted: Int) raises:
    assert_equal(model.valid, True)
    assert_equal(model.submitted_rows, submitted)
    for b in range(pool.blocks):
        assert_equal(pool.length(b), 5 + b if b < 3 else 0)


def test_invalid_batched_steps_change_nothing() raises:
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    var model = _fixture_model(ctx, 4)
    var pool = KVPool(ctx, 8, CONTEXT, model.kv_geometry())
    for s in range(3):
        var ids = List[Int]()
        for p in range(5 + s):
            ids.append(_token(s, p))
        model.forward(ctx, StepBatch.sequence(ids, 0, s, CONTEXT), pool, baseline_plan(5 + s, 5 + s))
    var submitted = model.submitted_rows
    var fused = configured_plan(DECODER_FUSED_DECODE, 2, 8, 2)
    # A prefill chunk beside a decode.
    with assert_raises():
        model.forward(ctx, StepBatch([1, 2, 3], [5, 6, 6], [0, 2, 3], 0, [7, 7], 1, [0, 1], [5, 71, 136], [1, 2]),
                      pool, fused)
    _unchanged(model, pool, submitted)
    # More sequences than the model holds.
    with assert_raises():
        model.forward(ctx, StepBatch([1, 2, 3, 4, 5], [5, 6, 7, 0, 0], [0, 1, 2, 3, 4, 5], 5, [6, 7, 8, 1, 1], 1,
                                     [0, 1, 2, 3, 4], [5, 71, 137, 195, 260], [0, 1, 2, 3, 4]),
                      pool, configured_plan(DECODER_FUSED_DECODE, 5, 8, 5))
    _unchanged(model, pool, submitted)
    var pair = StepBatch([1, 2], [5, 6], [0, 1, 2], 2, [6, 7], 1, [0, 1], [5, 71], [0, 1])
    # Only configuration 26 steps several sequences.
    with assert_raises():
        model.forward(ctx, pair, pool, baseline_plan(2, 7))
    _unchanged(model, pool, submitted)
    # A position that disagrees with its block.
    with assert_raises():
        model.forward(ctx, StepBatch([1, 2], [5, 7], [0, 1, 2], 2, [6, 8], 1, [0, 1], [5, 72], [0, 1]), pool, fused)
    _unchanged(model, pool, submitted)
    # A pool whose block size is not the model's.
    var other = KVPool(ctx, 8, CONTEXT - 1, model.kv_geometry())
    with assert_raises():
        model.forward(ctx, StepBatch([1, 2], [0, 0], [0, 1, 2], 2, [1, 1], 1, [0, 1], [0, 64], [0, 1]), other, fused)
    assert_equal(other.length(0), 0)
    _unchanged(model, pool, submitted)
    # Captures cover one sequence.
    with assert_raises():
        model.forward_captured(ctx, pair, pool, fused, CaptureRequest("build/test_decode_batch", False))
    _unchanged(model, pool, submitted)
    # A valid step selects one token per sequence; the single-sequence reader refuses it.
    model.forward(ctx, pair, pool, fused)
    with assert_raises():
        _ = model.greedy(ctx)
    assert_equal(model.valid, True)
    assert_equal(len(model.greedy_tokens(ctx)), 2)
    assert_equal(pool.length(0), 6)
    assert_equal(pool.length(1), 7)
    assert_equal(model.submitted_rows, submitted + 2 * FIXTURE_LAYERS)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
