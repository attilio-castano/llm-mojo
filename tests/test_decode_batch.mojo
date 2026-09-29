"""Batched decode: every row of a batched step equals its sequence decoded alone.

Each batched kernel is compared bit for bit with the same work done one row or
one sequence at a time, with guard rows and unwritten pool rows poisoned:
- every decode projection arrangement against one-row launches at every decode
  projection width, with signed zeros and subnormals among the values: the
  one-row kernel for the exact arrangements, and the arrangement's own one-row
  launches for the reordered ones, which must also stay within the worst-case
  FP32 summation error of an FP64 reference;
- residual RMSNorm and argmax against single-row launches;
- whole batched decode steps against each sequence decoded alone, on three
  layers of the verified decoder fixture, with invalid batches rejected before
  any state changes;
- sequences held in blocks of 32, 64 and 128 slots, in both orders within a
  block: their prefill through each prefill configuration and their batched
  decode steps against one block per sequence decoded alone, with every row no
  sequence wrote still poisoned.
The paged fused QKV/RoPE/append and decode attention that these steps launch
are checked against the unfused path and route 4 in tests/test_paged_kv.mojo.
"""
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from llm_mojo.kernels.linear import (
    DECODE_ARRANGEMENTS, decode_arrangement_reordered, enqueue_linear_apple_gpu, enqueue_linear_decode_rows_apple_gpu,
)
from llm_mojo.kernels.residual_norm import enqueue_residual_norm
from llm_mojo.kernels.token_selection import enqueue_argmax
from llm_mojo.layers.decoder_layer import DECODER_FUSED_DECODE
from llm_mojo.models.qwen2.model import CaptureRequest, QwenModel
from llm_mojo.models.qwen2.plan import DECODE_PROJECTION, baseline_plan, configured_plan
from std.math import ceildiv
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVPool
from decoder_layer_support import decoder_support, load_decoder

comptime POISON = UInt16(0x7FC1)
# Pools of scattered one-sequence blocks.
comptime BLOCKS = 33
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


def _block(s: Int) -> Int:
    return (s*7+3) % BLOCKS


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


def _launch[ARRANGEMENT: Int, HAS_BIAS: Bool](ctx: DeviceContext, mut x: DeviceBuffer[DType.bfloat16],
                                              mut w: DeviceBuffer[DType.bfloat16], mut b: DeviceBuffer[DType.bfloat16],
                                              mut y: DeviceBuffer[DType.bfloat16], first: Int, rows: Int, n: Int,
                                              k: Int) raises:
    """Rows first..first+rows of x into rows first+1.. of y, which keeps a guard row on each side."""
    var input = TileTensor(x.unsafe_ptr().unsafe_offset(first*k),row_major(rows,k))
    var weight = TileTensor(w,row_major(n,k))
    var output = TileTensor(y.unsafe_ptr().unsafe_offset((first+1)*n),row_major(rows,n))
    comptime if HAS_BIAS:
        enqueue_linear_decode_rows_apple_gpu[ARRANGEMENT](ctx,input,weight,TileTensor(b,row_major(n)),output)
    else:
        enqueue_linear_decode_rows_apple_gpu[ARRANGEMENT](ctx,input,weight,output)


def _reference(mut x: DeviceBuffer[DType.bfloat16], mut w: DeviceBuffer[DType.bfloat16],
               mut b: DeviceBuffer[DType.bfloat16], rows: Int, n: Int, k: Int, has_bias: Bool) raises -> List[Float64]:
    """For each output, its FP64 sum and the sum of its terms' magnitudes."""
    var result = List[Float64](capacity=2*rows*n)
    with x.map_to_host() as xs:
        with w.map_to_host() as ws:
            with b.map_to_host() as bs:
                for r in range(rows):
                    for c in range(n):
                        var total: Float64 = 0
                        var magnitude: Float64 = 0
                        for f in range(k):
                            var term = (xs.unsafe_ptr()[unsafe_offset=r*k+f].cast[DType.float64]()
                                        * ws.unsafe_ptr()[unsafe_offset=c*k+f].cast[DType.float64]())
                            total += term
                            magnitude += abs(term)
                        if has_bias:
                            var bias = bs.unsafe_ptr()[unsafe_offset=c].cast[DType.float64]()
                            total += bias
                            magnitude += abs(bias)
                        result.append(total)
                        result.append(magnitude)
    return result^


def _within_bound(mut y: DeviceBuffer[DType.bfloat16], reference: List[Float64], rows: Int, n: Int, k: Int,
                  label: String) raises:
    """Any correct FP32 summation order lies within (k+1)u times the terms' magnitudes, then one BF16 rounding."""
    with y.map_to_host() as ys:
        for i in range(rows*n):
            var got = ys.unsafe_ptr()[unsafe_offset=n+i].cast[DType.float64]()
            var total = reference[2*i]
            var bound = 2.0 * Float64(k+1) * 5.960464477539063e-08 * reference[2*i+1] + 0.00390625 * abs(total)
            if abs(got-total) > bound + 1e-30:
                raise Error(label + " exceeds the summation bound at output " + String(i))


def _arranged[ARRANGEMENT: Int, HAS_BIAS: Bool](ctx: DeviceContext, mut x: DeviceBuffer[DType.bfloat16],
                                                mut w: DeviceBuffer[DType.bfloat16], mut b: DeviceBuffer[DType.bfloat16],
                                                mut solo: DeviceBuffer[DType.bfloat16], rows: Int, n: Int, k: Int,
                                                reference: List[Float64]) raises:
    """Batched rows equal one-row results: the one-row kernel's, or a reordered arrangement's own."""
    var label = "arrangement "+String(ARRANGEMENT)+" rows "+String(rows)+" outputs "+String(n)+" inputs "+String(k)
    var batch = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*n)
    batch.enqueue_fill(-123)
    _launch[ARRANGEMENT,HAS_BIAS](ctx,x,w,b,batch,0,rows,n,k)
    comptime if decode_arrangement_reordered(ARRANGEMENT):
        var own = ctx.enqueue_create_buffer[DType.bfloat16]((rows+2)*n)
        own.enqueue_fill(-123)
        for row in range(rows):
            var input = TileTensor(x.unsafe_ptr().unsafe_offset(row*k),row_major(1,k))
            var weight = TileTensor(w,row_major(n,k))
            var output = TileTensor(own.unsafe_ptr().unsafe_offset((row+1)*n),row_major(1,n))
            comptime if HAS_BIAS:
                enqueue_linear_decode_rows_apple_gpu[ARRANGEMENT](ctx,input,weight,TileTensor(b,row_major(n)),output)
            else:
                enqueue_linear_decode_rows_apple_gpu[ARRANGEMENT](ctx,input,weight,output)
        _same(own,batch,label)
    else:
        _same(solo,batch,label)
    if len(reference):
        _within_bound(batch,reference,rows,n,k,label)


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
        # Three rows also check every arrangement against the FP64 summation bound.
        var reference = _reference(x,w,b,rows,n,k,HAS_BIAS) if rows == 3 else List[Float64]()
        comptime for arrangement in range(FIRST, LAST):
            _arranged[arrangement,HAS_BIAS](ctx,x,w,b,solo,rows,n,k,reference)


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
    _widths[True, 9, 11](ctx,1150,520)
    var x = ctx.enqueue_create_buffer[DType.bfloat16](2*512)
    var w = ctx.enqueue_create_buffer[DType.bfloat16](1150*512)
    var y = ctx.enqueue_create_buffer[DType.bfloat16](2*1150)
    _fill(x,1)
    _fill(w,2)
    y.enqueue_fill(-123)
    var input = TileTensor(x,row_major(2,512))
    var weight = TileTensor(w,row_major(1150,512))
    var output = TileTensor(y,row_major(2,1150))
    # Arrangements 3, 5-8 and 11 fix the width; 4-8 and 11 compute four columns at a time.
    comptime for arrangement in range(3, 9):
        with assert_raises():
            enqueue_linear_decode_rows_apple_gpu[arrangement](ctx,input,weight,output)
    with assert_raises():
        enqueue_linear_decode_rows_apple_gpu[11](ctx,input,weight,output)
    with y.map_to_host() as mapped:
        for i in range(len(y)):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i].cast[DType.float32](), Float32(-123))


def test_raw_offset_arrangements_need_contiguous_aligned_rows() raises:
    """Arrangements 8 and 11 address input and weight rows by raw offset; 8 loads four values at once."""
    var ctx = DeviceContext()
    var x = ctx.enqueue_create_buffer[DType.bfloat16](4*1792+1)
    var w = ctx.enqueue_create_buffer[DType.bfloat16](8*1792)
    var y = ctx.enqueue_create_buffer[DType.bfloat16](4*8)
    var exact = ctx.enqueue_create_buffer[DType.bfloat16](4*8)
    _fill(x,1)
    _fill(w,2)
    y.enqueue_fill(-123)
    var input = TileTensor(x,row_major(4,896))
    var weight = TileTensor(w,row_major(8,896))
    var output = TileTensor(y,row_major(4,8))
    # Rows of 896 values, 1,792 apart; and a contiguous input one element into its buffer.
    var padded_input = TileTensor(x,row_major(4,1792)).tile[4,896](0,0)
    var padded_weight = TileTensor(w,row_major(8,1792)).tile[8,896](0,0)
    var shifted = TileTensor(x.unsafe_ptr().unsafe_offset(1),row_major(4,896))
    with assert_raises(): enqueue_linear_decode_rows_apple_gpu[8](ctx,padded_input,weight,output)
    with assert_raises(): enqueue_linear_decode_rows_apple_gpu[8](ctx,input,padded_weight,output)
    with assert_raises(): enqueue_linear_decode_rows_apple_gpu[11](ctx,padded_input,weight,output)
    with assert_raises(): enqueue_linear_decode_rows_apple_gpu[11](ctx,input,padded_weight,output)
    with assert_raises(): enqueue_linear_decode_rows_apple_gpu[8](ctx,shifted,weight,output)
    with y.map_to_host() as mapped:
        for i in range(len(y)):
            assert_equal(mapped.unsafe_ptr()[unsafe_offset=i].cast[DType.float32](), Float32(-123))
    # Scalar loads need no more than the element's alignment: 11 accepts the shifted input and equals 5,
    # which indexes through its layout and equals the one-row kernel.
    enqueue_linear_decode_rows_apple_gpu[11](ctx,shifted,weight,output)
    enqueue_linear_decode_rows_apple_gpu[5](ctx,shifted,weight,TileTensor(exact,row_major(4,8)))
    _same(exact,y,"arrangement 11 on a shifted input")


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
    model.forward(ctx, StepBatch.sequence(ids, 0, [_block(s)], CONTEXT), pool, baseline_plan(length, length))


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


def _steps_equal_solo[ARRANGEMENT: Int = DECODE_PROJECTION](ctx: DeviceContext, sequences: Int,
                                                          steps: Int = DECODE_STEPS) raises:
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
    for step in range(steps):
        batched.forward[False, ARRANGEMENT](ctx, _decode(sequences, step, lengths), batched_pool,
                                            configured_plan(DECODER_FUSED_DECODE, sequences, longest + step + 1,
                                                            sequences))
        var tokens = batched.greedy_tokens(ctx)
        assert_equal(len(tokens), sequences)
        assert_equal(batched.last_route.sequences, sequences)
        # Launches do not depend on the number of sequences.
        assert_equal(batched.last_route.decode_launches, 1 + 10 + 9 * (FIXTURE_LAYERS - 1) + FIXTURE_LAYERS + 3)
        for s in range(sequences):
            var p = lengths[s] + step
            solo.forward[False, ARRANGEMENT](ctx, StepBatch.sequence([_token(s, p)], p, [_block(s)], CONTEXT), solo_pool,
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


def test_exact_and_reordered_steps_equal_each_sequence_decoded_alone() raises:
    """A reordered arrangement also decodes one sequence in its own order, so batched still equals solo.

    Exact arrangement 5, which decode parity pins, is checked here too, whichever arrangement is the default.
    """
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    for sequences in [3, 16]:
        _steps_equal_solo[5](ctx, sequences, 4)
    comptime for arrangement in range(8, 11):
        for sequences in [3, 16]:
            _steps_equal_solo[arrangement](ctx, sequences, 4)


def _arranged_step[ARRANGEMENT: Int](ctx: DeviceContext, mut model: QwenModel, mut pool: KVPool, sequences: Int,
                                     lengths: List[Int], longest: Int) raises -> List[Int]:
    for s in range(sequences):
        pool.truncate(_block(s), lengths[s])
    model.forward[False, ARRANGEMENT](ctx, _decode(sequences, 0, lengths), pool,
                                      configured_plan(DECODER_FUSED_DECODE, sequences, longest + 1, sequences))
    assert_equal(model.last_route.decode_launches, 1 + 10 + 9 * (FIXTURE_LAYERS - 1) + FIXTURE_LAYERS + 3)
    return model.greedy_tokens[True](ctx)


def _same_arrangement[ARRANGEMENT: Int](ctx: DeviceContext, mut model: QwenModel, mut pool: KVPool, sequences: Int,
                                        lengths: List[Int], longest: Int, expected: List[Int],
                                        mut logits: DeviceBuffer[DType.bfloat16], mut norms: DeviceBuffer[DType.bfloat16],
                                        mut storage: DeviceBuffer[DType.bfloat16]) raises:
    var selected = _arranged_step[ARRANGEMENT](ctx, model, pool, sequences, lengths, longest)
    var label = "arrangement " + String(ARRANGEMENT) + " sequences " + String(sequences)
    _same(logits, model.logits, label + " logits")
    _same(norms, model.normalized, label + " final norms")
    _same(storage, pool.storage, label + " pool")
    for s in range(sequences):
        assert_equal(selected[s], expected[s])


def test_arrangements_give_identical_batched_steps() raises:
    """Exact arrangements change how rows share work only: each reproduces arrangement 0's bytes."""
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    var model = _fixture_model(ctx, 32)
    var pool = KVPool(ctx, BLOCKS, CONTEXT, model.kv_geometry())
    _poison(pool.storage)
    var lengths = List[Int]()
    for s in range(32):
        lengths.append(_prefix(s))
        _prefill(ctx, model, pool, s, _prefix(s))
    var logits = ctx.enqueue_create_buffer[DType.bfloat16](len(model.logits))
    var norms = ctx.enqueue_create_buffer[DType.bfloat16](len(model.normalized))
    var storage = ctx.enqueue_create_buffer[DType.bfloat16](len(pool.storage))
    for sequences in [2, 3, 8, 16, 32]:
        var longest = 0
        for s in range(sequences):
            longest = max(longest, lengths[s])
        var expected = _arranged_step[0](ctx, model, pool, sequences, lengths, longest)
        ctx.enqueue_copy(dst_buf=logits, src_buf=model.logits)
        ctx.enqueue_copy(dst_buf=norms, src_buf=model.normalized)
        ctx.enqueue_copy(dst_buf=storage, src_buf=pool.storage)
        comptime for arrangement in range(1, DECODE_ARRANGEMENTS):
            comptime if not decode_arrangement_reordered(arrangement):
                _same_arrangement[arrangement](ctx, model, pool, sequences, lengths, longest, expected,
                                               logits, norms, storage)
    # Observed selection records every host mark in order.
    for i in range(1, 10):
        assert_true(model.observation[i] >= model.observation[i - 1] or i == 6)


def _paged_prefill(ctx: DeviceContext, mut model: QwenModel, mut pool: KVPool, s: Int, length: Int,
                   table: List[Int], configuration: Int) raises:
    """A sequence's prefix in chunks of at most MAX_PREFIX rows, through one prefill configuration."""
    var offset = 0
    while offset < length:
        var rows = min(MAX_PREFIX, length - offset)
        var ids = List[Int]()
        for p in range(offset, offset + rows):
            ids.append(_token(s, p))
        model.forward(ctx, StepBatch.sequence(ids, offset, table, pool.block_size), pool,
                      configured_plan(configuration, rows, offset + rows))
        offset += rows


def _same_sequence(pool: KVPool, table: List[Int], one: KVPool, block: Int, length: Int, label: String) raises:
    """Every written K and V row of a paged sequence equals its one-block copy, in every layer."""
    var size = pool.block_size
    for layer in range(FIXTURE_LAYERS):
        for kv in range(2):
            var reference = one.view(block, layer, kv)
            with reference.map_to_host() as r:
                for b in range(ceildiv(length, size)):
                    var view = pool.view(table[b], layer, kv)
                    with view.map_to_host() as m:
                        for slot in range(min(size, length - b * size)):
                            for head in range(2):
                                var start = (head * size + slot) * 64 if pool.head_major else (slot * 2 + head) * 64
                                var row = (b * size + slot) * 128 + head * 64
                                for d in range(64):
                                    if (bitcast[DType.uint16](m.unsafe_ptr()[unsafe_offset=start + d])
                                            != bitcast[DType.uint16](r.unsafe_ptr()[unsafe_offset=row + d])):
                                        raise Error(label + ": layer " + String(layer) + " position "
                                                    + String(b * size + slot) + " differs")


def _paged_steps[HEAD_MAJOR: Bool](ctx: DeviceContext, size: Int) raises:
    """Eight sequences in blocks of `size` slots against one block per sequence, decoded alone.

    Prefixes end before, at and after the 32- and 64-slot boundaries, and six
    decode steps cross them, up to the fixture's 65 positions. Each sequence
    prefills through configuration 0, 2, 3 or 21. Tables interleave the
    sequences' blocks in reverse, as sequences growing in turn would.
    """
    var prefixes: List[Int] = [1, 8, 29, 31, 32, 36, 53, 59]
    var configurations: List[Int] = [0, 2, 3, 21]
    var sequences = len(prefixes)
    var steps = 6
    var width = ceildiv(CONTEXT, size)
    var model = _fixture_model(ctx, sequences)
    var solo = _fixture_model(ctx, 1)
    var pool = KVPool(ctx, sequences * width + 2, size, model.kv_geometry(), HEAD_MAJOR)
    var one = KVPool(ctx, sequences, CONTEXT, solo.kv_geometry())
    _poison(pool.storage)
    _poison(one.storage)
    var tables = List[Int](capacity=sequences * width)
    for s in range(sequences):
        for b in range(width):
            tables.append(sequences * width - 1 - (b * sequences + s))
    var label = String(size) + "-slot " + ("head-major" if HEAD_MAJOR else "slot-major")
    for s in range(sequences):
        var table = List[Int](capacity=width)
        for b in range(width):
            table.append(tables[s * width + b])
        _paged_prefill(ctx, model, pool, s, prefixes[s], table, configurations[s % 4])
        _paged_prefill(ctx, solo, one, s, prefixes[s], [s], configurations[s % 4])
        _same_sequence(pool, table, one, s, prefixes[s], label + " prefill of sequence " + String(s))
    for step in range(steps):
        var ids = List[Int]()
        var positions = List[Int]()
        var starts = List[Int]()
        var seq_lens = List[Int]()
        var slots = List[Int]()
        var rows = List[Int]()
        var longest = 0
        for s in range(sequences):
            var p = prefixes[s] + step
            ids.append(_token(s, p))
            positions.append(p)
            starts.append(s)
            seq_lens.append(p + 1)
            slots.append(tables[s * width + p // size] * size + p % size)
            rows.append(s)
            longest = max(longest, p + 1)
        starts.append(sequences)
        model.forward(ctx, StepBatch(ids^, positions^, starts^, sequences, seq_lens^, width, tables.copy(), slots^,
                                     rows^), pool, configured_plan(DECODER_FUSED_DECODE, sequences, longest, sequences))
        var tokens = model.greedy_tokens(ctx)
        for s in range(sequences):
            var p = prefixes[s] + step
            solo.forward(ctx, StepBatch.sequence([_token(s, p)], p, [s], CONTEXT), one,
                         configured_plan(DECODER_FUSED_DECODE, 1, p + 1))
            var at = label + " step " + String(step) + " sequence " + String(s)
            assert_equal(tokens[s], solo.greedy(ctx))
            _same_rows(model.logits, s * VOCABULARY, solo.logits, VOCABULARY, at + " logits")
            _same_rows(model.normalized, s * 896, solo.normalized, 896, at + " final norm")
            assert_equal(pool.length(tables[s * width + p // size]), p % size + 1)
    var written = 0
    for s in range(sequences):
        var table = List[Int](capacity=width)
        for b in range(width):
            table.append(tables[s * width + b])
        _same_sequence(pool, table, one, s, prefixes[s] + steps, label + " sequence " + String(s))
        written += prefixes[s] + steps
    # Only the sequences' rows changed: every other slot of every block keeps its poison.
    var changed = 0
    with pool.storage.map_to_host() as mapped:
        for i in range(len(pool.storage)):
            if bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]) != POISON:
                changed += 1
    assert_equal(changed, written * FIXTURE_LAYERS * 2 * 128)


def test_paged_sequences_equal_one_block_each() raises:
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    for size in [32, 64, 128]:
        _paged_steps[False](ctx, size)
        _paged_steps[True](ctx, size)


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
        model.forward(ctx, StepBatch.sequence(ids, 0, [s], CONTEXT), pool, baseline_plan(5 + s, 5 + s))
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
    # Several blocks of a size that is not a multiple of 32, whose 32-row tiles would straddle blocks.
    var uneven = KVPool(ctx, 8, 48, model.kv_geometry())
    with assert_raises():
        model.forward(ctx, StepBatch([1, 2], [0, 0], [0, 1, 2], 2, [1, 1], 2, [0, 1, 2, 3], [0, 96], [0, 1]),
                      uneven, fused)
    assert_equal(uneven.length(0), 0)
    _unchanged(model, pool, submitted)
    # A table wider than the model's context in 32-slot blocks.
    var narrow = KVPool(ctx, 16, 32, model.kv_geometry())
    var single = configured_plan(DECODER_FUSED_DECODE, 1, 1, 1)
    with assert_raises():
        model.forward(ctx, StepBatch([1], [0], [0, 1], 1, [1], 4, [0, 1, 2, 3], [0], [0]), narrow, single)
    # A position past the model's context.
    with assert_raises():
        model.forward(ctx, StepBatch([1], [65], [0, 1], 1, [66], 3, [4, 5, 6], [193], [0]), narrow, single)
    assert_equal(narrow.length(0), 0)
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
