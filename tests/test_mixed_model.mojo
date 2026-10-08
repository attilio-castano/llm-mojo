"""Explicit mixed reference route: selection, paged isolation and operation parity.

Comparisons use identical token histories and the same route/projection for
mixed and separately submitted sequences. They do not require equality with
the Fast prefill route or a different prompt-chunk schedule.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises
from llm_mojo.layers.decoder_layer import DECODER_MIXED
from llm_mojo.models.qwen2.model import QwenModel, VOCABULARY
from llm_mojo.models.qwen2.plan import configured_plan
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVPool
from decoder_layer_support import decoder_support, load_decoder

comptime CASE = "h896_i4864_nq14_nk2_d64_t65_s4001_base"
comptime LAYERS = 3
comptime CONTEXT = 65
comptime MAX_ROWS = 40
comptime POISON = UInt16(0x7FC1)


def _model(ctx: DeviceContext, sequences: Int) raises -> QwenModel:
    var model = QwenModel.allocate(ctx, LAYERS, CONTEXT, MAX_ROWS, sequences)
    model.embedding.enqueue_fill(0)
    load_decoder(model.embedding, CASE, "input_X", 0, CONTEXT * 896)
    load_decoder(model.norm, CASE, "input_post_norm", 0, 896)
    load_decoder(model.attention.cosine, CASE, "full_cosine", 0, CONTEXT * 64)
    load_decoder(model.attention.sine, CASE, "full_sine", 0, CONTEXT * 64)
    for i in range(LAYERS):
        load_decoder(model.layers[i].attention.norm, CASE, "input_input_norm", 0, 896)
        load_decoder(model.layers[i].attention.qkv, CASE, "input_qkv", 0, 1152 * 896)
        load_decoder(model.layers[i].attention.bias, CASE, "input_bias", 0, 1152)
        load_decoder(model.layers[i].attention.output, CASE, "input_wo", 0, 896 * 896)
        load_decoder(model.layers[i].mlp.norm, CASE, "input_post_norm", 0, 896)
        load_decoder(model.layers[i].mlp.gate, CASE, "input_gate", 0, 4864 * 896)
        load_decoder(model.layers[i].mlp.up, CASE, "input_up", 0, 4864 * 896)
        load_decoder(model.layers[i].mlp.down, CASE, "input_down", 0, 896 * 4864)
    return model^


def _ids(sequence: Int, past: Int, rows: Int) -> List[Int]:
    var ids = List[Int](capacity=rows)
    for p in range(past, past + rows):
        ids.append((p + 13 * sequence) % CONTEXT)
    return ids^


def _table(sequence: Int) -> List[Int]:
    if sequence == 0:
        return [8, 4, 0]
    if sequence == 1:
        return [7, 3, 11]
    return [6, 2, 10]


def _batch(var selected: List[Int]) -> StepBatch:
    # Two singleton rows, then a five-row tail crossing a 32-slot boundary.
    return StepBatch([31, 46, 56, 57, 58, 59, 60], [31, 33, 30, 31, 32, 33, 34], [0, 1, 2, 7], 2,
        [32, 34, 35], 3, [8, 4, 0, 7, 3, 11, 6, 2, 10],
        [8 * 32 + 31, 3 * 32 + 1, 6 * 32 + 30, 6 * 32 + 31, 2 * 32, 2 * 32 + 1, 2 * 32 + 2], selected^)


def _snapshot(buffer: DeviceBuffer[DType.bfloat16], count: Int) raises -> List[UInt16]:
    var bits = List[UInt16](capacity=count)
    with buffer.map_to_host() as mapped:
        for i in range(count):
            bits.append(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]))
    return bits^


def _equals(bits: List[UInt16], first: Int, buffer: DeviceBuffer[DType.bfloat16], count: Int,
            label: String) raises:
    with buffer.map_to_host() as mapped:
        for i in range(count):
            if bits[first + i] != bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]):
                raise Error(label + " differs at element " + String(i))


def _poisoned(buffer: DeviceBuffer[DType.bfloat16], start: Int = 0) raises:
    with buffer.map_to_host() as mapped:
        for i in range(start, len(buffer)):
            assert_equal(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]), POISON)


def _rewind(mut pool: KVPool) raises:
    var prefixes: List[Int] = [31, 33, 30]
    for s in range(3):
        pool.truncate_table(_table(s), prefixes[s])


def _pools_equal_and_guarded(paged: KVPool, solo: KVPool, lengths: List[Int]) raises:
    for layer in range(LAYERS):
        for kv in range(2):
            for s in range(3):
                var expected = solo.view(s, layer, kv)
                with expected.map_to_host() as a:
                    var table = _table(s)
                    for b in range(3):
                        var actual = paged.view(table[b], layer, kv)
                        with actual.map_to_host() as m:
                            for slot in range(32):
                                for head in range(2):
                                    var target = (head * 32 + slot) * 64 if paged.head_major else (slot * 2 + head) * 64
                                    for d in range(64):
                                        var bits = bitcast[DType.uint16](m.unsafe_ptr()[unsafe_offset=target + d])
                                        var position = b * 32 + slot
                                        if position < lengths[s]:
                                            assert_equal(bits, bitcast[DType.uint16](
                                                a.unsafe_ptr()[unsafe_offset=position * 128 + head * 64 + d]))
                                        else:
                                            assert_equal(bits, POISON)
            for block in [1, 5, 9]:
                var spare = paged.view(block, layer, kv)
                _poisoned(spare)


def _run[HEAD_MAJOR: Bool](ctx: DeviceContext) raises:
    var model = _model(ctx, 3)
    var solo = _model(ctx, 1)
    var paged = KVPool(ctx, 12, 32, model.kv_geometry(), HEAD_MAJOR)
    var single = KVPool(ctx, 3, CONTEXT, solo.kv_geometry())
    paged.storage.enqueue_fill(bitcast[DType.bfloat16](POISON))
    single.storage.enqueue_fill(bitcast[DType.bfloat16](POISON))
    var prefixes: List[Int] = [31, 33, 30]
    for s in range(3):
        var prefix = _ids(s, 0, prefixes[s])
        var a = StepBatch.sequence(prefix, 0, _table(s), 32)
        a.logits_rows = List[Int]()
        model.forward(ctx, a, paged, configured_plan(DECODER_MIXED, prefixes[s], prefixes[s]))
        assert_equal(model.sampled_rows, 0)
        assert_equal(len(model.greedy_tokens(ctx)), 0)
        var b = StepBatch.sequence(prefix, 0, [s], CONTEXT)
        b.logits_rows = List[Int]()
        solo.forward(ctx, b, single, configured_plan(DECODER_MIXED, prefixes[s], prefixes[s]))
    var batch = _batch([0, 1, 6])
    model.forward(ctx, batch, paged, configured_plan(DECODER_MIXED, 7, 35, 3))
    assert_equal(model.sampled_rows, 3)
    assert_equal(model.last_route.sequences, 3)
    assert_equal(model.last_route.configuration, DECODER_MIXED)
    var tokens = model.greedy_tokens(ctx)
    assert_equal(len(tokens), 3)
    var logits = _snapshot(model.logits, 3 * VOCABULARY)
    for s in range(3):
        var rows = 1 if s < 2 else 5
        var ids = _ids(s, prefixes[s], rows)
        solo.forward(ctx, StepBatch.sequence(ids, prefixes[s], [s], CONTEXT), single,
            configured_plan(DECODER_MIXED, rows, prefixes[s] + rows))
        assert_equal(tokens[s], solo.greedy(ctx))
        _equals(logits, s * VOCABULARY, solo.logits, VOCABULARY, "same-reference mixed logits")
    _pools_equal_and_guarded(paged, single, [32, 34, 35])
    # Re-execute identical rows while sampling only the second singleton and tail.
    _rewind(paged)
    model.logits.enqueue_fill(bitcast[DType.bfloat16](POISON))
    model.normalized.enqueue_fill(bitcast[DType.bfloat16](POISON))
    model.forward(ctx, _batch([1, 6]), paged, configured_plan(DECODER_MIXED, 7, 35, 3))
    var subset = model.greedy_tokens(ctx)
    assert_equal(model.sampled_rows, 2)
    assert_equal(len(subset), 2)
    assert_equal(subset[0], tokens[1])
    assert_equal(subset[1], tokens[2])
    _equals(logits, VOCABULARY, model.logits, 2 * VOCABULARY, "selected logits in row order")
    _poisoned(model.logits, 2 * VOCABULARY)
    _poisoned(model.normalized, 2 * 896)
    # An unfinished prompt need not materialize or select vocabulary logits.
    _rewind(paged)
    model.logits.enqueue_fill(bitcast[DType.bfloat16](POISON))
    model.normalized.enqueue_fill(bitcast[DType.bfloat16](POISON))
    model.forward(ctx, _batch(List[Int]()), paged, configured_plan(DECODER_MIXED, 7, 35, 3))
    assert_equal(model.sampled_rows, 0)
    assert_equal(len(model.greedy_tokens(ctx)), 0)
    with assert_raises():
        _ = model.greedy(ctx)
    _poisoned(model.logits)
    _poisoned(model.normalized)
    _pools_equal_and_guarded(paged, single, [32, 34, 35])
    # A shape consisting only of singletons includes one-token prefill remainders.
    var all_singletons = StepBatch([32, 47, 61], [32, 34, 35], [0, 1, 2, 3], 3, [33, 35, 36], 3,
        [8, 4, 0, 7, 3, 11, 6, 2, 10], [4 * 32, 3 * 32 + 2, 2 * 32 + 3], [0, 2])
    model.forward(ctx, all_singletons, paged, configured_plan(DECODER_MIXED, 3, 36, 3))
    var selected = model.greedy_tokens(ctx)
    assert_equal(len(selected), 2)
    var final_logits = _snapshot(model.logits, 2 * VOCABULARY)
    var before: List[Int] = [32, 34, 35]
    for s in range(3):
        solo.forward(ctx, StepBatch.sequence(_ids(s, before[s], 1), before[s], [s], CONTEXT), single,
            configured_plan(DECODER_MIXED, 1, before[s] + 1))
        if s == 0 or s == 2:
            var index = 0 if s == 0 else 1
            assert_equal(selected[index], solo.greedy(ctx))
            _equals(final_logits, index * VOCABULARY, solo.logits, VOCABULARY, "singleton selective logits")
    _pools_equal_and_guarded(paged, single, [33, 35, 36])
    # A structurally valid StepBatch with two multi-row tails is rejected before enqueue.
    var submitted = model.submitted_rows
    var previous_samples = model.sampled_rows
    var previous_pool = _snapshot(paged.storage, len(paged.storage))
    var bad = StepBatch([33, 34, 48, 49], [33, 34, 35, 36], [0, 2, 4], 0, [35, 37], 3,
        [8, 4, 0, 7, 3, 11], [4 * 32 + 1, 4 * 32 + 2, 3 * 32 + 3, 3 * 32 + 4], [1, 3])
    with assert_raises():
        model.forward(ctx, bad, paged, configured_plan(DECODER_MIXED, 4, 37, 2))
    assert_equal(model.valid, True)
    assert_equal(model.submitted_rows, submitted)
    assert_equal(model.sampled_rows, previous_samples)
    _equals(previous_pool, 0, paged.storage, len(paged.storage), "rejected multi-tail pool")


def test_mixed_reference_model_selection_and_isolation() raises:
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    _run[False](ctx)
    _run[True](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
