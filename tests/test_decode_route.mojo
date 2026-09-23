"""The production single-row route runs end to end, exactly, without model weights.

Three layers at Qwen dimensions use the verified decoder fixture weights, so the
first, middle and last layer positions all occur. Fast decode must produce the
same bytes as the baseline route, and its fused kernels must leave the scratch
buffers of the unfused path untouched: a silent fallback would overwrite them.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.testing import TestSuite, assert_equal
from llm_mojo.models.qwen2.model import QwenModel
from decoder_layer_support import decoder_support, load_decoder, poison_decoder

comptime CASE = "h896_i4864_nq14_nk2_d64_t65_s4001_base"
comptime LAYERS = 3
comptime PREFIX = 53
comptime STEPS = 12
comptime SENTINEL = UInt16(0x42F6)


def _model(ctx: DeviceContext) raises -> QwenModel:
    var model = QwenModel.allocate(ctx, LAYERS, PREFIX + STEPS, PREFIX)
    model.embedding.enqueue_fill(0)
    model.attention.cosine.enqueue_fill(0)
    model.attention.sine.enqueue_fill(0)
    # Token i embeds fixture row i; later positions reuse the fixture tables.
    load_decoder(model.embedding, CASE, "input_X", 0, (PREFIX + STEPS) * 896)
    load_decoder(model.norm, CASE, "input_post_norm", 0, 896)
    load_decoder(model.attention.cosine, CASE, "full_cosine", 0, (PREFIX + STEPS) * 64)
    load_decoder(model.attention.sine, CASE, "full_sine", 0, (PREFIX + STEPS) * 64)
    for i in range(LAYERS):
        load_decoder(model.layers[i].attention.norm, CASE, "input_input_norm", 0, 896)
        load_decoder(model.layers[i].attention.qkv, CASE, "input_qkv", 0, 1152 * 896)
        load_decoder(model.layers[i].attention.bias, CASE, "input_bias", 0, 1152)
        load_decoder(model.layers[i].attention.output, CASE, "input_wo", 0, 896 * 896)
        load_decoder(model.layers[i].mlp.norm, CASE, "input_post_norm", 0, 896)
        load_decoder(model.layers[i].mlp.gate, CASE, "input_gate", 0, 4864 * 896)
        load_decoder(model.layers[i].mlp.up, CASE, "input_up", 0, 4864 * 896)
        load_decoder(model.layers[i].mlp.down, CASE, "input_down", 0, 896 * 4864)
        model.layers[i].cache.key.enqueue_fill(0)
        model.layers[i].cache.value.enqueue_fill(0)
    return model^


def _same(mut left: DeviceBuffer[DType.bfloat16], mut right: DeviceBuffer[DType.bfloat16],
          count: Int, label: String) raises:
    with left.map_to_host() as a:
        with right.map_to_host() as b:
            for i in range(count):
                if (bitcast[DType.uint16](a.unsafe_ptr()[unsafe_offset=i])
                        != bitcast[DType.uint16](b.unsafe_ptr()[unsafe_offset=i])):
                    raise Error(label + " differs at element " + String(i))


def _sentinels(mut buffer: DeviceBuffer[DType.bfloat16]) raises -> Int:
    var count = 0
    with buffer.map_to_host() as mapped:
        for i in range(len(buffer)):
            if bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=i]) == SENTINEL:
                count += 1
    return count


def _poison_scratch(mut model: QwenModel) raises:
    poison_decoder(model.attention.raw_query, 0)
    poison_decoder(model.attention.raw_key, 0)
    poison_decoder(model.attention.raw_value, 0)
    poison_decoder(model.attention.rotated_key, 0)
    poison_decoder(model.mlp.activated, 0)


def test_fast_decode_matches_baseline_and_runs_fused() raises:
    var support = decoder_support()
    support.verify_case(CASE)
    var ctx = DeviceContext()
    assert_equal(ctx.api(), "metal")
    var fast = _model(ctx)
    var baseline = _model(ctx)
    var prompt = List[Int]()
    for i in range(PREFIX):
        prompt.append(i)
    fast.forward(ctx, prompt, 0)
    baseline.forward(ctx, prompt, 0)
    assert_equal(fast.greedy(ctx), baseline.greedy(ctx))
    _poison_scratch(fast)
    _poison_scratch(baseline)
    for step in range(STEPS):
        # Teacher-forced fixture tokens keep both routes on identical inputs.
        var ids: List[Int] = [PREFIX + step]
        fast.forward(ctx, ids, 26, "", 1, False, True, True)
        baseline.forward(ctx, ids, 0)
        assert_equal(fast.greedy(ctx), baseline.greedy(ctx))
        _same(fast.logits, baseline.logits, 151936, "logits")
        _same(fast.normalized, baseline.normalized, 896, "final norm")
        _same(fast.mlp.output, baseline.mlp.output, 896, "final hidden state")
        for i in range(LAYERS):
            var name = "layer " + String(i)
            _same(fast.layers[i].cache.key, baseline.layers[i].cache.key, fast.capacity * 128, name + " keys")
            _same(fast.layers[i].cache.value, baseline.layers[i].cache.value, fast.capacity * 128, name + " values")
        assert_equal(fast.length, PREFIX + step + 1)
        assert_equal(fast.submitted_rows, baseline.submitted_rows)
    # Fused QKV/RoPE/cache and SiLU/multiply never write the unfused scratch.
    assert_equal(_sentinels(fast.attention.raw_query), len(fast.attention.raw_query))
    assert_equal(_sentinels(fast.attention.raw_key), len(fast.attention.raw_key))
    assert_equal(_sentinels(fast.attention.raw_value), len(fast.attention.raw_value))
    assert_equal(_sentinels(fast.attention.rotated_key), len(fast.attention.rotated_key))
    assert_equal(_sentinels(fast.mlp.activated), len(fast.mlp.activated))
    # Negative control: the baseline route overwrites each of these buffers.
    assert_equal(_sentinels(baseline.attention.raw_query) < len(baseline.attention.raw_query), True)
    assert_equal(_sentinels(baseline.attention.raw_key) < len(baseline.attention.raw_key), True)
    assert_equal(_sentinels(baseline.attention.raw_value) < len(baseline.attention.raw_value), True)
    assert_equal(_sentinels(baseline.attention.rotated_key) < len(baseline.attention.rotated_key), True)
    assert_equal(_sentinels(baseline.mlp.activated) < len(baseline.mlp.activated), True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
