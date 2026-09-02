"""BF16 grouped-query attention reference and Apple GPU baseline."""

from layout import TensorLayout, TileTensor
from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import ceildiv, exp, rsqrt
from std.sys.info import is_apple_gpu


comptime ATTENTION_APPLE_GPU_BLOCK_SIZE = 128


def _validate_grouped_query_attention[
    QueryLayout: TensorLayout,
    KeyLayout: TensorLayout,
    ValueLayout: TensorLayout,
    ScratchLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    query: TileTensor[DType.bfloat16, QueryLayout, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KeyLayout, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, ValueLayout, MutAnyOrigin],
    scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    """Validate the shared host and enqueue contract."""

    comptime assert query.flat_rank == 3, "query must have rank 3"
    comptime assert key.flat_rank == 3, "key must have rank 3"
    comptime assert value.flat_rank == 3, "value must have rank 3"
    comptime assert scratch.flat_rank == 3, "scratch must have rank 3"
    comptime assert output.flat_rank == 3, "output must have rank 3"

    var query_rows = Int(query.dim[0]())
    var query_heads = Int(query.dim[1]())
    var head_dim = Int(query.dim[2]())
    var key_value_rows = Int(key.dim[0]())
    var key_value_heads = Int(key.dim[1]())
    if (
        query_rows <= 0
        or query_heads <= 0
        or head_dim <= 0
        or key_value_rows <= 0
        or key_value_heads <= 0
    ):
        raise Error("attention dimensions must be positive")
    if query_rows > key_value_rows:
        raise Error("query rows must be an active key/value suffix")
    if query_heads % key_value_heads != 0:
        raise Error("query heads must divide evenly across key/value heads")
    if Int(key.dim[2]()) != head_dim:
        raise Error("key head dimension must match query head dimension")
    if (
        Int(value.dim[0]()) != key_value_rows
        or Int(value.dim[1]()) != key_value_heads
        or Int(value.dim[2]()) != head_dim
    ):
        raise Error("value shape must match key shape")
    if (
        Int(scratch.dim[0]()) != query_rows
        or Int(scratch.dim[1]()) != query_heads
        or Int(scratch.dim[2]()) != key_value_rows
    ):
        raise Error(
            "scratch shape must be (query rows, query heads, key/value rows)"
        )
    if (
        Int(output.dim[0]()) != query_rows
        or Int(output.dim[1]()) != query_heads
        or Int(output.dim[2]()) != head_dim
    ):
        raise Error("output shape must match query shape")


def grouped_query_attention_reference[
    QueryLayout: TensorLayout,
    KeyLayout: TensorLayout,
    ValueLayout: TensorLayout,
    ScratchLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    query: TileTensor[DType.bfloat16, QueryLayout, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KeyLayout, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, ValueLayout, MutAnyOrigin],
    scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    """Run serial causal GQA, leaving BF16 probabilities in scratch.

    The caller owns five non-overlapping storage regions.
    """

    comptime assert query.flat_rank == 3, "query must have rank 3"
    comptime assert key.flat_rank == 3, "key must have rank 3"
    comptime assert value.flat_rank == 3, "value must have rank 3"
    comptime assert scratch.flat_rank == 3, "scratch must have rank 3"
    comptime assert output.flat_rank == 3, "output must have rank 3"
    _validate_grouped_query_attention(query, key, value, scratch, output)

    var query_rows = Int(query.dim[0]())
    var query_heads = Int(query.dim[1]())
    var head_dim = Int(query.dim[2]())
    var key_value_rows = Int(key.dim[0]())
    var key_value_heads = Int(key.dim[1]())
    var queries_per_key_value_head = query_heads // key_value_heads
    var past = key_value_rows - query_rows
    var scale = rsqrt(Float32(head_dim))

    # Stage 1: materialize scaled QK scores. Future positions are zeroed so
    # scratch has deterministic contents even though softmax never reads them.
    for row in range(query_rows):
        var visible_key_count = past + row + 1
        for query_head in range(query_heads):
            var key_value_head = query_head // queries_per_key_value_head
            for key_position in range(key_value_rows):
                var score: Scalar[DType.float32] = 0.0
                if key_position < visible_key_count:
                    for dimension in range(head_dim):
                        var query_value = rebind[Scalar[DType.bfloat16]](
                            query[row, query_head, dimension]
                        )
                        var key_value = rebind[Scalar[DType.bfloat16]](
                            key[key_position, key_value_head, dimension]
                        )
                        score += (
                            query_value.cast[DType.float32]()
                            * key_value.cast[DType.float32]()
                        )
                    score *= scale
                var stored_score = score.cast[DType.bfloat16]()
                scratch[row, query_head, key_position] = rebind[
                    scratch.ElementType
                ](stored_score)

    # Stage 2: stable causal softmax in FP32, materialized back to BF16.
    for row in range(query_rows):
        var visible_key_count = past + row + 1
        for query_head in range(query_heads):
            var first_score = rebind[Scalar[DType.bfloat16]](
                scratch[row, query_head, 0]
            )
            var max_score = first_score.cast[DType.float32]()
            for key_position in range(1, visible_key_count):
                var stored_score = rebind[Scalar[DType.bfloat16]](
                    scratch[row, query_head, key_position]
                )
                var score = stored_score.cast[DType.float32]()
                if score > max_score:
                    max_score = score

            var denominator: Scalar[DType.float32] = 0.0
            for key_position in range(visible_key_count):
                var stored_score = rebind[Scalar[DType.bfloat16]](
                    scratch[row, query_head, key_position]
                )
                denominator += exp(
                    stored_score.cast[DType.float32]() - max_score
                )

            for key_position in range(key_value_rows):
                var probability: Scalar[DType.float32] = 0.0
                if key_position < visible_key_count:
                    var stored_score = rebind[Scalar[DType.bfloat16]](
                        scratch[row, query_head, key_position]
                    )
                    probability = (
                        exp(stored_score.cast[DType.float32]() - max_score)
                        / denominator
                    )
                var stored_probability = probability.cast[DType.bfloat16]()
                scratch[row, query_head, key_position] = rebind[
                    scratch.ElementType
                ](stored_probability)

    # Stage 3: reduce probability times V into each output element.
    for row in range(query_rows):
        var visible_key_count = past + row + 1
        for query_head in range(query_heads):
            var key_value_head = query_head // queries_per_key_value_head
            for dimension in range(head_dim):
                var accumulator: Scalar[DType.float32] = 0.0
                for key_position in range(visible_key_count):
                    var probability = rebind[Scalar[DType.bfloat16]](
                        scratch[row, query_head, key_position]
                    )
                    var value_element = rebind[Scalar[DType.bfloat16]](
                        value[key_position, key_value_head, dimension]
                    )
                    accumulator += (
                        probability.cast[DType.float32]()
                        * value_element.cast[DType.float32]()
                    )
                var result = accumulator.cast[DType.bfloat16]()
                output[row, query_head, dimension] = rebind[output.ElementType](
                    result
                )


def _grouped_query_attention_qk_apple_gpu_kernel[
    QueryLayout: TensorLayout,
    KeyLayout: TensorLayout,
    ScratchLayout: TensorLayout,
](
    query: TileTensor[DType.bfloat16, QueryLayout, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KeyLayout, MutAnyOrigin],
    scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],
    query_rows: Int32,
    key_value_rows: Int32,
    query_heads: Int32,
    key_value_heads: Int32,
    head_dim: Int32,
):
    """Map one scaled QK score to one Apple GPU thread."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert query.flat_rank == 3, "query must have rank 3"
    comptime assert key.flat_rank == 3, "key must have rank 3"
    comptime assert scratch.flat_rank == 3, "scratch must have rank 3"

    var query_row_count = Int(query_rows)
    var key_value_row_count = Int(key_value_rows)
    var query_head_count = Int(query_heads)
    var key_value_head_count = Int(key_value_heads)
    var dimension_count = Int(head_dim)
    var score_count = query_row_count * query_head_count * key_value_row_count
    var flat_score = global_idx.x
    if flat_score < score_count:
        var key_position = flat_score % key_value_row_count
        var row_head = flat_score // key_value_row_count
        var query_head = row_head % query_head_count
        var row = row_head // query_head_count
        var past = key_value_row_count - query_row_count
        var visible_key_count = past + row + 1
        var score: Scalar[DType.float32] = 0.0
        if key_position < visible_key_count:
            var queries_per_key_value_head = (
                query_head_count // key_value_head_count
            )
            var key_value_head = query_head // queries_per_key_value_head
            for dimension in range(dimension_count):
                var query_value = rebind[Scalar[DType.bfloat16]](
                    query[row, query_head, dimension]
                )
                var key_value = rebind[Scalar[DType.bfloat16]](
                    key[key_position, key_value_head, dimension]
                )
                score += (
                    query_value.cast[DType.float32]()
                    * key_value.cast[DType.float32]()
                )
            score *= rsqrt(Float32(dimension_count))
        var stored_score = score.cast[DType.bfloat16]()
        scratch[row, query_head, key_position] = rebind[scratch.ElementType](
            stored_score
        )


def _grouped_query_attention_softmax_apple_gpu_kernel[
    ScratchLayout: TensorLayout,
](
    scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],
    query_rows: Int32,
    key_value_rows: Int32,
    query_heads: Int32,
):
    """Map one stable causal softmax row to one Apple GPU thread."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert scratch.flat_rank == 3, "scratch must have rank 3"

    var query_row_count = Int(query_rows)
    var key_value_row_count = Int(key_value_rows)
    var query_head_count = Int(query_heads)
    var row_head_count = query_row_count * query_head_count
    var row_head = global_idx.x
    if row_head < row_head_count:
        var query_head = row_head % query_head_count
        var row = row_head // query_head_count
        var past = key_value_row_count - query_row_count
        var visible_key_count = past + row + 1
        var first_score = rebind[Scalar[DType.bfloat16]](
            scratch[row, query_head, 0]
        )
        var max_score = first_score.cast[DType.float32]()
        for key_position in range(1, visible_key_count):
            var stored_score = rebind[Scalar[DType.bfloat16]](
                scratch[row, query_head, key_position]
            )
            var score = stored_score.cast[DType.float32]()
            if score > max_score:
                max_score = score

        var denominator: Scalar[DType.float32] = 0.0
        for key_position in range(visible_key_count):
            var stored_score = rebind[Scalar[DType.bfloat16]](
                scratch[row, query_head, key_position]
            )
            denominator += exp(stored_score.cast[DType.float32]() - max_score)

        for key_position in range(key_value_row_count):
            var probability: Scalar[DType.float32] = 0.0
            if key_position < visible_key_count:
                var stored_score = rebind[Scalar[DType.bfloat16]](
                    scratch[row, query_head, key_position]
                )
                probability = (
                    exp(stored_score.cast[DType.float32]() - max_score)
                    / denominator
                )
            var stored_probability = probability.cast[DType.bfloat16]()
            scratch[row, query_head, key_position] = rebind[
                scratch.ElementType
            ](stored_probability)


def _grouped_query_attention_pv_apple_gpu_kernel[
    ValueLayout: TensorLayout,
    ScratchLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    value: TileTensor[DType.bfloat16, ValueLayout, MutAnyOrigin],
    scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
    query_rows: Int32,
    key_value_rows: Int32,
    query_heads: Int32,
    key_value_heads: Int32,
    head_dim: Int32,
):
    """Map one probability-times-V reduction to one Apple GPU thread."""

    comptime assert is_apple_gpu(), "kernel requires an Apple GPU target"
    comptime assert value.flat_rank == 3, "value must have rank 3"
    comptime assert scratch.flat_rank == 3, "scratch must have rank 3"
    comptime assert output.flat_rank == 3, "output must have rank 3"

    var query_row_count = Int(query_rows)
    var key_value_row_count = Int(key_value_rows)
    var query_head_count = Int(query_heads)
    var key_value_head_count = Int(key_value_heads)
    var dimension_count = Int(head_dim)
    var output_count = query_row_count * query_head_count * dimension_count
    var flat_output = global_idx.x
    if flat_output < output_count:
        var dimension = flat_output % dimension_count
        var row_head = flat_output // dimension_count
        var query_head = row_head % query_head_count
        var row = row_head // query_head_count
        var queries_per_key_value_head = (
            query_head_count // key_value_head_count
        )
        var key_value_head = query_head // queries_per_key_value_head
        var past = key_value_row_count - query_row_count
        var visible_key_count = past + row + 1
        var accumulator: Scalar[DType.float32] = 0.0
        for key_position in range(visible_key_count):
            var probability = rebind[Scalar[DType.bfloat16]](
                scratch[row, query_head, key_position]
            )
            var value_element = rebind[Scalar[DType.bfloat16]](
                value[key_position, key_value_head, dimension]
            )
            accumulator += (
                probability.cast[DType.float32]()
                * value_element.cast[DType.float32]()
            )
        var result = accumulator.cast[DType.bfloat16]()
        output[row, query_head, dimension] = rebind[output.ElementType](result)


def enqueue_grouped_query_attention_apple_gpu[
    QueryLayout: TensorLayout,
    KeyLayout: TensorLayout,
    ValueLayout: TensorLayout,
    ScratchLayout: TensorLayout,
    OutputLayout: TensorLayout,
](
    context: DeviceContext,
    query: TileTensor[DType.bfloat16, QueryLayout, MutAnyOrigin],
    key: TileTensor[DType.bfloat16, KeyLayout, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, ValueLayout, MutAnyOrigin],
    scratch: TileTensor[DType.bfloat16, ScratchLayout, MutAnyOrigin],
    output: TileTensor[DType.bfloat16, OutputLayout, MutAnyOrigin],
) raises:
    """Enqueue materialized GQA without allocating or synchronizing.

    The caller owns five non-overlapping storage regions and must keep them
    alive through their last queued use.
    """

    comptime assert query.flat_rank == 3, "query must have rank 3"
    comptime assert key.flat_rank == 3, "key must have rank 3"
    comptime assert value.flat_rank == 3, "value must have rank 3"
    comptime assert scratch.flat_rank == 3, "scratch must have rank 3"
    comptime assert output.flat_rank == 3, "output must have rank 3"
    _validate_grouped_query_attention(query, key, value, scratch, output)
    if context.api() != "metal":
        raise Error("Apple GPU grouped-query attention requires Metal")

    var query_rows = Int(query.dim[0]())
    var query_heads = Int(query.dim[1]())
    var head_dim = Int(query.dim[2]())
    var key_value_rows = Int(key.dim[0]())
    var key_value_heads = Int(key.dim[1]())
    var score_count = query_rows * query_heads * key_value_rows
    var row_head_count = query_rows * query_heads
    var output_count = query_rows * query_heads * head_dim

    comptime qk_kernel = _grouped_query_attention_qk_apple_gpu_kernel[
        QueryLayout, KeyLayout, ScratchLayout
    ]
    context.enqueue_function[qk_kernel](
        query,
        key,
        scratch,
        Int32(query_rows),
        Int32(key_value_rows),
        Int32(query_heads),
        Int32(key_value_heads),
        Int32(head_dim),
        grid_dim=ceildiv(score_count, ATTENTION_APPLE_GPU_BLOCK_SIZE),
        block_dim=ATTENTION_APPLE_GPU_BLOCK_SIZE,
    )

    comptime softmax_kernel = (
        _grouped_query_attention_softmax_apple_gpu_kernel[ScratchLayout]
    )
    context.enqueue_function[softmax_kernel](
        scratch,
        Int32(query_rows),
        Int32(key_value_rows),
        Int32(query_heads),
        grid_dim=ceildiv(row_head_count, ATTENTION_APPLE_GPU_BLOCK_SIZE),
        block_dim=ATTENTION_APPLE_GPU_BLOCK_SIZE,
    )

    comptime pv_kernel = _grouped_query_attention_pv_apple_gpu_kernel[
        ValueLayout, ScratchLayout, OutputLayout
    ]
    context.enqueue_function[pv_kernel](
        value,
        scratch,
        output,
        Int32(query_rows),
        Int32(key_value_rows),
        Int32(query_heads),
        Int32(key_value_heads),
        Int32(head_dim),
        grid_dim=ceildiv(output_count, ATTENTION_APPLE_GPU_BLOCK_SIZE),
        block_dim=ATTENTION_APPLE_GPU_BLOCK_SIZE,
    )
