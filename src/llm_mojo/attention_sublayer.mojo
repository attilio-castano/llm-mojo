"""One inspectable Qwen attention sublayer; allocation is separate from enqueue.

See docs/attention-sublayer.md for rounding, lifetime and cache contracts.
The caller initializes weights and rotary tables, retains all objects, and uses
one context/stream through completion. Rotary tables are explicit model inputs.
"""
from layout import TensorLayout, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import global_idx
from std.math import ceildiv
from llm_mojo.rms_norm import enqueue_rms_norm_apple_gpu
from llm_mojo.linear import enqueue_linear_apple_gpu
from llm_mojo.rope import enqueue_rope_apple_gpu
from llm_mojo.attention import enqueue_grouped_query_attention_apple_gpu
from llm_mojo.attention_decode import (
    enqueue_grouped_query_attention_decode_apple_gpu,
)
from llm_mojo.attention_prefill import (
    enqueue_grouped_query_attention_prefill_apple_gpu,
)
from llm_mojo.residual import enqueue_residual_apple_gpu


struct AttentionWeights(Movable):
    """Q/K/V source-compatible regions in one weight allocation, plus Wo/norm.
    """

    var query_heads: Int
    var kv_heads: Int
    var head_dim: Int
    var hidden: Int
    var qkv: DeviceBuffer[DType.bfloat16]
    var bias: DeviceBuffer[DType.bfloat16]
    var output: DeviceBuffer[DType.bfloat16]
    var norm: DeviceBuffer[DType.bfloat16]

    def __init__(
        out self,
        ctx: DeviceContext,
        query_heads: Int = 14,
        kv_heads: Int = 2,
        head_dim: Int = 64,
    ) raises:
        if (
            query_heads <= 0
            or kv_heads <= 0
            or query_heads % kv_heads != 0
            or head_dim <= 0
            or head_dim % 2 != 0
        ):
            raise Error("invalid attention head dimensions")
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.hidden = query_heads * head_dim
        var n = self.hidden + 2 * kv_heads * head_dim
        self.qkv = ctx.enqueue_create_buffer[DType.bfloat16](n * self.hidden)
        self.bias = ctx.enqueue_create_buffer[DType.bfloat16](n)
        self.output = ctx.enqueue_create_buffer[DType.bfloat16](
            self.hidden * self.hidden
        )
        self.norm = ctx.enqueue_create_buffer[DType.bfloat16](self.hidden)


struct AttentionCache(Movable):
    """Fixed storage; length counts the prefix whose writes have been enqueued.
    """

    var capacity: Int
    var length: Int
    var kv_heads: Int
    var head_dim: Int
    var key: DeviceBuffer[DType.bfloat16]
    var value: DeviceBuffer[DType.bfloat16]

    def __init__(
        out self,
        ctx: DeviceContext,
        capacity: Int,
        kv_heads: Int = 2,
        head_dim: Int = 64,
    ) raises:
        if capacity < 1 or capacity > 4096 or kv_heads <= 0 or head_dim <= 0:
            raise Error("invalid cache capacity or shape")
        self.capacity = capacity
        self.length = 0
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.key = ctx.enqueue_create_buffer[DType.bfloat16](
            capacity * kv_heads * head_dim
        )
        self.value = ctx.enqueue_create_buffer[DType.bfloat16](
            capacity * kv_heads * head_dim
        )

    def reset(mut self, ctx: DeviceContext) raises:
        """Finish pending use before making all stored entries logically absent.
        """
        ctx.synchronize()
        self.length = 0


struct AttentionWorkspace(Movable):
    """Reusable intermediates and caller-initialized BF16 rotary tables.

    The caller fills cosine/sine[capacity,head_dim] before enqueue, like weights.
    Table generation is outside this operation and its timing boundary. The
    compatibility baseline uses tables from the pinned upstream implementation.
    """

    var max_rows: Int
    var capacity: Int
    var query_heads: Int
    var kv_heads: Int
    var head_dim: Int
    var materialized: Bool
    var normalized: DeviceBuffer[DType.bfloat16]
    var raw_query: DeviceBuffer[DType.bfloat16]
    var raw_key: DeviceBuffer[DType.bfloat16]
    var raw_value: DeviceBuffer[DType.bfloat16]
    var query: DeviceBuffer[DType.bfloat16]
    var rotated_key: DeviceBuffer[DType.bfloat16]
    var attention: DeviceBuffer[DType.bfloat16]
    var projected: DeviceBuffer[DType.bfloat16]
    var output: DeviceBuffer[DType.bfloat16]
    var packed: DeviceBuffer[DType.bfloat16]
    var scratch: DeviceBuffer[DType.bfloat16]
    var fp32_scratch: DeviceBuffer[DType.float32]
    var fp32_materialized: Bool
    var split: DeviceBuffer[DType.float32]
    var cosine: DeviceBuffer[DType.bfloat16]
    var sine: DeviceBuffer[DType.bfloat16]

    def __init__(
        out self,
        ctx: DeviceContext,
        max_rows: Int,
        capacity: Int,
        query_heads: Int = 14,
        kv_heads: Int = 2,
        head_dim: Int = 64,
        materialized: Bool = False,
        fp32_materialized: Bool = True,
    ) raises:
        if (
            max_rows < 1
            or max_rows > capacity
            or capacity > 4096
            or query_heads <= 0
            or kv_heads <= 0
            or query_heads % kv_heads != 0
            or head_dim <= 0
            or head_dim % 2 != 0
        ):
            raise Error("invalid workspace shape")
        self.max_rows = max_rows
        self.capacity = capacity
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.materialized = materialized
        self.fp32_materialized = fp32_materialized
        var h = query_heads * head_dim
        var k = kv_heads * head_dim
        self.normalized = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * h
        )
        self.raw_query = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * h)
        self.raw_key = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * k)
        self.raw_value = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * k)
        self.query = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * h)
        self.rotated_key = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * k
        )
        self.attention = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * h)
        self.projected = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * h)
        self.output = ctx.enqueue_create_buffer[DType.bfloat16](max_rows * h)
        self.packed = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * (h + 2 * k)
        )
        self.scratch = ctx.enqueue_create_buffer[DType.bfloat16](
            max_rows * query_heads * capacity if materialized else 1
        )
        self.fp32_scratch = ctx.enqueue_create_buffer[DType.float32](
            max_rows * query_heads * capacity if fp32_materialized else 1
        )
        self.split = ctx.enqueue_create_buffer[DType.float32](14 * 64 * 66)
        self.cosine = ctx.enqueue_create_buffer[DType.bfloat16](
            capacity * head_dim
        )
        self.sine = ctx.enqueue_create_buffer[DType.bfloat16](
            capacity * head_dim
        )


def _append[
    KL: TensorLayout, VL: TensorLayout, CL: TensorLayout
](
    key: TileTensor[DType.bfloat16, KL, MutAnyOrigin],
    value: TileTensor[DType.bfloat16, VL, MutAnyOrigin],
    cache_key: TileTensor[DType.bfloat16, CL, MutAnyOrigin],
    cache_value: TileTensor[DType.bfloat16, CL, MutAnyOrigin],
    rows: Int32,
    width: Int32,
    past: Int32,
):
    comptime assert (
        key.flat_rank == 2
        and value.flat_rank == 2
        and cache_key.flat_rank == 2
        and cache_value.flat_rank == 2
    )
    var i = global_idx.x
    if i < Int(rows) * Int(width):
        var r = i // Int(width)
        var d = i % Int(width)
        cache_key[Int(past) + r, d] = key[r, d]
        cache_value[Int(past) + r, d] = value[r, d]


def enqueue_attention_sublayer[
    XL: TensorLayout
](
    ctx: DeviceContext,
    mut weights: AttentionWeights,
    mut cache: AttentionCache,
    mut work: AttentionWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
    route: Int = 3,
) raises -> Int:
    """Enqueue one sublayer and advance cache length; return the launched route.

    Default route 3 materializes FP32 scores/probabilities, with BF16 I/O.
    Route 0 materializes BF16 GQA. Route 1 uses G32 decode / original MMA prefill;
    route 2 uses split64 H4 decode / rolled-QK MMA prefill. Surroundings match.
    Routes 0-2 remain explicit BF16 compatibility comparisons.
    X must not overlap any writable workspace/cache region. Read output from
    work.output only after completion and before its next overwrite.
    """
    comptime assert x.flat_rank == 2
    var r = Int(x.dim[0]())
    var h = weights.hidden
    var d = weights.head_dim
    var nq = weights.query_heads
    var nk = weights.kv_heads
    var k = nk * d
    var p = cache.length
    var t = p + r
    if route < 0 or route > 3:
        raise Error("unknown attention sublayer route")
    if (
        r <= 0
        or r > work.max_rows
        or Int(x.dim[1]()) != h
        or p < 0
        or t > cache.capacity
        or t > work.capacity
    ):
        raise Error("invalid sublayer rows, hidden size or cache overflow")
    if (
        cache.kv_heads != nk
        or cache.head_dim != d
        or work.query_heads != nq
        or work.kv_heads != nk
        or work.head_dim != d
    ):
        raise Error("sublayer storage dimensions do not agree")
    if route == 0 and not work.materialized:
        raise Error("materialized route requires probability scratch")
    if route == 3 and not work.fp32_materialized:
        raise Error("FP32 route requires FP32 probability scratch")
    if (route == 1 or route == 2) and (nq != 14 or nk != 2 or d != 64):
        raise Error("optimized GQA requires Qwen dimensions")
    if ctx.api() != "metal":
        raise Error("attention sublayer requires Metal")

    var normal = TileTensor(work.normalized, row_major(r, h))
    enqueue_rms_norm_apple_gpu(
        ctx, x, TileTensor(weights.norm, row_major(h)), normal
    )
    var q2 = TileTensor(work.raw_query, row_major(r, h))
    var k2 = TileTensor(work.raw_key, row_major(r, k))
    var v2 = TileTensor(work.raw_value, row_major(r, k))
    enqueue_linear_apple_gpu(
        ctx,
        normal,
        TileTensor(weights.qkv, row_major(h, h)),
        TileTensor(weights.bias, row_major(h)),
        q2,
    )
    enqueue_linear_apple_gpu(
        ctx,
        normal,
        TileTensor(
            weights.qkv.unsafe_ptr().unsafe_offset(h * h), row_major(k, h)
        ),
        TileTensor(weights.bias.unsafe_ptr().unsafe_offset(h), row_major(k)),
        k2,
    )
    enqueue_linear_apple_gpu(
        ctx,
        normal,
        TileTensor(
            weights.qkv.unsafe_ptr().unsafe_offset((h + k) * h), row_major(k, h)
        ),
        TileTensor(
            weights.bias.unsafe_ptr().unsafe_offset(h + k), row_major(k)
        ),
        v2,
    )
    var q = TileTensor(work.query, row_major(r, nq, d))
    var c = TileTensor(work.cosine, row_major(work.capacity, d))
    var s = TileTensor(work.sine, row_major(work.capacity, d))
    enqueue_rope_apple_gpu(
        ctx, TileTensor(work.raw_query, row_major(r, nq, d)), c, s, q, p
    )
    enqueue_rope_apple_gpu(
        ctx,
        TileTensor(work.raw_key, row_major(r, nk, d)),
        c,
        s,
        TileTensor(work.rotated_key, row_major(r, nk, d)),
        p,
    )
    var kr = TileTensor(work.rotated_key, row_major(r, k))
    var ck = TileTensor(cache.key, row_major(cache.capacity, k))
    var cv = TileTensor(cache.value, row_major(cache.capacity, k))
    ctx.enqueue_function[
        _append[type_of(kr.layout), type_of(v2.layout), type_of(ck.layout)]
    ](
        kr,
        v2,
        ck,
        cv,
        Int32(r),
        Int32(k),
        Int32(p),
        grid_dim=ceildiv(r * k, 128),
        block_dim=128,
    )
    var keys = TileTensor(cache.key, row_major(t, nk, d))
    var values = TileTensor(cache.value, row_major(t, nk, d))
    var a = TileTensor(work.attention, row_major(r, nq, d))
    if route == 0:
        enqueue_grouped_query_attention_apple_gpu(
            ctx,
            q,
            keys,
            values,
            TileTensor(work.scratch, row_major(r, nq, t)),
            a,
        )
    elif route == 3:
        enqueue_grouped_query_attention_apple_gpu(
            ctx, q, keys, values,
            TileTensor(work.fp32_scratch, row_major(r, nq, t)), a,
        )
    elif r == 1:
        if route == 1:
            enqueue_grouped_query_attention_decode_apple_gpu[32, 1, 1](
                ctx,
                q,
                keys,
                values,
                a,
                TileTensor(work.split, row_major(14, 1, 66)),
            )
        else:
            enqueue_grouped_query_attention_decode_apple_gpu[1, 4, 64](
                ctx,
                q,
                keys,
                values,
                a,
                TileTensor(work.split, row_major(14, 64, 66)),
            )
    else:
        if route == 1:
            enqueue_grouped_query_attention_prefill_apple_gpu[32, 32, MMA=True](
                ctx, q, keys, values, a
            )
        else:
            enqueue_grouped_query_attention_prefill_apple_gpu[
                32, 32, MMA=True, SCHEDULE=2
            ](ctx, q, keys, values, a)
    var projected = TileTensor(work.projected, row_major(r, h))
    enqueue_linear_apple_gpu(
        ctx,
        TileTensor(work.attention, row_major(r, h)),
        TileTensor(weights.output, row_major(h, h)),
        projected,
    )
    enqueue_residual_apple_gpu(
        ctx, x, projected, TileTensor(work.output, row_major(r, h))
    )
    cache.length = t
    return route
