"""The single-layer Fast/Deterministic policy lookup from the decoder policy campaign.

studies/decoder_layer/policies.md measured these cells for one decoder layer in hot
and ring24 reuse modes; tests/fixtures/decoder_policies.json pins them. The model
never calls this lookup: its measured choices live in models/qwen2/plan.mojo. The
deterministic lookup is the starting point for full-model schedule determinism.
"""
from layout import TensorLayout, TileTensor
from max.gpu.host import DeviceContext
from llm_mojo.layers.attention_sublayer import AttentionWeights, AttentionCache, AttentionWorkspace
from llm_mojo.layers.mlp import MLPWeights, MLPWorkspace
from llm_mojo.layers.decoder_layer import enqueue_decoder_layer_configuration


struct DecoderCache[DETERMINISTIC: Bool](Movable):
    """A decoder cache whose execution policy is fixed in its type.

    Rebuilding under a different policy starts an empty prefix. The contained
    AttentionCache remains available to the existing explicit storage API.
    """
    var storage: AttentionCache
    var reuse_layers: Int

    def __init__(out self, ctx: DeviceContext, capacity: Int,
                 reuse_layers: Int = 1) raises:
        if reuse_layers != 1 and reuse_layers != 24:
            raise Error("decoder reuse mode must be hot or ring24")
        self.storage = AttentionCache(ctx, capacity)
        self.reuse_layers = reuse_layers

    def reset(mut self, ctx: DeviceContext) raises:
        self.storage.reset(ctx)

    def prefill_splits(self) -> Int:
        """Required allocation capacity, independent of the next call's shape."""
        return 1 if Self.DETERMINISTIC else 8


def decoder_policy_configuration(deterministic: Bool, rows: Int,
                                  total_rows: Int, reuse_layers: Int = 1) raises -> Int:
    """Exact measured cells, with an explicit unmeasured-workload fallback."""
    if rows < 1 or rows > total_rows or total_rows > 4096:
        raise Error("invalid decoder policy workload")
    if reuse_layers != 1 and reuse_layers != 24:
        raise Error("decoder reuse mode must be hot or ring24")
    if deterministic:
        if rows == 16 and total_rows == 16:
            return 22
        if rows == 256 and total_rows == 256:
            return 22
        if rows == 4096 and total_rows == 4096:
            return 22
        if rows == 16 and total_rows == 256:
            return 22
        if rows == 64 and total_rows == 4096:
            return 22
        return 20
    if rows == 16 and total_rows == 256 and reuse_layers == 1:
        return 21
    if rows == 64 and total_rows == 4096:
        return 3
    return 0


def enqueue_decoder_layer_policy[DETERMINISTIC: Bool, XL: TensorLayout](
    ctx: DeviceContext, mut aw: AttentionWeights,
    mut cache: DecoderCache[DETERMINISTIC], mut attention: AttentionWorkspace,
    mut mw: MLPWeights, mut mlp: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin],
) raises -> Int:
    var reuse_layers = cache.reuse_layers
    return _enqueue_decoder_layer_policy_storage[DETERMINISTIC](ctx, aw,
        cache.storage, attention, mw, mlp, x, reuse_layers)


def _enqueue_decoder_layer_policy_storage[DETERMINISTIC: Bool, XL: TensorLayout](
    ctx: DeviceContext, mut aw: AttentionWeights, mut cache: AttentionCache,
    mut attention: AttentionWorkspace, mut mw: MLPWeights, mut mlp: MLPWorkspace,
    x: TileTensor[DType.bfloat16, XL, MutAnyOrigin], reuse_layers: Int,
) raises -> Int:
    # Shared with the numerical harness, which observes every stored stage.
    var rows = Int(x.dim[0]())
    var variant = decoder_policy_configuration(DETERMINISTIC, rows,
        cache.length + rows, reuse_layers)
    return enqueue_decoder_layer_configuration(ctx, aw, cache,
        attention, mw, mlp, x, variant)
