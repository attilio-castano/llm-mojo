"""KV storage owned outside the model: one block-major allocation and its views.

Pool[block, layer, kv, slot, head, dim] in BF16. Each (block, layer) pair has
AttentionCache views of its K and V ranges, created once, so existing attention
code reads and appends through them unchanged. A view's `length` counts the
rows whose writes have been enqueued for that layer; the model requires all
layer views of a block to agree before it submits work.

The pool knows no model: its geometry below the block comes from the model it
serves, such as `QwenModel.kv_geometry()`.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from llm_mojo.layers.attention_sublayer import AttentionCache


@fieldwise_init
struct KVGeometry(ImplicitlyCopyable, Movable):
    """What one token stores: K and V rows of kv_heads x head_dim in each layer."""
    var layers: Int
    var kv_heads: Int
    var head_dim: Int

    def __eq__(self, other: Self) -> Bool:
        return self.layers == other.layers and self.kv_heads == other.kv_heads and self.head_dim == other.head_dim

    def __ne__(self, other: Self) -> Bool:
        return not self == other


struct KVPool(Movable):
    var blocks: Int
    var block_size: Int
    var geometry: KVGeometry
    var storage: DeviceBuffer[DType.bfloat16]
    var caches: List[AttentionCache]

    def __init__(out self, ctx: DeviceContext, blocks: Int, block_size: Int, geometry: KVGeometry) raises:
        if (blocks < 1 or block_size < 1 or block_size > 4096 or geometry.layers < 1
                or geometry.kv_heads < 1 or geometry.head_dim < 1):
            raise Error("invalid KV pool geometry")
        self.blocks = blocks
        self.block_size = block_size
        self.geometry = geometry
        var rows = block_size * geometry.kv_heads * geometry.head_dim
        var views = blocks * geometry.layers
        self.storage = ctx.enqueue_create_buffer[DType.bfloat16](views * 2 * rows)
        self.caches = List[AttentionCache](capacity=views)
        for view in range(views):
            self.caches.append(AttentionCache(
                self.storage.create_sub_buffer[DType.bfloat16](2 * view * rows, rows),
                self.storage.create_sub_buffer[DType.bfloat16]((2 * view + 1) * rows, rows),
                block_size, geometry.kv_heads, geometry.head_dim))

    def index(self, block: Int, layer: Int) raises -> Int:
        """Position of a (block, layer) view in `caches`."""
        if block < 0 or block >= self.blocks or layer < 0 or layer >= self.geometry.layers:
            raise Error("KV pool view out of range")
        return block * self.geometry.layers + layer

    def key_offset(self, block: Int, layer: Int) raises -> Int:
        """Element offset of a K view in `storage`; its V view follows directly."""
        return 2 * self.index(block, layer) * self.block_size * self.geometry.kv_heads * self.geometry.head_dim

    def length(self, block: Int) raises -> Int:
        """Rows written for the block's first layer; forward checks all layers agree."""
        return self.caches[self.index(block, 0)].length

    def truncate(mut self, block: Int, length: Int) raises:
        """Shorten a block's logical length without clearing rows.

        The caller guarantees that no queued work still depends on the removed
        rows. Rejection leaves every layer's length unchanged.
        """
        if length < 0:
            raise Error("truncation length must be nonnegative")
        for layer in range(self.geometry.layers):
            if length > self.caches[self.index(block, layer)].length:
                raise Error("truncation cannot extend a block")
        for layer in range(self.geometry.layers):
            self.caches[self.index(block, layer)].length = length

    def reset(mut self, ctx: DeviceContext) raises:
        """Finish pending use, then make every row of every block logically absent."""
        ctx.synchronize()
        for i in range(len(self.caches)):
            self.caches[i].length = 0
