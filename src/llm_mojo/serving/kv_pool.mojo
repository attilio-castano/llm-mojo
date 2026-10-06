"""KV storage owned outside the model: one block-major allocation and its written slots.

Pool[block, layer, kv, slot, head, dim] in BF16, or Pool[block, layer, kv, head,
slot, dim] when head_major; kernels/paged_kv.mojo maps a row to its offset. A
block holds block_size consecutive positions of one sequence in every layer,
and `written` counts the slots of each block whose writes, in all layers, have
been enqueued. The model checks each step against those counts and advances
them; the pool itself submits no work except its allocation.

The pool knows no model: its geometry below the block comes from the model it
serves, such as `QwenModel.kv_geometry()`.
"""
from max.gpu.host import DeviceBuffer, DeviceContext


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
    var head_major: Bool
    var storage: DeviceBuffer[DType.bfloat16]
    var written: List[Int]

    def __init__(out self, ctx: DeviceContext, blocks: Int, block_size: Int, geometry: KVGeometry,
                 head_major: Bool = False) raises:
        if (blocks < 1 or block_size < 1 or block_size > 4096 or geometry.layers < 1
                or geometry.kv_heads < 1 or geometry.head_dim < 1):
            raise Error("invalid KV pool geometry")
        self.blocks = blocks
        self.block_size = block_size
        self.geometry = geometry
        self.head_major = head_major
        var region = block_size * geometry.kv_heads * geometry.head_dim
        self.storage = ctx.enqueue_create_buffer[DType.bfloat16](blocks * geometry.layers * 2 * region)
        self.written = List[Int](capacity=blocks)
        for _ in range(blocks):
            self.written.append(0)

    def region(self) -> Int:
        """Elements of one block's K or V in one layer."""
        return self.block_size * self.geometry.kv_heads * self.geometry.head_dim

    def key_offset(self, block: Int, layer: Int) raises -> Int:
        """Element offset of a block's K in one layer; its V follows directly."""
        if block < 0 or block >= self.blocks or layer < 0 or layer >= self.geometry.layers:
            raise Error("KV pool view out of range")
        return 2 * (block * self.geometry.layers + layer) * self.region()

    def view(self, block: Int, layer: Int, kv: Int) raises -> DeviceBuffer[DType.bfloat16]:
        """A block's K (kv 0) or V (kv 1) in one layer, for diagnostics and tests."""
        if kv < 0 or kv > 1:
            raise Error("KV pool view out of range")
        return self.storage.create_sub_buffer[DType.bfloat16](self.key_offset(block, layer) + kv * self.region(),
                                                             self.region())

    def length(self, block: Int) raises -> Int:
        """Slots of a block whose writes have been enqueued."""
        if block < 0 or block >= self.blocks:
            raise Error("KV pool block out of range")
        return self.written[block]

    def truncate(mut self, block: Int, length: Int) raises:
        """Shorten a block's written slots without clearing them.

        The caller guarantees that no queued work still depends on the removed
        slots. Rejection leaves the count unchanged.
        """
        if length < 0 or length > self.length(block):
            raise Error("truncation cannot extend a block")
        self.written[block] = length

    def truncate_table(mut self, table: List[Int], length: Int) raises:
        """Shorten the sequence whose blocks `table` lists to `length` positions.

        Every block past the new length becomes empty, so it can hold another
        sequence's first rows. Rejection leaves every count unchanged.
        """
        var counts = List[Int](capacity=len(table))
        for b in range(len(table)):
            var count = min(max(length - b * self.block_size, 0), self.block_size)
            if length < 0 or count > self.length(table[b]):
                raise Error("truncation cannot extend a block")
            counts.append(count)
        for b in range(len(table)):
            self.written[table[b]] = counts[b]

    def reset(mut self, ctx: DeviceContext) raises:
        """Finish pending use, then make every slot of every block logically absent."""
        ctx.synchronize()
        for block in range(self.blocks):
            self.written[block] = 0

    def relayout(mut self, ctx: DeviceContext, block_size: Int, head_major: Bool) raises:
        """Hold the same storage in blocks of `block_size` slots, in the given order within a block.

        Pending use finishes first, and every block becomes empty: the bytes stay,
        but no slot means anything until it is written again. The pool's slots
        must split into whole blocks. Rejection changes nothing.
        """
        var slots = self.blocks * self.block_size
        if block_size < 1 or block_size > 4096 or slots % block_size != 0:
            raise Error("invalid KV pool layout")
        ctx.synchronize()
        self.blocks = slots // block_size
        self.block_size = block_size
        self.head_major = head_major
        self.written = List[Int](length=self.blocks, fill=0)
