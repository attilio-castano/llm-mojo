"""Where a sequence's K and V rows live in a paged, block-major pool.

The pool is one BF16 allocation of blocks, each holding every layer's K and V
rows for block_size consecutive positions of one sequence:

    slot-major  Pool[block, layer, kv, slot, head, dim]   serving phase 1's order
    head-major  Pool[block, layer, kv, head, slot, dim]

A sequence's row at position t of layer l lives in block table[t // block_size]
at slot t % block_size. Kernels that read the pool visit rows in 32-row tiles
starting at multiples of 32, or in residue classes modulo 32, so a tile never
straddles two blocks when block_size is a multiple of 32, and a kernel reads the
table once per tile or block. A sequence held in a single block may use any
block size. See docs/paged-kv-plan.md.
"""


@always_inline
def kv_row[KV_HEADS: Int, HEAD_DIM: Int, HEAD_MAJOR: Bool](
    block: Int, layer: Int, layers: Int, kv: Int, slot: Int, head: Int, block_size: Int,
) -> Int:
    """Element offset of (block, layer, kv, slot, head, dim 0); kv is 0 for K and 1 for V."""
    var region = ((block * layers + layer) * 2 + kv) * block_size * KV_HEADS * HEAD_DIM
    comptime if HEAD_MAJOR:
        return region + (head * block_size + slot) * HEAD_DIM
    else:
        return region + (slot * KV_HEADS + head) * HEAD_DIM


@always_inline
def kv_slot_stride[KV_HEADS: Int, HEAD_DIM: Int, HEAD_MAJOR: Bool]() -> Int:
    """Elements between one head's rows in consecutive slots of a block."""
    return HEAD_DIM if HEAD_MAJOR else KV_HEADS * HEAD_DIM


def validate_paged_pool[KV_HEADS: Int, HEAD_DIM: Int](
    elements: Int, layer: Int, layers: Int, block_size: Int, table_width: Int,
) raises:
    """Host checks shared by every paged launch, before it enqueues anything.

    A table wider than one block needs a block size that is a multiple of 32, so
    that no 32-row tile straddles two blocks.
    """
    if layers < 1 or layer < 0 or layer >= layers or block_size < 1 or block_size > 4096:
        raise Error("paged KV requires a valid layer and a block size of 1 to 4096")
    if elements < 1 or elements % (2 * layers * block_size * KV_HEADS * HEAD_DIM) != 0:
        raise Error("paged KV pool is not a whole number of blocks")
    if table_width < 1 or (table_width > 1 and block_size % 32 != 0):
        raise Error("paged KV tables of several blocks need a block size that is a multiple of 32")
