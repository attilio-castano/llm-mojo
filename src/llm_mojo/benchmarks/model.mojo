"""Real Qwen decode on the Fast route: fixed history, normal stream, optional host observations.

The token-profile modes (bench, verify, profile) time one sequence. The batch
mode and the MODEL_BATCH_PROFILE build time decode steps of B sequences: the
batch-size study (1c) pairs row tiles, the projection study (1d) pairs exact
batched projection arrangements, the reordered study (1e) pairs arrangements
with other summation orders, whose accuracy the accuracy mode records, and the
addressing check (1f) pairs raw-pointer and vector loads with arrangement 5
(docs/history/batched-decode-plan.md). The paged KV study (2d,
docs/paged-kv-plan.md) pairs KV layouts in the batch mode, its paged-prefill
mode and the MODEL_PAGED_PROFILE build.
Completed decode experiments (fusion, selection, buffer swap, composition,
projection arrangement, scheduling and launch probes) are replay-only; their
collectors exist through commit edb610a. See studies/model_generation/README.md.
"""
from std.sys import argv, is_defined, get_defined_int, get_defined_string
from std.time import sleep
from std.math import ceildiv
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceGraph, DeviceGraphBuilder
from layout import TileTensor, TensorLayout, row_major
from max.gpu import global_idx
from llm_mojo.kernels.linear import DECODE_ARRANGEMENTS, decode_arrangement_reordered, enqueue_linear_decode_rows_apple_gpu
from llm_mojo.kernels.paged_kv import kv_row
from llm_mojo.models.qwen2.model import HEAD_DIM, KV_HEADS, KV_WIDTH, LAYERS, QwenModel, save_bf16
from llm_mojo.models.qwen2.plan import fast_plan
from llm_mojo.models.qwen2.tokenizer import Tokenizer, TokenizerWorkspace
from llm_mojo.runtime.clock import now
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.blocks import BlockManager
from llm_mojo.serving.kv_pool import KVPool


def rewind(mut model: QwenModel, mut kv: KVPool, prefix: Int) raises:
    # Previous greedy readback completed model computation. Only the logical suffix is rewound.
    model.submitted_rows = prefix * 24
    kv.truncate(0, prefix)


def snapshot(mut model: QwenModel, kv: KVPool, path: String) raises:
    save_bf16(model.logits,path+"-logits.bin",151936)
    for layer in range(24):
        save_bf16(kv.view(0,layer,0),path+"-k"+String(layer)+".bin",4096*128)
        save_bf16(kv.view(0,layer,1),path+"-v"+String(layer)+".bin",4096*128)


def poison_outputs(mut model: QwenModel, mut kv: KVPool, prefix: Int) raises:
    """Untimed verification: stale logits/cache appends must not pass parity."""
    var sentinel = bitcast[DType.bfloat16](UInt16(0x7FC0))
    model.logits.enqueue_fill(sentinel)
    model.mlp.activated.enqueue_fill(sentinel)
    model.mlp.gated.enqueue_fill(sentinel)
    for layer in range(24):
        var key = kv.view(0,layer,0)
        with key.map_to_host() as mapped:
            for column in range(128):
                mapped.unsafe_ptr()[unsafe_offset=prefix*128+column] = sentinel
        var value = kv.view(0,layer,1)
        with value.map_to_host() as mapped:
            for column in range(128):
                mapped.unsafe_ptr()[unsafe_offset=prefix*128+column] = sentinel


def step[OBSERVE: Bool](mut model: QwenModel, mut kv: KVPool, ctx: DeviceContext, ids: List[Int]) raises -> Int:
    """One single-row Fast decode step: configuration 26, GPU argmax, swap, fused norms."""
    var cached = kv.length(0)
    model.forward[OBSERVE](ctx,StepBatch.sequence(ids,cached,[0],kv.block_size),kv,fast_plan(1,cached+1,ctx.name()))
    return model.greedy[OBSERVE](ctx)


comptime BATCH_POOL = 64
comptime BATCH_MIXED = 32


def frozen_history(tables: String) raises -> List[Int]:
    var tokenizer = Tokenizer(tables)
    var work = TokenizerWorkspace()
    var text = String()
    for _ in range(200):
        text += "A train travels sixty kilometers in forty-five minutes. Explain how to calculate its average speed, keeping track of distance, time, and units. The passengers compare their calculations and check each step.\n"
    return tokenizer.encode(text,work)


def batch_contexts(context: Int) -> List[Int]:
    """Cached tokens per sequence: the declared context, or 64 to 3968 spread evenly (context 0)."""
    var result = List[Int](capacity=BATCH_POOL)
    for s in range(BATCH_POOL):
        if context == 0:
            result.append(64 + (min(s,BATCH_MIXED-1)*(3968-64))//(BATCH_MIXED-1))
        else:
            result.append(context)
    return result^


def batch_setup(ctx: DeviceContext, mut model: QwenModel, mut kv: KVPool, history: List[Int], longest: Int) raises:
    """Prefill the frozen history into block 0 with Fast chunks, then copy block 0 to every block."""
    kv.storage.enqueue_fill(0)
    var offset = 0
    while offset < longest:
        var count = min(256,longest-offset)
        var chunk = List[Int](capacity=count)
        for i in range(count):
            chunk.append(history[offset+i])
        model.forward(ctx,StepBatch.sequence(chunk,offset,[0],kv.block_size),kv,fast_plan(count,offset+count,ctx.name()))
        offset += count
    ctx.synchronize()
    var block = len(kv.storage)//kv.blocks
    var first = kv.storage.create_sub_buffer[DType.bfloat16](0,block)
    for b in range(1,kv.blocks):
        var target = kv.storage.create_sub_buffer[DType.bfloat16](b*block,block)
        ctx.enqueue_copy(dst_buf=target,src_buf=first)
    ctx.synchronize()


def batch_rewind(mut model: QwenModel, mut kv: KVPool, contexts: List[Int], sequences: Int) raises:
    # Earlier readback completed every step. Only logical lengths are rewound.
    model.submitted_rows = 0
    for s in range(sequences):
        kv.written[s] = contexts[s]


def batch_step[OBSERVE: Bool, ARRANGEMENT: Int](mut model: QwenModel, mut kv: KVPool, ctx: DeviceContext,
                                                tokens: List[Int], contexts: List[Int], sequences: Int) raises -> List[Int]:
    """One decode token for each of the first `sequences` blocks, then one token per sequence."""
    var ids = List[Int](capacity=sequences)
    var positions = List[Int](capacity=sequences)
    var starts = List[Int](capacity=sequences+1)
    var seq_lens = List[Int](capacity=sequences)
    var blocks = List[Int](capacity=sequences)
    var slots = List[Int](capacity=sequences)
    var rows = List[Int](capacity=sequences)
    var longest = 0
    for s in range(sequences):
        ids.append(tokens[s])
        positions.append(contexts[s])
        starts.append(s)
        seq_lens.append(contexts[s]+1)
        blocks.append(s)
        slots.append(s*kv.block_size+contexts[s])
        rows.append(s)
        longest = max(longest,contexts[s]+1)
    starts.append(sequences)
    var batch = StepBatch(ids^,positions^,starts^,sequences,seq_lens^,1,blocks^,slots^,rows^)
    model.forward[OBSERVE, ARRANGEMENT](ctx,batch,kv,fast_plan(sequences,longest,ctx.name(),sequences))
    return model.greedy_tokens[OBSERVE](ctx)


def batch_arms(study: String) raises -> List[Int]:
    """The control arrangement, then each comparison's candidate; -1 is the control with host marks.

    Comparison 0 pairs the control with itself. size: 1c's tiles 4, 8 and 16 and
    the observed arm. projections: 1d's screen of arrangements 3-6 against 0.
    confirm:A: 1d's confirmation of A. reordered: 1e's screen of 7-10 against 5.
    reordered-confirm:A: 1e's confirmation of A. addressing: 1f's arrangements 11 and 8
    against 5.
    """
    if study == "size":
        return [0, 1, 2, -1]
    if study == "projections":
        return [0, 3, 4, 5, 6]
    if study == "reordered":
        return [5, 7, 8, 9, 10]
    if study == "addressing":
        return [5, 11, 8]
    var parts = study.split(":")
    if len(parts) == 2:
        var arrangement = Int(String(parts[1]))
        if String(parts[0]) == "confirm" and arrangement >= 3 and arrangement <= 6:
            return [0, arrangement]
        if String(parts[0]) == "reordered-confirm" and arrangement >= 7 and arrangement < DECODE_ARRANGEMENTS:
            return [5, arrangement]
    raise Error("unknown batch study")


def batch_arm(arrangement: Int, mut model: QwenModel, mut kv: KVPool, ctx: DeviceContext,
              tokens: List[Int], contexts: List[Int], sequences: Int) raises -> List[Int]:
    if arrangement == -1:
        return batch_step[True, 0](model,kv,ctx,tokens,contexts,sequences)
    comptime for a in range(DECODE_ARRANGEMENTS):
        if arrangement == a:
            return batch_step[False, a](model,kv,ctx,tokens,contexts,sequences)
    raise Error("unknown projection arrangement")


def batch_check(model: QwenModel, selected: List[Int], expected: List[Int], sequences: Int) raises:
    if (len(selected) != sequences or model.submitted_rows != 24*sequences
            or model.last_route.decode_launches != 245 or model.last_route.sequences != sequences):
        raise Error("batched step changed its route or accounting")
    for s in range(sequences):
        if selected[s] != expected[s]:
            raise Error("batched step changed a sequence's token")


def batch_bench(prepared: String, tables: String, context: Int, first: Int, study: String) raises:
    """Paired arms per batch size: the control against itself, then each of the study's candidates.

    Tiles 4, 8 and 16 are projection arrangements 0, 1 and 2.
    """
    var arms = batch_arms(study)
    var ctx = DeviceContext()
    if ctx.api() != "metal" or ctx.name() != "Apple M4 Pro":
        raise Error("study requires Apple M4 Pro / Metal")
    var history = frozen_history(tables)
    var contexts = batch_contexts(context)
    var sizes: List[Int] = [1, 2, 4, 8, 16, 32, 64]
    if context == 0:
        sizes = [BATCH_MIXED]
    var longest = 0
    for c in contexts:
        longest = max(longest,c)
    if len(history) < longest+BATCH_POOL+1:
        raise Error("insufficient frozen token history")
    var tokens = List[Int](capacity=BATCH_POOL)
    for s in range(BATCH_POOL):
        tokens.append(history[contexts[s]+s])
    var model = QwenModel(ctx,prepared,4096,256,BATCH_POOL)
    var kv = KVPool(ctx,BATCH_POOL,4096,model.kv_geometry())
    batch_setup(ctx,model,kv,history,longest)
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("context:",context,"pool blocks:",BATCH_POOL)
    print("study:",study)
    var records = String()
    for index in range(len(sizes)):
        var sequences = sizes[len(sizes)-1-index] if first == 1 else sizes[index]
        # Each comparison's arrangement records its own untimed tokens. Exact arrangements
        # must select the control's; a reordered one may differ, and the count is reported.
        var expected = List[List[Int]]()
        for position in range(len(arms)):
            batch_rewind(model,kv,contexts,sequences)
            var reference = batch_arm(arms[position] if arms[position] != -1 else arms[0],model,kv,ctx,
                                      tokens,contexts,sequences)
            batch_check(model,reference,reference,sequences)
            var differing = 0
            for s in range(sequences):
                if reference[s] != expected[0][s] if position else False:
                    differing += 1
            if differing and not decode_arrangement_reordered(arms[position]):
                raise Error("an exact arrangement changed a sequence's token")
            print("tokens:",sequences,arms[position],differing)
            expected.append(reference^)
        print("sequences:",sequences,"first token:",expected[0][0],"last token:",expected[0][sequences-1])
        for position in range(len(arms)):
            var comparison = len(arms)-1-position if first == 1 else position
            for arm_index in range(2):
                var arm = (first+arm_index)%2
                for sample in range(20):
                    batch_rewind(model,kv,contexts,sequences)
                    var start = now()
                    var arrangement = arms[comparison] if arm == 1 else arms[0]
                    var selected = batch_arm(arrangement,model,kv,ctx,tokens,contexts,sequences)
                    var elapsed = now()-start
                    batch_check(model,selected,expected[comparison if arm == 1 else 0],sequences)
                    if sample >= 10:
                        records += ("BATCH "+String(sequences)+" "+String(comparison)+" "+String(arm)+" "
                                    +String(sample-10)+" "+String(elapsed))
                        if arrangement == -1:
                            for i in range(10):
                                records += " "+String(model.observation[i]-start)
                        records += "\n"
    print(records,end="")
    print("BATCH_COMPLETE")


def _accuracy_values(mut buffer: DeviceBuffer[DType.bfloat16], seed: Int) raises:
    """The kernel tests' values: mixed signs and exponents, with every fifth a signed zero,
    a subnormal or a neighbour of one."""
    var edges: List[UInt16] = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x3F80, 0x3F81, 0xBF7F]
    with buffer.map_to_host() as mapped:
        for i in range(len(buffer)):
            var raw = UInt32((i*1664525+seed*1013904223) & 0xffffffff)
            var bits = UInt16((raw >> 16) & 0x807f) | UInt16((119+Int((raw >> 7)%UInt32(16))) << 7)
            if i % 5 == seed % 5:
                bits = edges[(i // 5) % len(edges)]
            mapped.unsafe_ptr()[unsafe_offset=i] = bitcast[DType.bfloat16](bits)


def _bf16_ulp(value: Float64) -> Float64:
    """One BF16 unit in the last place at value, with the smallest normal's below it."""
    var exponent = max(Int((bitcast[DType.uint64](value) >> 52) & 0x7FF) - 1023, -126)
    return bitcast[DType.float64](UInt64(exponent - 7 + 1023) << 52)


def _accuracy_shape[HAS_BIAS: Bool](ctx: DeviceContext, rows: Int, n: Int, k: Int) raises:
    """Each screened arrangement's errors against the FP64 sum of the same BF16 operands, in BF16 ulps."""
    var x = ctx.enqueue_create_buffer[DType.bfloat16](rows*k)
    var w = ctx.enqueue_create_buffer[DType.bfloat16](n*k)
    var b = ctx.enqueue_create_buffer[DType.bfloat16](n)
    var y = ctx.enqueue_create_buffer[DType.bfloat16](rows*n)
    _accuracy_values(x,rows)
    _accuracy_values(w,n)
    _accuracy_values(b,k)
    var exact = List[Float64](capacity=rows*n)
    with x.map_to_host() as xs:
        with w.map_to_host() as ws:
            with b.map_to_host() as bs:
                for r in range(rows):
                    for c in range(n):
                        var total: Float64 = 0
                        for f in range(k):
                            total += (xs.unsafe_ptr()[unsafe_offset=r*k+f].cast[DType.float64]()
                                      * ws.unsafe_ptr()[unsafe_offset=c*k+f].cast[DType.float64]())
                        comptime if HAS_BIAS:
                            total += bs.unsafe_ptr()[unsafe_offset=c].cast[DType.float64]()
                        exact.append(total)
    var input = TileTensor(x,row_major(rows,k))
    var weight = TileTensor(w,row_major(n,k))
    var output = TileTensor(y,row_major(rows,n))
    comptime for arrangement in [5, 7, 8, 9, 10]:
        y.enqueue_fill(0)
        comptime if HAS_BIAS:
            enqueue_linear_decode_rows_apple_gpu[arrangement](ctx,input,weight,TileTensor(b,row_major(n)),output)
        else:
            enqueue_linear_decode_rows_apple_gpu[arrangement](ctx,input,weight,output)
        var above_half = 0
        var above_one = 0
        var above_two = 0
        var worst: Float64 = 0
        with y.map_to_host() as ys:
            for i in range(rows*n):
                var error = abs(ys.unsafe_ptr()[unsafe_offset=i].cast[DType.float64]() - exact[i]) / _bf16_ulp(exact[i])
                worst = max(worst, error)
                above_half += 1 if error > 0.5 else 0
                above_one += 1 if error > 1 else 0
                above_two += 1 if error > 2 else 0
        print("ACCURACY",arrangement,rows,n,k,rows*n,above_half,above_one,above_two,worst)


def accuracy_census() raises:
    """1e's accuracy evidence: the five decode projection shapes with the kernel tests' values."""
    var ctx = DeviceContext()
    if ctx.api() != "metal" or ctx.name() != "Apple M4 Pro":
        raise Error("study requires Apple M4 Pro / Metal")
    print("device:",ctx.name())
    print("api:",ctx.api())
    _accuracy_shape[True](ctx,8,1152,896)
    _accuracy_shape[False](ctx,8,896,896)
    _accuracy_shape[False](ctx,8,4864,896)
    _accuracy_shape[False](ctx,8,896,4864)
    _accuracy_shape[False](ctx,2,151936,896)
    print("ACCURACY_COMPLETE")


def batch_profile[ARRANGEMENT: Int](prepared: String, tables: String, context: Int, sequences: Int,
                                    workload: String) raises:
    """Trace target: ten warmups and eight plain batched steps inside the profile region."""
    var ctx = DeviceContext()
    if ctx.api() != "metal" or ctx.name() != "Apple M4 Pro":
        raise Error("study requires Apple M4 Pro / Metal")
    var history = frozen_history(tables)
    var contexts = batch_contexts(context)
    var tokens = List[Int](capacity=BATCH_POOL)
    for s in range(BATCH_POOL):
        tokens.append(history[contexts[s]+s])
    var model = QwenModel(ctx,prepared,4096,256,BATCH_POOL)
    var kv = KVPool(ctx,BATCH_POOL,4096,model.kv_geometry())
    batch_setup(ctx,model,kv,history,context)
    batch_rewind(model,kv,contexts,sequences)
    var expected = batch_step[False, ARRANGEMENT](model,kv,ctx,tokens,contexts,sequences)
    for _ in range(10):
        batch_rewind(model,kv,contexts,sequences)
        batch_check(model,batch_step[False, ARRANGEMENT](model,kv,ctx,tokens,contexts,sequences),expected,sequences)
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("correctness: passed")
    print("profile implementation:","QwenModel.forward+greedy_tokens")
    print("rows:",sequences)
    print("hidden: 896")
    print("key value rows:",context+1)
    print("profile workload:",workload)
    print("profile dispatches per iteration:",245)
    print("warmup iterations: 10")
    print("profile iterations: 8")
    print("post-profile idle milliseconds: 250")
    print("PROFILE_REGION_BEGIN")
    for _ in range(8):
        batch_rewind(model,kv,contexts,sequences)
        batch_check(model,batch_step[False, ARRANGEMENT](model,kv,ctx,tokens,contexts,sequences),expected,sequences)
    print("PROFILE_REGION_END")
    sleep(0.25)


# The paged KV study (2d). Layout 0, the control, holds each sequence in one block of
# the whole context, slot-major; layouts 1-6 use blocks of 32, 64 and 128 slots, each
# slot-major and then head-major. Every pool's table comes from a block manager whose
# free list is a seeded permutation, so a sequence's blocks are scattered.
comptime PAGED_LAYOUTS = 7
comptime PAGED_SEED = 2026
# Elements of one sequence's 4,096 slots in every layer, K and V: 48 MiB.
comptime PAGED_SINGLE = 4096 * LAYERS * 2 * KV_WIDTH


def paged_block_size(layout: Int) raises -> Int:
    if layout == 0:
        return 4096
    if layout < 1 or layout >= PAGED_LAYOUTS:
        raise Error("unknown paged KV layout")
    return 32 << ((layout - 1) // 2)


def paged_head_major(layout: Int) -> Bool:
    return layout > 0 and (layout - 1) % 2 == 1


def paged_arms(study: String) raises -> List[Int]:
    """The arms' layouts: for paged, 0 against itself, then 1-6 against 0; for paged-confirm:L, 0 against L."""
    if study == "paged":
        return [0, 1, 2, 3, 4, 5, 6]
    var parts = study.split(":")
    if len(parts) == 2 and String(parts[0]) == "paged-confirm":
        var layout = Int(String(parts[1]))
        if layout >= 1 and layout < PAGED_LAYOUTS:
            return [0, layout]
    raise Error("unknown paged KV study")


def paged_table(blocks: Int, size: Int, length: Int) raises -> List[Int]:
    """The table of one sequence of `length` positions, alone in a pool of `blocks` blocks."""
    var manager = BlockManager(blocks, size, 4096, PAGED_SEED)
    var sequence = manager.add()
    manager.reserve(sequence, length)
    return manager.table(sequence)


def paged_row(pool: KVPool, table: List[Int], layer: Int, kv: Int, position: Int, head: Int) -> Int:
    var block = table[position // pool.block_size]
    var slot = position % pool.block_size
    if pool.head_major:
        return kv_row[KV_HEADS, HEAD_DIM, True](block, layer, LAYERS, kv, slot, head, pool.block_size)
    return kv_row[KV_HEADS, HEAD_DIM, False](block, layer, LAYERS, kv, slot, head, pool.block_size)


def paged_rows(pool: KVPool, table: List[Int], length: Int) raises -> List[UInt16]:
    """A sequence's K/V rows below `length` in position order: layer, K then V, position, head."""
    var rows = List[UInt16](capacity=LAYERS*2*length*KV_WIDTH)
    var storage = pool.storage.create_sub_buffer[DType.bfloat16](0,len(pool.storage))
    with storage.map_to_host() as mapped:
        for layer in range(LAYERS):
            for kv in range(2):
                for position in range(length):
                    for head in range(KV_HEADS):
                        var row = paged_row(pool,table,layer,kv,position,head)
                        for d in range(HEAD_DIM):
                            rows.append(bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=row+d]))
    return rows^


def paged_singles(ctx: DeviceContext, mut model: QwenModel, history: List[Int], arms: List[Int],
                  length: Int) raises -> DeviceBuffer[DType.bfloat16]:
    """Each arm's layout holding `length` positions of the frozen history, one 48 MiB region per layout.

    The history is prefilled in Fast chunks once per layout, through a scratch
    pool of one sequence, and its K/V rows must equal layout 0's byte for byte.
    """
    var singles = ctx.enqueue_create_buffer[DType.bfloat16](PAGED_LAYOUTS*PAGED_SINGLE)
    var scratch = KVPool(ctx,1,4096,model.kv_geometry())
    var reference = List[UInt16]()
    for layout in range(PAGED_LAYOUTS):
        var used = layout == 0
        for arm in arms:
            used = used or arm == layout
        if not used:
            continue
        var size = paged_block_size(layout)
        scratch.relayout(ctx,size,paged_head_major(layout))
        scratch.storage.enqueue_fill(0)
        var table = paged_table(scratch.blocks,size,4096)
        var offset = 0
        while offset < length:
            var count = min(256,length-offset)
            var chunk = List[Int](capacity=count)
            for i in range(count):
                chunk.append(history[offset+i])
            model.forward(ctx,StepBatch.sequence(chunk,offset,table,size),scratch,fast_plan(count,offset+count,ctx.name()))
            offset += count
        ctx.synchronize()
        var rows = paged_rows(scratch,table,length)
        if layout == 0:
            reference = rows^
        else:
            for i in range(len(reference)):
                if rows[i] != reference[i]:
                    raise Error("a paged layout's history differs from one block per sequence")
        ctx.enqueue_copy(dst_buf=singles.create_sub_buffer[DType.bfloat16](layout*PAGED_SINGLE,PAGED_SINGLE),
                         src_buf=scratch.storage)
    ctx.synchronize()
    return singles^


def paged_rebuild(ctx: DeviceContext, mut kv: KVPool, singles: DeviceBuffer[DType.bfloat16], layout: Int,
                  contexts: List[Int], sequences: Int) raises -> BlockManager:
    """Before an arm, outside timing: hold the working pool in `layout`, let a seeded manager
    allocate each sequence's blocks for its cached rows and the step's row, and copy the
    cached blocks from the layout's history."""
    var size = paged_block_size(layout)
    kv.relayout(ctx,size,paged_head_major(layout))
    var manager = BlockManager(kv.blocks,size,4096,PAGED_SEED)
    var single = paged_table(4096//size,size,4096)
    var block = len(kv.storage)//kv.blocks
    for s in range(sequences):
        var sequence = manager.add()
        manager.reserve(sequence,contexts[s]+1)
        var table = manager.table(sequence)
        for j in range(ceildiv(contexts[s],size)):
            ctx.enqueue_copy(dst_buf=kv.storage.create_sub_buffer[DType.bfloat16](table[j]*block,block),
                             src_buf=singles.create_sub_buffer[DType.bfloat16](layout*PAGED_SINGLE+single[j]*block,block))
            kv.written[table[j]] = min(size,contexts[s]-j*size)
        manager.commit(sequence,contexts[s])
    ctx.synchronize()
    manager.check_pool(kv)
    return manager^


def paged_rewind(mut model: QwenModel, mut kv: KVPool, manager: BlockManager, contexts: List[Int],
                 sequences: Int) raises:
    # Earlier readback completed every step. Only the step's block is rewound.
    model.submitted_rows = 0
    for s in range(sequences):
        var table = manager.table(s)
        kv.written[table[contexts[s]//kv.block_size]] = contexts[s]%kv.block_size


def paged_step(mut model: QwenModel, mut kv: KVPool, ctx: DeviceContext, manager: BlockManager,
               tokens: List[Int], contexts: List[Int], sequences: Int) raises -> List[Int]:
    """One decode token per sequence, as batch_step, with each table taken from the block manager."""
    var tables = List[List[Int]](capacity=sequences)
    var width = 0
    for s in range(sequences):
        tables.append(manager.table(s))
        width = max(width,len(tables[s]))
    var ids = List[Int](capacity=sequences)
    var positions = List[Int](capacity=sequences)
    var starts = List[Int](capacity=sequences+1)
    var seq_lens = List[Int](capacity=sequences)
    var blocks = List[Int](capacity=sequences*width)
    var slots = List[Int](capacity=sequences)
    var rows = List[Int](capacity=sequences)
    var longest = 0
    for s in range(sequences):
        ids.append(tokens[s])
        positions.append(contexts[s])
        starts.append(s)
        seq_lens.append(contexts[s]+1)
        for b in range(width):
            blocks.append(tables[s][b] if b < len(tables[s]) else 0)
        slots.append(tables[s][contexts[s]//kv.block_size]*kv.block_size+contexts[s]%kv.block_size)
        rows.append(s)
        longest = max(longest,contexts[s]+1)
    starts.append(sequences)
    var batch = StepBatch(ids^,positions^,starts^,sequences,seq_lens^,width,blocks^,slots^,rows^)
    model.forward(ctx,batch,kv,fast_plan(sequences,longest,ctx.name(),sequences))
    return model.greedy_tokens(ctx)


def paged_bench(prepared: String, tables: String, context: Int, first: Int, study: String) raises:
    """The batch mode across KV layouts: per batch size, layout 0 against itself, then each candidate
    against 0. The working pool, the 3 GiB of 64 full contexts, is rebuilt before every arm."""
    var arms = paged_arms(study)
    var ctx = DeviceContext()
    if ctx.api() != "metal" or ctx.name() != "Apple M4 Pro":
        raise Error("study requires Apple M4 Pro / Metal")
    var history = frozen_history(tables)
    var contexts = batch_contexts(context)
    var sizes: List[Int] = [1, 2, 4, 8, 16, 32, 64]
    if context == 0:
        sizes = [BATCH_MIXED]
    var longest = 0
    for c in contexts:
        longest = max(longest,c)
    if len(history) < longest+BATCH_POOL+1:
        raise Error("insufficient frozen token history")
    var tokens = List[Int](capacity=BATCH_POOL)
    for s in range(BATCH_POOL):
        tokens.append(history[contexts[s]+s])
    var model = QwenModel(ctx,prepared,4096,256,BATCH_POOL)
    var singles = paged_singles(ctx,model,history,arms,longest)
    var kv = KVPool(ctx,BATCH_POOL,4096,model.kv_geometry())
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("context:",context,"pool slots:",BATCH_POOL*4096)
    print("study:",study)
    print("history rows equal to layout 0:",longest)
    var records = String()
    for index in range(len(sizes)):
        var sequences = sizes[len(sizes)-1-index] if first == 1 else sizes[index]
        # Each layout's untimed step must select layout 0's tokens.
        var expected = List[List[Int]]()
        for position in range(len(arms)):
            var manager = paged_rebuild(ctx,kv,singles,arms[position],contexts,sequences)
            paged_rewind(model,kv,manager,contexts,sequences)
            var reference = paged_step(model,kv,ctx,manager,tokens,contexts,sequences)
            batch_check(model,reference,reference,sequences)
            var differing = 0
            if position > 0:
                for s in range(sequences):
                    if reference[s] != expected[0][s]:
                        differing += 1
            if differing > 0:
                raise Error("a paged layout changed a sequence's token")
            print("tokens:",sequences,arms[position],differing)
            expected.append(reference^)
        print("sequences:",sequences,"first token:",expected[0][0],"last token:",expected[0][sequences-1])
        for position in range(len(arms)):
            var comparison = len(arms)-1-position if first == 1 else position
            for arm_index in range(2):
                var arm = (first+arm_index)%2
                var layout = arms[comparison] if arm == 1 else arms[0]
                var manager = paged_rebuild(ctx,kv,singles,layout,contexts,sequences)
                for sample in range(20):
                    paged_rewind(model,kv,manager,contexts,sequences)
                    var start = now()
                    var selected = paged_step(model,kv,ctx,manager,tokens,contexts,sequences)
                    var elapsed = now()-start
                    batch_check(model,selected,expected[0],sequences)
                    if sample >= 10:
                        records += ("BATCH "+String(sequences)+" "+String(comparison)+" "+String(arm)+" "
                                    +String(sample-10)+" "+String(elapsed)+"\n")
    print(records,end="")
    print("BATCH_COMPLETE")


def paged_prefill_workloads() -> List[Int]:
    """2d's prefill chunks as rows, total pairs: the runtime study's eleven cells, then
    256-row chunks after 256 and 2,816 cached tokens."""
    return [16, 1024, 16, 4096, 15, 256, 17, 256, 64, 1024, 64, 4096, 256, 1024, 256, 4096,
            65, 4096, 255, 4096, 16, 256, 256, 512, 256, 3072]


def paged_restore(ctx: DeviceContext, mut kv: KVPool, singles: DeviceBuffer[DType.bfloat16], layout: Int,
                  length: Int) raises -> List[Int]:
    """Before an arm, outside timing: the one-sequence pool in `layout`, a copy of the layout's
    history with `length` positions written. Returns its table."""
    var size = paged_block_size(layout)
    kv.relayout(ctx,size,paged_head_major(layout))
    ctx.enqueue_copy(dst_buf=kv.storage,src_buf=singles.create_sub_buffer[DType.bfloat16](layout*PAGED_SINGLE,PAGED_SINGLE))
    ctx.synchronize()
    var table = paged_table(kv.blocks,size,4096)
    for j in range(len(table)):
        kv.written[table[j]] = min(max(length-j*size,0),size)
    return table^


def paged_prefill(prepared: String, tables: String, first: Int, study: String) raises:
    """2d's prefill workloads, one process per block: for each chunk, layout 0 against itself, then
    each candidate against 0, timed from the token upload to device synchronization."""
    var arms = paged_arms(study)
    var ctx = DeviceContext()
    if ctx.api() != "metal" or ctx.name() != "Apple M4 Pro":
        raise Error("study requires Apple M4 Pro / Metal")
    var history = frozen_history(tables)
    var workloads = paged_prefill_workloads()
    var count = len(workloads)//2
    var longest = 0
    for w in range(count):
        longest = max(longest,workloads[2*w+1]-workloads[2*w])
    if len(history) < 4096:
        raise Error("insufficient frozen token history")
    var model = QwenModel(ctx,prepared,4096,256)
    var singles = paged_singles(ctx,model,history,arms,longest)
    var kv = KVPool(ctx,1,4096,model.kv_geometry())
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("study:",study)
    print("history rows equal to layout 0:",longest)
    var records = String()
    for index in range(count):
        var w = count-1-index if first == 1 else index
        var rows = workloads[2*w]
        var total = workloads[2*w+1]
        var prefix = total-rows
        var ids = List[Int](capacity=rows)
        for i in range(rows):
            ids.append(history[prefix+i])
        var plan = fast_plan(rows,total,ctx.name())
        print("prefill workload:",w,rows,total,plan.configuration)
        # Each layout's untimed chunk must select layout 0's token.
        var expected = List[Int]()
        for position in range(len(arms)):
            var table = paged_restore(ctx,kv,singles,arms[position],prefix)
            model.forward(ctx,StepBatch.sequence(ids,prefix,table,kv.block_size),kv,plan)
            var token = model.greedy(ctx)
            var differing = 1 if position > 0 and token != expected[0] else 0
            if differing > 0:
                raise Error("a paged layout changed a prefill token")
            print("prefill tokens:",w,arms[position],differing)
            expected.append(token)
        for position in range(len(arms)):
            var comparison = len(arms)-1-position if first == 1 else position
            for arm_index in range(2):
                var arm = (first+arm_index)%2
                var layout = arms[comparison] if arm == 1 else arms[0]
                var table = paged_restore(ctx,kv,singles,layout,prefix)
                for sample in range(20):
                    # Earlier readback completed every forward; only the chunk's positions are rewound.
                    kv.truncate_table(table,prefix)
                    model.submitted_rows = prefix*LAYERS
                    var batch = StepBatch.sequence(ids,prefix,table,kv.block_size)
                    var start = now()
                    model.forward(ctx,batch,kv,plan)
                    ctx.synchronize()
                    var elapsed = now()-start
                    if (model.greedy(ctx) != expected[0] or model.submitted_rows != total*LAYERS
                            or model.last_route.configuration != plan.configuration):
                        raise Error("a prefill sample changed its token, accounting or route")
                    if sample >= 10:
                        records += ("PREFILL "+String(w)+" "+String(comparison)+" "+String(arm)+" "
                                    +String(sample-10)+" "+String(elapsed)+"\n")
    print(records,end="")
    print("PREFILL_COMPLETE")


def paged_profile(prepared: String, tables: String, context: Int, sequences: Int, layout: Int,
                  workload: String) raises:
    """Trace target for one layout: ten warmups and eight plain batched steps inside the profile region."""
    var ctx = DeviceContext()
    if ctx.api() != "metal" or ctx.name() != "Apple M4 Pro":
        raise Error("study requires Apple M4 Pro / Metal")
    var history = frozen_history(tables)
    var contexts = batch_contexts(context)
    var tokens = List[Int](capacity=BATCH_POOL)
    for s in range(BATCH_POOL):
        tokens.append(history[contexts[s]+s])
    var model = QwenModel(ctx,prepared,4096,256,BATCH_POOL)
    var arms: List[Int] = [0, layout]
    var singles = paged_singles(ctx,model,history,arms,context)
    var kv = KVPool(ctx,BATCH_POOL,4096,model.kv_geometry())
    var control = paged_rebuild(ctx,kv,singles,0,contexts,sequences)
    paged_rewind(model,kv,control,contexts,sequences)
    var expected = paged_step(model,kv,ctx,control,tokens,contexts,sequences)
    var manager = paged_rebuild(ctx,kv,singles,layout,contexts,sequences)
    for _ in range(10):
        paged_rewind(model,kv,manager,contexts,sequences)
        batch_check(model,paged_step(model,kv,ctx,manager,tokens,contexts,sequences),expected,sequences)
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("correctness: passed")
    print("profile implementation:","QwenModel.forward+greedy_tokens")
    print("rows:",sequences)
    print("hidden: 896")
    print("key value rows:",context+1)
    print("profile workload:",workload)
    print("profile dispatches per iteration:",245)
    print("warmup iterations: 10")
    print("profile iterations: 8")
    print("post-profile idle milliseconds: 250")
    print("PROFILE_REGION_BEGIN")
    for _ in range(8):
        paged_rewind(model,kv,manager,contexts,sequences)
        batch_check(model,paged_step(model,kv,ctx,manager,tokens,contexts,sequences),expected,sequences)
    print("PROFILE_REGION_END")
    sleep(0.25)


def _batch_advance[LT: TensorLayout](x: TileTensor[DType.int32, LT, MutAnyOrigin]):
    comptime assert x.flat_rank == 1
    if global_idx.x == 0:
        x[0] = x[0]*2+1

def batch_support() raises:
    var ctx = DeviceContext()
    print("device:",ctx.name()); print("api:",ctx.api())
    if ctx.name() != "Apple M4 Pro" or ctx.api() != "metal": raise Error("requires M4 Pro / Metal")
    var buf = ctx.enqueue_create_buffer[DType.int32](1)
    buf.enqueue_fill(3)
    var x = TileTensor(buf,row_major(1))
    var compiled = ctx.compile_function[_batch_advance[type_of(x.layout)]]()
    ctx.enqueue_function(compiled,x,grid_dim=1,block_dim=32)
    ctx.enqueue_function(compiled,x,grid_dim=1,block_dim=32)
    with buf.map_to_host() as mapped:
        if mapped.unsafe_ptr()[unsafe_offset=0] != 15: raise Error("eager control failed")
    print("BATCH_EAGER_PASS 15")
    buf.enqueue_fill(3);ctx.synchronize()
    def build(mut builder: DeviceGraphBuilder) raises {mut x, imm compiled}:
        print("BATCH_BUILDER_ENTERED")
        var first = builder.add_function(compiled,x,grid_dim=1,block_dim=32,dependencies=[])
        _ = builder.add_function(compiled,x,grid_dim=1,block_dim=32,dependencies=[first])
    try:
        var graph = DeviceGraph.create(ctx,build)
        graph.replay();ctx.synchronize()
        with buf.map_to_host() as mapped:
            if mapped.unsafe_ptr()[unsafe_offset=0] != 15: raise Error("graph first replay failed")
        graph.replay();ctx.synchronize()
        with buf.map_to_host() as mapped:
            if mapped.unsafe_ptr()[unsafe_offset=0] != 63: raise Error("graph second replay failed")
        print("BATCH_GRAPH_PASS 15 63")
    except e:
        print("BATCH_GRAPH_ERROR",e)
    print("BATCH_SUPPORT_COMPLETE")


def main() raises:
    comptime if is_defined["MODEL_BATCH_SUPPORT"]():
        batch_support()
        return
    comptime if is_defined["MODEL_BATCH_PROFILE"]():
        batch_profile[get_defined_int["MODEL_BATCH_ARRANGEMENT"]()](String(get_defined_string["MODEL_PREPARED"]()),
            String(get_defined_string["MODEL_TABLES"]()),get_defined_int["MODEL_BATCH_PROFILE"](),
            get_defined_int["MODEL_BATCH_SEQUENCES"](),String(get_defined_string["MODEL_BATCH_WORKLOAD"]()))
        return
    comptime if is_defined["MODEL_PAGED_PROFILE"]():
        paged_profile(String(get_defined_string["MODEL_PREPARED"]()),String(get_defined_string["MODEL_TABLES"]()),
            get_defined_int["MODEL_PAGED_PROFILE"](),get_defined_int["MODEL_BATCH_SEQUENCES"](),
            get_defined_int["MODEL_BATCH_LAYOUT"](),String(get_defined_string["MODEL_BATCH_WORKLOAD"]()))
        return
    var cli = argv()
    if len(cli) == 2 and String(cli[1]) == "accuracy":
        accuracy_census()
        return
    if len(cli) == 7 and String(cli[1]) == "batch":
        var context = Int(String(cli[4]))
        var first = Int(String(cli[5]))
        if (context != 0 and context != 64 and context != 1024 and context != 3968) or first < 0 or first > 1:
            raise Error("invalid batch-size workload")
        if String(cli[6]).startswith("paged"):
            paged_bench(String(cli[2]),String(cli[3]),context,first,String(cli[6]))
        else:
            batch_bench(String(cli[2]),String(cli[3]),context,first,String(cli[6]))
        return
    if len(cli) == 6 and String(cli[1]) == "paged-prefill":
        var first = Int(String(cli[4]))
        if first < 0 or first > 1:
            raise Error("invalid paged prefill order")
        paged_prefill(String(cli[2]),String(cli[3]),first,String(cli[5]))
        return
    var args = List[String]()
    comptime if is_defined["MODEL_PROFILE_PREFIX"]():
        args = ["model","profile",String(get_defined_string["MODEL_PREPARED"]()),
                String(get_defined_string["MODEL_TABLES"]()),
                String(get_defined_int["MODEL_PROFILE_PREFIX"]()),"0","0",""]
    else:
        for arg in argv():
            args.append(String(arg))
    if len(args) != 8:
        raise Error("model mode prepared tables prefix first comparison output-directory")
    var mode = args[1]
    var prefix = Int(args[4])
    var first = Int(args[5])
    var comparison = Int(args[6])
    if ((mode != "bench" and mode != "verify" and mode != "profile")
        or (prefix != 64 and prefix != 1024 and prefix != 3968)
        or first < 0 or first > 1 or comparison < 0 or comparison > 1):
        raise Error("invalid frozen model profiling workload")
    var ctx = DeviceContext()
    if ctx.api() != "metal" or ctx.name() != "Apple M4 Pro":
        raise Error("study requires Apple M4 Pro / Metal")
    var history = frozen_history(args[3])
    if len(history) < prefix+1:
        raise Error("insufficient frozen token history")
    var model = QwenModel(ctx,args[2],4096,256)
    var kv = KVPool(ctx,1,4096,model.kv_geometry())
    # Define all inactive storage for exact before/after comparisons.
    kv.storage.enqueue_fill(0)
    var offset = 0
    while offset < prefix:
        var count = min(256,prefix-offset)
        var chunk = List[Int](capacity=count)
        for i in range(count):
            chunk.append(history[offset+i])
        model.forward(ctx,StepBatch.sequence(chunk,offset,[0],4096),kv,fast_plan(count,offset+count,ctx.name()))
        offset += count
    ctx.synchronize()
    var ids: List[Int] = [history[prefix]]
    var winner = step[False](model,kv,ctx,ids)
    print("device:",ctx.name())
    print("api:",ctx.api())
    print("prefix:",prefix,"token:",ids[0],"winner:",winner)
    if mode == "verify":
        rewind(model,kv,prefix)
        snapshot(model,kv,args[7]+"/before")
        poison_outputs(model,kv,prefix)
        var plain = step[False](model,kv,ctx,ids)
        snapshot(model,kv,args[7]+"/plain")
        rewind(model,kv,prefix)
        poison_outputs(model,kv,prefix)
        var observed = step[True](model,kv,ctx,ids)
        snapshot(model,kv,args[7]+"/observed")
        if plain != observed or kv.length(0) != prefix+1 or model.submitted_rows != 24*(prefix+1):
            raise Error("instrumentation changed token or accounting")
        var record = String()
        for i in range(prefix+1):
            record += String(history[i])+"\n"
        var file = open(args[7]+"/history.txt","w")
        file.write(record)
        print("VERIFY_COMPLETE")
        return
    if mode == "profile":
        for _ in range(10):
            rewind(model,kv,prefix)
            if step[False](model,kv,ctx,ids) != winner:
                raise Error("unstable profile prediction")
        print("correctness: passed")
        print("profile implementation:","QwenModel.forward+greedy-all-three")
        print("rows: 1")
        print("hidden: 896")
        print("key value rows:",prefix+1)
        print("profile workload:","model-p"+String(prefix)+"-all-three")
        print("profile dispatches per iteration:",245)
        print("warmup iterations: 10")
        print("profile iterations: 8")
        print("post-profile idle milliseconds: 250")
        print("PROFILE_REGION_BEGIN")
        for _ in range(8):
            rewind(model,kv,prefix)
            if step[False](model,kv,ctx,ids) != winner:
                raise Error("unstable profile prediction")
        print("PROFILE_REGION_END")
        sleep(0.25)
        return
    var records = String()
    for arm_index in range(2):
        var arm = (first+arm_index)%2
        var observe = comparison == 1 and arm == 1
        for sample in range(20):
            rewind(model,kv,prefix)
            var start = now()
            var selected: Int
            if observe:
                selected = step[True](model,kv,ctx,ids)
            else:
                selected = step[False](model,kv,ctx,ids)
            var elapsed = now()-start
            if selected != winner or model.submitted_rows != 24*(prefix+1):
                raise Error("measurement prediction/accounting changed")
            if sample >= 10:
                records += "SAMPLE "+String(arm)+" "+String(sample-10)+" "+String(elapsed)
                if observe:
                    for i in range(10):
                        records += " "+String(model.observation[i]-start)
                records += "\n"
    print(records,end="")
    print("BENCHMARK_COMPLETE")
