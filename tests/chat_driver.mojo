"""Actual checkpoint: persistent chat versus full-history replay, with raw caches."""
from std.memory import bitcast
from std.sys import argv
from std.testing import assert_equal, assert_raises
from max.gpu.host import DeviceContext
from llm_mojo.models.qwen2.chat import ChatSession, DEFAULT_SYSTEM
from llm_mojo.models.qwen2.model import QwenModel, save_bf16
from llm_mojo.models.qwen2.plan import fast_plan
from llm_mojo.models.qwen2.tokenizer import Tokenizer, TokenizerWorkspace
from llm_mojo.runtime.clock import now
from llm_mojo.serving.batch import StepBatch
from llm_mojo.serving.kv_pool import KVPool


def caches(session: ChatSession, directory: String) raises:
    """Every layer's K and V rows in position order through the session's capacity.

    Positions outside the session's blocks read as the pool's fill, 123, as they
    do in a pool of one block, so the dumps do not depend on the block size.
    """
    var table = session.table()
    var size = session.kv.block_size
    var fill = bitcast[DType.uint16](Scalar[DType.bfloat16](123))
    for layer in range(24):
        for index in range(2):
            var data = List[UInt8](capacity=session.model.capacity*256)
            for b in range((session.model.capacity+size-1)//size):
                var slots = min(size,session.model.capacity-b*size)
                if b < len(table):
                    var view = session.kv.view(table[b],layer,index)
                    with view.map_to_host() as mapped:
                        for slot in range(slots):
                            for head in range(2):
                                var start = (head*size+slot)*64 if session.kv.head_major else (slot*2+head)*64
                                for d in range(64):
                                    var bits = bitcast[DType.uint16](mapped.unsafe_ptr()[unsafe_offset=start+d])
                                    data.append(UInt8(bits & 255))
                                    data.append(UInt8(bits >> 8))
                else:
                    for _ in range(slots*128):
                        data.append(UInt8(fill & 255))
                        data.append(UInt8(fill >> 8))
            var file = open(directory+("/key_" if index == 0 else "/value_")+String(layer)+".bin","w")
            file.write_bytes(data)


def replay_history(mut model: QwenModel, mut kv: KVPool, ctx: DeviceContext, ids: List[Int]) raises:
    while kv.length(0) < len(ids):
        var cached = kv.length(0)
        var rows = min(model.max_rows,len(ids)-cached)
        var suffix = List[Int]()
        for i in range(rows):
            suffix.append(ids[cached+i])
        model.forward(ctx,StepBatch.sequence(suffix,cached,[0],kv.block_size),kv,fast_plan(rows,cached+rows,ctx.name()))
    ctx.synchronize()


def main() raises:
    var args = argv()
    if len(args) < 4 or len(args) > 6:
        raise Error("chat_driver prepared tokenizer output [block-size [head-major]]")
    var tokenizer = Tokenizer(args[2])
    var work = TokenizerWorkspace()
    var ctx = DeviceContext()
    print("device",ctx.name(),"backend",ctx.api())
    var session = ChatSession(ctx,args[1],tokenizer,work,String(DEFAULT_SYSTEM),256,512,
                              Int(args[4]) if len(args) > 4 else 0,len(args) > 5 and args[5] == "head-major")
    var replay = QwenModel(ctx,args[1],512,256)
    var replay_kv = KVPool(ctx,1,512,replay.kv_geometry())
    session.kv.storage.enqueue_fill(123)
    var prompts: List[String] = ["My name is Ada. Reply briefly.","What is my name?", "Scrivi una frase sul caffè. ☕"]
    for turn in range(len(prompts)):
        var directory = args[3]+"/turn"+String(turn)
        var before = session.length()
        caches(session,directory+"/before")
        session.begin(tokenizer,work,prompts[turn],12)
        var prompt_length = len(session.history.tokens)
        var ids_file = open(directory+"/prompt.txt","w")
        for id in session.history.tokens:
            ids_file.write(String(id)+"\n")
        var started = now()
        while session.length() < len(session.history.tokens):
            session.submit_next(ctx)
        ctx.synchronize()
        var cached_ns = now()-started
        save_bf16(session.model.logits,directory+"/cached.bin",151936)
        started = now()
        replay.reset(ctx)
        replay_kv.reset(ctx)
        replay_history(replay,replay_kv,ctx,session.history.tokens)
        var replay_ns = now()-started
        save_bf16(replay.logits,directory+"/replay.bin",151936)
        print("prefill",turn,"before",before,"prompt",prompt_length,"cached_ns",cached_ns,"replay_ns",replay_ns)
        # Paired cached-suffix versus full-history forwards. Logical reset and
        # prior-prefix preparation are outside timing; weights stay resident.
        for block in range(4):
            for order in range(2):
                var arm = order if block == 0 or block == 3 else 1-order
                for sample in range(-3,5):
                    ctx.synchronize()
                    if arm == 0:
                        session.model.submitted_rows = before*24
                        session.truncate(before)
                    else:
                        replay.reset(ctx)
                        replay_kv.reset(ctx)
                    started = now()
                    if arm == 0:
                        while session.length()<len(session.history.tokens):
                            session.submit_next(ctx)
                        ctx.synchronize()
                    else:
                        replay_history(replay,replay_kv,ctx,session.history.tokens)
                    var elapsed = now()-started
                    if sample>=0:
                        print("sample",turn,block,arm,sample,elapsed)
        while session.history.generating:
            if session.length()<len(session.history.tokens):
                session.submit_next(ctx)
            else:
                _ = session.sample(ctx)
        assert_equal(session.model.submitted_rows,session.length()*24)
        caches(session,directory+"/after")
        print("finish",turn,"cached",session.length(),"history",len(session.history.tokens),"generated",session.history.generated,"reason",session.history.reason)
    var old_length = session.length()
    var old_history = len(session.history.tokens)
    with assert_raises():
        session.begin(tokenizer,work,"too big",4096)
    assert_equal(session.length(),old_length)
    assert_equal(len(session.history.tokens),old_history)
    session.fail()
    with assert_raises():
        session.begin(tokenizer,work,"invalid",1)
    session.reset(ctx)
    assert_equal(session.length(),0)
    session.begin(tokenizer,work,prompts[0],12)
    while session.length()<len(session.history.tokens):
        session.submit_next(ctx)
    save_bf16(session.model.logits,args[3]+"/reset.bin",151936)
    session.reset(ctx)
    session.begin(tokenizer,work,prompts[0],12)
    # Interrupt before any submission: the whole user turn remains pending.
    session.history.finish("interrupted")
    session.begin(tokenizer,work,"Continue.",1)
    while session.length()<len(session.history.tokens):
        session.submit_next(ctx)
    _ = session.sample(ctx)
    assert_equal(session.history.reason,"limit")
    assert_equal(session.model.submitted_rows,session.length()*24)
    print("chat lifecycle passed: prefix append reset failure recovery interruption pending closure")
