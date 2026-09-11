"""Actual checkpoint: persistent chat versus full-history replay, with raw caches."""
from std.sys import argv
from std.testing import assert_equal, assert_raises
from max.gpu.host import DeviceContext
from llm_mojo.chat import ChatSession, DEFAULT_SYSTEM
from llm_mojo.model import QwenModel, save_bf16, select_configuration
from llm_mojo.tokenizer import Tokenizer, TokenizerWorkspace
from llm_mojo.generate_cli import now


def caches(model: QwenModel, directory: String) raises:
    for i in range(24):
        save_bf16(model.layers[i].cache.key,directory+"/key_"+String(i)+".bin",model.capacity*128)
        save_bf16(model.layers[i].cache.value,directory+"/value_"+String(i)+".bin",model.capacity*128)


def replay_history(mut model: QwenModel, ctx: DeviceContext, ids: List[Int]) raises:
    while model.length < len(ids):
        var rows = min(model.max_rows,len(ids)-model.length)
        var suffix = List[Int]()
        for i in range(rows):
            suffix.append(ids[model.length+i])
        model.forward(ctx,suffix,select_configuration("fast",rows,model.length+rows,ctx.name()))
    ctx.synchronize()


def main() raises:
    var args = argv()
    if len(args)!=4:
        raise Error("chat_driver prepared tokenizer output")
    var tokenizer = Tokenizer(args[2])
    var work = TokenizerWorkspace()
    var ctx = DeviceContext()
    print("device",ctx.name(),"backend",ctx.api())
    var session = ChatSession(ctx,args[1],tokenizer,work,String(DEFAULT_SYSTEM),256,512)
    var replay = QwenModel(ctx,args[1],512,256)
    for i in range(24):
        session.model.layers[i].cache.key.enqueue_fill(123)
        session.model.layers[i].cache.value.enqueue_fill(123)
    var prompts: List[String] = ["My name is Ada. Reply briefly.","What is my name?", "Scrivi una frase sul caffè. ☕"]
    for turn in range(len(prompts)):
        var directory = args[3]+"/turn"+String(turn)
        var before = session.model.length
        caches(session.model,directory+"/before")
        session.begin(tokenizer,work,prompts[turn],12)
        var prompt_length = len(session.history.tokens)
        var ids_file = open(directory+"/prompt.txt","w")
        for id in session.history.tokens:
            ids_file.write(String(id)+"\n")
        var started = now()
        while session.model.length < len(session.history.tokens):
            session.submit_next(ctx)
        ctx.synchronize()
        var cached_ns = now()-started
        save_bf16(session.model.logits,directory+"/cached.bin",151936)
        started = now()
        replay.reset(ctx)
        replay_history(replay,ctx,session.history.tokens)
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
                        session.model.length = before
                        session.model.submitted_rows = before*24
                        for i in range(24):
                            session.model.layers[i].cache.length = before
                    else:
                        replay.reset(ctx)
                    started = now()
                    if arm == 0:
                        while session.model.length<len(session.history.tokens):
                            session.submit_next(ctx)
                        ctx.synchronize()
                    else:
                        replay_history(replay,ctx,session.history.tokens)
                    var elapsed = now()-started
                    if sample>=0:
                        print("sample",turn,block,arm,sample,elapsed)
        while session.history.generating:
            if session.model.length<len(session.history.tokens):
                session.submit_next(ctx)
            else:
                _ = session.sample(ctx)
        assert_equal(session.model.submitted_rows,session.model.length*24)
        caches(session.model,directory+"/after")
        print("finish",turn,"cached",session.model.length,"history",len(session.history.tokens),"generated",session.history.generated,"reason",session.history.reason)
    var old_length = session.model.length
    var old_history = len(session.history.tokens)
    with assert_raises():
        session.begin(tokenizer,work,"too big",4096)
    assert_equal(session.model.length,old_length)
    assert_equal(len(session.history.tokens),old_history)
    session.fail()
    with assert_raises():
        session.begin(tokenizer,work,"invalid",1)
    session.reset(ctx)
    assert_equal(session.model.length,0)
    session.begin(tokenizer,work,prompts[0],12)
    while session.model.length<len(session.history.tokens):
        session.submit_next(ctx)
    save_bf16(session.model.logits,args[3]+"/reset.bin",151936)
    session.reset(ctx)
    session.begin(tokenizer,work,prompts[0],12)
    # Interrupt before any submission: the whole user turn remains pending.
    session.history.finish("interrupted")
    session.begin(tokenizer,work,"Continue.",1)
    while session.model.length<len(session.history.tokens):
        session.submit_next(ctx)
    _ = session.sample(ctx)
    assert_equal(session.history.reason,"limit")
    assert_equal(session.model.submitted_rows,session.model.length*24)
    print("chat lifecycle passed: prefix append reset failure recovery interruption pending closure")
