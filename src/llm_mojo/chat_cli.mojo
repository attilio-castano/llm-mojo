"""A resident Mojo Qwen terminal chat; Python is only the verifying launcher."""
from llm_mojo.model import is_projection_policy
from std.sys import argv, is_defined
from max.gpu.host import DeviceContext
from llm_mojo.chat import ChatSession, DEFAULT_SYSTEM
from llm_mojo.tokenizer import Tokenizer, TokenizerWorkspace, TokenizerDecoder
from llm_mojo.terminal import block_interrupt, interrupted, read_line
from llm_mojo.generate_cli import now, is_stop


def main() raises:
    var args = argv()
    comptime STUDY = is_defined["MODEL_FUSION_STUDY"]()
    if len(args) != (8 if STUDY else 7):
        raise Error("chat prepared tokenizer maximum chunk-rows system-file report-file (empty = defaults)")
    var maximum = Int(args[3])
    var chunk = Int(args[4])
    if maximum < 1 or maximum > 4096 or chunk < 1 or chunk > 4096:
        raise Error("invalid chat limits")
    var started = now()
    var observed = args[6].byte_length()>0
    var events = String("event\tturn\tindex\tvalue\tnanoseconds\n")
    block_interrupt()
    print("Loading Qwen2.5-0.5B-Instruct…",flush=True)
    var tokenizer = Tokenizer(args[2])
    var work = TokenizerWorkspace()
    var system = String(DEFAULT_SYSTEM)
    if args[5].byte_length()>0:
        system = String(from_utf8=open(args[5],"r").read_bytes())
    var ctx = DeviceContext()
    var session = ChatSession(ctx,args[1],tokenizer,work,system,chunk)
    comptime if STUDY:
        if not is_projection_policy(args[7]) and args[7] != "fast" and args[7] != "fusion" and args[7] != "combined" and args[7] != "unfused" and args[7] != "gpu-argmax" and args[7] != "fused-head" and args[7] != "buffer-swap" and args[7] != "residual-norm" and args[7] != "swap-argmax" and args[7] != "all-three":
            raise Error("unknown native study arm")
        session.policy = String(args[7])
    print("Ready — Fast on",ctx.name(),"/",ctx.api(),flush=True)
    print("/reset: new conversation · /exit: quit · Ctrl-C: stop reply or clear input",flush=True)
    if observed:
        events += "load\t0\t0\t"+ctx.name()+"/"+ctx.api()+"\t"+String(now()-started)+"\n"
    var turn = 0
    var bytes = List[UInt8]()
    while True:
        print("\nYou: ",end="",flush=True)
        var status = read_line(bytes)
        if status == 0:
            break
        if status == -1:
            print("\nInput cancelled.",flush=True)
            continue
        if status == -2:
            print("Input too long (maximum 65536 bytes).",flush=True)
            continue
        var message: String
        try:
            message = String(from_utf8=bytes)
        except error:
            print("Input must be valid UTF-8.",flush=True)
            continue
        if message == "/exit":
            break
        if message == "/reset":
            session.reset(ctx)
            if observed:
                events += "reset\t"+String(turn)+"\t0\t0\t0\n"
            print("Conversation reset.",flush=True)
            continue
        if message.byte_length()==0:
            continue
        var turn_started = now()
        try:
            session.begin(tokenizer,work,message,maximum)
        except error:
            print(error,flush=True)
            if observed:
                events += "rejected\t"+String(turn)+"\t0\t"+String(session.model.length)+"\t0\n"
            continue
        turn += 1
        var cached_before = session.model.length
        var prompt_length = len(session.history.tokens)
        if observed:
            events += "begin\t"+String(turn)+"\t"+String(cached_before)+"\t"+String(prompt_length)+"\t0\n"
            for i in range(prompt_length):
                events += "prompt\t"+String(turn)+"\t"+String(i)+"\t"+String(session.history.tokens[i])+"\t0\n"
        var decoder = TokenizerDecoder()
        print("Assistant: ",end="",flush=True)
        try:
            while session.history.generating:
                if interrupted():
                    session.history.finish("interrupted")
                    break
                if session.model.length < len(session.history.tokens):
                    session.submit_next(ctx)
                    continue
                var token = session.sample(ctx)
                var output = List[UInt8]()
                if not is_stop(token):
                    decoder.push(tokenizer,token,output,True)
                    if len(output)>0:
                        print(String(from_utf8=output),end="",flush=True)
                        if observed:
                            events += "text\t"+String(turn)+"\t"+String(session.history.generated-1)+"\t"+String(len(output))+"\t"+String(now()-turn_started)+"\n"
                if observed:
                    events += "token\t"+String(turn)+"\t"+String(session.history.generated-1)+"\t"+String(token)+"\t"+String(now()-turn_started)+"\n"
            var output = List[UInt8]()
            decoder.finish(output)
            if len(output)>0:
                print(String(from_utf8=output),end="",flush=True)
                if observed:
                    events += "text\t"+String(turn)+"\t"+String(session.history.generated-1)+"\t"+String(len(output))+"\t"+String(now()-turn_started)+"\n"
        except error:
            session.fail()
            print("\nExecution failed:",error,"Use /reset before continuing.",flush=True)
        print()
        if session.history.reason == "limit":
            print("[Reply limit reached]",flush=True)
        elif session.history.reason == "interrupted":
            print("[Reply stopped]",flush=True)
        if observed:
            events += "finish\t"+String(turn)+"\t"+String(session.model.length)+"\t"+session.history.reason+"\t"+String(now()-turn_started)+"\n"
            events += "submitted\t"+String(turn)+"\t0\t"+String(session.model.submitted_rows)+"\t0\n"
            for i in range(len(session.history.tokens)):
                events += "history\t"+String(turn)+"\t"+String(i)+"\t"+String(session.history.tokens[i])+"\t0\n"
        if observed:
            var report = open(args[6],"w")
            report.write(events)
    ctx.synchronize()
    print("Goodbye.",flush=True)
    if observed:
        var report = open(args[6],"w")
        report.write(events)
