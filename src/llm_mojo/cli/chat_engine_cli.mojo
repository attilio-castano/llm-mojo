"""Optional synchronous EngineCore chat with resident weights and exact history."""
from std.sys import argv
from llm_mojo.models.qwen2.chat import DEFAULT_SYSTEM
from llm_mojo.models.qwen2.engine_chat import EngineChatSession
from llm_mojo.serving.engine import TOKEN_EVENT
from llm_mojo.models.qwen2.tokenizer import Tokenizer, TokenizerWorkspace, TokenizerDecoder
from llm_mojo.runtime.terminal import block_interrupt, interrupted, read_line
from llm_mojo.runtime.clock import now
from llm_mojo.models.qwen2.tokens import is_stop
from llm_mojo.models.qwen2.plan import MAX_CONTEXT


def main() raises:
    var args = argv()
    if len(args) != 7:
        raise Error("chat prepared tokenizer maximum chunk-rows system-file report-file (empty = defaults)")
    var maximum = Int(args[3])
    var chunk = Int(args[4])
    if maximum < 1 or maximum > MAX_CONTEXT or chunk < 1 or chunk > MAX_CONTEXT:
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
    var session = EngineChatSession(args[1],tokenizer,work,system,chunk)
    print("Ready — engine reference on",session.runner.ctx.name(),"/",session.runner.ctx.api(),flush=True)
    print("/reset: new conversation · /exit: quit · Ctrl-C: stop reply or clear input",flush=True)
    if observed:
        events += "load\t0\t0\t"+session.runner.ctx.name()+"/"+session.runner.ctx.api()+"\t"+String(now()-started)+"\n"
    if observed:
        events += "engine_mode\t0\t0\treference-27/recompute-history\t0\n"
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
            session.reset()
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
                events += "rejected\t"+String(turn)+"\t0\t"+String(len(session.history.tokens))+"\t0\n"
            continue
        turn += 1
        var cached_before = 0
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
                    session.abort()
                var record = session.step()
                for event in record.events:
                    if event.kind != TOKEN_EVENT:
                        continue
                    var token = event.token_id
                    var output = List[UInt8]()
                    if not is_stop(token):
                        decoder.push(tokenizer,token,output,True)
                        if len(output)>0:
                            print(String(from_utf8=output),end="",flush=True)
                            if observed:
                                events += "text\t"+String(turn)+"\t"+String(event.generated_tokens-1)+"\t"+String(len(output))+"\t"+String(now()-turn_started)+"\n"
                    if observed:
                        events += "token\t"+String(turn)+"\t"+String(event.generated_tokens-1)+"\t"+String(token)+"\t"+String(now()-turn_started)+"\n"
            var output = List[UInt8]()
            decoder.finish(output)
            if len(output)>0:
                print(String(from_utf8=output),end="",flush=True)
                if observed:
                    events += "text\t"+String(turn)+"\t"+String(session.history.generated-1)+"\t"+String(len(output))+"\t"+String(now()-turn_started)+"\n"
        except error:
            session.fail()
            print("\nExecution failed:",error,"Restart the chat process.",flush=True)
            if observed:
                events += "finish\t"+String(turn)+"\t0\terror\t"+String(now()-turn_started)+"\n"
                var written = 0
                for count in session.kv.written:
                    written += count
                events += "kv_free\t"+String(turn)+"\t0\t"+String(session.engine.blocks.free_blocks())+"\t0\n"
                events += "kv_total\t"+String(turn)+"\t0\t"+String(session.kv.blocks)+"\t0\n"
                events += "kv_owned\t"+String(turn)+"\t0\t"+String(session.kv.blocks-session.engine.blocks.free_blocks())+"\t0\n"
                events += "kv_written\t"+String(turn)+"\t0\t"+String(written)+"\t0\n"
                events += "engine_live\t"+String(turn)+"\t0\t"+String(session.engine.live())+"\t0\n"
                for i in range(len(session.history.tokens)):
                    events += "history\t"+String(turn)+"\t"+String(i)+"\t"+String(session.history.tokens[i])+"\t0\n"
                var report = open(args[6],"w")
                report.write(events)
            raise error
        print()
        if session.history.reason == "limit":
            print("[Reply limit reached]",flush=True)
        elif session.history.reason == "interrupted":
            print("[Reply stopped]",flush=True)
        session.check_drained()
        if observed:
            var written = 0
            for count in session.kv.written:
                written += count
            events += "finish\t"+String(turn)+"\t"+"0"+"\t"+session.history.reason+"\t"+String(now()-turn_started)+"\n"
            events += "submitted\t"+String(turn)+"\t0\t"+String(session.runner.model.submitted_rows-session.submitted_before)+"\t0\n"
            events += "engine_rows\t"+String(turn)+"\t0\t"+String(session.turn_rows)+"\t0\n"
            events += "engine_steps\t"+String(turn)+"\t0\t"+String(session.turn_steps)+"\t0\n"
            events += "kv_free\t"+String(turn)+"\t0\t"+String(session.engine.blocks.free_blocks())+"\t0\n"
            events += "kv_total\t"+String(turn)+"\t0\t"+String(session.kv.blocks)+"\t0\n"
            events += "kv_owned\t"+String(turn)+"\t0\t"+String(session.kv.blocks-session.engine.blocks.free_blocks())+"\t0\n"
            events += "kv_written\t"+String(turn)+"\t0\t"+String(written)+"\t0\n"
            events += "engine_live\t"+String(turn)+"\t0\t"+String(session.engine.live())+"\t0\n"
            for i in range(len(session.history.tokens)):
                events += "history\t"+String(turn)+"\t"+String(i)+"\t"+String(session.history.tokens[i])+"\t0\n"
        if observed:
            var report = open(args[6],"w")
            report.write(events)
    session.check_drained()
    session.runner.ctx.synchronize()
    print("Goodbye.",flush=True)
    if observed:
        var report = open(args[6],"w")
        report.write(events)
