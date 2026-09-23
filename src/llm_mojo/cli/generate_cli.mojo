"""Native plain-text greedy generation with optional diagnostic events."""
from std.sys import argv
from llm_mojo.runtime.clock import now
from llm_mojo.models.qwen2.tokens import is_stop
from max.gpu.host import DeviceContext
from llm_mojo.models.qwen2.model import QwenModel, generation_budget
from llm_mojo.models.qwen2.plan import execution_plan
from llm_mojo.models.qwen2.tokenizer import Tokenizer, TokenizerWorkspace, TokenizerDecoder


def main() raises:
    var started = now()
    var args = argv()
    if len(args) != 7 and len(args) != 8:
        raise Error("generate prepared-model tokenizer-tables prompt-file max-new-tokens chunk-rows mode [report]")
    var diagnostics = len(args) == 8
    var maximum = Int(args[4])
    var chunk_rows = Int(args[5])
    var mode = String(args[6])
    if maximum < 0 or maximum > 4096 or chunk_rows < 0 or chunk_rows > 4096:
        raise Error("invalid generation or chunk limit")
    _ = execution_plan(mode,1,1,"")
    var tokenizer = Tokenizer(args[2])
    var workspace = TokenizerWorkspace()
    var text = open(args[3],"r").read_bytes()
    var history = tokenizer.encode_bytes(text,workspace)
    var prompt_length = len(history)
    if prompt_length < 1 or prompt_length > 4096:
        raise Error("prompt must encode to 1..4096 tokens")
    var budget = generation_budget(prompt_length,maximum)
    var events = String("event\tindex\tvalue\tnanoseconds\n")
    if diagnostics:
        events += "mode\t0\t"+mode+"\t0\n"
        for i in range(prompt_length):
            events += "prompt\t"+String(i)+"\t"+String(history[i])+"\t0\n"
    if budget == 0:
        if len(args) == 8:
            var report = open(args[7],"w")
            report.write(events+"finish\t0\tlimit\t"+String(now()-started)+"\n")
        return
    var max_rows = min(chunk_rows,prompt_length) if chunk_rows > 0 else prompt_length
    var ctx = DeviceContext()
    var model = QwenModel(ctx,args[1],4096,max_rows)
    if diagnostics:
        events += "device\t0\t"+ctx.name()+"/"+ctx.api()+"\t0\n"
        events += "load\t0\t0\t"+String(now()-started)+"\n"
    var prefill_started = now()
    var calls = 0
    var offset = 0
    while offset < prompt_length:
        var rows = min(max_rows,prompt_length-offset)
        var ids = List[Int](capacity=rows)
        for i in range(rows):
            ids.append(history[offset+i])
        var plan = execution_plan(mode,rows,offset+rows,ctx.name())
        model.forward(ctx,ids,plan)
        if diagnostics:
            events += "configuration\t"+String(offset)+"\t"+String(plan.configuration)+"\t0\n"
            events += "route\t"+String(calls)+"\t"+model.last_route.describe()+"\t0\n"
        calls += 1
        offset += rows
    if diagnostics:
        ctx.synchronize()
        events += "prefill\t0\t"+String(prompt_length)+"\t"+String(now()-prefill_started)+"\n"
    var decoder = TokenizerDecoder()
    var finish_reason = String("limit")
    for step in range(budget):
        var token = model.greedy(ctx)
        history.append(token)
        if diagnostics:
            events += "token\t"+String(step)+"\t"+String(token)+"\t"+String(now()-started)+"\n"
        if is_stop(token):
            finish_reason = "stop"
            break
        var bytes = List[UInt8]()
        decoder.push(tokenizer,token,bytes,True)
        if len(bytes) > 0:
            print(String(from_utf8=bytes),end="",flush=True)
        if step+1 < budget:
            var ids: List[Int] = [token]
            var decode_started = now()
            model.forward(ctx,ids,execution_plan(mode,1,model.length+1,ctx.name()))
            if diagnostics:
                ctx.synchronize()
                events += "decode\t"+String(step)+"\t1\t"+String(now()-decode_started)+"\n"
                events += "route\t"+String(calls)+"\t"+model.last_route.describe()+"\t0\n"
            calls += 1
    var bytes = List[UInt8]()
    decoder.finish(bytes)
    if len(bytes) > 0:
        print(String(from_utf8=bytes),end="",flush=True)
    if diagnostics:
        events += "cache\t0\t"+String(model.length)+"\t0\n"
        events += "submitted\t0\t"+String(model.submitted_rows)+"\t0\n"
        events += "finish\t"+String(len(history)-prompt_length)+"\t"+finish_reason+"\t"+String(now()-started)+"\n"
        var report = open(args[7],"w")
        report.write(events)
