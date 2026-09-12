"""Native plain-text greedy generation with optional diagnostic events."""
from std.sys import argv
from std.ffi import external_call
from max.gpu.host import DeviceContext
from llm_mojo.model import QwenModel, select_configuration, select_token_selection, select_copy_free, select_residual_norm
from llm_mojo.tokenizer import Tokenizer, TokenizerWorkspace, TokenizerDecoder


def now() -> UInt64:
    return external_call["clock_gettime_nsec_np", UInt64](UInt32(8))


def generation_budget(prompt_length: Int, maximum: Int) raises -> Int:
    if prompt_length < 1 or prompt_length > 4096 or maximum < 0 or maximum > 4096:
        raise Error("invalid prompt or generation limit")
    return min(maximum,4096-prompt_length)


def is_stop(token: Int) -> Bool:
    return token == 151645 or token == 151643


def main() raises:
    var started = now()
    var args = argv()
    if len(args) != 7 and len(args) != 8:
        raise Error("generate prepared-model tokenizer-tables prompt-file max-new-tokens chunk-rows policy")
    var diagnostics = len(args) == 8
    var maximum = Int(args[4])
    var chunk_rows = Int(args[5])
    if maximum < 0 or maximum > 4096 or chunk_rows < 0 or chunk_rows > 4096:
        raise Error("invalid generation or chunk limit")
    _ = select_configuration(args[6],1,1,"")
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
    var offset = 0
    while offset < prompt_length:
        var rows = min(max_rows,prompt_length-offset)
        var ids = List[Int](capacity=rows)
        for i in range(rows):
            ids.append(history[offset+i])
        var configuration = select_configuration(args[6],rows,offset+rows,ctx.name())
        model.forward(ctx,ids,configuration,"",select_token_selection(args[6],rows,ctx.name()),False,select_copy_free(args[6],rows,ctx.name()),select_residual_norm(args[6],rows,ctx.name()))
        if diagnostics:
            events += "configuration\t"+String(offset)+"\t"+String(configuration)+"\t0\n"
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
            model.forward(ctx,ids,select_configuration(args[6],1,model.length+1,ctx.name()),"",select_token_selection(args[6],1,ctx.name()),False,select_copy_free(args[6],1,ctx.name()),select_residual_norm(args[6],1,ctx.name()))
            if diagnostics:
                ctx.synchronize()
                events += "decode\t"+String(step)+"\t1\t"+String(now()-decode_started)+"\n"
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
