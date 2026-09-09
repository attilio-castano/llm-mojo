"""Native plain-text greedy generation; development candidate pending acceptance."""
from std.sys import argv
from max.gpu.host import DeviceContext
from llm_mojo.model import QwenModel, select_configuration
from llm_mojo.tokenizer import Tokenizer, TokenizerWorkspace, TokenizerDecoder


def main() raises:
    var args = argv()
    if len(args) != 7:
        raise Error("generate prepared-model tokenizer-tables prompt-file max-new-tokens chunk-rows policy")
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
    var budget = min(maximum,4096-prompt_length)
    if budget == 0:
        return
    var max_rows = min(chunk_rows,prompt_length) if chunk_rows > 0 else prompt_length
    var ctx = DeviceContext()
    var model = QwenModel(ctx,args[1],4096,max_rows)
    var offset = 0
    while offset < prompt_length:
        var rows = min(max_rows,prompt_length-offset)
        var ids = List[Int](capacity=rows)
        for i in range(rows):
            ids.append(history[offset+i])
        model.forward(ctx,ids,select_configuration(args[6],rows,offset+rows,ctx.name()))
        offset += rows
    var decoder = TokenizerDecoder()
    for step in range(budget):
        var token = model.greedy(ctx)
        history.append(token)
        if token == 151645 or token == 151643:
            break
        var bytes = List[UInt8]()
        decoder.push(tokenizer,token,bytes,True)
        if len(bytes) > 0:
            print(String(from_utf8=bytes),end="",flush=True)
        if step+1 < budget:
            var ids: List[Int] = [token]
            model.forward(ctx,ids,select_configuration(args[6],1,model.length+1,ctx.name()))
    var bytes = List[UInt8]()
    decoder.finish(bytes)
    if len(bytes) > 0:
        print(String(from_utf8=bytes),end="",flush=True)
