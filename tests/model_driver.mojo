"""Development-only full-model entrypoint; requires externally verified assets."""
from std.sys import argv
from max.gpu.host import DeviceContext
from llm_mojo.model import QwenModel
from model_operation_support import capture_operations


def integers(text: String) raises -> List[Int]:
    var values = List[Int]()
    for item in text.split(","):
        values.append(Int(String(item)))
    return values^


def main() raises:
    var args = argv()
    if len(args) == 5 and args[1] == "--operations":
        var ctx = DeviceContext()
        capture_operations(ctx,args[2],args[3],args[4])
        return
    if len(args) != 6:
        raise Error("model_driver prepared-dir comma-token-ids schedule configurations capture-root")
    var ids = integers(args[2])
    var schedule = integers(args[3])
    var configurations = integers(args[4])
    if len(schedule) != len(configurations):
        raise Error("one configuration is required per call")
    var maximum = 0
    var total = 0
    for rows in schedule:
        if rows < 1:
            raise Error("empty schedule call")
        maximum = max(maximum,rows)
        total += rows
    if total != len(ids) or total > 4096:
        raise Error("schedule does not cover token IDs")
    var ctx = DeviceContext()
    print("model device",ctx.name(),"backend",ctx.api())
    var model = QwenModel(ctx,args[1],min(4096,len(ids)+3),maximum)
    # Exact untouched-cache checks use a finite recognizable poison pattern.
    for i in range(24):
        model.layers[i].cache.key.enqueue_fill(123)
        model.layers[i].cache.value.enqueue_fill(123)
    var offset = 0
    for i in range(len(schedule)):
        var chunk = List[Int]()
        for j in range(schedule[i]):
            chunk.append(ids[offset+j])
        var capture = args[5]+"/call_"+String(i) if args[5] != "-" else String("")
        model.forward(ctx,chunk,configurations[i],capture)
        print("call",i,"token",model.greedy(ctx),"cache_length",model.length,"submitted_layer_rows",model.submitted_rows)
        offset += schedule[i]
