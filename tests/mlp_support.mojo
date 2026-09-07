"""Host-only NumPy transport; never imported by engine operations."""
from std.python import Python, PythonObject
from max.gpu.host import DeviceBuffer
from llm_mojo.mlp import MLPWorkspace


def mlp_support() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, "tests")
    return Python.import_module("mlp_support")


def load_mlp(
    mut buffer: DeviceBuffer[DType.bfloat16],
    path: String,
    start: Int,
    count: Int,
) raises:
    var support = mlp_support()
    with buffer.map_to_host() as mapped:
        support.load_slice(Int(mapped.unsafe_ptr()), path, start, count)


def poison_mlp(
    mut buffer: DeviceBuffer[DType.bfloat16], active: Int, capacity: Int
) raises:
    var support = mlp_support()
    with buffer.map_to_host() as mapped:
        support.poison(Int(mapped.unsafe_ptr()), active, capacity)


def check_mlp(
    mut buffer: DeviceBuffer[DType.bfloat16],
    case_id: String,
    stage: String,
    mode: String,
    start: Int,
    rows: Int,
    width: Int,
    capacity: Int,
) raises:
    var support = mlp_support()
    with buffer.map_to_host() as mapped:
        support.stage_check(
            Int(mapped.unsafe_ptr()),
            case_id,
            stage,
            mode,
            start,
            rows,
            width,
            capacity,
        )


def poison_work(mut work: MLPWorkspace, rows: Int) raises:
    var h = work.hidden
    var i = work.intermediate
    var cap = work.max_rows
    poison_mlp(work.normalized, rows * h, cap * h)
    poison_mlp(work.gate, rows * i, cap * i)
    poison_mlp(work.up, rows * i, cap * i)
    poison_mlp(work.activated, rows * i, cap * i)
    poison_mlp(work.gated, rows * i, cap * i)
    poison_mlp(work.down, rows * h, cap * h)
    poison_mlp(work.output, rows * h, cap * h)


def check_work(
    mut work: MLPWorkspace, case_id: String, mode: String, start: Int, rows: Int
) raises:
    var h = work.hidden
    var i = work.intermediate
    var cap = work.max_rows
    check_mlp(work.normalized, case_id, "N", mode, start, rows, h, cap * h)
    check_mlp(work.gate, case_id, "G", mode, start, rows, i, cap * i)
    check_mlp(work.up, case_id, "U", mode, start, rows, i, cap * i)
    check_mlp(work.activated, case_id, "A", mode, start, rows, i, cap * i)
    check_mlp(work.gated, case_id, "S", mode, start, rows, i, cap * i)
    check_mlp(work.down, case_id, "D", mode, start, rows, h, cap * h)
    check_mlp(work.output, case_id, "Y", mode, start, rows, h, cap * h)
