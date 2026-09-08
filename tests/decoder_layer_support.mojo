"""Test-only NumPy transport; the engine does not import Python."""
from std.python import Python, PythonObject
from max.gpu.host import DeviceBuffer


def decoder_support() raises -> PythonObject:
    Python.add_to_path("tests")
    return Python.import_module("decoder_layer_support")


def load_decoder(
    mut buffer: DeviceBuffer[DType.bfloat16], name: String, label: String,
    start: Int, count: Int,
) raises:
    var support = decoder_support()
    with buffer.map_to_host() as mapped:
        support.load(Int(mapped.unsafe_ptr()), name, label, start, count)


def poison_decoder(mut buffer: DeviceBuffer[DType.bfloat16], active: Int) raises:
    var support = decoder_support()
    with buffer.map_to_host() as mapped:
        support.poison(Int(mapped.unsafe_ptr()), active, len(buffer))


def decoder_snapshot(mut buffer: DeviceBuffer[DType.bfloat16], count: Int) raises -> PythonObject:
    var support = decoder_support()
    with buffer.map_to_host() as mapped:
        return support.snapshot(Int(mapped.unsafe_ptr()), count)


def check_decoder(
    mut buffer: DeviceBuffer[DType.bfloat16], name: String, stage: String,
    schedule: String, start: Int, rows: Int, width: Int, mode: String = "layer",
) raises:
    var support = decoder_support()
    with buffer.map_to_host() as mapped:
        support.check(Int(mapped.unsafe_ptr()), name, stage, schedule, start, rows,
                      width, len(buffer), mode)


def exact_decoder(mut buffer: DeviceBuffer[DType.bfloat16], expected: PythonObject, label: String) raises:
    var support = decoder_support()
    with buffer.map_to_host() as mapped:
        support.exact(Int(mapped.unsafe_ptr()), expected, label)


def check_decoder_cache(mut buffer: DeviceBuffer[DType.bfloat16],
                        mut produced: DeviceBuffer[DType.bfloat16], previous: PythonObject,
                        name: String, label: String, start: Int, rows: Int,
                        width: Int, capacity: Int) raises:
    var support = decoder_support()
    var current = decoder_snapshot(produced, rows*width)
    with buffer.map_to_host() as mapped:
        support.check_cache(Int(mapped.unsafe_ptr()), current, previous, name, label,
                            start, rows, width, capacity)


def check_decoder_active(mut buffer: DeviceBuffer[DType.bfloat16], name: String,
                         stage: String, schedule: String, start: Int,
                         rows: Int, width: Int) raises:
    var support = decoder_support()
    with buffer.map_to_host() as mapped:
        support.check(Int(mapped.unsafe_ptr()),name,stage,schedule,start,rows,width,rows*width,"async")


def check_decoder_slice(mut buffer: DeviceBuffer[DType.bfloat16], name: String,
                        stage: String, start: Int, rows: Int, width: Int) raises:
    var support = decoder_support()
    with buffer.map_to_host() as mapped:
        support.check(Int(mapped.unsafe_ptr()),name,stage,"full_slice",start,rows,width,rows*width,"operation")
