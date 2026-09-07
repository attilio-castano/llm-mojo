"""Test-only NumPy fixture readback; no Python in the inference/timed path."""
from max.gpu.host import DeviceBuffer
from layout import TileTensor, row_major
from std.python import Python
from std.math import isfinite


def load_sublayer_fixture(
    mut buffer: DeviceBuffer[DType.bfloat16],
    case_id: Int,
    name: String,
    upstream: Bool = False,
    reference: String = "",
) raises:
    var np = Python.import_module("numpy")
    var a = np.load(
        "build/oracle_data/attention_sublayer/"
        + (reference + "_" if reference != "" else ("upstream_" if upstream else ""))
        + String(case_id)
        + "_"
        + name
        + ".npy",
        allow_pickle=False,
    ).reshape(-1)
    var count = Int(py=a.size)
    var src = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(py=a.ctypes.data)
    )
    with buffer.map_to_host() as mapped:
        var dst = TileTensor(mapped, row_major(count))
        comptime assert dst.flat_rank == 1
        for i in range(count):
            dst[i] = src[unsafe_offset=i].cast[DType.bfloat16]()


def assert_sublayer_fixture(
    mut buffer: DeviceBuffer[DType.bfloat16],
    case_id: Int,
    name: String,
    start: Int,
    rows: Int,
    width: Int,
    tol: Float32,
    require_close: Bool = True,
    reference: String = "upstream",
) raises:
    var np = Python.import_module("numpy")
    var a = np.load(
        "build/oracle_data/attention_sublayer/" + reference + "_"
        + String(case_id)
        + "_"
        + name
        + ".npy",
        allow_pickle=False,
    ).reshape(-1)
    var expected = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(py=a.ctypes.data)
    )
    var worst: Float32 = 0
    var worst_scaled: Float32 = 0
    var failures = 0
    with buffer.map_to_host() as mapped:
        var actual = TileTensor(mapped, row_major(rows * width))
        comptime assert actual.flat_rank == 1
        for i in range(rows * width):
            var want = expected[unsafe_offset=start * width + i]
            var got = rebind[Float32](actual[i].cast[DType.float32]())
            var error = abs(got - want)
            if error > worst:
                worst = error
            var scaled = error / (1 + abs(want))
            if scaled > worst_scaled:
                worst_scaled = scaled
            if not isfinite(got):
                raise Error("nonfinite sublayer output")
            if error > tol + tol * abs(want):
                if failures == 0:
                    print(
                        "first difference",
                        case_id,
                        name,
                        "row",
                        start + i // width,
                        "column",
                        i % width,
                        "got",
                        got,
                        "want",
                        want,
                    )
                failures += 1
    print(
        "checked" if require_close else "composition diagnostic",
        case_id,
        name,
        start,
        rows,
        "max abs",
        worst,
        "max scaled",
        worst_scaled,
        "outside stage budget",
        failures,
    )
    if require_close and failures:
        raise Error("pinned upstream sublayer compatibility mismatch")


def assert_cache_append(
    mut buffer: DeviceBuffer[DType.bfloat16],
    mut source: DeviceBuffer[DType.bfloat16],
    start: Int,
    rows: Int,
    width: Int,
    capacity: Int,
) raises:
    # Exact data-flow requirement, independent of upstream numerical variation.
    with buffer.map_to_host() as mapped:
        var a = TileTensor(mapped, row_major(capacity * width))
        comptime assert a.flat_rank == 1
        var actual_bits = a.ptr.bitcast[UInt16]()
        with source.map_to_host() as sm:
            var s = TileTensor(sm, row_major(rows * width))
            comptime assert s.flat_rank == 1
            var source_bits = s.ptr.bitcast[UInt16]()
            for i in range(rows * width):
                if (
                    actual_bits[unsafe_offset=start * width + i]
                    != source_bits[unsafe_offset=i]
                ):
                    raise Error("cache append changed a source value")
        for i in range((start + rows) * width, capacity * width):
            if rebind[Float32](a[i].cast[DType.float32]()) != 123:
                raise Error("cache write touched unused capacity")


def snapshot(
    mut buffer: DeviceBuffer[DType.bfloat16], count: Int
) raises -> List[UInt16]:
    var values = List[UInt16](capacity=count)
    with buffer.map_to_host() as mapped:
        var a = TileTensor(mapped, row_major(count))
        comptime assert a.flat_rank == 1
        var bits = a.ptr.bitcast[UInt16]()
        for i in range(count):
            values.append(bits[unsafe_offset=i])
    return values^
