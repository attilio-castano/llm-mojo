"""Native CPU timings, excluding setup/oracles and including call allocations."""
from std.sys import argv
from std.ffi import external_call
from llm_mojo.tokenizer import (
    Tokenizer,
    TokenizerWorkspace,
    TokenizerDecoder,
    TableReader,
)


def cpu_time_ns() -> UInt64:
    # Darwin SDK: CLOCK_UPTIME_RAW=8, clockid_t is unsigned int. Mojo's
    # perf_counter_ns rounds to microseconds on this host, losing short calls.
    return external_call["clock_gettime_nsec_np", UInt64](UInt32(8))


def read_bytes(mut reader: TableReader) raises -> List[UInt8]:
    var integers = reader.ints()
    var result = List[UInt8]()
    for i in integers:
        if i > 255:
            raise Error("invalid fixture byte")
        result.append(UInt8(i))
    return result^


def perform(
    mode: String,
    variant: Int,
    table: String,
    tokenizer: Tokenizer,
    text: List[UInt8],
    pieces: List[List[UInt8]],
    ids: List[Int],
    mut work: TokenizerWorkspace,
    mut result_ids: List[Int],
    mut result_bytes: List[UInt8],
) raises -> Int:
    if mode == "load":
        var loaded = Tokenizer(table)
        return len(loaded.token_bytes)
    if mode == "decode":
        result_bytes = tokenizer.decode_bytes(ids)
        return len(result_bytes)
    if mode == "stream":
        var stream = TokenizerDecoder()
        result_bytes.clear()
        for id in ids:
            stream.push(tokenizer, id, result_bytes)
        stream.finish(result_bytes)
        return len(result_bytes)
    if mode == "encode":
        result_ids = tokenizer.encode_bytes(text, work, variant)
    elif mode == "bpe":
        result_ids.clear()
        for piece in pieces:
            tokenizer.bpe(piece, work, result_ids, variant)
    else:
        raise Error("unknown tokenizer benchmark mode")
    return len(result_ids)


def main() raises:
    var args = argv()
    if len(args) != 9:
        raise Error(
            "expected table fixture case mode candidate first reps warmup"
        )
    var index = Int(args[3])
    var mode = args[4]
    var candidate = Int(args[5])
    var first = Int(args[6])
    var reps = Int(args[7])
    var warmup = Int(args[8])
    var tokenizer = Tokenizer(args[1])
    var reader = TableReader(args[2])
    var count = reader.u32()
    if index < 1 or index > count or candidate < 0 or candidate > 1:
        raise Error("invalid benchmark case")
    var text = List[UInt8]()
    var ids = List[Int]()
    var expected_bytes = List[UInt8]()
    var pieces = List[List[UInt8]]()
    for _ in range(index):
        text = read_bytes(reader)
        ids = reader.ints()
        expected_bytes = read_bytes(reader)
        pieces.clear()
        var number = reader.u32()
        for _ in range(number):
            pieces.append(read_bytes(reader))
    var work = TokenizerWorkspace()
    for variant in range(2):
        var actual = tokenizer.encode_bytes(text, work, variant)
        if len(actual) != len(ids):
            raise Error("benchmark encode length")
        for i in range(len(ids)):
            if actual[i] != ids[i]:
                raise Error("benchmark encode IDs")
        actual.clear()
        for piece in pieces:
            tokenizer.bpe(piece, work, actual, variant)
        if len(actual) != len(ids):
            raise Error("benchmark BPE length")
        for i in range(len(ids)):
            if actual[i] != ids[i]:
                raise Error("benchmark BPE IDs")
    var decoded = tokenizer.decode_bytes(ids)
    if len(decoded) != len(expected_bytes):
        raise Error("benchmark decode length")
    for i in range(len(decoded)):
        if decoded[i] != expected_bytes[i]:
            raise Error("benchmark decoded bytes")
    print("api: cpu")
    print("timer: clock_gettime_nsec_np CLOCK_UPTIME_RAW")
    print("operation: tokenizer")
    print("case:", index, "mode:", mode)
    print("correctness: passed")
    print(
        "workspace capacity:",
        work.ids.capacity(),
        work.previous.capacity(),
        work.next.capacity(),
        work.heap.capacity(),
    )
    var sink = 0
    var result_ids = List[Int]()
    var result_bytes = List[UInt8]()
    for arm in range(2):
        var is_candidate = (arm == 0) == (first == 1)
        var variant = candidate if is_candidate else 0
        var label = "candidate" if is_candidate else "control"
        for sample in range(warmup + reps):
            var start = cpu_time_ns()
            var value = perform(
                mode,
                variant,
                args[1],
                tokenizer,
                text,
                pieces,
                ids,
                work,
                result_ids,
                result_bytes,
            )
            var elapsed = Float64(cpu_time_ns() - start) / 1000.0
            sink += value
            # Consume every output after the timing boundary. The compiler must
            # preserve the full output, not only its length or final element.
            for id in result_ids:
                sink += id
            for byte in result_bytes:
                sink += Int(byte)
            if sample >= warmup:
                print("SAMPLE", label, variant, sample - warmup, elapsed)
    print("checksum:", sink)
    print("BENCHMARK_COMPLETE")
