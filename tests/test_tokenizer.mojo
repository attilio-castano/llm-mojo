"""Exact Rust-oracle parity, both BPE algorithms, Unicode, and streaming."""
from std.testing import assert_equal, assert_true, assert_raises
from std.sys import argv
from llm_mojo.tokenizer import (
    Tokenizer,
    TokenizerWorkspace,
    TokenizerDecoder,
    TableReader,
    codepoints,
    append_utf8,
    pair_key,
)


def fixture_bytes(mut reader: TableReader) raises -> List[UInt8]:
    var ints = reader.ints()
    var result = List[UInt8]()
    for x in ints:
        if x > 255:
            raise Error("invalid fixture byte")
        result.append(UInt8(x))
    return result^


def equal_bytes(actual: List[UInt8], expected: List[UInt8]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])


def equal_ids(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])


def verify_decode(
    tokenizer: Tokenizer, ids: List[Int], expected: List[UInt8], skip: Bool
) raises:
    equal_bytes(tokenizer.decode_bytes(ids, skip), expected)
    var stream = TokenizerDecoder()
    var output = List[UInt8]()
    for id in ids:
        stream.push(tokenizer, id, output, skip)
        assert_true(len(stream.pending) <= 3)
        # Every emitted prefix must already be valid UTF-8.
        _ = codepoints(output, 0, len(output))
    stream.finish(output)
    equal_bytes(output, expected)
    stream.finish(output)
    equal_bytes(output, expected)


def synthetic_merges(mut tokenizer: Tokenizer) raises:
    var work = TokenizerWorkspace()
    var a = tokenizer.byte_ids[97]
    var b = tokenizer.byte_ids[98]
    var c = tokenizer.byte_ids[99]
    var bytes: List[UInt8] = [97, 98, 99]
    tokenizer.merges.clear()
    tokenizer.merges[pair_key(a, b)] = pair_key(5, 1000)
    tokenizer.merges[pair_key(b, c)] = pair_key(1, 1001)
    var expected: List[Int] = [a, 1001]
    for variant in range(2):
        var result = List[Int]()
        tokenizer.bpe(bytes, work, result, variant)
        equal_ids(result, expected)
    # Reversing rank must change this discriminating result.
    tokenizer.merges[pair_key(a, b)] = pair_key(0, 1000)
    expected = [1000, c]
    for variant in range(2):
        var result = List[Int]()
        tokenizer.bpe(bytes, work, result, variant)
        equal_ids(result, expected)
    # Overlapping equal-rank occurrences must choose the leftmost pair.
    tokenizer.merges.clear()
    tokenizer.merges[pair_key(a, a)] = pair_key(0, 1000)
    tokenizer.merges[pair_key(1000, a)] = pair_key(1, 1001)
    bytes = [97, 97, 97]
    expected = [1001]
    for variant in range(2):
        var result = List[Int]()
        tokenizer.bpe(bytes, work, result, variant)
        equal_ids(result, expected)


def malformed_tables(path: String) raises:
    var original = open(path, "r").read_bytes()
    var target = String("build/oracle_data/tokenizer/malformed.bin")
    var truncated = List[UInt8]()
    truncated.append(UInt8(1))
    with open(target, "w") as f:
        f.write(truncated)
    with assert_raises():
        _ = Tokenizer(target)
    # Corrupt the first byte-token ID without changing dimensions.
    for j in range(4):
        original[36 + j] = UInt8(255)
    with open(target, "w") as f:
        f.write(original)
    with assert_raises():
        _ = Tokenizer(target)


def main() raises:
    var args = argv()
    var root = String(
        "build/checkpoints/qwen2.5-0.5b-instruct/7ae557604adf67be50417f59c2c2f167def9a775/prepared-v1/tables.bin"
    )
    var fixture = String("build/oracle_data/tokenizer/development.bin")
    if len(args) > 1:
        fixture = args[1]
    var tokenizer = Tokenizer(root)
    var work = TokenizerWorkspace()
    var reader = TableReader(fixture)
    var count = reader.u32()
    for test in range(count):
        var text = fixture_bytes(reader)
        var expected_ids = reader.ints()
        var decoded = fixture_bytes(reader)
        var skipped = fixture_bytes(reader)
        var normalized = fixture_bytes(reader)
        var boundaries = reader.ints()
        try:
            for variant in range(2):
                equal_ids(
                    tokenizer.encode_bytes(text, work, variant), expected_ids
                )
            verify_decode(tokenizer, expected_ids, decoded, False)
            verify_decode(tokenizer, expected_ids, skipped, True)
            var points = tokenizer.normalize(codepoints(text, 0, len(text)))
            var bytes = List[UInt8]()
            for cp in points:
                append_utf8(cp, bytes)
            equal_bytes(bytes, normalized)
            var ends = List[Int]()
            var i = 0
            while i < len(points):
                i = tokenizer.piece_end(points, i)
                ends.append(i)
            equal_ids(ends, boundaries)
        except e:
            print("FAIL text fixture", test)
            raise e
    var decode_count = reader.u32()
    for test in range(decode_count):
        var ids = reader.ints()
        var decoded = fixture_bytes(reader)
        var skipped = fixture_bytes(reader)
        try:
            verify_decode(tokenizer, ids, decoded, False)
            verify_decode(tokenizer, ids, skipped, True)
        except e:
            print("FAIL decode fixture", test)
            raise e
    assert_equal(reader.pos, len(reader.data))
    var invalid = List[UInt8]()
    invalid.append(UInt8(0xC0))
    with assert_raises():
        _ = tokenizer.encode_bytes(invalid, work)
    var empty = List[UInt8]()
    with assert_raises():
        _ = tokenizer.encode_bytes(empty, work, 99)
    var bad_ids: List[Int] = [-1]
    with assert_raises():
        _ = tokenizer.decode_bytes(bad_ids)
    bad_ids[0] = 151665
    with assert_raises():
        _ = tokenizer.decode_bytes(bad_ids)
    malformed_tables(root)
    synthetic_merges(tokenizer)
    print(
        "Tokenizer exact parity:",
        count,
        "text cases;",
        decode_count,
        "decode cases; scan, heap and streaming passed",
    )
