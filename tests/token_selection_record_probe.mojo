"""Finite host execution probe of the actual maintained UInt32 merge.

Input/output records contain words only; score arithmetic is deliberately
absent. The independent Python runner computes arbitrary-precision scores.
Finite success does not establish an arbitrary-input proof or GPU refinement.
"""
from std.sys import argv
from llm_mojo.kernels.token_selection_contract import (
    ArgmaxRecord, empty_argmax, merge_argmax,
)


def _record(fields: List[Int], offset: Int) raises -> ArgmaxRecord:
    var rank = fields[offset]
    var token = fields[offset + 1]
    var bad = fields[offset + 2]
    if rank < 0 or rank > 4294967295 or token < 0 or token > 4294967295:
        raise Error("Probe input exceeds UInt32")
    if bad != 0 and bad != 1:
        raise Error("Probe flag must be Boolean")
    return ArgmaxRecord(UInt32(rank), UInt32(token), UInt32(bad))


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("Expected one generated case-file argument")
    var count = 0
    # Retain owners for both levels of StringSlice; no slice refers to a
    # String temporary that expired before the case was decoded.
    var input_text = open(args[1], "r").read()
    # The pinned runtime's splitlines() also splits TABs. Use the observed
    # literal newline delimiter so a TSV record remains one complete row.
    var lines = input_text.split("\n")
    for line_index in range(len(lines)):
        var line_text = String(lines[line_index])
        if line_text.byte_length() == 0:
            # A trailing newline may produce one empty EOF entry. Empty
            # interior rows are malformed, rather than silently skipped.
            if line_index != len(lines) - 1:
                raise Error("Empty interior case line: " + String(line_index))
            continue
        var fields = line_text.split("\t")
        if len(fields) < 2:
            raise Error("Missing case kind or identity at line " + String(line_index))
        var kind = String(fields[0])
        var identity = Int(String(fields[1]))
        if identity != count:
            raise Error("Case identities must be consecutive")
        var words = List[Int]()
        for index in range(2, len(fields)):
            words.append(Int(String(fields[index])))
        if kind == "E":
            if len(fields) != 2:
                raise Error("Malformed empty case")
            var value = empty_argmax()
            print("E", identity, value.rank, value.token, value.invalid)
        elif kind == "U":
            if len(fields) != 5:
                raise Error("Malformed unary case")
            var a = _record(words, 0)
            var same = merge_argmax(a, a)
            var left = merge_argmax(empty_argmax(), a)
            var right = merge_argmax(a, empty_argmax())
            print("U", identity, same.rank, same.token, same.invalid,
                left.rank, left.token, left.invalid, right.rank, right.token, right.invalid)
        elif kind == "P":
            if len(fields) != 8:
                raise Error("Malformed pair case")
            var a = _record(words, 0)
            var b = _record(words, 3)
            var value = merge_argmax(a, b)
            print("P", identity, value.rank, value.token, value.invalid)
        elif kind == "T":
            if len(fields) != 11:
                raise Error("Malformed triple case")
            var a = _record(words, 0)
            var b = _record(words, 3)
            var c = _record(words, 6)
            var left = merge_argmax(merge_argmax(a, b), c)
            var right = merge_argmax(a, merge_argmax(b, c))
            print("T", identity, left.rank, left.token, left.invalid,
                right.rank, right.token, right.invalid)
        else:
            raise Error("Unknown probe case kind")
        count += 1
    print("Native UInt32 record diagnostic completed", count)
