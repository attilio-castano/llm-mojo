"""Exact CPU Qwen byte-level tokenizer. No Python or Rust runtime calls.

Prepared tables retain the pinned artifact's IDs/ranks. Workspace ownership is
per call; Tokenizer is immutable after loading. Variant 0 scans; variant 1 uses
a rank/position heap and stable neighbor indices.
"""
from std.collections import Dict

comptime MISSING = UInt64(0xFFFFFFFFFFFFFFFF)


def pair_key(a: Int, b: Int) -> UInt64:
    return (UInt64(a) << 32) | UInt64(b)


struct TableReader(Movable):
    var data: List[UInt8]
    var pos: Int

    def __init__(out self, path: String) raises:
        self.data = open(path, "r").read_bytes()
        self.pos = 0

    def u32(mut self) raises -> Int:
        if self.pos > len(self.data) - 4:
            raise Error("truncated tokenizer table")
        var n = UInt32(0)
        for j in range(4):
            n |= UInt32(self.data[self.pos + j]) << UInt32(8 * j)
        self.pos += 4
        return Int(n)

    def bytes(mut self, n: Int) raises -> List[UInt8]:
        if n < 0 or n > len(self.data) - self.pos:
            raise Error("invalid tokenizer byte extent")
        var result = List[UInt8](capacity=n)
        for i in range(n):
            result.append(self.data[self.pos + i])
        self.pos += n
        return result^

    def align(mut self) raises:
        while self.pos % 4 != 0:
            if self.pos >= len(self.data) or self.data[self.pos] != 0:
                raise Error("invalid tokenizer padding")
            self.pos += 1

    def ints(mut self) raises -> List[Int]:
        var n = self.u32()
        if n > (len(self.data) - self.pos) // 4:
            raise Error("invalid tokenizer integer extent")
        var result = List[Int](capacity=n)
        for _ in range(n):
            result.append(self.u32())
        return result^


@fieldwise_init
struct Candidate(ImplicitlyCopyable):
    var rank: Int
    var pos: Int
    var right: Int
    var left_id: Int
    var right_id: Int
    var result: Int

    def before(self, other: Self) -> Bool:
        return self.rank < other.rank or (
            self.rank == other.rank and self.pos < other.pos
        )


struct TokenizerWorkspace(Movable):
    var ids: List[Int]
    var previous: List[Int]
    var next: List[Int]
    var heap: List[Candidate]

    def __init__(out self):
        self.ids = List[Int]()
        self.previous = List[Int]()
        self.next = List[Int]()
        self.heap = List[Candidate]()

    def push(mut self, candidate: Candidate):
        self.heap.append(candidate)
        var i = len(self.heap) - 1
        while i > 0:
            var parent = (i - 1) // 2
            if not self.heap[i].before(self.heap[parent]):
                break
            var swap = self.heap[parent]
            self.heap[parent] = self.heap[i]
            self.heap[i] = swap
            i = parent

    def pop(mut self) -> Candidate:
        var result = self.heap[0]
        var last = self.heap.pop()
        if len(self.heap) != 0:
            self.heap[0] = last
            var i = 0
            while 2 * i + 1 < len(self.heap):
                var child = 2 * i + 1
                if child + 1 < len(self.heap) and self.heap[child + 1].before(
                    self.heap[child]
                ):
                    child += 1
                if not self.heap[child].before(self.heap[i]):
                    break
                var swap = self.heap[i]
                self.heap[i] = self.heap[child]
                self.heap[child] = swap
                i = child
        return result


def append_utf8(cp: Int, mut result: List[UInt8]):
    if cp < 0x80:
        result.append(UInt8(cp))
    elif cp < 0x800:
        result.append(UInt8(0xC0 | (cp >> 6)))
        result.append(UInt8(0x80 | (cp & 63)))
    elif cp < 0x10000:
        result.append(UInt8(0xE0 | (cp >> 12)))
        result.append(UInt8(0x80 | ((cp >> 6) & 63)))
        result.append(UInt8(0x80 | (cp & 63)))
    else:
        result.append(UInt8(0xF0 | (cp >> 18)))
        result.append(UInt8(0x80 | ((cp >> 12) & 63)))
        result.append(UInt8(0x80 | ((cp >> 6) & 63)))
        result.append(UInt8(0x80 | (cp & 63)))


def codepoints(data: List[UInt8], start: Int, end: Int) raises -> List[Int]:
    var result = List[Int]()
    var i = start
    while i < end:
        var a = Int(data[i])
        var n = 1
        var cp = a
        if a >= 0xC2 and a <= 0xDF:
            n = 2
            cp = a & 31
        elif a >= 0xE0 and a <= 0xEF:
            n = 3
            cp = a & 15
        elif a >= 0xF0 and a <= 0xF4:
            n = 4
            cp = a & 7
        elif a >= 0x80:
            raise Error("invalid UTF-8 input")
        if i + n > end:
            raise Error("incomplete UTF-8 input")
        for j in range(1, n):
            var b = Int(data[i + j])
            if b < 0x80 or b > 0xBF:
                raise Error("invalid UTF-8 continuation")
            cp = (cp << 6) | (b & 63)
        if (
            (n == 2 and cp < 0x80)
            or (n == 3 and cp < 0x800)
            or (n == 4 and cp < 0x10000)
            or (cp >= 0xD800 and cp <= 0xDFFF)
            or cp > 0x10FFFF
        ):
            raise Error("invalid UTF-8 scalar")
        result.append(cp)
        i += n
    return result^


struct Tokenizer(Movable):
    var byte_ids: List[Int]
    var offsets: List[Int]
    var token_bytes: List[UInt8]
    var special: List[UInt8]
    var merges: Dict[UInt64, UInt64]
    var properties: List[Int]
    var decompositions: Dict[Int, List[Int]]
    var compositions: Dict[UInt64, Int]
    var added: List[Int]

    def __init__(out self, path: String) raises:
        self.byte_ids = List[Int]()
        self.offsets = List[Int]()
        self.token_bytes = List[UInt8]()
        self.special = List[UInt8]()
        self.merges = Dict[UInt64, UInt64]()
        self.properties = List[Int]()
        self.decompositions = Dict[Int, List[Int]]()
        self.compositions = Dict[UInt64, Int]()
        self.added = List[Int]()
        var reader = TableReader(path)
        if reader.u32() != 0x51425431 or reader.u32() != 1:
            raise Error("unsupported tokenizer table format")
        var count = reader.u32()
        var byte_count = reader.u32()
        var merge_count = reader.u32()
        var property_count = reader.u32()
        var decomposition_count = reader.u32()
        var composition_count = reader.u32()
        var added_count = reader.u32()
        if (
            count != 151665
            or merge_count != 151387
            or property_count != 0x110000
            or added_count != 22
        ):
            raise Error("tokenizer table dimensions differ from pinned Qwen")
        if (
            byte_count > len(reader.data)
            or decomposition_count > 0x110000
            or composition_count > 0x110000
        ):
            raise Error("invalid tokenizer table dimensions")
        self.byte_ids.reserve(256)
        for _ in range(256):
            var id = reader.u32()
            if id >= count:
                raise Error("invalid byte token ID")
            self.byte_ids.append(id)
        self.offsets.reserve(count + 1)
        var previous = 0
        for i in range(count + 1):
            var offset = reader.u32()
            if (
                offset > byte_count
                or (i > 0 and offset <= previous)
                or (i == 0 and offset != 0)
            ):
                raise Error("invalid tokenizer byte offset")
            self.offsets.append(offset)
            previous = offset
        if previous != byte_count:
            raise Error("tokenizer final offset differs")
        self.token_bytes = reader.bytes(byte_count)
        reader.align()
        self.special = reader.bytes(count)
        reader.align()
        for b in self.special:
            if b > 1:
                raise Error("invalid special-token flag")
        for rank in range(merge_count):
            var left = reader.u32()
            var right = reader.u32()
            var result = reader.u32()
            if left >= count or right >= count or result >= count:
                raise Error("invalid merge token ID")
            var key = pair_key(left, right)
            if key in self.merges:
                raise Error("duplicate merge pair")
            self.merges[key] = pair_key(rank, result)
        self.properties.reserve(property_count)
        for _ in range(property_count):
            var value = reader.u32()
            if value >> 24 != 0 or (value & 0xFFFF) >= 2048:
                raise Error("invalid Unicode property")
            self.properties.append(value)
        for _ in range(decomposition_count):
            var cp = reader.u32()
            var seq = reader.ints()
            if (
                cp >= property_count
                or cp in self.decompositions
                or len(seq) == 0
            ):
                raise Error("invalid Unicode decomposition")
            for v in seq:
                if v >= property_count or (v >= 0xD800 and v < 0xE000):
                    raise Error("invalid decomposition scalar")
            self.decompositions[cp] = seq^
        for _ in range(composition_count):
            var a = reader.u32()
            var b = reader.u32()
            var c = reader.u32()
            if (
                a >= property_count
                or b >= property_count
                or c >= property_count
                or pair_key(a, b) in self.compositions
            ):
                raise Error("invalid Unicode composition")
            self.compositions[pair_key(a, b)] = c
        for _ in range(added_count):
            var id = reader.u32()
            if id < 151643 or id >= count:
                raise Error("invalid added token ID")
            for existing in self.added:
                if existing == id:
                    raise Error("duplicate added token ID")
            self.added.append(id)
        if reader.pos != len(reader.data):
            raise Error("trailing tokenizer table data")
        for b in range(256):
            var id = self.byte_ids[b]
            if (
                self.offsets[id + 1] - self.offsets[id] != 1
                or Int(self.token_bytes[self.offsets[id]]) != b
            ):
                raise Error("byte-token mapping differs")

    def combining(self, cp: Int) -> Int:
        return self.properties[cp] >> 16

    def has(self, cp: Int, mask: Int) -> Bool:
        return (self.properties[cp] & mask) != 0

    def compose(self, a: Int, b: Int) -> Int:
        if a >= 0x1100 and a < 0x1113 and b >= 0x1161 and b < 0x1176:
            return 0xAC00 + ((a - 0x1100) * 21 + b - 0x1161) * 28
        if (
            a >= 0xAC00
            and a < 0xD7A4
            and (a - 0xAC00) % 28 == 0
            and b > 0x11A7
            and b < 0x11C3
        ):
            return a + b - 0x11A7
        return self.compositions.get(pair_key(a, b), -1)

    def normalize(self, points: List[Int]) raises -> List[Int]:
        var decomposed = List[Int]()
        for cp in points:
            if cp >= 0xAC00 and cp < 0xD7A4:
                var syllable = cp - 0xAC00
                decomposed.append(0x1100 + syllable // 588)
                decomposed.append(0x1161 + (syllable % 588) // 28)
                if syllable % 28 != 0:
                    decomposed.append(0x11A7 + syllable % 28)
            elif cp in self.decompositions:
                for v in self.decompositions[cp]:
                    decomposed.append(v)
            else:
                decomposed.append(cp)
        # Stable canonical ordering. This intentionally inspectable baseline
        # can be quadratic on a pathological run of combining marks (not BPE).
        for i in range(1, len(decomposed)):
            var cc = self.combining(decomposed[i])
            if cc == 0:
                continue
            var j = i
            while j > 0 and self.combining(decomposed[j - 1]) > cc:
                var swap = decomposed[j - 1]
                decomposed[j - 1] = decomposed[j]
                decomposed[j] = swap
                j -= 1
        var result = List[Int]()
        var starter = -1
        var last_cc = 0
        for cp in decomposed:
            var cc = self.combining(cp)
            var combined = -1
            if starter >= 0 and (last_cc == 0 or last_cc < cc):
                combined = self.compose(result[starter], cp)
            if combined >= 0:
                result[starter] = combined
            else:
                if cc == 0:
                    starter = len(result)
                result.append(cp)
                last_cc = cc
        return result^

    def piece_end(self, points: List[Int], start: Int) -> Int:
        var n = len(points)
        var c = points[start]
        if c == 39 and start + 1 < n:
            var flags = self.properties[points[start + 1]]
            # s,t,m,d; re,ve,ll (case classes come from the Rust regex engine).
            if (flags & ((1 << 3) | (1 << 4) | (1 << 8) | (1 << 10))) != 0:
                return start + 2
            if start + 2 < n:
                var after = self.properties[points[start + 2]]
                if (
                    (flags & ((1 << 5) | (1 << 7))) != 0
                    and (after & (1 << 6)) != 0
                ) or ((flags & (1 << 9)) != 0 and (after & (1 << 9)) != 0):
                    return start + 3
        var i = start
        if (
            not self.has(c, 1)
            and c != 10
            and c != 13
            and not self.has(c, 2)
            and i + 1 < n
            and self.has(points[i + 1], 1)
        ):
            i += 1
        if self.has(points[i], 1):
            while i < n and self.has(points[i], 1):
                i += 1
            return i
        if self.has(c, 2):
            return start + 1
        i = start
        if c == 32 and i + 1 < n and not self.has(points[i + 1], 7):
            i += 1
        if not self.has(points[i], 7):
            while i < n and not self.has(points[i], 7):
                i += 1
            while i < n and (points[i] == 10 or points[i] == 13):
                i += 1
            return i
        i = start
        var last_newline = -1
        while i < n and self.has(points[i], 4):
            if points[i] == 10 or points[i] == 13:
                last_newline = i
            i += 1
        if last_newline >= 0:
            return last_newline + 1
        if i == n or i - start == 1:
            return i
        return i - 1

    def candidate(self, mut work: TokenizerWorkspace, left: Int):
        if left < 0:
            return
        var right = work.next[left]
        if right < 0:
            return
        var value = self.merges.get(
            pair_key(work.ids[left], work.ids[right]), MISSING
        )
        if value != MISSING:
            work.push(
                Candidate(
                    Int(value >> 32),
                    left,
                    right,
                    work.ids[left],
                    work.ids[right],
                    Int(value & 0xFFFFFFFF),
                )
            )

    def bpe(
        self,
        bytes: List[UInt8],
        mut work: TokenizerWorkspace,
        mut output: List[Int],
        variant: Int = 1,
    ) raises:
        if variant != 0 and variant != 1:
            raise Error("unsupported BPE variant")
        work.ids.clear()
        for b in bytes:
            work.ids.append(self.byte_ids[Int(b)])
        if variant == 0:
            while len(work.ids) > 1:
                var best = MISSING
                var position = -1
                for i in range(len(work.ids) - 1):
                    var value = self.merges.get(
                        pair_key(work.ids[i], work.ids[i + 1]), MISSING
                    )
                    if value != MISSING and (
                        position < 0 or (value >> 32) < (best >> 32)
                    ):
                        best = value
                        position = i
                if position < 0:
                    break
                work.ids[position] = Int(best & 0xFFFFFFFF)
                for j in range(position + 1, len(work.ids) - 1):
                    work.ids[j] = work.ids[j + 1]
                _ = work.ids.pop()
            for id in work.ids:
                output.append(id)
            return
        work.previous.clear()
        work.next.clear()
        work.heap.clear()
        for i in range(len(work.ids)):
            work.previous.append(i - 1)
            work.next.append(i + 1 if i + 1 < len(work.ids) else -1)
        for i in range(len(work.ids)):
            self.candidate(work, i)
        while len(work.heap) > 0:
            var item = work.pop()
            if (
                work.ids[item.pos] != item.left_id
                or work.next[item.pos] != item.right
                or work.ids[item.right] != item.right_id
            ):
                continue
            var after = work.next[item.right]
            work.ids[item.pos] = item.result
            work.ids[item.right] = -1
            work.next[item.pos] = after
            if after >= 0:
                work.previous[after] = item.pos
            self.candidate(work, work.previous[item.pos])
            self.candidate(work, item.pos)
        for id in work.ids:
            if id >= 0:
                output.append(id)

    def encode_span(
        self,
        bytes: List[UInt8],
        start: Int,
        end: Int,
        mut work: TokenizerWorkspace,
        mut output: List[Int],
        variant: Int,
    ) raises:
        var points = self.normalize(codepoints(bytes, start, end))
        var i = 0
        var piece = List[UInt8]()
        while i < len(points):
            var stop = self.piece_end(points, i)
            piece.clear()
            for j in range(i, stop):
                append_utf8(points[j], piece)
            self.bpe(piece, work, output, variant)
            i = stop

    def encode_bytes(
        self, bytes: List[UInt8], mut work: TokenizerWorkspace, variant: Int = 1
    ) raises -> List[Int]:
        if variant != 0 and variant != 1:
            raise Error("unsupported BPE variant")
        # Validate the entire caller input even when an added token splits it.
        _ = codepoints(bytes, 0, len(bytes))
        var output = List[Int]()
        var i = 0
        var plain = 0
        while i < len(bytes):
            var best_id = -1
            var best_length = 0
            if bytes[i] == UInt8(60):
                for id in self.added:
                    var size = self.offsets[id + 1] - self.offsets[id]
                    if size <= best_length or i + size > len(bytes):
                        continue
                    var matches = True
                    for j in range(size):
                        if (
                            bytes[i + j]
                            != self.token_bytes[self.offsets[id] + j]
                        ):
                            matches = False
                            break
                    if matches:
                        best_id = id
                        best_length = size
            if best_id >= 0:
                self.encode_span(bytes, plain, i, work, output, variant)
                output.append(best_id)
                i += best_length
                plain = i
            else:
                i += 1
        self.encode_span(bytes, plain, len(bytes), work, output, variant)
        return output^

    def encode(
        self, text: String, mut work: TokenizerWorkspace, variant: Int = 1
    ) raises -> List[Int]:
        var bytes = List[UInt8]()
        for b in text.as_bytes():
            bytes.append(b)
        return self.encode_bytes(bytes, work, variant)

    def decode_bytes(
        self, ids: List[Int], skip_special: Bool = False
    ) raises -> List[UInt8]:
        var stream = TokenizerDecoder()
        var output = List[UInt8]()
        for id in ids:
            stream.push(self, id, output, skip_special)
        stream.finish(output)
        return output^

    def decode(
        self, ids: List[Int], skip_special: Bool = False
    ) raises -> String:
        var bytes = self.decode_bytes(ids, skip_special)
        return String(from_utf8=bytes)


struct TokenizerDecoder(Movable):
    var pending: List[UInt8]

    def __init__(out self):
        self.pending = List[UInt8]()

    def push(
        mut self,
        tokenizer: Tokenizer,
        id: Int,
        mut output: List[UInt8],
        skip_special: Bool = False,
    ) raises:
        if id < 0 or id >= len(tokenizer.special):
            raise Error("invalid tokenizer token ID")
        if skip_special and tokenizer.special[id] != 0:
            return
        for i in range(tokenizer.offsets[id], tokenizer.offsets[id + 1]):
            self.pending.append(tokenizer.token_bytes[i])
        self.emit(output, False)

    def finish(mut self, mut output: List[UInt8]):
        self.emit(output, True)

    def emit(mut self, mut output: List[UInt8], final: Bool):
        var i = 0
        while i < len(self.pending):
            var a = Int(self.pending[i])
            var n = 0
            if a < 0x80:
                n = 1
            elif a >= 0xC2 and a <= 0xDF:
                n = 2
            elif a >= 0xE0 and a <= 0xEF:
                n = 3
            elif a >= 0xF0 and a <= 0xF4:
                n = 4
            if n == 0:
                append_utf8(0xFFFD, output)
                i += 1
                continue
            var j = 1
            while j < n and i + j < len(self.pending):
                var b = Int(self.pending[i + j])
                if (
                    b < 0x80
                    or b > 0xBF
                    or (
                        j == 1
                        and (
                            (a == 0xE0 and b < 0xA0)
                            or (a == 0xED and b >= 0xA0)
                            or (a == 0xF0 and b < 0x90)
                            or (a == 0xF4 and b >= 0x90)
                        )
                    )
                ):
                    break
                j += 1
            if j == n:
                for k in range(n):
                    output.append(self.pending[i + k])
                i += n
            elif i + j == len(self.pending) and not final:
                break
            else:
                append_utf8(0xFFFD, output)
                i += j
        var remaining = len(self.pending) - i
        for j in range(remaining):
            self.pending[j] = self.pending[i + j]
        while len(self.pending) > remaining:
            _ = self.pending.pop()
