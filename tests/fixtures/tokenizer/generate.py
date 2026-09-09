"""Frozen Rust-backend oracle, Unicode tables, and deterministic tokenizer cases."""
import argparse
import array
import hashlib
import json
from pathlib import Path
import random
import sys
import unicodedata as ud

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from llm_mojo.tokenizer_assets import (
    asset_directory,
    ensure_source,
    prepare_tables,
    atomic_write,
    sha,
    PATTERN,
)
import tokenizers
from tokenizers import Tokenizer, Regex, normalizers, pre_tokenizers


def unicode_tables():
    if tokenizers.__version__ != "0.19.1" or ud.unidata_version != "15.0.0":
        raise RuntimeError(
            "Unicode preparation requires locked tokenizers 0.19.1 and Python Unicode 15.0.0"
        )
    props = array.array("I", [0]) * 0x110000
    # Query the actual Rust regex engine. Offsets returned by this Python API
    # count Unicode scalars, not UTF-8 bytes. Chunk boundaries affect no property.
    patterns = [r"[^\p{L}]+", r"[^\p{N}]+", r"[^\s]+"] + [
        f'(?i:[^{x}]+)' for x in "strev mld".replace(" ", "")
    ]
    for bit, pattern in enumerate(patterns):
        splitter = pre_tokenizers.Split(Regex(pattern), "removed")
        for first, last in [(0, 0xD800), (0xE000, 0x110000)]:
            for start in range(first, last, 65536):
                text = "".join(map(chr, range(start, min(start + 65536, last))))
                for _, (a, b) in splitter.pre_tokenize_str(text):
                    for cp in range(start + a, start + b):
                        props[cp] |= 1 << bit
    nfd = normalizers.NFD()
    nfc = normalizers.NFC()
    decomp = {}
    compose = {}
    differences = []
    ccc_differences = []
    for cp in range(0x110000):
        if 0xD800 <= cp < 0xE000:
            continue
        c = chr(cp)
        normalized = nfd.normalize_str(c)
        if normalized != ud.normalize("NFD", c):
            differences.append(cp)
        if normalized != c and not 0xAC00 <= cp < 0xD7A4:
            decomp[cp] = list(map(ord, normalized))
        ccc = ud.combining(c)
        if ccc and normalized == c:
            probe = "a\u0345" + c + "\u0334"
            if nfd.normalize_str(probe) != ud.normalize("NFD", probe):
                # The pinned Rust normalizer predates some Unicode 15 marks.
                # A ccc=0 character is a barrier to canonical reordering.
                if nfd.normalize_str(probe) != probe:
                    raise RuntimeError(
                        f'unresolved reference combining class U+{cp:04X}'
                    )
                ccc_differences.append(cp)
                ccc = 0
        props[cp] |= ccc << 16
        raw = ud.decomposition(c)
        if raw and not raw.startswith("<"):
            pair = tuple(int(x, 16) for x in raw.split())
            if (
                len(pair) == 2
                and nfc.normalize_str("".join(map(chr, pair))) == c
            ):
                compose[pair] = cp
    info = dict(
        tokenizers=tokenizers.__version__,
        oracle_python=sys.version.split()[0],
        backend_extension_sha256=sha(
            Path(sys.modules["tokenizers.tokenizers"].__file__)
        ),
        python_unicode=ud.unidata_version,
        classification="exhaustive scalar queries against tokenizers.Regex",
        decomposition="exhaustive scalar tokenizers.normalizers.NFD queries; algorithmic Hangul",
        composition="Unicode 15 canonical pairs filtered by Rust NFC",
        combining_classes="Unicode 15 with Rust NFD barrier probes",
        decomposition_differences=differences,
        combining_class_differences=ccc_differences,
    )
    print(
        "Unicode tables verified against Rust:",
        len(differences),
        "decomposition differences;",
        len(ccc_differences),
        "combining-class differences",
        flush=True,
    )
    return props, decomp, compose, info


def cases(holdout=False):
    fixed = [
        "",
        "hello",
        "Hello world",
        " Hello world",
        "We're I'M he's I'LL we'd they've can't",
        "'ſ 'S 't 're 'VE",
        "0 1234567890 ١٢٣ Ⅷ ²",
        "\r\n\n \t  x   ",
        "café cafe\u0301 Å A\u030a \u212b",
        "中文 日本語 한국어 مرحبا नमस्ते",
        "🙂👨\u200d👩\u200d👧\u200d👦🏳️\u200d🌈",
        "a\0b\x01\x7f\u0085\u00a0",
        "a\u0345\u0300\u0334",
        "\u1193",
        "\U00011938\U00011935\U00011930",
        "\u1100\u1161\u11a8 각",
        "def f(x):\n    return x ** 2\n",
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    ]
    if holdout:
        fixed = [
            "မြန်မာစာ Ελληνικά עברית ไทย",
            "\U0001e4ec\u0301",
            "\u0344\u0f73\u1f82",
        ]
    rng = random.Random(871019 if holdout else 571019)
    alphabet = list("abcXYZ012 ' \t\r\n!.,_<>|/é中🙂") + [
        "\u0301",
        "\u0345",
        "\u0334",
        "\U00011935",
        "\U00011930",
        "\u1100",
        "\u1161",
        "\u11a8",
        "\u00a0",
        "\u2028",
        "\U0001e4ec",
    ]
    fixed += [
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 200)))
        for _ in range(256)
    ]
    fixed += [
        c * n
        for c in ["a", " ", "!", "中"]
        for n in ([137, 1031] if holdout else [128, 512, 2048, 4096])
    ]
    return fixed


def write_fixtures(source, target, holdout=False, supplemental=False):
    tok = Tokenizer.from_file(str(source))
    data = json.loads(source.read_text())
    texts = cases(holdout)
    for t in data["added_tokens"]:
        texts += [
            t["content"],
            "a" + t["content"] + "b",
            t["content"] * 2,
            "é\u0301" + t["content"] + "\u0301",
        ]
    # Boundary stress around all nontrivial canonical decompositions and CCCs.
    if not holdout:
        texts += [
            chr(cp)
            for cp in range(0x110000)
            if ud.decomposition(chr(cp))
            and not ud.decomposition(chr(cp)).startswith("<")
        ]
        texts += [
            "a\u0345" + chr(cp) + "\u0334"
            for cp in range(0x110000)
            if ud.combining(chr(cp))
        ]
    if supplemental:
        texts = [
            ud.normalize("NFD", chr(cp))
            for cp in range(0x110000)
            if ud.decomposition(chr(cp))
            and not ud.decomposition(chr(cp)).startswith("<")
        ]
    words = array.array("I", [len(texts)])
    identities = []

    def put(items):
        words.append(len(items))
        words.extend(items)

    for text in texts:
        ids = tok.encode(text, add_special_tokens=False).ids
        normalized = tok.normalizer.normalize_str(text)
        pieces = tok.pre_tokenizer.pre_tokenize_str(normalized)
        put(text.encode())
        put(ids)
        put(tok.decode(ids, skip_special_tokens=False).encode())
        put(tok.decode(ids, skip_special_tokens=True).encode())
        put(normalized.encode())
        # Save normalized scalar boundaries for a separate splitter assertion.
        put([b for _, (_, b) in pieces])
        identities.append([hashlib.sha256(text.encode()).hexdigest(), ids])
    # Independent token-ID sequences, including all vocabulary entries separately.
    sequences = [
        [i] for i in range(tok.get_vocab_size())
    ] if not holdout and not supplemental else []
    rng = random.Random(101777 if holdout else 91771)
    sequences += [
        [rng.randrange(256) for _ in range(rng.randrange(1, 48))]
        for _ in range(256)
    ]
    sequences += [[t["id"] for t in data["added_tokens"]], list(range(256))]
    words.append(len(sequences))
    for ids in sequences:
        put(ids)
        put(tok.decode(ids, skip_special_tokens=False).encode())
        put(tok.decode(ids, skip_special_tokens=True).encode())
    if sys.byteorder != "little":
        words.byteswap()
    raw = words.tobytes()
    atomic_write(target, raw)
    manifest = dict(
        tokenizers=tokenizers.__version__,
        source_sha256=sha(source),
        fixture_sha256=hashlib.sha256(raw).hexdigest(),
        text_cases=len(texts),
        decode_cases=len(sequences),
        holdout=holdout,
        recipes_sha256=hashlib.sha256(
            json.dumps(
                identities, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    )
    frozen = Path(__file__).with_name(
        "unicode_checksums.json" if supplemental else "holdout_checksums.json" if holdout else "checksums.json"
    )
    if frozen.exists() and json.loads(frozen.read_text()) != manifest:
        raise RuntimeError("tokenizer oracle changed; frozen contract mismatch")
    if not frozen.exists():
        # First held-out capture stays outside Git so the candidate stays clean.
        destination = target.with_suffix(".json") if holdout else frozen
        atomic_write(
            destination,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
    print(json.dumps(manifest), flush=True)


def benchmark_fixtures(source):
    tok = Tokenizer.from_file(str(source))
    families = {
        "prose": "The tokenizer converts text into IDs. We measure complete calls and retain every sample. ",
        "multilingual": "中文 日本語 한국어 مرحبا नमस्ते café 🙂 ",
        "code": "def square(x):\n    return x * x  # simple function\n",
        "whitespace": "  a\t b\r\n\n    c         ",
    }
    texts = []
    for family, base in families.items():
        for size in (64, 256, 4096, 16384):
            raw = (base * (size // len(base.encode()) + 2)).encode()[:size]
            texts.append((family, raw.decode("utf-8", errors="ignore")))
    texts += [("long_piece", "a" * n) for n in (128, 512, 2048, 4096)]
    words = array.array("I", [len(texts)])
    metadata = []

    def put(items):
        words.append(len(items))
        words.extend(items)

    for case, (family, text) in enumerate(texts, 1):
        normalized = tok.normalizer.normalize_str(text)
        pieces = [
            normalized[a:b].encode()
            for _, (a, b) in tok.pre_tokenizer.pre_tokenize_str(normalized)
        ]
        ids = tok.encode(text, add_special_tokens=False).ids
        put(text.encode())
        put(ids)
        put(tok.decode(ids, skip_special_tokens=False).encode())
        words.append(len(pieces))
        for piece in pieces:
            put(piece)
        metadata.append(
            dict(
                case=case,
                family=family,
                input_bytes=len(text.encode()),
                pieces=len(pieces),
                max_piece_bytes=max(map(len, pieces), default=0),
                output_tokens=len(ids),
            )
        )
    if sys.byteorder != "little":
        words.byteswap()
    raw = words.tobytes()
    target = ROOT / "build/oracle_data/tokenizer/benchmark.bin"
    atomic_write(target, raw)
    manifest = dict(
        tokenizers=tokenizers.__version__,
        source_sha256=sha(source),
        fixture_sha256=hashlib.sha256(raw).hexdigest(),
        cases=metadata,
    )
    frozen = Path(__file__).with_name("benchmark_checksums.json")
    if frozen.exists() and json.loads(frozen.read_text()) != manifest:
        raise RuntimeError("benchmark fixtures changed")
    if not frozen.exists():
        atomic_write(frozen, (json.dumps(manifest, indent=2) + "\n").encode())
    print("Benchmark fixtures:", len(texts), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset-dir", type=Path, default=asset_directory())
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--unicode", action="store_true")
    a = p.parse_args()
    source = ensure_source(a.asset_dir)
    if a.prepare_only:
        prepare_tables(a.asset_dir, unicode_tables())
        return
    if a.benchmark:
        benchmark_fixtures(source)
        return
    target = (
        ROOT
        / "build/oracle_data/tokenizer"
        / (
            "unicode.bin" if a.unicode else "holdout.bin" if a.holdout else "development.bin"
        )
    )
    write_fixtures(source, target, a.holdout, a.unicode)


if __name__ == "__main__":
    main()
