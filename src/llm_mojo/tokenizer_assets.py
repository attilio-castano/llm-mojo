"""Reproducible tokenizer preparation; text computation lives in Mojo.

The public setup entrypoint may download the single pinned model artifact. Test
and benchmark paths call ensure_source(download=False) and never use networking.
"""
from __future__ import annotations

import argparse
import array
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

from ._repository import repository_root, environment_tool

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
URL = f'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/{REVISION}/tokenizer.json'
FORMAT = 1
MAGIC = 0x51425431
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def asset_directory():
    return (
        repository_root() / "build/checkpoints/qwen2.5-0.5b-instruct" / REVISION
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def setup_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def ensure_source(directory, *, download=False):
    path = Path(directory) / "tokenizer.json"
    if path.exists():
        if sha(path) != SOURCE_SHA:
            raise ValueError(
                f'tokenizer checksum mismatch: {path}; remove the damaged file and rerun setup'
            )
        return path
    if not download:
        raise FileNotFoundError(f'{path}; run llm-mojo-tokenizer setup first')
    print(
        "Downloading pinned tokenizer.json (7 MB)...",
        file=sys.stderr,
        flush=True,
    )
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read(8_000_001)
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError("downloaded tokenizer checksum mismatch")
    atomic_write(path, data)
    return path


def byte_decoder():
    original = (
        list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    )
    return {chr(b): b for b in original} | {
        chr(256 + i): b
        for i, b in enumerate(x for x in range(256) if x not in original)
    }


def validate_config(data):
    model = data["model"]
    expected_model = dict(
        type="BPE",
        dropout=None,
        unk_token=None,
        continuing_subword_prefix="",
        end_of_word_suffix="",
        fuse_unk=False,
        byte_fallback=False,
    )
    if {
        k: v for k, v in model.items() if k not in ("vocab", "merges")
    } != expected_model:
        raise ValueError("unsupported BPE configuration")
    bytelevel = dict(
        type="ByteLevel",
        add_prefix_space=False,
        trim_offsets=False,
        use_regex=False,
    )
    expected_split = dict(
        type="Sequence",
        pretokenizers=[
            dict(
                type="Split",
                pattern={"Regex": PATTERN},
                behavior="Isolated",
                invert=False,
            ),
            bytelevel,
        ],
    )
    if (
        data["normalizer"] != {"type": "NFC"}
        or data["pre_tokenizer"] != expected_split
    ):
        raise ValueError("unsupported normalization or splitting configuration")
    if data["decoder"] != bytelevel or data["post_processor"] != bytelevel:
        raise ValueError("unsupported byte-level configuration")
    if data["padding"] is not None or data["truncation"] is not None:
        raise ValueError("padding and truncation are outside this contract")
    for t in data["added_tokens"]:
        if any(t[k] for k in ("single_word", "lstrip", "rstrip", "normalized")):
            raise ValueError("unsupported added-token flags")


def generator_identity():
    root = repository_root()
    names = [
        "src/llm_mojo/tokenizer_assets.py",
        "tests/fixtures/tokenizer/generate.py",
        "tests/fixtures/generate.py.lock",
        "tests/fixtures/tokenizer_reference.py",
    ]
    return {name: sha(root / name) for name in names}


def prepared_directory(directory):
    return Path(directory) / f'prepared-v{FORMAT}'


def prepared_valid(directory):
    target = prepared_directory(directory)
    try:
        manifest = json.loads((target / "manifest.json").read_text())
        return (
            manifest["format"] == FORMAT
            and manifest["source_sha256"] == SOURCE_SHA
            and manifest["generators"] == generator_identity()
            and manifest["tables_sha256"] == sha(target / "tables.bin")
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_tables(directory, unicode):
    data = json.loads(ensure_source(directory).read_text())
    validate_config(data)
    vocab = data["model"]["vocab"]
    reverse = byte_decoder()
    all_vocab = dict(vocab)
    special = {}
    for t in data["added_tokens"]:
        if t["content"] in all_vocab or t["id"] in all_vocab.values():
            raise ValueError("duplicate added token")
        all_vocab[t["content"]] = t["id"]
        special[t["id"]] = t["special"]
    count = len(all_vocab)
    if set(all_vocab.values()) != set(range(count)):
        raise ValueError("vocabulary IDs must be dense and unique")
    strings = [""] * count
    for token, index in all_vocab.items():
        strings[index] = token
    token_bytes = []
    for token in strings:
        token_bytes.append(
            bytes(reverse[c] for c in token) if all(
                c in reverse for c in token
            ) else token.encode()
        )
    offsets = [0]
    for b in token_bytes:
        offsets.append(offsets[-1] + len(b))
    byte_ids = [0] * 256
    for char, byte in reverse.items():
        byte_ids[byte] = vocab[char]
    merges = []
    seen = set()
    for rule in data["model"]["merges"]:
        left, right = rule.split(" ")
        item = (vocab[left], vocab[right], vocab[left + right])
        if item[:2] in seen:
            raise ValueError("duplicate merge pair")
        seen.add(item[:2])
        merges.append(item)
    props, decomp, compose, unicode_info = unicode
    words = array.array(
        "I",
        [
            MAGIC,
            FORMAT,
            count,
            offsets[-1],
            len(merges),
            len(props),
            len(decomp),
            len(compose),
            len(data["added_tokens"]),
        ],
    )
    words.extend(byte_ids)
    words.extend(offsets)

    def packed(a):
        if sys.byteorder != "little":
            a = a[:]
            a.byteswap()
        return a.tobytes()

    out = bytearray(packed(words))
    out.extend(b"".join(token_bytes))
    out.extend(b"\0" * (-len(out) % 4))
    out.extend(bytes(int(special.get(i, False)) for i in range(count)))
    out.extend(b"\0" * (-len(out) % 4))
    words = array.array("I")
    for item in merges:
        words.extend(item)
    words.extend(props)
    for cp, seq in sorted(decomp.items()):
        words.extend([cp, len(seq), *seq])
    for (a, b), cp in sorted(compose.items()):
        words.extend([a, b, cp])
    for t in data["added_tokens"]:
        words.append(t["id"])
    out.extend(packed(words))
    target = prepared_directory(directory)
    manifest = dict(
        format=FORMAT,
        source_sha256=SOURCE_SHA,
        source_url=URL,
        revision=REVISION,
        generators=generator_identity(),
        tables_sha256=hashlib.sha256(out).hexdigest(),
        tables_bytes=len(out),
        vocabulary=count,
        base_vocabulary=len(vocab),
        merges=len(merges),
        unicode=unicode_info,
    )
    atomic_write(target / "tables.bin", out)
    atomic_write(
        target / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return target


def ensure_prepared(directory=None, *, download=True):
    directory = Path(directory) if directory else asset_directory()
    with setup_lock(directory):
        ensure_source(directory, download=download)
        if not prepared_valid(directory):
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--locked",
                    "--script",
                    "tests/fixtures/tokenizer_reference.py",
                    "--prepare-only",
                    "--asset-dir",
                    str(directory),
                ],
                cwd=repository_root(),
                check=True,
            )
        if not prepared_valid(directory):
            raise ValueError("prepared tokenizer verification failed")
    return prepared_directory(directory) / "tables.bin"


def ensure_binary():
    root = repository_root()
    directory = root / "build/tokenizer"
    source_paths = [
        root / "src/llm_mojo/tokenizer.mojo",
        root / "src/llm_mojo/tokenizer_cli.mojo",
        root / "uv.lock",
    ]
    identity = {str(p.relative_to(root)): sha(p) for p in source_paths}
    binary = directory / "tokenizer"
    manifest_path = directory / "binary.json"
    with setup_lock(directory):
        try:
            previous = json.loads(manifest_path.read_text())
            valid = previous["sources"] == identity and previous[
                "binary_sha256"
            ] == sha(binary)
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            fd, name = tempfile.mkstemp(
                prefix="tokenizer.", suffix=".part", dir=directory
            )
            os.close(fd)
            temporary = Path(name)
            try:
                subprocess.run(
                    [
                        environment_tool("mojo"),
                        "build",
                        "-I",
                        "src",
                        str(source_paths[1]),
                        "-o",
                        str(temporary),
                    ],
                    cwd=root,
                    check=True,
                )
                if identity != {
                    str(p.relative_to(root)): sha(p) for p in source_paths
                }:
                    raise ValueError(
                        "tokenizer source changed during compilation"
                    )
                digest = sha(temporary)
                os.replace(temporary, binary)
                atomic_write(
                    manifest_path,
                    (
                        json.dumps(
                            dict(sources=identity, binary_sha256=digest),
                            indent=2,
                        )
                        + "\n"
                    ).encode(),
                )
            finally:
                temporary.unlink(missing_ok=True)
    return binary


def main():
    p = argparse.ArgumentParser(
        description="Prepare or execute the pure Mojo Qwen tokenizer."
    )
    p.add_argument("operation", choices=["setup", "encode", "decode"])
    p.add_argument(
        "value",
        nargs="?",
        help="Text or comma-separated token IDs; omit encode text to read stdin.",
    )
    p.add_argument("--asset-dir", type=Path)
    p.add_argument("--skip-special-tokens", action="store_true")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Require existing tokenizer.json; never download it.",
    )
    a = p.parse_args()
    table = ensure_prepared(a.asset_dir, download=not a.offline)
    binary = ensure_binary()
    if a.operation == "setup":
        print(table)
        print(binary)
        return
    if a.operation == "encode":
        # Preserve stdin bytes (including CRLF/NUL) and avoid argv size limits.
        data = (
            a.value.encode("utf-8") if a.value
            is not None else sys.stdin.buffer.read()
        )
        with tempfile.TemporaryDirectory(
            prefix="llm-mojo-tokenizer-"
        ) as temporary:
            path = Path(temporary) / "input.txt"
            path.write_bytes(data)
            subprocess.run(
                [str(binary), str(table), "encode-file", str(path), "0"],
                check=True,
            )
    else:
        value = a.value if a.value is not None else sys.stdin.read()
        subprocess.run(
            [
                str(binary),
                str(table),
                "decode",
                value,
                str(int(a.skip_special_tokens)),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
