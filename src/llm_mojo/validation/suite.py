"""Prepare numerical references, check frozen anchors, then run all tests."""

import argparse
import hashlib
import json
import os
import subprocess
import sys

from .._repository import environment_tool, repository_root
from . import fixtures as large_fixtures


def run(*args):
    print("+", *args, flush=True)
    subprocess.run(args, cwd=repository_root(), check=True,
                   env={**os.environ, "MODULAR_DEBUG": "device-sync-mode"})


def prepare(cache=True, regenerate=False):
    root = repository_root()
    fixtures = root / "build/oracle_data"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "__init__.mojo").touch()
    run("uv", "run", "--locked", "--script", "tests/fixtures/generate.py")
    for name in ("rms_norm", "linear", "rope", "attention"):
        (fixtures / name / "__init__.mojo").touch()
    run(sys.executable, "tests/fixtures/attention/generate_decode.py")
    run(sys.executable, "tests/fixtures/attention/generate_prefill.py")
    anchors = json.loads((root / "tests/fixtures/checksums.json").read_text())
    for name, expected in anchors["sha256"].items():
        actual = hashlib.sha256((fixtures / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"oracle changed: {name}; review the numerical contract before updating anchors")
    manifest = json.loads((fixtures / 'attention/prefill_manifest.json').read_text())
    for name, expected in manifest['array_sha256'].items():
        if hashlib.sha256((fixtures / 'attention' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'prefill oracle array changed: {name}')
    # The three large families come from the shared store unless --no-fixture-cache.
    sources = large_fixtures.inputs(root) if cache else None

    def large(name):
        family = large_fixtures.FAMILIES[name]
        if cache:
            large_fixtures.ensure(family, root, sources, run, regenerate=regenerate)
        else:
            large_fixtures.generate_locally(family, root, run)
    large("attention_sublayer")
    run("uv", "run", "--locked", "--script", "tests/fixtures/generate.py", "mlp", "--", "--self-test")
    large("mlp")
    run("uv", "run", "--locked", "--script", "tests/fixtures/decoder_reference.py", "--self-test")
    run("uv", "run", "--locked", "--script", "tests/fixtures/model_calibration.py", "--self-test")
    large("decoder_layer")
    from llm_mojo.models.qwen2.tokenizer_assets import ensure_prepared
    ensure_prepared(download=False)
    run("uv", "run", "--locked", "--script", "tests/fixtures/tokenizer_reference.py")
    run("uv", "run", "--locked", "--script", "tests/fixtures/tokenizer_reference.py", "--unicode")
    run(sys.executable, "tests/fixtures/chat_reference.py", "--pack")
    print("All oracles match the frozen anchors.", flush=True)
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    fixture_source = parser.add_mutually_exclusive_group()
    fixture_source.add_argument("--regenerate-fixtures", action="store_true",
                                help="regenerate the cached large oracles and require a byte-for-byte match")
    fixture_source.add_argument("--no-fixture-cache", action="store_true",
                                help="generate the large oracles in this checkout instead of the shared store")
    args = parser.parse_args(argv)
    sources = prepare(cache=not args.no_fixture_cache, regenerate=args.regenerate_fixtures)
    if not args.prepare_only:
        run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
        for test in sorted((repository_root() / "tests").glob("test_*.mojo")):
            run(environment_tool("mojo"), "run", "-I", "src", "-I", "build",
                "-I", "tests", str(test.relative_to(repository_root())))
            if test.name == "test_tokenizer.mojo":
                run(environment_tool("mojo"), "run", "-I", "src", str(test.relative_to(repository_root())),
                    "build/oracle_data/tokenizer/unicode.bin")

        run(sys.executable, "-m", "llm_mojo.benchmarks.smoke")
    # A pass describes the checkout as it stands only if the oracles still match its inputs.
    if sources is not None:
        large_fixtures.confirm_unchanged(repository_root(), sources)


if __name__ == "__main__":
    main()
