"""Prepare independent oracles, check frozen anchors, then run all tests."""

import argparse
import hashlib
import json
import os
import subprocess
import sys

from ._repository import environment_tool, repository_root


def run(*args):
    print("+", *args, flush=True)
    subprocess.run(args, cwd=repository_root(), check=True,
                   env={**os.environ, "MODULAR_DEBUG": "device-sync-mode"})


def prepare():
    fixtures = repository_root() / "build/oracle_data"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "__init__.mojo").touch()
    run("uv", "run", "--locked", "--script", "tests/fixtures/generate.py")
    for name in ("rms_norm", "linear", "rope", "attention"):
        (fixtures / name / "__init__.mojo").touch()
    run(sys.executable, "tests/fixtures/attention/generate_decode.py")
    run(sys.executable, "tests/fixtures/attention/generate_prefill.py")
    anchors = json.loads((repository_root() / "tests/fixtures/checksums.json").read_text())
    for name, expected in anchors["sha256"].items():
        actual = hashlib.sha256((fixtures / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"oracle changed: {name}; review the numerical contract before updating anchors")
    manifest = json.loads((fixtures / 'attention/prefill_manifest.json').read_text())
    for name, expected in manifest['array_sha256'].items():
        if hashlib.sha256((fixtures / 'attention' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'prefill oracle array changed: {name}')
    print("All generated oracles match the frozen anchors.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    prepare()
    if not args.prepare_only:
        run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
        for test in sorted((repository_root() / "tests").glob("test_*.mojo")):
            run(environment_tool("mojo"), "run", "-I", "src", "-I", "build",
                "-I", "tests", str(test.relative_to(repository_root())))

        run(sys.executable, "-m", "llm_mojo.benchmarks.smoke")


if __name__ == "__main__":
    main()
