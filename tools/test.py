"""Prepare independent oracles, check frozen anchors, then run all tests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("+", *args, flush=True)
    subprocess.run(args, cwd=ROOT, check=True,
                   env={**os.environ, "MODULAR_DEBUG": "device-sync-mode"})


def prepare():
    fixtures = ROOT / "build/oracle_data"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "__init__.mojo").touch()
    for name in ("rms_norm", "linear", "rope", "attention"):
        run("uv", "run", "--script", f"tests/fixtures/{name}/generate.py")
        (fixtures / name / "__init__.mojo").touch()
    run("uv", "run", "--locked", "python", "tests/fixtures/attention/generate_decode.py")
    anchors = json.loads((ROOT / "tests/fixtures/checksums.json").read_text())
    for name, expected in anchors["sha256"].items():
        actual = hashlib.sha256((fixtures / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"oracle changed: {name}; review the numerical contract before updating anchors")
    print("All generated oracles match the landed fixtures.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    prepare()
    if not args.prepare_only:
        run(sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
        for test in sorted((ROOT / "tests").glob("test_*.mojo")):
            run("uv", "run", "--locked", "mojo", "run", "-I", "src", "-I", "build",
                "-I", "tests", "-I", "benchmarks", str(test.relative_to(ROOT)))

        run(sys.executable, "tools/smoke.py")


if __name__ == "__main__":
    main()
