"""Retain a compact receipt from the existing host compiler contract tests.

Run with the Python beside an installed, locked Mojo toolchain, for example:
  /absolute/environment/bin/python docs/explorer/verify_contract.py \
      --compiler /absolute/environment/bin/mojo

This runs the unchanged test harness, retaining its child-process logs under
ignored build/explorer/. It does not run GPU work or measure performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
HARNESS = "tests/test_token_selection_compile.py"
SOURCE_PATHS = [
    "uv.lock", "pyproject.toml", "docs/explorer/verify_contract.py",
    "docs/token-selection-contract.md", HARNESS,
    "src/llm_mojo/__init__.py", "src/llm_mojo/_repository.py",
    "src/llm_mojo/__init__.mojo",
    "src/llm_mojo/kernels/__init__.mojo",
    "src/llm_mojo/kernels/token_selection_contract.mojo",
    "src/llm_mojo/kernels/argmax_reduction.mojo",
    "tests/test_token_selection_contract.mojo", "tests/test_reduction_plan.mojo",
    "tests/reduction_plan_cases.mojo", "tests/test_explorer_compiler.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths(root: Path) -> list[str]:
    return SOURCE_PATHS + [str(path.relative_to(root)) for path in sorted(
        (root / "tests/compile_fail").glob("*.mojo")
    ) if path.name.startswith(("reduction_plan_", "token_selection_"))]


def git_output(root: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True,
                          check=True).stdout


def committed_sources(root: Path) -> tuple[str, dict[str, str]]:
    """Bind every tested input to one commit, independently of the Git index."""
    commit = git_output(root, "rev-parse", "HEAD").decode().strip()
    committed_fixtures = [path for path in git_output(
        root, "ls-tree", "-rz", "--name-only", commit, "--", "tests/compile_fail"
    ).decode().split("\0") if Path(path).parent == Path("tests/compile_fail")
        and path.endswith(".mojo")
        and Path(path).name.startswith(("reduction_plan_", "token_selection_"))]
    paths = source_paths(root)
    if set(paths) != set(SOURCE_PATHS + committed_fixtures):
        raise RuntimeError("Contract source set differs from commit; commit added or removed fixtures before verification")
    sources = {}
    for path in paths:
        try:
            committed = git_output(root, "show", f"{commit}:{path}")
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Contract source is absent from commit {commit}: {path}") from error
        try:
            actual = digest(root / path)
        except OSError as error:
            raise RuntimeError(f"Contract source is missing or unreadable: {path}") from error
        if actual != hashlib.sha256(committed).hexdigest():
            raise RuntimeError(f"Contract source differs from commit {commit}: {path}; commit source changes before verification")
        sources[path] = actual
    return commit, sources


def verify_sources_unchanged(root: Path, commit: str, sources: dict[str, str]) -> None:
    """Reject persistent input changes, fixture additions/removals, and moving HEAD."""
    if git_output(root, "rev-parse", "HEAD").decode().strip() != commit:
        raise RuntimeError("HEAD changed during verification; rerun from the committed sources")
    try:
        unchanged = (set(source_paths(root)) == set(sources)
                     and all(digest(root / path) == expected for path, expected in sources.items()))
    except OSError:
        unchanged = False
    if not unchanged:
        raise RuntimeError("Contract source inputs changed during verification; receipt was not replaced")


def checked_compiler_version(output: str, locked_version: str) -> str:
    """Require the selected executable, not just its Python package, to match."""
    match = re.search(
        r"(?m)^Mojo[ \t]+(\d+\.\d+\.\d+)(?:[ \t]+\([^\r\n)]+\))?[ \t]*$",
        output,
    )
    if match is None:
        raise RuntimeError(f"Cannot parse Mojo compiler --version output: {output.strip()!r}")
    actual = match.group(1)
    if actual != locked_version:
        raise RuntimeError(f"Compiler Mojo {actual} differs from uv.lock {locked_version}")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path,
                        default=Path(sys.executable).parent / "mojo")
    args = parser.parse_args()
    compiler = args.compiler.absolute()
    source_commit, sources = committed_sources(ROOT)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    logs = ROOT / "build/explorer/contract-logs" / run_id
    logs.mkdir(parents=True, exist_ok=True)
    paths = list(sources)
    locked = {package["name"]: package["version"]
              for package in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    toolchain = {name: metadata.version(name) for name in ("mojo", "max")}
    for name, version in toolchain.items():
        if version != locked[name]:
            raise RuntimeError(f"Installed {name} {version} differs from uv.lock {locked[name]}")
    version = subprocess.run([str(compiler), "--version"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30, check=True)
    actual_version = checked_compiler_version(version.stdout, locked["mojo"])
    toolchain.update(compiler=str(compiler), compiler_version=version.stdout.strip(),
                     compiler_release=actual_version,
                     python=sys.version.split()[0], python_executable=sys.executable)
    environment = {name: os.environ[name] for name in ("DEVELOPER_DIR",)
                   if name in os.environ}
    spec = util.spec_from_file_location("explorer_contract_tests", ROOT / HARNESS)
    module = util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(module)
    # Preserve every test assertion and command in the existing harness. The
    # explicit compiler selects a preinstalled toolchain without changing it.
    module.environment_tool = lambda name: str(compiler) if name == "mojo" else None
    real_run = subprocess.run
    executions = []
    binaries = {}

    def recorded_run(command, *positional, **kwargs):
        completed = real_run(command, *positional, **kwargs)
        command = list(map(str, command))
        source = next((item for item in command if item.endswith(".mojo")), None)
        if source is not None:
            binaries[command[command.index("-o") + 1]] = source
        else:
            source = binaries.get(command[0])
        content = (completed.stdout or "") + (completed.stderr or "")
        path = logs / f"{len(executions) + 1:02d}-{Path(source or command[0]).stem}.log"
        path.write_text(content)
        diagnostic = re.search(
            r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*(.+?)\s*$", content)
        executions.append(dict(command=command, source=source,
                               stage="compile" if "build" in command else "execute",
                               exit_code=completed.returncode,
                               log=str(path.relative_to(ROOT)), log_sha256=digest(path),
                               diagnostic=diagnostic.group(1) if diagnostic else None))
        return completed

    output = io.StringIO()
    with patch.object(module.subprocess, "run", side_effect=recorded_run):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.TokenSelectionCompileTests)
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    test_log = logs / "unittest.log"
    test_log.write_text(output.getvalue())
    verify_sources_unchanged(ROOT, source_commit, sources)

    def passed(source):
        runs = [entry for entry in executions if entry["source"] == source]
        return (len(runs) == 2 and {entry["stage"] for entry in runs} == {"compile", "execute"}
                and all(entry["exit_code"] == 0 for entry in runs))

    plans = passed("tests/test_reduction_plan.mojo")
    scalars = passed("tests/test_token_selection_contract.mojo")
    results = [dict(id=name, status="passed" if plans else "failed",
                    detail=detail if plans else "The valid plan suite did not compile and execute successfully; see retained logs.")
               for name, detail in (
                   ("sequential", "Compile-time accepted; host evaluation agrees with the independent oracle."),
                   ("balanced", "Compile-time accepted; host evaluation agrees with the independent oracle."),
                   ("reordered-repeated", "Compile-time accepted with all eight inputs covered; host oracle agrees."),
                   ("earlier-root", "Compile-time accepted with an earlier complete root; host oracle agrees."),
               )]
    for source in paths:
        if "compile_fail/" not in source:
            continue
        entry = next((item for item in executions if item["source"] == source), None)
        name = Path(source).stem.replace("reduction_plan_", "").replace("_", "-")
        if entry is None:
            results.append(dict(id=name, status="not-run", source=source, diagnostic=None,
                                detail="Not run because the valid suite did not compile and execute successfully."))
            continue
        expected = ("argmax reduction plan rejected" if "reduction_plan_" in source else
                    "BF16 ordering contract failed" if "signed_zero" in source else
                    "BF16 classification contract failed" if "ignored_nan" in source else
                    "argmax representative laws failed")
        rejected = entry["exit_code"] != 0 and entry["diagnostic"] == expected
        if name == "missing-input":
            detail = "Input 7 replaced by a second input 6; compilation rejected with the intended diagnostic."
        else:
            detail = f"Negative fixture rejected with the intended diagnostic: {expected}."
        results.append(dict(id=name, status="passed" if rejected else "failed", detail=detail,
                            source=source, diagnostic=entry["diagnostic"]))
    results += [dict(id="scalar-contract", status="passed" if scalars else "failed",
                     detail="Finite BF16 ordering/classification and representative merge-law checks compiled and ran on the host."
                     if scalars else "The valid scalar suite did not compile and execute successfully; see retained logs."),
                dict(id="ties-and-nonfinite", status="passed" if plans and scalars else "failed",
                     detail="Host assertions cover lowest-token-ID ties and invalid flags from nonwinning records.")]
    success = result.wasSuccessful()
    command = shlex.join([sys.executable, "docs/explorer/verify_contract.py", "--compiler", str(compiler)])
    command = "".join(f"{name}={shlex.quote(value)} " for name, value in environment.items()) + command
    receipt = dict(
        schema_version=1, status="passed" if success else "failed",
        checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        scope="Host compilation and execution only; no GPU run or performance measurement.",
        command=command, environment=environment,
        harness=HARNESS, toolchain=toolchain,
        source_commit=source_commit,
        sources_unchanged_during_run=True, sources=sources,
        tests_run=result.testsRun, results=results, executions=executions,
        suite_log=str(test_log.relative_to(ROOT)), suite_log_sha256=digest(test_log),
        limits=["Reduction plans describe exactly eight symbolic inputs.",
                "Merge-law generalization uses the documented comparison-only mathematical argument.",
                "Compiler lowering, GPU scheduling, synchronization and full-model determinism are outside this receipt."],
    )
    (ROOT / "docs/explorer/contract-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output.getvalue(), end="")
    print(f"Receipt: docs/explorer/contract-receipt.json ({receipt['status']})")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
