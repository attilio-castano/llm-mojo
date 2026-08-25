"""Capture a standalone GPU profile through a verified temporary launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


REPOSITORY = Path(__file__).resolve().parents[1]
STAGING_ROOT = Path("/private/tmp")
DEFAULT_TEMPLATE = "Metal System Trace"
DEFAULT_TIME_LIMIT = "1s"
PROFILE_REGION_BEGIN = "PROFILE_REGION_BEGIN"
PROFILE_REGION_END = "PROFILE_REGION_END"
TIME_LIMIT = re.compile(r"^[1-9][0-9]*(?:ms|s|m|h)$")
CAPTURE_ID = re.compile(r"^(?:rmsnorm|linear)-[0-9a-f]{32}$")
IMPLEMENTATION_ENTRYPOINTS = {
    "apple_gpu_shared_tree_v0": "enqueue_rms_norm_apple_gpu_shared_tree",
    "apple_gpu_simdgroup_v1": "enqueue_rms_norm_apple_gpu",
    "apple_gpu_one_output_simdgroup_v0": "enqueue_linear_apple_gpu",
    "apple_gpu_two_output_simdgroup_v1": (
        "enqueue_linear_apple_gpu_two_output"
    ),
}
Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_capture_id(operation: str = "rms_norm") -> str:
    prefix = "linear" if operation == "linear_projection" else "rmsnorm"
    return f"{prefix}-{uuid4().hex}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_external_path(path: Path, *, label: str) -> None:
    resolved = path.resolve()
    if resolved == REPOSITORY or resolved.is_relative_to(REPOSITORY):
        raise RuntimeError(f"{label} must be outside the repository")


def profile_provenance(
    profile_binary: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    if not profile_binary.is_file():
        raise RuntimeError(f"profile binary does not exist: {profile_binary}")
    if profile_binary.stat().st_mode & 0o111 == 0:
        raise RuntimeError(f"profile binary is not executable: {profile_binary}")

    provenance_path = profile_binary.with_name(
        profile_binary.name + ".provenance.json"
    )
    if not provenance_path.is_file():
        raise RuntimeError(
            f"profile provenance does not exist: {provenance_path}"
        )
    try:
        provenance_bytes = provenance_path.read_bytes()
        provenance = json.loads(provenance_bytes)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"could not read profile provenance: {provenance_path}"
        ) from error

    binary = provenance.get("binary")
    if not isinstance(binary, dict):
        raise RuntimeError("profile provenance has no binary identity")
    expected_hash = binary.get("sha256")
    expected_bytes = binary.get("bytes")
    if not isinstance(expected_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_hash
    ):
        raise RuntimeError("profile provenance has an invalid binary SHA-256")
    actual_hash = sha256_file(profile_binary)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "profile binary SHA-256 does not match its provenance"
        )
    if profile_binary.stat().st_size != expected_bytes:
        raise RuntimeError("profile binary size does not match its provenance")
    binary_identity = {
        "bytes": profile_binary.stat().st_size,
        "sha256": actual_hash,
    }
    provenance_identity = {
        "bytes": len(provenance_bytes),
        "sha256": hashlib.sha256(provenance_bytes).hexdigest(),
    }
    return provenance, provenance_path, binary_identity, provenance_identity


def profile_contract(
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if provenance.get("schema_version") != 1:
        raise RuntimeError("profile provenance has an unsupported schema")

    operation = provenance.get("operation", "rms_norm")
    if operation not in ("rms_norm", "linear_projection"):
        raise RuntimeError("profile provenance has an unsupported operation")
    configuration: dict[str, Any] = {"operation": operation}
    for key in (
        "profile_warmup_iterations",
        "profile_iterations",
        "profile_post_idle_milliseconds",
    ):
        value = provenance.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(
                f"profile provenance has an invalid {key.replace('_', ' ')}"
            )
        configuration[key] = value
    if configuration["profile_iterations"] <= 0:
        raise RuntimeError("profile provenance iterations must be positive")

    if operation == "rms_norm":
        shape_keys = ("profile_rows", "hidden_size")
        for key in shape_keys:
            value = provenance.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RuntimeError(
                    f"profile provenance has an invalid {key.replace('_', ' ')}"
                )
            configuration[key] = value
        configuration["dispatches_per_iteration"] = 1
    else:
        workload = provenance.get("profile_workload")
        if not isinstance(workload, str) or not workload:
            raise RuntimeError("profile provenance has an invalid workload")
        configuration["profile_workload"] = workload
        linear_keys = (
            "rows",
            "input_features",
            "output_features",
            "layers",
            "dispatches_per_iteration",
        )
        for key in linear_keys:
            value = provenance.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise RuntimeError(
                    f"profile provenance has an invalid {key.replace('_', ' ')}"
                )
            configuration[key] = value
        configuration["profile_rows"] = configuration["rows"]
        configuration["hidden_size"] = configuration["input_features"]

    implementation = provenance.get("implementation")
    entrypoint = provenance.get("entrypoint")
    if (
        not isinstance(implementation, str)
        or implementation not in IMPLEMENTATION_ENTRYPOINTS
        or entrypoint != IMPLEMENTATION_ENTRYPOINTS[implementation]
    ):
        raise RuntimeError(
            "profile provenance has an inconsistent implementation"
        )
    configuration["implementation"] = implementation
    configuration["entrypoint"] = entrypoint

    repository = provenance.get("repository")
    if not isinstance(repository, dict):
        raise RuntimeError("profile provenance has no repository identity")
    commit = repository.get("commit")
    dirty = repository.get("dirty")
    branch = repository.get("branch")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise RuntimeError("profile provenance has an invalid repository commit")
    if not isinstance(dirty, bool):
        raise RuntimeError("profile provenance has an invalid dirty state")
    if not isinstance(branch, str) or not branch:
        raise RuntimeError("profile provenance has an invalid repository branch")
    repository_identity = {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }

    hardware = provenance.get("hardware")
    if not isinstance(hardware, dict):
        raise RuntimeError("profile provenance has no hardware identity")
    chip = hardware.get("chip")
    gpu_api = hardware.get("gpu_api")
    if not isinstance(chip, str) or not chip.startswith("Apple "):
        raise RuntimeError("profile provenance has an invalid Apple GPU identity")
    if gpu_api != "metal":
        raise RuntimeError("profile provenance does not identify the Metal API")
    hardware_identity = {"chip": chip, "gpu_api": gpu_api}
    return configuration, repository_identity, hardware_identity


def output_field(output: str, label: str) -> str:
    matches = re.findall(
        rf"^{re.escape(label)}:\s*(.*?)\s*$", output, flags=re.MULTILINE
    )
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"expected exactly one {label!r} line")
    return matches[0]


def parse_target_identity(output: str) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "entrypoint": output_field(output, "profile implementation"),
        "device": output_field(output, "device"),
        "backend": output_field(output, "api"),
    }
    for label, key in (
        ("rows", "rows"),
        ("hidden", "hidden_size"),
        ("warmup iterations", "warmup_iterations"),
        ("profile iterations", "profile_iterations"),
        (
            "post-profile idle milliseconds",
            "post_profile_idle_milliseconds",
        ),
    ):
        value = output_field(output, label)
        if re.fullmatch(r"[0-9]+", value) is None:
            raise ValueError(f"target {label} is not a non-negative integer")
        identity[key] = int(value)
    workload_matches = re.findall(
        r"^profile workload:\s*(.*?)\s*$", output, flags=re.MULTILINE
    )
    if workload_matches:
        if len(workload_matches) != 1 or not workload_matches[0]:
            raise ValueError("expected exactly one 'profile workload' line")
        identity["profile_workload"] = workload_matches[0]
        for label, key in (
            ("output features", "output_features"),
            ("profile dispatches per iteration", "dispatches_per_iteration"),
        ):
            value = output_field(output, label)
            if re.fullmatch(r"[0-9]+", value) is None or int(value) <= 0:
                raise ValueError(f"target {label} is not a positive integer")
            identity[key] = int(value)
    return identity


def validate_target_identity(
    identity: dict[str, Any],
    configuration: dict[str, Any],
    hardware: dict[str, str],
) -> dict[str, Any]:
    expected = {
        "entrypoint": configuration["entrypoint"],
        "device": hardware["chip"],
        "backend": hardware["gpu_api"],
        "rows": configuration["profile_rows"],
        "hidden_size": configuration["hidden_size"],
        "warmup_iterations": configuration["profile_warmup_iterations"],
        "profile_iterations": configuration["profile_iterations"],
        "post_profile_idle_milliseconds": configuration[
            "profile_post_idle_milliseconds"
        ],
    }
    if configuration["operation"] == "linear_projection":
        expected.update(
            {
                "profile_workload": configuration["profile_workload"],
                "output_features": configuration["output_features"],
                "dispatches_per_iteration": configuration[
                    "dispatches_per_iteration"
                ],
            }
        )
    for key, expected_value in expected.items():
        if identity.get(key) != expected_value:
            raise ValueError(
                f"target {key.replace('_', ' ')} does not match provenance"
            )
    return {"implementation": configuration["implementation"], **identity}


def template_identity(template: str) -> dict[str, Any]:
    candidate = Path(template).expanduser()
    if candidate.is_file():
        return {
            "kind": "file",
            "name": candidate.name,
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate),
        }
    return {"kind": "installed_name", "name": template}


def stage_profile_binary(
    source: Path,
    directory: Path,
    *,
    expected_hash: str,
    capture_id: str,
) -> tuple[Path, str]:
    if CAPTURE_ID.fullmatch(capture_id) is None:
        raise RuntimeError("capture ID is invalid")
    staged = directory / capture_id
    with source.open("rb") as source_handle, staged.open("xb") as staged_handle:
        for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
            staged_handle.write(chunk)
    staged.chmod(0o700)
    staged_hash = sha256_file(staged)
    if staged_hash != expected_hash:
        raise RuntimeError(
            "staged profile binary does not match the provenance SHA-256"
        )
    return staged, staged_hash


def xctrace_version(runner: Runner) -> str:
    try:
        result = runner(
            ["xcrun", "xctrace", "version"],
            cwd=REPOSITORY,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as error:
        return f"unavailable: {error}"
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        return f"unavailable: exit {result.returncode}: {output}"
    return output


def scrubbed_command(time_limit: str, capture_id: str) -> str:
    return shlex.join(
        [
            "xcrun",
            "xctrace",
            "record",
            "--no-prompt",
            "--run-name",
            capture_id,
            "--template",
            "<template>",
            "--time-limit",
            time_limit,
            "--output",
            "<external-trace>",
            "--target-stdout",
            "-",
            "--launch",
            "--",
            "<ephemeral-profile-binary>",
        ]
    )


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    with path.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


def capture_trace(
    *,
    profile_binary: Path,
    output_trace: Path,
    template: str = DEFAULT_TEMPLATE,
    time_limit: str = DEFAULT_TIME_LIMIT,
    receipt_path: Path | None = None,
    staging_root: Path = STAGING_ROOT,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    if not template.strip():
        raise RuntimeError("trace template must not be empty")
    if TIME_LIMIT.fullmatch(time_limit) is None:
        raise RuntimeError(
            "time limit must be a positive integer followed by ms, s, m, or h"
        )

    profile_binary = profile_binary.expanduser().resolve()
    output_trace = output_trace.expanduser().resolve()
    receipt_path = (
        receipt_path.expanduser().resolve()
        if receipt_path is not None
        else output_trace.with_name(output_trace.name + ".capture.json")
    )
    staging_root = staging_root.expanduser().resolve()

    ensure_external_path(profile_binary, label="profile binary")
    ensure_external_path(output_trace, label="raw trace")
    ensure_external_path(receipt_path, label="capture receipt")
    if output_trace.exists():
        raise RuntimeError(f"refusing to overwrite existing trace: {output_trace}")
    if receipt_path.exists():
        raise RuntimeError(
            f"refusing to overwrite existing capture receipt: {receipt_path}"
        )
    if not staging_root.is_dir():
        raise RuntimeError(f"staging root does not exist: {staging_root}")

    provenance, _, binary_identity, provenance_identity = profile_provenance(
        profile_binary
    )
    configuration, repository, hardware = profile_contract(provenance)
    capture_id = new_capture_id(configuration["operation"])
    if CAPTURE_ID.fullmatch(capture_id) is None:
        raise RuntimeError("generated capture ID is invalid")
    canonical_hash = binary_identity["sha256"]
    canonical_bytes = binary_identity["bytes"]
    recorded_template = template_identity(template)
    output_trace.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    capture_output = ""
    capture_returncode: int | None = None
    failures: list[str] = []
    staged_hash = ""
    staged_bytes = 0
    command = ""
    target_identity: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(
        prefix=(
            "llm-mojo-linear-"
            if configuration["operation"] == "linear_projection"
            else "llm-mojo-rmsnorm-"
        ),
        dir=staging_root,
    ) as temporary:
        temporary_path = Path(temporary)
        temporary_path.chmod(0o700)
        staged_binary, staged_hash = stage_profile_binary(
            profile_binary,
            temporary_path,
            expected_hash=canonical_hash,
            capture_id=capture_id,
        )
        staged_bytes = staged_binary.stat().st_size
        command_args = [
            "xcrun",
            "xctrace",
            "record",
            "--no-prompt",
            "--run-name",
            capture_id,
            "--template",
            template,
            "--time-limit",
            time_limit,
            "--output",
            str(output_trace),
            "--target-stdout",
            "-",
            "--launch",
            "--",
            str(staged_binary),
        ]
        command = scrubbed_command(time_limit, capture_id)
        print(
            "Launching a verified ephemeral profile binary "
            f"(sha256 {staged_hash})...",
            flush=True,
        )
        try:
            result = runner(
                command_args,
                cwd=REPOSITORY,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as error:
            failures.append(f"xctrace could not start: {error}")
        else:
            capture_returncode = result.returncode
            capture_output = result.stdout or ""
            print(capture_output, end="", flush=True)
            if capture_returncode != 0:
                failures.append(f"xctrace exited with {capture_returncode}")
            begin = capture_output.find(PROFILE_REGION_BEGIN)
            end = capture_output.find(PROFILE_REGION_END)
            markers_complete = (
                capture_output.count(PROFILE_REGION_BEGIN) == 1
                and capture_output.count(PROFILE_REGION_END) == 1
                and begin >= 0
                and end > begin
            )
            if not markers_complete:
                failures.append(
                    "target output does not contain the complete profile region"
                )
            try:
                parsed_identity = parse_target_identity(capture_output)
                target_identity = validate_target_identity(
                    parsed_identity, configuration, hardware
                )
            except ValueError as error:
                failures.append(f"target output identity invalid: {error}")

    if not output_trace.exists():
        failures.append("xctrace did not create the requested trace")

    output_bytes = capture_output.encode()
    receipt = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "capture": {
            "capture_id": capture_id,
            "run_name": capture_id,
            "status": "complete" if not failures else "invalid",
            "command": command,
            "template": recorded_template,
            "time_limit": time_limit,
            "xctrace_version": xctrace_version(runner),
            "xctrace_returncode": capture_returncode,
            "profile_region_markers_complete": (
                capture_output.count(PROFILE_REGION_BEGIN) == 1
                and capture_output.count(PROFILE_REGION_END) == 1
                and capture_output.find(PROFILE_REGION_BEGIN)
                < capture_output.find(PROFILE_REGION_END)
            ),
            "target_identity": target_identity,
            "target_output": {
                "bytes": len(output_bytes),
                "sha256": hashlib.sha256(output_bytes).hexdigest(),
                "content_note": "Output omitted from the receipt.",
            },
            "failures": failures,
        },
        "profile": {
            "binary": {
                "bytes": canonical_bytes,
                "sha256": canonical_hash,
                "path_note": "Canonical external artifact; path omitted.",
            },
            "staged_binary": {
                "bytes": staged_bytes,
                "sha256": staged_hash,
                "path_note": (
                    "Byte-identical ephemeral launch copy under /private/tmp; "
                    "removed after capture."
                ),
            },
            "provenance": {
                **provenance_identity,
                "schema_version": provenance.get("schema_version"),
                "path_note": "Canonical external artifact; path omitted.",
            },
            "configuration": configuration,
            "repository": repository,
            "hardware": hardware,
        },
        "trace": {
            "created": output_trace.exists(),
            "path_note": (
                "External raw Instruments trace; path intentionally omitted."
            ),
        },
    }
    write_receipt(receipt_path, receipt)
    if failures:
        raise RuntimeError(
            "; ".join(failures) + f"; capture receipt: {receipt_path}"
        )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-binary", required=True, type=Path)
    parser.add_argument("--output-trace", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--time-limit", default=DEFAULT_TIME_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = capture_trace(
        profile_binary=args.profile_binary,
        output_trace=args.output_trace,
        receipt_path=args.receipt,
        template=args.template,
        time_limit=args.time_limit,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    receipt_path = args.receipt or args.output_trace.with_name(
        args.output_trace.name + ".capture.json"
    )
    print(f"trace: {args.output_trace.resolve()}")
    print(f"capture receipt: {receipt_path.resolve()}")


if __name__ == "__main__":
    main()
