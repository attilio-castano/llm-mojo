"""Capture a standalone RMSNorm profile through a verified temporary launch."""

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


REPOSITORY = Path(__file__).resolve().parents[1]
STAGING_ROOT = Path("/private/tmp")
DEFAULT_TEMPLATE = "Metal System Trace"
DEFAULT_TIME_LIMIT = "1s"
PROFILE_REGION_BEGIN = "PROFILE_REGION_BEGIN"
PROFILE_REGION_END = "PROFILE_REGION_END"
TIME_LIMIT = re.compile(r"^[1-9][0-9]*(?:ms|s|m|h)$")
Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


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
    source: Path, directory: Path, *, expected_hash: str
) -> tuple[Path, str]:
    staged = directory / "rmsnorm-profile"
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


def scrubbed_command(time_limit: str) -> str:
    return shlex.join(
        [
            "xcrun",
            "xctrace",
            "record",
            "--no-prompt",
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

    with tempfile.TemporaryDirectory(
        prefix="llm-mojo-rmsnorm-", dir=staging_root
    ) as temporary:
        temporary_path = Path(temporary)
        temporary_path.chmod(0o700)
        staged_binary, staged_hash = stage_profile_binary(
            profile_binary,
            temporary_path,
            expected_hash=canonical_hash,
        )
        staged_bytes = staged_binary.stat().st_size
        command_args = [
            "xcrun",
            "xctrace",
            "record",
            "--no-prompt",
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
        command = scrubbed_command(time_limit)
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
            if begin < 0 or end <= begin:
                failures.append(
                    "target output does not contain the complete profile region"
                )

    if not output_trace.exists():
        failures.append("xctrace did not create the requested trace")

    output_bytes = capture_output.encode()
    receipt = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "capture": {
            "status": "complete" if not failures else "invalid",
            "command": command,
            "template": recorded_template,
            "time_limit": time_limit,
            "xctrace_version": xctrace_version(runner),
            "xctrace_returncode": capture_returncode,
            "profile_region_markers_complete": (
                PROFILE_REGION_BEGIN in capture_output
                and PROFILE_REGION_END in capture_output
                and capture_output.find(PROFILE_REGION_BEGIN)
                < capture_output.find(PROFILE_REGION_END)
            ),
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
            "configuration": {
                key: provenance.get(key)
                for key in (
                    "profile_rows",
                    "hidden_size",
                    "profile_warmup_iterations",
                    "profile_iterations",
                    "profile_post_idle_milliseconds",
                    "implementation",
                    "entrypoint",
                )
            },
            "repository": provenance.get("repository"),
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
