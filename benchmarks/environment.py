"""Shared curated machine conditions and source identity for measurements."""
import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path
from datetime import UTC, datetime
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]

def command(*args: str) -> str:
    result = subprocess.run(
        args,
        cwd=REPOSITORY,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip()


def optional_command(*args: str) -> str:
    try:
        return command(*args)
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def displays() -> list[dict[str, str]]:
    raw = optional_command("system_profiler", "SPDisplaysDataType", "-json")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return [{"status": raw}]

    result: list[dict[str, str]] = []
    for gpu in payload.get("SPDisplaysDataType", []):
        for display in gpu.get("spdisplays_ndrvs", []):
            result.append(
                {
                    "name": display.get("_name", "unknown"),
                    "connection": display.get(
                        "spdisplays_connection_type", "unknown"
                    ),
                    "resolution": display.get(
                        "_spdisplays_resolution", "unknown"
                    ),
                    "online": display.get("spdisplays_online", "unknown"),
                }
            )
    return result


def battery() -> dict[str, str]:
    raw = optional_command("pmset", "-g", "batt")
    source_match = re.search(r"Now drawing from '([^']+)'", raw)
    charge_match = re.search(r"(\d+)%;\s*([^;]+);\s*([^\n]+)", raw)
    return {
        "power_source": source_match.group(1) if source_match else "unknown",
        "charge": charge_match.group(1) + "%" if charge_match else "unknown",
        "state": charge_match.group(2).strip() if charge_match else "unknown",
        "estimate": charge_match.group(
            3
        ).strip() if charge_match else "unknown",
    }


def power_mode() -> str:
    raw = optional_command("pmset", "-g")
    match = re.search(r"^\s*powermode\s+(\d+)\s*$", raw, re.MULTILINE)
    return match.group(1) if match else "unavailable"


def memory() -> dict[str, Any]:
    raw = optional_command("memory_pressure", "-Q")
    total_match = re.search(r"system has (\d+)", raw)
    free_match = re.search(r"free percentage:\s*(\d+)%", raw)
    return {
        "physical_bytes": int(total_match.group(1)) if total_match else None,
        "free_percent": int(free_match.group(1)) if free_match else None,
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def repository_state() -> dict[str, Any]:
    status = command("git", "status", "--porcelain")
    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def stable_environment() -> dict[str, Any]:
    xcode = optional_command("xcodebuild", "-version").splitlines()
    metal = optional_command("xcrun", "metal", "--version").splitlines()
    return {
        "hardware": {
            "model": optional_command("sysctl", "-n", "hw.model"),
            "chip": optional_command(
                "sysctl", "-n", "machdep.cpu.brand_string"
            ),
            "logical_cpu_count": optional_command("sysctl", "-n", "hw.ncpu"),
            "gpu_api": optional_command(
                "uv", "run", "--locked", "gpu-query", "--api"
            ),
            "gpu_target": optional_command(
                "uv",
                "run",
                "--locked",
                "gpu-query",
                "--target-accelerator",
            ),
            "physical_memory_bytes": memory()["physical_bytes"],
        },
        "software": {
            "macos": optional_command("sw_vers", "-productVersion"),
            "macos_build": optional_command("sw_vers", "-buildVersion"),
            "darwin": optional_command("uname", "-r"),
            "xcode": xcode,
            "metal": metal[:3],
            "uv": optional_command("uv", "--version"),
            "mojo": package_version("mojo"),
            "max": package_version("max"),
        },
    }


def conditions_snapshot() -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "battery": battery(),
        "power_mode_raw": power_mode(),
        "thermal": optional_command("pmset", "-g", "therm").splitlines(),
        "memory": memory(),
        "displays": displays(),
    }


def require_ac(conditions: dict[str, Any]) -> None:
    source = conditions["battery"]["power_source"]
    if source != "AC Power":
        raise RuntimeError(
            f"recorded run requires AC Power; observed {source!r}"
        )


def require_nominal_thermal_state(conditions: dict[str, Any]) -> None:
    thermal = "\n".join(conditions["thermal"])
    required_lines = (
        "No thermal warning level has been recorded",
        "No performance warning level has been recorded",
    )
    if not all(line in thermal for line in required_lines):
        raise RuntimeError(
            "recorded run requires no thermal or performance warning; "
            f"observed {conditions['thermal']!r}"
        )


def ensure_record_location(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == REPOSITORY or resolved.is_relative_to(REPOSITORY):
        raise RuntimeError(
            "recorded run output must be outside the repository; promote only "
            "the compact reviewed evidence afterward"
        )
