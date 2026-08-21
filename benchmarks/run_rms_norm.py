"""Record a curated Apple GPU environment and run the RMSNorm benchmark."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
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
        "estimate": charge_match.group(3).strip() if charge_match else "unknown",
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


def main() -> None:
    status = command("git", "status", "--porcelain")
    xcode = optional_command("xcodebuild", "-version").splitlines()
    metal = optional_command("xcrun", "metal", "--version").splitlines()
    metadata = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "repository": {
            "commit": command("git", "rev-parse", "HEAD"),
            "branch": command("git", "branch", "--show-current"),
            "dirty": bool(status),
        },
        "hardware": {
            "model": optional_command("sysctl", "-n", "hw.model"),
            "chip": optional_command("sysctl", "-n", "machdep.cpu.brand_string"),
            "logical_cpu_count": optional_command("sysctl", "-n", "hw.ncpu"),
            "gpu_api": optional_command(
                "uv", "run", "--locked", "gpu-query", "--api"
            ),
            "gpu_target": optional_command(
                "uv", "run", "--locked", "gpu-query", "--target-accelerator"
            ),
            "memory": memory(),
            "displays": displays(),
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
        "conditions": {
            "battery": battery(),
            "power_mode_raw": power_mode(),
            "thermal": optional_command("pmset", "-g", "therm").splitlines(),
        },
        "benchmark_environment": {
            "MODULAR_DEBUG": "device-sync-mode",
        },
    }
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)

    environment = os.environ.copy()
    environment["MODULAR_DEBUG"] = "device-sync-mode"
    subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "mojo",
            "run",
            "-I",
            "src",
            "benchmarks/rms_norm.mojo",
        ],
        cwd=REPOSITORY,
        check=True,
        env=environment,
    )


if __name__ == "__main__":
    main()
