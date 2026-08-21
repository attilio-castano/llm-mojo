"""Summarize exported RMSNorm Metal trace tables without host identifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


Cell = tuple[str, str]
PROFILE_WARMUP_ITERATIONS = 1_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "path_note": "External local artifact; path intentionally omitted.",
    }


def read_table(path: Path) -> list[dict[str, Cell]]:
    root = ET.parse(path).getroot()
    schema = root.find(".//schema")
    if schema is None:
        raise ValueError(f"{path.name} has no table schema")
    columns = [column.findtext("mnemonic") for column in schema.findall("col")]
    if not columns or any(column is None for column in columns):
        raise ValueError(f"{path.name} has an incomplete table schema")

    references = {
        element.attrib["id"]: element
        for element in root.iter()
        if "id" in element.attrib
    }

    def resolve(element: ET.Element) -> Cell:
        seen: set[str] = set()
        while "ref" in element.attrib:
            reference = element.attrib["ref"]
            if reference in seen or reference not in references:
                raise ValueError(
                    f"{path.name} has an invalid XML reference {reference!r}"
                )
            seen.add(reference)
            element = references[reference]
        text = (element.text or "").strip()
        return text, element.attrib.get("fmt", text)

    rows: list[dict[str, Cell]] = []
    for row in root.findall(".//row"):
        cells = list(row)
        if len(cells) != len(columns):
            raise ValueError(
                f"{path.name} row has {len(cells)} cells for "
                f"{len(columns)} columns"
            )
        rows.append(
            {
                str(column): resolve(cell)
                for column, cell in zip(columns, cells, strict=True)
            }
        )
    return rows


def integer(row: dict[str, Cell], column: str) -> int:
    value = row[column][0]
    return int(value) if value else 0


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def duration_summary(rows: list[dict[str, Cell]]) -> dict[str, Any]:
    durations = [integer(row, "duration") for row in rows]
    if not durations:
        raise ValueError("duration summary requires at least one interval")
    median = statistics.median(durations)
    deviations = [abs(value - median) for value in durations]
    return {
        "count": len(durations),
        "unit": "nanoseconds",
        "minimum": min(durations),
        "median": median,
        "median_absolute_deviation": statistics.median(deviations),
        "p95": percentile(durations, 0.95),
        "maximum": max(durations),
    }


def trace_metadata(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    summary = root.find(".//summary")
    if summary is None:
        raise ValueError(f"{path.name} has no trace summary")
    gpu_settings: list[str] = []
    for instrument in root.findall(".//instrument"):
        if instrument.attrib.get("name") != "Metal Application":
            continue
        for key in instrument.findall(".//key"):
            if key.attrib.get("name") == "GPU":
                gpu_settings.extend(
                    value.text or "" for value in key.findall("value")
                )
    counter_table = root.find('.//table[@schema="gpu-counter-info"]')
    counter_configuration = None
    if counter_table is not None:
        counter_configuration = {
            key: counter_table.attrib[key]
            for key in ("counter-profile", "counter-device", "shader-profiler")
            if key in counter_table.attrib
        }
    return {
        "template": summary.findtext("template-name"),
        "recording_duration_seconds": float(summary.findtext("duration", "0")),
        "end_reason": summary.findtext("end-reason"),
        "instruments_version": summary.findtext("instruments-version"),
        "metal_application_gpu_settings": gpu_settings,
        "counter_configuration": counter_configuration,
    }


def performance_states(path: Path) -> dict[str, Any]:
    rows = read_table(path)
    states: Counter[str] = Counter()
    induced: Counter[str] = Counter()
    duration_by_state: Counter[str] = Counter()
    for row in rows:
        state = row["gpu-performance-state"][1]
        states[state] += 1
        duration_by_state[state] += integer(row, "duration")
        induced["yes" if integer(row, "is-induced") else "no"] += 1
    return {
        "interval_count": len(rows),
        "states": dict(sorted(states.items())),
        "duration_nanoseconds_by_state": dict(
            sorted(duration_by_state.items())
        ),
        "induced_intervals": dict(sorted(induced.items())),
        "scope": "Trace-level device condition; not target-process timing.",
    }


def spill_events(
    path: Path, command_buffer_ids: set[int], target_process: str
) -> dict[str, Any]:
    rows = read_table(path)
    matched = [
        row
        for row in rows
        if integer(row, "cmdbuffer-id") in command_buffer_ids
        and row["process"][1] == target_process
    ]
    spilled_bytes = [integer(row, "spilled-bytes") for row in matched]
    return {
        "target_event_count": len(matched),
        "target_spilled_bytes_total": sum(spilled_bytes),
        "target_spilled_bytes_maximum_event": max(spilled_bytes, default=0),
        "interpretation": (
            "No target spill event was reported in this capture. This does "
            "not prove that all executions are spill-free." if not matched else "The trace reported target spill events; inspect before use."
        ),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile_iterations <= 0:
        raise ValueError("--profile-iterations must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be non-negative")

    submissions = read_table(args.submissions_xml)
    encoded_submissions = [
        row for row in submissions if integer(row, "num-encoders") > 0
    ]
    command_buffer_ids = {
        integer(row, "cmdbuffer-id") for row in encoded_submissions
    }
    if len(command_buffer_ids) != len(encoded_submissions):
        raise ValueError("target command-buffer identifiers are not unique")
    target_processes = {
        row["process"][1] for row in encoded_submissions if row["process"][1]
    }
    if len(target_processes) != 1:
        raise ValueError(
            "command-buffer submissions do not identify exactly one target "
            "process"
        )
    target_process = next(iter(target_processes))
    gpu_intervals = read_table(args.gpu_intervals_xml)
    id_matched_intervals = [
        row
        for row in gpu_intervals
        if integer(row, "cmdbuffer-id") in command_buffer_ids
    ]
    target_intervals = [
        row
        for row in id_matched_intervals
        if target_process in row["event-label"][1]
    ]
    target_intervals.sort(key=lambda row: integer(row, "start"))
    interval_count_by_command_buffer = Counter(
        integer(row, "cmdbuffer-id") for row in target_intervals
    )
    duplicate_interval_groups = sum(
        count > 1 for count in interval_count_by_command_buffer.values()
    )
    target_channels = Counter(
        row["channel-name"][0] for row in target_intervals
    )
    compute_channel = [
        row for row in target_intervals if row["channel-name"][0] == "Compute"
    ]
    compute_commands = [
        row
        for row in compute_channel
        if ":Compute Command" in row["event-label"][1]
    ]
    expected_compute_commands = (
        1 + args.warmup_iterations + args.profile_iterations
    )
    if len(compute_commands) != expected_compute_commands:
        raise ValueError(
            "cannot segment the fixed RMSNorm profile sequence: expected "
            f"{expected_compute_commands} compute commands, observed "
            f"{len(compute_commands)}"
        )

    correctness = compute_commands[:1]
    warmup_start = 1
    profile_start = warmup_start + args.warmup_iterations
    warmup = compute_commands[warmup_start:profile_start]
    profile = compute_commands[profile_start:]
    command_kinds: Counter[str] = Counter()
    for row in compute_channel:
        label = row["event-label"][1]
        if ":Compute Command" in label:
            command_kinds["compute"] += 1
        elif ":Blit Command" in label:
            command_kinds["blit"] += 1
        else:
            command_kinds["other"] += 1

    inputs = [
        artifact(args.submissions_xml, "command_buffer_submissions_xml"),
        artifact(args.gpu_intervals_xml, "gpu_intervals_xml"),
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "rmsnorm_metal_trace",
        "inputs": inputs,
        "validated_sequence": {
            "target_submissions_with_encoders": len(encoded_submissions),
            "matched_target_gpu_intervals": len(target_intervals),
            "duplicate_command_buffer_interval_groups": duplicate_interval_groups,
            "excluded_command_buffer_id_collisions": (
                len(id_matched_intervals) - len(target_intervals)
            ),
            "join_rule": (
                "Command-buffer identifier plus the target-process label; "
                "the label itself is not retained."
            ),
            "target_gpu_interval_channels": dict(
                sorted(target_channels.items())
            ),
            "compute_channel_command_kinds": dict(
                sorted(command_kinds.items())
            ),
            "correctness_dispatches": len(correctness),
            "warmup_dispatches": len(warmup),
            "profile_dispatches": len(profile),
            "segmentation_rule": (
                "Fixed program order: one correctness dispatch, declared "
                "warmup dispatches, then declared profile dispatches."
            ),
        },
        "instrumented_gpu_interval_duration": {
            "correctness": duration_summary(correctness),
            "warmup": duration_summary(warmup) if warmup else None,
            "profile": duration_summary(profile),
            "evidence_boundary": (
                "Diagnostic Instruments intervals, not headline benchmark "
                "latency."
            ),
        },
        "caveats": [
            "Raw trace exports remain external because they may contain host identifiers and unrelated process activity.",
            "No logical or source-requested byte count is presented as observed hardware traffic.",
            "A missing profiler event is bounded to this capture and is not a universal absence claim.",
        ],
    }
    if args.toc_xml is not None:
        inputs.append(artifact(args.toc_xml, "trace_toc_xml"))
        result["trace"] = trace_metadata(args.toc_xml)
    if args.performance_state_xml is not None:
        inputs.append(
            artifact(args.performance_state_xml, "gpu_performance_state_xml")
        )
        result["gpu_performance_state"] = performance_states(
            args.performance_state_xml
        )
    if args.spill_xml is not None:
        inputs.append(artifact(args.spill_xml, "compiler_spill_events_xml"))
        result["compiler_spills"] = spill_events(
            args.spill_xml, command_buffer_ids, target_process
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions-xml", type=Path, required=True)
    parser.add_argument("--gpu-intervals-xml", type=Path, required=True)
    parser.add_argument("--toc-xml", type=Path)
    parser.add_argument("--performance-state-xml", type=Path)
    parser.add_argument("--spill-xml", type=Path)
    parser.add_argument("--profile-iterations", type=int, required=True)
    parser.add_argument(
        "--warmup-iterations", type=int, default=PROFILE_WARMUP_ITERATIONS
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"trace summary: {args.output.resolve()}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    try:
        main()
    except (ET.ParseError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
