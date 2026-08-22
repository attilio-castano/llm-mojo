"""Compare four scrubbed RMSNorm counter summaries in ABBA order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


EXPECTED_TEMPLATE = "LLM_Mojo_Metal_Limiters"
EXPECTED_GPU_SETTINGS = {
    "Counter Set: Performance Limiters",
    "Shader Timeline: Disabled",
    "Induced GPU Performance State: Default",
}
DEFAULT_WARMUP_ITERATIONS = 100
DEFAULT_PROFILE_ITERATIONS = 500
DEFAULT_MINIMUM_TIMESTAMPS = 10
DEFAULT_MINIMUM_SAMPLE_SPAN = 0.80
DEFAULT_MATERIAL_CHANGE = 0.05
CAPTURE_SPECS = (
    ("capture-01-baseline", "baseline"),
    ("capture-02-variant", "variant"),
    ("capture-03-variant", "variant"),
    ("capture-04-baseline", "baseline"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, capture_id: str) -> dict[str, Any]:
    return {
        "capture_id": capture_id,
        "kind": "scrubbed_trace_summary_json",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "path_note": "External local artifact; path intentionally omitted.",
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_capture(
    summary: dict[str, Any],
    *,
    capture_id: str,
    role: str,
    warmup_iterations: int,
    profile_iterations: int,
    minimum_timestamps: int,
    minimum_sample_span: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(
        summary.get("analysis") == "rmsnorm_metal_trace",
        f"{capture_id} is not an RMSNorm Metal trace summary",
    )
    trace = summary.get("trace", {})
    require(
        trace.get("end_reason") == "Target app exited",
        f"{capture_id} target did not exit normally",
    )
    require(
        trace.get("template") == EXPECTED_TEMPLATE,
        f"{capture_id} used an unexpected Instruments template",
    )
    settings = set(trace.get("metal_application_gpu_settings", []))
    require(
        EXPECTED_GPU_SETTINGS <= settings,
        f"{capture_id} does not have the frozen Metal settings",
    )

    sequence = summary.get("validated_sequence", {})
    expected_compute = 1 + warmup_iterations + profile_iterations
    require(
        sequence.get("setup_compute_commands") == 0,
        f"{capture_id} contains unexpected setup compute commands",
    )
    require(
        sequence.get("correctness_dispatches") == 1
        and sequence.get("warmup_dispatches") == warmup_iterations
        and sequence.get("profile_dispatches") == profile_iterations
        and sequence.get("compute_channel_command_kinds", {}).get("compute")
        == expected_compute,
        f"{capture_id} does not contain the exact declared compute sequence",
    )

    profile_counters = summary.get("profile_gpu_counters", {})
    samples = profile_counters.get("samples", {})
    timestamp_count = samples.get("timestamp_count", 0)
    sample_span = samples.get("sample_span_fraction", 0.0)
    require(
        timestamp_count >= minimum_timestamps,
        f"{capture_id} has too few profile-window counter timestamps",
    )
    require(
        math.isfinite(sample_span) and sample_span >= minimum_sample_span,
        f"{capture_id} counter samples do not span enough of the profile",
    )
    defined_count = profile_counters.get("defined_counter_count", 0)
    sampled_count = profile_counters.get("sampled_counter_count", 0)
    require(
        defined_count > 0 and sampled_count == defined_count,
        f"{capture_id} does not sample every named counter",
    )

    counters_by_name: dict[str, dict[str, Any]] = {}
    for counter in profile_counters.get("counters", []):
        name = counter.get("name")
        median = counter.get("median")
        require(
            isinstance(name, str) and name,
            f"{capture_id} contains an unnamed counter",
        )
        require(
            name not in counters_by_name,
            f"{capture_id} contains duplicate counter {name!r}",
        )
        require(
            isinstance(median, (int, float)) and math.isfinite(median),
            f"{capture_id} counter {name!r} has a non-finite median",
        )
        counters_by_name[name] = counter
    require(
        len(counters_by_name) == sampled_count,
        f"{capture_id} counter metadata is incomplete",
    )

    profile_window = profile_counters.get("profile_window", {})
    return (
        {
            "capture_id": capture_id,
            "role": role,
            "trace_duration_seconds": trace.get(
                "recording_duration_seconds"
            ),
            "profile_window_duration_nanoseconds": profile_window.get(
                "duration_nanoseconds"
            ),
            "target_gpu_busy_fraction": profile_window.get(
                "target_gpu_busy_fraction"
            ),
            "counter_timestamp_count": timestamp_count,
            "counter_sample_span_fraction": sample_span,
            "gpu_performance_state": summary.get("gpu_performance_state"),
            "target_spill_event_count": summary.get(
                "compiler_spills", {}
            ).get("target_event_count"),
            "instrumented_profile_duration": summary.get(
                "instrumented_gpu_interval_duration", {}
            ).get("profile"),
        },
        counters_by_name,
    )


def compare(
    summaries: list[dict[str, Any]],
    *,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    profile_iterations: int = DEFAULT_PROFILE_ITERATIONS,
    minimum_timestamps: int = DEFAULT_MINIMUM_TIMESTAMPS,
    minimum_sample_span: float = DEFAULT_MINIMUM_SAMPLE_SPAN,
    material_change: float = DEFAULT_MATERIAL_CHANGE,
) -> dict[str, Any]:
    require(len(summaries) == 4, "comparison requires exactly four captures")
    require(warmup_iterations >= 0, "warmup iterations must be non-negative")
    require(profile_iterations > 0, "profile iterations must be positive")
    require(minimum_timestamps > 0, "minimum timestamps must be positive")
    require(
        0.0 <= minimum_sample_span <= 1.0,
        "minimum sample span must be between zero and one",
    )
    require(material_change >= 0.0, "material change must be non-negative")

    captures = []
    counter_maps = []
    for summary, (capture_id, role) in zip(
        summaries, CAPTURE_SPECS, strict=True
    ):
        capture, counters = validate_capture(
            summary,
            capture_id=capture_id,
            role=role,
            warmup_iterations=warmup_iterations,
            profile_iterations=profile_iterations,
            minimum_timestamps=minimum_timestamps,
            minimum_sample_span=minimum_sample_span,
        )
        captures.append(capture)
        counter_maps.append(counters)

    reference_names = set(counter_maps[0])
    for capture, counters in zip(captures[1:], counter_maps[1:], strict=True):
        require(
            set(counters) == reference_names,
            f"{capture['capture_id']} has a different named-counter set",
        )

    comparisons = []
    for name in sorted(reference_names):
        observed = [counter_map[name] for counter_map in counter_maps]
        counter_ids = {counter.get("counter_id") for counter in observed}
        counter_types = {counter.get("type") for counter in observed}
        require(
            len(counter_ids) == 1 and len(counter_types) == 1,
            f"counter metadata changed across captures for {name!r}",
        )
        medians = [float(counter["median"]) for counter in observed]
        pair_ratios = None
        median_ratio = None
        relative_change_percent = None
        direction = None
        classification = "ineligible_nonpositive_median"
        if all(median > 0.0 for median in medians):
            pair_ratios = [medians[1] / medians[0], medians[2] / medians[3]]
            median_ratio = statistics.median(pair_ratios)
            relative_change_percent = (median_ratio - 1.0) * 100.0
            if pair_ratios[0] > 1.0 and pair_ratios[1] > 1.0:
                direction = "variant_higher"
            elif pair_ratios[0] < 1.0 and pair_ratios[1] < 1.0:
                direction = "variant_lower"
            if direction is None:
                classification = "directionally_inconsistent"
            elif abs(median_ratio - 1.0) < material_change:
                classification = "below_material_threshold"
            else:
                classification = "repeatable_difference"
        comparisons.append(
            {
                "counter_id": next(iter(counter_ids)),
                "name": name,
                "type": next(iter(counter_types)),
                "capture_medians": {
                    capture_id: median
                    for (capture_id, _), median in zip(
                        CAPTURE_SPECS, medians, strict=True
                    )
                },
                "pair_variant_over_baseline_ratios": pair_ratios,
                "median_variant_over_baseline_ratio": median_ratio,
                "relative_change_percent": relative_change_percent,
                "direction": direction,
                "classification": classification,
            }
        )

    repeatable = [
        counter
        for counter in comparisons
        if counter["classification"] == "repeatable_difference"
    ]
    return {
        "schema_version": 1,
        "analysis": "rmsnorm_limiter_counter_comparison",
        "protocol": {
            "capture_order": [role for _, role in CAPTURE_SPECS],
            "pairs": [
                "capture-02-variant / capture-01-baseline",
                "capture-03-variant / capture-04-baseline",
            ],
            "warmup_dispatches": warmup_iterations,
            "profile_dispatches": profile_iterations,
            "minimum_counter_timestamps": minimum_timestamps,
            "minimum_counter_sample_span_fraction": minimum_sample_span,
            "material_relative_change_threshold": material_change,
            "repeatable_rule": (
                "Positive medians in all captures, both pair ratios in the "
                "same direction, and median absolute relative change at or "
                "above the material threshold."
            ),
        },
        "captures": captures,
        "counter_summary": {
            "compared_count": len(comparisons),
            "eligible_positive_median_count": sum(
                counter["classification"]
                != "ineligible_nonpositive_median"
                for counter in comparisons
            ),
            "repeatable_difference_count": len(repeatable),
            "repeatable_difference_names": [
                counter["name"] for counter in repeatable
            ],
        },
        "comparisons": comparisons,
        "caveats": [
            "Counters are device-wide samples inside the target profile "
            "window, not command-buffer-exclusive measurements.",
            "The named counter set has no direct workgroup-barrier stall counter.",
            "Instrumented command durations are diagnostic and do not replace "
            "ordinary paired timing.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-first", type=Path, required=True)
    parser.add_argument("--variant-second", type=Path, required=True)
    parser.add_argument("--variant-third", type=Path, required=True)
    parser.add_argument("--baseline-fourth", type=Path, required=True)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=DEFAULT_PROFILE_ITERATIONS,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        args.baseline_first,
        args.variant_second,
        args.variant_third,
        args.baseline_fourth,
    ]
    summaries = [json.loads(path.read_text()) for path in paths]
    result = compare(
        summaries,
        warmup_iterations=args.warmup_iterations,
        profile_iterations=args.profile_iterations,
    )
    result["inputs"] = [
        artifact(path, capture_id)
        for path, (capture_id, _) in zip(paths, CAPTURE_SPECS, strict=True)
    ]
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"counter comparison: {args.output.resolve()}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    try:
        main()
    except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
