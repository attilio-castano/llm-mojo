"""Build and record paired output-only GQA decode experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

try:
    from benchmarks.run_rms_norm import (
        REPOSITORY,
        conditions_snapshot,
        ensure_record_location,
        repository_state,
        require_ac,
        require_nominal_thermal_state,
        stable_environment,
        utc_now,
    )
    from benchmarks.attention_decode_contract import VARIANTS
except ModuleNotFoundError:
    from run_rms_norm import (
        REPOSITORY,
        conditions_snapshot,
        ensure_record_location,
        repository_state,
        require_ac,
        require_nominal_thermal_state,
        stable_environment,
        utc_now,
    )
    from attention_decode_contract import VARIANTS

LENGTHS = (1, 16, 64, 256, 1024, 4096)
SOURCES = (
    "benchmarks/attention_decode.mojo",
    "tests/attention_decode_support.mojo",
    "benchmarks/attention_decode_contract.py",
    "src/llm_mojo/attention.mojo",
    "src/llm_mojo/attention_decode.mojo",
    "uv.lock",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def name(variant):
    return "materialized" if variant == 0 else (
        ("g%d-h%d-s%d" % VARIANTS[variant])
        + ("-conditional" if variant in (11, 12) else "")
    )


def specification(variant, rows):
    g, h, s = VARIANTS[variant]
    return {
        "id": variant,
        "name": name(variant),
        "groups": g,
        "conditional_rescale": variant in (11, 12),
        "heads": h,
        "splits": s,
        "dispatches": 3 if variant == 0 else (2 if s > 1 else 1),
        "workspace_bytes": rows * 14 * 2 if variant
        == 0 else (14 * s * 66 * 4 if s > 1 else 0),
    }


def build(binary):
    ensure_record_location(binary)
    if binary.exists() or binary.with_suffix(".provenance.json").exists():
        raise RuntimeError("refusing to overwrite binary/provenance")
    repo = repository_state()
    if repo["dirty"]:
        raise RuntimeError("retained binary build requires a clean repository")
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "--locked",
        "mojo",
        "build",
        "-I",
        "src",
        "-I",
        "tests",
        "benchmarks/attention_decode.mojo",
        "-o",
        str(binary),
    ]
    subprocess.run(command, cwd=REPOSITORY, check=True)
    record = {
        "repository": repo,
        "binary_sha256": sha(binary),
        "source_sha256": {p: sha(REPOSITORY / p) for p in SOURCES},
        "command": command[:-1] + ["<external-binary>"],
        "created_utc": utc_now(),
        "variants": {v: specification(v, 4096) for v in VARIANTS},
    }
    binary.with_suffix(".provenance.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


def build_profile(args):
    binary = args.build_profile_binary.resolve()
    ensure_record_location(binary)
    if binary.exists() or Path(str(binary) + ".provenance.json").exists():
        raise RuntimeError("refusing to overwrite profile artifacts")
    repo = repository_state()
    spec = specification(args.profile_variant, args.profile_rows)
    if (
        repo["dirty"]
        or not 1 <= args.profile_rows <= 4096
        or not 1 <= args.profile_iterations * spec["dispatches"] <= 5000
    ):
        raise RuntimeError(
            "profile requires clean source, bounded shape and dispatch count"
        )
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "--locked",
        "mojo",
        "build",
        "-I",
        "src",
        "-I",
        "tests",
        "-D",
        f'GQA_PROFILE_ROWS={args.profile_rows}',
        "-D",
        f'GQA_PROFILE_VARIANT={args.profile_variant}',
        "-D",
        f'GQA_PROFILE_ITERATIONS={args.profile_iterations}',
        "benchmarks/attention_decode.mojo",
        "-o",
        str(binary),
    ]
    environment = os.environ.copy()
    environment.pop("MODULAR_DEBUG", None)
    subprocess.run(command, cwd=REPOSITORY, check=True, env=environment)
    record = {
        "schema_version": 1,
        "operation": "grouped_query_attention_decode",
        "repository": repo,
        **stable_environment(),
        "created_utc": utc_now(),
        "implementation": f'gqa_decode_{args.profile_variant}',
        "entrypoint": "enqueue_grouped_query_attention_apple_gpu" if args.profile_variant
        == 0 else "enqueue_grouped_query_attention_decode_apple_gpu",
        "profile_rows": 1,
        "hidden_size": 64,
        "key_value_rows": args.profile_rows,
        "query_heads": 14,
        "key_value_heads": 2,
        "groups": spec["groups"],
        "heads": spec["heads"],
        "splits": spec["splits"],
        "conditional_rescale": spec["conditional_rescale"],
        "profile_workload": f'decode-t{args.profile_rows}-v{args.profile_variant}',
        "dispatches_per_iteration": spec["dispatches"],
        "profile_warmup_iterations": 100,
        "profile_iterations": args.profile_iterations,
        "profile_post_idle_milliseconds": 250,
        "source_sha256": {p: sha(REPOSITORY / p) for p in SOURCES},
        "binary": {"bytes": binary.stat().st_size, "sha256": sha(binary)},
        "command": command[:-1] + ["<external-profile-binary>"],
    }
    Path(str(binary) + ".provenance.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


def parse_output(
    output,
    control,
    candidate,
    first,
    repetitions=10,
    *,
    rows=None,
    layers=None,
    seed=None,
):
    fields = {}
    samples = []
    for line in output.splitlines():
        if line.startswith(("device:", "api:", "correctness:")):
            k, v = line.split(":", 1)
            fields[k] = v.strip()
        if line.startswith("SAMPLE "):
            _, arm, variant, rep, value = line.split()
            sample = {
                "arm": arm,
                "variant": int(variant),
                "repetition": int(rep),
                "us": float(value),
            }
            if not math.isfinite(sample["us"]) or sample["us"] <= 0:
                raise ValueError("non-finite or non-positive timing")
            samples.append(sample)
    if (
        fields.get("api") != "metal"
        or not fields.get("device", "").startswith("Apple ")
        or fields.get("correctness") != "passed"
    ):
        raise ValueError("missing Metal device or correctness evidence")
    arms = [("control", control), ("candidate", candidate)]
    if first:
        arms.reverse()
    expected = [(arm, v, r) for arm, v in arms for r in range(repetitions)]
    if [(s["arm"], s["variant"], s["repetition"]) for s in samples] != expected:
        raise ValueError("unexpected sample order, count or implementation")
    if not output.rstrip().endswith("BENCHMARK_COMPLETE"):
        raise ValueError("benchmark did not complete")
    if rows is not None:
        lines = output.splitlines()
        if (
            lines.count(f"shape: {rows} {layers} seed: {seed}") != 1
            or lines.count(
                f"variants: {control} {candidate} candidate-first: {int(first)}"
            )
            != 1
        ):
            raise ValueError(
                "runtime workload identity disagrees with requested pair"
            )
    return fields, samples


def summarize(samples, noise=None):
    grouped = defaultdict(list)
    for s in samples:
        grouped[(s["candidate"], s["rows"], s["layers"])].append(s)
    report = []
    for (candidate, rows, layers), group in grouped.items():
        ratios, controls, candidates = [], [], []
        for block in sorted({s["block"] for s in group}):
            arms = [
                [
                    s["us"]
                    for s in group
                    if s["block"] == block and s["arm"] == arm
                ]
                for arm in ("control", "candidate")
            ]
            if any(len(a) != 10 for a in arms):
                raise ValueError("summary requires 10 samples per arm/block")
            c, v = map(statistics.median, arms)
            ratios.append(v / c)
            controls.append(c)
            candidates.append(v)
        effect = 1 - statistics.median(ratios)
        floor = (noise or {}).get(f'{rows}/{layers}', 0)
        report.append(
            {
                "candidate": candidate,
                "name": name(candidate),
                "rows": rows,
                "layers": layers,
                "control_us": statistics.median(controls),
                "candidate_us": statistics.median(candidates),
                "block_ratios": ratios,
                "improvement_fraction": effect,
                "noise_floor": floor,
                "accepted": len(ratios) == 4
                and effect >= max(0.05, floor)
                and all(r < 1 for r in ratios),
                "regression": effect < -0.05,
                **specification(candidate, rows),
            }
        )
    return report


def run(args):
    output = args.output_dir.resolve()
    ensure_record_location(output)
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    provenance = json.loads(binary.with_suffix(".provenance.json").read_text())
    repo = repository_state()
    if provenance["binary_sha256"] != sha(binary):
        raise RuntimeError("binary identity changed")
    if provenance["source_sha256"] != {p: sha(REPOSITORY / p) for p in SOURCES}:
        raise RuntimeError("binary sources do not match current checkout")
    if args.recorded and (
        repo["dirty"]
        or provenance["repository"]["commit"] != repo["commit"]
        or args.blocks != 4
    ):
        raise RuntimeError(
            "recorded run requires matching clean commit and four blocks"
        )
    candidates = [int(v) for v in args.candidates.split(",")]
    lengths = [int(v) for v in args.lengths.split(",")]
    if (
        not candidates
        or len(candidates) != len(set(candidates))
        or any(v not in VARIANTS for v in candidates)
        or args.control not in VARIANTS
    ):
        raise ValueError("invalid variants")
    if (
        not lengths
        or len(lengths) != len(set(lengths))
        or any(t < 1 or t > 4096 for t in lengths)
    ):
        raise ValueError("invalid lengths")
    noise = {}
    if args.noise:
        for row in json.loads(args.noise.read_text()):
            noise[f"{row['rows']}/{row['layers']}"] = max(
                abs(1 - r) for r in row["block_ratios"]
            )
    metadata = {
        "schema_version": 1,
        "experiment": "EXP-0014",
        "recorded": args.recorded,
        "repository": repo,
        "build": provenance,
        **stable_environment(),
        "seed": args.seed,
        "control": args.control,
        "candidates": candidates,
        "lengths": lengths,
        "timing": "host monotonic enqueue through synchronization, us per attention; ring24 divides sweep by 24",
        "dtype": "BF16 Q/K/V/O/scaled scores; FP32 online state",
        "layout": "contiguous row major Q/O[1,14,64], K/V[T,2,64]",
        "warmup": 10,
        "repetitions": 10,
        "conditions": [],
        "started_utc": utc_now(),
    }
    all_samples = []
    for block in range(args.blocks):
        before = conditions_snapshot()
        if args.recorded:
            require_ac(before)
            require_nominal_thermal_state(before)
        first = block % 4 in (1, 2)
        workloads = [(t, layers) for t in lengths for layers in (1, 24)]
        if first:
            workloads.reverse()
        for rows, layers in workloads:
            for candidate in reversed(candidates) if first else candidates:
                command = [
                    str(binary),
                    str(rows),
                    str(layers),
                    str(candidate),
                    str(args.control),
                    str(int(first)),
                    str(args.seed),
                    "bench",
                    "10",
                    "10",
                ]
                environment = os.environ.copy()
                environment.pop("MODULAR_DEBUG", None)
                process = subprocess.run(
                    command, capture_output=True, text=True, env=environment
                )
                prefix = f'b{block + 1}-t{rows}-l{layers}-c{candidate}'
                (output / (prefix + ".txt")).write_text(
                    process.stdout + process.stderr
                )
                process.check_returncode()
                identity, samples = parse_output(
                    process.stdout,
                    args.control,
                    candidate,
                    first,
                    rows=rows,
                    layers=layers,
                    seed=args.seed,
                )
                if "runtime" in metadata and metadata["runtime"] != identity:
                    raise RuntimeError("runtime identity changed during run")
                metadata["runtime"] = identity
                all_samples.extend(
                    {
                        **s,
                        "block": block + 1,
                        "rows": rows,
                        "layers": layers,
                        "candidate": candidate,
                        "control": args.control,
                        "candidate_first": first,
                    }
                    for s in samples
                )
                (output / "samples.jsonl").write_text(
                    "".join(json.dumps(s) + "\n" for s in all_samples)
                )
                print(prefix, "complete", flush=True)
        after = conditions_snapshot()
        if args.recorded:
            require_ac(after)
            require_nominal_thermal_state(after)
        metadata["conditions"].append(
            {"block": block + 1, "before": before, "after": after}
        )
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    if repository_state() != repo:
        raise RuntimeError("repository changed during benchmark")
    metadata["completed_utc"] = utc_now()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = summarize(all_samples, noise)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-binary", type=Path)
    p.add_argument("--build-profile-binary", type=Path)
    p.add_argument(
        "--profile-variant", type=int, choices=tuple(VARIANTS), default=0
    )
    p.add_argument("--profile-rows", type=int, default=16)
    p.add_argument("--profile-iterations", type=int, default=500)
    p.add_argument("--binary", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--recorded", action="store_true")
    p.add_argument("--blocks", type=int, choices=(1, 4), default=4)
    p.add_argument("--candidates", default="0")
    p.add_argument("--control", type=int, default=0)
    p.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--noise", type=Path)
    args = p.parse_args()
    if args.build_binary:
        build(args.build_binary.resolve())
    elif args.build_profile_binary:
        build_profile(args)
    elif args.binary and args.output_dir:
        run(args)
    else:
        p.error("provide --build-binary or both --binary and --output-dir")


if __name__ == "__main__":
    main()
