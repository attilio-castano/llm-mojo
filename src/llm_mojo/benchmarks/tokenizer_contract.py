"""CPU tokenizer adapter for the shared paired protocol and sample format."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import platform
import subprocess

from .._repository import repository_root, environment_tool
from ..tokenizer_assets import (
    ensure_prepared,
    prepared_directory,
    asset_directory,
)
from .environment import (
    repository_state,
    command,
    utc_now,
    ensure_record_location,
)
from .study import (
    BLOCKS,
    REPETITIONS,
    WARMUP,
    sha,
    write_json,
    encode_samples,
    read_samples,
    summarize,
    comparisons,
)

MODES = ("bpe", "encode", "decode", "stream", "load")


def sources():
    root = repository_root()
    names = [
        "src/llm_mojo/tokenizer.mojo",
        "src/llm_mojo/tokenizer_assets.py",
        "src/llm_mojo/benchmarks/tokenizer.mojo",
        "src/llm_mojo/benchmarks/tokenizer_contract.py",
        "src/llm_mojo/benchmarks/study.py",
        "src/llm_mojo/benchmarks/run.py",
        "tests/fixtures/tokenizer/generate.py",
        "tests/fixtures/tokenizer/benchmark_checksums.json",
        "tests/fixtures/tokenizer_reference.py",
        "tests/fixtures/generate.py.lock",
        "uv.lock",
    ]
    return {name: sha(root / name) for name in names}


def environment():
    if platform.system() != "Darwin":
        raise ValueError("the CPU tokenizer measurement harness requires Darwin")
    return dict(
        backend="cpu",
        os=platform.platform(),
        machine=platform.machine(),
        cpu=command("sysctl", "-n", "machdep.cpu.brand_string"),
        logical_cpus=command("sysctl", "-n", "hw.ncpu"),
        mojo=command(environment_tool("mojo"), "--version"),
    )


def fixtures():
    root = repository_root()
    path = root / "build/oracle_data/tokenizer/benchmark.bin"
    manifest = json.loads(
        (root / "tests/fixtures/tokenizer/benchmark_checksums.json").read_text()
    )
    if sha(path) != manifest["fixture_sha256"]:
        raise ValueError("tokenizer benchmark fixture mismatch")
    return path, manifest


def build(directory):
    ensure_record_location(directory)
    table = ensure_prepared(download=False)
    subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "--script",
            "tests/fixtures/tokenizer_reference.py",
            "--benchmark",
        ],
        cwd=repository_root(),
        check=True,
    )
    _, inputs = fixtures()
    repo = repository_state()
    source = sources()
    env = environment()
    if repo["dirty"]:
        raise ValueError("CPU benchmark build requires a clean source commit")
    directory.mkdir(parents=True, exist_ok=False)
    binary = directory / "tokenizer"
    cmd = [
        environment_tool("mojo"),
        "build",
        "-I",
        "src",
        "src/llm_mojo/benchmarks/tokenizer.mojo",
        "-o",
        str(binary),
    ]
    subprocess.run(
        cmd,
        cwd=repository_root(),
        check=True,
        env={k: v for k, v in os.environ.items() if k != "MODULAR_DEBUG"},
    )
    if repo != repository_state() or source != sources():
        raise ValueError("source changed during build")
    write_json(
        directory / "build.json",
        dict(
            repository=repo,
            sources=source,
            environment=env,
            binary_sha256=sha(binary),
            tables_sha256=sha(table),
            inputs=inputs,
            table_manifest=json.loads(
                (table.parent / "manifest.json").read_text()
            ),
            command=[
                "mojo",
                "build",
                "-I",
                "src",
                "src/llm_mojo/benchmarks/tokenizer.mojo",
                "-o",
                "<binary>",
            ],
        ),
    )


def specification(mode, inputs):
    # The shared sample format calls its workload index `rows`. For this CPU
    # adapter it is an opaque case ID; actual byte/piece/token sizes live in cases.
    return dict(
        operation="tokenizer",
        mode=mode,
        control=0,
        candidates=[0, 1] if mode in ("bpe", "encode") else [0],
        names={0: "scan", 1: "heap"} if mode
        in ("bpe", "encode") else {0: mode},
        layers=[1],
        rows=[1] if mode == "load" else [c["case"] for c in inputs["cases"]],
        row_semantics="case ID in cases; layers=1 means one sequence",
    )


def parse_output(output, case, mode, candidate, first):
    lines = output.splitlines()
    required = [
        "api: cpu",
        "timer: clock_gettime_nsec_np CLOCK_UPTIME_RAW",
        "operation: tokenizer",
        f'case: {case} mode: {mode}',
        "correctness: passed",
        "BENCHMARK_COMPLETE",
    ]
    if any(
        lines.count(x) != 1 for x in required
    ) or not output.rstrip().endswith("BENCHMARK_COMPLETE"):
        raise ValueError("missing CPU tokenizer runtime/completion contract")
    caps = [x for x in lines if x.startswith("workspace capacity: ")]
    if len(caps) != 1:
        raise ValueError("missing workspace metadata")
    capacities = list(map(int, caps[0].split(": ")[1].split()))
    if len(capacities) != 4 or any(x < 0 for x in capacities):
        raise ValueError("invalid workspace metadata")
    observations = []
    for line in lines:
        if line.startswith("SAMPLE "):
            _, arm, variant, rep, value = line.split()
            us = float(value)
            if not math.isfinite(us) or us <= 0:
                raise ValueError("invalid CPU sample")
            observations.append(
                dict(arm=arm, variant=int(variant), repetition=int(rep), us=us)
            )
    arms = [("control", 0), ("candidate", candidate)]
    if first:
        arms.reverse()
    expected = [(arm, v, r) for arm, v in arms for r in range(REPETITIONS)]
    if [
        (s["arm"], s["variant"], s["repetition"]) for s in observations
    ] != expected:
        raise ValueError("wrong CPU sample order or implementation")
    return observations, capacities


def run(directory, output):
    ensure_record_location(output)
    from .run import checked_conditions

    table = ensure_prepared(download=False)
    fixture, inputs = fixtures()
    provenance = json.loads((directory / "build.json").read_text())
    repo = repository_state()
    env = environment()

    def verify():
        if (
            repo["dirty"]
            or repository_state() != repo
            or provenance["repository"] != repo
            or provenance["sources"] != sources()
        ):
            raise ValueError("CPU run requires the same clean source commit")
        if provenance["environment"] != environment() or provenance[
            "binary_sha256"
        ] != sha(directory / "tokenizer"):
            raise ValueError("CPU environment or binary mismatch")
        if (
            provenance["tables_sha256"] != sha(table)
            or provenance["inputs"] != fixtures()[1]
        ):
            raise ValueError("CPU tokenizer table/fixture mismatch")

    verify()
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    samples = {}
    for mode in MODES:
        spec = specification(mode, inputs)
        target = output / mode
        target.mkdir()
        records[mode] = dict(
            schema=1,
            study="tokenizer_" + mode,
            specification=spec,
            cases=inputs["cases"],
            repository=repo,
            build=provenance,
            runtime=dict(api="cpu", device=env["cpu"]),
            timer="Darwin clock_gettime_nsec_np(CLOCK_UPTIME_RAW), nanoseconds",
            representation=dict(
                input="UTF-8 UInt8",
                token_ids="Mojo Int on recorded 64-bit host",
                merge_key_value="UInt64",
                prepared_fields="little-endian UInt32",
            ),
            blocks=BLOCKS,
            repetitions=REPETITIONS,
            warmup=WARMUP,
            timing="Native monotonic wall time per complete CPU call, including call allocations and prior-result destruction. Setup, fixtures, output consumption and printing excluded. Symbol/heap buffers reused; BPE and streaming output buffers reused. BPE mode receives precomputed pieces; encode mode starts from UTF-8 bytes.",
            started_utc=utc_now(),
            conditions=[],
            workspace_capacities={},
        )
        samples[mode] = []
    for block in range(1, BLOCKS + 1):
        before = checked_conditions()
        first = block in (2, 3)
        modes = list(MODES)
        if first:
            modes.reverse()
        for mode in modes:
            record = records[mode]
            spec = record["specification"]
            target = output / mode
            cases = comparisons(spec)
            if first:
                cases.reverse()
            for workload, layers, candidate in cases:
                case = workload["rows"]
                cmd = [
                    str(directory / "tokenizer"),
                    str(table),
                    str(fixture),
                    str(case),
                    mode,
                    str(candidate),
                    str(int(first)),
                    str(REPETITIONS),
                    str(WARMUP),
                ]
                process = subprocess.run(
                    cmd,
                    cwd=repository_root(),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env={
                        k: v
                        for k, v in os.environ.items()
                        if k != "MODULAR_DEBUG"
                    },
                )
                (target / "last-process.txt").write_text(
                    process.stdout + process.stderr
                )
                process.check_returncode()
                observations, capacities = parse_output(
                    process.stdout, case, mode, candidate, first
                )
                record["workspace_capacities"][str(case)] = capacities
                samples[mode].extend(
                    dict(
                        block=block,
                        rows=case,
                        layers=1,
                        candidate=candidate,
                        **s,
                    )
                    for s in observations
                )
            (target / "samples.csv.gz").write_bytes(
                encode_samples(samples[mode])
            )
            write_json(target / "run.json", record)
        after = checked_conditions()
        for mode in MODES:
            records[mode]["conditions"].append(
                dict(block=block, before=before, after=after)
            )
            write_json(output / mode / "run.json", records[mode])
        verify()
        print("Tokenizer CPU block", block, "complete", flush=True)
    for mode in MODES:
        record = records[mode]
        summarize(samples[mode], record["specification"])
        record.update(
            completed_utc=utc_now(),
            samples_sha256=sha(output / mode / "samples.csv.gz"),
        )
        write_json(output / mode / "run.json", record)
    report(output)


def load(directory):
    record = json.loads((directory / "run.json").read_text())
    if (
        record.get("schema") != 1
        or not record.get("completed_utc")
        or record["repository"]["dirty"]
        or record["runtime"]["api"] != "cpu"
        or record["build"]["environment"]["backend"] != "cpu"
        or record["runtime"]["device"] != record["build"]["environment"]["cpu"]
        or record["repository"] != record["build"]["repository"]
    ):
        raise ValueError("unverified CPU run")
    if (record["blocks"], record["repetitions"], record["warmup"]) != (
        BLOCKS,
        REPETITIONS,
        WARMUP,
    ):
        raise ValueError("CPU protocol mismatch")
    if [c["block"] for c in record["conditions"]] != list(
        range(1, BLOCKS + 1)
    ) or any("after" not in c for c in record["conditions"]):
        raise ValueError("missing CPU conditions")
    if sha(directory / "samples.csv.gz") != record["samples_sha256"]:
        raise ValueError("CPU samples changed")
    samples = read_samples(directory / "samples.csv.gz")
    return record, summarize(samples, record["specification"])


def report(output):
    for mode in MODES:
        record, summary = load(output / mode)
        cases = {c["case"]: c for c in record["cases"]}
        rows = [
            dict(
                mode=mode,
                **cases[s["rows"]],
                **{k: v for k, v in s.items() if k not in ("rows", "layers")},
            )
            for s in summary
        ]
        with (output / mode / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(
            mode,
            {
                decision: sum(s["decision"] == decision for s in summary)
                for decision in (
                    "faster",
                    "slower",
                    "inconclusive",
                    "calibration",
                )
            },
        )


def plot(output):
    """Regenerate the study question figure solely from validated raw samples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    colors = dict(
        prose="#2563eb",
        multilingual="#059669",
        code="#d97706",
        whitespace="#9333ea",
        long_piece="#dc2626",
    )
    for ax, mode, title in zip(
        axes,
        ("bpe", "encode"),
        ("BPE on prepared pieces", "Complete text encoding"),
    ):
        record, summary = load(output / mode)
        cases = {c["case"]: c for c in record["cases"]}
        for family, color in colors.items():
            points = [
                (cases[s["rows"]]["input_bytes"], s)
                for s in summary
                if s["candidate"] == 1 and cases[s["rows"]]["family"] == family
            ]
            points.sort(key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1]["ratio"] for p in points]
            ax.plot(
                xs,
                ys,
                color=color,
                label=family.replace("_", " "),
                linewidth=1.5,
            )
            ax.fill_between(
                xs,
                [p[1]["ratio_min"] for p in points],
                [p[1]["ratio_max"] for p in points],
                color=color,
                alpha=0.12,
            )
            for x, s in points:
                ax.scatter(
                    [x],
                    [s["ratio"]],
                    color=color,
                    facecolors=color if s["decision"] == "faster" else "white",
                    s=30,
                    zorder=3,
                )
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Input UTF-8 bytes")
        ax.set_title(title)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Heap / scan latency (lower is faster)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Does faster BPE improve the complete tokenizer?")
    fig.text(
        0.5,
        0.075,
        "Bands: range of four paired block ratios, not confidence intervals. Filled points pass the frozen gain gate.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])
    fig.savefig(output / "heap-vs-scan.png", dpi=180)
    plt.close(fig)
