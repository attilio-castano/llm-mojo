import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks import run_attention_decode as runner
from benchmarks.run_attention_decode import load_noise, parse_output, summarize
from benchmarks.attention_decode_contract import (
    configuration,
    ENTRYPOINTS,
    VARIANTS,
)
from benchmarks.capture_rms_norm_trace import (
    profile_contract,
    parse_target_identity,
    validate_target_identity,
)
from benchmarks.analyze_rms_norm_trace import segment_compute_commands


class DecodeRunnerTests(unittest.TestCase):
    def noise_metadata(self):
        return {
            "recorded": True, "completed_utc": "2026-09-05T12:00:00Z",
            "repository": {"commit": "a" * 40, "dirty": False},
            "build": {"binary_sha256": "b" * 64, "source_sha256": {"kernel": "c" * 64}},
            "control": 0, "candidates": [0], "lengths": [64], "seed": 17,
            "hardware": {"chip": "Apple Test GPU"}, "software": {"mojo": "test"},
            "timing": "enqueue through sync", "dtype": "BF16", "layout": "contiguous",
            "warmup": 10, "repetitions": 10,
            "runtime": {"device": "Apple Test GPU", "api": "metal", "correctness": "passed"},
        }

    def load_calibration(self, meta=None, rows=None, current=None):
        if meta is None:
            meta = self.noise_metadata()
        if rows is None:
            rows = [
                {"candidate": 0, "rows": 64, "layers": layers,
                 "block_ratios": [1.01, 0.98, 1.03, 0.99]}
                for layers in (1, 24)
            ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(rows))
            path.with_name("metadata.json").write_text(json.dumps(meta))
            return load_noise(path, current or self.noise_metadata())

    def test_noise_binds_baseline_calibration_and_retains_reproducible_floors(self):
        current = self.noise_metadata()
        current.update(control=4, candidates=[9])
        floors, record = self.load_calibration(current=current)
        self.assertAlmostEqual(floors["64/24"], 0.03)
        self.assertEqual(record["identity"]["control"], 0)
        self.assertEqual(record["identity"]["candidates"], [0])
        self.assertEqual(record["metadata_sha256"], hashlib.sha256(
            json.dumps(self.noise_metadata()).encode()).hexdigest())
        self.assertEqual(len(record["summary_sha256"]), 64)
        self.assertEqual(floors, {
            key: max(abs(1 - ratio) for ratio in ratios)
            for key, ratios in record["block_ratios"].items()
        })

    def test_noise_rejects_comparisons_and_incompatible_identity(self):
        for field, value in (
            ("control", 4), ("candidates", [9]), ("recorded", False),
            ("completed_utc", None), ("seed", 101), ("warmup", 5),
            ("repetitions", 5), ("hardware", {}), ("software", {}),
            ("timing", "different boundary"), ("dtype", "FP32"),
            ("layout", "strided"), ("runtime", {}),
            ("repository", {"commit": "d" * 40, "dirty": False}),
            ("repository", {"commit": "a" * 40, "dirty": True}),
            ("build", {"binary_sha256": "d" * 64, "source_sha256": {"kernel": "c" * 64}}),
            ("build", {"binary_sha256": "b" * 64, "source_sha256": {"kernel": "d" * 64}}),
        ):
            with self.subTest(field=field, value=value):
                meta = self.noise_metadata()
                meta[field] = value
                with self.assertRaises(ValueError):
                    self.load_calibration(meta=meta)

    def test_noise_rejects_missing_duplicate_and_malformed_workloads(self):
        good = [
            {"candidate": 0, "rows": 64, "layers": layers,
             "block_ratios": [1.0] * 4}
            for layers in (1, 24)
        ]
        bad_rows = [[], good[:1], good + good[:1]]
        for field, value in (
            ("candidate", 9), ("rows", 128), ("layers", 2),
            ("block_ratios", [1.0] * 3),
            ("block_ratios", [1, 1, 1, float("nan")]),
            ("block_ratios", [1, 1, 1, float("inf")]),
            ("block_ratios", [1, 1, 1, 0]),
            ("block_ratios", [1, 1, 1, -1]),
        ):
            rows = copy.deepcopy(good)
            rows[0][field] = value
            bad_rows.append(rows)
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load_calibration(rows=rows)

    def test_run_checks_calibrated_runtime_before_retaining_samples(self):
        runtime = self.noise_metadata()["runtime"]
        for device in ("Apple Test GPU", "Apple Different GPU"):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binary = root / "benchmark"
                binary.write_bytes(b"test binary")
                repo = {"commit": "a" * 40, "dirty": False}
                provenance = {
                    "repository": repo, "binary_sha256": runner.sha(binary),
                    "source_sha256": {p: runner.sha(runner.REPOSITORY / p) for p in runner.SOURCES},
                }
                binary.with_suffix(".provenance.json").write_text(json.dumps(provenance))
                args = SimpleNamespace(
                    output_dir=root / "run", binary=binary, recorded=False,
                    blocks=1, candidates="1", control=0, lengths="64", seed=17,
                    noise=root / "noise" / "summary.json",
                )
                record = {"identity": {"runtime": runtime}, "floors": {"64/1": 0.2, "64/24": 0.2}}
                process = SimpleNamespace(stdout="", stderr="", check_returncode=lambda: None)
                identity = {**runtime, "device": device}
                samples = parse_output(self.output(), 0, 1, False)[1]
                with (
                    patch.object(runner, "ensure_record_location"),
                    patch.object(runner, "repository_state", return_value=repo),
                    patch.object(runner, "stable_environment", return_value={}),
                    patch.object(runner, "conditions_snapshot", return_value={}),
                    patch.object(runner, "load_noise", return_value=(record["floors"], record)),
                    patch.object(runner.subprocess, "run", return_value=process),
                    patch.object(runner, "parse_output", return_value=(identity, samples)),
                    patch("builtins.print"),
                ):
                    if device != runtime["device"]:
                        with self.assertRaisesRegex(RuntimeError, "noise calibration"):
                            runner.run(args)
                        self.assertFalse((args.output_dir / "samples.jsonl").exists())
                    else:
                        runner.run(args)
                        meta = json.loads((args.output_dir / "metadata.json").read_text())
                        self.assertEqual(meta["noise"], record)
                        summary = json.loads((args.output_dir / "summary.json").read_text())
                        self.assertTrue(all(row["noise_floor"] == 0.2 for row in summary))

    def test_runtime_shape_and_seed_must_match_requested_pair(self):
        text = (
            "shape: 64 24 seed: 17\nvariants: 0 1 candidate-first: 0\n"
            + self.output()
        )
        parse_output(text, 0, 1, False, rows=64, layers=24, seed=17)
        with self.assertRaises(ValueError):
            parse_output(text, 0, 1, False, rows=64, layers=24, seed=37)

    def output(self, first=False):
        arms = [("control", 0), ("candidate", 1)]
        if first:
            arms.reverse()
        return "\n".join(
            ["device: Apple Test GPU", "api: metal", "correctness: passed"]
            + [
                f'SAMPLE {arm} {v} {r} {10 if arm == "control" else 8}'
                for arm, v in arms
                for r in range(10)
            ]
            + ["BENCHMARK_COMPLETE"]
        )

    def test_parser_rejects_missing_identity_truncation_nan_and_wrong_order(
        self,
    ):
        text = self.output()
        self.assertEqual(len(parse_output(text, 0, 1, False)[1]), 20)
        for bad in (
            text.replace("metal", "cpu"),
            text.replace("Apple Test GPU", "CPU"),
            text.replace("correctness: passed", ""),
            text.replace("BENCHMARK_COMPLETE", ""),
            text.replace("candidate 1 0 8", "candidate 1 0 nan"),
            self.output(True),
        ):
            with self.assertRaises(ValueError):
                parse_output(bad, 0, 1, False)

    def samples(self):
        return [
            {**s, "block": b, "candidate": 1, "rows": 64, "layers": 1}
            for b in range(1, 5)
            for s in parse_output(self.output(), 0, 1, False)[1]
        ]

    def test_acceptance_requires_all_blocks_and_exceeds_noise(self):
        samples = self.samples()
        self.assertTrue(summarize(samples)[0]["accepted"])
        self.assertFalse(summarize(samples, {"64/1": 0.25})[0]["accepted"])
        self.assertFalse(
            summarize([s for s in samples if s["block"] != 4])[0]["accepted"]
        )
        for s in samples:
            if s["block"] == 4 and s["arm"] == "candidate":
                s["us"] = 11
        self.assertFalse(summarize(samples)[0]["accepted"])

    def profile(self, variant=7):
        g, h, s = VARIANTS[variant]
        return {
            "schema_version": 1,
            "operation": "grouped_query_attention_decode",
            "implementation": f'gqa_decode_{variant}',
            "entrypoint": ENTRYPOINTS[f'gqa_decode_{variant}'],
            "profile_rows": 1,
            "hidden_size": 64,
            "key_value_rows": 256,
            "query_heads": 14,
            "key_value_heads": 2,
            "groups": g,
            "heads": h,
            "splits": s,
            "profile_workload": f'decode-t256-v{variant}',
            "dispatches_per_iteration": 3 if variant
            == 0 else (2 if s > 1 else 1),
            "profile_warmup_iterations": 100,
            "profile_iterations": 500,
            "profile_post_idle_milliseconds": 250,
            "repository": {
                "commit": "a" * 40,
                "dirty": False,
                "branch": "codex/test",
            },
            "hardware": {"chip": "Apple Test GPU", "gpu_api": "metal"},
        }

    def test_profile_runtime_matches_shape_parameters_and_dispatches(self):
        for variant in (0, 1, 7, 8, 9, 10, 11, 12):
            p = self.profile(variant)
            cfg, _, hardware = profile_contract(p)
            text = "\n".join(
                [
                    f'profile implementation: {p["entrypoint"]}',
                    "device: Apple Test GPU",
                    "api: metal",
                    "rows: 1",
                    "hidden: 64",
                    "warmup iterations: 100",
                    "profile iterations: 500",
                    "post-profile idle milliseconds: 250",
                ]
                + [
                    f'{label}: {p[k]}'
                    for label, k in (
                        ("profile workload", "profile_workload"),
                        (
                            "profile dispatches per iteration",
                            "dispatches_per_iteration",
                        ),
                        ("key value rows", "key_value_rows"),
                        ("query heads", "query_heads"),
                        ("key value heads", "key_value_heads"),
                        ("groups", "groups"),
                        ("heads", "heads"),
                        ("splits", "splits"),
                    )
                ]
            )
            target = parse_target_identity(text)
            validate_target_identity(target, cfg, hardware)
            target["key_value_rows"] = 16
            with self.assertRaises(ValueError):
                validate_target_identity(target, cfg, hardware)

    def test_profile_rejects_forged_shapes_counts_and_unbounded_runs(self):
        p = self.profile()
        for key, value in (
            ("splits", 4),
            ("key_value_rows", 0),
            ("query_heads", 2),
            ("profile_iterations", 3000),
            ("dispatches_per_iteration", 1),
            ("groups", True),
        ):
            bad = copy.deepcopy(p)
            bad[key] = value
            with self.assertRaises(ValueError):
                configuration(bad)
        setup, correctness, warmup, profile = segment_compute_commands(
            list(range(17)), 2, 5, 2, False
        )
        self.assertEqual(
            (len(setup), len(correctness), len(warmup), len(profile)),
            (3, 0, 4, 10),
        )


if __name__ == "__main__":
    unittest.main()
