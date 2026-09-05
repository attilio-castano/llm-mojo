import copy
import unittest

from benchmarks.run_attention_decode import parse_output, summarize
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
