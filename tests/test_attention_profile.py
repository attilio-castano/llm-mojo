import copy
import unittest
from llm_mojo.benchmarks.attention_decode_contract import configuration, ENTRYPOINTS, VARIANTS
from llm_mojo.benchmarks.capture_trace import profile_contract, parse_target_identity, validate_target_identity
from llm_mojo.benchmarks.analyze_trace import segment_compute_commands

class AttentionProfileTests(unittest.TestCase):
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

