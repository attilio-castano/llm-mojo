import unittest

from benchmarks.compare_rms_norm_counters import compare


def counter(counter_id: int, name: str, median: float):
    return {
        "counter_id": counter_id,
        "name": name,
        "type": "Percentage",
        "description": "Synthetic counter percentage",
        "unit": "percent",
        "median": median,
    }


def summary(
    medians: list[float],
    *,
    capture_index: int,
    role: str | None = None,
    sample_span: float = 0.9,
    commit: str = "c" * 40,
    device: str = "Apple Test GPU",
    binary_sha256: str | None = None,
    provenance_sha256: str | None = None,
):
    roles = ("baseline", "variant", "variant", "baseline")
    role = role or roles[capture_index - 1]
    if role == "baseline":
        implementation = "apple_gpu_shared_tree_v0"
        entrypoint = "enqueue_rms_norm_apple_gpu_shared_tree"
        binary_sha256 = binary_sha256 or "a" * 64
        provenance_sha256 = provenance_sha256 or "b" * 64
    else:
        implementation = "apple_gpu_simdgroup_v1"
        entrypoint = "enqueue_rms_norm_apple_gpu"
        binary_sha256 = binary_sha256 or "d" * 64
        provenance_sha256 = provenance_sha256 or "e" * 64
    capture_id = f"rmsnorm-{capture_index:032x}"
    return {
        "schema_version": 3,
        "analysis": "rmsnorm_metal_trace",
        "capture_identity": {
            "verification": "capture_receipt_v2_and_trace_target",
            "capture_id": capture_id,
            "implementation": implementation,
            "entrypoint": entrypoint,
            "binary": {"bytes": 100, "sha256": binary_sha256},
            "provenance": {"bytes": 200, "sha256": provenance_sha256},
            "capture_receipt": {
                "bytes": 300,
                "sha256": f"{capture_index:064x}",
            },
            "repository": {
                "commit": commit,
                "branch": "test",
                "dirty": False,
            },
            "runtime": {"device": device, "backend": "metal"},
            "workload": {
                "rows": 512,
                "hidden_size": 896,
                "warmup_iterations": 100,
                "profile_iterations": 500,
                "post_profile_idle_milliseconds": 250,
            },
        },
        "inputs": [
            {
                "kind": "capture_receipt_json",
                "bytes": 300,
                "sha256": f"{capture_index:064x}",
            }
        ],
        "trace": {
            "end_reason": "Target app exited",
            "template": "LLM_Mojo_Metal_Limiters",
            "recording_duration_seconds": 1.0,
            "metal_application_gpu_settings": [
                "Counter Set: Performance Limiters",
                "Shader Timeline: Disabled",
                "Induced GPU Performance State: Default",
            ],
            "target": {"capture_id_verified": True},
        },
        "validated_sequence": {
            "setup_compute_commands": 0,
            "correctness_dispatches": 1,
            "warmup_dispatches": 100,
            "profile_dispatches": 500,
            "compute_channel_command_kinds": {"compute": 601, "blit": 2},
        },
        "profile_gpu_counters": {
            "defined_counter_count": 4,
            "sampled_counter_count": 4,
            "samples": {
                "timestamp_count": 10,
                "sample_span_fraction": sample_span,
            },
            "profile_window": {
                "duration_nanoseconds": 1000,
                "target_gpu_busy_fraction": 0.9,
            },
            "counters": [
                counter(1, "repeatable", medians[0]),
                counter(2, "inconsistent", medians[1]),
                counter(3, "small", medians[2]),
                counter(4, "zero", medians[3]),
            ],
        },
        "gpu_performance_state": {"states": {"Minimum": 1}},
        "compiler_spills": {"target_event_count": 0},
        "instrumented_gpu_interval_duration": {
            "profile": {"count": 500, "median": 10}
        },
    }


class CounterComparisonTest(unittest.TestCase):
    def captures(self, medians: list[list[float]]):
        return [
            summary(values, capture_index=index)
            for index, values in enumerate(medians, start=1)
        ]

    def test_applies_frozen_pairing_and_repeatability_rule(self):
        result = compare(
            self.captures(
                [
                    [100.0, 100.0, 100.0, 0.0],
                    [80.0, 80.0, 97.0, 1.0],
                    [90.0, 120.0, 96.0, 2.0],
                    [100.0, 100.0, 100.0, 0.0],
                ]
            )
        )
        comparisons = {
            item["name"]: item for item in result["comparisons"]
        }

        self.assertEqual(
            comparisons["repeatable"]["classification"],
            "repeatable_difference",
        )
        self.assertEqual(
            comparisons["repeatable"][
                "pair_variant_over_baseline_ratios"
            ],
            [0.8, 0.9],
        )
        self.assertAlmostEqual(
            comparisons["repeatable"]["relative_change_percent"], -15.0
        )
        self.assertEqual(
            comparisons["inconsistent"]["classification"],
            "directionally_inconsistent",
        )
        self.assertEqual(
            comparisons["small"]["classification"],
            "below_material_threshold",
        )
        self.assertEqual(
            comparisons["zero"]["classification"],
            "ineligible_nonpositive_median",
        )

    def test_rejects_insufficient_profile_window_coverage(self):
        captures = self.captures([[1.0, 1.0, 1.0, 1.0]] * 4)
        captures[2] = summary(
            [1.0, 1.0, 1.0, 1.0],
            capture_index=3,
            sample_span=0.79,
        )

        with self.assertRaisesRegex(ValueError, "do not span enough"):
            compare(captures)

    def test_rejects_swapped_summaries_instead_of_inverting_roles(self):
        captures = self.captures([[1.0, 1.0, 1.0, 1.0]] * 4)
        captures[0], captures[1] = captures[1], captures[0]

        with self.assertRaisesRegex(ValueError, "frozen ABBA order"):
            compare(captures)

    def test_rejects_mixed_repository_commits(self):
        captures = self.captures([[1.0, 1.0, 1.0, 1.0]] * 4)
        captures[2] = summary(
            [1.0, 1.0, 1.0, 1.0],
            capture_index=3,
            commit="f" * 40,
        )

        with self.assertRaisesRegex(ValueError, "same repository commit"):
            compare(captures)

    def test_rejects_mixed_gpu_devices(self):
        captures = self.captures([[1.0, 1.0, 1.0, 1.0]] * 4)
        captures[3] = summary(
            [1.0, 1.0, 1.0, 1.0],
            capture_index=4,
            device="Apple Other GPU",
        )

        with self.assertRaisesRegex(ValueError, "same GPU device"):
            compare(captures)

    def test_rejects_different_binaries_for_the_same_role(self):
        captures = self.captures([[1.0, 1.0, 1.0, 1.0]] * 4)
        captures[3] = summary(
            [1.0, 1.0, 1.0, 1.0],
            capture_index=4,
            binary_sha256="f" * 64,
        )

        with self.assertRaisesRegex(ValueError, "baseline captures"):
            compare(captures)

    def test_rejects_stale_provenance_for_the_same_role(self):
        captures = self.captures([[1.0, 1.0, 1.0, 1.0]] * 4)
        captures[3] = summary(
            [1.0, 1.0, 1.0, 1.0],
            capture_index=4,
            provenance_sha256="f" * 64,
        )

        with self.assertRaisesRegex(ValueError, "same provenance"):
            compare(captures)

    def test_rejects_summary_not_bound_to_its_receipt_input(self):
        captures = self.captures([[1.0, 1.0, 1.0, 1.0]] * 4)
        captures[1]["inputs"][0]["sha256"] = "f" * 64

        with self.assertRaisesRegex(ValueError, "analyzer receipt input"):
            compare(captures)


if __name__ == "__main__":
    unittest.main()
