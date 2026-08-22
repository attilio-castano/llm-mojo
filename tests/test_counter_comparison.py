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


def summary(medians: list[float], *, sample_span: float = 0.9):
    return {
        "analysis": "rmsnorm_metal_trace",
        "trace": {
            "end_reason": "Target app exited",
            "template": "LLM_Mojo_Metal_Limiters",
            "recording_duration_seconds": 1.0,
            "metal_application_gpu_settings": [
                "Counter Set: Performance Limiters",
                "Shader Timeline: Disabled",
                "Induced GPU Performance State: Default",
            ],
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
    def test_applies_frozen_pairing_and_repeatability_rule(self):
        result = compare(
            [
                summary([100.0, 100.0, 100.0, 0.0]),
                summary([80.0, 80.0, 97.0, 1.0]),
                summary([90.0, 120.0, 96.0, 2.0]),
                summary([100.0, 100.0, 100.0, 0.0]),
            ]
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
        captures = [summary([1.0, 1.0, 1.0, 1.0]) for _ in range(4)]
        captures[2] = summary(
            [1.0, 1.0, 1.0, 1.0], sample_span=0.79
        )

        with self.assertRaisesRegex(ValueError, "do not span enough"):
            compare(captures)


if __name__ == "__main__":
    unittest.main()
