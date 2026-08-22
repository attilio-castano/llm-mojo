import unittest

from benchmarks.run_rms_norm import (
    BASELINE_IMPLEMENTATION,
    BENCHMARK_RESULTS_BEGIN,
    BENCHMARK_RESULTS_END,
    HIDDEN_SIZE,
    VARIANT_IMPLEMENTATION,
    WORKLOAD_ROWS,
    benchmark_command,
    parse_samples,
    summarize,
)


def paired_output(*, variant_first: bool = False) -> str:
    lines = [
        "implementation: enqueue_rms_norm_apple_gpu",
        (
            "comparison implementation: "
            "enqueue_rms_norm_apple_gpu_simdgroup"
        ),
        "device: Apple Test GPU",
        "api: metal",
        BENCHMARK_RESULTS_BEGIN,
        "name,met (ms),iters",
    ]
    names = (
        ["rms_norm_apple_gpu_simdgroup", "rms_norm_apple_gpu"]
        if variant_first
        else ["rms_norm_apple_gpu", "rms_norm_apple_gpu_simdgroup"]
    )
    for rows in WORKLOAD_ROWS:
        for name in names:
            for repetition in range(10):
                lines.append(
                    f"{name}/input_id:rows={rows} hidden={HIDDEN_SIZE},"
                    f"{0.01 + repetition / 1000000.0},1000"
                )
    lines.append(BENCHMARK_RESULTS_END)
    return "\n".join(lines)


def synthetic_samples() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for block_number in range(1, 5):
        for rows in WORKLOAD_ROWS:
            if rows == 1:
                ratio = 0.90
            elif rows == 4:
                ratio = (0.80, 0.80, 1.10, 1.10)[block_number - 1]
            elif rows == 128:
                ratio = 1.10
            else:
                ratio = 1.00
            for implementation, value in (
                (BASELINE_IMPLEMENTATION, 1.0),
                (VARIANT_IMPLEMENTATION, ratio),
            ):
                for repetition in range(1, 11):
                    samples.append(
                        {
                            "valid": True,
                            "workload": f"r{rows}-h{HIDDEN_SIZE}",
                            "rows": rows,
                            "hidden_size": HIDDEN_SIZE,
                            "block_id": f"block-{block_number:02d}",
                            "implementation": implementation["id"],
                            "implementation_entrypoint": implementation[
                                "entrypoint"
                            ],
                            "repetition": repetition,
                            "value": value,
                        }
                    )
    return samples


class ParsePairedSamplesTest(unittest.TestCase):
    def test_accepts_frozen_pair_order_and_assigns_unique_ids(self):
        identity, samples = parse_samples(
            paired_output(),
            experiment_id="EXP-0002",
            run_id="RUN-001",
            block_id="block-01",
            block_order="ascending",
            implementation_order="baseline_then_variant",
            variant_comparison=True,
        )

        self.assertEqual(identity["device"], "Apple Test GPU")
        self.assertEqual(len(samples), 140)
        self.assertEqual(len({sample["sample_id"] for sample in samples}), 140)
        self.assertEqual(
            samples[0]["implementation"], BASELINE_IMPLEMENTATION["id"]
        )
        self.assertEqual(
            samples[10]["implementation"], VARIANT_IMPLEMENTATION["id"]
        )

    def test_rejects_an_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "workload order mismatch"):
            parse_samples(
                paired_output(variant_first=True),
                experiment_id="EXP-0002",
                run_id="RUN-001",
                block_id="block-01",
                block_order="ascending",
                implementation_order="baseline_then_variant",
                variant_comparison=True,
            )


class PairedSummaryTest(unittest.TestCase):
    def test_applies_ratio_and_direction_rules(self):
        result = summarize(synthetic_samples(), variant_comparison=True)
        paired = result["paired_comparison"]
        by_rows = {row["rows"]: row for row in paired["workloads"]}

        self.assertEqual(by_rows[1]["classification"], "material_improvement")
        self.assertEqual(by_rows[4]["classification"], "inconclusive")
        self.assertEqual(by_rows[128]["classification"], "material_regression")
        self.assertEqual(paired["timing_decision"], "retain_baseline")
        self.assertEqual(paired["relevant_material_improvements"], ["r1-h896"])
        self.assertEqual(paired["relevant_material_regressions"], ["r128-h896"])

    def test_comparison_command_selects_both_compile_time_modes(self):
        command = benchmark_command(
            reverse=True, variant_comparison=True, variant_first=True
        )

        self.assertIn("RMS_NORM_BENCH_REVERSE=true", command)
        self.assertIn("RMS_NORM_BENCH_VARIANT_COMPARISON=true", command)
        self.assertIn("RMS_NORM_BENCH_VARIANT_FIRST=true", command)


if __name__ == "__main__":
    unittest.main()
