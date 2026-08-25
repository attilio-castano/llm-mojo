import unittest

from benchmarks.run_linear import (
    BASELINE_IMPLEMENTATION,
    BENCHMARK_RESULTS_BEGIN,
    BENCHMARK_RESULTS_END,
    PRIMARY_WORKLOAD,
    VARIANT_IMPLEMENTATION,
    WORKLOAD_ORDER,
    benchmark_command,
    parse_samples,
    summarize,
)


def benchmark_name(workload: str, *, variant: bool) -> str:
    if workload == PRIMARY_WORKLOAD:
        base = "linear_decode_qkv3_ring24_apple_gpu"
    elif workload.startswith("qkv3-hot"):
        base = "linear_decode_qkv3_apple_gpu"
    else:
        base = "linear_decode_apple_gpu"
    return base + ("_two_output" if variant else "")


def synthetic_output(
    *, comparison: bool, variant_first: bool = False, reverse: bool = False
) -> str:
    lines = [
        "implementation: enqueue_linear_apple_gpu",
        "device: Apple Test GPU",
        "api: metal",
    ]
    if comparison:
        lines.insert(
            1,
            "comparison implementation: enqueue_linear_apple_gpu_two_output",
        )
    lines.extend([BENCHMARK_RESULTS_BEGIN, "name,met (ms),iters"])
    workloads = list(reversed(WORKLOAD_ORDER)) if reverse else WORKLOAD_ORDER
    variants = (True, False) if variant_first else (False, True)
    for workload in workloads:
        for variant in variants if comparison else (False,):
            for repetition in range(10):
                value = 0.01 + repetition / 1_000_000.0
                lines.append(
                    f"{benchmark_name(workload, variant=variant)}/"
                    f"input_id:{workload},{value},100"
                )
    lines.append(BENCHMARK_RESULTS_END)
    return "\n".join(lines)


def comparison_samples() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for block_number in range(1, 5):
        for workload in WORKLOAD_ORDER:
            ratio = 0.90 if workload == PRIMARY_WORKLOAD else 1.00
            for implementation, value in (
                (BASELINE_IMPLEMENTATION, 1.0),
                (VARIANT_IMPLEMENTATION, ratio),
            ):
                for repetition in range(1, 11):
                    samples.append(
                        {
                            "valid": True,
                            "workload": workload,
                            "block_id": f"block-{block_number:02d}",
                            "implementation": implementation["id"],
                            "implementation_entrypoint": implementation[
                                "entrypoint"
                            ],
                            "value": value,
                            "repetition": repetition,
                        }
                    )
    return samples


class ProjectionBenchmarkParserTest(unittest.TestCase):
    def test_baseline_mode_assigns_unique_sample_ids(self):
        identity, samples = parse_samples(
            synthetic_output(comparison=False),
            experiment_id="EXP-0004",
            run_id="RUN-001",
            block_id="block-01",
            block_order="ascending",
            implementation_order="baseline_only",
            variant_comparison=False,
        )

        self.assertEqual(identity["device"], "Apple Test GPU")
        self.assertEqual(len(samples), 40)
        self.assertEqual(len({sample["sample_id"] for sample in samples}), 40)
        self.assertTrue(
            all(
                sample["implementation"] == BASELINE_IMPLEMENTATION["id"]
                for sample in samples
            )
        )

    def test_accepts_reversed_candidate_first_order(self):
        _, samples = parse_samples(
            synthetic_output(
                comparison=True, variant_first=True, reverse=True
            ),
            experiment_id="EXP-0005",
            run_id="RUN-001",
            block_id="block-02",
            block_order="descending",
            implementation_order="variant_then_baseline",
            variant_comparison=True,
        )

        self.assertEqual(len(samples), 80)
        self.assertEqual(
            samples[0]["implementation"], VARIANT_IMPLEMENTATION["id"]
        )

    def test_rejects_an_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "workload order mismatch"):
            parse_samples(
                synthetic_output(comparison=True, variant_first=True),
                experiment_id="EXP-0005",
                run_id="RUN-001",
                block_id="block-01",
                block_order="ascending",
                implementation_order="baseline_then_variant",
                variant_comparison=True,
            )


class ProjectionBenchmarkDecisionTest(unittest.TestCase):
    def test_primary_win_without_secondary_regression_promotes_candidate(self):
        result = summarize(comparison_samples(), variant_comparison=True)
        comparison = result["paired_comparison"]

        self.assertTrue(comparison["primary_rule_passed"])
        self.assertEqual(comparison["secondary_material_regressions"], [])
        self.assertEqual(
            comparison["timing_decision"], "promote_two_output_m1"
        )

    def test_comparison_command_selects_all_compile_time_modes(self):
        command = benchmark_command(
            reverse=True, variant_comparison=True, variant_first=True
        )

        self.assertIn("LINEAR_BENCH_REVERSE=true", command)
        self.assertIn("LINEAR_BENCH_VARIANT_COMPARISON=true", command)
        self.assertIn("LINEAR_BENCH_VARIANT_FIRST=true", command)


if __name__ == "__main__":
    unittest.main()
