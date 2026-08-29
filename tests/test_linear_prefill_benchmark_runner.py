import unittest

from benchmarks.run_linear_prefill import (
    BASELINE_IMPLEMENTATION,
    BENCHMARK_RESULTS_BEGIN,
    BENCHMARK_RESULTS_END,
    DIRECT_IMPLEMENTATION,
    EXPECTED_REPETITIONS,
    INPUT_FEATURES,
    WORKLOAD_ORDER,
    WORKLOADS,
    benchmark_command,
    parse_samples,
    summarize,
)


def synthetic_output(
    *,
    reverse: bool = False,
    direct_comparison: bool = False,
    direct_first: bool = False,
) -> str:
    lines = [
        "implementation: enqueue_linear_apple_gpu",
        "device: Apple Test GPU",
        "api: metal",
    ]
    if direct_comparison:
        lines.insert(
            1,
            "comparison implementation: "
            "enqueue_linear_prefill_direct_apple_gpu",
        )
    lines.extend([BENCHMARK_RESULTS_BEGIN, "name,met (ms),iters"])
    workloads = reversed(WORKLOAD_ORDER) if reverse else WORKLOAD_ORDER
    implementations = (
        ("direct", "rowwise") if direct_first else ("rowwise", "direct")
    )
    for workload in workloads:
        for implementation in implementations if direct_comparison else ("rowwise",):
            for repetition in range(EXPECTED_REPETITIONS):
                value = 0.01 + repetition / 1_000_000.0
                lines.append(
                    f"linear_prefill_{implementation}_apple_gpu/input_id:"
                    f"{workload},{value},20"
                )
    lines.append(BENCHMARK_RESULTS_END)
    return "\n".join(lines)


class PrefillProjectionBenchmarkParserTest(unittest.TestCase):
    def test_records_all_shapes_with_unique_sample_ids(self):
        identity, samples = parse_samples(
            synthetic_output(),
            experiment_id="EXP-0007",
            run_id="RUN-001",
            block_id="block-01",
            block_order="ascending",
        )

        self.assertEqual(identity["device"], "Apple Test GPU")
        self.assertEqual(
            len(samples), len(WORKLOAD_ORDER) * EXPECTED_REPETITIONS
        )
        self.assertEqual(len({sample["sample_id"] for sample in samples}), len(samples))
        self.assertTrue(
            all(
                sample["implementation"] == BASELINE_IMPLEMENTATION["id"]
                for sample in samples
            )
        )

    def test_accepts_the_reversed_workload_order(self):
        _, samples = parse_samples(
            synthetic_output(reverse=True),
            experiment_id="EXP-0007",
            run_id="RUN-001",
            block_id="block-02",
            block_order="descending",
        )

        self.assertEqual(samples[0]["workload"], WORKLOAD_ORDER[-1])

    def test_rejects_an_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "workload order mismatch"):
            parse_samples(
                synthetic_output(reverse=True),
                experiment_id="EXP-0007",
                run_id="RUN-001",
                block_id="block-01",
                block_order="ascending",
            )


class PrefillProjectionBenchmarkSummaryTest(unittest.TestCase):
    def test_reports_shape_aware_compute_metrics(self):
        _, samples = parse_samples(
            synthetic_output(),
            experiment_id="EXP-0007",
            run_id="RUN-001",
            block_id="block-01",
            block_order="ascending",
        )
        result = summarize(samples)

        self.assertEqual(len(result["workloads"]), len(WORKLOAD_ORDER))
        first = result["workloads"][0]
        specification = WORKLOADS[WORKLOAD_ORDER[0]]
        expected = (
            2
            * int(specification["macs"])
            / (first["median_ms_per_workload_iteration"] * 1_000_000.0)
        )
        self.assertEqual(first["input_features"], INPUT_FEATURES)
        self.assertAlmostEqual(first["effective_gflop_per_second"], expected)

    def test_reverse_command_selects_the_compile_time_order(self):
        command = benchmark_command(reverse=True)

        self.assertIn("LINEAR_PREFILL_BENCH_REVERSE=true", command)

    def test_direct_comparison_parses_abba_samples_and_reports_ratios(self):
        samples = []
        for block_number, (reverse, direct_first) in enumerate(
            ((False, False), (True, True), (True, True), (False, False)),
            start=1,
        ):
            _, block_samples = parse_samples(
                synthetic_output(
                    reverse=reverse,
                    direct_comparison=True,
                    direct_first=direct_first,
                ),
                experiment_id="EXP-0007",
                run_id="RUN-001",
                block_id=f"block-{block_number:02d}",
                block_order="descending" if reverse else "ascending",
                implementation_order=(
                    "variant_then_baseline"
                    if direct_first
                    else "baseline_then_variant"
                ),
                direct_comparison=True,
            )
            samples.extend(block_samples)

        result = summarize(samples, direct_comparison=True)
        paired = result["paired_comparison"]
        self.assertEqual(len(paired["workloads"]), len(WORKLOAD_ORDER))
        self.assertEqual(
            paired["timing_decision"], "control_only_no_dispatch_decision"
        )
        self.assertTrue(
            all(
                item["classification"] == "inconclusive"
                for item in paired["workloads"]
            )
        )

    def test_direct_comparison_command_selects_both_compile_time_modes(self):
        command = benchmark_command(
            reverse=True, direct_comparison=True, direct_first=True
        )

        self.assertIn("LINEAR_PREFILL_BENCH_DIRECT_COMPARISON=true", command)
        self.assertIn("LINEAR_PREFILL_BENCH_DIRECT_FIRST=true", command)
        self.assertEqual(
            DIRECT_IMPLEMENTATION["id"], "apple_gpu_prefill_direct_8x16_v0"
        )


if __name__ == "__main__":
    unittest.main()
