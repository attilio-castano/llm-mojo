import unittest

from benchmarks.run_linear_prefill import (
    BENCHMARK_RESULTS_BEGIN,
    BENCHMARK_RESULTS_END,
    EXPECTED_REPETITIONS,
    IMPLEMENTATION,
    INPUT_FEATURES,
    WORKLOAD_ORDER,
    WORKLOADS,
    benchmark_command,
    parse_samples,
    summarize,
)


def synthetic_output(*, reverse: bool = False) -> str:
    lines = [
        "implementation: enqueue_linear_apple_gpu",
        "device: Apple Test GPU",
        "api: metal",
        BENCHMARK_RESULTS_BEGIN,
        "name,met (ms),iters",
    ]
    workloads = reversed(WORKLOAD_ORDER) if reverse else WORKLOAD_ORDER
    for workload in workloads:
        for repetition in range(EXPECTED_REPETITIONS):
            value = 0.01 + repetition / 1_000_000.0
            lines.append(
                "linear_prefill_rowwise_apple_gpu/input_id:"
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
                sample["implementation"] == IMPLEMENTATION["id"]
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


if __name__ == "__main__":
    unittest.main()
