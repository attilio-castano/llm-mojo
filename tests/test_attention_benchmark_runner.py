import unittest

from benchmarks.run_attention import (
    BENCHMARK_RESULTS_BEGIN,
    BENCHMARK_RESULTS_END,
    EXPECTED_REPETITIONS,
    IMPLEMENTATION,
    STAGE_IMPLEMENTATIONS,
    STAGE_ORDER,
    WORKLOAD_ORDER,
    benchmark_command,
    parse_samples,
    stage_metrics,
    summarize,
    workload_metrics,
)


def synthetic_output(
    *,
    reverse: bool = False,
    stage_attribution: bool = False,
    reverse_stages: bool = False,
    device: str = "Apple Test GPU",
    api: str = "metal",
) -> str:
    lines = [
        "implementation: enqueue_grouped_query_attention_apple_gpu",
        f"device: {device}",
        f"api: {api}",
        f"mode: {'stage-attribution' if stage_attribution else 'end-to-end'}",
        BENCHMARK_RESULTS_BEGIN,
        "name,met (ms),iters",
    ]
    if stage_attribution:
        lines.insert(4, f"stage order: {'reverse' if reverse_stages else 'forward'}")
    workloads = list(reversed(WORKLOAD_ORDER)) if reverse else WORKLOAD_ORDER
    stages = list(reversed(STAGE_ORDER)) if reverse_stages else STAGE_ORDER
    if not stage_attribution:
        stages = ("end-to-end",)
    for workload in workloads:
        for stage in stages:
            benchmark_name = STAGE_IMPLEMENTATIONS[stage]["benchmark_name"]
            for repetition in range(EXPECTED_REPETITIONS):
                value = 0.01 + repetition / 1_000_000.0
                lines.append(f"{benchmark_name}/input_id:{workload},{value},100")
    lines.append(BENCHMARK_RESULTS_END)
    return "\n".join(lines)


class AttentionWorkloadAccountingTest(unittest.TestCase):
    def test_decode_t1_accounting(self):
        metrics = workload_metrics(1, 1)

        self.assertEqual(metrics["visible_scores"], 14)
        self.assertEqual(metrics["materialized_scores"], 14)
        self.assertEqual(metrics["output_elements"], 896)
        self.assertEqual(metrics["total_macs"], 1_792)
        self.assertEqual(metrics["scratch_bytes"], 28)
        self.assertEqual(metrics["allocated_footprint_bytes"], 4_124)
        self.assertEqual(metrics["program_requested_traffic_bytes"], 9_100)

        qk = stage_metrics(metrics, "qk")
        softmax = stage_metrics(metrics, "softmax")
        pv = stage_metrics(metrics, "pv")
        self.assertEqual(qk["program_requested_traffic_bytes"], 3_612)
        self.assertEqual(softmax["program_requested_traffic_bytes"], 112)
        self.assertEqual(pv["program_requested_traffic_bytes"], 5_376)
        self.assertEqual(
            qk["program_requested_traffic_bytes"]
            + softmax["program_requested_traffic_bytes"]
            + pv["program_requested_traffic_bytes"],
            metrics["program_requested_traffic_bytes"],
        )

    def test_rejects_query_rows_beyond_active_key_value_rows(self):
        with self.assertRaisesRegex(ValueError, "invalid attention workload"):
            workload_metrics(2, 1)


class AttentionBenchmarkParserTest(unittest.TestCase):
    def test_assigns_unique_samples_for_the_full_ascending_sweep(self):
        identity, samples = parse_samples(
            synthetic_output(),
            experiment_id="exploration",
            run_id="RUN-001",
            block_id="block-01",
            block_order="ascending",
        )

        self.assertEqual(identity["device"], "Apple Test GPU")
        self.assertEqual(len(samples), len(WORKLOAD_ORDER) * EXPECTED_REPETITIONS)
        self.assertEqual(len({sample["sample_id"] for sample in samples}), len(samples))
        self.assertTrue(
            all(sample["implementation"] == IMPLEMENTATION["id"] for sample in samples)
        )

    def test_accepts_the_exact_reversed_sweep(self):
        _, samples = parse_samples(
            synthetic_output(reverse=True),
            experiment_id="exploration",
            run_id="RUN-002",
            block_id="block-02",
            block_order="descending",
        )

        self.assertEqual(samples[0]["workload"], WORKLOAD_ORDER[-1])

    def test_parses_the_full_stage_attribution_sweep(self):
        identity, samples = parse_samples(
            synthetic_output(stage_attribution=True),
            experiment_id="exploration",
            run_id="RUN-STAGES-001",
            block_id="block-01",
            block_order="ascending",
            stage_attribution=True,
            block_stage_order="forward",
        )

        self.assertEqual(identity["mode"], "stage-attribution")
        self.assertEqual(
            len(samples),
            len(WORKLOAD_ORDER) * len(STAGE_ORDER) * EXPECTED_REPETITIONS,
        )
        self.assertEqual(len({sample["sample_id"] for sample in samples}), len(samples))
        self.assertEqual(
            [samples[index * EXPECTED_REPETITIONS]["stage"] for index in range(4)],
            list(STAGE_ORDER),
        )
        self.assertEqual(samples[0]["dispatches_per_iteration"], 3)
        self.assertEqual(samples[EXPECTED_REPETITIONS]["dispatches_per_iteration"], 1)

    def test_accepts_reversed_workloads_and_stages(self):
        _, samples = parse_samples(
            synthetic_output(
                reverse=True,
                stage_attribution=True,
                reverse_stages=True,
            ),
            experiment_id="exploration",
            run_id="RUN-STAGES-002",
            block_id="block-02",
            block_order="descending",
            stage_attribution=True,
            block_stage_order="reverse",
        )

        self.assertEqual(samples[0]["workload"], WORKLOAD_ORDER[-1])
        self.assertEqual(samples[0]["stage"], STAGE_ORDER[-1])

    def test_rejects_a_stage_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "stage order mismatch"):
            parse_samples(
                synthetic_output(
                    stage_attribution=True,
                    reverse_stages=True,
                ),
                experiment_id="exploration",
                run_id="RUN-STAGES-003",
                block_id="block-01",
                block_order="ascending",
                stage_attribution=True,
                block_stage_order="forward",
            )

    def test_rejects_a_workload_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "workload order mismatch"):
            parse_samples(
                synthetic_output(reverse=True),
                experiment_id="exploration",
                run_id="RUN-003",
                block_id="block-01",
                block_order="ascending",
            )

    def test_rejects_a_non_metal_identity(self):
        with self.assertRaisesRegex(ValueError, "Apple GPU Metal"):
            parse_samples(
                synthetic_output(api="cpu"),
                experiment_id="exploration",
                run_id="RUN-004",
                block_id="block-01",
                block_order="ascending",
            )

    def test_summary_retains_the_primary_metric_and_scope(self):
        _, samples = parse_samples(
            synthetic_output(),
            experiment_id="exploration",
            run_id="RUN-005",
            block_id="block-01",
            block_order="ascending",
        )

        result = summarize(samples)
        first = result["workloads"][0]
        self.assertEqual(
            result["statistics"]["primary_metric"],
            "synchronized milliseconds per attention call",
        )
        self.assertEqual(first["workload"], WORKLOAD_ORDER[0])
        self.assertEqual(first["count"], EXPECTED_REPETITIONS)
        self.assertEqual(first["total_macs"], 1_792)
        self.assertEqual(first["scratch_bytes"], 28)
        self.assertIn("no 24-layer cache pressure", result["scope"])

    def test_stage_summary_uses_the_isolated_sum_for_fractions(self):
        _, samples = parse_samples(
            synthetic_output(stage_attribution=True),
            experiment_id="exploration",
            run_id="RUN-STAGES-004",
            block_id="block-01",
            block_order="ascending",
            stage_attribution=True,
            block_stage_order="forward",
        )

        result = summarize(samples, stage_attribution=True)
        first = result["workloads"][0]
        self.assertEqual(first["workload"], WORKLOAD_ORDER[0])
        self.assertAlmostEqual(sum(first["isolated_stage_fractions"].values()), 1.0)
        self.assertAlmostEqual(first["isolated_sum_over_end_to_end_ratio"], 3.0)
        self.assertIsNone(first["stages"]["softmax"]["effective_gmac_per_second"])

    def test_reverse_command_selects_the_compile_time_order(self):
        command = benchmark_command(reverse=True)

        self.assertIn("ATTENTION_BENCH_REVERSE=true", command)
        self.assertEqual(command[-1], "benchmarks/attention.mojo")

    def test_stage_command_selects_attribution_and_reverse_stage_order(self):
        command = benchmark_command(
            reverse=True,
            stage_attribution=True,
            reverse_stages=True,
        )

        self.assertIn("ATTENTION_BENCH_STAGE_ATTRIBUTION=true", command)
        self.assertIn("ATTENTION_BENCH_REVERSE_STAGES=true", command)
        self.assertEqual(command[-1], "benchmarks/attention.mojo")


if __name__ == "__main__":
    unittest.main()
