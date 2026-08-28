import unittest

from benchmarks.run_linear import (
    BASELINE_IMPLEMENTATION,
    BENCHMARK_RESULTS_BEGIN,
    BENCHMARK_RESULTS_END,
    PRIMARY_WORKLOAD,
    QKV_FUSION_IMPLEMENTATION,
    QKV_FUSION_WORKLOAD_ORDER,
    VARIANT_IMPLEMENTATION,
    WORKLOADS,
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
                            "dispatches_per_iteration": WORKLOADS[workload][
                                "dispatches"
                            ],
                        }
                    )
    return samples


def fusion_output(*, candidate_first: bool, reverse: bool) -> str:
    lines = [
        "implementation: enqueue_linear_apple_gpu",
        "comparison implementation: enqueue_linear_apple_gpu",
        "device: Apple Test GPU",
        "api: metal",
        BENCHMARK_RESULTS_BEGIN,
        "name,met (ms),iters",
    ]
    workloads = (
        list(reversed(QKV_FUSION_WORKLOAD_ORDER))
        if reverse
        else QKV_FUSION_WORKLOAD_ORDER
    )
    implementations = (
        (QKV_FUSION_IMPLEMENTATION, BASELINE_IMPLEMENTATION)
        if candidate_first
        else (BASELINE_IMPLEMENTATION, QKV_FUSION_IMPLEMENTATION)
    )
    for workload in workloads:
        for implementation in implementations:
            fused = implementation == QKV_FUSION_IMPLEMENTATION
            ring = workload == PRIMARY_WORKLOAD
            base = (
                "linear_decode_qkv3_ring24_apple_gpu"
                if ring
                else "linear_decode_qkv3_apple_gpu"
            )
            name = base + ("_fused" if fused else "")
            for repetition in range(10):
                value = 0.01 + repetition / 1_000_000.0
                lines.append(f"{name}/input_id:{workload},{value},100")
    lines.append(BENCHMARK_RESULTS_END)
    return "\n".join(lines)


def fusion_samples() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for block_number in range(1, 5):
        for workload in QKV_FUSION_WORKLOAD_ORDER:
            ratio = 0.90 if workload == PRIMARY_WORKLOAD else 1.00
            for implementation, value in (
                (BASELINE_IMPLEMENTATION, 1.0),
                (QKV_FUSION_IMPLEMENTATION, ratio),
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
                            "dispatches_per_iteration": (
                                WORKLOADS[workload]["layers"]
                                if implementation == QKV_FUSION_IMPLEMENTATION
                                else WORKLOADS[workload]["dispatches"]
                            ),
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

    def test_fusion_mode_records_candidate_dispatch_reduction(self):
        _, samples = parse_samples(
            fusion_output(candidate_first=True, reverse=True),
            experiment_id="EXP-0006",
            run_id="RUN-001",
            block_id="block-02",
            block_order="descending",
            implementation_order="variant_then_baseline",
            variant_comparison=False,
            qkv_fusion_comparison=True,
        )

        self.assertEqual(len(samples), 40)
        fused = [
            sample
            for sample in samples
            if sample["implementation"] == QKV_FUSION_IMPLEMENTATION["id"]
        ]
        dispatches = {
            sample["workload"]: sample["dispatches_per_iteration"]
            for sample in fused
        }
        self.assertEqual(dispatches["qkv3-hot-m1-k896-n1152"], 1)
        self.assertEqual(dispatches[PRIMARY_WORKLOAD], 24)


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

    def test_fusion_primary_win_promotes_single_enqueue(self):
        result = summarize(
            fusion_samples(),
            variant_comparison=False,
            qkv_fusion_comparison=True,
        )

        comparison = result["paired_comparison"]
        self.assertTrue(comparison["primary_rule_passed"])
        self.assertEqual(comparison["secondary_material_regressions"], [])
        self.assertEqual(
            comparison["timing_decision"],
            "promote_packed_qkv_single_enqueue",
        )

    def test_fusion_command_has_a_distinct_compile_time_mode(self):
        command = benchmark_command(
            reverse=False,
            variant_comparison=False,
            variant_first=False,
            qkv_fusion_comparison=True,
        )

        self.assertIn("LINEAR_BENCH_QKV_FUSION_COMPARISON=true", command)
        self.assertNotIn("LINEAR_BENCH_VARIANT_COMPARISON=true", command)


if __name__ == "__main__":
    unittest.main()
