import unittest

from benchmarks.run_linear import BENCHMARK_RESULTS_BEGIN, BENCHMARK_RESULTS_END
from benchmarks.run_linear_prefill_register_rowwise import (
    BLOCK_IMPLEMENTATION_ORDERS,
    BLOCK_ORDERS,
    EXPECTED_REPETITIONS,
    REGISTER_IMPLEMENTATION,
    ROWWISE_IMPLEMENTATION,
    WORKLOAD_ORDER,
    benchmark_command,
    parse_samples,
    summarize,
)


def synthetic_output(
    *, block_number: int, register_multiplier: float = 1.0
) -> str:
    reverse = BLOCK_ORDERS[block_number - 1] == "descending"
    implementation_order = BLOCK_IMPLEMENTATION_ORDERS[block_number - 1]
    emitted_order = (
        "register_2x2,rowwise"
        if implementation_order == "register_2x2_then_rowwise"
        else "rowwise,register_2x2"
    )
    lines = [
        "implementation: enqueue_linear_apple_gpu",
        (
            "comparison implementation: "
            "enqueue_linear_prefill_register_2x2_apple_gpu"
        ),
        "device: Apple Test GPU",
        "api: metal",
        f"implementation order: {emitted_order}",
        BENCHMARK_RESULTS_BEGIN,
        "name,met (ms),iters",
    ]
    implementations = (
        ("register_2x2", "rowwise")
        if implementation_order == "register_2x2_then_rowwise"
        else ("rowwise", "register_2x2")
    )
    workloads = reversed(WORKLOAD_ORDER) if reverse else WORKLOAD_ORDER
    for workload_index, workload in enumerate(workloads):
        base = 1.0 + workload_index / 100.0
        for implementation in implementations:
            multiplier = (
                register_multiplier
                if implementation == "register_2x2"
                else 1.0
            )
            for repetition in range(EXPECTED_REPETITIONS):
                value = base * multiplier + repetition / 1_000_000.0
                lines.append(
                    f"linear_prefill_{implementation}_apple_gpu/input_id:"
                    f"{workload},{value},20"
                )
    lines.append(BENCHMARK_RESULTS_END)
    return "\n".join(lines)


def parsed_blocks(*, register_multiplier: float) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for block_number in range(1, 5):
        _, block_samples = parse_samples(
            synthetic_output(
                block_number=block_number,
                register_multiplier=register_multiplier,
            ),
            experiment_id="EXP-0012",
            run_id="EXP-0012-RUN-TEST",
            block_id=f"block-{block_number:02d}",
            block_order=BLOCK_ORDERS[block_number - 1],
            implementation_order=BLOCK_IMPLEMENTATION_ORDERS[
                block_number - 1
            ],
        )
        samples.extend(block_samples)
    return samples


class PrefillRegisterRowwiseParserTest(unittest.TestCase):
    def test_records_both_arithmetic_mappings_and_source_costs(self):
        _, samples = parse_samples(
            synthetic_output(block_number=1),
            experiment_id="EXP-0012",
            run_id="RUN-001",
            block_id="block-01",
            block_order="ascending",
            implementation_order="rowwise_then_register_2x2",
        )

        self.assertEqual(
            len(samples), len(WORKLOAD_ORDER) * 2 * EXPECTED_REPETITIONS
        )
        self.assertEqual(
            len({sample["sample_id"] for sample in samples}), len(samples)
        )
        rowwise = next(
            sample
            for sample in samples
            if sample["implementation"] == ROWWISE_IMPLEMENTATION["id"]
        )
        register = next(
            sample
            for sample in samples
            if sample["implementation"] == REGISTER_IMPLEMENTATION["id"]
        )
        self.assertEqual(rowwise["threads_per_threadgroup"], 128)
        self.assertEqual(register["threads_per_threadgroup"], 32)
        self.assertEqual(rowwise["simd_groups_per_threadgroup"], 4)
        self.assertEqual(register["simd_groups_per_threadgroup"], 1)
        self.assertEqual(rowwise["outputs_per_simd_group"], 1)
        self.assertEqual(register["outputs_per_simd_group"], 128)
        self.assertEqual(rowwise["k_elements_per_lane"], 28)
        self.assertEqual(register["k_elements_per_lane"], 896)
        self.assertEqual(rowwise["simd_group_reductions_per_output"], 1)
        self.assertEqual(register["simd_group_reductions_per_output"], 0)
        self.assertEqual(rowwise["threadgroup_operand_scratch_bytes"], 0)
        self.assertEqual(register["threadgroup_operand_scratch_bytes"], 0)
        self.assertLess(
            register["program_requested_traffic_bytes"],
            rowwise["program_requested_traffic_bytes"],
        )

    def test_accepts_reversed_candidate_first_block(self):
        _, samples = parse_samples(
            synthetic_output(block_number=2),
            experiment_id="EXP-0012",
            run_id="RUN-001",
            block_id="block-02",
            block_order="descending",
            implementation_order="register_2x2_then_rowwise",
        )

        self.assertEqual(samples[0]["workload"], WORKLOAD_ORDER[-1])
        self.assertEqual(
            samples[0]["implementation"], REGISTER_IMPLEMENTATION["id"]
        )

    def test_rejects_an_implementation_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "implementation order"):
            parse_samples(
                synthetic_output(block_number=2),
                experiment_id="EXP-0012",
                run_id="RUN-001",
                block_id="block-02",
                block_order="descending",
                implementation_order="rowwise_then_register_2x2",
            )


class PrefillRegisterRowwiseProtocolTest(unittest.TestCase):
    def test_commands_select_abba_and_workload_order(self):
        first = benchmark_command(block_number=1)
        second = benchmark_command(block_number=2)
        fourth = benchmark_command(block_number=4)

        self.assertIn(
            "LINEAR_PREFILL_REGISTER_ROWWISE_COMPARISON=true", first
        )
        self.assertNotIn("LINEAR_PREFILL_REGISTER_FIRST=true", first)
        self.assertIn("LINEAR_PREFILL_REGISTER_FIRST=true", second)
        self.assertIn("LINEAR_PREFILL_BK_SWEEP_REVERSE=true", second)
        self.assertNotIn("LINEAR_PREFILL_BK_SWEEP_REVERSE=true", fourth)

    def test_summary_advances_a_material_large_m_improvement(self):
        result = summarize(parsed_blocks(register_multiplier=0.9))

        self.assertTrue(result["large_prefill_rule"]["qualified"])
        self.assertEqual(
            result["crossover"][
                "smallest_no_larger_material_regression_row"
            ],
            8,
        )
        self.assertEqual(
            result["crossover"][
                "smallest_all_larger_material_improvement_row"
            ],
            8,
        )
        self.assertEqual(
            result["timing_decision"],
            "advance_register_2x2_as_manual_prefill_candidate",
        )
        self.assertEqual(result["production_dispatch_decision"], "none")

    def test_summary_rejects_a_material_large_m_regression(self):
        result = summarize(parsed_blocks(register_multiplier=1.1))

        self.assertTrue(
            result["large_prefill_rule"]["rejected_at_required_rows"]
        )
        self.assertIsNone(
            result["crossover"][
                "smallest_no_larger_material_regression_row"
            ]
        )
        self.assertEqual(
            result["timing_decision"],
            "reject_register_2x2_against_public_rowwise",
        )
        self.assertEqual(result["production_dispatch_decision"], "none")


if __name__ == "__main__":
    unittest.main()
