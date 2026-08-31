import unittest

from benchmarks.run_linear import BENCHMARK_RESULTS_BEGIN, BENCHMARK_RESULTS_END
from benchmarks.run_linear_prefill_register_2x2 import (
    BLOCK_IMPLEMENTATION_ORDERS,
    BLOCK_ORDERS,
    DIRECT_IMPLEMENTATION,
    EXPECTED_REPETITIONS,
    REGISTER_IMPLEMENTATION,
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
        "register_2x2,direct"
        if implementation_order == "register_2x2_then_direct"
        else "direct,register_2x2"
    )
    lines = [
        "implementation: enqueue_linear_prefill_direct_apple_gpu",
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
        ("register_2x2", "direct")
        if implementation_order == "register_2x2_then_direct"
        else ("direct", "register_2x2")
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
            experiment_id="EXP-0011",
            run_id="EXP-0011-RUN-TEST",
            block_id=f"block-{block_number:02d}",
            block_order=BLOCK_ORDERS[block_number - 1],
            implementation_order=BLOCK_IMPLEMENTATION_ORDERS[block_number - 1],
        )
        samples.extend(block_samples)
    return samples


class PrefillRegister2x2ParserTest(unittest.TestCase):
    def test_records_both_ownership_mappings_and_source_costs(self):
        _, samples = parse_samples(
            synthetic_output(block_number=1),
            experiment_id="EXP-0011",
            run_id="RUN-001",
            block_id="block-01",
            block_order="ascending",
            implementation_order="direct_then_register_2x2",
        )

        self.assertEqual(
            len(samples), len(WORKLOAD_ORDER) * 2 * EXPECTED_REPETITIONS
        )
        self.assertEqual(
            len({sample["sample_id"] for sample in samples}), len(samples)
        )
        direct = next(
            sample
            for sample in samples
            if sample["implementation"] == DIRECT_IMPLEMENTATION["id"]
        )
        register = next(
            sample
            for sample in samples
            if sample["implementation"] == REGISTER_IMPLEMENTATION["id"]
        )
        self.assertEqual(direct["threads_per_threadgroup"], 128)
        self.assertEqual(register["threads_per_threadgroup"], 32)
        self.assertEqual(direct["outputs_per_thread"], 1)
        self.assertEqual(register["outputs_per_thread"], 4)
        self.assertEqual(direct["fp32_accumulators_per_thread"], 1)
        self.assertEqual(register["fp32_accumulators_per_thread"], 4)
        self.assertEqual(direct["threadgroup_operand_scratch_bytes"], 0)
        self.assertEqual(register["threadgroup_operand_scratch_bytes"], 0)
        self.assertEqual(direct["barriers_per_dispatch"], 0)
        self.assertEqual(register["barriers_per_dispatch"], 0)
        self.assertLess(
            register["program_requested_traffic_bytes"],
            direct["program_requested_traffic_bytes"],
        )

    def test_accepts_reversed_candidate_first_block(self):
        _, samples = parse_samples(
            synthetic_output(block_number=2),
            experiment_id="EXP-0011",
            run_id="RUN-001",
            block_id="block-02",
            block_order="descending",
            implementation_order="register_2x2_then_direct",
        )

        self.assertEqual(samples[0]["workload"], WORKLOAD_ORDER[-1])
        self.assertEqual(
            samples[0]["implementation"], REGISTER_IMPLEMENTATION["id"]
        )

    def test_rejects_an_implementation_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "implementation order"):
            parse_samples(
                synthetic_output(block_number=2),
                experiment_id="EXP-0011",
                run_id="RUN-001",
                block_id="block-02",
                block_order="descending",
                implementation_order="direct_then_register_2x2",
            )


class PrefillRegister2x2ProtocolTest(unittest.TestCase):
    def test_commands_select_abba_and_workload_order(self):
        first = benchmark_command(block_number=1)
        second = benchmark_command(block_number=2)
        fourth = benchmark_command(block_number=4)

        self.assertIn("LINEAR_PREFILL_REGISTER_DIRECT_COMPARISON=true", first)
        self.assertNotIn("LINEAR_PREFILL_REGISTER_FIRST=true", first)
        self.assertIn("LINEAR_PREFILL_REGISTER_FIRST=true", second)
        self.assertIn("LINEAR_PREFILL_BK_SWEEP_REVERSE=true", second)
        self.assertNotIn("LINEAR_PREFILL_BK_SWEEP_REVERSE=true", fourth)

    def test_summary_advances_a_material_large_m_improvement(self):
        result = summarize(parsed_blocks(register_multiplier=0.9))

        self.assertTrue(result["advance_rule"]["qualified"])
        self.assertEqual(
            result["timing_decision"],
            "advance_register_2x2_to_public_rowwise_comparison",
        )
        self.assertEqual(result["production_dispatch_decision"], "none")

    def test_summary_rejects_a_material_large_m_regression(self):
        result = summarize(parsed_blocks(register_multiplier=1.1))

        self.assertTrue(result["advance_rule"]["rejected_at_required_rows"])
        self.assertEqual(
            result["timing_decision"],
            "reject_register_2x2_for_prefill_direct_mapping",
        )
        self.assertEqual(result["production_dispatch_decision"], "none")


if __name__ == "__main__":
    unittest.main()
