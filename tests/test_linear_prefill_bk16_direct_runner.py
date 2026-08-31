import unittest

from benchmarks.run_linear import BENCHMARK_RESULTS_BEGIN, BENCHMARK_RESULTS_END
from benchmarks.run_linear_prefill_bk16_direct import (
    BK16_IMPLEMENTATION,
    BLOCK_IMPLEMENTATION_ORDERS,
    BLOCK_ORDERS,
    DIRECT_IMPLEMENTATION,
    EXPECTED_REPETITIONS,
    WORKLOAD_ORDER,
    benchmark_command,
    parse_samples,
    summarize,
)


def synthetic_output(*, block_number: int, bk16_multiplier: float = 1.0) -> str:
    reverse = BLOCK_ORDERS[block_number - 1] == "descending"
    implementation_order = BLOCK_IMPLEMENTATION_ORDERS[block_number - 1]
    emitted_order = (
        "bk16,direct"
        if implementation_order == "bk16_then_direct"
        else "direct,bk16"
    )
    lines = [
        "implementation: enqueue_linear_prefill_direct_apple_gpu",
        "comparison implementation: enqueue_linear_prefill_tiled_apple_gpu_bk",
        "comparison BK: 16",
        "device: Apple Test GPU",
        "api: metal",
        f"implementation order: {emitted_order}",
        BENCHMARK_RESULTS_BEGIN,
        "name,met (ms),iters",
    ]
    implementations = (
        ("tiled_bk16", "direct")
        if implementation_order == "bk16_then_direct"
        else ("direct", "tiled_bk16")
    )
    workloads = reversed(WORKLOAD_ORDER) if reverse else WORKLOAD_ORDER
    for workload_index, workload in enumerate(workloads):
        base = 1.0 + workload_index / 100.0
        for implementation in implementations:
            multiplier = bk16_multiplier if implementation == "tiled_bk16" else 1.0
            for repetition in range(EXPECTED_REPETITIONS):
                value = base * multiplier + repetition / 1_000_000.0
                lines.append(
                    f"linear_prefill_{implementation}_apple_gpu/input_id:"
                    f"{workload},{value},20"
                )
    lines.append(BENCHMARK_RESULTS_END)
    return "\n".join(lines)


def parsed_blocks(*, bk16_multiplier: float) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for block_number in range(1, 5):
        _, block_samples = parse_samples(
            synthetic_output(
                block_number=block_number,
                bk16_multiplier=bk16_multiplier,
            ),
            experiment_id="EXP-0010",
            run_id="EXP-0010-RUN-TEST",
            block_id=f"block-{block_number:02d}",
            block_order=BLOCK_ORDERS[block_number - 1],
            implementation_order=BLOCK_IMPLEMENTATION_ORDERS[block_number - 1],
        )
        samples.extend(block_samples)
    return samples


class PrefillBK16DirectParserTest(unittest.TestCase):
    def test_records_both_implementations_and_source_costs(self):
        _, samples = parse_samples(
            synthetic_output(block_number=1),
            experiment_id="EXP-0010",
            run_id="RUN-001",
            block_id="block-01",
            block_order="ascending",
            implementation_order="direct_then_bk16",
        )

        self.assertEqual(
            len(samples), len(WORKLOAD_ORDER) * 2 * EXPECTED_REPETITIONS
        )
        self.assertEqual(len({sample["sample_id"] for sample in samples}), len(samples))
        direct = next(
            sample
            for sample in samples
            if sample["implementation"] == DIRECT_IMPLEMENTATION["id"]
        )
        bk16 = next(
            sample
            for sample in samples
            if sample["implementation"] == BK16_IMPLEMENTATION["id"]
        )
        self.assertEqual(direct["threadgroup_operand_scratch_bytes"], 0)
        self.assertEqual(direct["barriers_per_dispatch"], 0)
        self.assertEqual(bk16["threadgroup_operand_scratch_bytes"], 768)
        self.assertEqual(bk16["barriers_per_dispatch"], 112)
        self.assertLess(
            bk16["program_requested_traffic_bytes"],
            direct["program_requested_traffic_bytes"],
        )

    def test_accepts_reversed_candidate_first_block(self):
        _, samples = parse_samples(
            synthetic_output(block_number=2),
            experiment_id="EXP-0010",
            run_id="RUN-001",
            block_id="block-02",
            block_order="descending",
            implementation_order="bk16_then_direct",
        )

        self.assertEqual(samples[0]["workload"], WORKLOAD_ORDER[-1])
        self.assertEqual(samples[0]["implementation"], BK16_IMPLEMENTATION["id"])

    def test_rejects_an_implementation_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "implementation order"):
            parse_samples(
                synthetic_output(block_number=2),
                experiment_id="EXP-0010",
                run_id="RUN-001",
                block_id="block-02",
                block_order="descending",
                implementation_order="direct_then_bk16",
            )


class PrefillBK16DirectProtocolTest(unittest.TestCase):
    def test_commands_select_abba_and_workload_order(self):
        first = benchmark_command(block_number=1)
        second = benchmark_command(block_number=2)
        fourth = benchmark_command(block_number=4)

        self.assertIn("LINEAR_PREFILL_BK16_DIRECT_COMPARISON=true", first)
        self.assertNotIn("LINEAR_PREFILL_BK16_FIRST=true", first)
        self.assertIn("LINEAR_PREFILL_BK16_FIRST=true", second)
        self.assertIn("LINEAR_PREFILL_BK_SWEEP_REVERSE=true", second)
        self.assertNotIn("LINEAR_PREFILL_BK_SWEEP_REVERSE=true", fourth)

    def test_summary_advances_a_material_large_m_improvement(self):
        result = summarize(parsed_blocks(bk16_multiplier=0.9))

        self.assertTrue(result["advance_rule"]["qualified"])
        self.assertEqual(
            result["timing_decision"],
            "advance_bk16_to_public_rowwise_comparison",
        )
        self.assertEqual(result["production_dispatch_decision"], "none")

    def test_summary_rejects_a_material_large_m_regression(self):
        result = summarize(parsed_blocks(bk16_multiplier=1.1))

        self.assertTrue(result["advance_rule"]["rejected_at_required_rows"])
        self.assertEqual(
            result["timing_decision"],
            "reject_bk16_shared_staging_for_scalar_output_mapping",
        )
        self.assertEqual(result["production_dispatch_decision"], "none")


if __name__ == "__main__":
    unittest.main()
