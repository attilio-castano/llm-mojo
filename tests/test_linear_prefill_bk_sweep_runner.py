import unittest

from benchmarks.run_linear import BENCHMARK_RESULTS_BEGIN, BENCHMARK_RESULTS_END
from benchmarks.run_linear_prefill_bk_sweep import (
    BK_VALUES,
    BLOCK_BK_ORDERS,
    BLOCK_ORDERS,
    EXPECTED_REPETITIONS,
    WORKLOAD_ORDER,
    benchmark_command,
    parse_samples,
    summarize,
)


def synthetic_output(
    *,
    block_number: int,
    bk_multipliers: dict[int, float] | None = None,
) -> str:
    bk_order = BLOCK_BK_ORDERS[block_number - 1]
    reverse = BLOCK_ORDERS[block_number - 1] == "descending"
    multipliers = bk_multipliers or {bk: 1.0 for bk in BK_VALUES}
    lines = [
        "implementation: enqueue_linear_prefill_tiled_apple_gpu_bk",
        "device: Apple Test GPU",
        "api: metal",
        "BK order: " + ",".join(str(bk) for bk in bk_order),
        BENCHMARK_RESULTS_BEGIN,
        "name,met (ms),iters",
    ]
    workloads = reversed(WORKLOAD_ORDER) if reverse else WORKLOAD_ORDER
    for workload_index, workload in enumerate(workloads):
        base = 1.0 + workload_index / 100.0
        for bk in bk_order:
            for repetition in range(EXPECTED_REPETITIONS):
                value = base * multipliers[bk] + repetition / 1_000_000.0
                lines.append(
                    f"linear_prefill_tiled_bk{bk}_apple_gpu/input_id:"
                    f"{workload},{value},20"
                )
    lines.append(BENCHMARK_RESULTS_END)
    return "\n".join(lines)


def parsed_blocks(
    *, bk_multipliers: dict[int, float] | None = None
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for block_number in range(1, 5):
        _, block_samples = parse_samples(
            synthetic_output(
                block_number=block_number,
                bk_multipliers=bk_multipliers,
            ),
            experiment_id="EXP-0009",
            run_id="EXP-0009-RUN-TEST",
            block_id=f"block-{block_number:02d}",
            block_order=BLOCK_ORDERS[block_number - 1],
            bk_order=BLOCK_BK_ORDERS[block_number - 1],
        )
        samples.extend(block_samples)
    return samples


class PrefillBKSweepParserTest(unittest.TestCase):
    def test_records_all_bks_with_source_level_costs(self):
        _, samples = parse_samples(
            synthetic_output(block_number=1),
            experiment_id="EXP-0009",
            run_id="RUN-001",
            block_id="block-01",
            block_order="ascending",
            bk_order=BLOCK_BK_ORDERS[0],
        )

        self.assertEqual(
            len(samples),
            len(WORKLOAD_ORDER) * len(BK_VALUES) * EXPECTED_REPETITIONS,
        )
        self.assertEqual(len({sample["sample_id"] for sample in samples}), len(samples))
        by_bk = {int(sample["bk"]): sample for sample in samples}
        self.assertEqual(
            {
                bk: int(by_bk[bk]["threadgroup_operand_scratch_bytes"])
                for bk in BK_VALUES
            },
            {16: 768, 32: 1_536, 64: 3_072, 128: 6_144},
        )
        self.assertEqual(
            {bk: int(by_bk[bk]["barriers_per_dispatch"]) for bk in BK_VALUES},
            {16: 112, 32: 56, 64: 28, 128: 14},
        )
        self.assertEqual(
            len(
                {
                    int(sample["program_requested_traffic_bytes"])
                    for sample in samples
                    if sample["workload"] == WORKLOAD_ORDER[0]
                }
            ),
            1,
        )

    def test_accepts_reversed_workloads_and_second_sequence(self):
        _, samples = parse_samples(
            synthetic_output(block_number=2),
            experiment_id="EXP-0009",
            run_id="RUN-001",
            block_id="block-02",
            block_order="descending",
            bk_order=BLOCK_BK_ORDERS[1],
        )

        self.assertEqual(samples[0]["workload"], WORKLOAD_ORDER[-1])
        self.assertEqual(int(samples[0]["bk"]), BLOCK_BK_ORDERS[1][0])

    def test_rejects_a_declared_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "unexpected BK order"):
            parse_samples(
                synthetic_output(block_number=2),
                experiment_id="EXP-0009",
                run_id="RUN-001",
                block_id="block-02",
                block_order="descending",
                bk_order=BLOCK_BK_ORDERS[0],
            )


class PrefillBKSweepProtocolTest(unittest.TestCase):
    def test_sequences_balance_every_bk_across_every_position(self):
        for bk in BK_VALUES:
            positions = [order.index(bk) for order in BLOCK_BK_ORDERS]
            self.assertEqual(sorted(positions), list(range(len(BK_VALUES))))

    def test_commands_select_the_block_order_and_sequence(self):
        first = benchmark_command(block_number=1)
        fourth = benchmark_command(block_number=4)

        self.assertNotIn("LINEAR_PREFILL_BK_SWEEP_REVERSE=true", first)
        self.assertNotIn("LINEAR_PREFILL_BK_SWEEP_SEQUENCE_1=true", first)
        self.assertNotIn("LINEAR_PREFILL_BK_SWEEP_REVERSE=true", fourth)
        self.assertIn("LINEAR_PREFILL_BK_SWEEP_SEQUENCE_4=true", fourth)

    def test_summary_applies_predeclared_advance_rule(self):
        samples = parsed_blocks(
            bk_multipliers={16: 1.0, 32: 1.0, 64: 0.9, 128: 1.1}
        )
        result = summarize(samples)

        self.assertEqual(
            result["advance_rule"]["eligible_candidates"],
            [64],
        )
        self.assertEqual(result["advance_rule"]["selected_candidate_bk"], 64)
        self.assertEqual(
            result["timing_decision"],
            "advance_bk64_to_direct_control_comparison",
        )
        self.assertEqual(result["production_dispatch_decision"], "none")


if __name__ == "__main__":
    unittest.main()
