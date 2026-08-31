import unittest

from benchmarks.run_linear import BENCHMARK_RESULTS_BEGIN, BENCHMARK_RESULTS_END
from benchmarks.run_linear_mma_phase import (
    BLOCK_IMPLEMENTATION_ORDERS,
    BLOCK_ORDERS,
    EXPECTED_REPETITIONS,
    MMA_IMPLEMENTATION,
    REGISTER_IMPLEMENTATION,
    ROWWISE_IMPLEMENTATION,
    WORKLOAD_ORDER,
    WORKLOADS,
    benchmark_command,
    control_for_rows,
    parse_samples,
    summarize,
)


def synthetic_output(
    *,
    block_number: int,
    prefill_mma_multiplier: float = 1.0,
    decode_mma_multiplier: float = 1.0,
) -> str:
    reverse = BLOCK_ORDERS[block_number - 1] == "descending"
    implementation_order = BLOCK_IMPLEMENTATION_ORDERS[block_number - 1]
    emitted_order = (
        "mma,control"
        if implementation_order == "mma_then_control"
        else "control,mma"
    )
    lines = [
        "implementation: enqueue_linear_prefill_mma_8x16_apple_gpu",
        "M=1,8 control: enqueue_linear_apple_gpu",
        "M>=16 control: enqueue_linear_prefill_register_2x2_apple_gpu",
        "device: Apple Test GPU",
        "api: metal",
        f"workload order: {'descending' if reverse else 'ascending'}",
        f"implementation order: {emitted_order}",
        BENCHMARK_RESULTS_BEGIN,
        "name,met (ms),iters",
    ]
    workloads = list(reversed(WORKLOAD_ORDER)) if reverse else WORKLOAD_ORDER
    for workload_index, workload in enumerate(workloads):
        rows = int(WORKLOADS[workload]["rows"])
        control = control_for_rows(rows)
        control_name = (
            "rowwise"
            if control == ROWWISE_IMPLEMENTATION
            else "register_2x2"
        )
        implementations = (
            ("mma_8x16", control_name)
            if implementation_order == "mma_then_control"
            else (control_name, "mma_8x16")
        )
        base = 1.0 + workload_index / 100.0
        for implementation in implementations:
            multiplier = 1.0
            if implementation == "mma_8x16":
                multiplier = (
                    decode_mma_multiplier
                    if rows == 1
                    else prefill_mma_multiplier
                )
            for repetition in range(EXPECTED_REPETITIONS):
                value = base * multiplier + repetition / 1_000_000.0
                lines.append(
                    f"linear_prefill_{implementation}_apple_gpu/input_id:"
                    f"{workload},{value},20"
                )
    lines.append(BENCHMARK_RESULTS_END)
    return "\n".join(lines)


def parsed_blocks(
    *, prefill_mma_multiplier: float, decode_mma_multiplier: float
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for block_number in range(1, 5):
        _, block_samples = parse_samples(
            synthetic_output(
                block_number=block_number,
                prefill_mma_multiplier=prefill_mma_multiplier,
                decode_mma_multiplier=decode_mma_multiplier,
            ),
            experiment_id="EXP-0013",
            run_id="EXP-0013-RUN-TEST",
            block_id=f"block-{block_number:02d}",
            block_order=BLOCK_ORDERS[block_number - 1],
            implementation_order=BLOCK_IMPLEMENTATION_ORDERS[
                block_number - 1
            ],
        )
        samples.extend(block_samples)
    return samples


class LinearMmaPhaseParserTest(unittest.TestCase):
    def test_records_phase_controls_and_mma_structure(self):
        _, samples = parse_samples(
            synthetic_output(block_number=1),
            experiment_id="EXP-0013",
            run_id="RUN-001",
            block_id="block-01",
            block_order="ascending",
            implementation_order="control_then_mma",
        )

        self.assertEqual(
            len(samples), len(WORKLOAD_ORDER) * 2 * EXPECTED_REPETITIONS
        )
        self.assertEqual(
            len({sample["sample_id"] for sample in samples}), len(samples)
        )
        m1_control = next(
            sample
            for sample in samples
            if sample["rows"] == 1 and sample["role"] == "control"
        )
        m16_control = next(
            sample
            for sample in samples
            if sample["rows"] == 16 and sample["role"] == "control"
        )
        mma = next(
            sample
            for sample in samples
            if sample["implementation"] == MMA_IMPLEMENTATION["id"]
        )
        self.assertEqual(
            m1_control["implementation"], ROWWISE_IMPLEMENTATION["id"]
        )
        self.assertEqual(
            m16_control["implementation"], REGISTER_IMPLEMENTATION["id"]
        )
        self.assertEqual(mma["threads_per_threadgroup"], 32)
        self.assertEqual(mma["outputs_per_simd_group"], 128)
        self.assertEqual(mma["mma_k"], 8)
        self.assertEqual(mma["k_phases_per_dispatch"], 112)
        self.assertEqual(mma["mma_operations_per_k_phase"], 2)
        self.assertEqual(mma["fp32_accumulators_per_thread"], 4)
        self.assertEqual(mma["barriers_per_dispatch"], 0)

    def test_accepts_reversed_mma_first_block(self):
        _, samples = parse_samples(
            synthetic_output(block_number=2),
            experiment_id="EXP-0013",
            run_id="RUN-001",
            block_id="block-02",
            block_order="descending",
            implementation_order="mma_then_control",
        )

        self.assertEqual(samples[0]["workload"], WORKLOAD_ORDER[-1])
        self.assertEqual(samples[0]["implementation"], MMA_IMPLEMENTATION["id"])

    def test_rejects_an_implementation_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "implementation order"):
            parse_samples(
                synthetic_output(block_number=2),
                experiment_id="EXP-0013",
                run_id="RUN-001",
                block_id="block-02",
                block_order="descending",
                implementation_order="control_then_mma",
            )


class LinearMmaPhaseProtocolTest(unittest.TestCase):
    def test_commands_select_abba_and_workload_order(self):
        first = benchmark_command(block_number=1)
        second = benchmark_command(block_number=2)
        fourth = benchmark_command(block_number=4)

        self.assertNotIn("LINEAR_MMA_DIAGNOSTIC_MMA_FIRST=true", first)
        self.assertIn("LINEAR_MMA_DIAGNOSTIC_MMA_FIRST=true", second)
        self.assertIn("LINEAR_MMA_DIAGNOSTIC_REVERSE=true", second)
        self.assertNotIn("LINEAR_MMA_DIAGNOSTIC_REVERSE=true", fourth)

    def test_summary_advances_prefill_and_rejects_batch_1_decode(self):
        result = summarize(
            parsed_blocks(
                prefill_mma_multiplier=0.9,
                decode_mma_multiplier=1.1,
            )
        )

        self.assertTrue(result["large_prefill_rule"]["qualified"])
        self.assertEqual(
            result["timing_decision"],
            "advance_mma_as_large_prefill_candidate",
        )
        self.assertEqual(
            result["decode"]["disposition"],
            "reject_mma_for_batch_1_decode",
        )
        self.assertEqual(result["production_dispatch_decision"], "none")

    def test_summary_rejects_a_material_large_m_regression(self):
        result = summarize(
            parsed_blocks(
                prefill_mma_multiplier=1.1,
                decode_mma_multiplier=1.1,
            )
        )

        self.assertTrue(
            result["large_prefill_rule"]["rejected_at_required_rows"]
        )
        self.assertEqual(result["timing_decision"], "reject_mma_for_large_prefill")
        self.assertEqual(result["production_dispatch_decision"], "none")


if __name__ == "__main__":
    unittest.main()
