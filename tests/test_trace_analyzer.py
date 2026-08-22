import unittest

from benchmarks.analyze_rms_norm_trace import segment_compute_commands


def commands(count: int):
    return [{"index": (str(index), str(index))} for index in range(count)]


class SegmentComputeCommandsTest(unittest.TestCase):
    def test_exact_sequence_has_no_setup_prefix(self):
        setup, correctness, warmup, profile = segment_compute_commands(
            commands(6), warmup_iterations=2, profile_iterations=3
        )

        self.assertEqual(setup, [])
        self.assertEqual(correctness[0]["index"][0], "0")
        self.assertEqual([row["index"][0] for row in warmup], ["1", "2"])
        self.assertEqual(
            [row["index"][0] for row in profile], ["3", "4", "5"]
        )

    def test_setup_compute_commands_are_a_prefix(self):
        setup, correctness, warmup, profile = segment_compute_commands(
            commands(8), warmup_iterations=2, profile_iterations=3
        )

        self.assertEqual([row["index"][0] for row in setup], ["0", "1"])
        self.assertEqual(correctness[0]["index"][0], "2")
        self.assertEqual([row["index"][0] for row in warmup], ["3", "4"])
        self.assertEqual(
            [row["index"][0] for row in profile], ["5", "6", "7"]
        )

    def test_incomplete_sequence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected at least 6"):
            segment_compute_commands(
                commands(5), warmup_iterations=2, profile_iterations=3
            )


if __name__ == "__main__":
    unittest.main()
