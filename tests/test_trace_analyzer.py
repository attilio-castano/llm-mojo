import unittest

from benchmarks.analyze_rms_norm_trace import (
    segment_compute_commands,
    summarize_profile_counters,
)


def commands(count: int):
    return [{"index": (str(index), str(index))} for index in range(count)]


def cell(value: object):
    text = str(value)
    return text, text


def interval(start: int, duration: int):
    return {"start": cell(start), "duration": cell(duration)}


def counter_info(counter_id: int, name: str):
    return {
        "accelerator-id": cell(7),
        "counter-id": cell(counter_id),
        "name": cell(name),
        "type": cell("Percentage"),
    }


def counter_value(timestamp: int, counter_id: int, value: float):
    return {
        "accelerator-id": cell(7),
        "counter-id": cell(counter_id),
        "timestamp": cell(timestamp),
        "value": cell(value),
    }


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


class ProfileCounterSummaryTest(unittest.TestCase):
    def test_summarizes_named_samples_inside_profile_window(self):
        result = summarize_profile_counters(
            [counter_info(3, "Kernel Occupancy")],
            [
                counter_value(99, 3, 90.0),
                counter_value(105, 3, 25.0),
                counter_value(115, 3, 75.0),
                counter_value(121, 3, 10.0),
            ],
            [interval(100, 10), interval(112, 8)],
        )

        self.assertEqual(result["defined_counter_count"], 1)
        self.assertEqual(result["sampled_counter_count"], 1)
        self.assertEqual(result["samples"]["value_count"], 2)
        self.assertEqual(result["samples"]["timestamp_count"], 2)
        self.assertEqual(result["counters"][0]["median"], 50.0)

    def test_rejects_counter_window_without_profile_overlap(self):
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            summarize_profile_counters(
                [counter_info(3, "Kernel Occupancy")],
                [counter_value(99, 3, 90.0)],
                [interval(100, 20)],
            )


if __name__ == "__main__":
    unittest.main()
