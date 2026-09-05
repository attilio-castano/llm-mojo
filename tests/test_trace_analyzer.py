import json
import tempfile
import unittest
from pathlib import Path

from llm_mojo.benchmarks.analyze_trace import (
    capture_identity,
    segment_compute_commands,
    summarize_profile_counters,
    validate_trace_binding,
)


CAPTURE_ID = "rmsnorm-" + "1" * 32
LINEAR_CAPTURE_ID = "linear-" + "2" * 32
SHA = "a" * 64


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
        "description": cell("Synthetic counter percentage"),
        "type": cell("Percentage"),
    }


def counter_value(timestamp: int, counter_id: int, value: float):
    return {
        "accelerator-id": cell(7),
        "counter-id": cell(counter_id),
        "timestamp": cell(timestamp),
        "value": cell(value),
    }


def capture_receipt(*, target_rows: int = 512):
    target = {
        "implementation": "apple_gpu_simdgroup_v1",
        "entrypoint": "enqueue_rms_norm_apple_gpu",
        "device": "Apple Test GPU",
        "backend": "metal",
        "rows": target_rows,
        "hidden_size": 896,
        "warmup_iterations": 100,
        "profile_iterations": 500,
        "post_profile_idle_milliseconds": 250,
    }
    return {
        "schema_version": 2,
        "capture": {
            "capture_id": CAPTURE_ID,
            "run_name": CAPTURE_ID,
            "status": "complete",
            "command": (
                "xcrun xctrace record --run-name "
                f"{CAPTURE_ID} --launch -- '<ephemeral-profile-binary>'"
            ),
            "template": {
                "kind": "installed_name",
                "name": "LLM_Mojo_Metal_Limiters",
            },
            "xctrace_returncode": 0,
            "profile_region_markers_complete": True,
            "target_identity": target,
            "target_output": {"bytes": 100, "sha256": "d" * 64},
            "failures": [],
        },
        "profile": {
            "binary": {"bytes": 100, "sha256": SHA},
            "staged_binary": {"bytes": 100, "sha256": SHA},
            "provenance": {
                "bytes": 200,
                "sha256": "b" * 64,
                "schema_version": 1,
            },
            "configuration": {
                "profile_rows": 512,
                "hidden_size": 896,
                "profile_warmup_iterations": 100,
                "profile_iterations": 500,
                "profile_post_idle_milliseconds": 250,
                "implementation": "apple_gpu_simdgroup_v1",
                "entrypoint": "enqueue_rms_norm_apple_gpu",
            },
            "repository": {
                "commit": "c" * 40,
                "branch": "test",
                "dirty": False,
            },
            "hardware": {"chip": "Apple Test GPU", "gpu_api": "metal"},
        },
        "trace": {"created": True},
    }


def linear_capture_receipt():
    receipt = capture_receipt(target_rows=1)
    receipt["capture"]["capture_id"] = LINEAR_CAPTURE_ID
    receipt["capture"]["run_name"] = LINEAR_CAPTURE_ID
    receipt["capture"]["command"] = (
        "xcrun xctrace record --run-name "
        f"{LINEAR_CAPTURE_ID} --launch -- '<ephemeral-profile-binary>'"
    )
    receipt["capture"]["target_identity"] = {
        "implementation": "apple_gpu_one_output_simdgroup_v0",
        "entrypoint": "enqueue_linear_apple_gpu",
        "device": "Apple Test GPU",
        "backend": "metal",
        "rows": 1,
        "hidden_size": 896,
        "profile_workload": "qkv-ring24",
        "output_features": 1152,
        "dispatches_per_iteration": 72,
        "warmup_iterations": 1,
        "profile_iterations": 2,
        "post_profile_idle_milliseconds": 0,
    }
    receipt["profile"]["configuration"] = {
        "operation": "linear_projection",
        "profile_rows": 1,
        "hidden_size": 896,
        "profile_workload": "qkv-ring24",
        "output_features": 1152,
        "layers": 24,
        "dispatches_per_iteration": 72,
        "profile_warmup_iterations": 1,
        "profile_iterations": 2,
        "profile_post_idle_milliseconds": 0,
        "implementation": "apple_gpu_one_output_simdgroup_v0",
        "entrypoint": "enqueue_linear_apple_gpu",
    }
    return receipt


def trace_metadata():
    return {
        "run_name": CAPTURE_ID,
        "template": "LLM_Mojo_Metal_Limiters",
        "end_reason": "Target app exited",
        "target": {
            "process_name": CAPTURE_ID,
            "return_exit_status": "0",
            "termination_reason": "exit(0)",
            "device_platform": "macOS",
            "device_model": "MacBook Pro",
            "device_os_version": "test",
        },
    }


class SegmentComputeCommandsTest(unittest.TestCase):
    def test_multi_dispatch_iterations_are_segmented_as_groups(self):
        setup, correctness, warmup, profile = segment_compute_commands(
            commands(20),
            warmup_iterations=2,
            profile_iterations=3,
            dispatches_per_iteration=3,
        )

        self.assertEqual(len(setup), 2)
        self.assertEqual(len(correctness), 3)
        self.assertEqual(len(warmup), 6)
        self.assertEqual(len(profile), 9)

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

    def test_sequence_can_start_after_receipt_proven_correctness_gate(self):
        setup, correctness, warmup, profile = segment_compute_commands(
            commands(5),
            warmup_iterations=2,
            profile_iterations=3,
            trace_correctness_dispatches=False,
        )

        self.assertEqual(setup, [])
        self.assertEqual(correctness, [])
        self.assertEqual([row["index"][0] for row in warmup], ["0", "1"])
        self.assertEqual(
            [row["index"][0] for row in profile], ["2", "3", "4"]
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
        self.assertEqual(result["counters"][0]["unit"], "percent")

    def test_rejects_counter_window_without_profile_overlap(self):
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            summarize_profile_counters(
                [counter_info(3, "Kernel Occupancy")],
                [counter_value(99, 3, 90.0)],
                [interval(100, 20)],
            )


class CaptureIdentityTest(unittest.TestCase):
    def test_projection_receipt_retains_multi_dispatch_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            path.write_text(json.dumps(linear_capture_receipt()) + "\n")

            identity, _ = capture_identity(path)

        self.assertEqual(identity["operation"], "linear_projection")
        self.assertEqual(identity["capture_id"], LINEAR_CAPTURE_ID)
        self.assertEqual(
            identity["workload"]["dispatches_per_iteration"], 72
        )
        self.assertEqual(identity["workload"]["profile_workload"], "qkv-ring24")

    def test_projection_receipt_accepts_packed_qkv_identity(self):
        receipt = linear_capture_receipt()
        receipt["capture"]["target_identity"].update(
            {
                "implementation": "apple_gpu_packed_qkv_single_enqueue_v1",
                "dispatches_per_iteration": 24,
            }
        )
        receipt["profile"]["configuration"].update(
            {
                "implementation": "apple_gpu_packed_qkv_single_enqueue_v1",
                "dispatches_per_iteration": 24,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            path.write_text(json.dumps(receipt) + "\n")

            identity, _ = capture_identity(path)

        self.assertEqual(
            identity["implementation"],
            "apple_gpu_packed_qkv_single_enqueue_v1",
        )
        self.assertEqual(identity["workload"]["dispatches_per_iteration"], 24)

    def test_receipt_identity_is_carried_into_the_summary_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            path.write_text(json.dumps(capture_receipt()) + "\n")

            identity, template = capture_identity(path)

        self.assertEqual(identity["capture_id"], CAPTURE_ID)
        self.assertEqual(identity["implementation"], "apple_gpu_simdgroup_v1")
        self.assertEqual(identity["runtime"]["device"], "Apple Test GPU")
        self.assertEqual(identity["workload"]["profile_iterations"], 500)
        self.assertEqual(template, "LLM_Mojo_Metal_Limiters")

    def test_receipt_rejects_runtime_identity_that_disagrees_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            path.write_text(
                json.dumps(capture_receipt(target_rows=4)) + "\n"
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                capture_identity(path)

    def test_trace_process_must_match_receipt_generated_capture_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.json"
            path.write_text(json.dumps(capture_receipt()) + "\n")
            identity, template = capture_identity(path)

        with self.assertRaisesRegex(ValueError, "submissions"):
            validate_trace_binding(
                trace_metadata(),
                identity,
                "rmsnorm-" + "2" * 32 + " (42)",
                template,
            )


if __name__ == "__main__":
    unittest.main()
