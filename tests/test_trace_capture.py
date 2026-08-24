import hashlib
import json
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmarks.capture_rms_norm_trace import (
    CAPTURE_ID,
    capture_trace,
    stage_profile_binary,
)


PROFILE_OUTPUT = "\n".join(
    [
        "profile implementation: enqueue_rms_norm_apple_gpu",
        "device: Apple Test GPU",
        "api: metal",
        "rows: 1",
        "hidden: 896",
        "warmup iterations: 1",
        "profile iterations: 10",
        "post-profile idle milliseconds: 0",
        "PROFILE_REGION_BEGIN",
        "PROFILE_REGION_END",
        "",
    ]
)


def write_profile(directory: Path) -> Path:
    directory.mkdir(parents=True)
    binary = directory / "rmsnorm-profile"
    binary.write_bytes(b"test RMSNorm profile binary\n")
    binary.chmod(0o755)
    provenance = {
        "schema_version": 1,
        "repository": {
            "commit": "a" * 40,
            "branch": "test",
            "dirty": False,
        },
        "profile_rows": 1,
        "hidden_size": 896,
        "profile_warmup_iterations": 1,
        "profile_iterations": 10,
        "profile_post_idle_milliseconds": 0,
        "implementation": "apple_gpu_simdgroup_v1",
        "entrypoint": "enqueue_rms_norm_apple_gpu",
        "hardware": {
            "chip": "Apple Test GPU",
            "gpu_api": "metal",
        },
        "binary": {
            "bytes": binary.stat().st_size,
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        },
    }
    binary.with_name(binary.name + ".provenance.json").write_text(
        json.dumps(provenance) + "\n"
    )
    return binary


class TraceCaptureTest(unittest.TestCase):
    def test_documents_artifact_is_launched_from_verified_temporary_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_profile(root / "home" / "Documents" / "profiles")
            staging_root = root / "private-tmp"
            staging_root.mkdir()
            output = root / "external" / "capture.trace"
            calls: list[list[str]] = []
            staged_path: Path | None = None

            def runner(command: list[str], **_: object):
                nonlocal staged_path
                calls.append(command)
                if command[-1] == "version":
                    return subprocess.CompletedProcess(
                        command, 0, "xctrace version test\n"
                    )
                staged_path = Path(command[-1])
                self.assertTrue(
                    staged_path.is_relative_to(staging_root.resolve())
                )
                self.assertNotEqual(staged_path, source)
                self.assertEqual(staged_path.read_bytes(), source.read_bytes())
                self.assertEqual(
                    stat.S_IMODE(staged_path.stat().st_mode), 0o700
                )
                output.mkdir(parents=True)
                return subprocess.CompletedProcess(command, 0, PROFILE_OUTPUT)

            receipt = capture_trace(
                profile_binary=source,
                output_trace=output,
                template="Metal System Trace",
                time_limit="1s",
                staging_root=staging_root,
                runner=runner,
            )

            self.assertEqual(receipt["capture"]["status"], "complete")
            capture_id = receipt["capture"]["capture_id"]
            self.assertIsNotNone(CAPTURE_ID.fullmatch(capture_id))
            self.assertEqual(receipt["capture"]["run_name"], capture_id)
            self.assertEqual(
                receipt["capture"]["target_identity"],
                {
                    "implementation": "apple_gpu_simdgroup_v1",
                    "entrypoint": "enqueue_rms_norm_apple_gpu",
                    "device": "Apple Test GPU",
                    "backend": "metal",
                    "rows": 1,
                    "hidden_size": 896,
                    "warmup_iterations": 1,
                    "profile_iterations": 10,
                    "post_profile_idle_milliseconds": 0,
                },
            )
            self.assertEqual(
                receipt["profile"]["binary"]["sha256"],
                receipt["profile"]["staged_binary"]["sha256"],
            )
            self.assertEqual(len(calls), 2)
            capture_command = calls[0]
            self.assertEqual(
                capture_command[capture_command.index("--run-name") + 1],
                capture_id,
            )
            self.assertIsNotNone(staged_path)
            self.assertEqual(staged_path.name, capture_id)
            self.assertFalse(staged_path.exists())
            receipt_path = output.with_name(output.name + ".capture.json")
            receipt_text = receipt_path.read_text()
            self.assertNotIn(str(source), receipt_text)
            self.assertNotIn(str(staged_path), receipt_text)
            self.assertNotIn(str(output), receipt_text)

    def test_rejects_a_binary_that_does_not_match_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_profile(root / "external" / "profiles")
            source.write_bytes(b"changed after provenance\n")
            staging_root = root / "private-tmp"
            staging_root.mkdir()

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                capture_trace(
                    profile_binary=source,
                    output_trace=root / "external" / "capture.trace",
                    staging_root=staging_root,
                    runner=lambda *_args, **_kwargs: self.fail(
                        "xctrace must not run for a mismatched binary"
                    ),
                )

    def test_staged_copy_is_bound_to_the_validated_provenance_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_profile(root / "external" / "profiles")
            staging_directory = root / "private-tmp" / "capture"
            staging_directory.mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "provenance SHA-256"):
                stage_profile_binary(
                    source,
                    staging_directory,
                    expected_hash="0" * 64,
                    capture_id="rmsnorm-" + "0" * 32,
                )

    def test_target_identity_mismatch_is_recorded_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_profile(root / "external" / "profiles")
            staging_root = root / "private-tmp"
            staging_root.mkdir()
            output = root / "external" / "capture.trace"
            mismatched = PROFILE_OUTPUT.replace("rows: 1", "rows: 4")

            def runner(command: list[str], **_: object):
                if command[-1] == "version":
                    return subprocess.CompletedProcess(
                        command, 0, "xctrace version test\n"
                    )
                output.mkdir(parents=True)
                return subprocess.CompletedProcess(command, 0, mismatched)

            with self.assertRaisesRegex(RuntimeError, "does not match provenance"):
                capture_trace(
                    profile_binary=source,
                    output_trace=output,
                    staging_root=staging_root,
                    runner=runner,
                )

            receipt = json.loads(
                output.with_name(output.name + ".capture.json").read_text()
            )
            self.assertEqual(receipt["capture"]["status"], "invalid")
            self.assertIsNone(receipt["capture"]["target_identity"])
            self.assertIn(
                "target rows does not match provenance",
                receipt["capture"]["failures"][0],
            )

    def test_incomplete_target_output_is_recorded_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_profile(root / "external" / "profiles")
            staging_root = root / "private-tmp"
            staging_root.mkdir()
            output = root / "external" / "capture.trace"
            staged_path: Path | None = None

            def runner(command: list[str], **_: object):
                nonlocal staged_path
                if command[-1] == "version":
                    return subprocess.CompletedProcess(
                        command, 0, "xctrace version test\n"
                    )
                staged_path = Path(command[-1])
                output.mkdir(parents=True)
                return subprocess.CompletedProcess(command, 0, "recorded\n")

            with self.assertRaisesRegex(RuntimeError, "complete profile region"):
                capture_trace(
                    profile_binary=source,
                    output_trace=output,
                    staging_root=staging_root,
                    runner=runner,
                )

            receipt_path = output.with_name(output.name + ".capture.json")
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["capture"]["status"], "invalid")
            self.assertFalse(receipt["capture"]["profile_region_markers_complete"])
            self.assertIsNotNone(staged_path)
            self.assertFalse(staged_path.exists())

    def test_refuses_to_overwrite_a_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_profile(root / "external" / "profiles")
            staging_root = root / "private-tmp"
            staging_root.mkdir()
            output = root / "external" / "capture.trace"
            output.mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                capture_trace(
                    profile_binary=source,
                    output_trace=output,
                    staging_root=staging_root,
                )


if __name__ == "__main__":
    unittest.main()
