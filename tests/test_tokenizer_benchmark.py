"""CPU evidence must retain paired calibration and prove the executed route."""
import json
from pathlib import Path
import tempfile
import unittest
from llm_mojo.benchmarks import tokenizer_contract as contract
from llm_mojo.benchmarks.study import (
    BLOCKS,
    REPETITIONS,
    WARMUP,
    encode_samples,
    sha,
)


class TokenizerBenchmarkTests(unittest.TestCase):
    def output(self, first=False):
        lines = [
            "api: cpu",
            "operation: tokenizer",
            "case: 1 mode: encode",
            "correctness: passed",
            "workspace capacity: 8 8 8 16",
        ]
        arms = [("control", 0), ("candidate", 1)]
        if first:
            arms.reverse()
        lines += [
            f'SAMPLE {arm} {v} {r} 12.0'
            for arm, v in arms
            for r in range(REPETITIONS)
        ]
        return "\n".join(lines + ["BENCHMARK_COMPLETE"])

    def test_runtime_route_order_completion_and_finite_samples(self):
        for first in [False, True]:
            observations, caps = contract.parse_output(
                self.output(first), 1, "encode", 1, first
            )
            self.assertEqual(len(observations), 20)
            self.assertEqual(caps, [8, 8, 8, 16])
        valid = self.output()
        for output in [
            valid.replace("api: cpu", "api: metal"),
            valid.replace("12.0", "nan"),
            valid.replace("BENCHMARK_COMPLETE", ""),
            valid.replace("SAMPLE candidate 1", "SAMPLE candidate 0"),
        ]:
            with self.assertRaises(ValueError):
                contract.parse_output(output, 1, "encode", 1, False)
        with self.assertRaises(ValueError):
            contract.parse_output(valid, 1, "encode", 1, True)

    def test_incomplete_calibration_or_modified_payload_cannot_be_reported(
        self,
    ):
        inputs = {"cases": [{"case": 1}]}
        spec = contract.specification("encode", inputs)
        samples = [
            dict(
                block=b,
                rows=1,
                layers=1,
                candidate=c,
                arm=arm,
                variant=0 if arm == "control" else c,
                repetition=r,
                us=10.0 if arm == "control" or c == 0 else 8.0,
            )
            for b in range(1, BLOCKS + 1)
            for c in (0, 1)
            for arm in ("control", "candidate")
            for r in range(REPETITIONS)
        ]
        repo = {"dirty": False, "commit": "fixed"}
        record = dict(
            schema=1,
            completed_utc="done",
            repository=repo,
            runtime={"api": "cpu", "device": "Apple M4 Pro"},
            build={
                "repository": repo,
                "environment": {"backend": "cpu", "cpu": "Apple M4 Pro"},
            },
            blocks=BLOCKS,
            repetitions=REPETITIONS,
            warmup=WARMUP,
            conditions=[
                dict(block=b, before={}, after={}) for b in range(1, BLOCKS + 1)
            ],
            specification=spec,
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "samples.csv.gz"

            def save(observations):
                path.write_bytes(encode_samples(observations))
                record["samples_sha256"] = sha(path)
                (directory / "run.json").write_text(json.dumps(record))

            save(samples)
            self.assertEqual(
                contract.load(directory)[1][1]["decision"], "faster"
            )
            save([s for s in samples if s["candidate"] != 0])
            with self.assertRaisesRegex(ValueError, "grid"):
                contract.load(directory)
            save(samples)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "samples"):
                contract.load(directory)


if __name__ == "__main__":
    unittest.main()
