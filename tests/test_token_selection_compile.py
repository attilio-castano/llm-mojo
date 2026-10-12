"""Build valid host contracts and reject scalar mutants and malformed reduction plans."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from llm_mojo._repository import environment_tool


ROOT = Path(__file__).resolve().parents[1]
COMPILE_TIMEOUT_SECONDS = 180


class TokenSelectionCompileTests(unittest.TestCase):
    def _check_contract(self, positive_source, mutants):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-contract-") as directory:
            output = Path(directory)

            def compile_source(source, name):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", source,
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(positive_source, "valid")
            self.assertEqual(
                accepted.returncode, 0,
                "The valid contract must compile before rejection is checked.\n"
                + accepted.stdout + accepted.stderr,
            )
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            for source, diagnostic in mutants:
                name = Path(source).stem
                with self.subTest(mutant=name):
                    rejected = compile_source(source, name)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    # Match the diagnostic itself, not an echoed source line
                    # containing the assertion's message after an unrelated error.
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape(diagnostic) + r"\s*$",
                    )
                    self.assertFalse((output / name).exists())

    def test_valid_contract_builds_and_mutants_are_rejected(self):
        self._check_contract("tests/test_token_selection_contract.mojo", (
            ("tests/compile_fail/token_selection_signed_zero.mojo", "BF16 ordering contract failed"),
            ("tests/compile_fail/token_selection_ignored_nan.mojo", "BF16 classification contract failed"),
            ("tests/compile_fail/token_selection_larger_token.mojo", "argmax representative laws failed"),
            ("tests/compile_fail/token_selection_dropped_invalid.mojo", "argmax representative laws failed"),
        ))

    def test_reduction_plan_builds_and_invalid_plans_are_rejected(self):
        self._check_contract("tests/test_reduction_plan.mojo", tuple(
            (f"tests/compile_fail/reduction_plan_{name}.mojo", "argmax reduction plan rejected")
            for name in (
                "missing_input", "foreign_input", "forward_reference",
                "negative_reference", "unsupported_operation", "invalid_root",
            )
        ))


if __name__ == "__main__":
    unittest.main()
