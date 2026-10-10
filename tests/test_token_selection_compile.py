"""Build the valid host contract and require deliberate mutants to fail compilation."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from llm_mojo._repository import environment_tool


ROOT = Path(__file__).resolve().parents[1]
COMPILE_TIMEOUT_SECONDS = 180


class TokenSelectionCompileTests(unittest.TestCase):
    def test_valid_contract_builds_and_mutants_are_rejected(self):
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

            accepted = compile_source("tests/test_token_selection_contract.mojo", "valid")
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

            for name, diagnostic in (
                ("signed_zero", "BF16 ordering contract failed"),
                ("ignored_nan", "BF16 classification contract failed"),
                ("larger_token", "argmax representative laws failed"),
                ("dropped_invalid", "argmax representative laws failed"),
            ):
                with self.subTest(mutant=name):
                    rejected = compile_source(
                        f"tests/compile_fail/token_selection_{name}.mojo", name,
                    )
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


if __name__ == "__main__":
    unittest.main()
