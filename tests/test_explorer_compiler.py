"""Compiler overrides must not be mislabeled by the Python package version."""
from importlib import util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = util.spec_from_file_location(
    "explorer_verify_contract", ROOT / "docs/explorer/verify_contract.py")
verifier = util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class ExplorerCompilerVersionTests(unittest.TestCase):
    def test_accepts_pinned_compiler_release(self):
        for output in ("Mojo 1.1.0 (8189361e)\n", "Mojo 1.1.0\n"):
            with self.subTest(output=output):
                self.assertEqual(verifier.checked_compiler_version(output, "1.1.0"), "1.1.0")

    def test_rejects_different_compiler_even_if_python_package_is_pinned(self):
        with self.assertRaisesRegex(RuntimeError, r"Compiler Mojo 1\.0\.0 differs from uv\.lock 1\.1\.0"):
            verifier.checked_compiler_version("Mojo 1.0.0 (other-build)\n", "1.1.0")

    def test_rejects_unparseable_compiler_output(self):
        for output in ("", "clang version 1.1.0\n", "Mojo nightly\n", "Mojo 1.1.0dev\n"):
            with self.subTest(output=output):
                with self.assertRaisesRegex(RuntimeError, "Cannot parse Mojo compiler --version output"):
                    verifier.checked_compiler_version(output, "1.1.0")


if __name__ == "__main__":
    unittest.main()
