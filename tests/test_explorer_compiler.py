"""Bind explorer evidence to its committed sources and selected compiler."""
import hashlib
from importlib import util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


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


class ExplorerSourceProvenanceTests(unittest.TestCase):
    INPUT = "tests/input.mojo"
    FIXTURE = "tests/compile_fail/reduction_plan_example.mojo"

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="explorer-provenance-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        source_paths = patch.object(verifier, "SOURCE_PATHS", [self.INPUT])
        source_paths.start()
        self.addCleanup(source_paths.stop)
        for path in (self.INPUT, self.FIXTURE,
                     "docs/explorer/contract-receipt.json", "docs/explorer/index.html"):
            self.write(path, "original\n")
        self.git("init", "--quiet")
        self.git("add", ".")
        self.commit("initial sources")

    def write(self, relative_path, text):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def git(self, *arguments):
        return subprocess.run(
            ["git", *arguments], cwd=self.root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()

    def commit(self, message):
        self.git("-c", "user.name=Explorer regression test",
                 "-c", "user.email=explorer-test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--quiet",
                 "--allow-empty", "-m", message)

    def test_clean_sources_bind_to_head_and_exact_working_bytes(self):
        commit, sources = verifier.committed_sources(self.root)
        self.assertEqual(commit, self.git("rev-parse", "HEAD"))
        self.assertEqual(sources, {
            path: hashlib.sha256((self.root / path).read_bytes()).hexdigest()
            for path in (self.INPUT, self.FIXTURE)
        })
        verifier.verify_sources_unchanged(self.root, commit, sources)

    def test_unstaged_source_edit_is_rejected(self):
        self.write(self.INPUT, "changed\n")
        with self.assertRaisesRegex(RuntimeError, self.INPUT):
            verifier.committed_sources(self.root)

    def test_main_rejects_dirty_sources_before_logs_or_receipt_are_written(self):
        receipt = self.root / "docs/explorer/contract-receipt.json"
        historical_receipt = b'{"historical": "preserve exactly"}\n'
        receipt.write_bytes(historical_receipt)
        self.write(self.INPUT, "changed\n")
        with patch.object(verifier, "ROOT", self.root), \
                patch.object(verifier.sys, "argv", ["verify_contract.py"]):
            with self.assertRaisesRegex(RuntimeError, self.INPUT):
                verifier.main()
        self.assertEqual(receipt.read_bytes(), historical_receipt)
        self.assertFalse((self.root / "build").exists())

    def test_staged_source_edit_is_rejected_even_when_index_matches_worktree(self):
        self.write(self.INPUT, "changed and staged\n")
        self.git("add", self.INPUT)
        self.assertEqual(self.git("diff", "--", self.INPUT), "")
        self.assertNotEqual(self.git("diff", "--cached", "--", self.INPUT), "")
        with self.assertRaisesRegex(RuntimeError, self.INPUT):
            verifier.committed_sources(self.root)

    def test_untracked_declared_input_is_rejected(self):
        new_input = "tests/new_input.mojo"
        self.write(new_input, "untracked\n")
        with patch.object(verifier, "SOURCE_PATHS", [self.INPUT, new_input]):
            with self.assertRaisesRegex(RuntimeError, new_input):
                verifier.committed_sources(self.root)

    def test_removed_declared_input_is_rejected(self):
        (self.root / self.INPUT).unlink()
        with self.assertRaisesRegex(RuntimeError, self.INPUT):
            verifier.committed_sources(self.root)

    def test_untracked_relevant_fixture_is_rejected(self):
        self.write("tests/compile_fail/token_selection_new.mojo", "new fixture\n")
        with self.assertRaises(RuntimeError):
            verifier.committed_sources(self.root)

    def test_deleted_fixture_cannot_disappear_from_the_initial_manifest(self):
        (self.root / self.FIXTURE).unlink()
        with self.assertRaises(RuntimeError):
            verifier.committed_sources(self.root)

    def test_unrelated_edits_allow_receipt_and_page_generation(self):
        self.write("docs/explorer/contract-receipt.json", "fresh receipt\n")
        self.git("add", "docs/explorer/contract-receipt.json")
        self.write("docs/explorer/index.html", "rebuilt page\n")
        self.write("notes/unrelated.md", "untracked unrelated note\n")
        commit, sources = verifier.committed_sources(self.root)
        self.write("docs/explorer/index.html", "page edited during verification\n")
        verifier.verify_sources_unchanged(self.root, commit, sources)

    def test_changed_source_after_capture_is_rejected(self):
        commit, sources = verifier.committed_sources(self.root)
        self.write(self.INPUT, "changed while tests ran\n")
        with self.assertRaisesRegex(RuntimeError, "source inputs changed during verification"):
            verifier.verify_sources_unchanged(self.root, commit, sources)

    def test_removed_source_after_capture_is_rejected(self):
        commit, sources = verifier.committed_sources(self.root)
        (self.root / self.INPUT).unlink()
        with self.assertRaisesRegex(RuntimeError, "source inputs changed during verification"):
            verifier.verify_sources_unchanged(self.root, commit, sources)

    def test_added_relevant_fixture_after_capture_is_rejected(self):
        commit, sources = verifier.committed_sources(self.root)
        self.write("tests/compile_fail/token_selection_new.mojo", "new fixture\n")
        with self.assertRaises(RuntimeError):
            verifier.verify_sources_unchanged(self.root, commit, sources)

    def test_removed_relevant_fixture_after_capture_is_rejected(self):
        commit, sources = verifier.committed_sources(self.root)
        (self.root / self.FIXTURE).unlink()
        with self.assertRaises(RuntimeError):
            verifier.verify_sources_unchanged(self.root, commit, sources)

    def test_unrelated_compile_fail_fixture_is_allowed(self):
        commit, sources = verifier.committed_sources(self.root)
        self.write("tests/compile_fail/other_contract.mojo", "unrelated fixture\n")
        verifier.verify_sources_unchanged(self.root, commit, sources)

    def test_moved_head_after_capture_is_rejected(self):
        commit, sources = verifier.committed_sources(self.root)
        self.commit("move HEAD without changing the declared inputs")
        with self.assertRaises(RuntimeError):
            verifier.verify_sources_unchanged(self.root, commit, sources)


class ExplorerSourceManifestTests(unittest.TestCase):
    def test_python_package_initializer_is_a_declared_input(self):
        self.assertIn("src/llm_mojo/__init__.py", verifier.SOURCE_PATHS)


if __name__ == "__main__":
    unittest.main()
