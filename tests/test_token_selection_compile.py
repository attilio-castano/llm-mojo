"""Build host contracts and reject malformed primitives, reductions and finishing proofs."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from llm_mojo._repository import environment_tool


ROOT = Path(__file__).resolve().parents[1]
COMPILE_TIMEOUT_SECONDS = 180
# Full ownership/oracle control checks five geometries and both reduction plans.
# These gates also check the immutable Nat arithmetic certificate. Keep their
# compiler budgets separate from the smaller native execution timeout.
FOLD_COMPILE_TIMEOUT_SECONDS = 3600
TYPED_FOLD_FIXTURE_TIMEOUT_SECONDS = 3600


class TokenSelectionCompileTests(unittest.TestCase):
    def test_native_record_merge_matches_big_integer_oracle(self):
        # Load the sibling by its exact path: this also supports unittest
        # loaders that do not add tests/ to Python's module search path.
        import importlib.util

        oracle_path = ROOT / "tests/token_selection_record_oracle.py"
        specification = importlib.util.spec_from_file_location("token_selection_record_oracle", oracle_path)
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader)
        oracle = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(oracle)
        rows = oracle.cases()
        self.assertEqual(oracle.check_case_inventory(rows), 65)
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-native-record-") as directory:
            output = Path(directory)
            binary = output / "native-record"
            case_file = output / "cases.tsv"
            oracle.write_cases(case_file, rows)
            built = subprocess.run(
                [compiler, "build", "-I", "src", "tests/token_selection_record_probe.mojo",
                 "-o", str(binary)],
                cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            self.assertTrue(binary.is_file())
            executed = subprocess.run([str(binary), str(case_file)], cwd=ROOT,
                                      capture_output=True, text=True,
                                      timeout=COMPILE_TIMEOUT_SECONDS)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            checked_records = oracle.check_output(executed.stdout, rows)
            self.assertEqual(checked_records, oracle.EXPECTED_OUTPUT_RECORDS)
            print("Native UInt32 record oracle cases:", len(rows))
            print("Native UInt32 record oracle output records:", checked_records)
            print("Native UInt32 record oracle maximum Python encoded bits:", 65)

    def _check_contract(self, positive_source, mutants):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-contract-") as directory:
            output = Path(directory)

            def compile_source(source, name):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", source,
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=(FOLD_COMPILE_TIMEOUT_SECONDS
                             if source == "tests/test_argmax_schedule.mojo"
                             else TYPED_FOLD_FIXTURE_TIMEOUT_SECONDS
                             if Path(source).name in {
                                 "finishing_proof_unsupported_rule.mojo",
                                 "finishing_proof_weaker_cap.mojo",
                                 "token_selection_invalid_items.mojo",
                                 "argmax_fold_missing_input.mojo",
                             } else COMPILE_TIMEOUT_SECONDS),
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

    def test_schedule_builds_and_invalid_launch_shapes_are_rejected(self):
        self._check_contract("tests/test_argmax_schedule.mojo", (
            ("tests/compile_fail/token_selection_invalid_threads.mojo", "argmax schedule rejected"),
            ("tests/compile_fail/token_selection_invalid_items.mojo", "argmax schedule rejected"),
        ))

    def test_tiled_fold_builds_and_invalid_descriptions_are_rejected(self):
        self._check_contract("tests/test_argmax_schedule.mojo", tuple(
            (f"tests/compile_fail/argmax_fold_{name}.mojo", "argmax fold plan rejected")
            for name in (
                "zero_stride", "backward_stride", "overlap_stride", "gap_stride",
                "missing_input",
            )
        ))

    def test_finishing_proof_builds_and_invalid_derivations_are_rejected(self):
        self._check_contract("tests/test_finishing_proof.mojo", tuple(
            (f"tests/compile_fail/finishing_proof_{name}.mojo", "finishing proof rejected")
            for name in (
                "unsupported_rule", "forward_reference", "negative_reference",
                "dead_invalid_node", "wrong_equation", "wrong_descent",
                "missing_root", "invalid_root", "bad_start", "bad_guard",
                "bad_index", "bad_stride", "bad_headroom", "wrong_statement", "weaker_cap",
                "hole", "retired_coverage", "retired_visitation", "invalid_substitution",
                "induction_base", "induction_step", "eigenvariable", "discharge",
                "witness", "owner_boundary", "uniqueness", "capture", "missing_witness",
                "induction_context", "overflow", "negative_nat", "zero_divisor",
                "visited_count", "constant_claim_kind", "scope_escape", "open_root",
                "retired_termination_empty", "retired_termination_named", "retired_termination_atom",
                "stop_direction", "stop_round", "stop_count", "step_guard", "step_round", "step_count",
                "step_suffix", "step_cycle", "cases_order", "cases_conclusion", "cases_context",
                "termination_fuel", "termination_ih_bound", "termination_bound",
                "termination_open_assumption", "done_count_scope",
            )
        ))


if __name__ == "__main__":
    unittest.main()
