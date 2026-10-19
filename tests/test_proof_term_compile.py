"""Compile native typed proofs before checking focused admission failures."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from llm_mojo._repository import environment_tool


ROOT = Path(__file__).resolve().parents[1]
COMPILE_TIMEOUT_SECONDS = 180
# Compilation includes the immutable Nat arithmetic bootstrap and O3 codegen.
# Native execution has its own smaller bound. A valid concurrent coverage
# compilation exceeded 1,600 seconds; timeout is never a rejection diagnostic.
PROOF_COMPILE_TIMEOUT_SECONDS = 3000
TYPED_COVERAGE_COMPILE_TIMEOUT_SECONDS = 3600
NAT_ARITHMETIC_COMPILE_TIMEOUT_SECONDS = 3000
FINISHING_RECORD_COMPILE_TIMEOUT_SECONDS = 7200


class ProofTermCompileTests(unittest.TestCase):
    def test_finishing_checked_exports_are_bound(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-finishing-exports-") as directory:
            binary = Path(directory) / "valid"
            built = subprocess.run(
                [compiler, "build", "-j", "2", "-I", "src", "-I", "tests",
                 "tests/test_finishing_checked_exports.mojo", "-o", str(binary)],
                cwd=ROOT, capture_output=True, text=True,
                timeout=PROOF_COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            self.assertTrue(binary.is_file())
            executed = subprocess.run([str(binary)], cwd=ROOT, capture_output=True,
                                      text=True, timeout=COMPILE_TIMEOUT_SECONDS)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            self.assertEqual(executed.stdout.splitlines(),
                ["Checked immutable finishing exports and 19 changed-input controls passed"])
            for fixture in ("finishing_export_host_plan", "finishing_export_program_gate"):
                with self.subTest(fixture=fixture):
                    rejected_binary = Path(directory) / fixture
                    rejected = subprocess.run(
                        [compiler, "build", "-j", "2", "-I", "src", "-I", "tests",
                         f"tests/compile_fail/{fixture}.mojo", "-o", str(rejected_binary)],
                        cwd=ROOT, capture_output=True, text=True,
                        timeout=PROOF_COMPILE_TIMEOUT_SECONDS,
                    )
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(details, r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                                     + re.escape("argmax fold plan rejected") + r"\s*$")
                    self.assertFalse(rejected_binary.exists())

    def test_finishing_record_execution_is_checked(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-finishing-record-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=FINISHING_RECORD_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_record_execution.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run([str(output / "valid")], cwd=ROOT,
                                      capture_output=True, text=True,
                                      timeout=COMPILE_TIMEOUT_SECONDS)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            required_lines = (
                "Candidate183 native-control workspace 65536",
                "Earlier148 native-control workspace 40960",
                "Checked earlier148 Boolean semantic controls: 30",
                "Checked earlier148 Boolean resource preflights: 3",
                "Checked earlier148 assumed-bridge full typing positives: 1",
                "Checked record execution semantic rejection controls: 44",
                "Checked record execution resource preflight controls: 3",
                "Checked final-statement full-environment typing positives: 3",
                "Checked wrong-scope data declaration-prefix typing positives: 4",
                "Checked correlated record execution and independent controls passed",
            )
            self.assertEqual(executed.stdout.splitlines(), list(required_lines), executed.stdout)
            fixtures = sorted((ROOT / "tests/compile_fail").glob("finishing_record_*.mojo"))
            expected_names = {
                "finishing_record_assumed_final.mojo",
                "finishing_record_correlated_components.mojo",
                "finishing_record_global_ready_validity.mojo",
                "finishing_record_lost_bad.mojo",
                "finishing_record_metadata_budget.mojo",
                "finishing_record_unread_validity.mojo",
                "finishing_record_unused_malformed.mojo",
                "finishing_record_wrong_old_source.mojo",
            }
            self.assertEqual({source.name for source in fixtures}, expected_names)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertGreater(rejected.returncode, 0, details)
                    self.assertRegex(details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("checked correlated record execution rejected") + r"\s*$")
                    self.assertFalse((output / source.stem).exists())

    def test_finishing_join_is_checked(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-finishing-join-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=TYPED_COVERAGE_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_finishing_join.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run([str(output / "valid")], cwd=ROOT,
                                      capture_output=True, text=True,
                                      timeout=COMPILE_TIMEOUT_SECONDS)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            self.assertIn("Checked total Nat join and canonical finishing theorem passed",
                          executed.stdout)
            fixtures = sorted((ROOT / "tests/compile_fail").glob("finishing_join_*.mojo"))
            self.assertEqual(len(fixtures), 8)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertGreater(rejected.returncode, 0, details)
                    self.assertRegex(details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("checked finishing Nat join rejected") + r"\s*$")
                    self.assertFalse((output / source.stem).exists())

    def test_finishing_execution_production_is_bound(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-finishing-production-") as directory:
            binary = Path(directory) / "valid"
            accepted = subprocess.run(
                [compiler, "build", "-I", "src", "-I", "tests",
                 "tests/test_finishing_execution_production.mojo", "-o", str(binary)],
                cwd=ROOT, capture_output=True, text=True,
                timeout=TYPED_COVERAGE_COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue(binary.is_file())
            executed = subprocess.run([str(binary)], cwd=ROOT,
                                      capture_output=True, text=True,
                                      timeout=COMPILE_TIMEOUT_SECONDS)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            self.assertIn("Production finishing descriptor and certificate host bindings passed",
                          executed.stdout)

    def test_finishing_execution_is_checked(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-finishing-execution-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=TYPED_COVERAGE_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_finishing_execution.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run([str(output / "valid")], cwd=ROOT,
                                      capture_output=True, text=True,
                                      timeout=COMPILE_TIMEOUT_SECONDS)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            self.assertIn("Checked whole finishing execution and independent controls passed",
                          executed.stdout)
            fixtures = sorted((ROOT / "tests/compile_fail").glob("finishing_execution_*.mojo"))
            self.assertEqual(len(fixtures), 8)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertGreater(rejected.returncode, 0, details)
                    self.assertRegex(details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("checked finishing execution rejected") + r"\s*$")
                    self.assertFalse((output / source.stem).exists())

    def test_private_finishing_phase_commutation_is_checked(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-finishing-phase-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=PROOF_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_finishing_phase.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run([str(output / "valid")], cwd=ROOT,
                                      capture_output=True, text=True,
                                      timeout=COMPILE_TIMEOUT_SECONDS)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            fixtures = sorted((ROOT / "tests/compile_fail").glob("finishing_phase_*.mojo"))
            self.assertEqual(len(fixtures), 8)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertGreater(rejected.returncode, 0, details)
                    self.assertRegex(details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("checked finishing phase commutation rejected") + r"\s*$")
                    self.assertFalse((output / source.stem).exists())

    def test_headroom_is_checked_without_affine_or_order_reflection(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-nat-headroom-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=TYPED_COVERAGE_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_headroom.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("headroom_*.mojo"))
            self.assertEqual(len(fixtures), 8)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("checked Nat headroom rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())

    def test_literal_arithmetic_is_checked_through_raw_nat_conversion(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-literal-arithmetic-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=NAT_ARITHMETIC_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_literal_arithmetic.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("literal_arithmetic_*.mojo"))
            self.assertEqual(len(fixtures), 8)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("checked literal arithmetic conversion rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())

    def test_compact_numerals_are_checked_from_raw_nat_definitions(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-compact-numeral-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=NAT_ARITHMETIC_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_compact_numeral.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("compact_numeral_*.mojo"))
            self.assertEqual(len(fixtures), 8)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("checked compact numeral bridge rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())

    def test_operator_representations_are_checked_from_certified_nat_definitions(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-operator-representation-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=NAT_ARITHMETIC_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_operator_representation.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("operator_representation_*.mojo"))
            self.assertEqual(len(fixtures), 16)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("checked operator representation rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())

    def test_export_types_are_instantiated_from_checked_definitions(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-export-instantiation-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=NAT_ARITHMETIC_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_export_instantiation.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("export_instantiation_*.mojo"))
            self.assertEqual(len(fixtures), 12)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("checked export instantiation rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())

    def test_base_definition_and_derived_driver_equations_are_checked(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-base-definition-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=NAT_ARITHMETIC_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_base_definition.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("base_definition_*.mojo"))
            self.assertEqual(len(fixtures), 14)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("checked Base definition rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())

    def test_nat_arithmetic_definitions_and_derived_laws_are_checked(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-nat-arithmetic-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=NAT_ARITHMETIC_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_nat_arithmetic.mojo"), "valid")
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("nat_arithmetic_*.mojo"))
            self.assertEqual(len(fixtures), 29)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("native Nat arithmetic rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())

    def test_typed_proofs_build_and_malformed_terms_are_rejected(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-proof-terms-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=(TYPED_COVERAGE_COMPILE_TIMEOUT_SECONDS
                             if source.name == "proof_term_fold_admission.mojo"
                             else PROOF_COMPILE_TIMEOUT_SECONDS),
                )

            accepted = compile_source(Path("tests/test_proof_terms.mojo"), "valid")
            self.assertEqual(
                accepted.returncode, 0,
                "Valid proof terms must compile before rejection is checked.\n"
                + accepted.stdout + accepted.stderr,
            )
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("proof_term_*.mojo"))
            self.assertEqual(len(fixtures), 30)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    # Match the actual compiler diagnostic. Echoing an assertion
                    # after an unrelated import or parse error is insufficient.
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("proof term rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())

    def test_inductive_definitions_build_and_invalid_admission_is_rejected(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-inductive-proofs-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=(TYPED_COVERAGE_COMPILE_TIMEOUT_SECONDS
                             if source.name == "inductive_legacy_done_in_coverage.mojo"
                             else PROOF_COMPILE_TIMEOUT_SECONDS),
                )

            accepted = compile_source(Path("tests/test_inductive_proofs.mojo"), "valid")
            self.assertEqual(
                accepted.returncode, 0,
                "Valid generic inductives must compile before rejection is checked.\n"
                + accepted.stdout + accepted.stderr,
            )
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("inductive_*.mojo"))
            self.assertEqual(len(fixtures), 28)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("inductive admission rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())

    def test_inductive_eliminators_build_and_invalid_admission_is_rejected(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-inductive-eliminators-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=PROOF_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_inductive_eliminator.mojo"), "valid")
            self.assertEqual(
                accepted.returncode, 0,
                "Valid dependent eliminators must compile before rejection is checked.\n"
                + accepted.stdout + accepted.stderr,
            )
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("eliminator_*.mojo"))
            self.assertEqual(len(fixtures), 28)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("eliminator admission rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())


    def test_nat_definitions_build_and_invalid_induction_is_rejected(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-nat-inductive-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=PROOF_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_nat_inductive.mojo"), "valid")
            self.assertEqual(
                accepted.returncode, 0,
                "Valid Nat induction must compile before rejection is checked.\n"
                + accepted.stdout + accepted.stderr,
            )
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("nat_inductive_*.mojo"))
            self.assertEqual(len(fixtures), 33)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("Nat inductive admission rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())


    def test_typed_coverage_builds_and_invalid_ownership_is_rejected(self):
        compiler = environment_tool("mojo")
        with tempfile.TemporaryDirectory(prefix="llm-mojo-typed-coverage-") as directory:
            output = Path(directory)

            def compile_source(source: Path, name: str):
                return subprocess.run(
                    [compiler, "build", "-I", "src", "-I", "tests", str(source),
                     "-o", str(output / name)],
                    cwd=ROOT, capture_output=True, text=True,
                    timeout=TYPED_COVERAGE_COMPILE_TIMEOUT_SECONDS,
                )

            accepted = compile_source(Path("tests/test_typed_coverage.mojo"), "valid")
            self.assertEqual(
                accepted.returncode, 0,
                "Valid typed coverage must compile before rejection is checked.\n"
                + accepted.stdout + accepted.stderr,
            )
            self.assertTrue((output / "valid").is_file())
            executed = subprocess.run(
                [str(output / "valid")], cwd=ROOT, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

            fixtures = sorted((ROOT / "tests/compile_fail").glob("typed_coverage_*.mojo"))
            self.assertEqual(len(fixtures), 33)
            for source in fixtures:
                with self.subTest(mutant=source.stem):
                    rejected = compile_source(source, source.stem)
                    details = rejected.stdout + rejected.stderr
                    self.assertNotEqual(rejected.returncode, 0, details)
                    self.assertRegex(
                        details,
                        r"(?m)^.*\b(?:error|note):\s*constraint failed:\s*"
                        + re.escape("typed coverage rejected") + r"\s*$",
                    )
                    self.assertFalse((output / source.stem).exists())


if __name__ == "__main__":
    unittest.main()
