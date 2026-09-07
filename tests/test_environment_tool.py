"""Direct launches must stay in the selected environment, regardless of PATH."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_mojo._repository import environment_tool


class EnvironmentToolTests(unittest.TestCase):
    def test_uses_current_python_environment_without_path_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active, other = root / "active", root / "other"
            active.mkdir()
            other.mkdir()
            (other / "mojo").touch()
            with patch("llm_mojo._repository.sys.executable", str(active / "python")), \
                 patch.dict("os.environ", {"PATH": str(other)}):
                with self.assertRaisesRegex(RuntimeError, "Missing mojo"):
                    environment_tool("mojo")
                (active / "mojo").touch()
                self.assertEqual(environment_tool("mojo"), str(active / "mojo"))


class FixtureDriverTests(unittest.TestCase):
    def driver(self):
        path = Path(__file__).resolve().parent / 'fixtures/generate.py'
        spec = importlib.util.spec_from_file_location('fixture_driver', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_default_never_downloads_a_checkpoint_and_explicit_arguments_are_forwarded(self):
        module = self.driver()
        with patch.object(module.sys, 'argv', ['generate.py']), patch.object(module.subprocess, 'run') as run:
            module.main()
            self.assertEqual([Path(c.args[0][1]).parent.name for c in run.call_args_list],
                             ['rms_norm', 'linear', 'rope', 'attention'])
            self.assertTrue(all(len(c.args[0]) == 2 for c in run.call_args_list))
        with (patch.object(module.sys, 'argv', ['generate.py', 'attention_checkpoint', '--', '--attention-prefix']),
              patch.object(module.subprocess, 'run') as run):
            module.main()
            command = run.call_args.args[0]
            self.assertEqual(Path(command[1]).name, 'checkpoint.py')
            self.assertEqual(command[2:], ['--attention-prefix'])
            self.assertNotIn('--download', command)

    def test_invalid_batch_is_rejected_before_any_generator_runs(self):
        module = self.driver()
        for args in (['rms_norm', 'unknown'], ['--', '--download'],
                     ['attention_sublayer', 'attention_precision', '--', '--holdout']):
            with (patch.object(module.sys, 'argv', ['generate.py', *args]),
                  patch.object(module.subprocess, 'run') as run,
                  contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit):
                    module.main()
                run.assert_not_called()
