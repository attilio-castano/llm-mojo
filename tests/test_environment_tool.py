"""Direct launches must stay in the selected environment, regardless of PATH."""

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
