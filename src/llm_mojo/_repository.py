"""Locate the source checkout for commands that build or validate it."""
from pathlib import Path
import sys


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file() or not (root / "uv.lock").is_file():
        raise RuntimeError(
            "This command requires the llm-mojo source checkout; "
            "run it there with uv run --locked python -m llm_mojo.<command>."
        )
    return root


def environment_tool(name: str) -> str:
    """Use tools installed beside this Python, even if PATH names another toolchain."""
    tool = Path(sys.executable).parent / name
    if not tool.is_file():
        raise RuntimeError(f"Missing {name} in the current environment; run uv sync --locked.")
    return str(tool)
