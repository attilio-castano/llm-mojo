"""Locate the source checkout for commands that build or validate it."""
from pathlib import Path


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file() or not (root / "uv.lock").is_file():
        raise RuntimeError(
            "This command requires the llm-mojo source checkout; "
            "run it there with uv run --locked python -m llm_mojo.<command>."
        )
    return root
