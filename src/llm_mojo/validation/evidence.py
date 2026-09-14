"""Shared source identity and JSON receipts for numerical validation."""
import hashlib
import json
from pathlib import Path

from .._repository import repository_root
from ..benchmarks.environment import repository_state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_identity():
    root = repository_root()
    paths = [p for base in ('src', 'tests') for p in (root / base).rglob('*')
             if p.is_file() and p.suffix in ('.mojo', '.py', '.json', '.gz', '.lock')]
    paths += [root / 'pyproject.toml', root / 'uv.lock']
    return dict(repository=repository_state(),
                sources={str(p.relative_to(root)): sha(p) for p in sorted(paths)})
