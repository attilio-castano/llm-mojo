"""Cached native builds with an explicit local Mojo import closure."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from llm_mojo._repository import repository_root, environment_tool
from llm_mojo.runtime.artifacts import setup_lock, atomic_write, sha


def build_sources(entrypoint, root=None):
    root = root or repository_root()
    source_root = root / 'src'
    pending = [root / entrypoint]
    for parent in (root / entrypoint).parents:
        if parent == source_root:
            break
        initializer = parent / "__init__.mojo"
        if initializer.is_file():
            pending.append(initializer)
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise ValueError('missing native build dependency: ' + str(path))
        visited.add(path)
        # Native project imports use absolute module names, including imports
        # inside comptime branches. Package initializers are dependencies too.
        content = path.read_text()
        if re.search(r'^\s*from\s+\.', content, re.MULTILINE):
            raise ValueError('native build tracking requires absolute imports: ' + str(path))
        imports = []
        content = content.replace('\\\n', '')
        for match in re.finditer(r'^\s*from\s+(llm_mojo(?:\.\w+)*)\s+import\s+(\([^)]*\)|[^\n#]+)', content, re.MULTILINE):
            name, members = match.groups()
            imports.append(name)
            package = source_root.joinpath(*name.split('.'))
            if package.is_dir():
                for member in members.strip('()').split(','):
                    words = member.strip().split()
                    if not words:
                        continue
                    candidate = package / words[0]
                    if candidate.with_suffix('.mojo').is_file() or (candidate / '__init__.mojo').is_file():
                        imports.append(name + '.' + words[0])
        for statement in re.findall(r'^\s*import\s+([^\n#]+)', content, re.MULTILINE):
            for item in statement.split(','):
                words = item.strip().split()
                if words and (words[0] == 'llm_mojo' or words[0].startswith('llm_mojo.')):
                    imports.append(words[0])
        for name in imports:
            module = source_root.joinpath(*name.split('.'))
            dependency = module.with_suffix('.mojo')
            pending.append(dependency if dependency.is_file() else module / '__init__.mojo')
            for parent in module.parents:
                if parent == source_root:
                    break
                initializer = parent / '__init__.mojo'
                if initializer.is_file():
                    pending.append(initializer)
    visited.add(root / 'uv.lock')
    return {str(p.relative_to(root)): sha(p) for p in sorted(visited)}


def _current(binary, receipt, identity):
    if not binary.exists() or not receipt.exists():
        return False
    try:
        record = json.loads(receipt.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and record.get('sources') == identity and record.get('binary_sha256') == sha(binary)


def binary_status(name, entrypoint):
    """current, stale or missing, without locking, creating or building anything."""
    root = repository_root()
    directory = root / 'build' / name
    binary, receipt = directory / name, directory / 'binary.json'
    if not binary.exists() or not receipt.exists():
        return 'missing'
    try:
        identity = build_sources(entrypoint, root)
    except ValueError:
        return 'stale'
    return 'current' if _current(binary, receipt, identity) else 'stale'


def ensure_binary(name, entrypoint):
    root = repository_root()
    directory = root / 'build' / name
    binary, receipt = directory / name, directory / 'binary.json'
    with setup_lock(directory):
        identity = build_sources(entrypoint, root)
        if _current(binary, receipt, identity):
            return binary
        print(f'Building native {name} (reused on subsequent launches)…', flush=True, file=sys.stderr)
        fd, filename = tempfile.mkstemp(prefix=name + '.', suffix='.part', dir=directory)
        os.close(fd)
        temporary = Path(filename)
        try:
            subprocess.run([environment_tool('mojo'), 'build', '-I', 'src', entrypoint,
                            '-o', str(temporary)], cwd=root, check=True)
            if identity != build_sources(entrypoint, root):
                raise ValueError('native source changed during compilation')
            digest = sha(temporary)
            os.replace(temporary, binary)
            atomic_write(receipt, (json.dumps(dict(sources=identity, binary_sha256=digest), indent=2) + '\n').encode())
        finally:
            temporary.unlink(missing_ok=True)
    return binary
