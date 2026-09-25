"""Package imports point down the dependency direction in docs/cli.md#code-ownership."""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src' / 'llm_mojo'
IGNORED = {'__init__', '__pycache__', 'py.typed'}

# Wiring and development tooling pick a model to run, measure or validate, so
# they may import anything; nothing below imports them.
TOP = {'cli', 'benchmarks', 'validation', 'configuration', 'validate', 'decoder_validation',
       'mlp_validation', 'model_validation'}
# Services may be used by any package and import only other services.
SERVICES = {'runtime', '_repository'}
# Beyond services, each remaining package imports only these. A model family may
# also import its own package, never another family.
BELOW = {
    'models': {'serving', 'layers', 'kernels'},
    'serving': {'layers', 'kernels'},
    'layers': {'kernels'},
    'kernels': set(),
}


def owner(path):
    """(package, model family) of a source file under src/llm_mojo."""
    parts = path.relative_to(PACKAGE).parts
    if len(parts) == 1:
        return Path(parts[0]).stem, None
    return parts[0], parts[1] if parts[0] == 'models' and len(parts) > 2 else None


def target(module):
    """The package an llm_mojo module belongs to; model imports keep their family."""
    parts = module.split('.')[1:]
    if not parts:
        return None
    return '.'.join(parts[:2]) if parts[0] == 'models' and len(parts) > 1 else parts[0]


def python_imports(path):
    """(line, module) for every llm_mojo import, including imports inside functions."""
    package = ['llm_mojo', *path.relative_to(PACKAGE).parent.parts]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                base = '.'.join(package[:len(package) - node.level + 1] + ([base] if base else []))
            modules = [base + '.' + alias.name for alias in node.names]
        else:
            continue
        for module in modules:
            if module.split('.')[0] == 'llm_mojo':
                yield node.lineno, module


def mojo_imports(path):
    """(line, module) for every llm_mojo import; docstrings and continuations are removed first."""
    text = path.read_text().replace('\\\n', '')
    text = re.sub(r'"""[\s\S]*?"""', lambda match: '\n' * match.group().count('\n'), text)
    for match in re.finditer(r'^\s*from\s+(llm_mojo(?:\.\w+)*)\s+import\s+(\([^)]*\)|[^\n#]+)', text, re.M):
        line = text.count('\n', 0, match.start()) + 1
        for member in match.group(2).strip('()').split(','):
            if member.split():
                yield line, match.group(1) + '.' + member.split()[0]
    for match in re.finditer(r'^\s*import\s+([^\n#]+)', text, re.M):
        line = text.count('\n', 0, match.start()) + 1
        for item in match.group(1).split(','):
            if item.split() and item.split()[0].split('.')[0] == 'llm_mojo':
                yield line, item.split()[0]


def edges():
    """(source path, line, source package, family, target) for every package import."""
    for path in sorted(PACKAGE.rglob('*')):
        if path.suffix not in ('.py', '.mojo') or '__pycache__' in path.parts:
            continue
        found = python_imports(path) if path.suffix == '.py' else mojo_imports(path)
        package, family = owner(path)
        for line, module in found:
            if target(module) is not None:
                yield path.relative_to(PACKAGE).as_posix(), line, package, family, target(module)


def allowed(package, family, imported):
    base = imported.split('.')[0]
    if package in TOP or base in SERVICES:
        return True
    if package == 'models':
        return imported == 'models.' + str(family) or base in BELOW['models']
    return base == package or base in BELOW.get(package, set())


def violations():
    return [(source, line, imported) for source, line, package, family, imported in edges()
            if not allowed(package, family, imported)]


class PackageBoundaryTests(unittest.TestCase):
    def test_every_package_has_a_place(self):
        placed = TOP | SERVICES | set(BELOW)
        found = {path.stem if path.suffix in ('.py', '.mojo') else path.name for path in PACKAGE.iterdir()}
        self.assertEqual(found - IGNORED - placed, set())

    def test_imports_follow_the_dependency_direction(self):
        found = list(edges())
        for suffix in ('.py', '.mojo'):
            self.assertGreater(sum(source.endswith(suffix) for source, *_ in found), 20)
        self.assertEqual([f'{source}:{line} imports llm_mojo.{imported}' for source, line, imported in violations()], [])

    def test_rules(self):
        self.assertTrue(allowed('cli', None, 'models.qwen2'))
        self.assertTrue(allowed('validation', None, 'benchmarks'))
        self.assertTrue(allowed('models', 'qwen2', 'models.qwen2'))
        self.assertTrue(allowed('models', 'qwen2', 'serving'))
        self.assertTrue(allowed('serving', None, 'layers'))
        self.assertTrue(allowed('kernels', None, 'runtime'))
        self.assertFalse(allowed('models', 'qwen2', 'models.llama'))
        self.assertFalse(allowed('serving', None, 'models.qwen2'))
        self.assertFalse(allowed('layers', None, 'serving'))
        self.assertFalse(allowed('kernels', None, 'layers'))
        self.assertFalse(allowed('runtime', None, 'models.qwen2'))
        self.assertFalse(allowed('runtime', None, 'validation'))
        self.assertFalse(allowed('models', 'qwen2', 'benchmarks'))
        self.assertEqual(target('llm_mojo.models.qwen2.assets.verify_prepared'), 'models.qwen2')
        self.assertEqual(target('llm_mojo.runtime.build.ensure_binary'), 'runtime')
        self.assertEqual(target('llm_mojo.configuration'), 'configuration')


if __name__ == '__main__':
    unittest.main()
