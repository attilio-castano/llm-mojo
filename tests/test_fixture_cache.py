"""The shared fixture cache with stand-in generators: no Torch, Metal or network."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from llm_mojo.runtime import toolchain
from llm_mojo.validation import fixtures

REPOSITORY = Path(__file__).resolve().parents[1]
# Stand-ins for the real families: one writes to its fixed checkout path, one to {output}.
IN_PLACE = fixtures.Family('flat', (('generate', 'flat'),), in_place=True, provenance=('last_run.json',),
                           regenerable=('metal_*_checks*.json',), estimated_bytes=1)
OUTPUT = fixtures.Family('nested', (('generate', 'nested', '--output', '{output}'),), in_place=False,
                         estimated_bytes=1)
TREE = {'top.npy': b'top', 'case/inner/deep.npy': b'deep' * 100, 'manifest.json': b'{"status": "complete"}'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *arguments):
    subprocess.run(['git', '-c', 'user.name=Fixture Test', '-c', 'user.email=fixtures@example.invalid',
                    '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', '-c', 'init.defaultBranch=main',
                    *arguments], cwd=root, check=True, capture_output=True)


def contents(directory):
    """Relative path -> bytes, or the target of a link, for everything below directory."""
    return {path.relative_to(directory).as_posix(): ('link', os.readlink(path)) if path.is_symlink()
            else path.read_bytes() if path.is_file() else 'directory'
            for path in sorted(Path(directory).rglob('*'))}


def writable(*paths):
    for path in paths:
        for directory, _, _ in os.walk(path):
            os.chmod(directory, 0o755)


class Generator:
    """Stands in for a Torch script: writes a nested tree and records each run."""

    def __init__(self, root):
        self.root, self.calls, self.variant, self.during = root, [], b'', None

    def __call__(self, *command):
        self.calls.append(command)
        if '--output' in command:
            output = Path(command[command.index('--output') + 1])
            output.mkdir(parents=True)
        else:
            output = self.root / 'build/oracle_data' / command[1]
            output.mkdir(parents=True, exist_ok=True)
            (output / 'last_run.json').write_text(json.dumps({'command': str(output)}))
        for name, data in TREE.items():
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            (output / name).write_bytes(data + self.variant if name == 'top.npy' else data)
        if self.during:
            self.during()


class CacheTests(unittest.TestCase):
    """A temporary store and git checkouts whose generator inputs live under gen/."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.store = self.base / 'store'
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(fixtures, 'SCOPE', ('gen',)))
        stack.enter_context(patch.object(toolchain, 'disk_check', return_value=toolchain.Check('space', True, 'ok')))
        self.generators = {}
        self.root = self.checkout('one')
        self.checkout_dir = self.root / 'build/oracle_data/flat'

    def checkout(self, name):
        root = self.base / name
        (root / 'gen/sub').mkdir(parents=True)
        (root / 'gen/generate.py').write_text('print("oracle")\n')
        (root / 'gen/sub/anchor.json').write_text('{"sha256": {}}\n')
        (root / 'gen/generate.py.lock').symlink_to('generate.py')
        (root / '.gitignore').write_text('__pycache__/\nbuild/\n')
        git(root, 'init', '-q')
        git(root, 'add', '.')
        git(root, 'commit', '-q', '-m', 'fixture inputs')
        return root

    def generator(self, root):
        return self.generators.setdefault(root, Generator(root))

    def key(self, root=None, family=IN_PLACE):
        return fixtures.identity(family, fixtures.inputs(root or self.root))

    def entry(self, family=IN_PLACE):
        return fixtures.entry_paths(self.store, family, self.key(family=family))

    def ensure(self, root=None, family=IN_PLACE, runner=None, **options):
        root = root or self.root
        with redirect_stdout(io.StringIO()) as output, redirect_stderr(output):
            try:
                return fixtures.ensure(family, root, fixtures.inputs(root), runner or self.generator(root),
                                       store_dir=self.store, **options)
            finally:
                self.output = output.getvalue()

    def test_the_key_follows_generator_inputs_only(self):
        first = self.key()
        self.assertEqual(self.key(self.checkout('two')), first)
        (self.root / 'gen/__pycache__').mkdir()
        (self.root / 'gen/__pycache__/generate.cpython-312.pyc').write_bytes(b'bytecode')
        (self.root / 'outside.txt').write_text('not a generator input')
        self.assertEqual(self.key(), first)
        seen = {first}

        def relink():
            (self.root / 'gen/generate.py.lock').unlink()
            (self.root / 'gen/generate.py.lock').symlink_to('sub/anchor.json')
        for edit in (lambda: (self.root / 'gen/generate.py').write_text('print("changed")\n'),
                     lambda: (self.root / 'gen/untracked.py').write_text(''),
                     relink,
                     lambda: (self.root / 'gen/sub/anchor.json').unlink()):
            edit()
            self.assertNotIn(self.key(), seen)
            seen.add(self.key())
        recipe = fixtures.Family('flat', (('generate', 'flat', '--compare'),), in_place=True)
        self.assertNotEqual(fixtures.identity(recipe, fixtures.inputs(self.root)), self.key())

    def test_a_miss_publishes_a_read_only_tree_and_links_the_checkout(self):
        self.assertEqual(self.ensure(), 'generated')
        entry = self.entry()
        self.assertEqual(os.readlink(self.checkout_dir), str(entry.tree))
        modes = {path.relative_to(entry.tree).as_posix(): stat.S_IMODE(path.lstat().st_mode)
                 for path in [entry.tree, *entry.tree.rglob('*')]}
        self.assertEqual(modes, {'.': 0o555, 'case': 0o555, 'case/inner': 0o555, 'case/inner/deep.npy': 0o444,
                                 'manifest.json': 0o444, 'top.npy': 0o444})
        record = json.loads(entry.record.read_text())
        self.assertEqual(record['files'], {name: dict(bytes=len(data), sha256=sha256(data))
                                           for name, data in TREE.items()})
        self.assertEqual(record['generation']['run_files'], {'last_run.json': {'command': str(self.checkout_dir)}})
        self.assertEqual(record['generation']['worktree'], str(self.root))
        self.assertEqual(record['inputs'], fixtures.inputs(self.root))
        self.assertEqual(stat.S_IMODE(entry.record.stat().st_mode), 0o444)
        self.assertEqual(list((self.store / '.staging').iterdir()), [])
        self.assertEqual(fixtures.verify(entry, self.key()), (record, None))

    def test_a_hit_runs_nothing_and_writes_nothing_to_the_store(self):
        self.ensure()
        entry = self.entry()
        entry.lock.unlink()
        for directory in (self.store, self.store / 'fixtures', entry.tree.parent, self.store / '.staging'):
            os.chmod(directory, 0o555)
        self.addCleanup(writable, self.store)
        other = self.checkout('two')
        self.assertEqual(self.ensure(other), 'hit')
        self.assertEqual(self.generator(other).calls, [])
        self.assertFalse(entry.lock.exists())
        self.assertEqual(os.readlink(other / 'build/oracle_data/flat'), str(entry.tree))
        self.assertIn('verified cached fixtures', self.output)
        self.assertNotIn('note:', self.output)
        with patch.object(fixtures, 'machine', return_value=dict(python='3.12.99', mac_ver='99.0', platform='x')):
            self.assertEqual(self.ensure(other), 'hit')
        self.assertIn('this Mac runs macOS 99.0 with Python 3.12.99', self.output)

    def test_an_output_family_generates_only_into_fresh_staging(self):
        seen = []

        def generate(*command):
            output = Path(command[-1])
            seen.append((output.parent, output.exists()))
            self.generator(self.root)(*command)
        self.assertEqual(self.ensure(family=OUTPUT, runner=generate), 'generated')
        self.assertEqual(seen, [(self.store / '.staging', False)])
        entry = self.entry(OUTPUT)
        self.assertEqual(os.readlink(self.root / 'build/oracle_data/nested'), str(entry.tree))
        self.assertEqual(contents(entry.tree), {**{name: data for name, data in TREE.items()},
                                                'case': 'directory', 'case/inner': 'directory'})
        self.assertEqual(list((self.store / '.staging').iterdir()), [])

    def test_damaged_entries_are_set_aside_and_regenerated(self):
        def replace_with_link(tree, record):
            (tree / 'top.npy').unlink()
            (tree / 'top.npy').symlink_to(tree / 'manifest.json')
        damage = {'byte': lambda tree, record: (tree / 'top.npy').write_bytes(b'toq'),
                  'missing': lambda tree, record: (tree / 'manifest.json').unlink(),
                  'extra': lambda tree, record: (tree / 'case/extra.npy').write_bytes(b''),
                  'link': replace_with_link,
                  'record': lambda tree, record: record.write_text('{')}
        generator = self.generator(self.root)
        for kind, apply in damage.items():
            with self.subTest(damage=kind):
                self.ensure()
                entry = self.entry()
                writable(entry.tree)
                for path in (entry.record, *entry.tree.rglob('*')):
                    if not path.is_symlink():
                        os.chmod(path, 0o755 if path.is_dir() else 0o644)
                apply(entry.tree, entry.record)
                runs = len(generator.calls)
                self.assertEqual(self.ensure(), 'generated')
                self.assertEqual(len(generator.calls), runs + 1)
                self.assertIn('the cached entry is damaged', self.output)
                self.assertIsNone(fixtures.verify(entry, self.key())[1])
                self.assertTrue(list(entry.tree.parent.glob(f'{entry.tree.name}.invalid-*')))

    def test_failures_and_interrupts_publish_nothing_and_restore_the_checkout(self):
        (self.base / 'elsewhere').mkdir()
        generator = self.generator(self.root)

        def prepare(previous):
            if self.checkout_dir.is_symlink():
                self.checkout_dir.unlink()
            elif self.checkout_dir.exists():
                shutil.rmtree(self.checkout_dir)
            self.checkout_dir.parent.mkdir(parents=True, exist_ok=True)
            if previous == 'link':
                self.checkout_dir.symlink_to(self.base / 'elsewhere')
            elif previous == 'directory':
                self.checkout_dir.mkdir()
                (self.checkout_dir / 'capture.npy').write_bytes(b'hand made')
        for previous in ('missing', 'link', 'directory'):
            for failure in (RuntimeError('generator failed'), KeyboardInterrupt()):
                with self.subTest(previous=previous, failure=type(failure).__name__):
                    prepare(previous)
                    before = contents(self.root / 'build')

                    def failing(*command):
                        generator(*command)
                        raise failure
                    with self.assertRaises(type(failure)):
                        self.ensure(runner=failing)
                    self.assertEqual(contents(self.root / 'build'), before)
                    entry = self.entry()
                    self.assertFalse(entry.tree.exists() or entry.record.exists())
                    self.assertEqual(list((self.store / '.staging').iterdir()), [])

    def test_inputs_edited_during_generation_are_refused(self):
        self.generator(self.root).during = lambda: (self.root / 'gen/generate.py').write_text('edited\n')
        key = self.key()
        with self.assertRaisesRegex(RuntimeError, 'inputs changed while generating flat'):
            self.ensure()
        entry = fixtures.entry_paths(self.store, IN_PLACE, key)
        self.assertFalse(entry.tree.exists() or entry.record.exists())
        self.assertFalse(self.checkout_dir.exists() or self.checkout_dir.is_symlink())

    def test_real_directories_are_removed_only_when_regenerable(self):
        self.ensure()
        entry = self.entry()

        def copy_from_validate(*extra):
            self.checkout_dir.unlink()
            self.generator(self.root)('generate', 'flat')
            for name in ('metal_development_mlp_checks_v0.json', '.DS_Store', 'case/.DS_Store', *extra):
                (self.checkout_dir / name).write_text('{}')
        copy_from_validate()
        self.assertEqual(self.ensure(), 'hit')
        self.assertEqual(os.readlink(self.checkout_dir), str(entry.tree))
        self.assertEqual(list(self.checkout_dir.parent.glob('flat.local-*')), [])
        for extra in ('holdout_manifest.json', 'top.npy'):
            with self.subTest(extra=extra):
                copy_from_validate(extra)
                before = contents(self.checkout_dir)
                self.assertEqual(self.ensure(), 'hit')
                [kept] = self.checkout_dir.parent.glob('flat.local-*')
                self.assertEqual(contents(kept), before)
                self.assertIn(f'kept {kept}', self.output)
                shutil.rmtree(kept)
        (self.base / 'elsewhere').mkdir()
        for target in (self.base / 'elsewhere', self.base / 'missing'):
            with self.subTest(link=target.name):
                self.checkout_dir.unlink()
                self.checkout_dir.symlink_to(target)
                self.assertEqual(self.ensure(), 'hit')
                self.assertEqual(os.readlink(self.checkout_dir), str(entry.tree))

    def test_a_stashed_copy_is_settled_after_generation(self):
        self.generator(self.root)('generate', 'flat')
        (self.checkout_dir / 'capture.npy').write_bytes(b'hand made')
        self.assertEqual(self.ensure(), 'generated')
        [kept] = self.checkout_dir.parent.glob('flat.local-*')
        self.assertEqual((kept / 'capture.npy').read_bytes(), b'hand made')
        self.assertEqual(os.readlink(self.checkout_dir), str(self.entry().tree))

    def test_regeneration_accepts_identical_output_and_keeps_a_differing_tree_aside(self):
        self.ensure()
        entry, generator = self.entry(), self.generator(self.root)
        self.assertEqual(self.ensure(regenerate=True), 'reproduced')
        self.assertIn('reproduced byte-for-byte', self.output)
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(list(entry.tree.parent.glob('*.regenerated-*')), [])
        before = contents(entry.tree)
        generator.variant = b'!'
        with self.assertRaisesRegex(RuntimeError, r'differ from the cached entry: top\.npy\. The cached entry'):
            self.ensure(regenerate=True)
        [kept] = entry.tree.parent.glob(f'{entry.tree.name}.regenerated-*')
        self.assertEqual((kept / 'top.npy').read_bytes(), b'top!')
        self.assertEqual(contents(entry.tree), before)
        self.assertEqual(os.readlink(self.checkout_dir), str(entry.tree))
        self.assertIsNone(fixtures.verify(entry, self.key())[1])
        self.assertEqual(list((self.store / '.staging').iterdir()), [])

    def test_concurrent_checkouts_generate_once(self):
        other = self.checkout('two')
        started, release, results = threading.Event(), threading.Event(), {}
        first = self.generator(self.root)

        def slow(*command):
            started.set()
            release.wait(10)
            first(*command)

        def run(name, root, runner):
            results[name] = fixtures.ensure(IN_PLACE, root, fixtures.inputs(root), runner, store_dir=self.store)
        with redirect_stdout(io.StringIO()) as output:
            threads = [threading.Thread(target=run, args=('first', self.root, slow)),
                       threading.Thread(target=run, args=('second', other, self.generator(other)))]
            threads[0].start()
            started.wait(10)
            threads[1].start()
            deadline = time.monotonic() + 10
            while 'waiting for another validation' not in output.getvalue() and time.monotonic() < deadline:
                time.sleep(0.01)
            release.set()
            for thread in threads:
                thread.join(10)
        self.assertEqual(results, {'first': 'generated', 'second': 'hit'})
        self.assertEqual((len(first.calls), self.generator(other).calls), (1, []))
        self.assertIn('waiting for another validation', output.getvalue())

    def test_low_disk_changes_nothing(self):
        self.generator(self.root)('generate', 'flat')
        before = contents(self.root / 'build')
        short = toolchain.Check('space', False, '0.1 GB free, 1.0 GB needed at /store')
        with patch.object(toolchain, 'disk_check', return_value=short):
            with self.assertRaisesRegex(RuntimeError, 'Not enough free space to generate flat'):
                self.ensure()
        self.assertEqual(contents(self.root / 'build'), before)
        entry = self.entry()
        self.assertFalse(entry.tree.exists() or entry.record.exists() or (self.store / '.staging').exists())
        self.assertEqual(len(self.generator(self.root).calls), 1)

    def test_a_store_on_another_volume_receives_a_clone(self):
        real = os.rename

        def rename(source, destination, *args, **kwargs):
            if Path(source) == self.checkout_dir and Path(destination).parent == self.store / '.staging':
                raise OSError(errno.EXDEV, 'Cross-device link')
            return real(source, destination, *args, **kwargs)
        with patch('os.rename', rename):
            self.assertEqual(self.ensure(), 'generated')
        entry = self.entry()
        self.assertIsNone(fixtures.verify(entry, self.key())[1])
        self.assertEqual(os.readlink(self.checkout_dir), str(entry.tree))
        self.assertEqual(list((self.store / '.staging').iterdir()), [])

    def test_without_the_cache_the_store_is_never_touched(self):
        self.ensure()
        before = contents(self.store)
        generator = self.generator(self.root)
        fixtures.generate_locally(IN_PLACE, self.root, generator)
        self.assertEqual(contents(self.store), before)
        self.assertFalse(self.checkout_dir.is_symlink())
        self.assertEqual((self.checkout_dir / 'top.npy').read_bytes(), b'top')
        self.assertTrue((self.checkout_dir / 'last_run.json').is_file())
        calls = []
        fixtures.generate_locally(OUTPUT, self.root, lambda *command: calls.append(command))
        self.assertEqual(calls, [('generate', 'nested', '--output', str(self.root / 'build/oracle_data/nested'))])

    def test_detach_swaps_the_link_for_a_writable_copy_until_the_next_validation(self):
        with patch.dict(fixtures.FAMILIES, {'flat': IN_PLACE}):
            self.assertIn('is not linked', fixtures.detach('flat', self.root))
            self.checkout_dir.parent.mkdir(parents=True)
            self.checkout_dir.symlink_to(self.base / 'pruned')
            self.assertIn('removed a dangling link', fixtures.detach('flat', self.root))
            self.assertFalse(self.checkout_dir.is_symlink())
            self.ensure()
            entry = self.entry()
            published = contents(entry.tree)
            self.assertIn('is now a writable copy', fixtures.detach('flat', self.root))
            self.assertFalse(self.checkout_dir.is_symlink())
            (self.checkout_dir / 'case/inner/capture.npy').write_bytes(b'manual')
            (self.checkout_dir / 'top.npy').write_bytes(b'new')
            self.assertEqual(contents(entry.tree), published)
            self.assertIn('already a writable copy', fixtures.detach('flat', self.root))
            with self.assertRaisesRegex(ValueError, 'unknown fixture family'):
                fixtures.detach('decoder', self.root)
        self.assertEqual(self.ensure(), 'hit')
        self.assertEqual(os.readlink(self.checkout_dir), str(entry.tree))
        [kept] = self.checkout_dir.parent.glob('flat.local-*')
        self.assertEqual((kept / 'case/inner/capture.npy').read_bytes(), b'manual')

    def test_the_sublayer_anchor_check_catches_a_changed_contract_or_array(self):
        root, directory = self.base / 'anchors', self.base / 'anchors/arrays'
        (root / 'tests/fixtures/attention_sublayer').mkdir(parents=True)
        directory.mkdir()
        frozen = dict(cases=[[2, 1, 4, 7, 17]], atol={'output': 0.03125}, rtol={'output': 0.03125},
                      array_sha256={'0_output.npy': sha256(b'output')}, numerical_contract={}, upstream_contract={})
        (root / 'tests/fixtures/attention_sublayer/checksums.json').write_text(json.dumps(frozen))
        (directory / '0_output.npy').write_bytes(b'output')
        manifest = directory / 'manifest.json'
        manifest.write_text(json.dumps({**frozen, 'upstream_reference': {'platform': 'any'}}))
        fixtures.check_sublayer_anchors(root, directory)
        manifest.write_text(json.dumps({**frozen, 'atol': {'output': 1}}))
        with self.assertRaisesRegex(RuntimeError, 'sublayer oracle changed'):
            fixtures.check_sublayer_anchors(root, directory)
        manifest.write_text(json.dumps(frozen))
        (directory / '0_output.npy').write_bytes(b'outpuT')
        with self.assertRaisesRegex(RuntimeError, 'sublayer oracle array changed: 0_output.npy'):
            fixtures.check_sublayer_anchors(root, directory)


class ValidationWiringTests(unittest.TestCase):
    def test_validation_takes_each_large_family_from_the_cache_in_the_old_order(self):
        from llm_mojo.models.qwen2 import tokenizer_assets
        from llm_mojo.validation import suite
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tests/fixtures').mkdir(parents=True)
            (root / 'tests/fixtures/checksums.json').write_text('{"sha256": {}}')
            for name in ('rms_norm', 'linear', 'rope', 'attention'):
                (root / 'build/oracle_data' / name).mkdir(parents=True)
            (root / 'build/oracle_data/attention/prefill_manifest.json').write_text('{"array_sha256": {}}')
            for cache, regenerate in ((True, False), (True, True), (False, False)):
                events = []
                with patch.object(suite, 'repository_root', return_value=root), \
                     patch.object(suite, 'run', side_effect=lambda *command: events.append(command[-1])), \
                     patch.object(fixtures, 'inputs', return_value={'inputs': 'identity'}), \
                     patch.object(fixtures, 'ensure', side_effect=lambda family, root, sources, runner, **options:
                                  events.append(('ensure', family.name, root, sources, runner is suite.run, options))), \
                     patch.object(fixtures, 'generate_locally', side_effect=lambda family, root, runner:
                                  events.append(('local', family.name, root, runner is suite.run))), \
                     patch.object(tokenizer_assets, 'ensure_prepared'), redirect_stdout(io.StringIO()):
                    suite.prepare(cache=cache, regenerate=regenerate)
                large = [event for event in events if isinstance(event, tuple)]
                if cache:
                    self.assertEqual(large, [('ensure', name, root, {'inputs': 'identity'}, True,
                                              {'regenerate': regenerate})
                                             for name in ('attention_sublayer', 'mlp', 'decoder_layer')])
                else:
                    self.assertEqual(large, [('local', name, root, True)
                                             for name in ('attention_sublayer', 'mlp', 'decoder_layer')])
                order = [event if isinstance(event, str) else event[1] for event in events]
                self.assertEqual(order[order.index('attention_sublayer'):order.index('decoder_layer') + 1],
                                 ['attention_sublayer', '--self-test', 'mlp', '--self-test', '--self-test',
                                  'decoder_layer'])
                self.assertNotIn('attention_precision', order)

    def test_suite_flags_choose_the_fixture_source(self):
        from llm_mojo.validation import suite
        with patch.object(suite, 'prepare') as prepare:
            for arguments in (['--prepare-only'], ['--prepare-only', '--regenerate-fixtures'],
                              ['--prepare-only', '--no-fixture-cache']):
                suite.main(arguments)
        self.assertEqual([call.kwargs for call in prepare.call_args_list],
                         [dict(cache=True, regenerate=False), dict(cache=True, regenerate=True),
                          dict(cache=False, regenerate=False)])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            suite.main(['--regenerate-fixtures', '--no-fixture-cache'])

    def test_mlp_check_records_never_write_into_the_linked_family(self):
        import mlp_support
        with tempfile.TemporaryDirectory() as directory:
            linked, records = Path(directory, 'oracle_data/mlp'), Path(directory, 'oracle_records/mlp')
            linked.mkdir(parents=True)
            os.chmod(linked, 0o555)
            environment = {k: v for k, v in os.environ.items() if k not in ('MLP_RECORD_DIR', 'MLP_SPLIT')}
            with patch.object(mlp_support, 'ROOT', linked), patch.object(mlp_support, 'RECORD_ROOT', records), \
                 patch.object(mlp_support, 'RECORDS', []), patch.dict(os.environ, environment, clear=True):
                mlp_support.record(dict(probe='silu_sweep', failed=0))
            self.assertEqual([path.name for path in records.iterdir()], ['metal_development_probes_checks.json'])
            self.assertEqual(list(linked.iterdir()), [])
        self.assertEqual(mlp_support.RECORD_ROOT, REPOSITORY / 'build/oracle_records/mlp')


class RecipeTests(unittest.TestCase):
    def test_recipes_are_the_suite_commands(self):
        uv = ('uv', 'run', '--locked', '--script')
        self.assertEqual({name: family.steps for name, family in fixtures.FAMILIES.items()}, {
            'attention_sublayer': ((*uv, 'tests/fixtures/generate.py', 'attention_sublayer'),
                                   ('check', 'sublayer_anchors'),
                                   (*uv, 'tests/fixtures/generate.py', 'attention_precision')),
            'mlp': ((*uv, 'tests/fixtures/generate.py', 'mlp'),),
            'decoder_layer': ((*uv, 'tests/fixtures/decoder_reference.py', '--output', '{output}'),),
        })

    def test_the_generators_own_source_lists_lie_inside_the_key_scope(self):
        mlp = json.loads((REPOSITORY / 'tests/fixtures/mlp/checksums.json').read_text())['source_sha256']
        decoder = json.loads((REPOSITORY / 'tests/fixtures/decoder_layer/checksums.json').read_text())['sources']
        sources = fixtures.inputs(REPOSITORY)
        for name in [*mlp, *decoder]:
            with self.subTest(source=name):
                self.assertTrue(any(name == scope or name.startswith(scope + '/') for scope in fixtures.SCOPE))
                self.assertIn(name, sources)
        self.assertEqual(sources['tests/fixtures/decoder_reference.py.lock'], 'link generate.py.lock')
        self.assertFalse([name for name in sources if '__pycache__' in name])


if __name__ == '__main__':
    unittest.main()
