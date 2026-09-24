"""Shared model store, toolchain checks and one-step setup without network or Metal."""
import ast
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from llm_mojo.models.qwen2 import assets, tokenizer_assets
from llm_mojo.runtime import store, toolchain


class Stream(io.BytesIO):
    """A download response that records how much was read."""
    consumed = 0

    def read(self, *args):
        data = super().read(*args)
        self.consumed += len(data)
        return data


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_store_root_precedence(self):
        with patch.dict(os.environ, {'LLM_MOJO_CACHE_DIR': str(self.root / 'env'),
                                     'XDG_CACHE_HOME': str(self.root / 'xdg')}):
            self.assertEqual(store.store_root(self.root / 'explicit'), self.root / 'explicit')
            self.assertEqual(store.store_root(), self.root / 'env')
            del os.environ['LLM_MOJO_CACHE_DIR']
            self.assertEqual(store.store_root(), self.root / 'xdg/llm-mojo')
            del os.environ['XDG_CACHE_HOME']
            with patch.object(store.Path, 'home', return_value=self.root / 'home'):
                self.assertEqual(store.store_root(), self.root / 'home/.cache/llm-mojo')

    def download(self, payload, size=None, digest=None, response=None):
        target = self.root / 'store/file.bin'
        opener = lambda url, timeout: response or Stream(payload)
        return target, store.download('https://example.invalid/file', target,
                                      len(payload) if size is None else size,
                                      digest or hashlib.sha256(payload).hexdigest(), opener=opener)

    def test_download_publishes_only_a_verified_read_only_file(self):
        target, published = self.download(b'pinned bytes' * 1000)
        self.assertEqual(published, target)
        self.assertEqual(target.read_bytes(), b'pinned bytes' * 1000)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o444)
        self.assertTrue(store.verified(target, 12000, hashlib.sha256(b'pinned bytes' * 1000).hexdigest()))
        self.assertEqual(list(target.parent.glob('*.part')), [])

    def test_mismatch_oversize_and_interruption_leave_nothing(self):
        class Interrupted(io.BytesIO):
            def read(self, *args):
                raise ConnectionError('interrupted')

        cases = [
            (dict(payload=b'abc', digest='0' * 64), ValueError),
            (dict(payload=b'abcdef', size=3), ValueError),
            (dict(payload=b'ab', size=3), ValueError),
            (dict(payload=b'abc', response=Interrupted()), ConnectionError),
        ]
        for arguments, error in cases:
            with self.subTest(arguments=arguments), self.assertRaises(error):
                self.download(**arguments)
            directory = self.root / 'store'
            self.assertFalse((directory / 'file.bin').exists())
            self.assertEqual(list(directory.glob('*.part')), [])

    def test_oversize_download_stops_reading(self):
        response = Stream(b'x' * (3 * store.CHUNK))
        with self.assertRaisesRegex(ValueError, 'pinned size'):
            self.download(b'x', size=store.CHUNK, response=response)
        self.assertLessEqual(response.consumed, 2 * store.CHUNK)

    def test_links_never_replace_real_entries(self):
        target = self.root / 'store/model'
        target.mkdir(parents=True)
        other = self.root / 'store/other'
        other.mkdir()
        link = self.root / 'checkout/build/model'
        self.assertEqual(store.link_state(link, target), 'missing')
        self.assertEqual(store.ensure_link(link, target), 'linked')
        self.assertEqual(store.ensure_link(link, target), 'linked')
        self.assertEqual(link.resolve(), target)
        link.unlink(); link.symlink_to(other)
        self.assertEqual(store.link_state(link, target), 'foreign')
        self.assertEqual(store.ensure_link(link, target), 'relinked')
        self.assertEqual(link.resolve(), target)
        link.unlink(); link.symlink_to(self.root / 'gone')
        self.assertEqual(store.link_state(link, target), 'dangling')
        self.assertEqual(store.ensure_link(link, target), 'relinked')
        for real in ('file', 'directory'):
            with self.subTest(real=real):
                path = self.root / 'checkout/build' / real
                path.write_text('local') if real == 'file' else path.mkdir()
                self.assertEqual(store.ensure_link(path, target), 'local')
                self.assertFalse(path.is_symlink())
        self.assertEqual(sorted(p.name for p in link.parent.iterdir()), ['directory', 'file', 'model'])

    def test_clone_falls_back_to_an_ordinary_copy(self):
        source = self.root / 'source'
        source.mkdir()
        (source / 'tensor.bin').write_bytes(b'data')
        destination = store.staging_path(self.root / 'store', 'model')
        with patch.object(store.subprocess, 'run', side_effect=OSError('no clonefile')):
            store.clone(source, destination)
        self.assertEqual((destination / 'tensor.bin').read_bytes(), b'data')
        self.assertEqual((source / 'tensor.bin').read_bytes(), b'data')

    def test_published_directories_are_read_only_and_staging_is_removable(self):
        staged = store.staging_path(self.root / 'store', 'model')
        staged.mkdir()
        (staged / 'tensor.bin').write_bytes(b'data')
        final = store.publish_directory(staged, self.root / 'store/qwen/model')
        self.assertEqual(stat.S_IMODE(final.stat().st_mode), 0o555)
        self.assertEqual(stat.S_IMODE((final / 'tensor.bin').stat().st_mode), 0o444)
        with self.assertRaises(PermissionError):
            (final / 'tensor.bin').unlink()
        with redirect_stderr(io.StringIO()) as message:
            aside = store.set_aside(final)
        self.assertIn(str(aside), message.getvalue())
        self.assertFalse(final.exists())
        store.remove_staged(aside)
        self.assertFalse(aside.exists())

    def test_nested_trees_are_read_only_at_every_depth_and_removable(self):
        staged = store.staging_path(self.root / 'store', 'fixtures')
        (staged / 'case/inner').mkdir(parents=True)
        (staged / 'top.bin').write_bytes(b'a')
        (staged / 'case/inner/deep.npy').write_bytes(b'b')
        final = store.publish_directory(staged, self.root / 'store/fixtures/family/key')
        modes = {path.relative_to(final).as_posix(): stat.S_IMODE(path.stat().st_mode)
                 for path in [final, *final.rglob('*')]}
        self.assertEqual(modes, {'.': 0o555, 'top.bin': 0o444, 'case': 0o555,
                                 'case/inner': 0o555, 'case/inner/deep.npy': 0o444})
        with self.assertRaises(PermissionError):
            (final / 'case/inner/new.npy').write_bytes(b'c')
        with redirect_stderr(io.StringIO()):
            aside = store.set_aside(final)
        store.remove_staged(aside)
        self.assertFalse(aside.exists())


class ToolchainTests(unittest.TestCase):
    OUTPUTS = {
        ('xcode-select', '-p'): (0, '/Applications/Xcode.app/Contents/Developer'),
        ('xcodebuild', '-version'): (0, 'Xcode 26.6\nBuild version 17F113'),
        ('xcrun', '-f', 'metal'): (0, '/toolchain/usr/bin/metal'),
        ('xcrun', '-f', 'metallib'): (0, '/toolchain/usr/bin/metallib'),
        ('xcodebuild', '-showComponent', 'MetalToolchain', '-json'):
            (0, '{\n  "buildVersion" : "17F109",\n  "status" : "installed"\n}'),
        ('sysctl', '-n', 'machdep.cpu.brand_string'): (0, 'Apple M4 Pro'),
    }

    def runner(self, changes=None):
        outputs = {**self.OUTPUTS, **(changes or {})}
        return lambda *command: outputs.get(command, (1, 'not found'))

    def test_documented_toolchain_passes(self):
        checks = toolchain.xcode_checks(self.runner())
        self.assertEqual([c.name for c in checks], ['Xcode selected', 'Xcode 16 or later',
                                                    'Metal compiler', 'Metal toolchain component'])
        self.assertTrue(all(c.ok for c in checks), checks)

    def test_each_missing_prerequisite_names_its_remedy(self):
        cases = {
            ('xcode-select', '-p'): (0, '/Library/Developer/CommandLineTools'),
            ('xcodebuild', '-version'): (1, 'xcode-select: error: tool requires Xcode'),
            ('xcrun', '-f', 'metallib'): (1, 'unable to find utility'),
            ('xcodebuild', '-showComponent', 'MetalToolchain', '-json'): (0, '{"status" : "uninstalled"}'),
        }
        for command, output in cases.items():
            with self.subTest(command=command):
                failed = [c for c in toolchain.xcode_checks(self.runner({command: output})) if not c.ok]
                self.assertEqual(len(failed), 1, failed)
                self.assertTrue(failed[0].remedy)
                self.assertEqual(failed[0].blocks, 'build')

    def test_older_xcode_has_no_separate_metal_component(self):
        checks = toolchain.xcode_checks(self.runner({('xcodebuild', '-version'): (0, 'Xcode 16.4')}))
        self.assertNotIn('Metal toolchain component', [c.name for c in checks])

    def test_platform_and_device_notes(self):
        self.assertTrue(toolchain.platform_check(lambda: 'Darwin', lambda: 'arm64').ok)
        self.assertEqual(toolchain.platform_check(lambda: 'Linux', lambda: 'x86_64').blocks, 'all')
        self.assertIn('measured configurations apply', toolchain.device_check(self.runner()).detail)
        other = toolchain.device_check(self.runner({
            ('sysctl', '-n', 'machdep.cpu.brand_string'): (0, 'Apple M1')}))
        self.assertTrue(other.ok)
        self.assertIn('baseline configuration', other.detail)

    def test_disk_check_uses_the_nearest_existing_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'not/yet/created'
            self.assertTrue(toolchain.disk_check(missing, 1).ok)
            self.assertFalse(toolchain.disk_check(missing, 1 << 62).ok)


REPOSITORY = Path(__file__).resolve().parents[1]


def commands(action):
    """Stand in for the subprocess module as seen by assets.py; store.clone keeps real cp."""
    def run(*args, **kwargs):
        stub.calls += 1
        return action(*args, **kwargs)
    stub = types.SimpleNamespace(run=run, calls=0, SubprocessError=subprocess.SubprocessError,
                                 CalledProcessError=subprocess.CalledProcessError)
    return stub


def literal(path, name):
    """A module-level literal assignment from a fixture script, without importing it."""
    for node in ast.parse((REPOSITORY / path).read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class PinTests(unittest.TestCase):
    def test_checkpoint_pins_agree_everywhere(self):
        pinned = {name: digest for name, (_, digest) in assets.CHECKPOINT_FILES.items()}
        self.assertEqual(pinned, literal('tests/fixtures/attention_sublayer/checkpoint.py', 'ASSET_HASHES'))
        self.assertIn('988097824', (REPOSITORY / 'tests/fixtures/attention_sublayer/checkpoint.py').read_text())
        self.assertEqual(assets.CHECKPOINT_FILES['model.safetensors'][0], 988097824)
        self.assertEqual(pinned['model.safetensors'], assets.CHECKPOINT_SHA)
        self.assertEqual(pinned['model.safetensors'], literal('tests/fixtures/model_reference.py', 'WEIGHT_SHA'))
        self.assertEqual(pinned['config.json'], literal('tests/fixtures/model_reference.py', 'CONFIG_SHA'))
        self.assertEqual(pinned['tokenizer.json'], tokenizer_assets.SOURCE_SHA)
        self.assertEqual(assets.CHECKPOINT_FILES['tokenizer.json'][0], tokenizer_assets.SOURCE_BYTES)
        for name, digest in literal('tests/fixtures/chat_reference.py', 'HASHES').items():
            self.assertEqual(pinned[name], digest)
        documented = dict(reversed(line.split()) for line in
                          (REPOSITORY / 'docs/model.md').read_text().split('```text\n', 1)[1].split('```')[0].splitlines())
        self.assertEqual(pinned, documented)
        self.assertEqual(assets.store_checkpoint(REPOSITORY / 'store'),
                         REPOSITORY / 'store' / assets.MODEL_ID / assets.REVISION / 'checkpoint')

    def test_tensor_identity_ignores_manifest_provenance(self):
        records = {'b': dict(sha256='2' * 64), 'a': dict(sha256='1' * 64)}
        first = assets.prepared_tensors_sha256(dict(tensors=records, source_sha256='x'))
        self.assertEqual(first, assets.prepared_tensors_sha256(dict(tensors=dict(reversed(records.items())))))
        self.assertEqual(first, hashlib.sha256(f"a {'1' * 64}\nb {'2' * 64}\n".encode()).hexdigest())


class ProvisionTests(unittest.TestCase):
    """A temporary store, this checkout and another worktree holding verified assets."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkout = self.root / 'checkout'
        (self.checkout / 'build').mkdir(parents=True)
        self.store = self.root / 'store'
        self.files = {'config.json': b'{"hidden_size": 896}', 'model.safetensors': b'weights' * 100}
        self.tensors = {'embedding': b'\x80\x3f\x00\x40', 'final_norm': b'\x00\x3f\x80\xbf'}
        self.manifest = dict(format='qwen-model-prepared-v1', model_revision=assets.REVISION,
                             checkpoint_sha256=assets.CHECKPOINT_SHA, source_sha256='provenance',
                             tensors={name: dict(shape=[2], dtype='BF16', bytes=len(data),
                                                 sha256=hashlib.sha256(data).hexdigest())
                                      for name, data in self.tensors.items()})
        pins = {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in self.files.items()}
        for module, name, value in [(assets, 'CHECKPOINT_FILES', pins),
                                    (assets, 'PREPARED_TENSORS_SHA256', assets.prepared_tensors_sha256(self.manifest)),
                                    (assets, 'tensor_shapes', lambda: {name: (2,) for name in self.tensors}),
                                    (assets, 'repository_root', lambda: self.checkout),
                                    (tokenizer_assets, 'repository_root', lambda: self.checkout),
                                    (assets, 'ensure_prepared', lambda download: self.checkout / 'tables.bin')]:
            patcher = patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def worktree(self, name='other'):
        """Another checkout that prepared real (unlinked) assets before the store existed."""
        root = self.root / name
        checkpoint = root / 'build/checkpoints' / assets.MODEL_ID / assets.REVISION
        checkpoint.mkdir(parents=True)
        for filename, data in self.files.items():
            (checkpoint / filename).write_bytes(data)
        self.write_model(root / 'build/model-prepared-v1')
        return root

    def write_model(self, directory):
        directory.mkdir(parents=True)
        for name, data in self.tensors.items():
            (directory / (name + '.bin')).write_bytes(data)
        (directory / 'manifest.json').write_text(json.dumps(self.manifest))

    def provision(self, sources=((), ()), **options):
        with patch.object(assets, 'import_sources', return_value=sources), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return assets.provision(self.store, **options)

    def sources(self, root):
        return ([root / 'build/checkpoints' / assets.MODEL_ID / assets.REVISION], [root / 'build/model-prepared-v1'])

    def passing_toolchain(self, **overrides):
        """Every toolchain check passes unless named, e.g. uv_check=<a failing Check>."""
        passing = toolchain.Check('ok', True, 'ok')
        stack = ExitStack()
        checks = {'platform_check': passing, 'xcode_checks': [passing], 'mojo_check': passing,
                  'uv_check': passing, **overrides}
        for name, value in checks.items():
            stack.enter_context(patch.object(toolchain, name, return_value=value))
        stack.enter_context(patch.object(toolchain, 'device', return_value='Apple M4 Pro'))
        stack.enter_context(patch.object(assets, 'prepared_valid', return_value=True))
        stack.enter_context(patch('llm_mojo.runtime.build.binary_status', return_value='current'))
        return stack

    def snapshot(self, *roots):
        return {str(p): (p.is_symlink(), stat.S_IMODE(p.lstat().st_mode),
                         p.read_bytes() if p.is_file() and not p.is_symlink() else None)
                for root in roots for p in sorted(root.rglob('*'))}

    def test_import_links_checkout_and_leaves_source_untouched(self):
        other = self.worktree()
        before = self.snapshot(other)
        sources = ([other / 'build/checkpoints' / assets.MODEL_ID / assets.REVISION],
                   [other / 'build/model-prepared-v1'])
        result = self.provision(sources)
        self.assertEqual(self.snapshot(other), before)
        self.assertEqual(set(result['checkpoint_links'].values()), {'linked'})
        self.assertEqual(result['model_link'], 'linked')
        prepared = self.checkout / 'build/model-prepared-v1'
        self.assertTrue(prepared.is_symlink())
        self.assertEqual(prepared.resolve(), assets.store_prepared(self.store))
        self.assertEqual(stat.S_IMODE(assets.store_prepared(self.store).stat().st_mode), 0o555)
        for filename, data in self.files.items():
            link = tokenizer_assets.asset_directory() / filename
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.read_bytes(), data)
        assets.verify_prepared(prepared)
        # Idempotent: a second run finds the store and links in place.
        self.assertEqual(self.provision(sources)['model_link'], 'linked')
        self.assertEqual(list((self.store / '.staging').iterdir()), [])

    def test_a_read_only_store_copy_can_be_imported_into_another_store(self):
        other = self.worktree()
        self.provision(([other / 'build/checkpoints' / assets.MODEL_ID / assets.REVISION],
                        [other / 'build/model-prepared-v1']))
        published = assets.store_prepared(self.store)
        self.assertEqual(stat.S_IMODE(published.stat().st_mode), 0o555)
        second = self.root / 'second-store'
        with patch.object(assets, 'import_sources', return_value=([assets.store_checkpoint(self.store)], [published])), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            assets.provision(second)
        assets.verify_pinned_model(assets.store_prepared(second))
        self.assertEqual(stat.S_IMODE(assets.store_prepared(second).stat().st_mode), 0o555)
        self.assertEqual(stat.S_IMODE(published.stat().st_mode), 0o555)

    def test_second_checkout_reuses_the_store_without_sources(self):
        other = self.worktree()
        self.provision(([other / 'build/checkpoints' / assets.MODEL_ID / assets.REVISION],
                        [other / 'build/model-prepared-v1']))
        second = self.root / 'second'
        (second / 'build').mkdir(parents=True)
        with patch.object(assets, 'repository_root', lambda: second), \
             patch.object(tokenizer_assets, 'repository_root', lambda: second):
            self.provision(download=False)
            self.assertEqual((second / 'build/model-prepared-v1').resolve(), assets.store_prepared(self.store))

    def test_downloads_only_missing_files_then_offline_requires_the_store(self):
        with self.assertRaisesRegex(FileNotFoundError, 'uv run llm-mojo setup'):
            self.provision(download=False)
        requested = []

        def urlopen(url, timeout):
            requested.append(url.rsplit('/', 1)[1])
            return io.BytesIO(self.files[requested[-1]])

        def prepare(command, **kwargs):
            self.assertIn('model_reference.py', command[4])
            # Preparation reads the checkpoint through this checkout's links.
            self.assertTrue((tokenizer_assets.asset_directory() / 'model.safetensors').is_symlink())
            self.write_model(Path(command[command.index('--output') + 1]))

        with patch.object(store.urllib.request, 'urlopen', side_effect=urlopen), \
             patch.object(assets, 'subprocess', commands(prepare)) as run:
            self.provision(download=True)
        self.assertEqual(sorted(requested), sorted(self.files))
        self.assertEqual(run.calls, 1)
        assets.verify_pinned_model(assets.store_prepared(self.store))

    def test_failed_preparation_publishes_nothing(self):
        other = self.worktree()
        sources = ([other / 'build/checkpoints' / assets.MODEL_ID / assets.REVISION], [])

        def interrupted(command, **kwargs):
            output = Path(command[command.index('--output') + 1])
            output.mkdir()
            (output / 'embedding.bin').write_bytes(self.tensors['embedding'])
            raise subprocess.CalledProcessError(1, command)

        with patch.object(assets, 'subprocess', commands(interrupted)):
            with self.assertRaises(subprocess.CalledProcessError):
                self.provision(sources)
        self.assertFalse(assets.store_prepared(self.store).exists())
        self.assertEqual(list((self.store / '.staging').iterdir()), [])

    def test_real_entries_are_kept_when_valid_and_refused_when_not(self):
        other = self.worktree()
        sources = ([other / 'build/checkpoints' / assets.MODEL_ID / assets.REVISION],
                   [other / 'build/model-prepared-v1'])
        self.write_model(self.checkout / 'build/model-prepared-v1')
        self.assertEqual(self.provision(sources)['model_link'], 'local')
        local = tokenizer_assets.asset_directory() / 'config.json'
        local.unlink()
        local.write_bytes(b'{"hidden_size": 1}')
        with self.assertRaisesRegex(ValueError, 'mv "'):
            self.provision(sources)
        self.assertEqual(local.read_bytes(), b'{"hidden_size": 1}')

    def test_other_worktrees_are_discovered_from_git(self):
        listing = f'worktree {self.checkout}\nHEAD abc\n\nworktree {self.root / "other"}\nbranch x\n'
        with patch.object(assets, 'subprocess', commands(lambda *a, **k: subprocess.CompletedProcess([], 0, listing))):
            checkpoints, models = assets.import_sources([self.root / 'explicit'])
        self.assertEqual(models, [self.root / 'explicit/build/model-prepared-v1',
                                  self.checkout / 'build/model-prepared-v1',
                                  self.root / 'other/build/model-prepared-v1'])
        self.assertEqual(len(checkpoints), 3)

    def test_check_is_read_only_and_reports_readiness(self):
        other = self.worktree()
        with self.passing_toolchain():
            before = self.snapshot(self.root)
            lines = []
            self.assertEqual(assets.setup(self.store, check=True, log=lines.append), 1)
            self.assertEqual(self.snapshot(self.root), before)
            self.provision(([other / 'build/checkpoints' / assets.MODEL_ID / assets.REVISION],
                            [other / 'build/model-prepared-v1']))
            before = self.snapshot(self.root)
            self.assertEqual(assets.setup(self.store, check=True, log=lines.append), 0)
            self.assertEqual(self.snapshot(self.root), before)
        self.assertEqual(lines[-1], 'Ready: uv run llm-mojo chat')

    def test_check_verifies_kept_local_copies(self):
        self.write_model(self.checkout / 'build/model-prepared-v1')
        self.assertEqual(self.provision(self.sources(self.worktree()))['model_link'], 'local')
        lines = []
        with self.passing_toolchain():
            self.assertEqual(assets.setup(self.store, check=True, log=lines.append), 0)
            (self.checkout / 'build/model-prepared-v1/embedding.bin').write_bytes(b'damaged!')
            self.assertEqual(assets.setup(self.store, check=True, log=lines.append), 1)
            self.assertEqual(assets.status(self.store)['links']['model-prepared-v1'], 'local but invalid')
            config = tokenizer_assets.asset_directory() / 'config.json'
            config.unlink()
            config.write_bytes(b'{"hidden_size": 1}')
            self.assertEqual(assets.status(self.store)['links']['config.json'], 'local but invalid')
        self.assertEqual(lines[-1], 'Not ready: run uv run llm-mojo setup')

    def test_failed_preparation_prerequisites_stop_setup_before_any_change(self):
        failures = {'uv_check': toolchain.Check('uv on PATH', False, 'not found', 'Install uv', 'prepare'),
                    'disk_check': toolchain.Check('Free space for the store', False, '0.1 GB free',
                                                  'Free space', 'prepare')}
        other = self.worktree()
        for name, failure in failures.items():
            with self.subTest(check=name), self.passing_toolchain(**{name: failure}), \
                 patch.object(assets, 'import_sources', return_value=self.sources(other)):
                before = self.snapshot(self.root)
                lines = []
                self.assertEqual(assets.setup(self.store, build=False, log=lines.append), 1)
                self.assertEqual(self.snapshot(self.root), before)
                self.assertIn('Stopped before changing anything', lines[-1])

    def test_space_requirement_counts_what_the_store_lacks(self):
        requested = []

        def disk_check(directory, required):
            requested.append(required)
            return toolchain.Check('Free space for the store', True, 'enough', blocks='prepare')

        with patch.object(toolchain, 'disk_check', disk_check):
            assets.preparation_checks(self.store)
            self.provision(self.sources(self.worktree()))
            assets.preparation_checks(self.store)
            # With the checkpoint cached, a prepared model moved aside still means writing a whole model.
            with redirect_stderr(io.StringIO()):
                store.set_aside(assets.store_prepared(self.store))
            assets.preparation_checks(self.store)
        downloads = sum(size for size, _ in assets.CHECKPOINT_FILES.values())
        self.assertEqual(requested, [downloads + assets.PREPARED_BYTES, 0, assets.PREPARED_BYTES])

    def test_separate_copy_creates_its_parent_and_clones(self):
        sources = self.sources(self.worktree())
        self.provision(sources)
        output = self.root / 'new/parent/model'
        # Without the parent, cp -c fails and the copy silently becomes a full byte copy.
        with self.passing_toolchain(), patch.object(assets, 'import_sources', return_value=sources), \
             patch.object(store.shutil, 'copytree', side_effect=AssertionError('fell back to a full copy')), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(assets.prepare(output, root=self.store), output)
        assets.verify_prepared(output)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o755)
        self.assertEqual([p.name for p in output.parent.iterdir()], ['model'])

    def test_prepare_refuses_when_preparation_prerequisites_fail(self):
        missing_uv = toolchain.Check('uv on PATH', False, 'not found', 'Install uv', 'prepare')
        with patch.object(toolchain, 'uv_check', return_value=missing_uv), \
             patch.object(assets, 'provision') as provision:
            with self.assertRaisesRegex(RuntimeError, 'uv on PATH: not found'):
                assets.prepare(root=self.store)
        provision.assert_not_called()


if __name__ == '__main__':
    unittest.main()
