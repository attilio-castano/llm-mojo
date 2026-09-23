"""Shared model store, toolchain checks and one-step setup without network or Metal."""
from contextlib import redirect_stderr
import hashlib
import io
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

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


if __name__ == '__main__':
    unittest.main()
