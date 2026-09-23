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

from llm_mojo.runtime import store


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


if __name__ == '__main__':
    unittest.main()
