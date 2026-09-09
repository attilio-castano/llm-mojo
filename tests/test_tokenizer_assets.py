"""First-run, offline reuse, interruption, and provenance regression tests."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_mojo import tokenizer_assets as assets


class TokenizerAssetTests(unittest.TestCase):
    def test_first_download_is_verified_then_reused_without_network(self):
        payload = b"valid pinned tokenizer"
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            assets, "SOURCE_SHA", hashlib.sha256(payload).hexdigest()
        ):
            directory = Path(tmp)
            with patch.object(
                assets.urllib.request,
                "urlopen",
                return_value=io.BytesIO(payload),
            ) as download:
                path = assets.ensure_source(directory, download=True)
                self.assertEqual(path.read_bytes(), payload)
                download.assert_called_once()
            with patch.object(
                assets.urllib.request,
                "urlopen",
                side_effect=AssertionError("unexpected network"),
            ):
                self.assertEqual(
                    assets.ensure_source(directory, download=True), path
                )
                self.assertEqual(
                    assets.ensure_source(directory, download=False), path
                )
            self.assertEqual(list(directory.glob("*.part")), [])

    def test_interrupted_and_bad_downloads_never_publish(self):
        class Interrupted(io.BytesIO):
            def read(self, *args):
                raise ConnectionError("interrupted")

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for response, error in [
                (Interrupted(), ConnectionError),
                (io.BytesIO(b"wrong"), ValueError),
            ]:
                with patch.object(
                    assets.urllib.request, "urlopen", return_value=response
                ):
                    with self.assertRaises(error):
                        assets.ensure_source(directory, download=True)
                self.assertFalse((directory / "tokenizer.json").exists())
                self.assertEqual(list(directory.glob("*.part")), [])

    def test_missing_offline_and_corrupt_cached_source_fail_without_network(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            assets.urllib.request,
            "urlopen",
            side_effect=AssertionError("unexpected network"),
        ):
            directory = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                assets.ensure_source(directory)
            (directory / "tokenizer.json").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                assets.ensure_source(directory, download=True)

    def test_stale_partial_file_does_not_hide_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "tokenizer.json.old.part").write_bytes(b"partial")
            with self.assertRaises(FileNotFoundError):
                assets.ensure_source(directory)

    def test_failed_atomic_publication_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tables.bin"
            path.write_bytes(b"old")
            with patch.object(
                assets.os, "replace", side_effect=OSError("interrupted")
            ):
                with self.assertRaises(OSError):
                    assets.atomic_write(path, b"new")
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(list(Path(tmp).glob("*.part")), [])

    def test_prepared_cache_requires_matching_source_generator_and_payload(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            assets, "generator_identity", return_value={"generator": "a"}
        ):
            root = Path(tmp)
            target = assets.prepared_directory(root)
            target.mkdir()
            (target / "tables.bin").write_bytes(b"table")
            manifest = dict(
                format=assets.FORMAT,
                source_sha256=assets.SOURCE_SHA,
                generators={"generator": "a"},
                tables_sha256=assets.sha(target / "tables.bin"),
            )

            def save():
                (target / "manifest.json").write_text(json.dumps(manifest))

            save()
            self.assertTrue(assets.prepared_valid(root))
            for key, value in [
                ("format", 999),
                ("source_sha256", "wrong"),
                ("generators", {"generator": "b"}),
                ("tables_sha256", "wrong"),
            ]:
                old = manifest[key]
                manifest[key] = value
                save()
                self.assertFalse(assets.prepared_valid(root))
                manifest[key] = old
            save()
            (target / "tables.bin").write_bytes(b"corrupt")
            self.assertFalse(assets.prepared_valid(root))

    def test_encode_stdin_preserves_raw_bytes_and_uses_native_file_input(self):
        payload = b"a\r\n\x00\xf0\x9f\x99\x82"
        fake_stdin = io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8")
        observed = []

        def launch(command, **kwargs):
            self.assertEqual(command[2], "encode-file")
            observed.append(Path(command[3]).read_bytes())

        with patch.object(
            assets.sys, "argv", ["tokenizer", "encode"]
        ), patch.object(assets.sys, "stdin", fake_stdin), patch.object(
            assets, "ensure_prepared", return_value=Path("tables.bin")
        ), patch.object(
            assets, "ensure_binary", return_value=Path("native-tokenizer")
        ), patch.object(
            assets.subprocess, "run", side_effect=launch
        ):
            assets.main()
        self.assertEqual(observed, [payload])

    def test_all_byte_symbols_are_reversible(self):
        mapping = assets.byte_decoder()
        self.assertEqual(len(mapping), 256)
        self.assertEqual(set(mapping.values()), set(range(256)))


if __name__ == "__main__":
    unittest.main()
