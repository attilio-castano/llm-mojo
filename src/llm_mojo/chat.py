"""Verify local assets, build when needed, then replace this launcher with Mojo."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ._repository import repository_root, environment_tool
from .model_assets import prepared_directory, verify_prepared
from .tokenizer_assets import ensure_prepared, setup_lock, atomic_write, sha


def build_sources():
    root = repository_root()
    return {str(p.relative_to(root)): sha(p) for p in sorted(
        [*root.joinpath('src/llm_mojo').rglob('*.mojo'), root/'uv.lock'])}


def ensure_binary():
    root = repository_root()
    directory = root/'build/chat'
    binary, receipt = directory/'chat', directory/'binary.json'
    with setup_lock(directory):
        identity = build_sources()
        if binary.exists() and receipt.exists():
            record = json.loads(receipt.read_text())
            if record.get('sources') == identity and record.get('binary_sha256') == sha(binary):
                return binary
        print('Building native chat (reused on subsequent launches)…', file=sys.stderr, flush=True)
        fd, name = tempfile.mkstemp(prefix='chat.', suffix='.part', dir=directory)
        os.close(fd)
        temporary = Path(name)
        try:
            subprocess.run([environment_tool('mojo'), 'build', '-I', 'src',
                            'src/llm_mojo/chat_cli.mojo', '-o', str(temporary)], cwd=root, check=True)
            if identity != build_sources():
                raise ValueError('chat source changed during compilation')
            digest = sha(temporary)
            os.replace(temporary, binary)
            atomic_write(receipt, (json.dumps(dict(sources=identity, binary_sha256=digest), indent=2)+'\n').encode())
        finally:
            temporary.unlink(missing_ok=True)
    return binary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Native Qwen terminal chat with persistent KV caches and Fast inference.')
    parser.add_argument('--prepared', type=Path, default=prepared_directory(),
                        help='Prepared model directory (default: this checkout\'s build/model-prepared-v1)')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--chunk-rows', type=int, default=256)
    parser.add_argument('--system-file', type=Path, help='UTF-8 system message; otherwise use the pinned Qwen default')
    parser.add_argument('--report', type=Path, help='Optional token/cache/timing TSV; contains the conversation')
    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 4096 or not 1 <= args.chunk_rows <= 4096:
        parser.error('reply and chunk limits must be in 1..4096')
    if args.system_file:
        args.system_file = args.system_file.resolve()
        args.system_file.read_text(encoding='utf-8')
    if args.report:
        args.report = args.report.resolve()
        if args.report.exists() or not args.report.parent.is_dir():
            parser.error('report must be a new file in an existing directory')
    print('Verifying local Qwen weights and tokenizer…', file=sys.stderr, flush=True)
    prepared, _ = verify_prepared(args.prepared.resolve())
    tables = ensure_prepared(download=False)
    binary = ensure_binary()
    command = [str(binary), str(prepared), str(tables), str(args.max_new_tokens),
               str(args.chunk_rows), str(args.system_file or ''), str(args.report or '')]
    # exec preserves the foreground process group: SIGINT reaches the native
    # loop directly, with no Python parent to interrupt or inference interop.
    os.execv(binary, command)


if __name__ == '__main__':
    main()
