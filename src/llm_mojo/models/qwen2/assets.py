"""Verify the fixed Qwen prepared-weight boundary before native execution.

Preparation and verification are development/initialization work. Native model
execution receives only checked binary tensors and runs without Python interop.
Pinned downloads and the prepared model live once per machine in the shared
store; each checkout links its build/ paths to them. Tokenizer tables and native
binaries stay per checkout because they depend on the checkout's sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from llm_mojo._repository import environment_tool, repository_root
from llm_mojo.models.qwen2.tokenizer_assets import (
    MODEL_ID, REVISION, asset_directory, ensure_prepared, prepared_valid)
from llm_mojo.runtime import store, toolchain
from llm_mojo.runtime.artifacts import setup_lock

CONTEXT_CAPACITY = 4096
APPLICATION_MODE = 'fast'
CHECKPOINT_SHA = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
MODEL_URL = f'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/{REVISION}/'
# Pinned revision files: name -> (bytes, SHA-256). Kept in sync with docs/model.md
# and the fixture scripts by tests/test_setup.py.
CHECKPOINT_FILES = {
    'config.json': (659, '18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45'),
    'generation_config.json': (242, 'e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6'),
    'merges.txt': (1671839, '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3'),
    'model.safetensors': (988097824, CHECKPOINT_SHA),
    'tokenizer.json': (7031645, 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'),
    'tokenizer_config.json': (7305, '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583'),
    'vocab.json': (2776833, 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'),
}
# Digest of the 196 sorted "name sha256" tensor records. Manifests also record
# preparation provenance, so their bytes differ between otherwise identical copies.
PREPARED_TENSORS_SHA256 = 'a3151e7d635038ea0c1d7940e0a5fad725ad009503db2b9ffdac34470967fddb'
PREPARED_BYTES = 989114112
BINARIES = {'chat': 'src/llm_mojo/cli/chat_cli.mojo', 'generate': 'src/llm_mojo/cli/generate_cli.mojo'}


def capabilities():
    """The implemented application contract, also enforced by native Qwen."""
    return dict(model=MODEL_ID, revision=REVISION, checkpoint_sha256=CHECKPOINT_SHA,
                mode=APPLICATION_MODE, backend='metal', dtype='BF16', batch=1,
                context_capacity=CONTEXT_CAPACITY, decoding='greedy',
                chat='plain system/user/assistant')


def prepared_directory():
    """Default output of the documented model preparation, rooted in this checkout."""
    return repository_root() / 'build/model-prepared-v1'


def store_directory(root=None):
    return store.store_root(root) / MODEL_ID / REVISION


def store_checkpoint(root=None):
    return store_directory(root) / 'checkpoint'


def store_prepared(root=None):
    return store_directory(root) / 'model-prepared-v1'


def tensor_shapes():
    shapes = {'embedding': (151936, 896), 'final_norm': (896,),
              'cosine': (CONTEXT_CAPACITY, 64), 'sine': (CONTEXT_CAPACITY, 64)}
    for layer in range(24):
        for name, shape in {
            'attention_norm': (896,), 'mlp_norm': (896,), 'qkv': (1152,896),
            'bias': (1152,), 'wo': (896,896), 'gate': (4864,896),
            'up': (4864,896), 'down': (896,4864),
        }.items():
            shapes[f'layer_{layer}_{name}'] = shape
    return shapes


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError('prepared manifest must be an object')
    if (manifest.get('format') != 'qwen-model-prepared-v1'
            or manifest.get('model_revision') != REVISION
            or manifest.get('checkpoint_sha256') != CHECKPOINT_SHA):
        raise ValueError('prepared model identity mismatch')
    expected = tensor_shapes()
    actual = manifest.get('tensors', {})
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError('missing or extra prepared model tensors')
    for name, shape in expected.items():
        record = actual[name]
        if not isinstance(record, dict):
            raise ValueError('invalid prepared tensor record: ' + name)
        size = 2
        for dim in shape:
            size *= dim
        digest = record.get('sha256')
        if (record.get('shape') != list(shape) or record.get('dtype') != 'BF16'
                or record.get('bytes') != size
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('invalid prepared tensor geometry/identity: '+name)
    return expected


def verify_prepared(directory=None):
    directory = Path(directory) if directory is not None else prepared_directory()
    if not (directory/'manifest.json').is_file():
        detail = 'its store link is broken' if directory.is_symlink() and not directory.exists() else 'no manifest.json'
        raise FileNotFoundError(f'No prepared model at {directory} ({detail}); run: uv run llm-mojo setup')
    manifest = json.loads((directory/'manifest.json').read_text())
    expected = validate_manifest(manifest)
    for name in expected:
        path = directory/(name+'.bin')
        record = manifest['tensors'][name]
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('prepared tensor checksum/extent mismatch: '+name)
    return directory, manifest


def prepared_tensors_sha256(manifest):
    records = manifest['tensors']
    return hashlib.sha256(''.join(f'{name} {records[name]["sha256"]}\n'
                                  for name in sorted(records)).encode()).hexdigest()


def verify_pinned_model(directory):
    """verify_prepared plus the pinned tensor identity required of shared copies."""
    directory, manifest = verify_prepared(directory)
    if prepared_tensors_sha256(manifest) != PREPARED_TENSORS_SHA256:
        raise ValueError(f'prepared tensors differ from the pinned model: {directory}')
    return directory, manifest


def generate(prepared, prompt, maximum, chunk_rows=0, policy='fast', report=None):
    """Verify inputs before launching the native generation driver.

    Prepared artifacts must remain unchanged during the launch and execution.
    Python performs initialization only; the child owns native inference.
    """
    if not 0 <= maximum <= 4096 or not 0 <= chunk_rows <= 4096:
        raise ValueError('invalid generation or chunk limit')
    if policy not in ('baseline', 'auto', 'candidate', 'consistent', '0', '2', '3', '20', '21', 'fast'):
        raise ValueError('unknown generation configuration policy')
    prompt = Path(prompt).resolve()
    if not prompt.is_file():
        raise FileNotFoundError(prompt)
    directory, _ = verify_prepared(Path(prepared).resolve())
    tables = ensure_prepared(download=False)
    root = repository_root()
    subprocess.run([
        environment_tool('mojo'), 'run', '-I', 'src',
        str(root / 'src/llm_mojo/cli/generate_cli.mojo'), str(directory), str(tables),
        str(prompt), str(maximum), str(chunk_rows), policy,
        *([str(Path(report).resolve())] if report is not None else []),
    ], cwd=root, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Verify artifacts and run native BF16 Qwen plain-text greedy generation on Metal.')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, required=True)
    parser.add_argument('--chunk-rows', type=int, default=0)
    parser.add_argument('--policy', default='fast',
                        choices=['baseline', 'auto', 'candidate', 'consistent', '0', '2', '3', '20', '21', 'fast'])
    parser.add_argument('--report', type=Path, help='Write native timing, token and cache events as TSV')
    args = parser.parse_args(argv)
    generate(args.prepared, args.prompt, args.max_new_tokens, args.chunk_rows, args.policy, args.report)


def import_sources(import_from=()):
    """Places that may already hold verified assets: explicit paths, this checkout, other worktrees."""
    roots = [Path(p).expanduser().resolve() for p in import_from]
    root = repository_root()
    roots.append(root)
    try:
        listing = subprocess.run(['git', 'worktree', 'list', '--porcelain'], cwd=root,
                                 capture_output=True, text=True, check=True, timeout=30).stdout
        roots += [Path(line[len('worktree '):]) for line in listing.splitlines() if line.startswith('worktree ')]
    except (OSError, subprocess.SubprocessError):
        pass
    checkpoints, models, seen = [], [], set()
    for path in roots:
        if path in seen:
            continue
        seen.add(path)
        if (path/'manifest.json').is_file():
            models.append(path)
        elif (path/'model.safetensors').exists():
            checkpoints.append(path)
        else:
            checkpoints.append(path/'build/checkpoints'/MODEL_ID/REVISION)
            models.append(path/'build/model-prepared-v1')
    return checkpoints, models


def ensure_checkpoint(root, *, download, sources, log):
    """Every pinned file in the store: already there, cloned from a verified copy, or downloaded."""
    directory = store_checkpoint(root)
    for name, (size, digest) in CHECKPOINT_FILES.items():
        target = directory/name
        if store.verified(target, size, digest):
            continue
        if target.exists() or target.is_symlink():
            store.set_aside(target)
        for source in sources:
            candidate = source/name
            if candidate.resolve() != target.resolve() and store.verified(candidate, size, digest):
                staged = store.staging_path(store.store_root(root), name)
                try:
                    store.clone(candidate, staged)
                    if not store.verified(staged, size, digest):
                        raise ValueError(f'copy of {candidate} changed while importing')
                    store.publish_file(staged, target)
                finally:
                    store.remove_staged(staged)
                log(f'Imported {name} from {source}')
                break
        else:
            if not download:
                raise FileNotFoundError(f'Missing pinned checkpoint file {name}; run: uv run llm-mojo setup '
                                        '(downloads about 1 GB), or pass --import-from with a verified copy')
            log(f'Downloading {name} ({size / 1e6:,.1f} MB) from the pinned revision…')
            staged = store.staging_path(store.store_root(root), name)
            try:
                store.download(MODEL_URL + name, staged, size, digest)
                store.publish_file(staged, target)
            finally:
                store.remove_staged(staged)
    return directory


def ensure_model(root, *, sources, log):
    """The pinned prepared model in the store: already there, cloned from a verified copy, or prepared."""
    target = store_prepared(root)
    if target.exists() or target.is_symlink():
        try:
            verify_pinned_model(target)
            return target
        except (OSError, ValueError, KeyError, TypeError):
            store.set_aside(target)
    staged = store.staging_path(store.store_root(root), 'model-prepared-v1')
    try:
        for source in sources:
            if not source.exists() or source.resolve() == target.resolve():
                continue
            try:
                verify_pinned_model(source)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            store.clone(source, staged)
            verify_pinned_model(staged)
            store.publish_directory(staged, target)
            log(f'Imported the prepared model from {source}')
            return target
        # Preparation reads the checkpoint through this checkout's build/ links.
        link_checkpoint(root)
        log('Preparing 196 BF16 tensors with the pinned reference environment…')
        subprocess.run(['uv', 'run', '--locked', '--script', 'tests/fixtures/model_reference.py',
                        'prepare', '--output', str(staged)], cwd=repository_root(), check=True)
        verify_pinned_model(staged)
        store.publish_directory(staged, target)
        return target
    finally:
        store.remove_staged(staged)


def link_checkpoint(root):
    states = {}
    for name, (size, digest) in CHECKPOINT_FILES.items():
        path = asset_directory()/name
        states[name] = store.ensure_link(path, store_checkpoint(root)/name)
        if states[name] == 'local' and not store.verified(path, size, digest):
            raise ValueError(f'{path} is not the pinned file; move it aside and rerun setup: mv "{path}" "{path}.old"')
    return states


def link_model(root):
    path = prepared_directory()
    state = store.ensure_link(path, store_prepared(root))
    if state == 'local':
        try:
            verify_pinned_model(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f'{path} is not the pinned prepared model ({error}); move it aside and rerun '
                             f'setup: mv "{path}" "{path}.old"') from error
    return state


def provision(root=None, *, download=False, import_from=(), log=print):
    """Pinned checkpoint and prepared model in the store, linked here, plus tokenizer tables."""
    checkpoints, models = import_sources(import_from)
    with setup_lock(store_directory(root)):
        ensure_checkpoint(root, download=download, sources=checkpoints, log=log)
        ensure_model(root, sources=models, log=log)
        checkpoint_links = link_checkpoint(root)
        model_link = link_model(root)
    tables = ensure_prepared(download=False)
    return dict(checkpoint_links=checkpoint_links, model_link=model_link, tables=tables)


def store_state(root=None):
    """The pinned files the store lacks, and why its prepared model fails verification (None if it passes)."""
    checkpoint = store_checkpoint(root)
    missing = [name for name, (size, digest) in CHECKPOINT_FILES.items()
               if not store.verified(checkpoint/name, size, digest)]
    try:
        verify_pinned_model(store_prepared(root))
        model_error = None
    except (OSError, ValueError, KeyError, TypeError) as error:
        model_error = str(error)
    return missing, model_error


def preparation_checks(root=None, state=None):
    """What provisioning needs: uv, and room for everything the store still lacks.

    The space is an upper bound. An import may clone instead of copying, and an
    invalid entry set aside keeps its bytes.
    """
    missing, model_error = state or store_state(root)
    required = sum(CHECKPOINT_FILES[name][0] for name in missing) + (PREPARED_BYTES if model_error else 0)
    return [toolchain.uv_check(), toolchain.disk_check(store.store_root(root), required)]


def status(root=None, state=None):
    """Read-only readiness report: nothing is created, locked, downloaded or built."""
    from llm_mojo.runtime.build import binary_status
    checkpoint = store_checkpoint(root)
    missing, model_error = state or store_state(root)
    model = 'missing or invalid: ' + model_error if model_error else 'verified'
    # A real entry kept in the checkout is what chat reads, so it must still verify.
    links = {}
    for name, (size, digest) in CHECKPOINT_FILES.items():
        path = asset_directory()/name
        links[name] = store.link_state(path, checkpoint/name)
        if links[name] == 'local' and not store.verified(path, size, digest):
            links[name] = 'local but invalid'
    path = prepared_directory()
    links['model-prepared-v1'] = store.link_state(path, store_prepared(root))
    if links['model-prepared-v1'] == 'local':
        try:
            verify_pinned_model(path)
        except (OSError, ValueError, KeyError, TypeError):
            links['model-prepared-v1'] = 'local but invalid'
    device = toolchain.device()
    return dict(store=str(store_directory(root)),
                checkpoint='verified' if not missing else 'missing: ' + ', '.join(missing),
                prepared_model=model, links=links,
                tokenizer_tables='ready' if prepared_valid(asset_directory()) else 'not prepared',
                binaries={name: binary_status(name, entry) for name, entry in BINARIES.items()},
                device=device, fast_selection=device == toolchain.MEASURED_DEVICE)


def setup(root=None, *, offline=False, check=False, import_from=(), build=True, log=print):
    """Check the toolchain, provision the shared store, link this checkout and build; return an exit status."""
    state = store_state(root)
    checks = [toolchain.platform_check(), *toolchain.xcode_checks(), toolchain.mojo_check(),
              *preparation_checks(root, state), toolchain.device_check()]
    for item in checks:
        log(f"{'ok  ' if item.ok else 'FAIL'}  {item.name}: {item.detail}"
            + ('' if item.ok else f'\n      fix: {item.remedy}'))
    blocked = {item.blocks for item in checks if not item.ok}
    if 'all' in blocked:
        return 1
    if check:
        report = status(root, state)
        log(json.dumps(report, indent=2))
        ready = (report['checkpoint'] == 'verified' and report['prepared_model'] == 'verified'
                 and all(state in ('linked', 'local') for state in report['links'].values())
                 and report['tokenizer_tables'] == 'ready'
                 and all(state == 'current' for state in report['binaries'].values()))
        log('Ready: uv run llm-mojo chat' if ready and not blocked else 'Not ready: run uv run llm-mojo setup')
        return 0 if ready and not blocked else 1
    if 'prepare' in blocked:
        log('Stopped before changing anything: fix the checks marked FAIL above.')
        return 1
    log(f'Shared store: {store_directory(root)}')
    result = provision(root, download=not offline, import_from=import_from, log=log)
    local = [name for name, state in [*result['checkpoint_links'].items(), ('model-prepared-v1', result['model_link'])]
             if state == 'local']
    log(f'Linked {asset_directory()} and {prepared_directory()} to the store'
        + (f' (kept verified local copies: {", ".join(local)})' if local else ''))
    if not build:
        return 1 if blocked else 0
    if 'build' in blocked:
        log('Skipped building native binaries until the toolchain checks above pass.')
        return 1
    from llm_mojo.runtime.build import ensure_binary
    for name, entry in BINARIES.items():
        ensure_binary(name, entry)
    log('Ready: uv run llm-mojo chat')
    return 0


def prepare(output=None, *, download=False, root=None):
    """Provision the shared store and link this checkout, or write a separate verified copy to output."""
    failed = [item for item in preparation_checks(root) if not item.ok]
    if failed:
        raise RuntimeError('cannot prepare the model: ' + '; '.join(
            f'{item.name}: {item.detail} (fix: {item.remedy})' for item in failed))
    provision(root, download=download)
    if output is None:
        target = prepared_directory()
        print('Prepared model:', target, '->', store_prepared(root))
        return target
    target = Path(output).resolve()
    if target.exists():
        verify_prepared(target)
        print('Verified prepared model:', target)
        return target
    # Stage beside the target so publishing is one rename; cloning needs the parent to exist.
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f'.{target.name}.partial')
    store.remove_staged(staged)
    try:
        store.clone(store_prepared(root), staged)
        verify_pinned_model(staged)
        # A requested extra copy belongs to the caller; only the store is read-only.
        staged.chmod(0o755)
        for child in staged.iterdir():
            child.chmod(0o644)
        staged.rename(target)
    finally:
        store.remove_staged(staged)
    print('Prepared model:', target)
    return target


if __name__ == '__main__':
    main(sys.argv[1:])
