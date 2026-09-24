"""A per-user store of pinned downloads and prepared models shared by checkouts.

Standard library only: tokenizer preparation imports this module inside the
pinned Torch script environment. Files enter the store only by atomic rename
after size and SHA-256 verification, and are read-only afterwards. Checkouts
reach them through symbolic links; nothing writes through those links.
"""
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import uuid

CHUNK = 1 << 20


def store_root(explicit=None):
    """--store, then LLM_MOJO_CACHE_DIR, then $XDG_CACHE_HOME/llm-mojo, then ~/.cache/llm-mojo."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    configured = os.environ.get('LLM_MOJO_CACHE_DIR')
    if configured:
        return Path(configured).expanduser().resolve()
    cache = os.environ.get('XDG_CACHE_HOME')
    return ((Path(cache).expanduser() if cache else Path.home() / '.cache') / 'llm-mojo').resolve()


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified(path, size, sha256):
    """True for an existing regular file (or a link to one) with the pinned size and digest."""
    path = Path(path)
    try:
        return path.is_file() and path.stat().st_size == size and file_sha256(path) == sha256
    except OSError:
        return False


def download(url, destination, size, sha256, *, opener=None, read_only=True):
    """Stream url to destination, publishing only a complete, verified file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
    temporary = Path(name)
    try:
        digest = hashlib.sha256()
        written = 0
        with os.fdopen(fd, 'wb') as output, (opener or urllib.request.urlopen)(url, timeout=60) as response:
            while chunk := response.read(CHUNK):
                written += len(chunk)
                if written > size:
                    raise ValueError(f'download exceeds its pinned size: {destination.name}')
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if written != size or digest.hexdigest() != sha256:
            raise ValueError(f'downloaded size or checksum mismatch: {destination.name}')
        if read_only:
            os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def link_state(path, target):
    """missing, linked, foreign (a link elsewhere), dangling, or local (a real entry)."""
    path = Path(path)
    if path.is_symlink():
        if not path.exists():
            return 'dangling'
        return 'linked' if path.resolve() == Path(target).resolve() else 'foreign'
    return 'local' if path.exists() else 'missing'


def ensure_link(path, target):
    """Point path at target. Real files and directories are reported, never replaced."""
    path = Path(path)
    state = link_state(path, target)
    if state in ('linked', 'local'):
        return state
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.link')
    temporary.unlink(missing_ok=True)
    os.symlink(Path(target), temporary)
    try:
        # rename(2) replaces a link itself; it never follows it into the store.
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return 'linked' if state == 'missing' else 'relinked'


def staging_path(store, name):
    """A fresh, not yet created path on the store's volume, so publication is one rename."""
    directory = Path(store) / '.staging'
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f'{name}.{os.getpid()}.{uuid.uuid4().hex[:12]}'


def clone(source, destination):
    """Copy-on-write clone on APFS (cp -c), falling back to an ordinary copy."""
    source, destination = Path(source).resolve(), Path(destination)
    try:
        subprocess.run(['cp', '-c', *(['-R'] if source.is_dir() else []), str(source), str(destination)],
                       check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        remove_staged(destination)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    return destination


def publish_file(staged, final):
    os.chmod(staged, 0o444)
    Path(final).parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, final)
    return Path(final)


def publish_directory(staged, final):
    """Publish a complete directory read-only; an existing final entry must be set aside first."""
    staged, final = Path(staged), Path(final)
    for child in staged.iterdir():
        if child.is_file() and not child.is_symlink():
            os.chmod(child, 0o444)
    # Moving a directory to a new parent rewrites its '..' entry, which needs write
    # permission; a clone of another store's read-only directory would lack it.
    os.chmod(staged, 0o755)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staged, final)
    os.chmod(final, 0o555)
    return final


def set_aside(path):
    """Move an invalid store entry out of the way without deleting anything."""
    path = Path(path)
    aside = path.with_name(f'{path.name}.invalid-{datetime.now():%Y%m%d-%H%M%S}')
    os.rename(path, aside)
    print(f'Moved an invalid store entry aside: {aside} (delete it when no longer needed)', file=sys.stderr)
    return aside


def remove_staged(path):
    """Delete a staging entry, including one already marked read-only."""
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        os.chmod(path, 0o755)
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
