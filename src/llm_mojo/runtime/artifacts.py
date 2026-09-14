"""Hashing and atomic publication for local prepared assets and binaries."""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def setup_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
