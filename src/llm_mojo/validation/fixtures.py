"""One verified, shared copy of each large oracle fixture family.

decoder_layer, attention_sublayer and mlp hold about 8.4 GB and take about ten
minutes to generate. They are deterministic, frozen anchors under tests/fixtures
pin them, and every consumer reads them through build/oracle_data/<family>. So
the store (runtime/store.py) keeps one read-only copy per generator version:

    fixtures/<family>/<key>/        the generated tree, read-only at every depth
    fixtures/<family>/<key>.json    its record: each file's size and SHA-256, the
                                    inputs, the steps and the generating run
    fixtures/<family>/.<key>.lock   held only while generating or repairing it

and build/oracle_data/<family> is a link to it. The key hashes the steps and
every file git lists under SCOPE, so an edit to a generator or an anchor selects
a new entry. A hit re-hashes every file against its record and never writes to
the store. A miss generates under the lock, refuses inputs that changed during
the run, and publishes by rename.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import time

from .._repository import repository_root
from ..runtime import store, toolchain

FORMAT = 'llm-mojo-fixtures-v1'
# Everything the three generators read from the checkout: their scripts, the
# locked script environment (decoder_reference.py.lock links to it) and the
# frozen anchors beside them. tests/test_fixture_cache.py checks the generators'
# own source lists against it.
SCOPE = ('tests/fixtures/generate.py', 'tests/fixtures/generate.py.lock',
         'tests/fixtures/decoder_reference.py', 'tests/fixtures/decoder_reference.py.lock',
         'tests/fixtures/attention_sublayer', 'tests/fixtures/mlp', 'tests/fixtures/decoder_layer')


@dataclass(frozen=True)
class Family:
    name: str
    # Commands run from the checkout; {output} names the directory to generate.
    # ('check', name) runs CHECKS[name] on the directory instead.
    steps: tuple
    # The generators write to their fixed build/oracle_data/<name>, not {output}.
    in_place: bool
    # Generated files that describe the run rather than the oracle; the record keeps them.
    provenance: tuple = ()
    # Other files a validation writes into a checkout copy of the family.
    regenerable: tuple = ()
    estimated_bytes: int = 0


def script(*arguments):
    return ('uv', 'run', '--locked', '--script', *arguments)


FAMILIES = {family.name: family for family in (
    Family('attention_sublayer', (script('tests/fixtures/generate.py', 'attention_sublayer'),
                                  ('check', 'sublayer_anchors'),
                                  script('tests/fixtures/generate.py', 'attention_precision')),
           in_place=True, estimated_bytes=2_900_000_000),
    Family('mlp', (script('tests/fixtures/generate.py', 'mlp'),), in_place=True,
           provenance=('last_run.json',), regenerable=('metal_*_checks*.json',),
           estimated_bytes=2_250_000_000),
    Family('decoder_layer', (script('tests/fixtures/decoder_reference.py', '--output', '{output}'),),
           in_place=False, estimated_bytes=4_000_000_000),
)}


@dataclass(frozen=True)
class Entry:
    tree: Path
    record: Path
    lock: Path


def entry_paths(store_dir, family, key):
    base = Path(store_dir) / 'fixtures' / family.name
    return Entry(base / key, base / f'{key}.json', base / f'.{key}.lock')


def checkout_path(root, family):
    return Path(root) / 'build/oracle_data' / family.name


def inputs(root):
    """Path -> identity of every file git lists under SCOPE, tracked or untracked but not ignored.

    An uncommitted edit selects a new entry; __pycache__ never does. Links count
    by their target, and tracked files missing from the checkout as deleted.
    """
    try:
        listed = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard',
                                 '--', *SCOPE], cwd=root, capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError('The fixture cache identifies generator inputs with git; run validate in a '
                           'git checkout, or pass --no-fixture-cache.') from error
    found = {}
    for name in sorted({os.fsdecode(name) for name in listed.split(b'\0') if name}):
        path = Path(root, name)
        if path.is_symlink():
            found[name] = 'link ' + os.readlink(path)
        elif path.is_file():
            found[name] = 'sha256 ' + store.file_sha256(path)
        else:
            found[name] = 'deleted'
    return found


def identity(family, sources):
    """The entry key: SHA-256 of canonical JSON naming the format, family, steps and inputs."""
    description = dict(format=FORMAT, family=family.name, steps=[list(step) for step in family.steps],
                       inputs=sources)
    return hashlib.sha256(json.dumps(description, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def scan(directory):
    """Relative path -> size of every file below directory; None marks links and special files."""
    found = {}
    for parent, directories, files in os.walk(directory):
        for name in directories + files:
            path = Path(parent, name)
            status = path.lstat()
            if not stat.S_ISDIR(status.st_mode):
                found[path.relative_to(directory).as_posix()] = (
                    status.st_size if stat.S_ISREG(status.st_mode) else None)
    return found


def digests(directory, names):
    """SHA-256 of each named file, in parallel: hashlib releases the GIL while hashing."""
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
        return dict(zip(names, pool.map(lambda name: store.file_sha256(Path(directory, name)), names)))


def table(directory):
    """The record's file table for a freshly generated tree."""
    sizes = scan(directory)
    odd = sorted(name for name, size in sizes.items() if size is None)
    if odd:
        raise RuntimeError('Generated fixtures contain links or special files: ' + ', '.join(odd[:5]))
    hashes = digests(directory, sorted(sizes))
    return {name: dict(bytes=sizes[name], sha256=hashes[name]) for name in sorted(sizes)}


def verify(entry, key):
    """(record, None) when the entry holds exactly the files its record lists, else (None, why)."""
    try:
        record = json.loads(entry.record.read_text())
    except FileNotFoundError:
        return None, 'not cached'
    except (OSError, ValueError) as error:
        return None, f'unreadable record: {error}'
    if not isinstance(record, dict) or record.get('format') != FORMAT or record.get('key') != key:
        return None, 'the record describes another entry'
    if entry.tree.is_symlink() or not entry.tree.is_dir():
        return None, 'the tree is missing'
    try:
        files = record['files']
        sizes = scan(entry.tree)
        differ = sorted(name for name in files.keys() | sizes.keys()
                        if name not in files or sizes.get(name) != files[name]['bytes'])
        if not differ:
            hashes = digests(entry.tree, sorted(files))
            differ = sorted(name for name in files if hashes[name] != files[name]['sha256'])
    except (OSError, KeyError, TypeError, AttributeError) as error:
        return None, f'unreadable entry: {error}'
    if differ:
        return None, 'changed files: ' + ', '.join(differ[:5]) + (' ...' if len(differ) > 5 else '')
    return record, None


@contextmanager
def locked(path, waiting):
    """Hold an exclusive lock on path, saying so when another process holds it first."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open('a')
    except OSError as error:
        raise RuntimeError(f'Cannot write to the fixture store at {path.parent}: {error}. Fix its permissions, '
                           'set LLM_MOJO_CACHE_DIR, or pass --no-fixture-cache.') from error
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(waiting, flush=True)
            fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def check_sublayer_anchors(root, directory):
    """The frozen checks the suite runs between the sublayer and precision steps."""
    directory = Path(directory)
    sublayer = json.loads((directory / 'manifest.json').read_text())
    frozen = json.loads((Path(root) / 'tests/fixtures/attention_sublayer/checksums.json').read_text())
    if any(sublayer[key] != frozen[key] for key in (
        'cases', 'atol', 'rtol', 'array_sha256', 'numerical_contract', 'upstream_contract'
    )):
        raise RuntimeError('sublayer oracle changed: review its numerical contract')
    actual = digests(directory, list(frozen['array_sha256']))
    for name, expected in frozen['array_sha256'].items():
        if actual[name] != expected:
            raise RuntimeError(f'sublayer oracle array changed: {name}')


CHECKS = {'sublayer_anchors': check_sublayer_anchors}


def run_steps(family, root, output, runner):
    for step in family.steps:
        if step[0] == 'check':
            CHECKS[step[1]](root, output)
        else:
            runner(*(part.replace('{output}', str(output)) for part in step))


def machine():
    return dict(python=platform.python_version(), mac_ver=platform.mac_ver()[0], platform=platform.platform())


def generation(root, run_files):
    """Where and when an entry was generated, with the run files the family moves out of its tree."""
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True,
                                check=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return dict(time=datetime.now(timezone.utc).isoformat(timespec='seconds'), commit=commit,
                worktree=str(root), **machine(), run_files=run_files)


def stash(path):
    """Clear a generator's fixed output path; return what to put back if generation fails."""
    if path.is_symlink():
        target = os.readlink(path)
        path.unlink()
        return 'link', target
    if path.exists():
        kept = store.aside_path(path, 'local')
        os.rename(path, kept)
        return 'directory', kept
    return None


def restore(path, previous):
    if previous is None:
        return
    kind, value = previous
    if kind == 'link':
        os.symlink(value, path)
    else:
        os.rename(value, path)


def move(source, destination):
    """Rename a tree into store staging, cloning then removing it when the store is on another volume."""
    try:
        os.rename(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        store.clone(source, destination)
        store.remove_staged(source)


def generate(family, root, store_dir, key, sources, runner):
    """Run the steps into a fresh staging tree; return it, its run files and the stashed checkout.

    Nothing is published. On any failure, including an interrupt, the partial
    output is removed and build/oracle_data/<family> is put back as it was.
    """
    checkout = checkout_path(root, family)
    staged = store.staging_path(store_dir, f'fixtures-{family.name}-{key}')
    output = checkout if family.in_place else staged
    previous = stash(checkout) if family.in_place else None
    try:
        run_steps(family, root, output, runner)
        if inputs(root) != sources:
            raise RuntimeError(f'Fixture generator inputs changed while generating {family.name}; '
                               'nothing was published. Run validate again.')
        run_files = {}
        for name in family.provenance:
            run_files[name] = json.loads((output / name).read_text())
            (output / name).unlink()
        if family.in_place:
            move(checkout, staged)
    except BaseException:
        if family.in_place:
            store.remove_staged(checkout)
            restore(checkout, previous)
        store.remove_staged(staged)
        raise
    return staged, run_files, previous


def regenerable(family, directory, record):
    """True when a validation wrote every file: an entry file of the same size, a run file or .DS_Store."""
    if directory.is_symlink() or not directory.is_dir():
        return False
    patterns = family.provenance + family.regenerable
    for name, size in scan(directory).items():
        if size is None:
            return False
        described = record['files'].get(name)
        if described is not None and described['bytes'] == size:
            continue
        if Path(name).name != '.DS_Store' and not any(fnmatch.fnmatchcase(name, p) for p in patterns):
            return False
    return True


def discard(family, directory, record):
    """Remove a set-aside checkout copy that validate can regenerate; keep anything more."""
    if regenerable(family, directory, record):
        store.remove_staged(directory)
    else:
        print(f'{family.name}: kept {directory}; it holds files validate does not generate. '
              'Delete it when you no longer need them.', flush=True)


def attach(family, root, entry, record):
    """Link build/oracle_data/<family> to the entry, never discarding what validate cannot regenerate."""
    checkout = checkout_path(root, family)
    if store.link_state(checkout, entry.tree) == 'local':
        kept = store.aside_path(checkout, 'local')
        os.rename(checkout, kept)
        discard(family, kept, record)
    store.ensure_link(checkout, entry.tree)


def summary(record):
    files = record['files'].values()
    return f"{len(files)} files, {sum(file['bytes'] for file in files) / 1e9:.2f} GB"


def write_record(path, record):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
        store.publish_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_space(family, root, store_dir):
    """Room for one generation where it runs and where it is published (one check on one volume)."""
    places = [Path(store_dir)] + ([Path(root) / 'build'] if family.in_place else [])
    short = [check.detail for check in (toolchain.disk_check(place, family.estimated_bytes) for place in places)
             if not check.ok]
    if short:
        raise RuntimeError(f'Not enough free space to generate {family.name} fixtures: {"; ".join(short)}. '
                           'Free space, or set LLM_MOJO_CACHE_DIR to another volume.')


def rebuild(family, root, store_dir, key, entry, sources, runner, cached, problem):
    """Under the entry's lock: generate, then publish, or compare with a valid cached entry."""
    for stale in (Path(store_dir) / '.staging').glob(f'fixtures-{family.name}-{key}.*'):
        store.remove_staged(stale)
    require_space(family, root, store_dir)
    if cached is None and (entry.tree.exists() or entry.tree.is_symlink()):
        print(f'{family.name}: the cached entry is damaged ({problem}); regenerating it.', flush=True)
        store.set_aside(entry.tree)
        if entry.record.exists():
            store.set_aside(entry.record)
    print(f'{family.name}: generating fixtures {key[:12]}', flush=True)
    staged, run_files, previous = generate(family, root, store_dir, key, sources, runner)
    differ = []
    try:
        generated = table(staged)
        if cached is None:
            record = dict(format=FORMAT, key=key, family=family.name, steps=[list(step) for step in family.steps],
                          inputs=sources, files=generated, generation=generation(root, run_files))
            write_record(entry.record, record)
            store.publish_directory(staged, entry.tree)
            outcome = 'generated'
        else:
            record = cached
            differ = sorted(name for name in generated.keys() | cached['files'].keys()
                            if generated.get(name) != cached['files'].get(name))
            if differ:
                kept = store.publish_directory(staged, store.aside_path(entry.tree, 'regenerated'))
            else:
                store.remove_staged(staged)
            outcome = 'reproduced'
    except BaseException:
        store.remove_staged(staged)
        restore(checkout_path(root, family), previous)
        raise
    attach(family, root, entry, record)
    if previous is not None and previous[0] == 'directory':
        discard(family, previous[1], record)
    if differ:
        raise RuntimeError(f"{family.name}: regenerated fixtures differ from the cached entry: {', '.join(differ[:5])}"
                           f"{' and more' if len(differ) > 5 else ''}. The cached entry stays linked; the new "
                           f'tree is kept at {kept} for comparison.')
    print(f'{family.name}: {"reproduced byte-for-byte" if outcome == "reproduced" else "published"} '
          f'({summary(record)}) at {entry.tree}', flush=True)
    return outcome


def ensure(family, root, sources, runner, *, store_dir=None, regenerate=False):
    """Link build/oracle_data/<family> to a verified store entry, generating it on a miss.

    With regenerate, generate even on a hit and require a byte-for-byte match.
    Returns 'hit', 'generated' or 'reproduced'.
    """
    store_dir = store.store_root(store_dir)
    key = identity(family, sources)
    entry = entry_paths(store_dir, family, key)
    started = time.monotonic()
    record, problem = verify(entry, key)
    if record is None or regenerate:
        with locked(entry.lock, f'{family.name}: waiting for another validation to finish generating it'):
            started = time.monotonic()
            record, problem = verify(entry, key)
            if record is None or regenerate:
                return rebuild(family, root, store_dir, key, entry, sources, runner, record, problem)
    attach(family, root, entry, record)
    print(f'{family.name}: verified cached fixtures {key[:12]} ({summary(record)}) '
          f'in {time.monotonic() - started:.1f} s', flush=True)
    then, now = record.get('generation', {}), machine()
    if (then.get('mac_ver'), then.get('python')) != (now['mac_ver'], now['python']):
        print(f"{family.name}: note: generated on macOS {then.get('mac_ver')} with Python {then.get('python')}; "
              f"this Mac runs macOS {now['mac_ver']} with Python {now['python']}. The entry still matches its "
              'record; --regenerate-fixtures checks that this Mac reproduces it.', flush=True)
    return 'hit'


def generate_locally(family, root, runner):
    """Without the cache: run the steps into build/oracle_data/<family> as before.

    A cache link there is removed first; the store is never touched.
    """
    checkout = checkout_path(root, family)
    if checkout.is_symlink():
        checkout.unlink()
    run_steps(family, root, checkout, runner)


def family_named(name):
    if name not in FAMILIES:
        raise ValueError(f"unknown fixture family {name!r}; choose from {', '.join(FAMILIES)}")
    return FAMILIES[name]


def detach(name, root=None):
    """Swap build/oracle_data/<name>'s link for a writable copy-on-write clone of its entry.

    For generator commands run by hand, which write into the family directory.
    The next validate links the family again, keeping any extra files beside it.
    """
    family = family_named(name)
    checkout = checkout_path(root or repository_root(), family)
    if not checkout.is_symlink():
        if checkout.exists():
            return f'{name}: {checkout} is already a writable copy'
        return f'{name}: {checkout} is not linked; generators create it'
    target = checkout.resolve()
    if not target.is_dir():
        checkout.unlink()
        return f'{name}: removed a dangling link at {checkout}; generators create the directory'
    copy = store.aside_path(checkout, 'detaching')
    try:
        store.clone(target, copy)
        for directory, _, files in os.walk(copy):
            os.chmod(directory, 0o755)
            for file in files:
                os.chmod(Path(directory, file), 0o644)
        checkout.unlink()
        os.rename(copy, checkout)
    except BaseException:
        store.remove_staged(copy)
        if not checkout.exists() and not checkout.is_symlink():
            os.symlink(target, checkout)
        raise
    return (f'{name}: {checkout} is now a writable copy of {target}. The next validate links it again '
            'and keeps any files it does not generate beside it.')


KEY = re.compile(r'[0-9a-f]{64}')
ASIDE = re.compile(r'\.(invalid|regenerated)-\d{8}-\d{6}(-\d+)?$')
STAGED = re.compile(r'fixtures-(?P<family>\w+)-(?P<key>[0-9a-f]{64})\.')


def worktrees(root):
    """This repository's worktrees that still exist, this checkout first."""
    found = [Path(root)]
    try:
        listing = subprocess.run(['git', 'worktree', 'list', '--porcelain'], cwd=root, capture_output=True,
                                 text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return found
    for line in listing.splitlines():
        path = Path(line[len('worktree '):]) if line.startswith('worktree ') else None
        if path is not None and path.is_dir() and path.resolve() != Path(root).resolve():
            found.append(path)
    return found


def usage(root, store_dir):
    """(family, key) -> worktrees linking that entry, and -> worktrees whose current inputs select it."""
    linked, selected = {}, {}
    base = (Path(store_dir) / 'fixtures').resolve()
    for tree in worktrees(root):
        for link in (tree / 'build/oracle_data').glob('*'):
            if link.is_symlink():
                target = link.parent / os.readlink(link)
                if target.parent.resolve() == base / link.name:
                    linked.setdefault((link.name, target.name), []).append(tree)
        try:
            sources = inputs(tree)
        except RuntimeError:
            continue
        for family in FAMILIES.values():
            selected.setdefault((family.name, identity(family, sources)), []).append(tree)
    return linked, selected


def survey(store_dir):
    """(family, name, path, kind, key) for everything the cache keeps in the store.

    Kinds: entry, set aside, orphaned record, orphaned lock and staging. Files the
    cache did not create are never listed, so nothing here can remove them.
    """
    base = Path(store_dir) / 'fixtures'
    for directory in sorted(base.iterdir()) if base.is_dir() else []:
        if directory.is_symlink() or not directory.is_dir():
            continue
        names = {path.name for path in directory.iterdir()}
        for path in sorted(directory.iterdir()):
            name = path.name
            if ASIDE.search(name):
                yield directory.name, name, path, 'set aside', None
            elif KEY.fullmatch(name):
                yield directory.name, name, path, 'entry', name
            elif name.endswith('.json') and KEY.fullmatch(name[:-5]) and name[:-5] not in names:
                yield directory.name, name, path, 'orphaned record', name[:-5]
            elif (name.startswith('.') and name.endswith('.lock') and KEY.fullmatch(name[1:-5])
                  and name[1:-5] not in names and name[1:-5] + '.json' not in names):
                yield directory.name, name, path, 'orphaned lock', name[1:-5]
    staging = Path(store_dir) / '.staging'
    for path in sorted(staging.iterdir()) if staging.is_dir() else []:
        match = STAGED.match(path.name)
        if match:
            yield match['family'], path.name, path, 'staging', match['key']


def footprint(path, record=None):
    if record is not None:
        return sum(file['bytes'] for file in record['files'].values())
    if path.is_dir() and not path.is_symlink():
        return sum(size or 0 for size in scan(path).values())
    return path.lstat().st_size


def stored_record(path):
    try:
        record = json.loads(path.with_name(path.name + '.json').read_text())
        return record if record.get('format') == FORMAT and record.get('key') == path.name else None
    except (OSError, ValueError, AttributeError):
        return None


@contextmanager
def unless_locked(path):
    """Yield True while holding path's lock without waiting, or False when another process holds it."""
    try:
        stream = path.open('r')
    except FileNotFoundError:
        yield True
        return
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def label(tree, root):
    tree = Path(tree)
    if tree.resolve() == Path(root).resolve():
        return 'this checkout'
    return '~/' + tree.relative_to(Path.home()).as_posix() if tree.is_relative_to(Path.home()) else str(tree)


def report(root=None, store_dir=None):
    """Each cached entry with its size, origin and the worktrees that link or select it."""
    store_dir, root = store.store_root(store_dir), Path(root or repository_root())
    linked, selected = usage(root, store_dir)
    lines = [f"Shared oracle fixtures in {Path(store_dir) / 'fixtures'}"]
    for family, name, path, kind, key in survey(store_dir):
        if kind == 'entry':
            record = stored_record(path)
            origin = record.get('generation', {}) if record else {}
            users = ', '.join(label(tree, root) for tree in linked.get((family, key), [])) or 'none'
            current = ', '.join(label(tree, root) for tree in selected.get((family, key), [])) or 'none'
            lines += [f"{family}/{key[:12]}  {footprint(path, record) / 1e9:.2f} GB  generated "
                      f"{origin.get('time', 'at an unknown time')} at {(origin.get('commit') or 'unknown')[:7]}",
                      f'  linked by: {users}', f'  current for: {current}']
        else:
            lines.append(f'{family}/{name}  {footprint(path) / 1e9:.2f} GB  {kind}')
    return '\n'.join(lines if len(lines) > 1 else [*lines, 'No entries.'])


def prune(yes=False, root=None, store_dir=None):
    """Remove set-aside and stale items and entries no worktree links or selects; list them unless yes.

    Entries being generated or repaired are skipped. The worktrees considered
    are this repository's; run it when no validation is using the store.
    """
    store_dir, root = store.store_root(store_dir), Path(root or repository_root())
    linked, selected = usage(root, store_dir)
    lines, total, count = [], 0, 0
    for family, name, path, kind, key in survey(store_dir):
        if kind == 'entry' and ((family, key) in linked or (family, key) in selected):
            continue
        lock = Path(store_dir) / 'fixtures' / family / f'.{key}.lock' if key else None
        with unless_locked(lock) if lock else nullcontext(True) as free:
            if not free:
                lines.append(f'Skipped {family}/{name}: a validation is generating it.')
                continue
            size = footprint(path, stored_record(path) if kind == 'entry' else None)
            total, count = total + size, count + 1
            reason = 'no worktree links or selects it' if kind == 'entry' else kind
            lines.append(f"{'Removed' if yes else 'Would remove'} {family}/{name}  {size / 1e9:.2f} GB  ({reason})")
            if yes:
                store.remove_staged(path)
                if kind == 'entry':
                    path.with_name(name + '.json').unlink(missing_ok=True)
                if kind in ('entry', 'orphaned record', 'orphaned lock'):
                    lock.unlink(missing_ok=True)
    if not count:
        return '\n'.join([*lines, 'Nothing to prune.'])
    summary_line = f"{'Removed' if yes else 'Would remove'} {count} item{'s' * (count != 1)}, {total / 1e9:.2f} GB."
    return '\n'.join([*lines, summary_line] + ([] if yes else ['Run again with --yes to remove them.']))
