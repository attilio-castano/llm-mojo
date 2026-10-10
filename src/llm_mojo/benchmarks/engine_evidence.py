"""Verify and restore the original Phase 3 evidence layout without GPU execution.

The catalog changes storage only. Historical records are reconstructed byte for
byte; their source identities, numerical gates, commands, and exits stay intact.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


CATALOG = 'engine-evidence.json'


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _relative(name):
    if (type(name) is not str or not name or '\\' in name
            or PurePosixPath(name).is_absolute()
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('evidence path must be a normalized relative path')
    return Path(name)


def _checked(data, record, label):
    if len(data) != record['bytes'] or _sha(data) != record['sha256']:
        raise ValueError('evidence identity mismatch: ' + label)
    return data


def original_files(root):
    """Load every original file after verifying its retained storage and identity."""
    root = Path(root).resolve()
    catalog = json.loads((root / CATALOG).read_bytes())
    if (catalog.get('kind') != 'engine-evidence-catalog-v1'
            or not isinstance(catalog.get('files'), dict) or not catalog['files']):
        raise ValueError('invalid engine evidence catalog')
    stored, archives, result = {}, {}, {}

    def storage(source, expected=None):
        name = source['path']
        path = _relative(name)
        if name not in stored:
            resolved = (root / path).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError('evidence storage escapes the catalog root')
            stored[name] = resolved.read_bytes()
        return _checked(stored[name], expected or source, name)

    for name, record in catalog['files'].items():
        _relative(name)
        source = record['storage']
        form = source['format']
        encoded = storage(source, record if form == 'file' else None)
        if form == 'file':
            raw = encoded
        elif form == 'gzip':
            raw = gzip.decompress(encoded)
        elif form == 'archive-member':
            archive_name = source['path']
            manifest_raw = storage(source['manifest'])
            if archive_name not in archives:
                manifest = json.loads(manifest_raw)
                unpacked = gzip.decompress(encoded)
                if (len(encoded) != manifest['bytes'] or _sha(encoded) != manifest['sha256']
                        or len(unpacked) != manifest['uncompressed_bytes']
                        or _sha(unpacked) != manifest['uncompressed_sha256']):
                    raise ValueError('validation archive manifest mismatch: ' + archive_name)
                files = json.loads(unpacked)['files']
                members = {}
                for entry in files:
                    member = entry['name']
                    _relative(member)
                    if member in members:
                        raise ValueError('duplicate validation archive member: ' + member)
                    members[member] = _checked(entry['text'].encode('utf-8'), entry, member)
                archives[archive_name] = (manifest_raw, members)
            elif manifest_raw != archives[archive_name][0]:
                raise ValueError('conflicting validation archive manifests: ' + archive_name)
            member = source['member']
            _relative(member)
            raw = archives[archive_name][1][member]
        else:
            raise ValueError('unknown evidence storage format: ' + str(form))
        result[name] = _checked(raw, record, name)
    return catalog, result


def verify(root):
    catalog, files = original_files(root)
    return dict(kind='engine-evidence-verification-v1', basis_commit=catalog['basis_commit'],
                catalog_sha256=_sha((Path(root) / CATALOG).read_bytes()),
                original_files=len(files), original_bytes=sum(map(len, files.values())),
                reconstructed_files=[dict(path=name, bytes=len(raw), sha256=_sha(raw))
                                     for name, raw in files.items()
                                     if catalog['files'][name]['storage']['format'] != 'file'],
                native_execution=False)


def restore(root, directory):
    """Create a fresh original-layout directory only after every file verifies."""
    catalog, files = original_files(root)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        destination = directory / _relative(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(raw)
    return dict(kind='engine-evidence-restoration-v1', basis_commit=catalog['basis_commit'],
                catalog_sha256=_sha((Path(root) / CATALOG).read_bytes()),
                restored_directory=str(directory.resolve()), original_files=len(files),
                original_bytes=sum(map(len, files.values())), native_execution=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'restore'))
    parser.add_argument('--root', type=Path, default=Path('studies/model_generation'))
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.action == 'restore':
        if args.directory is None:
            parser.error('restore requires --directory')
        result = restore(args.root, args.directory)
    else:
        if args.directory is not None:
            parser.error('--directory is only valid for restore')
        result = verify(args.root)
    print(json.dumps(result, indent=2) + '\n', end='')


if __name__ == '__main__':
    main()
