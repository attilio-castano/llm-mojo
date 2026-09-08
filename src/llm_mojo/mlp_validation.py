"""Build and execute the exact MLP numerical candidate, retaining run receipts."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from ._repository import environment_tool, repository_root
from .benchmarks.environment import ensure_record_location, repository_state, utc_now

SPLITS = ('holdout', 'optimization_holdout', 'decode_holdout')
STAGES = ('N', 'G', 'U', 'A', 'S', 'D', 'Y')
REUSE_CASE = 'h896_i4864_r17_s1601'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_identity():
    root = repository_root()
    paths = [p for base in ('src', 'tests') for p in (root / base).rglob('*')
             if p.is_file() and p.suffix in ('.mojo', '.py', '.json', '.gz', '.lock')]
    paths += [root / 'pyproject.toml', root / 'uv.lock']
    return dict(repository=repository_state(),
                sources={str(p.relative_to(root)): sha(p) for p in sorted(paths)})


def execution_environment():
    # Explicit selection must not inherit a case filter, another split, stale
    # result directory, or debug synchronization from the invoking shell.
    return {k: v for k, v in os.environ.items()
            if not k.startswith('MLP_') and k != 'MODULAR_DEBUG'}


def build(binary):
    binary = Path(binary).resolve()
    ensure_record_location(binary)
    receipt = Path(str(binary) + '.provenance.json')
    if binary.exists() or receipt.exists():
        raise ValueError('refusing to overwrite a numerical build')
    before = source_identity()
    if before['repository']['dirty']:
        raise ValueError('numerical build requires clean source')
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [environment_tool('mojo'), 'build', '-I', 'src', '-I', 'tests',
               'tests/test_mlp.mojo', '-o', str(binary)]
    subprocess.run(command, cwd=repository_root(), env=execution_environment(), check=True)
    if source_identity() != before:
        raise ValueError('source changed during numerical build')
    write(receipt, dict(schema=1, kind='mlp_numerical_build', source=before,
                        command=command, binary_sha256=sha(binary), created_utc=utc_now()))


def verify_build(binary):
    """Also used by the isolated capture script before exposing holdouts."""
    binary = Path(binary).resolve()
    receipt = Path(str(binary) + '.provenance.json')
    record = json.loads(receipt.read_text())
    if (record.get('schema') != 1 or record.get('kind') != 'mlp_numerical_build'
        or record['binary_sha256'] != sha(binary)
        or record['source']['repository']['dirty']
        or record['source'] != source_identity()):
        raise ValueError('candidate binary or build source does not match its receipt')
    return record


def inputs(split):
    root = repository_root()
    directory = root / 'build/oracle_data' / ('mlp_' + split)
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    payload = {k: v for k, v in manifest.items() if k != 'manifest_payload_sha256'}
    if (manifest.get('status') != 'complete'
        or hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        != manifest.get('manifest_payload_sha256')):
        raise ValueError('incomplete or changed holdout capture')
    anchor = root / 'tests/fixtures/mlp/checksums.json'
    if sha(anchor) != manifest['candidate']['reference_sha256']:
        raise ValueError('holdout reference changed')
    if split != 'holdout':
        declaration = root / 'tests/fixtures' / ('mlp_' + split + '.json')
        if sha(declaration) != manifest['candidate']['holdout_spec_sha256']:
            raise ValueError('holdout declaration changed')
    hashes = {'manifest.json': sha(path)}
    for name, spec in manifest['arrays'].items():
        if sha(directory / name) != spec['sha256']:
            raise ValueError('holdout array changed: ' + name)
        hashes[name] = spec['sha256']
    # The suite's asynchronous reuse checks always use this development case.
    frozen = json.loads(gzip.decompress((root / 'tests/fixtures/mlp/development.json.gz').read_bytes()))
    for name, spec in frozen['arrays'].items():
        if name.startswith(REUSE_CASE + '/'):
            actual = sha(root / 'build/oracle_data/mlp' / name)
            if actual != spec['sha256']:
                raise ValueError('reuse fixture changed: ' + name)
            hashes['development/' + name] = actual
    return manifest, hashes


def expected_checks(cases, variants):
    expected = Counter()
    for variant in variants:
        for case, spec in cases.items():
            rows = 1 if variant >= 8 else spec['rows']
            for mode in ('local', 'full'):
                for stage in STAGES:
                    expected[variant, case, mode, stage, 0, rows] += 1
            chunks = [1] * rows if rows <= 17 else [rows - 18, 17, 1]
            start = 0
            for count in chunks:
                for stage in STAGES:
                    expected[variant, case, 'chunk', stage, start, count] += 1
                start += count
        lengths = [1] * 12 if variant >= 8 else [17, 1, 7, 15, 16, 1, 17, 7, 1, 16, 15, 1]
        for j, rows in enumerate(lengths):
            for stage in ('D', 'Y'):
                expected[variant, REUSE_CASE, 'reuse', stage, (j * 3) % (18 - rows), rows] += 1
    return expected


def validate_results(directory, cases, variants):
    observed = Counter()
    paths = sorted(Path(directory).glob('*.jsonl'))
    if len(paths) != 1:
        raise ValueError('evaluation requires one numerical result stream')
    for line in paths[0].read_text().splitlines():
        row = json.loads(line)
        key = tuple(row[k] for k in ('mapping', 'case', 'mode', 'stage', 'start', 'rows'))
        observed[key] += 1
        spec = dict(hidden=896, intermediate=4864) if row['mode'] == 'reuse' else cases[row['case']]
        width = spec['hidden'] if row['stage'] in ('N', 'D', 'Y') else spec['intermediate']
        if row['elements'] != row['rows'] * width:
            raise ValueError('numerical check has incomplete element coverage')
        gated = row['mode'] == 'local' or row['stage'] in ('D', 'Y')
        if row.get('failed', 0) != 0 or (gated and row.get('failed') != 0):
            raise ValueError('numerical gate failed or missing')
        if row['mode'] == 'chunk' and row['stage'] in ('D', 'Y'):
            if row.get('full_chunked', {}).get('failed') != 0:
                raise ValueError('full/chunk composition gate failed or missing')
    if observed != expected_checks(cases, variants):
        raise ValueError('incomplete, duplicate, or unexpected numerical coverage')
    return dict(checks=sum(observed.values()), results_sha256={p.name: sha(p) for p in paths})


def evaluate(binary, output, split, variants, *, regression=False):
    from .benchmarks.mlp_contract import VARIANTS
    if split not in SPLITS or not variants or len(set(variants)) != len(variants) or any(
        type(v) is not int or v not in VARIANTS for v in variants
    ):
        raise ValueError('explicit holdout split and unique supported mappings required')
    binary, output = Path(binary).resolve(), Path(output).resolve()
    ensure_record_location(output)
    build_record = verify_build(binary)
    manifest, fixture_hashes = inputs(split)
    if not regression and (manifest['candidate']['binary_sha256'] != build_record['binary_sha256']
                           or manifest['candidate']['commit'] != build_record['source']['repository']['commit']):
        raise ValueError('capture names a different candidate; use --regression for observed fixtures')
    output.mkdir(parents=True, exist_ok=False)
    results = output / 'checks'
    results.mkdir()
    env = execution_environment()
    env.update(MLP_SPLIT=split, MLP_VARIANTS=','.join(map(str, variants)), MLP_RECORD_DIR=str(results))
    record = dict(schema=1, kind='mlp_numerical_evaluation', status='started',
                  purpose='observed_holdout_regression' if regression else 'frozen_candidate_evaluation',
                  build=build_record, fixtures=fixture_hashes, command=[str(binary)],
                  python=sys.executable, environment={k: v for k, v in env.items() if k.startswith('MLP_')},
                  started_utc=utc_now())
    receipt = output / 'evaluation.json'
    write(receipt, record)
    log = output / 'output.log'
    try:
        with log.open('w') as stream:
            process = subprocess.run([str(binary)], cwd=repository_root(), env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
        record['exit_code'] = process.returncode
        record['source_unchanged'] = verify_build(binary) == build_record
        if not record['source_unchanged'] or inputs(split) != (manifest, fixture_hashes):
            raise ValueError('build or fixtures changed during evaluation')
        if process.returncode:
            raise ValueError('candidate numerical suite failed')
        runtime = re.findall(r'^MLP runtime: (Apple .+) metal (\S+) mapping (\d+)$', log.read_text(), re.M)
        if (Counter((case, int(v)) for _, case, v in runtime)
            != Counter((case, v) for case in manifest['cases'] for v in variants)
            or len({device for device, _, _ in runtime}) != 1):
            raise ValueError('missing or inconsistent Metal runtime coverage')
        record.update(validate_results(results, manifest['cases'], variants))
        record.update(status='passed', runtime=dict(device=runtime[0][0], backend='metal'))
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        record.update(finished_utc=utc_now(), output_sha256=sha(log) if log.exists() else None)
        record['result_files_sha256'] = {p.name: sha(p) for p in sorted(results.glob('*.jsonl'))}
        write(receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build_parser = commands.add_parser('build')
    build_parser.add_argument('--binary', type=Path, required=True)
    run_parser = commands.add_parser('evaluate')
    run_parser.add_argument('--binary', type=Path, required=True)
    run_parser.add_argument('--output', type=Path, required=True)
    run_parser.add_argument('--split', choices=SPLITS, required=True)
    run_parser.add_argument('--variants', type=int, nargs='+', required=True)
    run_parser.add_argument('--regression', action='store_true')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.binary)
    else:
        evaluate(args.binary, args.output, args.split, args.variants, regression=args.regression)


if __name__ == '__main__':
    main()
