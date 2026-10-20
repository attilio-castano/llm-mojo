"""Cold-cache finishing admission measurements, separate from GPU timings.

Run with the locked environment. Outputs are create-only and outside Git.
Every invocation uses a new compiler cache, retains raw pass/resource reports,
hashes all Mojo inputs before/after, and runs the resulting checked probe.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

from llm_mojo._repository import environment_tool, repository_root


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(source_root, probe):
    return dict(sources={str(path.relative_to(source_root)): digest(path)
                         for path in sorted(source_root.rglob('*.mojo'))},
                probe_sha256=digest(probe))


def checked_source_revision(root, revision, sources):
    """A revision label is evidence only after every source byte matches Git."""
    commit = subprocess.check_output(
        ['git', 'rev-parse', '--verify', revision + '^{commit}'], cwd=root, text=True).strip()
    archive = subprocess.check_output(['git', 'archive', commit, 'src'], cwd=root)
    expected = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as entries:
        for member in entries:
            if member.isfile() and member.name.startswith('src/') and member.name.endswith('.mojo'):
                expected[member.name[4:]] = hashlib.sha256(entries.extractfile(member).read()).hexdigest()
    if expected != sources:
        raise ValueError('source-root does not exactly match the requested Git revision')
    return commit


def file_revision(root, path, revision):
    if not path.is_relative_to(root):
        return None
    result = subprocess.run(['git', 'show', f'{revision}:{path.relative_to(root)}'],
                            cwd=root, capture_output=True)
    return revision if result.returncode == 0 and result.stdout == path.read_bytes() else None


def run(args):
    if sys.platform != 'darwin':
        raise RuntimeError('this resource profile uses macOS /usr/bin/time -l')
    root = repository_root()
    source_root, output = (path.resolve() for path in (args.source_root, args.output))
    default_probes = {'proof': root / 'tests/finishing_compilation_probe.mojo',
                      'operations': source_root / 'llm_mojo/benchmarks/operations.mojo',
                      'numerical': root / 'tests/test_token_selection.mojo'}
    probe = (args.probe or default_probes[args.target]).resolve()
    if args.target != 'proof' and (args.stage != 4 or args.copies != 1 or args.reuse != 1):
        raise ValueError('stage/copies/reuse select the proof probe, not a GPU target')
    if output.is_relative_to(root):
        raise ValueError('keep compiler caches, binaries and full reports outside the repository')
    if not (source_root / 'llm_mojo/kernels/proof_term_kernel.mojo').is_file():
        raise ValueError('source-root must contain the llm_mojo package')
    identity = source_identity(source_root, probe)
    checkout = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    try:
        source_revision = checked_source_revision(root, args.source_revision or checkout, identity['sources'])
    except ValueError:
        if args.source_revision:
            raise
        source_revision = None
    output.mkdir(parents=True, exist_ok=False)
    compiler = environment_tool('mojo')
    command = [compiler, 'build', '-j', str(args.threads), '--mlir-timing',
               '--timing-json', '--timing-file', str(output / 'passes.json'),
               '-I', str(source_root)]
    if args.target == 'proof':
        command += ['-D', f'PROFILE_STAGE={args.stage}', '-D', f'PROFILE_COPIES={args.copies}',
                    '-D', f'PROFILE_REUSE={args.reuse}']
    command += [str(probe), '-o', str(output / 'probe')]
    record = dict(schema_version=1, purpose='host compilation; no GPU performance claim',
                  captured_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  compiler=subprocess.check_output([compiler, '--version'], text=True).strip(),
                  host=dict(platform=platform.platform(), machine=platform.machine(),
                            processor=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                            memory_bytes=int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
                            power=subprocess.check_output(['pmset', '-g', 'batt'], text=True).strip()),
                  checkout_commit=checkout, source_revision=source_revision,
                  driver_revision=file_revision(root, Path(__file__).resolve(), checkout),
                  probe_revision=(source_revision if probe.is_relative_to(source_root)
                                  else file_revision(root, probe, checkout)),
                  checkout_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True),
                  source_root=str(source_root), probe_source=str(probe), inputs=identity,
                  compiler_cwd=str(source_root.parent),
                  mojo_path_environment=os.environ.get('MOJO_PATH'),
                  driver_sha256=digest(Path(__file__)), lock_sha256=digest(root / 'uv.lock'),
                  cache_policy='fresh directory for this invocation',
                  target=args.target,
                  stage=args.stage if args.target == 'proof' else None,
                  copies=args.copies if args.target == 'proof' else None,
                  reuse=args.reuse if args.target == 'proof' else None,
                  compiler_threads=args.threads, optimization_level=3,
                  command=command)
    (output / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    environment = {**os.environ, 'MODULAR_CACHE_DIR': str(output / 'cache')}
    start = time.monotonic_ns()
    with (output / 'compiler.log').open('wb') as log:
        compiled = subprocess.run(['/usr/bin/time', '-l', *command], env=environment,
                                  stdout=log, stderr=subprocess.STDOUT, cwd=source_root.parent)
    record.update(compile_wall_ns=time.monotonic_ns() - start,
                  compiler_exit_code=compiled.returncode,
                  inputs_unchanged=source_identity(source_root, probe) == identity)
    if compiled.returncode == 0:
        run_command = [str(output / 'probe')]
        run_env = environment.copy()
        if args.target != 'proof':
            run_env['MODULAR_DEBUG'] = 'device-sync-mode'
        if args.target == 'operations':
            run_command += ['token_selection', '1', '1', '2', '0', '1', '53', 'bench', '10', '0']
        run_start = time.monotonic_ns()
        executed = subprocess.run(run_command, capture_output=True, text=True, env=run_env,
                                  cwd=source_root.parent, timeout=180)
        run_wall_ns = time.monotonic_ns() - run_start
        (output / 'probe.log').write_text(executed.stdout + executed.stderr)
        if args.target == 'proof':
            expected = [f'Checked compilation profile caller {i} stage {args.stage}'
                        for i in range(args.copies)]
            matches = executed.stdout.splitlines() == expected
        elif args.target == 'operations':
            from llm_mojo.benchmarks.study import parse_output
            try:
                parse_output(executed.stdout, 0, 2, True, rows=1, layers=1, seed=53,
                             operation='token_selection')
                matches = True
            except ValueError as error:
                matches = False
                record['probe_validation_error'] = str(error)
        else:
            lines = executed.stdout.splitlines()
            matches = (len(lines) >= 2 and lines[0].startswith('device: Apple ')
                       and lines[0].endswith('api: metal')
                       and lines[-1] == 'Token selection ordering, rounded ties, checked schedules, runtime folds and rejection tests passed')
        record.update(binary_sha256=digest(output / 'probe'),
                      run_command=run_command,
                      run_wall_ns=run_wall_ns,
                      probe_exit_code=executed.returncode,
                      probe_output_matches=matches)
    record['inputs_unchanged'] = source_identity(source_root, probe) == identity
    record['passed'] = (compiled.returncode == 0 and record['inputs_unchanged']
                        and record.get('probe_exit_code') == 0
                        and record.get('probe_output_matches') is True)
    record['artifact_sha256'] = {name: digest(output / name)
                                 for name in ('passes.json', 'compiler.log', 'probe.log')
                                 if (output / name).is_file()}
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: record[key] for key in
                      ('passed', 'compile_wall_ns', 'compiler_exit_code', 'inputs_unchanged')}))
    return 0 if record['passed'] else 1


def main():
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=root / 'src')
    parser.add_argument('--source-revision', help='require source bytes to match this Git commit')
    parser.add_argument('--target', choices=('proof', 'operations', 'numerical'), default='proof')
    parser.add_argument('--probe', type=Path, help='override the selected target source')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', type=int, choices=range(5), default=4)
    parser.add_argument('--copies', type=int, choices=range(1, 33), default=1)
    parser.add_argument('--reuse', type=int, choices=(0, 1), default=1)
    parser.add_argument('--threads', type=int, choices=range(1, 33), default=2)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
