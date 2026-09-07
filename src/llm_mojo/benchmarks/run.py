"""Build once and record bounded, paired kernel studies on the Apple GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from .._repository import environment_tool, repository_root

from .environment import (conditions_snapshot, ensure_record_location,
                         repository_state, require_ac, require_nominal_thermal_state,
                         stable_environment, utc_now)
from .study import (STUDIES, BLOCKS, REPETITIONS, WARMUP, sha, write_json,
                   encode_samples, parse_output, summarize, workloads, load_run,
                   select_parallelism_finalists, select_projection_tile)
from .attention_sublayer_contract import fixture_identity
from .mlp_contract import fixture_identity as mlp_fixture_identity



def source_hashes():
    root = repository_root()
    paths = [*root.glob('src/**/*.mojo'), *root.glob('src/**/*.py'),
             *root.glob('tests/fixtures/**/*.py'), *root.glob('tests/fixtures/**/*.json'),
             *root.glob('tests/fixtures/**/*.lock'), root / 'pyproject.toml', root / 'uv.lock']
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}



def build(directory):
    ensure_record_location(directory)
    repo = repository_state()
    if repo['dirty']:
        raise RuntimeError('recorded builds require a clean commit')
    directory.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    fixtures = fixture_identity()
    mlp_fixtures = mlp_fixture_identity()
    commands, binaries = {}, {}
    env = {k: v for k, v in os.environ.items() if k != 'MODULAR_DEBUG'}
    for name, source in [('operations', 'operations.mojo'), ('gqa_decode', 'attention_decode.mojo'), ('gqa_prefill','attention_prefill.mojo'), ('attention_sublayer','attention_sublayer.mojo'), ('mlp','mlp.mojo')]:
        command = [environment_tool('mojo'), 'build', '-I', 'src',
                   f'src/llm_mojo/benchmarks/{source}', '-o', str(directory / name)]
        subprocess.run(command, cwd=repository_root(), env=env, check=True)
        binaries[name] = sha(directory / name)
        commands[name] = ['mojo', *command[1:-1], '<binary>']
    if repository_state() != repo or source_hashes() != sources or fixture_identity() != fixtures or mlp_fixture_identity() != mlp_fixtures:
        raise RuntimeError('source changed during build')
    write_json(directory / 'build.json', dict(repository=repo, sources=sources, binaries=binaries,
                                             commands=commands, environment=stable_environment(), attention_fixtures=fixtures, mlp_fixtures=mlp_fixtures))


def checked_conditions():
    conditions = conditions_snapshot()
    require_ac(conditions)
    require_nominal_thermal_state(conditions)
    # pmset uses lowpowermode on this machine; retain and reject its active setting.
    raw = subprocess.check_output(['pmset', '-g'], text=True)
    conditions['power_settings'] = [line.strip() for line in raw.splitlines()
                                    if 'lowpowermode' in line or 'powermode' in line]
    if any(line.split()[-1] == '1' for line in conditions['power_settings']):
        raise RuntimeError('recorded run requires Low Power Mode off')
    return conditions


def run(build_dir, output, study_names, *, parallelism_screen=None, tile_screen=None, tile_kernel_screen=None):
    ensure_record_location(output)
    provenance = json.loads((build_dir / 'build.json').read_text())
    repo, sources = repository_state(), source_hashes()
    environment = stable_environment()
    if repo['dirty'] or repo != provenance['repository'] or sources != provenance['sources']:
        raise RuntimeError('build requires the same clean source commit')
    if environment != provenance['environment']:
        raise RuntimeError('hardware/software changed since build')
    for name, digest in provenance['binaries'].items():
        if sha(build_dir / name) != digest:
            raise RuntimeError('binary identity changed')
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if k != 'MODULAR_DEBUG'}
    for name in study_names:
        spec = STUDIES[name]
        seed = spec.get('seed',53)
        if spec['operation'] == 'mlp' and mlp_fixture_identity() != provenance.get('mlp_fixtures'):
            raise RuntimeError('MLP benchmark inputs changed')
        selection = None
        if name == 'attention_sublayer_parallelism':
            if parallelism_screen is None:
                raise ValueError('parallelism full run requires its completed screen')
            screen, _, summary = load_run(parallelism_screen)
            declared = json.loads(json.dumps(STUDIES['attention_sublayer_parallelism_screen']))
            if (screen['study'] != 'attention_sublayer_parallelism_screen'
                or screen['specification'] != declared or screen['build'] != provenance):
                raise ValueError('parallelism screen must match this specification and build')
            finalists = select_parallelism_finalists(summary)
            if not finalists:
                raise ValueError('no parallelism candidate qualified; stop at the screen')
            spec = {**spec,'candidates':[9,*finalists]}
            selection = dict(finalists=finalists,screen_run_sha256=sha(parallelism_screen / 'run.json'),
                             screen_samples_sha256=screen['samples_sha256'],
                             rule='Both modes faster at (64,4096); minimum worst-mode ratio per family; lower-ID tie break.')
        if name in ('attention_sublayer_tiles','attention_sublayer_tiles_qkv'):
            records=[]
            summaries=[]
            for path,study in ((tile_screen,'attention_sublayer_tiles_screen'),
                               (tile_kernel_screen,'attention_sublayer_tiles_kernel_screen')):
                if path is None:
                    raise ValueError('projection follow-up requires both completed screens')
                screen,_,summary=load_run(path)
                if (screen['study']!=study or screen['build']!=provenance
                    or screen['specification']!=json.loads(json.dumps(STUDIES[study]))):
                    raise ValueError('projection screen must match this specification and build')
                records.append(dict(study=study,run_sha256=sha(path/'run.json'),
                                    samples_sha256=screen['samples_sha256']))
                summaries.append(summary)
            winner=select_projection_tile(*summaries)
            if winner is None:
                raise ValueError('no projection tile qualified; stop at screens')
            variant=winner+2 if name.endswith('_qkv') else winner
            spec={**spec,'candidates':[9,variant]}
            selection=dict(wo_finalist=winner,variant=variant,screens=records,
                           rule='Full 1024 faster for isolated Wo and whole attention in both modes; minimum worst-mode whole-block ratio, lower-ID tie break.')
        if spec['operation'] == 'attention_sublayer' and fixture_identity() != provenance.get('attention_fixtures'):
            raise RuntimeError('attention benchmark input identity changed')
        directory = output / name
        directory.mkdir()
        record = dict(schema=1, study=name, specification=spec, repository=repo,
                      build=provenance, seed=seed, repetitions=REPETITIONS, warmup=WARMUP, blocks=BLOCKS,
                      dtype=spec.get('arithmetic','BF16 operands/output; FP32 accumulation. GQA rounds scaled scores to BF16.'),
                      timing=spec.get('timing','Host monotonic enqueue through one synchronization per sample; microseconds per call. 24 distinct input or weight buffers, divided by 24; output/scratch reused.'),
                      inputs=spec.get('inputs','GQA deterministic signed recipe, seed + 13*layer; other operations analytical constants varying by layer, see operations.mojo. Numerical suites cover nonuniform data.'),
                      started_utc=utc_now(), conditions=[])
        if selection is not None:
            record['selection'] = selection
        samples = []
        for block in range(1, BLOCKS + 1):
            before = checked_conditions()
            record['conditions'].append(dict(block=block, before=before))
            first = block in (2, 3)
            cases = [(w, l, c) for w in workloads(spec) for l in spec.get('layers',(1,24)) for c in spec['candidates']]
            if first:
                cases.reverse()
            for workload, layers, candidate in cases:
                rows = workload['rows']
                binary_name = spec['operation'] if spec['operation'].startswith('gqa_') or spec['operation'] in ('attention_sublayer','mlp') else 'operations'
                command = [str(build_dir / binary_name)]
                if binary_name == 'operations':
                    command.append(spec['operation'])
                if binary_name in ('gqa_prefill', 'attention_sublayer'):
                    command.append(str(workload['query_rows']))
                command += list(map(str, [rows, layers, candidate, spec['control'], int(first), seed,
                                          spec.get('mode','bench'), REPETITIONS, WARMUP]))
                # A 4096-row MLP ring process performs 960 complete blocks.
                # The rowwise baseline needs about ten minutes on this host.
                timeout = 1200 if spec['operation'] == 'mlp' else 600 if spec['operation'] == 'attention_sublayer' else 300
                process = subprocess.run(command, cwd=repository_root(), capture_output=True, text=True, env=env, timeout=timeout)
                # Local diagnostic logs are useful during execution; compact samples are the retained evidence.
                (directory / 'last-process.txt').write_text(process.stdout + process.stderr)
                process.check_returncode()
                identity, observations = parse_output(process.stdout, spec['control'], candidate, first,
                                                      rows=rows, layers=layers, seed=seed, operation=spec['operation'], query_rows=workload.get('query_rows'), measurement=spec.get('measurement'))
                if record.get('runtime', identity) != identity:
                    raise RuntimeError('runtime identity changed')
                record['runtime'] = identity
                samples.extend(dict(block=block, **workload, layers=layers, candidate=candidate, **s) for s in observations)
                # Preserve completed cases if a later process fails. The
                # loader still rejects any run without completed_utc.
                (directory / 'samples.csv.gz').write_bytes(encode_samples(samples))
                write_json(directory / 'run.json', record)
            after = checked_conditions()
            record['conditions'][-1]['after'] = after
            (directory / 'samples.csv.gz').write_bytes(encode_samples(samples))
            write_json(directory / 'run.json', record)
            print(name, 'block', block, 'complete:', len(samples), 'observations', flush=True)
        summarize(samples, spec)  # Fail before marking completion if any measurement is missing.
        if spec['operation'] == 'attention_sublayer' and fixture_identity() != provenance['attention_fixtures']:
            raise RuntimeError('attention benchmark inputs changed during measurement')
        if spec['operation'] == 'mlp' and mlp_fixture_identity() != provenance['mlp_fixtures']:
            raise RuntimeError('MLP inputs changed during measurement')
        if repository_state() != repo or source_hashes() != sources or stable_environment() != environment:
            raise RuntimeError('source or hardware/software changed during measurement')
        record.update(completed_utc=utc_now(), samples_sha256=sha(directory / 'samples.csv.gz'))
        write_json(directory / 'run.json', record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['build', 'run'])
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--parallelism-screen', type=Path)
    p.add_argument('--tile-screen', type=Path)
    p.add_argument('--tile-kernel-screen', type=Path)
    p.add_argument('--studies', nargs='+', choices=list(STUDIES),
                   default=[name for name in STUDIES if not name.endswith('_screen') and not STUDIES[name].get('opt_in')
                            and name not in ('attention_sublayer_wo','attention_sublayer_decode','attention_sublayer_prefill',
                                             'attention_sublayer_projections','attention_sublayer_integrated',
                                             'attention_sublayer_parallelism')])
    args = p.parse_args()
    if args.command == 'build':
        build(args.build_dir.resolve())
    elif args.output is None:
        p.error('run requires --output')
    else:
        run(args.build_dir.resolve(), args.output.resolve(), args.studies,
            parallelism_screen=args.parallelism_screen.resolve() if args.parallelism_screen else None,
            tile_screen=args.tile_screen.resolve() if args.tile_screen else None,
            tile_kernel_screen=args.tile_kernel_screen.resolve() if args.tile_kernel_screen else None)


if __name__ == '__main__':
    main()
