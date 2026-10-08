"""Collect the Fast-route token profile and batch-size study; replay every retained full-model archive.

Build/run/capture require a clean local checkout and verified local assets and
measure the current Fast route. The batch-size commands time decode steps of 1
to 64 sequences: --study size (1c) pairs three projection row tiles, --study
projections (1d) screens and confirms exact batched projection arrangements, and
--study reordered (1e) screens arrangements with other summation orders, with an
accuracy census and model-level diagnostics, and --study paged (2d) screens KV
block sizes and orders in decode and prefill. single-sequence compares two
generator builds on one sequence.
Completed decode experiments are replay-only: their archives, parsers and
summaries remain, their collectors do not.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
import random
from pathlib import Path
import statistics as stats
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from .._repository import environment_tool, repository_root
from ..validation.evidence import source_identity, sha, write
from llm_mojo.models.qwen2.assets import verify_prepared
from ..validation.model import environment, generation_events
from llm_mojo.models.qwen2.tokenizer_assets import ensure_prepared
from .environment import stable_environment, conditions_snapshot, require_ac, require_nominal_thermal_state, ensure_record_location
from . import model_contract as contract


def execute(command, log, timeout=600):
    result = subprocess.run(list(map(str, command)), cwd=repository_root(), env=environment(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}); see {log}')
    return result.stdout


def conditions():
    result = conditions_snapshot()
    require_ac(result)
    require_nominal_thermal_state(result)
    if result['power_mode_raw'] != '0':
        raise ValueError('profiling requires verified normal power mode')
    return result


def assets(prepared):
    prepared = Path(prepared).resolve()
    verify_prepared(prepared)
    tables = ensure_prepared(download=False)
    return dict(prepared=str(prepared), tables=str(tables),
                prepared_sha256=sha(prepared/'manifest.json'), tables_sha256=sha(tables))


def build(output, prepared):
    """Fast-route executables: model driver, terminal, and one trace binary per context."""
    ensure_record_location(output)
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    if source['repository']['dirty']:
        raise ValueError('model profiling build requires clean source')
    identity = assets(prepared)
    binaries = {}
    for name, entry in [('model', 'benchmarks/model.mojo'), ('terminal', 'cli/chat_cli.mojo')]:
        command = [environment_tool('mojo'), 'build', '-I', 'src', 'src/llm_mojo/'+entry, '-o', output/name]
        execute(command, output/f'{name}-build.log')
        binaries[name] = dict(sha256=sha(output/name), bytes=(output/name).stat().st_size)
    machine = stable_environment()
    implementation = contract.COMPOSITION_IMPLEMENTATIONS[contract.FAST_DECODE_VARIANT]
    for prefix in contract.PREFIXES:
        name = f'profile-{prefix}'
        command = [environment_tool('mojo'), 'build', '-I', 'src',
                   '-D', f'MODEL_PROFILE_PREFIX={prefix}',
                   '-D', 'MODEL_PREPARED='+identity['prepared'],
                   '-D', 'MODEL_TABLES='+identity['tables'],
                   'src/llm_mojo/benchmarks/model.mojo', '-o', output/name]
        execute(command, output/f'{name}-build.log')
        binary = dict(sha256=sha(output/name), bytes=(output/name).stat().st_size)
        binaries[name] = binary
        provenance = dict(schema_version=1, operation=contract.OPERATION,
                          implementation=implementation,
                          entrypoint=contract.ENTRYPOINTS[implementation],
                          repository=source['repository'], source_sha256=source['sources'],
                          **machine, **contract.specification(prefix,*contract.options(implementation)),
                          profile_warmup_iterations=10, profile_iterations=8,
                          profile_post_idle_milliseconds=250, binary=binary,
                          assets={k:v for k,v in identity.items() if k.endswith('_sha256')})
        contract.configuration(provenance)
        write(output/(name+'.provenance.json'), provenance)
    if source_identity() != source or assets(prepared) != identity:
        raise ValueError('source or assets changed during compilation')
    write(output/'build.json', dict(source=source, assets=identity, environment=machine,
                                    declaration=contract.DECLARATION, binaries=binaries))


def verify_build(directory):
    receipt = json.loads((directory/'build.json').read_text())
    if receipt['source'] != source_identity() or receipt['source']['repository']['dirty']:
        raise ValueError('profile source differs from clean build')
    for name, expected in receipt['binaries'].items():
        if sha(directory/name) != expected['sha256'] or (directory/name).stat().st_size != expected['bytes']:
            raise ValueError('profile executable changed')
    if assets(receipt['assets']['prepared']) != receipt['assets']:
        raise ValueError('prepared assets changed')
    if stable_environment() != receipt['environment']:
        raise ValueError('hardware/software differs from build')
    return receipt


def verify_snapshots(directory, prefix, appended=1):
    records = []
    for name in ['logits'] + [f'{kind}{layer}' for kind in ('k', 'v') for layer in range(24)]:
        paths = [directory/f'{mode}-{name}.bin' for mode in ('before', 'plain', 'observed')]
        arrays = [np.fromfile(p, dtype='<u2') for p in paths]
        size = 151936 if name == 'logits' else 4096*128
        if any(a.size != size for a in arrays):
            raise ValueError('incomplete profiling numerical capture')
        before, plain, observed = arrays
        exact = np.array_equal(plain, observed)
        finite = all(np.isfinite((a.astype(np.uint32) << 16).view(np.float32)).all() for a in arrays)
        prefix_exact = name == 'logits' or np.array_equal(before[:prefix*128], plain[:prefix*128])
        inactive_exact = name == 'logits' or np.array_equal(before[(prefix+appended)*128:], plain[(prefix+appended)*128:])
        if not all((exact, finite, prefix_exact, inactive_exact)):
            raise ValueError('profiling changed numerical/cache invariants')
        records.append(dict(name=name, exact=bool(exact), finite=bool(finite),
                            prefix_exact=bool(prefix_exact), inactive_exact=bool(inactive_exact),
                            hashes={p.name: sha(p) for p in paths}))
    history = list(map(int, (directory/'history.txt').read_text().splitlines()))
    if len(history) != prefix+1 or any(not 0 <= token < 151936 for token in history):
        raise ValueError('invalid frozen history')
    return dict(prefix=prefix, history=history, observations=records)


def parse_samples(stdout, prefix, block, comparison, observed=True):
    if 'device: Apple M4 Pro\napi: metal\n' not in stdout or stdout.count('BENCHMARK_COMPLETE') != 1:
        raise ValueError('missing measured device or completion')
    records = []
    for line in stdout.splitlines():
        if not line.startswith('SAMPLE '):
            continue
        values = list(map(int, line.split()[1:]))
        if len(values) not in (3, 13):
            raise ValueError('invalid host observation record')
        arm, sample, elapsed, *marks = values
        if elapsed <= 0 or bool(marks) != (observed and comparison == 1 and arm == 1):
            raise ValueError('incorrect observation arm')
        if marks and (marks != sorted(marks) or marks[0] < 0 or marks[-1] > elapsed):
            raise ValueError('invalid host timing sequence')
        records.append(dict(prefix=prefix, block=block, comparison=comparison, arm=arm,
                            sample=sample, elapsed_ns=elapsed, marks=marks))
    if Counter((r['arm'], r['sample']) for r in records) != Counter((a, s) for a in range(2) for s in range(10)):
        raise ValueError('incomplete paired timing census')
    return records


def swap_capture_names(extra_norm=False):
    return ([f'hidden_{i}.bin' for i in range(25)]+['final_norm.bin','logits.bin']
            +[f'{stage}_{kind}_{i}.bin' for i in range(24) for stage in ('cache','append') for kind in ('key','value')]
            +([f'{name}_{i}.bin' for i in range(24) for name in ('attention_norm','mlp_norm','attention_residual')] if extra_norm else []))


def swap_lifecycle_names():
    return ([f'step-{i}' for i in (0,1,2,3,4,6,7,8)]
            +['final-'+name for name in ['logits']+[f'{k}{i}' for k in ('k','v') for i in range(24)]])


def validate_swap_checks(checks,extra_norm=False):
    if checks.get('owners_checked') is not True or checks.get('rejection_checked') is not True:
        raise ValueError('missing buffer ownership/lifecycle checks')
    for field,names in [('layers',swap_capture_names(extra_norm)),('lifecycle',swap_lifecycle_names())]:
        records=checks.get(field,[])
        if Counter(r['name'] for r in records)!=Counter(names) or any(
            r['exact'] is not True or r['bytes']<=0 or len(r['sha256'])!=64 for r in records):
            raise ValueError('incomplete or changed buffer-swap numerical evidence')
    states=checks.get('states',[])
    expected=[(0,3,3),(1,1,4),(2,1,5),(3,2,7),(4,1,8),(6,1,1),(7,2,3),(8,1,4)]
    if ([tuple(row[:3]) for row in states]!=expected
        or any(len(row)!=4 or not 0<=row[3]<151936 for row in states)):
        raise ValueError('buffer-swap lifecycle state changed')


def collect(directory, output):
    ensure_record_location(output)
    receipt = verify_build(directory)
    if receipt['declaration'] != contract.DECLARATION:
        raise ValueError('only current Fast-route builds can be collected; completed experiments are replay-only')
    output.mkdir(parents=True, exist_ok=False)
    args = receipt['assets']
    numerical = []
    for prefix in contract.PREFIXES:
        target = output/f'verify-{prefix}'
        target.mkdir()
        stdout = execute([directory/'model', 'verify', args['prepared'], args['tables'], prefix, 0, 0, target],
                         target/'driver.log')
        if 'VERIFY_COMPLETE' not in stdout:
            raise ValueError('native verification incomplete')
        numerical.append(verify_snapshots(target, prefix))
    samples, blocks = [], []
    for block in range(4):
        before = conditions()
        reverse = block in (1, 2)
        prefixes = list(reversed(contract.PREFIXES)) if reverse else contract.PREFIXES
        for prefix in prefixes:
            for comparison in ([1, 0] if reverse else [0, 1]):
                stdout = execute([directory/'model', 'bench', args['prepared'], args['tables'],
                                  prefix, int(reverse), comparison, ''],
                                 output/f'p{prefix}-b{block}-c{comparison}.log')
                samples.extend(parse_samples(stdout, prefix, block, comparison))
        blocks.append(dict(block=block, before=before, after=conditions()))
        print(f'Completed timing block {block+1}/4', flush=True)
    if verify_build(directory) != receipt:
        raise ValueError('build changed during collection')
    write(output/'timings.json', dict(build=receipt, numerical=numerical, blocks=blocks, samples=samples))


def capture(directory, output):
    ensure_record_location(output)
    from .capture_trace import capture_trace
    receipt = verify_build(directory)
    output.mkdir(parents=True, exist_ok=False)
    for repeat in range(2):
        for prefix in (contract.PREFIXES if repeat == 0 else reversed(contract.PREFIXES)):
            target = output/f'p{prefix}-r{repeat}'
            target.mkdir()
            before = conditions()
            capture_trace(profile_binary=directory/f'profile-{prefix}', output_trace=target/'raw.trace',
                          receipt_path=target/'capture.json', time_limit='30s')
            write(target/'conditions.json', dict(before=before, after=conditions()))
            provenance = directory/f'profile-{prefix}.provenance.json'
            (target/'profile.provenance.json').write_bytes(provenance.read_bytes())
            print(f'Captured prefix {prefix}, repeat {repeat+1}/2', flush=True)
    if verify_build(directory) != receipt:
        raise ValueError('build changed during profiling')


def terminal(directory, output):
    ensure_record_location(output)
    """Real streaming conversations; report actual context and stop lengths."""
    receipt = verify_build(directory)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repository_root()/'tests'))
    from chat_terminal import events, validate, run as lifecycle
    identity = receipt['assets']
    before = conditions()
    interactions = lifecycle(directory/'terminal', Path(identity['prepared']), Path(identity['tables']), output/'lifecycle')
    records = []
    passage = ('A train travels sixty kilometers in forty-five minutes. Explain how to calculate its average speed, '
               'keeping track of distance, time, and units. The passengers compare their calculations and check each step. ')
    prompts = ['Explain why keeping units consistent matters when calculating speed. Give three examples.',
               passage*26+'\nExplain the calculation and give three other examples.',
               passage*100+'\nExplain the calculation and give three other examples.']
    # One line per user turn; resetting leaves weights resident but clears context.
    inputs = ('\n/reset\n'.join(p.replace('\n',' ') for p in prompts)+'\n/exit\n').encode()
    for repeat in range(4):
        report = output/f'terminal-{repeat}.tsv'
        base = list(map(str, [directory/'terminal', identity['prepared'], identity['tables'], 128, 256, '']))
        outputs = {}
        for observed in ([True,False] if repeat in (1,2) else [False,True]):
            command = base+[str(report) if observed else '']
            result = subprocess.run(command, cwd=repository_root(), env=environment(), input=inputs,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
            (output/f'terminal-{repeat}-{int(observed)}.txt').write_bytes(result.stdout)
            if result.returncode:
                raise ValueError('terminal execution failed')
            outputs[observed] = result.stdout
        if outputs[True] != outputs[False]:
            raise ValueError('reporting changed terminal output')
        turns = validate(events(report), 128)
        if len(turns) != 3:
            raise ValueError('terminal workload rejected or missing')
        records.append(dict(repeat=repeat, turns=turns, reporting_output_exact=True,
                            events=events(report), output_sha256=hashlib.sha256(outputs[True]).hexdigest()))
    write(output/'terminal.json', dict(build=receipt, prompts=prompts, records=records, lifecycle=interactions,
                                      conditions=dict(before=before, after=conditions())))
    verify_build(directory)


def export_trace(target):
    """Use xctrace's observed schema names; raw exports remain outside Git."""
    import xml.etree.ElementTree as ET
    trace = target/'raw.trace'
    execute(['xcrun', 'xctrace', 'export', '--input', trace, '--toc', '--output', target/'toc.xml'], target/'export-toc.log')
    root = ET.parse(target/'toc.xml').getroot()
    schemas = {node.attrib['schema'] for node in root.iter('table') if 'schema' in node.attrib}
    for key, needle in [('submissions', 'command-buffer-submission'), ('gpu-intervals', 'gpu-interval')]:
        matches = [s for s in schemas if needle in s]
        if len(matches) != 1:
            raise ValueError(f'ambiguous {needle} schema: {sorted(schemas)}')
        xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{matches[0]}"]'
        execute(['xcrun', 'xctrace', 'export', '--input', trace, '--xpath', xpath,
                 '--output', target/(key+'.xml')], target/f'export-{key}.log')


def curate(target, prefix, repeat, fused=None, combined=None):
    provenance = json.loads((target/'profile.provenance.json').read_text())
    if fused is None:
        fused = provenance['implementation'] != 'qwen_model_fast'
    if combined is None:
        combined = provenance['implementation'] == 'qwen_model_combined'
    _,_,_,copy_free,residual_norm = contract.options(provenance['implementation'])
    selection = contract.SELECTIONS.get(provenance['implementation'],0)
    from .analyze_trace import analyze, read_table, integer, coalesce_compute_commands, segment_compute_commands
    arguments = SimpleNamespace(capture_receipt=target/'capture.json', submissions_xml=target/'submissions.xml',
                                gpu_intervals_xml=target/'gpu-intervals.xml', toc_xml=target/'toc.xml',
                                performance_state_xml=None, spill_xml=None, counter_info_xml=None, counter_values_xml=None)
    report = analyze(arguments)
    write(target/'summary.json', report)
    submissions = [r for r in read_table(arguments.submissions_xml) if integer(r, 'num-encoders') > 0]
    # Match the analyzer's target process, excluding unrelated trace activity.
    capture_id = json.loads((target/'capture.json').read_text())['capture']['capture_id']
    submissions = [r for r in submissions if capture_id in r['process'][1]]
    ids = {integer(r, 'cmdbuffer-id') for r in submissions}
    intervals = [r for r in read_table(arguments.gpu_intervals_xml)
                 if integer(r, 'cmdbuffer-id') in ids and capture_id in r['event-label'][1]
                 and r['channel-name'][0] == 'Compute' and any(
                     kind in r['event-label'][1] for kind in (':Compute Command',':Blit Command'))]
    fragments = defaultdict(list)
    for row in intervals:
        key = tuple(integer(row,k) for k in ('cmdbuffer-id','encoder-id'))
        fragments[key].append([integer(row,'start'),integer(row,'duration')])
    stages = contract.command_stages(fused, combined and fused, selection, copy_free, residual_norm)
    count = len(stages)
    intervals, joined = coalesce_compute_commands(intervals, submissions, 18*count, join_resubmissions=True)
    if joined != report['validated_sequence']['interval_coalescing']:
        raise ValueError('curation differs from validated dispatch join')
    *_, measured = segment_compute_commands(intervals, 10, 8, count, False)
    contract.validate_command_sequence(measured,fused,combined and fused,selection,copy_free,residual_norm)
    by_command = {integer(r,'cmdbuffer-id'):r for r in submissions}
    rows = []
    for index, row in enumerate(measured):
        layer, stage, kind = stages[index % count]
        submitted = by_command[integer(row,'cmdbuffer-id')]
        rows.append(dict(prefix=prefix, repeat=repeat, iteration=index//count, dispatch=index%count,
                         layer=layer, stage=stage, kind=kind, start_ns=integer(row, 'start'), end_ns=integer(row, 'end'),
                         duration_ns=integer(row, 'duration'), segments=integer(row, 'active-segments'),
                         active_intervals=sorted(fragments[tuple(integer(row,k) for k in ('cmdbuffer-id','encoder-id'))]),
                         submission_start_ns=integer(submitted,'start'),
                         submission_duration_ns=integer(submitted,'duration'),
                         encoder_duration_ns=integer(submitted,'encoder-time')))
    return dict(prefix=prefix, repeat=repeat, analysis=report,
                provenance=json.loads((target/'profile.provenance.json').read_text()),
                conditions=json.loads((target/'conditions.json').read_text()), samples=rows)


def summarize(samples, observed=True):
    expected = Counter((p,b,c,a,s) for p in contract.PREFIXES for b in range(4)
                       for c in range(2) for a in range(2) for s in range(10))
    if Counter(tuple(r[k] for k in ('prefix','block','comparison','arm','sample')) for r in samples) != expected:
        raise ValueError('incomplete study timing census')
    for row in samples:
        marks = row['marks']
        if (row['elapsed_ns'] <= 0 or bool(marks) != (observed and row['comparison']==1 and row['arm']==1)
                or (marks and (len(marks)!=10 or marks != sorted(marks) or marks[0]<0 or marks[-1]>row['elapsed_ns']))):
            raise ValueError('invalid retained host observation')
    result = []
    for prefix in contract.PREFIXES:
        subset = [r for r in samples if r['prefix'] == prefix]
        medians = {(b,c,a): stats.median(r['elapsed_ns'] for r in subset
                                       if (r['block'],r['comparison'],r['arm']) == (b,c,a))
                   for b in range(4) for c in range(2) for a in range(2)}
        calibration = [medians[b,0,1]/medians[b,0,0] for b in range(4)]
        ratios = [medians[b,1,1]/medians[b,1,0] for b in range(4)]
        baseline = stats.median(medians[b,1,0] for b in range(4))/1e6
        phases = defaultdict(list)
        labels = ['preflight','token staging','embedding enqueue','decoder stack enqueue',
                  'final norm/head enqueue','forward return','map/wait','host winner processing','unmap']
        for block in range(4):
            records = [r for r in subset if r['block'] == block and r['marks']]
            for i, label in enumerate(labels if records else []):
                phases[label].append(stats.median(r['marks'][i+1]-r['marks'][i] for r in records)/1e6)
        result.append(dict(prefix=prefix, baseline_ms=baseline, equivalent_steps_per_second=1000/baseline,
                           baseline_block_ms=[medians[b,1,0]/1e6 for b in range(4)],
                           observation_ratios=ratios, calibration_ratios=calibration,
                           noise_floor=max(.05, max(abs(r-1) for r in calibration)),
                           host_phase_ms={k:stats.median(v) for k,v in phases.items()}))
    return result


def archive(timings, traces, output, terminal_path):
    timing = json.loads((timings/'timings.json').read_text())
    summarize(timing['samples'])
    captures = []
    for prefix in contract.PREFIXES:
        for repeat in range(2):
            target = traces/f'p{prefix}-r{repeat}'
            if not (target/'submissions.xml').exists():
                export_trace(target)
            captures.append(curate(target, prefix, repeat))
    terminal_record = json.loads((terminal_path/'terminal.json').read_text())
    record = dict(kind='qwen-token-profile-v1', timing=timing, captures=captures, terminal=terminal_record,
                  analysis_source_sha256={str(p.relative_to(repository_root())):sha(p) for p in
                    [Path(__file__).resolve(), Path(__file__).with_name('analyze_trace.py').resolve(),
                     Path(__file__).with_name('model_contract.py').resolve()]},
                  rejected_captures=json.loads((traces/'rejections.json').read_text()) if (traces/'rejections.json').exists() else [])
    # Asset identities remain available without publishing machine-local paths.
    def scrub(value):
        if isinstance(value, dict):
            return {k: ('<verified-local-asset>' if k in ('prepared','tables') else scrub(v)) for k,v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value
    record = scrub(record)
    raw = json.dumps(record, separators=(',', ':'), allow_nan=False).encode()
    packed = gzip.compress(raw, mtime=0)
    output.mkdir(parents=True, exist_ok=True)
    (output/'token-profile.json.gz').write_bytes(packed)
    write(output/'token-profile.json', dict(kind=record['kind'], sha256=hashlib.sha256(packed).hexdigest(),
                                           uncompressed_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(packed)))
    replay(output)


def replay(directory):
    manifest = json.loads((directory/'token-profile.json').read_text())
    packed = (directory/'token-profile.json.gz').read_bytes()
    raw = gzip.decompress(packed)
    if sha(directory/'token-profile.json.gz') != manifest['sha256'] or hashlib.sha256(raw).hexdigest() != manifest['uncompressed_sha256']:
        raise ValueError('profile archive hash mismatch')
    record = json.loads(raw)
    summary = summarize(record['timing']['samples'])
    numerical = record['timing']['numerical']
    if Counter(n['prefix'] for n in numerical) != Counter(contract.PREFIXES):
        raise ValueError('incomplete numerical context census')
    names = ['logits']+[f'{k}{i}' for k in ('k','v') for i in range(24)]
    for check in numerical:
        if (len(check['history']) != check['prefix']+1
                or Counter(r['name'] for r in check['observations']) != Counter(names)
                or any(r[k] is not True for r in check['observations'] for k in ('exact','finite','prefix_exact','inactive_exact'))):
            raise ValueError('invalid retained numerical invariants')
    build_record = record['timing']['build']
    if build_record['declaration'] != contract.DECLARATION:
        raise ValueError('profiling workload declaration changed')
    if record['terminal']['build'] != build_record:
        raise ValueError('terminal and timing executable identity differ')
    terminal_records = record['terminal']['records']
    if (Counter(r['repeat'] for r in terminal_records) != Counter(range(4))
            or any(r['reporting_output_exact'] is not True or len(r['turns'])!=3 for r in terminal_records)):
        raise ValueError('incomplete terminal comparison census')
    if Counter((c['prefix'],c['repeat']) for c in record['captures']) != Counter((p,r) for p in contract.PREFIXES for r in range(2)):
        raise ValueError('incomplete trace capture census')
    stage_totals = []
    for capture in record['captures']:
        provenance = capture['provenance']
        contract.configuration(provenance)
        if (provenance['repository'] != build_record['source']['repository']
                or provenance['source_sha256'] != build_record['source']['sources']
                or provenance['assets'] != {k:v for k,v in build_record['assets'].items() if k.endswith('_sha256')}
                or provenance['binary'] != build_record['binaries'][f"profile-{capture['prefix']}"]):
            raise ValueError('trace differs from frozen model build')
        canonical = (json.dumps(provenance, indent=2, allow_nan=False)+'\n').encode()
        if hashlib.sha256(canonical).hexdigest() != capture['analysis']['capture_identity']['provenance']['sha256']:
            raise ValueError('retained provenance differs from captured build receipt')
        rows = capture['samples']
        stages = contract.command_stages(*contract.options(provenance['implementation']))
        if Counter((r['iteration'],r['dispatch']) for r in rows) != Counter((i,d) for i in range(8) for d in range(len(stages))):
            raise ValueError('incomplete captured dispatch census')
        for row in rows:
            if (row['layer'],row['stage'],row['kind']) != stages[row['dispatch']] or row['duration_ns'] <= 0 or row['end_ns'] < row['start_ns']+row['duration_ns']:
                raise ValueError('invalid dispatch timing or stage')
            segments = row['active_intervals']
            if (len(segments)!=row['segments'] or sum(d for _,d in segments)!=row['duration_ns']
                    or segments[0][0]!=row['start_ns'] or sum(segments[-1])!=row['end_ns']
                    or any(d<=0 for _,d in segments) or any(a+d>b for (a,d),(b,_) in zip(segments,segments[1:]))):
                raise ValueError('invalid active command fragments')
        by_stage = defaultdict(list)
        for iteration in range(8):
            step = [r for r in rows if r['iteration']==iteration]
            for stage in sorted({r['stage'] for r in step}):
                by_stage[stage].append(sum(r['duration_ns'] for r in step if r['stage']==stage)/1e6)
            by_stage['GPU active total'].append(sum(r['duration_ns'] for r in step)/1e6)
            by_stage['GPU enclosing span'].append((max(r['end_ns'] for r in step)-min(r['start_ns'] for r in step))/1e6)
            compute = [r for r in step if r['kind']=='compute']
            by_stage['GPU compute enclosing span'].append((max(r['end_ns'] for r in compute)-min(r['start_ns'] for r in compute))/1e6)
            by_stage['Metal submission intervals'].append(sum(r['submission_duration_ns'] for r in step)/1e6)
            by_stage['Metal encoder intervals'].append(sum(r['encoder_duration_ns'] for r in step)/1e6)
        for stage, values in by_stage.items():
            stage_totals.append(dict(prefix=capture['prefix'],repeat=capture['repeat'],stage=stage,median_ms=stats.median(values)))
    write(directory/'token-profile-summary.json', dict(timing=summary, gpu=stage_totals))
    print(json.dumps(summary, indent=2))
    return record


def plot(directory):
    """Regenerate publication figures exclusively from the checked archive."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    record = replay(directory)
    summary = json.loads((directory/'token-profile-summary.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
    fig, axes = plt.subplots(1, 2, figsize=(11,4.6), constrained_layout=True)
    for ax in axes:
        ax.set_xticks(range(3), [str(p) for p in contract.PREFIXES])
        ax.set_xlabel('Previously cached tokens')
        ax.set_ylabel('Milliseconds per decode step')
    baseline = [r['baseline_ms'] for r in summary['timing']]
    ranges = [r['baseline_block_ms'] for r in summary['timing']]
    errors = [[m-min(v) for m,v in zip(baseline,ranges)], [max(v)-m for m,v in zip(baseline,ranges)]]
    axes[0].bar(range(3), baseline, color='#244b69', width=.55, yerr=errors, capsize=4)
    for i, value in enumerate(baseline):
        axes[0].text(i,max(ranges[i])+.15,f'{value:.2f}',ha='center')
    axes[0].set_ylim(0,max(max(r) for r in ranges)*1.18)
    axes[0].set_title('Ordinary execution: complete token step\nMedian of four block medians; whiskers show range')
    groups = [('Decoder projections/MLP',{'packed QKV projection','output projection','gate projection','up projection','down projection'},'#244b69'),
              ('Attention',{'FP32 GQA'},'#27a89b'),
              ('Vocabulary projection',{'vocabulary projection'},'#e8a044'),
              ('Inter-layer copies',{'inter-layer copy'},'#b86f85')]
    used = set().union(*(s for _,s,_ in groups))
    commands=[r for capture in record['captures'] for r in capture['samples']]
    groups.append(('Other GPU operations',{r['stage'] for r in commands if r['kind']=='compute'}-used,'#abb9c7'))
    groups.append(('Buffer transfers',{r['stage'] for r in commands if r['kind']=='blit'},'#678d75'))
    bottoms = [0.]*3
    for name, stages, color in groups:
        values = []
        for prefix in contract.PREFIXES:
            totals = [sum(r['median_ms'] for r in summary['gpu'] if r['prefix']==prefix and r['repeat']==repeat and r['stage'] in stages)
                      for repeat in range(2)]
            values.append(stats.mean(totals))
        axes[1].bar(range(3),values,bottom=bottoms,color=color,width=.55,label=name)
        bottoms = [a+b for a,b in zip(bottoms,values)]
    axes[1].set_title('Separate traces: active GPU stage time')
    axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,1))
    axes[1].set_ylim(0,max(bottoms)*1.7)
    fig.suptitle('Qwen2.5-0.5B · BF16 · M4 Pro / Metal\nTrace time explains execution; it is not an additive part of the untraced measurement.',fontsize=12)
    fig.savefig(directory/'token-profile-context.png',dpi=170)
    plt.close(fig)
    capture = next(c for c in record['captures'] if c['prefix']==1024 and c['repeat']==0)
    rows = [r for r in capture['samples'] if r['iteration']==3]
    origin = min(min(r['start_ns'],r['submission_start_ns']) for r in rows)
    fig, ax = plt.subplots(figsize=(11,4.5),constrained_layout=True)
    for index,(name,stages,color) in enumerate(groups):
        spans = [((start-origin)/1e6,duration/1e6) for r in rows if r['stage'] in stages
                 for start,duration in r['active_intervals']]
        ax.broken_barh(spans,(index-.3,.6),facecolors=color)
    spans = [((r['submission_start_ns']-origin)/1e6,r['submission_duration_ns']/1e6) for r in rows]
    ax.broken_barh(spans,(len(groups)-.3,.6),facecolors='#6b5ca5')
    ax.set_yticks(range(len(groups)+1),[name for name,_,_ in groups]+['Metal submissions (host)'])
    ax.set_xlabel('Milliseconds on the shared trace clock, relative to this token step')
    ax.set_title('One traced token at 1,024 cached tokens\nHost submission overlaps active GPU work; preemption gaps remain visible.')
    fig.savefig(directory/'token-profile-timeline.png',dpi=170)
    plt.close(fig)


def fusion_summary(samples):
    result = summarize(samples,observed=False)
    for r in result:
        ratios = r.pop('observation_ratios')
        r.pop('host_phase_ms')
        r['candidate_ratios'] = ratios
        r['median_reduction'] = 1-stats.median(ratios)
        r['promote'] = all(x<1 for x in ratios) and r['median_reduction']>r['noise_floor']
        r['candidate_block_ms'] = [a*b for a,b in zip(r['baseline_block_ms'],ratios)]
        r['candidate_ms'] = stats.median(r['candidate_block_ms'])
    return result


def combined_ablation(samples):
    expected = Counter((p,b,c,a,s) for p in contract.PREFIXES for b in range(4)
                       for c in range(3) for a in range(2) for s in range(10))
    if Counter(tuple(r[k] for k in ('prefix','block','comparison','arm','sample')) for r in samples)!=expected:
        raise ValueError('incomplete combined fusion timing census')
    if any(r['marks'] or r['elapsed_ns']<=0 for r in samples):
        raise ValueError('invalid combined fusion timing sample')
    result=[]
    for prefix in contract.PREFIXES:
        pairs=[]
        for block in range(4):
            pairs.append([stats.median(r['elapsed_ns'] for r in samples if
                (r['prefix'],r['block'],r['comparison'],r['arm'])==(prefix,block,2,arm))/1e6 for arm in range(2)])
        ratios=[b/a for a,b in pairs]
        result.append(dict(prefix=prefix,qkv_block_ms=[a for a,b in pairs],combined_block_ms=[b for a,b in pairs],
                           ratios=ratios,median_reduction=1-stats.median(ratios),all_faster=all(r<1 for r in ratios)))
    return result


def fusion_plot(directory, combined=False, copy_free=False):
    stem = "buffer-swap" if copy_free else ("combined-fusion" if combined else "qkv-fusion")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fusion_replay(directory,combined,copy_free)
    rows=json.loads((directory/(stem+'-summary.json')).read_text())['timing']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for i,r in enumerate(rows):
        for offset,key,color,label in [(-.18,'baseline','#64798c','Control'),(.18,'candidate','#188f82',('Owner swap' if copy_free else 'Fused'))]:
            value=r[key+'_ms'];values=r[key+'_block_ms']
            axes[0].bar(i+offset,value,.34,color=color,label=label if i==0 else None,
                        yerr=[[value-min(values)],[max(values)-value]],capsize=3)
            axes[0].text(i+offset,max(values)+.2,f'{value:.2f}',ha='center',fontsize=9)
        axes[1].scatter([i-.12,i-.04,i+.04,i+.12],r['candidate_ratios'],color='#188f82',s=30)
        axes[1].plot([i-.3,i+.3],[1-r['noise_floor']]*2,color='#bd6a44',linewidth=2,
                     label='Required median-ratio threshold' if i==0 else None)
    axes[0].set_ylim(0,max(max(r['baseline_block_ms']+r['candidate_block_ms']) for r in rows)*1.18)
    axes[0].set_ylabel('Milliseconds per complete token step')
    axes[0].set_title('Untraced complete-step latency; block range')
    axes[0].legend(loc='upper center',ncol=2)
    axes[1].axhline(1,color='#64798c',linestyle='--')
    axes[1].set_title('Four paired block ratios per context')
    axes[1].set_ylabel(('Owner swap' if copy_free else 'Fused')+' / control latency; lower is better')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(3),[str(r['prefix']) for r in rows])
        ax.set_xlabel('Previously cached tokens')
    fig.suptitle('Qwen2.5-0.5B · BF16 · batch one · M4 Pro / Metal',fontsize=13)
    fig.savefig(directory/(stem+'.png'),dpi=170)
    plt.close(fig)


def fusion_replay(directory, combined=False, copy_free=False):
    stem = "buffer-swap" if copy_free else ("combined-fusion" if combined else "qkv-fusion")
    manifest = json.loads((directory/(stem+'.json')).read_text())
    packed = (directory/(stem+'.json.gz')).read_bytes()
    raw = gzip.decompress(packed)
    if hashlib.sha256(packed).hexdigest()!=manifest['sha256'] or hashlib.sha256(raw).hexdigest()!=manifest['uncompressed_sha256']:
        raise ValueError('fusion evidence hash mismatch')
    record = json.loads(raw)
    if record['kind']!=('qwen-buffer-swap-v1' if copy_free else ('qwen-combined-fusion-v1' if combined else 'qwen-qkv-fusion-v1')): raise ValueError('unexpected fusion evidence kind')
    timing = record['timing']
    build_record = timing['build']
    if build_record['declaration']!=(contract.COPY_FREE_DECLARATION if copy_free else (contract.COMBINED_DECLARATION if combined else contract.FUSION_DECLARATION)) or record['terminal']['build']!=build_record:
        raise ValueError('fusion build declaration changed')
    summary = fusion_summary([r for r in timing['samples'] if not combined or r['comparison']!=2])
    ablation = combined_ablation(timing['samples']) if combined else []
    if Counter(n['prefix'] for n in timing['numerical'])!=Counter(contract.PREFIXES):
        raise ValueError('incomplete fusion numerical coverage')
    names = ['logits']+[f'{k}{i}' for k in ('k','v') for i in range(24)]
    for n in timing['numerical']:
        if (len(n['history'])!=n['prefix']+1 or Counter(r['name'] for r in n['observations'])!=Counter(names)
                or any(r[k] is not True for r in n['observations'] for k in ('exact','finite','prefix_exact','inactive_exact'))):
            raise ValueError('fusion numerical invariant failed')
    if copy_free:
        validate_swap_checks(next(n for n in timing['numerical'] if n['prefix']==64).get('swap_checks',{}))
        if [b['block'] for b in timing['blocks']] != list(range(4)):
            raise ValueError('incomplete buffer-swap timing conditions')
        for block in [*timing['blocks'],*record['terminal']['blocks'],*[c['conditions'] for c in record['captures']]]:
            for side in ('before','after'):
                require_ac(block[side])
                require_nominal_thermal_state(block[side])
                if block[side]['power_mode_raw'] != '0':
                    raise ValueError('buffer-swap power mode changed')
    if len(record['captures'])!=2: raise ValueError('incomplete fusion capture census')
    for fused,capture in zip((False,True),record['captures']):
        provenance = capture['provenance']
        contract.configuration(provenance)
        name = 'profile-1024'+('-fused' if fused else '')
        if (provenance['repository']!=build_record['source']['repository']
                or provenance['source_sha256']!=build_record['source']['sources']
                or provenance['binary']!=build_record['binaries'][name]
                or provenance['assets']!={k:v for k,v in build_record['assets'].items() if k.endswith('_sha256')}
                or provenance['implementation']!=(('qwen_model_buffer_swap' if fused else 'qwen_model_combined') if copy_free else ('qwen_model_combined' if combined and fused else ('qwen_model_fused' if fused else 'qwen_model_fast')))):
            raise ValueError('fusion trace build identity mismatch')
        canonical = (json.dumps(provenance,indent=2,allow_nan=False)+'\n').encode()
        if hashlib.sha256(canonical).hexdigest()!=capture['analysis']['capture_identity']['provenance']['sha256']:
            raise ValueError('fusion trace capture provenance mismatch')
        stages = contract.command_stages(fused, copy_free or combined and fused, copy_free=copy_free and fused)
        if Counter((r['iteration'],r['dispatch']) for r in capture['samples'])!=Counter((i,d) for i in range(8) for d in range(len(stages))):
            raise ValueError('incomplete fusion dispatch census')
        for row in capture['samples']:
            if (row['layer'],row['stage'],row['kind'])!=stages[row['dispatch']]:
                raise ValueError('fusion stage attribution changed')
            parts = row['active_intervals']
            if (not parts or len(parts)!=row['segments'] or sum(d for _,d in parts)!=row['duration_ns']
                    or parts[0][0]!=row['start_ns'] or sum(parts[-1])!=row['end_ns']
                    or any(d<=0 for _,d in parts) or any(a+d>b for (a,d),(b,_) in zip(parts,parts[1:]))):
                raise ValueError('invalid fusion trace fragments')
    blocks = record['terminal']['blocks']
    if Counter(b['block'] for b in blocks)!=Counter(range(4)):
        raise ValueError('incomplete fusion terminal blocks')
    sys.path.insert(0,str(repository_root()/'tests'))
    from chat_terminal import validate
    for b in blocks:
        if b['output_exact'] is not True or Counter(a['fused'] for a in b['arms'])!=Counter([False,True]):
            raise ValueError('fusion terminal arm mismatch')
        for a in b['arms']:
            if validate(a['events'],128)!=a['turns'] or len(a['turns'])!=3:
                raise ValueError('fusion terminal event evidence changed')
        if (b['arms'][0]['output_sha256']!=b['arms'][1]['output_sha256'] or any(
                a['generated']!=c['generated'] or a['history']!=c['history'] for a,c in zip(b['arms'][0]['turns'],b['arms'][1]['turns']))):
            raise ValueError('fusion terminal outputs changed')
    write(directory/(stem+'-summary.json'),dict(timing=summary,promote=all(r['promote'] for r in summary) and all(r['all_faster'] for r in ablation), **({'ablation':ablation} if combined else {})))
    print(json.dumps(summary,indent=2))
    return record


def selection_summary(samples):
    expected = Counter((p,b,c,a,s) for p in contract.PREFIXES for b in range(4)
                       for c in range(4) for a in range(2) for s in range(10))
    if Counter(tuple(r[k] for k in ('prefix','block','comparison','arm','sample')) for r in samples)!=expected:
        raise ValueError('incomplete selection timing census')
    if any(type(r['elapsed_ns']) is not int or r['elapsed_ns']<=0 or r['marks'] for r in samples):
        raise ValueError('invalid selection timing record')
    result = []
    for prefix in contract.PREFIXES:
        rows = [r for r in samples if r['prefix']==prefix]
        median = {(b,c,a):stats.median(r['elapsed_ns'] for r in rows if (r['block'],r['comparison'],r['arm'])==(b,c,a))
                  for b in range(4) for c in range(4) for a in range(2)}
        calibration = [median[b,0,1]/median[b,0,0] for b in range(4)]
        noise = max(.05,max(abs(r-1) for r in calibration))
        comparisons = []
        for c in range(1,4):
            ratios = [median[b,c,1]/median[b,c,0] for b in range(4)]
            reduction = 1-stats.median(ratios)
            comparisons.append(dict(comparison=c,ratios=ratios,median_reduction=reduction,
                control_block_ms=[median[b,c,0]/1e6 for b in range(4)],candidate_block_ms=[median[b,c,1]/1e6 for b in range(4)],
                qualifies=all(r<1 for r in ratios) and reduction>noise))
        result.append(dict(prefix=prefix,calibration_ratios=calibration,noise_floor=noise,comparisons=comparisons))
    qualifies = [all(r['comparisons'][c]['qualifies'] for r in result) for c in range(3)]
    choice = 1 if qualifies[0] else 0
    if qualifies[1] and (not qualifies[0] or qualifies[2]):
        choice = 2
    return dict(contexts=result,selected=choice,policy=['combined','gpu-argmax','fused-head'][choice])


def composition_summary(samples):
    expected=Counter((p,b,c,a,s) for p in contract.PREFIXES for b in range(4)
                     for c in range(6) for a in range(2) for s in range(10))
    if Counter(tuple(r[k] for k in ('prefix','block','comparison','arm','sample')) for r in samples)!=expected:
        raise ValueError('incomplete residual-norm comparison census')
    if any(type(r['elapsed_ns']) is not int or r['elapsed_ns']<=0 or r['marks'] for r in samples):
        raise ValueError('invalid residual-norm timing sample')
    contexts=[]
    for prefix in contract.PREFIXES:
        rows=[r for r in samples if r['prefix']==prefix]
        medians={(b,c,a):stats.median(r['elapsed_ns'] for r in rows if (r['block'],r['comparison'],r['arm'])==(b,c,a))
                 for b in range(4) for c in range(6) for a in range(2)}
        calibration=[medians[b,0,1]/medians[b,0,0] for b in range(4)]
        noise=max(.05,max(abs(x-1) for x in calibration))
        comparisons=[]
        for c in range(1,6):
            ratios=[medians[b,c,1]/medians[b,c,0] for b in range(4)]
            reduction=1-stats.median(ratios)
            comparisons.append(dict(comparison=c,arms=list(contract.COMPOSITION_PAIRS[c]),ratios=ratios,
                median_reduction=reduction,control_block_ms=[medians[b,c,0]/1e6 for b in range(4)],
                candidate_block_ms=[medians[b,c,1]/1e6 for b in range(4)],all_faster=all(x<1 for x in ratios),
                qualifies=all(x<1 for x in ratios) and reduction>noise))
        contexts.append(dict(prefix=prefix,calibration_ratios=calibration,noise_floor=noise,comparisons=comparisons))
    qualifiers=[v for v in range(1,4) if all(c['comparisons'][v-1]['qualifies'] for c in contexts)]
    selected=0
    if 3 in qualifiers and all(all(c['comparisons'][4 if v==1 else 3]['all_faster'] for c in contexts) for v in qualifiers if v!=3):
        selected=3
    elif len(qualifiers)==1:
        selected=qualifiers[0]
    return dict(contexts=contexts,qualifiers=qualifiers,selected=selected,policy=contract.COMPOSITION_ARMS[selected])


def projection_summary(samples):
    expected=Counter((p,b,c,a,i) for p in contract.PREFIXES for b in range(4)
                     for c in range(6) for a in range(2) for i in range(10))
    if Counter(tuple(r[k] for k in ('prefix','block','comparison','arm','sample')) for r in samples)!=expected:
        raise ValueError('incomplete projection timing census')
    if any(type(r['elapsed_ns']) is not int or r['elapsed_ns']<=0 or r['marks'] for r in samples):
        raise ValueError('invalid projection timing sample')
    contexts=[]
    for prefix in contract.PREFIXES:
        rows=[r for r in samples if r['prefix']==prefix]
        med={(b,c,a):stats.median(r['elapsed_ns'] for r in rows if (r['block'],r['comparison'],r['arm'])==(b,c,a))
             for b in range(4) for c in range(6) for a in range(2)}
        calibration=[med[b,0,1]/med[b,0,0] for b in range(4)]
        noise=max(.05,max(abs(x-1) for x in calibration));comparisons=[]
        for v in range(1,6):
            ratios=[med[b,v,1]/med[b,v,0] for b in range(4)]
            reduction=1-stats.median(ratios)
            comparisons.append(dict(variant=v,label=contract.PROJECTION_LABELS[v],ratios=ratios,
                median_reduction=reduction,control_block_ms=[med[b,v,0]/1e6 for b in range(4)],
                candidate_block_ms=[med[b,v,1]/1e6 for b in range(4)],qualifies=all(x<1 for x in ratios) and reduction>noise))
        contexts.append(dict(prefix=prefix,noise_floor=noise,calibration_ratios=calibration,comparisons=comparisons))
    qualifiers=[v for v in range(1,6) if all(c['comparisons'][v-1]['qualifies'] for c in contexts)]
    def rank(v):
        ratios=[1-c['comparisons'][v-1]['median_reduction'] for c in contexts]
        return max(ratios),stats.mean(ratios),v
    selected=min(qualifiers,key=rank) if qualifiers else 0
    return dict(contexts=contexts,qualifiers=qualifiers,screen_selected=selected,promote=False,
                confirmation_required=bool(selected),policy=contract.PROJECTION_ARMS[selected])


def projection_plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=selection_replay(directory,projection=True)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    colors=['#227eaa','#a98230','#8068a3','#188f82','#bc6653']
    for i,c in enumerate(summary['contexts']):
        for v,r in enumerate(c['comparisons']):
            x=i+(v-2)*.13
            ax.scatter([x]*4,r['ratios'],color=colors[v],s=26,label=r['label'] if i==0 else None)
            ax.plot([x-.045,x+.045],[stats.median(r['ratios'])]*2,color=colors[v])
        ax.plot([i-.38,i+.38],[1-c['noise_floor']]*2,color='black',linestyle=':')
    ax.axhline(1,color='gray',linestyle='--');ax.set_xticks(range(3),contract.PREFIXES)
    ax.set_xlabel('Previously cached tokens');ax.set_ylabel('Paired complete-token latency ratio; lower is better')
    ax.set_title('Projection arrangements over all-three Fast · M4 Pro / Metal · BF16')
    ax.legend(ncol=3);fig.savefig(directory/'projection-arrangements.png',dpi=170);plt.close(fig)


def selection_plot(directory, composition=False):
    stem="residual-norm" if composition else "token-selection"
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=selection_replay(directory,composition)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for index,context in enumerate(summary['contexts']):
        plotted=([(0,-.22,'#277eaa','Norm / Fast'),(1,0,'#a98230','Swap + argmax / Fast'),(2,.22,'#188f82','All three / Fast')] if composition else [(0,-.13,'#277eaa','GPU argmax / CPU'),(1,.13,'#188f82','Fused head / CPU')])
        for comparison,offset,color,label in plotted:
            ratios=context['comparisons'][comparison]['ratios']
            axes[0].scatter([index+offset]*4,ratios,color=color,s=32,label=label if index==0 else None)
            axes[0].plot([index+offset-.08,index+offset+.08],[stats.median(ratios)]*2,color=color,linewidth=2)
        axes[0].plot([index-.35,index+.35],[1-context['noise_floor']]*2,color='#bd6a44',linewidth=1.5,
                     label='Required median threshold' if index==0 else None)
        direct_pairs=[(3,-.12,'#6d63a6','All three / swap + argmax'),(4,.12,'#277eaa','All three / norm')] if composition else [(2,0,'#6d63a6','Fused head / argmax')]
        for comparison,offset,color,label in direct_pairs:
            direct=context['comparisons'][comparison]['ratios']
            axes[1].scatter([index+offset]*4,direct,color=color,s=32,label=label if index==0 else None)
            axes[1].plot([index+offset-.08,index+offset+.08],[stats.median(direct)]*2,color=color,linewidth=2)
        if not composition:
            axes[1].plot([index-.35,index+.35],[1-context['noise_floor']]*2,color='#bd6a44',linewidth=1.5)
    for ax in axes:
        ax.axhline(1,color='#64798c',linestyle='--')
        ax.set_xticks(range(3),[str(r['prefix']) for r in summary['contexts']])
        ax.set_xlabel('Previously cached tokens')
        ax.set_ylabel('Paired complete-token latency ratio; lower is better')
    axes[0].set_title('Independent and combined changes versus Fast' if composition else 'Do GPU selectors improve the current Fast path?')
    axes[0].legend(fontsize=8)
    axes[1].set_title('Does combining all three add value?' if composition else 'Does fusing the head improve GPU argmax?')
    if composition: axes[1].legend(fontsize=8)
    fig.suptitle('Qwen2.5-0.5B · BF16 · M4 Pro / Metal · four paired blocks',fontsize=13)
    fig.savefig(directory/(stem+'.png'),dpi=170)
    plt.close(fig)


def selection_replay(directory, composition=False, projection=False):
    stem='projection-arrangements' if projection else 'residual-norm' if composition else 'token-selection'
    variant_count=6 if projection else 4 if composition else 3
    packed = (directory/(stem+'.json.gz')).read_bytes()
    raw = gzip.decompress(packed)
    manifest = json.loads((directory/(stem+'.json')).read_text())
    if manifest != dict(sha256=hashlib.sha256(packed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()):
        raise ValueError('selection evidence hash mismatch')
    record = json.loads(raw)
    timing = record['timing']
    build_record = timing['build']
    if record['kind']!=('qwen-projection-arrangements-v1' if projection else 'qwen-residual-norm-v1' if composition else 'qwen-token-selection-v1') or build_record['declaration']!=(contract.PROJECTION_DECLARATION if projection else contract.COMPOSITION_DECLARATION if composition else contract.SELECTION_DECLARATION) or record['terminal']['build']!=build_record:
        raise ValueError('selection build identity mismatch')
    summary = projection_summary(timing['samples']) if projection else composition_summary(timing['samples']) if composition else selection_summary(timing['samples'])
    if [b['block'] for b in timing['blocks']] != list(range(4)):
        raise ValueError('incomplete selection timing conditions')
    for block in [*timing['blocks'],*record['terminal']['blocks'],*([c['conditions'] for c in record['captures']] if composition or projection else [])]:
        for side in ('before','after'):
            require_ac(block[side])
            require_nominal_thermal_state(block[side])
            if block[side]['power_mode_raw'] != '0':
                raise ValueError('selection power mode changed')
    if Counter((n['prefix'],n['variant' if composition or projection else 'selection']) for n in timing['numerical'])!=Counter((p,s) for p in contract.PREFIXES for s in range(1,variant_count)):
        raise ValueError('incomplete selection numerical coverage')
    names = {'logits'}|{f'{k}{l}' for k in ('k','v') for l in range(24)}
    for check in timing['numerical']:
        if (not composition and not projection and not check['nonfinite_invalidates']) or len(check['history'])!=check['prefix']+1:
            raise ValueError('invalid selection lifecycle evidence')
        if projection and check['prefix']==64:
            layers=check.get('layers',[])
            if Counter(r['name'] for r in layers)!=Counter(swap_capture_names(extra_norm=True)) or not all(r['exact'] and r['bytes']>0 and len(r['sha256'])==64 for r in layers):
                raise ValueError('incomplete projection layer evidence')
        if composition and check['prefix']==64:
            validate_swap_checks(check.get('swap_checks',{}),extra_norm=True)
        for field in (('observations',) if composition or projection else ('observations','actual')):
            observations = check[field]
            if len(observations)!=49 or {r['name'] for r in observations}!=names or not all(r['exact'] for r in observations):
                raise ValueError('selection numerical invariant failed')
        if not all(r['finite'] and r['prefix_exact'] and r['inactive_exact'] for r in check['observations']):
            raise ValueError('selection cache invariant failed')
    if len(record['captures'])!=variant_count:
        raise ValueError('incomplete selection trace census')
    for selection,capture in enumerate(record['captures']):
        provenance = capture['provenance']
        implementation = 'qwen_model_all_three' if projection else contract.COMPOSITION_IMPLEMENTATIONS[selection] if composition else ['qwen_model_combined','qwen_model_gpu_argmax','qwen_model_fused_head'][selection]
        contract.configuration(provenance)
        if (provenance['implementation']!=implementation or provenance['binary']!=build_record['binaries'][f'profile-1024-s{selection}']
            or provenance['repository']!=build_record['source']['repository'] or provenance['source_sha256']!=build_record['source']['sources']
            or provenance['assets']!={k:v for k,v in build_record['assets'].items() if k.endswith('_sha256')}):
            raise ValueError('selection trace provenance mismatch')
        if projection and (provenance.get('projection_variant')!=selection or capture['analysis']['capture_identity']['workload'].get('projection_variant')!=selection):
            raise ValueError('projection variant identity changed')
        if composition or projection:
            original=capture.get('provenance_text','').encode()
            identity=capture['analysis']['capture_identity']['provenance']
            if (not original or hashlib.sha256(original).hexdigest()!=identity['sha256']
                or len(original)!=identity['bytes'] or json.loads(original)!=provenance):
                raise ValueError('residual-norm trace capture provenance changed')
        stages = contract.command_stages(*contract.options(implementation))
        if len(capture['samples'])!=8*len(stages):
            raise ValueError('incomplete selection command census')
        for i,row in enumerate(capture['samples']):
            if (row['iteration'],row['dispatch'],row['layer'],row['stage'],row['kind'])!=(i//len(stages),i%len(stages),*stages[i%len(stages)]):
                raise ValueError('selection trace stage changed')
            intervals=row['active_intervals']
            if row['duration_ns']<=0 or not intervals or sum(d for _,d in intervals)!=row['duration_ns'] or any(d<=0 for _,d in intervals):
                raise ValueError('invalid selection trace fragments')
            if (composition or projection) and (len(intervals)!=row['segments'] or intervals[0][0]!=row['start_ns']
                or sum(intervals[-1])!=row['end_ns'] or any(a+d>b for (a,d),(b,_) in zip(intervals,intervals[1:]))):
                raise ValueError('residual-norm active fragment geometry changed')
    sys.path.insert(0,str(repository_root()/'tests'))
    from chat_terminal import validate
    blocks=record['terminal']['blocks']
    if [b['block'] for b in blocks]!=list(range(4)):
        raise ValueError('incomplete selection terminal blocks')
    for block in blocks:
        arms=block['arms']
        if len(arms)!=variant_count or {a['selection'] for a in arms}!=set(range(variant_count)) or len({a['output_sha256'] for a in arms})!=1:
            raise ValueError('selection terminal arms changed')
        for arm in arms:
            if validate(arm['events'],128)!=arm['turns'] or len(arm['turns'])!=3:
                raise ValueError('selection terminal events changed')
            if any(a['generated']!=b['generated'] or a['history']!=b['history'] for a,b in zip(arm['turns'],arms[0]['turns'])):
                raise ValueError('selection terminal tokens changed')
    if projection:
        confirmation=record.get('confirmation')
        if bool(confirmation)!=bool(summary['screen_selected']):
            raise ValueError('projection confirmation does not match screen decision')
        if confirmation:
            if (confirmation['build']!=build_record or confirmation['selected']!=summary['screen_selected']
                or confirmation['screen_sha256']!=record['screen_timing_sha256']):
                raise ValueError('projection confirmation identity changed')
            if [b['block'] for b in confirmation['blocks']]!=list(range(4)):
                raise ValueError('incomplete projection confirmation conditions')
            for block in confirmation['blocks']:
                for side in ('before','after'):
                    require_ac(block[side]);require_nominal_thermal_state(block[side])
                    if block[side]['power_mode_raw']!='0': raise ValueError('confirmation power mode changed')
            confirmed=fusion_summary(confirmation['samples'])
            if confirmed!=confirmation['summary'] or confirmation['promote']!=all(c['promote'] for c in confirmed):
                raise ValueError('projection confirmation result changed')
            summary.update(confirmation=confirmed,promote=confirmation['promote'],confirmation_required=False)
    write(directory/(stem+'-summary.json'),summary)
    print(json.dumps(summary,indent=2))
    return summary


SCHEDULING_MODES = ['fixed','advance','observed-fixed','observed-advance']
SCHEDULING_DECLARATION = dict(kind='projection-scheduling-v1',variants=[0,1],prefixes=[64,1024,3968],
    modes=SCHEDULING_MODES,blocks=4,warmups=16,samples=64,comparisons=['self','candidate'],
    trace_repeats=2,trace_warmups=10,trace_samples=8,promotion=False)


def scheduling_parse(stdout, mode, prefix, block, comparison):
    if 'device: Apple M4 Pro\napi: metal\n' not in stdout or stdout.count('SCHEDULING_COMPLETE')!=1:
        raise ValueError('missing scheduling runtime identity/completion')
    observed=mode.startswith('observed-');rows=[]
    for line in stdout.splitlines():
        if not line.startswith('SCHED_SAMPLE '): continue
        values=list(map(int,line.split()[1:]))
        if len(values)!=(14 if observed else 4): raise ValueError('invalid scheduling sample width')
        arm,sample,token,elapsed,*marks=values
        if elapsed<=0 or not 0<=token<151936 or (observed and (marks!=sorted(marks) or marks[0]<0 or marks[-1]>elapsed)):
            raise ValueError('invalid scheduling sample')
        rows.append(dict(mode=mode,prefix=prefix,block=block,comparison=comparison,arm=arm,
                         sample=sample,token=token,elapsed_ns=elapsed,marks=marks))
    if Counter((x['arm'],x['sample']) for x in rows)!=Counter((a,i) for a in (0,1) for i in range(64)):
        raise ValueError('incomplete scheduling pair')
    if [x['token'] for x in rows if x['arm']==0]!=[x['token'] for x in rows if x['arm']==1]:
        raise ValueError('scheduling trajectory changed')
    if mode.endswith('fixed') and len({x['token'] for x in rows})!=1:
        raise ValueError('fixed-position prediction changed')
    return rows


def scheduling_host(stdout):
    rows=[]
    for line in stdout.splitlines():
        if line.startswith('SCHED_HOST '):
            values=list(map(int,line.split()[1:]))
            if len(values)!=12: raise ValueError('invalid scheduling host record width')
            iteration,elapsed,*marks=values
            if elapsed<=0 or marks!=sorted(marks) or marks[0]<0 or marks[-1]>elapsed:
                raise ValueError('invalid scheduling host marks')
            rows.append(dict(iteration=iteration,elapsed_ns=elapsed,marks=marks))
    if [x['iteration'] for x in rows]!=list(range(8)): raise ValueError('incomplete scheduling host records')
    return rows


def scheduling_summary(rows):
    expected=Counter((m,p,b,c,a,i) for m in SCHEDULING_MODES for p in contract.PREFIXES
                     for b in range(4) for c in (0,1) for a in (0,1) for i in range(64))
    if Counter(tuple(x[k] for k in ('mode','prefix','block','comparison','arm','sample')) for x in rows)!=expected:
        raise ValueError('incomplete scheduling timing census')
    for x in rows:
        marks=x['marks']
        if (type(x['elapsed_ns']) is not int or x['elapsed_ns']<=0 or not 0<=x['token']<151936
            or len(marks)!=(10 if x['mode'].startswith('observed-') else 0)
            or (marks and (marks!=sorted(marks) or marks[0]<0 or marks[-1]>x['elapsed_ns']))):
            raise ValueError('invalid scheduling retained sample')
    # Observation clocks and block order must not change the generated workload.
    for p in contract.PREFIXES:
        for mode in ('fixed','advance'):
            groups=defaultdict(list)
            for x in rows:
                if x['prefix']==p and x['mode'].removeprefix('observed-')==mode:
                    groups[x['mode'],x['block'],x['comparison'],x['arm']].append(x)
            sequences={tuple(x['token'] for x in sorted(xs,key=lambda x:x['sample'])) for xs in groups.values()}
            if len(sequences)!=1: raise ValueError('scheduling trajectory differs across observation modes/blocks')
    results=[]
    for m in SCHEDULING_MODES:
        for p in contract.PREFIXES:
            rr=[x for x in rows if x['mode']==m and x['prefix']==p]
            med={(b,c,a):stats.median(x['elapsed_ns'] for x in rr if (x['block'],x['comparison'],x['arm'])==(b,c,a))
                 for b in range(4) for c in (0,1) for a in (0,1)}
            for b in range(4):
                for c in (0,1):
                    tokens=[[x['token'] for x in sorted(rr,key=lambda x:x['sample']) if (x['block'],x['comparison'],x['arm'])==(b,c,a)] for a in (0,1)]
                    if tokens[0]!=tokens[1]: raise ValueError('retained scheduling tokens differ')
                    if m.endswith('fixed') and len(set(tokens[0]))!=1: raise ValueError('retained fixed prediction changed')
            cal=[med[b,0,1]/med[b,0,0] for b in range(4)];ratios=[med[b,1,1]/med[b,1,0] for b in range(4)]
            noise=max(.05,max(abs(x-1) for x in cal));reduction=1-stats.median(ratios)
            result=dict(mode=m,prefix=p,calibration_ratios=cal,ratios=ratios,noise_floor=noise,
                median_reduction=reduction,qualifies=all(x<1 for x in ratios) and reduction>noise,
                control_block_ms=[med[b,1,0]/1e6 for b in range(4)],candidate_block_ms=[med[b,1,1]/1e6 for b in range(4)])
            result['quarters']=[dict(arm=a,block=b,first_ms=stats.median(x['elapsed_ns'] for x in rr if x['comparison']==1 and x['block']==b and x['arm']==a and x['sample']<16)/1e6,
                last_ms=stats.median(x['elapsed_ns'] for x in rr if x['comparison']==1 and x['block']==b and x['arm']==a and x['sample']>=48)/1e6) for a in (0,1) for b in range(4)]
            if m.startswith('observed-'):
                result['host']=[dict(arm=a,block=b,
                    forward_ms=stats.median(x['marks'][5] for x in rr if x['comparison']==1 and x['block']==b and x['arm']==a)/1e6,
                    readback_ms=stats.median(x['marks'][7]-x['marks'][6] for x in rr if x['comparison']==1 and x['block']==b and x['arm']==a)/1e6) for a in (0,1) for b in range(4)]
            results.append(result)
    return results


def scheduling_timeline(capture):
    rows=capture['samples'];result=[]
    matrix={'packed QKV projection','output projection','gate projection','up projection','down projection','vocabulary projection'}
    for i in range(8):
        rr=[x for x in rows if x['iteration']==i and x['kind']=='compute']
        intervals=sorted((a,a+d) for x in rr for a,d in x['active_intervals'])
        end=intervals[0][0];union=0
        for a,b in intervals:
            union+=max(0,b-max(a,end));end=max(end,b)
        span=end-intervals[0][0]
        result.append(dict(iteration=i,active_ns=union,span_ns=span,uncovered_ns=span-union,
            projections_ns=sum(x['duration_ns'] for x in rr if x['stage'] in matrix),
            attention_ns=sum(x['duration_ns'] for x in rr if x['stage']=='FP32 GQA'),
            submission_span_ns=max(x['submission_start_ns']+x['submission_duration_ns'] for x in rr)-min(x['submission_start_ns'] for x in rr),
            fragmented_commands=sum(x['segments']>1 for x in rr)))
    return result


def scheduling_replay(directory):
    packed=(directory/'projection-scheduling.json.gz').read_bytes();raw=gzip.decompress(packed)
    if json.loads((directory/'projection-scheduling.json').read_text())!=dict(sha256=hashlib.sha256(packed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()):
        raise ValueError('scheduling archive hash mismatch')
    r=json.loads(raw);t=r['timing'];build=t['build']
    if r['kind']!='projection-scheduling-v1' or build['declaration']!=SCHEDULING_DECLARATION or build['source']['repository']['dirty']:
        raise ValueError('scheduling declaration/source changed')
    summary=scheduling_summary(t['samples'])
    if summary!=t['summary']: raise ValueError('scheduling summary changed')
    if [b['block'] for b in t['blocks']]!=list(range(4)): raise ValueError('scheduling conditions missing')
    if Counter((x['mode'],x['prefix']) for x in t['numerical'])!=Counter((m,p) for m in SCHEDULING_MODES for p in contract.PREFIXES):
        raise ValueError('scheduling numerical coverage missing')
    for n in t['numerical']:
        obs=n['observations'];names={'logits'}|{f'{k}{i}' for k in ('k','v') for i in range(24)}
        if (len(n['history'])!=n['prefix']+1 or len(obs)!=49 or {x['name'] for x in obs}!=names
            or not all(all(x[k] for k in ('exact','finite','prefix_exact','inactive_exact')) for x in obs)):
            raise ValueError('scheduling numerical invariant changed')
    if Counter((c['prefix'],c['provenance']['projection_variant'],c['repeat']) for c in r['captures'])!=Counter((p,v,r) for p in contract.PREFIXES for v in (0,1) for r in range(2)):
        raise ValueError('scheduling capture coverage missing')
    for block in [*t['blocks'],*[c['conditions'] for c in r['captures']]]:
        for side in ('before','after'):
            require_ac(block[side]);require_nominal_thermal_state(block[side])
            if block[side]['power_mode_raw']!='0': raise ValueError('scheduling power changed')
    traces=[]
    for c in r['captures']:
        p=c['prefix'];prov=c['provenance'];v=prov['projection_variant'];contract.configuration(prov)
        identity=c['analysis']['capture_identity'];original=c['provenance_text'].encode()
        if (prov['implementation']!='qwen_model_all_three' or prov['binary']!=build['binaries'][f'profile-{p}-s{v}']
            or prov['repository']!=build['source']['repository'] or prov['source_sha256']!=build['source']['sources']
            or prov['assets']!={k:x for k,x in build['assets'].items() if k.endswith('_sha256')}
            or identity['workload']['projection_variant']!=v or prov['profile_workload']!=f'model-p{p}-all-three'
            or hashlib.sha256(original).hexdigest()!=identity['provenance']['sha256']
            or len(original)!=identity['provenance']['bytes'] or json.loads(original)!=prov):
            raise ValueError('scheduling trace provenance changed')
        receipt_bytes=c['capture_receipt_text'].encode()
        receipt_capture=json.loads(receipt_bytes)['capture']
        if (hashlib.sha256(receipt_bytes).hexdigest()!=identity['capture_receipt']['sha256']
            or len(receipt_bytes)!=identity['capture_receipt']['bytes']
            or receipt_capture['capture_id']!=identity['capture_id']
            or receipt_capture['target_output']!=c['target_output']):
            raise ValueError('scheduling capture receipt changed')
        output=c['target_text'].encode()
        if c['target_output']['sha256']!=hashlib.sha256(output).hexdigest() or c['target_output']['bytes']!=len(output) or scheduling_host(c['target_text'])!=c['host']:
            raise ValueError('scheduling host capture changed')
        stages=contract.command_stages(*contract.options('qwen_model_all_three'))
        if len(c['samples'])!=8*len(stages): raise ValueError('scheduling command census changed')
        for i,x in enumerate(c['samples']):
            if (x['iteration'],x['dispatch'],x['layer'],x['stage'],x['kind'])!=(i//len(stages),i%len(stages),*stages[i%len(stages)]):
                raise ValueError('scheduling command identity changed')
            parts=x['active_intervals']
            if (not parts or len(parts)!=x['segments'] or sum(d for _,d in parts)!=x['duration_ns'] or parts[0][0]!=x['start_ns']
                or sum(parts[-1])!=x['end_ns'] or any(d<=0 for _,d in parts) or any(a+d>b for (a,d),(b,_) in zip(parts,parts[1:]))):
                raise ValueError('scheduling command fragments changed')
        traces.append(dict(prefix=p,variant=v,repeat=c['repeat'],timeline=scheduling_timeline(c),host=c['host']))
    result=dict(timing=summary,traces=traces,promote=False)
    write(directory/'projection-scheduling-summary.json',result)
    print('Verified scheduling archive: 12288 timings, 12 numerical cases, 23904 commands, 96 host records')
    return result


def scheduling_plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=scheduling_replay(directory)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors=['#247da3','#cf8531'];positions=list(range(3))
    for mode,color in zip(['fixed','advance'],colors):
        values=[next(x for x in result['timing'] if x['mode']==mode and x['prefix']==p) for p in contract.PREFIXES]
        for arm,style in [('control','--'),('candidate','-')]:
            axes[0,0].plot(positions,[stats.median(x[arm+'_block_ms']) for x in values],style,marker='o',color=color,label=mode+' '+arm)
        for i,x in enumerate(values):
            pos=i+(-.1 if mode=='fixed' else .1)
            axes[0,1].scatter([pos]*4,x['ratios'],color=color,label=mode if i==0 else None)
    axes[0,0].set_title('Untraced complete-token latency');axes[0,0].set_ylabel('ms/token')
    axes[0,0].legend(fontsize=8);axes[0,1].axhline(1,color='gray',linestyle='--')
    axes[0,1].set_title('Paired fixed-width / original ratios');axes[0,1].set_ylabel('Lower is better');axes[0,1].legend()
    for i,p in enumerate(contract.PREFIXES):
        obs=next(x for x in result['timing'] if x['mode']=='observed-fixed' and x['prefix']==p)
        for v in (0,1):
            pos=i+(-.17 if v==0 else .17)
            hh=[x for x in obs['host'] if x['arm']==v]
            forward=stats.median(x['forward_ms'] for x in hh);wait=stats.median(x['readback_ms'] for x in hh)
            axes[1,0].bar(pos,forward,.3,color='#247da3',label='Forward wall interval' if i==v==0 else None)
            axes[1,0].bar(pos,wait,.3,bottom=forward,color='#b7cfda',label='Readback wait interval' if i==v==0 else None)
            tt=[x for c in result['traces'] if c['prefix']==p and c['variant']==v for x in c['timeline']]
            active=stats.median(x['active_ns'] for x in tt)/1e6;gap=stats.median(x['uncovered_ns'] for x in tt)/1e6
            axes[1,1].bar(pos,active,.3,color='#31877c',label='Target compute active' if i==v==0 else None)
            axes[1,1].bar(pos,gap,.3,bottom=active,color='#c9dbce',label='Uncovered by target compute' if i==v==0 else None)
    axes[1,0].set_title('Untraced observed fixed-position host intervals');axes[1,0].set_ylabel('ms; GPU work overlaps forward')
    axes[1,1].set_title('Separate fixed-position traces');axes[1,1].set_ylabel('ms; medians of interval components')
    for ax in axes[1]: ax.legend(fontsize=8);ax.set_xlabel('Cached tokens · original left, fixed-width right')
    for ax in axes.flat: ax.set_xticks(positions,list(contract.PREFIXES))
    fig.suptitle('Why does fixed-width projection speedup depend on the measurement?\nQwen2.5-0.5B · M4 Pro / Metal · BF16 · fixed 128-thread blocks',fontsize=13)
    fig.savefig(directory/'projection-scheduling.png',dpi=170);plt.close(fig)


ENQUEUE_DECLARATION = dict(kind='runtime-enqueue-v1', blocks=4, prefixes=[64,1024,3968],
    states=['plain','disabled','enabled'], model_samples=64, model_warmups=16,
    shapes=['tiny','down'], batches=[1,256], micro_samples=10, micro_warmups=10,
    queue_passes=2, expected_model_calls=245, promotion=False)
PROBE_SOURCE = 'src/llm_mojo/benchmarks/enqueue_probe.c'


def enqueue_windows(stdout, kind, **metadata):
    if kind=='model':
        rows=scheduling_parse(stdout,'observed-fixed',metadata['prefix'],metadata['block'],1)
        windows={}
        for line in stdout.splitlines():
            if line.startswith('ENQUEUE_WINDOW '):
                a,i,t=map(int,line.split()[1:])
                if (a,i) in windows: raise ValueError('duplicate enqueue window')
                windows[a,i]=t
        if set(windows)!={(r['arm'],r['sample']) for r in rows}: raise ValueError('missing enqueue windows')
        for r in rows:
            t=windows[r['arm'],r['sample']]
            r.update(start_ns=t+r['marks'][2],end_ns=t+r['marks'][5])
    else:
        if 'device: Apple M4 Pro\napi: metal\n' not in stdout or stdout.count('LAUNCH_MICRO_COMPLETE')!=1:
            raise ValueError('missing micro runtime identity/completion')
        rows=[]
        for line in stdout.splitlines():
            if line.startswith('LAUNCH_SAMPLE '):
                a,i,start,end,done=map(int,line.split()[1:])
                if not 0<start<end<done: raise ValueError('invalid launch timing')
                rows.append(dict(arm=a,sample=i,start_ns=start,end_ns=end,elapsed_ns=done-start))
        if Counter((r['arm'],r['sample']) for r in rows)!=Counter((a,i) for a in (0,1) for i in range(10)):
            raise ValueError('incomplete launch microbenchmark')
    rows.sort(key=lambda r:r['start_ns'])
    if any(r['start_ns']<=0 or r['end_ns']<=r['start_ns'] for r in rows) or any(a['end_ns']>b['start_ns'] for a,b in zip(rows,rows[1:])):
        raise ValueError('invalid/overlapping enqueue windows')
    return rows


def enqueue_partition(rows, calls, expected):
    # Calls are complete runtime-wall intervals; reject overlaps or straddling.
    if any(len(c)!=11 or c[0]>=c[1] or c[-1]!=0 or min(c[3:10])<=0 for c in calls):
        raise ValueError('invalid enqueue call/error')
    if any(a[1]>b[0] for a,b in zip(calls,calls[1:])) or len({c[2] for c in calls})!=1:
        raise ValueError('overlapping or multithreaded enqueue calls')
    selected=[];index=0
    for r in rows:
        start,end=r['start_ns'],r['end_ns']
        while index<len(calls) and calls[index][1]<=start: index+=1
        group=[]
        while index<len(calls) and calls[index][0]<end:
            c=calls[index]
            if c[0]<start or c[1]>end: raise ValueError('enqueue straddles timing boundary')
            group.append(c);index+=1
        if len(group)!=expected: raise ValueError('unexpected enqueue count per window')
        selected.append(group)
    return selected


def enqueue_summary(record):
    if record['kind']!=ENQUEUE_DECLARATION['kind'] or record['build']['declaration']!=ENQUEUE_DECLARATION or record['build']['source']['repository']['dirty']:
        raise ValueError('invalid enqueue study identity')
    runs=record['runs']; model=[];micro=[];queue=[]
    expected=Counter([('model',b,p,s) for b in range(4) for p in contract.PREFIXES for s in ENQUEUE_DECLARATION['states']]
        +[('micro',b,s,n,c) for b in range(4) for s in ('tiny','down') for n in (1,256) for c in (0,1)]
        +[('queue',r,s) for r in range(2) for s in ('tiny','down')])
    keys=[]
    for run in runs:
        kind=run['kind']
        if (kind=='micro' and run['state']!='plain') or (kind=='queue' and
            (run['state']!='enabled' or run['batch']!=256 or run['comparison']!=0)):
            raise ValueError('enqueue workload conditions changed')
        if kind=='model': keys.append((kind,run['block'],run['prefix'],run['state']))
        elif kind=='micro': keys.append((kind,run['block'],run['shape'],run['batch'],run['comparison']))
        elif kind=='queue': keys.append((kind,run['repeat'],run['shape']))
        else: raise ValueError('unknown enqueue run')
        if run['stdout_sha256']!=hashlib.sha256(run['stdout'].encode()).hexdigest(): raise ValueError('enqueue stdout hash changed')
        rows=enqueue_windows(run['stdout'],kind,**{k:v for k,v in run.items() if k not in ('kind','stdout')})
        if run['state']=='enabled':
            probe=run['probe'];groups=probe['groups'];count=245 if kind=='model' else run['batch']
            if probe['dropped']!=0 or len(groups)!=len(rows) or probe['total_calls']<len(rows)*count:
                raise ValueError('incomplete retained probe')
            if enqueue_partition(rows,[c for g in groups for c in g],count)!=groups:
                raise ValueError('retained enqueue partition changed')
            for r,g in zip(rows,groups):
                r['runtime_ns']=sum(c[1]-c[0] for c in g)
                r['early_ns']=stats.median(c[1]-c[0] for c in g[:16])
                r['late_ns']=stats.median(c[1]-c[0] for c in g[-16:])
        elif run['probe'] is not None or run['state'] not in ('plain','disabled'):
            raise ValueError('unexpected probe')
        for r in rows:
            r.update({k:v for k,v in run.items() if k not in ('stdout','stdout_sha256','probe')})
            (model if kind=='model' else micro if kind=='micro' else queue).append(r)
    if Counter(keys)!=expected: raise ValueError('incomplete enqueue census')
    if [(x.get('block'),x.get('repeat')) for x in record['blocks']]!=[(b,None) for b in range(4)]+[(None,r) for r in range(2)]:
        raise ValueError('missing enqueue conditions')
    for block in record['blocks']:
        for side in ('before','after'):
            require_ac(block[side]);require_nominal_thermal_state(block[side])
            if block[side]['power_mode_raw']!='0': raise ValueError('enqueue power mode changed')
    numerical=record['numerical']
    if Counter((n['prefix'],n['state']) for n in numerical)!=Counter((p,s) for p in contract.PREFIXES for s in ENQUEUE_DECLARATION['states']):
        raise ValueError('missing enqueue numerical coverage')
    names={'logits'}|{f'{k}{i}' for k in ('k','v') for i in range(24)}
    for n in numerical:
        if len(n['history'])!=n['prefix']+1 or len(n['observations'])!=49 or {x['name'] for x in n['observations']}!=names or not all(all(x[k] for k in ('exact','finite','prefix_exact','inactive_exact')) for x in n['observations']):
            raise ValueError('enqueue numerical invariant changed')
    for p in contract.PREFIXES:
        if len({x['token'] for x in model if x['prefix']==p})!=1: raise ValueError('enqueue trajectory changed')
        captures=[n for n in numerical if n['prefix']==p]
        if any(n['history']!=captures[0]['history'] for n in captures): raise ValueError('enqueue history changed')
        for name in names:
            hashes=[next(x for x in n['observations'] if x['name']==name)['hashes'] for n in captures]
            if any(h!=hashes[0] for h in hashes): raise ValueError('probe changed complete output bytes')
    attribution=[]
    for p in contract.PREFIXES:
        for arm in (0,1):
            entry=dict(prefix=p,variant=arm)
            for state in ENQUEUE_DECLARATION['states']:
                subset=[x for x in model if x['prefix']==p and x['arm']==arm and x['state']==state]
                metric=lambda fn:stats.median(stats.median(fn(x) for x in subset if x['block']==b) for b in range(4))
                entry[state]=dict(token_ms=metric(lambda x:x['elapsed_ns'])/1e6,
                    forward_ms=metric(lambda x:x['marks'][5]-x['marks'][0])/1e6,
                    enqueue_window_ms=metric(lambda x:x['end_ns']-x['start_ns'])/1e6)
                if state=='enabled':
                    entry[state].update(runtime_ms=metric(lambda x:x['runtime_ns'])/1e6,
                        outside_runtime_ms=metric(lambda x:x['end_ns']-x['start_ns']-x['runtime_ns'])/1e6,
                        runtime_fraction=metric(lambda x:x['runtime_ns']/(x['end_ns']-x['start_ns'])))
            for state in ('disabled','enabled'):
                entry[state]['token_ratios_to_plain']=[stats.median(x['elapsed_ns'] for x in model if x['prefix']==p and x['arm']==arm and x['state']==state and x['block']==b)/stats.median(x['elapsed_ns'] for x in model if x['prefix']==p and x['arm']==arm and x['state']=='plain' and x['block']==b) for b in range(4)]
            attribution.append(entry)
    cache=[]
    for shape in ('tiny','down'):
        for batch in (1,256):
            s=[x for x in micro if x['shape']==shape and x['batch']==batch]
            entry=dict(shape=shape,batch=batch)
            for boundary in ('submit','complete'):
                def med(b,c,a):
                    return stats.median((x['end_ns']-x['start_ns'] if boundary=='submit' else x['elapsed_ns'])/batch for x in s if x['block']==b and x['comparison']==c and x['arm']==a)
                ratios=[med(b,1,1)/med(b,1,0) for b in range(4)]
                noise=max(abs(med(b,0,1)/med(b,0,0)-1) for b in range(4))
                entry[boundary]=dict(original_us=stats.median(med(b,1,0) for b in range(4))/1e3,
                    compiled_us=stats.median(med(b,1,1) for b in range(4))/1e3,ratios=ratios,
                    self_deviation=noise,qualifies=all(r<1 for r in ratios) and 1-stats.median(ratios)>max(.05,noise))
            cache.append(entry)
    pressure=[]
    for shape in ('tiny','down'):
        s=[x for x in queue if x['shape']==shape]
        pressure.append(dict(shape=shape,early_call_us=stats.median(x['early_ns'] for x in s)/1e3,
            late_call_us=stats.median(x['late_ns'] for x in s)/1e3,
            runtime_per_call_us=stats.median(x['runtime_ns']/256 for x in s)/1e3,
            submission_per_call_us=stats.median((x['end_ns']-x['start_ns'])/256 for x in s)/1e3))
    return dict(model_samples=len(model),micro_samples=len(micro),queue_samples=len(queue),
        measured_runtime_calls=sum(len(g) for r in runs if r['probe'] for g in r['probe']['groups']),
        attribution=attribution,compiled_handle=cache,queue_pressure=pressure,promote=False)


def enqueue_replay(directory):
    packed=(directory/'runtime-enqueue.json.gz').read_bytes();raw=gzip.decompress(packed)
    if json.loads((directory/'runtime-enqueue.json').read_text())!=dict(sha256=hashlib.sha256(packed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()):
        raise ValueError('enqueue archive hash mismatch')
    r=json.loads(raw);summary=enqueue_summary(r)
    if summary!=r['summary']: raise ValueError('enqueue archived summary changed')
    return summary


def enqueue_plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=enqueue_replay(directory)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    rows=summary['attribution'];positions=np.arange(len(rows))
    axes[0].barh(positions,[r['enabled']['runtime_ms'] for r in rows],color='#306998',label='Inside runtime')
    axes[0].barh(positions,[r['enabled']['outside_runtime_ms'] for r in rows],
        left=[r['enabled']['runtime_ms'] for r in rows],color='#e5a43c',label='Outside runtime')
    axes[0].set_yticks(positions,[str(r['prefix'])+(' original' if r['variant']==0 else ' fixed') for r in rows])
    axes[0].invert_yaxis();axes[0].set_xlabel('Milliseconds per token (recording enabled)')
    axes[0].set_title('Where does launch submission spend time?')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2)
    axes[0].set_xlim(0,9)
    pressure=summary['queue_pressure'];x=np.arange(2)
    axes[1].bar(x-.18,[r['early_call_us'] for r in pressure],.36,label='First 16 enqueues',color='#306998')
    axes[1].bar(x+.18,[r['late_call_us'] for r in pressure],.36,label='Last 16 enqueues',color='#e5a43c')
    axes[1].set_xticks(x,['Tiny projection','Down projection']);axes[1].set_ylabel('Runtime wall time per call (microseconds)')
    axes[1].set_title('Enqueues can wait as GPU work accumulates');axes[1].legend()
    fig.suptitle('M4 Pro / Metal: runtime time includes blocking, not only CPU work',fontsize=12)
    fig.savefig(directory/'runtime-enqueue.png',dpi=160);plt.close(fig)


BATCH_SUPPORT_DECLARATION = dict(kind='metal-batch-support-v1',repeats=2,
    control='two ordered int32 x=2*x+1 dispatches from x=3',
    graph='same dependent dispatches, replay twice: 15 then 63',timing=False)


def batch_support_parse(stdout):
    if 'device: Apple M4 Pro\napi: metal\n' not in stdout or stdout.count('BATCH_EAGER_PASS 15')!=1 or stdout.count('BATCH_SUPPORT_COMPLETE')!=1:
        raise ValueError('missing batching control/device/completion')
    errors=[l for l in stdout.splitlines() if l.startswith('BATCH_GRAPH_ERROR ')]
    if errors:
        if len(errors)!=1 or 'createGraphBuilder() not supported on this device context' not in errors[0] or 'BATCH_BUILDER_ENTERED' in stdout or 'BATCH_GRAPH_PASS' in stdout:
            raise ValueError('unexpected graph failure')
        return dict(status='graph-unsupported',eager_value=15,builder_entered=False,error=errors[0])
    if stdout.count('BATCH_BUILDER_ENTERED')!=1 or stdout.count('BATCH_GRAPH_PASS 15 63')!=1:
        raise ValueError('incomplete graph replay validation')
    return dict(status='graph-replay-supported',eager_value=15,builder_entered=True,graph_values=[15,63])


def batch_support_collect(output):
    ensure_record_location(output);output.mkdir(parents=True,exist_ok=False)
    source=source_identity();machine=stable_environment();before=conditions()
    if source['repository']['dirty']: raise ValueError('batch support requires clean source')
    if machine['software']['max']!='26.5.0' or machine['software']['mojo']!='1.0.0' or machine['hardware']['chip']!='Apple M4 Pro':
        raise ValueError('batch support probe targets pinned M4 Pro / MAX 26.5.0')
    runtime=repository_root()/'.venv/lib/python3.12/site-packages/modular/lib/libAsyncRTMojoBindings.dylib'
    command=[environment_tool('mojo'),'build','-I','src','-D','MODEL_BATCH_SUPPORT',
        'src/llm_mojo/benchmarks/model.mojo','-o',output/'probe']
    execute(command,output/'build.log')
    exports=execute(['xcrun','dyld_info','-exports',runtime],output/'exports.log')
    entrypoints=[line.split()[-1] for line in exports.splitlines()
                 if '_AsyncRT_DeviceContext_' in line or '_AsyncRT_DeviceStream_' in line or '_AsyncRT_DeviceGraph' in line]
    runs=[]
    for repeat in range(2):
        stdout=execute([output/'probe'],output/f'run-{repeat}.log')
        runs.append(dict(repeat=repeat,stdout=stdout,stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),result=batch_support_parse(stdout)))
    if source_identity()!=source or stable_environment()!=machine: raise ValueError('batch probe source/environment changed')
    record=dict(declaration=BATCH_SUPPORT_DECLARATION,source=source,environment=machine,
        conditions=dict(before=before,after=conditions()),command=[str(x) for x in command],
        binary=dict(sha256=sha(output/'probe'),bytes=(output/'probe').stat().st_size),
        runtime=dict(sha256=sha(runtime),bytes=runtime.stat().st_size),
        entrypoints=entrypoints,exports_sha256=sha(output/'exports.log'),runs=runs)
    write(output/'batch-support.json',record)
    print(json.dumps([r['result'] for r in runs],indent=2))


def batch_support_replay(path):
    record=json.loads(path.read_text())
    if record['declaration']!=BATCH_SUPPORT_DECLARATION or record['source']['repository']['dirty']:
        raise ValueError('batch support declaration/source changed')
    if [r['repeat'] for r in record['runs']]!=[0,1]: raise ValueError('batch support repeats missing')
    if record['environment']['hardware']['chip']!='Apple M4 Pro' or record['environment']['hardware']['gpu_api']!='metal' or record['environment']['software']['max']!='26.5.0':
        raise ValueError('batch support device/backend changed')
    for side in ('before','after'):
        require_ac(record['conditions'][side]);require_nominal_thermal_state(record['conditions'][side])
        if record['conditions'][side]['power_mode_raw']!='0': raise ValueError('batch probe power mode changed')
    results=[]
    for r in record['runs']:
        if hashlib.sha256(r['stdout'].encode()).hexdigest()!=r['stdout_sha256']: raise ValueError('batch probe output hash changed')
        result=batch_support_parse(r['stdout'])
        if result!=r['result']: raise ValueError('batch probe result changed')
        results.append(result)
    if results[0]!=results[1]: raise ValueError('batch capability differed between processes')
    if '_AsyncRT_DeviceContext_createGraphBuilder' not in record['entrypoints'] or '_AsyncRT_DeviceContext_enqueueFunctionDirect' not in record['entrypoints']:
        raise ValueError('missing runtime entrypoints')
    return dict(status=results[0]['status'],timing_run=False,promote=False)


# Collectors for completed decode experiments built arms that are no longer in
# the engine. Their retained archives still replay; re-collection needs the
# commit recorded in each archive (collectors exist through edb610a).
BATCH_STEM = 'batch-size'
BATCH_KIND = 'qwen-batch-size-v1'
HOST_PHASES = ['preflight','step upload','embedding enqueue','decoder stack enqueue','head enqueue',
               'forward return','readback wait','host selection','unmap']
PROJECTION_BATCH_STEM = 'batch-projections'
PROJECTION_BATCH_KIND = 'qwen-batch-projections-v1'
REORDERED_BATCH_STEM = 'batch-reordered'
REORDERED_BATCH_KIND = 'qwen-batch-reordered-v1'
ADDRESSING_BATCH_STEM = 'batch-addressing'
ADDRESSING_BATCH_KIND = 'qwen-batch-addressing-v1'
PAGED_STEM = 'paged-kv'
PAGED_KIND = 'qwen-paged-kv-v1'
PAGED_LOOP_STEM = 'paged-kv-loop'
PAGED_LOOP_KIND = 'qwen-paged-kv-loop-v1'
PAGED_STUDIES = ('paged', 'paged-loop')


def batch_study(name):
    """1c's row tiles ('size') or 1d's projection arrangements ('projections').

    Each trace is (context, sequences, arrangement, binary name, provenance arm).
    """
    if name == 'size':
        traces = [(c, b, contract.BATCH_TILE_ARRANGEMENTS[t], f'batch-profile-{c}-{b}-{t}', dict(row_tile=t))
                  for c, b, t in contract.BATCH_TRACES]
        return dict(stem=BATCH_STEM, kind=BATCH_KIND, declaration=contract.BATCH_DECLARATION,
                    argument='size', traces=traces)
    if name == 'projections':
        traces = [(c, b, a, f'batch-profile-{c}-{b}-a{a}', dict(arrangement=a))
                  for c, b, a in contract.BATCH_PROJECTION_TRACES]
        return dict(stem=PROJECTION_BATCH_STEM, kind=PROJECTION_BATCH_KIND,
                    declaration=contract.BATCH_PROJECTION_DECLARATION, argument='projections', traces=traces)
    if name == 'reordered':
        traces = [(c, b, a, f'batch-profile-{c}-{b}-a{a}', dict(arrangement=a))
                  for c, b, a in contract.BATCH_REORDERED_TRACES]
        return dict(stem=REORDERED_BATCH_STEM, kind=REORDERED_BATCH_KIND,
                    declaration=contract.BATCH_REORDERED_DECLARATION, argument='reordered', traces=traces)
    if name == 'addressing':
        traces = [(c, b, a, f'batch-profile-{c}-{b}-a{a}', dict(arrangement=a))
                  for c, b, a in contract.BATCH_ADDRESSING_TRACES]
        return dict(stem=ADDRESSING_BATCH_STEM, kind=ADDRESSING_BATCH_KIND,
                    declaration=contract.BATCH_ADDRESSING_DECLARATION, argument='addressing', traces=traces)
    if name in PAGED_STUDIES:
        # 2d and its rerun share the benchmark's paged study; the build's kernel differs.
        traces = [(c, b, layout, f'paged-profile-{c}-{b}-l{layout}', dict(layout=layout))
                  for c, b, layout in contract.PAGED_TRACES]
        if name == 'paged':
            return dict(stem=PAGED_STEM, kind=PAGED_KIND, declaration=contract.PAGED_DECLARATION, argument='paged',
                        traces=traces)
        return dict(stem=PAGED_LOOP_STEM, kind=PAGED_LOOP_KIND, declaration=contract.PAGED_LOOP_DECLARATION,
                    argument='paged', traces=traces)
    raise ValueError('unknown batch study')


def batch_comparisons(argument):
    """(comparisons, observed comparison) run by the batch mode for one study argument."""
    if argument == 'size':
        return 4, 3
    if argument == 'projections':
        return 1+len(contract.BATCH_PROJECTION_ARRANGEMENTS), None
    if argument == 'reordered':
        return 1+len(contract.BATCH_REORDERED_ARRANGEMENTS), None
    if argument == 'addressing':
        return 1+len(contract.BATCH_ADDRESSING_ARRANGEMENTS), None
    if argument.startswith('confirm:') and argument[8:].isdigit() and int(argument[8:]) in contract.BATCH_PROJECTION_ARRANGEMENTS:
        return 2, None
    prefix = 'reordered-confirm:'
    if (argument.startswith(prefix) and argument[len(prefix):].isdigit()
            and int(argument[len(prefix):]) in contract.BATCH_REORDERED_ARRANGEMENTS):
        return 2, None
    if argument == 'paged':
        return 1+len(contract.PAGED_CANDIDATES), None
    prefix = 'paged-confirm:'
    if (argument.startswith(prefix) and argument[len(prefix):].isdigit()
            and int(argument[len(prefix):]) in contract.PAGED_CANDIDATES):
        return 2, None
    raise ValueError('unknown batch study argument')


def batch_trace_specification(context, sequences, arm):
    if 'layout' in arm:
        return contract.paged_specification(context, sequences, arm['layout'])
    if 'arrangement' in arm:
        return contract.batch_projection_specification(context, sequences, arm['arrangement'])
    return contract.batch_specification(context, sequences, arm['row_tile'])


def batch_build(output, prepared, study='size'):
    """Batch executables: the model driver and one trace binary per trace workload of the study."""
    spec = batch_study(study)
    ensure_record_location(output)
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    if source['repository']['dirty']:
        raise ValueError('model profiling build requires clean source')
    identity = assets(prepared)
    command = [environment_tool('mojo'), 'build', '-I', 'src', 'src/llm_mojo/benchmarks/model.mojo', '-o', output/'model']
    execute(command, output/'model-build.log')
    binaries = dict(model=dict(sha256=sha(output/'model'), bytes=(output/'model').stat().st_size))
    machine = stable_environment()
    for context, sequences, arrangement, name, arm in spec['traces']:
        specification = batch_trace_specification(context, sequences, arm)
        # A paged trace's third field is its layout; it decodes in the default arrangement.
        target = (['-D', f'MODEL_PAGED_PROFILE={context}', '-D', f'MODEL_BATCH_LAYOUT={arrangement}'] if 'layout' in arm
                  else ['-D', f'MODEL_BATCH_PROFILE={context}', '-D', f'MODEL_BATCH_ARRANGEMENT={arrangement}'])
        command = [environment_tool('mojo'), 'build', '-I', 'src', *target,
                   '-D', f'MODEL_BATCH_SEQUENCES={sequences}',
                   '-D', 'MODEL_BATCH_WORKLOAD='+specification['profile_workload'],
                   '-D', 'MODEL_PREPARED='+identity['prepared'],
                   '-D', 'MODEL_TABLES='+identity['tables'], 'src/llm_mojo/benchmarks/model.mojo', '-o', output/name]
        execute(command, output/f'{name}-build.log')
        binary = dict(sha256=sha(output/name), bytes=(output/name).stat().st_size)
        binaries[name] = binary
        provenance = dict(schema_version=1, operation=contract.OPERATION,
                          implementation=contract.BATCH_IMPLEMENTATION,
                          entrypoint=contract.ENTRYPOINTS[contract.BATCH_IMPLEMENTATION],
                          repository=source['repository'], source_sha256=source['sources'],
                          **machine, **specification, **arm,
                          profile_warmup_iterations=10, profile_iterations=8,
                          profile_post_idle_milliseconds=250, binary=binary,
                          assets={k:v for k,v in identity.items() if k.endswith('_sha256')})
        contract.configuration(provenance)
        write(output/(name+'.provenance.json'), provenance)
    if source_identity() != source or assets(prepared) != identity:
        raise ValueError('source or assets changed during compilation')
    write(output/'build.json', dict(source=source, assets=identity, environment=machine,
                                    declaration=spec['declaration'], binaries=binaries))


def batch_sizes(context):
    return list(contract.BATCH_SIZES) if context else [contract.BATCH_MIXED]


def parse_batch_samples(stdout, context, block, comparisons=4, observed=3, argument=None):
    """BATCH <sequences> <comparison> <arm> <sample> <elapsed>[ ten marks] records from one context process.

    Only the observed comparison's second arm carries host marks; 1d observes none.
    """
    if 'device: Apple M4 Pro\napi: metal\n' not in stdout or stdout.count('BATCH_COMPLETE') != 1:
        raise ValueError('missing measured device or completion')
    if argument is not None and f'\nstudy: {argument}\n' not in stdout:
        raise ValueError('batch process ran another study')
    records = []
    for line in stdout.splitlines():
        if not line.startswith('BATCH '):
            continue
        values = list(map(int, line.split()[1:]))
        if len(values) not in (5, 15):
            raise ValueError('invalid batch timing record')
        sequences, comparison, arm, sample, elapsed, *marks = values
        if elapsed <= 0 or bool(marks) != (comparison == observed and arm == 1):
            raise ValueError('incorrect observation arm')
        if marks and (marks != sorted(marks) or marks[0] < 0 or marks[-1] > elapsed):
            raise ValueError('invalid host timing sequence')
        records.append(dict(context=context, sequences=sequences, block=block, comparison=comparison,
                            arm=arm, sample=sample, elapsed_ns=elapsed, marks=marks))
    expected = Counter((b, c, a, s) for b in batch_sizes(context) for c in range(comparisons)
                       for a in range(2) for s in range(10))
    if Counter((r['sequences'], r['comparison'], r['arm'], r['sample']) for r in records) != expected:
        raise ValueError('incomplete batch timing census')
    return records


def parse_batch_tokens(stdout, context, block):
    """tokens: <sequences> <arrangement> <differing> lines: how many sequences an arm's untimed step
    selected differently from the control's. Exact arrangements are checked to differ in none."""
    records = []
    for line in stdout.splitlines():
        if line.startswith('tokens: '):
            sequences, arrangement, differing = map(int, line.split()[1:])
            if not 0 <= differing <= sequences:
                raise ValueError('invalid token record')
            records.append(dict(context=context, sequences=sequences, block=block, arrangement=arrangement,
                                differing=differing))
    return records


def parse_accuracy(stdout):
    """ACCURACY <arrangement> <rows> <outputs> <inputs> <count> <above half> <above one> <above two> <worst> lines."""
    if 'device: Apple M4 Pro\napi: metal\n' not in stdout or stdout.count('ACCURACY_COMPLETE') != 1:
        raise ValueError('missing measured device or accuracy completion')
    records = []
    for line in stdout.splitlines():
        if line.startswith('ACCURACY '):
            fields = line.split()[1:]
            arrangement, rows, outputs, inputs, count, half, one, two = map(int, fields[:8])
            worst = float(fields[8])
            if count != rows*outputs or not 0 <= two <= one <= half <= count or not worst >= 0:
                raise ValueError('invalid accuracy record')
            records.append(dict(arrangement=arrangement, rows=rows, outputs=outputs, inputs=inputs, count=count,
                                above_half_ulp=half, above_one_ulp=one, above_two_ulps=two, worst_ulps=worst))
    expected = Counter((a, *shape) for a in (contract.BATCH_REORDERED_CONTROL,) + contract.BATCH_REORDERED_ARRANGEMENTS
                       for shape in contract.BATCH_ACCURACY_SHAPES)
    if Counter((r['arrangement'], r['rows'], r['outputs'], r['inputs']) for r in records) != expected:
        raise ValueError('incomplete accuracy census')
    return records


def accuracy_gate(census):
    """Per candidate: its worst error in every shape does not exceed arrangement 5's."""
    worst = {(r['arrangement'], r['outputs'], r['inputs']): r['worst_ulps'] for r in census}
    control = contract.BATCH_REORDERED_CONTROL
    return {a: all(worst[a, n, k] <= worst[control, n, k] for _, n, k in contract.BATCH_ACCURACY_SHAPES)
            for a in contract.BATCH_REORDERED_ARRANGEMENTS}


def batch_collect(directory, output, study='size', argument=None):
    spec = batch_study(study)
    argument = argument or spec['argument']
    comparisons, observed = batch_comparisons(argument)
    ensure_record_location(output)
    receipt = verify_build(directory)
    if receipt['declaration'] != spec['declaration']:
        raise ValueError(f'{study} collection requires a {study} build')
    output.mkdir(parents=True, exist_ok=False)
    args = receipt['assets']
    contexts = list(contract.BATCH_CONTEXTS) + [0]
    samples, blocks, tokens = [], [], []
    extra = {}
    if argument == 'reordered':
        extra['accuracy'] = parse_accuracy(execute([directory/'model', 'accuracy'], output/'accuracy.log'))
    # 2d adds one prefill process per block, after the decode contexts or, in reversed blocks, before them.
    paged = study in PAGED_STUDIES
    if paged:
        extra.update(prefill_samples=[], prefill_token_differences=[])
    for block in range(4):
        before = conditions()
        reverse = block in (1, 2)
        processes = contexts + (['prefill'] if paged else [])
        for context in (list(reversed(processes)) if reverse else processes):
            if context == 'prefill':
                stdout = execute([directory/'model', 'paged-prefill', args['prepared'], args['tables'], int(reverse),
                                  argument], output/f'prefill-b{block}.log', timeout=3600)
                records, differences = parse_paged_prefill(stdout, block, comparisons, argument)
                extra['prefill_samples'].extend(records)
                extra['prefill_token_differences'].extend(differences)
                continue
            stdout = execute([directory/'model', 'batch', args['prepared'], args['tables'], context, int(reverse),
                              argument], output/f'c{context}-b{block}.log', timeout=3600 if paged else 600)
            samples.extend(parse_batch_samples(stdout, context, block, comparisons, observed, argument))
            tokens.extend(parse_batch_tokens(stdout, context, block))
        blocks.append(dict(block=block, before=before, after=conditions()))
        print(f'Completed {argument} block {block+1}/4', flush=True)
    if verify_build(directory) != receipt:
        raise ValueError('build changed during collection')
    write(output/'timings.json', dict(build=receipt, argument=argument, blocks=blocks, samples=samples,
                                      token_differences=tokens, **extra))


def screen_decision(study, timing):
    """The frozen selection of 1d (study projections), 1e (study reordered) or 2d (study paged) from its screen."""
    if study in PAGED_STUDIES:
        return paged_decision(paged_summarize(timing, contract.PAGED_CANDIDATES))
    if study == 'projections':
        return projection_decision(projection_summarize(timing['samples'], contract.BATCH_PROJECTION_ARRANGEMENTS))
    return projection_decision(projection_summarize(timing['samples'], contract.BATCH_REORDERED_ARRANGEMENTS),
                               regression_from=1, gain_from=16, eligible=accuracy_gate(timing['accuracy']))


def batch_confirm(directory, screen, output, study='projections'):
    """A fresh four-block run of the screen's selected arrangement against the screen's control."""
    timing = json.loads((screen/'timings.json').read_text())
    if timing.get('argument') != batch_study(study)['argument'] or timing['build'] != verify_build(directory):
        raise ValueError(f'confirmation requires the {study} screen of this build')
    decision = screen_decision(study, timing)
    if decision['selected'] is None:
        raise ValueError('no arrangement qualified, so there is nothing to confirm')
    print(f"Confirming {'layout' if study in PAGED_STUDIES else 'arrangement'} {decision['selected']}", flush=True)
    prefix = dict(projections='confirm:', reordered='reordered-confirm:', paged='paged-confirm:',
                  **{'paged-loop': 'paged-confirm:'})[study]
    batch_collect(directory, output, study, f"{prefix}{decision['selected']}")


def batch_capture(directory, output, study='size'):
    spec = batch_study(study)
    ensure_record_location(output)
    from .capture_trace import capture_trace
    receipt = verify_build(directory)
    if receipt['declaration'] != spec['declaration']:
        raise ValueError(f'{study} capture requires a {study} build')
    output.mkdir(parents=True, exist_ok=False)
    for repeat in range(2):
        for context, sequences, arrangement, name, arm in (spec['traces'] if repeat == 0 else reversed(spec['traces'])):
            target = output/f'{name}-r{repeat}'
            target.mkdir()
            before = conditions()
            capture_trace(profile_binary=directory/name, output_trace=target/'raw.trace',
                          receipt_path=target/'capture.json', time_limit='30s')
            write(target/'conditions.json', dict(before=before, after=conditions()))
            (target/'profile.provenance.json').write_bytes((directory/(name+'.provenance.json')).read_bytes())
            print(f'Captured {name}, repeat {repeat+1}/2', flush=True)
    if verify_build(directory) != receipt:
        raise ValueError('build changed during profiling')


def _outcome(ratios, noise):
    middle = stats.median(ratios)
    if all(r < 1 for r in ratios) and 1-middle > noise:
        return 'faster'
    if all(r > 1 for r in ratios) and middle-1 > noise:
        return 'slower'
    return 'inconclusive'


def batch_summarize(samples):
    expected = Counter((c, b, block, comparison, arm, sample) for c, b in contract.batch_workloads()
                       for block in range(4) for comparison in range(4) for arm in range(2) for sample in range(10))
    keys = ('context','sequences','block','comparison','arm','sample')
    if Counter(tuple(r[k] for k in keys) for r in samples) != expected:
        raise ValueError('incomplete batch-size timing census')
    for row in samples:
        marks = row['marks']
        if (row['elapsed_ns'] <= 0 or bool(marks) != (row['comparison'] == 3 and row['arm'] == 1)
                or (marks and (len(marks) != 10 or marks != sorted(marks) or marks[0] < 0 or marks[-1] > row['elapsed_ns']))):
            raise ValueError('invalid retained host observation')
    groups = defaultdict(list)
    for r in samples:
        groups[r['context'], r['sequences'], r['block'], r['comparison'], r['arm']].append(r)
    result = []
    for context, sequences in contract.batch_workloads():
        medians = {(b, c, a): stats.median(r['elapsed_ns'] for r in groups[context, sequences, b, c, a])
                   for b in range(4) for c in range(4) for a in range(2)}
        control = [medians[b, 0, 0]/1e6 for b in range(4)]
        step_ms = stats.median(control)
        calibration = [medians[b, 0, 1]/medians[b, 0, 0] for b in range(4)]
        noise = max(.05, max(abs(r-1) for r in calibration))
        tiles = {}
        for comparison, tile in ((1, 8), (2, 16)):
            ratios = [medians[b, comparison, 1]/medians[b, comparison, 0] for b in range(4)]
            tiles[str(tile)] = dict(step_ms=stats.median(medians[b, comparison, 1] for b in range(4))/1e6,
                                    block_ratios=ratios, median_ratio=stats.median(ratios), outcome=_outcome(ratios, noise))
        phases = defaultdict(list)
        for block in range(4):
            records = groups[context, sequences, block, 3, 1]
            for i, label in enumerate(HOST_PHASES):
                phases[label].append(stats.median(r['marks'][i+1]-r['marks'][i] for r in records)/1e6)
        result.append(dict(context=context, sequences=sequences, step_ms=step_ms, step_block_ms=control,
                           tokens_per_second=sequences*1000/step_ms, calibration_ratios=calibration, noise_floor=noise,
                           tiles=tiles, observation_ratios=[medians[b, 3, 1]/medians[b, 3, 0] for b in range(4)],
                           host_phase_ms={k: stats.median(v) for k, v in phases.items()}))
    single = {r['context']: r['tokens_per_second'] for r in result if r['sequences'] == 1}
    for r in result:
        r['throughput_vs_one'] = r['tokens_per_second']/single[r['context']] if r['context'] in single else None
    return result


def projection_summarize(samples, arrangements):
    """Per workload: arrangement 0's step and calibration, then each candidate's paired ratios and outcome.

    Comparison 0 pairs arrangement 0 with itself; comparison i pairs it with arrangements[i-1].
    """
    count = 1+len(arrangements)
    expected = Counter((c, b, block, comparison, arm, sample) for c, b in contract.batch_workloads()
                       for block in range(4) for comparison in range(count) for arm in range(2) for sample in range(10))
    keys = ('context','sequences','block','comparison','arm','sample')
    if Counter(tuple(r[k] for k in keys) for r in samples) != expected:
        raise ValueError('incomplete projection timing census')
    if any(r['elapsed_ns'] <= 0 or r['marks'] for r in samples):
        raise ValueError('invalid projection timing sample')
    groups = defaultdict(list)
    for r in samples:
        groups[r['context'], r['sequences'], r['block'], r['comparison'], r['arm']].append(r['elapsed_ns'])
    result = []
    for context, sequences in contract.batch_workloads():
        medians = {(b, c, a): stats.median(groups[context, sequences, b, c, a])
                   for b in range(4) for c in range(count) for a in range(2)}
        control = [medians[b, 0, 0]/1e6 for b in range(4)]
        step_ms = stats.median(control)
        calibration = [medians[b, 0, 1]/medians[b, 0, 0] for b in range(4)]
        noise = max(.05, max(abs(r-1) for r in calibration))
        candidates = {}
        for comparison, arrangement in enumerate(arrangements, 1):
            ratios = [medians[b, comparison, 1]/medians[b, comparison, 0] for b in range(4)]
            candidate = stats.median(medians[b, comparison, 1] for b in range(4))/1e6
            candidates[str(arrangement)] = dict(step_ms=candidate, tokens_per_second=sequences*1000/candidate,
                                                block_ratios=ratios, median_ratio=stats.median(ratios),
                                                outcome=_outcome(ratios, noise))
        result.append(dict(context=context, sequences=sequences, step_ms=step_ms, step_block_ms=control,
                           tokens_per_second=sequences*1000/step_ms, calibration_ratios=calibration,
                           noise_floor=noise, arrangements=candidates))
    return result


def projection_decision(summary, regression_from=2, gain_from=4, eligible=None):
    """The frozen rule: qualify every arrangement, then select at most one.

    Qualifying needs eligibility (1e's accuracy gate), no regression from B =
    regression_from and a gain in every workload from B = gain_from. The lowest
    worst-case median ratio from gain_from wins, then the lowest mean ratio, then
    the lower ID. 1d uses 2 and 4; 1e uses 1 and 16.
    """
    qualified = []
    for arrangement in sorted(int(a) for a in summary[0]['arrangements']):
        cells = [(r['sequences'], r['arrangements'][str(arrangement)]) for r in summary]
        if ((eligible is None or eligible[arrangement])
                and all(x['outcome'] != 'slower' for b, x in cells if b >= regression_from)
                and all(x['outcome'] == 'faster' for b, x in cells if b >= gain_from)):
            ratios = [x['median_ratio'] for b, x in cells if b >= gain_from]
            qualified.append((max(ratios), stats.mean(ratios), arrangement))
    qualified.sort()
    return dict(qualified=[dict(arrangement=a, worst_median_ratio=w, mean_median_ratio=m) for w, m, a in qualified],
                selected=qualified[0][2] if qualified else None)


def batch_archive(timings, traces, output, study='size', confirmation=None, diagnostics=None):
    spec = batch_study(study)
    timing = json.loads((timings/'timings.json').read_text())
    record = dict(kind=spec['kind'], timing=timing)
    if study == 'size':
        batch_summarize(timing['samples'])
    else:
        decision = screen_decision(study, timing)
        if (decision['selected'] is None) != (confirmation is None):
            raise ValueError('the archive holds a confirmation exactly when an arrangement is selected')
        record['confirmation'] = json.loads((confirmation/'timings.json').read_text()) if confirmation else None
        if study == 'reordered':
            needed = decision['selected'] is not None and reordered_arrangement(decision['selected'])
            if needed != (diagnostics is not None):
                raise ValueError('the reordered archive holds diagnostics exactly when a reordered arrangement is selected')
            record['diagnostics'] = json.loads((diagnostics/'diagnostics.json').read_text()) if diagnostics else None
    captures = []
    for context, sequences, arrangement, name, arm in spec['traces']:
        for repeat in range(2):
            target = traces/f'{name}-r{repeat}'
            if not (target/'submissions.xml').exists():
                export_trace(target)
            capture = curate(target, context, repeat)
            capture.update(sequences=sequences, **arm)
            captures.append(capture)
    record.update(captures=captures,
                  analysis_source_sha256={str(p.relative_to(repository_root())):sha(p) for p in
                    [Path(__file__).resolve(), Path(__file__).with_name('analyze_trace.py').resolve(),
                     Path(__file__).with_name('model_contract.py').resolve()]},
                  rejected_captures=json.loads((traces/'rejections.json').read_text()) if (traces/'rejections.json').exists() else [])
    def scrub(value):
        if isinstance(value, dict):
            return {k: ('<verified-local-asset>' if k in ('prepared','tables') else scrub(v)) for k,v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value
    raw = json.dumps(scrub(record), separators=(',', ':'), allow_nan=False).encode()
    packed = gzip.compress(raw, mtime=0)
    output.mkdir(parents=True, exist_ok=True)
    (output/(spec['stem']+'.json.gz')).write_bytes(packed)
    write(output/(spec['stem']+'.json'), dict(kind=spec['kind'], sha256=hashlib.sha256(packed).hexdigest(),
                                             uncompressed_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(packed)))
    dict(size=batch_replay, projections=projection_replay, reordered=reordered_replay, paged=paged_replay,
         **{'paged-loop': lambda directory: paged_replay(directory, 'paged-loop')})[study](output)


def batch_capture_totals(capture, build_record):
    """Validate one retained trace against the frozen build; per-step GPU totals."""
    provenance = capture['provenance']
    contract.configuration(provenance)
    arm = next(key for key in ('row_tile', 'arrangement', 'layout') if key in capture)
    suffix = dict(row_tile=str(capture.get('row_tile')), arrangement=f"a{capture.get('arrangement')}",
                  layout=f"l{capture.get('layout')}")[arm]
    name = f"{'paged' if arm == 'layout' else 'batch'}-profile-{capture['prefix']}-{capture['sequences']}-{suffix}"
    if (provenance['repository'] != build_record['source']['repository']
            or provenance['source_sha256'] != build_record['source']['sources']
            or provenance['assets'] != {k:v for k,v in build_record['assets'].items() if k.endswith('_sha256')}
            or provenance['binary'] != build_record['binaries'][name]
            or (provenance['profile_rows'], provenance.get(arm)) != (capture['sequences'], capture[arm])):
        raise ValueError('trace differs from frozen batch-size build')
    canonical = (json.dumps(provenance, indent=2, allow_nan=False)+'\n').encode()
    if hashlib.sha256(canonical).hexdigest() != capture['analysis']['capture_identity']['provenance']['sha256']:
        raise ValueError('retained provenance differs from captured build receipt')
    rows = capture['samples']
    stages = contract.command_stages(*contract.options(provenance['implementation']))
    if Counter((r['iteration'],r['dispatch']) for r in rows) != Counter((i,d) for i in range(8) for d in range(len(stages))):
        raise ValueError('incomplete captured dispatch census')
    for row in rows:
        segments = row['active_intervals']
        if ((row['layer'],row['stage'],row['kind']) != stages[row['dispatch']] or row['duration_ns'] <= 0
                or len(segments) != row['segments'] or sum(d for _,d in segments) != row['duration_ns']
                or segments[0][0] != row['start_ns'] or sum(segments[-1]) != row['end_ns']
                or any(d <= 0 for _,d in segments) or any(a+d > b for (a,d),(b,_) in zip(segments,segments[1:]))):
            raise ValueError('invalid dispatch timing, stage or active fragments')
    totals = defaultdict(list)
    for iteration in range(8):
        step = [r for r in rows if r['iteration'] == iteration]
        for stage in sorted({r['stage'] for r in step}):
            totals[stage].append(sum(r['duration_ns'] for r in step if r['stage'] == stage)/1e6)
        totals['GPU active total'].append(sum(r['duration_ns'] for r in step)/1e6)
        totals['GPU enclosing span'].append((max(r['end_ns'] for r in step)-min(r['start_ns'] for r in step))/1e6)
        totals['Metal submission intervals'].append(sum(r['submission_duration_ns'] for r in step)/1e6)
    return {stage: stats.median(values) for stage, values in totals.items()}


def batch_replay(directory):
    manifest = json.loads((directory/(BATCH_STEM+'.json')).read_text())
    packed = (directory/(BATCH_STEM+'.json.gz')).read_bytes()
    raw = gzip.decompress(packed)
    if (hashlib.sha256(packed).hexdigest() != manifest['sha256']
            or hashlib.sha256(raw).hexdigest() != manifest['uncompressed_sha256']):
        raise ValueError('batch-size archive hash mismatch')
    record = json.loads(raw)
    if record['kind'] != BATCH_KIND or manifest['kind'] != BATCH_KIND:
        raise ValueError('not a batch-size archive')
    summary = batch_summarize(record['timing']['samples'])
    build_record = record['timing']['build']
    if build_record['declaration'] != contract.BATCH_DECLARATION:
        raise ValueError('batch-size declaration changed')
    if [b['block'] for b in record['timing']['blocks']] != list(range(4)):
        raise ValueError('incomplete batch-size block conditions')
    expected = Counter((c, b, t, r) for c, b, t in contract.BATCH_TRACES for r in range(2))
    if Counter((c['prefix'], c['sequences'], c['row_tile'], c['repeat']) for c in record['captures']) != expected:
        raise ValueError('incomplete batch-size trace census')
    for block in [*record['timing']['blocks'], *[c['conditions'] for c in record['captures']]]:
        for side in ('before', 'after'):
            require_ac(block[side])
            require_nominal_thermal_state(block[side])
            if block[side]['power_mode_raw'] != '0':
                raise ValueError('batch-size power mode changed')
    # A rejected attempt must be a trace of a declared workload with the frozen binary.
    for rejected in record['rejected_captures']:
        workload = (rejected['prefix'], rejected['sequences'], rejected['row_tile'])
        binary = rejected['receipt']['profile']['binary']
        if (workload not in contract.BATCH_TRACES or rejected['status'] != 'rejected by analysis'
                or {k: binary[k] for k in ('sha256', 'bytes')} != build_record['binaries']['batch-profile-%d-%d-%d' % workload]):
            raise ValueError('rejected capture is not an attempt of the frozen batch-size build')
    gpu = []
    for capture in record['captures']:
        for stage, value in batch_capture_totals(capture, build_record).items():
            gpu.append(dict(context=capture['prefix'], sequences=capture['sequences'], row_tile=capture['row_tile'],
                            repeat=capture['repeat'], stage=stage, median_ms=value))
    write(directory/(BATCH_STEM+'-summary.json'), dict(timing=summary, gpu=gpu))
    for r in summary:
        print(f"context {r['context']:4d} sequences {r['sequences']:2d}: {r['step_ms']:8.2f} ms/step, "
              f"{r['tokens_per_second']:7.1f} tokens/s, tile 8 {r['tiles']['8']['outcome']}, tile 16 {r['tiles']['16']['outcome']}")
    return record


def projection_replay(directory):
    """Verify 1d's archive and regenerate its summary: screen, decision, confirmation and traces."""
    manifest = json.loads((directory/(PROJECTION_BATCH_STEM+'.json')).read_text())
    packed = (directory/(PROJECTION_BATCH_STEM+'.json.gz')).read_bytes()
    raw = gzip.decompress(packed)
    if (hashlib.sha256(packed).hexdigest() != manifest['sha256']
            or hashlib.sha256(raw).hexdigest() != manifest['uncompressed_sha256']):
        raise ValueError('projection archive hash mismatch')
    record = json.loads(raw)
    if record['kind'] != PROJECTION_BATCH_KIND or manifest['kind'] != PROJECTION_BATCH_KIND:
        raise ValueError('not a projection archive')
    screen = record['timing']
    build_record = screen['build']
    if build_record['declaration'] != contract.BATCH_PROJECTION_DECLARATION or screen.get('argument') != 'projections':
        raise ValueError('projection declaration changed')
    summary = projection_summarize(screen['samples'], contract.BATCH_PROJECTION_ARRANGEMENTS)
    decision = projection_decision(summary)
    timings, confirmation, confirmed = [screen], None, None
    if decision['selected'] is None:
        if record['confirmation'] is not None:
            raise ValueError('confirmation without a selected arrangement')
    else:
        run = record['confirmation']
        if run is None or run['build'] != build_record or run.get('argument') != f"confirm:{decision['selected']}":
            raise ValueError('missing or mismatched confirmation of the selected arrangement')
        confirmation = projection_summarize(run['samples'], (decision['selected'],))
        confirmed = projection_decision(confirmation)['selected'] == decision['selected']
        timings.append(run)
    if any([b['block'] for b in timing['blocks']] != list(range(4)) for timing in timings):
        raise ValueError('incomplete projection block conditions')
    expected = Counter((c, b, a, r) for c, b, a in contract.BATCH_PROJECTION_TRACES for r in range(2))
    if Counter((c['prefix'], c['sequences'], c['arrangement'], c['repeat']) for c in record['captures']) != expected:
        raise ValueError('incomplete projection trace census')
    for block in [*[b for timing in timings for b in timing['blocks']], *[c['conditions'] for c in record['captures']]]:
        for side in ('before', 'after'):
            require_ac(block[side])
            require_nominal_thermal_state(block[side])
            if block[side]['power_mode_raw'] != '0':
                raise ValueError('projection power mode changed')
    for rejected in record['rejected_captures']:
        workload = (rejected['prefix'], rejected['sequences'], rejected['arrangement'])
        binary = rejected['receipt']['profile']['binary']
        if (workload not in contract.BATCH_PROJECTION_TRACES or rejected['status'] != 'rejected by analysis'
                or {k: binary[k] for k in ('sha256', 'bytes')} != build_record['binaries']['batch-profile-%d-%d-a%d' % workload]):
            raise ValueError('rejected capture is not an attempt of the frozen projection build')
    gpu = []
    for capture in record['captures']:
        for stage, value in batch_capture_totals(capture, build_record).items():
            gpu.append(dict(context=capture['prefix'], sequences=capture['sequences'], arrangement=capture['arrangement'],
                            repeat=capture['repeat'], stage=stage, median_ms=value))
    write(directory/(PROJECTION_BATCH_STEM+'-summary.json'),
          dict(timing=summary, decision=decision, confirmation=confirmation, confirmed=confirmed, gpu=gpu))
    for r in summary:
        print(f"context {r['context']:4d} sequences {r['sequences']:2d}: {r['step_ms']:8.2f} ms/step; "
              + ', '.join(f"{a}: {x['median_ratio']:.3f} {x['outcome']}" for a, x in r['arrangements'].items()))
    print('qualified:', [q['arrangement'] for q in decision['qualified']], 'selected:', decision['selected'],
          'confirmed:', confirmed)
    return record


def reordered_arrangement(arrangement):
    """Arrangements 8-10 sum in another order than the one-row kernel (kernels/linear.mojo)."""
    return 8 <= arrangement <= 10


def diagnostic_stop(diagnostics):
    """1e's stop rule: the selected arrangement agrees with HF on more than one fewer decode choice than
    arrangement 5, or its largest KL divergence more than doubles."""
    exact = diagnostics['hf'][str(contract.BATCH_REORDERED_CONTROL)]
    candidate = diagnostics['hf'][str(diagnostics['selected'])]
    return candidate['agree'] < exact['agree']-1 or candidate['max_kl_nats'] > 2*exact['max_kl_nats']


def hf_choices(result):
    """Every decode call's next-token comparison with HF in one diagnose result."""
    choices = [dict(case=r['case'], call=r['call'], token=r['token'], reference_token=r['reference_token'],
                    kl_nats=r['kl_nats'], total_variation=r['total_variation'], reference_margin=r['reference_margin'])
               for r in result['diagnostics'] if r['comparison'] == 'hf_same_history' and r['stage'] == 'logits'
               and r['mode'] == 'scheduled' and r['configuration'] == 26]
    if not choices:
        raise ValueError('no decode choices in the diagnostics')
    return choices


def hf_summary(choices):
    return dict(choices=choices, decode_choices=len(choices), agree=sum(c['token'] == c['reference_token'] for c in choices),
                max_kl_nats=max(c['kl_nats'] for c in choices), max_total_variation=max(c['total_variation'] for c in choices))


def reordered_diagnostics(screen, confirmation, output, prepared=None):
    """1e's model-level diagnostics of the confirmed reordered arrangement against arrangement 5.

    Builds the model driver and generator in both arrangements, runs the
    teacher-forced decode comparison, then HF same-history comparisons on the
    specification's decode cases and on each arrangement's own generations.
    Only compact per-choice records are kept.
    """
    from ..validation import model as validation
    timing = json.loads((screen/'timings.json').read_text())
    selected = screen_decision('reordered', timing)['selected']
    run = json.loads((confirmation/'timings.json').read_text())
    if selected is None or not reordered_arrangement(selected) or run.get('argument') != f'reordered-confirm:{selected}':
        raise ValueError('diagnostics follow the confirmation of a selected reordered arrangement')
    gate = accuracy_gate(timing['accuracy'])
    if projection_decision(projection_summarize(run['samples'], (selected,)), 1, 16, {selected: gate[selected]})['selected'] != selected:
        raise ValueError('the selected arrangement was not confirmed')
    ensure_record_location(output)
    output.mkdir(parents=True, exist_ok=False)
    control = contract.BATCH_REORDERED_CONTROL
    for arrangement in (control, selected):
        validation.build(output/f'model-{arrangement}', projection=arrangement)
        validation.build(output/f'generator-{arrangement}', generation=True, projection=arrangement)
    validation.decode_comparison(output/f'model-{control}', output/f'model-{selected}', output/'decode-comparison.json',
                                 prepared)
    def reference(specification, target):
        # uv is not installed into the environment; the reference runs as its own locked script, as in assets.py.
        # It runs on CPU and can take longer than the default limit.
        execute(['uv', 'run', '--locked', '--script', 'tests/fixtures/model_reference.py', 'diagnose',
                 '--specification', specification, '--output', target], output/f'{target.name}.log', timeout=3*3600)
    validation.runtime_specification(output/'decode-specification.json', decode_only=True)
    reference(output/'decode-specification.json', output/'decode-reference')
    hf = {}
    for arrangement in (control, selected):
        validation.generation_study(output/f'generator-{arrangement}', output/f'generation-{arrangement}', prepared)
        validation.runtime_specification(output/f'history-specification-{arrangement}.json',
                                         generations=output/f'generation-{arrangement}'/'result.json')
        reference(output/f'history-specification-{arrangement}.json', output/f'history-reference-{arrangement}')
        choices = []
        for name, source in (('decode', output/'decode-reference'), ('history', output/f'history-reference-{arrangement}')):
            target = output/f'{name}-diagnostics-{arrangement}'
            validation.diagnose(output/f'model-{arrangement}', source, target, prepared, policy='fast')
            choices += [dict(c, source=name) for c in hf_choices(json.loads((target/'result.json').read_text()))]
        hf[str(arrangement)] = hf_summary(choices)
    diagnostics = dict(kind='qwen-batch-reordered-diagnostics-v1', control=control, selected=selected,
                       decode_comparison=json.loads((output/'decode-comparison.json').read_text()), hf=hf)
    diagnostics['stop'] = diagnostic_stop(diagnostics)
    write(output/'diagnostics.json', diagnostics)
    print('decode choices agreeing with HF:', {a: f"{s['agree']}/{s['decode_choices']}" for a, s in hf.items()},
          'stop:', diagnostics['stop'])


def reordered_replay(directory):
    """Verify 1e's archive: screen, accuracy gate, decision, confirmation, diagnostics and traces."""
    manifest = json.loads((directory/(REORDERED_BATCH_STEM+'.json')).read_text())
    packed = (directory/(REORDERED_BATCH_STEM+'.json.gz')).read_bytes()
    raw = gzip.decompress(packed)
    if (hashlib.sha256(packed).hexdigest() != manifest['sha256']
            or hashlib.sha256(raw).hexdigest() != manifest['uncompressed_sha256']):
        raise ValueError('reordered archive hash mismatch')
    record = json.loads(raw)
    if record['kind'] != REORDERED_BATCH_KIND or manifest['kind'] != REORDERED_BATCH_KIND:
        raise ValueError('not a reordered archive')
    screen = record['timing']
    build_record = screen['build']
    if build_record['declaration'] != contract.BATCH_REORDERED_DECLARATION or screen.get('argument') != 'reordered':
        raise ValueError('reordered declaration changed')
    census = screen['accuracy']
    expected = Counter((a, *shape) for a in (contract.BATCH_REORDERED_CONTROL,) + contract.BATCH_REORDERED_ARRANGEMENTS
                       for shape in contract.BATCH_ACCURACY_SHAPES)
    if (Counter((r['arrangement'], r['rows'], r['outputs'], r['inputs']) for r in census) != expected
            or any(not 0 <= r['above_two_ulps'] <= r['above_one_ulp'] <= r['above_half_ulp'] <= r['count']
                   or r['count'] != r['rows']*r['outputs'] or not r['worst_ulps'] >= 0 for r in census)):
        raise ValueError('incomplete or invalid accuracy census')
    gate = accuracy_gate(census)
    summary = projection_summarize(screen['samples'], contract.BATCH_REORDERED_ARRANGEMENTS)
    decision = projection_decision(summary, 1, 16, gate)
    tokens = Counter((r['context'], r['sequences'], r['block'], r['arrangement']) for r in screen['token_differences'])
    arms = (contract.BATCH_REORDERED_CONTROL,) + contract.BATCH_REORDERED_ARRANGEMENTS
    if tokens != Counter((c, b, block, a) for c, b in contract.batch_workloads() for block in range(4) for a in arms):
        raise ValueError('incomplete token census')
    if any(r['differing'] for r in screen['token_differences'] if not reordered_arrangement(r['arrangement'])):
        raise ValueError('an exact arrangement changed a token')
    selected, timings, confirmation, confirmed, diagnostics = decision['selected'], [screen], None, None, None
    if selected is None:
        if record['confirmation'] is not None or record['diagnostics'] is not None:
            raise ValueError('confirmation or diagnostics without a selected arrangement')
    else:
        run = record['confirmation']
        if run is None or run['build'] != build_record or run.get('argument') != f'reordered-confirm:{selected}':
            raise ValueError('missing or mismatched confirmation of the selected arrangement')
        confirmation = projection_summarize(run['samples'], (selected,))
        confirmed = projection_decision(confirmation, 1, 16, {selected: gate[selected]})['selected'] == selected
        timings.append(run)
        diagnostics = record['diagnostics']
        if (diagnostics is not None) != reordered_arrangement(selected):
            raise ValueError('diagnostics belong to a reordered selection')
        if diagnostics is not None:
            comparison = diagnostics['decode_comparison']
            if (diagnostics['selected'] != selected or diagnostics['control'] != contract.BATCH_REORDERED_CONTROL
                    or comparison['projections'] != dict(reference=contract.BATCH_REORDERED_CONTROL, candidate=selected)
                    or set(diagnostics['hf']) != {str(contract.BATCH_REORDERED_CONTROL), str(selected)}
                    or any(hf_summary(s['choices']) != s for s in diagnostics['hf'].values())
                    or diagnostics['stop'] != diagnostic_stop(diagnostics)):
                raise ValueError('inconsistent model-level diagnostics')
    if any([b['block'] for b in timing['blocks']] != list(range(4)) for timing in timings):
        raise ValueError('incomplete reordered block conditions')
    expected = Counter((c, b, a, r) for c, b, a in contract.BATCH_REORDERED_TRACES for r in range(2))
    if Counter((c['prefix'], c['sequences'], c['arrangement'], c['repeat']) for c in record['captures']) != expected:
        raise ValueError('incomplete reordered trace census')
    for block in [*[b for timing in timings for b in timing['blocks']], *[c['conditions'] for c in record['captures']]]:
        for side in ('before', 'after'):
            require_ac(block[side])
            require_nominal_thermal_state(block[side])
            if block[side]['power_mode_raw'] != '0':
                raise ValueError('reordered power mode changed')
    for rejected in record['rejected_captures']:
        workload = (rejected['prefix'], rejected['sequences'], rejected['arrangement'])
        binary = rejected['receipt']['profile']['binary']
        capture = rejected['receipt']['capture']
        # An attempt is set aside by the coverage analysis, or because xctrace failed and its receipt says so.
        failed = rejected['status'] == 'failed during capture' and capture['status'] == 'invalid' and bool(capture['failures'])
        if (workload not in contract.BATCH_REORDERED_TRACES or (rejected['status'] != 'rejected by analysis' and not failed)
                or {k: binary[k] for k in ('sha256', 'bytes')} != build_record['binaries']['batch-profile-%d-%d-a%d' % workload]):
            raise ValueError('rejected capture is not an attempt of the frozen reordered build')
    gpu = []
    for capture in record['captures']:
        for stage, value in batch_capture_totals(capture, build_record).items():
            gpu.append(dict(context=capture['prefix'], sequences=capture['sequences'], arrangement=capture['arrangement'],
                            repeat=capture['repeat'], stage=stage, median_ms=value))
    diagnostic_summary = None if diagnostics is None else dict(
        selected=diagnostics['selected'], stop=diagnostics['stop'],
        decode_comparison=dict(tokens_agree=diagnostics['decode_comparison']['tokens_agree'],
                               steps=diagnostics['decode_comparison']['steps'],
                               summary=diagnostics['decode_comparison']['summary']),
        hf={a: {k: v for k, v in s.items() if k != 'choices'} for a, s in diagnostics['hf'].items()})
    write(directory/(REORDERED_BATCH_STEM+'-summary.json'),
          dict(timing=summary, accuracy=census, accuracy_gate={str(a): v for a, v in gate.items()}, decision=decision,
               confirmation=confirmation, confirmed=confirmed, diagnostics=diagnostic_summary,
               token_differences=screen['token_differences'], gpu=gpu))
    for r in summary:
        print(f"context {r['context']:4d} sequences {r['sequences']:2d}: {r['step_ms']:8.2f} ms/step; "
              + ', '.join(f"{a}: {x['median_ratio']:.3f} {x['outcome']}" for a, x in r['arrangements'].items()))
    print('accuracy gate:', gate, 'qualified:', [q['arrangement'] for q in decision['qualified']],
          'selected:', selected, 'confirmed:', confirmed, 'stop:', None if diagnostics is None else diagnostics['stop'])
    return record


def addressing_archive(timings, traces, output):
    """1f's compact record: every screen sample, and each accepted trace's per-stage totals, not its intervals."""
    spec = batch_study('addressing')
    timing = json.loads((timings/'timings.json').read_text())
    if timing.get('argument') != spec['argument'] or timing['build']['declaration'] != spec['declaration']:
        raise ValueError('the addressing record requires the addressing screen')
    captures = []
    for context, sequences, arrangement, name, arm in spec['traces']:
        for repeat in range(2):
            target = traces/f'{name}-r{repeat}'
            if not (target/'submissions.xml').exists():
                export_trace(target)
            capture = curate(target, context, repeat)
            capture.update(sequences=sequences, **arm)
            captures.append(dict(prefix=context, sequences=sequences, arrangement=arrangement, repeat=repeat,
                                 conditions=capture['conditions'], binary=capture['provenance']['binary'],
                                 stages=batch_capture_totals(capture, timing['build'])))
    record = dict(kind=spec['kind'], timing=timing, captures=captures,
                  analysis_source_sha256={str(p.relative_to(repository_root())): sha(p) for p in
                    [Path(__file__).resolve(), Path(__file__).with_name('analyze_trace.py').resolve(),
                     Path(__file__).with_name('model_contract.py').resolve()]},
                  rejected_captures=json.loads((traces/'rejections.json').read_text()) if (traces/'rejections.json').exists() else [])
    def scrub(value):
        if isinstance(value, dict):
            return {k: ('<verified-local-asset>' if k in ('prepared','tables') else scrub(v)) for k,v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value
    raw = json.dumps(scrub(record), separators=(',', ':'), allow_nan=False).encode()
    packed = gzip.compress(raw, mtime=0)
    output.mkdir(parents=True, exist_ok=True)
    (output/(spec['stem']+'.json.gz')).write_bytes(packed)
    write(output/(spec['stem']+'.json'), dict(kind=spec['kind'], sha256=hashlib.sha256(packed).hexdigest(),
                                             uncompressed_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(packed)))
    addressing_replay(output)


def addressing_replay(directory):
    """Verify 1f's compact record and recompute its declared analysis; it selects nothing."""
    manifest = json.loads((directory/(ADDRESSING_BATCH_STEM+'.json')).read_text())
    packed = (directory/(ADDRESSING_BATCH_STEM+'.json.gz')).read_bytes()
    raw = gzip.decompress(packed)
    if (hashlib.sha256(packed).hexdigest() != manifest['sha256']
            or hashlib.sha256(raw).hexdigest() != manifest['uncompressed_sha256']):
        raise ValueError('addressing record hash mismatch')
    record = json.loads(raw)
    if record['kind'] != ADDRESSING_BATCH_KIND or manifest['kind'] != ADDRESSING_BATCH_KIND:
        raise ValueError('not an addressing record')
    timing = record['timing']
    build_record = timing['build']
    if build_record['declaration'] != contract.BATCH_ADDRESSING_DECLARATION or timing.get('argument') != 'addressing':
        raise ValueError('addressing declaration changed')
    if [b['block'] for b in timing['blocks']] != list(range(4)):
        raise ValueError('incomplete addressing block conditions')
    for side in ([b[s] for b in timing['blocks'] for s in ('before', 'after')]
                 + [c['conditions'][s] for c in record['captures'] for s in ('before', 'after')]):
        require_ac(side)
        require_nominal_thermal_state(side)
        if side['power_mode_raw'] != '0':
            raise ValueError('addressing power mode changed')
    arms = (contract.BATCH_ADDRESSING_CONTROL,) + contract.BATCH_ADDRESSING_ARRANGEMENTS
    tokens = Counter((r['context'], r['sequences'], r['block'], r['arrangement']) for r in timing['token_differences'])
    if tokens != Counter((c, b, block, a) for c, b in contract.batch_workloads() for block in range(4) for a in arms):
        raise ValueError('incomplete token census')
    if any(r['differing'] for r in timing['token_differences'] if not reordered_arrangement(r['arrangement'])):
        raise ValueError('an exact arrangement changed a token')
    summary = projection_summarize(timing['samples'], contract.BATCH_ADDRESSING_ARRANGEMENTS)
    # The declared analysis: where arrangement 8 is a gain, the share of it that arrangement 11 reaches.
    workloads = []
    for r in summary:
        raw11, wide = r['arrangements']['11'], r['arrangements']['8']
        share = (1-raw11['median_ratio'])/(1-wide['median_ratio']) if wide['outcome'] == 'faster' else None
        workloads.append(dict(context=r['context'], sequences=r['sequences'], share=share))
    large = [w['share'] for w in workloads if w['sequences'] >= 16 and w['share'] is not None]
    slower = [[r['context'], r['sequences']] for r in summary if r['arrangements']['11']['outcome'] == 'slower']
    expected = Counter((c, b, a, repeat) for c, b, a in contract.BATCH_ADDRESSING_TRACES for repeat in range(2))
    if Counter((c['prefix'], c['sequences'], c['arrangement'], c['repeat']) for c in record['captures']) != expected:
        raise ValueError('incomplete addressing trace census')
    for capture in record['captures']:
        name = f"batch-profile-{capture['prefix']}-{capture['sequences']}-a{capture['arrangement']}"
        if capture['binary'] != build_record['binaries'][name]:
            raise ValueError('addressing trace differs from the frozen build')
    for rejected in record['rejected_captures']:
        workload = (rejected['prefix'], rejected['sequences'], rejected['arrangement'])
        binary = rejected['receipt']['profile']['binary']
        if (workload not in contract.BATCH_ADDRESSING_TRACES or rejected['status'] != 'rejected by analysis'
                or {k: binary[k] for k in ('sha256', 'bytes')} != build_record['binaries']['batch-profile-%d-%d-a%d' % workload]):
            raise ValueError('rejected capture is not an attempt of the frozen addressing build')
    stages = defaultdict(list)
    for capture in record['captures']:
        for stage, value in capture['stages'].items():
            stages[stage, capture['arrangement']].append(value)
    means = {f'{stage}/{a}': stats.mean(values) for (stage, a), values in stages.items()}
    projections = ('packed QKV projection', 'output projection', 'gate projection', 'up projection',
                   'down projection', 'vocabulary projection')
    total = {a: sum(means[f'{s}/{a}'] for s in projections) for a in arms}
    analysis = dict(share_from_16=dict(minimum=min(large), median=stats.median(large), maximum=max(large),
                                       workloads=len(large)),
                    hypothesis=all(s < 1/3 for s in large),
                    consequence=not slower and all(s >= 0.8 for s in large), slower_workloads=slower,
                    projections_ms={str(a): total[a] for a in arms},
                    projection_share=(total[5]-total[11])/(total[5]-total[8]))
    write(directory/(ADDRESSING_BATCH_STEM+'-summary.json'),
          dict(timing=summary, workloads=workloads, analysis=analysis, stage_means_ms=means,
               token_differences=timing['token_differences']))
    for r, w in zip(summary, workloads):
        print(f"context {r['context']:4d} sequences {r['sequences']:2d}: {r['step_ms']:8.2f} ms/step; "
              + ', '.join(f"{a}: {x['median_ratio']:.3f} {x['outcome']}" for a, x in r['arrangements'].items())
              + ('' if w['share'] is None else f"; share {w['share']:.2f}"))
    print('from B = 16, share of arrangement 8\'s gain:', analysis['share_from_16'], 'hypothesis:', analysis['hypothesis'],
          'consequence:', analysis['consequence'], 'projection share:', round(analysis['projection_share'], 3))
    return record


def parse_paged_prefill(stdout, block, comparisons, argument):
    """PREFILL <workload> <comparison> <arm> <sample> <elapsed> records of one block's prefill process,
    and its per-layout token checks. Every chunk must run Fast's declared configuration."""
    if ('device: Apple M4 Pro\napi: metal\n' not in stdout or stdout.count('PREFILL_COMPLETE') != 1
            or f'\nstudy: {argument}\n' not in stdout):
        raise ValueError('missing measured device, study or prefill completion')
    workloads, records, tokens = {}, [], []
    for line in stdout.splitlines():
        if line.startswith('prefill workload: '):
            index, rows, total, configuration = map(int, line.split()[2:])
            workloads[index] = (rows, total, configuration)
        elif line.startswith('prefill tokens: '):
            index, layout, differing = map(int, line.split()[2:])
            tokens.append(dict(block=block, workload=index, layout=layout, differing=differing))
        elif line.startswith('PREFILL '):
            values = list(map(int, line.split()[1:]))
            if len(values) != 5 or values[4] <= 0:
                raise ValueError('invalid prefill timing record')
            workload, comparison, arm, sample, elapsed = values
            records.append(dict(workload=workload, block=block, comparison=comparison, arm=arm, sample=sample,
                                elapsed_ns=elapsed))
    if workloads != dict(enumerate(contract.PAGED_PREFILL_WORKLOADS)):
        raise ValueError('prefill workloads or their configurations changed')
    count = len(contract.PAGED_PREFILL_WORKLOADS)
    expected = Counter((w, c, a, s) for w in range(count) for c in range(comparisons) for a in range(2) for s in range(10))
    if Counter((r['workload'], r['comparison'], r['arm'], r['sample']) for r in records) != expected:
        raise ValueError('incomplete prefill timing census')
    if len(tokens) != count*comparisons or any(t['differing'] for t in tokens):
        raise ValueError('a paged layout changed a prefill token')
    return records, tokens


def _paired_rows(groups, cells, layouts):
    """Per cell: layout 0's median time and calibration, then each layout's paired ratios and outcome.

    groups maps (cell, block, comparison, arm) to elapsed times; comparison 0 pairs layout 0 with
    itself and comparison i pairs it with layouts[i-1].
    """
    rows = []
    for cell in cells:
        medians = {(b, c, a): stats.median(groups[cell, b, c, a])
                   for b in range(4) for c in range(1+len(layouts)) for a in range(2)}
        control = [medians[b, 0, 0]/1e6 for b in range(4)]
        calibration = [medians[b, 0, 1]/medians[b, 0, 0] for b in range(4)]
        noise = max(.05, max(abs(r-1) for r in calibration))
        candidates = {}
        for comparison, layout in enumerate(layouts, 1):
            ratios = [medians[b, comparison, 1]/medians[b, comparison, 0] for b in range(4)]
            candidates[str(layout)] = dict(ms=stats.median(medians[b, comparison, 1] for b in range(4))/1e6,
                                           block_ratios=ratios, median_ratio=stats.median(ratios),
                                           outcome=_outcome(ratios, noise))
        rows.append(dict(ms=stats.median(control), block_ms=control, calibration_ratios=calibration,
                         noise_floor=noise, layouts=candidates))
    return rows


def paged_summarize(timing, layouts):
    """2d's 22 decode and 13 prefill workloads: layout 0's time and calibration, and each layout's ratios."""
    count = 1+len(layouts)
    decode = Counter((c, b, block, comparison, arm, sample) for c, b in contract.batch_workloads()
                     for block in range(4) for comparison in range(count) for arm in range(2) for sample in range(10))
    keys = ('context', 'sequences', 'block', 'comparison', 'arm', 'sample')
    if Counter(tuple(r[k] for k in keys) for r in timing['samples']) != decode:
        raise ValueError('incomplete paged decode timing census')
    if any(r['elapsed_ns'] <= 0 or r['marks'] for r in timing['samples']):
        raise ValueError('invalid paged decode timing sample')
    workloads = range(len(contract.PAGED_PREFILL_WORKLOADS))
    prefill = Counter((w, block, comparison, arm, sample) for w in workloads
                      for block in range(4) for comparison in range(count) for arm in range(2) for sample in range(10))
    keys = ('workload', 'block', 'comparison', 'arm', 'sample')
    if Counter(tuple(r[k] for k in keys) for r in timing['prefill_samples']) != prefill:
        raise ValueError('incomplete paged prefill timing census')
    groups = defaultdict(list)
    for r in timing['samples']:
        groups[(r['context'], r['sequences']), r['block'], r['comparison'], r['arm']].append(r['elapsed_ns'])
    for r in timing['prefill_samples']:
        groups[('prefill', r['workload']), r['block'], r['comparison'], r['arm']].append(r['elapsed_ns'])
    cells = contract.batch_workloads()
    rows = _paired_rows(groups, cells, layouts)
    for (context, sequences), row in zip(cells, rows):
        row.update(context=context, sequences=sequences, tokens_per_second=sequences*1000/row['ms'])
        for x in row['layouts'].values():
            x['tokens_per_second'] = sequences*1000/x['ms']
    chunks = _paired_rows(groups, [('prefill', w) for w in workloads], layouts)
    for (rows_, total, configuration), row in zip(contract.PAGED_PREFILL_WORKLOADS, chunks):
        row.update(rows=rows_, total=total, configuration=configuration)
    return dict(decode=rows, prefill=chunks)


def paged_decision(summary):
    """2d's frozen rule: a layout qualifies with a regression in none of the 35 workloads. The
    smallest qualifying block size is selected, slot-major unless head-major also qualifies and is
    a gain in at least one workload, or head-major when only it qualifies at that size."""
    rows = summary['decode'] + summary['prefill']
    outcomes = {int(layout): [r['layouts'][layout]['outcome'] for r in rows] for layout in rows[0]['layouts']}
    qualified = [layout for layout in sorted(outcomes) if 'slower' not in outcomes[layout]]
    gains = {str(layout): outcomes[layout].count('faster') for layout in sorted(outcomes)}
    if not qualified:
        return dict(qualified=[], selected=None, gains=gains)
    size = min(contract.PAGED_LAYOUTS[layout]['block_size'] for layout in qualified)
    at_size = {contract.PAGED_LAYOUTS[layout]['order']: layout for layout in qualified
               if contract.PAGED_LAYOUTS[layout]['block_size'] == size}
    head = at_size.get('head-major')
    if head is not None and ('slot-major' not in at_size or 'faster' in outcomes[head]):
        selected = head
    else:
        selected = at_size['slot-major']
    return dict(qualified=qualified, selected=selected, gains=gains)


def paged_census(timing, layouts):
    """Every decode step and prefill chunk of every layout selected layout 0's tokens."""
    arms = (contract.PAGED_CONTROL,) + tuple(layouts)
    tokens = Counter((r['context'], r['sequences'], r['block'], r['arrangement']) for r in timing['token_differences'])
    if tokens != Counter((c, b, block, a) for c, b in contract.batch_workloads() for block in range(4) for a in arms):
        raise ValueError('incomplete paged decode token census')
    chunks = Counter((r['workload'], r['block'], r['layout']) for r in timing['prefill_token_differences'])
    if chunks != Counter((w, block, a) for w in range(len(contract.PAGED_PREFILL_WORKLOADS)) for block in range(4)
                         for a in arms):
        raise ValueError('incomplete paged prefill token census')
    if any(r['differing'] for r in timing['token_differences'] + timing['prefill_token_differences']):
        raise ValueError('a paged layout changed a token')


def paged_replay(directory, study='paged'):
    """Verify 2d's archive, or its rerun's, and regenerate its summary: screen, decision, confirmation and traces."""
    spec = batch_study(study)
    manifest = json.loads((directory/(spec['stem']+'.json')).read_text())
    packed = (directory/(spec['stem']+'.json.gz')).read_bytes()
    raw = gzip.decompress(packed)
    if (hashlib.sha256(packed).hexdigest() != manifest['sha256']
            or hashlib.sha256(raw).hexdigest() != manifest['uncompressed_sha256']):
        raise ValueError('paged KV archive hash mismatch')
    record = json.loads(raw)
    if record['kind'] != spec['kind'] or manifest['kind'] != spec['kind']:
        raise ValueError('not a paged KV archive')
    screen = record['timing']
    build_record = screen['build']
    if build_record['declaration'] != spec['declaration'] or screen.get('argument') != 'paged':
        raise ValueError('paged KV declaration changed')
    paged_census(screen, contract.PAGED_CANDIDATES)
    summary = paged_summarize(screen, contract.PAGED_CANDIDATES)
    decision = paged_decision(summary)
    timings, confirmation, confirmed = [screen], None, None
    if decision['selected'] is None:
        if record['confirmation'] is not None:
            raise ValueError('confirmation without a selected layout')
    else:
        run = record['confirmation']
        if run is None or run['build'] != build_record or run.get('argument') != f"paged-confirm:{decision['selected']}":
            raise ValueError('missing or mismatched confirmation of the selected layout')
        paged_census(run, (decision['selected'],))
        confirmation = paged_summarize(run, (decision['selected'],))
        confirmed = paged_decision(confirmation)['selected'] == decision['selected']
        timings.append(run)
    if any([b['block'] for b in timing['blocks']] != list(range(4)) for timing in timings):
        raise ValueError('incomplete paged KV block conditions')
    expected = Counter((c, b, layout, r) for c, b, layout in contract.PAGED_TRACES for r in range(2))
    if Counter((c['prefix'], c['sequences'], c['layout'], c['repeat']) for c in record['captures']) != expected:
        raise ValueError('incomplete paged KV trace census')
    for block in [*[b for timing in timings for b in timing['blocks']], *[c['conditions'] for c in record['captures']]]:
        for side in ('before', 'after'):
            require_ac(block[side])
            require_nominal_thermal_state(block[side])
            if block[side]['power_mode_raw'] != '0':
                raise ValueError('paged KV power mode changed')
    for rejected in record['rejected_captures']:
        workload = (rejected['prefix'], rejected['sequences'], rejected['layout'])
        binary = rejected['receipt']['profile']['binary']
        if (workload not in contract.PAGED_TRACES or rejected['status'] != 'rejected by analysis'
                or {k: binary[k] for k in ('sha256', 'bytes')} != build_record['binaries']['paged-profile-%d-%d-l%d' % workload]):
            raise ValueError('rejected capture is not an attempt of the frozen paged KV build')
    gpu = []
    for capture in record['captures']:
        for stage, value in batch_capture_totals(capture, build_record).items():
            gpu.append(dict(context=capture['prefix'], sequences=capture['sequences'], layout=capture['layout'],
                            repeat=capture['repeat'], stage=stage, median_ms=value))
    write(directory/(spec['stem']+'-summary.json'),
          dict(timing=summary, decision=decision, confirmation=confirmation, confirmed=confirmed, gpu=gpu))
    for r in summary['decode']:
        print(f"context {r['context']:4d} sequences {r['sequences']:2d}: {r['ms']:8.2f} ms/step; "
              + ', '.join(f"{l}: {x['median_ratio']:.3f} {x['outcome']}" for l, x in r['layouts'].items()))
    for r in summary['prefill']:
        print(f"prefill {r['rows']:3d} rows to {r['total']:4d}: {r['ms']:8.2f} ms; "
              + ', '.join(f"{l}: {x['median_ratio']:.3f} {x['outcome']}" for l, x in r['layouts'].items()))
    print('qualified:', decision['qualified'], 'selected:', decision['selected'], 'confirmed:', confirmed)
    return record


def single_sequence_summary(record):
    """Block ratios, their median and the verdict of a single-sequence record, from its raw decode steps."""
    runs = record['runs']
    if ([r['run'] for r in runs] != list(range(1, 17))
            or [[r['arm'] for r in runs if r['block'] == b] for b in range(1, 5)] != contract.SINGLE_SEQUENCE['blocks']):
        raise ValueError('the runs are not four alternating blocks')
    if any(r.get('device') != 'Apple M4 Pro/metal' for r in runs):
        raise ValueError('a run does not prove the M4 Pro Metal device')
    if any(len(r['decode_step_ns']) != record['prompt']['max_new_tokens']-1
           or r['median_ms'] != stats.median(r['decode_step_ns'])/1e6 for r in runs):
        raise ValueError('a run lost a decode step or misstates its median')
    ratios = [stats.mean(r['median_ms'] for r in runs if r['block'] == b and r['arm'] == 'candidate')
              / stats.mean(r['median_ms'] for r in runs if r['block'] == b and r['arm'] == 'baseline')
              for b in range(1, 5)]
    median = stats.median(ratios)
    verdict = ('regression' if all(r > 1 for r in ratios) and median > 1.05 else
               'consistent slowdown below the floor' if all(r > 1 for r in ratios) else 'no regression')
    return ratios, median, verdict


def single_sequence_run(report, declared):
    """One run's device and decode steps, from a report that passes the generation validator.

    The validator requires the M4 Pro's Metal device, the token limit or a true stop, the cache length and
    submitted rows, one decode per token after the first and a route of the declared policy for every call.
    """
    validated = generation_events(report, declared['max_new_tokens'], declared['policy'])
    if len(validated['prompt_ids']) != declared['prompt']['tokens']:
        raise ValueError('the prompt encoded to another length')
    device, = [e['value'] for e in validated['events'] if e['event'] == 'device']
    return device, [int(e['nanoseconds']) for e in validated['events'] if e['event'] == 'decode']


def single_sequence(baseline, candidate, prepared, output, purpose):
    """1e's single-sequence check between two receipted generator builds; the record keeps every decode step."""
    ensure_record_location(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = assets(prepared)
    binaries, paths = {}, {}
    for arm, binary in (('baseline', baseline), ('candidate', candidate)):
        binary = Path(binary).resolve()
        receipt_path = Path(str(binary)+'.provenance.json')
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('kind') != 'model-development-build' or receipt['binary_sha256'] != sha(binary)
                or receipt['source']['repository']['dirty'] or not receipt['command'][-3].endswith('generate_cli.mojo')):
            raise ValueError(f'the {arm} is not a clean receipted generator build')
        paths[arm] = binary
        binaries[arm] = dict(sha256=receipt['binary_sha256'], receipt_sha256=sha(receipt_path),
                             commit=receipt['source']['repository']['commit'],
                             archive=receipt['source']['repository'].get('archive', False),
                             command=receipt['command'][1:-1]+[f'<external>/{binary.name}'],
                             decode_projection=receipt['decode_projection'],
                             kv_block_size=receipt.get('kv_block_size'), kv_head_major=receipt.get('kv_head_major', False))
    prompt = output/'prompt.txt'
    prompt.write_text(contract.single_sequence_prompt())
    declared = contract.SINGLE_SEQUENCE
    if sha(prompt) != declared['prompt']['sha256']:
        raise ValueError('the single-sequence prompt changed')
    def snapshot():
        return dict(conditions(), load_average=list(os.getloadavg()))
    runs, blocks, run = [], [], 0
    for block, order in enumerate(declared['blocks'], 1):
        before = snapshot()
        for arm in order:
            run += 1
            report = output/f'run{run}-{arm}.tsv'
            result = subprocess.run([str(paths[arm]), identity['prepared'], identity['tables'], str(prompt),
                                     str(declared['max_new_tokens']), str(declared['chunk_rows']), declared['policy'],
                                     str(report)], cwd=repository_root(), env=environment(), capture_output=True, check=True)
            device, steps = single_sequence_run(report, declared)
            runs.append(dict(run=run, block=block, arm=arm, device=device, decode_step_ns=steps,
                             median_ms=stats.median(steps)/1e6, text_sha256=hashlib.sha256(result.stdout).hexdigest()))
            print(f'run {run} {arm}: median decode step {runs[-1]["median_ms"]:.3f} ms', flush=True)
        blocks.append(dict(block=block, before=before, after=snapshot()))
    for arm in paths:
        if sha(paths[arm]) != binaries[arm]['sha256']:
            raise ValueError('a generator changed during the check')
    record = dict(kind='qwen-single-sequence-check-v1', purpose=purpose, procedure=declared['procedure'],
                  rule=declared['rule'], prompt=dict(declared['prompt'], max_new_tokens=declared['max_new_tokens'],
                                                     chunk_rows=declared['chunk_rows'], policy=declared['policy']),
                  binaries=binaries, assets={k: v for k, v in identity.items() if k.endswith('_sha256')},
                  environment=stable_environment(), conditions=blocks, runs=runs)
    ratios, median, verdict = single_sequence_summary(record)
    record.update(block_ratios=ratios, median_block_ratio=median,
                  texts_identical=len({r['text_sha256'] for r in runs}) == 1, verdict=verdict)
    write(output/'single-sequence.json', record)
    print('block ratios:', ', '.join(f'{r:.3f}' for r in ratios), 'median:', f'{median:.3f}', 'verdict:', verdict,
          'texts identical:', record['texts_identical'])


def batch_plot(directory):
    """Regenerate the batch-size figures exclusively from the checked archive."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    batch_replay(directory)
    summary = json.loads((directory/(BATCH_STEM+'-summary.json')).read_text())
    timing = summary['timing']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
    colors = {64:'#27a89b', 1024:'#244b69', 3968:'#e8a044', 0:'#b86f85'}
    sizes = list(contract.BATCH_SIZES)
    fig, axes = plt.subplots(1, 2, figsize=(11,4.6), constrained_layout=True)
    for context in contract.BATCH_CONTEXTS:
        rows = [r for r in timing if r['context'] == context]
        axes[0].plot(sizes, [r['tokens_per_second'] for r in rows], marker='o', color=colors[context],
                     label=f'{context} cached tokens')
        for tile, style in (('8','--'), ('16',':')):
            axes[0].plot(sizes, [r['sequences']*1000/r['tiles'][tile]['step_ms'] for r in rows],
                         linestyle=style, color=colors[context], linewidth=1)
        middle = [r['step_ms'] for r in rows]
        errors = [[m-min(r['step_block_ms']) for m,r in zip(middle,rows)], [max(r['step_block_ms'])-m for m,r in zip(middle,rows)]]
        axes[1].errorbar(sizes, middle, yerr=errors, marker='o', color=colors[context], capsize=3,
                         label=f'{context} cached tokens')
    mixed = next(r for r in timing if r['context'] == 0)
    for ax, value in ((axes[0], mixed['tokens_per_second']), (axes[1], mixed['step_ms'])):
        ax.plot([mixed['sequences']], [value], marker='D', color=colors[0], linestyle='none', label='mixed, 64 to 3968')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xticks(sizes, [str(b) for b in sizes])
        ax.set_xlabel('Sequences decoding in one step')
    axes[0].set_ylabel('Tokens per second, all sequences')
    axes[0].set_title('Aggregate throughput\nsolid: tile 4; dashed: tile 8; dotted: tile 16')
    axes[0].legend(fontsize=8)
    axes[1].set_yscale('log', base=2)
    axes[1].set_ylabel("Milliseconds per step, each sequence's token latency")
    axes[1].set_title('Step latency with tile 4\nmedian of four block medians; whiskers show range')
    fig.suptitle('Batched decode · Qwen2.5-0.5B · BF16 · M4 Pro / Metal', fontsize=12)
    fig.savefig(directory/(BATCH_STEM+'-throughput.png'), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11,4.6), constrained_layout=True)
    rows = [r for r in timing if r['context'] == 1024]
    parts = [('Host enqueue', ['preflight','step upload','embedding enqueue','decoder stack enqueue','head enqueue'], '#244b69'),
             ('Readback wait', ['readback wait'], '#e8a044'),
             ('Other host', ['forward return','host selection','unmap'], '#abb9c7')]
    bottoms = [0.]*len(rows)
    for name, labels, color in parts:
        values = [sum(r['host_phase_ms'][label] for label in labels) for r in rows]
        axes[0].bar(range(len(rows)), values, bottom=bottoms, color=color, width=.6, label=name)
        bottoms = [a+b for a,b in zip(bottoms, values)]
    axes[0].set_xticks(range(len(rows)), [str(r['sequences']) for r in rows])
    axes[0].set_xlabel('Sequences decoding in one step')
    axes[0].set_ylabel('Milliseconds, median observed step')
    axes[0].set_title('Host intervals at 1,024 cached tokens\nenqueue lengthens while the GPU works')
    axes[0].legend(fontsize=8, loc='upper left')
    groups = [('Decoder projections', {'packed QKV projection','output projection','gate projection','up projection','down projection'}, '#244b69'),
              ('Attention', {'FP32 GQA'}, '#27a89b'),
              ('Vocabulary projection', {'vocabulary projection'}, '#e8a044')]
    traced = [(c, b) for c, b, _ in contract.BATCH_TRACES]
    used = set().union(*(stages for _, stages, _ in groups))
    everything = {r['stage'] for r in summary['gpu']} - {'GPU active total','GPU enclosing span','Metal submission intervals'}
    groups.append(('Other GPU operations', everything-used, '#abb9c7'))
    bottoms = [0.]*len(traced)
    for name, stage_set, color in groups:
        values = [stats.mean(sum(r['median_ms'] for r in summary['gpu'] if (r['context'], r['sequences'], r['repeat']) == (c, b, repeat)
                                 and r['stage'] in stage_set) for repeat in range(2)) for c, b in traced]
        axes[1].bar(range(len(traced)), values, bottom=bottoms, color=color, width=.6, label=name)
        bottoms = [a+b for a,b in zip(bottoms, values)]
    axes[1].set_xticks(range(len(traced)), [str(b) for _, b in traced])
    axes[1].set_xlabel('Sequences decoding in one step')
    axes[1].set_ylabel('Milliseconds of active GPU time per step')
    axes[1].set_title('Separate traces at 1,024 cached tokens, tile 4\nactive GPU time by stage')
    axes[1].legend(fontsize=8, loc='upper left')
    fig.suptitle('Where a batched step spends its time', fontsize=12)
    fig.savefig(directory/(BATCH_STEM+'-breakdown.png'), dpi=170)
    plt.close(fig)


PLOT_STUDIES = dict(
    projections=dict(stem=PROJECTION_BATCH_STEM, control=0, control_name='0: tile 4',
                     colors={'3':'#e8a044', '4':'#27a89b', '5':'#244b69', '6':'#b86f85'},
                     names={'3':'3: early loads', '4':'4: four columns', '5':'5: both', '6':'6: both, column-block order'},
                     title='Exact batched projection arrangements against tile 4',
                     ylabel='Step time relative to arrangement 0 (tile 4)'),
    reordered=dict(stem=REORDERED_BATCH_STEM, control=5, control_name='5: the batched default',
                   colors={'7':'#244b69', '8':'#27a89b', '9':'#e8a044', '10':'#b86f85'},
                   names={'7':'7: eight rows, same order', '8':'8: four-wide lanes', '9':'9: matrix units 8x32',
                          '10':'10: matrix units 16x16'},
                   title='Batched projection arrangements against arrangement 5',
                   ylabel='Step time relative to arrangement 5'))


# Block sizes by hue (validated categorical slots on a light surface); head-major marks are hollow.
PAGED_COLORS = {32: '#2a78d6', 64: '#eb6834', 128: '#1baf7a'}
PAGED_INK = dict(primary='#0b0b0b', secondary='#52514e', muted='#898781', band='#f0efec', axis='#c3c2b7')


def paged_plot(directory, study='paged'):
    """Regenerate 2d's figures, or its rerun's, exclusively from the checked archive."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paged_replay(directory, study)
    stem = batch_study(study)['stem']
    summary = json.loads((directory/(stem+'-summary.json')).read_text())
    timing = summary['timing']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white',
                         'axes.edgecolor':PAGED_INK['axis'],'xtick.color':PAGED_INK['secondary'],
                         'ytick.color':PAGED_INK['secondary'],'text.color':PAGED_INK['primary']})
    candidates = list(contract.PAGED_CANDIDATES)
    def name(layout):
        return f"{contract.PAGED_LAYOUTS[layout]['block_size']}-slot blocks, {contract.PAGED_LAYOUTS[layout]['order']}"
    def style(layout):
        color = PAGED_COLORS[contract.PAGED_LAYOUTS[layout]['block_size']]
        head = contract.PAGED_LAYOUTS[layout]['order'] == 'head-major'
        return dict(color=color, marker='o', markersize=5, markeredgecolor=color, markeredgewidth=1.2,
                    markerfacecolor='white' if head else color, linewidth=1 if head else 1.6, elinewidth=.8, capsize=0)
    offset = {layout: (i-(len(candidates)-1)/2)*.07 for i, layout in enumerate(candidates)}
    def ratios(cell):
        m = cell['median_ratio']
        return m, [[m-min(cell['block_ratios'])], [max(cell['block_ratios'])-m]]
    sizes = list(contract.BATCH_SIZES)
    fig = plt.figure(figsize=(14, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[3, 3, 3, 1.7])
    top = [fig.add_subplot(grid[0, i]) for i in range(4)]
    prefill = fig.add_subplot(grid[1, :])
    for ax, context in zip(top, contract.BATCH_CONTEXTS):
        rows = [r for r in timing['decode'] if r['context'] == context]
        ax.fill_between(sizes, [1-r['noise_floor'] for r in rows], [1+r['noise_floor'] for r in rows],
                        color=PAGED_INK['band'], linewidth=0)
        for layout in candidates:
            for b, row in zip(sizes, rows):
                m, error = ratios(row['layouts'][str(layout)])
                ax.errorbar(b*2**offset[layout], m, yerr=error, **style(layout))
            ax.plot([b*2**offset[layout] for b in sizes], [r['layouts'][str(layout)]['median_ratio'] for r in rows],
                    color=style(layout)['color'], linewidth=style(layout)['linewidth'])
        ax.set_xscale('log', base=2)
        ax.set_xticks(sizes, [str(b) for b in sizes])
        ax.minorticks_off()
        ax.set_xlabel('Sequences decoding in one step')
        ax.set_title(f'Decode, {context:,} cached tokens')
    mixed = next(r for r in timing['decode'] if r['context'] == 0)
    top[3].axhspan(1-mixed['noise_floor'], 1+mixed['noise_floor'], color=PAGED_INK['band'], linewidth=0)
    for i, layout in enumerate(candidates):
        m, error = ratios(mixed['layouts'][str(layout)])
        top[3].errorbar(i, m, yerr=error, linestyle='none', **style(layout))
    top[3].set_xticks(range(len(candidates)), [str(contract.PAGED_LAYOUTS[l]['block_size'])
                                                + ('h' if contract.PAGED_LAYOUTS[l]['order'] == 'head-major' else '')
                                                for l in candidates])
    top[3].set_xlabel('Block size; h: head-major')
    top[3].set_title('Decode, mixed batch')
    rows = timing['prefill']
    for i, row in enumerate(rows):
        prefill.fill_between([i-.42, i+.42], 1-row['noise_floor'], 1+row['noise_floor'], color=PAGED_INK['band'], linewidth=0)
        for layout in candidates:
            m, error = ratios(row['layouts'][str(layout)])
            prefill.errorbar(i+offset[layout]*1.6, m, yerr=error, linestyle='none', **style(layout))
    prefill.set_xticks(range(len(rows)), [f"{r['rows']} rows\nafter {r['total']-r['rows']:,}\nconfiguration {r['configuration']}"
                                          for r in rows], fontsize=8)
    prefill.set_xlim(-.6, len(rows)-.4)
    prefill.set_title('Prefill: one chunk of one sequence, from the token upload to device synchronization')
    # Decode panels share one scale; prefill, whose effects are small, has its own.
    for axes, part in (([*top], 'decode'), ([prefill], 'prefill')):
        values = [v for r in timing[part] for x in r['layouts'].values() for v in x['block_ratios']]
        values += [1+sign*r['noise_floor'] for r in timing[part] for sign in (-1, 1)]
        low, high = min(values), max(values)
        for ax in axes:
            ax.axhline(1, color=PAGED_INK['muted'], linewidth=.8)
            ax.set_ylim(low-(high-low)*.06, high+(high-low)*.06)
    top[0].set_ylabel('Step time relative to one block per sequence')
    prefill.set_ylabel('Chunk time relative to one block,\nits own scale')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], **{k: v for k, v in style(layout).items() if k not in ('elinewidth', 'capsize')})
               for layout in candidates] + [Patch(color=PAGED_INK['band'])]
    fig.legend(handles, [name(layout) for layout in candidates]+['within the noise floor: inconclusive'],
               loc='outside lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle('Paged KV layouts against one block per sequence · median of four paired block ratios; '
                 'whiskers show their range', fontsize=12)
    fig.savefig(directory/(stem+'-ratios.png'), dpi=170)
    plt.close(fig)
    # Where translation would show: attention and the KV writes in the traces.
    gpu = {(r['layout'], r['stage'], r['repeat']): r['median_ms'] for r in summary['gpu']}
    def traced_ms(layout, stage):
        return stats.mean(gpu[layout, stage, r] for r in range(2))
    stages = [('Attention', 'FP32 GQA'), ('KV writes', 'fused QKV/RoPE/cache'), ('All active GPU time', 'GPU active total')]
    traced = [layout for _, _, layout in contract.PAGED_TRACES if layout != contract.PAGED_CONTROL]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True, gridspec_kw=dict(width_ratios=[3, 2]))
    width = .24
    for g, (_, stage) in enumerate(stages):
        base = traced_ms(contract.PAGED_CONTROL, stage)
        for j, layout in enumerate(traced):
            x = g+(j-(len(traced)-1)/2)*width
            repeats = [gpu[layout, stage, r]/base for r in range(2)]
            axes[0].bar(x, stats.mean(repeats)-1, bottom=1, width=width*.85,
                        color=PAGED_COLORS[contract.PAGED_LAYOUTS[layout]['block_size']],
                        label=name(layout) if g == 0 else None)
            axes[0].scatter([x, x], repeats, color=PAGED_INK['primary'], s=9, zorder=3,
                            label='each trace repeat' if g == 0 and j == 0 else None)
    axes[0].axhline(1, color=PAGED_INK['muted'], linewidth=.8)
    axes[0].set_xticks(range(len(stages)), [f'{label}\n{traced_ms(contract.PAGED_CONTROL, stage):.2f} ms per step\nwith one block'
                                            for label, stage in stages])
    axes[0].set_ylabel('Active GPU time relative to one block')
    axes[0].set_title('Relative to one block per sequence')
    # Where the added time sits: attention, the KV writes and every other dispatch, stacked.
    parts = [('Attention', lambda l: traced_ms(l, 'FP32 GQA'), PAGED_INK['primary']),
             ('KV writes', lambda l: traced_ms(l, 'fused QKV/RoPE/cache'), PAGED_INK['secondary']),
             ('Every other dispatch', lambda l: traced_ms(l, 'GPU active total') - traced_ms(l, 'FP32 GQA')
              - traced_ms(l, 'fused QKV/RoPE/cache'), PAGED_INK['axis'])]
    labels = [str(contract.PAGED_LAYOUTS[l]['block_size']) + '-slot' for l in traced]
    # Additions stack up from zero and reductions down from it.
    above, below = [0.]*len(traced), [0.]*len(traced)
    for label, value, color in parts:
        added = [value(l) - value(contract.PAGED_CONTROL) for l in traced]
        axes[1].bar(range(len(traced)), added, bottom=[u if a >= 0 else d for a, u, d in zip(added, above, below)],
                    width=.55, color=color, label=label, edgecolor='white', linewidth=1)
        above = [u + max(a, 0) for u, a in zip(above, added)]
        below = [d + min(a, 0) for d, a in zip(below, added)]
    for i, (u, d) in enumerate(zip(above, below)):
        axes[1].annotate(f'{u + d:+.1f} ms', (i, u), textcoords='offset points', xytext=(0, 4), ha='center',
                         fontsize=9, color=PAGED_INK['primary'])
    axes[1].axhline(0, color=PAGED_INK['muted'], linewidth=.8)
    low, high = min(below), max(above)
    span = (high - low) or 1
    axes[1].set_ylim(low - span*.08, high + span*.18)
    axes[1].set_xticks(range(len(traced)), labels)
    axes[1].set_xlabel('Slot-major blocks')
    axes[1].set_ylabel('Active GPU milliseconds added per step')
    axes[1].set_title('Added to one block per sequence')
    axes[1].legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, fontsize=8, frameon=False)
    fig.suptitle('Traces of 64 sequences at 3,968 cached tokens, two repeats per layout', fontsize=12)
    fig.savefig(directory/(stem+'-traces.png'), dpi=170)
    plt.close(fig)


def projection_plot(directory, study='projections'):
    """Regenerate 1d's or 1e's figures exclusively from the checked archive."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings = PLOT_STUDIES[study]
    (projection_replay if study == 'projections' else reordered_replay)(directory)
    summary = json.loads((directory/(settings['stem']+'-summary.json')).read_text())
    timing = summary['timing']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
    colors, names = settings['colors'], settings['names']
    sizes = list(contract.BATCH_SIZES)
    fig, axes = plt.subplots(1, 4, figsize=(14,4.4), constrained_layout=True, sharey=True,
                             gridspec_kw=dict(width_ratios=[3,3,3,1.5]))
    for ax, context in zip(axes, contract.BATCH_CONTEXTS):
        rows = [r for r in timing if r['context'] == context]
        for arrangement, color in colors.items():
            cells = [r['arrangements'][arrangement] for r in rows]
            middle = [x['median_ratio'] for x in cells]
            errors = [[m-min(x['block_ratios']) for m, x in zip(middle, cells)],
                      [max(x['block_ratios'])-m for m, x in zip(middle, cells)]]
            ax.errorbar(sizes, middle, yerr=errors, marker='o', color=color, capsize=2, linewidth=1.2,
                        label=names[arrangement])
        ax.plot(sizes, [1-r['noise_floor'] for r in rows], color='#abb9c7', linestyle=':', label='gain threshold')
        ax.axhline(1, color='black', linestyle='--', linewidth=.8)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sizes, [str(b) for b in sizes])
        ax.set_xlabel('Sequences decoding in one step')
        ax.set_title(f'{context:,} cached tokens')
    mixed = next(r for r in timing if r['context'] == 0)
    for i, (arrangement, color) in enumerate(colors.items()):
        cell = mixed['arrangements'][arrangement]
        m = cell['median_ratio']
        axes[3].bar(i, m, color=color, width=.7)
        axes[3].errorbar(i, m, yerr=[[m-min(cell['block_ratios'])], [max(cell['block_ratios'])-m]], color='black', capsize=2)
    axes[3].axhline(1, color='black', linestyle='--', linewidth=.8)
    axes[3].axhline(1-mixed['noise_floor'], color='#abb9c7', linestyle=':')
    axes[3].set_xticks(range(len(colors)), list(colors))
    axes[3].set_xlabel('Arrangement')
    axes[3].set_title('Mixed batch of 32')
    axes[0].set_ylabel(settings['ylabel'])
    axes[0].legend(fontsize=7, loc='lower left')
    fig.suptitle(f"{settings['title']} · median of four paired block ratios; whiskers show their range", fontsize=11)
    fig.savefig(directory/(settings['stem']+'-ratios.png'), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11,4.6), constrained_layout=True)
    rows = [r for r in timing if r['context'] == 1024]
    axes[0].plot(sizes, [r['tokens_per_second'] for r in rows], marker='o', color='black', label=settings['control_name'])
    for arrangement, color in colors.items():
        axes[0].plot(sizes, [r['arrangements'][arrangement]['tokens_per_second'] for r in rows], marker='o',
                     color=color, label=names[arrangement])
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(sizes, [str(b) for b in sizes])
    axes[0].set_xlabel('Sequences decoding in one step')
    axes[0].set_ylabel('Tokens per second, all sequences')
    axes[0].set_title('Aggregate throughput at 1,024 cached tokens')
    axes[0].legend(fontsize=7, loc='upper left')
    groups = [('QKV and output projections', {'packed QKV projection','output projection'}, '#8fb3cf'),
              ('Gate and up projections', {'gate projection','up projection'}, '#244b69'),
              ('Down projection', {'down projection'}, '#4f7fa3'),
              ('Vocabulary projection', {'vocabulary projection'}, '#e8a044'),
              ('Attention', {'FP32 GQA'}, '#27a89b')]
    everything = {r['stage'] for r in summary['gpu']} - {'GPU active total','GPU enclosing span','Metal submission intervals'}
    groups.append(('Other GPU operations', everything-set().union(*(s for _, s, _ in groups)), '#abb9c7'))
    arms = [settings['control'], *map(int, colors)]
    bottoms = [0.]*len(arms)
    for name, stage_set, color in groups:
        values = [stats.mean(sum(r['median_ms'] for r in summary['gpu'] if (r['arrangement'], r['repeat']) == (a, repeat)
                                 and r['stage'] in stage_set) for repeat in range(2)) for a in arms]
        axes[1].bar(range(len(arms)), values, bottom=bottoms, color=color, width=.6, label=name)
        bottoms = [x+y for x, y in zip(bottoms, values)]
    axes[1].set_xticks(range(len(arms)), [str(a) for a in arms])
    axes[1].set_xlabel('Arrangement')
    axes[1].set_ylabel('Milliseconds of active GPU time per step')
    axes[1].set_title('Separate traces: 64 sequences at 1,024 cached tokens')
    axes[1].legend(fontsize=7)
    fig.suptitle('Where the batched projections recover time', fontsize=12)
    fig.savefig(directory/(settings['stem']+'-breakdown.png'), dpi=170)
    plt.close(fig)
    if study == 'reordered':
        fig, ax = plt.subplots(figsize=(11,3.8), constrained_layout=True)
        shapes = [(n, k) for _, n, k in contract.BATCH_ACCURACY_SHAPES]
        worst = {(r['arrangement'], r['outputs'], r['inputs']): r['worst_ulps'] for r in summary['accuracy']}
        width = 0.16
        for i, arrangement in enumerate([settings['control'], *map(int, colors)]):
            color = 'black' if arrangement == settings['control'] else colors[str(arrangement)]
            label = settings['control_name'] if arrangement == settings['control'] else names[str(arrangement)]
            ax.bar([s+(i-2)*width for s in range(len(shapes))], [worst[arrangement, n, k] for n, k in shapes],
                   width=width, color=color, label=label)
        ax.set_xticks(range(len(shapes)), [f'{n:,} outputs\nfrom {k:,} inputs' for n, k in shapes])
        ax.set_ylabel('Worst error, BF16 ulps')
        ax.set_yscale('log')
        ax.legend(fontsize=7, ncol=5, loc='upper left')
        ax.set_title('Worst error against the FP64 sum of the same BF16 operands, per projection shape')
        fig.savefig(directory/(settings['stem']+'-accuracy.png'), dpi=170)
        plt.close(fig)


RETIRED = ['enqueue-build', 'enqueue-collect', 'enqueue-archive', 'scheduling-build', 'scheduling-collect',
           'scheduling-capture', 'scheduling-archive', 'projection-confirm', 'selection-capture',
           'selection-terminal', 'selection-archive', 'fusion-capture', 'fusion-terminal', 'fusion-archive']


def engine_specification(output, seed=7, arrival_rate=None):
    """Freeze a bounded synthetic token trace; no fitted rate or latency target."""
    if arrival_rate is not None and (not np.isfinite(arrival_rate) or arrival_rate <= 0):
        raise ValueError('arrival rate must be positive and finite')
    rng = random.Random(seed)
    arrival = 0
    requests = []
    for i, length in enumerate((32, 128, 64, 512, 96, 1024, 256, 64)):
        if arrival_rate is not None and i:
            arrival += max(1, round(rng.expovariate(arrival_rate) * 1e9))
        requests.append(dict(request_id=i, arrival_offset_ns=arrival,
                             prompt_ids=[(p * 103 + i * 17 + 42) % 151643 for p in range(length)],
                             max_new_tokens=32, stop_ids=[151643, 151645], abort_offset_ns=None,
                             output_script=None))
    trace = dict(kind='engine-token-trace-v1', schema_version=1,
                 generator='synthetic-affine-v1', seed=seed,
                 arrival_rate=arrival_rate, mode='offline' if arrival_rate is None else 'online',
                 scripted_tokens=[11, 13, 17, 19, 23], requests=requests)
    validate_engine_trace(trace)
    if output.exists():
        raise ValueError('refusing to replace a token trace')
    write(output, trace)
    return trace


def validate_engine_trace(trace):
    if (trace.get('kind') != 'engine-token-trace-v1' or trace.get('schema_version') != 1
            or trace.get('mode') not in ('offline', 'online')):
        raise ValueError('invalid engine trace declaration')
    requests = trace.get('requests', [])
    if not 1 <= len(requests) <= 128:
        raise ValueError('trace must contain 1..128 requests')
    seen = set()
    previous = -1
    for request in requests:
        identifier = request.get('request_id')
        at = request.get('arrival_offset_ns')
        prompt = request.get('prompt_ids', [])
        maximum = request.get('max_new_tokens')
        stops = request.get('stop_ids', [])
        if (type(identifier) is not int or identifier < 0 or identifier in seen
                or type(at) is not int or at < previous or at < 0
                or type(maximum) is not int or maximum < 0 or not prompt
                or len(prompt) + maximum > 4096):
            raise ValueError('invalid trace request identity, time or budget')
        if any(type(t) is not int or not 0 <= t < 151936 for t in [*prompt, *stops]):
            raise ValueError('invalid trace token')
        abort = request.get('abort_offset_ns')
        if abort is not None and (type(abort) is not int or abort < at):
            raise ValueError('abort precedes arrival')
        if request.get('output_script') is not None:
            raise ValueError('per-request teacher forcing is not implemented')
        if trace['mode'] == 'offline' and at:
            raise ValueError('offline requests must arrive at zero')
        seen.add(identifier)
        previous = at
    script = trace.get('scripted_tokens', [])
    if not script or any(type(t) is not int or not 0 <= t < 151936 for t in script):
        raise ValueError('invalid simulated token oracle')
    return trace


def engine_trace_tsv(trace):
    validate_engine_trace(trace)
    def csv(values): return ','.join(map(str, values)) if values else '-'
    lines = ['script ' + csv(trace['scripted_tokens'])]
    for request in trace['requests']:
        lines.append('request {} {} {} {} {} -'.format(
            request['request_id'], request['arrival_offset_ns'], request['max_new_tokens'],
            csv(request['stop_ids']), csv(request['prompt_ids'])))
        if request['abort_offset_ns'] is not None:
            lines.append(f"abort {request['request_id']} {request['abort_offset_ns']}")
    return '\n'.join(lines) + '\n'


def parse_engine_run(stdout, trace, arm, blocks, maximum_sequences, mode, policy=None):
    """Reject malformed or incomplete native traces before computing any metric."""
    validate_engine_trace(trace)
    if arm not in (*contract.ENGINE_ARMS, 'adaptive') or mode not in ('greedy', 'scripted'):
        raise ValueError('invalid engine arm or mode')
    metadata, arrivals, events, steps = {}, {}, [], []
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[0]
        if name in ('device', 'mode', 'config', 'drained', 'policy'):
            if name in metadata:
                raise ValueError('duplicate engine execution identity')
            metadata[name] = fields[1:]
        elif name == 'arrival' and len(fields) == 4:
            identifier, scheduled, actual = map(int, fields[1:])
            if identifier in arrivals:
                raise ValueError('duplicate request arrival')
            arrivals[identifier] = dict(scheduled_ns=scheduled, actual_ns=actual)
        elif name in ('token', 'finish') and len(fields) == 7:
            identifier = int(fields[1])
            events.append(dict(kind=name, request_id=identifier,
                               token_id=int(fields[2]) if name == 'token' else None,
                               reason=fields[2] if name == 'finish' else None,
                               prompt_tokens=int(fields[3]), generated_tokens=int(fields[4]),
                               arrival_ns=int(fields[5]), emitted_ns=int(fields[6]),
                               step_id=steps[-1]['step_id'] if steps else None))
        elif name == 'step' and len(fields) == len(contract.ENGINE_STEP_FIELDS) + 1:
            steps.append(dict(zip(contract.ENGINE_STEP_FIELDS, map(int, fields[1:]))))
        else:
            raise ValueError('unknown or malformed engine trace line: ' + line[:120])
    device = 'Apple M4 Pro/metal' if mode == 'greedy' else 'simulated/virtual'
    budget = 256 if arm in ('chunked', 'adaptive') else 4096
    sequences = 1 if arm == 'serial' else maximum_sequences
    if (metadata.get('device') != device.split() or metadata.get('mode') != [mode]
            or metadata.get('config') != [arm, str(blocks), str(budget), str(sequences)]):
        raise ValueError('engine device, mode or configuration differs from declaration')
    if policy is None:
        if arm == 'adaptive' or 'policy' in metadata:
            raise ValueError('adaptive run requires a frozen declared policy')
    elif (arm != 'adaptive' or set(policy) != set(contract.ENGINE_POLICY_FIELDS)
          or any(type(v) is not int or v < 0 for v in policy.values()) or policy['target_ns'] < 1
          or metadata.get('policy') != [str(policy[k]) for k in contract.ENGINE_POLICY_FIELDS]):
        raise ValueError('engine adaptive policy differs from declaration')
    expected = {r['request_id']: r for r in trace['requests']}
    if set(arrivals) != set(expected):
        raise ValueError('incomplete engine arrivals')
    if [s['step_id'] for s in steps] != list(range(len(steps))):
        raise ValueError('missing or unordered engine step')
    previous_end = 0
    for step in steps:
        if any(value < 0 for value in step.values()):
            raise ValueError('negative engine counter or timestamp')
        if (step['begin_ns'] < previous_end or step['end_ns'] < step['begin_ns']
                or sum(step[k] for k in ('schedule_ns', 'build_ns', 'execute_ns', 'postprocess_ns'))
                > step['end_ns'] - step['begin_ns']):
            raise ValueError('inconsistent synchronous step timing')
        if (step['prefill_seqs'] > 1 or step['decode_seqs'] + step['prefill_seqs'] > sequences
                or step['total_tokens'] != step['decode_seqs'] + step['prefill_tokens']
                or step['total_tokens'] > budget or step['blocks_free'] > blocks
                or step['aborted'] > step['finished'] or step['budget_limited'] not in (0, 1)):
            raise ValueError('engine exceeded its declared work or pool budget')
        if policy is None and (step['predicted_ns'] or step['budget_limited']):
            raise ValueError('fixed-budget trace unexpectedly used a fitted policy')
        previous_end = step['end_ns']
    histories = {identifier: [] for identifier in expected}
    terminal, last_time = {}, {}
    for event in events:
        identifier = event['request_id']
        if identifier not in expected or identifier in terminal:
            raise ValueError('unknown request or event after terminal result')
        request = expected[identifier]
        arrival = arrivals[identifier]
        if (arrival['scheduled_ns'] != request['arrival_offset_ns']
                or arrival['actual_ns'] < arrival['scheduled_ns']
                or event['arrival_ns'] != arrival['scheduled_ns']
                or event['emitted_ns'] < max(arrival['actual_ns'], last_time.get(identifier, 0))
                or event['prompt_tokens'] != len(request['prompt_ids'])):
            raise ValueError('engine event has invalid time or prompt accounting')
        if event['step_id'] is None:
            raise ValueError('engine event is outside a declared step')
        step = steps[event['step_id']]
        if not step['begin_ns'] <= event['emitted_ns'] <= step['end_ns']:
            raise ValueError('engine event time is outside its step')
        last_time[identifier] = event['emitted_ns']
        tokens = histories[identifier]
        if event['kind'] == 'token':
            token = event['token_id']
            if (not 0 <= token < 151936 or event['generated_tokens'] != len(tokens) + 1
                    or len(tokens) >= request['max_new_tokens']
                    or (tokens and tokens[-1]['token_id'] in request['stop_ids'])):
                raise ValueError('engine token delivery violates order, stop or budget')
            tokens.append(event)
        else:
            reason = event['reason']
            if event['generated_tokens'] != len(tokens) or reason not in ('stop', 'length', 'abort', 'error'):
                raise ValueError('engine terminal token accounting differs')
            if reason == 'length' and len(tokens) != request['max_new_tokens']:
                raise ValueError('engine terminated before its output limit')
            if reason == 'stop' and (not tokens or tokens[-1]['token_id'] not in request['stop_ids']):
                raise ValueError('engine reported a false stop')
            if reason == 'abort' and (request['abort_offset_ns'] is None
                                     or event['emitted_ns'] < request['abort_offset_ns']):
                raise ValueError('engine reported an undeclared or early abort')
            if reason == 'length' and tokens and tokens[-1]['token_id'] in request['stop_ids']:
                raise ValueError('engine ignored its final stop token')
            terminal[identifier] = event
    if set(terminal) != set(expected):
        raise ValueError('missing engine terminal result')
    for step in steps:
        local = [e for e in events if e['step_id'] == step['step_id']]
        if (sum(e['kind'] == 'finish' for e in local) != step['finished']
                or sum(e['reason'] == 'abort' for e in local) != step['aborted']):
            raise ValueError('engine per-step terminal accounting differs')
        if policy is not None and step['total_tokens']:
            partitions = int(step['decode_seqs'] > 0 or step['prefill_tokens'] == 1) + int(step['prefill_tokens'] > 1)
            logits = sum(e['kind'] == 'token' for e in local)
            prediction = (policy['fixed_ns'] + policy['per_row_ns'] * step['total_tokens']
                          + policy['per_position_ns'] * step['attended_positions']
                          + policy['per_partition_ns'] * partitions + policy['per_logit_ns'] * logits)
            if prediction != step['predicted_ns']:
                raise ValueError('engine prediction differs from frozen coefficients')
    drained = metadata.get('drained', [])
    if len(drained) != 4:
        raise ValueError('missing engine drain receipt')
    request_count, step_count, free_blocks, end_ns = map(int, drained)
    if (request_count != len(expected) or step_count != len(steps) or free_blocks != blocks
            or end_ns < max([s['end_ns'] for s in steps] + [e['emitted_ns'] for e in events] + [0])):
        raise ValueError('engine drain receipt differs from complete trace')
    if (sum(s['finished'] for s in steps) != len(terminal)
            or sum(s['aborted'] for s in steps) != sum(e['reason'] == 'abort' for e in terminal.values())
            or (steps and (steps[-1]['blocks_free'] != blocks or steps[-1]['waiting']))):
        raise ValueError('engine did not drain or step/event counts differ')
    for identifier, request in expected.items():
        arrival = arrivals[identifier]
        if (arrival['scheduled_ns'] != request['arrival_offset_ns']
                or arrival['actual_ns'] < arrival['scheduled_ns']):
            raise ValueError('request arrival differs from frozen trace')
    return dict(device=device, mode=mode, arm=arm, blocks=blocks, token_budget=budget,
                max_sequences=sequences, arrivals={str(k): v for k, v in arrivals.items()}, events=events, steps=steps,
                drained=dict(requests=request_count, steps=step_count, free_blocks=free_blocks, elapsed_ns=end_ns),
                policy=policy)


def engine_run_summary(run, trace):
    """Metrics are derived from complete ordered raw events, including the drain."""
    def distribution(values):
        return dict(count=len(values), p50_ns=float(np.percentile(values, 50)),
                    p95_ns=float(np.percentile(values, 95)), p99_ns=float(np.percentile(values, 99))) if values else None
    requests = []
    intervals = []
    for request in trace['requests']:
        identifier = request['request_id']
        tokens = [e for e in run['events'] if e['request_id'] == identifier and e['kind'] == 'token']
        finish = next(e for e in run['events'] if e['request_id'] == identifier and e['kind'] == 'finish')
        gaps = [b['emitted_ns'] - a['emitted_ns'] for a, b in zip(tokens, tokens[1:])]
        intervals.extend(gaps)
        requests.append(dict(request_id=identifier, reason=finish['reason'],
                             delivered_tokens=len(tokens), token_ids=[t['token_id'] for t in tokens],
                             ttft_ns=tokens[0]['emitted_ns'] - request['arrival_offset_ns'] if tokens else None,
                             tpot_ns=stats.mean(gaps) if gaps else None,
                             end_to_end_ns=finish['emitted_ns'] - request['arrival_offset_ns']))
    end = run['drained']['elapsed_ns']
    start = min(r['arrival_offset_ns'] for r in trace['requests'])
    duration = end - start
    delivered = sum(r['delivered_tokens'] for r in requests)
    return dict(duration_ns=duration, delivered_tokens=delivered,
                tokens_per_second=delivered * 1e9 / duration if duration else None,
                clock='host-monotonic' if run['mode'] == 'greedy' else 'virtual',
                arrival_window_ns=max(r['arrival_offset_ns'] for r in trace['requests']) - start,
                drain_tail_ns=end - max(r['arrival_offset_ns'] for r in trace['requests']),
                terminal_reasons=dict(Counter(r['reason'] for r in requests)), requests=requests,
                ttft=distribution([r['ttft_ns'] for r in requests if r['ttft_ns'] is not None]),
                tpot=distribution([r['tpot_ns'] for r in requests if r['tpot_ns'] is not None]),
                inter_token=distribution(intervals),
                end_to_end=distribution([r['end_to_end_ns'] for r in requests]),
                steps=len(run['steps']), preemptions=sum(s['preempted'] for s in run['steps']),
                total_tokens=sum(s['total_tokens'] for s in run['steps']),
                attended_positions=sum(s['attended_positions'] for s in run['steps']))


def engine_study_summary(record):
    trace = validate_engine_trace(record['trace'])
    if record.get('declaration') != contract.ENGINE_DECLARATION:
        raise ValueError('engine measurement declaration changed')
    if (record.get('mode') not in ('greedy', 'scripted')
            or type(record.get('blocks')) is not int or not 1 <= record['blocks'] <= 8192
            or type(record.get('max_sequences')) is not int or not 1 <= record['max_sequences'] <= 64
            or record.get('warmup_steps') != contract.ENGINE_DECLARATION['warmup_steps']):
        raise ValueError('engine collection configuration changed')
    runs = record.get('runs', [])
    keys = [(r['block'], r['arm'], r['calibration']) for r in runs]
    expected = {(b, arm, False) for b in range(4) for arm in contract.ENGINE_ARMS}
    expected |= {(b, 'serial', True) for b in range(4)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('incomplete or duplicate paired engine trace grid')
    summaries = []
    for run in runs:
        for key in ('conditions_before', 'conditions_after'):
            snapshot = run.get(key, {})
            try:
                require_ac(snapshot)
                require_nominal_thermal_state(snapshot)
                if snapshot['power_mode_raw'] != '0':
                    raise ValueError('non-nominal power mode')
            except (KeyError, RuntimeError) as error:
                raise ValueError('incomplete or non-nominal engine conditions') from error
        parsed = parse_engine_run(run['stdout'], trace, run['arm'], record['blocks'],
                                  record['max_sequences'], record['mode'])
        if run.get('parsed') != parsed:
            raise ValueError('engine parsed records differ from native trace')
        summaries.append(dict(block=run['block'], arm=run['arm'], calibration=run['calibration'],
                              **engine_run_summary(parsed, trace)))
    by_key = {(r['block'], r['arm'], r['calibration']): r for r in summaries}
    if any(r['duration_ns'] <= 0 for r in summaries):
        raise ValueError('paired engine makespan must be positive')
    calibrations = [by_key[b, 'serial', True]['duration_ns'] / by_key[b, 'serial', False]['duration_ns'] for b in range(4)]
    noise = max(.05, max(abs(r - 1) for r in calibrations))
    comparisons = []
    for arm in contract.ENGINE_ARMS[1:]:
        ratios, same_history = [], True
        for block in range(4):
            control, candidate = by_key[block, 'serial', False], by_key[block, arm, False]
            ratios.append(candidate['duration_ns'] / control['duration_ns'])
            same_history &= all(a['token_ids'] == b['token_ids'] for a, b in zip(control['requests'], candidate['requests']))
        outcome = ('virtual-clock' if record['mode'] == 'scripted' else
                   'different-greedy-histories' if not same_history else
                   'distribution-only' if trace['mode'] == 'online' else _outcome(ratios, noise))
        comparisons.append(dict(arm=arm, block_ratios=ratios, median_ratio=stats.median(ratios),
                                same_generated_histories=same_history, outcome=outcome))
    return dict(runs=summaries, comparisons=comparisons, calibration_ratios=calibrations,
                noise_floor=noise, target=None, goodput=None, quantile_method='linear')


def engine_build(output, prepared):
    ensure_record_location(output)
    source = source_identity()
    if source['repository']['dirty']:
        raise ValueError('engine build requires clean source')
    identity = assets(prepared)
    output.mkdir(parents=True, exist_ok=False)
    command = [environment_tool('mojo'), 'build', '-I', 'src',
               'src/llm_mojo/benchmarks/engine_trace.mojo', '-o', output/'engine']
    execute(command, output/'engine-build.log')
    if source_identity() != source or assets(prepared) != identity:
        raise ValueError('engine source or assets changed during build')
    write(output/'build.json', dict(source=source, assets=identity, environment=stable_environment(),
                                    declaration=contract.ENGINE_DECLARATION, command=list(map(str, command)),
                                    binaries={'engine': dict(sha256=sha(output/'engine'), bytes=(output/'engine').stat().st_size)}))


def engine_collect(directory, trace_path, output, blocks=128, maximum_sequences=8, mode='greedy', warmup_steps=10):
    if (type(blocks) is not int or not 1 <= blocks <= 8192
            or type(maximum_sequences) is not int or not 1 <= maximum_sequences <= 64
            or mode not in ('greedy', 'scripted') or warmup_steps != contract.ENGINE_DECLARATION['warmup_steps']):
        raise ValueError('invalid bounded engine collection')
    receipt = verify_build(directory)
    if receipt.get('declaration') != contract.ENGINE_DECLARATION or set(receipt['binaries']) != {'engine'}:
        raise ValueError('not a current engine trace build')
    trace_document = trace_path.read_text()
    trace_file_sha256 = hashlib.sha256(trace_document.encode()).hexdigest()
    trace = validate_engine_trace(json.loads(trace_document))
    if any((max(len(r['prompt_ids']),len(r['prompt_ids'])+r['max_new_tokens']-1)+31)//32 > blocks
           for r in trace['requests']):
        raise ValueError('a declared request cannot fit the pool alone')
    ensure_record_location(output)
    output.mkdir(parents=True, exist_ok=False)
    native_trace = output/'trace.tsv'
    native_trace.write_text(engine_trace_tsv(trace))
    runs = []
    for block in range(4):
        order = [('serial', False), ('serial', True), *[(a, False) for a in contract.ENGINE_ARMS[1:]]]
        if block in (1, 2):
            order.reverse()
        for arm, calibration in order:
            before = conditions()
            budget, sequences = (256 if arm == 'chunked' else 4096), (1 if arm == 'serial' else maximum_sequences)
            command = [directory/'engine', receipt['assets']['prepared'], native_trace, arm,
                       blocks, budget, sequences, warmup_steps, mode]
            log = output/f'block-{block}-{arm}{"-calibration" if calibration else ""}.log'
            stdout = execute(command, log)
            parsed = parse_engine_run(stdout, trace, arm, blocks, maximum_sequences, mode)
            runs.append(dict(block=block, arm=arm, calibration=calibration, stdout=stdout, parsed=parsed,
                             conditions_before=before, conditions_after=conditions()))
    if verify_build(directory) != receipt or sha(trace_path) != trace_file_sha256:
        raise ValueError('engine build or trace changed during collection')
    record = dict(kind='qwen-engine-core-v1', declaration=contract.ENGINE_DECLARATION,
                  build=receipt, trace=trace, trace_document=trace_document, trace_sha256=trace_file_sha256,
                  blocks=blocks, max_sequences=maximum_sequences, mode=mode,
                  warmup_steps=warmup_steps, runs=runs)
    record['summary'] = engine_study_summary(record)
    payload = json.dumps(record, separators=(',', ':')).encode()
    compressed = gzip.compress(payload, mtime=0)
    (output/'engine-core.json.gz').write_bytes(compressed)
    write(output/'engine-core.json', dict(kind='qwen-engine-core-v1', sha256=hashlib.sha256(compressed).hexdigest(),
                                         uncompressed_sha256=hashlib.sha256(payload).hexdigest(), bytes=len(compressed)))
    write(output/'engine-core-summary.json', record['summary'])
    return engine_replay(output)


def validate_engine_record_build(record):
    """Validate retained provenance without touching current source or assets."""
    build = record.get('build', {})
    if (record.get('kind') != 'qwen-engine-core-v1' or build.get('source', {}).get('repository', {}).get('dirty') is not False
            or not build.get('binaries', {}).get('engine', {}).get('sha256')
            or build.get('declaration') != contract.ENGINE_DECLARATION
            or not build.get('assets', {}).get('prepared_sha256')
            or not build.get('assets', {}).get('tables_sha256')):
        raise ValueError('incomplete engine source, binary or asset provenance')
    source = build['source']
    def digest(value, width=64):
        return isinstance(value, str) and len(value) == width and all(c in '0123456789abcdef' for c in value)
    if (not digest(source['repository'].get('commit'), 40)
            or not source.get('sources') or not digest(source['sources'].get('uv.lock'))
            or not all(digest(v) for v in source['sources'].values())
            or not digest(build['binaries']['engine']['sha256'])
            or type(build['binaries']['engine'].get('bytes')) is not int or build['binaries']['engine']['bytes'] < 1
            or not digest(build['assets']['prepared_sha256']) or not digest(build['assets']['tables_sha256'])
            or not build.get('environment', {}).get('hardware')
            or not build.get('environment', {}).get('software')):
        raise ValueError('invalid engine provenance identity')
    return build


def engine_replay(directory):
    archive = directory if directory.is_file() else directory/'engine-core.json.gz'
    manifest_path = archive.with_suffix('')
    summary_path = archive.with_name(archive.name.removesuffix('.json.gz')+'-summary.json')
    manifest = json.loads(manifest_path.read_text())
    compressed = archive.read_bytes()
    payload = gzip.decompress(compressed)
    if (manifest.get('kind') != 'qwen-engine-core-v1' or len(compressed) != manifest['bytes']
            or hashlib.sha256(compressed).hexdigest() != manifest['sha256']
            or hashlib.sha256(payload).hexdigest() != manifest['uncompressed_sha256']):
        raise ValueError('engine archive hash mismatch')
    record = json.loads(payload)
    document = record.get('trace_document', '')
    if (hashlib.sha256(document.encode()).hexdigest() != record.get('trace_sha256')
            or json.loads(document) != record.get('trace')):
        raise ValueError('engine token trace identity changed')
    validate_engine_record_build(record)
    summary = engine_study_summary(record)
    if summary != record.get('summary'):
        raise ValueError('engine archived summary changed')
    write(summary_path, summary)
    return summary


def main():
    # Trace capture and the reused terminal lifecycle helper inherit this process.
    os.environ.pop('MODULAR_DEBUG', None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['batch-support','batch-support-replay','enqueue-plot','enqueue-replay',
                                            'scheduling-plot','scheduling-replay','build','collect','capture',
                                            'terminal','archive','replay','plot','fusion-replay','fusion-plot',
                                            'selection-replay','selection-plot','batch-size-build','batch-size-collect',
                                            'batch-size-confirm','batch-size-diagnose','batch-size-capture','batch-size-archive',
                                            'batch-size-replay','batch-size-plot','single-sequence',
                                            'engine-specification','engine-build','engine-collect','engine-replay',
                                            *RETIRED])
    parser.add_argument('--projections',action='store_true',help='Replay/plot the projection arrangement study')
    parser.add_argument('--study', choices=['size','projections','reordered','addressing','paged','paged-loop'],
                        default='size',
                        help='batch-size-* study: 1c row tiles (size), 1d exact arrangements (projections), '
                             '1e reordered arrangements (reordered), 1f addressing (addressing), 2d KV layouts (paged) '
                             'or their rerun with decode attention in one loop (paged-loop)')
    parser.add_argument('--baseline', type=Path, help='single-sequence: the receipted baseline generator')
    parser.add_argument('--candidate', type=Path, help='single-sequence: the receipted candidate generator')
    parser.add_argument('--purpose', help='single-sequence: what the check decides')
    parser.add_argument('--screen', type=Path, help='screen timings for batch-size-confirm and batch-size-diagnose')
    parser.add_argument('--confirmation', type=Path, help='confirmation timings for batch-size-archive and -diagnose')
    parser.add_argument('--diagnostics', type=Path, help='1e model-level diagnostics for batch-size-archive')
    parser.add_argument('--residual-norm',action='store_true',help='Replay/plot the residual normalization study')
    parser.add_argument('--copy-free',action='store_true',help='Replay/plot the buffer ownership study')
    parser.add_argument('--selection',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--combined', action='store_true', help='Replay/plot the combined fusion study')
    parser.add_argument('--fusion', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timings', type=Path)
    parser.add_argument('--traces', type=Path)
    parser.add_argument('--terminal', type=Path)
    parser.add_argument('--trace', type=Path, help='engine-collect: frozen token-trace JSON')
    parser.add_argument('--seed', type=int, default=7, help='engine-specification: trace generator seed')
    parser.add_argument('--arrival-rate', type=float, help='engine-specification: seeded Poisson requests/s; omission is offline')
    parser.add_argument('--blocks', type=int, default=128, help='engine-collect: explicit 32-slot KV pool capacity')
    parser.add_argument('--max-sequences', type=int, default=8, help='engine-collect: maximum concurrent sequences')
    parser.add_argument('--engine-mode', choices=['greedy', 'scripted'], default='greedy')
    parser.add_argument('--warmup-steps', type=int, default=10)
    args = parser.parse_args()
    if args.command in RETIRED or (args.command == 'build' and (args.fusion or args.combined or args.selection
                                   or args.copy_free or args.residual_norm or args.projections)):
        parser.error(args.command + ' belongs to a completed experiment; its retained archive replays, and '
                     're-collection needs the commit recorded in that archive (collectors exist through edb610a)')
    if args.command == 'engine-specification':
        engine_specification(args.output.resolve(), args.seed, args.arrival_rate)
    elif args.command == 'engine-build':
        if args.prepared is None: parser.error('engine-build needs --prepared')
        engine_build(args.output.resolve(), args.prepared)
    elif args.command == 'engine-collect':
        if args.build is None or args.trace is None: parser.error('engine-collect needs --build and --trace')
        engine_collect(args.build.resolve(), args.trace.resolve(), args.output.resolve(), args.blocks,
                       args.max_sequences, args.engine_mode, args.warmup_steps)
    elif args.command == 'engine-replay': print(json.dumps(engine_replay(args.output.resolve()), indent=2))
    elif args.command == 'batch-support': batch_support_collect(args.output.resolve())
    elif args.command == 'batch-support-replay': print(json.dumps(batch_support_replay(args.output),indent=2))
    elif args.command == 'enqueue-plot': enqueue_plot(args.output)
    elif args.command == 'enqueue-replay': print(json.dumps(enqueue_replay(args.output),indent=2))
    elif args.command == 'scheduling-plot': scheduling_plot(args.output)
    elif args.command == 'scheduling-replay': scheduling_replay(args.output)
    elif args.command == 'build': build(args.output.resolve(), args.prepared)
    elif args.command == 'batch-size-build': batch_build(args.output.resolve(), args.prepared, args.study)
    elif args.command == 'batch-size-collect': batch_collect(args.build.resolve(), args.output.resolve(), args.study)
    elif args.command == 'batch-size-confirm':
        batch_confirm(args.build.resolve(), args.screen.resolve(), args.output.resolve(),
                      'projections' if args.study == 'size' else args.study)
    elif args.command == 'batch-size-diagnose':
        reordered_diagnostics(args.screen.resolve(), args.confirmation.resolve(), args.output.resolve(), args.prepared)
    elif args.command == 'batch-size-capture': batch_capture(args.build.resolve(), args.output.resolve(), args.study)
    elif args.command == 'batch-size-archive':
        if args.study == 'addressing':
            addressing_archive(args.timings, args.traces, args.output)
        else:
            batch_archive(args.timings, args.traces, args.output, args.study, args.confirmation, args.diagnostics)
    elif args.command == 'batch-size-replay':
        dict(size=batch_replay, projections=projection_replay, reordered=reordered_replay,
             addressing=addressing_replay, paged=paged_replay,
             **{'paged-loop': lambda directory: paged_replay(directory, 'paged-loop')})[args.study](args.output)
    elif args.command == 'batch-size-plot':
        if args.study == 'size': batch_plot(args.output)
        elif args.study in PAGED_STUDIES: paged_plot(args.output, args.study)
        else: projection_plot(args.output, args.study)
    elif args.command == 'single-sequence':
        if not (args.baseline and args.candidate and args.prepared and args.purpose):
            parser.error('single-sequence needs --baseline, --candidate, --prepared and --purpose')
        single_sequence(args.baseline, args.candidate, args.prepared, args.output.resolve(), args.purpose)
    elif args.command == 'selection-plot':
        if args.projections: projection_plot(args.output)
        else: selection_plot(args.output,args.residual_norm)
    elif args.command == 'selection-replay': selection_replay(args.output,args.residual_norm,args.projections)
    elif args.command == 'fusion-replay': fusion_replay(args.output,args.combined,args.copy_free)
    elif args.command == 'fusion-plot': fusion_plot(args.output,args.combined,args.copy_free)
    elif args.command == 'collect': collect(args.build.resolve(), args.output.resolve())
    elif args.command == 'capture': capture(args.build.resolve(), args.output.resolve())
    elif args.command == 'terminal': terminal(args.build.resolve(), args.output.resolve())
    elif args.command == 'archive': archive(args.timings, args.traces, args.output, args.terminal)
    elif args.command == 'plot': plot(args.output)
    else: replay(args.output)


if __name__ == '__main__':
    main()
