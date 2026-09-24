"""Collect the Fast-route token profile; replay every retained full-model archive.

Build/run/capture require a clean local checkout and verified local assets and
measure the current Fast route. Completed decode experiments are replay-only:
their archives, parsers and summaries remain, their collectors do not.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics as stats
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from .._repository import environment_tool, repository_root
from ..validation.evidence import source_identity, sha, write
from llm_mojo.models.qwen2.assets import verify_prepared
from ..validation.model import environment
from llm_mojo.models.qwen2.tokenizer_assets import ensure_prepared
from .environment import stable_environment, conditions_snapshot, require_ac, require_nominal_thermal_state, ensure_record_location
from . import model_contract as contract


def execute(command, log):
    result = subprocess.run(list(map(str, command)), cwd=repository_root(), env=environment(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
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
RETIRED = ['enqueue-build', 'enqueue-collect', 'enqueue-archive', 'scheduling-build', 'scheduling-collect',
           'scheduling-capture', 'scheduling-archive', 'projection-confirm', 'selection-capture',
           'selection-terminal', 'selection-archive', 'fusion-capture', 'fusion-terminal', 'fusion-archive']


def main():
    # Trace capture and the reused terminal lifecycle helper inherit this process.
    os.environ.pop('MODULAR_DEBUG', None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['batch-support','batch-support-replay','enqueue-plot','enqueue-replay',
                                            'scheduling-plot','scheduling-replay','build','collect','capture',
                                            'terminal','archive','replay','plot','fusion-replay','fusion-plot',
                                            'selection-replay','selection-plot',*RETIRED])
    parser.add_argument('--projections',action='store_true',help='Replay/plot the projection arrangement study')
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
    args = parser.parse_args()
    if args.command in RETIRED or (args.command == 'build' and (args.fusion or args.combined or args.selection
                                   or args.copy_free or args.residual_norm or args.projections)):
        parser.error(args.command + ' belongs to a completed experiment; its retained archive replays, and '
                     're-collection needs the commit recorded in that archive (collectors exist through edb610a)')
    if args.command == 'batch-support': batch_support_collect(args.output.resolve())
    elif args.command == 'batch-support-replay': print(json.dumps(batch_support_replay(args.output),indent=2))
    elif args.command == 'enqueue-plot': enqueue_plot(args.output)
    elif args.command == 'enqueue-replay': print(json.dumps(enqueue_replay(args.output),indent=2))
    elif args.command == 'scheduling-plot': scheduling_plot(args.output)
    elif args.command == 'scheduling-replay': scheduling_replay(args.output)
    elif args.command == 'build': build(args.output.resolve(), args.prepared)
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
