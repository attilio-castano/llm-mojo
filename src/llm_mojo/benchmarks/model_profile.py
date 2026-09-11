"""Collect and replay the bounded full-model token profiling study.

Build/run/capture require a clean local checkout and verified local assets.
Replay needs only the retained archive. The fusion experiment keeps an explicit control.
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
from ..mlp_validation import source_identity, sha, write
from ..model_assets import verify_prepared
from ..model_validation import environment
from ..tokenizer_assets import ensure_prepared
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


def build(output, prepared, fusion=False):
    ensure_record_location(output)
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    if source['repository']['dirty']:
        raise ValueError('model profiling build requires clean source')
    identity = assets(prepared)
    binaries = {}
    for name, entry in [('model', 'benchmarks/model.mojo'), ('terminal', 'chat_cli.mojo')]:
        command = [environment_tool('mojo'), 'build', '-I', 'src',
                   *(['-D','MODEL_FUSION_STUDY'] if fusion else []),
                   'src/llm_mojo/'+entry, '-o', output/name]
        execute(command, output/f'{name}-build.log')
        binaries[name] = dict(sha256=sha(output/name), bytes=(output/name).stat().st_size)
    machine = stable_environment()
    for prefix, fused in ([(1024,False),(1024,True)] if fusion else [(p,False) for p in contract.PREFIXES]):
        name = f'profile-{prefix}'+('-fused' if fused else '')
        command = [environment_tool('mojo'), 'build', '-I', 'src',
                   *(['-D','MODEL_FUSION_STUDY'] if fusion else []),
                   *(['-D','MODEL_FUSION_PROFILE'] if fused else []),
                   '-D', f'MODEL_PROFILE_PREFIX={prefix}',
                   '-D', 'MODEL_PREPARED='+identity['prepared'],
                   '-D', 'MODEL_TABLES='+identity['tables'],
                   'src/llm_mojo/benchmarks/model.mojo', '-o', output/name]
        execute(command, output/f'{name}-build.log')
        binary = dict(sha256=sha(output/name), bytes=(output/name).stat().st_size)
        binaries[name] = binary
        provenance = dict(schema_version=1, operation=contract.OPERATION,
                          implementation='qwen_model_fused' if fused else 'qwen_model_fast',
                          entrypoint=contract.ENTRYPOINTS['qwen_model_fused' if fused else 'qwen_model_fast'],
                          repository=source['repository'], source_sha256=source['sources'],
                          **machine, **contract.specification(prefix,fused),
                          profile_warmup_iterations=10, profile_iterations=8,
                          profile_post_idle_milliseconds=250, binary=binary,
                          assets={k:v for k,v in identity.items() if k.endswith('_sha256')})
        contract.configuration(provenance)
        write(output/(name+'.provenance.json'), provenance)
    if source_identity() != source or assets(prepared) != identity:
        raise ValueError('source or assets changed during compilation')
    write(output/'build.json', dict(source=source, assets=identity, environment=machine,
                                    declaration=contract.FUSION_DECLARATION if fusion else contract.DECLARATION, binaries=binaries))


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


def verify_snapshots(directory, prefix):
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
        inactive_exact = name == 'logits' or np.array_equal(before[(prefix+1)*128:], plain[(prefix+1)*128:])
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


def collect(directory, output):
    ensure_record_location(output)
    receipt = verify_build(directory)
    output.mkdir(parents=True, exist_ok=False)
    args = receipt['assets']
    base = [directory/'model', 'verify', args['prepared'], args['tables']]
    numerical = []
    for prefix in contract.PREFIXES:
        target = output/f'verify-{prefix}'
        target.mkdir()
        stdout = execute([*base, prefix, 0, 0, target], target/'driver.log')
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
                samples.extend(parse_samples(stdout, prefix, block, comparison,
                                             observed=receipt['declaration']==contract.DECLARATION))
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


def curate(target, prefix, repeat, fused=False):
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
    stages = contract.command_stages(fused)
    count = len(stages)
    intervals, joined = coalesce_compute_commands(intervals, submissions, 18*count, join_resubmissions=True)
    if joined != report['validated_sequence']['interval_coalescing']:
        raise ValueError('curation differs from validated dispatch join')
    *_, measured = segment_compute_commands(intervals, 10, 8, count, False)
    contract.validate_command_sequence(measured,fused)
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
                  'final norm/head enqueue','forward return','map/wait','CPU argmax scan','unmap']
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
        stages = contract.command_stages()
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
    groups.append(('Other GPU operations',set(s for _,s in contract.stages())-used,'#abb9c7'))
    groups.append(('Buffer transfers',set(s for _,s,k in contract.command_stages() if k=='blit'),'#678d75'))
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


def fusion_capture(directory, output):
    """One control and one candidate capture at the representative context."""
    from .capture_trace import capture_trace
    receipt = verify_build(directory)
    if receipt['declaration'] != contract.FUSION_DECLARATION:
        raise ValueError('not a fusion build')
    ensure_record_location(output)
    output.mkdir(parents=True,exist_ok=False)
    for fused in (False,True):
        name = 'profile-1024'+('-fused' if fused else '')
        target = output/('fused' if fused else 'control')
        target.mkdir()
        before = conditions()
        capture_trace(profile_binary=directory/name,output_trace=target/'raw.trace',
                      receipt_path=target/'capture.json',time_limit='30s')
        write(target/'conditions.json',dict(before=before,after=conditions()))
        (target/'profile.provenance.json').write_bytes((directory/(name+'.provenance.json')).read_bytes())
        export_trace(target)
        curated = curate(target,1024,0,fused)
        write(target/'curated.json',curated)
        print('Verified fusion capture',name,len(curated['samples']),flush=True)
    verify_build(directory)


def fusion_terminal(directory, output):
    receipt = verify_build(directory)
    if receipt['declaration'] != contract.FUSION_DECLARATION:
        raise ValueError('not a fusion build')
    ensure_record_location(output)
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(repository_root()/'tests'))
    from chat_terminal import events, validate
    original = json.loads(gzip.decompress((repository_root()/'studies/model_generation/token-profile.json.gz').read_bytes()))
    prompts = original['terminal']['prompts']
    inputs = ('\n/reset\n'.join(p.replace('\n',' ') for p in prompts)+'\n/exit\n').encode()
    identity = receipt['assets']
    blocks = []
    for block in range(4):
        before = conditions()
        arms = []
        outputs = []
        for fused in ([True,False] if block in (1,2) else [False,True]):
            report = output/f'b{block}-f{int(fused)}.tsv'
            command = list(map(str,[directory/'terminal',identity['prepared'],identity['tables'],128,256,
                                    '',report,'fusion' if fused else 'fast']))
            result = subprocess.run(command,cwd=repository_root(),env=environment(),input=inputs,
                                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
            (output/f'b{block}-f{int(fused)}.txt').write_bytes(result.stdout)
            if result.returncode:
                raise ValueError('fusion terminal failed')
            ev = events(report)
            turns = validate(ev,128)
            if len(turns)!=3:
                raise ValueError('missing fusion terminal turns')
            arms.append(dict(fused=fused,events=ev,turns=turns,output_sha256=hashlib.sha256(result.stdout).hexdigest()))
            outputs.append(result.stdout)
        if outputs[0]!=outputs[1] or any(a['generated']!=b['generated'] or a['history']!=b['history']
                                       for a,b in zip(arms[0]['turns'],arms[1]['turns'])):
            raise ValueError('fusion changed terminal text, tokens or history')
        blocks.append(dict(block=block,before=before,after=conditions(),arms=arms,output_exact=True))
        print('Verified fusion terminal block',block+1,flush=True)
    verify_build(directory)
    write(output/'terminal.json',dict(build=receipt,prompts=prompts,blocks=blocks))


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


def fusion_archive(timings, traces, terminal_path, output):
    record = dict(kind='qwen-qkv-fusion-v1',timing=json.loads((timings/'timings.json').read_text()),
                  terminal=json.loads((terminal_path/'terminal.json').read_text()),
                  captures=[json.loads((traces/name/'curated.json').read_text()) for name in ('control','fused')])
    # Keep exact evidence and hashes; omit local asset paths from publication.
    def scrub(value):
        if isinstance(value,dict):
            return {k:('<verified-local-asset>' if k in ('prepared','tables') else scrub(v)) for k,v in value.items()}
        if isinstance(value,list): return [scrub(v) for v in value]
        return value
    raw = json.dumps(scrub(record),separators=(',',':'),allow_nan=False).encode()
    packed = gzip.compress(raw,mtime=0)
    output.mkdir(parents=True,exist_ok=True)
    (output/'qkv-fusion.json.gz').write_bytes(packed)
    write(output/'qkv-fusion.json',dict(kind=record['kind'],sha256=hashlib.sha256(packed).hexdigest(),
                                      uncompressed_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(packed)))
    fusion_replay(output)


def fusion_plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fusion_replay(directory)
    rows=json.loads((directory/'qkv-fusion-summary.json').read_text())['timing']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for i,r in enumerate(rows):
        for offset,key,color,label in [(-.18,'baseline','#64798c','Control'),(.18,'candidate','#188f82','Fused')]:
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
    axes[1].set_ylabel('Fused / control latency; lower is better')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(3),[str(r['prefix']) for r in rows])
        ax.set_xlabel('Previously cached tokens')
    fig.suptitle('Qwen2.5-0.5B · BF16 · batch one · M4 Pro / Metal',fontsize=13)
    fig.savefig(directory/'qkv-fusion.png',dpi=170)
    plt.close(fig)


def fusion_replay(directory):
    manifest = json.loads((directory/'qkv-fusion.json').read_text())
    packed = (directory/'qkv-fusion.json.gz').read_bytes()
    raw = gzip.decompress(packed)
    if hashlib.sha256(packed).hexdigest()!=manifest['sha256'] or hashlib.sha256(raw).hexdigest()!=manifest['uncompressed_sha256']:
        raise ValueError('fusion evidence hash mismatch')
    record = json.loads(raw)
    if record['kind']!='qwen-qkv-fusion-v1': raise ValueError('unexpected fusion evidence kind')
    timing = record['timing']
    build_record = timing['build']
    if build_record['declaration']!=contract.FUSION_DECLARATION or record['terminal']['build']!=build_record:
        raise ValueError('fusion build declaration changed')
    summary = fusion_summary(timing['samples'])
    if Counter(n['prefix'] for n in timing['numerical'])!=Counter(contract.PREFIXES):
        raise ValueError('incomplete fusion numerical coverage')
    names = ['logits']+[f'{k}{i}' for k in ('k','v') for i in range(24)]
    for n in timing['numerical']:
        if (len(n['history'])!=n['prefix']+1 or Counter(r['name'] for r in n['observations'])!=Counter(names)
                or any(r[k] is not True for r in n['observations'] for k in ('exact','finite','prefix_exact','inactive_exact'))):
            raise ValueError('fusion numerical invariant failed')
    if len(record['captures'])!=2: raise ValueError('incomplete fusion capture census')
    for fused,capture in zip((False,True),record['captures']):
        provenance = capture['provenance']
        contract.configuration(provenance)
        name = 'profile-1024'+('-fused' if fused else '')
        if (provenance['repository']!=build_record['source']['repository']
                or provenance['source_sha256']!=build_record['source']['sources']
                or provenance['binary']!=build_record['binaries'][name]
                or provenance['assets']!={k:v for k,v in build_record['assets'].items() if k.endswith('_sha256')}
                or provenance['implementation']!=('qwen_model_fused' if fused else 'qwen_model_fast')):
            raise ValueError('fusion trace build identity mismatch')
        canonical = (json.dumps(provenance,indent=2,allow_nan=False)+'\n').encode()
        if hashlib.sha256(canonical).hexdigest()!=capture['analysis']['capture_identity']['provenance']['sha256']:
            raise ValueError('fusion trace capture provenance mismatch')
        stages = contract.command_stages(fused)
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
    write(directory/'qkv-fusion-summary.json',dict(timing=summary,promote=all(r['promote'] for r in summary)))
    print(json.dumps(summary,indent=2))
    return record


def main():
    # Trace capture and the reused terminal lifecycle helper inherit this process.
    os.environ.pop('MODULAR_DEBUG', None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build','collect','capture','terminal','archive','replay','plot','fusion-capture','fusion-terminal','fusion-archive','fusion-replay','fusion-plot'])
    parser.add_argument('--fusion', action='store_true', help='Build the bounded fusion experiment')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timings', type=Path)
    parser.add_argument('--traces', type=Path)
    parser.add_argument('--terminal', type=Path)
    args = parser.parse_args()
    if args.command == 'build': build(args.output.resolve(), args.prepared, args.fusion)
    elif args.command == 'fusion-capture': fusion_capture(args.build.resolve(),args.output.resolve())
    elif args.command == 'fusion-terminal': fusion_terminal(args.build.resolve(),args.output.resolve())
    elif args.command == 'fusion-archive': fusion_archive(args.timings,args.traces,args.terminal,args.output)
    elif args.command == 'fusion-replay': fusion_replay(args.output)
    elif args.command == 'fusion-plot': fusion_plot(args.output)
    elif args.command == 'collect': collect(args.build.resolve(), args.output.resolve())
    elif args.command == 'capture': capture(args.build.resolve(), args.output.resolve())
    elif args.command == 'terminal': terminal(args.build.resolve(), args.output.resolve())
    elif args.command == 'archive': archive(args.timings, args.traces, args.output, args.terminal)
    elif args.command == 'plot': plot(args.output)
    else: replay(args.output)


if __name__ == '__main__':
    main()
