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


def build(output, prepared, fusion=False, combined=False, selection=False, copy_free=False):
    if copy_free and (combined or selection):
        raise ValueError("copy-free is a separate bounded study")
    if selection:
        return selection_build(output,prepared)
    fusion = fusion or combined or copy_free
    default_combined = not fusion and contract.FAST_DECODE_CONFIGURATION == 26
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
                   *(['-D','MODEL_COPY_FREE_STUDY'] if copy_free else []),
                   *(['-D','MODEL_COMBINED_STUDY'] if combined else []),
                   'src/llm_mojo/'+entry, '-o', output/name]
        execute(command, output/f'{name}-build.log')
        binaries[name] = dict(sha256=sha(output/name), bytes=(output/name).stat().st_size)
    machine = stable_environment()
    for prefix, fused in ([(1024,False),(1024,True)] if fusion else [(p,default_combined) for p in contract.PREFIXES]):
        name = f'profile-{prefix}'+('-fused' if fusion and fused else '')
        command = [environment_tool('mojo'), 'build', '-I', 'src',
                   *(['-D','MODEL_FUSION_STUDY'] if fusion else []),
                   *(['-D','MODEL_COPY_FREE_STUDY'] if copy_free else []),
                   *(['-D','MODEL_COMBINED_STUDY'] if combined or default_combined else []),
                   *(['-D','MODEL_FUSION_PROFILE'] if fused else []),
                   '-D', f'MODEL_PROFILE_PREFIX={prefix}',
                   '-D', 'MODEL_PREPARED='+identity['prepared'],
                   '-D', 'MODEL_TABLES='+identity['tables'],
                   'src/llm_mojo/benchmarks/model.mojo', '-o', output/name]
        execute(command, output/f'{name}-build.log')
        binary = dict(sha256=sha(output/name), bytes=(output/name).stat().st_size)
        binaries[name] = binary
        implementation = 'qwen_model_combined' if fused and (combined or default_combined) else ('qwen_model_fused' if fused else 'qwen_model_fast')
        if copy_free:
            implementation = 'qwen_model_buffer_swap' if fused else 'qwen_model_combined'
        provenance = dict(schema_version=1, operation=contract.OPERATION,
                          implementation=implementation,
                          entrypoint=contract.ENTRYPOINTS[implementation],
                          repository=source['repository'], source_sha256=source['sources'],
                          **machine, **contract.specification(prefix,fused,copy_free or (combined or default_combined) and fused,copy_free=copy_free and fused),
                          profile_warmup_iterations=10, profile_iterations=8,
                          profile_post_idle_milliseconds=250, binary=binary,
                          assets={k:v for k,v in identity.items() if k.endswith('_sha256')})
        contract.configuration(provenance)
        write(output/(name+'.provenance.json'), provenance)
    if source_identity() != source or assets(prepared) != identity:
        raise ValueError('source or assets changed during compilation')
    write(output/'build.json', dict(source=source, assets=identity, environment=machine,
                                    declaration=contract.COPY_FREE_DECLARATION if copy_free else (contract.COMBINED_DECLARATION if combined else (contract.FUSION_DECLARATION if fusion else contract.DECLARATION)), binaries=binaries))


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


def swap_capture_names():
    return ([f'hidden_{i}.bin' for i in range(25)]+['final_norm.bin','logits.bin']
            +[f'{stage}_{kind}_{i}.bin' for i in range(24) for stage in ('cache','append') for kind in ('key','value')])


def swap_lifecycle_names():
    return ([f'step-{i}' for i in (0,1,2,3,4,6,7,8)]
            +['final-'+name for name in ['logits']+[f'{k}{i}' for k in ('k','v') for i in range(24)]])


def validate_swap_checks(checks):
    if checks.get('owners_checked') is not True or checks.get('rejection_checked') is not True:
        raise ValueError('missing buffer ownership/lifecycle checks')
    for field,names in [('layers',swap_capture_names()),('lifecycle',swap_lifecycle_names())]:
        records=checks.get(field,[])
        if Counter(r['name'] for r in records)!=Counter(names) or any(
            r['exact'] is not True or r['bytes']<=0 or len(r['sha256'])!=64 for r in records):
            raise ValueError('incomplete or changed buffer-swap numerical evidence')
    states=checks.get('states',[])
    expected=[(0,3,3),(1,1,4),(2,1,5),(3,2,7),(4,1,8),(6,1,1),(7,2,3),(8,1,4)]
    if ([tuple(row[:3]) for row in states]!=expected
        or any(len(row)!=4 or not 0<=row[3]<151936 for row in states)):
        raise ValueError('buffer-swap lifecycle state changed')


def verify_swap_snapshots(directory):
    if (directory/'driver.log').read_text().count('SWAP_LIFECYCLE_COMPLETE')!=1:
        raise ValueError('native buffer-swap lifecycle incomplete')
    def compare(name,left,right,size):
        a=left.read_bytes(); b=right.read_bytes()
        if a!=b or len(a)!=size:
            raise ValueError('buffer-swap byte parity failed: '+name)
        return dict(name=name,exact=True,bytes=len(a),sha256=hashlib.sha256(a).hexdigest())
    layers=[]
    for name in swap_capture_names():
        size = 1048576 if name.startswith('cache_') else (256 if name.startswith('append_') else (303872 if name=='logits.bin' else 1792))
        layers.append(compare(name,directory/'layers-control'/name,directory/'layers-candidate'/name,size))
    lifecycle=[]
    for name in swap_lifecycle_names():
        if name.startswith('step-'):
            index=name.split('-')[1]
            left=directory/f'lifecycle-0-{index}.bin';right=directory/f'lifecycle-1-{index}.bin'
            size=303872
        else:
            tensor=name.removeprefix('final-')
            left=directory/f'lifecycle-final-0-{tensor}.bin';right=directory/f'lifecycle-final-1-{tensor}.bin'
            size=303872 if tensor=='logits' else 1048576
        lifecycle.append(compare(name,left,right,size))
    state=(directory/'lifecycle-0.txt').read_text()
    if state!=(directory/'lifecycle-1.txt').read_text():
        raise ValueError('buffer-swap lifecycle tokens/accounting changed')
    checks=dict(layers=layers,lifecycle=lifecycle,states=[list(map(int,line.split())) for line in state.splitlines()],
                owners_checked=True,rejection_checked=True)
    validate_swap_checks(checks)
    return checks


def collect(directory, output):
    ensure_record_location(output)
    receipt = verify_build(directory)
    output.mkdir(parents=True, exist_ok=False)
    args = receipt['assets']
    base = [directory/'model', 'verify', args['prepared'], args['tables']]
    numerical = []
    selection = receipt['declaration']==contract.SELECTION_DECLARATION
    copy_free = receipt['declaration']==contract.COPY_FREE_DECLARATION
    for prefix in contract.PREFIXES:
        for candidate in ([1,2] if selection else [0]):
            target = output/(f'verify-{prefix}'+(f'-s{candidate}' if selection else ''))
            target.mkdir()
            if copy_free and prefix == 64:
                for name in ('layers-control','layers-candidate'):
                    (target/name).mkdir()
            stdout = execute([*base, prefix, 0, candidate, target], target/'driver.log')
            if 'VERIFY_COMPLETE' not in stdout:
                raise ValueError('native verification incomplete')
            check = verify_snapshots(target, prefix)
            if selection:
                actual = []
                for observation in check['observations']:
                    name = observation['name']
                    path = target/f'actual-{name}.bin'
                    got = np.fromfile(path,dtype='<u2')
                    expected = np.full(151936,0x7FC0,dtype='<u2') if name=='logits' and candidate==2 else np.fromfile(target/f'plain-{name}.bin',dtype='<u2')
                    if not np.array_equal(got,expected):
                        raise ValueError('actual selection path changed storage')
                    actual.append(dict(name=name,exact=True,sha256=sha(path)))
                check.update(selection=candidate,actual=actual,nonfinite_invalidates=True)
            if copy_free and prefix == 64:
                check["swap_checks"] = verify_swap_snapshots(target)
            numerical.append(check)
    combined = receipt['declaration']==contract.COMBINED_DECLARATION
    samples, blocks = [], []
    for block in range(4):
        before = conditions()
        reverse = block in (1, 2)
        prefixes = list(reversed(contract.PREFIXES)) if reverse else contract.PREFIXES
        for prefix in prefixes:
            for comparison in (([3,2,1,0] if reverse else [0,1,2,3]) if selection else (([2,1,0] if reverse else [0,1,2]) if combined else ([1, 0] if reverse else [0, 1]))):
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


def curate(target, prefix, repeat, fused=None, combined=None):
    provenance = json.loads((target/'profile.provenance.json').read_text())
    if fused is None:
        fused = provenance['implementation'] != 'qwen_model_fast'
    if combined is None:
        combined = provenance['implementation'] == 'qwen_model_combined'
    copy_free = provenance['implementation']=='qwen_model_buffer_swap'
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
    stages = contract.command_stages(fused, combined and fused, selection, copy_free)
    count = len(stages)
    intervals, joined = coalesce_compute_commands(intervals, submissions, 18*count, join_resubmissions=True)
    if joined != report['validated_sequence']['interval_coalescing']:
        raise ValueError('curation differs from validated dispatch join')
    *_, measured = segment_compute_commands(intervals, 10, 8, count, False)
    contract.validate_command_sequence(measured,fused,combined and fused,selection,copy_free)
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
        stages = contract.command_stages(provenance['implementation']!='qwen_model_fast',
                                         provenance['implementation']=='qwen_model_combined')
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
    groups.append(('Other GPU operations',(set(s for _,s in contract.stages()) | set(s for _,s in contract.stages(True,True)))-used,'#abb9c7'))
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
    if receipt['declaration'] not in (contract.FUSION_DECLARATION,contract.COMBINED_DECLARATION,contract.COPY_FREE_DECLARATION):
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
        curated = curate(target,1024,0)
        write(target/'curated.json',curated)
        print('Verified fusion capture',name,len(curated['samples']),flush=True)
    verify_build(directory)


def fusion_terminal(directory, output):
    receipt = verify_build(directory)
    if receipt['declaration'] not in (contract.FUSION_DECLARATION,contract.COMBINED_DECLARATION,contract.COPY_FREE_DECLARATION):
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
                                    '',report,(('buffer-swap' if fused else 'combined') if receipt['declaration']==contract.COPY_FREE_DECLARATION else (('combined' if receipt['declaration']==contract.COMBINED_DECLARATION else 'fusion') if fused else 'unfused'))]))
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


def fusion_archive(timings, traces, terminal_path, output, combined=False, copy_free=False):
    stem = "buffer-swap" if copy_free else ("combined-fusion" if combined else "qkv-fusion")
    record = dict(kind='qwen-buffer-swap-v1' if copy_free else ('qwen-combined-fusion-v1' if combined else 'qwen-qkv-fusion-v1'),timing=json.loads((timings/'timings.json').read_text()),
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
    (output/(stem+'.json.gz')).write_bytes(packed)
    write(output/(stem+'.json'),dict(kind=record['kind'],sha256=hashlib.sha256(packed).hexdigest(),
                                      uncompressed_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(packed)))
    fusion_replay(output,combined,copy_free)


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


def selection_build(output, prepared):
    ensure_record_location(output)
    output.mkdir(parents=True,exist_ok=False)
    source = source_identity()
    if source['repository']['dirty']:
        raise ValueError('selection study requires clean source')
    identity = assets(prepared)
    machine = stable_environment()
    binaries = {}
    for name, entry, flags in [('model','benchmarks/model.mojo',['MODEL_SELECTION_STUDY']),
                               ('terminal','chat_cli.mojo',['MODEL_FUSION_STUDY'])] + [
        (f'profile-1024-s{s}','benchmarks/model.mojo',['MODEL_SELECTION_STUDY',f'MODEL_SELECTION_PROFILE={s}',
          'MODEL_PROFILE_PREFIX=1024','MODEL_PREPARED='+identity['prepared'],'MODEL_TABLES='+identity['tables']]) for s in range(3)]:
        execute([environment_tool('mojo'),'build','-I','src',*[item for flag in flags for item in ['-D',flag]],
                 'src/llm_mojo/'+entry,'-o',output/name],output/f'{name}-build.log')
        binary = dict(sha256=sha(output/name),bytes=(output/name).stat().st_size)
        binaries[name] = binary
        if name.startswith('profile'):
            selection = int(name[-1])
            implementation = ['qwen_model_combined','qwen_model_gpu_argmax','qwen_model_fused_head'][selection]
            provenance = dict(schema_version=1,operation=contract.OPERATION,implementation=implementation,
                entrypoint=contract.ENTRYPOINTS[implementation],repository=source['repository'],source_sha256=source['sources'],
                **machine,**contract.specification(1024,True,True,selection),profile_warmup_iterations=10,
                profile_iterations=8,profile_post_idle_milliseconds=250,binary=binary,
                assets={k:v for k,v in identity.items() if k.endswith('_sha256')})
            contract.configuration(provenance)
            write(output/(name+'.provenance.json'),provenance)
    if source_identity()!=source or assets(prepared)!=identity:
        raise ValueError('source or assets changed during selection build')
    write(output/'build.json',dict(source=source,assets=identity,environment=machine,
                                  declaration=contract.SELECTION_DECLARATION,binaries=binaries))


def selection_capture(directory, output):
    from .capture_trace import capture_trace
    receipt = verify_build(directory)
    if receipt['declaration'] != contract.SELECTION_DECLARATION:
        raise ValueError('not a token selection build')
    ensure_record_location(output)
    output.mkdir(parents=True,exist_ok=False)
    for selection in range(3):
        name = f'profile-1024-s{selection}'
        target = output/f's{selection}'
        target.mkdir()
        before = conditions()
        capture_trace(profile_binary=directory/name,output_trace=target/'raw.trace',
                      receipt_path=target/'capture.json',time_limit='30s')
        write(target/'conditions.json',dict(before=before,after=conditions()))
        (target/'profile.provenance.json').write_bytes((directory/(name+'.provenance.json')).read_bytes())
        export_trace(target)
        write(target/'curated.json',curate(target,1024,0))
        print('Verified selection trace',selection,flush=True)
    verify_build(directory)


def selection_terminal(directory, output):
    receipt = verify_build(directory)
    if receipt['declaration'] != contract.SELECTION_DECLARATION:
        raise ValueError('not a token selection build')
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
        for selection in ([2,1,0] if block in (1,2) else [0,1,2]):
            report = output/f'b{block}-s{selection}.tsv'
            command = list(map(str,[directory/'terminal',identity['prepared'],identity['tables'],128,256,
                                    '',report,['combined','gpu-argmax','fused-head'][selection]]))
            result = subprocess.run(command,cwd=repository_root(),env=environment(),input=inputs,
                                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
            (output/f'b{block}-s{selection}.txt').write_bytes(result.stdout)
            if result.returncode:
                raise ValueError('selection terminal failed')
            ev = events(report)
            turns = validate(ev,128)
            if len(turns)!=3:
                raise ValueError('missing selection terminal turns')
            arms.append(dict(selection=selection,events=ev,turns=turns,output_sha256=hashlib.sha256(result.stdout).hexdigest()))
        if len({a['output_sha256'] for a in arms})!=1 or any(
            a['generated']!=b['generated'] or a['history']!=b['history']
            for arm in arms[1:] for a,b in zip(arms[0]['turns'],arm['turns'])):
            raise ValueError('selection changed terminal text, tokens or history')
        blocks.append(dict(block=block,before=before,after=conditions(),arms=arms,output_exact=True))
        print('Verified selection terminal block',block+1,flush=True)
    verify_build(directory)
    write(output/'terminal.json',dict(build=receipt,prompts=prompts,blocks=blocks))


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


def selection_archive(timings, traces, terminal_path, output):
    record = dict(kind='qwen-token-selection-v1',timing=json.loads((timings/'timings.json').read_text()),
                  terminal=json.loads((terminal_path/'terminal.json').read_text()),
                  captures=[json.loads((traces/f's{s}/curated.json').read_text()) for s in range(3)])
    # Retain hashes/provenance while removing machine-specific model asset paths.
    for build_record in [record['timing']['build'],record['terminal']['build']]:
        for key in ('prepared','tables'):
            build_record['assets'].pop(key,None)
    output.mkdir(parents=True,exist_ok=True)
    raw = json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    packed = gzip.compress(raw,mtime=0)
    (output/'token-selection.json.gz').write_bytes(packed)
    write(output/'token-selection.json',dict(sha256=hashlib.sha256(packed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
    selection_replay(output)


def selection_plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=selection_replay(directory)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for index,context in enumerate(summary['contexts']):
        for comparison,offset,color,label in [(0,-.13,'#277eaa','GPU argmax / CPU'),(1,.13,'#188f82','Fused head / CPU')]:
            ratios=context['comparisons'][comparison]['ratios']
            axes[0].scatter([index+offset]*4,ratios,color=color,s=32,label=label if index==0 else None)
            axes[0].plot([index+offset-.08,index+offset+.08],[stats.median(ratios)]*2,color=color,linewidth=2)
        axes[0].plot([index-.35,index+.35],[1-context['noise_floor']]*2,color='#bd6a44',linewidth=1.5,
                     label='Required median threshold' if index==0 else None)
        direct=context['comparisons'][2]['ratios']
        axes[1].scatter([index]*4,direct,color='#6d63a6',s=32)
        axes[1].plot([index-.08,index+.08],[stats.median(direct)]*2,color='#6d63a6',linewidth=2)
        axes[1].plot([index-.35,index+.35],[1-context['noise_floor']]*2,color='#bd6a44',linewidth=1.5)
    for ax in axes:
        ax.axhline(1,color='#64798c',linestyle='--')
        ax.set_xticks(range(3),[str(r['prefix']) for r in summary['contexts']])
        ax.set_xlabel('Previously cached tokens')
        ax.set_ylabel('Paired complete-token latency ratio; lower is better')
    axes[0].set_title('Do GPU selectors improve the current Fast path?')
    axes[0].legend(fontsize=8)
    axes[1].set_title('Does fusing the head improve GPU argmax?')
    fig.suptitle('Qwen2.5-0.5B · BF16 · M4 Pro / Metal · four paired blocks',fontsize=13)
    fig.savefig(directory/'token-selection.png',dpi=170)
    plt.close(fig)


def selection_replay(directory):
    packed = (directory/'token-selection.json.gz').read_bytes()
    raw = gzip.decompress(packed)
    manifest = json.loads((directory/'token-selection.json').read_text())
    if manifest != dict(sha256=hashlib.sha256(packed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()):
        raise ValueError('selection evidence hash mismatch')
    record = json.loads(raw)
    timing = record['timing']
    build_record = timing['build']
    if record['kind']!='qwen-token-selection-v1' or build_record['declaration']!=contract.SELECTION_DECLARATION or record['terminal']['build']!=build_record:
        raise ValueError('selection build identity mismatch')
    summary = selection_summary(timing['samples'])
    if [b['block'] for b in timing['blocks']] != list(range(4)):
        raise ValueError('incomplete selection timing conditions')
    for block in [*timing['blocks'],*record['terminal']['blocks']]:
        for side in ('before','after'):
            require_ac(block[side])
            require_nominal_thermal_state(block[side])
            if block[side]['power_mode_raw'] != '0':
                raise ValueError('selection power mode changed')
    if Counter((n['prefix'],n['selection']) for n in timing['numerical'])!=Counter((p,s) for p in contract.PREFIXES for s in (1,2)):
        raise ValueError('incomplete selection numerical coverage')
    names = {'logits'}|{f'{k}{l}' for k in ('k','v') for l in range(24)}
    for check in timing['numerical']:
        if not check['nonfinite_invalidates'] or len(check['history'])!=check['prefix']+1:
            raise ValueError('invalid selection lifecycle evidence')
        for field in ('observations','actual'):
            observations = check[field]
            if len(observations)!=49 or {r['name'] for r in observations}!=names or not all(r['exact'] for r in observations):
                raise ValueError('selection numerical invariant failed')
        if not all(r['finite'] and r['prefix_exact'] and r['inactive_exact'] for r in check['observations']):
            raise ValueError('selection cache invariant failed')
    if len(record['captures'])!=3:
        raise ValueError('incomplete selection trace census')
    for selection,capture in enumerate(record['captures']):
        provenance = capture['provenance']
        implementation = ['qwen_model_combined','qwen_model_gpu_argmax','qwen_model_fused_head'][selection]
        contract.configuration(provenance)
        if (provenance['implementation']!=implementation or provenance['binary']!=build_record['binaries'][f'profile-1024-s{selection}']
            or provenance['repository']!=build_record['source']['repository'] or provenance['source_sha256']!=build_record['source']['sources']
            or provenance['assets']!={k:v for k,v in build_record['assets'].items() if k.endswith('_sha256')}):
            raise ValueError('selection trace provenance mismatch')
        stages = contract.command_stages(True,True,selection)
        if len(capture['samples'])!=8*len(stages):
            raise ValueError('incomplete selection command census')
        for i,row in enumerate(capture['samples']):
            if (row['iteration'],row['dispatch'],row['layer'],row['stage'],row['kind'])!=(i//len(stages),i%len(stages),*stages[i%len(stages)]):
                raise ValueError('selection trace stage changed')
            intervals=row['active_intervals']
            if row['duration_ns']<=0 or not intervals or sum(d for _,d in intervals)!=row['duration_ns'] or any(d<=0 for _,d in intervals):
                raise ValueError('invalid selection trace fragments')
    sys.path.insert(0,str(repository_root()/'tests'))
    from chat_terminal import validate
    blocks=record['terminal']['blocks']
    if [b['block'] for b in blocks]!=list(range(4)):
        raise ValueError('incomplete selection terminal blocks')
    for block in blocks:
        arms=block['arms']
        if len(arms)!=3 or {a['selection'] for a in arms}!={0,1,2} or len({a['output_sha256'] for a in arms})!=1:
            raise ValueError('selection terminal arms changed')
        for arm in arms:
            if validate(arm['events'],128)!=arm['turns'] or len(arm['turns'])!=3:
                raise ValueError('selection terminal events changed')
            if any(a['generated']!=b['generated'] or a['history']!=b['history'] for a,b in zip(arm['turns'],arms[0]['turns'])):
                raise ValueError('selection terminal tokens changed')
    write(directory/'token-selection-summary.json',summary)
    print(json.dumps(summary,indent=2))
    return summary


def main():
    # Trace capture and the reused terminal lifecycle helper inherit this process.
    os.environ.pop('MODULAR_DEBUG', None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build','collect','capture','terminal','archive','replay','plot','fusion-capture','fusion-terminal','fusion-archive','fusion-replay','fusion-plot','selection-capture','selection-terminal','selection-archive','selection-replay','selection-plot'])
    parser.add_argument('--copy-free',action='store_true',help='Compare buffer ownership swapping with inter-layer copies')
    parser.add_argument('--selection',action='store_true',help='Compare CPU, GPU argmax and fused vocabulary head')
    parser.add_argument('--combined', action='store_true', help='Combine QKV and SiLU/multiply fusion')
    parser.add_argument('--fusion', action='store_true', help='Build the bounded fusion experiment')
    parser.add_argument('--build', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timings', type=Path)
    parser.add_argument('--traces', type=Path)
    parser.add_argument('--terminal', type=Path)
    args = parser.parse_args()
    if args.command == 'build': build(args.output.resolve(), args.prepared, args.fusion, args.combined, args.selection, args.copy_free)
    elif args.command == 'selection-capture': selection_capture(args.build.resolve(),args.output.resolve())
    elif args.command == 'selection-terminal': selection_terminal(args.build.resolve(),args.output.resolve())
    elif args.command == 'selection-archive': selection_archive(args.timings,args.traces,args.terminal,args.output)
    elif args.command == 'selection-plot': selection_plot(args.output)
    elif args.command == 'selection-replay': selection_replay(args.output)
    elif args.command == 'fusion-capture': fusion_capture(args.build.resolve(),args.output.resolve())
    elif args.command == 'fusion-terminal': fusion_terminal(args.build.resolve(),args.output.resolve())
    elif args.command == 'fusion-archive': fusion_archive(args.timings,args.traces,args.terminal,args.output,args.combined,args.copy_free)
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
