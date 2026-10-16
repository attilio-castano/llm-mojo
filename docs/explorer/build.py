#!/usr/bin/env python3
"""Render the offline explorer from pinned, retained evidence (standard library only).

No model execution, downloads, or writes to historical archives occur here. The
small manifest records the exact excerpt selections and a pinned tokenizer subset.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path('docs/explorer')
MARKER = '__EXPLORER_DATA__'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_manifest(root=ROOT):
    return json.loads((Path(root) / DIRECTORY / 'recordings.json').read_text())


def load_asset(root, manifest, name):
    spec = manifest['assets'][name]
    path = Path(root) / spec['path']
    raw = path.read_bytes()
    require(digest(raw) == spec['sha256'], f"asset checksum mismatch: {spec['path']}")
    if spec['path'].endswith('.gz'):
        expanded = gzip.decompress(raw)
        require(digest(expanded) == spec['uncompressedSha256'],
                f"uncompressed asset checksum mismatch: {spec['path']}")
        if 'hashManifest' in spec:
            receipt = json.loads((Path(root) / spec['hashManifest']).read_text())
            require(receipt['sha256'] == spec['sha256']
                    and receipt['uncompressed_sha256'] == spec['uncompressedSha256'],
                    f"historical archive manifest mismatch: {spec['hashManifest']}")
        raw = expanded
    return json.loads(raw)


def source_records(root, manifest):
    result = {}
    for path, spec in manifest['sources'].items():
        raw = (Path(root) / path).read_bytes()
        require(digest(raw) == spec['sha256'], f'source checksum mismatch: {path}')
        lines = raw.decode().splitlines()
        start = spec.get('startLine', 1)
        end = spec.get('endLine', min(start + 69, len(lines)))
        require(type(start) is int and type(end) is int and 1 <= start <= end <= len(lines),
                f'invalid source excerpt range: {path}')
        result[path] = dict(path=path, revision=spec['revision'], sha256=spec['sha256'],
                            startLine=start, endLine=end,
                            excerpt='\n'.join(lines[start - 1:end]),
                            url=f"{manifest['repository']}/blob/{spec['revision']}/{path}#L{start}")
    return result


def provenance(record, paths=()):
    build = record.get('timing', record).get('build', {})
    env = build.get('environment', {})
    software = env.get('software', {})
    hardware = env.get('hardware', {})
    revision = build.get('source', {}).get('repository', {}).get('commit')
    tracked = build.get('source', {}).get('sources', {})
    if paths:
        source_paths = list(paths)
    elif 'src/llm_mojo/serving/engine.mojo' in tracked:
        source_paths = [p for p in (
            'src/llm_mojo/serving/engine.mojo', 'src/llm_mojo/serving/batch.mojo',
            'src/llm_mojo/serving/blocks.mojo', 'src/llm_mojo/serving/kv_pool.mojo',
            'src/llm_mojo/models/qwen2/runner.mojo', 'src/llm_mojo/models/qwen2/model.mojo',
        ) if p in tracked]
    else:
        # These campaigns predate the package reorganization. Link the paths
        # actually present in the measured build, rather than today's paths.
        source_paths = [p for p in (
            'src/llm_mojo/model.mojo', 'src/llm_mojo/decoder_layer.mojo',
            'src/llm_mojo/residual_norm.mojo', 'src/llm_mojo/token_selection.mojo',
            'src/llm_mojo/swiglu.mojo', 'src/llm_mojo/linear.mojo',
        ) if p in tracked]
    require(source_paths and all(p in tracked for p in source_paths), 'measured source link absent from build receipt')
    return dict(revision=revision, toolchain={'mojo': software.get('mojo'), 'max': software.get('max')},
                device=hardware.get('chip', 'Apple M4 Pro') + ' / ' + hardware.get('gpu_api', 'metal'),
                dtype='BF16 weights, stored boundaries and KV; FP32 reductions',
                timingBoundary='Token upload through greedy completion; preparation and allocation excluded',
                sourcePaths=source_paths,
                sourceUrls=[f'https://github.com/attilio-castano/llm-mojo/blob/{revision}/{p}' for p in source_paths])


def chat_data(record, manifest):
    arm = next(a for a in record['terminal']['blocks'][0]['arms'] if a['selection'] == 3)
    turn = next(t for t in arm['turns'] if t['turn'] == 1)
    events = [{k: int(v) if k in ('index', 'nanoseconds', 'turn') else v for k, v in e.items()}
              for e in arm['events'] if int(e['turn']) == 1]
    require(len(turn['prompt']) == 44 and len(turn['generated']) == 128, 'chat selection changed')
    require(turn['history'] == turn['prompt'] + turn['generated'] + [151645, 198], 'chat history disagrees')
    require(turn['cached'] == len(turn['prompt']) + len(turn['generated']) - 1,
            'chat pending-token accounting disagrees')
    pieces = manifest['tokenizer']['pieces']
    def tokens(ids):
        result = []
        for token in ids:
            require(str(token) in pieces, f'tokenizer subset misses ID {token}')
            raw = bytes.fromhex(pieces[str(token)])
            result.append(dict(id=token, text=raw.decode('utf-8', errors='replace'), bytes=list(raw)))
        return result
    generated = tokens(turn['generated'])
    prompt = tokens(turn['prompt'])
    chat_provenance = provenance({'build': record['terminal']['build']}, (
        'src/llm_mojo/chat.mojo', 'src/llm_mojo/tokenizer.mojo',
        'src/llm_mojo/model.mojo', 'src/llm_mojo/token_selection.mojo'))
    chat_provenance['timingBoundary'] = (
        'Native terminal events from turn begin: first-token and first-visible timestamps include prefill with resident weights; '
        'subsequent output rate excludes the first token and includes detokenization and flush. Loading is separate.')
    return dict(prompt=record['terminal']['prompts'][0], promptIds=turn['prompt'],
                generatedIds=turn['generated'], historyIds=turn['history'], promptTokens=prompt,
                closingIds=[151645, 198],
                tokens=generated, text=b''.join(bytes(t['bytes']) for t in generated).decode('utf-8'),
                events=events, cachedBefore=turn['cached_before'], cached=turn['cached'],
                firstTokenMs=turn['first_token_ms'], firstVisibleMs=turn['first_visible_ms'],
                reason=turn['reason'], selection=3, asset=manifest['assets']['residualNorm']['path'],
                provenance=chat_provenance,
                textProvenance='Derived by decoding recorded IDs with the pinned tokenizer byte vocabulary; event timestamps and byte counts are recorded.')


def serving_data(record, manifest):
    run = next(r for r in record['runs'] if r['arm'] == 'chunked' and not r['calibration'])
    parsed = run['parsed']
    require(len(parsed['steps']) == 46, 'serving selection changed')
    require(parsed['drained']['free_blocks'] == parsed['blocks'] == 128, 'serving pool did not drain')
    require(len(parsed['events']) == 264 and len(record['trace']['requests']) == 8, 'serving events changed')
    require([s['step_id'] for s in parsed['steps']] == list(range(46)), 'serving steps are incomplete')
    require(all(s['total_tokens'] == s['decode_seqs'] + s['prefill_tokens'] <= 256
                for s in parsed['steps']), 'serving token budget disagrees')
    proof = provenance(record)
    proof['timingBoundary'] = record['declaration']['timing_boundary']
    return dict(arm='chunked', device=parsed['device'], mode=parsed['mode'],
                blocks=parsed['blocks'], blockSize=record['declaration']['block_size'],
                tokenBudget=parsed['token_budget'], maxSequences=parsed['max_sequences'],
                requests=record['trace']['requests'], arrivals=parsed['arrivals'],
                steps=parsed['steps'], events=parsed['events'], drained=parsed['drained'],
                traceSha256=record['trace_sha256'], asset=manifest['assets']['engineOffline']['path'],
                policy=record['declaration']['policy'], provenance=proof,
                jumps=[0, 1, 3, 5, 15, next(s['step_id'] for s in parsed['steps'] if s['finished']), 45],
                scope='Recorded Metal run with frozen synthetic token prompts; reference configuration 27, synchronous stepping. Physical block identities are not recorded.')


def paired_contexts(record):
    """Recompute the campaign's four-block paired statistic from raw samples."""
    samples = record['timing']['samples']
    contexts = []
    for prefix in sorted({s['prefix'] for s in samples}):
        comparisons = []
        for comparison in sorted({s['comparison'] for s in samples if s['prefix'] == prefix}):
            selected = [s for s in samples if s['prefix'] == prefix and s['comparison'] == comparison]
            arms = sorted({s['arm'] for s in selected})
            # Self pairs use two encoded arms 0 and 1, like every other comparison.
            require(len(arms) == 2, 'paired timing requires two arms')
            blocks = sorted({s['block'] for s in selected})
            require(blocks == [0, 1, 2, 3], 'timing block missing')
            values = []
            for arm in arms:
                medians = []
                for block in blocks:
                    group = [s for s in selected if s['arm'] == arm and s['block'] == block]
                    require(len(group) == 10 and sorted(s['sample'] for s in group) == list(range(10)),
                            'timing sample missing')
                    medians.append(statistics.median(s['elapsed_ns'] for s in group) / 1e6)
                values.append(medians)
            ratios = [b/a for a, b in zip(*values)]
            comparisons.append(dict(comparison=comparison, arms=arms, ratios=ratios,
                                    control_block_ms=values[0], candidate_block_ms=values[1],
                                    median_reduction=1-statistics.median(ratios)))
        calibration = comparisons[0]['ratios']
        floor = max(0.05, max(abs(r-1) for r in calibration))
        for c in comparisons:
            c['qualifies'] = all(r < 1 for r in c['ratios']) and c['median_reduction'] > floor
        contexts.append(dict(prefix=prefix, noise_floor=floor, comparisons=comparisons,
                             calibration_ratios=calibration))
    return contexts


def close(a, b):
    return abs(a-b) < 1e-11


def check_summary(contexts, summary, projection=False):
    require(len(contexts) == len(summary['contexts']) == 3, 'summary context count changed')
    for got, expected in zip(contexts, summary['contexts']):
        require(got['prefix'] == expected['prefix'] and close(got['noise_floor'], expected['noise_floor']),
                'retained summary calibration disagrees with raw timing')
        for c in expected['comparisons']:
            identity = c['variant'] if projection else c['comparison']
            actual = next(x for x in got['comparisons'] if x['comparison'] == identity)
            require(all(close(a, b) for a, b in zip(actual['ratios'], c['ratios']))
                    and close(actual['median_reduction'], c['median_reduction'])
                    and actual['qualifies'] == c['qualifies'], 'retained summary disagrees with raw timing')


def study_data(accepted, rejected, accepted_summary, rejected_summary, manifest):
    a = paired_contexts(accepted)
    r = paired_contexts(rejected)
    check_summary(a, accepted_summary)
    check_summary(r, rejected_summary, True)
    require(accepted_summary['selected'] == 3 and not rejected_summary['promote'], 'study decision changed')
    def row(context, comparison, label):
        c = next(c for c in context['comparisons'] if c['comparison'] == comparison)
        return dict(history=context['prefix'], label=label,
                    controlMs=statistics.median(c['control_block_ms']),
                    candidateMs=statistics.median(c['candidate_block_ms']),
                    reductionPct=c['median_reduction']*100, requiredPct=context['noise_floor']*100,
                    pairedRatios=c['ratios'], qualifies=c['qualifies'])
    def streaming(record, selection):
        return [statistics.median(next(a for a in b['arms'] if a['selection'] == selection)['turns'][i]
                                  ['output_tokens_per_second'] for b in record['terminal']['blocks'])
                for i in range(3)]
    accepted_rows = [row(c, 3, 'Residual/RMSNorm + owner swap + GPU argmax') for c in a]
    # Exact printed report values are checked independently of summary JSON.
    require([round(x['reductionPct'], 2) for x in accepted_rows] == [22.98, 24.51, 17.08],
            'accepted report table changed')
    labels = {c['variant']: c['label'] for c in rejected_summary['contexts'][0]['comparisons']}
    rejected_rows = []
    expected = [[2.43, .56, 8.77], [-2.16, .01, -.22], [.50, -.66, -.75],
                [2.20, 2.55, 8.75], [1.35, -.03, 8.13]]
    for i in range(1, 6):
        cells = [row(c, i, labels[i]) for c in r]
        require([round(c['reductionPct'], 2) for c in cells] == expected[i-1], 'rejected report table changed')
        rejected_rows.append(dict(id=i, label=labels[i], reductionsPct=[c['reductionPct'] for c in cells],
                                  qualifies=all(c['qualifies'] for c in cells), cells=cells))
    return dict(accepted=dict(title='Compose fewer launches without changing stored values',
                             decision='Promoted', rows=accepted_rows, streamingTokensPerSecond=streaming(accepted, 3),
                             commandsBefore=314, commandsAfter=245, numericalComparisons=len(accepted['timing']['numerical']),
                             mechanism='Fuse 48 residual/RMSNorm boundaries; swap 23 inter-layer owners; select the token on GPU.',
                             provenance=provenance(accepted), asset=manifest['assets']['residualNorm']['path']),
                rejected=dict(title='A faster projection kernel misses the whole-token gate', decision='Not promoted',
                              rows=rejected_rows, histories=[c['prefix'] for c in r],
                              requiredPct=[c['noise_floor']*100 for c in r],
                              streamingTokensPerSecond=streaming(rejected, 1),
                              numericalComparisons=len(rejected['timing']['numerical']),
                              explanation='Fixed-width scheduling helped traced projection work, but short and medium complete-token gains missed the declared minimum. Correctness checks passed.',
                              provenance=provenance(rejected), asset=manifest['assets']['projectionArrangements']['path']))


def integer_expression(text, variables):
    node = ast.parse(text.strip(), mode='eval').body
    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.Name) and n.id in variables:
            return variables[n.id]
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)):
            a, b = visit(n.left), visit(n.right)
            return a+b if isinstance(n.op, ast.Add) else a-b if isinstance(n.op, ast.Sub) else a*b
        raise ValueError('unsupported expression in closed plan fixture: ' + text)
    return visit(node)


def derive_plans(root):
    """Interpret only the tiny constructor language actually used by the fixtures."""
    paths = ['tests/reduction_plan_cases.mojo', 'tests/compile_fail/reduction_plan_missing_input.mojo']
    functions = {}
    for path in paths:
        source = (Path(root) / path).read_text()
        matches = list(re.finditer(r'^def (\w+)\(\) -> ArgmaxPlan\[(\d+)\]:\n', source, re.M))
        for match in matches:
            following = source[match.end():]
            body = re.split(r'\n(?=def )', following, maxsplit=1)[0]
            functions[match[1]] = (int(match[2]), body.splitlines())
    def make(name):
        size, lines = functions[name]
        state, variables = {}, {}
        def run(block):
            i = 0
            while i < len(block):
                line = block[i].strip()
                i += 1
                if not line or line.startswith('#'):
                    continue
                loop = re.fullmatch(r'for (\w+) in range\((.+)\):', line)
                if loop:
                    indent = len(block[i-1]) - len(block[i-1].lstrip())
                    nested = []
                    while i < len(block) and (not block[i].strip() or len(block[i])-len(block[i].lstrip()) > indent):
                        nested.append(block[i]); i += 1
                    for value in range(integer_expression(loop[2], variables)):
                        variables[loop[1]] = value; run(nested)
                    continue
                if re.fullmatch(r'var nodes = Array\[ReductionNode, \d+\]\(uninitialized=True\)', line):
                    state['nodes'] = [None]*size
                    continue
                base = re.fullmatch(r'var plan = (\w+)\(\)', line)
                if base:
                    state.update(copy.deepcopy(make(base[1]))); continue
                assignment = re.fullmatch(r'(?:plan\.)?nodes\[(.+)\] = (input_node|empty_node|merge_node)\((.*)\)', line)
                if assignment:
                    args = [integer_expression(a, variables) for a in assignment[3].split(',') if a.strip()]
                    value = (1,args[0],0) if assignment[2]=='input_node' else (0,0,0) if assignment[2]=='empty_node' else (2,*args)
                    state['nodes'][integer_expression(assignment[1], variables)] = value
                    continue
                root_assignment = re.fullmatch(r'plan.root = (\d+)', line)
                if root_assignment:
                    state['root'] = int(root_assignment[1]); continue
                returned = re.fullmatch(r'return ArgmaxPlan\[\d+\]\(nodes\^, (\d+)\)', line)
                if returned:
                    state['root'] = int(returned[1]); continue
                if line == 'return plan^':
                    continue
                raise ValueError('unsupported closed-plan fixture statement: ' + line)
        run(lines)
        require(len(state['nodes']) == size and all(n is not None for n in state['nodes']), 'incomplete plan')
        return state
    result = []
    labels = [('sequential_plan','Sequential'), ('balanced_plan','Balanced'),
              ('repeated_permuted_plan','Repeated and permuted'), ('earlier_root_plan','Earlier complete root'),
              ('invalid_plan','Missing input seven')]
    for name, label in labels:
        plan = make(name)
        supports = []
        nodes = []
        for index, (operation,left,right) in enumerate(plan['nodes']):
            if operation == 0:
                require(left == right == 0, 'malformed empty node'); support = set()
            elif operation == 1:
                require(0 <= left < 8 and right == 0, 'malformed input node'); support = {left}
            else:
                require(operation == 2 and 0 <= left < index and 0 <= right < index, 'malformed merge node')
                support = supports[left] | supports[right]
            supports.append(support)
            nodes.append(dict(id=index, kind=['empty','input','merge'][operation], input=left if operation==1 else None,
                              left=left, right=right, tuple=[operation,left,right], support=sorted(support)))
        support = sorted(supports[plan['root']])
        accepted = support == list(range(8))
        require(accepted == (name != 'invalid_plan'), 'fixture acceptance changed')
        result.append(dict(id=name, name=label, accepted=accepted, root=plan['root'], nodes=nodes,
                           support=support, missing=sorted(set(range(8))-set(support))))
    require(result[0]['nodes'][9]['tuple'] == [2,8,0] and result[1]['nodes'][16]['tuple'] == [2,15,8]
            and result[2]['nodes'][16]['tuple'] == [2,15,15] and result[3]['root'] == 15
            and result[4]['missing'] == [7], 'expected plan edges changed')
    test_source = (Path(root) / 'tests/test_reduction_plan.mojo').read_text().split('def main() raises:', 1)[1]
    initial_inputs = test_source.split('var inputs:', 1)[1].split('= [', 1)[1].split(']', 1)[0]
    inputs = [dict(rank=int(rank), token=int(token), invalid=int(invalid))
              for rank, token, invalid in re.findall(r'ArgmaxRecord\((\d+), (\d+), (\d+)\)', initial_inputs)]
    require(len(inputs) == 8, 'reduction fixture input records changed')
    return dict(inputCount=8, plans=result, inputs=inputs, expected=dict(rank=3, token=31, invalid=1),
                contract='Every node must be well formed, references go backward, and the selected root covers all eight symbolic inputs. Repeated inputs are allowed by idempotence.',
                scope='The browser displays fixture-derived plans. Mojo compile-time acceptance and rejection are separate compiler evidence; this view does not prove GPU or full-model behavior.')


def contract_receipt(root):
    path = Path(root) / DIRECTORY / 'contract-receipt.json'
    receipt = json.loads(path.read_text())
    require(receipt['schema_version'] == 1 and receipt['status'] in ('passed', 'failed'),
            'unknown compiler receipt schema or status')
    require(receipt['sources_unchanged_during_run'] is True, 'compiler receipt sources changed during run')
    require(receipt['sources'] and receipt['tests_run'] > 0, 'compiler receipt has no executed checks')
    for source, expected in receipt['sources'].items():
        require(digest((Path(root) / source).read_bytes()) == expected,
                f'compiler receipt source checksum mismatch: {source}')
    if receipt['status'] == 'passed':
        require(all(r['status'] == 'passed' for r in receipt['results']),
                'compiler receipt claims a pass with incomplete results')
        executions = receipt['executions']
        for accepted in ('tests/test_reduction_plan.mojo', 'tests/test_token_selection_contract.mojo'):
            for stage in ('compile', 'execute'):
                require(any(e['source'] == accepted and e['stage'] == stage and e['exit_code'] == 0
                            for e in executions), 'compiler receipt lacks accepted compilation and host execution')
        expected_negative = {
            **{f'tests/compile_fail/reduction_plan_{name}.mojo': 'argmax reduction plan rejected'
               for name in ('missing_input', 'foreign_input', 'forward_reference', 'negative_reference',
                            'unsupported_operation', 'invalid_root')},
            'tests/compile_fail/token_selection_signed_zero.mojo': 'BF16 ordering contract failed',
            'tests/compile_fail/token_selection_ignored_nan.mojo': 'BF16 classification contract failed',
            'tests/compile_fail/token_selection_larger_token.mojo': 'argmax representative laws failed',
            'tests/compile_fail/token_selection_dropped_invalid.mojo': 'argmax representative laws failed',
        }
        for rejected, diagnostic in expected_negative.items():
            require(any(e['source'] == rejected and e['stage'] == 'compile' and e['exit_code'] != 0
                        and e.get('diagnostic') == diagnostic for e in executions),
                    f'compiler receipt lacks intended rejection: {rejected}')
    return dict(path=str(DIRECTORY / 'contract-receipt.json'), sha256=digest(path.read_bytes()),
                status=receipt['status'], checkedAt=receipt['checked_at'],
                scope=receipt['scope'], sourceRevision=receipt['source_commit'],
                toolchain=receipt['toolchain'], testsRun=receipt['tests_run'],
                results=receipt['results'], executions=receipt['executions'], limits=receipt['limits'])


def build_data(root=ROOT):
    root = Path(root)
    manifest = load_manifest(root)
    sources = source_records(root, manifest)
    # Verify every declared asset, including report summaries and original hash manifests.
    assets = {name: load_asset(root, manifest, name) for name in manifest['assets']}
    tokenizer = manifest['tokenizer']
    require(digest(json.dumps(tokenizer['pieces'], sort_keys=True, separators=(',', ':')).encode())
            == tokenizer['subsetSha256'], 'tokenizer subset checksum mismatch')
    accepted, rejected = assets['residualNorm'], assets['projectionArrangements']
    proof = derive_plans(root)
    proof['receipt'] = contract_receipt(root)
    return dict(meta=dict(revision=manifest['sourceRevision'], model='Qwen2.5-0.5B-Instruct',
                          device='Apple M4 Pro / Metal', recordedToolchain='Mojo 1.0.0 / MAX 26.5.0',
                          generatedFrom='Pinned recorded archives; no live inference in this page',
                          sourceRepository=manifest['repository']), sources=sources,
                chat=chat_data(accepted, manifest), serving=serving_data(assets['engineOffline'], manifest),
                studies=study_data(accepted, rejected, assets['residualNormSummary'],
                                   assets['projectionArrangementsSummary'], manifest), proof=proof)


def render(root=ROOT):
    root = Path(root)
    template = (root / DIRECTORY / 'explorer.template.html').read_text()
    require(template.count(MARKER) == 1, 'template must contain exactly one data marker')
    manifest = load_manifest(root)
    vendor = manifest['vendor']
    source = (root / vendor['sourcePath']).read_bytes()
    license_text = (root / vendor['licensePath']).read_bytes()
    require(digest(source) == vendor['sourceSha256'] and digest(license_text) == vendor['licenseSha256'],
            'vendored Pretext source or license checksum mismatch')
    require(template.count('__PRETEXT_SOURCE__') == template.count('__PRETEXT_LICENSE__') == 1,
            'template must contain exactly one Pretext source and license marker')
    module = source.decode()
    require('X1 as prepare' in module and 'Z1 as layout' in module and '</script' not in module.lower(),
            'vendored Pretext export bindings changed')
    module += '\n;globalThis.Pretext={prepare:X1,layout:Z1};window.dispatchEvent(new Event("pretext-ready"));\n'
    template = template.replace('__PRETEXT_SOURCE__', module).replace('__PRETEXT_LICENSE__', license_text.decode())
    data = json.dumps(build_data(root), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    # JSON lives in a script element: neutralize HTML closing sequences and JS line separators.
    data = data.replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    return template.replace(MARKER, data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='fail if sources, evidence or generated HTML changed')
    modes.add_argument('--data-only', action='store_true', help='print compact derived JSON without writing files')
    args = parser.parse_args(argv)
    try:
        if args.data_only:
            print(json.dumps(build_data(), ensure_ascii=False, sort_keys=True, separators=(',', ':')))
        else:
            output = ROOT / DIRECTORY / 'index.html'
            expected = render()
            if args.check:
                require(output.exists() and output.read_text() == expected, 'generated explorer is stale; run python3 docs/explorer/build.py')
                print('Explorer sources, recordings and generated HTML match.')
            else:
                output.write_text(expected)
                print(f'Wrote {output}')
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f'explorer: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
