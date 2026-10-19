#!/usr/bin/env python3
"""Reproduce the independent prefix-cache audit without GPU work or file writes.

Run from the measured checkout using its locked environment, for example:
  .venv/bin/python -B build/prefix-cache/independent-audit/reproduce.py \
    /private/tmp/llm-mojo-prefix-20261018-v3-measurements/engine-prefix.json.gz

The archive's sibling manifest and summary must be present. JSON goes to stdout.
The recorded native engine binary and source bytes must still be available.
This helper supplements the sealed archive; it does not modify that archive.
"""

import argparse
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

from llm_mojo.benchmarks import engine_prefix


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def history(phase, request_ids):
    result = []
    for request_id in request_ids:
        events = [e for e in phase['events'] if e['request_id'] == request_id]
        finishes = [e['reason'] for e in events if e['kind'] == 'finish']
        require(len(finishes) == 1, 'exactly one finish per request')
        result.append({
            'request_id': request_id,
            'tokens': [e['token_id'] for e in events if e['kind'] == 'token'],
            'reason': finishes[0],
        })
    return result


def audit(archive, source_root):
    compressed = archive.read_bytes()
    payload = gzip.decompress(compressed)
    record = json.loads(payload)
    manifest = json.loads(archive.with_suffix('').read_text())
    standalone = json.loads((archive.parent / 'engine-prefix-summary.json').read_text())
    require(manifest['bytes'] == len(compressed), 'compressed byte count')
    require(manifest['sha256'] == digest(compressed), 'compressed SHA256')
    require(manifest['uncompressed_sha256'] == digest(payload), 'uncompressed SHA256')
    computed = engine_prefix.summary(record)
    require(computed == record['summary'] == standalone, 'strict summary replay')
    require(computed['measured_runs'] == 96, '96 timed invocations')
    require(computed['untimed_preflights'] == 16, '16 untimed preflights')

    qualification = json.loads(record['qualification_document'])
    source = record['build']['source']
    require(source['repository']['dirty'] is False, 'measured clean source')
    require(
        qualification['build']['source'] == source
        == qualification['checkpoint_receipt']['build']['source'],
        'qualification/checkpoint/timing source binding',
    )
    for path, expected in source['sources'].items():
        require(digest((source_root / path).read_bytes()) == expected,
                'live source bytes: ' + path)
    binary = Path(record['build']['command'][-1])
    binary_info = record['build']['binaries']['engine']
    require(binary.stat().st_size == binary_info['bytes'], 'native binary byte count')
    require(digest(binary.read_bytes()) == binary_info['sha256'], 'native binary SHA256')
    require(digest(record['qualification_document'].encode())
            == record['qualification_sha256'], 'qualification document SHA256')

    expected_grid = {
        'prefix-%d-reuse-%d-blocks-%d' % cell
        for cell in ((256, 0, 384), (256, 4, 384), (256, 8, 384),
                     (1024, 0, 384), (1024, 4, 384), (1024, 8, 384),
                     (1024, 4, 40), (1024, 8, 40))
    }
    require({c['cell_id'] for c in record['cells']} == expected_grid, 'frozen eight cells')
    namespace = 'qwen-reference27-projection8/' + record['build']['assets']['prepared']
    counts = collections.Counter()
    preemptions = collections.Counter()
    conditions = []
    devices = set()
    ownership_snapshots = raw_hits = history_runs = work_runs = 0
    cells = []

    for cell, summary in zip(record['cells'], computed['cells']):
        cell_id = cell['cell_id']
        require(cell_id == summary['cell_id'], 'cell summary order')
        require(len(cell['runs']) == 12, '12 timed invocations per cell')
        requests = {r['request_id']: r for r in cell['trace']['requests']}
        for block in range(4):
            actual = [(r['cache'], r['calibration']) for r in cell['runs']
                      if r['block'] == block]
            expected = [('disabled', False), ('disabled', True), ('enabled', False)]
            if block in (1, 2):
                expected.reverse()
            require(actual == expected, 'frozen alternating arm order')
        enabled, disabled, controls = [], [], []
        grouped, cleanup = {}, {}
        input_histories = [
            requests[h['request_id']]['prompt_ids'] + h['tokens'][:-1]
            for h in summary['frozen_histories']['measure']
        ]

        for raw, metrics in zip(cell['runs'], summary['runs']):
            key = (metrics['block'], metrics['cache'], metrics['calibration'])
            require(key == (raw['block'], raw['cache'], raw['calibration']), 'run order')
            require(raw['native_execution']['exit_code'] == 0, 'native exit zero')
            parsed = raw['parsed']
            devices.add(parsed['device'])
            measure = parsed['phases']['measure']
            histories = history(measure, requests)
            require(histories == summary['frozen_histories']['measure'],
                    'literal measured histories')
            require(history(parsed['phases']['seed'], [128])
                    == summary['frozen_histories']['seed'], 'literal seed history')
            history_runs += 1
            hits = {h['request_id']: h['cached_tokens'] for h in measure['hits']}
            require(len(hits) == len(measure['hits']), 'unique hit per request')
            rows = attended = 0
            for h in histories:
                request_id = h['request_id']
                total = len(requests[request_id]['prompt_ids']) + len(h['tokens']) - 1
                cached = hits.get(request_id, 0)
                rows += total - cached
                attended += total * (total + 1) // 2 - cached * (cached + 1) // 2
            require(rows == sum(s['total_tokens'] for s in measure['steps'])
                    == metrics['total_tokens'], 'independent executed row formula')
            require(attended == sum(s['attended_positions'] for s in measure['steps'])
                    == metrics['attended_positions'], 'independent attention formula')
            work_runs += 1
            for phase in parsed['phases'].values():
                for state in phase['cache_states']:
                    require(state['free_blocks'] + state['cached_blocks']
                            + state['private_blocks'] == parsed['blocks'], 'pool partition')
                    ownership_snapshots += 1
                drained = phase['drained']
                require(drained['references'] == drained['private_blocks']
                        == drained['pending'] == 0, 'drain ownership')
            times = parsed['phase_times']
            require(metrics['episode_ns'] == times['measure']['end_ns']
                    - times['seed']['begin_ns'], 'episode timing boundary')
            cleanup[key] = times['clear']['end_ns'] - times['seed']['begin_ns']
            require(metrics['cache_cleared'], 'final cache clear')
            preemptions[metrics['preemptions']] += 1
            conditions.extend((raw['conditions_before'], raw['conditions_after']))
            grouped[key] = metrics
            (enabled if metrics['cache'] == 'enabled' else
             controls if metrics['calibration'] else disabled).append(metrics)

            # Independent lexer/replay of the native stdout protocol. This does
            # not consume engine_prefix.parse_run or its parsed event objects.
            resident, event_id, phase, seed_rows = set(), 0, None, 0
            for line in raw['stdout'].splitlines():
                fields = line.split()
                tag = fields[0]
                if tag == 'phase':
                    phase = fields[1]
                elif tag == 'namespace':
                    require(fields[1] == namespace, 'runtime namespace binding')
                elif tag == 'step' and phase == 'seed':
                    seed_rows += int(fields[5])
                elif tag == 'cache_event':
                    require(len(fields) == 7 and int(fields[1]) == event_id,
                            'monotonic cache event IDs')
                    event_id += 1
                    kind = fields[3]
                    counts[kind] += 1
                    if kind == 'cleared':
                        require(fields[4:] == ['-', '-', '-'] and not resident,
                                'global clear event')
                        continue
                    cache_key = (fields[4], fields[5], tuple(map(int, fields[6].split(','))))
                    require(cache_key[0] == namespace and cache_key[1] == 'prefix-study'
                            and len(cache_key[2]) % 32 == 0, 'exact cache identity')
                    if kind == 'stored':
                        require(cache_key not in resident, 'unique resident key')
                        if phase == 'seed':
                            require(len(cache_key[2]) <= seed_rows,
                                    'seed completed row publication bound')
                        else:
                            require(phase == 'measure' and any(
                                tuple(v[:len(cache_key[2])]) == cache_key[2]
                                for v in input_histories), 'known measured input prefix')
                        resident.add(cache_key)
                    else:
                        require(kind == 'removed' and cache_key in resident,
                                'remove resident key exactly once')
                        resident.remove(cache_key)
                elif tag == 'cache_hit':
                    request_id, length = int(fields[2]), int(fields[3])
                    raw_hits += 1
                    for end in range(32, length + 1, 32):
                        require((namespace, 'prefix-study',
                                 tuple(requests[request_id]['prompt_ids'][:end])) in resident,
                                'every hit ancestor resident')
                elif tag == 'cache_state':
                    require(int(fields[3]) == len(resident), 'raw index occupancy')
                elif tag == 'cache_cleared':
                    require(fields[1:] == [str(cell['trace']['prefix']['blocks']),
                                          '0', '0', '0', '0', '0']
                            and not resident, 'raw final clear ownership')
            require(not resident, 'no residual cache identity')

        comparisons = {}
        for metric in ('duration_ns', 'episode_ns'):
            ratios = [grouped[(b, 'enabled', False)][metric]
                      / grouped[(b, 'disabled', False)][metric] for b in range(4)]
            self_ratios = [grouped[(b, 'disabled', True)][metric]
                           / grouped[(b, 'disabled', False)][metric] for b in range(4)]
            noise = max(.05, max(abs(r - 1) for r in self_ratios))
            median = statistics.median(ratios)
            verdict = ('faster' if all(r < 1 for r in ratios) and 1 - median > noise
                       else 'slower' if all(r > 1 for r in ratios) and median - 1 > noise
                       else 'inconclusive')
            comparisons[metric] = {
                'block_ratios': ratios, 'median_ratio': median,
                'calibration_ratios': self_ratios, 'noise_floor': noise,
                'speed_verdict': verdict,
            }
            require(comparisons[metric] == summary['comparisons'][metric],
                    'independent paired ratios/noise/verdict')
        first_enabled, first_disabled = enabled[0], disabled[0]
        hit_sets = [tuple((q['request_id'], q['cached_prompt_tokens']) for q in m['requests']
                         if q['cached_prompt_tokens']) for m in enabled]
        require(all(h == hit_sets[0] for h in hit_sets), 'repeatable actual hit identities')
        require(all(m['total_tokens'] == first_enabled['total_tokens']
                    and m['cache_hit_tokens'] == first_enabled['cache_hit_tokens']
                    and m['evictions'] == first_enabled['evictions'] for m in enabled),
                'repeatable actual work and eviction counts')
        require(all(m['cache_hit_tokens'] == 0 for m in disabled + controls), 'disabled no hits')
        cells.append({
            'cell_id': cell_id,
            'disabled_median_duration_ns': statistics.median(m['duration_ns'] for m in disabled),
            'enabled_median_duration_ns': statistics.median(m['duration_ns'] for m in enabled),
            'comparisons': comparisons,
            'actual_hits': [dict(request_id=i, cached_tokens=n) for i, n in hit_sets[0]],
            'enabled_hit_tokens': first_enabled['cache_hit_tokens'],
            'rows_disabled': first_disabled['total_tokens'],
            'rows_enabled': first_enabled['total_tokens'],
            'attention_disabled': first_disabled['attended_positions'],
            'attention_enabled': first_enabled['attended_positions'],
            'enabled_evictions': first_enabled['evictions'],
            'enabled_peak_cached_blocks': first_enabled['peak_cached_blocks'],
            'enabled_peak_private_blocks': first_enabled['peak_private_blocks'],
            'enabled_peak_references': first_enabled['peak_references'],
            'enabled_clear_ns_range': [min(m['clear_ns'] for m in enabled),
                                       max(m['clear_ns'] for m in enabled)],
            'exploratory_cleanup_inclusive_median_ratio': statistics.median(
                cleanup[(b, 'enabled', False)] / cleanup[(b, 'disabled', False)]
                for b in range(4)),
        })

    return {
        'kind': 'engine-prefix-independent-audit-v1',
        'status': 'passed',
        'blocking_findings': [],
        'archive': {
            'path': str(archive), 'compressed_bytes': len(compressed),
            'sha256': digest(compressed), 'uncompressed_sha256': digest(payload),
            'manifest_and_strict_summary_replay': 'passed',
        },
        'source': {
            'commit': source['repository']['commit'], 'clean': True,
            'live_source_hashes_verified': len(source['sources']),
            'qualification_checkpoint_timing_identity_equal': True,
            'native_engine': {'path': str(binary), **binary_info},
        },
        'checks': {
            'frozen_cells': 8, 'measured_runs': 96, 'untimed_preflights': 16,
            'frozen_grid_and_arm_order': 'passed',
            'exact_seed_and_measure_history_runs': history_runs,
            'independent_actual_work_formula_runs': work_runs,
            'row_formula': 'sum(P + G - 1 - H)',
            'attention_formula': 'sum(T*(T+1)/2 - H*(H+1)/2), T=P+G-1',
            'formula_symbols': {'P': 'prompt tokens', 'G': 'delivered tokens',
                                'H': 'actual cached prompt tokens'},
            'bounded_ownership_snapshots': ownership_snapshots,
            'all_drains_refs_private_pending_zero': True,
            'raw_stdout_logical_events': dict(counts),
            'raw_hit_receipts_with_every_exact_ancestor_resident': raw_hits,
            'namespace': namespace, 'salt': 'prefix-study',
            'raw_event_ids_residency_occupancy_final_clear': 'passed',
            'paired_ratios_noise_and_verdicts_independently_recomputed': True,
            'preemption_counts': dict(preemptions),
            'numerical_qualification': computed['qualification']['checks'],
        },
        'conditions': {
            'devices': sorted(devices),
            'power_sources': sorted({c['battery']['power_source'] for c in conditions}),
            'power_modes': sorted({c['power_mode_raw'] for c in conditions}),
            'free_memory_percent_range': [min(c['memory']['free_percent'] for c in conditions),
                                          max(c['memory']['free_percent'] for c in conditions)],
            'thermal_messages': [list(x) for x in sorted({tuple(c['thermal']) for c in conditions})],
            'first_before_utc': min(c['conditions_before']['timestamp_utc']
                                    for cell in record['cells'] for c in cell['runs']),
            'last_after_utc': max(c['conditions_after']['timestamp_utc']
                                  for cell in record['cells'] for c in cell['runs']),
        },
        'cells': cells,
        'caveats': [
            'Both arms share cached seed preparation; this is not cold application startup.',
            'Measured duration excludes seed/handoff; episode includes them but excludes final clear.',
            'Cleanup-inclusive ratios are descriptive, not a new confirmatory metric; conclusions persist.',
            'Self-control deviation reaches 7.86 percent; noise is not uniformly five percent.',
            'All requests finish at the 16-token length cap; stop behavior is not performance tested.',
            'Finite offline all-arrivals-zero traces, lifetime reservation, synchronous reference27 only.',
            'No sustainable capacity, SLO, goodput, async/Fast, or default-promotion claim.',
            'Measured CPU replay checks known prefix identity and total work, not independent per-request step completion.',
            'Publication completion relies on the native ownership and numerical evidence.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    result = audit(args.archive.resolve(), args.source_root.resolve())
    helper = Path(__file__).resolve()
    result['reproduction'] = {
        'command': [sys.executable, '-B', str(helper), str(args.archive.resolve()),
                    '--source-root', str(args.source_root.resolve())],
        'exit_code': 0,
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'helper_sha256': digest(helper.read_bytes()),
        'writes': 'JSON stdout only; no GPU invocation, source or archive mutation',
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
