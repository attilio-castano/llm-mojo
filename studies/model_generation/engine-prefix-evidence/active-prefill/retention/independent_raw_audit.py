"""Independent raw cache-key, ownership, history and work audit; no model execution.

Optional live source/binary checks require those original files. Without them,
the audit checks the retained measurement alone; guarded source replay is separate.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--native-binary', type=Path)
    args = parser.parse_args()
    raw_bytes = args.archive.read_bytes()
    manifest = json.loads(args.archive.with_suffix('').read_bytes())
    unpacked = gzip.decompress(raw_bytes)
    assert len(raw_bytes) == manifest['bytes'] and sha(raw_bytes) == manifest['sha256']
    assert sha(unpacked) == manifest['uncompressed_sha256']
    r = json.loads(unpacked)
    assert r['kind'] == 'qwen-engine-prefix-active-v1'
    assert len(r['cells']) == len(r['summary']['cells']) == 5
    counts = collections.Counter()
    states = hits = work = history_count = 0
    namespace = 'qwen-reference27-projection8/' + r['build']['assets']['prepared']
    for c, s in zip(r['cells'], r['summary']['cells']):
        assert c['cell_id'] == s['cell_id'] and len(c['runs']) == len(s['runs']) == 16
        req = {q['request_id']: q for q in c['trace']['requests']}
        frozen = s['frozen_histories']
        hist_inputs = [req[h['request_id']]['prompt_ids'] + h['tokens'][:-1]
                       for h in frozen['measure']]
        for raw, m in zip(c['runs'], s['runs']):
            assert raw['native_execution']['exit_code'] == 0
            p = raw['parsed']
            h = {x['request_id']: x['cached_tokens'] for x in p['phases']['measure']['hits']}
            rows = positions = 0
            for phase, request_ids in [('seed', [128]), ('measure', list(req))]:
                data = p['phases'][phase]
                hist = []
                for i in request_ids:
                    es = [e for e in data['events'] if e['request_id'] == i]
                    tok = [e['token_id'] for e in es if e['kind'] == 'token']
                    finish = [e for e in es if e['kind'] == 'finish']
                    assert len(finish) == 1
                    hist.append(dict(request_id=i, tokens=tok, reason=finish[0]['reason']))
                    if phase == 'measure':
                        total = len(req[i]['prompt_ids']) + len(tok) - 1
                        cached = h.get(i, 0)
                        rows += total - cached
                        positions += total * (total + 1) // 2 - cached * (cached + 1) // 2
                assert hist == frozen[phase]
                for state in data['cache_states']:
                    assert (state['free_blocks'] + state['cached_blocks']
                            + state['private_blocks']) == p['blocks']
                    states += 1
                assert data['drained']['references'] == data['drained']['private_blocks'] == data['drained']['pending'] == 0
            history_count += 1
            assert rows == m['total_tokens'] and positions == m['attended_positions']
            work += 1
            resident = set()
            eid = seedrows = 0
            phase = None
            for line in raw['stdout'].splitlines():
                f = line.split()
                tag = f[0]
                if tag == 'phase':
                    phase = f[1]
                elif tag == 'namespace':
                    assert f[1] == namespace
                elif tag == 'step' and phase == 'seed':
                    seedrows += int(f[5])
                elif tag == 'cache_event':
                    assert int(f[1]) == eid
                    eid += 1
                    kind = f[3]
                    counts[kind] += 1
                    if kind == 'cleared':
                        assert not resident and f[4:] == ['-', '-', '-']
                        continue
                    key = (f[4], f[5], tuple(map(int, f[6].split(','))))
                    assert key[0] == namespace and key[1] == 'prefix-study' and len(key[2]) % 32 == 0
                    if kind == 'stored':
                        assert key not in resident
                        if phase == 'seed':
                            assert len(key[2]) <= seedrows
                        else:
                            assert phase == 'measure' and any(tuple(v[:len(key[2])]) == key[2] for v in hist_inputs)
                        resident.add(key)
                    else:
                        assert kind == 'removed' and key in resident
                        resident.remove(key)
                elif tag == 'cache_hit':
                    rid, length = int(f[2]), int(f[3])
                    hits += 1
                    for n in range(32, length + 1, 32):
                        assert (namespace, 'prefix-study', tuple(req[rid]['prompt_ids'][:n])) in resident
                elif tag == 'cache_state':
                    assert int(f[3]) == len(resident)
                elif tag == 'cache_cleared':
                    assert f[1:] == [str(p['blocks']), '0', '0', '0', '0', '0'] and not resident
            assert not resident
    assert work == history_count == 80
    source_hashes = None
    if args.source_root:
        changed = [f for f, h in r['build']['source']['sources'].items()
                   if sha((args.source_root / f).read_bytes()) != h]
        assert not changed, changed
        source_hashes = len(r['build']['source']['sources'])
    if args.native_binary:
        info = r['build']['binaries']['engine']
        assert args.native_binary.stat().st_size == info['bytes']
        assert sha(args.native_binary.read_bytes()) == info['sha256']
    print(json.dumps(dict(status='passed', source_commit=r['build']['source']['repository']['commit'],
                          measurement_sha256=sha(raw_bytes), runs=work, histories=history_count,
                          snapshots=states, hits=hits, events=dict(counts),
                          source_hashes=source_hashes,
                          native_binary='matched' if args.native_binary else 'not requested',
                          native_execution=False), indent=2))


if __name__ == '__main__':
    main()
