"""Frozen prefix-cache workloads, native collection and strict CPU replay.

Identity uses exact namespace, salt and full token prefixes, never a digest
surrogate. Native execution remains in engine_trace.mojo. Seed, handoff, warm
trace and final clear have distinct retained host timing boundaries.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

from . import model_contract as contract
from . import model_profile as profile
from .environment import ensure_record_location
from ..validation.evidence import sha, write


SALT = 'prefix-study'
CHECKPOINT_IDENTITY = 'engine-prefix-cache-reference27-v1'
CHECKPOINT_GROUPS = ('prefix-chunk-schedule-kv-logits-greedy',
                     'readonly-prefix-divergent-consumers-cold-recompute',
                     'prefix-engine-lifetime-incremental',
                     'prefix-engine-numeric-fault-cache-clear')
CELL_CONFIGS = tuple((length, reuse, 384) for length in (256, 1024) for reuse in (0, 4, 8)) + ((1024, 4, 40), (1024, 8, 40))
PHASES = ('seed', 'handoff', 'measure', 'clear')
SOURCE_TEXT = (
    'You are reviewing a local inference service. Explain ownership, scheduling, '
    'and measured evidence using the supplied engineering notes. A request owns '
    'its token history. Full immutable cache blocks may be shared by requests. '
    'Private suffix blocks receive new key and value rows. A healthy completed '
    'step may publish a full block. Cancellation preserves delivered tokens and '
    'releases references. Capacity pressure evicts unused entries before a '
    'request receives private storage. Report what was checked and what remains '
    'unknown. Keep source identity and device identity with each measurement.\n'
)


def _same(a, b):
    return profile._engine_same_json(a, b)


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _digest(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def cells(block):
    if type(block) is not int or block not in range(4):
        raise ValueError('prefix block must be 0..3')
    order = [('disabled', False), ('disabled', True), ('enabled', False)]
    return list(reversed(order)) if block in (1, 2) else order


def cell_id(length, reuse, blocks):
    return f'prefix-{length}-reuse-{reuse}-blocks-{blocks}'


def _request(identifier, ids, maximum=16):
    return dict(request_id=identifier, arrival_offset_ns=0, prompt_ids=list(ids),
                max_new_tokens=maximum, stop_ids=[151643, 151645], abort_offset_ns=None, output_script=None)


def validate_trace(trace):
    profile.validate_engine_trace(trace)
    prefix = trace.get('prefix', {})
    length, reuse, blocks = (prefix.get(k) for k in ('tokens', 'reuse_requests', 'blocks'))
    if ((length, reuse, blocks) not in CELL_CONFIGS
            or any(type(v) is not int for v in (length, reuse, blocks))
            or trace.get('mode') != 'offline' or prefix.get('salt') != SALT
            or len(trace['requests']) != 8):
        raise ValueError('prefix trace differs from frozen cell declaration')
    seed = prefix.get('seed_request', {})
    profile.validate_engine_trace(dict(kind='engine-token-trace-v1', schema_version=1, mode='offline',
                                     scripted_tokens=[11], requests=[seed]))
    common = seed['prompt_ids'][:-1]
    if (len(common) != length or seed['request_id'] != 128 or seed['max_new_tokens'] != 1
            or seed['arrival_offset_ns'] != 0 or seed['abort_offset_ns'] is not None):
        raise ValueError('prefix seed must execute full prefix and one private query')
    tails, first_blocks = set(), set()
    for i, request in enumerate(trace['requests']):
        ids = request['prompt_ids']
        if (request['request_id'] != i or len(ids) != length + 64
                or request['max_new_tokens'] != 16 or request['stop_ids'] != [151643, 151645]
                or request['arrival_offset_ns'] != 0 or request['abort_offset_ns'] is not None
                or (ids[:length] == common) != (i < reuse)):
            raise ValueError('prefix request length, order or intended reuse changed')
        tail = tuple(ids[length:length+32])
        if tail in tails:
            raise ValueError('private suffixes must differ within their first full block')
        tails.add(tail)
        if i >= reuse:
            first = tuple(ids[:32])
            if first == tuple(common[:32]) or first in first_blocks:
                raise ValueError('nonmatching prefixes must differ within first full block')
            first_blocks.add(first)
    return trace


def validate_campaign(campaign):
    if (campaign.get('kind') != 'engine-prefix-campaign-v1' or campaign.get('schema_version') != 1
            or campaign.get('source_text') != SOURCE_TEXT
            or campaign.get('source_text_sha256') != _hash(SOURCE_TEXT)
            or campaign.get('text_provenance') != 'repository-authored engineering text; no external corpus'
            or not isinstance(campaign.get('tokenizer'), dict)
            or set(campaign['tokenizer']) != {'tables_sha256','binary_sha256','generator_source_sha256'}
            or not all(_digest(value) for value in campaign['tokenizer'].values())):
        raise ValueError('prefix campaign provenance changed')
    traces = campaign.get('traces', [])
    if len(traces) != len(CELL_CONFIGS):
        raise ValueError('prefix campaign must contain all eight frozen cells')
    for trace, expected in zip(traces, CELL_CONFIGS):
        validate_trace(trace)
        if tuple(trace['prefix'][k] for k in ('tokens', 'reuse_requests', 'blocks')) != expected:
            raise ValueError('prefix campaign cells reordered or duplicated')
    return campaign


def specification(output):
    """Tokenize owned text once with the verified native tokenizer, then freeze IDs."""
    from ..models.qwen2.tokenizer_assets import ensure_binary, ensure_prepared
    if output.exists():
        raise ValueError('refusing to replace a prefix campaign')
    ensure_record_location(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tables, binary = ensure_prepared(download=False), ensure_binary()
    tokenizer_identity = dict(tables_sha256=sha(tables), binary_sha256=sha(binary),generator_source_sha256=sha(Path(__file__)))
    def encode(text):
        result = profile.subprocess.run([str(binary), str(tables), 'encode', text, '0'],
            cwd=profile.repository_root(), env=profile.environment(), text=True,
            stdout=profile.subprocess.PIPE, stderr=profile.subprocess.PIPE, timeout=180, check=True)
        ids = json.loads(result.stdout)
        if not ids or any(type(t) is not int or not 0 <= t < 151936 for t in ids):
            raise ValueError('native tokenizer returned invalid IDs')
        return ids
    common = encode(SOURCE_TEXT * 32)
    variants = [encode(f'Independent engineering case {i}.\n' + SOURCE_TEXT * 32) for i in range(8)]
    suffixes = [encode(f'Question {i}: explain request {i} in detail.\n' + SOURCE_TEXT * 4)[:64] for i in range(8)]
    if min(map(len, [common, *variants])) < 1025 or any(len(t) != 64 for t in suffixes):
        raise ValueError('owned text did not produce declared prompt lengths')
    traces = []
    for length, reuse, blocks in CELL_CONFIGS:
        traces.append(dict(kind='engine-token-trace-v1', schema_version=1, mode='offline',
            generator='owned-engineering-text-prefix-v1', seed=0, arrival_rate=None, scripted_tokens=[11],
            prefix=dict(tokens=length, reuse_requests=reuse, blocks=blocks, salt=SALT,
                        seed_request=_request(128, common[:length+1], 1)),
            requests=[_request(i, (common if i < reuse else variants[i])[:length] + suffixes[i]) for i in range(8)]))
    result = dict(kind='engine-prefix-campaign-v1', schema_version=1, source_text=SOURCE_TEXT,
        source_text_sha256=_hash(SOURCE_TEXT), text_provenance='repository-authored engineering text; no external corpus',
        tokenizer=tokenizer_identity, traces=traces)
    validate_campaign(result)
    if tokenizer_identity != dict(tables_sha256=sha(tables), binary_sha256=sha(binary),generator_source_sha256=sha(Path(__file__))):
        raise ValueError('tokenizer changed while freezing prefix campaign')
    write(output, result)
    return result


def trace_tsv(trace):
    validate_trace(trace)
    seed = trace['prefix']['seed_request']
    seed_line = 'seed {} {} {} {} {} {}\n'.format(seed['request_id'], seed['arrival_offset_ns'],
        seed['max_new_tokens'], ','.join(map(str, seed['stop_ids'])),
        ','.join(map(str, seed['prompt_ids'])), SALT)
    return seed_line + profile.engine_trace_tsv(trace)


def _ids(value):
    if value == '-':
        return []
    result = list(map(int, value.split(',')))
    if any(not 0 <= x < 151936 for x in result):
        raise ValueError('invalid exact prefix token')
    return result


def _phase_data():
    return dict(mode='greedy', arrivals={}, events=[], steps=[], cache_steps=[], cache_states=[], hits=[], drained=None)


def _validate_phase(data, trace, blocks, start_step):
    requests = {r['request_id']: r for r in trace['requests']}
    if set(data['arrivals']) != {str(i) for i in requests}:
        raise ValueError('incomplete prefix phase arrivals')
    steps, events = data['steps'], data['events']
    if [s['step_id'] for s in steps] != list(range(start_step, start_step+len(steps))):
        raise ValueError('missing or unordered prefix step')
    previous = 0
    for step in steps:
        if (any(v < 0 for v in step.values()) or step['begin_ns'] < previous or step['end_ns'] < step['begin_ns']
                or sum(step[k] for k in ('schedule_ns','build_ns','execute_ns','postprocess_ns')) > step['end_ns']-step['begin_ns']
                or step['prefill_seqs'] != int(step['prefill_tokens'] > 0)
                or step['decode_seqs']+step['prefill_seqs'] > 8
                or step['total_tokens'] != step['decode_seqs']+step['prefill_tokens']
                or step['total_tokens'] > 256 or step['blocks_free'] > blocks
                or step['preempted'] or step['aborted'] or step['predicted_ns'] or step['budget_limited']):
            raise ValueError('prefix step violates timing, budget or reservation')
        previous = step['end_ns']
    if [s['step_id'] for s in data['cache_steps']] != [s['step_id'] for s in steps]:
        raise ValueError('missing prefix step counters')
    if [s['step_id'] for s in data['cache_states']] != [s['step_id'] for s in steps]:
        raise ValueError('missing prefix physical capacity observations')
    for state,step in zip(data['cache_states'],steps):
        if (state['free_blocks']!=step['blocks_free']
                or state['free_blocks']+state['cached_blocks']+state['private_blocks']!=blocks
                or any(v<0 for v in state.values()) or state['references']>blocks*8):
            raise ValueError('prefix physical capacity partition differs')
    by_step = {s['step_id']: s for s in steps}
    histories, terminal, last = {i: [] for i in requests}, {}, {}
    for event in events:
        i = event['request_id']
        if i not in requests or i in terminal or event['step_id'] not in by_step:
            raise ValueError('unknown prefix event or delivery after finish')
        request, arrival, step = requests[i], data['arrivals'][str(i)], by_step[event['step_id']]
        if (arrival['scheduled_ns'] != request['arrival_offset_ns'] or arrival['actual_ns'] < arrival['scheduled_ns']
                or event['arrival_ns'] != request['arrival_offset_ns'] or event['prompt_tokens'] != len(request['prompt_ids'])
                or not max(arrival['actual_ns'], last.get(i,0), step['begin_ns']) <= event['emitted_ns'] <= step['end_ns']):
            raise ValueError('prefix event time or prompt differs')
        last[i] = event['emitted_ns']
        history = histories[i]
        if event['kind'] == 'token':
            if (not 0 <= event['token_id'] < 151936 or event['generated_tokens'] != len(history)+1
                    or len(history) >= request['max_new_tokens'] or (history and history[-1] in request['stop_ids'])):
                raise ValueError('prefix delivered token order or stop differs')
            history.append(event['token_id'])
        else:
            if (event['generated_tokens'] != len(history) or event['reason'] not in ('stop','length')
                    or (event['reason'] == 'length' and (len(history) != request['max_new_tokens'] or (history and history[-1] in request['stop_ids'])))
                    or (event['reason'] == 'stop' and (not history or history[-1] not in request['stop_ids']))):
                raise ValueError('prefix terminal reason or accounting differs')
            terminal[i] = event
    if set(terminal) != set(requests):
        raise ValueError('missing prefix terminal result')
    for step in steps:
        if sum(e['kind'] == 'finish' and e['step_id'] == step['step_id'] for e in events) != step['finished']:
            raise ValueError('prefix per-step finish accounting differs')
    hit_by_request = Counter()
    for hit in data['hits']:
        if hit['request_id'] in hit_by_request:
            raise ValueError('duplicate reserved prefix admission')
        hit_by_request[hit['request_id']] = hit['cached_tokens']
    for counters in data['cache_steps']:
        if counters['cache_hit_tokens'] != sum(h['cached_tokens'] for h in data['hits'] if h['step_id'] == counters['step_id']):
            raise ValueError('prefix hit counters differ from request receipts')
    necessary = sum(len(requests[i]['prompt_ids'])+len(tokens)-1-hit_by_request[i] if tokens else 0 for i,tokens in histories.items())
    if sum(s['total_tokens'] for s in steps) != necessary:
        raise ValueError('prefix executed rows differ from actual hit-adjusted necessary work')
    drain = data['drained']
    if (drain is None or drain['requests'] != len(requests) or drain['steps'] != len(steps)
            or drain['references'] or drain['private_blocks'] or drain['pending']
            or drain['free_blocks']+drain['cached_blocks'] != blocks
            or drain['elapsed_ns'] < max([s['end_ns'] for s in steps]+[e['emitted_ns'] for e in events]+[0])
            or (steps and (steps[-1]['blocks_free'] != drain['free_blocks'] or steps[-1]['waiting']))):
        raise ValueError('prefix phase did not drain ownership')
    return data


def parse_run(stdout, trace, cache_mode, blocks):
    """Reconstruct exact logical entries and complete request/phase accounting."""
    validate_trace(trace)
    if cache_mode not in ('enabled','disabled') or blocks != trace['prefix']['blocks']:
        raise ValueError('prefix execution differs from declared cell')
    metadata, phase_times, phases = {}, {}, {name:_phase_data() for name in ('seed','measure')}
    entries, events, current, phase_order = set(), [], None, []
    next_event = 0
    cleared = None
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[0]
        if current in phase_times and name != 'phase':
            raise ValueError('prefix record appeared after its closed phase boundary')
        if name in ('device','mode','config','admission','study','work_capacity','cache','namespace','salt'):
            if current is not None or name in metadata:
                raise ValueError('duplicate or late prefix identity')
            metadata[name] = fields[1:]
        elif name == 'phase' and len(fields) == 2:
            if current is not None and current not in phase_times:
                raise ValueError('prefix phase lacks its closing boundary')
            current = fields[1]
            if current not in PHASES or current in phase_order:
                raise ValueError('invalid or repeated prefix phase')
            phase_order.append(current)
            if current == 'measure' and cache_mode == 'disabled' and entries:
                raise ValueError('disabled measurement retained seed index')
        elif name == 'phase_time' and len(fields) == 4:
            phase, begin, end = fields[1], int(fields[2]), int(fields[3])
            # Keeping the seed cache is a no-op; adjacent clock reads may agree.
            zero_work_handoff = (phase == 'handoff' and cache_mode == 'enabled'
                                 and not any(event['phase'] == phase for event in events))
            if (phase != current or phase in phase_times or begin < 0 or end < begin
                    or (end == begin and not zero_work_handoff)):
                raise ValueError('invalid prefix phase boundary')
            phase_times[phase] = dict(begin_ns=begin,end_ns=end)
        elif name == 'cache_event' and len(fields) == 7:
            event_id, step_id = int(fields[1]),int(fields[2])
            kind, namespace, salt, ids = fields[3],fields[4],fields[5],_ids(fields[6])
            if event_id != next_event or current not in PHASES or kind not in ('stored','removed','cleared'):
                raise ValueError('missing, duplicated or invalid prefix cache event')
            next_event += 1
            if current in phases:
                if not phases[current]['steps'] or step_id != phases[current]['steps'][-1]['step_id']:
                    raise ValueError('prefix event outside its completed step')
                if current == 'measure' and cache_mode == 'disabled':
                    raise ValueError('disabled cache emitted an index event')
            elif step_id != -1:
                raise ValueError('cache handoff/clear event must have no step owner')
            key = (namespace,salt,tuple(ids))
            if kind == 'cleared' and (namespace,salt) == ('-','-'):
                namespace,salt='',''
            elif namespace != metadata.get('namespace',[''])[0] or salt != SALT:
                raise ValueError('cache event namespace or salt differs')
            if kind == 'cleared':
                if ids or entries:
                    raise ValueError('cache clear omitted individual removals')
            elif not ids or len(ids)%32:
                raise ValueError('partial block cannot enter prefix index')
            elif kind == 'stored':
                if key in entries or (len(ids)>32 and (namespace,salt,tuple(ids[:-32])) not in entries):
                    raise ValueError('duplicate prefix identity or missing parent')
                entries.add(key)
            else:
                if key not in entries:
                    raise ValueError('removed unknown exact prefix identity')
                entries.remove(key)
            events.append(dict(event_id=event_id,step_id=step_id,phase=current,kind=kind,
                               namespace=namespace,salt=salt,tokens=ids))
        elif current in phases:
            data = phases[current]
            if name == 'step' and len(fields) == len(contract.ENGINE_STEP_FIELDS)+1:
                data['steps'].append(dict(zip(contract.ENGINE_STEP_FIELDS,map(int,fields[1:]))))
            elif name == 'prefix_step' and len(fields) == 6:
                value = dict(zip(('step_id','cache_hit_tokens','lookup_ns','publish_ns','admission_ns'),map(int,fields[1:])))
                if any(v<0 for v in value.values()):
                    raise ValueError('negative prefix counter')
                data['cache_steps'].append(value)
            elif name == 'cache_hit' and len(fields) == 4:
                step_id,i,count = map(int,fields[1:])
                requests = [trace['prefix']['seed_request']] if current == 'seed' else trace['requests']
                request = next((r for r in requests if r['request_id']==i),None)
                if (request is None or not data['steps'] or step_id != data['steps'][-1]['step_id']
                        or count <= 0 or count%32 or count > ((len(request['prompt_ids'])-1)//32)*32
                        or any((metadata.get('namespace',[''])[0],SALT,tuple(request['prompt_ids'][:n])) not in entries
                               for n in range(32,count+1,32))
                        or (current=='measure' and cache_mode=='disabled')):
                    raise ValueError('cache hit is absent, partial, out of range or wrong exact identity')
                data['hits'].append(dict(step_id=step_id,request_id=i,cached_tokens=count))
            elif name == 'arrival' and len(fields) == 4:
                i,scheduled,actual = map(int,fields[1:])
                if str(i) in data['arrivals']:
                    raise ValueError('duplicate prefix arrival')
                data['arrivals'][str(i)] = dict(scheduled_ns=scheduled,actual_ns=actual)
            elif name == 'cache_state' and len(fields)==6:
                value=dict(zip(('step_id','free_blocks','cached_blocks','references','private_blocks'),map(int,fields[1:])))
                if value['cached_blocks']!=len(entries):
                    raise ValueError('prefix logical index and physical cached capacity differ')
                data['cache_states'].append(value)
            elif name in ('token','finish') and len(fields) == 7:
                if not data['steps']:
                    raise ValueError('prefix event before step')
                data['events'].append(dict(kind=name,request_id=int(fields[1]),
                    token_id=int(fields[2]) if name=='token' else None,reason=fields[2] if name=='finish' else None,
                    prompt_tokens=int(fields[3]),generated_tokens=int(fields[4]),arrival_ns=int(fields[5]),
                    emitted_ns=int(fields[6]),step_id=data['steps'][-1]['step_id']))
            elif name == 'prefix_drained' and len(fields) == 9:
                if data['drained'] is not None:
                    raise ValueError('duplicate prefix drain')
                data['drained'] = dict(zip(('requests','steps','free_blocks','cached_blocks','references','private_blocks','pending','elapsed_ns'),map(int,fields[1:])))
                if any(v<0 for v in data['drained'].values()) or data['drained']['cached_blocks'] != len(entries):
                    raise ValueError('cache index differs from drain occupancy')
            else:
                raise ValueError('unknown prefix phase line: '+line[:100])
        elif name == 'cache_cleared' and current=='clear' and len(fields)==7:
            if cleared is not None:
                raise ValueError('duplicate final clear')
            cleared = list(map(int,fields[1:]))
        else:
            raise ValueError('unknown prefix line: '+line[:100])
    expected = dict(device=['Apple','M4','Pro/metal'],mode=['greedy'],config=['chunked',str(blocks),'256','8'],
        admission=['reserved'],study=[contract.ENGINE_PREFIX_STUDY],work_capacity=['256','8'],cache=[cache_mode],
        salt=[SALT],namespace=metadata.get('namespace'))
    if metadata != expected or not metadata.get('namespace') or len(metadata['namespace'])!=1:
        raise ValueError('prefix device/backend or configuration changed')
    if phase_order != list(PHASES) or set(phase_times)!=set(PHASES) or entries or cleared != [blocks,0,0,0,0,0]:
        raise ValueError('prefix omitted phase or final complete cache clear')
    for left,right in zip(PHASES,PHASES[1:]):
        if phase_times[left]['end_ns'] > phase_times[right]['begin_ns']:
            raise ValueError('prefix phase times overlap or reverse')
    seed_trace = dict(kind='engine-token-trace-v1',schema_version=1,mode='offline',scripted_tokens=[11],requests=[trace['prefix']['seed_request']])
    _validate_phase(phases['seed'],seed_trace,blocks,0)
    _validate_phase(phases['measure'],trace,blocks,len(phases['seed']['steps']))
    for event in events:
        if event['kind']!='stored':
            continue
        if event['phase']=='seed':
            completed_rows=sum(s['total_tokens'] for s in phases['seed']['steps'] if s['step_id']<=event['step_id'])
            if len(event['tokens'])>completed_rows:
                raise ValueError('seed stored prefix exceeds cumulative completed rows')
        requests=seed_trace['requests'] if event['phase']=='seed' else trace['requests']
        phase_events=phases[event['phase']]['events']
        grounded=False
        for request in requests:
            delivered=[e['token_id'] for e in phase_events if e['kind']=='token'
                       and e['request_id']==request['request_id'] and e['step_id']<=event['step_id']]
            history=request['prompt_ids']+delivered
            committed_cap=len(request['prompt_ids'])+max(0,len(delivered)-1)
            if len(event['tokens'])<=committed_cap and event['tokens']==history[:len(event['tokens'])]:
                grounded=True
                break
        if not grounded:
            raise ValueError('stored prefix identity is outside known computed input histories')
    for name in ('seed','measure'):
        if phases[name]['drained']['elapsed_ns'] > phase_times[name]['end_ns']-phase_times[name]['begin_ns']:
            raise ValueError('prefix phase execution exceeds episode boundary')
    return dict(device='Apple M4 Pro/metal',mode='greedy',cache=cache_mode,blocks=blocks,
                namespace=metadata['namespace'][0],phases=phases,phase_times=phase_times,cache_events=events)


def run_summary(parsed, trace):
    result = profile.engine_run_summary(parsed['phases']['measure'],trace)
    phases = parsed['phase_times']
    result.update(seed_ns=phases['seed']['end_ns']-phases['seed']['begin_ns'],
        handoff_ns=phases['handoff']['end_ns']-phases['handoff']['begin_ns'],
        episode_ns=phases['measure']['end_ns']-phases['seed']['begin_ns'],
        clear_ns=phases['clear']['end_ns']-phases['clear']['begin_ns'],
        seed_rows=sum(s['total_tokens'] for s in parsed['phases']['seed']['steps']),
        cache_hit_tokens=sum(h['cached_tokens'] for h in parsed['phases']['measure']['hits']),
        lookup_ns=sum(s['lookup_ns'] for s in parsed['phases']['measure']['cache_steps']),
        publish_ns=sum(s['publish_ns'] for s in parsed['phases']['measure']['cache_steps']),
        cache_admission_ns=sum(s['admission_ns'] for s in parsed['phases']['measure']['cache_steps']),
        evictions=sum(e['kind']=='removed' and e['phase']=='measure' for e in parsed['cache_events']),
        peak_cached_blocks=max((s['cached_blocks'] for s in parsed['phases']['measure']['cache_states']),default=0),
        peak_private_blocks=max((s['private_blocks'] for s in parsed['phases']['measure']['cache_states']),default=0),
        peak_references=max((s['references'] for s in parsed['phases']['measure']['cache_states']),default=0),
        cache_cleared=True)
    hits={h['request_id']:h['cached_tokens'] for h in parsed['phases']['measure']['hits']}
    for request in result['requests']:
        request['cached_prompt_tokens']=hits.get(request['request_id'],0)
    return result


def _execution(value, command, timeout, minimum_ns=0):
    if (value.get('command') != list(map(str,command)) or type(value.get('exit_code')) is not int
            or value['exit_code'] != 0 or type(value.get('timeout_seconds')) is not int
            or value['timeout_seconds'] != timeout or type(value.get('wall_elapsed_ns')) is not int
            or not max(1,minimum_ns) <= value['wall_elapsed_ns'] <= timeout*1_000_000_000):
        raise ValueError('prefix native execution receipt changed or failed')


def parse_checkpoint(stdout, required_groups=CHECKPOINT_GROUPS):
    """Check native arithmetic identity and the explicitly required coverage."""
    identity, checks = {},[]
    for line in stdout.splitlines():
        fields = line.split()
        if fields and fields[0] in ('device','qualification','configuration') and fields[0] not in identity:
            identity[fields[0]]=fields[1:]
        elif len(fields)==8 and fields[0]=='check' and fields[2::2]==['checked','mismatches','unexpected-nonfinite']:
            checks.append(dict(name=fields[1],checked=int(fields[3]),mismatches=int(fields[5]),nonfinite=int(fields[7])))
        else:
            raise ValueError('unknown prefix checkpoint qualification line')
    expected=dict(device=['Apple','M4','Pro/metal'],qualification=[CHECKPOINT_IDENTITY],
                  configuration=['27','projection','8','dtype','BF16/FP32','block-size','32'])
    if (identity != expected or tuple(c['name'] for c in checks)!=required_groups
            or any(c['checked']<1 or c['mismatches'] or c['nonfinite'] for c in checks)):
        raise ValueError('prefix checkpoint qualification omitted or failed a required group')
    return dict(numerical_passed=True,checks=checks)


def _qualified_namespace(build):
    return 'qwen-reference27-projection8/'+build['assets']['prepared']


def _qualification(record, require_preflights=True):
    build = profile.validate_engine_record_build(record)
    document = record.get('qualification_document')
    if (type(document) is not str or _hash(document) != record.get('qualification_sha256')
            or not _same(json.loads(document),record.get('qualification'))):
        raise ValueError('prefix qualification document changed')
    qualification = record['qualification']
    checkpoint = qualification.get('checkpoint_receipt',{})
    actual = checkpoint.get('build',{})
    if (qualification.get('kind') != 'engine-prefix-qualification-v1'
            or type(qualification.get('schema_version')) is not int or qualification['schema_version']!=1
            or not _same(qualification.get('build'),build)
            or any(not _same(actual.get(k),build.get(k)) for k in ('source','assets','environment'))):
        raise ValueError('prefix qualification source/assets/environment changed')
    command = actual.get('command',[])
    if (len(command)!=9 or command[1:7] != ['build','-I','src','-I','tests','tests/engine_metal_driver.mojo']
            or command[0]!=build.get('command',[''])[0]
            or command[-2]!='-o' or not Path(command[-1]).is_absolute()
            or set(actual.get('binaries',{}))!={'checkpoint-driver'}
            or not _digest(actual['binaries']['checkpoint-driver'].get('sha256'))
            or type(actual['binaries']['checkpoint-driver'].get('bytes')) is not int
            or actual['binaries']['checkpoint-driver']['bytes']<1
            or not _digest(build['source']['sources'].get('tests/engine_metal_driver.mojo'))):
        raise ValueError('prefix checkpoint build command changed')
    _execution(checkpoint.get('build_execution',{}),command,600)
    _execution(checkpoint.get('execution',{}),[command[-1],build['assets']['prepared'],'prefix-cache-qualification'],180)
    stdout = checkpoint.get('stdout')
    if type(stdout) is not str or _hash(stdout)!=checkpoint.get('stdout_sha256'):
        raise ValueError('prefix qualification stdout changed')
    checked=parse_checkpoint(stdout)
    if not require_preflights:
        return checked
    document=qualification.get('campaign_document')
    if (type(document) is not str or _hash(document)!=qualification.get('campaign_sha256')
            or not _same(json.loads(document),qualification.get('campaign'))):
        raise ValueError('prefix qualification campaign document changed')
    campaign=validate_campaign(qualification['campaign'])
    if ('trace' in record and not _same(record['trace'],campaign)):
        raise ValueError('prefix qualification does not bind measured campaign')
    if (campaign['tokenizer']['tables_sha256']!=build['assets']['tables_sha256']
            or campaign['tokenizer']['generator_source_sha256']!=build['source']['sources'].get('src/llm_mojo/benchmarks/engine_prefix.py')):
        raise ValueError('prefix qualification tokenizer/generator changed')
    preflights=qualification.get('preflight_cells',[])
    if len(preflights)!=8:
        raise ValueError('prefix qualification requires all sixteen untimed preflights')
    expected={}
    for cell,trace in zip(preflights,campaign['traces']):
        params=trace['prefix']; name=cell_id(params['tokens'],params['reuse_requests'],params['blocks'])
        if (cell.get('cell_id')!=name or not _same(cell.get('trace'),trace)
                or cell.get('native_trace_sha256')!=_hash(trace_tsv(trace))
                or not Path(cell.get('native_trace_path','')).is_absolute()
                or [r.get('cache') for r in cell.get('runs',[])]!=['disabled','enabled']):
            raise ValueError('prefix qualification preflight identity/census changed')
        oracle=None
        for run in cell['runs']:
            parsed=parse_run(run['stdout'],trace,run['cache'],params['blocks'])
            if not _same(parsed,run.get('parsed')) or parsed['namespace']!=_qualified_namespace(build):
                raise ValueError('prefix preflight parsed identity differs from qualified source')
            minimum=parsed['phase_times']['clear']['end_ns']-parsed['phase_times']['seed']['begin_ns']
            _execution(run.get('native_execution',{}),_command(record,cell,run),180,minimum)
            histories=_histories(parsed,trace)
            if oracle is None:
                oracle=histories
            if not _same(histories,oracle):
                raise ValueError('prefix untimed cache hit/miss natural histories differ')
        if not _same(cell.get('expected_histories'),oracle):
            raise ValueError('prefix qualification frozen histories changed')
        expected[name]=oracle
    return dict(**checked,untimed_preflights=16,expected_histories=expected,campaign_sha256=qualification['campaign_sha256'])


def qualify(directory, output, trace_path):
    build=profile.verify_engine_build(directory,contract.ENGINE_PREFIX_DECLARATION)
    ensure_record_location(output)
    output.mkdir(parents=True,exist_ok=False)
    binary=output/'checkpoint-driver'
    command=[profile.environment_tool('mojo'),'build','-I','src','-I','tests','tests/engine_metal_driver.mojo','-o',binary]
    _,compiled=profile.checked_execution(command,output/'checkpoint-build.log',600)
    if profile.verify_build(directory)!=build:
        raise ValueError('prefix build changed during qualification compilation')
    numerical=dict(source=build['source'],assets=build['assets'],environment=build['environment'],
        command=list(map(str,command)),binaries={'checkpoint-driver':dict(sha256=sha(binary),bytes=binary.stat().st_size)})
    stdout,execution=profile.checked_execution([binary,build['assets']['prepared'],'prefix-cache-qualification'],output/'checkpoint-native.log',180)
    if profile.verify_build(directory)!=build or sha(binary)!=numerical['binaries']['checkpoint-driver']['sha256']:
        raise ValueError('prefix build changed during qualification execution')
    qualification=dict(kind='engine-prefix-qualification-v1',schema_version=1,build=build,
        checkpoint_receipt=dict(build=numerical,build_execution=compiled,stdout=stdout,stdout_sha256=_hash(stdout),execution=execution))
    def checker(require_preflights=True):
        document=json.dumps(qualification,indent=2,allow_nan=False)+'\n'
        record=dict(kind=contract.ENGINE_PREFIX_DECLARATION['kind'],declaration=contract.ENGINE_PREFIX_DECLARATION,
            build=build,qualification=qualification,qualification_document=document,qualification_sha256=_hash(document))
        return _qualification(record,require_preflights)
    checker(False)
    campaign_document=trace_path.read_text()
    campaign=validate_campaign(json.loads(campaign_document))
    if any(ch.isspace() for ch in build['assets']['prepared']):
        raise ValueError('bounded prefix benchmark requires whitespace-free prepared path')
    qualification.update(campaign=campaign,campaign_document=campaign_document,campaign_sha256=_hash(campaign_document),preflight_cells=[])
    try:
        for trace in campaign['traces']:
            params=trace['prefix']; name=cell_id(params['tokens'],params['reuse_requests'],params['blocks'])
            cell_dir=output/name; cell_dir.mkdir()
            native=cell_dir/'trace.tsv'; native.write_text(trace_tsv(trace))
            cell=dict(cell_id=name,trace=trace,native_trace_path=str(native),native_trace_sha256=sha(native),runs=[])
            qualification['preflight_cells'].append(cell)
            oracle=None
            for cache in ('disabled','enabled'):
                run=dict(cache=cache)
                command=_command(dict(build=build),cell,run)
                stdout,execution=profile.checked_execution(command,cell_dir/f'preflight-{cache}.log',180)
                parsed=parse_run(stdout,trace,cache,params['blocks'])
                run.update(stdout=stdout,native_execution=execution,parsed=parsed)
                cell['runs'].append(run)
                actual=_histories(parsed,trace)
                if oracle is None: oracle=actual
                if not _same(oracle,actual):
                    raise ValueError('prefix qualification histories differ before timing')
            cell['expected_histories']=oracle
            if sha(native)!=cell['native_trace_sha256']:
                raise ValueError('prefix qualification input changed')
        if (profile.verify_build(directory)!=build or sha(binary)!=numerical['binaries']['checkpoint-driver']['sha256']
                or sha(trace_path)!=qualification['campaign_sha256']):
            raise ValueError('prefix qualification source/binary/campaign changed during preflights')
        result=checker()
    except Exception:
        write(output/'incomplete-qualification.json',qualification)
        raise
    document=json.dumps(qualification,indent=2,allow_nan=False)+'\n'
    (output/'qualification.json').write_text(document)
    write(output/'qualification-summary.json',result)
    return result


def _command(record, cell, run):
    return [record['build']['command'][-1],record['build']['assets']['prepared'],cell['native_trace_path'],
            'chunked',cell['trace']['prefix']['blocks'],256,8,10,'greedy','reserved',contract.ENGINE_PREFIX_STUDY,run['cache']]


def _histories(parsed,trace):
    def histories(phase,requests):
        return [dict(request_id=r['request_id'],tokens=[e['token_id'] for e in parsed['phases'][phase]['events']
                 if e['kind']=='token' and e['request_id']==r['request_id']],
                 reason=next(e['reason'] for e in parsed['phases'][phase]['events']
                 if e['kind']=='finish' and e['request_id']==r['request_id'])) for r in requests]
    return dict(seed=histories('seed',[trace['prefix']['seed_request']]),measure=histories('measure',trace['requests']))


def summary(record):
    if record.get('declaration')!=contract.ENGINE_PREFIX_DECLARATION or record.get('kind')!=contract.ENGINE_PREFIX_DECLARATION['kind']:
        raise ValueError('prefix campaign declaration changed')
    qualification=_qualification(record)
    campaign=validate_campaign(record['trace'])
    if (type(record.get('trace_document')) is not str or _hash(record['trace_document'])!=record.get('trace_sha256')
            or not _same(json.loads(record['trace_document']),campaign)
            or len(record.get('cells',[]))!=8):
        raise ValueError('prefix campaign document or cell census changed')
    if campaign['tokenizer']['tables_sha256']!=record['build']['assets']['tables_sha256']:
        raise ValueError('prefix workload tokenizer differs from qualified build')
    if campaign['tokenizer']['generator_source_sha256']!=record['build']['source']['sources'].get('src/llm_mojo/benchmarks/engine_prefix.py'):
        raise ValueError('prefix workload generator differs from qualified source')
    cells_out=[]
    for cell,trace in zip(record['cells'],campaign['traces']):
        params=trace['prefix']
        name=cell_id(params['tokens'],params['reuse_requests'],params['blocks'])
        if (cell.get('cell_id')!=name or not _same(cell.get('trace'),trace)
                or cell.get('native_trace_sha256')!=_hash(trace_tsv(trace))
                or not Path(cell.get('native_trace_path','')).is_absolute()):
            raise ValueError('prefix native trace identity changed')
        runs=cell.get('runs',[])
        expected=[(b,cache,calibration) for b in range(4) for cache,calibration in cells(b)]
        if [(r.get('block'),r.get('cache'),r.get('calibration')) for r in runs]!=expected:
            raise ValueError('incomplete, duplicate or reordered prefix paired grid')
        histories=qualification['expected_histories'][name]
        summaries=[]
        for run in runs:
            parsed=parse_run(run['stdout'],trace,run['cache'],params['blocks'])
            if not _same(parsed,run.get('parsed')):
                raise ValueError('prefix parsed record differs from native stdout')
            if parsed['namespace']!=_qualified_namespace(record['build']):
                raise ValueError('prefix measured namespace differs from qualified build')
            minimum=parsed['phase_times']['clear']['end_ns']-parsed['phase_times']['seed']['begin_ns']
            _execution(run.get('native_execution',{}),_command(record,cell,run),180,minimum)
            actual=_histories(parsed,trace)
            if not _same(histories,actual):
                raise ValueError('prefix natural greedy histories or terminal reasons diverged')
            if type(run['block']) is not int or type(run['calibration']) is not bool:
                raise ValueError('prefix block/calibration types changed')
            for key in ('conditions_before','conditions_after'):
                conditions=run.get(key,{})
                try:
                    profile.require_ac(conditions)
                    profile.require_nominal_thermal_state(conditions)
                    if conditions['power_mode_raw']!='0':
                        raise ValueError('non-nominal power mode')
                except (KeyError,RuntimeError) as error:
                    raise ValueError('missing or non-nominal prefix conditions') from error
            summaries.append(dict(block=run['block'],cache=run['cache'],calibration=run['calibration'],**run_summary(parsed,trace)))
        by_key={(r['block'],r['cache'],r['calibration']):r for r in summaries}
        comparisons={}
        for metric in ('duration_ns','episode_ns'):
            self_ratios=[by_key[b,'disabled',True][metric]/by_key[b,'disabled',False][metric] for b in range(4)]
            ratios=[by_key[b,'enabled',False][metric]/by_key[b,'disabled',False][metric] for b in range(4)]
            floor=max(.05,max(abs(v-1) for v in self_ratios))
            comparisons[metric]=dict(block_ratios=ratios,median_ratio=statistics.median(ratios),
                calibration_ratios=self_ratios,noise_floor=floor,speed_verdict=profile._outcome(ratios,floor))
        cells_out.append(dict(cell_id=name,runs=summaries,comparisons=comparisons,frozen_histories=histories))
    return dict(cells=cells_out,qualification=qualification,measured_runs=96,untimed_preflights=16,
                same_delivered_histories=True,asynchronous=False,promotion=False,target=None,goodput=None,quantile_method='linear')


def collect(directory,trace_path,output,qualification_path):
    """Run the frozen eight-cell campaign once, retaining every actual invocation."""
    build=profile.verify_engine_build(directory,contract.ENGINE_PREFIX_DECLARATION)
    document=trace_path.read_text()
    campaign=validate_campaign(json.loads(document))
    if campaign['tokenizer']['tables_sha256']!=build['assets']['tables_sha256']:
        raise ValueError('prefix campaign tokenizer differs from clean build')
    if campaign['tokenizer']['generator_source_sha256']!=build['source']['sources'].get('src/llm_mojo/benchmarks/engine_prefix.py'):
        raise ValueError('prefix campaign generator differs from clean build')
    if any(ch.isspace() for ch in build['assets']['prepared']):
        raise ValueError('bounded native benchmark requires a whitespace-free prepared path')
    qualification_document=qualification_path.read_text()
    record=dict(kind=contract.ENGINE_PREFIX_DECLARATION['kind'],declaration=contract.ENGINE_PREFIX_DECLARATION,
        build=build,trace=campaign,trace_document=document,trace_sha256=_hash(document),
        qualification=json.loads(qualification_document),qualification_document=qualification_document,
        qualification_sha256=_hash(qualification_document),cells=[])
    accepted=_qualification(record)
    numerical=record['qualification']['checkpoint_receipt']['build']
    binary=Path(numerical['command'][-1])
    identity=numerical['binaries']['checkpoint-driver']
    if sha(binary)!=identity['sha256'] or binary.stat().st_size!=identity['bytes']:
        raise ValueError('prefix qualified checkpoint binary changed')
    ensure_record_location(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'qualification.json').write_text(qualification_document)
    deadline=time.monotonic()+7200
    def execute(cell,run,log):
        if time.monotonic()+180>deadline:
            raise TimeoutError('prefix campaign reached its two-hour bound')
        before=profile.conditions()
        stdout,execution=profile.checked_execution(_command(record,cell,run),log,180)
        run.update(stdout=stdout,native_execution=execution,conditions_before=before,
                   conditions_after=profile.conditions(),parsed=parse_run(stdout,cell['trace'],run['cache'],cell['trace']['prefix']['blocks']))
        return run
    try:
        for trace in campaign['traces']:
            params=trace['prefix']
            name=cell_id(params['tokens'],params['reuse_requests'],params['blocks'])
            cell_dir=output/name
            cell_dir.mkdir()
            native=cell_dir/'trace.tsv'
            native.write_text(trace_tsv(trace))
            cell=dict(cell_id=name,trace=trace,native_trace_path=str(native),native_trace_sha256=sha(native),runs=[])
            record['cells'].append(cell)
            oracle=accepted['expected_histories'][name]
            for block in range(4):
                for cache,calibration in cells(block):
                    if sha(native)!=cell['native_trace_sha256']:
                        raise ValueError('prefix native trace changed during campaign')
                    run=execute(cell,dict(block=block,cache=cache,calibration=calibration),
                                cell_dir/f'block-{block}-{cache}{"-calibration" if calibration else ""}.log')
                    cell['runs'].append(run)
                    if _histories(run['parsed'],trace)!=oracle:
                        raise ValueError('prefix measured natural history differs from frozen preflight')
            if sha(native)!=cell['native_trace_sha256']:
                raise ValueError('prefix native input changed after cell')
        if (profile.verify_build(directory)!=build or sha(trace_path)!=record['trace_sha256']
                or sha(qualification_path)!=record['qualification_sha256'] or sha(binary)!=identity['sha256']):
            raise ValueError('prefix campaign source/assets/qualification changed')
        record['summary']=summary(record)
    except Exception:
        write(output/'incomplete-campaign.json',record)
        raise
    raw=json.dumps(record,separators=(',',':'),allow_nan=False).encode()
    compressed=gzip.compress(raw,mtime=0)
    (output/'engine-prefix.json.gz').write_bytes(compressed)
    write(output/'engine-prefix.json',dict(kind=record['kind'],bytes=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
    write(output/'engine-prefix-summary.json',record['summary'])
    return profile.engine_replay(output)
