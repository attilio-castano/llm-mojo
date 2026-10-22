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
ACTIVE_CELL_CONFIGS = ((1024, 4, 384), (1024, 8, 384), (1024, 4, 40), (1024, 8, 40), (1024, 0, 384))
ACTIVE_CAPS = (0, 32, 16)
ADAPTIVE_MODES = ('current','fixed32','adaptive')
FREQUENCY_MODES = ('current','adaptive','frequency')
FREQUENCY_CHECKPOINT_IDENTITY = 'engine-prefix-frequency-reference27-v1'
FREQUENCY_CHECKPOINT_GROUPS = ('prefix-frequency-engine-cohort-kv-logits-greedy',
    'prefix-frequency-engine-invalidation-terminal-cache-clear')
FREQUENCY_SNAPSHOT_INTS = ('cohort_id','anchor_request_id','anchor_ticket','last_ticket','request_count',
    'gap_budget','run_count','predicted_interruptions','threshold','matched_tokens','remaining_prompt_tokens',
    'anchor_maximum_outputs','required_blocks','available_blocks','lookup_count')
FREQUENCY_SNAPSHOT_LISTS = ('request_ids','tickets','prompt_lengths','maximum_outputs','run_request_ids','run_tickets')
ADAPTIVE_CHECKPOINT_IDENTITY = 'engine-prefix-adaptive-reference27-v1'
ADAPTIVE_CHECKPOINT_GROUPS = ('prefix-adaptive-mixed-kv-logits-greedy-cold-recompute',
    'prefix-adaptive-zero-head-partial','prefix-adaptive-engine-decisions-kv-logits-greedy',
    'prefix-adaptive-engine-terminal-fault-cache-clear')
DECISION_FIELDS = ('request_id','prompt_length','maximum_outputs','matched_tokens','free_blocks',
    'reclaimable_blocks','watermark_blocks','available_blocks','required_blocks','selected_decodes',
    'incumbent_request_id','incumbent_remaining_outputs','remaining_prompt_tokens','row_budget',
    'max_sequences','baseline_tokens','chosen_tokens')
# A separate native identity preserves the original checkpoint coverage scope.
ACTIVE_CHECKPOINT_IDENTITY = 'engine-prefix-active-reference27-v1'
ACTIVE_CHECKPOINT_GROUPS = ('prefix-active-mixed-kv-logits-greedy-cold-recompute',
                            'prefix-active-zero-head-partial', 'prefix-active-engine-cap-lifecycle',
                            'prefix-active-engine-fault-cache-clear')
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


def active_cells(block):
    if type(block) is not int or block not in range(4):
        raise ValueError('active prefix block must be 0..3')
    arms = [(0, False), (0, True), (32, False), (16, False)]
    return [arms[i] for i in contract.ENGINE_PREFIX_ACTIVE_DECLARATION['balanced_orders'][block]]


def _declaration(active=False,adaptive=False,frequency=False):
    return contract.ENGINE_PREFIX_FREQUENCY_DECLARATION if frequency else contract.ENGINE_PREFIX_ADAPTIVE_DECLARATION if adaptive else contract.ENGINE_PREFIX_ACTIVE_DECLARATION if active else contract.ENGINE_PREFIX_DECLARATION


def _active(record):
    return record.get('kind') == contract.ENGINE_PREFIX_ACTIVE_DECLARATION['kind']


def _adaptive(record):
    return record.get('kind') == contract.ENGINE_PREFIX_ADAPTIVE_DECLARATION['kind']


def _frequency(record):
    return record.get('kind') == contract.ENGINE_PREFIX_FREQUENCY_DECLARATION['kind']


def adaptive_cells(block):
    if type(block) is not int or block not in range(4): raise ValueError('adaptive block must be 0..3')
    arms=[('current',False),('current',True),('fixed32',False),('adaptive',False)]
    return [arms[i] for i in contract.ENGINE_PREFIX_ADAPTIVE_DECLARATION['balanced_orders'][block]]


def frequency_cells(block):
    if type(block) is not int or block not in range(4): raise ValueError('frequency block must be 0..3')
    arms=[('current',False),('current',True),('adaptive',False),('frequency',False)]
    return [arms[i] for i in contract.ENGINE_PREFIX_FREQUENCY_DECLARATION['balanced_orders'][block]]


def _cell_name(trace,adaptive=False,frequency=False):
    p=trace['prefix']; name=cell_id(p['tokens'],p['reuse_requests'],p['blocks'])
    if frequency: return name+f"-suffix-{p['suffix_tokens']}-out-{p['max_new_tokens']}-requests-{p['cohort_requests']}-{p['match_order']}"
    return name+f"-suffix-{p['suffix_tokens']}-out-{p['max_new_tokens']}-{p['match_order']}" if adaptive else name


def cell_id(length, reuse, blocks):
    return f'prefix-{length}-reuse-{reuse}-blocks-{blocks}'


def _request(identifier, ids, maximum=16):
    return dict(request_id=identifier, arrival_offset_ns=0, prompt_ids=list(ids),
                max_new_tokens=maximum, stop_ids=[151643, 151645], abort_offset_ns=None, output_script=None)


def validate_trace(trace,adaptive_prefill=False,frequency_prefill=False):
    adaptive_prefill=adaptive_prefill or frequency_prefill
    profile.validate_engine_trace(trace)
    prefix = trace.get('prefix', {})
    length, reuse, blocks = (prefix.get(k) for k in ('tokens', 'reuse_requests', 'blocks'))
    config=[reuse,blocks,prefix.get('suffix_tokens'),prefix.get('max_new_tokens'),prefix.get('match_order')]
    valid=(config+[prefix.get('cohort_requests')] in contract.ENGINE_PREFIX_FREQUENCY_CELLS and length==1024 if frequency_prefill else
           config in contract.ENGINE_PREFIX_ADAPTIVE_CELLS and length==1024 if adaptive_prefill else (length,reuse,blocks) in CELL_CONFIGS)
    suffix=prefix['suffix_tokens'] if adaptive_prefill else 64
    maximum=prefix['max_new_tokens'] if adaptive_prefill else 16
    if (not valid
            or any(type(v) is not int for v in ((length,reuse,blocks,suffix,maximum) if adaptive_prefill else (length,reuse,blocks)))
            or trace.get('mode') != 'offline' or prefix.get('salt') != SALT
            or (frequency_prefill and type(prefix.get('cohort_requests')) is not int)
            or len(trace['requests']) != (prefix.get('cohort_requests') if frequency_prefill else 8)):
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
        matching=i%2==0 if adaptive_prefill and prefix['match_order']=='even' else i<reuse
        if (request['request_id'] != i or len(ids) != length + suffix
                or request['max_new_tokens'] != maximum or request['stop_ids'] != [151643, 151645]
                or request['arrival_offset_ns'] != 0 or request['abort_offset_ns'] is not None
                or (ids[:length] == common) != matching):
            raise ValueError('prefix request length, order or intended reuse changed')
        tail = tuple(ids[length:length+32])
        if tail in tails:
            raise ValueError('private suffixes must differ within their first full block')
        tails.add(tail)
        if not matching:
            first = tuple(ids[:32])
            if first == tuple(common[:32]) or first in first_blocks:
                raise ValueError('nonmatching prefixes must differ within first full block')
            first_blocks.add(first)
    return trace


def validate_campaign(campaign, active_prefill=False,adaptive_prefill=False,frequency_prefill=False):
    adaptive_prefill=adaptive_prefill or frequency_prefill
    configs = contract.ENGINE_PREFIX_FREQUENCY_CELLS if frequency_prefill else contract.ENGINE_PREFIX_ADAPTIVE_CELLS if adaptive_prefill else ACTIVE_CELL_CONFIGS if active_prefill else CELL_CONFIGS
    kind = 'engine-prefix-frequency-campaign-v1' if frequency_prefill else 'engine-prefix-adaptive-campaign-v1' if adaptive_prefill else 'engine-prefix-active-campaign-v1' if active_prefill else 'engine-prefix-campaign-v1'
    if (campaign.get('kind') != kind or type(campaign.get('schema_version')) is not int or campaign['schema_version'] != 1
            or campaign.get('source_text') != SOURCE_TEXT
            or campaign.get('source_text_sha256') != _hash(SOURCE_TEXT)
            or campaign.get('text_provenance') != 'repository-authored engineering text; no external corpus'
            or not isinstance(campaign.get('tokenizer'), dict)
            or set(campaign['tokenizer']) != {'tables_sha256','binary_sha256','generator_source_sha256'}
            or not all(_digest(value) for value in campaign['tokenizer'].values())):
        raise ValueError('prefix campaign provenance changed')
    traces = campaign.get('traces', [])
    if len(traces) != len(configs):
        raise ValueError('prefix campaign must contain its complete frozen cells')
    for trace, expected in zip(traces, configs):
        validate_trace(trace,adaptive_prefill,frequency_prefill)
        observed=[trace['prefix'][k] for k in ('reuse_requests','blocks','suffix_tokens','max_new_tokens','match_order')] if adaptive_prefill else tuple(trace['prefix'][k] for k in ('tokens','reuse_requests','blocks'))
        if frequency_prefill: observed.append(trace['prefix']['cohort_requests'])
        if observed != expected:
            raise ValueError('prefix campaign cells reordered or duplicated')
    return campaign


def specification(output, active_prefill=False,adaptive_prefill=False,frequency_prefill=False):
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
    adaptive_prefill=adaptive_prefill or frequency_prefill
    suffix_length=65 if adaptive_prefill else 64
    count=9 if frequency_prefill else 8
    if frequency_prefill: variants.append(encode('Independent engineering case 8.\n'+SOURCE_TEXT*32))
    suffixes = [encode(f'Question {i}: explain request {i} in detail.\n' + SOURCE_TEXT * 4)[:suffix_length] for i in range(count)]
    if min(map(len, [common, *variants])) < 1025 or any(len(t) != suffix_length for t in suffixes):
        raise ValueError('owned text did not produce declared prompt lengths')
    traces = []
    configs=([(1024,*values) for values in contract.ENGINE_PREFIX_FREQUENCY_CELLS] if frequency_prefill else
             [(1024,*values,8) for values in contract.ENGINE_PREFIX_ADAPTIVE_CELLS] if adaptive_prefill else
             [(length,reuse,blocks,64,16,'first',8) for length,reuse,blocks in (ACTIVE_CELL_CONFIGS if active_prefill else CELL_CONFIGS)])
    for length,reuse,blocks,suffix,maximum,order,request_count in configs:
        traces.append(dict(kind='engine-token-trace-v1', schema_version=1, mode='offline',
            generator='owned-engineering-text-prefix-v1', seed=0, arrival_rate=None, scripted_tokens=[11],
            prefix=dict(tokens=length, reuse_requests=reuse, blocks=blocks, salt=SALT,
                        seed_request=_request(128, common[:length+1], 1)),
            requests=[_request(i,(common if (i%2==0 if order=='even' else i<reuse) else variants[i])[:length]+suffixes[i][:suffix],maximum) for i in range(request_count)]))
        if adaptive_prefill: traces[-1]['prefix'].update(suffix_tokens=suffix,max_new_tokens=maximum,match_order=order)
        if frequency_prefill: traces[-1]['prefix']['cohort_requests']=request_count
    result = dict(kind='engine-prefix-frequency-campaign-v1' if frequency_prefill else 'engine-prefix-adaptive-campaign-v1' if adaptive_prefill else 'engine-prefix-active-campaign-v1' if active_prefill else 'engine-prefix-campaign-v1', schema_version=1, source_text=SOURCE_TEXT,
        source_text_sha256=_hash(SOURCE_TEXT), text_provenance='repository-authored engineering text; no external corpus',
        tokenizer=tokenizer_identity, traces=traces)
    validate_campaign(result,active_prefill,adaptive_prefill,frequency_prefill)
    if tokenizer_identity != dict(tables_sha256=sha(tables), binary_sha256=sha(binary),generator_source_sha256=sha(Path(__file__))):
        raise ValueError('tokenizer changed while freezing prefix campaign')
    write(output, result)
    return result


def trace_tsv(trace,adaptive_prefill=False,frequency_prefill=False):
    validate_trace(trace,adaptive_prefill,frequency_prefill)
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


def _phase_data(adaptive=False,frequency=False):
    result=dict(mode='greedy', arrivals={}, events=[], steps=[], cache_steps=[], cache_states=[], hits=[], drained=None)
    if adaptive: result.update(prefill_decisions=[],selections=[])
    if frequency: result.update(frequency_work=[],frequency_snapshots=[],frequency_events=[],frequency_decisions=[])
    return result


def _validate_phase(data, trace, blocks, start_step, active_prefill_cap=None,adaptive=False):
    requests = {r['request_id']: r for r in trace['requests']}
    if set(data['arrivals']) != {str(i) for i in requests}:
        raise ValueError('incomplete prefix phase arrivals')
    steps, events = data['steps'], data['events']
    if [s['step_id'] for s in steps] != list(range(start_step, start_step+len(steps))):
        raise ValueError('missing or unordered prefix step')
    if active_prefill_cap is not None and not adaptive:
        first=trace['requests'][0]; prefix=trace.get('prefix',{})
        hit=prefix.get('tokens',0) if first['request_id'] < prefix.get('reuse_requests',0) else 0
        if (not steps or steps[0]['decode_seqs'] or steps[0]['prefill_tokens']!=min(256,len(first['prompt_ids'])-hit)):
            raise ValueError('active prefix idle first chunk differs from unchanged policy')
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
        if (active_prefill_cap and step['decode_seqs'] > 0
                and step['prefill_tokens'] > active_prefill_cap):
            raise ValueError('active prefill exceeded its mixed-step cap')
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
    last_token={}; token_steps=set()
    for event in events:
        i = event['request_id']
        if i not in requests or i in terminal or event['step_id'] not in by_step:
            raise ValueError('unknown prefix event or delivery after finish')
        request, arrival, step = requests[i], data['arrivals'][str(i)], by_step[event['step_id']]
        if (arrival['scheduled_ns'] != request['arrival_offset_ns'] or arrival['actual_ns'] < arrival['scheduled_ns']
                or event['arrival_ns'] != request['arrival_offset_ns'] or event['prompt_tokens'] != len(request['prompt_ids'])
                or not max(arrival['actual_ns'], last.get(i,0), step['begin_ns']) <= event['emitted_ns'] <= step['end_ns']):
            raise ValueError('prefix event time or prompt differs')
        if active_prefill_cap is not None and event['emitted_ns'] != step['end_ns']-step['postprocess_ns']:
            raise ValueError('active prefix event differs from synchronous postprocess timestamp')
        last[i] = event['emitted_ns']
        history = histories[i]
        if event['kind'] == 'token':
            if active_prefill_cap is not None and (i,event['step_id']) in token_steps:
                raise ValueError('active prefix delivered two tokens for one request in a step')
            token_steps.add((i,event['step_id']))
            if (not 0 <= event['token_id'] < 151936 or event['generated_tokens'] != len(history)+1
                    or len(history) >= request['max_new_tokens'] or (history and history[-1] in request['stop_ids'])):
                raise ValueError('prefix delivered token order or stop differs')
            history.append(event['token_id'])
            last_token[i]=(event['step_id'],event['emitted_ns'])
        else:
            if (event['generated_tokens'] != len(history) or event['reason'] not in ('stop','length')
                    or (event['reason'] == 'length' and (len(history) != request['max_new_tokens'] or (history and history[-1] in request['stop_ids'])))
                    or (event['reason'] == 'stop' and (not history or history[-1] not in request['stop_ids']))):
                raise ValueError('prefix terminal reason or accounting differs')
            if active_prefill_cap is not None and last_token.get(i)!=(event['step_id'],event['emitted_ns']):
                raise ValueError('active prefix finish differs from its final delivered token')
            terminal[i] = event
    if set(terminal) != set(requests):
        raise ValueError('missing prefix terminal result')
    for step in steps:
        if sum(e['kind'] == 'finish' and e['step_id'] == step['step_id'] for e in events) != step['finished']:
            raise ValueError('prefix per-step finish accounting differs')
        if active_prefill_cap is not None:
            heads=sum(e['kind']=='token' and e['step_id']==step['step_id'] for e in events)
            if not step['decode_seqs']<=heads<=step['decode_seqs']+step['prefill_seqs']:
                raise ValueError('active prefix per-step delivered head census differs')
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
    if active_prefill_cap is not None:
        prefix = trace.get('prefix',{})
        for i in ([] if adaptive else requests):
            expected_hit = prefix.get('tokens',0) if i < prefix.get('reuse_requests',0) else 0
            if hit_by_request[i] != expected_hit:
                raise ValueError('active prefix per-request reuse differs from frozen workload')
        attended = sum((len(requests[i]['prompt_ids'])+len(tokens)-1)*(len(requests[i]['prompt_ids'])+len(tokens))//2
                       - hit_by_request[i]*(hit_by_request[i]+1)//2 for i,tokens in histories.items() if tokens)
        if sum(s['attended_positions'] for s in steps) != attended:
            raise ValueError('active prefix attended positions differ from necessary work')
    drain = data['drained']
    if (drain is None or drain['requests'] != len(requests) or drain['steps'] != len(steps)
            or drain['references'] or drain['private_blocks'] or drain['pending']
            or drain['free_blocks']+drain['cached_blocks'] != blocks
            or drain['elapsed_ns'] < max([s['end_ns'] for s in steps]+[e['emitted_ns'] for e in events]+[0])
            or (steps and (steps[-1]['blocks_free'] != drain['free_blocks'] or steps[-1]['waiting']))):
        raise ValueError('prefix phase did not drain ownership')
    return data


def parse_run(stdout, trace, cache_mode, blocks, active_prefill_cap=None,adaptive_prefill_mode=None,frequency_prefill_mode=None):
    """Reconstruct exact logical entries and complete request/phase accounting."""
    frequency=frequency_prefill_mode is not None
    if frequency and adaptive_prefill_mode is not None: raise ValueError('conflicting frequency/adaptive study modes')
    mode=frequency_prefill_mode if frequency else adaptive_prefill_mode
    adaptive=adaptive_prefill_mode is not None or frequency
    validate_trace(trace,adaptive,frequency)
    if adaptive:
        if mode not in (FREQUENCY_MODES if frequency else ADAPTIVE_MODES) or active_prefill_cap is not None:
            raise ValueError('invalid adaptive policy mode or conflicting fixed cap')
        active_prefill_cap=32 if mode=='fixed32' else 0
    if cache_mode not in ('enabled','disabled') or blocks != trace['prefix']['blocks']:
        raise ValueError('prefix execution differs from declared cell')
    active = active_prefill_cap is not None
    if active and (type(active_prefill_cap) is not int or active_prefill_cap not in ACTIVE_CAPS
                   or cache_mode != 'enabled' or (not adaptive and tuple(trace['prefix'][k] for k in ('tokens','reuse_requests','blocks')) not in ACTIVE_CELL_CONFIGS)):
        raise ValueError('active prefix execution differs from declared cell or policy')
    metadata, phase_times, phases = {}, {}, {name:_phase_data(adaptive,frequency) for name in ('seed','measure')}
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
        if name in ('device','mode','config','admission','study','work_capacity','cache','namespace','salt') or (active and not adaptive and name=='active_prefill_cap') or (adaptive and name=='prefill_policy'):
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
            if frequency and name.startswith('frequency_'):
                if len(fields)<2 or not data['steps'] or int(fields[1])!=data['steps'][-1]['step_id']:
                    raise ValueError('frequency observation outside its step')
                sid=int(fields[1])
                if name=='frequency_work' and len(fields)==3:
                    data['frequency_work'].append(dict(step_id=sid,ns=int(fields[2])))
                elif name=='frequency_snapshot' and len(fields)==29 and all(v in ('0','1') for v in fields[17:20]):
                    data['frequency_snapshots'].append(dict(step_id=sid,**dict(zip(FREQUENCY_SNAPSHOT_INTS,map(int,fields[2:17]))),
                        **dict(zip(('valid','dense','enabled'),(v=='1' for v in fields[17:20]))),
                        reason=fields[20],namespace=fields[21],cache_salt=fields[22],
                        **dict(zip(FREQUENCY_SNAPSHOT_LISTS,(_ids(v) for v in fields[23:29])))))
                elif name=='frequency_event' and len(fields)==9:
                    data['frequency_events'].append(dict(step_id=sid,**dict(zip(('event_id','cohort_id','request_id','ticket'),map(int,fields[2:6]))),kind=fields[6],phase=fields[7],reason=fields[8]))
                elif name=='frequency_decision' and len(fields)==14 and all(v in ('0','1') for v in fields[8:13]):
                    data['frequency_decisions'].append(dict(step_id=sid,**dict(zip(('request_id','ticket','cohort_id','gap_budget','predicted_interruptions','threshold'),map(int,fields[2:8]))),
                        **dict(zip(('member','in_run','active','dense','allowed'),(v=='1' for v in fields[8:13]))),reason=fields[13]))
                else: raise ValueError('invalid frequency observation schema')
            elif adaptive and name=='prefill_decision' and len(fields)==22:
                if not data['steps'] or int(fields[1])!=data['steps'][-1]['step_id'] or fields[19] not in ('0','1') or fields[20] not in ('0','1'):
                    raise ValueError('invalid adaptive decision step or boolean')
                data['prefill_decisions'].append(dict(step_id=int(fields[1]),**dict(zip(DECISION_FIELDS,map(int,fields[2:19]))),eligible=fields[19]=='1',applied=fields[20]=='1',reason=fields[21]))
            elif adaptive and name=='batch_selection' and len(fields)==7:
                if not data['steps'] or int(fields[1])!=data['steps'][-1]['step_id']:
                    raise ValueError('adaptive selection outside its step')
                data['selections'].append(dict(zip(('step_id','order','request_id','past','query_tokens','selected_head'),map(int,fields[1:]))))
            elif name == 'step' and len(fields) == len(contract.ENGINE_STEP_FIELDS)+1:
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
        admission=['reserved'],study=[contract.ENGINE_PREFIX_FREQUENCY_STUDY if frequency else contract.ENGINE_PREFIX_ADAPTIVE_STUDY if adaptive else contract.ENGINE_PREFIX_ACTIVE_STUDY if active else contract.ENGINE_PREFIX_STUDY],work_capacity=['256','8'],cache=[cache_mode],
        salt=[SALT],namespace=metadata.get('namespace'))
    if adaptive: expected['prefill_policy']=[mode]
    elif active:
        expected['active_prefill_cap']=[str(active_prefill_cap)]
    if metadata != expected or not metadata.get('namespace') or len(metadata['namespace'])!=1:
        raise ValueError('prefix device/backend or configuration changed')
    if phase_order != list(PHASES) or set(phase_times)!=set(PHASES) or entries or cleared != [blocks,0,0,0,0,0]:
        raise ValueError('prefix omitted phase or final complete cache clear')
    for left,right in zip(PHASES,PHASES[1:]):
        if phase_times[left]['end_ns'] > phase_times[right]['begin_ns']:
            raise ValueError('prefix phase times overlap or reverse')
    seed_trace = dict(kind='engine-token-trace-v1',schema_version=1,mode='offline',scripted_tokens=[11],requests=[trace['prefix']['seed_request']])
    _validate_phase(phases['seed'],seed_trace,blocks,0,0 if active else None,adaptive)
    _validate_phase(phases['measure'],trace,blocks,len(phases['seed']['steps']),active_prefill_cap,adaptive)
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
    result = dict(device='Apple M4 Pro/metal',mode='greedy',cache=cache_mode,blocks=blocks,
                  namespace=metadata['namespace'][0],phases=phases,phase_times=phase_times,cache_events=events)
    if adaptive:
        result['prefill_policy']=mode
        result['policy_replay']=_validate_adaptive_replay(result,trace,frequency)
    elif active:
        result['active_prefill_cap']=active_prefill_cap
    return result


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
    if 'prefill_policy' in parsed:
        result.update(prefill_policy=parsed['prefill_policy'],**_active_metrics(parsed),policy_replay=parsed['policy_replay'])
        if 'frequency_work' in parsed['phases']['measure']: _frequency_metrics(result,parsed,trace)
    elif 'active_prefill_cap' in parsed:
        result.update(active_prefill_cap=parsed['active_prefill_cap'],**_active_metrics(parsed))
    return result


def _frequency_metrics(result,parsed,trace):
    """Expose rare stalls per request and per labelled group, within each run only."""
    phase=parsed['phases']['measure']; gaps={}; hits={h['request_id']:h['cached_tokens'] for h in phase['hits']}
    def distribution(values):
        return dict(count=len(values),p50_ns=float(profile.np.percentile(values,50)) if values else None,
            p95_ns=float(profile.np.percentile(values,95)) if values else None,
            p99_ns=float(profile.np.percentile(values,99)) if values else None,maximum_ns=max(values) if values else None)
    for request in result['requests']:
        tokens=[e['emitted_ns'] for e in phase['events'] if e['kind']=='token' and e['request_id']==request['request_id']]
        values=[b-a for a,b in zip(tokens,tokens[1:])]; gaps[request['request_id']]=values
        request.update(gap_p95_ns=distribution(values)['p95_ns'],gap_maximum_ns=max(values) if values else None,
            gap_count=len(values),gaps_above_20ms=sum(v>20_000_000 for v in values),gaps_above_50ms=sum(v>50_000_000 for v in values))
    common=trace['prefix']['seed_request']['prompt_ids'][:-1]
    intended={r['request_id'] for r in trace['requests'] if r['prompt_ids'][:len(common)]==common}
    groups={}
    for name,ids in (('actual_hit',{i for i in gaps if hits.get(i,0)>0}),('actual_miss',{i for i in gaps if not hits.get(i,0)}),
                     ('intended_match',intended),('intended_miss',set(gaps)-intended)):
        selected=[r for r in result['requests'] if r['request_id'] in ids]; values=[v for i in ids for v in gaps[i]]
        groups[name]=dict(request_ids=sorted(ids),delivered_tokens=sum(r['delivered_tokens'] for r in selected),
            gap_count=len(values),gaps=distribution(values),
            maximum_request_p95_ns=max((r['gap_p95_ns'] for r in selected if r['gap_p95_ns'] is not None),default=None),
            maximum_request_gap_ns=max((r['gap_maximum_ns'] for r in selected if r['gap_maximum_ns'] is not None),default=None))
    events=phase['frequency_events']; snapshots=phase['frequency_snapshots']; decisions=phase['frequency_decisions']
    result.update(groups=groups,frequency_summary=dict(snapshots=snapshots,forecast_ns=sum(v['ns'] for v in phase['frequency_work']),
        lookup_count=sum(s['lookup_count'] for s in snapshots),event_counts=dict(Counter(e['kind'] for e in events)),
        invalidations=[dict(step_id=e['step_id'],reason=e['reason'],request_id=e['request_id'],ticket=e['ticket']) for e in events if e['kind']=='invalidated'],
        allowed_admissions=sum(d['allowed'] for d in decisions),applied_admissions=sum(d['applied'] for d in phase['prefill_decisions']),
        final_state=events[-1]['kind'] if events else 'not_executed'))


def _active_metrics(parsed):
    """Host gaps and the synchronous steps they crossed; no GPU timing inference."""
    phase=parsed['phases']['measure']; steps={s['step_id']:s for s in phase['steps']}
    def kind(step):
        return 'mixed' if step['prefill_tokens'] and step['decode_seqs'] else 'prefill_only' if step['prefill_tokens'] else 'decode_only'
    def describe(values):
        return dict(count=len(values),total_ns=sum(values),maximum_ns=max(values,default=0))
    previous={}; requests={}; attributed={name:[] for name in ('mixed','prefill_only','decode_only')}
    for event in phase['events']:
        if event['kind']!='token': continue
        i=event['request_id']; requests.setdefault(i,[])
        if i in previous:
            old=previous[i]; gap=event['emitted_ns']-old['emitted_ns']
            requests[i].append(gap)
            crossed={kind(steps[s]) for s in range(old['step_id']+1,event['step_id']+1)}
            category='mixed' if 'mixed' in crossed else 'prefill_only' if 'prefill_only' in crossed else 'decode_only'
            attributed[category].append(gap)
        previous[i]=event
    all_gaps=[gap for values in requests.values() for gap in values]
    def tails(values):
        return dict(describe(values),above_20ms=sum(v>20_000_000 for v in values),above_50ms=sum(v>50_000_000 for v in values))
    classes={}
    for name in attributed:
        selected=[s for s in steps.values() if kind(s)==name]
        classes[name]=dict(count=len(selected),prefill_rows=sum(s['prefill_tokens'] for s in selected),
            decode_rows=sum(s['decode_seqs'] for s in selected),execute=describe([s['execute_ns'] for s in selected]),
            host_step=describe([s['end_ns']-s['begin_ns'] for s in selected]),
            schedule_ns=sum(s['schedule_ns'] for s in selected),build_ns=sum(s['build_ns'] for s in selected),
            prefill_rows_max=max([s['prefill_tokens'] for s in selected],default=0))
    return dict(gap_tail=dict(tails(all_gaps),requests=[dict(request_id=i,**tails(values)) for i,values in sorted(requests.items())],
                    attribution={name:tails(values) for name,values in attributed.items()}),step_classes=classes)


def adaptive_decision(inputs,mode):
    """Literal frozen predicate; no timing observations or learned parameters."""
    d=inputs; a,k=d['available_blocks'],d['required_blocks']; count=d['selected_decodes']
    checks=[(k>a,'admission_blocked'),(k<=0,'no_private_blocks'),(count!=1,'decode_count'),
        (d['incumbent_remaining_outputs']<2,'incumbent_terminal'),
        (not 33<=d['remaining_prompt_tokens']<=64,'tail_length'),
        (count+1>=d['max_sequences'],'last_sequence_slot'),(a-k>=k,'capacity_headroom')]
    reason=next((reason for failed,reason in checks if failed),None)
    eligible=reason is None
    baseline=d['baseline_tokens']
    chosen=0 if k>a else min(baseline,32) if (mode=='fixed32' and count>0) or (mode=='adaptive' and eligible) else baseline
    applied=mode=='adaptive' and chosen<baseline and eligible
    if eligible:
        reason='adaptive_disabled' if mode!='adaptive' else 'adaptive_short_tail' if applied else 'baseline_at_most_32'
    return dict(chosen_tokens=chosen,eligible=eligible,applied=applied,reason=reason)


def frequency_threshold(gaps):
    if type(gaps) is not int or gaps<0 or gaps>(1<<63)-1: raise ValueError('invalid frequency gap budget')
    return gaps-(19*(gaps-1)//20) if gaps else 0


def _prefix_match(prompt,entries):
    result=[]
    for n in range(32,((len(prompt)-1)//32)*32+1,32):
        key=tuple(prompt[:n])
        if key not in entries: break
        result.append(key)
    return result


def frequency_snapshot(requests,tickets,waiting,entries,inputs,namespace,pool,step_id,cohort_id,
                       resident=False,watermark=0,budget=256,maximum_sequences=8):
    """Pure forecast from already-arrived FIFO inputs and the current exact index.

    No delivered future tokens, future trace arrivals, timing samples, index touches
    or ownership operations enter this calculation. Native integers are bounded.
    """
    anchor=inputs['request_id']; head=requests[anchor]; h=inputs['matched_tokens']; k=inputs['required_blocks']
    gap_budget=0; overflow=False
    for i in waiting:
        gaps=max(requests[i]['max_new_tokens']-1,0)
        if gaps>(1<<63)-1-gap_budget: overflow=True
        else: gap_budget+=gaps
    result=dict(step_id=step_id,cohort_id=cohort_id,anchor_request_id=anchor,anchor_ticket=tickets[anchor],
        last_ticket=tickets[waiting[-1]] if waiting else -1,request_count=len(waiting),gap_budget=gap_budget,
        run_count=0,predicted_interruptions=0,threshold=frequency_threshold(gap_budget),matched_tokens=h,
        remaining_prompt_tokens=len(head['prompt_ids'])-h,anchor_maximum_outputs=head['max_new_tokens'],
        required_blocks=k,available_blocks=inputs['available_blocks'],lookup_count=0,valid=False,dense=False,enabled=False,
        reason='dense',namespace=namespace,cache_salt=head.get('cache_salt',SALT),request_ids=list(waiting),
        tickets=[tickets[i] for i in waiting],prompt_lengths=[len(requests[i]['prompt_ids']) for i in waiting],
        maximum_outputs=[requests[i]['max_new_tokens'] for i in waiting],run_request_ids=[],run_tickets=[])
    reason=next((reason for failed,reason in (
        (resident or not waiting or waiting[0]!=anchor,'incompatible_initial_state'),(watermark!=0,'watermark'),
        (not 33<=result['remaining_prompt_tokens']<=64,'tail_length'),(head['max_new_tokens']<3,'output_limit'),
        (budget<result['remaining_prompt_tokens']+1,'row_budget'),(maximum_sequences<3,'sequence_slots'),
        (k<=0 or (pool-h//32)//max(k,1)!=2,'geometry'),(overflow,'gap_overflow'),(gap_budget==0,'zero_gap_budget')) if failed),None)
    if reason is None:
        anchor_keys=_prefix_match(head['prompt_ids'],entries)
        for position,i in enumerate(waiting):
            r=requests[i]
            if (r.get('cache_salt',SALT)!=result['cache_salt'] or len(r['prompt_ids'])!=len(head['prompt_ids'])
                    or r['max_new_tokens']!=head['max_new_tokens']): break
            keys=anchor_keys if i==anchor else _prefix_match(r['prompt_ids'],entries)
            if i!=anchor: result['lookup_count']+=1
            if keys!=anchor_keys: break
            if any(requests[previous]['prompt_ids'][h:h+32]==r['prompt_ids'][h:h+32] for previous in waiting[:position]):
                reason='duplicate_private_key'; break
            result['run_request_ids'].append(i); result['run_tickets'].append(tickets[i])
    result['run_count']=len(result['run_request_ids']); result['predicted_interruptions']=max(result['run_count']-1,0)
    result['valid']=reason is None
    result['dense']=result['valid'] and result['predicted_interruptions']>=result['threshold']
    result['enabled']=result['dense']
    result['reason']=reason or ('dense' if result['dense'] else 'sparse')
    return result


def _validate_adaptive_replay(parsed,trace,frequency=False):
    """Independent logical ownership/FIFO replay, including exact LRU and private reservations."""
    blocks=parsed['blocks']; entries={}; private={}; clock=0; stats=Counter()
    cohort=None; next_cohort=0; next_frequency_event=0
    def require(value,message):
        if not value: raise ValueError('adaptive replay: '+message)
    def free(): return blocks-len(entries)-sum(private.values())
    def refs(): return sum(private.values())+sum(len(e['owners']) for e in entries.values())
    for phase_name in ('seed','measure'):
        phase=parsed['phases'][phase_name]
        requests=[trace['prefix']['seed_request']] if phase_name=='seed' else trace['requests']
        requests={r['request_id']:r for r in requests}; ids=list(requests)
        generated={i:[] for i in ids}; computed={}; finished=set()
        tickets={i:(0 if phase_name=='seed' else i+1) for i in ids}
        if frequency:
            require(phase['steps'] and all(phase['arrivals'][str(i)]['actual_ns']<=phase['steps'][0]['begin_ns'] for i in ids),
                    'frozen cohort included a request observed after its first step began')
        mode='current' if phase_name=='seed' else parsed['prefill_policy']
        for step in phase['steps']:
            sid=step['step_id']; expected_events=[]
            expected_snapshots=[]; expected_frequency_decisions=[]; expected_frequency_events=[]
            def frequency_event(i,kind,stage,reason):
                nonlocal next_frequency_event
                expected_frequency_events.append(dict(step_id=sid,event_id=next_frequency_event,
                    cohort_id=cohort['cohort_id'],request_id=i,ticket=tickets[i] if i>=0 else -1,kind=kind,phase=stage,reason=reason))
                next_frequency_event+=1
            decodes=[i for i in ids if i in computed and i not in finished and generated[i]][:8]
            budget=256-len(decodes); chosen=[(i,1) for i in decodes]; incumbent=decodes[0] if len(decodes)==1 else -1
            expected_decisions=[]; expected_hits=[]; admission=0
            resident=next((i for i in ids if i in computed and i not in finished and not generated[i]),None)
            if len(decodes)<8 and resident is not None:
                remaining=len(requests[resident]['prompt_ids'])-computed[resident]
                count=min(budget,remaining,32 if mode=='fixed32' and decodes else 256)
                chosen.append((resident,count))
            elif len(decodes)<8:
                waiting=next((i for i in ids if i not in computed),None)
                if waiting is not None:
                    r=requests[waiting]; prompt=r['prompt_ids']; matched=[]
                    for n in range(32,((len(prompt)-1)//32)*32+1,32):
                        key=tuple(prompt[:n])
                        if key not in entries: break
                        matched.append(key)
                    h=len(matched)*32; required=(len(prompt)+r['max_new_tokens']-1+31)//32-len(matched)
                    reclaimable=sum(not e['owners'] and key not in matched for key,e in entries.items())
                    inputs=dict(request_id=waiting,prompt_length=len(prompt),maximum_outputs=r['max_new_tokens'],matched_tokens=h,
                        free_blocks=free(),reclaimable_blocks=reclaimable,watermark_blocks=0,available_blocks=free()+reclaimable,
                        required_blocks=required,selected_decodes=len(decodes),incumbent_request_id=incumbent,
                        incumbent_remaining_outputs=requests[incumbent]['max_new_tokens']-len(generated[incumbent]) if incumbent>=0 else -1,
                        remaining_prompt_tokens=len(prompt)-h,row_budget=budget,max_sequences=8,baseline_tokens=min(budget,len(prompt)-h))
                    frequency_allowed=False
                    if frequency:
                        if mode!='frequency':
                            expected_frequency_decisions.append(dict(step_id=sid,request_id=waiting,ticket=tickets[waiting],
                                cohort_id=-1,gap_budget=-1,predicted_interruptions=-1,threshold=-1,
                                member=False,in_run=False,active=False,dense=False,allowed=False,reason='not_executed'))
                        else:
                            if cohort is None and required<=inputs['available_blocks']:
                                cohort=frequency_snapshot(requests,tickets,[i for i in ids if i not in computed],entries,inputs,
                                    parsed['namespace'],blocks,sid,next_cohort,resident=bool(computed.keys()-finished))
                                next_cohort+=1; expected_snapshots.append(dict(cohort))
                                frequency_event(waiting,'frozen','admission',cohort['reason'])
                            member=cohort is not None and tickets[waiting] in cohort['tickets']
                            in_run=cohort is not None and tickets[waiting] in cohort['run_tickets']
                            if cohort is not None and cohort['valid'] and in_run:
                                anchor=requests[cohort['anchor_request_id']]['prompt_ids']
                                if (h!=cohort['matched_tokens'] or len(prompt)-h!=cohort['remaining_prompt_tokens']
                                        or r['max_new_tokens']!=cohort['anchor_maximum_outputs'] or required!=cohort['required_blocks']
                                        or prompt[:h]!=anchor[:cohort['matched_tokens']]):
                                    cohort['valid']=False; cohort['enabled']=False; cohort['reason']='geometry_changed'
                                    frequency_event(waiting,'invalidated','admission','geometry_changed')
                            active=cohort is not None and cohort['valid']; dense=cohort is not None and cohort['dense']
                            frequency_allowed=active and dense and in_run
                            reason=('unfrozen' if cohort is None else cohort['reason'] if not active else 'not_member' if not member else
                                    'outside_run' if not in_run else 'sparse' if not dense else 'allowed')
                            expected_frequency_decisions.append(dict(step_id=sid,request_id=waiting,ticket=tickets[waiting],
                                cohort_id=cohort['cohort_id'] if cohort else -1,gap_budget=cohort['gap_budget'] if cohort else 0,
                                predicted_interruptions=cohort['predicted_interruptions'] if cohort else 0,threshold=cohort['threshold'] if cohort else 0,
                                member=member,in_run=in_run,active=active,dense=dense,allowed=frequency_allowed,reason=reason))
                    policy='adaptive' if mode=='frequency' and frequency_allowed else 'current' if mode=='frequency' else mode
                    choice=adaptive_decision(inputs,policy)
                    if mode=='frequency' and choice['eligible'] and not frequency_allowed: choice['reason']='frequency_fallback'
                    decision=dict(step_id=sid,**inputs,**choice); expected_decisions.append(decision)
                    stats['decisions']+=1; stats['eligible']+=decision['eligible']; stats['applied']+=decision['applied']
                    if decision['chosen_tokens']>0:
                        while free()<required:
                            candidates=[key for key,e in entries.items() if not e['owners'] and key not in matched]
                            require(candidates,'admission cannot reclaim its claimed capacity')
                            victim=min(candidates,key=lambda key:entries[key]['stamp'])
                            del entries[victim]; expected_events.append(('removed',victim))
                        private[waiting]=required; computed[waiting]=h; clock+=1
                        for key in matched: entries[key]['owners'].add(waiting); entries[key]['stamp']=clock
                        chosen.append((waiting,decision['chosen_tokens'])); admission=1
                        if h: expected_hits.append(dict(step_id=sid,request_id=waiting,cached_tokens=h))
            actual_decisions=[d for d in phase['prefill_decisions'] if d['step_id']==sid]
            require(_same(actual_decisions,expected_decisions),'decision inputs, capacity, predicate or reason differs')
            actual_hits=[h for h in phase['hits'] if h['step_id']==sid]
            require(_same(actual_hits,expected_hits),'matched prefix differs from exact live index')
            expected_selections=[]; offset=0; positions=0
            for order,(i,count) in enumerate(chosen):
                past=computed[i]; history=requests[i]['prompt_ids']+generated[i]
                head=offset+count-1 if past+count==len(history) else -1
                expected_selections.append(dict(step_id=sid,order=order,request_id=i,past=past,query_tokens=count,selected_head=head))
                positions+=count*past+count*(count+1)//2; offset+=count
            actual_selections=[s for s in phase['selections'] if s['step_id']==sid]
            require(_same(actual_selections,expected_selections),'ordered request/past/query/head selection differs')
            require(step['decode_seqs']==len(decodes) and step['prefill_seqs']==int(len(chosen)>len(decodes))
                and step['prefill_tokens']==sum(c for _,c in chosen[len(decodes):]) and step['total_tokens']==offset
                and step['attended_positions']==positions and step['admitted']==admission,'step counters differ from ordered inputs')
            tokens=[e for e in phase['events'] if e['step_id']==sid and e['kind']=='token']
            require([e['request_id'] for e in tokens]==[s['request_id'] for s in expected_selections if s['selected_head']>=0],
                    'sampled heads are omitted, duplicated or reordered')
            by_id={e['request_id']:e for e in tokens}
            for selection in expected_selections:
                i=selection['request_id']; computed[i]+=selection['query_tokens']; history=requests[i]['prompt_ids']+generated[i]
                for n in range(32,computed[i]//32*32+1,32):
                    key=tuple(history[:n])
                    if key in entries: continue
                    require(private[i]>0,'publication has no completed private block')
                    private[i]-=1; clock+=1; entries[key]=dict(owners={i},stamp=clock)
                    expected_events.append(('stored',key))
                if i in by_id:
                    token=by_id[i]['token_id']; generated[i].append(token)
                    terminal=token in requests[i]['stop_ids'] or len(generated[i])==requests[i]['max_new_tokens']
                    finishes=[e for e in phase['events'] if e['step_id']==sid and e['kind']=='finish' and e['request_id']==i]
                    require(len(finishes)==int(terminal),'terminal event differs from actual selected token/limit')
                    if terminal:
                        if frequency and mode=='frequency' and cohort is not None and cohort['valid'] and len(generated[i])<requests[i]['max_new_tokens']:
                            cohort['valid']=False; cohort['enabled']=False; cohort['reason']='early_termination'
                            frequency_event(i,'invalidated','postprocess','early_termination')
                        finished.add(i); del private[i]
                        for entry in entries.values(): entry['owners'].discard(i)
            actual_events=[(e['kind'],tuple(e['tokens'])) for e in parsed['cache_events'] if e['phase']==phase_name and e['step_id']==sid]
            require(actual_events==expected_events,'publication or idle LRU eviction differs from completed ownership')
            state=next(s for s in phase['cache_states'] if s['step_id']==sid)
            require(state['free_blocks']==free() and state['cached_blocks']==len(entries) and state['references']==refs()
                and state['private_blocks']==sum(private.values()),'physical capacity/reference snapshot differs')
            require(step['waiting']==sum(i not in computed for i in ids),'FIFO waiting census differs')
            if frequency:
                if mode=='frequency' and cohort is not None and finished==set(ids):
                    frequency_event(-1,'closed','close','drained'); cohort=None
                work=[w for w in phase['frequency_work'] if w['step_id']==sid]
                require(len(work)==1 and 0<=work[0]['ns']<=step['schedule_ns'] and (mode=='frequency' or work[0]['ns']==0)
                    and (expected_frequency_decisions or work[0]['ns']==0),
                    'forecast cost omitted, outside scheduling or executed in control')
                for name,expected in (('frequency_snapshots',expected_snapshots),('frequency_decisions',expected_frequency_decisions),('frequency_events',expected_frequency_events)):
                    require(_same([v for v in phase[name] if v['step_id']==sid],expected),'frequency snapshot, admission or lifecycle differs')
                stats['forecast_snapshots']+=len(expected_snapshots)
                stats['forecast_invalidations']+=sum(e['kind']=='invalidated' for e in expected_frequency_events)
        require(finished==set(ids) and not private and refs()==0,'phase does not release every request owner')
    require(not [e for e in parsed['cache_events'] if e['phase']=='handoff'],'enabled handoff changed the cache')
    clears=[(e['kind'],tuple(e['tokens'])) for e in parsed['cache_events'] if e['phase']=='clear']
    require(clears==[('removed',key) for key in entries]+[('cleared',())],'final clear order or completeness differs')
    return dict(stats,logical_schedule_sha256=_hash(json.dumps(_schedule_signature(parsed),sort_keys=True,separators=(',',':'))))


def _schedule_signature(parsed):
    """Complete logical ordered schedule/cache observations, excluding time and policy diagnostics."""
    result={}
    for name in ('seed','measure'):
        data=parsed['phases'][name]
        result[name]=[dict(work={k:v for k,v in step.items() if not k.endswith('_ns') and k not in ('predicted_ns','budget_limited')},
            selections=[s for s in data['selections'] if s['step_id']==step['step_id']],
            hits=[h for h in data['hits'] if h['step_id']==step['step_id']],
            cache=[dict(kind=e['kind'],tokens=e['tokens']) for e in parsed['cache_events'] if e['phase']==name and e['step_id']==step['step_id']]) for step in data['steps']]
    return result


def _execution(value, command, timeout, minimum_ns=0):
    if (value.get('command') != list(map(str,command)) or type(value.get('exit_code')) is not int
            or value['exit_code'] != 0 or type(value.get('timeout_seconds')) is not int
            or value['timeout_seconds'] != timeout or type(value.get('wall_elapsed_ns')) is not int
            or not max(1,minimum_ns) <= value['wall_elapsed_ns'] <= timeout*1_000_000_000):
        raise ValueError('prefix native execution receipt changed or failed')


def parse_checkpoint(stdout, required_groups=None, active_prefill=False,adaptive_prefill=False,frequency_prefill=False):
    """Check native arithmetic identity and the explicitly required coverage."""
    identity, checks = {},[]
    if required_groups is None:
        required_groups=FREQUENCY_CHECKPOINT_GROUPS if frequency_prefill else ADAPTIVE_CHECKPOINT_GROUPS if adaptive_prefill else ACTIVE_CHECKPOINT_GROUPS if active_prefill else CHECKPOINT_GROUPS
    for line in stdout.splitlines():
        fields = line.split()
        if fields and fields[0] in (('device','qualification','configuration','numerical-scope') if frequency_prefill else ('device','qualification','configuration')) and fields[0] not in identity:
            identity[fields[0]]=fields[1:]
        elif len(fields)==8 and fields[0]=='check' and fields[2::2]==['checked','mismatches','unexpected-nonfinite']:
            checks.append(dict(name=fields[1],checked=int(fields[3]),mismatches=int(fields[5]),nonfinite=int(fields[7])))
        else:
            raise ValueError('unknown prefix checkpoint qualification line')
    expected=dict(device=['Apple','M4','Pro/metal'],qualification=[FREQUENCY_CHECKPOINT_IDENTITY if frequency_prefill else ADAPTIVE_CHECKPOINT_IDENTITY if adaptive_prefill else ACTIVE_CHECKPOINT_IDENTITY if active_prefill else CHECKPOINT_IDENTITY],
                  configuration=['27','projection','8','dtype','BF16/FP32','block-size','32'])
    if frequency_prefill: expected['numerical-scope']=['scaled-prefix-32','pool','7','context','128']
    if (identity != expected or tuple(c['name'] for c in checks)!=required_groups
            or any(c['checked']<1 or c['mismatches'] or c['nonfinite'] for c in checks)):
        raise ValueError('prefix checkpoint qualification omitted or failed a required group')
    return dict(numerical_passed=True,checks=checks)


def _qualified_namespace(build):
    return 'qwen-reference27-projection8/'+build['assets']['prepared']


def _parse_policy(stdout,trace,run,active=False,adaptive=False,frequency=False):
    return parse_run(stdout,trace,run['cache'],trace['prefix']['blocks'],
                     run['active_prefill_cap'] if active and not adaptive else None,
                     run['prefill_policy'] if adaptive and not frequency else None,
                     run['prefill_policy'] if frequency else None)


def _schedule_matches(trace,schedules,frequency=False):
    """Frozen mechanism claims concern all logical selections, not timings."""
    p=trace['prefix']; known=p['suffix_tokens']==64 and p['max_new_tokens']==16 and p['match_order']=='first'
    if (not known or p['blocks'] not in (40,384) or (frequency and (p['cohort_requests']!=8
            or p['reuse_requests'] not in ((4,8) if p['blocks']==40 else (0,4,8))))): return dict(required=False)
    candidate='frequency' if frequency else 'adaptive'
    reference=('adaptive' if p['reuse_requests']==8 else 'current') if frequency and p['blocks']==40 else 'fixed32' if p['blocks']==40 else 'current'
    expected_steps=({4:113 if frequency else 115,8:69} if p['blocks']==40 else {0:55,4:39,8:23})[p['reuse_requests']]
    if not _same(schedules[candidate],schedules[reference]) or len(schedules[candidate]['measure'])!=expected_steps:
        raise ValueError('adaptive known-cell logical schedule differs from frozen reference')
    result=dict(required=True,reference=reference,matched=True,measured_steps=expected_steps)
    if frequency and p['blocks']==40:
        mixed=sum(bool(step['work']['decode_seqs'] and step['work']['prefill_tokens']) for step in schedules[candidate]['measure'])
        if mixed!=({4:3,8:14}[p['reuse_requests']]): raise ValueError('frequency known mixed-step census changed')
        result['mixed_steps']=mixed
    return result


def _qualification(record, require_preflights=True):
    frequency=_frequency(record); adaptive=_adaptive(record) or frequency; active=_active(record) or adaptive
    build = profile.validate_engine_record_build(record)
    document = record.get('qualification_document')
    if (type(document) is not str or _hash(document) != record.get('qualification_sha256')
            or not _same(json.loads(document),record.get('qualification'))):
        raise ValueError('prefix qualification document changed')
    qualification = record['qualification']
    checkpoint = qualification.get('checkpoint_receipt',{})
    actual = checkpoint.get('build',{})
    if (qualification.get('kind') != ('engine-prefix-frequency-qualification-v1' if frequency else 'engine-prefix-adaptive-qualification-v1' if adaptive else 'engine-prefix-active-qualification-v1' if active else 'engine-prefix-qualification-v1')
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
    _execution(checkpoint.get('execution',{}),[command[-1],build['assets']['prepared'],
               'prefix-frequency-qualification' if frequency else 'prefix-adaptive-qualification' if adaptive else 'prefix-active-qualification' if active else 'prefix-cache-qualification'],180)
    stdout = checkpoint.get('stdout')
    if type(stdout) is not str or _hash(stdout)!=checkpoint.get('stdout_sha256'):
        raise ValueError('prefix qualification stdout changed')
    checked=parse_checkpoint(stdout,active_prefill=active,adaptive_prefill=adaptive,frequency_prefill=frequency)
    if not require_preflights:
        return checked
    document=qualification.get('campaign_document')
    if (type(document) is not str or _hash(document)!=qualification.get('campaign_sha256')
            or not _same(json.loads(document),qualification.get('campaign'))):
        raise ValueError('prefix qualification campaign document changed')
    campaign=validate_campaign(qualification['campaign'],active,adaptive,frequency)
    if ('trace' in record and not _same(record['trace'],campaign)):
        raise ValueError('prefix qualification does not bind measured campaign')
    if (campaign['tokenizer']['tables_sha256']!=build['assets']['tables_sha256']
            or campaign['tokenizer']['generator_source_sha256']!=build['source']['sources'].get('src/llm_mojo/benchmarks/engine_prefix.py')):
        raise ValueError('prefix qualification tokenizer/generator changed')
    preflights=qualification.get('preflight_cells',[])
    if len(preflights)!=(20 if frequency else 12 if adaptive else 5 if active else 8):
        raise ValueError('prefix qualification requires all declared untimed preflights')
    expected={}; expected_work={}; expected_schedules={}; matches={}
    for cell,trace in zip(preflights,campaign['traces']):
        params=trace['prefix']; name=_cell_name(trace,adaptive,frequency)
        observed_policies=[(r.get('cache'),r.get('prefill_policy')) for r in cell.get('runs',[])] if adaptive else [(r.get('cache'),r.get('active_prefill_cap')) for r in cell.get('runs',[])] if active else [r.get('cache') for r in cell.get('runs',[])]
        policies=[('enabled',mode) for mode in (FREQUENCY_MODES if frequency else ADAPTIVE_MODES)] if adaptive else [('enabled',cap) for cap in ACTIVE_CAPS] if active else ['disabled','enabled']
        if (cell.get('cell_id')!=name or not _same(cell.get('trace'),trace)
                or cell.get('native_trace_sha256')!=_hash(trace_tsv(trace,adaptive,frequency))
                or not Path(cell.get('native_trace_path','')).is_absolute()
                or observed_policies!=policies):
            raise ValueError('prefix qualification preflight identity/census changed')
        oracle=None; work=None; schedules={}
        for run in cell['runs']:
            parsed=_parse_policy(run['stdout'],trace,run,active,adaptive,frequency)
            if not _same(parsed,run.get('parsed')) or parsed['namespace']!=_qualified_namespace(build):
                raise ValueError('prefix preflight parsed identity differs from qualified source')
            minimum=parsed['phase_times']['clear']['end_ns']-parsed['phase_times']['seed']['begin_ns']
            _execution(run.get('native_execution',{}),_command(record,cell,run),180,minimum)
            histories=_histories(parsed,trace)
            if oracle is None:
                oracle=histories
            if not _same(histories,oracle):
                raise ValueError('prefix untimed cache hit/miss natural histories differ')
            if active:
                current_work=_work_signature(parsed,trace)
                if work is None: work=current_work
                if not _same(work,current_work):
                    raise ValueError('active prefix untimed per-request reuse or total work differs')
            if adaptive: schedules[run['prefill_policy']]=_schedule_signature(parsed)
        if not _same(cell.get('expected_histories'),oracle):
            raise ValueError('prefix qualification frozen histories changed')
        expected[name]=oracle
        if active:
            if not _same(cell.get('expected_work'),work):
                raise ValueError('active prefix qualification frozen work changed')
            expected_work[name]=work
        if adaptive:
            if not _same(cell.get('expected_schedules'),schedules): raise ValueError('adaptive frozen logical schedules changed')
            expected_schedules[name]=schedules; matches[name]=_schedule_matches(trace,schedules,frequency)
    result=dict(**checked,untimed_preflights=60 if frequency else 36 if adaptive else 15 if active else 16,expected_histories=expected,campaign_sha256=qualification['campaign_sha256'])
    if active: result['expected_work']=expected_work
    if adaptive: result.update(expected_schedules=expected_schedules,logical_schedule_matches=matches)
    return result


def qualify(directory, output, trace_path, active_prefill=False,adaptive_prefill=False,frequency_prefill=False):
    adaptive_prefill=adaptive_prefill or frequency_prefill
    frequency=frequency_prefill
    active=active_prefill or adaptive_prefill
    declaration=_declaration(active,adaptive_prefill,frequency_prefill)
    build=profile.verify_engine_build(directory,declaration)
    ensure_record_location(output)
    output.mkdir(parents=True,exist_ok=False)
    binary=output/'checkpoint-driver'
    command=[profile.environment_tool('mojo'),'build','-I','src','-I','tests','tests/engine_metal_driver.mojo','-o',binary]
    _,compiled=profile.checked_execution(command,output/'checkpoint-build.log',600)
    if profile.verify_build(directory)!=build:
        raise ValueError('prefix build changed during qualification compilation')
    numerical=dict(source=build['source'],assets=build['assets'],environment=build['environment'],
        command=list(map(str,command)),binaries={'checkpoint-driver':dict(sha256=sha(binary),bytes=binary.stat().st_size)})
    stdout,execution=profile.checked_execution([binary,build['assets']['prepared'],
        'prefix-frequency-qualification' if frequency_prefill else 'prefix-adaptive-qualification' if adaptive_prefill else 'prefix-active-qualification' if active else 'prefix-cache-qualification'],output/'checkpoint-native.log',180)
    if profile.verify_build(directory)!=build or sha(binary)!=numerical['binaries']['checkpoint-driver']['sha256']:
        raise ValueError('prefix build changed during qualification execution')
    qualification=dict(kind='engine-prefix-frequency-qualification-v1' if frequency_prefill else 'engine-prefix-adaptive-qualification-v1' if adaptive_prefill else 'engine-prefix-active-qualification-v1' if active else 'engine-prefix-qualification-v1',schema_version=1,build=build,
        checkpoint_receipt=dict(build=numerical,build_execution=compiled,stdout=stdout,stdout_sha256=_hash(stdout),execution=execution))
    def checker(require_preflights=True):
        document=json.dumps(qualification,indent=2,allow_nan=False)+'\n'
        record=dict(kind=declaration['kind'],declaration=declaration,
            build=build,qualification=qualification,qualification_document=document,qualification_sha256=_hash(document))
        return _qualification(record,require_preflights)
    checker(False)
    campaign_document=trace_path.read_text()
    campaign=validate_campaign(json.loads(campaign_document),active,adaptive_prefill,frequency_prefill)
    if any(ch.isspace() for ch in build['assets']['prepared']):
        raise ValueError('bounded prefix benchmark requires whitespace-free prepared path')
    qualification.update(campaign=campaign,campaign_document=campaign_document,campaign_sha256=_hash(campaign_document),preflight_cells=[])
    try:
        for trace in campaign['traces']:
            params=trace['prefix']; name=_cell_name(trace,adaptive_prefill,frequency_prefill)
            cell_dir=output/name; cell_dir.mkdir()
            native=cell_dir/'trace.tsv'; native.write_text(trace_tsv(trace,adaptive_prefill,frequency_prefill))
            cell=dict(cell_id=name,trace=trace,native_trace_path=str(native),native_trace_sha256=sha(native),runs=[])
            qualification['preflight_cells'].append(cell)
            oracle=None; work=None; schedules={}
            policies=([dict(cache='enabled',prefill_policy=mode) for mode in (FREQUENCY_MODES if frequency_prefill else ADAPTIVE_MODES)] if adaptive_prefill else
                      [dict(cache='enabled',active_prefill_cap=cap) for cap in ACTIVE_CAPS] if active else [dict(cache=cache) for cache in ('disabled','enabled')])
            for run in policies:
                command=_command(dict(build=build,kind=declaration['kind']),cell,run)
                label=run['prefill_policy'] if adaptive_prefill else f"cap-{run['active_prefill_cap']}" if active else run['cache']
                log=cell_dir/f'preflight-{label}.log'
                if active: cell['runs'].append(run)
                try:
                    stdout,execution=profile.checked_execution(command,log,180)
                    run.update(stdout=stdout,native_execution=execution)
                    parsed=_parse_policy(stdout,trace,run,active,adaptive_prefill,frequency_prefill)
                except Exception as error:
                    if active: _retain_failure(run,log,error)
                    raise
                run.update(stdout=stdout,native_execution=execution,parsed=parsed)
                if not active: cell['runs'].append(run)
                actual=_histories(parsed,trace)
                if oracle is None: oracle=actual
                if not _same(oracle,actual):
                    raise ValueError('prefix qualification histories differ before timing')
                if active:
                    actual_work=_work_signature(parsed,trace)
                    if work is None: work=actual_work
                    if not _same(work,actual_work):
                        raise ValueError('active prefix qualification work differs before timing')
                if adaptive_prefill: schedules[run['prefill_policy']]=_schedule_signature(parsed)
            cell['expected_histories']=oracle
            if active: cell['expected_work']=work
            if adaptive_prefill:
                cell['expected_schedules']=schedules
                _schedule_matches(trace,schedules,frequency)
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
    frequency=_frequency(record); adaptive=_adaptive(record); active=_active(record)
    result=[record['build']['command'][-1],record['build']['assets']['prepared'],cell['native_trace_path'],
            'chunked',cell['trace']['prefix']['blocks'],256,8,10,'greedy','reserved',
            contract.ENGINE_PREFIX_FREQUENCY_STUDY if frequency else contract.ENGINE_PREFIX_ADAPTIVE_STUDY if adaptive else contract.ENGINE_PREFIX_ACTIVE_STUDY if active else contract.ENGINE_PREFIX_STUDY,run['cache']]
    if adaptive or frequency: result.append(run['prefill_policy'])
    elif active: result.append(run['active_prefill_cap'])
    return result


def _histories(parsed,trace):
    def histories(phase,requests):
        return [dict(request_id=r['request_id'],tokens=[e['token_id'] for e in parsed['phases'][phase]['events']
                 if e['kind']=='token' and e['request_id']==r['request_id']],
                 reason=next(e['reason'] for e in parsed['phases'][phase]['events']
                 if e['kind']=='finish' and e['request_id']==r['request_id'])) for r in requests]
    return dict(seed=histories('seed',[trace['prefix']['seed_request']]),measure=histories('measure',trace['requests']))


def _work_signature(parsed,trace):
    result={}
    for name in ('seed','measure'):
        phase=parsed['phases'][name]
        ids=[r['request_id'] for r in ([trace['prefix']['seed_request']] if name=='seed' else trace['requests'])]
        hits={h['request_id']:h['cached_tokens'] for h in phase['hits']}
        result[name]=dict(rows=sum(s['total_tokens'] for s in phase['steps']),
            attended_positions=sum(s['attended_positions'] for s in phase['steps']),
            cached_tokens_by_request={str(i):hits.get(i,0) for i in ids})
    return result


def summary(record):
    frequency=_frequency(record); adaptive=_adaptive(record) or frequency; active=_active(record) or adaptive; declaration=_declaration(active,adaptive,frequency)
    if record.get('declaration')!=declaration or record.get('kind')!=declaration['kind']:
        raise ValueError('prefix campaign declaration changed')
    qualification=_qualification(record)
    campaign=validate_campaign(record['trace'],active,adaptive,frequency)
    if (type(record.get('trace_document')) is not str or _hash(record['trace_document'])!=record.get('trace_sha256')
            or not _same(json.loads(record['trace_document']),campaign)
            or len(record.get('cells',[]))!=(20 if frequency else 12 if adaptive else 5 if active else 8)):
        raise ValueError('prefix campaign document or cell census changed')
    if campaign['tokenizer']['tables_sha256']!=record['build']['assets']['tables_sha256']:
        raise ValueError('prefix workload tokenizer differs from qualified build')
    if campaign['tokenizer']['generator_source_sha256']!=record['build']['source']['sources'].get('src/llm_mojo/benchmarks/engine_prefix.py'):
        raise ValueError('prefix workload generator differs from qualified source')
    cells_out=[]
    for cell,trace in zip(record['cells'],campaign['traces']):
        params=trace['prefix']
        name=_cell_name(trace,adaptive,frequency)
        if (cell.get('cell_id')!=name or not _same(cell.get('trace'),trace)
                or cell.get('native_trace_sha256')!=_hash(trace_tsv(trace,adaptive,frequency))
                or not Path(cell.get('native_trace_path','')).is_absolute()):
            raise ValueError('prefix native trace identity changed')
        runs=cell.get('runs',[])
        expected=[(b,'enabled',mode,calibration) for b in range(4) for mode,calibration in (frequency_cells(b) if frequency else adaptive_cells(b))] if adaptive else [(b,'enabled',cap,calibration) for b in range(4) for cap,calibration in active_cells(b)] if active else [(b,cache,calibration) for b in range(4) for cache,calibration in cells(b)]
        observed=[(r.get('block'),r.get('cache'),r.get('prefill_policy' if adaptive else 'active_prefill_cap'),r.get('calibration')) for r in runs] if active else [(r.get('block'),r.get('cache'),r.get('calibration')) for r in runs]
        if observed!=expected:
            raise ValueError('incomplete, duplicate or reordered prefix paired grid')
        histories=qualification['expected_histories'][name]
        summaries=[]; parsed_runs={}
        for run in runs:
            parsed=_parse_policy(run['stdout'],trace,run,active,adaptive,frequency)
            if not _same(parsed,run.get('parsed')):
                raise ValueError('prefix parsed record differs from native stdout')
            if parsed['namespace']!=_qualified_namespace(record['build']):
                raise ValueError('prefix measured namespace differs from qualified build')
            minimum=parsed['phase_times']['clear']['end_ns']-parsed['phase_times']['seed']['begin_ns']
            _execution(run.get('native_execution',{}),_command(record,cell,run),180,minimum)
            actual=_histories(parsed,trace)
            if not _same(histories,actual):
                raise ValueError('prefix natural greedy histories or terminal reasons diverged')
            if active and not _same(_work_signature(parsed,trace),qualification['expected_work'][name]):
                raise ValueError('active prefix measured reuse or total work differs from preflight')
            if adaptive and not _same(_schedule_signature(parsed),qualification['expected_schedules'][name][run['prefill_policy']]):
                raise ValueError('adaptive measured logical schedule differs from preflight')
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
            if frequency: parsed_runs[run['block'],run['prefill_policy'],run['calibration']]=parsed
        if active:
            choices=('adaptive','frequency') if frequency else ('fixed32','adaptive') if adaptive else (32,16)
            comparisons={str(cap):_active_comparisons(summaries,cap,adaptive) for cap in choices}
            cells_out.append(dict(cell_id=name,runs=summaries,comparisons=comparisons,frozen_histories=histories,
                frozen_work=qualification['expected_work'][name],decisions={str(cap):_active_decision(comparisons[str(cap)]) for cap in choices}))
            if adaptive:
                cells_out[-1].update(logical_schedule_matches=qualification['logical_schedule_matches'][name],
                    schedule_sha256={mode:_hash(json.dumps(schedule,sort_keys=True,separators=(',',':'))) for mode,schedule in qualification['expected_schedules'][name].items()})
            if frequency:
                for choice in choices: cells_out[-1]['decisions'][choice]['p95_no_harm']=_p95_no_harm(comparisons[choice]['p95_gap_ns'])
                cells_out[-1]['forecast_vs_current']=[_forecast_vs_current(b,parsed_runs[b,'frequency',False],parsed_runs[b,'current',False],trace) for b in range(4)]
            continue
        by_key={(r['block'],r['cache'],r['calibration']):r for r in summaries}
        comparisons={}
        for metric in ('duration_ns','episode_ns'):
            self_ratios=[by_key[b,'disabled',True][metric]/by_key[b,'disabled',False][metric] for b in range(4)]
            ratios=[by_key[b,'enabled',False][metric]/by_key[b,'disabled',False][metric] for b in range(4)]
            floor=max(.05,max(abs(v-1) for v in self_ratios))
            comparisons[metric]=dict(block_ratios=ratios,median_ratio=statistics.median(ratios),
                calibration_ratios=self_ratios,noise_floor=floor,speed_verdict=profile._outcome(ratios,floor))
        cells_out.append(dict(cell_id=name,runs=summaries,comparisons=comparisons,frozen_histories=histories))
    result=dict(cells=cells_out,qualification=qualification,measured_runs=320 if frequency else 192 if adaptive else 80 if active else 96,untimed_preflights=60 if frequency else 36 if adaptive else 15 if active else 16,
                same_delivered_histories=True,asynchronous=False,promotion=False,target=None,goodput=None,quantile_method='linear')
    if active:
        primary=next(c for c in cells_out if c['cell_id']==declaration['usefulness']['primary_cell'])
        if frequency:
            broad=all(c['decisions']['frequency']['guards_pass'] for c in cells_out)
            p95=all(c['decisions']['frequency']['p95_no_harm']['status']=='pass' for c in cells_out)
            result.update(same_reuse_work=True,usefulness=dict(primary_cell=primary['cell_id'],primary_policy='frequency',
                primary=primary['decisions']['frequency'],broad_guard_pass=broad,broad_p95_guard_pass=p95,
                study_success=primary['decisions']['frequency']['hypothesis']=='supported' and broad and p95,
                adaptive_comparison='Descriptive paired ratios only; no equivalence claim.',
                uncertainty='Engineering median-ratio plus/minus self-control noise buffer; not a confidence interval.'))
            return result
        if adaptive:
            result.update(same_reuse_work=True,usefulness=dict(primary_cell=primary['cell_id'],primary_policy='adaptive',
                primary=primary['decisions']['adaptive'],broad_guard_pass=all(c['decisions']['adaptive']['guards_pass'] for c in cells_out),
                fixed32_comparison='Descriptive paired ratios only; no equivalence claim.',
                uncertainty='Engineering median-ratio plus/minus self-control noise buffer; not a confidence interval.'))
            return result
        result.update(same_reuse_work=True,usefulness=dict(primary_cell=primary['cell_id'],primary_cap=32,
            primary=primary['decisions']['32'],broad_guard_pass=all(c['decisions']['32']['guards_pass'] for c in cells_out),
            exploratory_cap16=dict(primary=primary['decisions']['16'],broad_guard_pass=all(c['decisions']['16']['guards_pass'] for c in cells_out)),
            uncertainty='Engineering median-ratio plus/minus self-control noise buffer; not a confidence interval.'))
    return result


def _active_comparisons(runs,cap,adaptive=False):
    by_key={(r['block'],r['prefill_policy' if adaptive else 'active_prefill_cap'],r['calibration']):r for r in runs}
    def value(run,metric):
        if metric=='p95_gap_ns': return (run['inter_token'] or {}).get('p95_ns')
        if metric=='p95_ttft_ns': return (run['ttft'] or {}).get('p95_ns')
        if metric=='maximum_gap_ns': return run['gap_tail']['maximum_ns']
        return run[metric]
    comparisons={}
    for metric in ('duration_ns','episode_ns','p95_ttft_ns','p95_gap_ns','maximum_gap_ns'):
        baseline=[value(by_key[b,'current' if adaptive else 0,False],metric) for b in range(4)]
        controls=[value(by_key[b,'current' if adaptive else 0,True],metric) for b in range(4)]
        candidate=[value(by_key[b,cap,False],metric) for b in range(4)]
        if any(v is None or v<=0 for v in baseline+controls+candidate):
            comparisons[metric]=dict(available=False,block_ratios=[],calibration_ratios=[],median_ratio=None,noise_floor=None,speed_verdict='inconclusive')
            continue
        ratios=[a/b for a,b in zip(candidate,baseline)]; self_ratios=[a/b for a,b in zip(controls,baseline)]
        noise=max(.05,max(abs(v-1) for v in self_ratios))
        comparisons[metric]=dict(available=True,block_ratios=ratios,median_ratio=statistics.median(ratios),
            calibration_ratios=self_ratios,noise_floor=noise,speed_verdict=profile._outcome(ratios,noise))
    return comparisons


def _active_decision(comparisons):
    rules=contract.ENGINE_PREFIX_ACTIVE_DECLARATION['usefulness']
    guards={}
    for name,metric,limit in (('completion','duration_ns',rules['completion_ratio_max']),('p95_ttft','p95_ttft_ns',rules['p95_ttft_ratio_max'])):
        c=comparisons[metric]
        if not c['available']:
            guards[name]=dict(status='inconclusive',buffer=None)
            continue
        low,high=c['median_ratio']-c['noise_floor'],c['median_ratio']+c['noise_floor']
        guards[name]=dict(status='pass' if high<=limit else 'fail' if low>limit else 'inconclusive',buffer=[low,high],limit=limit)
    worst=comparisons['maximum_gap_ns']
    status='inconclusive'
    if worst['available']:
        if worst['median_ratio']<=1+worst['noise_floor']: status='pass'
        elif all(v>1 for v in worst['block_ratios']): status='fail'
    guards['worst_gap']=dict(status=status)
    gap=comparisons['p95_gap_ns']; gap_status='inconclusive'
    if gap['available']:
        effect=1-gap['median_ratio']
        if all(v<1 for v in gap['block_ratios']) and effect>gap['noise_floor']:
            gap_status='pass' if gap['median_ratio']<=rules['p95_gap_ratio_max'] else 'fail'
        elif gap['speed_verdict']=='slower': gap_status='fail'
    statuses=[gap_status,*[g['status'] for g in guards.values()]]
    return dict(p95_gap=dict(status=gap_status,ratio_limit=rules['p95_gap_ratio_max']),guards=guards,
        guards_pass=all(g['status']=='pass' for g in guards.values()),
        hypothesis='supported' if all(s=='pass' for s in statuses) else 'failed' if 'fail' in statuses else 'inconclusive')


def _p95_no_harm(comparison):
    rules=contract.ENGINE_PREFIX_FREQUENCY_DECLARATION['usefulness']; status='inconclusive'; buffer=None
    if comparison['available']:
        m,n=comparison['median_ratio'],comparison['noise_floor']; buffer=[m-n,m+n]
        if m<=rules['p95_no_harm_ratio_max'] and m+n<=rules['p95_no_harm_buffer_max']: status='pass'
        elif all(r>1 for r in comparison['block_ratios']) and m-1>n: status='fail'
    return dict(status=status,median_limit=rules['p95_no_harm_ratio_max'],buffer_limit=rules['p95_no_harm_buffer_max'],buffer=buffer)


def _forecast_vs_current(block,candidate,current,trace):
    snapshots=candidate['phases']['measure']['frequency_snapshots']
    if len(snapshots)!=1: raise ValueError('frequency campaign requires one frozen measured busy period')
    snapshot=snapshots[0]; run_ids=set(snapshot['run_request_ids'])
    phase=current['phases']['measure']; steps={s['step_id']:s for s in phase['steps']}; previous={}; interruptions=0
    prompts={r['request_id']:len(r['prompt_ids']) for r in trace['requests']}
    for event in phase['events']:
        if event['kind']!='token': continue
        i=event['request_id']
        if i in previous and i in run_ids:
            for sid in range(previous[i]['step_id']+1,event['step_id']+1):
                step=steps[sid]
                if (step['decode_seqs'] and step['prefill_tokens'] and any(s['step_id']==sid and s['request_id'] in run_ids
                        and s['past']<prompts[s['request_id']] for s in phase['selections'])):
                    interruptions+=1; break
        previous[i]=event
    baseline=_active_metrics(current)['gap_tail']; observed=_active_metrics(candidate)['gap_tail']
    return dict(block=block,cohort_id=snapshot['cohort_id'],gap_budget=snapshot['gap_budget'],
        predicted_interruptions=snapshot['predicted_interruptions'],threshold=snapshot['threshold'],
        valid=snapshot['valid'],dense=snapshot['dense'],reason=snapshot['reason'],
        actual_current_gaps=baseline['count'],actual_current_mixed_gaps=baseline['attribution']['mixed']['count'],
        actual_current_run_interruptions=interruptions,actual_frequency_gaps=observed['count'],
        actual_frequency_mixed_gaps=observed['attribution']['mixed']['count'])


def collect(directory,trace_path,output,qualification_path,active_prefill=False,adaptive_prefill=False,frequency_prefill=False):
    adaptive_prefill=adaptive_prefill or frequency_prefill
    frequency=frequency_prefill
    """Run the selected frozen campaign once, retaining every actual invocation."""
    active=active_prefill or adaptive_prefill
    declaration=_declaration(active,adaptive_prefill,frequency_prefill)
    build=profile.verify_engine_build(directory,declaration)
    document=trace_path.read_text()
    campaign=validate_campaign(json.loads(document),active,adaptive_prefill,frequency_prefill)
    if campaign['tokenizer']['tables_sha256']!=build['assets']['tables_sha256']:
        raise ValueError('prefix campaign tokenizer differs from clean build')
    if campaign['tokenizer']['generator_source_sha256']!=build['source']['sources'].get('src/llm_mojo/benchmarks/engine_prefix.py'):
        raise ValueError('prefix campaign generator differs from clean build')
    if any(ch.isspace() for ch in build['assets']['prepared']):
        raise ValueError('bounded native benchmark requires a whitespace-free prepared path')
    qualification_document=qualification_path.read_text()
    record=dict(kind=declaration['kind'],declaration=declaration,
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
        try:
            if time.monotonic()+180>deadline:
                raise TimeoutError('prefix campaign reached its two-hour bound')
            before=profile.conditions()
            stdout,execution=profile.checked_execution(_command(record,cell,run),log,180)
            run.update(stdout=stdout,native_execution=execution,conditions_before=before)
            run['conditions_after']=profile.conditions()
            run['parsed']=_parse_policy(stdout,cell['trace'],run,active,adaptive_prefill,frequency_prefill)
        except Exception as error:
            if active: _retain_failure(run,log,error)
            raise
        return run
    try:
        for trace in campaign['traces']:
            params=trace['prefix']
            name=_cell_name(trace,adaptive_prefill,frequency_prefill)
            cell_dir=output/name
            cell_dir.mkdir()
            native=cell_dir/'trace.tsv'
            native.write_text(trace_tsv(trace,adaptive_prefill,frequency_prefill))
            cell=dict(cell_id=name,trace=trace,native_trace_path=str(native),native_trace_sha256=sha(native),runs=[])
            record['cells'].append(cell)
            oracle=accepted['expected_histories'][name]
            for block in range(4):
                policies=[('enabled',mode,calibration) for mode,calibration in (frequency_cells(block) if frequency_prefill else adaptive_cells(block))] if adaptive_prefill else [('enabled',cap,calibration) for cap,calibration in active_cells(block)] if active else [(cache,None,calibration) for cache,calibration in cells(block)]
                for cache,cap,calibration in policies:
                    if sha(native)!=cell['native_trace_sha256']:
                        raise ValueError('prefix native trace changed during campaign')
                    run=dict(block=block,cache=cache,calibration=calibration)
                    if active:
                        run['prefill_policy' if adaptive_prefill else 'active_prefill_cap']=cap
                        cell['runs'].append(run)
                    label=cap if adaptive_prefill else f'cap-{cap}' if active else cache
                    run=execute(cell,run,cell_dir/f'block-{block}-{label}{"-calibration" if calibration else ""}.log')
                    if not active: cell['runs'].append(run)
                    if _histories(run['parsed'],trace)!=oracle:
                        raise ValueError('prefix measured natural history differs from frozen preflight')
                    if active and not _same(_work_signature(run['parsed'],trace),accepted['expected_work'][name]):
                        raise ValueError('active prefix measured reuse/work differs from frozen preflight')
                    if adaptive_prefill and not _same(_schedule_signature(run['parsed']),accepted['expected_schedules'][name][cap]):
                        raise ValueError('adaptive measured schedule differs from frozen preflight')
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
    stem='engine-prefix-frequency' if frequency_prefill else 'engine-prefix-adaptive' if adaptive_prefill else 'engine-prefix-active' if active else 'engine-prefix'
    (output/(stem+'.json.gz')).write_bytes(compressed)
    write(output/(stem+'.json'),dict(kind=record['kind'],bytes=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest()))
    write(output/(stem+'-summary.json'),record['summary'])
    return profile.engine_replay(output)


def _retain_failure(run,log,error):
    """Keep the actual failed attempt in addition to its original log/receipt."""
    run['error']=str(error)
    if log.exists(): run['stdout']=log.read_text()
    receipt=log.with_suffix('.execution.json')
    if receipt.exists(): run['native_execution']=json.loads(receipt.read_text())
