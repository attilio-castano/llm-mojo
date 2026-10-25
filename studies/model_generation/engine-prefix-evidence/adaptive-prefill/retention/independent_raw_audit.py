#!/usr/bin/env python3
"""Independent standard-library audit of the frozen 192-call adaptive study.

Reads raw stdout first, reconstructs FIFO selection, exact full-prefix ownership,
LRU, histories, work and terminal release, then checks retained parsed records.
No production imports, subprocesses, model, GPU, network or asset loading.
Optional original-file checks bind retained source/binary identities to files.
This checks receipts and logical consistency, not independent tensor arithmetic,
compiler correctness, isolated GPU timing, performance gates or generalization.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

KIND = 'qwen-engine-prefix-adaptive-v1'
STUDY = 'engine-prefix-adaptive-v1'
SALT = 'prefix-study'
MODES = ('current', 'fixed32', 'adaptive')
CELLS = ((4,384,64,16,'first'), (8,384,64,16,'first'),
         (4,40,64,16,'first'), (8,40,64,16,'first'),
         (0,384,64,16,'first'), (8,41,64,16,'first'),
         (8,56,64,16,'first'), (8,36,33,16,'first'),
         (8,40,63,16,'first'), (8,40,65,16,'first'),
         (8,40,64,2,'first'), (4,40,64,16,'even'))
ORDERS = ((0,1,3,2), (1,2,0,3), (2,3,1,0), (3,0,2,1))
ARMS = (('current',False), ('current',True), ('fixed32',False), ('adaptive',False))
STEP = ('step_id','decode_seqs','prefill_seqs','prefill_tokens','total_tokens',
        'attended_positions','admitted','preempted','finished','aborted','waiting',
        'blocks_free','begin_ns','schedule_ns','build_ns','execute_ns',
        'postprocess_ns','end_ns','predicted_ns','budget_limited')
DECISION = ('request_id','prompt_length','maximum_outputs','matched_tokens',
            'free_blocks','reclaimable_blocks','watermark_blocks','available_blocks',
            'required_blocks','selected_decodes','incumbent_request_id',
            'incumbent_remaining_outputs','remaining_prompt_tokens','row_budget',
            'max_sequences','baseline_tokens','chosen_tokens')
CHECKS = ('prefix-adaptive-mixed-kv-logits-greedy-cold-recompute',
          'prefix-adaptive-zero-head-partial',
          'prefix-adaptive-engine-decisions-kv-logits-greedy',
          'prefix-adaptive-engine-terminal-fault-cache-clear')


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def name_of(values):
    reuse, pool, suffix, maximum, order = values
    return f'prefix-1024-reuse-{reuse}-blocks-{pool}-suffix-{suffix}-out-{maximum}-{order}'


def csv(values):
    return ','.join(map(str, values)) if values else '-'


def native_input(trace):
    seed = trace['prefix']['seed_request']
    lines = ['seed {} {} {} {} {} {}'.format(seed['request_id'],seed['arrival_offset_ns'],
             seed['max_new_tokens'],csv(seed['stop_ids']),csv(seed['prompt_ids']),SALT),
             'script '+csv(trace['scripted_tokens'])]
    for r in trace['requests']:
        lines.append('request {} {} {} {} {} -'.format(r['request_id'],r['arrival_offset_ns'],
                     r['max_new_tokens'],csv(r['stop_ids']),csv(r['prompt_ids'])))
    return '\n'.join(lines)+'\n'


def validate_trace(trace, values):
    reuse,pool,suffix,maximum,order = values
    p = trace['prefix']; seed = p['seed_request']; common = seed['prompt_ids'][:-1]
    need((p['tokens'],p['reuse_requests'],p['blocks'],p['suffix_tokens'],p['max_new_tokens'],p['match_order'])
         == (1024,*values), 'cell parameters')
    need(trace['mode']=='offline' and p['salt']==SALT and len(common)==1024
         and seed['request_id']==128 and seed['max_new_tokens']==1, 'seed/trace geometry')
    need(len(trace['requests'])==8, 'request census')
    tails=set(); firsts=set()
    for i,r in enumerate(trace['requests']):
        prompt=r['prompt_ids']; matching=(i%2==0 if order=='even' else i<reuse)
        need(r['request_id']==i and len(prompt)==1024+suffix and r['max_new_tokens']==maximum
             and (prompt[:1024]==common)==matching, 'literal request geometry/reuse')
        tail=tuple(prompt[1024:1056]); need(tail not in tails,'duplicate private suffix'); tails.add(tail)
        if not matching:
            first=tuple(prompt[:32]); need(first!=tuple(common[:32]) and first not in firsts,'nonmatch collision'); firsts.add(first)
    for r in [seed,*trace['requests']]:
        need(r['arrival_offset_ns']==0 and r['abort_offset_ns'] is None and r['output_script'] is None
             and r['stop_ids']==[151643,151645] and all(type(t) is int and 0<=t<151936 for t in r['prompt_ids']),
             'frozen arrivals, stop IDs or literal token domain')


def phase_data():
    return dict(mode='greedy',arrivals={},events=[],steps=[],cache_steps=[],cache_states=[],
                hits=[],drained=None,prefill_decisions=[],selections=[])


def parse_stdout(stdout, trace, mode, namespace):
    """Small independent raw-line decoder; never invokes the repository parser."""
    pool=trace['prefix']['blocks']; phases={p:phase_data() for p in ('seed','measure')}
    times={}; events=[]; meta={}; current=None; order=[]; cleared=None
    for lineno,line in enumerate(stdout.splitlines(),1):
        f=line.split()
        if not f: continue
        tag=f[0]
        need(current not in times or tag=='phase',f'line after phase close: {lineno}')
        if tag in ('device','mode','config','admission','study','work_capacity','cache','namespace','salt','prefill_policy'):
            need(current is None and tag not in meta,'late/duplicate metadata'); meta[tag]=f[1:]; continue
        if tag=='phase':
            need(len(f)==2 and (current is None or current in times),'unclosed phase')
            current=f[1]; need(current not in order,'duplicate phase'); order.append(current); continue
        if tag=='phase_time':
            need(len(f)==4 and f[1]==current,'phase boundary ownership')
            begin,end=map(int,f[2:]); need(0<=begin<=end and (begin<end or current=='handoff'),'phase times')
            times[current]=dict(begin_ns=begin,end_ns=end); continue
        if tag=='cache_event':
            need(len(f)==7 and current in ('seed','handoff','measure','clear'),'cache event shape')
            eid,sid=int(f[1]),int(f[2]); kind=f[3]; ns,salt=f[4:6]
            tokens=[] if f[6]=='-' else list(map(int,f[6].split(',')))
            need(eid==len(events) and kind in ('stored','removed','cleared'),'cache event IDs/kind')
            if kind=='cleared':
                need((ns,salt,tokens)==('-','-',[]),'clear identity'); ns=salt=''
            else:
                need(ns==namespace and salt==SALT and len(tokens)>0 and len(tokens)%32==0
                     and all(0<=t<151936 for t in tokens),'full exact cache identity')
            need(sid==(phases[current]['steps'][-1]['step_id'] if current in phases else -1),'cache event step')
            events.append(dict(event_id=eid,step_id=sid,phase=current,kind=kind,namespace=ns,salt=salt,tokens=tokens)); continue
        if tag=='cache_cleared':
            need(current=='clear' and len(f)==7 and cleared is None,'final clear receipt')
            cleared=list(map(int,f[1:])); continue
        need(current in phases,'unknown phase record: '+line[:80]); d=phases[current]
        if tag=='step' and len(f)==21:
            d['steps'].append(dict(zip(STEP,map(int,f[1:])))); continue
        if tag=='arrival' and len(f)==4:
            rid,scheduled,actual=map(int,f[1:]); need(str(rid) not in d['arrivals'],'duplicate arrival')
            d['arrivals'][str(rid)]=dict(scheduled_ns=scheduled,actual_ns=actual); continue
        need(bool(d['steps']),'record before step'); sid=d['steps'][-1]['step_id']
        if tag in ('prefix_step','cache_hit','cache_state','prefill_decision','batch_selection'):
            need(int(f[1])==sid,'explicit record has the wrong current step owner')
        if tag in ('token','finish') and len(f)==7:
            d['events'].append(dict(kind=tag,request_id=int(f[1]),token_id=int(f[2]) if tag=='token' else None,
                reason=f[2] if tag=='finish' else None,prompt_tokens=int(f[3]),generated_tokens=int(f[4]),
                arrival_ns=int(f[5]),emitted_ns=int(f[6]),step_id=sid))
        elif tag=='prefix_step' and len(f)==6:
            d['cache_steps'].append(dict(zip(('step_id','cache_hit_tokens','lookup_ns','publish_ns','admission_ns'),map(int,f[1:]))))
        elif tag=='cache_hit' and len(f)==4:
            d['hits'].append(dict(zip(('step_id','request_id','cached_tokens'),map(int,f[1:]))))
        elif tag=='cache_state' and len(f)==6:
            d['cache_states'].append(dict(zip(('step_id','free_blocks','cached_blocks','references','private_blocks'),map(int,f[1:]))))
        elif tag=='prefill_decision' and len(f)==22:
            need(f[19] in ('0','1') and f[20] in ('0','1'),'decision boolean')
            d['prefill_decisions'].append(dict(step_id=int(f[1]),**dict(zip(DECISION,map(int,f[2:19]))),
                eligible=f[19]=='1',applied=f[20]=='1',reason=f[21]))
        elif tag=='batch_selection' and len(f)==7:
            d['selections'].append(dict(zip(('step_id','order','request_id','past','query_tokens','selected_head'),map(int,f[1:]))))
        elif tag=='prefix_drained' and len(f)==9:
            need(d['drained'] is None,'duplicate drain')
            d['drained']=dict(zip(('requests','steps','free_blocks','cached_blocks','references','private_blocks','pending','elapsed_ns'),map(int,f[1:])))
        else:
            raise ValueError('unknown raw record: '+line[:100])
    need(order==['seed','handoff','measure','clear'] and set(times)==set(order),'phase census/order')
    need(cleared==[pool,0,0,0,0,0],'complete final clear')
    need(meta==dict(device=['Apple','M4','Pro/metal'],mode=['greedy'],config=['chunked',str(pool),'256','8'],
        admission=['reserved'],study=[STUDY],work_capacity=['256','8'],cache=['enabled'],namespace=[namespace],
        salt=[SALT],prefill_policy=[mode]),'raw metadata identity')
    need(all(times[a]['end_ns']<=times[b]['begin_ns'] for a,b in zip(order,order[1:])),'phase overlap')
    return dict(device='Apple M4 Pro/metal',mode='greedy',cache='enabled',blocks=pool,namespace=namespace,
                phases=phases,phase_times=times,cache_events=events,prefill_policy=mode)


def audit_run(run, trace, namespace):
    parsed=parse_stdout(run['stdout'],trace,run['prefill_policy'],namespace)
    retained=run['parsed']; need(set(retained)==set(parsed)|{'policy_replay'},'parsed fields')
    need(all(retained[k]==v for k,v in parsed.items()),'raw stdout differs from retained parsed records')
    pool=parsed['blocks']; index={}; private={}; clock=0; next_sid=0
    stats=Counter(); all_hist={}; all_work={}; signature={}; per_phase={}
    def free(): return pool-len(index)-sum(private.values())
    def refs(): return sum(private.values())+sum(len(v['owners']) for v in index.values())
    def key(history,n): return (namespace,SALT,tuple(history[:n]))
    for phase in ('seed','measure'):
        data=parsed['phases'][phase]; policy='current' if phase=='seed' else run['prefill_policy']
        reqs=[trace['prefix']['seed_request']] if phase=='seed' else trace['requests']
        req={r['request_id']:r for r in reqs}; ids=list(req); generated={i:[] for i in ids}
        completed={}; done=set(); hit_by_id={}; hist=[]; sig=[]; phase_counts=Counter()
        need(set(data['arrivals'])=={str(i) for i in ids},'arrival census')
        for arrival in data['arrivals'].values(): need(arrival['scheduled_ns']==0 and arrival['actual_ns']>=0,'arrival times')
        need([s['step_id'] for s in data['steps']]==list(range(next_sid,next_sid+len(data['steps']))),'ordered step IDs')
        for field in ('cache_steps','cache_states'):
            need([s['step_id'] for s in data[field]]==[s['step_id'] for s in data['steps']],field+' census')
        previous_end=0
        for step in data['steps']:
            sid=step['step_id']; need(all(type(v) is int and v>=0 for v in step.values()),'step domain')
            need(previous_end<=step['begin_ns']<=step['end_ns'] and sum(step[k] for k in
                ('schedule_ns','build_ns','execute_ns','postprocess_ns'))<=step['end_ns']-step['begin_ns'],'step timings')
            previous_end=step['end_ns']
            need(not any(step[k] for k in ('aborted','preempted','predicted_ns','budget_limited')),'unfrozen scheduler behavior')
            dec=[i for i in ids if i in completed and generated[i] and i not in done][:8]
            plan=[(i,1) for i in dec]; budget=256-len(dec); expected_decisions=[]; expected_hits=[]; changes=[]; admitted=0
            held=next((i for i in ids if i in completed and not generated[i] and i not in done),None)
            if len(dec)<8 and held is not None:
                n=min(budget,len(req[held]['prompt_ids'])-completed[held])
                if policy=='fixed32' and dec: n=min(n,32)
                plan.append((held,n))
            elif len(dec)<8:
                waiter=next((i for i in ids if i not in completed),None)
                if waiter is not None:
                    r=req[waiter]; prompt=r['prompt_ids']; matched=[]
                    for end in range(32,len(prompt),32):
                        candidate=key(prompt,end)
                        if candidate not in index: break
                        matched.append(candidate)
                    h=32*len(matched); k=(len(prompt)+r['max_new_tokens']+30)//32-len(matched)
                    reclaim=sum(not v['owners'] and candidate not in matched for candidate,v in index.items())
                    a=free()+reclaim; rem=len(prompt)-h; incumbent=dec[0] if len(dec)==1 else -1
                    remaining=req[incumbent]['max_new_tokens']-len(generated[incumbent]) if incumbent>=0 else -1
                    baseline=min(budget,rem)
                    failures=((k>a,'admission_blocked'),(k<=0,'no_private_blocks'),(len(dec)!=1,'decode_count'),
                        (remaining<2,'incumbent_terminal'),(rem<33 or rem>64,'tail_length'),
                        (len(dec)+1>=8,'last_sequence_slot'),(a-k>=k,'capacity_headroom'))
                    reason=next((text for failed,text in failures if failed),None); eligible=reason is None
                    n=0 if k>a else min(baseline,32) if policy=='fixed32' and dec or policy=='adaptive' and eligible else baseline
                    applied=policy=='adaptive' and eligible and n<baseline
                    if eligible: reason='adaptive_disabled' if policy!='adaptive' else 'adaptive_short_tail' if applied else 'baseline_at_most_32'
                    decision=dict(step_id=sid,**dict(zip(DECISION,[waiter,len(prompt),r['max_new_tokens'],h,free(),reclaim,0,a,k,
                        len(dec),incumbent,remaining,rem,budget,8,baseline,n])),eligible=eligible,applied=applied,reason=reason)
                    expected_decisions.append(decision); stats['decisions']+=1; stats['eligible']+=eligible; stats['applied']+=applied
                    if n:
                        while free()<k:
                            victims=[candidate for candidate,v in index.items() if not v['owners'] and candidate not in matched]
                            need(victims,'capacity cannot be reclaimed')
                            victim=min(victims,key=lambda candidate:index[candidate]['stamp'])
                            del index[victim]; changes.append(('removed',victim))
                        private[waiter]=k; completed[waiter]=h; clock+=1
                        for candidate in matched: index[candidate]['owners'].add(waiter); index[candidate]['stamp']=clock
                        plan.append((waiter,n)); admitted=1; hit_by_id[waiter]=h
                        if h: expected_hits.append(dict(step_id=sid,request_id=waiter,cached_tokens=h))
            need([d for d in data['prefill_decisions'] if d['step_id']==sid]==expected_decisions,'admission inputs/predicate/reason')
            need([d for d in data['hits'] if d['step_id']==sid]==expected_hits,'full resident prefix hit provenance')
            stats['hits']+=len(expected_hits)
            selections=[]; offset=0; positions=0
            for order,(i,n) in enumerate(plan):
                past=completed[i]; known=len(req[i]['prompt_ids'])+len(generated[i])
                need(0<n<=256 and past+n<=known,'queries exceed known request history')
                head=offset+n-1 if past+n==known else -1
                selections.append(dict(step_id=sid,order=order,request_id=i,past=past,query_tokens=n,selected_head=head))
                positions+=n*past+n*(n+1)//2; offset+=n
            need([s for s in data['selections'] if s['step_id']==sid]==selections,'ordered request/past/query/head rows')
            need((step['decode_seqs'],step['prefill_seqs'],step['prefill_tokens'],step['total_tokens'],step['attended_positions'],step['admitted'])
                 ==(len(dec),int(len(plan)>len(dec)),sum(n for _,n in plan[len(dec):]),offset,positions,admitted),'selected step work')
            step_events=[e for e in data['events'] if e['step_id']==sid]
            expected_head_ids=[s['request_id'] for s in selections if s['selected_head']>=0]
            need([e['request_id'] for e in step_events if e['kind']=='token']==expected_head_ids,'token head ownership/order')
            event_cursor=0; finished=0
            for selection in selections:
                i=selection['request_id']; completed[i]+=selection['query_tokens']; history=req[i]['prompt_ids']+generated[i]
                for end in range(32,completed[i]//32*32+1,32):
                    candidate=key(history,end)
                    if candidate in index: continue
                    need(private[i]>0 and len(candidate[2])==end,'publication lacks a completed private full block')
                    private[i]-=1; clock+=1; index[candidate]=dict(owners={i},stamp=clock); changes.append(('stored',candidate))
                if selection['selected_head']<0: continue
                event=step_events[event_cursor]; event_cursor+=1
                need(event['kind']=='token' and event['request_id']==i and 0<=event['token_id']<151936
                     and event['generated_tokens']==len(generated[i])+1,'delivered token ordinal/domain')
                need(event['prompt_tokens']==len(req[i]['prompt_ids']) and event['arrival_ns']==0
                     and max(data['arrivals'][str(i)]['actual_ns'],step['begin_ns'])<=event['emitted_ns']
                     ==step['end_ns']-step['postprocess_ns'],'token availability boundary')
                generated[i].append(event['token_id'])
                reason='stop' if event['token_id'] in req[i]['stop_ids'] else 'length' if len(generated[i])==req[i]['max_new_tokens'] else None
                if reason:
                    finish=step_events[event_cursor]; event_cursor+=1
                    need(finish==dict(event,kind='finish',token_id=None,reason=reason),'same-step terminal boundary/reason')
                    done.add(i); finished+=1; del private[i]
                    for v in index.values(): v['owners'].discard(i)
            need(event_cursor==len(step_events) and step['finished']==finished,'orphan token/finish events')
            actual_changes=[(e['kind'],(e['namespace'],e['salt'],tuple(e['tokens']))) for e in parsed['cache_events'] if e['phase']==phase and e['step_id']==sid]
            need(actual_changes==changes,'exact store/remove, completed-history provenance or LRU order')
            stats.update(kind for kind,_ in changes)
            state=next(s for s in data['cache_states'] if s['step_id']==sid)
            need(state==dict(step_id=sid,free_blocks=free(),cached_blocks=len(index),references=refs(),private_blocks=sum(private.values())),
                 'physical partition/references')
            need(free()>=0 and free()+len(index)+sum(private.values())==pool and step['blocks_free']==free()
                 and step['waiting']==sum(i not in completed for i in ids),'bounded pool/FIFO waiting census')
            counters=next(c for c in data['cache_steps'] if c['step_id']==sid)
            need(counters['cache_hit_tokens']==sum(h['cached_tokens'] for h in expected_hits)
                 and all(type(v) is int and v>=0 for v in counters.values()),'cache timing/hit counters')
            stats['snapshots']+=1; stats['selections']+=len(selections)
            kind='mixed' if dec and len(plan)>len(dec) else 'prefill_only' if len(plan)>len(dec) else 'decode_only'
            phase_counts[kind]+=1; phase_counts['decode_rows']+=len(dec)
            sig.append(dict(work={k:v for k,v in step.items() if not k.endswith('_ns') and k not in ('predicted_ns','budget_limited')},
                selections=selections,hits=expected_hits,cache=[dict(kind=k,tokens=list(v[2])) for k,v in changes]))
        next_sid+=len(data['steps']); need(done==set(ids) and not private and refs()==0,'all request ownership drained')
        for i in ids:
            terminal=[e for e in data['events'] if e['request_id']==i and e['kind']=='finish']
            need(len(terminal)==1,'exactly one terminal per request')
            hist.append(dict(request_id=i,tokens=generated[i],reason=terminal[0]['reason']))
        rows=positions=0
        for i in ids:
            t=len(req[i]['prompt_ids'])+len(generated[i])-1; h=hit_by_id[i]
            rows+=t-h; positions+=t*(t+1)//2-h*(h+1)//2
        need(rows==sum(s['total_tokens'] for s in data['steps']) and positions==sum(s['attended_positions'] for s in data['steps']),
             'independent necessary rows/causal-position formula')
        drain=data['drained']; need(drain is not None and drain['requests']==len(ids) and drain['steps']==len(data['steps'])
            and drain['free_blocks']==free() and drain['cached_blocks']==len(index)
            and drain['references']==drain['private_blocks']==drain['pending']==0
            and previous_end<=drain['elapsed_ns']<=parsed['phase_times'][phase]['end_ns']-parsed['phase_times'][phase]['begin_ns'],
            'complete physical drain and bounded elapsed interval')
        all_hist[phase]=hist; all_work[phase]=dict(rows=rows,attended_positions=positions,
            cached_tokens_by_request={str(i):hit_by_id[i] for i in ids}); signature[phase]=sig
        per_phase[phase]=dict(phase_counts,steps=len(data['steps']),rows=rows,attended_positions=positions)
    need(not any(e['phase']=='handoff' for e in parsed['cache_events']),'cache changed at enabled handoff')
    clear=[(e['kind'],(e['namespace'],e['salt'],tuple(e['tokens']))) for e in parsed['cache_events'] if e['phase']=='clear']
    need(clear==[('removed',candidate) for candidate in index]+[('cleared',('','',()))],'complete ordered final clear')
    stats['removed']+=len(index); stats['cleared']+=1
    index.clear(); need(free()==pool and not private and refs()==0,'final clear releases every modeled physical block')
    proof=dict(decisions=stats['decisions'],eligible=stats['eligible'],applied=stats['applied'],logical_schedule_sha256=digest(canonical(signature).encode()))
    need(proof==retained['policy_replay'],'retained policy replay proof differs')
    return dict(histories=all_hist,work=all_work,signature=signature,signature_sha256=proof['logical_schedule_sha256'],
                counts=dict(stats),phases=per_phase,parsed=parsed)


def execution(receipt, command, timeout, minimum=1):
    need(receipt['command']==list(map(str,command)) and type(receipt['exit_code']) is int and receipt['exit_code']==0
         and type(receipt['timeout_seconds']) is int and receipt['timeout_seconds']==timeout
         and type(receipt['wall_elapsed_ns']) is int and minimum<=receipt['wall_elapsed_ns']<=timeout*1_000_000_000,
         'actual native/compiler command, exit or elapsed bound')


def command(build, cell, mode):
    return [build['command'][-1],build['assets']['prepared'],cell['native_trace_path'],'chunked',
            str(cell['trace']['prefix']['blocks']),'256','8','10','greedy','reserved',STUDY,'enabled',mode]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive',type=Path); ap.add_argument('--manifest',type=Path)
    ap.add_argument('--source-root',type=Path); ap.add_argument('--native-binary',type=Path)
    ap.add_argument('--checkpoint-binary',type=Path)
    args=ap.parse_args(); packed=args.archive.read_bytes(); unpacked=gzip.decompress(packed)
    manifest=decode((args.manifest or args.archive.with_suffix('')).read_bytes())
    need(manifest['kind']==KIND and type(manifest['bytes']) is int and manifest['bytes']==len(packed)
         and manifest['sha256']==digest(packed) and manifest['uncompressed_sha256']==digest(unpacked),'archive hash/byte manifest')
    if 'uncompressed_bytes' in manifest:
        need(type(manifest['uncompressed_bytes']) is int and manifest['uncompressed_bytes']==len(unpacked),'uncompressed bytes')
    r=decode(unpacked); need(r['kind']==KIND,'archive kind')
    build=r['build']; q=r['qualification']; summary=r['summary']; campaign=r['trace']
    need(build['source']['repository']['dirty'] is False,'source must be clean')
    commit=build['source']['repository']['commit']; need(len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'commit identity')
    need(Path(build['assets']['prepared']).is_absolute(),'prepared identity path')
    for value in build['source']['sources'].values():
        need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'retained source digest format')
    namespace='qwen-reference27-projection8/'+build['assets']['prepared']
    need(build['declaration']==r['declaration'] and r['declaration']['study']==STUDY
         and r['declaration']['measured_runs']==summary['measured_runs']==192
         and r['declaration']['untimed_preflights']==summary['untimed_preflights']==36,'declaration/census')
    need(summary['promotion'] is False and summary['asynchronous'] is False and summary['target'] is None and summary['goodput'] is None,'claim boundary')
    need(digest(r['trace_document'].encode())==r['trace_sha256'] and decode(r['trace_document'])==campaign,'campaign document hash')
    need(campaign['source_text_sha256']==digest(campaign['source_text'].encode()),'owned source text hash')
    need(digest(r['qualification_document'].encode())==r['qualification_sha256'] and decode(r['qualification_document'])==q,'qualification document hash')
    need(q['build']==build and decode(q['campaign_document'])==campaign and q['campaign']==campaign
         and digest(q['campaign_document'].encode())==q['campaign_sha256']==r['trace_sha256'],'qualification/campaign binding')
    need(campaign['tokenizer']['tables_sha256']==build['assets']['tables_sha256']
         and campaign['tokenizer']['generator_source_sha256']==build['source']['sources']['src/llm_mojo/benchmarks/engine_prefix.py'],
         'tokenizer/generator source binding')
    cp=q['checkpoint_receipt']; cb=cp['build']
    need(all(cb[k]==build[k] for k in ('source','assets','environment')),'numerical source/assets/runtime binding')
    need(cb['command'][1:7]==['build','-I','src','-I','tests','tests/engine_metal_driver.mojo']
         and cb['command'][0]==build['command'][0] and cb['command'][-2]=='-o','checkpoint compilation command')
    execution(cp['build_execution'],cb['command'],600)
    execution(cp['execution'],[cb['command'][-1],build['assets']['prepared'],'prefix-adaptive-qualification'],180)
    need(digest(cp['stdout'].encode())==cp['stdout_sha256'],'checkpoint stdout hash')
    checks=[]; identity=[]
    for line in cp['stdout'].splitlines():
        f=line.split()
        if not f: continue
        if f[0]=='check':
            need(len(f)==8 and f[2]=='checked' and f[4]=='mismatches' and f[6]=='unexpected-nonfinite'
                 and int(f[3])>0 and int(f[5])==int(f[7])==0,'reported numerical group failed')
            checks.append(dict(name=f[1],checked=int(f[3]),mismatches=0,unexpected_nonfinite=0))
        else: identity.append(f)
    need(tuple(c['name'] for c in checks)==CHECKS and identity==[['device','Apple','M4','Pro/metal'],
         ['qualification','engine-prefix-adaptive-reference27-v1'],['configuration','27','projection','8','dtype','BF16/FP32','block-size','32']],
         'reported numerical device/backend/coverage identity')
    need(len(r['cells'])==len(summary['cells'])==len(campaign['traces'])==len(q['preflight_cells'])==12,'twelve complete cells')
    counts={'preflight':Counter(),'timed':Counter()}; cell_results=[]; execution_count=2
    for values,trace,cell,scell,pre in zip(CELLS,campaign['traces'],r['cells'],summary['cells'],q['preflight_cells']):
        name=name_of(values); validate_trace(trace,values)
        for c in (cell,pre):
            need(c['cell_id']==name and c['trace']==trace and c['native_trace_sha256']==digest(native_input(trace).encode())
                 and Path(c['native_trace_path']).is_absolute(),'native input/cell identity')
        need(scell['cell_id']==name and len(cell['runs'])==len(scell['runs'])==16,'cell summary census')
        need([(x['cache'],x['prefill_policy']) for x in pre['runs']]==[('enabled',m) for m in MODES],'three unique preflight policies')
        expected_order=[(block,*ARMS[i]) for block,order in enumerate(ORDERS) for i in order]
        need([(x['block'],x['prefill_policy'],x['calibration']) for x in cell['runs']]==expected_order,'balanced actual 192-call order')
        expected_hist=pre['expected_histories']; expected_work=pre['expected_work']; schedules={}; measured=[]
        for stage,runs in (('preflight',pre['runs']),('timed',cell['runs'])):
            for number,run in enumerate(runs):
                need(run['cache']=='enabled' and run['prefill_policy'] in MODES,'run policy')
                if stage=='timed':
                    need(type(run['block']) is int and type(run['calibration']) is bool,'typed block/calibration')
                audited=audit_run(run,trace,namespace); p=audited['parsed']
                elapsed=p['phase_times']['clear']['end_ns']-p['phase_times']['seed']['begin_ns']
                execution(run['native_execution'],command(build,pre if stage=='preflight' else cell,run['prefill_policy']),180,max(1,elapsed)); execution_count+=1
                need(audited['histories']==expected_hist==scell['frozen_histories'],'raw histories differ from frozen seed/measure histories')
                need(audited['work']==expected_work==scell['frozen_work'],'raw perrequest hits/necessary work differs across policies')
                counts[stage].update(audited['counts']); counts[stage]['runs']+=1; counts[stage]['histories']+=9
                if stage=='preflight':
                    schedules[run['prefill_policy']]=audited['signature']
                    need(audited['signature']==pre['expected_schedules'][run['prefill_policy']],'frozen preflight selection signature')
                else:
                    need(audited['signature']==schedules[run['prefill_policy']],'measured ordered schedule differs from preflight')
                    for label in ('conditions_before','conditions_after'):
                        condition=run[label]
                        thermal='\n'.join(condition['thermal'])
                        need(condition['battery']['power_source']=='AC Power' and condition['power_mode_raw']=='0'
                             and all(line in thermal for line in ('No thermal warning level has been recorded','No performance warning level has been recorded')),
                             'recorded AC/nominal power/thermal conditions')
                    sm=scell['runs'][number]
                    need((sm['block'],sm['prefill_policy'],sm['calibration'])==(run['block'],run['prefill_policy'],run['calibration'])
                         and sm['total_tokens']==audited['work']['measure']['rows']
                         and sm['attended_positions']==audited['work']['measure']['attended_positions']
                         and sm['steps']==audited['phases']['measure']['steps'],'summary identity/work/step binding')
                    measured.append(dict(block=run['block'],policy=run['prefill_policy'],calibration=run['calibration'],
                        logical_schedule_sha256=audited['signature_sha256'],**audited['phases']['measure']))
        known=values[2:]==(64,16,'first') and values[1] in (40,384)
        prediction=None
        if known:
            reference='fixed32' if values[1]==40 else 'current'
            expected=({4:115,8:69} if values[1]==40 else {0:55,4:39,8:23})[values[0]]
            need(schedules['adaptive']==schedules[reference] and len(schedules['adaptive']['measure'])==expected,
                 'known 40/static32 or 384/current full logical schedule prediction')
            prediction=dict(reference=reference,steps=expected,full_logical_equality=True)
        cell_results.append(dict(cell_id=name,known_prediction=prediction,runs=measured))
    need(counts['timed']['runs']==192 and counts['preflight']['runs']==36 and execution_count==230,'actual call/exit census')
    matched_sources=None
    if args.source_root:
        for filename,expected in build['source']['sources'].items():
            need(digest((args.source_root/filename).read_bytes())==expected,'source file changed: '+filename)
        matched_sources=len(build['source']['sources'])
    binary_checks={}
    for label,path,info in (('engine',args.native_binary,build['binaries']['engine']),
                            ('checkpoint',args.checkpoint_binary,cb['binaries']['checkpoint-driver'])):
        if path:
            raw=path.read_bytes(); need(len(raw)==info['bytes'] and digest(raw)==info['sha256'],'binary changed: '+label)
            binary_checks[label]='matched'
        else: binary_checks[label]='not requested; retained identity only'
    print(json.dumps(dict(kind='engine-prefix-adaptive-independent-raw-audit-v1',status='passed',
        script_sha256=digest(Path(__file__).read_bytes()),source_commit=commit,
        archive=dict(path=str(args.archive),bytes=len(packed),sha256=digest(packed),uncompressed_sha256=digest(unpacked)),
        actual_receipts=dict(zero_exit_receipts=execution_count,checkpoint_compilation=1,checkpoint_execution=1,preflights=36,timed=192,
            engine_compilation_exit='not separately embedded; retained build identity checked'),
        reported_numerical_groups=checks,raw_counts={k:dict(v) for k,v in counts.items()},cells=cell_results,
        source_files_matched=matched_sources,binary_files=binary_checks,
        formulas=dict(rows='T-H',causal_positions='T*(T+1)/2-H*(H+1)/2',T='prompt length + delivered outputs - 1'),
        ownership='Exact namespace/salt/full-token-prefix keys; private lifetime blocks, per-entry active references, completed publication, protected idle LRU, full drain and clear.',
        caveats=['Logical consistency and retained receipts do not independently prove tensor arithmetic or actual device execution.',
                 'Performance ratios/gates are outside this raw audit; no timing, capacity, online or generalization claim is added.',
                 'Replay is bounded to this frozen positive-output, zero-arrival, zero-watermark synchronous matrix.',
                 'Original source/binary file correspondence requires the optional file arguments.'],
        native_execution=False,network_access=False,production_imports=False),indent=2))


if __name__=='__main__':
    main()
