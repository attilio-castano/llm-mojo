"""Prefix replay checks exact identities, charged seed work and real phase bounds."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from llm_mojo.benchmarks import engine_prefix as prefix
from llm_mojo.benchmarks import model_contract as contract
from llm_mojo.benchmarks import model_profile as profile


def trace_fixture(length=256,reuse=4,blocks=384):
    common=list(range(length+1))
    requests=[]
    for i in range(8):
        base=common[:length] if i<reuse else [5000+i,*common[1:length]]
        requests.append(prefix._request(i,base+[6000+i]*64))
    return dict(kind='engine-token-trace-v1',schema_version=1,mode='offline',scripted_tokens=[11],
        prefix=dict(tokens=length,reuse_requests=reuse,blocks=blocks,salt=prefix.SALT,
                    seed_request=prefix._request(128,common,1)),requests=requests)


def native_fixture(trace=None,cache='enabled',active_prefill_cap=None):
    trace=trace or trace_fixture()
    blocks=trace['prefix']['blocks']
    namespace='qwen-reference27-projection8//prepared'
    lines=['device Apple M4 Pro/metal','mode greedy',f'config chunked {blocks} 256 8',
           'admission reserved','study engine-prefix-v1','work_capacity 256 8',f'cache {cache}',
           f'namespace {namespace}',f'salt {prefix.SALT}']
    if active_prefill_cap is not None:
        lines[4]='study engine-prefix-active-v1'
        lines.append(f'active_prefill_cap {active_prefill_cap}')
    indexed={}
    event_id=0
    step_id=0
    absolute=1000
    def event(kind,tokens):
        nonlocal event_id
        row=f'cache_event {event_id} {step_id if phase in ("seed","measure") else -1} {kind} '
        if kind=='cleared':
            row+='- - -'
        else:
            row+=f'{namespace} {prefix.SALT} '+','.join(map(str,tokens))
        event_id+=1
        return row
    def clear():
        result=[]
        for key in list(indexed):
            result.append(event('removed',key))
            del indexed[key]
        result.append(event('cleared',[]))
        return result
    def run_phase(requests,enabled):
        nonlocal step_id
        local=0
        steps=0
        for request in requests:
            lines.append(f'arrival {request["request_id"]} 0 0')
        for request in requests:
            ids=request['prompt_ids']
            hit=0
            if enabled:
                for size in range(32,((len(ids)-1)//32)*32+1,32):
                    if tuple(ids[:size]) not in indexed:
                        break
                    hit=size
            past=hit
            while past<len(ids):
                rows=min(256,len(ids)-past)
                end=past+rows
                last=end==len(ids)
                changes=[]
                if enabled:
                    wanted=[tuple(ids[:n]) for n in range(32,(end//32)*32+1,32)]
                    for key in wanted:
                        if key in indexed:
                            continue
                        while len(indexed)>=blocks:
                            victim=next(k for k in indexed if k not in wanted)
                            changes.append(event('removed',victim)); del indexed[victim]
                        indexed[key]=True
                        changes.append(event('stored',key))
                begin=local+1
                local=begin+10
                values=dict.fromkeys(contract.ENGINE_STEP_FIELDS,0)
                values.update(step_id=step_id,prefill_seqs=1,prefill_tokens=rows,total_tokens=rows,
                    attended_positions=rows*(past+1)+rows*(rows-1)//2,admitted=int(past==hit),
                    finished=int(last),blocks_free=blocks-len(indexed),begin_ns=begin,schedule_ns=1,
                    build_ns=1,execute_ns=4,postprocess_ns=1,end_ns=local)
                lines.append('step '+' '.join(str(values[k]) for k in contract.ENGINE_STEP_FIELDS))
                if last:
                    i=request['request_id']
                    lines.append(f'token {i} 151643 {len(ids)} 1 0 {local-1}')
                    lines.append(f'finish {i} stop {len(ids)} 1 0 {local-1}')
                lines.append(f'prefix_step {step_id} {hit if past==hit else 0} 1 1 0')
                if hit and past==hit:
                    lines.append(f'cache_hit {step_id} {request["request_id"]} {hit}')
                lines.extend(changes)
                lines.append(f'cache_state {step_id} {blocks-len(indexed)} {len(indexed)} 0 0')
                step_id+=1; steps+=1; past=end
        lines.append(f'prefix_drained {len(requests)} {steps} {blocks-len(indexed)} {len(indexed)} 0 0 0 {local+1}')
        return local+3
    for phase in prefix.PHASES:
        lines.append('phase '+phase)
        if phase=='seed':
            duration=run_phase([trace['prefix']['seed_request']],True)
        elif phase=='handoff':
            if cache=='disabled': lines.extend(clear())
            duration=5
        elif phase=='measure':
            duration=run_phase(trace['requests'],cache=='enabled')
        else:
            lines.extend(clear())
            lines.append(f'cache_cleared {blocks} 0 0 0 0 0')
            duration=5
        lines.append(f'phase_time {phase} {absolute} {absolute+duration}')
        absolute+=duration+2
    return '\n'.join(lines)


def adaptive_trace(values):
    reuse,blocks,suffix,maximum,order=values
    trace=trace_fixture(1024,reuse,blocks)
    trace['prefix'].update(suffix_tokens=suffix,max_new_tokens=maximum,match_order=order)
    for i,r in enumerate(trace['requests']):
        matching=i%2==0 if order=='even' else i<reuse
        r['prompt_ids']=(list(range(1024)) if matching else [5000+i,*range(1,1024)])+[6000+i]*suffix
        r['max_new_tokens']=maximum
    return trace


def adaptive_native_fixture(trace,mode='adaptive'):
    """Bounded synthetic receipts with explicit blocks and deterministic non-stop heads.

    This tests receipt replay, not model arithmetic. Physical numerical qualification
    remains a separate native gate. The producer never calls the replay oracle.
    """
    pool=trace['prefix']['blocks']; namespace='qwen-reference27-projection8//prepared'
    lines=['device Apple M4 Pro/metal','mode greedy',f'config chunked {pool} 256 8',
           'admission reserved','study engine-prefix-adaptive-v1','work_capacity 256 8','cache enabled',
           f'namespace {namespace}',f'salt {prefix.SALT}',f'prefill_policy {mode}']
    index={}; private={}; tick=0; eid=0; sid=0; absolute=1000
    def free(): return pool-len(index)-sum(private.values())
    def event(kind,key=()):
        nonlocal eid
        tail='- - -' if kind=='cleared' else namespace+' '+prefix.SALT+' '+','.join(map(str,key))
        row=f'cache_event {eid} {sid if phase in ("seed","measure") else -1} {kind} '+tail
        eid+=1
        return row
    for phase in prefix.PHASES:
        lines.append('phase '+phase); local=0; countsteps=0
        if phase in ('seed','measure'):
            current='current' if phase=='seed' else mode
            requests=[trace['prefix']['seed_request']] if phase=='seed' else trace['requests']
            req={r['request_id']:r for r in requests}; past={}; generated={i:0 for i in req}; done=set()
            for i in req: lines.append(f'arrival {i} 0 0')
            while len(done)<len(req):
                changes=[]; hits=[]; decisions=[]; selected=[]
                dec=[i for i in req if i in past and generated[i] and i not in done][:8]
                selected.extend((i,1) for i in dec); budget=256-len(dec)
                held=next((i for i in req if i in past and not generated[i] and i not in done),None)
                if len(dec)<8 and held is not None:
                    n=min(budget,len(req[held]['prompt_ids'])-past[held],32 if current=='fixed32' and dec else 256)
                    selected.append((held,n))
                elif len(dec)<8:
                    waiting=next((i for i in req if i not in past),None)
                    if waiting is not None:
                        r=req[waiting]; prompt=r['prompt_ids']; match=[]
                        for n in range(32,((len(prompt)-1)//32)*32+1,32):
                            key=tuple(prompt[:n])
                            if key not in index: break
                            match.append(key)
                        h=32*len(match); k=(len(prompt)+r['max_new_tokens']+30)//32-len(match)
                        reclaim=sum(not value[0] and key not in match for key,value in index.items())
                        a=free()+reclaim; remaining=len(prompt)-h; incumbent=dec[0] if len(dec)==1 else -1
                        rest=req[incumbent]['max_new_tokens']-generated[incumbent] if incumbent>=0 else -1
                        baseline=min(budget,remaining)
                        reason=next((name for failed,name in ((k>a,'admission_blocked'),(k<=0,'no_private_blocks'),
                            (len(dec)!=1,'decode_count'),(rest<2,'incumbent_terminal'),
                            (not 33<=remaining<=64,'tail_length'),(len(dec)+1>=8,'last_sequence_slot'),
                            (a-k>=k,'capacity_headroom')) if failed),None)
                        eligible=reason is None
                        n=0 if k>a else min(baseline,32) if (current=='fixed32' and dec) or (current=='adaptive' and eligible) else baseline
                        applied=current=='adaptive' and eligible and n<baseline
                        reason=reason or ('adaptive_disabled' if current!='adaptive' else 'adaptive_short_tail' if applied else 'baseline_at_most_32')
                        fields=[waiting,len(prompt),r['max_new_tokens'],h,free(),reclaim,0,a,k,len(dec),incumbent,rest,remaining,budget,8,baseline,n]
                        decisions.append(f'prefill_decision {sid} '+' '.join(map(str,fields))+f' {int(eligible)} {int(applied)} {reason}')
                        if n:
                            while free()<k:
                                victim=min((key for key,v in index.items() if not v[0] and key not in match),key=lambda key:index[key][1])
                                del index[victim]; changes.append(event('removed',victim))
                            private[waiting]=k; past[waiting]=h; tick+=1
                            for key in match: index[key][0].add(waiting); index[key][1]=tick
                            if h: hits.append((waiting,h))
                            selected.append((waiting,n))
                offset=0; positions=0; selections=[]; events=[]; finished=0
                for order,(i,n) in enumerate(selected):
                    before=past[i]; head=offset+n-1 if before+n==len(req[i]['prompt_ids'])+generated[i] else -1
                    selections.append(f'batch_selection {sid} {order} {i} {before} {n} {head}')
                    positions+=n*before+n*(n+1)//2; offset+=n; past[i]+=n
                    history=req[i]['prompt_ids']+[11]*generated[i]
                    for end in range(32,(past[i]//32)*32+1,32):
                        key=tuple(history[:end])
                        if key not in index:
                            private[i]-=1; tick+=1; index[key]=[{i},tick]; changes.append(event('stored',key))
                    if head>=0:
                        generated[i]+=1; events.append(f'token {i} 11 {len(req[i]["prompt_ids"])} {generated[i]} 0 {local+10}')
                        if generated[i]==req[i]['max_new_tokens']:
                            done.add(i); finished+=1; del private[i]
                            for owners,_ in index.values(): owners.discard(i)
                            events.append(f'finish {i} length {len(req[i]["prompt_ids"])} {generated[i]} 0 {local+10}')
                values=dict.fromkeys(contract.ENGINE_STEP_FIELDS,0)
                values.update(step_id=sid,decode_seqs=len(dec),prefill_seqs=int(len(selected)>len(dec)),
                    prefill_tokens=sum(n for _,n in selected[len(dec):]),total_tokens=offset,attended_positions=positions,
                    admitted=int(any('admission_blocked' not in d and int(d.split()[18])>0 for d in decisions)),
                    finished=finished,waiting=sum(i not in past for i in req),blocks_free=free(),
                    begin_ns=local+1,schedule_ns=1,build_ns=1,execute_ns=4,postprocess_ns=1,end_ns=local+11)
                lines.append('step '+' '.join(str(values[k]) for k in contract.ENGINE_STEP_FIELDS))
                lines.extend(events); lines.append(f'prefix_step {sid} {sum(h for _,h in hits)} 1 1 0')
                lines.extend(f'cache_hit {sid} {i} {h}' for i,h in hits)
                lines.extend(decisions); lines.extend(selections); lines.extend(changes)
                lines.append(f'cache_state {sid} {free()} {len(index)} {sum(private.values())+sum(len(v[0]) for v in index.values())} {sum(private.values())}')
                local+=12; sid+=1; countsteps+=1
            lines.append(f'prefix_drained {len(req)} {countsteps} {free()} {len(index)} 0 0 0 {local+1}')
        elif phase=='clear':
            for key in list(index): lines.append(event('removed',key)); del index[key]
            lines.append(event('cleared')); lines.append(f'cache_cleared {pool} 0 0 0 0 0')
        duration=local+3
        lines.append(f'phase_time {phase} {absolute} {absolute+duration}'); absolute+=duration+2
    return '\n'.join(lines)


def campaign_fixture(active_prefill=False,adaptive_prefill=False):
    return dict(kind='engine-prefix-adaptive-campaign-v1' if adaptive_prefill else 'engine-prefix-active-campaign-v1' if active_prefill else 'engine-prefix-campaign-v1',schema_version=1,source_text=prefix.SOURCE_TEXT,
        source_text_sha256=prefix._hash(prefix.SOURCE_TEXT),
        text_provenance='repository-authored engineering text; no external corpus',
        tokenizer=dict(tables_sha256='a'*64,binary_sha256='b'*64,generator_source_sha256='c'*64),
        traces=[adaptive_trace(values) for values in contract.ENGINE_PREFIX_ADAPTIVE_CELLS] if adaptive_prefill else [trace_fixture(*values) for values in (prefix.ACTIVE_CELL_CONFIGS if active_prefill else prefix.CELL_CONFIGS)])


def record_fixture(active_prefill=False,adaptive_prefill=False):
    active=active_prefill or adaptive_prefill
    declaration=prefix._declaration(active,adaptive_prefill)
    build=dict(source=dict(repository=dict(commit='a'*40,dirty=False),sources={'uv.lock':'b'*64,
        'src/llm_mojo/benchmarks/engine_prefix.py':'c'*64,'tests/engine_metal_driver.mojo':'d'*64}),
        assets=dict(prepared='/prepared',prepared_sha256='c'*64,tables_sha256='a'*64),
        environment=dict(hardware={'chip':'Apple M4 Pro'},software={'mojo':'1.1.0','max':'26.6.0'}),
        declaration=declaration,
        command=['/mojo','build','-I','src','src/llm_mojo/benchmarks/engine_trace.mojo','-o','/engine'],
        binaries={'engine':dict(sha256='e'*64,bytes=1)})
    numerical=dict(source=build['source'],assets=build['assets'],environment=build['environment'],
        command=['/mojo','build','-I','src','-I','tests','tests/engine_metal_driver.mojo','-o','/checkpoint-driver'],
        binaries={'checkpoint-driver':dict(sha256='f'*64,bytes=1)})
    output='\n'.join(['device Apple M4 Pro/metal','qualification '+(prefix.ADAPTIVE_CHECKPOINT_IDENTITY if adaptive_prefill else prefix.ACTIVE_CHECKPOINT_IDENTITY if active else prefix.CHECKPOINT_IDENTITY),
        'configuration 27 projection 8 dtype BF16/FP32 block-size 32',
        *[f'check {name} checked 4 mismatches 0 unexpected-nonfinite 0' for name in (prefix.ADAPTIVE_CHECKPOINT_GROUPS if adaptive_prefill else prefix.ACTIVE_CHECKPOINT_GROUPS if active else prefix.CHECKPOINT_GROUPS)]])
    def receipt(command,timeout):
        return dict(command=list(map(str,command)),exit_code=0,timeout_seconds=timeout,wall_elapsed_ns=10000000)
    qualification=dict(kind='engine-prefix-adaptive-qualification-v1' if adaptive_prefill else 'engine-prefix-active-qualification-v1' if active else 'engine-prefix-qualification-v1',schema_version=1,build=build,
        checkpoint_receipt=dict(build=numerical,build_execution=receipt(numerical['command'],600),
            stdout=output,stdout_sha256=prefix._hash(output),
            execution=receipt(['/checkpoint-driver','/prepared','prefix-adaptive-qualification' if adaptive_prefill else 'prefix-active-qualification' if active else 'prefix-cache-qualification'],180)))
    campaign=campaign_fixture(active_prefill,adaptive_prefill); document=json.dumps(campaign)
    qualification.update(campaign=campaign,campaign_document=document,campaign_sha256=prefix._hash(document),preflight_cells=[])
    record=dict(kind=declaration['kind'],declaration=declaration,
        build=build,trace=campaign,trace_document=document,trace_sha256=prefix._hash(document),
        qualification=qualification,cells=[])
    conditions=dict(battery={'power_source':'AC Power'},power_mode_raw='0',
        thermal=['No thermal warning level has been recorded','No performance warning level has been recorded'])
    for trace in campaign['traces']:
        params=trace['prefix']; name=prefix._cell_name(trace,adaptive_prefill)
        cell=dict(cell_id=name,trace=trace,native_trace_path=f'/records/{name}/trace.tsv',
                  native_trace_sha256=prefix._hash(prefix.trace_tsv(trace,adaptive_prefill)),runs=[])
        preflight=dict(cell_id=name,trace=trace,native_trace_path=f'/qualification/{name}/trace.tsv',
                       native_trace_sha256=prefix._hash(prefix.trace_tsv(trace,adaptive_prefill)),runs=[])
        policies=[('enabled',mode) for mode in prefix.ADAPTIVE_MODES] if adaptive_prefill else [('enabled',cap) for cap in prefix.ACTIVE_CAPS] if active else [(cache,None) for cache in ('disabled','enabled')]
        for cache,cap in policies:
            stdout=adaptive_native_fixture(trace,cap) if adaptive_prefill else native_fixture(trace,cache,cap)
            run=dict(cache=cache,stdout=stdout)
            if active: run['prefill_policy' if adaptive_prefill else 'active_prefill_cap']=cap
            run['parsed']=prefix._parse_policy(stdout,trace,run,active,adaptive_prefill)
            run['native_execution']=receipt(prefix._command(record,preflight,run),180)
            preflight['runs'].append(run)
        preflight['expected_histories']=prefix._histories(preflight['runs'][0]['parsed'],trace)
        if active: preflight['expected_work']=prefix._work_signature(preflight['runs'][0]['parsed'],trace)
        if adaptive_prefill: preflight['expected_schedules']={r['prefill_policy']:prefix._schedule_signature(r['parsed']) for r in preflight['runs']}
        qualification['preflight_cells'].append(preflight)
        for block in range(4):
            policies=[('enabled',mode,calibration) for mode,calibration in prefix.adaptive_cells(block)] if adaptive_prefill else [('enabled',cap,calibration) for cap,calibration in prefix.active_cells(block)] if active else [(cache,None,calibration) for cache,calibration in prefix.cells(block)]
            for cache,cap,calibration in policies:
                original=next(r for r in preflight['runs'] if r['cache']==cache and (not active or r['prefill_policy' if adaptive_prefill else 'active_prefill_cap']==cap))
                run=dict(original) if adaptive_prefill else copy.deepcopy(original)
                run.update(block=block,calibration=calibration,conditions_before=conditions,conditions_after=conditions)
                run['native_execution']=receipt(prefix._command(record,cell,run),180)
                cell['runs'].append(run)
        record['cells'].append(cell)
    record['qualification_document']=json.dumps(qualification)
    record['qualification_sha256']=prefix._hash(record['qualification_document'])
    record['summary']=prefix.summary(record)
    return record


class PrefixTraceTests(unittest.TestCase):
    def test_actual_initial_gate_stdout_keeps_coverage_scope(self):
        # Exact retained checkpoint-qualification.log from numerical-attempt-2.
        initial='\n'.join(['device Apple M4 Pro/metal',
            'qualification engine-prefix-cache-reference27-v1',
            'configuration 27 projection 8 dtype BF16/FP32 block-size 32',
            'check prefix-chunk-schedule-kv-logits-greedy checked 64252482 mismatches 0 unexpected-nonfinite 0',
            'check readonly-prefix-divergent-consumers-cold-recompute checked 25926962 mismatches 0 unexpected-nonfinite 0'])
        checked=prefix.parse_checkpoint(initial,prefix.CHECKPOINT_GROUPS[:2])
        self.assertEqual(sum(c['checked'] for c in checked['checks']),90179444)
        with self.assertRaises(ValueError):
            prefix.parse_checkpoint(initial)  # Production Engine groups are still required.
        for wrong in ('configuration 26','projection 5','dtype FP16/FP32','block-size 64'):
            fields=wrong.split(); current=' '.join(fields[:-1])
            changed=initial.replace(next(part for part in ('configuration 27','projection 8','dtype BF16/FP32','block-size 32') if part.startswith(current)),wrong)
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):
                prefix.parse_checkpoint(changed,prefix.CHECKPOINT_GROUPS[:2])

    def test_hit_rows_and_seed_cost_have_separate_observed_boundaries(self):
        trace=trace_fixture()
        warm=prefix.run_summary(prefix.parse_run(native_fixture(trace),trace,'enabled',384),trace)
        cold=prefix.run_summary(prefix.parse_run(native_fixture(trace,'disabled'),trace,'disabled',384),trace)
        self.assertEqual(warm['cache_hit_tokens'],4*256)
        self.assertEqual(cold['cache_hit_tokens'],0)
        self.assertEqual(cold['total_tokens']-warm['total_tokens'],4*256)
        self.assertEqual(warm['seed_rows'],257)
        self.assertGreater(warm['episode_ns'],warm['duration_ns']+warm['seed_ns'])
        self.assertTrue(warm['cache_cleared'])

    def test_seed_scheduled_arrival_is_distinct_from_observation_and_first_step(self):
        # The failed c959179 preflight observed the seed at42, then began its
        # first step at8500. Omitting EngineCore.add's arrival argument stamped
        # token/finish arrival8500, although the frozen seed is scheduled at0.
        trace=trace_fixture(reuse=0)
        lines=native_fixture(trace,'disabled').splitlines()
        delay=8499
        phase=None
        for i,line in enumerate(lines):
            fields=line.split()
            if fields[:1]==['phase']:
                phase=fields[1]
            elif phase=='seed' and fields[:1]==['arrival']:
                fields[3]='42'
            elif phase=='seed' and fields[:1]==['step']:
                for key in ('begin_ns','end_ns'):
                    index=contract.ENGINE_STEP_FIELDS.index(key)+1
                    fields[index]=str(int(fields[index])+delay)
            elif phase=='seed' and fields[:1] in (['token'],['finish'],['prefix_drained']):
                fields[-1]=str(int(fields[-1])+delay)
            elif fields[:1]==['phase_time']:
                if phase!='seed':
                    fields[2]=str(int(fields[2])+delay)
                fields[3]=str(int(fields[3])+delay)
            lines[i]=' '.join(fields)
        valid='\n'.join(lines)
        parsed=prefix.parse_run(valid,trace,'disabled',384)
        self.assertEqual(parsed['phases']['seed']['arrivals']['128'],dict(scheduled_ns=0,actual_ns=42))
        self.assertEqual(parsed['phases']['seed']['steps'][0]['begin_ns'],8500)
        for kind in ('token','finish'):
            changed=lines.copy()
            index=next(i for i,line in enumerate(lines) if line.startswith(kind+' 128 '))
            fields=changed[index].split(); fields[-2]='8500'
            changed[index]=' '.join(fields)
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'prefix event time or prompt differs'):
                prefix.parse_run('\n'.join(changed),trace,'disabled',384)

    def test_namespace_salt_tokens_and_event_gap_are_exact(self):
        stdout=native_fixture(); trace=trace_fixture()
        event=next(line for line in stdout.splitlines() if line.startswith('cache_event 0 '))
        mutations=[stdout.replace(event,event.replace('cache_event 0 ','cache_event 1 ')),
            stdout.replace(event,event.replace('prefix-study','another-salt')),
            stdout.replace(event,event.replace('/prepared','/other-checkpoint')),
            stdout.replace(event,event.rsplit(' ',1)[0]+' 0'),stdout.replace(event+'\n',''),
            stdout.replace('cache_hit 2 0 256','cache_hit 2 0 224')]
        for broken in mutations:
            with self.subTest(change=broken[:30]),self.assertRaises(ValueError):
                prefix.parse_run(broken,trace,'enabled',384)

    def test_only_enabled_noop_handoff_may_have_zero_duration(self):
        # The final v2 enabled 1024/reuse8/blocks40 preflight recorded exactly
        # phase_time handoff 52734680091916 52734680091916, with no cache events.
        trace=trace_fixture(1024,8,40)
        stdout=native_fixture(trace)
        def interval(raw,phase,delta=0):
            row=next(line for line in raw.splitlines() if line.startswith('phase_time '+phase+' '))
            fields=row.split(); fields[3]=str(int(fields[2])+delta)
            return raw.replace(row,' '.join(fields))
        parsed=prefix.parse_run(interval(stdout,'handoff'),trace,'enabled',40)
        self.assertEqual(prefix.run_summary(parsed,trace)['handoff_ns'],0)
        corruptions=[(interval(stdout,phase),'enabled') for phase in ('seed','measure','clear')]
        corruptions.extend([(interval(stdout,'handoff',-1),'enabled'),
            (interval(stdout.replace('cache enabled','cache disabled'),'handoff'),'disabled'),
            (interval(native_fixture(trace,'disabled').replace('cache disabled','cache enabled'),'handoff'),'enabled')])
        for broken,cache in corruptions:
            with self.subTest(cache=cache,phase=broken[-100:]),self.assertRaisesRegex(ValueError,'invalid prefix phase boundary'):
                prefix.parse_run(broken,trace,cache,40)

    def test_coherent_unused_index_identity_change_still_fails(self):
        trace=trace_fixture(reuse=0)
        lines=native_fixture(trace).splitlines()
        for i,line in enumerate(lines):
            fields=line.split()
            if fields[:1]==['cache_event'] and fields[3]!='cleared':
                fields[6]=','.join(str(int(token)+90000) for token in fields[6].split(','))
                lines[i]=' '.join(fields)
        with self.assertRaisesRegex(ValueError,'outside known computed input histories'):
            prefix.parse_run('\n'.join(lines),trace,'enabled',384)

    def test_seed_cannot_publish_later_blocks_before_their_completed_rows(self):
        trace=trace_fixture(1024,0,384)
        lines=native_fixture(trace).splitlines()
        first=lines.index('phase seed')+1
        last=next(i for i,line in enumerate(lines) if line.startswith('phase_time seed '))
        stored=[line for line in lines[first:last] if line.startswith('cache_event ')]
        self.assertEqual(len(stored),32)
        relocated=[]
        for line in stored:
            fields=line.split(); fields[2]='0'
            relocated.append(' '.join(fields))
        body=[]; inserted=False
        for line in lines[first:last]:
            fields=line.split()
            if fields[:1]==['cache_event']:
                continue
            if fields[:1]==['step']:
                fields[contract.ENGINE_STEP_FIELDS.index('blocks_free')+1]='352'
            elif fields[:1]==['cache_state']:
                fields[2:4]=['352','32']
                if not inserted:
                    body.extend(relocated); inserted=True
            body.append(' '.join(fields))
        changed='\n'.join([*lines[:first],*body,*lines[last:]])
        with self.assertRaisesRegex(ValueError,'seed stored prefix exceeds cumulative completed rows'):
            prefix.parse_run(changed,trace,'enabled',384)

    def test_clear_and_ownership_are_independent_from_request_completion(self):
        stdout=native_fixture()
        drained=next(line for line in stdout.splitlines() if line.startswith('prefix_drained '))
        fields=drained.split(); fields[5]='1'
        corruptions=[stdout.replace('cache_cleared 384 0 0 0 0 0','cache_cleared 383 1 0 0 0 32'),
            stdout.replace(drained,' '.join(fields)),
            stdout.replace('phase clear','phase handoff'),
            stdout.replace('device Apple M4 Pro/metal','device cpu/host')]
        for broken in corruptions:
            with self.subTest(change=broken[-100:]),self.assertRaises(ValueError):
                prefix.parse_run(broken,trace_fixture(),'enabled',384)

    def test_omitted_step_token_and_hit_adjusted_work_fail(self):
        stdout=native_fixture()
        lines=stdout.splitlines()
        for marker in ('step ','token ','prefix_step '):
            removed=next(line for line in lines if line.startswith(marker))
            with self.subTest(marker=marker),self.assertRaises(ValueError):
                prefix.parse_run(stdout.replace(removed+'\n','',1),trace_fixture(),'enabled',384)
        step=next(line for line in lines if line.startswith('step '))
        parts=step.split(); parts[contract.ENGINE_STEP_FIELDS.index('total_tokens')+1]='255'
        with self.assertRaises(ValueError):
            prefix.parse_run(stdout.replace(step,' '.join(parts)),trace_fixture(),'enabled',384)

    def test_frozen_census_and_nonmatching_first_blocks(self):
        campaign=campaign_fixture()
        self.assertIs(prefix.validate_campaign(campaign),campaign)
        for mutation in (lambda c:c['traces'].pop(),lambda c:c['traces'].reverse(),
                         lambda c:c['traces'][0]['requests'][0].update(prompt_ids=c['traces'][0]['prefix']['seed_request']['prompt_ids'][:-1]+[11]*64)):
            broken=copy.deepcopy(campaign); mutation(broken)
            with self.assertRaises(ValueError): prefix.validate_campaign(broken)


class PrefixCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record=record_fixture()

    def test_complete_grid_and_cpu_archive_replay(self):
        self.assertEqual(self.record['summary']['measured_runs'],96)
        self.assertEqual(len(self.record['summary']['cells']),8)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            raw=json.dumps(self.record).encode(); compressed=gzip.compress(raw,mtime=0)
            (root/'engine-prefix.json.gz').write_bytes(compressed)
            (root/'engine-prefix.json').write_text(json.dumps(dict(kind=self.record['kind'],bytes=len(compressed),
                sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
            with mock.patch.object(profile,'execute',side_effect=AssertionError('CPU replay must not execute native work')), \
                 mock.patch.object(profile,'assets',side_effect=AssertionError('CPU replay must not read checkpoint assets')):
                self.assertEqual(profile.engine_replay(root),self.record['summary'])

    def test_duplicate_missing_reordered_failed_and_changed_qualification_fail(self):
        mutations=[lambda r:r['cells'][0]['runs'].pop(),
            lambda r:r['cells'][0]['runs'].reverse(),
            lambda r:r['cells'][0]['runs'][0]['native_execution'].update(exit_code=True),
            lambda r:r['cells'][0]['runs'][0]['native_execution'].update(exit_code=1),
            lambda r:r['qualification'].update(kind='unqualified'),
            lambda r:r['cells'][0]['runs'][0]['parsed']['phase_times']['seed'].update(end_ns=1)]
        for mutate in mutations:
            broken=copy.deepcopy(self.record); mutate(broken)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError): prefix.summary(broken)

    def test_coherent_namespace_change_cannot_cross_qualified_checkpoint(self):
        for phase in ('qualification','measurement'):
            broken=copy.deepcopy(self.record)
            if phase=='qualification':
                cell=broken['qualification']['preflight_cells'][0]
            else:
                cell=broken['cells'][0]
            run=cell['runs'][0]
            run['stdout']=run['stdout'].replace('qwen-reference27-projection8//prepared','qwen-reference27-projection8//other')
            run['parsed']=prefix.parse_run(run['stdout'],cell['trace'],run['cache'],cell['trace']['prefix']['blocks'])
            if phase=='qualification':
                broken['qualification_document']=json.dumps(broken['qualification'])
                broken['qualification_sha256']=prefix._hash(broken['qualification_document'])
            with self.subTest(phase=phase),self.assertRaisesRegex(ValueError,'namespace|identity'):
                prefix.summary(broken)

    def test_coherently_resealed_qualification_still_requires_provenance(self):
        mutations=[lambda q:q.update(schema_version=True),
            lambda q:q['checkpoint_receipt']['build']['binaries']['checkpoint-driver'].update(sha256='x'),
            lambda q:q['checkpoint_receipt']['build']['binaries']['checkpoint-driver'].update(bytes=0),
            lambda q:q['checkpoint_receipt']['execution'].update(wall_elapsed_ns=0),
            lambda q:q['checkpoint_receipt']['build']['command'].__setitem__(0,'/other-compiler'),
            lambda q:q['preflight_cells'][0]['runs'].pop(),
            lambda q:q['preflight_cells'][0].update(expected_histories={})]
        for mutate in mutations:
            broken=copy.deepcopy(self.record); mutate(broken['qualification'])
            broken['qualification_document']=json.dumps(broken['qualification'])
            broken['qualification_sha256']=prefix._hash(broken['qualification_document'])
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                prefix.summary(broken)

    def test_cli_binds_campaign_before_qualification_and_timing(self):
        with mock.patch('sys.argv',['model_profile','engine-qualify','--prefix-study','--build','/build',
                        '--trace','/campaign.json','--output','/qualification']),mock.patch.object(prefix,'qualify') as qualify:
            profile.main()
            qualify.assert_called_once_with(Path('/build'),Path('/qualification'),Path('/campaign.json'))
        with mock.patch('sys.argv',['model_profile','engine-collect','--prefix-study','--build','/build',
                        '--trace','/campaign.json','--qualification','/qualification.json','--output','/measure']),mock.patch.object(prefix,'collect') as collect:
            profile.main()
            collect.assert_called_once_with(Path('/build'),Path('/campaign.json'),Path('/measure'),Path('/qualification.json'))


class ActivePrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record=record_fixture(True)

    def test_balanced_grid_and_complete_cpu_replay(self):
        orders=[prefix.active_cells(b) for b in range(4)]
        self.assertEqual(len({(a,b) for row in orders for a,b in zip(row,row[1:])}),12)
        for position in range(4): self.assertEqual(len({row[position] for row in orders}),4)
        self.assertEqual(self.record['summary']['measured_runs'],80)
        self.assertEqual(self.record['summary']['untimed_preflights'],15)
        self.assertFalse(self.record['summary']['promotion'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=json.dumps(self.record).encode(); compressed=gzip.compress(raw,mtime=0)
            (root/'engine-prefix-active.json.gz').write_bytes(compressed)
            (root/'engine-prefix-active.json').write_text(json.dumps(dict(kind=self.record['kind'],bytes=len(compressed),
                sha256=hashlib.sha256(compressed).hexdigest(),uncompressed_sha256=hashlib.sha256(raw).hexdigest())))
            with mock.patch.object(profile,'execute',side_effect=AssertionError('CPU replay cannot run native work')), \
                 mock.patch.object(profile,'assets',side_effect=AssertionError('CPU replay cannot read assets')):
                self.assertEqual(profile.engine_replay(root),self.record['summary'])

    def test_active_cap_identity_idle_work_and_delivery_guards(self):
        trace=trace_fixture(1024,8,40); stdout=native_fixture(trace,'enabled',32)
        prefix.parse_run(stdout,trace,'enabled',40,32)
        with self.assertRaises(ValueError): prefix.parse_run(stdout,trace,'enabled',40)
        with self.assertRaises(ValueError): prefix.parse_run(stdout,trace,'enabled',40,True)
        corruptions=[stdout.replace('active_prefill_cap 32','active_prefill_cap 16'),stdout.replace('cache enabled','cache disabled')]
        lines=stdout.splitlines(); start=lines.index('phase measure')
        index=next(i for i in range(start,len(lines)) if lines[i].startswith('step '))
        fields=lines[index].split()
        for key,value in [('prefill_tokens',16),('attended_positions',int(fields[contract.ENGINE_STEP_FIELDS.index('attended_positions')+1])+1)]:
            changed=lines.copy(); row=fields.copy(); row[contract.ENGINE_STEP_FIELDS.index(key)+1]=str(value); changed[index]=' '.join(row)
            corruptions.append('\n'.join(changed))
        changed=lines.copy(); row=fields.copy()
        row[contract.ENGINE_STEP_FIELDS.index('decode_seqs')+1]='1'; row[contract.ENGINE_STEP_FIELDS.index('prefill_tokens')+1]='63'
        changed[index]=' '.join(row); corruptions.append('\n'.join(changed))
        finish=next(i for i,line in enumerate(lines) if line.startswith('finish 0 '))
        changed=lines.copy(); row=changed[finish].split(); row[-1]=str(int(row[-1])+1); changed[finish]=' '.join(row)
        corruptions.append('\n'.join(changed))
        for broken in corruptions:
            with self.subTest(change=broken[:80]),self.assertRaises(ValueError): prefix.parse_run(broken,trace,'enabled',40,32)

    def test_actual_raw_trace_cannot_merge_two_token_deliveries_into_one_step(self):
        path=Path(__file__).resolve().parents[1]/'studies/model_generation/engine-prefix-evidence/measurements/engine-prefix.json.gz'
        recorded=json.loads(gzip.decompress(path.read_bytes()))
        cell=next(c for c in recorded['cells'] if c['cell_id']=='prefix-1024-reuse-8-blocks-40')
        stdout=next(r['stdout'] for r in cell['runs'] if r['cache']=='enabled')
        stdout=stdout.replace('study engine-prefix-v1','study engine-prefix-active-v1\nactive_prefill_cap 0')
        prefix.parse_run(stdout,cell['trace'],'enabled',40,0)
        with self.assertRaisesRegex(ValueError,'mixed-step cap'):
            prefix.parse_run(stdout.replace('active_prefill_cap 0','active_prefill_cap 32'),cell['trace'],'enabled',40,32)
        lines=stdout.splitlines(); indexes=[i for i,line in enumerate(lines) if line.startswith('token 0 ')]
        first=lines[indexes[0]].split(); second=lines[indexes[1]].split(); first[-1]=second[-1]
        lines[indexes[1]]=' '.join(first)+'\n'+lines[indexes[1]]; del lines[indexes[0]]
        with self.assertRaisesRegex(ValueError,'two tokens|head census'):
            prefix.parse_run('\n'.join(lines),cell['trace'],'enabled',40,0)

    def test_frozen_preflight_caps_work_grid_and_commands_cannot_change(self):
        for mutate in (lambda r:r['cells'][0]['runs'].pop(),lambda r:r['cells'][0]['runs'].reverse(),
                lambda r:r['cells'][0]['runs'][0]['native_execution']['command'].__setitem__(-1,'16'),
                lambda r:r['qualification']['preflight_cells'][0]['expected_work']['measure'].update(rows=1)):
            broken=copy.deepcopy(self.record); mutate(broken)
            broken['qualification_document']=json.dumps(broken['qualification']); broken['qualification_sha256']=prefix._hash(broken['qualification_document'])
            with self.assertRaises(ValueError): prefix.summary(broken)

    def test_gap_counts_and_crossed_step_attribution_are_exact(self):
        def step(i,decode,prefill,begin,end):
            return dict(step_id=i,decode_seqs=decode,prefill_tokens=prefill,begin_ns=begin,end_ns=end,execute_ns=end-begin,schedule_ns=0,build_ns=0)
        scale=1_000_000
        steps=[step(0,0,64,0,10*scale),step(1,1,32,10*scale,20*scale),step(2,2,0,20*scale,70*scale),step(3,3,0,70*scale,130*scale)]
        events=[dict(kind='token',request_id=i,step_id=s,emitted_ns=t*scale) for s,t,ids in [(0,0,[0,2]),(1,20,[0,1]),(2,70,[0,1]),(3,130,[0,1,2])] for i in ids]
        data=prefix._active_metrics(dict(phases=dict(measure=dict(steps=steps,events=events))))
        self.assertEqual(data['gap_tail']['count'],6)
        self.assertEqual(data['gap_tail']['above_20ms'],5)
        self.assertEqual(data['gap_tail']['above_50ms'],3)
        self.assertEqual(data['gap_tail']['maximum_ns'],130*scale)
        self.assertEqual([r['maximum_ns'] for r in data['gap_tail']['requests']],[60*scale,60*scale,130*scale])
        self.assertEqual(data['gap_tail']['attribution']['mixed']['count'],2)
        self.assertEqual(data['gap_tail']['attribution']['decode_only']['count'],4)
        self.assertEqual(data['step_classes']['mixed']['prefill_rows_max'],32)

    def test_decision_noise_buffers_thresholds_and_unavailable_tails(self):
        def comparison(ratio,noise=.05):
            return dict(available=True,block_ratios=[ratio]*4,median_ratio=ratio,noise_floor=noise,
                speed_verdict=profile._outcome([ratio]*4,noise))
        base={metric:comparison(ratio) for metric,ratio in [('duration_ns',1),('episode_ns',1),('p95_ttft_ns',1),('p95_gap_ns',.75),('maximum_gap_ns',.8)]}
        self.assertEqual(prefix._active_decision(base)['hypothesis'],'supported')
        for metric,ratio,status in [('duration_ns',1.08,'inconclusive'),('p95_ttft_ns',1.17,'failed'),('p95_gap_ns',.76,'failed'),('maximum_gap_ns',1.08,'failed')]:
            changed=copy.deepcopy(base); changed[metric]=comparison(ratio)
            self.assertEqual(prefix._active_decision(changed)['hypothesis'],status)
        changed=copy.deepcopy(base); changed['p95_gap_ns']=comparison(.75,.30)
        self.assertEqual(prefix._active_decision(changed)['hypothesis'],'inconclusive')
        changed=copy.deepcopy(base); changed['maximum_gap_ns']=dict(available=False)
        self.assertEqual(prefix._active_decision(changed)['hypothesis'],'inconclusive')
        runs=[]
        for block in range(4):
            for cap,calibration in prefix.active_cells(block):
                factor=1.30 if calibration else .70 if cap else 1
                runs.append(dict(block=block,active_prefill_cap=cap,calibration=calibration,duration_ns=100,episode_ns=100,
                    ttft=dict(p95_ns=100),inter_token=dict(p95_ns=100*factor),gap_tail=dict(maximum_ns=100*factor)))
        actual=prefix._active_comparisons(runs,32)
        self.assertAlmostEqual(actual['p95_gap_ns']['noise_floor'],.30)
        self.assertEqual(actual['duration_ns']['noise_floor'],.05)
        self.assertEqual(prefix._active_decision(actual)['hypothesis'],'inconclusive')

    def test_failed_native_call_is_retained_once_without_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); binary=root/'checkpoint-driver'; binary.write_bytes(b'fixture')
            q=copy.deepcopy(self.record['qualification']); q['checkpoint_receipt']['build']['command'][-1]=str(binary)
            q['checkpoint_receipt']['build']['binaries']['checkpoint-driver']=dict(sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),bytes=7)
            qualification=root/'qualification.json'; qualification.write_text(json.dumps(q))
            campaign=root/'campaign.json'; campaign.write_text(self.record['trace_document'])
            def fail(command,log,timeout):
                log.write_text('actual native failure\n')
                log.with_suffix('.execution.json').write_text(json.dumps(dict(command=list(map(str,command)),exit_code=9,timeout_seconds=timeout,wall_elapsed_ns=100)))
                raise profile.NativeCommandError(9,log)
            with mock.patch.object(profile,'verify_engine_build',return_value=self.record['build']), \
                 mock.patch.object(prefix,'_qualification',return_value=self.record['summary']['qualification']), \
                 mock.patch.object(profile,'conditions',return_value={}),mock.patch.object(profile,'checked_execution',side_effect=fail) as execute:
                with self.assertRaises(profile.NativeCommandError): prefix.collect(root/'build',campaign,root/'output',qualification,True)
                self.assertEqual(execute.call_count,1)
            failed=json.loads((root/'output/incomplete-campaign.json').read_text())['cells'][0]['runs']
            self.assertEqual(len(failed),1)
            self.assertEqual(failed[0]['native_execution']['exit_code'],9)
            self.assertEqual(failed[0]['stdout'],'actual native failure\n')


class AdaptivePrefixTests(unittest.TestCase):
    def test_literal_predicate_boundaries_and_first_failure_precedence(self):
        base=dict(required_blocks=3,available_blocks=5,selected_decodes=1,incumbent_remaining_outputs=15,
            remaining_prompt_tokens=64,max_sequences=8,baseline_tokens=64)
        self.assertEqual(prefix.adaptive_decision(base,'adaptive'),dict(chosen_tokens=32,eligible=True,applied=True,reason='adaptive_short_tail'))
        for changes,reason in [({'required_blocks':6},'admission_blocked'),({'required_blocks':0},'no_private_blocks'),
                ({'selected_decodes':0},'decode_count'),({'selected_decodes':2},'decode_count'),
                ({'incumbent_remaining_outputs':1},'incumbent_terminal'),({'remaining_prompt_tokens':32},'tail_length'),
                ({'remaining_prompt_tokens':65},'tail_length'),({'max_sequences':2},'last_sequence_slot'),
                ({'available_blocks':6},'capacity_headroom')]:
            with self.subTest(changes=changes):
                value=prefix.adaptive_decision(dict(base,**changes),'adaptive')
                self.assertEqual(value['reason'],reason); self.assertFalse(value['eligible']); self.assertFalse(value['applied'])
        for tail in (33,63,64): self.assertTrue(prefix.adaptive_decision(dict(base,remaining_prompt_tokens=tail),'adaptive')['eligible'])
        for available in (3,5): self.assertTrue(prefix.adaptive_decision(dict(base,available_blocks=available),'adaptive')['eligible'])
        self.assertEqual(prefix.adaptive_decision(dict(base,available_blocks=2),'adaptive')['reason'],'admission_blocked')
        for mode,rows in [('current',64),('fixed32',32)]:
            result=prefix.adaptive_decision(base,mode)
            self.assertEqual((result['chosen_tokens'],result['eligible'],result['applied'],result['reason']),(rows,True,False,'adaptive_disabled'))
        self.assertEqual(prefix.adaptive_decision(dict(base,baseline_tokens=32),'adaptive')['reason'],'baseline_at_most_32')
        self.assertEqual(prefix.adaptive_decision(dict(base,required_blocks=9,selected_decodes=0),'adaptive')['chosen_tokens'],0)

    def test_adaptive_ratios_keep_metric_noise_and_uncertainty(self):
        runs=[]
        for block in range(4):
            for mode,calibration in prefix.adaptive_cells(block):
                gap=100*(1.3 if calibration else .7 if mode=='adaptive' else 1)
                runs.append(dict(block=block,prefill_policy=mode,calibration=calibration,duration_ns=100,
                    episode_ns=100,ttft=dict(p95_ns=100),inter_token=dict(p95_ns=gap),gap_tail=dict(maximum_ns=gap)))
        comparisons=prefix._active_comparisons(runs,'adaptive',True)
        self.assertEqual(comparisons['p95_gap_ns']['block_ratios'],[.7]*4)
        self.assertAlmostEqual(comparisons['p95_gap_ns']['noise_floor'],.3)
        self.assertEqual(comparisons['duration_ns']['noise_floor'],.05)
        self.assertEqual(prefix._active_decision(comparisons)['hypothesis'],'inconclusive')
        for r in runs:
            if r['calibration']: r['inter_token']['p95_ns']=100; r['gap_tail']['maximum_ns']=100
        comparisons=prefix._active_comparisons(runs,'adaptive',True)
        self.assertEqual(prefix._active_decision(comparisons)['hypothesis'],'supported')

    def test_live_ownership_predicate_and_ordered_selection_mutations_are_rejected(self):
        trace=adaptive_trace([8,40,64,16,'first']); stdout=adaptive_native_fixture(trace)
        parsed=prefix.parse_run(stdout,trace,'enabled',40,adaptive_prefill_mode='adaptive')
        self.assertGreater(parsed['policy_replay']['applied'],0)
        lines=stdout.splitlines(); decision=next(i for i,line in enumerate(lines) if line.startswith('prefill_decision ') and line.endswith('adaptive_short_tail'))
        mutations=[]
        # A coherent self-reported free/A change cannot override reconstructed live ownership.
        changed=lines.copy(); fields=changed[decision].split()
        for index in (6,9): fields[index]=str(int(fields[index])+1)
        changed[decision]=' '.join(fields); mutations.append(changed)
        for index,value in [(12,'-1'),(13,'1'),(18,'64'),(19,'0'),(20,'0'),(21,'adaptive_disabled')]:
            changed=lines.copy(); fields=changed[decision].split(); fields[index]=value; changed[decision]=' '.join(fields); mutations.append(changed)
        selection=next(i for i,line in enumerate(lines) if line.startswith('batch_selection ') and line.split()[-1]!='-1')
        for index in (2,3,4,5,6):
            changed=lines.copy(); fields=changed[selection].split(); fields[index]=str(int(fields[index])+1); changed[selection]=' '.join(fields); mutations.append(changed)
        changed=lines.copy(); del changed[decision]; mutations.append(changed)
        changed=lines.copy(); changed.insert(selection,changed[selection]); mutations.append(changed)
        for changed in mutations:
            with self.subTest(row=changed[decision][:120]),self.assertRaises(ValueError):
                prefix.parse_run('\n'.join(changed),trace,'enabled',40,adaptive_prefill_mode='adaptive')
        with self.assertRaises(ValueError): prefix.parse_run(stdout,trace,'enabled',40,32,'adaptive')
        with self.assertRaises(ValueError): prefix.parse_run(stdout,trace,'enabled',40,adaptive_prefill_mode='unknown')

    def test_alternating_workload_freezes_observed_hits_and_large_pool_falls_back(self):
        trace=adaptive_trace([4,40,64,16,'even'])
        observed=[]
        for mode in prefix.ADAPTIVE_MODES:
            parsed=prefix.parse_run(adaptive_native_fixture(trace,mode),trace,'enabled',40,adaptive_prefill_mode=mode)
            observed.append(prefix._work_signature(parsed,trace))
        self.assertEqual(observed[0],observed[1]); self.assertEqual(observed[0],observed[2])
        # Intended matching text can miss after idle LRU eviction; no H>0 shortcut.
        self.assertEqual(observed[0]['measure']['cached_tokens_by_request']['2'],0)
        trace=adaptive_trace([8,384,64,16,'first'])
        current=prefix.parse_run(adaptive_native_fixture(trace,'current'),trace,'enabled',384,adaptive_prefill_mode='current')
        adaptive=prefix.parse_run(adaptive_native_fixture(trace),trace,'enabled',384,adaptive_prefill_mode='adaptive')
        self.assertEqual(prefix._schedule_signature(current),prefix._schedule_signature(adaptive))
        self.assertEqual(adaptive['policy_replay'].get('applied',0),0)

    def test_frozen_192_run_census_preflights_schedules_and_noise_decisions(self):
        record=record_fixture(adaptive_prefill=True); result=record['summary']
        self.assertEqual((result['measured_runs'],result['untimed_preflights'],len(result['cells'])),(192,36,12))
        self.assertEqual(sum(len(c['runs']) for c in record['qualification']['preflight_cells']),36)
        self.assertTrue(result['same_reuse_work']); self.assertFalse(result['promotion'])
        self.assertEqual(result['usefulness']['primary_policy'],'adaptive')
        primary=next(c for c in result['cells'] if c['cell_id']==result['usefulness']['primary_cell'])
        self.assertEqual(primary['logical_schedule_matches'],dict(required=True,reference='fixed32',matched=True,measured_steps=69))
        self.assertGreater(primary['runs'][0]['gap_tail']['count'],0)
        self.assertEqual(set(primary['comparisons']),{'fixed32','adaptive'})
        self.assertEqual(len(primary['comparisons']['adaptive']['p95_gap_ns']['block_ratios']),4)
        self.assertEqual(primary['comparisons']['adaptive']['duration_ns']['noise_floor'],.05)
        orders=[prefix.adaptive_cells(b) for b in range(4)]
        self.assertEqual(len({(a,b) for row in orders for a,b in zip(row,row[1:])}),12)
        # Census rejection occurs before parsing hundreds of run bodies.
        broken=dict(record,cells=[dict(record['cells'][0],runs=record['cells'][0]['runs'][:-1]),*record['cells'][1:]])
        with self.assertRaisesRegex(ValueError,'paired grid'): prefix.summary(broken)
        broken=dict(record,cells=[dict(record['cells'][0],runs=list(reversed(record['cells'][0]['runs']))),*record['cells'][1:]])
        with self.assertRaisesRegex(ValueError,'paired grid'): prefix.summary(broken)
        q=copy.deepcopy(record['qualification']); q['preflight_cells'][0]['expected_schedules']=copy.deepcopy(q['preflight_cells'][0]['expected_schedules'])
        q['preflight_cells'][0]['expected_schedules']['adaptive']['measure'][0]['selections'][0]['past']+=1
        doc=json.dumps(q); broken=dict(record,qualification=q,qualification_document=doc,qualification_sha256=prefix._hash(doc))
        with self.assertRaisesRegex(ValueError,'schedules changed'): prefix.summary(broken)

    def test_failed_adaptive_call_preserves_actual_first_attempt_and_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); binary=root/'checkpoint-driver'; binary.write_bytes(b'fixture')
            campaign=campaign_fixture(adaptive_prefill=True); trace_path=root/'campaign.json'; trace_path.write_text(json.dumps(campaign))
            build=dict(command=['/mojo','-o','/engine'],assets=dict(prepared='/prepared',tables_sha256='a'*64),
                source=dict(sources={'src/llm_mojo/benchmarks/engine_prefix.py':'c'*64}))
            q=dict(checkpoint_receipt=dict(build=dict(command=['/mojo','-o',str(binary)],
                binaries={'checkpoint-driver':dict(sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),bytes=7)})))
            qualification=root/'qualification.json'; qualification.write_text(json.dumps(q))
            accepted=dict(expected_histories={prefix._cell_name(t,True):{} for t in campaign['traces']})
            def fail(command,log,timeout):
                self.assertEqual(command[-2:],['enabled','current'])
                log.write_text('actual adaptive failure\n')
                log.with_suffix('.execution.json').write_text(json.dumps(dict(command=list(map(str,command)),exit_code=9,timeout_seconds=timeout,wall_elapsed_ns=100)))
                raise profile.NativeCommandError(9,log)
            with mock.patch.object(profile,'verify_engine_build',return_value=build), \
                 mock.patch.object(prefix,'_qualification',return_value=accepted), \
                 mock.patch.object(profile,'conditions',return_value={}),mock.patch.object(profile,'checked_execution',side_effect=fail) as execute:
                with self.assertRaises(profile.NativeCommandError):
                    prefix.collect(root/'build',trace_path,root/'output',qualification,adaptive_prefill=True)
                self.assertEqual(execute.call_count,1)
            failed=json.loads((root/'output/incomplete-campaign.json').read_text())['cells'][0]['runs']
            self.assertEqual(len(failed),1); self.assertEqual(failed[0]['prefill_policy'],'current')
            self.assertEqual(failed[0]['stdout'],'actual adaptive failure\n')
            self.assertEqual(failed[0]['native_execution']['exit_code'],9)


if __name__=='__main__':
    unittest.main()
