"""Actual native terminal integration, including a controlling PTY and Ctrl-C.

Explicit checkpoint test, not part of weight-free unittest discovery. The model
and all inference remain in the native process. Python only drives the terminal.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import time


def events(path):
    return list(csv.DictReader(Path(path).read_text().splitlines(),delimiter='\t'))


def validate(rows, maximum):
    if len([r for r in rows if r['event']=='load'])!=1:
        raise ValueError('missing single resident model load')
    if not any(r['event']=='load' and r['value']=='Apple M4 Pro/metal' for r in rows):
        raise ValueError('missing actual M4 Pro/Metal identity')
    previous=None
    turns=[]
    for row in rows:
        if row['event']=='reset': previous=None
        if row['event']!='begin': continue
        turn=row['turn'];group=[r for r in rows if r['turn']==turn]
        def ids(event):
            selected=[r for r in group if r['event']==event]
            if [int(r['index']) for r in selected]!=list(range(len(selected))):
                raise ValueError('incomplete ordered token history')
            return [int(r['value']) for r in selected]
        prompt,generated,history=ids('prompt'),ids('token'),ids('history')
        finish=next(r for r in group if r['event']=='finish')
        if finish['value']=='error': raise ValueError('native execution failed')
        before,after=int(row['index']),int(finish['index'])
        if int(row['value'])!=len(prompt) or not 0<=before<=after<=len(history)<=4096:
            raise ValueError('invalid chat lengths')
        if previous is None:
            if before!=0: raise ValueError('reset did not empty the cache')
        elif before!=previous['cached'] or prompt[:len(previous['history'])]!=previous['history']:
            raise ValueError('conversation prefix or persistent cache was lost')
        if not len(generated)<=maximum: raise ValueError('reply budget exceeded')
        tail=[] if generated and generated[-1]==151645 else [151645]
        if history!=prompt+generated+tail+[198]:
            raise ValueError('pending token or chat closure lost/duplicated')
        if generated:
            allowed={len(prompt)+len(generated)-1}
            if finish['value']=='interrupted': allowed.add(len(prompt)+len(generated))
            if after not in allowed: raise ValueError('generated token/cache accounting mismatch')
        submitted=next(r for r in group if r['event']=='submitted')
        if int(submitted['value'])!=24*after: raise ValueError('old rows were recomputed')
        token_events=[r for r in group if r['event']=='token']
        times=[int(r['nanoseconds']) for r in token_events]
        if times!=sorted(times) or (times and int(finish['nanoseconds'])<times[-1]):
            raise ValueError('invalid streaming timestamps')
        visible=[r for r in token_events if int(r['value']) not in (151643,151645)]
        # Tokens are timestamped after decoded output is flushed. Some byte
        # fragments need multiple tokens before text becomes visible.
        rate=(len(visible)-1)*1e9/(int(visible[-1]['nanoseconds'])-int(visible[0]['nanoseconds'])) if len(visible)>1 else None
        text_events=[r for r in group if r['event']=='text']
        first_visible_ms=int(text_events[0]['nanoseconds'])/1e6 if text_events else None
        turns.append(dict(first_visible_ms=first_visible_ms,turn=int(turn),cached_before=before,cached=after,prompt=prompt,history=history,
                          generated=generated,reason=finish['value'],first_token_ms=times[0]/1e6 if times else None,
                          request_ms=int(finish['nanoseconds'])/1e6,output_tokens_per_second=rate))
        previous=turns[-1]
    return turns


"""Stage this function in existing tests/chat_terminal.py; separate contract."""


def validate_engine(rows, maximum, async_steps=False):
    if type(maximum) is not int or not 1 <= maximum <= 4096:
        raise ValueError('invalid engine chat reply maximum')
    if type(async_steps) is not bool:
        raise ValueError('engine chat stepping declaration must be explicit')
    loads=[r for r in rows if r['event']=='load']
    modes=[r for r in rows if r['event']=='engine_mode']
    if len(loads)!=1 or loads[0]['value']!='Apple M4 Pro/metal':
        raise ValueError('missing single actual Metal resident engine load')
    expected_mode='reference-27/async-two-context/recompute-history' if async_steps else 'reference-27/recompute-history'
    if len(modes)!=1 or modes[0]['value']!=expected_mode:
        raise ValueError('engine chat mode or KV lifecycle changed')
    begins=[r for r in rows if r['event']=='begin']
    if [int(r['turn']) for r in begins]!=list(range(1,len(begins)+1)):
        raise ValueError('missing or duplicate engine chat turn')
    previous=None
    turns=[]
    for row in rows:
        if row['event']=='reset':
            previous=None
        if row['event']=='rejected' and previous is not None:
            if int(row['value'])!=len(previous['history']):
                raise ValueError('engine context rejection changed exact history')
        if row['event']!='begin':
            continue
        turn=row['turn']; group=[r for r in rows if r['turn']==turn]
        def one(name):
            found=[r for r in group if r['event']==name]
            if len(found)!=1:
                raise ValueError('missing or duplicate engine '+name+' receipt')
            return found[0]
        def value(name):
            return int(one(name)['value'])
        def ids(name):
            found=[r for r in group if r['event']==name]
            if [int(r['index']) for r in found]!=list(range(len(found))):
                raise ValueError('engine token order is incomplete')
            result=[int(r['value']) for r in found]
            if any(not 0<=token<151936 for token in result):
                raise ValueError('engine token is outside vocabulary')
            return result
        prompt,generated,history=ids('prompt'),ids('token'),ids('history')
        finish=one('finish'); reason=finish['value']
        if reason not in ('stop','limit','interrupted'):
            raise ValueError('native engine chat failed or has unknown terminal reason')
        if (int(row['index'])!=0 or int(finish['index'])!=0 or int(row['value'])!=len(prompt)
                or not prompt or len(history)>4096 or len(generated)>maximum):
            raise ValueError('invalid engine chat prompt/cache lengths')
        if previous is not None and prompt[:len(previous['history'])]!=previous['history']:
            raise ValueError('engine chat lost exact conversation token prefix')
        if reason=='stop' and (not generated or generated[-1] not in (151643,151645)):
            raise ValueError('engine chat reported a false stop')
        if reason=='limit' and (len(generated)!=maximum or generated[-1] in (151643,151645)):
            raise ValueError('engine chat limit accounting differs')
        if any(token in (151643,151645) for token in generated[:-1]):
            raise ValueError('engine chat generated after its stop token')
        closure=[] if generated and generated[-1]==151645 else [151645]
        if history!=prompt+generated+closure+[198]:
            raise ValueError('engine chat lost or duplicated raw generated tokens/closure')
        rows_count=value('engine_rows')
        async_counters={}
        names=('submissions','completions','peak_pending','pending','submitted_rows','selected_heads',
               'delivered_tokens','chained_rows','discarded_tokens','discarded_rows')
        if async_steps:
            async_counters={name:value('async_'+name) for name in names}
            if (any(v<0 for v in async_counters.values())
                    or async_counters['submissions']!=async_counters['completions']
                    or async_counters['pending']!=0 or not 0<=async_counters['peak_pending']<=2
                    or async_counters['submitted_rows']!=rows_count
                    or async_counters['selected_heads']!=len(generated)+async_counters['discarded_tokens']
                    or async_counters['delivered_tokens']!=len(generated)
                    or async_counters['selected_heads']>async_counters['submissions']
                    or async_counters['discarded_tokens']>1
                    or async_counters['chained_rows']>max(0,async_counters['selected_heads']-1)
                    or (rows_count>0 and (async_counters['submissions']<1 or async_counters['peak_pending']<1))
                    or (rows_count==0 and any(async_counters.values()))):
                raise ValueError('async chat submitted/completed/head/delivery census differs')
            necessary=len(prompt)+len(generated)-1 if generated else 0
            if rows_count!=necessary+async_counters['discarded_rows']:
                raise ValueError('async chat discarded rows were omitted from execution')
            if generated:
                if async_counters['discarded_rows']!=async_counters['discarded_tokens']:
                    raise ValueError('async chat terminal decode rows disagree with discarded heads')
            elif reason!='interrupted' or not 0<=rows_count<=len(prompt):
                raise ValueError('async chat cancelled prefill exceeded its prompt')
            elif async_counters['selected_heads']!=int(rows_count==len(prompt)):
                raise ValueError('async cancelled prefill head differs from submitted prompt completion')
            if reason=='limit' and (async_counters['discarded_tokens'] or async_counters['discarded_rows']):
                raise ValueError('async chat speculated past a known output limit')
        else:
            if any(r['event']=='async_'+name for r in group for name in names):
                raise ValueError('synchronous engine chat unexpectedly claimed async work')
            if generated:
                if rows_count!=len(prompt)+len(generated)-1:
                    raise ValueError('engine chat normal/decoded turn rows differ from P+G-1')
            elif reason!='interrupted' or not 0<=rows_count<len(prompt):
                raise ValueError('engine chat prefill abort executed invalid rows')
        if value('submitted')!=24*rows_count or value('engine_steps')<1:
            raise ValueError('engine chat submitted model rows disagree with executed engine rows')
        if (value('kv_free')!=value('kv_total') or value('kv_total')<1
                or value('kv_owned')!=0 or value('kv_written')!=0 or value('engine_live')!=0):
            raise ValueError('engine chat terminal retained valid KV or request ownership')
        token_events=[r for r in group if r['event']=='token']
        times=[int(r['nanoseconds']) for r in token_events]
        if times!=sorted(times) or any(t<0 for t in times) or int(finish['nanoseconds'])<max(times,default=0):
            raise ValueError('invalid engine streaming timestamps')
        texts=[r for r in group if r['event']=='text']
        if any(int(r['value'])<1 or int(r['nanoseconds'])<0 for r in texts):
            raise ValueError('invalid engine visible-byte receipt')
        visible=[r for r in token_events if int(r['value']) not in (151643,151645)]
        span=int(visible[-1]['nanoseconds'])-int(visible[0]['nanoseconds']) if len(visible)>1 else 0
        rate=(len(visible)-1)*1e9/span if span>0 else None
        turns.append(dict(turn=int(turn),cached_before=0,cached=0,prompt=prompt,history=history,
            generated=generated,reason=reason,executed_rows=rows_count,
            valid_kv_at_terminal=0,engine_live_at_terminal=0,
            first_visible_ms=int(texts[0]['nanoseconds'])/1e6 if texts else None,
            first_token_ms=times[0]/1e6 if times else None,
            request_ms=int(finish['nanoseconds'])/1e6,output_tokens_per_second=rate))
        if async_steps:
            turns[-1]['async']=async_counters
        previous=turns[-1]
    return turns


def run(binary,prepared,tables,output,engine=False,async_steps=False):
    if type(engine) is not bool or type(async_steps) is not bool or (async_steps and not engine):
        raise ValueError('async terminal validation requires explicit engine mode')
    output.mkdir(parents=True,exist_ok=False)
    basic=output/'basic.tsv'
    args=[str(binary),str(prepared),str(tables),'16','256','',str(basic)]
    if async_steps: args.append('async')
    message='My name is Ada. Reply in one short sentence.'
    input_bytes=(message+'\nWhat is my name?\n/reset\n'+message+'\n'+'a '*5000+'\nSay caffè ☕.\n').encode()+b'\xff\n/exit\n'
    start=time.monotonic()
    result=subprocess.run(args,input=input_bytes,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    text=result.stdout.decode('utf-8')
    (output/'basic.txt').write_text(text)
    if result.returncode or 'Conversation full:' not in text or 'Input must be valid UTF-8.' not in text:
        raise ValueError('input rejection or execution failed: '+text)
    checker=(lambda records,maximum:validate_engine(records,maximum,async_steps=async_steps)) if engine else validate
    turns=checker(events(basic),16)
    if len(turns)!=4 or turns[0]['generated']!=turns[2]['generated']:
        raise ValueError('multi-turn/reset replay mismatch')
    elapsed=time.monotonic()-start
    # Report-free output must use the same inference/streaming behavior.
    args[6]=''
    plain=subprocess.run(args,input=input_bytes,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    if plain.returncode or plain.stdout!=result.stdout:
        raise ValueError('reporting changed visible output')
    pty_report=output/'interrupt.tsv'
    args=[str(binary),str(prepared),str(tables),'256','256','',str(pty_report)]
    if async_steps: args.append('async')
    pid,fd=pty.fork()
    if pid==0:
        os.execv(binary,args)
    transcript=bytearray(); cursor=0; exit_status=None
    def until(marker,timeout=120):
        nonlocal cursor
        deadline=time.monotonic()+timeout
        while marker not in transcript[cursor:]:
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError('terminal did not emit '+repr(marker))
            ready,_,_=select.select([fd],[],[],min(remaining,1))
            if ready:
                data=os.read(fd,65536)
                if not data: raise ValueError('terminal closed early')
                transcript.extend(data)
        cursor=transcript.index(marker,cursor)+len(marker)
    try:
        until(b'You: ')
        # Typed Ctrl-C at the prompt cancels input without changing history.
        os.write(fd,b'\x03');until(b'Input cancelled.');until(b'You: ')
        os.write(fd,b'Write a very long story about a journey.\n')
        until(b'Assistant: ')
        # Wait for actual output, then deliver terminal-generated SIGINT.
        deadline=time.monotonic()+30
        while len(transcript)==cursor and time.monotonic()<deadline:
            if select.select([fd],[],[],1)[0]: transcript.extend(os.read(fd,65536))
        os.write(fd,b'\x03')
        until(b'[Reply stopped]');until(b'You: ')
        os.write(fd,b'What was the story about? Answer briefly.\n')
        until(b'Assistant: ');until(b'You: ')
        os.write(fd,b'/reset\n');until(b'Conversation reset.');until(b'You: ')
        os.write(fd,b'Say hello.\n');until(b'Assistant: ');until(b'You: ')
        os.write(fd,b'\x04');until(b'Goodbye.')
        _,exit_status=os.waitpid(pid,0)
    finally:
        if exit_status is None:
            os.kill(pid,signal.SIGKILL);os.waitpid(pid,0)
        os.close(fd)
        (output/'interrupt.txt').write_bytes(transcript)
    if not os.WIFEXITED(exit_status) or os.WEXITSTATUS(exit_status)!=0:
        raise ValueError('PTY child failed')
    interrupted=checker(events(pty_report),256)
    if len(interrupted)!=3 or interrupted[0]['reason']!='interrupted':
        raise ValueError('missing interrupted/resumed/reset conversation')
    return dict(kind='native-engine-async-chat-terminal-v1' if async_steps else 'native-engine-chat-terminal-v1' if engine else 'native-chat-terminal-v1',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                basic_turns=turns,interrupt_turns=interrupted,basic_wall_seconds=elapsed,
                report_free_output_exact=True,basic_output=text,pty_output=transcript.decode(),
                basic_events=events(basic),interrupt_events=events(pty_report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',required=True,type=Path)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--tables',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--engine',action='store_true',help='Validate the separate EngineCore chat lifecycle')
    parser.add_argument('--async-stepping',action='store_true',help='Require the engine two-context async lifecycle')
    a=parser.parse_args()
    if a.async_stepping and not a.engine: parser.error('--async-stepping requires --engine')
    result=run(a.binary.resolve(),a.prepared.resolve(),a.tables.resolve(),a.output,engine=a.engine,async_steps=a.async_stepping)
    (a.output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('Native terminal passed: four piped turns, three PTY turns, Ctrl-C, EOF, reset, Unicode, context rejection and report parity.')

if __name__=='__main__': main()
