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


def run(binary,prepared,tables,output):
    output.mkdir(parents=True,exist_ok=False)
    basic=output/'basic.tsv'
    args=[str(binary),str(prepared),str(tables),'16','256','',str(basic)]
    message='My name is Ada. Reply in one short sentence.'
    input_bytes=(message+'\nWhat is my name?\n/reset\n'+message+'\n'+'a '*5000+'\nSay caffè ☕.\n').encode()+b'\xff\n/exit\n'
    start=time.monotonic()
    result=subprocess.run(args,input=input_bytes,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    text=result.stdout.decode('utf-8')
    (output/'basic.txt').write_text(text)
    if result.returncode or 'Conversation full:' not in text or 'Input must be valid UTF-8.' not in text:
        raise ValueError('input rejection or execution failed: '+text)
    turns=validate(events(basic),16)
    if len(turns)!=4 or turns[0]['generated']!=turns[2]['generated']:
        raise ValueError('multi-turn/reset replay mismatch')
    elapsed=time.monotonic()-start
    # Report-free output must use the same inference/streaming behavior.
    args[-1]=''
    plain=subprocess.run(args,input=input_bytes,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    if plain.returncode or plain.stdout!=result.stdout:
        raise ValueError('reporting changed visible output')
    pty_report=output/'interrupt.tsv'
    args=[str(binary),str(prepared),str(tables),'256','256','',str(pty_report)]
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
    interrupted=validate(events(pty_report),256)
    if len(interrupted)!=3 or interrupted[0]['reason']!='interrupted':
        raise ValueError('missing interrupted/resumed/reset conversation')
    return dict(kind='native-chat-terminal-v1',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                basic_turns=turns,interrupt_turns=interrupted,basic_wall_seconds=elapsed,
                report_free_output_exact=True,basic_output=text,pty_output=transcript.decode(),
                basic_events=events(basic),interrupt_events=events(pty_report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',required=True,type=Path)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--tables',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args()
    result=run(a.binary.resolve(),a.prepared.resolve(),a.tables.resolve(),a.output)
    (a.output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('Native terminal passed: four piped turns, three PTY turns, Ctrl-C, EOF, reset, Unicode, context rejection and report parity.')

if __name__=='__main__': main()
