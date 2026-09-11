"""Collect the bounded native terminal-chat study from a clean local checkout."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
from llm_mojo._repository import environment_tool, repository_root
from llm_mojo.mlp_validation import source_identity, sha, write
from llm_mojo.model_assets import verify_prepared
from llm_mojo.tokenizer_assets import ensure_prepared
from llm_mojo.model_validation import bf16, numerical_diagnostic, prediction_diagnostic, environment
from llm_mojo.benchmarks.environment import stable_environment, conditions_snapshot, require_ac, require_nominal_thermal_state
from chat_terminal import run as terminal_run


def summarize_driver(directory,stdout):
    lines=[line.split() for line in stdout.splitlines()]
    if 'device Apple M4 Pro backend metal' not in stdout or 'chat lifecycle passed:' not in stdout:
        raise ValueError('missing actual device or lifecycle completion')
    prefill=[dict(turn=int(r[1]),before=int(r[3]),prompt=int(r[5])) for r in lines if r[0]=='prefill']
    finishes={int(r[1]):dict(cached=int(r[3]),history=int(r[5]),generated=int(r[7]),reason=r[9]) for r in lines if r[0]=='finish'}
    if [r['turn'] for r in prefill]!=[0,1,2] or set(finishes)!={0,1,2}:
        raise ValueError('incomplete native conversation')
    diagnostics=[];storage=[]
    for r in prefill:
        turn=r['turn'];base=directory/f'turn{turn}';finish=finishes[turn]
        a,b=bf16(base/'cached.bin',(1,151936)),bf16(base/'replay.bin',(1,151936))
        diagnostics.append(dict(**r,**finish,**numerical_diagnostic(a,b),**prediction_diagnostic(a,b)))
        for layer in range(24):
            for kind in ('key','value'):
                before=np.fromfile(base/'before'/f'{kind}_{layer}.bin',dtype='<u2').reshape(512,128)
                after=np.fromfile(base/'after'/f'{kind}_{layer}.bin',dtype='<u2').reshape(512,128)
                record=dict(turn=turn,layer=layer,kind=kind,
                    prefix_exact=before[:r['before']].tobytes()==after[:r['before']].tobytes(),
                    inactive_exact=before[finish['cached']:].tobytes()==after[finish['cached']:].tobytes(),
                    finite=bool(np.isfinite((after.astype(np.uint32)<<16).view(np.float32)).all()))
                if not all(record[k] for k in ('prefix_exact','inactive_exact','finite')):
                    raise ValueError('persistent chat cache corrupted')
                storage.append(record)
    samples=[dict(zip(('turn','block','arm','sample','nanoseconds'),map(int,r[1:]))) for r in lines if r[0]=='sample']
    expected=Counter((t,b,a,s) for t in range(3) for b in range(4) for a in range(2) for s in range(5))
    if Counter(tuple(r[k] for k in ('turn','block','arm','sample')) for r in samples)!=expected:
        raise ValueError('incomplete paired chat sample census')
    if any(r['nanoseconds']<=0 for r in samples): raise ValueError('invalid chat timing')
    if (directory/'reset.bin').read_bytes()!=(directory/'turn0/cached.bin').read_bytes():
        raise ValueError('reset did not reproduce identical prompt logits')
    return dict(diagnostics=diagnostics,storage=storage,samples=samples,reset_logits_exact=True,
                raw_files={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args();root=repository_root();output=a.output.resolve()
    source=source_identity()
    if source['repository']['dirty']: raise ValueError('chat study requires clean source')
    prepared,_=verify_prepared(a.prepared.resolve());tables=ensure_prepared(download=False)
    output.mkdir(parents=True,exist_ok=False)
    builds={}
    for name,entry in [('terminal','src/llm_mojo/chat_cli.mojo'),('driver','tests/chat_driver.mojo')]:
        binary=output/name
        command=[environment_tool('mojo'),'build','-I','src',entry,'-o',str(binary)]
        result=subprocess.run(command,cwd=root,env=environment(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (output/f'{name}-build.log').write_text(result.stdout)
        result.check_returncode()
        builds[name]=dict(command=command,binary_sha256=sha(binary),source=source)
    conditions=conditions_snapshot();require_ac(conditions);require_nominal_thermal_state(conditions)
    directory=output/'driver-output'
    for turn in range(3):
        for stage in ('before','after'): (directory/f'turn{turn}'/stage).mkdir(parents=True)
    command=[str(output/'driver'),str(prepared),str(tables),str(directory)]
    process=subprocess.run(command,cwd=root,env=environment(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240)
    (output/'driver.log').write_text(process.stdout);process.check_returncode()
    driver=summarize_driver(directory,process.stdout)
    terminal=terminal_run(output/'terminal',prepared,tables,output/'terminal-output')
    after=conditions_snapshot();require_ac(after);require_nominal_thermal_state(after)
    if source!=source_identity(): raise ValueError('source changed during chat collection')
    report=dict(kind='native-chat-study-v1',source=source,builds=builds,
        environment=stable_environment(),conditions_before=conditions,conditions_after=after,
        dtype='BF16 storage, existing FP32 reductions',layout='model row-major; 24 distinct [capacity,128] K/V caches',
        prepared_manifest_sha256=sha(prepared/'manifest.json'),tokenizer_tables_sha256=sha(tables),
        numerical_policy='diagnostic; exact history/cache accounting and finite outputs required',
        driver=driver,driver_command=command,driver_stdout=process.stdout,terminal=terminal,
        measurement=dict(blocks=4,warmups=3,samples_per_arm=5,arms=['cached suffix','full history'],
            boundary='resident 24-layer forward plus final device synchronization; reset, captures, allocation, and greedy readback excluded',
            chunk_rows=256,capacity=512),complete=True)
    write(output/'result.json',report)
    print('Complete chat study: 3 numerical comparisons, 144 cache observations, 120 timings and 7 terminal turns.')

if __name__=='__main__': main()
