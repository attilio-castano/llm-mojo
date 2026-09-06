#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==1.26.4", "torch==2.4.0", "transformers==4.43.1"]
# ///
"""Pinned Qwen compatibility fixtures plus an independent numerical diagnostic.

No checkpoint downloads. All arrays remain in ignored build/oracle_data/.
FP64 dot products deliberately do not reproduce GPU reduction/work ownership.
"""
import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import transformers
from upstream import (UpstreamAttention, errors, provenance, reference_regressions,
                      score_boundary_diagnostic, compare_attention_backends)

ROOT = Path(__file__).resolve().parents[3] / 'build/oracle_data/attention_sublayer'
BASE_CASES = [(2,1,4,7,17),(4,2,4,9,37)]
BASE_CASES += [(14,2,64,t,seed) for t,seed in ((1,17),(7,37),(33,53),(65,71),(257,17),(4096,53))]
# A second full-context seed is fixed before checking the revised composition.
BASE_CASES += [(14,2,64,4096,103)]
# Diagnostic calibration fixed before observing these additional GPU results.
CALIBRATION_CASES = [(14,2,64,4096,seed) for seed in (149,211,307,401)]
# Fixed after approval of the calibrated budget, before these GPU results.
HOLDOUT_CASES = [(14,2,64,4096,seed) for seed in (509,887)]
CASES = BASE_CASES + CALIBRATION_CASES + HOLDOUT_CASES
TOLERANCES = {'normalized':.0078125,'raw_query':.0078125,'raw_key':.0078125,
              'raw_value':.0078125,'query':.0078125,'rotated_key':.0078125,
              'attention':.03125,'projected':.03125,'output':.03125}
CONTRACT = dict(
    authority='pinned upstream CPU BF16 eager Qwen',
    operation_gates='identical upstream inputs; approved GQA compatibility budget 0.03125; other stage budgets unchanged',
    exact_operations=['rotary application with identical tables', 'residual addition',
                      'cache append and prefix preservation'],
    composition_gates=['projected', 'output'],
    composition_intermediates='reported against upstream; operation gates isolate local errors',
    independent_numpy='diagnostic; disagreement is investigated, not automatically a Mojo defect',
    rotary_tables='explicit upstream-derived BF16 input; construction outside enqueue',
)


def bf16(x):
    bits = np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((bits+0x7fff+((bits>>16)&1))&0xffff0000).view(np.float32)


def recipe(n, seed, scale):
    i = np.arange(n,dtype=np.uint32)
    z = i*np.uint32(2654435761)+np.uint32(seed*97)
    z ^= z>>16
    return bf16(((z&255).astype(np.int32)-128)/scale)


def fixtures(nq,nk,d,t,seed):
    h,k = nq*d,nk*d
    x = recipe(t*h,seed,64).reshape(t,h)
    w = recipe((h+2*k)*h,seed+3,4096 if h==896 else 256).reshape(h+2*k,h)
    bias = recipe(h+2*k,seed+7,1024)
    wo = recipe(h*h,seed+11,4096 if h==896 else 256).reshape(h,h)
    weight = bf16(1+recipe(h,seed+13,4096))
    normal = bf16(bf16(x.astype(np.float64)/np.sqrt((x.astype(np.float64)**2).mean(-1,keepdims=True)+1e-6))*weight)
    projected = bf16(normal.astype(np.float64) @ w.astype(np.float64).T+bias)
    rawq,rawk,rawv = np.split(projected,[h,h+k],axis=1)
    freq = (np.float32(1)/np.float32(1000000)**(np.arange(0,d,2,dtype=np.float32)/d))
    angles = np.arange(t,dtype=np.float32)[:,None]*freq[None,:]
    c,s = bf16(np.cos(angles)),bf16(np.sin(angles))
    def rotate(a,heads):
        a = a.reshape(t,heads,d)
        left,right = a[...,:d//2],a[...,d//2:]
        return np.concatenate((bf16(bf16(left*c[:,None,:])-bf16(right*s[:,None,:])),
                               bf16(bf16(right*c[:,None,:])+bf16(left*s[:,None,:]))),axis=-1)
    q,keys,values = rotate(rawq,nq),rotate(rawk,nk),rawv.reshape(t,nk,d)
    attention = np.empty((t,nq,d),np.float32)
    for head in range(nq):
        for start in range(0,t,32):
            stop = min(start+32,t)
            scores = bf16(q[start:stop,head].astype(np.float64) @ keys[:,head//(nq//nk)].astype(np.float64).T / np.sqrt(d)).astype(np.float64)
            scores[np.arange(t)[None,:]>np.arange(start,stop)[:,None]] = -np.inf
            prob = np.exp(scores-scores.max(-1,keepdims=True))
            prob /= prob.sum(-1,keepdims=True)
            attention[start:stop,head] = bf16(bf16(prob).astype(np.float64) @ values[:,head//(nq//nk)].astype(np.float64))
    branch = bf16(attention.reshape(t,h).astype(np.float64) @ wo.astype(np.float64).T)
    return dict(input=x,weight=w,bias=bias,output_weight=wo,norm_weight=weight,
                normalized=normal,raw_query=rawq,raw_key=rawk,raw_value=rawv,
                query=q,rotated_key=keys,attention=attention,projected=branch,output=bf16(x+branch))


def crosscheck(arrays,nq,nk,d,t):
    """Observe actual upstream full/chunked execution, including its KV cache."""
    full = UpstreamAttention(arrays,nq,nk,d).run(arrays['input'])
    diagnostic = {name:errors(arrays[name].reshape(full[name].shape),full[name],tol)
                  for name,tol in TOLERANCES.items()}
    chunks = [t-18,17,1] if t>65 else [1]*(t-1)+[1]
    chunked = UpstreamAttention(arrays,nq,nk,d)
    p = 0
    checks = []
    for r in chunks:
        result = chunked.run(arrays['input'][p:p+r])
        stages = {name:errors(result[name],full[name][p:p+r],tol)
                  for name,tol in TOLERANCES.items()}
        # Reduction association can differ with query-row count even upstream.
        for name in ('projected','output'):
            if stages[name]['failed']:
                raise RuntimeError(f'upstream full/chunked {name} mismatch at {p}')
        if not np.array_equal(result['cache_key'][p:],result['rotated_key']):
            raise RuntimeError('upstream capture disagrees with its rotated-key cache')
        if not np.array_equal(result['cache_value'][p:],result['raw_value'].reshape(r,nk,d)):
            raise RuntimeError('upstream capture disagrees with its value cache')
        checks.append(dict(start=p,rows=r,stages=stages))
        p += r
    return full,diagnostic,checks


def main(mode='primary', compare_backends=False):
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    regressions=reference_regressions()
    hashes={}
    checks={}
    chunk_checks={}
    boundaries={}
    backend_checks={}
    cases={'primary':CASES,'calibrate':CALIBRATION_CASES,'holdout':HOLDOUT_CASES}[mode]
    offset={'primary':0,'calibrate':len(BASE_CASES),
            'holdout':len(BASE_CASES)+len(CALIBRATION_CASES)}[mode]
    for case,(nq,nk,d,t,seed) in enumerate(cases,start=offset):
        arrays=fixtures(nq,nk,d,t,seed)
        upstream,checks[case],chunk_checks[case]=crosscheck(arrays,nq,nk,d,t)
        boundary_cases={8:(350,4,303),14:(667,0,665)}
        if case in boundary_cases:
            boundaries[case]=score_boundary_diagnostic(
                upstream,*boundary_cases[case],TOLERANCES['attention'])
            if compare_backends:
                backend_checks[case]=compare_attention_backends(
                    arrays,upstream,nq,nk,d,TOLERANCES)
        outputs = {f'{case}_{name}':a for name,a in arrays.items()}
        outputs.update({f'upstream_{case}_{name}':a for name,a in upstream.items()})
        for name,a in outputs.items():
            if not np.isfinite(a).all():raise RuntimeError('nonfinite fixture')
            path=ROOT/f'{name}.npy'
            np.save(path,a.astype(np.float32),allow_pickle=False)
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        print('sublayer fixture',case,'shape',nq,nk,d,t,'generated',flush=True)
    record=dict(model='Qwen/Qwen2.5-0.5B-Instruct',model_revision='7ae557604adf67be50417f59c2c2f167def9a775',
                weights='synthetic deterministic uint32 recipe; no checkpoint weights',
                oracle='Primary: pinned upstream CPU BF16 eager modules, every case. Diagnostic: independent NumPy FP64 operations with explicit BF16 boundaries.',
                upstream_reference=provenance(),
                upstream_contract={k:v for k,v in provenance().items() if k not in ('platform','machine')},
                numerical_contract=CONTRACT,reference_regressions=regressions,
                score_boundary_diagnostics=boundaries,
                upstream_eager_vs_sdpa=backend_checks,
                numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,
                cases=cases,case_offset=offset,atol=TOLERANCES,rtol=TOLERANCES,numpy_vs_upstream=checks,
                upstream_full_vs_chunked=chunk_checks,array_sha256=hashes,
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                capture_sha256=hashlib.sha256(Path(__file__).with_name('upstream.py').read_bytes()).hexdigest())
    name='manifest.json' if mode=='primary' else mode+'_manifest.json'
    (ROOT/name).write_text(json.dumps(record,indent=2)+'\n')
    print('all upstream sublayer fixtures and independent diagnostics generated')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--calibrate',action='store_true',help='generate the fixed additional diagnostic seeds')
    group.add_argument('--holdout',action='store_true',help='generate the two fixed validation seeds')
    parser.add_argument('--compare-backends',action='store_true',
                        help='also diagnose official eager versus math SDPA on the midpoint cases')
    args=parser.parse_args()
    main('calibrate' if args.calibrate else 'holdout' if args.holdout else 'primary',
         args.compare_backends)
