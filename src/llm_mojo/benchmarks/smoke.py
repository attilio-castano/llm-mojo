"""Exercise every maintained measurement route, output gate, and invalid selector."""
from .attention_sublayer_contract import PROFILE_CONTROLS

import os
import subprocess

from .._repository import environment_tool, repository_root
from .attention_prefill_contract import VARIANTS as PREFILL_VARIANTS


def main():
    target = repository_root() / 'build/operations-smoke'
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([environment_tool('mojo'), 'build', '-I', 'src',
                    'src/llm_mojo/benchmarks/operations.mojo', '-o', str(target)], cwd=repository_root(), check=True)
    env = {**os.environ, 'MODULAR_DEBUG': 'device-sync-mode'}
    for operation, variants in [('linear', range(7)), ('rms_norm', range(2)), ('rope', range(1))]:
        for variant in variants:
            for rows, layers in [(1, 1), (16 if variant not in (0, 2) else 1, 24)]:
                control = 1 if operation == 'linear' and rows > 1 else 0
                command = list(map(str, [target, operation, rows, layers, variant, control, 1, 53, 'bench', 1, 0]))
                result = subprocess.run(command, text=True, capture_output=True, env=env)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                if 'correctness: passed' not in result.stdout or not result.stdout.rstrip().endswith('BENCHMARK_COMPLETE'):
                    raise RuntimeError(result.stdout + result.stderr)
        invalid = subprocess.run(list(map(str, [target, operation, 1, 1, 99, 0, 0, 53, 'bench', 1, 0])),
                                 capture_output=True, env=env)
        if invalid.returncode == 0:
            raise RuntimeError('invalid route accepted')
        print(operation, 'all measurement routes passed', flush=True)
    target = repository_root() / 'build/prefill-smoke'
    subprocess.run([environment_tool('mojo'),'build','-I','src',
                    'src/llm_mojo/benchmarks/attention_prefill.mojo','-o',str(target)],
                   cwd=repository_root(),check=True)
    for variant in PREFILL_VARIANTS:
        for layers in (1,24):
            result = subprocess.run(list(map(str,[target,7,33,layers,variant,0,1,53,'bench',1,0])),
                                    text=True,capture_output=True,env=env,check=True)
            if (f'variants: 0 {variant} candidate-first: 1' not in result.stdout or
                f'SAMPLE candidate {variant} 0 ' not in result.stdout or
                'api: metal' not in result.stdout or 'correctness: passed' not in result.stdout or
                not result.stdout.rstrip().endswith('BENCHMARK_COMPLETE')):
                raise RuntimeError('prefill measurement route or completion mismatch')
    for rows,tokens,variant in [(34,33,0),(7,4097,0),(7,33,99)]:
        result = subprocess.run(list(map(str,[target,rows,tokens,1,variant,0,1,53,'bench',1,0])),
                                capture_output=True,env=env)
        if result.returncode == 0:
            raise RuntimeError('invalid prefill benchmark accepted')
    print('prefill all',len(PREFILL_VARIANTS),'measurement routes passed in both modes',flush=True)
    target = repository_root() / 'build/attention-sublayer-smoke'
    subprocess.run([environment_tool('mojo'),'build','-I','src',
                    'src/llm_mojo/benchmarks/attention_sublayer.mojo','-o',str(target)],
                   cwd=repository_root(),check=True)
    for r,t in ((1,64),(7,33),(33,33)):
        for layers in (1,24):
            for variant in ((3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19) if r == 1 else (3,4,7,8,9,10,11,12,13,14,15,16,17,18,19)):
                control = PROFILE_CONTROLS[variant]
                result = subprocess.run(list(map(str,[target,r,t,layers,variant,control,1,53,'bench',1,0])),
                                        cwd=repository_root(),text=True,capture_output=True,env=env,check=True)
                if (f'query rows: {r}' not in result.stdout or
                    f'variants: {control} {variant} candidate-first: 1' not in result.stdout or
                    f'SAMPLE candidate {variant} 0 ' not in result.stdout or
                    'api: metal' not in result.stdout or 'correctness: passed' not in result.stdout or
                    not result.stdout.rstrip().endswith('BENCHMARK_COMPLETE')):
                    raise RuntimeError('attention sublayer measurement identity or completion mismatch')
    for r,t,variant,seed in ((34,33,3,53),(7,4097,3,53),(7,33,0,53),(7,33,3,17),
                           (7,33,5,53),(7,33,6,53),(1,64,7,53),(7,33,7,53)):
        result = subprocess.run(list(map(str,[target,r,t,1,variant,3,1,seed,'bench',1,0])),
                                cwd=repository_root(),capture_output=True,env=env)
        if result.returncode == 0:
            raise RuntimeError('invalid attention sublayer benchmark accepted')
    # Exercise the combined-baseline comparison and both sides of the fixed
    # projection policy on the actual timed route, before recording samples.
    for r,t in ((4,64),(15,64),(16,64),(17,64)):
        for layers in (1,24):
            result = subprocess.run(list(map(str,[target,r,t,layers,9,3,0,53,'bench',1,0])),
                                    cwd=repository_root(),text=True,capture_output=True,env=env,check=True)
            if 'correctness: passed' not in result.stdout or not result.stdout.rstrip().endswith('BENCHMARK_COMPLETE'):
                raise RuntimeError('integrated attention benchmark boundary check failed')
    for r,t in ((15,64),(16,64),(17,64),(64,4096)):
        for layers in (1,24):
            for variant in (10,11,12,13,18):
                result = subprocess.run(list(map(str,[target,r,t,layers,variant,9,0,53,'bench',1,0])),
                                        cwd=repository_root(),text=True,capture_output=True,env=env,check=True)
                if 'correctness: passed' not in result.stdout or not result.stdout.rstrip().endswith('BENCHMARK_COMPLETE'):
                    raise RuntimeError('GQA parallelism benchmark boundary check failed')
    # Both actual comparison controls, self-pairs and the projection threshold.
    for r,t in ((1,64),(15,64),(16,64),(17,64),(64,4096)):
        for layers in (1,24):
            for control in (13,18):
                for variant in (control,19):
                    result=subprocess.run(list(map(str,[target,r,t,layers,variant,control,0,53,'bench',1,0])),
                        cwd=repository_root(),text=True,capture_output=True,env=env,check=True)
                    if ('correctness: passed' not in result.stdout
                        or f'SAMPLE candidate {variant} 0 ' not in result.stdout
                        or not result.stdout.rstrip().endswith('BENCHMARK_COMPLETE')):
                        raise RuntimeError('split8 plus projections benchmark boundary check failed')
    for mode, variants in [('wo',(9,14,15)),('buffered',(9,))]:
        for variant in variants:
            for layers in (1,24):
                result=subprocess.run(list(map(str,[target,17,64,layers,variant,9,1,53,mode,1,0])),
                    cwd=repository_root(),text=True,capture_output=True,env=env,check=True)
                measurement='isolated_wo' if mode=='wo' else 'whole_attention_buffered'
                if (f'measurement: {measurement}' not in result.stdout
                    or 'correctness: passed' not in result.stdout
                    or not result.stdout.rstrip().endswith('BENCHMARK_COMPLETE')):
                    raise RuntimeError('contained projection/timing boundary check failed')
    print('attention sublayer FP32, projections, GQA parallelism and decode measurement routes passed in both modes',flush=True)
    target = repository_root() / 'build/mlp-smoke'
    subprocess.run([environment_tool('mojo'),'build','-I','src',
                    'src/llm_mojo/benchmarks/mlp.mojo','-o',str(target)],cwd=repository_root(),check=True)
    from .mlp_contract import VARIANTS
    routes = [(r,l,'bench',v) for v in sorted(VARIANTS) for r,l in ((1,1),(17,24))]
    routes += [(17,1,f'stage{i}',0) for i in range(7)]
    routes += [(17,l,f'stage{s}',v) for s,vs in ((1,(0,1,2,3)),(2,(0,1,2,3)),(5,(0,4,5,6))) for v in vs for l in (1,24)]
    for rows,layers,mode,variant in routes:
        result = subprocess.run(list(map(str,[target,rows,layers,variant,0,variant%2,1601,mode,1,0])),
                                cwd=repository_root(),capture_output=True,text=True,env=env,check=True)
        boundary = 'whole_mlp' if mode=='bench' else 'mlp_stage_'+mode[5:]
        if (f'variants: 0 {variant} candidate-first: {variant%2}' not in result.stdout
            or 'correctness: passed' not in result.stdout or f'measurement: {boundary}' not in result.stdout
            or f'SAMPLE candidate {variant} 0 ' not in result.stdout
            or 'SAMPLE control 0 0 ' not in result.stdout
            or not result.stdout.rstrip().endswith('BENCHMARK_COMPLETE')):
            raise RuntimeError('MLP benchmark route or output check failed')
    # Exercise the reader itself on a non-self pair in both orders. The old
    # candidate/control header reversal escaped substring-only route checks.
    from .study import parse_output, REPETITIONS
    for candidate,control,mode,measurement in [(1,0,'stage1','mlp_stage_1'),(7,2,'bench','whole_mlp')]:
        for first in (0,1):
            result = subprocess.run(list(map(str,[target,1,1,candidate,control,first,1601,mode,REPETITIONS,0])),
                                    cwd=repository_root(),capture_output=True,text=True,env=env,check=True)
            parse_output(result.stdout,control,candidate,first,rows=1,layers=1,seed=1601,
                         operation='mlp',measurement=measurement)
    for rows,layers,variant,seed,mode in [(0,1,0,1601,'bench'),(4097,1,0,1601,'bench'),
        (1,2,0,1601,'bench'),(1,1,8,1601,'bench'),(1,1,0,53,'bench'),(1,1,0,1601,'stage7'),(1,24,0,1601,'stage0')]:
        result = subprocess.run(list(map(str,[target,rows,layers,variant,0,0,seed,mode,1,0])),
                                cwd=repository_root(),capture_output=True,env=env)
        if result.returncode == 0:
            raise RuntimeError('invalid MLP benchmark request accepted')
    print('MLP projection variants in both modes, all seven stages and isolated projection ring routes passed',flush=True)



if __name__ == '__main__':
    main()
