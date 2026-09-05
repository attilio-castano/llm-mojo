"""Exercise every maintained measurement route, output gate, and invalid selector."""
import os
import subprocess

from .._repository import repository_root


def main():
    target = repository_root() / 'build/operations-smoke'
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['uv', 'run', '--locked', 'mojo', 'build', '-I', 'src',
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
    subprocess.run(['uv','run','--locked','mojo','build','-I','src',
                    'src/llm_mojo/benchmarks/attention_prefill.mojo','-o',str(target)],
                   cwd=repository_root(),check=True)
    for variant in range(11):
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
    print('prefill all eleven measurement routes passed in both modes',flush=True)


if __name__ == '__main__':
    main()
