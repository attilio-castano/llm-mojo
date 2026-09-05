"""Exercise every maintained measurement route, output gate, and invalid selector."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / 'build/operations-smoke'
    subprocess.run(['uv', 'run', '--locked', 'mojo', 'build', '-I', 'src',
                    'benchmarks/operations.mojo', '-o', str(target)], cwd=ROOT, check=True)
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


if __name__ == '__main__':
    main()
