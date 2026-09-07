# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "torch==2.4.0",
#   "transformers==4.43.1",
# ]
# ///
"""Run the independent Torch oracles in one shared, locked script environment."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    defaults = ("rms_norm", "linear", "rope", "attention")
    generators = {name: f"{name}/generate.py" for name in defaults}
    generators.update(attention_sublayer="attention_sublayer/generate.py",
                      attention_precision="attention_sublayer/precision.py",
                      attention_checkpoint="attention_sublayer/checkpoint.py",
                      mlp="mlp/generate.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operations", nargs="*", help="Oracle names; defaults to the four standalone operations.")
    # Arguments after -- belong to one explicitly selected generator. This keeps
    # checkpoint downloads opt-in and lets the numerical diagnostics retain their CLI.
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    extra = argv[split + 1:]
    selected = args.operations or defaults
    if extra and len(selected) != 1:
        parser.error("generator arguments require exactly one oracle")
    for operation in selected:
        if operation not in generators:
            parser.error(f"unknown oracle {operation!r}; choose from {', '.join(generators)}")
    for operation in selected:
        subprocess.run([sys.executable, str(Path(__file__).parent / generators[operation]), *extra],
                       check=True)


if __name__ == "__main__":
    main()
