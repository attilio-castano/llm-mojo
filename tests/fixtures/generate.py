# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
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
    operations = ("rms_norm", "linear", "rope", "attention")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operations", nargs="*", help="Oracle names; defaults to all four.")
    args = parser.parse_args()
    for operation in args.operations or operations:
        if operation not in operations:
            parser.error(f"unknown oracle {operation!r}; choose from {', '.join(operations)}")
    for operation in args.operations or operations:
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / operation / "generate.py")],
            check=True,
        )


if __name__ == "__main__":
    main()
