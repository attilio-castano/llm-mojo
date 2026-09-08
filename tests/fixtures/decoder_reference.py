# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==1.26.4",
#   "torch==2.4.0",
#   "transformers==4.43.1",
# ]
# ///
"""Decoder references through the shared lock without editing frozen sources."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    directory = Path(__file__).resolve().parent/'decoder_layer'
    sys.path.insert(0, str(directory))
    runpy.run_path(str(directory/'generate.py'), run_name='__main__')
