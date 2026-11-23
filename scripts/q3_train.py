"""
Entry point for the Q3 GPU run.

Question: Q3.
How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q3_train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.pipeline import main

if __name__ == "__main__":
    main()
