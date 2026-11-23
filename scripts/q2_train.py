"""
Entry point for the Q2 GPU run.

Question: Q2.
How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q2_train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q2.pipeline import main

if __name__ == "__main__":
    main()
