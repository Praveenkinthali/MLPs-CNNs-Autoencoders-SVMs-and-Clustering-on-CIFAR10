"""
Regenerate Q1 figures and tables from the saved logs.

Question: Q1.
How to run, from the repository root:

    PYTHONNOUSERSITE=1 .venv/bin/python scripts/q1_figures.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.figures import write_q1_figures


if __name__ == "__main__":
    write_q1_figures(ROOT)
