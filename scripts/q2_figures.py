"""Redraw Q2 figures from results/q2. Question: Q2. How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q2_figures.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q2.figures import write_q2_figures

if __name__ == "__main__":
    write_q2_figures()
