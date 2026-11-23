"""Redraw Q4 figures from results/q4. Question: Q4. How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q4_figures.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q4.figures import write_q4_figures

if __name__ == "__main__":
    write_q4_figures()
