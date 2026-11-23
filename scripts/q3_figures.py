"""Redraw Q3 figures from results/q3. Question: Q3. How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q3_figures.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.figures import write_q3_figures

if __name__ == "__main__":
    write_q3_figures()
