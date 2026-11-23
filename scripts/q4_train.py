"""
Entry point for Q4. Feature extraction uses a GPU if the feature file is absent.

Question: Q4.
How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q4_train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q4.pipeline import main

if __name__ == "__main__":
    main()
