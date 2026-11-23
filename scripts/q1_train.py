"""
Run the full Q1 probe, the three-seed comparison, the test evaluation and Q1b.

Question: Q1.
How to run, from the repository root, on a GPU node:

    bash scripts/q1_srun.sh

The plan in configs/q1_fairness.json must already exist. This script does not write it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.q1.pipeline import main


if __name__ == "__main__":
    main()
