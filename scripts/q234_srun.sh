#!/bin/bash
# Q2, then Q3, then Q4, on one GPU allocation.
# The plan files configs/q2_plan.json, q3_plan.json and q4_plan.json
# must already be on disk. This script does not write them.
# Submit from the repository root:
#   sbatch --partition=medium --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=08:00:00 \
#     --job-name=q234_m26csa003 --output=results/q234.log --error=results/q234.log \
#     scripts/q234_srun.sh
echo "job start $(date) host $(hostname)"
set -e
module load python/3.10.pytorch
echo "module loaded"
cd /scratch/m26csa003/ML_assignment_3
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
nvidia-smi -L || true
.venv/bin/python scripts/q2_train.py
.venv/bin/python scripts/q3_train.py
.venv/bin/python scripts/q4_train.py
