#!/bin/bash
# Q1 on one GPU. The fairness file must already be on disk.
# Submit from the repository root with:
#   srun --partition=mtech --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=03:00:00 bash scripts/q1_srun.sh
echo "job start $(date) host $(hostname)"
module load python/3.10.pytorch
echo "module loaded"
cd /scratch/m26csa003/ML_assignment_3
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
nvidia-smi -L || true
.venv/bin/python scripts/q1_train.py
