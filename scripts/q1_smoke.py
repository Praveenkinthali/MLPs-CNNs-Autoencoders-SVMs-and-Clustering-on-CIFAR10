"""
One-epoch check that Q1 models, the noise mask and the training step run.

Question: Q1. This is a bug check, not a result. It writes under results/q1/smoke
and those numbers must not go in the report.
How to run, from the repository root:

    PYTHONNOUSERSITE=1 .venv/bin/python scripts/q1_smoke.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.common.data import apply_train_normalisation, load_experiment_arrays
from src.common.models import build_model, count_parameters
from src.common.train import train_classifier
from src.common.utils import set_seed
from src.q1.noise import corrupt_training_labels
from src.q1.pipeline import assert_parameter_counts, load_fairness


def main():
    fairness, digest, path = load_fairness(ROOT)
    assert_parameter_counts(fairness)
    print(f"parameter counts match {path.name} ({digest[:12]})")

    bundle = load_experiment_arrays(root=ROOT, device="cpu")
    labels = bundle["train_labels"].numpy()
    for fraction, expected in ((0.0, 0), (0.2, 900), (0.5, 2250), (1.0, 4500)):
        noisy, mask = corrupt_training_labels(labels, fraction, 26003)
        changed = noisy != labels
        if not (changed == mask).all():
            raise AssertionError("The noise mask and the changed labels disagree.")
        if mask[changed].size and (noisy[mask] == labels[mask]).any():
            raise AssertionError("A corrupted label stayed equal to the true label.")
        for class_id in range(10):
            count = int(((labels == class_id) & mask).sum())
            if count != expected:
                raise AssertionError(f"fraction {fraction} class {class_id} corrupted {count}, expected {expected}")
    print("label noise masks have the planned per-class counts")

    mean, std = bundle["mean"], bundle["std"]
    images = apply_train_normalisation(bundle["train_images"], mean, std)[:256]
    train_labels = bundle["train_labels"][:256]
    val_images = apply_train_normalisation(bundle["val_images"], mean, std)[:64]
    val_labels = bundle["val_labels"][:64]
    set_seed(26003)
    for model_name in ("mlp_matched", "cnn_matched", "mlp_wide", "mlp_regularised"):
        spec = fairness["models"][model_name]
        model = build_model(spec)
        result = train_classifier(
            model,
            images,
            train_labels,
            val_images,
            val_labels,
            {
                "lr": 1e-3,
                "weight_decay": spec["weight_decay"],
                "epochs": 1,
                "batch_size": 32,
                "use_amp": False,
                "channels_last": False,
                "batch_transform": None,
            },
        )
        if result["failed"] or result["best_state"] is None:
            raise AssertionError(f"{model_name} smoke run failed")
        accuracy = result["history"][0]["val_accuracy"]
        if not 0.0 <= accuracy <= 1.0:
            raise AssertionError(f"{model_name} validation accuracy {accuracy}")
        print(f"  {model_name:20} params {count_parameters(model):8d}  one-epoch val {accuracy:.3f}")
    print("Q1 smoke test passed")


if __name__ == "__main__":
    main()
