"""
CPU checks for Q2, Q3 and Q4. None of these numbers are reported.

Question: smoke test before the GPU run.
How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q234_smoke.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.models import build_model
from src.common.train import train_classifier
from src.common.utils import set_seed
from src.q1.pipeline import load_fairness
from src.q2.pipeline import load_plan as load_q2
from src.q2.transforms import apply_spatial_permutation, make_spatial_permutation, to_grayscale
from src.q3.models import ConvAutoencoder
from src.q3.pipeline import assert_architecture, load_plan as load_q3
from src.q3.probe import linear_probe
from src.q3.train_ae import train_autoencoder
from src.q4.analysis import (
    choose_k,
    choose_svm,
    fit_svc,
    ovo_decision,
    pair_column,
    pairwise_support_mask,
)
from src.q4.pipeline import load_plan as load_q4


def check_q2():
    root = Path(__file__).resolve().parents[1]
    plan, _digest, _path = load_q2(root)
    q1, q1_hash, _q1_path = load_fairness(root)
    if q1_hash != plan["q1_fairness_sha256"]:
        raise AssertionError("Q2 plan does not point at the current Q1 fairness file.")
    perm = make_spatial_permutation(26003)
    images = torch.rand(4, 3, 32, 32)
    shuffled = apply_spatial_permutation(images, perm)
    # A spatial permutation keeps the sorted values of each channel.
    for channel in range(3):
        before = torch.sort(images[:, channel].reshape(4, -1), dim=1).values
        after = torch.sort(shuffled[:, channel].reshape(4, -1), dim=1).values
        if not torch.allclose(before, after):
            raise AssertionError("Permutation changed pixel values.")
    gray = to_grayscale(images)
    if not torch.allclose(gray[:, 0], gray[:, 1]) or not torch.allclose(gray[:, 0], gray[:, 2]):
        raise AssertionError("Grayscale channels are not copies.")
    spec = q1["models"]["mlp_matched"]
    set_seed(26003)
    model = build_model(spec)
    images = torch.rand(64, 3, 32, 32)
    labels = torch.randint(0, 10, (64,))
    result = train_classifier(
        model,
        apply_spatial_permutation(images[:48], perm),
        labels[:48],
        to_grayscale(images[48:]),
        labels[48:],
        {"lr": 1e-3, "weight_decay": 0.0, "epochs": 1, "batch_size": 16, "use_amp": False, "channels_last": False},
    )
    if result["failed"] or not np.isfinite(result["best_val_accuracy"]):
        raise AssertionError("Q2 smoke training failed.")
    print("q2 smoke ok")


def check_q3():
    root = Path(__file__).resolve().parents[1]
    plan, _digest, _path = load_q3(root)
    assert_architecture(plan)
    set_seed(26003)
    model = ConvAutoencoder(32)
    clean = torch.rand(32, 3, 32, 32)
    result = train_autoencoder(
        model,
        clean[:24],
        clean[24:],
        clean[24:],
        {"lr": 1e-3, "weight_decay": 0.0, "epochs": 1, "batch_size": 8, "use_amp": False, "channels_last": False, "noise_sigma": 0.25},
    )
    if result["failed"] or result["best_val_mse"] is None:
        raise AssertionError("Q3 smoke autoencoder failed.")
    code = model.encode(clean[:4])
    if tuple(code.shape) != (4, 32):
        raise AssertionError(f"Code shape {tuple(code.shape)}.")
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(40, 8))
    val_x = rng.normal(size=(20, 8))
    test_x = rng.normal(size=(20, 8))
    train_y = rng.integers(0, 3, size=40)
    val_y = rng.integers(0, 3, size=20)
    test_y = rng.integers(0, 3, size=20)
    _rows, best, test_accuracy = linear_probe(train_x, train_y, val_x, val_y, test_x, test_y, [0.1, 1.0])
    if best["C"] not in (0.1, 1.0) or not 0.0 <= test_accuracy <= 1.0:
        raise AssertionError("Probe smoke failed.")
    print("q3 smoke ok")


def check_q4():
    root = Path(__file__).resolve().parents[1]
    plan, _digest, _path = load_q4(root)
    if plan["subset_per_class"] * 10 != plan["subset_size"]:
        raise AssertionError("Subset size does not match the per-class count.")
    if max(plan["C_values"]) / min(plan["C_values"]) < 1000:
        raise AssertionError("C values must span at least three orders of magnitude.")
    if "rbf" not in plan["kernels"] or len(plan["kernels"]) < 3:
        raise AssertionError("Need at least three kernels, including RBF.")
    rng = np.random.default_rng(0)
    features = rng.normal(size=(90, 4))
    labels = np.repeat(np.arange(3), 30)
    features[:30] += np.array([3.0, 0, 0, 0])
    features[30:60] += np.array([0, 3.0, 0, 0])
    features[60:] += np.array([0, 0, 3.0, 0])
    clf = fit_svc(features, labels, "linear", 1.0, 0)
    scores = ovo_decision(clf, features)
    if scores.shape != (90, 3):
        raise AssertionError(f"OvO shape {scores.shape}.")
    if pair_column(0, 1, 3) != 0 or pair_column(1, 2, 3) != 2:
        raise AssertionError("Pair column index is wrong.")
    binary = SVC(C=1.0, kernel="linear").fit(features[labels < 2], labels[labels < 2])
    column = scores[:, 0]
    binary_scores = binary.decision_function(features)
    correlation = np.corrcoef(column[labels < 2], binary_scores[labels < 2])[0, 1]
    if abs(correlation) < 0.99:
        raise AssertionError(f"OvO column does not match the binary SVM (corr {correlation}).")
    mask = pairwise_support_mask(clf, 0, 1)
    got = set(clf.support_[mask].tolist())
    subset_index = np.flatnonzero(labels < 2)
    expected = set(subset_index[binary.support_].tolist())
    if got != expected:
        raise AssertionError(f"Pair support vectors {sorted(got)} != binary {sorted(expected)}.")
    starts = np.cumsum(np.concatenate([[0], clf.n_support_]))
    strengths = np.zeros(mask.shape[0], dtype=np.float64)
    for owner, other in ((0, 1), (1, 0)):
        row = other if other < owner else other - 1
        sl = slice(int(starts[owner]), int(starts[owner + 1]))
        strengths[sl] = np.abs(np.asarray(clf.dual_coef_[row, sl], dtype=np.float64)).reshape(-1)
    if not np.any(strengths[mask] > 0):
        raise AssertionError("Support-vector dual coefficients were all zero.")
    rows = [
        {"kernel": "linear", "C": 1.0, "val_accuracy": 0.5, "support_fraction": 0.4},
        {"kernel": "rbf", "C": 1.0, "val_accuracy": 0.5, "support_fraction": 0.2},
        {"kernel": "rbf", "C": 10.0, "val_accuracy": 0.4, "support_fraction": 0.1},
    ]
    chosen = choose_svm(rows)
    if chosen["kernel"] != "rbf" or chosen["C"] != 1.0:
        raise AssertionError(f"SVM tie break chose {chosen}.")
    k_rows = [{"k": 8, "val_nmi": 0.2}, {"k": 12, "val_nmi": 0.2}, {"k": 10, "val_nmi": 0.2}]
    if int(choose_k(k_rows)["k"]) != 10:
        raise AssertionError("Cluster tie break should prefer k = 10.")
    print("q4 smoke ok")


if __name__ == "__main__":
    check_q2()
    check_q3()
    check_q4()
    print("smoke passed")
