"""
Download CIFAR-10, save the stratified split, and write the dataset figures.

Question: shared data setup, before Q1. Every later question imports this split.
How to run, from the repository root:

    module load python/3.10.pytorch
    PYTHONNOUSERSITE=1 .venv/bin/python scripts/prepare_data.py
"""

import json
import platform
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torchvision

from src.common.data import SEED, apply_train_normalisation, load_experiment_arrays, prepare_dataset
from src.common.utils import get_device, set_seed
from src.common.viz import write_dataset_figures


def check_libraries():
    """The cluster image ships a torchvision that does not match its torch."""
    if not torchvision.__version__.startswith("0.16."):
        raise RuntimeError(
            f"Need torchvision 0.16.x to match torch 2.1.2, found {torchvision.__version__} "
            f"at {torchvision.__file__}. Use the project .venv described in README.md."
        )


def write_environment_note(path, device):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    note = {
        "question": "shared data setup, before Q1",
        "seed": SEED,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "hostname": socket.gethostname(),
        "nondeterminism": (
            "The split and the channel statistics depend only on the CIFAR-10 "
            "files and numpy Generator(26003). They do not depend on the GPU. "
            "cuDNN nondeterminism does not apply to this phase."
        ),
    }
    path.write_text(json.dumps(note, indent=2) + "\n")


def check_loader(arrays):
    """Confirm the loader serves the same split the checks already accepted."""
    if tuple(arrays["train_images"].shape) != (45000, 3, 32, 32):
        raise AssertionError(arrays["train_images"].shape)
    if tuple(arrays["val_images"].shape) != (5000, 3, 32, 32):
        raise AssertionError(arrays["val_images"].shape)
    if tuple(arrays["test_images"].shape) != (10000, 3, 32, 32):
        raise AssertionError(arrays["test_images"].shape)
    if arrays["train_labels"].bincount().tolist() != [4500] * 10:
        raise AssertionError("loader train labels are not balanced")
    if arrays["val_labels"].bincount().tolist() != [500] * 10:
        raise AssertionError("loader val labels are not balanced")
    if arrays["test_labels"].bincount().tolist() != [1000] * 10:
        raise AssertionError("loader test labels are not balanced")
    # The float64 mean of the float32 tensors must match the saved constants.
    got = arrays["train_images"].double().mean(dim=(0, 2, 3))
    saved = arrays["mean"].double().cpu()
    if not torch.allclose(got.cpu(), saved, atol=1e-6):
        raise AssertionError(f"loader mean {got.tolist()} != saved {saved.tolist()}")
    # Normalisation is defined. A full-set check of the mean is enough here.
    normalised = apply_train_normalisation(arrays["train_images"], arrays["mean"], arrays["std"])
    centred = normalised.double().mean(dim=(0, 2, 3))
    if not torch.allclose(centred.cpu(), torch.zeros(3, dtype=torch.float64), atol=1e-5):
        raise AssertionError(f"normalised train mean is {centred.tolist()}, expected about 0")
    overlap = set(arrays["train_idx"].tolist()) & set(arrays["val_idx"].tolist())
    if overlap:
        raise AssertionError(f"loader still has {len(overlap)} overlapping indices")


def main():
    check_libraries()
    # Recorded for later training. The split uses its own Generator(SEED).
    set_seed(SEED)
    device = get_device()
    prepared = prepare_dataset(ROOT)
    write_dataset_figures(prepared, ROOT / "figures", ROOT / "results" / "phase1")
    write_environment_note(ROOT / "results" / "phase1" / "environment.json", device)
    arrays = load_experiment_arrays(ROOT, device="cpu")
    check_loader(arrays)

    report = prepared["report"]
    print("Phase 1 data is ready.")
    print(f"  split:  {report['split_status']}  train {report['train_size']}  val {report['val_size']}")
    print(f"  test:   {report['test_size']} official images, pixels not used for statistics")
    print(f"  counts: train {report['train_counts']}")
    print(f"          val   {report['val_counts']}")
    print(f"          test  {report['test_counts']}")
    print(f"  mean:   {report['mean_rgb']}")
    print(f"  std:    {report['std_rgb']}")
    print(f"  train index sha256: {report['train_idx_sha256']}")
    print(f"  device available for later training: {device}")
    print(f"  figures: {ROOT / 'figures'}")


if __name__ == "__main__":
    main()
