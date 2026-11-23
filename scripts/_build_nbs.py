"""One-shot builder for the two assignment notebooks. Not part of the experiments."""

import nbformat as nbf
from pathlib import Path

ROOT = Path("/scratch/m26csa003/ML_assignment_3/notebooks")

SETUP = r'''
from pathlib import Path
import sys

HERE = Path.cwd().resolve()
CANDIDATES = [HERE, *HERE.parents, Path("/scratch/m26csa003/ML_assignment_3")]
ROOT = next(path for path in CANDIDATES if (path / "src" / "common" / "data.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Saved logs are the reported results. Set this to True only on a GPU node
# when you intend to repeat training. The training entry points are the same
# functions the batch jobs called.
RECOMPUTE = False

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

from IPython.display import display, Image, Markdown, HTML

def show_rows(rows, columns=None, digits=4):
    """Render a list of dicts as an HTML table."""
    if not rows:
        display(Markdown("*No rows.*"))
        return
    columns = list(columns or rows[0].keys())

    def cell(value):
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return "" if value is None else str(value)

    head = "".join(f"<th style='text-align:left;padding:4px 8px'>{name}</th>" for name in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(
            f"<td style='padding:4px 8px'>{cell(row.get(name, ''))}</td>" for name in columns
        ) + "</tr>")
    display(HTML(
        "<table style='border-collapse:collapse;font-size:13px'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    ))

def show_figure(relative, width=720):
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    display(Image(filename=str(path), width=width))

print("Project root:", ROOT)
print("RECOMPUTE =", RECOMPUTE)
'''.strip()

NB1 = []

def md(text):
    NB1.append(("md", text.strip("\n")))

def code(text):
    NB1.append(("code", text.strip("\n")))

md("""
# Assignment 3, Part 1 — Data, MLPs, CNNs, and spatial structure

**Praveen, roll m26csa003. Split seed 26003.**

This notebook is the dataset check, Question 1, and Question 2. It loads the saved stratified split and the logs written by the training jobs. It does not draw a new split, and it does not read any variable from `Assignment_3_Part_2.ipynb`.

The model and training code that produced the logs is imported from `src/`. A second copy inside this notebook would be free to drift away from those logs, so the notebook calls that code rather than restating it. Set `RECOMPUTE = True` on a GPU node only if you intend to repeat training.
""")

md("""
## 1. Objectives

**Question 1.** Test one reason an MLP is weak on CIFAR-10, and separate that reason from "not enough parameters" and "it overfits". Then watch a wide MLP train on partly randomised labels, and decide from the trajectory whether useful structure appears before memorisation.

**Question 2.** Use that same matched MLP and matched CNN. Destroy spatial neighbourhoods while keeping the pixel values, and, separately, remove colour while keeping the neighbourhoods. The same transform is applied to train, validation, and test. Report test accuracy and confidence.
""")

md("## 2. Imports, paths, and the recompute switch")
code(SETUP)

md("""
## 3. Dataset

Seed 26003 is the digits of roll m26csa003, in order. `prepare_dataset` reloads `splits/split_seed26003.npz` and checks that a fresh draw from that seed matches it. It refuses to replace the file. Channel mean and population standard deviation are computed from the 45,000 training images only.
""")
code("""
import json
from src.common.data import prepare_dataset, SEED
from src.common.viz import write_dataset_figures

prepared = prepare_dataset(ROOT)
report = prepared["report"]
assert report["train_size"] == 45000
assert report["val_size"] == 5000
assert report["test_size"] == 10000
assert report["overlap"] == 0
assert report["train_counts"] == [4500] * 10
assert int(SEED) == 26003
print("split status:", report["split_status"])
print("train index sha256:", report["train_idx_sha256"])
print("validation index sha256:", report["val_idx_sha256"])
show_rows([{
    "train": report["train_size"],
    "validation": report["val_size"],
    "test": report["test_size"],
    "per class train": report["train_counts"][0],
    "per class validation": report["val_counts"][0],
    "overlap": report["overlap"],
}])
print("train mean RGB:", [round(v, 6) for v in report["mean_rgb"]])
print("train std RGB:", [round(v, 6) for v in report["std_rgb"]])

write_dataset_figures(prepared, ROOT / "figures", ROOT / "results" / "phase1")
for name in ("class_samples.png", "class_distribution.png", "mean_images.png", "pixel_statistics.png"):
    show_figure(Path_name := f"figures/{name}", width=640)
""".replace("Path_name := ", ""))

md("""
## 4. Question 1 — design locked before training

The comparison rules are in `configs/q1_fairness.json`. This notebook reads that file. It does not rewrite it. The matched CNN and the matched MLP are within 10% of each other in parameter count. The wide MLP has about 8.5 times as many parameters. The regularised MLP is the same width as the matched MLP, with dropout 0.3 and a stronger weight decay. No model in this comparison is augmented, and none uses batch normalisation.

One learning rate is shared. It was chosen by an 8-epoch validation probe of the matched MLP and the matched CNN. The score is the mean of their best validation accuracies. The wide and regularised models did not get a private search.
""")
code("""
import inspect
from src.common.models import MLP, ConvNet, build_model, count_parameters

fairness = json.loads((ROOT / "configs" / "q1_fairness.json").read_text())
print("written before training:", fairness["written_before_training"])
print(fairness["reason_under_test"])
rows = []
for name, spec in fairness["models"].items():
    built = count_parameters(build_model(spec))
    if built != int(spec["parameters"]):
        raise AssertionError(f"{name} has {built} parameters, the plan says {spec['parameters']}.")
    rows.append({
        "model": name,
        "parameters": built,
        "dropout": spec["dropout"],
        "weight_decay": spec["weight_decay"],
    })
show_rows(rows, digits=4)
display(Markdown("### MLP definition used for the logged runs"))
display(Markdown("```python\\n" + inspect.getsource(MLP) + "```"))
display(Markdown("### ConvNet definition used for the logged runs"))
display(Markdown("```python\\n" + inspect.getsource(ConvNet) + "```"))
""")

md("""
## 5. Question 1a — logged results

The tables below are read from `results/q1`. Training is not repeated. Final test numbers use the checkpoint with the highest validation accuracy, earliest epoch on a tie, for each of the three seeds 26003, 26004 and 26005. Sample standard deviation uses divisor \(n-1\).
""")
code("""
import csv
from src.q1.figures import write_q1_figures

chosen = json.loads((ROOT / "results" / "q1" / "chosen_lr.json").read_text())
print("probe rule:", chosen["rule"])
print("chosen learning rate:", chosen["chosen_lr"])
show_rows([
    {"learning_rate": key, "mean_best_validation": value}
    for key, value in chosen["scores_mean_best_val"].items()
])

write_q1_figures(ROOT)
table_path = ROOT / "results" / "q1" / "q1a_model_table.csv"
with table_path.open() as handle:
    q1_table = list(csv.DictReader(handle))
show_rows([
    {
        "model": row["display_name"],
        "parameters": int(row["parameters"]),
        "validation": float(row["val_accuracy_mean"]),
        "validation_std": float(row["val_accuracy_std"]),
        "test": float(row["test_accuracy_mean"]),
        "test_std": float(row["test_accuracy_std"]),
    }
    for row in q1_table
])

tests = json.loads((ROOT / "results" / "q1" / "test_results.json").read_text())["rows"]
for row in q1_table:
    subset = [item for item in tests if item["model"] == row["model"]]
    mean = sum(item["test_accuracy"] for item in subset) / len(subset)
    if abs(mean - float(row["test_accuracy_mean"])) > 1e-6:
        raise AssertionError(f"{row['model']} table does not match the per-seed test log.")

seconds = []
for path in sorted((ROOT / "results" / "q1" / "final").glob("*.json")):
    payload = json.loads(path.read_text())
    seconds.append({
        "run": path.stem,
        "best_epoch": payload["best_epoch"],
        "best_validation": payload["best_val_accuracy"],
        "train_seconds": payload["seconds"],
    })
show_rows(seconds)
for name in (
    "figures/q1/q1a_lr_probe.png",
    "figures/q1/q1a_val_accuracy.png",
    "figures/q1/q1a_train_accuracy.png",
    "figures/q1/q1a_test_accuracy.png",
):
    show_figure(name)
""")

md("""
## 6. Question 1b — structure and memorisation

The wide MLP from Question 1a is trained again with weight decay set to 0, which the fairness file says is not part of the fair comparison. A fixed fraction of training labels is replaced by a wrong class. Validation labels stay clean. Four curves are logged every epoch: accuracy on the untouched training rows, fit to the random labels, accuracy on the true class of the corrupted rows, and clean validation accuracy.

The rule written before training says that useful structure appeared before memorisation only if, at 20% or 50% noise, validation rises while the fit to the random labels is still near chance (10%), and that fit keeps rising after validation has peaked.
""")
code("""
with (ROOT / "results" / "q1" / "q1b_summary.csv").open() as handle:
    memo = list(csv.DictReader(handle))
show_rows(memo, digits=4)
with (ROOT / "results" / "q1" / "q1b_milestones.csv").open() as handle:
    milestones = list(csv.DictReader(handle))
show_rows(milestones, digits=4)
show_figure("figures/q1/q1b_memorisation.png", width=780)
""")

md("""
### Reading of Question 1

The matched CNN's test accuracy is 0.739 ± 0.030. The matched MLP is 0.504 ± 0.002. About 8.5 times as many parameters move the MLP test accuracy to 0.527 ± 0.003. Dropout and a stronger weight decay leave it at 0.505 ± 0.004. The extra parameters overfit: training accuracy goes high and the test loss stays high. The gap that remains is the one the plan attributed to missing spatial weight sharing.

At 20% label noise, validation rises from 0.393 to 0.458 in five epochs while the random-label fit stays at 0.070–0.081. It peaks at 0.463 on epoch 19, and the random-label fit then continues from 0.228 to 0.395. At 50% noise the validation peak is 0.318 on epoch 6, while the noise fit is still 0.090, and the noise fit later reaches 0.285 as validation falls to 0.282. That is the pre-written condition, with a small useful gain at 50%. At 100% noise the training loss stays at \(\\ln 10\) and nothing leaves chance, so this budget did not fit a fully random training set.
""")

md("""
## 7. Question 2 — permutation and grayscale

Part (a) does not train a new pair. It reuses the matched models from Question 1a. Parts (b) and (c) retrain both models with the same optimiser, the same learning rate 0.003, 30 epochs, and the three seeds. A single permutation of the 1,024 spatial positions is saved and applied to every image and every channel. Grayscale uses luma \(0.2989 R + 0.5870 G + 0.1140 B\), repeated on three channels so the parameter counts do not change. Mean and standard deviation for the grayscale images are computed on the training split only.
""")
code("""
from src.q2.figures import write_q2_figures

plan = json.loads((ROOT / "configs" / "q2_plan.json").read_text())
print("written before the retraining:", plan["written_before_training"])
print("Part (b) expectation:")
print(plan["expectations_before_training"]["part_b"])
print()
print("Part (c) expectation:")
print(plan["expectations_before_training"]["part_c"])
write_q2_figures(ROOT)
with (ROOT / "results" / "q2" / "q2_summary.csv").open() as handle:
    q2_rows = list(csv.DictReader(handle))
show_rows([
    {
        "model": row["model"],
        "condition": row["condition"],
        "parameters": int(float(row["parameters"])),
        "test": float(row["test_accuracy_mean"]),
        "test_std": float(row["test_accuracy_std"]),
        "confidence_correct": float(row["confidence_correct_mean"]),
        "confidence_wrong": float(row["confidence_incorrect_mean"]),
        "high_conf_error_fraction": float(row["high_confidence_error_fraction_mean"]),
    }
    for row in q2_rows
])
# The clean re-evaluation has to match the Question 1 test log.
q1_cnn = next(row for row in q1_table if row["model"] == "cnn_matched")
q2_cnn = next(row for row in q2_rows if row["model"] == "cnn_matched" and row["condition"] == "clean")
if abs(float(q1_cnn["test_accuracy_mean"]) - float(q2_cnn["test_accuracy_mean"])) > 1e-4:
    raise AssertionError("Clean CNN test accuracy disagrees between Question 1 and Question 2.")
for name in (
    "figures/q2/q2_test_accuracy.png",
    "figures/q2/q2_confidence.png",
    "figures/q2/q2_confusion_cnn_seed26003.png",
):
    show_figure(name, width=760)
""")

md("""
### Reading of Question 2

After the pixel permutation the MLP test accuracy is 0.501 ± 0.003, inside the band expected around its clean 0.504. The CNN falls from 0.739 to 0.486 ± 0.028. The spatial explanation holds: once a 3×3 filter no longer sits on a neighbourhood, the CNN loses the advantage.

On grayscale the MLP is 0.419 ± 0.005 and the CNN mean is 0.612 ± 0.122. The three CNN seeds are 0.715, 0.645 and 0.476. Two seeds stay well above the MLP and above 0.60. Seed 26005 has validation accuracy 0.490, so the "above 60%" sentence is not true of every seed. Colour is used by both models. It does not, on the two stable seeds, explain the clean-image gap.

Correct predictions have a higher mean max-softmax than incorrect ones in every condition. After the permutation, the fraction of CNN errors with confidence at least 0.9 falls from 0.148 to 0.008.
""")

md("""
## 8. Summary of Part 1

The matched CNN leads the matched MLP by about 23 test points under a matched parameter budget, a shared learning rate, and no augmentation. A much wider MLP and a regularised MLP do not close that gap. A fixed spatial permutation removes the CNN's lead and leaves the MLP essentially unchanged. Removing colour lowers both models; the CNN lead survives in two of three seeds and fails in the third.

Question 3 and Question 4 are in `Assignment_3_Part_2.ipynb`. That notebook reads these logs from disk. It does not need this kernel.
""")
code("""
if RECOMPUTE:
    from src.q1.pipeline import main as train_q1
    from src.q2.pipeline import main as train_q2
    train_q1()
    train_q2()
else:
    print("Using the saved Question 1 and Question 2 logs. Retraining is scripts/q1_srun.sh and the Q2 section of scripts/q234_srun.sh.")
""")

# ---- notebook 2, separate list ----
NB2 = []

def md2(text):
    NB2.append(("md", text.strip("\n")))

def code2(text):
    NB2.append(("code", text.strip("\n")))

md2("""
# Assignment 3, Part 2 — Autoencoders, SVMs, and clustering

**Praveen, roll m26csa003. Split seed 26003.**

This notebook is Question 3 and Question 4. It loads the saved split, the autoencoder logs, and the SVM logs from disk. It does not read variables from `Assignment_3_Part_1.ipynb`. The classifier and the 160-dimensional features come from the Question 1 matched CNN checkpoint that was selected by validation accuracy (seed 26003). That checkpoint is a file, first introduced in Question 1a.

The training and fitting code is imported from `src/` so it stays the code that wrote the logs. `RECOMPUTE = False` skips a repeat of that work.
""")

md2("""
## 1. Objectives

**Question 3.** Train three convolutional autoencoders that differ only in bottleneck width. If reconstruction error were the only criterion, choose one on the validation set. Train a denoiser for Gaussian noise and compare a frozen CNN on corrupted images and on reconstructions. Without fine-tuning the encoders, measure how much class information a linear probe can read, and compare that with reconstruction quality.

**Question 4.** Freeze one representation. Sweep at least five values of \(C\) across at least three orders of magnitude, and at least three kernels including RBF. Select on validation, then evaluate once on the test set. Inspect three hard class pairs, and cluster the same representation without labels during fitting.
""")

md2("## 2. Imports, paths, and the recompute switch")
code2(SETUP)

md2("""
## 3. Split check

This cell reloads the same split file Question 1 used and checks that a fresh draw from seed 26003 still matches it. Validation and test images are not used to recompute the channel statistics.
""")
code2("""
import json
from src.common.data import prepare_dataset, SEED

prepared = prepare_dataset(ROOT)
report = prepared["report"]
assert report["train_size"] == 45000
assert report["val_size"] == 5000
assert report["overlap"] == 0
assert report["train_counts"] == [4500] * 10
assert int(SEED) == 26003
print("split status:", report["split_status"])
print("train index sha256:", report["train_idx_sha256"])
show_rows([{
    "train": report["train_size"],
    "validation": report["val_size"],
    "test": report["test_size"],
    "overlap": report["overlap"],
}])
""")

md2("""
## 4. Question 3 — three bottleneck widths

The plan is `configs/q3_plan.json`, written before training. The convolutions are identical. Only `to_code.weight`, `to_code.bias`, and `from_code.weight` change with the bottleneck. Images stay in \([0, 1]\). The loss is pixel mean squared error. Weight decay is 0. The learning rate is chosen by a 4-epoch probe on the middle bottleneck (128): lowest validation MSE wins, and the smaller learning rate wins a tie.

The checkpoint is the lowest validation MSE, earliest epoch on a tie. The bottleneck is chosen the same way. Test reconstruction error of the selected model is written only after that choice.
""")
code2("""
import csv
import inspect
from src.q3.models import ConvAutoencoder
from src.q3.pipeline import assert_architecture
from src.q3.figures import write_q3_figures

plan = json.loads((ROOT / "configs" / "q3_plan.json").read_text())
print("written before training:", plan["written_before_training"])
assert_architecture(plan)
print(plan["architecture"])
display(Markdown("```python\\n" + inspect.getsource(ConvAutoencoder) + "```"))

chosen_lr = json.loads((ROOT / "results" / "q3" / "chosen_lr.json").read_text())
print("probe rule:", chosen_lr["rule"])
show_rows([
    {"learning_rate": row["lr"], "best_validation_mse": row["best_val_mse"]}
    for row in chosen_lr["rows"]
])
print("chosen learning rate:", chosen_lr["chosen_lr"])

selection = json.loads((ROOT / "results" / "q3" / "q3a_selection.json").read_text())
print("selection rule:", selection["rule"])
print("chosen bottleneck:", selection["chosen_bottleneck"])
show_rows(selection["rows"])
test_mse = json.loads((ROOT / "results" / "q3" / "q3a_test_mse.json").read_text())
print("selected test MSE:", test_mse["test_mse"], "PSNR:", test_mse["test_psnr"])
if int(test_mse["bottleneck"]) != int(selection["chosen_bottleneck"]):
    raise AssertionError("Test MSE file is not the selected bottleneck.")
""")

md2("""
## 5. Question 3b and 3c — denoising and linear probes

Corruption is additive Gaussian noise with standard deviation 0.25 on images in \([0, 1]\), then clipped. One frozen noisy validation set and one frozen noisy test set were drawn before training. The denoiser has the bottleneck selected above. Its inputs are noisy and its targets are clean. Class labels are not used.

The classifier is the Question 1a matched CNN with the highest validation accuracy. Ties would keep the lower seed. It is frozen. Reconstructions and noisy images are normalised with the Question 1 training mean and standard deviation before they are classified, which is the transform that CNN was trained with.

The linear probe is multinomial logistic regression on the frozen code. \(C\) is chosen on the validation codes. Encoders are not fine-tuned.
""")
code2("""
write_q3_figures(ROOT)
with (ROOT / "results" / "q3" / "q3b_classification.csv").open() as handle:
    class_rows = list(csv.DictReader(handle))
show_rows([
    {
        "condition": row["condition"],
        "test_accuracy": float(row["test_accuracy"]),
        "pixel_mse": float(row["pixel_mse_vs_clean"]),
        "psnr": row["psnr_vs_clean"],
    }
    for row in class_rows
])
with (ROOT / "results" / "q3" / "q3c_representation.csv").open() as handle:
    rep_rows = list(csv.DictReader(handle))
show_rows(rep_rows)
denoise = json.loads((ROOT / "results" / "q3" / "denoise" / "ae_b512_seed26003.json").read_text())
print(f"denoiser training seconds: {denoise['seconds']:.1f}, best epoch {denoise['best_epoch']}")
for name in (
    "figures/q3/q3a_val_mse.png",
    "figures/q3/q3b_classification.png",
    "figures/q3/q3c_mse_vs_probe.png",
):
    show_figure(name)
""")

md2("""
### Reading of Question 3

Validation MSE falls with the bottleneck: 0.01152 at width 32, 0.00557 at 128, and 0.00479 at 512. The written rule selects 512. Its test MSE is 0.00483 (PSNR 23.16). All three clean models were still best on epoch 20, the last epoch of the budget, so a longer run could lower every curve. The order is already clear.

The frozen CNN scores 0.761 on clean test images and 0.167 on the noisy images. The clean autoencoder's reconstruction of those noisy images scores 0.334. The denoiser scores 0.330 and has the better PSNR (21.21 against 20.12). Better pixels did not restore the classifier. Both reconstructions help relative to the raw noisy image and leave most of the gap to the clean image.

Probe test accuracy is 0.364, 0.444 and 0.545 as the code widens. Reconstruction and class information move in the same direction. Most of the MSE drop is from 32 to 128. The probe keeps rising from 128 to 512, where the MSE drop is small. Some logistic fits hit the 400-iteration limit, so an individual \(C\) can be slightly short of its optimum. The gaps between bottlenecks are larger than that.
""")

md2("""
## 6. Question 4 — a frozen code, then an SVM

The representation is the 160-dimensional vector after global average pooling in the Question 1a matched CNN, before the linear head. The checkpoint is seed 26003, chosen because it had the highest validation accuracy among the three Question 1 seeds. The network is not updated.

The SVM training subset is 200 images from each class inside the 45,000 training rows, drawn with `Generator(26003)` and saved before any fit (`splits/q4_svm_subset_seed26003.npy`). Every kernel and every \(C\) uses that file. Features are standardised with the mean and standard deviation of those 2,000 rows only.

Kernels are linear, polynomial (degree 3, `coef0` 1, `gamma='scale'`), and RBF (`gamma='scale'`). Degree and gamma are not searched. \(C\) is 0.01, 0.1, 1, 10 and 100. The highest validation accuracy wins. A tie keeps the smaller support-vector fraction, then the smaller \(C\), then the kernel name in alphabetical order. The winner is evaluated once on the official test set.
""")
code2("""
from src.q4.figures import write_q4_figures

q4 = json.loads((ROOT / "configs" / "q4_plan.json").read_text())
print("written before any fit:", q4["written_before_any_fit"])
print("subset size:", q4["subset_size"], "per class:", q4["subset_per_class"])
print("C values:", q4["C_values"])
print("kernels:", q4["kernels"])
subset = __import__("numpy").load(ROOT / "splits" / "q4_svm_subset_seed26003.npy")
if subset.shape != (2000,):
    raise AssertionError(f"Subset length is {subset.shape}, the plan says 2000.")

write_q4_figures(ROOT)
with (ROOT / "results" / "q4" / "svm_sweep.csv").open() as handle:
    sweep = list(csv.DictReader(handle))
show_rows([
    {
        "kernel": row["kernel"],
        "C": float(row["C"]),
        "train": float(row["train_accuracy"]),
        "validation": float(row["val_accuracy"]),
        "support_fraction": float(row["support_fraction"]),
    }
    for row in sweep
])
selected = json.loads((ROOT / "results" / "q4" / "svm_selection.json").read_text())
tested = json.loads((ROOT / "results" / "q4" / "svm_test.json").read_text())
print("selected:", selected["kernel"], "C =", selected["C"], "validation =", selected["val_accuracy"])
print("single test accuracy:", tested["test_accuracy"])
# The selected row must be the unique validation maximum in the sweep.
best_val = max(float(row["val_accuracy"]) for row in sweep)
if abs(best_val - float(selected["val_accuracy"])) > 1e-6:
    raise AssertionError("Selection file is not the best validation row.")
for kernel in ("linear", "poly", "rbf"):
    show_figure(f"figures/q4/q4_{kernel}.png")
""")

md2("""
## 7. Question 4b — three hard pairs

After the single test evaluation, unordered class pairs are ranked by the sum of the two off-diagonal confusion counts. The three largest are kept. A tie would keep the smaller class index. This uses test labels to choose which pairs to draw, which the question asks for, and it does not change \(C\) or the kernel.

For each pair the score is that column of the one-versus-one decision function. The figure shows up to eight training support vectors of that binary problem and eight test images of the two classes with the smallest absolute score.
""")
code2("""
pairs = json.loads((ROOT / "results" / "q4" / "hard_pairs.json").read_text())["pairs"]
show_rows([
    {
        "pair": f"{pair['class_i_name']} / {pair['class_j_name']}",
        "test_confusions": pair["test_confusions"],
        "pair_images": pair["test_pair_count"],
        "error_rate": pair["error_rate"],
        "support_vectors": pair["n_support_vectors"],
        "closest_test_score": pair["nearest_test"][0]["decision_score"],
        "closest_correct": pair["nearest_test"][0]["correct"],
    }
    for pair in pairs
])
show_figure("figures/q4/q4_hard_pairs.png", width=820)
""")

md2("""
## 8. Question 4c — clustering without labels

KMeans is fit on the same 2,000 standardised training features. Labels are not passed to the fitter. The number of clusters is chosen by validation NMI, over \(k \\in \\{5, 8, 10, 12, 15, 20\\}\). A tie keeps the \(k\) closer to 10, then the smaller \(k\). Test points are assigned by that model. Test labels are used only afterwards, for ARI, NMI, the contingency table, and the majority-vote accuracy.

The majority map from cluster id to class is fit on the training subset only. The disagreement class is the one with the largest SVM recall minus that cluster recall.
""")
code2("""
clustering = json.loads((ROOT / "results" / "q4" / "clustering.json").read_text())
show_rows([
    {"k": row["k"], "validation_nmi": row["val_nmi"], "validation_ari": row["val_ari"]}
    for row in clustering["k_rows"]
])
print("chosen k:", clustering["chosen_k"])
print("test ARI:", clustering["test_ari"])
print("test NMI:", clustering["test_nmi"])
print("cluster majority accuracy:", clustering["test_majority_accuracy"])
print("SVM test accuracy:", clustering["svm_test_accuracy"])
show_rows([
    {
        "class": row["class_name"],
        "svm_recall": row["svm_recall"],
        "cluster_recall": row["cluster_recall"],
        "svm_minus_cluster": row["svm_minus_cluster"],
    }
    for row in clustering["per_class"]
])
gap = clustering["disagreement_class"]
print(
    "disagreement class:", gap["class_name"],
    "SVM recall", gap["svm_recall"],
    "cluster recall", gap["cluster_recall"],
)
show_figure("figures/q4/q4_contingency.png")
show_figure("figures/q4/q4_recall_comparison.png")
""")

md2("""
### Reading of Question 4

Polynomial degree 3 with \(C = 1\) wins validation at 0.734, with training accuracy 0.999 and 68.6% of the 2,000 rows as support vectors. The one test accuracy is 0.7235.

Small \(C\) underfits. RBF at \(C = 0.01\) has training accuracy 0.693, validation 0.609, and every training point is a support vector. Linear overfits as \(C\) grows: validation is 0.730 at \(C = 0.01\) and about 0.710 once training accuracy has reached 1. Polynomial and RBF are unchanged from \(C = 10\) to \(C = 100\).

The hardest test pairs are cat/dog (358 confusions, error rate 0.179), bird/deer (178, 0.089) and airplane/ship (146, 0.073). The cat/dog test image closest to the boundary is an error, with a decision score near zero.

KMeans with \(k = 10\) has validation NMI 0.362. Test ARI is 0.228, test NMI is 0.361, and majority-vote accuracy is 0.460, against the SVM's 0.724. Bird is the disagreement class: SVM recall 0.615, and no training cluster has bird as its majority label, so the cluster map's bird recall is 0.
""")

md2("""
## 9. Summary of Part 2

A wider bottleneck reconstructs better and is easier for a linear probe to read, but the two gains are not proportional. Denoising improves pixel error relative to a clean autoencoder and does not improve the frozen CNN's accuracy on those reconstructions. On the frozen convolutional features, a polynomial SVM selected only on validation reaches 0.724 test accuracy, while label-free KMeans on the same vectors reaches 0.460 by a training-set majority map and does not recover the bird class.
""")
code2("""
if RECOMPUTE:
    from src.q3.pipeline import main as train_q3
    from src.q4.pipeline import main as train_q4
    train_q3()
    train_q4()
else:
    print("Using the saved Question 3 and Question 4 logs. Retraining is scripts/q234_srun.sh on a GPU node.")
""")


def write_notebook(cells, path):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    built = []
    for kind, source in cells:
        if kind == "md":
            built.append(nbf.v4.new_markdown_cell(source))
        else:
            built.append(nbf.v4.new_code_cell(source))
    nb.cells = built
    path.write_text(nbf.writes(nb))
    print("wrote", path, "cells", len(built))


write_notebook(NB1, ROOT / "Assignment_3_Part_1.ipynb")
write_notebook(NB2, ROOT / "Assignment_3_Part_2.ipynb")
