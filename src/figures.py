"""Figures for the report, saved as PNG to <runs_root>/figures/.

  class_distribution.png   images (and lesions) per diagnosis
  leakage_counts.png       val/test images whose lesion is also in train, per split
  confusion_matrices.png   one panel per run (seed 0), rows normalised to 100%
  h3_subgroups.png         malignant sensitivity and AUROC per subgroup, BiT vs ViT
  h4_subgroups.png         malignant sensitivity per subgroup, standard vs balanced
  training_curves.png      loss and val macro-F1 per epoch, every run and seed

Subgroup charts show each value as a dot with its 95% CI as a line. (Bars
must start at 0, which squeezes values like 0.85 vs 0.92 together and hides
the CIs.) A figure is skipped if its input files don't exist yet.

Inputs: <data_dir>/metadata.csv, split_lesion.csv, split_image.csv;
        <runs_root>/<run>/history.csv, test_predictions.csv, config.yaml;
        <runs_root>/evaluation/*.csv (from evaluate.py).

Usage (Colab):
    python src/figures.py --runs_root /content/drive/MyDrive/ham10000_runs --data_dir /content/data
    python src/figures.py --smoke
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # save to files, no screen needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from data import CLASSES

# ---- Colours (a colour-blind-safe palette, checked with a palette validator) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # main text
INK_2 = "#52514e"        # secondary text
MUTED = "#898781"        # axis ticks
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# One fixed colour per run, the same in every figure.
RUN_COLOURS = {"run1": "#2a78d6", "run3": "#2a78d6",     # BiT: blue
               "run2": "#eb6834", "run4": "#eb6834",     # ViT standard: orange
               "run5": "#1baf7a",                        # ViT balanced: aqua
               "run5b": "#eda100"}                       # ViT class-balanced: yellow
BIT, VIT, BALANCED, CLASS_BALANCED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL = "#52514e"      # single-series descriptive bars
SEEN = "#e34948"         # leaked (seen) images
UNSEEN = AXIS            # not leaked
# Single-hue blue ramp (light -> dark) for the confusion matrices.
BLUES = LinearSegmentedColormap.from_list(
    "blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

RUN_LABELS = {"run1": "Run 1 · BiT · lesion", "run2": "Run 2 · ViT · lesion",
              "run3": "Run 3 · BiT · image", "run4": "Run 4 · ViT · image",
              "run5": "Run 5 · ViT · lesion · balanced",
              "run5b": "Run 5b · ViT · lesion · class-balanced"}
GROUP_TITLES = {"age_band": "Age band", "sex": "Sex", "site_group": "Site group"}


def set_style():
    """Quiet axes and grid so the data stands out."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    })


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_run_folders(runs_root):
    """List of dicts, one per run folder: run id, seed, model, split, mode, folder."""
    found = []
    for name in sorted(os.listdir(runs_root)):
        folder = os.path.join(runs_root, name)
        cfg_path = os.path.join(folder, "config.yaml")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        found.append({"run": name.split("_")[0], "seed": int(cfg["seed"]), "model": cfg["model"],
                      "split": cfg["split"], "mode": cfg["mode"], "folder": folder})
    order = list(RUN_LABELS)
    return sorted(found, key=lambda r: (order.index(r["run"]) if r["run"] in order else 99, r["seed"]))


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def class_distribution(meta, out_dir):
    """Horizontal bars: images per diagnosis, labelled with images and lesions."""
    counts = pd.DataFrame({"images": meta.groupby("dx").size(),
                           "lesions": meta.groupby("dx")["lesion_id"].nunique()}).reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(6, 3))
    y = np.arange(len(CLASSES))[::-1]  # first class at the top
    ax.barh(y, counts["images"], height=0.6, color=NEUTRAL)
    for yi, (n_img, n_les) in zip(y, counts.to_numpy()):
        ax.text(n_img, yi, f"  {n_img:,} images · {n_les:,} lesions", va="center", color=INK_2, fontsize=8)
    ax.set_yticks(y, CLASSES)
    ax.set_xlabel("Images")
    ax.set_xlim(0, counts["images"].max() * 1.45)  # room for the labels
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(f"Class distribution ({len(meta):,} images, {meta['lesion_id'].nunique():,} lesions)",
                 loc="left")
    save(fig, out_dir, "class_distribution.png")


def leakage_counts(split_lesion, split_image, out_dir):
    """Stacked bars: val/test images whose lesion is (not) also in train."""
    rows = []
    for split_name, df in [("Lesion split", split_lesion), ("Image split", split_image)]:
        for part in ["val", "test"]:
            p = df[df["split"] == part]
            rows.append((f"{split_name} · {part}", int(p["seen_lesion"].sum()), int((~p["seen_lesion"]).sum())))
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    y = np.arange(len(rows))[::-1]
    for yi, (label, seen, unseen) in zip(y, rows):
        # A 2px surface-coloured edge separates the two stacked segments.
        ax.barh(yi, unseen, height=0.6, color=UNSEEN, edgecolor=SURFACE, linewidth=2)
        ax.barh(yi, seen, left=unseen, height=0.6, color=SEEN, edgecolor=SURFACE, linewidth=2)
        total = seen + unseen
        ax.text(total, yi, f"  {seen:,} of {total:,} seen ({seen / total:.0%})",
                va="center", color=INK_2, fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Images")
    ax.set_xlim(0, max(r[1] + r[2] for r in rows) * 1.5)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=UNSEEN), plt.Rectangle((0, 0), 1, 1, color=SEEN)],
              labels=["lesion not in train", "lesion also in train (seen)"],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    ax.set_title("Leakage: val/test images whose lesion has another image in train", loc="left", pad=24)
    save(fig, out_dir, "leakage_counts.png")


def confusion_matrices(runs, out_dir):
    """One row-normalised confusion matrix per run (seed 0)."""
    seed0 = [r for r in runs if r["seed"] == 0
             and os.path.exists(os.path.join(r["folder"], "test_predictions.csv"))]
    if not seed0:
        print("Skipped confusion_matrices.png: no test predictions")
        return
    n_cols = min(3, len(seed0))
    n_rows = int(np.ceil(len(seed0) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows), squeeze=False,
                             layout="constrained")  # leaves room for every label
    k = len(CLASSES)
    for ax, r in zip(axes.flat, seed0):
        p = pd.read_csv(os.path.join(r["folder"], "test_predictions.csv"))
        cm = np.bincount(p["y_true"] * k + p["y_pred"], minlength=k * k).reshape(k, k)
        # Each row (true class) sums to 100%: the diagonal is each class's recall.
        rows_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(rows_pct, cmap=BLUES, vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                if cm[i, j] > 0:
                    ax.text(j, i, f"{rows_pct[i, j]:.0%}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if rows_pct[i, j] > 0.55 else INK)
        ax.set_xticks(range(k), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(k), CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(RUN_LABELS.get(r["run"], r["run"]) + " (seed 0)", fontsize=9, loc="left")
    for ax in list(axes.flat)[len(seed0):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Share of the true class")
    save(fig, out_dir, "confusion_matrices.png")


def dot_ci_panel(ax, groups, series, labels):
    """Dots with 95% CI lines for several series side by side at each group.

    series: list of (name, colour, values, ci_low, ci_high) with one value per group.
    """
    x = np.arange(len(groups))
    width = 0.22
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for (name, colour, values, low, high), dx in zip(series, offsets):
        ax.vlines(x + dx, low, high, color=colour, linewidth=2)
        # A 2px surface ring keeps overlapping dots apart.
        ax.plot(x + dx, values, "o", color=colour, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.set_xticks(x, labels)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def legend_above(fig, ax, title, height_in):
    """Title, then one shared legend row, above the panels (never on top of the data)."""
    handles, labels = ax.get_legend_handles_labels()
    top = 1 - 0.75 / height_in           # leave ~0.75 inch for the title and the legend
    fig.tight_layout(rect=(0, 0, 1, top))  # lay out the panels first, then add title and legend
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=10)
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, top), ncol=len(labels),
               frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.5)


def h3_subgroups(eval_dir, out_dir):
    """Malignant sensitivity and AUROC per subgroup: BiT vs ViT."""
    path = os.path.join(eval_dir, "h3_subgroups.csv")
    if not os.path.exists(path):
        print("Skipped h3_subgroups.png: run evaluate.py first")
        return
    df = pd.read_csv(path)
    columns = [c for c in GROUP_TITLES if c in set(df["column"])]
    fig, axes = plt.subplots(2, len(columns), figsize=(4 * len(columns), 6),
                             sharey="row", squeeze=False,
                             gridspec_kw={"width_ratios": [df[df["column"] == c]["group"].nunique() for c in columns]})
    for row, metric in enumerate(["sensitivity", "auroc"]):
        for col, column in enumerate(columns):
            d = df[(df["column"] == column) & (df["metric"] == metric)]
            d = d[d["vit"].notna() | d["bit"].notna()]
            labels = [f"{g}\nn={m}" for g, m in zip(d["group"], d["n_malignant"])]
            ax = axes[row, col]
            dot_ci_panel(ax, list(d["group"]), [
                ("BiT ResNet-50 (run 1)", BIT, d["bit"], d["bit_ci_low"], d["bit_ci_high"]),
                ("ViT-B/16 (run 2)", VIT, d["vit"], d["vit_ci_low"], d["vit_ci_high"])], labels)
            if row == 0:
                ax.set_title(GROUP_TITLES[column], loc="left")
        axes[row, 0].set_ylabel("Malignant sensitivity" if metric == "sensitivity" else "Malignant AUROC")
    legend_above(fig, axes[0, 0], "H3: malignant vs benign by subgroup (mean over seeds, 95% lesion-bootstrap CI)\n"
                 "n = malignant test images in the group", height_in=6)
    save(fig, out_dir, "h3_subgroups.png")


def h4_subgroups(eval_dir, out_dir):
    """Malignant sensitivity per subgroup: standard vs balanced (and class-balanced)."""
    path = os.path.join(eval_dir, "h4_group_sensitivity.csv")
    if not os.path.exists(path):
        print("Skipped h4_subgroups.png: run evaluate.py first")
        return
    df = pd.read_csv(path)
    columns = [c for c in GROUP_TITLES if c in set(df["column"])]
    comparisons = list(dict.fromkeys(df["comparison"]))  # "balanced", maybe "class_balanced"
    colours = {"balanced": BALANCED, "class_balanced": CLASS_BALANCED}
    names = {"balanced": "Balanced sampler (run 5)", "class_balanced": "Class-only sampler (run 5b)"}
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.4), sharey=True, squeeze=False,
                             gridspec_kw={"width_ratios": [df[df["column"] == c]["group"].nunique() for c in columns]})
    for ax, column in zip(axes[0], columns):
        d = df[df["column"] == column]
        first = d[d["comparison"] == comparisons[0]]
        series = [("Standard (run 2)", VIT, first["standard"], first["standard_ci_low"], first["standard_ci_high"])]
        for comp in comparisons:
            c = d[d["comparison"] == comp]
            series.append((names.get(comp, comp), colours.get(comp, NEUTRAL),
                           c["variant"], c["variant_ci_low"], c["variant_ci_high"]))
        # Groups too small for the worst-group analysis are marked with *.
        labels = [f"{g}{'' if e else '*'}\nn={m}" for g, m, e in
                  zip(first["group"], first["n_malignant"], first["eligible"])]
        dot_ci_panel(ax, list(first["group"]), series, labels)
        ax.set_title(GROUP_TITLES[column], loc="left")
    axes[0, 0].set_ylabel("Malignant sensitivity")
    legend_above(fig, axes[0, 0], "H4: malignant sensitivity by subgroup, standard vs balanced training (ViT, lesion split)\n"
                 "n = malignant test images; * = too few to count in the worst-group analysis", height_in=3.4)
    save(fig, out_dir, "h4_subgroups.png")


def training_curves(runs, out_dir):
    """Per run: train/val loss (top) and val macro-F1 (bottom); every seed drawn."""
    by_run = {}
    for r in runs:
        h = os.path.join(r["folder"], "history.csv")
        if os.path.exists(h):
            by_run.setdefault(r["run"], []).append((r["seed"], pd.read_csv(h)))
    if not by_run:
        print("Skipped training_curves.png: no history.csv files")
        return
    n = len(by_run)
    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 5), sharey="row", squeeze=False)
    for col, (run_id, histories) in enumerate(by_run.items()):
        colour = RUN_COLOURS.get(run_id, NEUTRAL)
        for seed, h in histories:
            axes[0, col].plot(h["epoch"], h["train_loss"], color=colour, linewidth=1.5, linestyle="--")
            axes[0, col].plot(h["epoch"], h["val_loss"], color=colour, linewidth=2)
            axes[1, col].plot(h["epoch"], h["val_macro_f1"], color=colour, linewidth=2)
            best = h["val_macro_f1"].idxmax()  # the epoch whose weights were kept
            axes[1, col].plot(h.loc[best, "epoch"], h.loc[best, "val_macro_f1"], "o", color=colour,
                              markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        # "Run 5 · ViT · lesion · balanced" -> "Run 5" on one line, the rest below.
        title = RUN_LABELS.get(run_id, run_id).replace(" · ", "\n", 1)
        axes[0, col].set_title(f"{title}\n{len(histories)} seed(s)", loc="left", fontsize=8)
        axes[1, col].set_xlabel("Epoch")
        for ax in axes[:, col]:
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # whole epochs only
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Val macro-F1")
    axes[0, 0].legend(handles=[Line2D([], [], color=INK_2, linestyle="--", linewidth=1.5),
                               Line2D([], [], color=INK_2, linewidth=2)],
                      labels=["train", "val"], fontsize=8)
    axes[1, 0].legend(handles=[Line2D([], [], marker="o", color=INK_2, linestyle="", markersize=6)],
                      labels=["best epoch (kept)"], fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, out_dir, "training_curves.png")


# ----------------------------------------------------------------------------
# Smoke test data
# ----------------------------------------------------------------------------

def make_smoke_inputs():
    """Fake predictions + evaluation tables (via evaluate.py), splits and histories."""
    import subprocess
    import sys

    import data
    import splits

    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "evaluate.py"), "--smoke"],
                   check=True, stdout=subprocess.DEVNULL)
    work, runs_root = "data_fake", os.path.join("data_fake", "runs_eval")
    meta = data.build_metadata(work)
    meta.to_csv(os.path.join(work, "metadata.csv"), index=False)
    splits.lesion_level_split(meta, 42).to_csv(os.path.join(work, "split_lesion.csv"), index=False)
    splits.image_level_split(meta, 42).to_csv(os.path.join(work, "split_image.csv"), index=False)

    rng = np.random.default_rng(0)
    for name in os.listdir(runs_root):
        folder = os.path.join(runs_root, name)
        if not os.path.exists(os.path.join(folder, "config.yaml")):
            continue
        epochs = int(rng.integers(8, 16))  # as if early stopping ended at different epochs
        e = np.arange(epochs)
        f1 = 0.75 - 0.4 * np.exp(-e / 3) + rng.normal(0, 0.02, epochs)
        pd.DataFrame({"epoch": e, "train_loss": 1.6 * np.exp(-e / 4) + 0.2,
                      "val_loss": 1.4 * np.exp(-e / 4) + 0.4 + 0.02 * e,
                      "val_macro_f1": f1, "lr": 1e-4}).to_csv(os.path.join(folder, "history.csv"), index=False)
    return runs_root, work


def main():
    parser = argparse.ArgumentParser(description="Make the report figures.")
    parser.add_argument("--runs_root", default="/content/drive/MyDrive/ham10000_runs")
    parser.add_argument("--data_dir", default="/content/data",
                        help="folder with metadata.csv, split_lesion.csv, split_image.csv")
    parser.add_argument("--out_dir", default=None, help="default: <runs_root>/figures")
    parser.add_argument("--smoke", action="store_true", help="make figures from fake data")
    args = parser.parse_args()

    if args.smoke:
        args.runs_root, args.data_dir = make_smoke_inputs()

    out_dir = args.out_dir or os.path.join(args.runs_root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    set_style()

    meta_path = os.path.join(args.data_dir, "metadata.csv")
    if os.path.exists(meta_path):
        class_distribution(pd.read_csv(meta_path), out_dir)
    else:
        print(f"Skipped class_distribution.png: no {meta_path}")
    les, img = (os.path.join(args.data_dir, f"split_{s}.csv") for s in ["lesion", "image"])
    if os.path.exists(les) and os.path.exists(img):
        leakage_counts(pd.read_csv(les), pd.read_csv(img), out_dir)
    else:
        print("Skipped leakage_counts.png: split CSVs not found")

    runs = load_run_folders(args.runs_root)
    confusion_matrices(runs, out_dir)
    eval_dir = os.path.join(args.runs_root, "evaluation")
    h3_subgroups(eval_dir, out_dir)
    h4_subgroups(eval_dir, out_dir)
    training_curves(runs, out_dir)


if __name__ == "__main__":
    main()
