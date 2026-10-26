"""Sample outputs: 8 lesion-split test images with both models' predictions.

A grid of 8 test images (2 rows x 4), each with its true label, age band, and
the predicted label and confidence of ViT (run 2) and BiT (run 1), seed 0.
Wrong predictions are shown in red. The 8 images cover these cases:
    1. melanoma, both correct
    2. melanoma, both wrong
    3. melanoma, the models disagree
    4. malignant lesion in a patient under 40
    5. bcc or akiec that at least one model gets wrong
    6. benign lesion that at least one model calls melanoma
    7. nv, both correct
    8. other benign lesion (bkl, df, vasc), both correct
Inside a case one image is drawn at random (seed 0), so the examples are typical
rather than the most confident ones. Images already used for Grad-CAM, and a
second image of an already chosen lesion, are skipped. If a case has no
example, a random unused image is shown and marked "fallback".

Output: <runs_root>/figures/sample_outputs.png and sample_outputs.csv.

Usage (Colab):
    python src/sample_outputs.py --runs_root /content/drive/MyDrive/ham10000_runs \
                                 --image_dir /content/data/images_256
    python src/sample_outputs.py --smoke
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CLASSES, MALIGNANT
from gradcam import cropped_image, load_pair

MEL = CLASSES.index("mel")
NV = CLASSES.index("nv")
MAL_LABELS = [CLASSES.index(c) for c in MALIGNANT]
OTHER_BENIGN = [CLASSES.index(c) for c in ["bkl", "df", "vasc"]]
SURFACE, INK, INK_2, WRONG = "#fcfcfb", "#0b0b0b", "#52514e", "#d03b3b"


def pick_images(df, skip_ids, seed=0):
    """One random image per case; returns a table with a 'case' column."""
    t, b, v = df["y_true"], df["y_pred_bit"], df["y_pred_vit"]
    malignant = t.isin(MAL_LABELS)
    cases = [
        ("melanoma, both correct", (t == MEL) & (b == MEL) & (v == MEL)),
        ("melanoma, both wrong", (t == MEL) & (b != MEL) & (v != MEL)),
        ("melanoma, models disagree", (t == MEL) & ((b == MEL) != (v == MEL))),
        ("malignant, patient under 40", malignant & (df["age_band"] == "<40")),
        ("bcc/akiec, at least one wrong", t.isin([CLASSES.index("bcc"), CLASSES.index("akiec")])
         & ((b != t) | (v != t))),
        ("benign called melanoma", ~malignant & ((b == MEL) | (v == MEL))),
        ("nv, both correct", (t == NV) & (b == NV) & (v == NV)),
        ("other benign, both correct", t.isin(OTHER_BENIGN) & (b == t) & (v == t)),
    ]
    rng = np.random.default_rng(seed)
    used_images, used_lesions, chosen = set(skip_ids), set(), []
    for name, mask in cases:
        free = ~df["image_id"].isin(used_images) & ~df["lesion_id"].isin(used_lesions)
        options = df[mask & free]
        if len(options) == 0:  # no example of this case: any unused image
            options = df[free]
            name = f"{name}\n(fallback: no example)"  # own line, so titles never collide
        row = options.iloc[rng.integers(len(options))]
        used_images.add(row["image_id"])
        used_lesions.add(row["lesion_id"])
        chosen.append({"case": name, **row.to_dict()})
    return pd.DataFrame(chosen)


def plot(picks, image_dir, out_path):
    """2 x 4 grid: image, true label and age band, then each model's prediction."""
    fig, axes = plt.subplots(2, 4, figsize=(10, 6.6), facecolor=SURFACE)
    for ax, (_, row) in zip(axes.ravel(), picks.iterrows()):
        ax.imshow(cropped_image(image_dir, row["image_id"]))
        ax.axis("off")
        true = int(row["y_true"])
        ax.set_title(f"{row['case']}\ntrue: {CLASSES[true]} · age {row['age_band']}",
                     fontsize=8, loc="left", color=INK)
        # Predictions as text under the image; wrong ones in red.
        for k, (key, name) in enumerate([("vit", "ViT"), ("bit", "BiT")]):
            pred = int(row[f"y_pred_{key}"])
            mark = "correct" if pred == true else "wrong"
            ax.text(0, -0.06 - 0.08 * k, f"{name}: {CLASSES[pred]} ({row[f'conf_{key}']:.0%}, {mark})",
                    transform=ax.transAxes, fontsize=8, va="top",
                    color=INK_2 if pred == true else WRONG)
    fig.suptitle("Sample test outputs (lesion split, seed 0): true label vs ViT-B/16 and BiT ResNet-50",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08, hspace=0.55, wspace=0.08)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_smoke_inputs():
    """Fake runs 1 and 2 (from gradcam.py's smoke set-up) plus the age band column."""
    import data
    import gradcam

    runs_root, image_dir = gradcam.make_smoke_inputs()
    meta = data.build_metadata("data_fake")[["image_id", "age_band"]]
    for run in ["run1_bit_lesion_standard_seed0", "run2_vit_lesion_standard_seed0"]:
        path = os.path.join(runs_root, run, "test_predictions.csv")
        preds = pd.read_csv(path)
        preds.drop(columns="age_band", errors="ignore").merge(meta, on="image_id").to_csv(path, index=False)
    return runs_root, image_dir


def main():
    parser = argparse.ArgumentParser(description="Grid of 8 test images with ViT and BiT predictions.")
    parser.add_argument("--runs_root", default="/content/drive/MyDrive/ham10000_runs")
    parser.add_argument("--image_dir", default="/content/data/images_256")
    parser.add_argument("--bit_run", default="run1_bit_lesion_standard_seed0")
    parser.add_argument("--vit_run", default="run2_vit_lesion_standard_seed0")
    parser.add_argument("--out_dir", default=None, help="default: <runs_root>/figures")
    parser.add_argument("--smoke", action="store_true", help="fake data and fake predictions")
    args = parser.parse_args()

    if args.smoke:
        args.runs_root, args.image_dir = make_smoke_inputs()
    out_dir = args.out_dir or os.path.join(args.runs_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    df = load_pair(os.path.join(args.runs_root, args.bit_run), os.path.join(args.runs_root, args.vit_run))
    # load_pair keeps only the prediction columns; add the age band back.
    ages = pd.read_csv(os.path.join(args.runs_root, args.vit_run, "test_predictions.csv"))
    df = df.merge(ages[["image_id", "age_band"]], on="image_id")

    gradcam_csv = os.path.join(out_dir, "gradcam_images.csv")
    skip = set(pd.read_csv(gradcam_csv)["image_id"]) if os.path.exists(gradcam_csv) else set()

    picks = pick_images(df, skip)
    cols = ["case", "image_id", "lesion_id", "age_band", "y_true", "y_pred_vit", "conf_vit",
            "y_pred_bit", "conf_bit"]
    out = picks[cols].copy()
    out["case"] = out["case"].str.replace("\n", " ")
    for c in ["y_true", "y_pred_vit", "y_pred_bit"]:
        out[c] = [CLASSES[int(i)] for i in out[c]]
    out.to_csv(os.path.join(out_dir, "sample_outputs.csv"), index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    plot(picks, args.image_dir, os.path.join(out_dir, "sample_outputs.png"))


if __name__ == "__main__":
    main()
