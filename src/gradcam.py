"""Grad-CAM for both models on 6 lesion-split test images.

Grad-CAM shows which parts of an image pushed the model towards a class: it
weights a layer's feature maps by how much the class score depends on them.
Each map here explains the class THAT MODEL PREDICTED.

The 6 images are picked from the seed-0 test predictions of BiT (run 1) and
ViT (run 2), one per case (the most confident example of each):
    1. melanoma, both models correct
    2. melanoma, both models wrong
    3. melanoma, the models disagree
    4. benign lesion that at least one model calls melanoma
    5. other benign lesion (not nv), both correct
    6. nv, both correct
If a case has no example, another test image is used and marked "fallback".

Layers:
    ViT: the last block's first LayerNorm. Its output is a sequence of tokens
         (1 class token + 14x14 patch tokens); we drop the class token and
         reshape the patches back into a 14x14 grid.
    BiT: the final normalisation + ReLU after the last stage (7x7 feature
         maps; BiT is a "pre-activation" ResNet, so this is where the last
         stage's features are actually finished).

Output: <runs_root>/figures/gradcam.png and gradcam_images.csv (which images, why).

Usage (Colab):
    python src/gradcam.py --runs_root /content/drive/MyDrive/ham10000_runs \
                          --image_dir /content/data/images_256
    python src/gradcam.py --smoke
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data import CLASSES, MALIGNANT
from dataset import eval_transform, get_normalization
from models import MODEL_NAMES, create_model

MEL = CLASSES.index("mel")
NV = CLASSES.index("nv")
MAL_LABELS = [CLASSES.index(c) for c in MALIGNANT]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


# ----------------------------------------------------------------------------
# Picking the 6 images
# ----------------------------------------------------------------------------

def load_pair(bit_dir, vit_dir):
    """Both models' test predictions side by side, one row per image."""
    keep = ["image_id", "lesion_id", "y_true", "y_pred"] + [f"prob_{c}" for c in CLASSES]
    bit = pd.read_csv(os.path.join(bit_dir, "test_predictions.csv"))[keep]
    vit = pd.read_csv(os.path.join(vit_dir, "test_predictions.csv"))[keep]
    df = bit.merge(vit, on=["image_id", "lesion_id", "y_true"], suffixes=("_bit", "_vit"))
    assert len(df) == len(bit) == len(vit), "the two runs have different test images"
    for m in ["bit", "vit"]:
        probs = df[[f"prob_{c}_{m}" for c in CLASSES]].to_numpy()
        df[f"conf_{m}"] = probs.max(axis=1)  # probability of the predicted class
    df["conf"] = (df["conf_bit"] + df["conf_vit"]) / 2
    return df


def pick_images(df):
    """One image per case (most confident example); fallbacks if a case is empty."""
    t, b, v = df["y_true"], df["y_pred_bit"], df["y_pred_vit"]
    cases = [
        ("melanoma, both correct", (t == MEL) & (b == MEL) & (v == MEL)),
        ("melanoma, both wrong", (t == MEL) & (b != MEL) & (v != MEL)),
        ("melanoma, models disagree", (t == MEL) & ((b == MEL) != (v == MEL))),
        ("benign called melanoma", ~t.isin(MAL_LABELS) & ((b == MEL) | (v == MEL))),
        ("other benign (not nv), both correct", ~t.isin(MAL_LABELS) & (t != NV) & (b == t) & (v == t)),
        ("nv, both correct", (t == NV) & (b == NV) & (v == NV)),
    ]
    chosen, used = [], set()
    for name, mask in cases:
        options = df[mask & ~df["image_id"].isin(used)].sort_values("conf", ascending=False)
        if len(options) == 0:  # no example of this case: use the most confident unused image
            options = df[~df["image_id"].isin(used)].sort_values("conf", ascending=False)
            name = f"{name} (fallback: no example)"
        row = options.iloc[0]
        used.add(row["image_id"])
        chosen.append({"case": name, **row.to_dict()})
    return pd.DataFrame(chosen)


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------

def vit_reshape(model):
    """Turn ViT tokens (B, 1 + 196, C) into feature maps (B, C, 14, 14)."""
    def reshape(tokens):
        patches = tokens[:, model.num_prefix_tokens:, :]  # drop the class token
        side = int(patches.shape[1] ** 0.5)               # 196 patches -> 14 x 14
        grid = patches.reshape(patches.shape[0], side, side, patches.shape[2])
        return grid.permute(0, 3, 1, 2)                   # channels first, like a CNN
    return reshape


def load_model(key, run_dir, device):
    """Build the model and load its best checkpoint."""
    model = create_model(key, pretrained=False)
    state = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def make_cam(key, model):
    if key == "vit":
        return GradCAM(model=model, target_layers=[model.blocks[-1].norm1],
                       reshape_transform=vit_reshape(model))
    return GradCAM(model=model, target_layers=[model.norm])


def compute_cams(key, run_dir, picks, image_dir, device):
    """Grad-CAM map (224x224, values 0-1) for each picked image and this model's prediction."""
    model = load_model(key, run_dir, device)
    cam = make_cam(key, model)
    mean, std = get_normalization(MODEL_NAMES[key])
    transform = eval_transform(mean, std)  # the same center crop the model was tested on
    maps = []
    for _, row in picks.iterrows():
        img = Image.open(os.path.join(image_dir, row["image_id"] + ".jpg")).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        target = [ClassifierOutputTarget(int(row[f"y_pred_{key}"]))]
        maps.append(cam(input_tensor=x, targets=target)[0])
    return maps


def cropped_image(image_dir, image_id):
    """The 224x224 center crop as an RGB array in [0, 1], for display."""
    img = Image.open(os.path.join(image_dir, image_id + ".jpg")).convert("RGB")
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(img.crop((left, top, left + 224, top + 224))) / 255.0


def plot(picks, cams, image_dir, out_path):
    """6 rows (images) x 3 columns (image, BiT Grad-CAM, ViT Grad-CAM)."""
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.2, 2.5 * len(picks)), facecolor=SURFACE)
    for i, (_, row) in enumerate(picks.iterrows()):
        image = cropped_image(image_dir, row["image_id"])
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"{row['case']}\ntrue: {CLASSES[int(row['y_true'])]}",
                             fontsize=8, loc="left", color=INK)
        for j, (key, name) in enumerate([("bit", "BiT"), ("vit", "ViT")], start=1):
            pred = int(row[f"y_pred_{key}"])
            ok = "correct" if pred == row["y_true"] else "wrong"
            axes[i, j].imshow(image)
            # Perceptually uniform colour map: dark = little influence, bright = a lot.
            heat = axes[i, j].imshow(cams[key][i], cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            axes[i, j].set_title(f"{name}: {CLASSES[pred]} ({row[f'conf_{key}']:.0%}, {ok})",
                                 fontsize=8, loc="left", color=INK if ok == "correct" else "#d03b3b")
        for ax in axes[i]:
            ax.axis("off")
    fig.colorbar(heat, ax=axes, shrink=0.3, label="Grad-CAM (for the predicted class)")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


# ----------------------------------------------------------------------------
# Smoke test data
# ----------------------------------------------------------------------------

def make_smoke_inputs():
    """Fake images, random-weight checkpoints and fake test predictions for runs 1 and 2."""
    import data
    import fake_data
    import preprocess
    import splits

    work = "data_fake"
    fake_data.generate(work, n_lesions=120, per_class=12)
    meta = data.build_metadata(work)
    preprocess.cache_images(meta, os.path.join(work, "images_256"))
    test = splits.lesion_level_split(meta, 42)
    test = test[test["split"] == "test"].reset_index(drop=True)

    runs_root = os.path.join(work, "runs_gradcam")
    for run_name, key, seed in [("run1_bit_lesion_standard_seed0", "bit", 1),
                                ("run2_vit_lesion_standard_seed0", "vit", 2)]:
        folder = os.path.join(runs_root, run_name)
        os.makedirs(folder, exist_ok=True)
        torch.save(create_model(key, pretrained=False).state_dict(), os.path.join(folder, "best.pt"))
        rng = np.random.default_rng(seed)
        probs = rng.random((len(test), len(CLASSES)))
        probs[np.arange(len(test)), test["label"]] += 0.6
        probs /= probs.sum(axis=1, keepdims=True)
        preds = pd.DataFrame({"image_id": test["image_id"], "lesion_id": test["lesion_id"],
                              "y_true": test["label"], "y_pred": probs.argmax(axis=1)})
        for k, c in enumerate(CLASSES):
            preds[f"prob_{c}"] = probs[:, k]
        preds.to_csv(os.path.join(folder, "test_predictions.csv"), index=False)
    return runs_root, os.path.join(work, "images_256")


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for BiT and ViT on 6 test images.")
    parser.add_argument("--runs_root", default="/content/drive/MyDrive/ham10000_runs")
    parser.add_argument("--image_dir", default="/content/data/images_256")
    parser.add_argument("--bit_run", default="run1_bit_lesion_standard_seed0")
    parser.add_argument("--vit_run", default="run2_vit_lesion_standard_seed0")
    parser.add_argument("--out_dir", default=None, help="default: <runs_root>/figures")
    parser.add_argument("--smoke", action="store_true", help="fake data and random weights")
    args = parser.parse_args()

    if args.smoke:
        args.runs_root, args.image_dir = make_smoke_inputs()

    out_dir = args.out_dir or os.path.join(args.runs_root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bit_dir = os.path.join(args.runs_root, args.bit_run)
    vit_dir = os.path.join(args.runs_root, args.vit_run)

    picks = pick_images(load_pair(bit_dir, vit_dir))
    cols = ["case", "image_id", "lesion_id", "y_true", "y_pred_bit", "conf_bit", "y_pred_vit", "conf_vit"]
    picks[cols].to_csv(os.path.join(out_dir, "gradcam_images.csv"), index=False)
    print(picks[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    cams = {"bit": compute_cams("bit", bit_dir, picks, args.image_dir, device),
            "vit": compute_cams("vit", vit_dir, picks, args.image_dir, device)}
    plot(picks, cams, args.image_dir, os.path.join(out_dir, "gradcam.png"))


if __name__ == "__main__":
    main()
