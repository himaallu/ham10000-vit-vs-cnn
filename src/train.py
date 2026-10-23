"""Fine-tune one model on one split, driven by a YAML config.

Config keys (see configs/*.yaml):
    run_name       name of the output folder
    model          vit | bit
    split          lesion | image          (reads split_<split>.csv)
    mode           standard        class-weighted cross-entropy, shuffled batches
                   balanced        WeightedRandomSampler over diagnosis x age band
                                   x site group, plain cross-entropy
                   class_balanced  WeightedRandomSampler over diagnosis only,
                                   plain cross-entropy (run 5b)
    seed, lr, weight_decay, epochs, warmup_epochs, batch_size, patience,
    min_cell_size, num_workers, data_dir, split_dir, out_root

Training: AdamW, linear warmup then cosine decay (per batch), mixed precision
on GPU. After each epoch we compute val macro-F1; the best epoch's weights are
kept (best.pt) and training stops early after `patience` epochs without
improvement. last.pt is saved every epoch, and rerunning the same command
resumes from it (useful when Colab disconnects).

At the end the best weights are loaded and the test set is predicted.

Output folder <out_root>/<run_name>/:
    config.yaml          copy of the config used
    best.pt              weights of the best epoch (by val macro-F1)
    last.pt              full training state, for resuming
    history.csv          one row per epoch: losses, val macro-F1, lr
    test_predictions.csv image_id, lesion_id, y_true, y_pred, prob_<class> x7,
                         age_band, sex, site_group, seen_lesion
    summary.json         best epoch, best val macro-F1, test macro-F1, mel recall

Age, sex and site are NEVER model inputs. They are only used to build the
balanced sampler and copied into the predictions CSV for reporting.

Usage (Colab):
    python src/train.py --config configs/run1_seed0.yaml
    python src/train.py --config configs/run5_seed0.yaml --smoke   # CPU, fake data
"""

import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import CLASSES
from dataset import build_datasets
from models import MODEL_NAMES, NUM_CLASSES, create_model

MODES = ["standard", "balanced", "class_balanced"]
SPLITS = ["lesion", "image"]


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

def load_config(path):
    """Read the YAML config and check its values."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["model"] in MODEL_NAMES, f"model must be one of {list(MODEL_NAMES)}"
    assert cfg["split"] in SPLITS, f"split must be one of {SPLITS}"
    assert cfg["mode"] in MODES, f"mode must be one of {MODES}"
    # YAML can read "1e-4" as text, so make sure numbers really are numbers.
    for key in ["lr", "weight_decay"]:
        cfg[key] = float(cfg[key])
    for key in ["seed", "epochs", "warmup_epochs", "batch_size", "patience",
                "min_cell_size", "num_workers"]:
        cfg[key] = int(cfg[key])
    cfg.setdefault("pretrained", True)
    return cfg


def set_seed(seed):
    """Seed every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the GPU
    # Note: some GPU operations are still not bit-for-bit deterministic;
    # forcing that would make training much slower.


# ----------------------------------------------------------------------------
# Loss and sampler
# ----------------------------------------------------------------------------

def class_weights(train_df):
    """Weight per class = N / (7 * n_class), so rare classes count more.

    A class with n_class = N/7 images gets weight 1.
    """
    counts = train_df["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    counts = counts.clip(lower=1)  # avoid dividing by 0 if a class is missing
    weights = len(train_df) / (NUM_CLASSES * counts)
    return torch.tensor(weights.to_numpy(), dtype=torch.float32)


def sampling_cells(train_df, mode, min_cell_size):
    """Give every training image the name of its sampling cell.

    class_balanced: the cell is just the diagnosis.
    balanced:       the cell is diagnosis x age band x site group. A cell with
                    fewer than min_cell_size images is merged: first site is
                    dropped (dx|age|any); if still too small, age is dropped
                    too (dx|any|any). If even that leftover cell is too small,
                    its images join the biggest cell of the same diagnosis,
                    so no tiny cell is drawn over and over.
    """
    dx, age, site = train_df["dx"], train_df["age_band"], train_df["site_group"]
    if mode == "class_balanced":
        return dx.copy()

    cell = dx + "|" + age + "|" + site
    # Step 1: images in small cells lose their site.
    too_small = cell.map(cell.value_counts()) < min_cell_size
    cell[too_small] = dx[too_small] + "|" + age[too_small] + "|any"
    # Step 2: images still in small cells lose their age band as well.
    too_small = cell.map(cell.value_counts()) < min_cell_size
    cell[too_small] = dx[too_small] + "|any|any"
    # Step 3: a leftover cell that is still too small joins the biggest
    # other cell of its diagnosis (if the diagnosis has another cell).
    for d in dx.unique():
        leftover = cell == d + "|any|any"
        others = cell[(dx == d) & ~leftover]
        if 0 < leftover.sum() < min_cell_size and len(others) > 0:
            cell[leftover] = others.value_counts().idxmax()
    return cell


def sampler_weights(train_df, mode, min_cell_size):
    """Weight per training image for WeightedRandomSampler.

    Each diagnosis gets the same share of the samples (1/7), and inside a
    diagnosis each cell gets the same share. Inside a cell all images are
    equally likely. So:
        weight(image) = 1 / (n_classes * cells_in_its_dx * images_in_its_cell)
    """
    cell = sampling_cells(train_df, mode, min_cell_size)
    images_in_cell = cell.map(cell.value_counts())
    cells_in_dx = train_df["dx"].map(cell.groupby(train_df["dx"]).nunique())
    n_classes = train_df["dx"].nunique()
    weights = 1.0 / (n_classes * cells_in_dx * images_in_cell)

    # Short report: how many cells each diagnosis ended up with.
    print(f"Sampler ({mode}): {cell.nunique()} cells")
    report = pd.DataFrame({"dx": train_df["dx"], "cell": cell})
    per_dx = report.groupby("dx")["cell"].agg(cells="nunique")
    per_dx["smallest_cell"] = report.groupby(["dx", "cell"]).size().groupby("dx").min()
    print(per_dx.reindex(CLASSES).to_string())
    return torch.tensor(weights.to_numpy(), dtype=torch.double)


# ----------------------------------------------------------------------------
# Training and prediction
# ----------------------------------------------------------------------------

def save_safely(obj, path):
    """torch.save to a temporary file, then rename it to `path`.

    If Colab disconnects in the middle of saving, only the temporary file is
    broken; the previous checkpoint at `path` is still complete.
    """
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)  # the rename is all-or-nothing


def make_lr_lambda(warmup_steps, total_steps):
    """Learning-rate multiplier per batch: linear warmup, then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps                # 1/warmup ... 1
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))     # 1 ... 0
    return lr_lambda


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp):
    """One pass over the training data. Returns the mean training loss."""
    model.train()
    total_loss, n_images = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # Mixed precision: run the forward pass in float16 where it is safe
        # (faster, less memory). Turned off on CPU.
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # The scaler multiplies the loss before backward so small float16
        # gradients don't round to 0, then undoes it before the step.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # the learning rate changes every batch

        total_loss += loss.item() * len(labels)
        n_images += len(labels)
    return total_loss / n_images


@torch.no_grad()  # no gradients needed: faster and uses less memory
def predict(model, loader, device, use_amp):
    """Return (probabilities [N, 7], true labels [N], mean plain CE loss)."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images.to(device))
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    loss = F.cross_entropy(logits, labels).item()
    probs = torch.softmax(logits, dim=1).numpy()
    return probs, labels.numpy(), loss


def macro_f1(y_true, y_pred):
    """Macro-F1 over all 7 classes (a class never predicted counts as F1 = 0)."""
    return f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def prepare_smoke(cfg):
    """Make tiny fake data and shrink the run so it finishes quickly on CPU."""
    import data
    import fake_data
    import preprocess
    import splits

    work = "data_fake"
    fake_data.generate(work)
    meta = data.build_metadata(work)
    meta.to_csv(os.path.join(work, "metadata.csv"), index=False)
    splits.lesion_level_split(meta, seed=42).to_csv(os.path.join(work, "split_lesion.csv"), index=False)
    splits.image_level_split(meta, seed=42).to_csv(os.path.join(work, "split_image.csv"), index=False)
    preprocess.cache_images(meta, os.path.join(work, "images_256"))

    cfg.update(data_dir=work, split_dir=work, out_root=os.path.join(work, "runs"),
               pretrained=False,  # random weights: no download
               epochs=2, warmup_epochs=1, batch_size=4, num_workers=0,
               min_cell_size=2)   # small enough that some cells survive merging
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Fine-tune one model from a YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny fake data, random weights, 2 epochs on CPU")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = prepare_smoke(cfg)

    out_dir = os.path.join(cfg["out_root"], cfg["run_name"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Run {cfg['run_name']} -> {out_dir}")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}, mixed precision: {use_amp}")

    # ---- Data ----
    split_df = pd.read_csv(os.path.join(cfg["split_dir"], f"split_{cfg['split']}.csv"))
    image_dir = os.path.join(cfg["data_dir"], "images_256")
    datasets = build_datasets(split_df, image_dir, MODEL_NAMES[cfg["model"]])
    train_df = datasets["train"].df
    print(f"Images: train {len(datasets['train'])}, val {len(datasets['val'])}, "
          f"test {len(datasets['test'])}")

    # A seeded generator makes the shuffling/sampling order repeatable.
    g = torch.Generator().manual_seed(cfg["seed"])
    if cfg["mode"] == "standard":
        # Class-weighted CE; batches are a normal shuffle of the training set.
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_df).to(device))
        train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True,
                                  num_workers=cfg["num_workers"], generator=g)
    else:
        # Plain CE; the sampler does the balancing. It draws len(train) images
        # per epoch WITH replacement, so rare cells appear several times.
        criterion = torch.nn.CrossEntropyLoss()
        weights = sampler_weights(train_df, cfg["mode"], cfg["min_cell_size"])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True,
                                        generator=g)
        train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], sampler=sampler,
                                  num_workers=cfg["num_workers"])
    val_loader = DataLoader(datasets["val"], batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])
    test_loader = DataLoader(datasets["test"], batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"])

    # ---- Model, optimiser, schedule ----
    model = create_model(cfg["model"], pretrained=cfg["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(cfg["warmup_epochs"] * steps_per_epoch, cfg["epochs"] * steps_per_epoch))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # ---- Resume if a previous run of this config was interrupted ----
    last_path = os.path.join(out_dir, "last.pt")
    best_path = os.path.join(out_dir, "best.pt")
    start_epoch, best_f1, best_epoch, bad_epochs, stopped, history = 0, -1.0, -1, 0, False, []
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1, best_epoch = ckpt["best_f1"], ckpt["best_epoch"]
        bad_epochs, stopped, history = ckpt["bad_epochs"], ckpt["stopped"], ckpt["history"]
        # The model, optimiser and schedule continue exactly; the random order
        # of later batches is not restored, so a resumed run is not bit-for-bit
        # identical to an uninterrupted one.
        print(f"Resumed from {last_path}: epoch {start_epoch} next, best val macro-F1 {best_f1:.4f}")

    # ---- Training loop ----
    for epoch in range(start_epoch, cfg["epochs"]):
        if stopped:
            break
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                     scaler, device, use_amp)
        val_probs, val_true, val_loss = predict(model, val_loader, device, use_amp)
        val_f1 = macro_f1(val_true, val_probs.argmax(axis=1))

        # Keep the weights of the epoch with the best val macro-F1.
        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            save_safely(model.state_dict(), best_path)
        else:
            bad_epochs += 1
        stopped = bad_epochs >= cfg["patience"]  # early stopping

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_f1": val_f1, "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val macro-F1 {val_f1:.4f}{'  *best*' if epoch == best_epoch else ''}")

        # Save everything needed to continue from the next epoch.
        save_safely({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch,
                     "bad_epochs": bad_epochs, "stopped": stopped, "history": history},
                    last_path)
        if stopped:
            print(f"Early stopping: no improvement for {cfg['patience']} epochs.")

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # ---- Test predictions with the best weights ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_probs, test_true, _ = predict(model, test_loader, device, use_amp)
    test_pred = test_probs.argmax(axis=1)

    # Row i of the test dataset's table is row i of test_probs (shuffle=False).
    test_df = datasets["test"].df
    preds = pd.DataFrame({
        "image_id": test_df["image_id"],
        "lesion_id": test_df["lesion_id"],
        "y_true": test_true,
        "y_pred": test_pred,
    })
    for k, name in enumerate(CLASSES):
        preds[f"prob_{name}"] = test_probs[:, k]
    for column in ["age_band", "sex", "site_group", "seen_lesion"]:
        preds[column] = test_df[column]
    preds.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    mel = CLASSES.index("mel")
    summary = {
        "run_name": cfg["run_name"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "epochs_trained": len(history),
        "stopped_early": stopped,
        "test_macro_f1": macro_f1(test_true, test_pred),
        "test_mel_recall": recall_score(test_true == mel, test_pred == mel, zero_division=0),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
