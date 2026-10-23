"""PyTorch Dataset, image transforms and an augmentation preview.

- Train images get random augmentation, on the fly (a new random version
  every time an image is loaded): rotation, random resized crop to 224,
  horizontal + vertical flips, mild colour jitter.
- Val/test images get no randomness: center crop to 224.
- Normalisation (mean/std) comes from each timm model's own pretrained config.

The Dataset returns ONLY (image, label). Age, sex and site are never model
inputs; they stay in the split CSV for sampling and reporting.

Expects images already resized by preprocess.py (short side 256).

Usage (Colab):
    python src/dataset.py --split_csv /content/data/split_lesion.csv --model vit
    python src/dataset.py --smoke
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # save figures to files, no screen needed
import matplotlib.pyplot as plt
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# The two models we compare, as timm names. Both are pre-trained on
# ImageNet-21k only (no ImageNet-1k fine-tuning).
MODEL_NAMES = {
    "vit": "vit_base_patch16_224.augreg_in21k",  # ViT-B/16
    "bit": "resnetv2_50x1_bit.goog_in21k",       # BiT ResNet-50 (ResNetV2-50x1)
}

CROP_SIZE = 224


def get_normalization(model_name):
    """Return (mean, std) from a timm model's pretrained config.

    Reads the config only; no weights are downloaded.
    """
    cfg = timm.get_pretrained_cfg(model_name)
    return cfg.mean, cfg.std


def train_transform(mean, std):
    """Random augmentation for training images."""
    return transforms.Compose([
        # Lesions have no "up", so any rotation is realistic. Small black
        # corners can appear; the crop below removes most of them.
        transforms.RandomRotation(degrees=30),
        # Take a random part covering 60-100% of the image and resize it to
        # 224x224. Not smaller than 60% so the lesion stays mostly in view.
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # Mild colour changes (different cameras/lighting). Hue is kept very
        # small because colour itself helps the diagnosis.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),             # PIL image -> float tensor in [0, 1], shape (3, H, W)
        transforms.Normalize(mean, std),   # (x - mean) / std, per channel
    ])


def eval_transform(mean, std):
    """No randomness for val/test images: always the same center crop."""
    return transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class HAM10000Dataset(Dataset):
    """Images of one partition (train, val or test) of a split CSV.

    Row i of self.df is item i of the dataset, so predictions made with
    shuffle=False can be matched back to image_id, lesion_id, age_band, ...
    """

    def __init__(self, split_df, partition, image_dir, transform):
        # Keep only this partition's rows; reset the index so rows are 0..N-1.
        self.df = split_df[split_df["split"] == partition].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.image_dir, row["image_id"] + ".jpg")).convert("RGB")
        image = self.transform(img)
        label = int(row["label"])
        return image, label  # only the image and its label go to the model


def build_datasets(split_df, image_dir, model_name):
    """Return a dict {"train": ..., "val": ..., "test": ...} for one model."""
    mean, std = get_normalization(model_name)
    return {
        "train": HAM10000Dataset(split_df, "train", image_dir, train_transform(mean, std)),
        "val": HAM10000Dataset(split_df, "val", image_dir, eval_transform(mean, std)),
        "test": HAM10000Dataset(split_df, "test", image_dir, eval_transform(mean, std)),
    }


def save_augmentation_grid(train_dataset, out_path, n=8, seed=0):
    """Save a 2x4 grid of n augmented training images, titled by diagnosis."""
    # Get mean/std back from the dataset's Normalize step so we can undo it.
    normalize = train_dataset.transform.transforms[-1]
    mean = torch.tensor(normalize.mean).view(3, 1, 1)
    std = torch.tensor(normalize.std).view(3, 1, 1)

    # Pick n random training images (fixed seed so the same images are shown).
    g = torch.Generator().manual_seed(seed)
    picks = torch.randperm(len(train_dataset), generator=g)[:n].tolist()

    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6.5))
    for ax, i in zip(axes.flat, picks):
        image, _ = train_dataset[i]                      # a fresh random augmentation
        image = (image * std + mean).clamp(0, 1)         # undo normalisation -> [0, 1]
        ax.imshow(image.permute(1, 2, 0).numpy())        # (3, H, W) -> (H, W, 3) for matplotlib
        ax.set_title(train_dataset.df.loc[i, "dx"])
        ax.axis("off")
    for ax in list(axes.flat)[len(picks):]:              # hide unused panels if n is odd
        ax.axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved augmentation grid to {out_path}")


def check_loaders(datasets, batch_size):
    """Load one batch from each partition and print its shape."""
    for part, ds in datasets.items():
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(part == "train"))
        images, labels = next(iter(loader))
        print(f"  {part:5s}: {len(ds):5d} images | batch {tuple(images.shape)} | "
              f"pixel range [{images.min():.2f}, {images.max():.2f}] | labels {labels.tolist()}")


def run_smoke(work_dir):
    """Fake data -> metadata -> lesion split -> cache -> datasets -> grid."""
    import data
    import fake_data
    import preprocess
    import splits

    fake_data.generate(work_dir)
    meta = data.build_metadata(work_dir)
    split_df = splits.lesion_level_split(meta, seed=42)
    image_dir = os.path.join(work_dir, "images_256")
    preprocess.cache_images(meta, image_dir)

    for key, model_name in MODEL_NAMES.items():
        mean, std = get_normalization(model_name)
        print(f"\n{key}: {model_name}  mean={mean}  std={std}")
        datasets = build_datasets(split_df, image_dir, model_name)
        check_loaders(datasets, batch_size=4)

    datasets = build_datasets(split_df, image_dir, MODEL_NAMES["vit"])
    save_augmentation_grid(datasets["train"], os.path.join(work_dir, "augmentation_grid.png"))


def main():
    parser = argparse.ArgumentParser(description="Check the datasets and save an augmentation grid.")
    parser.add_argument("--split_csv", default="/content/data/split_lesion.csv")
    parser.add_argument("--image_dir", default="/content/data/images_256",
                        help="resized images from preprocess.py")
    parser.add_argument("--model", choices=list(MODEL_NAMES), default="vit")
    parser.add_argument("--grid_out", default="results/augmentation_grid.png")
    parser.add_argument("--smoke", action="store_true",
                        help="run the whole chain on tiny fake data in data_fake/")
    args = parser.parse_args()

    if args.smoke:
        run_smoke("data_fake")
        return

    split_df = pd.read_csv(args.split_csv)
    model_name = MODEL_NAMES[args.model]
    mean, std = get_normalization(model_name)
    print(f"{model_name}  mean={mean}  std={std}")
    datasets = build_datasets(split_df, args.image_dir, model_name)
    check_loaders(datasets, batch_size=8)
    save_augmentation_grid(datasets["train"], args.grid_out)


if __name__ == "__main__":
    main()
