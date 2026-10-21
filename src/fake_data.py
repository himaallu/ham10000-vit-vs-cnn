"""Generate a tiny fake HAM10000 dataset for smoke tests.

The output copies the layout of the real Kaggle download, so the rest of the
code can run on it unchanged:

    <out>/HAM10000_metadata.csv
    <out>/HAM10000_images_part_1/ISIC_xxxxxxx.jpg
    <out>/HAM10000_images_part_2/ISIC_xxxxxxx.jpg

The metadata has the same columns as the real file:
    lesion_id, image_id, dx, dx_type, age, sex, localization

The images are small random-noise JPEGs, so no model can learn anything from
them. They are only there to check that the code runs end to end.

Usage:
    python src/fake_data.py --out data_fake
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

# The 7 HAM10000 diagnosis codes.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Values that appear in the real metadata columns.
DX_TYPES = ["histo", "follow_up", "consensus", "confocal"]
SEXES = ["male", "female", "unknown"]
LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk", "upper extremity",
    "unknown",
]


def make_metadata(n_lesions, rng):
    """Build the fake metadata table: one row per image."""
    # Every class gets at least 4 lesions, so stratified splits work.
    # The remaining lesions are "nv", which is also the majority class in
    # the real data.
    lesion_dx = []
    for c in CLASSES:
        lesion_dx += [c] * 4
    lesion_dx += ["nv"] * (n_lesions - len(lesion_dx))

    rows = []
    image_counter = 0
    for i, dx in enumerate(lesion_dx):
        lesion_id = f"HAM_{i:07d}"

        # Age, sex and site belong to the patient/lesion, so every image of
        # the same lesion shares them (as in the real data).
        # About 10% of ages are missing (NaN), like in the real file.
        if rng.random() < 0.1:
            age = np.nan
        else:
            age = float(rng.choice(np.arange(5, 90, 5)))  # real ages are multiples of 5
        sex = rng.choice(SEXES, p=[0.5, 0.45, 0.05])
        localization = rng.choice(LOCALIZATIONS)
        dx_type = rng.choice(DX_TYPES)

        # Each lesion has 1 to 3 images. With 40 lesions this gives ~60-80 images.
        n_images = rng.choice([1, 2, 3], p=[0.6, 0.25, 0.15])
        for _ in range(n_images):
            rows.append({
                "lesion_id": lesion_id,
                "image_id": f"ISIC_{image_counter:07d}",
                "dx": dx,
                "dx_type": dx_type,
                "age": age,
                "sex": sex,
                "localization": localization,
            })
            image_counter += 1

    df = pd.DataFrame(rows)
    # Shuffle the rows so images of one lesion are not always next to each other.
    df = df.sample(frac=1, random_state=int(rng.integers(1_000_000)))
    return df.reset_index(drop=True)


def write_images(df, out_dir, rng, width, height):
    """Save one random-noise JPEG per image_id, split over two folders like Kaggle."""
    part1 = os.path.join(out_dir, "HAM10000_images_part_1")
    part2 = os.path.join(out_dir, "HAM10000_images_part_2")
    os.makedirs(part1, exist_ok=True)
    os.makedirs(part2, exist_ok=True)

    for k, image_id in enumerate(df["image_id"]):
        # Random RGB pixels, shape (height, width, 3), values 0-255.
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        folder = part1 if k % 2 == 0 else part2  # alternate between the two folders
        Image.fromarray(pixels).save(os.path.join(folder, image_id + ".jpg"))


def main():
    parser = argparse.ArgumentParser(description="Generate a tiny fake HAM10000.")
    parser.add_argument("--out", default="data_fake", help="output folder")
    parser.add_argument("--n_lesions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    # Real images are 600x450; small ones keep the smoke test fast.
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    args = parser.parse_args()

    if args.n_lesions < 4 * len(CLASSES):
        parser.error(f"--n_lesions must be at least {4 * len(CLASSES)}")

    # One random generator, seeded, so the fake data is the same every run.
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out, exist_ok=True)
    df = make_metadata(args.n_lesions, rng)
    df.to_csv(os.path.join(args.out, "HAM10000_metadata.csv"), index=False)
    write_images(df, args.out, rng, args.width, args.height)

    print(f"Wrote {len(df)} images from {df['lesion_id'].nunique()} lesions to {args.out}")
    print(df["dx"].value_counts().to_string())


if __name__ == "__main__":
    main()
