"""Download HAM10000 and build a clean metadata.csv.

Steps:
  1. Download the Kaggle dataset kmader/skin-cancer-mnist-ham10000 into the
     data folder (skipped if it is already there).
  2. Map each image_id to ONE file path. The Kaggle download contains every
     image twice (HAM10000_images_part_* and ham10000_images_part_*), so we
     keep a single copy.
  3. Add the columns we need later and save <data_dir>/metadata.csv:
       image_id, path, dx, label, lesion_id, age, age_band, sex,
       site_group, malignant
  4. Print counts and a table of group sizes.

Age, sex and site are kept ONLY for sampling and reporting. They are never
model inputs.

Usage (Colab):
    python src/data.py                 # real data into /content/data
    python src/data.py --smoke         # tiny fake data, no download
"""

import argparse
import glob
import os

import pandas as pd

KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"
RAW_METADATA = "HAM10000_metadata.csv"

# The 7 diagnosis codes in alphabetical order. A class's position in this
# list is its label: akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Classes counted as malignant in the malignant-vs-benign analysis.
MALIGNANT = ["mel", "bcc", "akiec"]

# Map the 15 HAM10000 'localization' values to 5 site groups.
# "acral" (palms/soles) does not say hand or foot, so it goes to "other".
SITE_GROUPS = {
    "face": "head_neck", "scalp": "head_neck", "ear": "head_neck", "neck": "head_neck",
    "back": "trunk", "chest": "trunk", "abdomen": "trunk", "trunk": "trunk",
    "upper extremity": "upper_limb", "hand": "upper_limb",
    "lower extremity": "lower_limb", "foot": "lower_limb",
    "acral": "other", "genital": "other", "unknown": "other",
}

# Order used when printing groups.
AGE_BANDS = ["<40", "40-59", "60+", "unknown"]
SITE_ORDER = ["head_neck", "trunk", "upper_limb", "lower_limb", "other"]


def download(data_dir):
    """Download and unzip the Kaggle dataset, unless it is already there."""
    if os.path.exists(os.path.join(data_dir, RAW_METADATA)):
        print(f"Found {RAW_METADATA} in {data_dir}, skipping download.")
        return

    # Imported here so --smoke works without Kaggle credentials.
    # The API reads your key from ~/.kaggle/kaggle.json (or the
    # KAGGLE_USERNAME / KAGGLE_KEY environment variables).
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} into {data_dir} ...")
    os.makedirs(data_dir, exist_ok=True)
    api.dataset_download_files(KAGGLE_DATASET, path=data_dir, unzip=True, quiet=False)


def find_image_paths(data_dir):
    """Return a dict {image_id: path}, with exactly one path per image_id."""
    # Find every .jpg inside a *images_part_* folder (at any depth) under
    # data_dir. Other folders, like the images_256 cache from preprocess.py,
    # are ignored. Sorting makes the choice of copy the same on every run.
    pattern = os.path.join(data_dir, "**", "*images_part_*", "*.jpg")
    all_paths = sorted(glob.glob(pattern, recursive=True))

    paths = {}
    for p in all_paths:
        image_id = os.path.splitext(os.path.basename(p))[0]  # "ISIC_0024306"
        if image_id not in paths:  # keep the first copy, ignore the duplicates
            paths[image_id] = p

    print(f"Found {len(all_paths)} .jpg files -> {len(paths)} unique image_ids")
    return paths


def age_to_band(age):
    """Turn an age in years (or NaN) into one of the 4 age bands."""
    # Age 0 in HAM10000 almost certainly means "not recorded", not a newborn,
    # so it counts as unknown. The raw age column is left unchanged.
    if pd.isna(age) or age == 0:
        return "unknown"
    if age < 40:
        return "<40"
    if age < 60:
        return "40-59"
    return "60+"


def build_metadata(data_dir):
    """Read the raw Kaggle metadata and return our clean table."""
    raw = pd.read_csv(os.path.join(data_dir, RAW_METADATA))

    # Attach one file path to every image; stop if any image is missing.
    paths = find_image_paths(data_dir)
    raw["path"] = raw["image_id"].map(paths)
    missing = raw["path"].isna().sum()
    if missing > 0:
        raise FileNotFoundError(f"{missing} image_ids in {RAW_METADATA} have no .jpg file")

    # Stop if the metadata has a value we have no mapping for.
    unknown_dx = set(raw["dx"]) - set(CLASSES)
    unknown_site = set(raw["localization"]) - set(SITE_GROUPS)
    if unknown_dx or unknown_site:
        raise ValueError(f"Unexpected values: dx={unknown_dx}, localization={unknown_site}")

    df = pd.DataFrame({
        "image_id": raw["image_id"],
        "path": raw["path"],
        "dx": raw["dx"],
        "label": raw["dx"].map(CLASSES.index),
        "lesion_id": raw["lesion_id"],
        "age": raw["age"],
        "age_band": raw["age"].map(age_to_band),
        # Missing sex values become "unknown" (the raw file already uses it).
        "sex": raw["sex"].fillna("unknown"),
        "site_group": raw["localization"].map(SITE_GROUPS),
        "malignant": raw["dx"].isin(MALIGNANT).astype(int),
    })
    return df


def group_table(df, column, order):
    """Count images, lesions and malignant images for each value of one column."""
    table = df.groupby(column).agg(
        images=("image_id", "count"),
        lesions=("lesion_id", "nunique"),
        malignant_images=("malignant", "sum"),
    )
    # Show groups in a fixed order, plus any value not in that order.
    order = [v for v in order if v in table.index] + [v for v in table.index if v not in order]
    return table.loc[order]


def print_summary(df):
    """Print overall counts and the group-size tables."""
    print("\n=== Overall ===")
    print(f"images:  {len(df)}")
    print(f"lesions: {df['lesion_id'].nunique()}")
    print(f"malignant images: {df['malignant'].sum()} ({df['malignant'].mean():.1%})")

    print("\n=== Diagnosis ===")
    print(group_table(df, "dx", CLASSES).to_string())
    print("\n=== Age band ===")
    print(group_table(df, "age_band", AGE_BANDS).to_string())
    print("\n=== Sex ===")
    print(group_table(df, "sex", ["male", "female", "unknown"]).to_string())
    print("\n=== Site group ===")
    print(group_table(df, "site_group", SITE_ORDER).to_string())

    # Images per diagnosis x age band x site group cell. These are the cells
    # the balanced sampler will use, so small cells matter.
    print("\n=== Images per diagnosis x age band ===")
    print(pd.crosstab(df["dx"], df["age_band"]).reindex(columns=AGE_BANDS, fill_value=0).to_string())
    print("\n=== Images per diagnosis x site group ===")
    print(pd.crosstab(df["dx"], df["site_group"]).reindex(columns=SITE_ORDER, fill_value=0).to_string())


def main():
    parser = argparse.ArgumentParser(description="Download HAM10000 and build metadata.csv.")
    parser.add_argument("--data_dir", default="/content/data")
    parser.add_argument("--smoke", action="store_true",
                        help="use tiny fake data instead of downloading (no Kaggle key needed)")
    args = parser.parse_args()

    if args.smoke:
        import fake_data  # src/fake_data.py
        if args.data_dir == "/content/data":
            args.data_dir = "data_fake"  # never write fake data over the real folder
        fake_data.generate(args.data_dir)
        print(f"Smoke mode: generated fake data in {args.data_dir}")
    else:
        download(args.data_dir)

    df = build_metadata(args.data_dir)
    out_path = os.path.join(args.data_dir, "metadata.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    print_summary(df)


if __name__ == "__main__":
    main()
