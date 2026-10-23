"""Create the two train/val/test splits and save them to CSV.

(a) Lesion-level split: all images of one lesion go to the same partition,
    so the model is never tested on a lesion it saw during training.
    Stratified by diagnosis (dx). 70/15/15 of the LESIONS in each class.
(b) Image-level split: images are split one by one, ignoring lesion_id.
    Stratified by dx. 70/15/15 of the IMAGES in each class. Different images
    of the same lesion can end up in train and test: this is the "leakage"
    we want to measure.

Both CSVs contain every column of metadata.csv plus:
    split        "train", "val" or "test"
    seen_lesion  True if ANOTHER image of the same lesion is in train

Usage (Colab):
    python src/splits.py                    # reads /content/data/metadata.csv
    python src/splits.py --smoke            # tiny fake data, no download
"""

import argparse
import os

import numpy as np
import pandas as pd

from data import AGE_BANDS, CLASSES, SITE_ORDER

PARTITIONS = ["train", "val", "test"]
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15  # train gets the rest (~70%)


def stratified_split(units, seed):
    """Assign each row of `units` to train/val/test, separately for each dx.

    `units` is a table with a "dx" column: one row per lesion for split (a),
    one row per image for split (b). Returns a Series of partition names with
    the same index as `units`.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=units.index)

    # Doing each class on its own keeps the class mix the same in every
    # partition (this is what "stratified" means).
    for dx in CLASSES:
        idx = units.index[units["dx"] == dx].to_numpy()
        idx = rng.permutation(idx)  # shuffle this class's rows

        n = len(idx)
        # "+ 0.5" then int() rounds half up, e.g. 4 * 0.15 = 0.6 -> 1.
        n_val = int(n * VAL_FRACTION + 0.5)
        n_test = int(n * TEST_FRACTION + 0.5)

        split[idx[:n_val]] = "val"
        split[idx[n_val:n_val + n_test]] = "test"
        split[idx[n_val + n_test:]] = "train"  # everything left over

    return split


def add_seen_lesion(df):
    """Add seen_lesion: True if ANOTHER image of the same lesion is in train."""
    is_train = (df["split"] == "train").astype(int)
    # Number of train images of each image's lesion.
    train_images_of_lesion = is_train.groupby(df["lesion_id"]).transform("sum")
    # Don't count the image itself: subtract 1 for train images.
    other_train_images = train_images_of_lesion - is_train
    df["seen_lesion"] = other_train_images > 0
    return df


def lesion_level_split(meta, seed):
    """Split (a): grouped by lesion_id, stratified by dx."""
    # One row per lesion. Every image of a lesion has the same dx (checked
    # here), so the lesion's dx is just the dx of its first image.
    assert (meta.groupby("lesion_id")["dx"].nunique() == 1).all(), "a lesion has more than one dx"
    lesions = meta.groupby("lesion_id", as_index=False)["dx"].first()

    lesions["split"] = stratified_split(lesions, seed)

    # Give each image the partition of its lesion.
    df = meta.copy()
    df["split"] = df["lesion_id"].map(lesions.set_index("lesion_id")["split"])
    df = add_seen_lesion(df)

    # Checks: every image has a partition, and no lesion is in two partitions.
    assert df["split"].isin(PARTITIONS).all(), "an image has no partition"
    partitions_per_lesion = df.groupby("lesion_id")["split"].nunique()
    n_overlap = int((partitions_per_lesion > 1).sum())
    assert n_overlap == 0, f"{n_overlap} lesions are in more than one partition"
    # (Train images CAN be seen_lesion: two train images of one lesion see
    # each other. What must never happen is a val/test image being seen.)
    not_train = df["split"] != "train"
    assert not df.loc[not_train, "seen_lesion"].any(), "a val/test image has its lesion in train"
    print("Lesion-level split: 0 lesions shared between partitions (checked).")
    return df


def image_level_split(meta, seed):
    """Split (b): each image on its own, stratified by dx."""
    df = meta.copy()
    df["split"] = stratified_split(df, seed)
    df = add_seen_lesion(df)
    assert df["split"].isin(PARTITIONS).all(), "an image has no partition"
    return df


def report_leakage(df):
    """Print how much the image-level split leaks lesions between partitions."""
    partitions_per_lesion = df.groupby("lesion_id")["split"].nunique()
    n_shared = int((partitions_per_lesion > 1).sum())

    train_lesions = set(df.loc[df["split"] == "train", "lesion_id"])
    test_lesions = set(df.loc[df["split"] == "test", "lesion_id"])

    print("\n=== Leakage in the image-level split ===")
    print(f"lesions in more than one partition: {n_shared} of {len(partitions_per_lesion)}")
    print(f"test lesions that also have an image in train: "
          f"{len(test_lesions & train_lesions)} of {len(test_lesions)}")
    for part in ["val", "test"]:
        rows = df[df["split"] == part]
        n_seen = int(rows["seen_lesion"].sum())
        print(f"{part} images whose lesion is seen in train: "
              f"{n_seen} of {len(rows)} ({n_seen / len(rows):.1%})")


def print_partition_counts(df, name):
    """Print image, lesion, class and subgroup counts per partition."""
    print(f"\n##### {name}: counts per partition #####")
    sizes = pd.DataFrame({
        "images": df.groupby("split")["image_id"].count(),
        "lesions": df.groupby("split")["lesion_id"].nunique(),
    }).reindex(PARTITIONS)
    sizes["% images"] = (100 * sizes["images"] / sizes["images"].sum()).round(1)
    print(sizes.to_string())

    # One table per column: rows = groups, columns = partitions (image counts).
    for column, order in [("dx", CLASSES), ("age_band", AGE_BANDS),
                          ("sex", ["male", "female", "unknown"]),
                          ("site_group", SITE_ORDER), ("malignant", [0, 1])]:
        table = pd.crosstab(df[column], df["split"])
        table = table.reindex(index=order, columns=PARTITIONS, fill_value=0)
        print(f"\n{column}")
        print(table.to_string())


def main():
    parser = argparse.ArgumentParser(description="Create lesion-level and image-level splits.")
    parser.add_argument("--data_dir", default="/content/data",
                        help="folder that contains metadata.csv (from data.py)")
    parser.add_argument("--out_dir", default=None,
                        help="where to save the split CSVs (default: same as --data_dir)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="build tiny fake data first, then split it")
    args = parser.parse_args()

    if args.smoke:
        import data
        import fake_data
        if args.data_dir == "/content/data":
            args.data_dir = "data_fake"  # never write fake data over the real folder
        fake_data.generate(args.data_dir)
        data.build_metadata(args.data_dir).to_csv(
            os.path.join(args.data_dir, "metadata.csv"), index=False)
        print(f"Smoke mode: fake data and metadata.csv in {args.data_dir}")

    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(args.data_dir, "metadata.csv"))

    # Same seed for both splits, so reruns give exactly the same CSVs.
    lesion_split = lesion_level_split(meta, args.seed)
    image_split = image_level_split(meta, args.seed)

    lesion_split.to_csv(os.path.join(out_dir, "split_lesion.csv"), index=False)
    image_split.to_csv(os.path.join(out_dir, "split_image.csv"), index=False)
    print(f"Saved split_lesion.csv and split_image.csv to {out_dir}")

    report_leakage(image_split)
    print_partition_counts(lesion_split, "(a) lesion-level split")
    print_partition_counts(image_split, "(b) image-level split")


if __name__ == "__main__":
    main()
