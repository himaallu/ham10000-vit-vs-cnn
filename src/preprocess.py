"""Resize every image once (short side 256 px) and cache it to disk.

The originals are 600x450. Training crops 224x224 patches, so decoding and
shrinking the full-size JPEG every epoch is wasted work. We do it once here:
600x450 -> 341x256, saved as <cache_dir>/<image_id>.jpg.

Images that are already in the cache are skipped, so rerunning is cheap.

Usage (Colab):
    python src/preprocess.py                # /content/data -> /content/data/images_256
    python src/preprocess.py --smoke        # tiny fake data
"""

import argparse
import os

import pandas as pd
from PIL import Image

SHORT_SIDE = 256


def resize_short_side(img, short_side=SHORT_SIDE):
    """Resize a PIL image so its shorter side is `short_side`, keeping the aspect ratio."""
    w, h = img.size
    scale = short_side / min(w, h)
    new_size = (round(w * scale), round(h * scale))
    return img.resize(new_size, Image.BICUBIC)


def cache_images(meta, cache_dir):
    """Resize every image listed in `meta` (needs image_id and path) into cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    n_done, n_skipped = 0, 0

    for i, row in enumerate(meta.itertuples()):
        out_path = os.path.join(cache_dir, row.image_id + ".jpg")
        if os.path.exists(out_path):
            n_skipped += 1
            continue

        # convert("RGB") makes sure we always have 3 channels.
        img = Image.open(row.path).convert("RGB")
        img = resize_short_side(img)
        img.save(out_path, quality=95)  # high JPEG quality: little extra loss
        n_done += 1

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} / {len(meta)}")

    print(f"Cached {n_done} images, skipped {n_skipped} already cached -> {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Resize images to short side 256 and cache them.")
    parser.add_argument("--data_dir", default="/content/data",
                        help="folder that contains metadata.csv (from data.py)")
    parser.add_argument("--cache_dir", default=None,
                        help="where to save resized images (default: <data_dir>/images_256)")
    parser.add_argument("--smoke", action="store_true",
                        help="build tiny fake data first, then resize it")
    args = parser.parse_args()

    if args.smoke:
        import data
        import fake_data
        if args.data_dir == "/content/data":
            args.data_dir = "data_fake"  # never write fake data over the real folder
        fake_data.generate(args.data_dir)
        data.build_metadata(args.data_dir).to_csv(
            os.path.join(args.data_dir, "metadata.csv"), index=False)

    cache_dir = args.cache_dir or os.path.join(args.data_dir, "images_256")
    meta = pd.read_csv(os.path.join(args.data_dir, "metadata.csv"))
    cache_images(meta, cache_dir)

    # Quick check: the short side of the first cached image is 256.
    first = Image.open(os.path.join(cache_dir, meta["image_id"].iloc[0] + ".jpg"))
    print(f"Example cached image size (w, h): {first.size}")
    assert min(first.size) == SHORT_SIDE


if __name__ == "__main__":
    main()
