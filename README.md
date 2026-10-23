# ViT vs CNN on HAM10000 — leakage, subgroups, balanced fine-tuning

Course project (MAAI901). Compares ViT-B/16 and BiT ResNet-50 (both ImageNet-21k
pre-trained, via `timm`) on HAM10000 skin-lesion classification, looking at:

- **Leakage:** lesion-level vs image-level train/val/test splits.
- **Subgroups:** performance by age band, sex and body site.
- **Balanced fine-tuning:** class-weighted loss vs a sampler balanced over
  diagnosis x age band x site group.

## Layout

```
src/        Python code
configs/    Run configurations
notebooks/  Colab notebooks
results/    Outputs (not committed; real runs save to Google Drive)
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

On Colab, put your Kaggle API key in `~/.kaggle/kaggle.json`, then:

```bash
python src/data.py
```

This downloads HAM10000 into `/content/data`, keeps one copy of each image, and
writes `/content/data/metadata.csv` with image_id, path, dx, label (0-6),
lesion_id, age, age_band, sex, site_group and malignant.

## Splits

```bash
python src/splits.py
```

Writes `split_lesion.csv` (70/15/15 of lesions per class; no lesion in two
partitions) and `split_image.csv` (70/15/15 of images per class; lesions can
leak) next to `metadata.csv`. Both add a `split` column (train/val/test) and a
`seen_lesion` column (another image of the same lesion is in train). Use
`--out_dir` to save them elsewhere, e.g. on Google Drive.

## Smoke test (no dataset, no GPU)

```bash
python src/data.py --smoke
```

This writes a tiny fake HAM10000 to `data_fake/` (40 lesions, ~60 small random
images, same file layout and metadata columns as the real one) and builds
`metadata.csv` from it. `python src/fake_data.py --out data_fake` makes only the
fake data. `python src/splits.py --smoke` does the same and then splits it.
