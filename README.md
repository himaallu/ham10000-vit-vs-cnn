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

## Running on Colab

Open `notebooks/run_colab.ipynb` in Colab (GPU runtime) and run it top to bottom.
It mounts Drive, clones the repo, installs requirements, sets up `kaggle.json`,
prepares the data once (cached on Drive in `MyDrive/ham10000_data`), offers a quick
learning-rate check (train + val only) and trains the config you choose.

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

## Resize cache and datasets

```bash
python src/preprocess.py
python src/dataset.py --split_csv /content/data/split_lesion.csv --model vit
```

`preprocess.py` resizes every image once to short side 256 into
`/content/data/images_256/`. `dataset.py` defines the PyTorch `Dataset` (train:
rotation, random resized crop 224, flips, colour jitter; val/test: center crop
224; mean/std from each timm model's config), checks one batch per partition
and saves a grid of 8 augmented training images.

## Training

```bash
python src/train.py --config configs/run1_seed0.yaml
```

One YAML per run and seed (runs 1-5 and 5b, see `CLAUDE.md`). Each run saves
`best.pt`, `last.pt`, `history.csv`, `test_predictions.csv` and `summary.json`
to `<out_root>/<run_name>/` (Google Drive by default). Balanced runs also print
the sampler groups before and after merging and save `sampler_groups.csv`
(group size, share, expected draws per epoch). If Colab disconnects,
run the same command again: it resumes from `last.pt`.

Modes: `standard` = class-weighted cross-entropy; `balanced` = WeightedRandomSampler
where every diagnosis gets an equal share and, inside a diagnosis, every
age band x site group cell gets an equal share (cells under `min_cell_size`
images are merged); `class_balanced` = the same sampler over diagnosis only.

## Evaluation

```bash
python src/evaluate.py --runs_root /content/drive/MyDrive/ham10000_runs
```

For every run folder with a `test_predictions.csv`: macro-F1, per-class F1,
melanoma recall, and malignant-vs-benign (mel, bcc, akiec) sensitivity and AUROC,
overall and per age band, sex, site group and seen_lesion. 95% CIs come from a
bootstrap that resamples lesions, not images. Writes `metrics_overall.csv`,
`metrics_per_class.csv` and `metrics_subgroups.csv` into each run folder, and
`summary_all_runs.csv` and `summary_by_run.csv` (mean/std over seeds) into the
runs folder.

## Smoke test (no dataset, no GPU)

```bash
python src/data.py --smoke
```

This writes a tiny fake HAM10000 to `data_fake/` (40 lesions, ~60 small random
images, same file layout and metadata columns as the real one) and builds
`metadata.csv` from it. `python src/fake_data.py --out data_fake` makes only the
fake data. `python src/splits.py --smoke` does the same and then splits it. `python src/preprocess.py --smoke` and `python src/dataset.py --smoke` run
the chain up to the datasets.
`python src/train.py --config configs/run5_seed0.yaml --smoke` trains 2 epochs
on CPU with random weights (no download). `python src/evaluate.py --smoke` evaluates fake
predictions for three fake runs.
