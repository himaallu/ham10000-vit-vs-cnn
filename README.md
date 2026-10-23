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

Reads every run's `test_predictions.csv` and writes to `<runs_root>/evaluation/`:

| File | Content |
|---|---|
| `table1_per_seed.csv`, `table1_overall.csv` | macro-F1, per-class F1, melanoma recall, malignant sensitivity/AUROC per run; mean ± SD over seeds |
| `h1_vit_vs_bit.csv` | H1: ViT (run 2) − BiT (run 1), lesion split |
| `h2_leaky_vs_fair.csv` | H2: image split − lesion split per model (seed 0) |
| `h2_seen_vs_unseen.csv` | H2: seen − unseen lesions inside the image-split runs |
| `h3_subgroups.csv` | H3: malignant sensitivity/AUROC per age band, sex, site group; ViT vs BiT |
| `h4_group_sensitivity.csv`, `h4_worst_group.csv` | H4: worst-group sensitivity and best−worst gap; balanced (run 5, 5b) − standard (run 2) |

A model's value is the mean of the metric over its seeds. 95% CIs come from 1000
bootstrap resamples of lesions (not images), shared by all runs on a test set, so
model differences use a paired bootstrap. H4 only uses groups with at least 10
malignant test images (`--min_malignant`).

## Figures and Grad-CAM

```bash
python src/figures.py --runs_root /content/drive/MyDrive/ham10000_runs --data_dir /content/data
python src/gradcam.py --runs_root /content/drive/MyDrive/ham10000_runs --image_dir /content/data/images_256
```

`figures.py` saves to `<runs_root>/figures/`: class distribution, leakage counts,
confusion matrices (seed 0 of each run), H3 and H4 subgroup charts (dots with 95%
CIs; needs `evaluate.py` first) and training curves. `gradcam.py` picks 6 test
images (melanoma correct / wrong / disagreement, a false melanoma alarm, two correct
benign lesions) and shows each image with BiT and ViT Grad-CAM for the predicted
class; it needs the seed-0 `best.pt` of runs 1 and 2.

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
predictions for all runs. `python src/figures.py --smoke` and
`python src/gradcam.py --smoke` make the figures from fake data.
