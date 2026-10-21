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

The real data goes in `/content/data` on Colab.

## Smoke test (no dataset, no GPU)

```bash
python src/fake_data.py --out data_fake
```

This writes a tiny fake HAM10000 (40 lesions, ~60 small random images) with the
same file layout and metadata columns as the real one.
