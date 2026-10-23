# Project: ViT vs CNN on HAM10000 — leakage, subgroups, balanced fine-tuning
Course project (MAAI901). Student must understand every line; write simple, well-commented PyTorch.
Runs on Google Colab (GPU) — data lives in /content/data, outputs saved to Google Drive.

Design (do not change without asking):
- Models: ViT-B/16 pre-trained on ImageNet-21k and ResNet-50 (BiT, ResNetV2-50x1) also pre-trained on ImageNet-21k, via timm. 7-class head.
- Splits: (a) lesion-level 70/15/15 grouped by lesion_id, stratified by diagnosis; (b) image-level 70/15/15 stratified by diagnosis. Fixed seed, saved to CSV.
- Augmentation only on the training set, on the fly, after splitting.
- Standard training = class-weighted cross-entropy. Balanced training = WeightedRandomSampler over diagnosis x age band x site group (sparse cells merged), plain cross-entropy.
  - Sampler rule: each diagnosis gets an equal share (1/7); inside a diagnosis, each age band x site group cell gets an equal share; images inside a cell are equally likely.
  - Merging cells under min_cell_size (10) training images: first drop site (dx|age|any); if still too small, drop age (dx|any|any); if that leftover is still too small, it joins the largest cell of the same diagnosis.
  - Class-only balanced sampler (run 5b) = same sampler with diagnosis-only cells.
- Age, sex, site are NEVER model inputs — only for sampling and reporting.
- Model selection: best validation macro-F1, early stopping.
- Every run saves predictions CSV: image_id, lesion_id, y_true, y_pred, 7 class probabilities, age_band, sex, site_group, seen_lesion flag.
- Metrics: overall macro-F1, per-class F1, melanoma recall; subgroups use malignant-vs-benign (mel, bcc, akiec = malignant) sensitivity and AUROC; bootstrap CIs resample lesions, not images.
- Groups: age <40, 40-59, 60+, unknown; site: head_neck, trunk, upper_limb, lower_limb, other.

Runs (keep this numbering — it matches the report; one config per run and seed in configs/):
1 = BiT ResNet-50 / lesion split / standard   (seeds 0, 1, 2)
2 = ViT-B/16      / lesion split / standard   (seeds 0, 1, 2)
3 = BiT ResNet-50 / image split  / standard   (seed 0)
4 = ViT-B/16      / image split  / standard   (seed 0)
5 = ViT-B/16      / lesion split / balanced   (seeds 0, 1, 2)
Optional 5b = ViT-B/16 / lesion split / class-only balanced sampler (seed 0), only if time allows.

Rules: one task at a time; show a short plan before writing code; add a --smoke flag that runs on tiny fake data so code can be tested without the real dataset or a GPU.
