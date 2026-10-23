"""Evaluate test predictions: overall, per-class and subgroup metrics with CIs.

Reads <run_dir>/test_predictions.csv (written by train.py) and computes:
  Overall:   macro-F1, per-class F1, melanoma recall,
             malignant-vs-benign sensitivity and AUROC
  Subgroups: malignant-vs-benign sensitivity and AUROC for every
             age band, sex, site group and seen_lesion value

Malignant = mel, bcc, akiec.
  - malignant prediction: the predicted class (argmax) is malignant
  - malignant score:      prob_mel + prob_bcc + prob_akiec (used for AUROC)

95% confidence intervals come from a bootstrap that resamples LESIONS, not
images: images of the same lesion are not independent, so they are always
drawn together. For a subgroup, lesions are resampled within that subgroup.

Files written to each run folder:
    metrics_overall.csv     metric, value, ci_low, ci_high, n_valid
    metrics_per_class.csv   F1 per class with CI
    metrics_subgroups.csv   counts, sensitivity and AUROC (with CIs) per group
And, when --runs_root is used, to the runs folder:
    summary_all_runs.csv    one row per run folder
    summary_by_run.csv      mean and std over seeds for each run

Usage (Colab):
    python src/evaluate.py --runs_root /content/drive/MyDrive/ham10000_runs
    python src/evaluate.py --run_dir /content/drive/MyDrive/ham10000_runs/run1_bit_lesion_standard_seed0
    python src/evaluate.py --smoke
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from data import AGE_BANDS, CLASSES, MALIGNANT, SITE_ORDER

MAL_LABELS = [CLASSES.index(c) for c in MALIGNANT]   # [4, 1, 0]
MEL = CLASSES.index("mel")

# Subgroup columns and the order their groups are shown in.
SUBGROUPS = {
    "age_band": AGE_BANDS,
    "sex": ["male", "female", "unknown"],
    "site_group": SITE_ORDER,
    "seen_lesion": [False, True],
}


# ----------------------------------------------------------------------------
# Metrics. Each takes a predictions table and returns one number (or NaN if
# the metric can't be computed on that table).
# ----------------------------------------------------------------------------

def macro_f1(df):
    """Mean F1 over the classes that appear (as true or predicted label).

    On the full test set all 7 classes appear, so this is the usual 7-class
    macro-F1. In a bootstrap sample a rare class can be missing completely;
    counting it as F1 = 0 would wrongly pull the CI down, so it is left out.
    """
    return f1_score(df["y_true"], df["y_pred"], average="macro", zero_division=0)


def class_f1(k):
    """Return a metric function: F1 of class k (one-vs-rest).

    NaN if class k has no true images in the table (F1 is then meaningless).
    """
    def f1_of_class(df):
        if (df["y_true"] == k).sum() == 0:
            return np.nan
        return f1_score(df["y_true"] == k, df["y_pred"] == k, zero_division=0)
    return f1_of_class


def mel_recall(df):
    """Of the true melanomas, the fraction predicted as melanoma."""
    is_mel = df["y_true"] == MEL
    if is_mel.sum() == 0:
        return np.nan
    return (df.loc[is_mel, "y_pred"] == MEL).mean()


def malignant_sensitivity(df):
    """Of the truly malignant images, the fraction predicted as any malignant class."""
    is_mal = df["y_true"].isin(MAL_LABELS)
    if is_mal.sum() == 0:
        return np.nan
    return df.loc[is_mal, "y_pred"].isin(MAL_LABELS).mean()


def malignant_score(df):
    """Probability of 'malignant' = sum of the 3 malignant class probabilities."""
    return df[[f"prob_{c}" for c in MALIGNANT]].sum(axis=1)


def malignant_auroc(df):
    """AUROC of the malignant score; NaN if only one of the two outcomes is present."""
    is_mal = df["y_true"].isin(MAL_LABELS)
    if is_mal.nunique() < 2:
        return np.nan
    return roc_auc_score(is_mal, malignant_score(df))


# ----------------------------------------------------------------------------
# Lesion-level bootstrap
# ----------------------------------------------------------------------------

def lesion_bootstrap(df, metric, n_boot, seed):
    """Value of `metric` on df, plus a 95% CI from resampling lesions.

    Each bootstrap sample draws as many lesions as df has, WITH replacement,
    and takes all images of every drawn lesion (a lesion drawn twice
    contributes its images twice).

    Returns (value, ci_low, ci_high, n_valid), where n_valid is the number of
    bootstrap samples on which the metric could be computed.
    """
    value = metric(df)

    # Row positions of each lesion's images, computed once.
    rows_of_lesion = list(df.groupby("lesion_id").indices.values())
    n_lesions = len(rows_of_lesion)

    rng = np.random.default_rng(seed)
    boot_values = []
    for _ in range(n_boot):
        picked = rng.integers(0, n_lesions, size=n_lesions)  # lesion numbers, with repeats
        rows = np.concatenate([rows_of_lesion[i] for i in picked])
        v = metric(df.iloc[rows])
        if not np.isnan(v):
            boot_values.append(v)

    if len(boot_values) == 0:
        return value, np.nan, np.nan, 0
    low, high = np.percentile(boot_values, [2.5, 97.5])
    return value, low, high, len(boot_values)


# ----------------------------------------------------------------------------
# Evaluate one run
# ----------------------------------------------------------------------------

def evaluate_run(run_dir, n_boot, seed):
    """Compute all metrics for one run folder and save them there."""
    df = pd.read_csv(os.path.join(run_dir, "test_predictions.csv"))

    def row(name, metric, data):
        value, low, high, n_valid = lesion_bootstrap(data, metric, n_boot, seed)
        return {"metric": name, "value": value, "ci_low": low, "ci_high": high, "n_valid": n_valid}

    overall = pd.DataFrame([
        row("macro_f1", macro_f1, df),
        row("mel_recall", mel_recall, df),
        row("malignant_sensitivity", malignant_sensitivity, df),
        row("malignant_auroc", malignant_auroc, df),
    ])
    overall.to_csv(os.path.join(run_dir, "metrics_overall.csv"), index=False)

    per_class = pd.DataFrame([row(f"f1_{c}", class_f1(k), df) for k, c in enumerate(CLASSES)])
    per_class.insert(0, "class", CLASSES)
    per_class = per_class.drop(columns="metric")
    per_class.to_csv(os.path.join(run_dir, "metrics_per_class.csv"), index=False)

    sub_rows = []
    for column, order in SUBGROUPS.items():
        for group in order:
            part = df[df[column] == group]
            if len(part) == 0:
                continue
            sens = lesion_bootstrap(part, malignant_sensitivity, n_boot, seed)
            auc = lesion_bootstrap(part, malignant_auroc, n_boot, seed)
            sub_rows.append({
                "column": column, "group": group,
                "n_images": len(part),
                "n_lesions": part["lesion_id"].nunique(),
                "n_malignant": int(part["y_true"].isin(MAL_LABELS).sum()),
                "sensitivity": sens[0], "sens_ci_low": sens[1], "sens_ci_high": sens[2],
                "auroc": auc[0], "auroc_ci_low": auc[1], "auroc_ci_high": auc[2],
                "auroc_n_valid": auc[3],
            })
    subgroups = pd.DataFrame(sub_rows)
    subgroups.to_csv(os.path.join(run_dir, "metrics_subgroups.csv"), index=False)

    return overall, per_class, subgroups


def print_run(name, overall, per_class, subgroups):
    """Print one run's results as compact tables."""
    fmt = lambda x: f"{x:.3f}"
    print(f"\n===== {name} =====")
    print(overall.to_string(index=False, float_format=fmt))
    print()
    print(per_class.to_string(index=False, float_format=fmt))
    print()
    cols = ["column", "group", "n_images", "n_lesions", "n_malignant",
            "sensitivity", "sens_ci_low", "sens_ci_high", "auroc", "auroc_ci_low", "auroc_ci_high"]
    print(subgroups[cols].to_string(index=False, float_format=fmt))


# ----------------------------------------------------------------------------
# Summaries across runs
# ----------------------------------------------------------------------------

def summarise_runs(results):
    """results: {run_name: overall table}. Returns (all runs table, by-run table)."""
    rows = []
    for run_name, overall in results.items():
        # run_name looks like "run1_bit_lesion_standard_seed0".
        match = re.match(r"(run\w+?)_.*_seed(\d+)$", run_name)
        r = {"run_name": run_name,
             "run": match.group(1) if match else run_name,
             "seed": int(match.group(2)) if match else np.nan}
        for _, m in overall.iterrows():
            r[m["metric"]] = m["value"]
            r[m["metric"] + "_ci"] = f"[{m['ci_low']:.3f}, {m['ci_high']:.3f}]"
        rows.append(r)
    all_runs = pd.DataFrame(rows).sort_values(["run", "seed"])

    # Mean and standard deviation over seeds (std is NaN for a single seed).
    metrics = ["macro_f1", "mel_recall", "malignant_sensitivity", "malignant_auroc"]
    by_run = all_runs.groupby("run")[metrics].agg(["mean", "std"])
    by_run.columns = [f"{m}_{stat}" for m, stat in by_run.columns]
    by_run.insert(0, "n_seeds", all_runs.groupby("run").size())
    return all_runs, by_run


# ----------------------------------------------------------------------------
# Smoke test data
# ----------------------------------------------------------------------------

def make_smoke_runs(root):
    """Write fake test_predictions.csv files for a few fake runs."""
    import fake_data
    import data
    import splits

    work = "data_fake"
    fake_data.generate(work)
    meta = data.build_metadata(work)
    for run_name, split_fn in [("run1_bit_lesion_standard_seed0", splits.lesion_level_split),
                               ("run1_bit_lesion_standard_seed1", splits.lesion_level_split),
                               ("run3_bit_image_standard_seed0", splits.image_level_split)]:
        seed = int(run_name[-1])
        test = split_fn(meta, seed=42)
        test = test[test["split"] == "test"].reset_index(drop=True)

        # Probabilities that lean towards the true class, so metrics are not 0.
        rng = np.random.default_rng(seed)
        probs = rng.random((len(test), len(CLASSES)))
        probs[np.arange(len(test)), test["label"]] += 0.4
        probs /= probs.sum(axis=1, keepdims=True)

        preds = pd.DataFrame({"image_id": test["image_id"], "lesion_id": test["lesion_id"],
                              "y_true": test["label"], "y_pred": probs.argmax(axis=1)})
        for k, c in enumerate(CLASSES):
            preds[f"prob_{c}"] = probs[:, k]
        for column in ["age_band", "sex", "site_group", "seen_lesion"]:
            preds[column] = test[column]

        os.makedirs(os.path.join(root, run_name), exist_ok=True)
        preds.to_csv(os.path.join(root, run_name, "test_predictions.csv"), index=False)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate test predictions with lesion-level bootstrap CIs.")
    parser.add_argument("--runs_root", default=None,
                        help="evaluate every subfolder that has a test_predictions.csv")
    parser.add_argument("--run_dir", default=None, help="evaluate a single run folder")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="seed for the bootstrap")
    parser.add_argument("--smoke", action="store_true", help="evaluate fake predictions")
    args = parser.parse_args()

    if args.smoke:
        args.runs_root = os.path.join("data_fake", "runs_eval")
        args.n_boot = 200
        make_smoke_runs(args.runs_root)

    if args.run_dir:
        run_dirs = [args.run_dir]
    elif args.runs_root:
        run_dirs = sorted(
            os.path.join(args.runs_root, d) for d in os.listdir(args.runs_root)
            if os.path.exists(os.path.join(args.runs_root, d, "test_predictions.csv")))
    else:
        parser.error("give --runs_root or --run_dir (or --smoke)")
    print(f"Evaluating {len(run_dirs)} run(s) with {args.n_boot} lesion bootstrap samples")

    results = {}
    for run_dir in run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        overall, per_class, subgroups = evaluate_run(run_dir, args.n_boot, args.seed)
        print_run(name, overall, per_class, subgroups)
        results[name] = overall

    if args.runs_root and not args.run_dir:
        all_runs, by_run = summarise_runs(results)
        all_runs.to_csv(os.path.join(args.runs_root, "summary_all_runs.csv"), index=False)
        by_run.to_csv(os.path.join(args.runs_root, "summary_by_run.csv"))
        print("\n===== All runs =====")
        print(all_runs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\n===== Mean and std over seeds =====")
        print(by_run.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"\nSaved summary_all_runs.csv and summary_by_run.csv to {args.runs_root}")


if __name__ == "__main__":
    main()
