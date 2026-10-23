"""Evaluate all runs and test the four hypotheses, with lesion-level bootstrap CIs.

Reads every <runs_root>/<run_name>/test_predictions.csv (and its config.yaml)
written by train.py, and saves these tables to <runs_root>/evaluation/:

  table1_per_seed.csv       every run and seed: macro-F1, per-class F1, mel recall,
                            malignant sensitivity and AUROC
  table1_overall.csv        the same, as mean and SD over seeds per run
  h1_vit_vs_bit.csv         H1: ViT (run 2) vs BiT (run 1), lesion split
  h2_leaky_vs_fair.csv      H2: image split vs lesion split, per model (seed 0)
  h2_seen_vs_unseen.csv     H2: seen vs unseen lesions inside the image-split runs
  h3_subgroups.csv          H3: malignant sensitivity/AUROC per subgroup, BiT vs ViT
  h4_group_sensitivity.csv  H4: malignant sensitivity per subgroup, standard vs balanced
  h4_worst_group.csv        H4: worst-group sensitivity and best-worst gap
  h4_excluded_groups.csv    H4: subgroups left out of the worst-group analysis, and why

Definitions
  malignant = mel, bcc, akiec.
  malignant sensitivity: of the truly malignant images, the fraction whose
                         predicted class is malignant.
  malignant AUROC:       AUROC of prob_mel + prob_bcc + prob_akiec.
  A model's value = the MEAN OVER ITS SEEDS of the metric.

Bootstrap (95% CIs)
  For each test set we draw n_boot resamples of LESIONS (with replacement; all
  images of a drawn lesion come along) and reuse the same resamples for every
  statistic. So when two models share a test set, each resample scores both
  on the same lesions: a paired bootstrap, the right one for a difference.
  Subgroup statistics use the subgroup's images inside each resample.
  Lesion-split and image-split runs have different test sets; their
  comparison (H2 leaky vs fair) uses independent resamples of each.

Usage (Colab):
    python src/evaluate.py --runs_root /content/drive/MyDrive/ham10000_runs
    python src/evaluate.py --smoke
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from data import AGE_BANDS, CLASSES, MALIGNANT, SITE_ORDER

N_CLASSES = len(CLASSES)
MAL_LABELS = [CLASSES.index(c) for c in MALIGNANT]
MEL = CLASSES.index("mel")

# Subgroup columns and the order their groups are shown in.
SUBGROUPS = {
    "age_band": AGE_BANDS,
    "sex": ["male", "female", "unknown"],
    "site_group": SITE_ORDER,
}


# ----------------------------------------------------------------------------
# Metrics. Each takes numpy arrays (true labels, predicted labels, malignant
# score) and returns one number, or NaN if it can't be computed (for example
# no malignant images in a bootstrap sample). Written with numpy instead of
# sklearn because they run hundreds of thousands of times in the bootstrap.
# ----------------------------------------------------------------------------

def f1_per_class(y_true, y_pred):
    """F1 of each class; NaN for a class that appears neither as true nor predicted."""
    # Confusion matrix: cm[t, p] = number of images with true class t, predicted p.
    cm = np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp   # predicted as the class, but wrong
    fn = cm.sum(axis=1) - tp   # truly the class, but missed
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * tp / denom, np.nan)


def macro_f1(y_true, y_pred, score):
    """Mean F1 over the classes that appear (on the full test set: all 7)."""
    f1 = f1_per_class(y_true, y_pred)
    return np.nan if np.all(np.isnan(f1)) else np.nanmean(f1)


def class_f1(k):
    """Return a metric: F1 of class k; NaN if class k has no true images."""
    def metric(y_true, y_pred, score):
        if not np.any(y_true == k):
            return np.nan
        return f1_per_class(y_true, y_pred)[k]
    return metric


def mel_recall(y_true, y_pred, score):
    """Of the true melanomas, the fraction predicted as melanoma."""
    is_mel = y_true == MEL
    return np.mean(y_pred[is_mel] == MEL) if is_mel.any() else np.nan


def malignant_sensitivity(y_true, y_pred, score):
    """Of the truly malignant images, the fraction predicted as any malignant class."""
    is_mal = np.isin(y_true, MAL_LABELS)
    return np.mean(np.isin(y_pred[is_mal], MAL_LABELS)) if is_mal.any() else np.nan


def malignant_auroc(y_true, y_pred, score):
    """AUROC of the malignant score (Mann-Whitney formula, ties count half)."""
    is_mal = np.isin(y_true, MAL_LABELS)
    n_pos, n_neg = is_mal.sum(), (~is_mal).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(score)  # 1 = lowest score; tied scores share their average rank
    return (ranks[is_mal].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


HEADLINE = {"macro_f1": macro_f1, "mel_recall": mel_recall,
            "malignant_sensitivity": malignant_sensitivity, "malignant_auroc": malignant_auroc}
ALL_METRICS = {"macro_f1": macro_f1,
               **{f"f1_{c}": class_f1(k) for k, c in enumerate(CLASSES)},
               "mel_recall": mel_recall,
               "malignant_sensitivity": malignant_sensitivity,
               "malignant_auroc": malignant_auroc}


# ----------------------------------------------------------------------------
# Loading runs
# ----------------------------------------------------------------------------

class TestSet:
    """The test images of one split, shared by every run on that split."""

    def __init__(self, preds):
        # Sort by image_id so all runs' predictions line up row by row.
        table = preds.sort_values("image_id").reset_index(drop=True)
        self.table = table[["image_id", "lesion_id", "y_true", "age_band", "sex",
                            "site_group", "seen_lesion"]]
        self.y_true = table["y_true"].to_numpy()
        self.all_rows = np.arange(len(table))
        self.samples = []  # bootstrap resamples (row indices), filled by draw_bootstrap

    def draw_bootstrap(self, n_boot, seed):
        """Draw n_boot lesion resamples; each is an array of row indices."""
        rows_of_lesion = list(self.table.groupby("lesion_id").indices.values())
        n_lesions = len(rows_of_lesion)
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(n_boot):
            picked = rng.integers(0, n_lesions, size=n_lesions)  # with repeats
            self.samples.append(np.concatenate([rows_of_lesion[i] for i in picked]))

    def rows_where(self, column, value):
        """Boolean mask: which rows belong to a subgroup."""
        return (self.table[column] == value).to_numpy()


def load_runs(runs_root):
    """Return (runs, test_sets).

    runs:      {run id: {"model", "split", "mode", "seeds": {seed: (y_pred, score)}}}
               e.g. runs["run2"]["seeds"][0] = predictions of ViT seed 0
    test_sets: {"lesion": TestSet, "image": TestSet}
    """
    runs, test_sets = {}, {}
    for name in sorted(os.listdir(runs_root)):
        folder = os.path.join(runs_root, name)
        pred_path = os.path.join(folder, "test_predictions.csv")
        if not os.path.exists(pred_path):
            continue
        with open(os.path.join(folder, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        preds = pd.read_csv(pred_path).sort_values("image_id").reset_index(drop=True)

        split = cfg["split"]
        if split not in test_sets:
            test_sets[split] = TestSet(preds)
        # Every run on a split must have exactly the same test images.
        ts = test_sets[split]
        assert (preds["image_id"].to_numpy() == ts.table["image_id"].to_numpy()).all(), \
            f"{name}: test images differ from other {split}-split runs"
        assert (preds["y_true"].to_numpy() == ts.y_true).all(), f"{name}: labels differ"

        run_id = name.split("_")[0]  # "run2_vit_lesion_standard_seed0" -> "run2"
        run = runs.setdefault(run_id, {"model": cfg["model"], "split": split,
                                       "mode": cfg["mode"], "seeds": {}})
        score = preds[[f"prob_{c}" for c in MALIGNANT]].sum(axis=1).to_numpy()
        run["seeds"][int(cfg["seed"])] = (preds["y_pred"].to_numpy(), score)

    for run_id, run in runs.items():
        print(f"{run_id}: {run['model']} / {run['split']} / {run['mode']} / "
              f"seeds {sorted(run['seeds'])}")
    return runs, test_sets


# ----------------------------------------------------------------------------
# Statistics and the bootstrap
# ----------------------------------------------------------------------------

def run_value(run, ts, metric, rows, seeds=None):
    """Metric of one run on the given rows, averaged over its seeds."""
    seeds = sorted(run["seeds"]) if seeds is None else seeds
    values = []
    for s in seeds:
        y_pred, score = run["seeds"][s]
        values.append(metric(ts.y_true[rows], y_pred[rows], score[rows]))
    values = np.array(values, dtype=float)
    return np.nan if np.all(np.isnan(values)) else np.nanmean(values)


def bootstrap(stat, ts, ts_b=None):
    """Point estimate and 95% CI for a statistic that returns an array.

    stat(rows) -> array of numbers. With a second test set ts_b,
    stat(rows, rows_b) is called with one resample of each (independent).
    Returns (point, ci_low, ci_high), each an array like stat's output.
    """
    # numpy warns about NaN values (e.g. no malignant images in a small
    # resample); NaN is the intended result there, so hide those warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if ts_b is None:
            point = stat(ts.all_rows)
            boots = np.array([stat(rows) for rows in ts.samples])
        else:
            point = stat(ts.all_rows, ts_b.all_rows)
            boots = np.array([stat(r, rb) for r, rb in zip(ts.samples, ts_b.samples)])
        # Resamples where a value could not be computed (NaN) are left out.
        low, high = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return np.asarray(point, dtype=float), low, high


def compare_row(a_name, b_name, point, low, high):
    """Turn [a, b, a-b] bootstrap results into one table row."""
    return {a_name: point[0], f"{a_name}_ci_low": low[0], f"{a_name}_ci_high": high[0],
            b_name: point[1], f"{b_name}_ci_low": low[1], f"{b_name}_ci_high": high[1],
            "diff": point[2], "diff_ci_low": low[2], "diff_ci_high": high[2]}


# ----------------------------------------------------------------------------
# The tables
# ----------------------------------------------------------------------------

def table1(runs, test_sets):
    """Per-seed metrics, and mean and SD over seeds, for every run."""
    rows = []
    for run_id, run in sorted(runs.items()):
        ts = test_sets[run["split"]]
        for seed in sorted(run["seeds"]):
            r = {"run": run_id, "model": run["model"], "split": run["split"],
                 "mode": run["mode"], "seed": seed}
            for name, metric in ALL_METRICS.items():
                r[name] = run_value(run, ts, metric, ts.all_rows, seeds=[seed])
            rows.append(r)
    per_seed = pd.DataFrame(rows)

    keys = ["run", "model", "split", "mode"]
    grouped = per_seed.groupby(keys)[list(ALL_METRICS)]
    overall = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_sd"))  # SD: NaN for 1 seed
    # Put each metric's mean and SD side by side.
    overall = overall[[f"{m}_{s}" for m in ALL_METRICS for s in ["mean", "sd"]]]
    overall.insert(0, "n_seeds", per_seed.groupby(keys).size())
    return per_seed, overall.reset_index()


def h1_vit_vs_bit(runs, test_sets):
    """ViT (run 2) minus BiT (run 1) on the lesion split, paired bootstrap."""
    vit, bit, ts = runs["run2"], runs["run1"], test_sets["lesion"]
    rows = []
    for name, metric in HEADLINE.items():
        def stat(r):
            a, b = run_value(vit, ts, metric, r), run_value(bit, ts, metric, r)
            return [a, b, a - b]
        rows.append({"metric": name, **compare_row("vit", "bit", *bootstrap(stat, ts))})
    return pd.DataFrame(rows)


def h2_leaky_vs_fair(runs, test_sets):
    """Image split (leaky) minus lesion split (fair), per model, seed 0 on both sides."""
    rows = []
    for model, leaky_id, fair_id in [("bit", "run3", "run1"), ("vit", "run4", "run2")]:
        if leaky_id not in runs or fair_id not in runs:
            continue
        leaky, fair = runs[leaky_id], runs[fair_id]
        for name, metric in HEADLINE.items():
            def stat(r_img, r_les):
                a = run_value(leaky, test_sets["image"], metric, r_img, seeds=[0])
                b = run_value(fair, test_sets["lesion"], metric, r_les, seeds=[0])
                return [a, b, a - b]
            result = bootstrap(stat, test_sets["image"], test_sets["lesion"])
            rows.append({"model": model, "metric": name, **compare_row("leaky", "fair", *result)})
    return pd.DataFrame(rows)


def h2_seen_vs_unseen(runs, test_sets):
    """Inside each image-split run: seen-lesion minus unseen-lesion test images."""
    ts = test_sets["image"]
    seen = ts.rows_where("seen_lesion", True)
    rows = []
    for run_id in ["run3", "run4"]:
        if run_id not in runs:
            continue
        run = runs[run_id]
        for name, metric in HEADLINE.items():
            def stat(r):
                a = run_value(run, ts, metric, r[seen[r]])    # resampled rows that are seen
                b = run_value(run, ts, metric, r[~seen[r]])   # ... and unseen
                return [a, b, a - b]
            rows.append({"run": run_id, "model": run["model"], "metric": name,
                         "n_seen_images": int(seen.sum()), "n_unseen_images": int((~seen).sum()),
                         **compare_row("seen", "unseen", *bootstrap(stat, ts))})
    return pd.DataFrame(rows)


def group_counts(ts, mask):
    return {"n_images": int(mask.sum()),
            "n_lesions": ts.table.loc[mask, "lesion_id"].nunique(),
            "n_malignant": int(np.isin(ts.y_true[mask], MAL_LABELS).sum())}


def h3_subgroups(runs, test_sets):
    """Malignant sensitivity and AUROC per subgroup: ViT (run 2) vs BiT (run 1)."""
    vit, bit, ts = runs["run2"], runs["run1"], test_sets["lesion"]
    rows = []
    for column, groups in SUBGROUPS.items():
        for group in groups:
            mask = ts.rows_where(column, group)
            if not mask.any():
                continue
            for name, metric in [("sensitivity", malignant_sensitivity), ("auroc", malignant_auroc)]:
                def stat(r):
                    r = r[mask[r]]  # this subgroup's images in the resample
                    a, b = run_value(vit, ts, metric, r), run_value(bit, ts, metric, r)
                    return [a, b, a - b]
                rows.append({"column": column, "group": group, **group_counts(ts, mask),
                             "metric": name, **compare_row("vit", "bit", *bootstrap(stat, ts))})
    return pd.DataFrame(rows)


def eligible_for_h4(ts, mask, min_malignant):
    """H4 uses a subgroup only if it has at least min_malignant malignant test images."""
    return np.isin(ts.y_true[mask], MAL_LABELS).sum() >= min_malignant


def h4_excluded_groups(test_sets, min_malignant):
    """Every subgroup left out of the H4 worst-group analysis, with its counts and reason."""
    ts = test_sets["lesion"]
    rows = []
    for column, groups in SUBGROUPS.items():
        for group in groups:
            mask = ts.rows_where(column, group)
            if mask.any() and eligible_for_h4(ts, mask, min_malignant):
                continue
            reason = (f"fewer than {min_malignant} malignant test images" if mask.any()
                      else "no test images")
            rows.append({"column": column, "group": group, **group_counts(ts, mask),
                         "min_malignant": min_malignant, "reason": reason})
    columns = ["column", "group", "n_images", "n_lesions", "n_malignant", "min_malignant", "reason"]
    return pd.DataFrame(rows, columns=columns)  # keeps the header even if nothing is excluded


def h4_balanced(runs, test_sets, min_malignant):
    """Worst-group sensitivity and best-worst gap: standard (run 2) vs balanced runs.

    "variant" is the balanced run being compared: run 5 (comparison
    "balanced") or run 5b (comparison "class_balanced"); diff = variant - standard.
    Only groups with at least min_malignant malignant test images take part
    in the worst/best comparison (decided once, on the full test set).
    """
    ts = test_sets["lesion"]
    standard = runs["run2"]
    comparisons = [(mode_name, runs[run_id]) for mode_name, run_id
                   in [("balanced", "run5"), ("class_balanced", "run5b")] if run_id in runs]

    group_rows, worst_rows = [], []
    for column, groups in SUBGROUPS.items():
        masks = {g: ts.rows_where(column, g) for g in groups}
        masks = {g: m for g, m in masks.items() if m.any()}
        eligible = [g for g, m in masks.items() if eligible_for_h4(ts, m, min_malignant)]

        def group_sens(run, r):
            """Sensitivity of every group, on resampled rows r."""
            return np.array([run_value(run, ts, malignant_sensitivity, r[masks[g][r]]) for g in masks])

        for mode_name, balanced in comparisons:
            # Per-group sensitivity, standard vs this balanced mode.
            def stat_groups(r):
                s, b = group_sens(standard, r), group_sens(balanced, r)
                return np.concatenate([s, b, b - s])
            point, low, high = bootstrap(stat_groups, ts)
            n = len(masks)
            for i, g in enumerate(masks):
                group_rows.append({"comparison": mode_name, "column": column, "group": g,
                                   **group_counts(ts, masks[g]), "eligible": g in eligible,
                                   **compare_row("standard", "variant",
                                                 point[[i, n + i, 2 * n + i]],
                                                 low[[i, n + i, 2 * n + i]],
                                                 high[[i, n + i, 2 * n + i]])})
            if len(eligible) < 2:
                print(f"H4 {column}: fewer than 2 groups with >= {min_malignant} malignant images, skipped")
                continue

            # Worst group and best-worst gap, using eligible groups only.
            idx = [list(masks).index(g) for g in eligible]

            def stat_worst(r):
                s, b = group_sens(standard, r)[idx], group_sens(balanced, r)[idx]
                s_worst, b_worst = np.nanmin(s), np.nanmin(b)
                s_gap, b_gap = np.nanmax(s) - s_worst, np.nanmax(b) - b_worst
                return [s_worst, b_worst, b_worst - s_worst, s_gap, b_gap, b_gap - s_gap]
            point, low, high = bootstrap(stat_worst, ts)
            s_full, b_full = group_sens(standard, ts.all_rows)[idx], group_sens(balanced, ts.all_rows)[idx]
            for j, quantity in enumerate(["worst_group_sensitivity", "best_worst_gap"]):
                k = [3 * j, 3 * j + 1, 3 * j + 2]
                worst_rows.append({
                    "comparison": mode_name, "column": column, "quantity": quantity,
                    "groups_used": ", ".join(map(str, eligible)),
                    "standard_worst_group": eligible[int(np.nanargmin(s_full))],
                    "variant_worst_group": eligible[int(np.nanargmin(b_full))],
                    **compare_row("standard", "variant", point[k], low[k], high[k])})
    return pd.DataFrame(group_rows), pd.DataFrame(worst_rows)


# ----------------------------------------------------------------------------
# Smoke test data
# ----------------------------------------------------------------------------

def make_smoke_runs(root):
    """Fake predictions for every run, on a larger fake dataset (400 lesions, 40+ per class)."""
    import data
    import fake_data
    import splits

    work = "data_fake"
    fake_data.generate(work, n_lesions=400, per_class=40)  # enough malignant lesions for subgroups
    meta = data.build_metadata(work)
    tests = {"lesion": splits.lesion_level_split(meta, seed=42),
             "image": splits.image_level_split(meta, seed=42)}
    tests = {k: v[v["split"] == "test"].reset_index(drop=True) for k, v in tests.items()}

    # (run id, model, split, mode, seeds, how strongly probs lean to the truth)
    fake_runs = [("run1", "bit", "lesion", "standard", [0, 1, 2], 0.5),
                 ("run2", "vit", "lesion", "standard", [0, 1, 2], 0.7),
                 ("run3", "bit", "image", "standard", [0], 0.5),
                 ("run4", "vit", "image", "standard", [0], 0.7),
                 ("run5", "vit", "lesion", "balanced", [0, 1, 2], 0.7)]
    for run_id, model, split, mode, seeds, strength in fake_runs:
        test = tests[split]
        for seed in seeds:
            rng = np.random.default_rng(hash((run_id, seed)) % 2**32)
            probs = rng.random((len(test), N_CLASSES))
            boost = strength + 0.8 * test["seen_lesion"].to_numpy()  # leakage: seen lesions are easier
            probs[np.arange(len(test)), test["label"]] += boost
            probs /= probs.sum(axis=1, keepdims=True)
            preds = pd.DataFrame({"image_id": test["image_id"], "lesion_id": test["lesion_id"],
                                  "y_true": test["label"], "y_pred": probs.argmax(axis=1)})
            for k, c in enumerate(CLASSES):
                preds[f"prob_{c}"] = probs[:, k]
            for column in ["age_band", "sex", "site_group", "seen_lesion"]:
                preds[column] = test[column]
            run_name = f"{run_id}_{model}_{split}_{mode}_seed{seed}"
            folder = os.path.join(root, run_name)
            os.makedirs(folder, exist_ok=True)
            preds.to_csv(os.path.join(folder, "test_predictions.csv"), index=False)
            with open(os.path.join(folder, "config.yaml"), "w") as f:
                yaml.safe_dump({"run_name": run_name, "model": model, "split": split,
                                "mode": mode, "seed": seed}, f)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate all runs and the four hypotheses.")
    parser.add_argument("--runs_root", default="/content/drive/MyDrive/ham10000_runs")
    parser.add_argument("--out_dir", default=None, help="default: <runs_root>/evaluation")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="seed for the bootstrap")
    parser.add_argument("--min_malignant", type=int, default=10,
                        help="H4: groups need this many malignant test images to count")
    parser.add_argument("--smoke", action="store_true", help="evaluate fake predictions")
    args = parser.parse_args()

    if args.smoke:
        args.runs_root = os.path.join("data_fake", "runs_eval")
        args.n_boot = 200
        args.min_malignant = 3
        make_smoke_runs(args.runs_root)

    out_dir = args.out_dir or os.path.join(args.runs_root, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    runs, test_sets = load_runs(args.runs_root)
    for ts in test_sets.values():
        ts.draw_bootstrap(args.n_boot, args.seed)
    print(f"{args.n_boot} lesion bootstrap samples per test set\n")

    def save(df, name):
        df.to_csv(os.path.join(out_dir, name), index=False)
        print(f"===== {name} =====")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"), "\n")

    per_seed, overall = table1(runs, test_sets)
    save(per_seed, "table1_per_seed.csv")
    save(overall, "table1_overall.csv")

    have = lambda *ids: all(i in runs for i in ids)
    if have("run1", "run2"):
        save(h1_vit_vs_bit(runs, test_sets), "h1_vit_vs_bit.csv")
        save(h3_subgroups(runs, test_sets), "h3_subgroups.csv")
    else:
        print("H1/H3 skipped: need run1 and run2\n")
    if have("run1", "run3") or have("run2", "run4"):
        save(h2_leaky_vs_fair(runs, test_sets), "h2_leaky_vs_fair.csv")
        save(h2_seen_vs_unseen(runs, test_sets), "h2_seen_vs_unseen.csv")
    else:
        print("H2 skipped: need an image-split run and its lesion-split partner\n")
    if have("run2") and (have("run5") or have("run5b")):
        groups, worst = h4_balanced(runs, test_sets, args.min_malignant)
        save(groups, "h4_group_sensitivity.csv")
        save(worst, "h4_worst_group.csv")
        save(h4_excluded_groups(test_sets, args.min_malignant), "h4_excluded_groups.csv")
    else:
        print("H4 skipped: need run2 and run5 (or run5b)\n")
    print(f"Saved tables to {out_dir}")


if __name__ == "__main__":
    main()
