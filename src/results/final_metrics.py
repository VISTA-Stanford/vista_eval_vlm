"""
Read result CSVs (same discovery as ct_experiment_plot) from results_dir in config,
and output figures/results_stats/results.csv with columns: task, model_name, experiment,
accuracy, weighted_f1, and (for binary Yes/No tasks) true_positive, true_negative,
false_positive, false_negative, sensitivity, specificity.
Includes all experiments for every model for every task.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure these imports exist in your environment
from results.results_analyzer import _extract_answer, is_answer_correct, map_label_to_answer
from context.normalize import experiment_names
# Single source of truth for the Yes/No-task predicate (dep-free module); re-exported here
# so existing `from results.final_metrics import is_binary_yes_no_task` callers keep working.
from results.task_mapping import is_binary_yes_no_task
from tasks.registry import load_valid_tasks


def extract_experiment_from_filename(filename):
    """Extract experiment name from filename like task_name_results_experiment.csv"""
    match = re.search(r'_results_(.+)\.csv$', filename)
    if match:
        return match.group(1)
    return "default"


def load_config(config_path):
    """Load results_dir, tasks, models, experiments from all_tasks.yaml."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = config.get("paths", {})
    results_dir = paths.get("results_dir") or "/home/dcunhrya/results"
    base_dir = paths.get("base_dir") or "/home/dcunhrya/vista_bench"
    tasks = list(config.get("tasks", []))
    models_list = config.get("models", [])
    valid_models = set()
    for model in models_list or []:
        if isinstance(model, dict) and "name" in model:
            name = model["name"]
            file_name = name.split("/")[-1].replace("/", "_")
            valid_models.add(file_name)
            valid_models.add(name.replace("/", "_"))
            if "/" in name:
                valid_models.add(name.split("/")[-1])
    # `experiments` may contain bare legacy strings AND block-list dicts; resolve
    # to normalized name tokens so `set(...)` never hits an unhashable dict.
    experiments = set(experiment_names(config))
    return results_dir, base_dir, tasks, valid_models, experiments


def calculate_accuracy_like_plot(df, mapping):
    """
    Same accuracy logic as ct_experiment_plot: is_answer_correct on every row (all rows, no filtering).
    Returns accuracy as percentage, or None if cannot compute.
    """
    if not mapping or "label" not in df.columns or "model_response" not in df.columns:
        return None
    df = df.copy()
    df["mapped_label"] = df["label"].apply(lambda lbl: map_label_to_answer(lbl, mapping))
    df["is_correct"] = df.apply(
        lambda x: is_answer_correct(x["model_response"], x["mapped_label"]),
        axis=1,
    )
    return df["is_correct"].mean() * 100


def binary_labels_and_scores(df, mapping):
    """
    Build y_true (binary) and y_score (prediction score 0/1 from extracted answer).
    Uses Yes=1 (positive) and No=0 (negative) per mapping["1"]="Yes", mapping["0"]="No".
    Excludes rows where label is -1 or maps to "Insufficient follow-up or missing data".

    If model output matches 'Yes' (positive) -> score 1.0.
    Otherwise (No, garbage, ambiguous) -> score 0.0.
    """
    if not mapping or "label" not in df.columns or "model_response" not in df.columns:
        return None, None
    df = df.copy()
    df["mapped_label"] = df["label"].apply(lambda lbl: map_label_to_answer(lbl, mapping))
    
    # 1. Filter Ground Truth: Keep only rows with valid labels
    insufficient = "Insufficient follow-up or missing data"
    valid = (
        df["label"].notna()
        & (df["label"].astype(str).str.strip() != "-1")
        & (df["mapped_label"] != insufficient)
        & (df["mapped_label"].notna())
    )
    df = df.loc[valid].copy()
    
    if df.empty:
        return None, None

    pos_str = mapping.get("1")
    # We don't strictly need neg_str for binary score calculation if we treat "not pos" as 0
    if not pos_str:
        return None, None

    df["cleaned_response"] = df["model_response"].apply(_extract_answer)
    
    # 2. Build y_true based on mapped_label: Yes=1, No=0
    y_true = (df["mapped_label"].str.strip().str.lower() == pos_str.strip().lower()).astype(int).values

    # 3. Build y_score based on prediction: Yes=1.0, No (or other)=0.0
    # If you have a 'probability' column in your CSV, you should use that here instead.
    y_score = (df["cleaned_response"].str.strip().str.lower() == pos_str.strip().lower()).astype(float).values

    return y_true, y_score


def _weighted_f1_binary(tp, tn, fp, fn):
    """
    Compute weighted F1 for binary classification from confusion matrix counts.
    F1 per class; weight by support (number of true samples per class).
    """
    support_pos = tp + fn
    support_neg = tn + fp
    n = support_pos + support_neg
    if n == 0:
        return float("nan")

    # Positive class: precision = TP/(TP+FP), recall = TP/(TP+FN) = sensitivity
    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = 2 * prec_pos * rec_pos / (prec_pos + rec_pos) if (prec_pos + rec_pos) > 0 else 0.0

    # Negative class: precision = TN/(TN+FN), recall = TN/(TN+FP) = specificity
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0

    return (support_pos * f1_pos + support_neg * f1_neg) / n


def compute_metrics(df, mapping):
    """
    Compute confusion matrix metrics and accuracy (%).
    Returns: (tp, tn, fp, fn, sensitivity, specificity, accuracy, weighted_f1).
    Accuracy: same as ct_experiment_plot (is_answer_correct on all rows).
    TP/TN/FP/FN, sensitivity, specificity, weighted_f1: on rows with valid label (exclude -1/insufficient).
    """
    # Accuracy: match ct_experiment_plot exactly (all rows, is_answer_correct)
    accuracy = calculate_accuracy_like_plot(df, mapping)
    if accuracy is None:
        return None, None, None, None, None, None, None, None

    y_true, y_score = binary_labels_and_scores(df, mapping)
    if y_true is None or len(y_true) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), accuracy, float("nan")

    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= 0.5).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Sensitivity = TP / (TP + FN); recall for positive class
    denom_sens = tp + fn
    sensitivity = tp / denom_sens if denom_sens > 0 else float("nan")

    # Specificity = TN / (TN + FP)
    denom_spec = tn + fp
    specificity = tn / denom_spec if denom_spec > 0 else float("nan")

    weighted_f1 = _weighted_f1_binary(tp, tn, fp, fn)

    return tp, tn, fp, fn, sensitivity, specificity, accuracy, weighted_f1


def collect_result_files(results_base, valid_tasks, valid_models, valid_experiments):
    """Discover and filter result files like ct_experiment_plot."""
    results_base = Path(results_base)
    # Use rglob to find all csvs, but be careful with permissions/depth if needed
    all_files = list(results_base.rglob("*_results_*.csv"))
    result_files = []
    for file_path in all_files:
        try:
            rel = file_path.relative_to(results_base).parts
        except ValueError:
            continue
        # Expect structure: .../results/source/task/model/filename.csv
        if len(rel) < 4:
            continue
        
        # Adjust indices based on your folder structure depth
        # Assuming: /results_dir/source_folder/task_name/model_name/filename
        task_name, model_name, filename = rel[1], rel[2], rel[-1]
        
        experiment = extract_experiment_from_filename(filename)
        
        if valid_tasks and task_name not in valid_tasks:
            continue
        if valid_models:
            model_ok = model_name in valid_models or any(
                model_name == m or model_name.endswith("_" + m) or model_name.endswith("/" + m)
                for m in valid_models
            )
            if not model_ok:
                continue
        if valid_experiments and experiment not in valid_experiments:
            continue
        result_files.append(file_path)
    return result_files


def main(config_path=None, output_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "all_tasks.yaml"
    config_path = Path(config_path)
    
    results_dir, base_dir, tasks, valid_models, experiments = load_config(config_path)
    results_base = Path(results_dir)
    base_path = Path(base_dir)

    # Task registry for mappings
    tasks_json = base_path / "tasks" / "valid_tasks.json"
    task_registry = {}
    if tasks_json.exists():
        with open(tasks_json, "r") as f:
            task_registry = {t["task_name"]: t for t in json.load(f)}

    result_files = collect_result_files(results_base, set(tasks), valid_models, experiments)
    if not result_files:
        print("No result files found matching config.")
        return

    # Group files by (task_name, model_name, experiment)
    by_task_model_exp = {}
    for file_path in result_files:
        try:
            rel = file_path.relative_to(results_base).parts
            task_name, model_name, filename = rel[1], rel[2], rel[-1]
            experiment = extract_experiment_from_filename(filename)
            key = (task_name, model_name, experiment)
            if key not in by_task_model_exp:
                by_task_model_exp[key] = []
            by_task_model_exp[key].append(file_path)
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    rows = []
    for task_name in tasks:
        mapping = task_registry.get(task_name, {}).get("mapping", {})
        
        # Find all models available for this task
        task_keys = [k for k in by_task_model_exp.keys() if k[0] == task_name]
        unique_models = list({k[1] for k in task_keys})
        
        for model_name in unique_models:
            # Include all experiments for this (task, model)
            relevant_keys = [k for k in task_keys if k[1] == model_name]

            for key in relevant_keys:
                experiment = key[2]
                files = by_task_model_exp[key]
                dfs = []
                for fp in files:
                    try:
                        df = pd.read_csv(fp)
                        if not df.empty:
                            dfs.append(df)
                    except Exception as e:
                        print(f"  Error reading {fp.name}: {e}")

                if not dfs:
                    continue

                combined = pd.concat(dfs, ignore_index=True)
                if "index" in combined.columns:
                    combined = combined.drop_duplicates(subset=["index"], keep="first")

                tp, tn, fp, fn, sensitivity, specificity, accuracy, weighted_f1 = compute_metrics(combined, mapping)
                if accuracy is None:
                    continue

                # Only compute confusion matrix for binary Yes/No tasks (Yes=1, No=0)
                include_confusion = (
                    ("died_of_cancer" in task_name or "has_recurrence" in task_name or 'pneumonitis_infection' in task_name)
                    and is_binary_yes_no_task(mapping)
                )
                row = {
                    "task": task_name,
                    "model_name": model_name,
                    "experiment": experiment,
                    "accuracy": accuracy,
                    "weighted_f1": weighted_f1,
                }
                if include_confusion:
                    row.update({
                        "true_positive": tp,
                        "true_negative": tn,
                        "false_positive": fp,
                        "false_negative": fn,
                        "sensitivity": sensitivity,
                        "specificity": specificity,
                    })
                rows.append(row)

    if not rows:
        print("No metrics computed.")
        return

    out_df = pd.DataFrame(rows)
    if output_path is None:
        output_path = Path(__file__).resolve().parents[2] / "figures" / "results_stats" / "results.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")



# ---------------------------------------------------------------------------
# Forced-answer scoring for the tumor-board questions (vista_bench_v1_6).
# Plan: planning/vista_eval_vlm/vlm-tumor-board-questions-v1_6.md, step 6 / V6.
# Scores the forced answer only (no probabilities, so no AUROC): accuracy, balanced
# accuracy, sensitivity/specificity (yes/no) or macro-F1 (multi-class), 95% bootstrap
# intervals, per split and pooled, with an always-majority baseline row.
# ---------------------------------------------------------------------------

TB_SPLITS = ("train", "valid", "test")


def answer_to_label(response, mapping, unscored=("-1",)):
    """Map a forced answer back to its integer label; None if it is not one of the options."""
    if response is None or (isinstance(response, float) and np.isnan(response)):
        return None
    r = str(response).strip().lower()
    for k, v in (mapping or {}).items():
        if str(k) in {str(u) for u in unscored}:
            continue
        if str(v).strip().lower() == r:
            return int(k)
    return None


def forced_answer_metrics(y_true, y_pred, classes):
    """Metrics from labels and predicted labels. A None prediction (unreadable) counts as wrong.

    Balanced accuracy and macro-F1 average over the classes present in ``y_true``.
    Sensitivity/specificity are recall of class 1 / class 0 when ``classes`` is {0, 1}.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    out = {"n": n, "n_unreadable": sum(p is None for p in y_pred)}
    if n == 0:
        return out
    correct = [t == p for t, p in zip(y_true, y_pred)]
    out["accuracy"] = float(np.mean(correct))
    present = [c for c in classes if c in set(y_true)]
    recalls, f1s = [], []
    for c in present:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        rec = tp / (tp + fn) if tp + fn else float("nan")
        prec = tp / (tp + fp) if tp + fp else 0.0
        recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    out["balanced_accuracy"] = float(np.mean(recalls)) if recalls else float("nan")
    if set(classes) == {0, 1}:
        rec = dict(zip(present, recalls))
        out["sensitivity"] = rec.get(1, float("nan"))
        out["specificity"] = rec.get(0, float("nan"))
    else:
        out["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    return out


def _stratified_draw(y_true, rng):
    """Bootstrap row indices resampled within each true class, so every resample keeps every
    class present in the sample (balanced accuracy then always averages the same classes)."""
    by_class = {}
    for i, t in enumerate(y_true):
        by_class.setdefault(t, []).append(i)
    return [i for members in by_class.values() for i in rng.choice(members, len(members))]


def bootstrap_ci(y_true, y_pred, classes, n_boot=1000, seed=0, alpha=0.05):
    """Percentile intervals for every metric ``forced_answer_metrics`` returns (class-stratified)."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    draws = {}
    for _ in range(n_boot):
        idx = _stratified_draw(y_true, rng)
        m = forced_answer_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx], classes)
        for k, v in m.items():
            if k in ("n", "n_unreadable"):
                continue
            draws.setdefault(k, []).append(v)
    return {k: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for k, v in draws.items()}


def gemini_cost_usd(df, prices):
    """Dollars for the API calls in a results frame, from measured usage (cached rows included).

    The whole request is billed at the long-context rate when its prompt is over
    ``prices['long_context_over']`` tokens; thinking tokens bill as output.
    """
    if not prices or "usage_prompt_tokens" not in df.columns:
        return None
    pt = pd.to_numeric(df["usage_prompt_tokens"], errors="coerce").fillna(0)
    ot = (pd.to_numeric(df.get("usage_output_tokens"), errors="coerce").fillna(0)
          + pd.to_numeric(df.get("usage_thinking_tokens"), errors="coerce").fillna(0))
    long = pt > prices.get("long_context_over", 200000)
    cost = np.where(long,
                    pt * prices["input_long"] + ot * prices["output_long"],
                    pt * prices["input"] + ot * prices["output"]) / 1e6
    return float(cost.sum())


def _score_rows(base, y_true, y_pred, classes, n_boot):
    row = dict(base)
    row.update(forced_answer_metrics(y_true, y_pred, classes))
    for k, (lo, hi) in bootstrap_ci(y_true, y_pred, classes, n_boot=n_boot).items():
        row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
    # No "above chance" flag: Phil reads the intervals against the baseline row (10-14).
    return row


def paired_difference(y_true, pred_a, pred_b, classes, metric="balanced_accuracy", n_boot=1000, seed=0):
    """metric(b) - metric(a) on the same rows, with a class-stratified paired bootstrap interval."""
    y_true, pred_a, pred_b = list(y_true), list(pred_a), list(pred_b)
    if not y_true:
        return {}
    point = forced_answer_metrics(y_true, pred_b, classes)[metric] - forced_answer_metrics(y_true, pred_a, classes)[metric]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _stratified_draw(y_true, rng)
        t = [y_true[i] for i in idx]
        diffs.append(forced_answer_metrics(t, [pred_b[i] for i in idx], classes)[metric]
                     - forced_answer_metrics(t, [pred_a[i] for i in idx], classes)[metric])
    return {"n": len(y_true), f"{metric}_diff": point,
            f"{metric}_diff_lo": float(np.nanpercentile(diffs, 2.5)),
            f"{metric}_diff_hi": float(np.nanpercentile(diffs, 97.5))}


# Plan comparisons: each arm against the trimmed EHR (`no_image`), on the rows both answered
# (the full-record arm holds only the test split, and drops overflow rows).
TB_COMPARISONS = (("no_image", "no_image_uncut"), ("no_image", "no_notes"), ("no_image", "summary"),
                  ("no_image", "axial_all_image"), ("summary", "summary_image"),
                  # Image-only arms: what the prior CT adds (radiology progression).
                  ("ct_current", "ct_pair"))


def expand_panels(tasks, registry):
    """Config task names with each panel replaced by its member tasks, in order."""
    out = []
    for t in tasks:
        info = registry[t]
        out += [m["task_name"] for m in info["panel"]] if info.get("panel") else [t]
    return out


def split_panel_results(results_dir, panel_info):
    """Write each member task's own results file from a panel's (one row per scan, all answers in one JSON).

    For every model and arm under the panel's folder, member ``<task>`` gets
    ``<results_dir>/<source>/<task>/<model>/<task>_results_<arm>.csv`` with the scan's index, ids, split,
    that task's label (``label__<task>``; scans with no row for the task are left out) and its answer.
    Rewritten from the panel file every time, so a rerun of the panel is picked up. Returns files written.
    """
    results_dir = Path(results_dir)
    src = results_dir / panel_info["task_source_csv"] / panel_info["task_name"]
    written = []
    if not src.exists():
        return written
    keep = [c for c in ("index", "person_id", "ct_accession_number", "split") ]
    for model_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        for fp in sorted(model_dir.glob(f"{panel_info['task_name']}_results_*.csv")):
            arm = extract_experiment_from_filename(fp.name)
            df = pd.read_csv(fp, dtype={"index": str, "ct_accession_number": str}).drop_duplicates(subset=["index"], keep="last")
            answers = df["model_response"].apply(lambda r: json.loads(r) if isinstance(r, str) and r.strip().startswith("{") else {})
            for m in panel_info["panel"]:
                name = m["task_name"]
                col = f"label__{name}"
                if col not in df.columns:
                    raise ValueError(f"{fp}: no {col} column; was it written by the panel loader?")
                out = df[[c for c in keep if c in df.columns]].copy()
                out["label"] = df[col]
                out["model_response"] = answers.apply(lambda a: a.get(name))
                out = out[out["label"].notna()]
                out["label"] = out["label"].astype(int)
                dest = results_dir / m["task_source_csv"] / name / model_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / f"{name}_results_{arm}.csv"
                out.to_csv(path, index=False)
                written.append(path)
    return written


def _split_all_panels(cfg, results_dir, registry, names):
    for t in names:
        if registry[t].get("panel"):
            n = len(split_panel_results(results_dir, registry[t]))
            print(f"[PANEL] {t}: wrote {n} per-task result files")


def score_tb(config_path, output_path=None, n_boot=1000):
    """One table: model x arm x task x split (train/valid/test/pooled), plus baselines.

    Reads ``<results_dir>/<source>/<task>/<model>/<task>_results_<arm>.csv`` and the
    overflow / missing / errors lists beside each. Writes the table to
    ``<results_dir>/tb_scores.csv`` (the mount) unless ``output_path`` is given.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    results_dir = Path(cfg["paths"]["results_dir"])
    registry = {t["task_name"]: t for t in load_valid_tasks(Path(cfg["paths"]["base_dir"]) / cfg["paths"]["valid_tasks"])}
    prices = (cfg.get("vertex") or {}).get("price_per_mtok")
    rows = []
    comparisons = []
    _split_all_panels(cfg, results_dir, registry, cfg.get("tasks", []))
    for task in expand_panels(cfg.get("tasks", []), registry):
        info = registry[task]
        mapping = info["mapping"]
        unscored = [str(u) for u in info.get("unscored_labels", [-1])]
        classes = sorted(int(k) for k in mapping if str(k) not in unscored)
        task_dir = results_dir / info["task_source_csv"] / task
        if not task_dir.exists():
            continue
        pooled_labels = {}
        preds_by_arm = {}
        for model_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            for fp in sorted(model_dir.glob(f"{task}_results_*.csv")):
                arm = extract_experiment_from_filename(fp.name)
                df = pd.read_csv(fp).drop_duplicates(subset=["index"], keep="last")
                df = df[df["label"].astype(str).isin([str(c) for c in classes])]
                df["pred"] = df["model_response"].apply(lambda r: answer_to_label(r, mapping, unscored))
                side = {}
                for kind in ("overflow", "missing", "errors"):
                    sp = fp.with_name(fp.name.replace("_results_", f"_{kind}_"))
                    side[f"n_{kind}"] = len(pd.read_csv(sp)) if sp.exists() else 0
                cost = gemini_cost_usd(df, prices) if "vertex" in str(cfg.get("models")) else None
                if "usage_total_tokens" in df.columns:
                    # Subscription-billed models (Codex): spend is tokens, not dollars.
                    side["tokens_total"] = int(pd.to_numeric(df["usage_total_tokens"], errors="coerce").fillna(0).sum())
                for split in TB_SPLITS + ("pooled",):
                    sub = df if split == "pooled" else df[df["split"] == split]
                    base = {"task": task, "model": model_dir.name, "arm": arm, "split": split, **side}
                    if split == "pooled" and cost is not None:
                        base["cost_usd"] = cost
                    preds = [None if pd.isna(p) else int(p) for p in sub["pred"]]
                    rows.append(_score_rows(base, sub["label"].astype(int).tolist(), preds, classes, n_boot))
                for rec in df[["index", "split", "label"]].to_dict("records"):
                    pooled_labels[rec["index"]] = rec
                preds_by_arm[(model_dir.name, arm)] = df.set_index("index")[["label", "pred"]]
        # Always-majority baseline over every row any arm scored.
        if pooled_labels:
            lab = pd.DataFrame(list(pooled_labels.values()))
            majority = int(lab["label"].astype(int).value_counts().idxmax())
            for split in TB_SPLITS + ("pooled",):
                sub = lab if split == "pooled" else lab[lab["split"] == split]
                base = {"task": task, "model": f"baseline_always_{mapping[str(majority)]}", "arm": "-", "split": split}
                rows.append(_score_rows(base, sub["label"].astype(int).tolist(), [majority] * len(sub), classes, n_boot))
        for (model, a), _ in list(preds_by_arm.items()):
            for base_arm, arm in TB_COMPARISONS:
                if a != arm or (model, base_arm) not in preds_by_arm:
                    continue
                pa, pb = preds_by_arm[(model, base_arm)], preds_by_arm[(model, arm)]
                both = pa.index.intersection(pb.index)
                y = pa.loc[both, "label"].astype(int).tolist()
                fa = [None if pd.isna(p) else int(p) for p in pa.loc[both, "pred"]]
                fb = [None if pd.isna(p) else int(p) for p in pb.loc[both, "pred"]]
                comparisons.append({"task": task, "model": model, "arm": arm, "versus": base_arm,
                                    **paired_difference(y, fa, fb, classes, n_boot=n_boot)})
    out = pd.DataFrame(rows)
    output_path = Path(output_path) if output_path else results_dir / "tb_scores.csv"
    if comparisons:
        cmp_path = output_path.with_name(output_path.stem + "_matched_comparisons.csv")
        pd.DataFrame(comparisons).to_csv(cmp_path, index=False)
        print(f"Wrote {len(comparisons)} matched comparisons to {cmp_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows to {output_path}")
    return out


# ---------------------------------------------------------------------------
# VLM vs linear probe on the same scans (radiology progression).
# Plan: planning/vista_eval_vlm/vlm-radiology-progression-codex.md.
# ---------------------------------------------------------------------------

def _probe_rows(path, shared_task, classes):
    """A vista-eval per-example parquet -> rows of (accession key, split, label, top-class prediction)."""
    from vistabench_eval.frames import normalize_accession
    pe = pd.read_parquet(path)
    if "task" in pe.columns:
        pe = pe[pe["task"] == shared_task]
    pe = pe[pe["label"].astype(int).isin(classes)].copy()
    pe["key"] = pe["accession_number"].map(normalize_accession)
    if pe["key"].duplicated().any():
        raise ValueError(f"{path}: more than one row per accession")
    pe["split"] = pe["split"].replace({"val": "valid"})
    return pe.set_index("key")[["split", "label", "pred_label"]]


def compare_to_probe(config_path, output_path=None, n_boot=1000):
    """Each image arm's VLM answers against the best probe variant, on the scans the VLM answered.

    Config ``probe_comparison``: ``task`` (the VLM task), ``key`` (the row column holding the scan's
    accession), ``arms`` ({arm: [per-example parquet, ...]}, one file per probe variant: the paired
    variant for ``ct_pair``, the current-only one for ``ct_current``), ``select_split`` (default
    valid) and ``score_split`` (default test). The headline variant is the one with the best
    balanced accuracy on ``select_split`` over all its rows, chosen before the score split is read;
    every variant gets a row, ``headline`` marks the chosen one. ``balanced_accuracy_diff`` is VLM
    minus probe, with a class-stratified paired bootstrap interval.
    """
    from vistabench_eval.frames import normalize_accession
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pc = cfg["probe_comparison"]
    results_dir = Path(cfg["paths"]["results_dir"])
    registry = {t["task_name"]: t for t in load_valid_tasks(Path(cfg["paths"]["base_dir"]) / cfg["paths"]["valid_tasks"])}
    names = pc["task"] if isinstance(pc["task"], list) else [pc["task"]]
    _split_all_panels(cfg, results_dir, registry, names)
    out = []
    for task in expand_panels(names, registry):
        out += _compare_one_task(task, registry[task], pc, results_dir, n_boot)
    df = pd.DataFrame(out)
    output_path = Path(output_path) if output_path else results_dir / "probe_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")
    return df


def _compare_one_task(task, info, pc, results_dir, n_boot):
    """compare_to_probe's rows for one task (see there)."""
    from vistabench_eval.frames import normalize_accession
    mapping = info["mapping"]
    unscored = [str(u) for u in info.get("unscored_labels", [-1])]
    classes = sorted(int(k) for k in mapping if str(k) not in unscored)
    shared_task = info.get("shared_task", task)
    select_split, score_split = pc.get("select_split", "valid"), pc.get("score_split", "test")
    key_col = pc.get("key", "ct_accession_number")
    task_dir = results_dir / info["task_source_csv"] / task
    out = []
    for arm, probe_files in pc["arms"].items():
        probes = {Path(p).name.replace("_per_example.parquet", ""): _probe_rows(p, shared_task, classes) for p in probe_files}
        val_ba = {}
        for name, pr in probes.items():
            v = pr[pr["split"] == select_split]
            val_ba[name] = forced_answer_metrics(v["label"].astype(int).tolist(), v["pred_label"].astype(int).tolist(), classes)["balanced_accuracy"]
        best = max(val_ba, key=val_ba.get)
        for model_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()) if task_dir.exists() else []:
            fp = model_dir / f"{task}_results_{arm}.csv"
            if not fp.exists():
                continue
            vlm = pd.read_csv(fp, dtype={key_col: str}).drop_duplicates(subset=["index"], keep="last")
            vlm = vlm[(vlm["split"] == score_split) & vlm["label"].astype(str).isin([str(c) for c in classes])].copy()
            vlm["key"] = vlm[key_col].map(normalize_accession)
            vlm["pred"] = vlm["model_response"].apply(lambda r: answer_to_label(r, mapping, unscored))
            vlm = vlm.set_index("key")
            for name, pr in probes.items():
                both = vlm.index.intersection(pr.index)
                lab_v, lab_p = vlm.loc[both, "label"].astype(int), pr.loc[both, "label"].astype(int)
                if (lab_v != lab_p).any():
                    raise ValueError(f"{name}: {int((lab_v != lab_p).sum())} scans carry a different label than the VLM rows")
                y = lab_v.tolist()
                p_vlm = [None if pd.isna(p) else int(p) for p in vlm.loc[both, "pred"]]
                p_probe = pr.loc[both, "pred_label"].astype(int).tolist()
                row = {"task": task, "model": model_dir.name, "arm": arm, "probe": name,
                       "headline": name == best, f"probe_{select_split}_balanced_accuracy": val_ba[name],
                       "n_vlm_rows": len(vlm), "n_matched": len(both),
                       "vlm_balanced_accuracy": forced_answer_metrics(y, p_vlm, classes).get("balanced_accuracy"),
                       "probe_balanced_accuracy": forced_answer_metrics(y, p_probe, classes).get("balanced_accuracy")}
                row.update(paired_difference(y, p_probe, p_vlm, classes, n_boot=n_boot))
                out.append(row)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--tb", action="store_true", help="tumor-board forced-answer scoring (score_tb)")
    ap.add_argument("--probe", action="store_true", help="VLM vs linear probe on the same scans (compare_to_probe)")
    args = ap.parse_args()
    if args.probe:
        compare_to_probe(args.config, args.output)
    elif args.tb:
        score_tb(args.config, args.output)
    else:
        main(args.config, args.output)
