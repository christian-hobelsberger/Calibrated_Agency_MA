"""
Evaluation pipeline for step-level and trajectory-level UQ metrics.

Trajectory-level metrics (ACC, ECE, AUROC, SelAcc@0.8, Cov@0.8, TR, PPV, FER; see
`evaluate_trajectory_uq` and `evaluate_snowball_detection`) are the ones reported in the
thesis. All final results use `task_success` (trajectory-level ground truth), never the
heuristic step-level `is_correct` labels from `uq/annotation.py`.

`evaluate_step_uq` and `calibrate_thresholds` compute metrics from step-level `is_correct`
labels instead. These were exploratory/diagnostic during development and are NOT used to
produce any metric reported in the thesis. The thesis's escalation policy uses the fixed
thresholds lambda_snow=0.6 / lambda_auto=0.8, not the conformal threshold computed here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_step_records(jsonl_path: str | Path) -> list[dict]:
    """
    Load all step records from a trajectories JSONL file.

    Each step dict gets a 'task_success' field injected from the parent record.
    """
    steps: list[dict] = []
    with open(jsonl_path) as fh:
        for line in fh:
            traj = json.loads(line)
            for step in traj["trajectory"]:
                step["task_success"] = traj["task_success"]
                steps.append(step)
    return steps


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (equal-width bins).

    ECE = Σ_b (|b|/n) |acc(b) - conf(b)|
    """
    ece = 0.0
    n = len(confidences)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correctness[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def evaluate_step_uq(
    steps: list[dict],
    confidence_key: str = "c_star_cocoa",
    n_bins: int = 10,
    bootstrap_n: int = 2000,
    trajectories: list[dict] | None = None,
) -> dict:
    """
    Compute ACC, ECE, AUROC, and selective accuracy from annotated step records.

    NOT used for any thesis-reported metric: relies on heuristic step-level `is_correct`
    labels (see `uq/annotation.py`), which the thesis methodology restricts to qualitative
    failure-mode illustration only. Kept as a diagnostic/exploratory tool from development;
    use `evaluate_trajectory_uq` for reportable numbers.

    Parameters
    ----------
    steps          : List of step dicts (from load_step_records).
    confidence_key : Key to use as confidence score.
    n_bins         : Number of ECE bins.
    bootstrap_n    : Bootstrap resamples for AUROC confidence interval.
    trajectories   : Optional list of raw trajectory dicts. When provided,
                     accuracy is task-level (fraction of successful trajectories)
                     instead of step-level.

    Returns
    -------
    dict with keys: n_steps, accuracy, ece, auroc, auroc_ci_95,
                    sel_acc_0.8, coverage_0.8
    """
    valid = [
        s for s in steps
        if s.get(confidence_key) is not None and s.get("is_correct") is not None
    ]
    if not valid:
        logger.warning("evaluate_step_uq: no valid steps for key '%s'", confidence_key)
        return {k: float("nan") for k in
                ["n_steps", "accuracy", "ece", "auroc", "sel_acc_0.8", "coverage_0.8"]}

    confidences = np.array([s[confidence_key] for s in valid], dtype=float)
    correctness = np.array([int(s["is_correct"]) for s in valid], dtype=float)

    if trajectories is not None:
        accuracy = float(np.mean([int(traj["task_success"]) for traj in trajectories]))
    else:
        accuracy = float(correctness.mean())
    ece = compute_ece(confidences, correctness, n_bins=n_bins)

    # AUROC
    if len(np.unique(correctness)) < 2:
        auroc = float("nan")
        auroc_ci: tuple[float, float] = (float("nan"), float("nan"))
    else:
        auroc = float(roc_auc_score(correctness, confidences))
        try:
            boot = bootstrap(
                (correctness, confidences),
                statistic=lambda y, s, axis: roc_auc_score(y, s),
                n_resamples=bootstrap_n,
                confidence_level=0.95,
                method="percentile",
                paired=True,
            )
            auroc_ci = (
                float(boot.confidence_interval.low),
                float(boot.confidence_interval.high),
            )
        except Exception as exc:
            logger.debug("Bootstrap CI failed: %s", exc)
            auroc_ci = (float("nan"), float("nan"))

    # Selective accuracy at C* > 0.8
    mask_08 = confidences > 0.8
    sel_acc  = float(correctness[mask_08].mean()) if mask_08.any() else float("nan")
    coverage = float(mask_08.mean())

    return {
        "n_steps":      len(valid),
        "accuracy":     accuracy,
        "ece":          ece,
        "auroc":        auroc,
        "auroc_ci_95":  auroc_ci,
        "sel_acc_0.8":  sel_acc,
        "coverage_0.8": coverage,
    }


def evaluate_trajectory_uq(
    trajectories: list[dict],
    confidence_key: str = "c_star_cocoa",
    aggregation: str = "min",
    n_bins: int = 10,
) -> dict:
    """
    Compute task-level metrics by aggregating step confidences per trajectory.

    This addresses the core use case: predicting task success from trajectory-level
    confidence aggregation. Unlike evaluate_step_uq (which lumps steps across
    trajectories), this method preserves trajectory boundaries and evaluates
    whether aggregated step confidence predicts task_success.

    Parameters
    ----------
    trajectories : List of trajectory dicts with 'trajectory' and 'task_success' fields.
    confidence_key : Step-level confidence field to aggregate (e.g., 'c_star_cocoa').
    aggregation : How to aggregate step confidences.
        - 'min': minimum step confidence in trajectory (captures worst-case step)
        - 'mean': average step confidence (overall coherence)
        - 'median': median step confidence (robust aggregate)
    n_bins : Number of ECE bins.

    Returns
    -------
    dict with keys: n_trajs, accuracy, auroc, ece, sel_acc_0.8, coverage_0.8
    """
    valid = []
    for traj in trajectories:
        steps = traj.get("trajectory", [])
        confs = [s.get(confidence_key) for s in steps
                 if s.get(confidence_key) is not None]

        if not confs:
            continue

        # Aggregate step confidences
        if aggregation == "min":
            agg_conf = float(np.min(confs))
        elif aggregation == "mean":
            agg_conf = float(np.mean(confs))
        elif aggregation == "median":
            agg_conf = float(np.median(confs))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        valid.append({
            "confidence": agg_conf,
            "task_success": int(traj.get("task_success", False)),
        })

    if not valid or len(valid) < 2:
        logger.warning(
            "evaluate_trajectory_uq: insufficient valid trajectories for '%s' (n=%d)",
            confidence_key, len(valid)
        )
        return {k: float("nan") for k in
                ["n_trajs", "accuracy", "auroc", "ece", "sel_acc_0.8", "coverage_0.8"]}

    confidences = np.array([v["confidence"] for v in valid], dtype=float)
    task_labels = np.array([v["task_success"] for v in valid], dtype=float)

    accuracy = float(task_labels.mean())
    ece = compute_ece(confidences, task_labels, n_bins=n_bins)

    # AUROC at trajectory level
    if len(np.unique(task_labels)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(task_labels, confidences))

    # Selective accuracy at 0.8
    mask_08 = confidences > 0.8
    sel_acc = float(task_labels[mask_08].mean()) if mask_08.any() else float("nan")
    coverage = float(mask_08.mean())

    return {
        "n_trajs":      len(valid),
        "accuracy":     accuracy,
        "auroc":        auroc,
        "ece":          ece,
        "sel_acc_0.8":  sel_acc,
        "coverage_0.8": coverage,
    }


# ---------------------------------------------------------------------------
# Snowball detection
# ---------------------------------------------------------------------------

def evaluate_snowball_detection(
    jsonl_path: str | Path,
    confidence_key: str = "c_star_cocoa",
    threshold: float = 0.6,
    min_warnings: int = 1,
) -> dict:
    """
    Evaluate snowball detection using trajectory-level correctness (task_success).

    Trigger rate (recall): fraction of FAILED trajectories where ≥min_warnings
    low-confidence steps fire.

    PPV (precision): fraction of trajectories that met the warning criterion which
    actually failed (P(fail | ≥min_warnings warnings)). Does not rely on step-level
    is_correct labels.

    Parameters
    ----------
    min_warnings : minimum number of low-confidence steps required to count a
                   trajectory as warned (default 1).

    Returns
    -------
    dict with keys:
      n_failed_trajectories : Number of failed trajectories in the file.
      ppv_early_warning     : Precision: P(trajectory failed | ≥min_warnings warnings).
      trigger_rate          : Recall: fraction of failed trajectories with ≥min_warnings warnings.
    """
    ppv_hits: list[int] = []
    trigger_fired: list[int] = []
    fer_hits: list[int] = []

    with open(jsonl_path) as fh:
        for line in fh:
            traj = json.loads(line)
            failed = not traj.get("task_success", True)

            steps = traj["trajectory"]
            confs = [s.get(confidence_key) if s.get(confidence_key) is not None
                     else 1.0 for s in steps]

            warned = sum(c < threshold for c in confs) >= min_warnings

            if failed:
                trigger_fired.append(int(warned))
            else:
                fer_hits.append(int(warned))
            if warned:
                ppv_hits.append(int(failed))

    return {
        "n_failed_trajectories": len(trigger_fired),
        "ppv_early_warning":     float(np.mean(ppv_hits)) if ppv_hits else float("nan"),
        "trigger_rate":          float(np.mean(trigger_fired)) if trigger_fired else float("nan"),
        "fer":                   float(np.mean(fer_hits)) if fer_hits else float("nan"),
    }


# ---------------------------------------------------------------------------
# Conformal threshold calibration
# ---------------------------------------------------------------------------

def calibrate_thresholds(
    calib_steps: list[dict],
    confidence_key: str = "c_star_cocoa",
    alpha: float = 0.1,
) -> dict:
    """
    Derive escalation thresholds using conformal risk control.

    NOT used for the thesis's reported escalation policy, which uses the fixed thresholds
    lambda_snow=0.6 / lambda_auto=0.8 (not tuned per-benchmark). This function relies on
    heuristic step-level `is_correct` labels and is kept as a diagnostic/exploratory tool
    from development, not a source of any reported thesis number.

    Given a calibration set of annotated step records, compute the threshold
    λ such that the FDR (fraction of wrong actions below threshold) is
    bounded by alpha with high probability.

    Also returns the fixed thresholds (lambda_snow=0.6, lambda_auto=0.8) for comparison.

    Returns
    -------
    dict with keys:
      lambda_conformal : Conformal threshold (escalate if C* < 1 - lambda).
      threshold_escalate_conformal : Equivalent to 1 - lambda_conformal.
      threshold_escalate_fixed     : Fixed threshold lambda_snow = 0.6.
      threshold_auto_fixed         : Fixed threshold lambda_auto = 0.8.
    """
    valid = [
        s for s in calib_steps
        if s.get(confidence_key) is not None and s.get("is_correct") is not None
    ]
    if not valid:
        logger.warning("calibrate_thresholds: no valid calibration steps.")
        return {}

    confidences = np.array([s[confidence_key] for s in valid])
    correctness = np.array([int(s["is_correct"]) for s in valid])

    # Conformity scores for incorrect predictions: s_i = 1 - C*(a_i)
    incorrect_confs = confidences[correctness == 0]
    if len(incorrect_confs) == 0:
        lambda_hat = 0.0
    else:
        scores = 1.0 - incorrect_confs
        # lambda_hat: (1-alpha) quantile of conformity scores of incorrect steps
        lambda_hat = float(np.quantile(scores, 1.0 - alpha))

    threshold_escalate_conformal = 1.0 - lambda_hat

    return {
        "lambda_conformal":                lambda_hat,
        "threshold_escalate_conformal":    threshold_escalate_conformal,
        "threshold_escalate_fixed":        0.6,
        "threshold_auto_fixed":            0.8,
        "alpha":                           alpha,
        "n_calib_steps":                   len(valid),
        "n_incorrect_calib_steps":         int((correctness == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Pretty-print utilities
# ---------------------------------------------------------------------------

def print_results_table(results_by_method: dict[str, dict]) -> None:
    """Print a formatted comparison table to stdout."""
    cols = ["Method", "ACC", "ECE", "AUROC", "SelAcc@.8", "Coverage"]
    widths = [32, 6, 6, 7, 10, 9]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print("\n" + header)
    print("-" * len(header))
    for method, r in results_by_method.items():
        row = [
            method[:widths[0]].ljust(widths[0]),
            f"{r.get('accuracy', float('nan')):.3f}",
            f"{r.get('ece', float('nan')):.3f}",
            f"{r.get('auroc', float('nan')):.3f}",
            f"{r.get('sel_acc_0.8', float('nan')):.3f}",
            f"{r.get('coverage_0.8', float('nan')):.3f}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print()


def print_trajectory_comparison(results_by_method: dict[str, dict]) -> None:
    """
    Print trajectory-level AUROC comparison: step-level vs trajectory-level (min/mean).
    Highlights differences between evaluating steps lumped together vs. per-trajectory.
    """
    print("\n" + "="*100)
    print("TRAJECTORY-LEVEL AUROC: Step-level vs Trajectory-level Aggregation")
    print("="*100)
    print(f"{'Method':<30} {'Step AUROC':<15} {'Traj Min':<15} {'Traj Mean':<15} {'Min vs Step':<15}")
    print("-"*100)

    for method, metrics in results_by_method.items():
        if not isinstance(metrics, dict):
            continue

        step_auc = metrics.get('step', {}).get('auroc', float('nan'))
        traj_min = metrics.get('traj_min', {}).get('auroc', float('nan'))
        traj_mean = metrics.get('traj_mean', {}).get('auroc', float('nan'))

        # For answer-level methods, traj_answer is the only trajectory metric
        if 'traj_answer' in metrics:
            traj_min = metrics.get('traj_answer', {}).get('auroc', float('nan'))
            traj_mean = float('nan')

        diff = traj_min - step_auc if not (np.isnan(step_auc) or np.isnan(traj_min)) else float('nan')

        diff_str = f"{diff:+.3f}" if not np.isnan(diff) else "N/A"

        row = [
            method[:30].ljust(30),
            f"{step_auc:.3f}".ljust(15),
            f"{traj_min:.3f}".ljust(15),
            f"{traj_mean:.3f}".ljust(15),
            diff_str.ljust(15),
        ]
        print("".join(row))
    print()
