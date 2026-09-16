"""
Answer-level baseline UQ methods, replicating the consulting project
(Hobelsberger 2026) in the agentic setting.

These methods apply MSP and CoCoA to the FINAL ANSWER of each trajectory (not
individual steps), providing the answer-level baselines (MSP-Answer, CoCoA-Answer)
reported in the thesis alongside the step-level StepCoCoA and BranchConsistency
estimators.

Each function takes a list of trajectory dicts (from JSONL) and returns
a list of step dicts with confidence scores under a new key so they can
be passed directly to `evaluate_step_uq` / `evaluate_trajectory_uq`.

Methods
-------
- msp_answer_level_proper : MSP-Answer (reported baseline; proper calib/test split).
- msp_answer_level        : Legacy single-split MSP-Answer, superseded by
                            msp_answer_level_proper; NOT used for any reported number.
- cocoa_answer_level      : CoCoA-Answer (reported baseline).
- vce_answer_level        : Verbalized confidence, NOT one of the thesis's four
                            reported estimators (StepCoCoA, BranchConsistency,
                            MSP-Answer, CoCoA-Answer); kept working as an
                            excluded/ablation baseline only.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MSP (answer-level)
# ---------------------------------------------------------------------------

def msp_answer_level_proper(
    calib_trajectories: list[dict],
    test_trajectories: list[dict],
    quantile_clip: float = 0.98,
) -> tuple[list[dict], dict]:
    """
    Compute answer-level MSP confidence with PROPER calib/test split (NO DATA LEAKAGE).

    Fits normalisation parameters on the calib set, evaluates on the held-out test set
    (the thesis's final results use a 50/50 calib/test split). This ensures fair
    comparison with step-wise methods which use the same split.

    Parameters
    ----------
    calib_trajectories : list[dict]
        Trajectories to fit normaliser on (calibration split)
    test_trajectories : list[dict]
        Trajectories to evaluate on (held-out test split)
    quantile_clip : float
        Quantile for clipping (default 0.98)

    Returns
    -------
    tuple[list[dict], dict]
        - List of step dicts with 'c_msp_answer' key (for evaluation)
        - Dict with fitted parameters (q98, u_min) for logging
    """
    # --- Step 1: Fit on calib set ---
    calib_raw_scores: list[float] = []
    for traj in calib_trajectories:
        steps = traj.get("trajectory", [])
        if not steps:
            continue
        last = steps[-1]
        raw_logprob = last.get("seq_logprob", 0.0)
        u_msp = -raw_logprob
        calib_raw_scores.append(u_msp)

    # Fit normalisation from calib scores only
    if calib_raw_scores:
        arr = np.array(calib_raw_scores)
        q98   = float(np.quantile(arr, quantile_clip))
        u_min = float(np.clip(arr, None, q98).min())
        denom = q98 - u_min if q98 - u_min > 1e-9 else 1.0
    else:
        q98, u_min, denom = 1.0, 0.0, 1.0

    logger.info(
        "MSP-Answer (proper split): Fitted q98=%.4f, u_min=%.4f on %d calib samples",
        q98, u_min, len(calib_raw_scores)
    )

    # --- Step 2: Evaluate on test set ---
    out: list[dict] = []
    for traj in test_trajectories:
        steps = traj.get("trajectory", [])
        if not steps:
            continue
        last = steps[-1]
        raw_logprob = last.get("seq_logprob", 0.0)
        u_msp = -raw_logprob

        u_clip = min(u_msp, q98)
        c_star = float(1.0 - np.clip((u_clip - u_min) / denom, 0.0, 1.0))

        out.append({
            "task_id":       traj["idx"] if "idx" in traj else traj.get("task_id"),
            "c_msp_answer":  c_star,
            "is_correct":    int(traj.get("task_success", False)),
            "task_success":  traj.get("task_success", False),
        })

    params = {"q98": q98, "u_min": u_min, "n_calib_samples": len(calib_raw_scores)}
    return out, params


def msp_answer_level(trajectories: list[dict], quantile_clip: float = 0.98) -> list[dict]:
    """
    Compute answer-level MSP confidence (LEGACY: single-split mode).

    NOT used for any thesis-reported number: superseded by msp_answer_level_proper(),
    which is what the final MSP-Answer results use.

    WARNING: This function fits and evaluates on the same data (potential data leakage).
    Only use when you have a SINGLE trajectories.jsonl file (--split-phase all).

    For a proper calib/test split, use msp_answer_level_proper() instead.

    Uses quantile clipping for normalization (consistent with step-wise methods).

    Each returned step dict has key 'c_msp_answer' and 'is_correct' from
    task-level success.
    """
    logger.warning(
        "msp_answer_level() called without proper split. "
        "This fits and evaluates on the same data (data leakage). "
        "Use msp_answer_level_proper(calib_trajs, test_trajs) for proper evaluation."
    )

    raw_scores: list[float] = []
    records: list[dict] = []

    for traj in trajectories:
        steps = traj.get("trajectory", [])
        if not steps:
            continue
        last = steps[-1]
        raw_logprob = last.get("seq_logprob", 0.0)
        u_msp = -raw_logprob

        raw_scores.append(u_msp)
        records.append({
            "task_id":     traj["idx"] if "idx" in traj else traj.get("task_id"),
            "u_msp_raw":   u_msp,
            "is_correct":  int(traj.get("task_success", False)),
            "task_success": traj.get("task_success", False),
        })

    # Fit normalisation from all collected scores (SAME DATA)
    if raw_scores:
        arr = np.array(raw_scores)
        q98   = float(np.quantile(arr, quantile_clip))
        u_min = float(np.clip(arr, None, q98).min())
        denom = q98 - u_min if q98 - u_min > 1e-9 else 1.0
    else:
        q98, u_min, denom = 1.0, 0.0, 1.0

    out: list[dict] = []
    for rec in records:
        u_raw = rec["u_msp_raw"]
        u_clip = min(u_raw, q98)
        c_star = float(1.0 - np.clip((u_clip - u_min) / denom, 0.0, 1.0))
        out.append({
            "task_id":       rec["task_id"],
            "c_msp_answer":  c_star,
            "is_correct":    rec["is_correct"],
            "task_success":  rec["task_success"],
        })
    return out


# ---------------------------------------------------------------------------
# VCE (answer-level): requires stored vce field or re-elicitation
# ---------------------------------------------------------------------------

def vce_answer_level(trajectories: list[dict]) -> list[dict]:
    """
    Extract verbalized confidence from stored 'confidence_vce' field on
    the last step (populated during experiment if elicited).

    NOT one of the thesis's four reported estimators (StepCoCoA, BranchConsistency,
    MSP-Answer, CoCoA-Answer). Kept working as an excluded/ablation baseline only.
    Never treat 'VCE-Answer' output from this function as a reported thesis result.

    Returns step dicts with key 'c_vce_answer'.
    """
    out: list[dict] = []
    for traj in trajectories:
        steps = traj.get("trajectory", [])
        if not steps:
            continue
        last = steps[-1]
        c_vce = last.get("confidence_vce")
        if c_vce is None:
            logger.debug(
                "VCE not stored for task %s, skipping.",
                traj["idx"] if "idx" in traj else traj.get("task_id"),
            )
            continue
        out.append({
            "task_id":      traj["idx"] if "idx" in traj else traj.get("task_id"),
            "c_vce_answer": float(c_vce),
            "is_correct":   int(traj.get("task_success", False)),
            "task_success": traj.get("task_success", False),
        })
    return out


# ---------------------------------------------------------------------------
# CoCoA (answer-level)
# ---------------------------------------------------------------------------

def cocoa_answer_level(
    trajectories: list[dict],
    quantile_clip: float = 0.98,
) -> list[dict]:
    """
    Extract answer-level CoCoA confidence from the last step's pre-computed
    c_star_cocoa (which is properly calibrated and normalized).

    The final step's c_star_cocoa is the authoritative answer-level confidence;
    it's already normalized via the StepwiseCoCoA calibration pipeline and
    shouldn't be recomputed from raw consistency_scores (which may be on a
    different scale depending on the similarity metric used).

    Returns step dicts with key 'c_cocoa_answer'.
    """
    out: list[dict] = []

    for traj in trajectories:
        steps = traj.get("trajectory", [])
        if not steps:
            continue
        last = steps[-1]

        # Use the pre-computed c_star_cocoa from the final step (already calibrated)
        c_cocoa = last.get("c_star_cocoa")
        if c_cocoa is None:
            # Fallback: estimate from seq_logprob if c_star_cocoa not available
            raw_logprob = last.get("seq_logprob", 0.0)
            u_belief = -raw_logprob
            c_cocoa = float(1.0 / (1.0 + u_belief)) if u_belief > 0 else 1.0

        out.append({
            "task_id":        traj["idx"] if "idx" in traj else traj.get("task_id"),
            "c_cocoa_answer": float(c_cocoa),
            "is_correct":     int(traj.get("task_success", False)),
            "task_success":   traj.get("task_success", False),
        })
    return out


# ---------------------------------------------------------------------------
# Smart loader for calib/test split or fallback to single split
# ---------------------------------------------------------------------------

def load_trajectories(jsonl_path: str | Path) -> list[dict]:
    """Load full trajectory records (not flattened steps) from a JSONL file."""
    trajectories: list[dict] = []
    with open(jsonl_path) as fh:
        for line in fh:
            trajectories.append(json.loads(line))
    return trajectories


def load_trajectories_with_split(results_dir: str | Path) -> tuple[list[dict], list[dict], bool]:
    """
    Load trajectories, preferring calib/test split if available.

    Intelligently detects whether results directory has:
    1. Proper calib/test split (calib/ and test/ subdirectories)
    2. Single trajectories.jsonl (legacy --split-phase all mode)

    Parameters
    ----------
    results_dir : str or Path
        Results directory (e.g., "results/gsm8k")

    Returns
    -------
    tuple[list[dict], list[dict], bool]
        - calib_trajectories (empty list if not found)
        - test_trajectories (all trajectories if single-split mode)
        - has_proper_split (True if calib/ and test/ exist, False otherwise)

    Raises
    ------
    FileNotFoundError
        If neither calib/test split nor single trajectories.jsonl exists
    """
    results_dir = Path(results_dir)

    calib_path = results_dir / "calib" / "trajectories.jsonl"
    test_path = results_dir / "test" / "trajectories.jsonl"
    single_path = results_dir / "trajectories.jsonl"

    # --- Check for proper calib/test split ---
    if calib_path.exists() and test_path.exists():
        logger.info(
            "Found proper calib/test split: calib=%s, test=%s",
            calib_path, test_path
        )
        calib_trajs = load_trajectories(calib_path)
        test_trajs = load_trajectories(test_path)
        return calib_trajs, test_trajs, True

    # --- Fallback to single trajectories.jsonl (legacy mode) ---
    if single_path.exists():
        logger.warning(
            "Using legacy single-split mode (trajectories.jsonl). "
            "For proper calib/test split evaluation, use --split-phase full "
            "which creates calib/ and test/ subdirectories."
        )
        all_trajs = load_trajectories(single_path)
        return [], all_trajs, False

    # --- Error: neither exists ---
    raise FileNotFoundError(
        f"Could not find trajectories in {results_dir}:\n"
        f"  Expected either:\n"
        f"    1. Proper split: {calib_path} AND {test_path}\n"
        f"    2. Legacy mode: {single_path}\n"
        f"  Run with --split-phase full for proper evaluation."
    )
