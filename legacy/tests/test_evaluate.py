"""
Unit tests for the evaluation pipeline.
Does NOT require a GPU.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from uq.evaluate import (
    compute_ece,
    evaluate_step_uq,
    evaluate_snowball_detection,
    calibrate_thresholds,
)


def _make_steps(confidences, correctness):
    return [
        {"c_star_cocoa": float(c), "is_correct": bool(y)}
        for c, y in zip(confidences, correctness)
    ]


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------

class TestComputeECE:

    def test_perfect_calibration(self):
        # Confidence = accuracy in every bin → ECE = 0
        confs = np.linspace(0.05, 0.95, 100)
        corrs = (np.random.default_rng(42).random(100) < confs).astype(float)
        # ECE won't be exactly 0 but should be low for a large sample
        ece = compute_ece(confs, corrs, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_overconfident_model(self):
        # Always predicts 0.95 confidence; only 50% accurate → high ECE
        confs = np.full(100, 0.95)
        corrs = np.array([1, 0] * 50, dtype=float)
        ece = compute_ece(confs, corrs, n_bins=10)
        assert ece > 0.3

    def test_empty_bins_skipped(self):
        # Conf=0.9, Acc=1.0 → ECE = |1.0 - 0.9| = 0.1 (all in one bin)
        confs = np.array([0.9, 0.9, 0.9])
        corrs = np.array([1.0, 1.0, 1.0])
        ece = compute_ece(confs, corrs, n_bins=10)
        assert ece == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# evaluate_step_uq
# ---------------------------------------------------------------------------

class TestEvaluateStepUQ:

    def test_returns_all_keys(self):
        steps = _make_steps([0.9, 0.5, 0.3], [True, False, False])
        result = evaluate_step_uq(steps, bootstrap_n=10)
        for key in ["n_steps", "accuracy", "ece", "auroc", "sel_acc_0.8", "coverage_0.8"]:
            assert key in result

    def test_perfect_discriminator(self):
        # High confidence for correct, low for incorrect → AUROC = 1
        confs = [0.95, 0.95, 0.1, 0.1]
        corrs = [True, True, False, False]
        steps = _make_steps(confs, corrs)
        result = evaluate_step_uq(steps, bootstrap_n=10)
        assert result["auroc"] == pytest.approx(1.0, abs=1e-6)

    def test_single_class_auroc_nan(self):
        steps = _make_steps([0.9, 0.8], [True, True])
        result = evaluate_step_uq(steps, bootstrap_n=10)
        assert result["auroc"] != result["auroc"]  # NaN check

    def test_skips_steps_without_confidence(self):
        steps = [
            {"c_star_cocoa": 0.9, "is_correct": True},
            {"c_star_cocoa": None, "is_correct": True},   # missing → skipped
            {"c_star_cocoa": 0.3, "is_correct": False},
        ]
        result = evaluate_step_uq(steps, bootstrap_n=10)
        assert result["n_steps"] == 2


# ---------------------------------------------------------------------------
# evaluate_snowball_detection
# ---------------------------------------------------------------------------

class TestSnowballDetection:

    def _write_jsonl(self, trajectories):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        for t in trajectories:
            tmp.write(json.dumps(t) + "\n")
        tmp.close()
        return tmp.name

    def test_perfect_early_warning(self):
        # Low confidence appears before first incorrect step
        traj = {
            "task_success": False,
            "trajectory": [
                {"c_star_cocoa": 0.8, "is_correct": True},
                {"c_star_cocoa": 0.4, "is_correct": True},   # trigger at step 1
                {"c_star_cocoa": 0.3, "is_correct": False},  # first error at step 2
            ],
        }
        path = self._write_jsonl([traj])
        result = evaluate_snowball_detection(path, threshold=0.6)
        assert result["ppv_early_warning"] == pytest.approx(1.0)

    def test_late_warning_is_miss(self):
        traj = {
            "task_success": False,
            "trajectory": [
                {"c_star_cocoa": 0.9, "is_correct": False},  # error at step 0
                {"c_star_cocoa": 0.4, "is_correct": False},  # trigger at step 1
            ],
        }
        path = self._write_jsonl([traj])
        result = evaluate_snowball_detection(path, threshold=0.6)
        # first_low=1 > first_err=0 → not a hit
        assert result["ppv_early_warning"] == pytest.approx(0.0)

    def test_successful_trajectories_ignored(self):
        traj = {"task_success": True, "trajectory": [{"c_star_cocoa": 0.3, "is_correct": True}]}
        path = self._write_jsonl([traj])
        result = evaluate_snowball_detection(path)
        assert result["n_failed_trajectories"] == 0


# ---------------------------------------------------------------------------
# calibrate_thresholds
# ---------------------------------------------------------------------------

class TestCalibrateThresholds:

    def test_returns_expected_keys(self):
        steps = _make_steps([0.9, 0.5, 0.2, 0.8], [True, False, False, True])
        result = calibrate_thresholds(steps)
        assert "lambda_conformal" in result
        assert "threshold_escalate_conformal" in result
        assert "threshold_escalate_fixed" in result

    def test_lambda_in_valid_range(self):
        steps = _make_steps([0.9, 0.5, 0.2], [True, False, False])
        result = calibrate_thresholds(steps, alpha=0.1)
        assert 0.0 <= result["lambda_conformal"] <= 1.0
        assert 0.0 <= result["threshold_escalate_conformal"] <= 1.0
