"""
ARCHIVED, see legacy/README.md. Never actually used to produce a thesis result: final
experiments were run via direct `python -m experiments.<benchmark>_agent` CLI invocations,
and evaluation was done via the notebooks (see notebooks/comprehensive_results_analysis.ipynb),
not this script. Also note: evaluate_experiment() below computes metrics via evaluate_step_uq
(heuristic step-level labels) for every method, which the thesis's methodology never reports.

Calibrated Agency — Main Experiment Runner (unused combined runner)

Runs all experiments (GSM8K, AgentBench-OS, AgentBench-DB, ToolBench-G1)
and produces the results table for the thesis.

Usage examples:

    # Run all experiments (full scale)
    python run_all_experiments.py

    # Run only GSM8K with a 50-sample pilot
    python run_all_experiments.py --experiment gsm8k --n-samples 50

    # Run specific experiment with a custom model
    python run_all_experiments.py --experiment agentbench_os --model meta-llama/Llama-3.1-70B-Instruct

    # Dry run: show experiment configs without running
    python run_all_experiments.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import mlflow

from config import DEFAULT_CONFIG
from experiments.agentbench_agent import run_agentbench_experiment
from experiments.gsm8k_agent import run_gsm8k_experiment
from experiments.hotpotqa_agent import run_hotpotqa_experiment
from legacy.toolbench_agent import run_toolbench_experiment
from uq.baselines import cocoa_answer_level, load_trajectories, msp_answer_level, vce_answer_level
from uq.evaluate import (
    calibrate_thresholds,
    evaluate_snowball_detection,
    evaluate_step_uq,
    load_step_records,
    print_results_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "gsm8k": {
        "coro":       run_gsm8k_experiment,
        "kwargs_key": "gsm8k",
        "result_dir": "results/gsm8k",
        "label":      "GSM8K",
    },
    "agentbench_os": {
        "coro":       run_agentbench_experiment,
        "kwargs_key": "agentbench_os",
        "result_dir": "results/agentbench_os",
        "label":      "AgentBench-OS",
    },
    "agentbench_db": {
        "coro":       run_agentbench_experiment,
        "kwargs_key": "agentbench_db",
        "result_dir": "results/agentbench_db",
        "label":      "AgentBench-DB",
    },
    "toolbench_g1": {
        "coro":       run_toolbench_experiment,
        "kwargs_key": "toolbench_g1",
        "result_dir": "results/toolbench_g1",
        "label":      "ToolBench-G1",
    },
    "hotpotqa": {
        "coro":       run_hotpotqa_experiment,
        "kwargs_key": "hotpotqa",
        "result_dir": "results/hotpotqa",
        "label":      "HotpotQA",
    },
}

# Default kwargs per experiment (overridden by CLI args)
def _build_kwargs(key: str, model_name: str, n_samples: int | None) -> dict:
    cfg = DEFAULT_CONFIG
    base = dict(model_name=model_name)

    if key == "gsm8k":
        return {**base,
                "n_samples":  n_samples or cfg.experiment.gsm8k_n,
                "output_dir": "results/gsm8k",
                "cocoa_m":    cfg.uq.cocoa_m,
                "branch_m":   cfg.uq.branch_m,
                "temperature": cfg.uq.temperature}

    if key == "agentbench_os":
        return {**base,
                "env_type":   "os",
                "n_samples":  n_samples or cfg.experiment.agentbench_os_n,
                "output_dir": "results/agentbench_os",
                "cocoa_m":    cfg.uq.cocoa_m,
                "temperature": cfg.uq.temperature}

    if key == "agentbench_db":
        return {**base,
                "env_type":   "db",
                "n_samples":  n_samples or cfg.experiment.agentbench_db_n,
                "output_dir": "results/agentbench_db",
                "cocoa_m":    cfg.uq.cocoa_m,
                "temperature": cfg.uq.temperature}

    if key == "toolbench_g1":
        return {**base,
                "split":      "G1",
                "n_samples":  n_samples or cfg.experiment.toolbench_g1_n,
                "output_dir": "results/toolbench_g1",
                "cocoa_m":    cfg.uq.cocoa_m,
                "temperature": cfg.uq.temperature}

    if key == "hotpotqa":
        return {**base,
                "n_samples":  n_samples or cfg.experiment.hotpotqa_n,
                "output_dir": "results/hotpotqa",
                "cocoa_m":    cfg.uq.cocoa_m,
                "branch_m":   cfg.uq.branch_m,
                "temperature": cfg.uq.temperature}

    raise ValueError(f"Unknown experiment key: {key}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_experiment(result_dir: str, label: str) -> dict:
    """Run full evaluation suite for one experiment's output directory."""
    traj_path = Path(result_dir) / "trajectories.jsonl"
    if not traj_path.exists():
        logger.warning("No trajectories file found at %s", traj_path)
        return {}

    steps = load_step_records(traj_path)
    trajectories = load_trajectories(traj_path)

    # --- Step-wise CoCoA (proposed method) ---
    metrics_stepcocoa = evaluate_step_uq(
        steps, confidence_key="c_star_cocoa",
        bootstrap_n=DEFAULT_CONFIG.experiment.bootstrap_n,
        trajectories=trajectories,
    )

    # --- Branching consistency (proposed: use branch_consistency as signal) ---
    metrics_branch = evaluate_step_uq(
        steps, confidence_key="branch_consistency",
        bootstrap_n=DEFAULT_CONFIG.experiment.bootstrap_n,
        trajectories=trajectories,
    )

    # --- Answer-level baselines ---
    msp_steps   = msp_answer_level(trajectories)
    vce_steps   = vce_answer_level(trajectories)
    cocoa_steps = cocoa_answer_level(trajectories)

    metrics_msp   = evaluate_step_uq(msp_steps,   confidence_key="c_msp_answer",   trajectories=trajectories)
    metrics_vce   = evaluate_step_uq(vce_steps,   confidence_key="c_vce_answer",   trajectories=trajectories) if vce_steps else {}
    metrics_cocoa = evaluate_step_uq(cocoa_steps, confidence_key="c_cocoa_answer", trajectories=trajectories)

    # --- Snowball detection ---
    snowball = evaluate_snowball_detection(
        traj_path, confidence_key="c_star_cocoa",
        threshold=DEFAULT_CONFIG.uq.threshold_warn,
    )

    # --- Conformal threshold calibration (using all steps as proxy calib set) ---
    calib_result = calibrate_thresholds(
        steps, confidence_key="c_star_cocoa",
        alpha=DEFAULT_CONFIG.uq.alpha,
    )

    all_metrics = {
        f"{label}/StepCoCoA":     metrics_stepcocoa,
        f"{label}/BranchConsistency": metrics_branch,
        f"{label}/MSP-Answer":    metrics_msp,
        f"{label}/VCE-Answer":    metrics_vce,
        f"{label}/CoCoA-Answer":  metrics_cocoa,
        f"{label}/Snowball":      snowball,
        f"{label}/ConformalCalib": calib_result,
    }
    return all_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri(DEFAULT_CONFIG.paths.mlflow_uri)
    mlflow.set_experiment(DEFAULT_CONFIG.mlflow_experiment)

    # Determine which experiments to run
    if args.experiment == "all":
        to_run = list(EXPERIMENTS.keys())
    else:
        if args.experiment not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment: {args.experiment!r}. "
                f"Choices: {list(EXPERIMENTS)} + 'all'"
            )
        to_run = [args.experiment]

    all_results: dict[str, dict] = {}

    for key in to_run:
        spec = EXPERIMENTS[key]
        label = spec["label"]
        kwargs = _build_kwargs(key, args.model, args.n_samples)

        print(f"\n{'='*64}")
        print(f"  Experiment: {label}")
        print(f"  Config:     {kwargs}")
        print(f"{'='*64}")

        if args.dry_run:
            logger.info("[DRY RUN] Skipping execution.")
            continue

        with mlflow.start_run(run_name=label):
            mlflow.log_params(kwargs)

            try:
                await spec["coro"](**kwargs)
            except Exception as exc:
                logger.error("Experiment %s failed: %s", label, exc, exc_info=True)
                mlflow.log_param("error", str(exc))
                continue

            # Evaluate
            metrics = evaluate_experiment(spec["result_dir"], label)
            all_results.update(metrics)

            # Log scalar metrics to MLflow
            for method, m in metrics.items():
                if isinstance(m, dict):
                    for metric_name, val in m.items():
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            safe_name = f"{method}/{metric_name}".replace("/", "_")
                            try:
                                mlflow.log_metric(safe_name, float(val))
                            except Exception:
                                pass

            logger.info("Results for %s:\n%s", label, json.dumps(metrics, indent=2))

    if args.dry_run:
        return

    # --- Combined results table ---
    if all_results:
        print("\n\n===  FINAL RESULTS TABLE  ===")
        # Flatten for the table: one row per (experiment, method) pair
        flat: dict[str, dict] = {}
        for full_key, m in all_results.items():
            if isinstance(m, dict) and "ece" in m:
                flat[full_key] = m
        if flat:
            print_results_table(flat)

        # Save to disk
        out_path = Path("results/all_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(all_results, fh, indent=2, default=str)
        logger.info("Combined results saved to %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrated Agency — Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS) + ["all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CONFIG.model.primary,
        help="HuggingFace model ID (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override number of samples (default: per-experiment config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment configs without running inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
