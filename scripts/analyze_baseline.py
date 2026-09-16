"""
Analyze GSM8K baseline results (25-sample pilot or full 500-sample).

Diagnostic/dev tool for inspecting a single results directory. The "STEP-LEVEL RESULTS"
table uses heuristic step-level `is_correct` labels and is for diagnostics only. The
trajectory-level metrics (traj_min/traj_mean, printed via print_trajectory_comparison)
are the ones consistent with the thesis's reported methodology.

VCE-Answer is computed but is NOT one of the thesis's four reported estimators
(StepCoCoA, BranchConsistency, MSP-Answer, CoCoA-Answer). Kept working as an
excluded/ablation baseline only.

Usage:
    python analyze_baseline.py
    python analyze_baseline.py --results-dir results/gsm8k_pilot_interactive_25
    python analyze_baseline.py --results-dir results/gsm8k_70b

Generates:
    - Results table (AUROC, ECE, etc.)
    - Reliability diagrams
    - Selective accuracy curves
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Add project root to path and set root as working directory
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from uq.evaluate import (
    load_step_records,
    evaluate_step_uq,
    evaluate_trajectory_uq,
    print_results_table,
    print_trajectory_comparison,
)
from uq.baselines import (
    load_trajectories,
    load_trajectories_with_split,
    msp_answer_level,
    msp_answer_level_proper,
    vce_answer_level,
    cocoa_answer_level,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze GSM8K baseline results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/gsm8k_pilot_interactive_25",
        help="Path to results directory (default: 25-sample pilot)",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save plots to PNG")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # =========================================================================
    # Load data (intelligently detect calib/test split or legacy mode)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Loading trajectories from: {results_dir}")
    print(f"{'='*70}\n")

    try:
        calib_trajectories, test_trajectories, has_proper_split = load_trajectories_with_split(
            results_dir
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # --- Load step records for evaluation ---
    if has_proper_split:
        # Proper split: evaluate on test set only
        test_traj_path = results_dir / "test" / "trajectories.jsonl"
        steps = load_step_records(str(test_traj_path))
        trajectories = test_trajectories
        steps_source = "test (proper split)"
    else:
        # Legacy mode: use single trajectories
        single_traj_path = results_dir / "trajectories.jsonl"
        steps = load_step_records(str(single_traj_path))
        trajectories = test_trajectories
        steps_source = "single (legacy)"

    n_steps = len(steps)
    n_traj = len(trajectories)
    success_rate = np.mean([t["task_success"] for t in trajectories])

    print(f"✅ Loaded {n_steps} steps from {n_traj} trajectories ({steps_source})")
    print(f"✅ Task success rate: {success_rate:.1%} ({sum(t['task_success'] for t in trajectories)}/{n_traj})")

    # =========================================================================
    # Evaluate all methods
    # =========================================================================
    print(f"\n{'='*70}")
    print("Evaluating UQ methods...")
    if has_proper_split:
        print("(Using proper calib/test split for all metrics)")
    else:
        print("(Using legacy single-split mode)")
    print(f"{'='*70}\n")

    results = {}

    # Step-wise methods (proposed)
    try:
        step_metrics = evaluate_step_uq(steps, confidence_key="c_star_cocoa", trajectories=trajectories)
        traj_min = evaluate_trajectory_uq(trajectories, confidence_key="c_star_cocoa", aggregation="min")
        traj_mean = evaluate_trajectory_uq(trajectories, confidence_key="c_star_cocoa", aggregation="mean")
        results["StepCoCoA"] = {
            "step": step_metrics,
            "traj_min": traj_min,
            "traj_mean": traj_mean,
        }
        print("✅ StepCoCoA (step-level + trajectory-level min/mean)")
    except Exception as e:
        print(f"⚠️  StepCoCoA failed: {e}")

    try:
        step_metrics = evaluate_step_uq(steps, confidence_key="branch_consistency", trajectories=trajectories)
        traj_min = evaluate_trajectory_uq(trajectories, confidence_key="branch_consistency", aggregation="min")
        traj_mean = evaluate_trajectory_uq(trajectories, confidence_key="branch_consistency", aggregation="mean")
        results["BranchConsistency"] = {
            "step": step_metrics,
            "traj_min": traj_min,
            "traj_mean": traj_mean,
        }
        print("✅ BranchConsistency (step-level + trajectory-level min/mean)")
    except Exception as e:
        print(f"⚠️  BranchConsistency failed: {e}")

    # Answer-level baselines
    try:
        if has_proper_split:
            # Proper split: fit on calib, evaluate on test (NO DATA LEAKAGE)
            msp_data, msp_params = msp_answer_level_proper(calib_trajectories, test_trajectories)
            step_metrics = evaluate_step_uq(msp_data, confidence_key="c_msp_answer", trajectories=trajectories)
            # For answer-level: only single value per trajectory, so min = mean = answer
            traj_answer = evaluate_trajectory_uq(trajectories, confidence_key="c_msp_answer", aggregation="mean")
            results["MSP-Answer"] = {
                "step": step_metrics,
                "traj_answer": traj_answer,
            }
            print(f"✅ MSP-Answer (proper split: q98={msp_params['q98']:.4f}, "
                  f"u_min={msp_params['u_min']:.4f}, n_calib={msp_params['n_calib_samples']})")
        else:
            # Legacy mode: fit and evaluate on same data (with warning)
            msp_data = msp_answer_level(trajectories)
            step_metrics = evaluate_step_uq(msp_data, confidence_key="c_msp_answer", trajectories=trajectories)
            traj_answer = evaluate_trajectory_uq(trajectories, confidence_key="c_msp_answer", aggregation="mean")
            results["MSP-Answer"] = {
                "step": step_metrics,
                "traj_answer": traj_answer,
            }
            print("✅ MSP-Answer (legacy mode: fit and eval on same data)")
    except Exception as e:
        print(f"⚠️  MSP-Answer failed: {e}")

    try:
        vce_data = vce_answer_level(trajectories)
        if vce_data:
            step_metrics = evaluate_step_uq(vce_data, confidence_key="c_vce_answer", trajectories=trajectories)
            traj_answer = evaluate_trajectory_uq(trajectories, confidence_key="c_vce_answer", aggregation="mean")
            results["VCE-Answer"] = {
                "step": step_metrics,
                "traj_answer": traj_answer,
            }
            print("✅ VCE-Answer")
    except Exception as e:
        print(f"⚠️  VCE-Answer failed: {e}")

    try:
        # CoCoA-Answer: inherits calibration from final step's c_star_cocoa
        # (which uses proper split if available, or legacy otherwise)
        cocoa_data = cocoa_answer_level(trajectories)
        step_metrics = evaluate_step_uq(cocoa_data, confidence_key="c_cocoa_answer", trajectories=trajectories)
        traj_answer = evaluate_trajectory_uq(trajectories, confidence_key="c_cocoa_answer", aggregation="mean")
        results["CoCoA-Answer"] = {
            "step": step_metrics,
            "traj_answer": traj_answer,
        }
        if has_proper_split:
            print("✅ CoCoA-Answer (inherits calibration from step-level proper split)")
        else:
            print("✅ CoCoA-Answer (inherits calibration from step-level legacy mode)")
    except Exception as e:
        print(f"⚠️  CoCoA-Answer failed: {e}")

    # =========================================================================
    # Print results tables
    # =========================================================================
    print(f"\n{'='*100}")
    print("STEP-LEVEL RESULTS (Step-level AUROC & ECE)")
    print(f"{'='*100}")

    # Extract step-level metrics for backward-compatible printing
    step_level_results = {}
    for method, metrics in results.items():
        if isinstance(metrics, dict) and 'step' in metrics:
            step_level_results[method] = metrics['step']
        elif isinstance(metrics, dict):
            # Legacy support for flat dicts
            step_level_results[method] = metrics

    print_results_table(step_level_results)

    # Print trajectory-level comparison
    print_trajectory_comparison(results)

    # =========================================================================
    # Reliability diagrams
    # =========================================================================
    if args.save_plots and len(results) > 0:
        print(f"\n{'='*70}")
        print("Generating plots...")
        print(f"{'='*70}\n")

        # Select top 3 methods for reliability diagram
        methods_to_plot = []
        for label in ["StepCoCoA", "MSP-Answer", "CoCoA-Answer"]:
            if label == "StepCoCoA":
                methods_to_plot.append((label, steps, "c_star_cocoa"))
            elif label == "MSP-Answer":
                if has_proper_split:
                    msp_data, _ = msp_answer_level_proper(calib_trajectories, test_trajectories)
                else:
                    msp_data = msp_answer_level(trajectories)
                methods_to_plot.append((label, msp_data, "c_msp_answer"))
            elif label == "CoCoA-Answer":
                methods_to_plot.append((label, cocoa_answer_level(trajectories), "c_cocoa_answer"))

        if methods_to_plot:
            fig, axes = plt.subplots(1, len(methods_to_plot), figsize=(5 * len(methods_to_plot), 4))
            if len(methods_to_plot) == 1:
                axes = [axes]
            fig.suptitle(f"Reliability Diagrams: {results_dir.name}", fontsize=14)

            for ax, (label, data, key) in zip(axes, methods_to_plot):
                valid = [s for s in data if s.get(key) is not None and s.get("is_correct") is not None]
                if not valid:
                    ax.set_title(f"{label}\n(no data)")
                    continue

                confs = np.array([s[key] for s in valid])
                corrs = np.array([int(s["is_correct"]) for s in valid])

                prob_true, prob_pred = calibration_curve(corrs, confs, n_bins=10, strategy="uniform")
                ax.plot(prob_pred, prob_true, "o-", label=label, linewidth=2, markersize=8)
                ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect")

                ece = results.get(label, {}).get("ece", float("nan"))
                ax.set_title(f"{label}\nECE={ece:.3f}")
                ax.set_xlabel("Mean confidence")
                ax.set_ylabel("Fraction correct")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            plot_path = results_dir / "reliability_diagrams.png"
            plt.savefig(plot_path, bbox_inches="tight", dpi=150)
            print(f"✅ Saved: {plot_path}")
            plt.close()

        # Selective accuracy curve
        if "StepCoCoA" in results:
            valid = [s for s in steps if s.get("c_star_cocoa") is not None and s.get("is_correct") is not None]
            if valid:
                confs = np.array([s["c_star_cocoa"] for s in valid])
                corrs = np.array([int(s["is_correct"]) for s in valid])

                thresholds = np.arange(0.5, 1.0, 0.05)
                sel_accs = [corrs[confs > t].mean() if (confs > t).any() else np.nan for t in thresholds]
                coverages = [(confs > t).mean() for t in thresholds]

                fig, ax1 = plt.subplots(figsize=(8, 5))
                ax2 = ax1.twinx()

                ax1.plot(thresholds, sel_accs, "o-", color="steelblue", linewidth=2, markersize=8, label="Selective accuracy")
                ax2.plot(thresholds, coverages, "s--", color="orange", linewidth=2, markersize=8, label="Coverage")

                ax1.set_xlabel("Confidence threshold", fontsize=11)
                ax1.set_ylabel("Selective accuracy", color="steelblue", fontsize=11)
                ax2.set_ylabel("Coverage", color="orange", fontsize=11)
                ax1.set_title(f"Selective Prediction: StepCoCoA ({results_dir.name})", fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim([0, 1])
                ax2.set_ylim([0, 1])

                fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
                plt.tight_layout()

                plot_path = results_dir / "selective_accuracy.png"
                plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                print(f"✅ Saved: {plot_path}")
                plt.close()

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()