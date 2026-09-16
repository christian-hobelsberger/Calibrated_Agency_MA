# Lazy imports to avoid loading torch/sentence-transformers at import time.
from .annotation import annotate_gsm8k_steps, annotate_steps_by_task_success
from .evaluate import (
    evaluate_step_uq,
    evaluate_trajectory_uq,
    evaluate_snowball_detection,
    load_step_records,
)

__all__ = [
    "StepwiseCoCoA",
    "BranchingConsistency",
    "annotate_gsm8k_steps",
    "annotate_steps_by_task_success",
    "evaluate_step_uq",
    "evaluate_trajectory_uq",
    "evaluate_snowball_detection",
    "load_step_records",
]


def __getattr__(name):
    if name == "StepwiseCoCoA":
        from .stepwise_cocoa import StepwiseCoCoA
        return StepwiseCoCoA
    if name == "BranchingConsistency":
        from .branching import BranchingConsistency
        return BranchingConsistency
    raise AttributeError(f"module 'uq' has no attribute {name!r}")
