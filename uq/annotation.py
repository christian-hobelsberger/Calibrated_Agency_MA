"""
Intermediate-step annotation protocol.

Converts binary task-level success/failure labels into per-step correctness
labels using two complementary strategies:

1. `annotate_steps_by_task_success`: generic back-propagation heuristic
   applicable to AgentBench and ToolBench.

2. `annotate_gsm8k_steps`: GSM8K-specific, verifies each arithmetic
   calculation result independently via safe_eval, giving higher-quality
   step-level labels than the generic back-propagation approach.

3. `annotate_hotpotqa_steps`: HotpotQA-specific, labels search/lookup steps
   as correct when they retrieve a gold supporting-fact paragraph (identified
   by the [Title] prefix in observations), and labels finish steps by F1 ≥ 0.5.

Reference: Hobelsberger (2026) Calibrated Agency, Section 3.7
"""
from __future__ import annotations

import ast
import operator
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.instrumented_agent import StepRecord

logger = logging.getLogger(__name__)

_SAFE_OPS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.Pow:      operator.pow,
    ast.Mod:      operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval_arithmetic(expr: str) -> float | None:
    """Evaluate a pure-arithmetic expression. Returns None on any error."""
    try:
        def _eval(node):
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.BinOp):
                op = type(node.op)
                if op not in _SAFE_OPS:
                    raise ValueError(f"Unsupported op: {op}")
                return _SAFE_OPS[op](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -_eval(node.operand)
            raise ValueError(f"Unsupported node: {type(node).__name__}")
        tree = ast.parse(expr.strip(), mode="eval")
        return _eval(tree.body)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generic annotation (AgentBench, ToolBench)
# ---------------------------------------------------------------------------

def annotate_steps_by_task_success(
    trajectory: list["StepRecord"],
    task_success: bool,
) -> list["StepRecord"]:
    """
    Back-propagate a binary task-level success label to individual steps.

    Heuristic:
    - If the task succeeds, ALL steps on the trajectory are labelled correct.
    - If the task fails, steps up to (not including) the FIRST suspicious step
      are labelled correct; all subsequent steps are labelled incorrect.

    A step is 'suspicious' when:
    - Its observation contains 'ERROR', 'not found', or similar error signals, OR
    - Its action could not be parsed (action_name == "none").

    This is an approximation; higher-quality labels require counterfactual replay
    (see thesis Section 3.7 for a detailed discussion of the limitation).
    """
    if task_success:
        for step in trajectory:
            step.is_correct = True
        return trajectory

    # Find the first error step
    first_error_idx = len(trajectory)
    for i, step in enumerate(trajectory):
        obs = (step.observation or "").lower()
        action_missing = step.action_name in ("none", "", None)
        has_error_signal = (
            "error" in obs
            or "not found" in obs
            or "blocked" in obs
            or "exception" in obs
            or "traceback" in obs
        )
        if action_missing or has_error_signal:
            first_error_idx = i
            break

    for i, step in enumerate(trajectory):
        step.is_correct = i < first_error_idx

    logger.debug(
        "annotate_steps_by_task_success: first_error_idx=%d / %d steps",
        first_error_idx, len(trajectory),
    )
    return trajectory


# ---------------------------------------------------------------------------
# GSM8K-specific annotation
# ---------------------------------------------------------------------------

def annotate_gsm8k_steps(
    trajectory: list["StepRecord"],
    task_success: bool,
) -> list["StepRecord"]:
    """
    GSM8K-specific step-level annotation.

    For each step that invokes the calculator:
    - Re-evaluate the expression independently via safe_eval.
    - Compare the result to the tool's observed output.
    - Label the step as correct iff: (a) no ERROR in observation, AND
      (b) the re-evaluated result matches the observation (within 1e-3).

    For non-calculator steps (planning thoughts, final answer), fall back to
    `task_success` as a proxy for correctness.

    This provides substantially better step-level labels than the generic
    back-propagation heuristic for arithmetic trajectories.
    """
    for step in trajectory:
        obs = (step.observation or "").strip()

        # Error observation → incorrect regardless
        if "ERROR" in obs.upper():
            step.is_correct = False
            continue

        if step.action_name == "calculate":
            expression = step.action_args.get("expression", "")
            expected = _safe_eval_arithmetic(expression)
            try:
                actual = float(obs.replace(",", ""))
                step.is_correct = (
                    expected is not None and abs(expected - actual) < 1e-3
                )
            except ValueError:
                # Observation is not a number (e.g. final FINAL ANSWER line)
                step.is_correct = task_success
        else:
            # Planning / final-answer steps: proxy by task success
            step.is_correct = task_success

    return trajectory


# ---------------------------------------------------------------------------
# HotpotQA-specific annotation
# ---------------------------------------------------------------------------

_OBS_TITLE_RE = re.compile(r'\[([^\]]+)\]')


def annotate_hotpotqa_steps(
    trajectory: list["StepRecord"],
    task_success: bool,
    supporting_fact_titles: list[str],
    final_f1: float = 0.0,
) -> list["StepRecord"]:
    """
    HotpotQA-specific step-level annotation.

    Annotation strategy per action type:

    - finish()  : correct iff final F1 ≥ 0.5 (lenient to handle paraphrasing).
    - search()  : correct iff the retrieved paragraph's title (extracted from the
                  '[Title] …' prefix in the observation) matches any supporting
                  fact title via case-insensitive substring containment.
    - lookup()  : same title-matching logic applied to the current-paragraph title
                  embedded in the observation (always present as '[Title] …' or
                  '… in [Title].' in ContextSearcher output).
    - Thought / none : proxy by task_success (no verifiable grounding available).

    This yields higher-quality step labels than the generic back-propagation
    heuristic for retrieval-based tasks, because the supporting facts are
    explicitly annotated in HotpotQA.
    """
    sup_lower = [t.lower() for t in supporting_fact_titles]

    def _title_is_supporting(obs: str) -> bool:
        """Return True if any [Title] in obs matches a supporting-fact title."""
        if not sup_lower:
            return task_success
        for m in _OBS_TITLE_RE.finditer(obs):
            candidate = m.group(1).lower()
            if any(sf in candidate or candidate in sf for sf in sup_lower):
                return True
        return False

    for step in trajectory:
        if step.action_name == "finish":
            step.is_correct = final_f1 >= 0.5

        elif step.action_name in ("search", "lookup"):
            obs = step.observation or ""
            step.is_correct = _title_is_supporting(obs)

        else:
            # Thought steps or unparsed responses: use task success as proxy
            step.is_correct = task_success

    logger.debug(
        "annotate_hotpotqa_steps: %d correct / %d steps (task_success=%s, F1=%.3f)",
        sum(1 for s in trajectory if s.is_correct),
        len(trajectory),
        task_success,
        final_f1,
    )
    return trajectory
