"""
Branching Consistency — forks the agent context at each pre-tool-call
decision point and generates M lightweight next-action candidates to detect
high-uncertainty steps without re-running full trajectory alternatives.

Computational complexity: O(M) additional LLM calls per step (vs. O(M·T)
for full trajectory re-sampling).

Reference: Hobelsberger (2026) Calibrated Agency, Section 3.5
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from scipy.special import expit  # sigmoid function

from uq.stepwise_cocoa import semantic_similarity_nli

if TYPE_CHECKING:
    from agent.instrumented_agent import InstrumentedVLLMModel

logger = logging.getLogger(__name__)


class BranchingConsistency:
    """
    At each pre-tool-call step, sample M alternative next-action strings
    and measure their semantic consistency with the greedy action.

        consistency = mean_m  s(greedy_action, alternative_m)

    High consistency → agent is certain → proceed (auto).
    Low consistency  → agent is uncertain → escalate.

    Parameters
    ----------
    model         : InstrumentedVLLMModel.
    M             : Number of alternative next actions to sample.
    temperature   : Sampling temperature for alternatives.
    nli_model     : NLI cross-encoder model for semantic similarity.
    device        : GPU/CPU device for the NLI model.
    threshold     : Default escalation threshold (can be overridden per call).
    similarity_fn : Optional custom similarity function.
    """

    def __init__(
        self,
        model: "InstrumentedVLLMModel",
        M: int = 5,
        temperature: float = 0.7,
        nli_model: str = "cross-encoder/nli-roberta-base",
        device: str = "cuda",
        threshold: float = 0.6,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self.model = model
        self.M = M
        self.temperature = temperature
        self.threshold = threshold
        self.sim_fn: Callable[[str, str], float] = similarity_fn or (
            lambda s1, s2: semantic_similarity_nli(s1, s2, nli_model, device)
        )
        self._trigger_log: list[dict] = []

    def check(
        self,
        context: list[dict],
        greedy_action: str,
        threshold: Optional[float] = None,
    ) -> tuple[bool, float, float]:
        """
        Fork the agent context, sample M next-action candidates, and check
        their consistency with the greedy action.

        Parameters
        ----------
        context       : Full message history at the current decision point.
        greedy_action : The greedy-decoded action string (already generated).
        threshold     : Escalation threshold (overrides self.threshold if given).

        Returns
        -------
        should_escalate     : True if consistency < threshold.
        consistency_raw     : Raw NLI logits (unbounded ℝ), stored for diagnostics.
        consistency         : Mean semantic similarity of M alternatives to greedy_action,
                              normalized to [0,1] via sigmoid (entailment probability).
                              Used for decision-making and escalation policy.
        """
        thr = threshold if threshold is not None else self.threshold

        # Generate M short alternatives (next action only, max_tokens=128)
        all_outputs = self.model(
            context,
            temperature=self.temperature,
            max_tokens=128,
            n_samples=self.M,
        )
        alternatives = all_outputs[1:]  # skip index 0 (greedy re-run)

        if not alternatives:
            logger.warning("BranchingConsistency: no alternatives generated.")
            return False, 0.0, 1.0

        # Get raw NLI logits (unbounded)
        sims_raw = [self.sim_fn(greedy_action, alt) for alt in alternatives]
        mean_raw = float(np.mean(sims_raw))

        # Normalize to [0,1] via sigmoid: converts logits to entailment probabilities
        consistency = float(expit(mean_raw))
        should_escalate = consistency < thr

        entry = {
            "consistency_raw":  mean_raw,      # Raw logit for diagnostics
            "consistency":      consistency,   # Normalized [0,1] for decision-making
            "should_escalate":  should_escalate,
            "threshold":        thr,
            "n_alternatives":   len(alternatives),
        }
        self._trigger_log.append(entry)

        if should_escalate:
            logger.debug(
                "ESCALATE triggered: consistency=%.3f < threshold=%.3f",
                consistency, thr,
            )

        return should_escalate, mean_raw, consistency

    @property
    def escalation_rate(self) -> float:
        """Fraction of steps that triggered an escalation."""
        if not self._trigger_log:
            return 0.0
        return sum(e["should_escalate"] for e in self._trigger_log) / len(self._trigger_log)

    @property
    def trigger_log(self) -> list[dict]:
        return self._trigger_log

    def reset_log(self) -> None:
        self._trigger_log.clear()
