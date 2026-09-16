"""
Step-wise CoCoA — extends the CoCoA (Confidence-Consistency Aggregation)
framework from single-response QA to individual steps in an agent trajectory.

Reference: Vashurin et al. (2025) CoCoA [original, single-response version]
           Hobelsberger (2026) Calibrated Agency [this extension]

Mathematical formulation (per step t):
    u_belief(a*_t | s_t)  = -log p(a*_t | s_t)          [MSP uncertainty term]
    u_cons(a*_t | s_t)    = 1 - (1/M) Σ_m s(a*_t, a^m_t)  [dissimilarity]
    u_cocoa(a*_t | s_t)   = u_belief · u_cons             [product]
    C*(a_t)               = 1 - normalise(u_cocoa)        [confidence ∈ [0,1]]

Normalisation: quantile-clip at 98th percentile, then min-max to [0,1].
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from scipy.special import expit  # sigmoid function

if TYPE_CHECKING:
    from agent.instrumented_agent import InstrumentedVLLMModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLI-based semantic similarity (lazy-loaded, no CUDA required at import)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_nli_model(model_name: str, device: str = "cuda"):
    """Load the cross-encoder NLI model once and cache it."""
    from sentence_transformers import CrossEncoder
    logger.info("Loading NLI model: %s (device=%s)", model_name, device)
    return CrossEncoder(model_name, device=device, max_length=512)


def semantic_similarity_nli(
    s1: str,
    s2: str,
    model_name: str = "cross-encoder/nli-roberta-base",
    device: str = "cuda",
) -> float:
    """
    Bidirectional entailment probability as a similarity score in [0, 1].

    Higher = more semantically similar.
    Uses the same RoBERTa-large cross-encoder as the consulting project.
    """
    if not s1.strip() or not s2.strip():
        return 0.0
    nli = _load_nli_model(model_name, device)
    # Predict both directions to avoid directionality bias
    scores = nli.predict([[s1, s2], [s2, s1]])
    # Index 1 = entailment class for this model
    return float((scores[0][1] + scores[1][1]) / 2.0)


# ---------------------------------------------------------------------------
# StepwiseCoCoA
# ---------------------------------------------------------------------------

class StepwiseCoCoA:
    """
    Compute step-level CoCoA confidence score C*(a_t | s_t).

    Parameters
    ----------
    model       : InstrumentedVLLMModel providing generation + logprobs.
    M           : Number of stochastic alternatives to sample per step.
    temperature : Sampling temperature for consistency alternatives.
    nli_model   : Name of the cross-encoder NLI model for similarity scoring.
    device      : Device for the NLI model ('cuda' or 'cpu').
    similarity_fn : Optional override for the semantic similarity function.
    """

    def __init__(
        self,
        model: "InstrumentedVLLMModel",
        M: int = 5,
        temperature: float = 0.7,
        nli_model: str = "cross-encoder/nli-roberta-base",
        device: str = "cuda",
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self.model = model
        self.M = M
        self.temperature = temperature
        self._nli_model_name = nli_model
        self._device = device
        self.sim_fn: Callable[[str, str], float] = similarity_fn or (
            lambda s1, s2: semantic_similarity_nli(s1, s2, nli_model, device)
        )

        # Running buffer of raw u_cocoa scores (filled online; used for calibration)
        self._raw_scores: list[float] = []

        # Calibration parameters (fitted via fit_normalisation)
        self._q98:    float = 1.0
        self._u_min:  float = 0.0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Main scoring method
    # ------------------------------------------------------------------

    def score_step(
        self,
        context: list[dict],
        greedy_output: str,
        raw_seq_logprob: float,
    ) -> tuple[float, list[float]]:
        """
        Compute step-level CoCoA confidence.

        Parameters
        ----------
        context         : Full message history up to and including the current step.
        greedy_output   : The greedy-decoded action string (already generated).
        raw_seq_logprob : Sum of per-token log-probs of greedy_output (≤ 0).

        Returns
        -------
        c_star              : Step-level CoCoA confidence in [0, 1].
        consistency_scores  : Per-alternative similarity scores normalized to [0, 1] via sigmoid
                              (entailment probabilities), length M.
        """
        # --- Belief term: MSP uncertainty (positive; higher = more uncertain) ---
        u_belief = -raw_seq_logprob  # negate: log-prob ≤ 0 → u_belief ≥ 0

        # --- Generate M stochastic alternatives at the same context state ---
        # Call with n_samples=M; index [0] is greedy (already have it), [1..M] are stochastic
        all_outputs = self.model(
            context,
            temperature=self.temperature,
            max_tokens=256,
            n_samples=self.M,
        )
        # Skip index 0 (greedy re-run), use indices 1..M as alternatives
        alternatives = all_outputs[1:]

        # --- Consistency term ---
        # Get raw NLI logits and normalize to [0,1] via sigmoid
        sims_raw = [self.sim_fn(greedy_output, alt) for alt in alternatives]
        consistency_scores: list[float] = [float(expit(s)) for s in sims_raw]

        mean_similarity = float(np.mean(consistency_scores)) if consistency_scores else 0.5
        u_cons = 1.0 - mean_similarity   # high = semantically inconsistent

        # --- CoCoA product ---
        u_cocoa = u_belief * u_cons
        self._raw_scores.append(u_cocoa)

        # --- Convert to confidence ---
        c_star = self._to_confidence(u_cocoa)
        return c_star, consistency_scores

    # ------------------------------------------------------------------
    # Normalisation / calibration
    # ------------------------------------------------------------------

    def fit_normalisation(self, raw_scores: Optional[list[float]] = None) -> None:
        """
        Fit quantile-clip threshold and min-max parameters.

        Call ONCE on a held-out calibration split before scoring the test set.

        Parameters
        ----------
        raw_scores : External list of raw u_cocoa values. If None, uses the
                     internally collected buffer (self._raw_scores).
        """
        scores = raw_scores if raw_scores is not None else self._raw_scores
        if not scores:
            logger.warning("fit_normalisation called with empty score buffer.")
            return

        arr = np.array(scores, dtype=float)
        self._q98 = float(np.quantile(arr, 0.98))
        clipped = np.clip(arr, None, self._q98)
        self._u_min = float(clipped.min())
        self._fitted = True
        logger.info(
            "[StepwiseCoCoA] Calibrated normaliser: q98=%.4f, u_min=%.4f",
            self._q98, self._u_min,
        )

    def _to_confidence(self, u_raw: float) -> float:
        """
        Clip → min-max → invert to get confidence C* ∈ [0, 1].

        If not yet calibrated, uses an online approximation.
        """
        if not self._fitted:
            # Online approximation: soft sigmoid inversion
            normalised = u_raw / (abs(u_raw) + 1.0)
            return float(np.clip(1.0 - normalised, 0.0, 1.0))

        u_clipped = min(u_raw, self._q98)
        denom = self._q98 - self._u_min
        if denom < 1e-9:
            return 1.0
        normalised = (u_clipped - self._u_min) / denom
        return float(np.clip(1.0 - normalised, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Convenience: compute cumulative trajectory uncertainty
    # ------------------------------------------------------------------

    def trajectory_confidence(
        self,
        step_confidences: list[float],
        strategy: str = "min",
    ) -> float:
        """
        Aggregate step-level confidences into a trajectory-level score.

        Parameters
        ----------
        strategy : 'min'  : conservative (worst-case step determines trajectory risk);
                            reported in the thesis.
                   'mean' : average across steps; reported in the thesis.
                   'ema'  : exponential moving average, alpha=0.7 (early steps weighted
                            more). Defined in the Methods section but not used in any
                            final reported table; kept for completeness/future use.
        """
        if not step_confidences:
            return 1.0
        arr = np.array(step_confidences)
        if strategy == "min":
            return float(arr.min())
        if strategy == "mean":
            return float(arr.mean())
        if strategy == "ema":
            alpha = 0.7
            ema = arr[0]
            for v in arr[1:]:
                ema = alpha * ema + (1 - alpha) * v
            return float(ema)
        raise ValueError(f"Unknown strategy: {strategy!r}")
