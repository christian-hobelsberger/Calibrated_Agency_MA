"""
InstrumentedVLLMModel — wraps vLLM to capture per-token log-probabilities.
StepRecord — immutable data container for one ReAct reasoning step.

Design notes:
- `generate()` / `__call__` is synchronous (vLLM generates synchronously).
- When n_samples > 1, the model makes TWO vLLM calls:
    1. Greedy (temperature=0.0) → captures last_logprobs / last_seq_logprob
    2. Stochastic (temperature=T, n=n_samples) → M alternative continuations
  This avoids contaminating the greedy logprob computation with sampling noise.
- Callers can then use `[1:]` to get exactly n_samples stochastic alternatives
  while index [0] is always the greedy output.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

# vLLM is imported lazily inside __init__ below. This is a heavy GPU dependency, and
# code paths that never instantiate InstrumentedVLLMModel shouldn't need it installed.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Full record of one agent reasoning step."""
    step_id:        int
    thought:        str
    action_name:    str
    action_args:    dict

    observation:    str = ""

    # Raw UQ signals
    logprobs:       list[float] = field(default_factory=list)
    seq_logprob:    float = 0.0  # sum of logprobs = raw MSP score (< 0)

    # Derived UQ scores (filled during / after step execution)
    consistency_scores: list[float] = field(default_factory=list)
    c_star_cocoa:       Optional[float] = None
    branch_consistency_raw: Optional[float] = None
    branch_consistency: Optional[float] = None
    escalated:          bool = False

    # Ground truth (filled by annotation protocol)
    is_correct: Optional[bool] = None


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class InstrumentedVLLMModel:
    """
    Drop-in replacement for smolagents-style model that uses vLLM directly.

    Captures per-token log-probabilities for MSP computation.

    Parameters
    ----------
    model_name : HuggingFace model ID (must be downloaded / accessible).
    dtype      : 'bfloat16' (recommended for Llama on A100).
    gpu_memory_utilization : fraction of GPU VRAM to allow vLLM to use.
    max_model_len : maximum context length in tokens.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.88,
        max_model_len: int = 8192,
        hf_token: str = "",
        tensor_parallel_size: int = 1,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. Run: pip install vllm"
            ) from exc

        self._SamplingParams = SamplingParams
        logger.info(
            "Loading vLLM model: %s (tensor_parallel_size=%d)",
            model_name, tensor_parallel_size,
        )

        # vLLM picks up HF_TOKEN from the environment for gated/private models.
        # Local model paths require no token.
        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)

        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=False,
        )
        self.model_name = model_name

        # Populated after each __call__
        self._last_logprobs: list[float] = []
        self._last_seq_logprob: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        n_samples: int = 1,
        stop_sequences: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Generate completions and return list of text outputs.

        Index [0] is always the greedy output (temperature=0.0).
        Indices [1 .. n_samples] are i.i.d. stochastic samples.

        After the call, `last_seq_logprob` and `last_logprobs` reflect the
        greedy output only.
        """
        prompt = self._format_messages(messages)
        stop = stop_sequences or []

        # --- Greedy pass (always run to capture logprobs) ---
        greedy_params = self._SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            stop=stop,
            logprobs=1,
            n=1,
        )
        greedy_result = self.llm.generate([prompt], greedy_params)[0]
        greedy_output = greedy_result.outputs[0]

        self._last_logprobs = [
            list(lp.values())[0].logprob
            for lp in (greedy_output.logprobs or [])
        ]
        self._last_seq_logprob = sum(self._last_logprobs)

        outputs: list[str] = [greedy_output.text]

        # --- Stochastic passes for consistency sampling ---
        if n_samples > 0:
            sample_params = self._SamplingParams(
                temperature=temperature if temperature > 0.0 else 0.7,
                top_p=0.9,
                max_tokens=max_tokens,
                stop=stop,
                n=n_samples,
            )
            sample_result = self.llm.generate([prompt], sample_params)[0]
            outputs += [o.text for o in sample_result.outputs]

        return outputs

    @property
    def last_seq_logprob(self) -> float:
        return self._last_seq_logprob

    @property
    def last_logprobs(self) -> list[float]:
        return self._last_logprobs

    # ------------------------------------------------------------------
    # Chat formatting
    # ------------------------------------------------------------------

    def _format_messages(self, messages: list[dict]) -> str:
        """Convert chat messages to Llama-3 prompt format."""
        parts = ["<|begin_of_text|>"]
        for m in messages:
            role = m["role"]
            content = m.get("content", "")
            parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                f"{content}<|eot_id|>"
            )
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)
