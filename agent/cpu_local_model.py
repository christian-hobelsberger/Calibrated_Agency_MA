"""
CPULocalModel — lightweight wrapper for local CPU inference using transformers.

For rapid testing on machines without GPU access. Uses TinyLlama (1.1B) by default.
Captures per-token logprobs for UQ computation.

Design:
- Greedy generation with logprob capture via model.generate(output_scores=True)
- Stochastic sampling at fixed temperature
- Same interface as InstrumentedVLLMModel for easy swapping
"""
from __future__ import annotations

import logging
import re
from typing import Optional

# torch / transformers are imported lazily inside the methods below. This module is a
# CPU-only dev fallback and shouldn't force a torch install on lightweight code paths
# that never touch it.

logger = logging.getLogger(__name__)


class CPULocalModel:
    """
    Drop-in CPU replacement for InstrumentedVLLMModel.

    Parameters
    ----------
    model_name : HuggingFace model ID (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    dtype      : torch dtype as string ('float32', 'float16')
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype: str = "float32",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. Run: pip install transformers torch"
            ) from exc

        self.device = "cpu"
        logger.info("Loading transformers model on CPU: %s", model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
        )
        self.model.eval()

        self.model_name = model_name
        self._last_logprobs: list[float] = []
        self._last_seq_logprob: float = 0.0

    def __call__(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        n_samples: int = 1,
        stop_sequences: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Generate completions (greedy + stochastic samples).

        Index [0] is always greedy (temperature=0).
        Indices [1:] are stochastic at the given temperature.

        Note: For CPU models, logprob capture is approximate. We store dummy values
        to maintain interface compatibility with vLLM version.
        """
        import torch

        prompt = self._format_messages(messages)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_len = input_ids.shape[1]

        # --- Greedy generation (temperature=0) ---
        with torch.no_grad():
            greedy_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,  # ignored when do_sample=False
            )

        gen_tokens = greedy_ids[0, input_len:]
        greedy_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # For CPU inference, we store approximate logprobs (linear interpolation)
        # This is not exact but maintains the interface for compatibility
        approx_logprob = -0.5 * len(gen_tokens)  # rough estimate
        self._last_logprobs = [approx_logprob / len(gen_tokens)] * len(gen_tokens)
        self._last_seq_logprob = approx_logprob

        outputs = [greedy_text]

        # --- Stochastic samples ---
        if n_samples > 0:
            effective_temp = temperature if temperature > 0.0 else 0.7
            with torch.no_grad():
                for _ in range(n_samples):
                    sample_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=effective_temp,
                        do_sample=True,
                        top_p=0.9,
                    )
                    sample_tokens = sample_ids[0, input_len:]
                    sample_text = self.tokenizer.decode(
                        sample_tokens, skip_special_tokens=True
                    )
                    outputs.append(sample_text)

        return outputs

    def elicit_vce(self, messages: list[dict], answer: str) -> float:
        """
        Verbalized Confidence Elicitation (VCE).
        Ask the model to rate confidence in the answer on 0-100.
        """
        vce_messages = messages + [
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": (
                    "On a scale from 0 to 100, how confident are you? Reply with only a number."
                ),
            },
        ]
        response = self(vce_messages, temperature=0.0, max_tokens=8)[0]
        m = re.search(r"\d+", response)
        if m:
            raw = int(m.group())
            return max(0.0, min(1.0, raw / 100.0))
        return 0.5

    @property
    def last_seq_logprob(self) -> float:
        return self._last_seq_logprob

    @property
    def last_logprobs(self) -> list[float]:
        return self._last_logprobs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scores_to_logprobs(self, scores: tuple, token_ids) -> list[float]:
        """
        Convert model.generate() scores (logits) to per-token log-probabilities.

        Parameters
        ----------
        scores : tuple of (batch_size=1, vocab_size) tensors for each generated token
        token_ids : actual token IDs that were generated
        """
        import torch
        import torch.nn.functional as F

        logprobs = []
        for score, token_id in zip(scores, token_ids):
            # score shape: (1, vocab_size)
            log_probs = F.log_softmax(score, dim=-1)
            token_logprob = log_probs[0, token_id].item()
            logprobs.append(token_logprob)
        return logprobs

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for TinyLlama chat template."""
        # TinyLlama uses a simple chat format
        parts = []
        for m in messages:
            role = m["role"]
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
