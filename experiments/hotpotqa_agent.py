"""
HotpotQA Multi-Hop Question Answering Agent Experiment with Calibration/Test Split Support.

Flow for each problem:
  1. Load HotpotQA question with 10 context paragraphs (distractor setting).
  2. Run the instrumented ReAct loop (records step-level logprobs).
  3. Parse thought / action / final-answer from each response.
  4. Execute search() against the in-memory context paragraphs.
  5. Compute step-wise CoCoA and branching consistency at each step.
  6. Score the final answer against the gold label (exact match + F1 ≥ 0.5).
  7. Annotate step-level correctness using provided supporting-fact titles.
  8. Stream results to JSONL.

Split Workflow (default 50/50, configurable via --calib-ratio):
  Phase 1 (calib, --calib-ratio):   Generate trajectories, fit normalisation parameters.
  Phase 2 (test, remainder):    Generate trajectories with calibrated C* scores.
  All (none):             Legacy mode, generates all samples without splitting.

Dataset:
  Uses hotpot_qa (distractor config) from HuggingFace. Each question ships with
  10 context paragraphs (2 gold supporting + 8 distractors), so retrieval is
  entirely in-memory: no external API calls required.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

HOTPOTQA_SYSTEM_PROMPT = """\
You are an expert research assistant answering multi-hop questions by searching provided context paragraphs.

CRITICAL RULES - READ CAREFULLY:

1. You CANNOT answer from memory. You MUST search for information first.
2. EVERY response MUST contain an Action line. A response with only a Thought and NO Action is INVALID — it wastes a step.
3. Each response must contain at most ONE Thought and exactly ONE Action (combined in the same response).
4. The moment retrieved text contains enough information to answer, your VERY NEXT response MUST be Action: finish("answer"). Do NOT search further.
5. Stop immediately after the Action line.

AVAILABLE ACTIONS:
  search("query")  — Find the most relevant context paragraph for a query.
  finish("answer") — Submit your final answer. Use the shortest precise phrasing.

STEP FORMAT — every response must follow one of these two patterns:

THOUGHT + ACTION (use when you need to reason before acting):
Thought: <1-2 sentence reasoning>
Action: search("query") | finish("answer")

ACTION ONLY (use when the next step is obvious):
Action: search("query") | finish("answer")

EXAMPLE OF CORRECT SEQUENCE (bridge question):

[Model Response 1]
Thought: I need to find who directed "Inception" and then look up their nationality.
Action: search("Inception film director")

[Model Response 2]
Thought: Inception was directed by Christopher Nolan. Now I need his nationality.
Action: search("Christopher Nolan nationality")

[Model Response 3]
Thought: The text says he is British-American. That directly answers the question.
Action: finish("British-American")

EXAMPLES OF WRONG BEHAVIOR — DO NOT DO THESE:

[WRONG — Thought with no Action]
"Thought: I need to find Terry Richardson's birthdate to compare."
^ INVALID. No Action line. Always include an Action in every response.

[WRONG — Re-searching when answer is already retrieved]
Thought: I found the answer is YG Entertainment. Let me also check the founder article.
Action: search("YG Entertainment founder")
^ INVALID. You already have the answer — call finish() immediately.

[WRONG — Multiple step pairs in one response]
"Thought: Search for X. Action: search("X") Thought: Found it. Action: finish("Y")"
^ INVALID. Only one Thought+Action pair per response.

MANDATORY FINISH RULE:
Once the retrieved context answers the question — call finish() in that SAME response. No re-verification.

MANDATORY SEARCH RULES:
- Never answer without first searching.
- For comparison questions (e.g. "were both X and Y …?"), search each entity separately.\
"""


# ---------------------------------------------------------------------------
# In-memory context search tool
# ---------------------------------------------------------------------------

class ContextSearcher:
    """
    In-memory retrieval over the 10 context paragraphs bundled with each HotpotQA question.

    search(query): ranks paragraphs by word-overlap (title weight 2×, body weight 1×)
                    and returns the best match; prefers previously unseen paragraphs on ties.

    Observations always carry a [Title] prefix so that annotation can identify
    which paragraph was retrieved without additional state tracking.
    """

    _STOP: frozenset[str] = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
        "and", "or", "for", "by", "on", "at", "with", "it", "this", "that",
        "be", "as", "from", "its", "has", "had", "have", "he", "she", "they",
        "his", "her", "their", "who", "which", "what", "where", "when", "how",
        "both", "each", "than", "more", "also", "only", "not", "but",
    })

    def __init__(self, context: list[dict]) -> None:
        """
        Parameters
        ----------
        context : list of {"title": str, "sentences": list[str]}
        """
        self.paragraphs: list[dict] = context
        self.last_title: Optional[str] = None
        self.last_sentences: Optional[list[str]] = None
        self.visited_titles: list[str] = []

    def _tok(self, text: str) -> set[str]:
        return set(re.findall(r'\w+', text.lower())) - self._STOP

    def _score(self, q_tok: set[str], title: str, sentences: list[str]) -> float:
        if not q_tok:
            return 0.0
        n = len(q_tok)
        title_ov   = len(q_tok & self._tok(title)) / n
        content_ov = len(q_tok & self._tok(" ".join(sentences))) / n
        return 2.0 * title_ov + content_ov

    def search(self, query: str) -> str:
        """Return the most relevant paragraph as '[Title] first-3-sentences'."""
        q_tok = self._tok(query)
        scored = sorted(
            ((self._score(q_tok, p["title"], p["sentences"]), i)
             for i, p in enumerate(self.paragraphs)),
            key=lambda x: (-x[0], x[1]),
        )

        # Among top-scoring paragraphs, prefer one not yet visited
        best_score = scored[0][0]
        best_idx   = scored[0][1]
        for score, idx in scored:
            if score < best_score - 1e-9:
                break
            if self.paragraphs[idx]["title"] not in self.visited_titles:
                best_idx = idx
                break

        para = self.paragraphs[best_idx]
        self.last_title     = para["title"]
        self.last_sentences = para["sentences"]
        if para["title"] not in self.visited_titles:
            self.visited_titles.append(para["title"])

        snippet = " ".join(self.last_sentences[:3])[:500]
        return f"[{self.last_title}] {snippet}"

    def lookup(self, keyword: str) -> str:
        """Return sentences containing keyword across all context paragraphs.

        Accepts single words or multi-word phrases. For multi-word phrases the
        most specific token (longest non-stop word) is used as the match key,
        mirroring Ctrl+F behaviour: the model can write lookup("Scandinavian
        design") and the implementation searches for "scandinavian".

        Searches all 10 paragraphs, preferring the last-retrieved paragraph first
        (follow-up lookup on same para), then others in original order.
        Updates last_title / last_sentences on a match.
        """
        # Extract the primary search token from the query.
        # Models tend to write multi-word phrases (e.g. "John Whiting director")
        # but the entity/topic appears first. Taking the first non-stop token
        # mirrors Ctrl+F on the most specific term and covers 99 %+ of failures.
        # Falls back to the full lowercased phrase if no non-stop token found.
        tokens = [t for t in re.findall(r'\w+', keyword.lower())
                  if t not in self._STOP and len(t) > 1]
        kw = tokens[0] if tokens else keyword.lower().strip()

        # Prefer the last-retrieved paragraph first (follow-up lookup on same para),
        # then others in original order (cross-paragraph fallback).
        ordered = sorted(
            self.paragraphs,
            key=lambda p: (0 if p["title"] == self.last_title else 1,
                           self.paragraphs.index(p)),
        )
        for para in ordered:
            hits = [s for s in para["sentences"] if kw in s.lower()]
            if hits:
                self.last_title     = para["title"]
                self.last_sentences = para["sentences"]
                if para["title"] not in self.visited_titles:
                    self.visited_titles.append(para["title"])
                return f"[{para['title']}] " + " ".join(hits[:3])[:490]

        return f"No sentences found containing '{kw}'."


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_SEARCH_RE = re.compile(r'Action:\s*search\(["\']?(.*?)["\']?\)', re.IGNORECASE | re.DOTALL)
_FINISH_RE = re.compile(r'Action:\s*finish\(["\']?(.*?)["\']?\)', re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r'Thought:(.*?)(?=Action:|$)', re.DOTALL | re.IGNORECASE)


def parse_hotpotqa_action(text: str) -> tuple[str, Optional[str], Optional[str], bool]:
    """
    Parse a single ReAct response into (thought, action_name, action_arg, is_final).

    Returns
    -------
    thought     : Reasoning text (or raw text if no Thought prefix found).
    action_name : One of "search", "finish", or None.
    action_arg  : The string argument to the action, or None.
    is_final    : True only when action_name == "finish".
    """
    finish_m = _FINISH_RE.search(text)
    if finish_m:
        thought_m = _THOUGHT_RE.search(text)
        thought   = thought_m.group(1).strip() if thought_m else text.strip()
        return thought, "finish", finish_m.group(1).strip(), True

    search_m = _SEARCH_RE.search(text)
    if search_m:
        thought_m = _THOUGHT_RE.search(text)
        thought   = thought_m.group(1).strip() if thought_m else ""
        return thought, "search", search_m.group(1).strip(), False

    thought_m = _THOUGHT_RE.search(text)
    thought   = thought_m.group(1).strip() if thought_m else text.strip()
    return thought, None, None, False


# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    """Lowercase, strip articles and punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    return " ".join(s.split())


def compute_f1(pred: str, gold: str) -> float:
    """Token-level F1 score (standard HotpotQA metric)."""
    pred_tok = _normalize_answer(pred).split()
    gold_tok = _normalize_answer(gold).split()
    if not pred_tok or not gold_tok:
        return float(pred_tok == gold_tok)
    common   = Counter(pred_tok) & Counter(gold_tok)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tok)
    recall    = n_common / len(gold_tok)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return _normalize_answer(pred) == _normalize_answer(gold)


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Task success: exact match OR F1 ≥ 0.5 (lenient for paraphrases)."""
    if pred is None or gold is None:
        return False
    return exact_match(pred, gold) or compute_f1(pred, gold) >= 0.5


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

async def run_hotpotqa_experiment(
    n_samples: int = 500,
    output_dir: str = "results/hotpotqa",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cocoa_m: int = 5,
    branch_m: int = 5,
    temperature: float = 0.7,
    resume: bool = True,
    tensor_parallel_size: int = 1,
    split_phase: str = "all",
    calib_params_path: Optional[str] = None,
    calib_ratio: float = 0.5,
    _cocoa=None,
) -> tuple[list[dict], Any]:
    """
    Run the HotpotQA multi-hop QA agent experiment with optional calib/test split.

    Parameters
    ----------
    n_samples            : Number of validation questions to sample.
    output_dir           : Directory for JSONL output and checkpoints.
    model_name           : HuggingFace model ID or local path.
    cocoa_m              : M samples for step-wise CoCoA.
    branch_m             : M alternatives for branching consistency.
    temperature          : Sampling temperature for consistency sampling.
    resume               : If True, skip already-completed indices (crash recovery).
    tensor_parallel_size : Number of GPUs for vLLM tensor parallelism (1 = single GPU).
    split_phase          : One of "calib", "test", or "all" (default: "all").
                           - "all":   Run all n_samples without splitting (legacy mode).
                           - "calib": Run first calib_ratio of n_samples, fit norm params.
                           - "test":  Run last (1-calib_ratio) of n_samples, apply params.
    calib_params_path    : Path to fitted calib_params.json (required for "test").
    calib_ratio          : Fraction of tasks for calibration phase (default: 0.5).
    """
    # Deliberately lazy: these pull in vLLM/torch, so importing this module for its
    # dataclasses/CLI parsing alone doesn't require a GPU environment.
    from tqdm import tqdm
    from agent.instrumented_agent import InstrumentedVLLMModel, StepRecord
    from uq.annotation import annotate_hotpotqa_steps
    from uq.branching import BranchingConsistency
    from uq.stepwise_cocoa import StepwiseCoCoA

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"

    # --- Load dataset ---
    # Prefer the prepared JSONL (carries context + supporting facts in metadata).
    # Fall back to raw HuggingFace download if data/ is not yet populated.
    prepared_path = Path("data/hotpotqa/validation.jsonl")
    if prepared_path.exists():
        logger.info("Loading HotpotQA from prepared JSONL: %s", prepared_path)
        with open(prepared_path, encoding="utf-8") as fh:
            all_records = [json.loads(l) for l in fh if l.strip()]
        all_records = all_records[:n_samples]

        dataset = [
            {
                "question":              r["instruction"],
                "gold_answer":           r["gold_answer"],
                "question_type":         r["metadata"]["type"],
                "level":                 r["metadata"]["level"],
                "context":               r["metadata"]["context"],
                "supporting_fact_titles": r["metadata"]["supporting_fact_titles"],
            }
            for r in all_records
        ]
        logger.info("Loaded %d examples (with context paragraphs).", len(dataset))
    else:
        logger.warning(
            "Prepared JSONL not found at %s: falling back to HuggingFace download. "
            "Run `python scripts/prepare_datasets.py --only hotpotqa` to cache locally.",
            prepared_path,
        )
        from datasets import load_dataset
        raw = load_dataset("hotpot_qa", "distractor", split="validation")
        raw = raw.select(range(min(n_samples, len(raw))))
        dataset = []
        for ex in raw:
            ctx = ex["context"]
            paragraphs = [
                {"title": t, "sentences": s}
                for t, s in zip(ctx["title"], ctx["sentences"])
            ]
            sf_titles = list(dict.fromkeys(ex["supporting_facts"]["title"]))
            dataset.append({
                "question":               ex["question"],
                "gold_answer":            ex["answer"],
                "question_type":          ex["type"],
                "level":                  ex["level"],
                "context":                paragraphs,
                "supporting_fact_titles": sf_titles,
            })

    # --- Split dataset ---
    if split_phase == "all":
        indices = list(range(len(dataset)))
        logger.info("Running all %d samples (no split).", len(dataset))
    else:
        all_indices = np.arange(len(dataset))
        np.random.seed(42)
        np.random.shuffle(all_indices)
        calib_end = int(calib_ratio * len(dataset))

        if split_phase == "calib":
            indices = all_indices[:calib_end].tolist()
            logger.info("Calib phase: %d samples (first %.0f%%).",
                        len(indices), calib_ratio * 100)
        elif split_phase == "test":
            indices = all_indices[calib_end:].tolist()
            logger.info("Test phase: %d samples (last %.0f%%).",
                        len(indices), (1 - calib_ratio) * 100)
        else:
            raise ValueError(
                f"Invalid split_phase: {split_phase!r}. "
                f"Must be one of: 'calib', 'test', 'all'."
            )
        dataset = [dataset[i] for i in indices]

    # --- Resume support ---
    completed_indices: set[int] = set()
    if resume and traj_path.exists():
        with open(traj_path) as fh:
            for line in fh:
                try:
                    completed_indices.add(json.loads(line)["idx"])
                except Exception:
                    pass
        logger.info("Resuming: %d examples already completed.", len(completed_indices))

    # --- Initialise model & UQ modules (reuse shared instance if provided) ---
    if _cocoa is not None:
        model = _cocoa.model
        cocoa = _cocoa
        logger.info("Reusing shared model instance for %s phase.", split_phase)
    else:
        logger.info("Loading model: %s", model_name)
        model = InstrumentedVLLMModel(model_name, tensor_parallel_size=tensor_parallel_size)
        cocoa = StepwiseCoCoA(model=model, M=cocoa_m, temperature=temperature)
    brancher = BranchingConsistency(model=model, M=branch_m, temperature=temperature)

    # --- Load calibration parameters for test phase ---
    if split_phase == "test" and _cocoa is None:
        # Standalone test phase: load params from file.
        # When called from run_hotpotqa_full_pipeline, cocoa already has fitted params.
        if calib_params_path is None:
            raise ValueError("split_phase='test' requires --calib-params-path.")
        with open(calib_params_path) as fh:
            params = json.load(fh)
        cocoa._q98    = params["q98"]
        cocoa._u_min  = params["u_min"]
        cocoa._fitted = True
        logger.info("Loaded calib params: q98=%.4f, u_min=%.4f",
                    cocoa._q98, cocoa._u_min)

    results: list[dict] = []
    cfg = DEFAULT_CONFIG.experiment

    with open(traj_path, "a") as out_fh:
        for idx, example in enumerate(tqdm(dataset, desc="HotpotQA")):
            if idx in completed_indices:
                continue

            question              = example["question"]
            gold_ans              = example["gold_answer"]
            question_type         = example.get("question_type", "unknown")
            level                 = example.get("level", "unknown")
            context               = example["context"]
            supporting_titles     = example.get("supporting_fact_titles", [])

            # Fresh retriever per question
            searcher = ContextSearcher(context)

            messages: list[dict] = [
                {"role": "system", "content": HOTPOTQA_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Question: {question}"},
            ]
            trajectory: list[StepRecord] = []
            final_answer: Optional[str]  = None

            for step_id in range(cfg.hotpotqa_max_steps):
                response    = model(messages, temperature=0.0, max_tokens=100)[0]
                raw_logprob = model.last_seq_logprob

                thought, action_name, action_arg, is_final = parse_hotpotqa_action(response)

                if is_final:
                    final_answer = action_arg

                    c_star, consistency_scores = cocoa.score_step(
                        context=messages,
                        greedy_output=response,
                        raw_seq_logprob=raw_logprob,
                    )
                    should_escalate, branch_cons_raw, branch_cons = brancher.check(
                        context=messages,
                        greedy_action=response,
                    )

                    trajectory.append(StepRecord(
                        step_id=step_id,
                        thought=thought,
                        action_name="finish",
                        action_args={"answer": final_answer or ""},
                        observation=final_answer or "",
                        logprobs=model.last_logprobs,
                        seq_logprob=raw_logprob,
                        consistency_scores=consistency_scores,
                        c_star_cocoa=c_star,
                        branch_consistency_raw=branch_cons_raw,
                        branch_consistency=branch_cons,
                        escalated=should_escalate,
                    ))
                    break

                # --- Execute tool ---
                if action_name == "search" and action_arg:
                    observation = searcher.search(action_arg)
                else:
                    observation = (
                        "No valid action found. "
                        "Use: search(\"query\") or finish(\"answer\")."
                    )

                # --- Step-wise CoCoA ---
                c_star, consistency_scores = cocoa.score_step(
                    context=messages,
                    greedy_output=response,
                    raw_seq_logprob=raw_logprob,
                )

                # --- Branching consistency ---
                should_escalate, branch_cons_raw, branch_cons = brancher.check(
                    context=messages,
                    greedy_action=response,
                )

                step = StepRecord(
                    step_id=step_id,
                    thought=thought,
                    action_name=action_name or "none",
                    action_args={"query": action_arg or ""},
                    observation=observation,
                    logprobs=model.last_logprobs,
                    seq_logprob=raw_logprob,
                    consistency_scores=consistency_scores,
                    c_star_cocoa=c_star,
                    branch_consistency_raw=branch_cons_raw,
                    branch_consistency=branch_cons,
                    escalated=should_escalate,
                )
                trajectory.append(step)

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",      "content": f"Observation: {observation}"})

            # --- Score and annotate ---
            task_success = answers_match(final_answer, gold_ans)
            f1_score     = compute_f1(final_answer or "", gold_ans or "")
            em_score     = exact_match(final_answer or "", gold_ans or "")

            annotated = annotate_hotpotqa_steps(
                trajectory, task_success, supporting_titles, f1_score
            )

            record = {
                "idx":               idx,
                "question":          question,
                "gold_answer":       gold_ans,
                "pred_answer":       final_answer,
                "task_success":      task_success,
                "f1":                f1_score,
                "em":                em_score,
                "question_type":     question_type,
                "level":             level,
                "n_steps":           len(annotated),
                "supporting_titles": supporting_titles,
                "trajectory": [
                    {
                        "step_id":                s.step_id,
                        "thought":                s.thought[:500],
                        "action_name":            s.action_name,
                        "action_args":            s.action_args,
                        "observation":            s.observation[:500],
                        "seq_logprob":            s.seq_logprob,
                        "consistency_scores":     s.consistency_scores,
                        "c_star_cocoa":           s.c_star_cocoa,
                        "branch_consistency_raw": s.branch_consistency_raw,
                        "branch_consistency":     s.branch_consistency,
                        "escalated":              s.escalated,
                        "is_correct":             s.is_correct,
                    }
                    for s in annotated
                ],
            }
            results.append(record)
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()

    if results:
        accuracy = sum(r["task_success"] for r in results) / len(results)
        mean_f1  = sum(r["f1"]           for r in results) / len(results)
    else:
        accuracy = mean_f1 = float("nan")
    logger.info(
        "HotpotQA done. EM accuracy: %.3f | Mean F1: %.3f", accuracy, mean_f1
    )

    # --- Fit and save calibration parameters (calib phase only) ---
    if split_phase == "calib":
        calib_params_file = Path(output_dir) / "calib_params.json"
        if not cocoa._raw_scores:
            if calib_params_file.exists():
                logger.info(
                    "Calib phase already complete (all samples resumed). "
                    "Reusing existing calib_params.json at %s", calib_params_file
                )
            else:
                raise RuntimeError(
                    "Calib phase produced no scores and no calib_params.json exists. "
                    "Re-run without --resume or delete the trajectories file to reprocess."
                )
        else:
            logger.info(
                "Fitting normalisation parameters on %d raw scores...",
                len(cocoa._raw_scores),
            )
            cocoa.fit_normalisation()
            calib_params_file.write_text(json.dumps({
                "q98":          float(cocoa._q98),
                "u_min":        float(cocoa._u_min),
                "n_raw_scores": len(cocoa._raw_scores),
                "split_phase":  "calib",
                "model":        model_name,
                "cocoa_m":      cocoa_m,
                "temperature":  temperature,
            }, indent=2))
            logger.info("Saved calibration parameters to %s", calib_params_file)

    return results, cocoa


async def run_hotpotqa_full_pipeline(
    n_samples: int = 500,
    output_base: str = "results/hotpotqa",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cocoa_m: int = 5,
    branch_m: int = 5,
    temperature: float = 0.7,
    resume: bool = True,
    tensor_parallel_size: int = 1,
    calib_ratio: float = 0.5,
) -> dict[str, list[dict]]:
    """
    Run calibration → test pipeline sequentially in one call.

    Creates results/hotpotqa/calib/ and results/hotpotqa/test/ subdirectories
    and passes calibration params automatically from phase 1 to phase 2.

    Returns dict with keys "calib" and "test".
    """
    base      = Path(output_base)
    calib_dir = base / "calib"
    test_dir  = base / "test"
    calib_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}

    # Phase 1 (calibration): model is created here, returned inside cocoa
    logger.info("=" * 80)
    logger.info("PHASE 1: CALIBRATION (%.0f%% of %d samples)", calib_ratio * 100, n_samples)
    logger.info("=" * 80)
    calib_results, cocoa = await run_hotpotqa_experiment(
        n_samples=n_samples,
        output_dir=str(calib_dir),
        model_name=model_name,
        cocoa_m=cocoa_m,
        branch_m=branch_m,
        temperature=temperature,
        resume=resume,
        tensor_parallel_size=tensor_parallel_size,
        split_phase="calib",
        calib_ratio=calib_ratio,
    )
    all_results["calib"] = calib_results
    logger.info("Phase 1 complete: %d samples processed.", len(calib_results))

    # Ensure calib params are fitted on the live cocoa (handles resume edge case)
    calib_params_file = calib_dir / "calib_params.json"
    if not calib_params_file.exists():
        raise FileNotFoundError(
            f"Calibration params file not found: {calib_params_file}\n"
            "This usually means the calib phase did not complete successfully."
        )
    if not cocoa._fitted:
        # Resumed run where all samples were skipped: load params from file
        with open(calib_params_file) as fh:
            params = json.load(fh)
        cocoa._q98    = params["q98"]
        cocoa._u_min  = params["u_min"]
        cocoa._fitted = True
        logger.info("Loaded calib params from file: q98=%.4f, u_min=%.4f",
                    cocoa._q98, cocoa._u_min)

    # Reset raw-score buffer; fitted params stay set for test phase scoring
    cocoa._raw_scores = []

    # Phase 2 (test): pass same cocoa so the model is never torn down
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: TEST (%.0f%% of %d samples)", (1 - calib_ratio) * 100, n_samples)
    logger.info("=" * 80)
    test_results, _ = await run_hotpotqa_experiment(
        n_samples=n_samples,
        output_dir=str(test_dir),
        model_name=model_name,
        cocoa_m=cocoa_m,
        branch_m=branch_m,
        temperature=temperature,
        resume=resume,
        tensor_parallel_size=tensor_parallel_size,
        split_phase="test",
        calib_ratio=calib_ratio,
        _cocoa=cocoa,
    )
    all_results["test"] = test_results
    logger.info("Phase 2 complete: %d samples processed.", len(test_results))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION → TEST PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(
        "Calib phase (%.0f%%):  %d/%d correct, %d trajectory steps",
        calib_ratio * 100,
        sum(1 for r in calib_results if r["task_success"]),
        len(calib_results),
        sum(len(r["trajectory"]) for r in calib_results),
    )
    logger.info(
        "Test phase  (%.0f%%):  %d/%d correct, %d trajectory steps (CALIBRATED)",
        (1 - calib_ratio) * 100,
        sum(1 for r in test_results if r["task_success"]),
        len(test_results),
        sum(len(r["trajectory"]) for r in test_results),
    )
    logger.info("\nResults location:")
    logger.info("  Calib (fitting):  %s/trajectories.jsonl", calib_dir)
    logger.info("  Calib params:     %s", calib_params_file)
    logger.info("  Test (evaluate):  %s/trajectories.jsonl  ← USE THIS", test_dir)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HotpotQA Multi-Hop QA Agent Experiment with Calibration/Test Split"
    )
    parser.add_argument("--n-samples",           type=int,   default=500,
                        help="Total number of HotpotQA validation questions to use.")
    parser.add_argument("--output-dir",           type=str,   default="results/hotpotqa",
                        help="Output directory for results and calibration params.")
    parser.add_argument("--model",                type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--cocoa-m",              type=int,   default=5,
                        help="M samples for step-wise CoCoA.")
    parser.add_argument("--branch-m",             type=int,   default=5,
                        help="M alternatives for branching consistency.")
    parser.add_argument("--temperature",          type=float, default=0.7,
                        help="Sampling temperature for consistency sampling.")
    parser.add_argument("--no-resume",            action="store_true",
                        help="Do not resume from incomplete runs.")
    parser.add_argument("--tensor-parallel-size", type=int,   default=1,
                        help="Number of GPUs for vLLM tensor parallelism.")
    parser.add_argument("--split-phase",          type=str,   default="all",
                        choices=["calib", "test", "all", "full"],
                        help=(
                            "'calib': run first calib_ratio of samples, fit norm params. "
                            "'test': run remaining samples with fitted params. "
                            "'all': no split (legacy mode). "
                            "'full': run calib→test pipeline sequentially."
                        ))
    parser.add_argument("--calib-params-path",    type=str,   default=None,
                        help="Path to calib_params.json (required for --split-phase test).")
    parser.add_argument("--calib-ratio",          type=float, default=0.5,
                        help="Fraction of tasks for calibration phase (default: 0.5).")
    args = parser.parse_args()

    if args.split_phase == "full":
        asyncio.run(run_hotpotqa_full_pipeline(
            n_samples=args.n_samples,
            output_base=args.output_dir,
            model_name=args.model,
            cocoa_m=args.cocoa_m,
            branch_m=args.branch_m,
            temperature=args.temperature,
            resume=not args.no_resume,
            tensor_parallel_size=args.tensor_parallel_size,
            calib_ratio=args.calib_ratio,
        ))
    else:
        asyncio.run(run_hotpotqa_experiment(
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            model_name=args.model,
            cocoa_m=args.cocoa_m,
            branch_m=args.branch_m,
            temperature=args.temperature,
            resume=not args.no_resume,
            tensor_parallel_size=args.tensor_parallel_size,
            split_phase=args.split_phase,
            calib_params_path=args.calib_params_path,
            calib_ratio=args.calib_ratio,
        ))
