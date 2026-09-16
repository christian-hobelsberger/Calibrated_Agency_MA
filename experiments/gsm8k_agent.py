"""
GSM8K Tool-Augmented Agent Experiment with Calibration/Test Split Support.

Flow for each problem:
  1. Load problem from GSM8K test split.
  2. Run the instrumented ReAct loop (records step-level logprobs).
  3. Parse thought / action / final-answer from each response.
  4. Execute calculator tool (MCP or direct fallback).
  5. Compute step-wise CoCoA and branching consistency at each step.
  6. Score the final answer against the gold label.
  7. Annotate step-level correctness (GSM8K-specific arithmetic verifier).
  8. Stream results to JSONL.

Split Workflow (default 50/50, configurable via --calib-ratio):
  Phase 1 (calib, --calib-ratio):   Generate trajectories, fit normalisation parameters
  Phase 2 (test, remainder):    Generate trajectories with calibrated C* scores
  All (none):             Legacy mode: generate all samples without splitting
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

import numpy as np

from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

GSM8K_SYSTEM_PROMPT = """\
You are a precise mathematical reasoning assistant solving word problems step by step.

CRITICAL RULES - READ CAREFULLY:

1. You CANNOT perform arithmetic in your head. EVER.
2. ALL numeric calculations MUST use the calculate() tool.
3. Each response must be ONLY ONE of these:
   - A single Thought (1-2 sentences max, no calculations)
   - A single Action (just the tool call, no extra text)
   - FINAL ANSWER (when you have the solution)
4. Do NOT output multiple steps in one response.
5. Do NOT mix Thought and Action in the same response.
6. Stop immediately after outputting one step.

STEP FORMAT (pick ONE per response):

THOUGHT FORMAT:
Thought: <1-2 sentence reasoning about what to calculate>

ACTION FORMAT:
Action: calculate("<mathematical_expression>")

FINAL FORMAT:
FINAL ANSWER: <number>

EXAMPLE OF CORRECT MULTI-STEP SEQUENCE:

[Model Response 1]
Thought: I need to calculate hat cost first.

[Model Response 2]
Action: calculate("25")

[Model Response 3]
Thought: Now jacket costs 3 times the hat.

[Model Response 4]
Action: calculate("25 * 3")

[Model Response 5]
Thought: Total is hat + jacket + pants.

[Model Response 6]
Action: calculate("25 + 75 + 50")

[Model Response 7]
FINAL ANSWER: 150

EXAMPLE OF WRONG BEHAVIOR (DO NOT DO THIS):
"Thought: I need to find the hat cost. Action: calculate("25") Observation: 25 
Thought: Now the jacket. Action: calculate("25 * 3")..."
^ This is WRONG - multiple steps in one response

MANDATORY TOOL USE:
- Never compute math in your head
- Examples of REQUIRED tool use:
  * "What is 2+2?" → Use calculate, don't say "4"
  * "Half of 100?" → Use calculate("100 / 2"), don't say "50"
  * Any percentage, multiplication, division, addition → ALWAYS calculate
- It is not optional. You must use calculate for every number operation.\
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the numeric answer from a model response or GSM8K gold label."""
    m = re.search(r"FINAL ANSWER:\s*([\d,.\-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip()
    # GSM8K gold format uses '#### <number>'
    m = re.search(r"####\s*([\d,.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last number in text
    nums = re.findall(r"[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Normalised numeric comparison (tolerance 1e-3)."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except ValueError:
        return pred.strip() == gold.strip()


def parse_react_output(text: str) -> tuple[str, Optional[str], Optional[str], bool]:
    """
    Parse a single ReAct response into (thought, action_name, expr, is_final).
    """
    if re.search(r"FINAL ANSWER:", text, re.IGNORECASE):
        return text, None, None, True

    thought_m = re.search(
        r"Thought:(.*?)(?:Action:|$)", text, re.MULTILINE | re.IGNORECASE
    )
    action_m = re.search(
        r'Action:\s*calculate\(["\']?(.*?)["\']?\)', text, re.IGNORECASE
    )

    thought = thought_m.group(1).strip() if thought_m else text.strip()
    expr    = action_m.group(1).strip() if action_m else None
    action  = "calculate" if expr else None

    return thought, action, expr, False


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

async def _run_phase(
    dataset: list[dict],
    phase_name: str,
    phase_indices: list[int],
    output_dir: str,
    model_name: str,
    cocoa_m: int,
    branch_m: int,
    temperature: float,
    use_mcp: bool,
    resume: bool,
    tensor_parallel_size: int,
    cocoa: Optional["StepwiseCoCoA"] = None,
) -> tuple[list[dict], "StepwiseCoCoA", "BranchingConsistency"]:
    """Run a single phase of the experiment and return results + UQ modules."""
    # Deliberately lazy: these pull in vLLM/torch/HF datasets, so importing this module
    # for its dataclasses/CLI parsing alone doesn't require a GPU environment.
    from datasets import load_dataset
    from tqdm import tqdm
    from agent.instrumented_agent import InstrumentedVLLMModel, StepRecord
    from agent.mcp_client import DirectCalculator
    from uq.annotation import annotate_gsm8k_steps
    from uq.branching import BranchingConsistency
    from uq.stepwise_cocoa import StepwiseCoCoA

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"

    # Filter dataset to phase indices
    phase_dataset = [dataset[i] for i in phase_indices]
    logger.info("%s phase: %d samples", phase_name.upper(), len(phase_dataset))

    # Resume support
    completed_indices: set[int] = set()
    if resume and traj_path.exists():
        with open(traj_path) as fh:
            for line in fh:
                try:
                    completed_indices.add(json.loads(line)["idx"])
                except Exception:
                    pass
        logger.info("Resuming: %d examples already completed.", len(completed_indices))

    # Initialize model and UQ modules (or reuse from previous phase)
    if cocoa is None:
        logger.info("Loading model: %s", model_name)
        model = InstrumentedVLLMModel(model_name, tensor_parallel_size=tensor_parallel_size)
        cocoa = StepwiseCoCoA(model=model, M=cocoa_m, temperature=temperature)
    else:
        model = cocoa.model

    brancher = BranchingConsistency(model=model, M=branch_m, temperature=temperature)
    calc = DirectCalculator()

    results: list[dict] = []
    cfg = DEFAULT_CONFIG.experiment

    async with AsyncExitStack() as stack:
        mcp_provider = None
        if use_mcp:
            from agent.mcp_client import MCPToolProvider
            mcp_provider = await stack.enter_async_context(
                MCPToolProvider("tools/calculator_server.py", "calculator")
            )

        with open(traj_path, "a") as out_fh:
            for idx, example in enumerate(tqdm(phase_dataset, desc=f"GSM8K-{phase_name}")):
                if idx in completed_indices:
                    continue

                question   = example["question"]
                gold_ans   = extract_gsm8k_answer(example["answer"])
                difficulty = example.get("difficulty")

                messages: list[dict] = [
                    {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Problem: {question}"},
                ]
                trajectory: list[StepRecord] = []
                final_answer: Optional[str] = None

                for step_id in range(cfg.gsm8k_max_steps):
                    response = model(messages, temperature=0.0, max_tokens=100)[0]
                    raw_logprob = model.last_seq_logprob

                    thought, action_name, expr, is_final = parse_react_output(response)

                    if is_final:
                        final_answer = extract_gsm8k_answer(response)
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
                            action_name="final_answer",
                            action_args={},
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

                    if expr:
                        if use_mcp and mcp_provider:
                            observation = await mcp_provider.call("calculate", expression=expr)
                        else:
                            observation = calc.calculate(expr)
                    else:
                        observation = "No calculation requested."

                    c_star, consistency_scores = cocoa.score_step(
                        context=messages,
                        greedy_output=response,
                        raw_seq_logprob=raw_logprob,
                    )
                    should_escalate, branch_cons_raw, branch_cons = brancher.check(
                        context=messages,
                        greedy_action=response,
                    )

                    step = StepRecord(
                        step_id=step_id,
                        thought=thought,
                        action_name=action_name or "none",
                        action_args={"expression": expr or ""},
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
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {observation}",
                    })

                task_success = answers_match(final_answer, gold_ans)
                annotated = annotate_gsm8k_steps(trajectory, task_success)

                record = {
                    "idx":          idx,
                    "question":     question,
                    "gold_answer":  gold_ans,
                    "pred_answer":  final_answer,
                    "task_success": task_success,
                    "difficulty":   difficulty,
                    "n_steps":      len(annotated),
                    "trajectory": [
                        {
                            "step_id":            s.step_id,
                            "thought":            s.thought[:500],
                            "action_name":        s.action_name,
                            "action_args":        s.action_args,
                            "observation":        s.observation[:500],
                            "seq_logprob":        s.seq_logprob,
                            "consistency_scores": s.consistency_scores,
                            "c_star_cocoa":       s.c_star_cocoa,
                            "branch_consistency_raw": s.branch_consistency_raw,
                            "branch_consistency": s.branch_consistency,
                            "escalated":          s.escalated,
                            "is_correct":         s.is_correct,
                        }
                        for s in annotated
                    ],
                }
                results.append(record)
                out_fh.write(json.dumps(record) + "\n")
                out_fh.flush()

    accuracy = (
        sum(r["task_success"] for r in results) / len(results)
        if results else float("nan")
    )
    logger.info("GSM8K-%s done. Task accuracy: %.3f", phase_name.upper(), accuracy)
    return results, cocoa, brancher


async def run_gsm8k_experiment(
    n_samples: int = 500,
    output_dir: str = "results/gsm8k",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cocoa_m: int = 5,
    branch_m: int = 5,
    temperature: float = 0.7,
    use_mcp: bool = False,
    resume: bool = True,
    tensor_parallel_size: int = 1,
    split_phase: str = "all",
    calib_params_path: Optional[str] = None,
    calib_ratio: float = 0.5,
    _shared_model=None,
) -> list[dict]:
    """
    Run the GSM8K tool-augmented agent experiment with optional train/calib/test split.

    Parameters
    ----------
    n_samples            : Total number of GSM8K test problems to sample from.
    output_dir           : Directory for JSONL output and checkpoints.
    model_name           : HuggingFace model ID or local path.
    cocoa_m              : M samples for step-wise CoCoA.
    branch_m             : M alternatives for branching consistency.
    temperature          : Sampling temperature for consistency sampling.
    use_mcp              : If True, use the MCP calculator server subprocess.
                           If False, use the DirectCalculator (faster, no subprocess).
    resume               : If True, skip already-completed indices (for crash recovery).
    tensor_parallel_size : Number of GPUs for vLLM tensor parallelism (1=single GPU).
    split_phase          : One of "calib", "test", or "all" (default: "all").
                           - "all": Run all n_samples without splitting (legacy mode).
                           - "calib": Run first calib_ratio of n_samples, fit normalisation params.
                           - "test": Run last (1-calib_ratio) of n_samples, use fitted params.
    calib_params_path    : Path to JSON file with fitted calibration params.
                           Required for split_phase="test".
    calib_ratio          : Fraction of tasks used for calibration phase (default: 0.5).
    """
    # Deliberately lazy: see _run_phase() above.
    from datasets import load_dataset
    from tqdm import tqdm
    from agent.instrumented_agent import InstrumentedVLLMModel, StepRecord
    from agent.mcp_client import DirectCalculator
    from uq.annotation import annotate_gsm8k_steps
    from uq.branching import BranchingConsistency
    from uq.stepwise_cocoa import StepwiseCoCoA

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"

    # --- Load dataset ---
    # Prefer the prepared JSONL (carries difficulty + unified schema).
    # Fall back to raw HuggingFace download if data/ is not yet populated.
    prepared_path = Path("data/gsm8k/test.jsonl")
    if prepared_path.exists():
        logger.info("Loading GSM8K from prepared JSONL: %s", prepared_path)
        with open(prepared_path, encoding="utf-8") as fh:
            all_records = [json.loads(l) for l in fh if l.strip()]
        all_records = all_records[:n_samples]

        # Normalise to a list of dicts with the same keys used below
        dataset = [
            {
                "question":   r["instruction"],
                "answer":     r["metadata"]["raw_answer"],
                "difficulty": r["metadata"]["difficulty"],
                "n_steps":    r["metadata"]["n_steps"],
            }
            for r in all_records
        ]
        logger.info("Loaded %d examples (with difficulty labels).", len(dataset))
    else:
        logger.warning(
            "Prepared JSONL not found at %s: falling back to HuggingFace. "
            "Run `python scripts/prepare_datasets.py --only gsm8k` to add "
            "difficulty labels.",
            prepared_path,
        )
        raw = load_dataset("gsm8k", "main", split="test")
        raw = raw.select(range(min(n_samples, len(raw))))
        dataset = [
            {"question": ex["question"], "answer": ex["answer"],
             "difficulty": None, "n_steps": None}
            for ex in raw
        ]

    # --- Split dataset based on split_phase ---
    if split_phase == "all":
        indices = list(range(len(dataset)))
        logger.info("Running all %d samples (no splitting)", len(dataset))
    else:
        # Use fixed seed for reproducibility
        all_indices = np.arange(len(dataset))
        np.random.seed(42)
        np.random.shuffle(all_indices)

        calib_end = int(calib_ratio * len(dataset))

        if split_phase == "calib":
            indices = all_indices[:calib_end]
            logger.info("Calib phase: running %d samples (0-%.0f%%)", len(indices), calib_ratio * 100)
        elif split_phase == "test":
            indices = all_indices[calib_end:]
            logger.info("Test phase: running %d samples (%.0f-100%%)", len(indices), calib_ratio * 100)
        else:
            raise ValueError(f"Invalid split_phase: {split_phase!r}. "
                           f"Must be one of: 'calib', 'test', 'all'")

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

    # --- Initialise model (reuse shared instance if provided by full pipeline) ---
    if _shared_model is not None:
        model = _shared_model
        logger.info("Reusing shared model instance for %s phase.", split_phase)
    else:
        logger.info("Loading model: %s", model_name)
        model = InstrumentedVLLMModel(model_name, tensor_parallel_size=tensor_parallel_size)

    # --- Set up UQ modules ---
    cocoa    = StepwiseCoCoA(model=model, M=cocoa_m, temperature=temperature)
    brancher = BranchingConsistency(model=model, M=branch_m, temperature=temperature)

    # --- Load calibration parameters if test phase ---
    if split_phase == "test":
        if calib_params_path is None:
            raise ValueError("split_phase='test' requires --calib-params-path")
        with open(calib_params_path) as f:
            params = json.load(f)
        cocoa._q98 = params["q98"]
        cocoa._u_min = params["u_min"]
        cocoa._fitted = True
        logger.info("Loaded calibration parameters: q98=%.4f, u_min=%.4f",
                   cocoa._q98, cocoa._u_min)

    # --- Calculator (MCP or direct) ---
    calc = DirectCalculator()

    results: list[dict] = []

    async with AsyncExitStack() as stack:
        # MCP provider is optional: only entered when use_mcp=True
        mcp_provider = None
        if use_mcp:
            from agent.mcp_client import MCPToolProvider
            mcp_provider = await stack.enter_async_context(
                MCPToolProvider("tools/calculator_server.py", "calculator")
            )

        with open(traj_path, "a") as out_fh:
            for idx, example in enumerate(tqdm(dataset, desc="GSM8K")):
                if idx in completed_indices:
                    continue

                question   = example["question"]
                gold_ans   = extract_gsm8k_answer(example["answer"])
                difficulty = example.get("difficulty")   # None if HuggingFace fallback

                messages: list[dict] = [
                    {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Problem: {question}"},
                ]
                trajectory: list[StepRecord] = []
                final_answer: Optional[str] = None

                cfg = DEFAULT_CONFIG.experiment

                for step_id in range(cfg.gsm8k_max_steps):
                    # Greedy generation + logprob capture
                    response = model(messages, temperature=0.0, max_tokens=100)[0] # Reduced max tokens to enforce tool use and prevent long-winded responses 
                    raw_logprob = model.last_seq_logprob

                    thought, action_name, expr, is_final = parse_react_output(response)

                    if is_final:
                        final_answer = extract_gsm8k_answer(response)

                        # Apply UQ scoring to final answer (sample M alternatives)
                        c_star, consistency_scores = cocoa.score_step(
                            context=messages,
                            greedy_output=response,
                            raw_seq_logprob=raw_logprob,
                        )

                        should_escalate, branch_cons_raw, branch_cons = brancher.check(
                            context=messages,
                            greedy_action=response,
                        )

                        # Record the final-answer step with UQ annotations
                        trajectory.append(StepRecord(
                            step_id=step_id,
                            thought=thought,
                            action_name="final_answer",
                            action_args={},
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

                    # Execute calculator
                    if expr:
                        if use_mcp and mcp_provider:
                            observation = await mcp_provider.call("calculate", expression=expr)
                        else:
                            observation = calc.calculate(expr)
                    else:
                        observation = "No calculation requested."

                    # Step-wise CoCoA (sync: no await needed)
                    c_star, consistency_scores = cocoa.score_step(
                        context=messages,
                        greedy_output=response,
                        raw_seq_logprob=raw_logprob,
                    )

                    # Branching consistency (sync: no await needed)
                    should_escalate, branch_cons_raw, branch_cons = brancher.check(
                        context=messages,
                        greedy_action=response,
                    )

                    step = StepRecord(
                        step_id=step_id,
                        thought=thought,
                        action_name=action_name or "none",
                        action_args={"expression": expr or ""},
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

                    # Update conversation with observation
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {observation}",
                    })

                # --- Score and annotate ---
                task_success = answers_match(final_answer, gold_ans)
                annotated = annotate_gsm8k_steps(trajectory, task_success)

                record = {
                    "idx":          idx,
                    "question":     question,
                    "gold_answer":  gold_ans,
                    "pred_answer":  final_answer,
                    "task_success": task_success,
                    "difficulty":   difficulty,
                    "n_steps":      len(annotated),
                    "trajectory": [
                        {
                            "step_id":            s.step_id,
                            "thought":            s.thought[:500],
                            "action_name":        s.action_name,
                            "action_args":        s.action_args,
                            "observation":        s.observation[:500],
                            "seq_logprob":        s.seq_logprob,
                            "consistency_scores": s.consistency_scores,
                            "c_star_cocoa":       s.c_star_cocoa,
                            "branch_consistency_raw": s.branch_consistency_raw,
                            "branch_consistency": s.branch_consistency,
                            "escalated":          s.escalated,
                            "is_correct":         s.is_correct,
                        }
                        for s in annotated
                    ],
                }
                results.append(record)

                out_fh.write(json.dumps(record) + "\n")
                out_fh.flush()

    accuracy = (
        sum(r["task_success"] for r in results) / len(results)
        if results else float("nan")
    )
    logger.info("GSM8K done. Task accuracy: %.3f", accuracy)

    # --- Fit and save calibration parameters (calib phase only) ---
    if split_phase == "calib":
        calib_params_file = Path(output_dir) / "calib_params.json"
        if not cocoa._raw_scores:
            # All samples were skipped (resume): reuse existing params if present.
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
            logger.info("Fitting normalisation parameters on %d raw scores...",
                       len(cocoa._raw_scores))
            cocoa.fit_normalisation()

            calib_params = {
                "q98": float(cocoa._q98),
                "u_min": float(cocoa._u_min),
                "n_raw_scores": len(cocoa._raw_scores),
                "split_phase": "calib",
                "model": model_name,
                "cocoa_m": cocoa_m,
                "temperature": temperature,
            }
            calib_params_file.write_text(json.dumps(calib_params, indent=2))
            logger.info("Saved calibration parameters to %s", calib_params_file)

    return results


async def run_gsm8k_full_pipeline(
    n_samples: int = 500,
    output_base: str = "results/gsm8k",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cocoa_m: int = 5,
    branch_m: int = 5,
    temperature: float = 0.7,
    use_mcp: bool = False,
    resume: bool = True,
    tensor_parallel_size: int = 1,
    calib_ratio: float = 0.5,
) -> dict[str, list[dict]]:
    """
    Run calibration→test pipeline in a single process without reinitialising vLLM.

    Uses _run_phase so the model is created once for calib and the same instance
    (carried inside the returned cocoa object) is passed directly into test.
    This avoids the CUDA worker hang that occurs when a second LLM engine is
    initialised in the same process.
    """
    base = Path(output_base)
    calib_dir = base / "calib"
    test_dir  = base / "test"
    calib_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset once, shared across both phases ---
    prepared_path = Path("data/gsm8k/test.jsonl")
    if prepared_path.exists():
        logger.info("Loading GSM8K from prepared JSONL: %s", prepared_path)
        with open(prepared_path, encoding="utf-8") as fh:
            all_records = [json.loads(l) for l in fh if l.strip()]
        all_records = all_records[:n_samples]
        dataset = [
            {
                "question":   r["instruction"],
                "answer":     r["metadata"]["raw_answer"],
                "difficulty": r["metadata"]["difficulty"],
                "n_steps":    r["metadata"]["n_steps"],
            }
            for r in all_records
        ]
    else:
        from datasets import load_dataset as hf_load
        raw = hf_load("gsm8k", "main", split="test")
        raw = raw.select(range(min(n_samples, len(raw))))
        dataset = [
            {"question": ex["question"], "answer": ex["answer"],
             "difficulty": None, "n_steps": None}
            for ex in raw
        ]

    # --- Deterministic split (same seed as run_gsm8k_experiment) ---
    all_idx = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(all_idx)
    calib_end     = int(calib_ratio * len(dataset))
    calib_indices = all_idx[:calib_end].tolist()
    test_indices  = all_idx[calib_end:].tolist()

    # ------------------------------------------------------------------ #
    # Phase 1: Calibration                                                #
    # _run_phase creates the model here and returns it inside cocoa.      #
    # ------------------------------------------------------------------ #
    logger.info("=" * 80)
    logger.info("PHASE 1: CALIBRATION (%.0f%% of %d samples)", calib_ratio * 100, n_samples)
    logger.info("=" * 80)
    calib_results, cocoa, _ = await _run_phase(
        dataset=dataset,
        phase_name="calib",
        phase_indices=calib_indices,
        output_dir=str(calib_dir),
        model_name=model_name,
        cocoa_m=cocoa_m,
        branch_m=branch_m,
        temperature=temperature,
        use_mcp=use_mcp,
        resume=resume,
        tensor_parallel_size=tensor_parallel_size,
        cocoa=None,
    )
    logger.info("Phase 1 complete: %d samples processed", len(calib_results))

    # --- Fit normalisation (or reload if all samples were resumed) ---
    calib_params_file = calib_dir / "calib_params.json"
    if not cocoa._raw_scores:
        if calib_params_file.exists():
            logger.info("Calib already complete: loading existing calib_params.json")
            with open(calib_params_file) as fh:
                params = json.load(fh)
            cocoa._q98    = params["q98"]
            cocoa._u_min  = params["u_min"]
            cocoa._fitted = True
        else:
            raise RuntimeError(
                "Calib phase produced no scores and no calib_params.json found. "
                "Delete the calib trajectories and re-run to regenerate."
            )
    else:
        logger.info("Fitting normalisation on %d raw scores...", len(cocoa._raw_scores))
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
        logger.info("Saved calib params to %s", calib_params_file)

    # Reset raw-score buffer; fitted params (_q98, _u_min, _fitted) stay set.
    cocoa._raw_scores = []

    # ------------------------------------------------------------------ #
    # Phase 2: Test                                                        #
    # Pass the same cocoa → _run_phase reuses cocoa.model, no re-init.   #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: TEST (%.0f%% of %d samples)", (1 - calib_ratio) * 100, n_samples)
    logger.info("=" * 80)
    test_results, _, _ = await _run_phase(
        dataset=dataset,
        phase_name="test",
        phase_indices=test_indices,
        output_dir=str(test_dir),
        model_name=model_name,
        cocoa_m=cocoa_m,
        branch_m=branch_m,
        temperature=temperature,
        use_mcp=use_mcp,
        resume=resume,
        tensor_parallel_size=tensor_parallel_size,
        cocoa=cocoa,
    )
    logger.info("Phase 2 complete: %d samples processed", len(test_results))

    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION → TEST PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info("  Calib (used for fitting):  %s/trajectories.jsonl", calib_dir)
    logger.info("  Calib params:              %s", calib_params_file)
    logger.info("  Test (for evaluation):     %s/trajectories.jsonl ← USE THIS", test_dir)

    return {"calib": calib_results, "test": test_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K Agent Experiment with Train/Calib/Test Split")
    parser.add_argument("--n-samples",   type=int, default=500,
                        help="Total number of GSM8K samples to use")
    parser.add_argument("--output-dir",  type=str, default="results/gsm8k",
                        help="Output directory for results and calibration params")
    parser.add_argument("--model",       type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--cocoa-m",     type=int, default=5,
                        help="M samples for step-wise CoCoA")
    parser.add_argument("--branch-m",    type=int, default=5,
                        help="M alternatives for branching consistency")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for consistency")
    parser.add_argument("--use-mcp",              action="store_true",
                        help="Use MCP calculator server instead of direct calculator")
    parser.add_argument("--no-resume",           action="store_true",
                        help="Don't resume from incomplete runs")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for vLLM tensor parallelism")
    parser.add_argument("--split-phase", type=str, default="all",
                        choices=["calib", "test", "all", "full"],
                        help="Which phase to run: 'calib'/'test' (individual), 'all' (no split, legacy), or 'full' (run calib→test sequentially)")
    parser.add_argument("--calib-params-path", type=str, default=None,
                        help="Path to JSON file with calibration parameters (required for --split-phase test)")
    parser.add_argument("--calib-ratio", type=float, default=0.5,
                        help="Fraction of tasks used for calibration phase (default: 0.5)")
    args = parser.parse_args()

    if args.split_phase == "full":
        # Run all three phases sequentially
        asyncio.run(run_gsm8k_full_pipeline(
            n_samples=args.n_samples,
            output_base=args.output_dir,
            model_name=args.model,
            cocoa_m=args.cocoa_m,
            branch_m=args.branch_m,
            temperature=args.temperature,
            use_mcp=args.use_mcp,
            resume=not args.no_resume,
            tensor_parallel_size=args.tensor_parallel_size,
            calib_ratio=args.calib_ratio,
        ))
    else:
        # Run a single phase
        asyncio.run(run_gsm8k_experiment(
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            model_name=args.model,
            cocoa_m=args.cocoa_m,
            branch_m=args.branch_m,
            temperature=args.temperature,
            use_mcp=args.use_mcp,
            resume=not args.no_resume,
            tensor_parallel_size=args.tensor_parallel_size,
            split_phase=args.split_phase,
            calib_params_path=args.calib_params_path,
            calib_ratio=args.calib_ratio,
        ))
