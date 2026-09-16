#!/usr/bin/env python
"""
Local CPU test of GSM8K pipeline.

Runs 5 samples with TinyLlama locally to validate the full UQ pipeline
before running on LRZ.

Usage:
    python test_local_gsm8k.py
"""
import asyncio
import json
import logging
import re
from pathlib import Path

from config import DEFAULT_CONFIG
from agent.cpu_local_model import CPULocalModel
from agent.mcp_client import DirectCalculator
from uq.stepwise_cocoa import StepwiseCoCoA
from uq.branching import BranchingConsistency
from uq.annotation import annotate_gsm8k_steps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# System prompt for GSM8K
GSM8K_SYSTEM_PROMPT = """\
You are a precise mathematical reasoning assistant.
Solve the given word problem step by step.

RULES:
- For EVERY arithmetic calculation — no matter how simple — you MUST call the
  calculate tool. Do NOT compute in your head.
- Format each step EXACTLY as:
    Thought: <your reasoning about what to compute next>
    Action: calculate("<expression>")
- After the final calculation, emit:
    FINAL ANSWER: <number>

EXAMPLE:
Thought: I need to find 80% of 10.
Action: calculate("(80/100) * 10")
Observation: 8
Thought: Now I add this to the base amount of 10.
Action: calculate("10 + 8")
Observation: 18
FINAL ANSWER: 18\
"""


def extract_gsm8k_answer(text: str):
    """Extract numeric answer from text."""
    m = re.search(r"FINAL ANSWER:\s*([\d,.\-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def answers_match(pred, gold):
    """Normalize and compare numeric answers."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except ValueError:
        return pred.strip() == gold.strip()


def parse_react_output(text: str):
    """Parse ReAct response into (thought, action_name, expr, is_final)."""
    if re.search(r"FINAL ANSWER:", text, re.IGNORECASE):
        return text, None, None, True

    thought_m = re.search(r"Thought:(.*?)(?:Action:|$)", text, re.DOTALL | re.IGNORECASE)
    action_m = re.search(
        r'Action:\s*calculate\(["\']?(.*?)["\']?\)', text, re.DOTALL | re.IGNORECASE
    )

    thought = thought_m.group(1).strip() if thought_m else text.strip()
    expr = action_m.group(1).strip() if action_m else None
    action = "calculate" if expr else None

    return thought, action, expr, False


async def test_gsm8k_local():
    """Run 5-sample GSM8K test with local model."""
    from datasets import load_dataset
    from tqdm import tqdm

    # Load dataset
    logger.info("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(5))  # Only 5 samples
    dataset = [
        {"question": ex["question"], "answer": ex["answer"]}
        for ex in dataset
    ]
    logger.info("Loaded %d examples.", len(dataset))

    # Initialize model
    logger.info("Loading TinyLlama model on CPU...")
    model = CPULocalModel()

    # UQ modules (simplified: reduced M for speed)
    cocoa = StepwiseCoCoA(model=model, M=2, temperature=0.7)
    brancher = BranchingConsistency(model=model, M=2, temperature=0.7)

    # Calculator
    calc = DirectCalculator()

    # Output
    out_dir = Path("results/gsm8k_local_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"

    results = []
    try:
        for idx, example in enumerate(tqdm(dataset, desc="GSM8K Local Test")):
            question = example["question"]
            gold_ans = extract_gsm8k_answer(example["answer"])

            messages = [
                {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                {"role": "user", "content": f"Problem: {question}"},
            ]
            trajectory = []
            final_answer = None

            for step_id in range(12):  # max 12 steps
                logger.info(f"  Sample {idx}, step {step_id}...")
                response = model(messages, temperature=0.0, max_tokens=256)[0]
                raw_logprob = model.last_seq_logprob

                thought, action_name, expr, is_final = parse_react_output(response)

                if is_final:
                    final_answer = extract_gsm8k_answer(response)
                    break

                # Execute calculator
                if expr:
                    observation = calc.calculate(expr)
                else:
                    observation = "No calculation requested."

                # Compute UQ
                c_star, consistency_scores = cocoa.score_step(
                    context=messages,
                    greedy_output=response,
                    raw_seq_logprob=raw_logprob,
                )

                should_escalate, branch_cons_raw, branch_cons = brancher.check(
                    context=messages,
                    greedy_action=response,
                )

                step = {
                    "step_id": step_id,
                    "thought": thought[:500],
                    "action_name": action_name or "none",
                    "action_args": {"expression": expr or ""},
                    "observation": observation[:500],
                    "seq_logprob": raw_logprob,
                    "consistency_scores": consistency_scores,
                    "c_star_cocoa": c_star,
                    "branch_consistency": branch_cons,
                    "branch_consistency_raw": branch_cons_raw,
                    "escalated": should_escalate,
                }
                trajectory.append(step)

                # Update conversation
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )

            # Score
            task_success = answers_match(final_answer, gold_ans)

            record = {
                "idx": idx,
                "question": question[:200],
                "gold_answer": gold_ans,
                "pred_answer": final_answer,
                "task_success": task_success,
                "n_steps": len(trajectory),
                "trajectory": trajectory,
            }
            results.append(record)

            with open(traj_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")

            logger.info(
                f"  Sample {idx}: {'✓' if task_success else '✗'} "
                f"(pred={final_answer}, gold={gold_ans})"
            )

    finally:
        pass

    accuracy = (
        sum(r["task_success"] for r in results) / len(results)
        if results
        else float("nan")
    )
    logger.info("=" * 60)
    logger.info(f"Local test complete. Accuracy: {accuracy:.1%} ({sum(r['task_success'] for r in results)}/{len(results)})")
    logger.info(f"Results saved to: {traj_path}")


if __name__ == "__main__":
    asyncio.run(test_gsm8k_local())
