#!/usr/bin/env python
"""
Simplified local CPU test of GSM8K pipeline.

Uses hardcoded examples (no datasets dependency) to validate:
- Model loading and inference
- Calculator execution
- UQ module integration

Usage:
    python test_local_gsm8k_simple.py
"""
import asyncio
import json
import logging
import re
from pathlib import Path

from config import DEFAULT_CONFIG
from agent.cpu_local_model import CPULocalModel
from agent.mcp_client import DirectCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Simple GSM8K examples for local testing
TEST_PROBLEMS = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 eggs per day and bakes muffins with 4 eggs per day. She sells the remainder at the farmers market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers market?",
        "gold_answer": "18",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does the robe take?",
        "gold_answer": "3",
    },
    {
        "question": "Carla is downloading a file that is 2 GB. So far she has downloaded 400 MB. If her internet speed is 2 MB/s, how much longer in seconds will the download take?",
        "gold_answer": "800",
    },
]

GSM8K_SYSTEM_PROMPT = """\
You are a precise mathematical reasoning assistant.
Solve the given word problem step by step.

RULES:
- For EVERY arithmetic calculation you MUST call the calculate tool.
- Format each step as:
    Thought: <reasoning>
    Action: calculate("<expression>")
- After final calculation emit:
    FINAL ANSWER: <number>"""


def extract_answer(text: str):
    """Extract numeric answer from text."""
    m = re.search(r"FINAL ANSWER:\s*([\d,.\-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def answers_match(pred, gold):
    """Compare answers with tolerance."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except ValueError:
        return pred.strip() == gold.strip()


def parse_react(text: str):
    """Parse ReAct output."""
    if re.search(r"FINAL ANSWER:", text, re.IGNORECASE):
        return text, None, None, True

    thought_m = re.search(r"Thought:(.*?)(?:Action:|$)", text, re.DOTALL | re.IGNORECASE)
    action_m = re.search(
        r'Action:\s*calculate\(["\']?(.*?)["\']?\)', text, re.DOTALL | re.IGNORECASE
    )

    thought = thought_m.group(1).strip() if thought_m else text.strip()
    expr = action_m.group(1).strip() if action_m else None

    return thought, "calculate" if expr else None, expr, False


async def main():
    """Run simple local test."""
    logger.info("=" * 70)
    logger.info("GSM8K Local CPU Test (simplified)")
    logger.info("=" * 70)

    # Load model
    logger.info("\n[1/4] Loading TinyLlama model on CPU...")
    try:
        model = CPULocalModel()
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return

    # Test calculator
    logger.info("\n[2/4] Testing calculator tool...")
    calc = DirectCalculator()
    try:
        result = calc.calculate("2 + 3")
        logger.info(f"✓ Calculator works: 2 + 3 = {result}")
    except Exception as e:
        logger.error(f"✗ Calculator failed: {e}")
        return

    # Run inference on one problem
    logger.info("\n[3/4] Testing model inference...")
    messages = [
        {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {TEST_PROBLEMS[0]['question']}"},
    ]
    try:
        outputs = model(messages, temperature=0.0, max_tokens=256, n_samples=2)
        logger.info(f"✓ Generated {len(outputs)} outputs (greedy + 1 sample)")
        logger.info(f"  Greedy response (first 200 chars): {outputs[0][:200]}...")
        logger.info(f"  Last seq logprob: {model.last_seq_logprob:.4f}")
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        return

    # Run full pipeline on 1-2 problems
    logger.info("\n[4/4] Running full GSM8K pipeline...")
    out_dir = Path("results/gsm8k_local_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"

    results = []
    for prob_idx, problem in enumerate(TEST_PROBLEMS[:2]):
        logger.info(f"\n  Problem {prob_idx + 1}/2: {problem['question'][:60]}...")

        messages = [
            {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem['question']}"},
        ]
        trajectory = []
        final_answer = None

        for step_id in range(10):
            try:
                response = model(messages, temperature=0.0, max_tokens=256)[0]
                thought, action_name, expr, is_final = parse_react(response)

                if is_final:
                    final_answer = extract_answer(response)
                    logger.info(f"    ✓ Final answer: {final_answer}")
                    break

                if expr:
                    observation = calc.calculate(expr)
                    logger.info(f"    Step {step_id}: calc({expr}) = {observation}")
                else:
                    observation = "No calculation."

                trajectory.append({
                    "step_id": step_id,
                    "action": expr or "none",
                    "observation": observation,
                })

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            except Exception as e:
                logger.error(f"    ✗ Step {step_id} failed: {e}")
                break

        task_success = answers_match(final_answer, problem["gold_answer"])
        record = {
            "idx": prob_idx,
            "question": problem["question"][:100],
            "gold_answer": problem["gold_answer"],
            "pred_answer": final_answer,
            "success": task_success,
            "steps": len(trajectory),
        }
        results.append(record)

        with open(traj_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

        status = "✓" if task_success else "✗"
        logger.info(f"    {status} Expected: {problem['gold_answer']}, Got: {final_answer}")

    # Summary
    logger.info("\n" + "=" * 70)
    accuracy = sum(r["success"] for r in results) / len(results) if results else 0
    logger.info(f"Test complete. Accuracy: {accuracy:.0%} ({sum(r['success'] for r in results)}/{len(results)})")
    logger.info(f"Results saved to: {traj_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
