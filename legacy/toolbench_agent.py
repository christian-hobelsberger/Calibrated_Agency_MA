"""
ToolBench G1 + G2 Experiment.

Uses the ToolBench mock/real API server for deterministic, offline tool calls.
Evaluates step-wise UQ on real REST API call chains.

Setup required:
    git clone https://github.com/OpenBMB/ToolBench.git
    cd ToolBench && pip install -r requirements.txt
    export RAPIDAPI_KEY="..." (or use mock server)

See scripts/setup_toolbench.sh for full setup.

References:
    Qin et al. (2023) — ToolLLM / ToolBench
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

import requests

from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

TOOLBENCH_SYSTEM_PROMPT = """\
You are an AI assistant that answers questions using external APIs.

Available tools:
{tool_descriptions}

At each step, reason about what to do, then call ONE tool:
  Thought: <your reasoning>
  Action: <tool_name>
  API: <api_endpoint_name>
  Action Input: {{"param1": "value1", "param2": "value2"}}

When you have the final answer:
  Final Answer: <your answer>

IMPORTANT:
- Do NOT fabricate API responses. Only use what the Observation returns.
- Call exactly one API per step.\
"""


# ---------------------------------------------------------------------------
# ToolBench client
# ---------------------------------------------------------------------------

class ToolBenchClient:
    """Client for the ToolBench mock/real API server."""

    def __init__(self, server_url: Optional[str] = None):
        cfg = DEFAULT_CONFIG.experiment
        self.server_url = (server_url or cfg.toolbench_server).rstrip("/")
        self._call_log: list[dict] = []

    def list_tools(self, category: str) -> list[dict]:
        """Get available tools for a query category."""
        try:
            resp = requests.get(
                f"{self.server_url}/tools",
                params={"category": category},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("tools", [])
        except requests.RequestException as exc:
            logger.warning("ToolBench list_tools failed: %s", exc)
            return []

    def call_tool(
        self,
        tool_name: str,
        api_name: str,
        params: dict,
    ) -> dict:
        """Invoke a specific API endpoint via the ToolBench server."""
        payload = {
            "tool_name": tool_name,
            "api_name":  api_name,
            "tool_input": json.dumps(params),
        }
        try:
            resp = requests.post(
                f"{self.server_url}/virtual",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as exc:
            result = {"error": str(exc), "status_code": -1}

        self._call_log.append({
            "tool_name": tool_name,
            "api_name":  api_name,
            "params":    params,
            "response":  result,
        })
        return result

    @property
    def call_log(self) -> list[dict]:
        return self._call_log


# ---------------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------------

def load_toolbench_queries(
    split: str = "G1",
    n: int = 300,
    toolbench_root: str = "ToolBench",
) -> list[dict]:
    """Load ToolBench query instances from the official data files."""
    path = (
        Path(toolbench_root) / "data" / "test_instructions"
        / f"{split}_instruction.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"ToolBench data not found at {path}. "
            "Run scripts/setup_toolbench.sh first."
        )
    with open(path) as fh:
        data = json.load(fh)
    return data[:n]


def format_tool_descriptions(tools: list[dict]) -> str:
    """Format tool list for the system prompt."""
    if not tools:
        return "(no tools available)"
    lines: list[str] = []
    for t in tools:
        lines.append(f"Tool: {t.get('tool_name', 'unknown')}")
        for api in t.get("api_list", []):
            desc = api.get("description", "")[:120]
            lines.append(f"  - {api['name']}: {desc}")
    return "\n".join(lines)


def parse_toolbench_action(response: str) -> tuple[str, str, dict]:
    """Parse Action / API / Action Input from a ReAct response."""
    tool_m  = re.search(r"Action:\s*(.+?)(?:\n|$)", response)
    api_m   = re.search(r"API:\s*(.+?)(?:\n|$)", response)
    input_m = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)

    tool_name = tool_m.group(1).strip() if tool_m else ""
    api_name  = api_m.group(1).strip()  if api_m  else ""
    try:
        params = json.loads(input_m.group(1)) if input_m else {}
    except json.JSONDecodeError:
        params = {}

    return tool_name, api_name, params


def check_toolbench_success(
    pred_answer: str,
    gold_answer: Optional[str],
) -> bool:
    """Token-F1-based success check (ToolBench evaluation protocol)."""
    if not pred_answer:
        return False
    if gold_answer is None:
        return "Final Answer:" in pred_answer

    pred_tokens = set(pred_answer.lower().split())
    gold_tokens = set(gold_answer.lower().split())
    common = pred_tokens & gold_tokens
    if not common:
        return False
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall    = len(common) / len(gold_tokens) if gold_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return f1 > 0.5


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

async def run_toolbench_experiment(
    split: str = "G1",
    n_samples: int = 300,
    output_dir: str = "results/toolbench_g1",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cocoa_m: int = 5,
    temperature: float = 0.7,
    resume: bool = True,
) -> list[dict]:
    """
    Run the ToolBench G1 or G2 experiment.

    Parameters
    ----------
    split      : 'G1' (single-tool) or 'G2' (intra-category multi-tool).
    n_samples  : Number of queries to evaluate.
    output_dir : Output directory for JSONL.
    resume     : Skip already-completed indices.
    """
    from tqdm import tqdm
    from agent.instrumented_agent import InstrumentedVLLMModel, StepRecord
    from uq.annotation import annotate_steps_by_task_success
    from uq.branching import BranchingConsistency
    from uq.stepwise_cocoa import StepwiseCoCoA

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"

    logger.info("Loading ToolBench %s queries (%d).", split, n_samples)
    queries = load_toolbench_queries(split=split, n=n_samples)

    # Resume support
    completed: set[int] = set()
    if resume and traj_path.exists():
        with open(traj_path) as fh:
            for line in fh:
                try:
                    completed.add(json.loads(line)["idx"])
                except Exception:
                    pass
        logger.info("Resuming: %d queries already done.", len(completed))

    logger.info("Loading model: %s", model_name)
    model     = InstrumentedVLLMModel(model_name)
    tb_client = ToolBenchClient()
    cocoa     = StepwiseCoCoA(model=model, M=cocoa_m, temperature=temperature)
    brancher  = BranchingConsistency(model=model, M=5, temperature=temperature)

    results: list[dict] = []
    cfg = DEFAULT_CONFIG.experiment

    for idx, query in enumerate(tqdm(queries, desc=f"ToolBench-{split}")):
        if idx in completed:
            continue

        tools     = tb_client.list_tools(query.get("category", ""))
        tool_desc = format_tool_descriptions(tools)
        system_prompt = TOOLBENCH_SYSTEM_PROMPT.format(
            tool_descriptions=tool_desc
        )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Question: {query['query']}"},
        ]
        trajectory: list[StepRecord] = []
        final_answer: Optional[str] = None

        for step_id in range(cfg.toolbench_max_steps):
            response = model(messages, temperature=0.0, max_tokens=512)[0]
            raw_logprob = model.last_seq_logprob

            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()

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

                # Record the final answer step with UQ annotations
                step = StepRecord(
                    step_id=step_id,
                    thought=response,
                    action_name="final_answer",
                    action_args={},
                    observation=final_answer,
                    logprobs=model.last_logprobs,
                    seq_logprob=raw_logprob,
                    consistency_scores=consistency_scores,
                    c_star_cocoa=c_star,
                    branch_consistency_raw=branch_cons_raw,
                    branch_consistency=branch_cons,
                    escalated=should_escalate,
                )
                trajectory.append(step)
                break

            tool_name, api_name, params = parse_toolbench_action(response)
            if tool_name and api_name:
                obs_dict   = tb_client.call_tool(tool_name, api_name, params)
                observation = json.dumps(obs_dict)[:1000]
            else:
                observation = (
                    "No valid action parsed. "
                    "Use: Action: <tool_name>, API: <endpoint>, "
                    "Action Input: {\"key\": \"value\"}"
                )

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
                thought=response,
                action_name=f"{tool_name}/{api_name}" if tool_name else "none",
                action_args=params,
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
            messages.append({"role": "user",
                             "content": f"Observation: {observation}"})

        task_success = check_toolbench_success(
            final_answer or "", query.get("answer")
        )
        annotated = annotate_steps_by_task_success(trajectory, task_success)

        record = {
            "idx":          idx,
            "query":        query["query"],
            "gold_answer":  query.get("answer"),
            "pred_answer":  final_answer,
            "task_success": task_success,
            "n_steps":      len(annotated),
            "trajectory": [
                {
                    "step_id":           s.step_id,
                    "thought":           s.thought[:500],
                    "action_name":       s.action_name,
                    "action_args":       s.action_args,
                    "observation":       s.observation[:500],
                    "seq_logprob":       s.seq_logprob,
                    "consistency_scores": s.consistency_scores,
                    "c_star_cocoa":      s.c_star_cocoa,
                    "branch_consistency_raw": s.branch_consistency_raw,
                    "branch_consistency": s.branch_consistency,
                    "escalated":         s.escalated,
                    "is_correct":        s.is_correct,
                }
                for s in annotated
            ],
        }
        results.append(record)
        with open(traj_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    logger.info(
        "ToolBench-%s done. Success rate: %.3f",
        split,
        sum(r["task_success"] for r in results) / max(len(results), 1),
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToolBench Agent Experiment")
    parser.add_argument("--split",       choices=["G1", "G2"], default="G1")
    parser.add_argument("--n-samples",   type=int, default=300)
    parser.add_argument("--output-dir",  type=str, default="results/toolbench_g1")
    parser.add_argument("--model",       type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cocoa-m",     type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-resume",   action="store_true")
    args = parser.parse_args()

    asyncio.run(run_toolbench_experiment(
        split=args.split,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        model_name=args.model,
        cocoa_m=args.cocoa_m,
        temperature=args.temperature,
        resume=not args.no_resume,
    ))
