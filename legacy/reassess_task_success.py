#!/usr/bin/env python3
"""
Re-assess task_success using the fixed evaluate_agentbench_task function.

This script loads existing trajectories and re-evaluates them with the updated
evaluation logic (comma/float handling + SQL spacing normalization).

Usage:
    python reassess_task_success.py <trajectories.jsonl> [--output <output.jsonl>]
"""
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
from collections import defaultdict


@dataclass
class AgentBenchTask:
    task_id: str
    env_type: str
    instruction: str
    gold_answer: Optional[Union[str, list[str]]]
    initial_state: dict


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for loose comparison: lowercase, collapse whitespace, strip quotes.

    Also strips backslashes so that \\`col\\` and `col` normalise identically,
    since the LLM sometimes emits backslash-escaped backticks copied from prompt examples.

    Normalizes spaces around = so that `col = val` and `col=val` compare as equal.
    """
    sql = sql.lower()
    sql = sql.replace("\\`", "`")      # unescape \` → ` before stripping
    sql = re.sub(r'[`"\'\\]', '', sql) # remove quoting chars AND stray backslashes
    sql = re.sub(r'\s*=\s*', '=', sql) # col = val  →  col=val  (purely cosmetic diff)
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql


def evaluate_agentbench_task(task: AgentBenchTask, trajectory: list) -> bool:
    """Check if task was successfully completed (FIXED VERSION).

    Handles both string gold answers (OS tasks) and list gold answers (DB tasks).

    For DB SELECT tasks (gold answer is a scalar value like "7" or "Chelsea"):
      Returns True if the gold value appears in a sql_query/sql_execute observation
      (NOT describe_table or list_tables, those now contain sample rows that could
      accidentally match the gold answer), OR in any FINAL ANSWER line in a thought.
      Also handles numeric gold answers stored as floats (e.g. '1.0' matches '1').
      Handles comma-formatted numbers: '32502.0' matches '32,502'.

    For DB INSERT/UPDATE tasks (gold answer is a SQL string):
      Returns True if the agent executed a SQL command that normalized-matches the gold SQL.
      Normalized comparison handles spaces around = and quote variations.

    For OS tasks (gold answer is a string):
      Returns True if the gold value appears in any observation or FINAL ANSWER.
    """
    if task.gold_answer is None:
        return any(
            "TASK COMPLETE" in str(s.get("thought", ""))
            or "FINAL ANSWER" in str(s.get("thought", "")).upper()
            for s in trajectory
        )

    gold_answers = (
        task.gold_answer if isinstance(task.gold_answer, list)
        else [task.gold_answer]
    )

    all_thoughts = [s.get("thought", "") for s in trajectory]
    all_action_cmds = [s.get("action_cmd", "") for s in trajectory]

    # Split observations by step type so describe_table sample rows
    # don't accidentally match the gold answer.
    sql_obs    = []   # from sql_query / sql_execute: valid answer sources
    other_obs  = []   # list_tables, describe_table, etc.: NOT valid answer sources
    for s in trajectory:
        obs = s.get("observation", "") or ""
        an  = s.get("action_name", "") or ""
        if an in ("sql_query", "sql_execute"):
            sql_obs.append(obs)
        else:
            other_obs.append(obs)

    # Extract FINAL ANSWER values from thoughts
    final_answer_texts = []
    for thought in all_thoughts:
        fa_m = re.search(r"FINAL\s+ANSWER:\s*(.+?)(?:\n|$)", thought, re.IGNORECASE)
        if fa_m:
            final_answer_texts.append(fa_m.group(1).strip())

    # For INSERT/UPDATE gold answers: compare against executed SQL commands
    for gold in gold_answers:
        gold_upper = gold.strip().upper()
        if gold_upper.startswith("INSERT") or gold_upper.startswith("UPDATE") or gold_upper.startswith("DELETE"):
            gold_norm = _normalize_sql(gold)
            for cmd in all_action_cmds:
                if cmd and _normalize_sql(cmd) == gold_norm:
                    return True
            # Loose match: ≥95% structural token overlap AND all quoted literal
            # values from gold must appear in the agent command (ensures the agent
            # used the correct values even if quoting/spacing differs).
            gold_tokens = set(gold_norm.split())
            gold_literals = set(
                v.lower() for v in re.findall(r"'([^']*)'", gold)
            )
            for cmd in all_action_cmds:
                if not cmd:
                    continue
                cmd_norm = _normalize_sql(cmd)
                cmd_tokens = set(cmd_norm.split())
                token_overlap = (
                    len(gold_tokens & cmd_tokens) / len(gold_tokens)
                    if gold_tokens else 0
                )
                # All gold literal values must be present in the raw (unnormalized)
                # command to ensure correct values were used.
                values_preserved = all(lit in cmd.lower() for lit in gold_literals)
                if token_overlap >= 0.95 and values_preserved:
                    return True
            continue

        # For SELECT-type / scalar answers:
        # Only search sql_query results + FINAL ANSWER (never describe_table obs).
        gold_clean = gold.strip().lower()
        searchable = sql_obs + final_answer_texts
        if any(gold_clean in text.lower() for text in searchable if text):
            return True

        # Numeric tolerance: gold='1.0' should match agent FINAL ANSWER '1' etc.
        # Also handles comma-formatted numbers: gold='32502.0' matches '32,502'.
        # The model copies value formats directly from the DB (e.g. "2,859") while
        # the gold label stores a plain float ("2859.0"); strip commas before
        # converting so both sides compare as the same number.
        try:
            gold_float = float(gold_clean.replace(',', ''))
            for fa_text in final_answer_texts:
                try:
                    if abs(float(fa_text.strip().replace(',', '')) - gold_float) < 1e-6:
                        return True
                except ValueError:
                    pass
            # Also check SQL query observations for comma-formatted numbers.
            # e.g. observation "{'Capacity': '2,859'}" should match gold 2859.0
            for obs_text in sql_obs:
                for num_m in re.finditer(r'\b[\d,]+(?:\.\d+)?\b', obs_text):
                    try:
                        if abs(float(num_m.group().replace(',', '')) - gold_float) < 1e-6:
                            return True
                    except ValueError:
                        pass
        except ValueError:
            pass

    return False


def get_task_type(gold_answer) -> str:
    """Determine task type from gold_answer."""
    if isinstance(gold_answer, list):
        gold_str = gold_answer[0] if gold_answer else ""
    else:
        gold_str = str(gold_answer) if gold_answer else ""

    if gold_str.upper().startswith("INSERT"):
        return "INSERT"
    elif gold_str.upper().startswith("UPDATE"):
        return "UPDATE"
    elif gold_str.upper().startswith("DELETE"):
        return "DELETE"
    else:
        return "SELECT"


def main():
    if len(sys.argv) < 2:
        print("Usage: python reassess_task_success.py <trajectories.jsonl> [--output <output.jsonl>]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = None

    # Parse optional --output flag
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--output" and i + 2 < len(sys.argv):
            output_file = Path(sys.argv[i + 2])
            break

    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    print(f"Reading trajectories from {input_file}...")

    # Load and re-evaluate trajectories
    records = []
    changes = []  # Track (task_id, old_success, new_success)
    task_type_stats = defaultdict(lambda: {"before": 0, "after": 0, "total": 0})

    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} ({e})")
                continue

            task = AgentBenchTask(
                task_id=record["task_id"],
                env_type=record["env_type"],
                instruction=record["instruction"],
                gold_answer=record["gold_answer"],
                initial_state={},
            )

            # Extract trajectory in evaluator format
            trajectory = [
                {
                    "thought": step["thought"],
                    "observation": step["observation"],
                    "action_name": step["action_name"],
                    "action_cmd": step["action_cmd"],
                }
                for step in record["trajectory"]
            ]

            old_success = record["task_success"]
            new_success = evaluate_agentbench_task(task, trajectory)

            task_type = get_task_type(record["gold_answer"])
            task_type_stats[task_type]["total"] += 1
            if old_success:
                task_type_stats[task_type]["before"] += 1
            if new_success:
                task_type_stats[task_type]["after"] += 1

            if old_success != new_success:
                changes.append((record["task_id"], old_success, new_success, task_type))

            # Update record with new success status
            record["task_success"] = new_success
            records.append(record)

    # Print summary
    print("\n" + "=" * 90)
    print("RE-ASSESSMENT SUMMARY")
    print("=" * 90)

    total_before = sum(1 for r in records if r["task_success"])
    total_after = sum(1 for r in records if r["task_success"])  # Already updated above

    print(f"\nTotal tasks: {len(records)}")
    print(f"Changed: {len(changes)} tasks")

    if changes:
        print("\n" + "-" * 90)
        print("Task Type Breakdown (Before -> After):")
        print("-" * 90)
        for task_type in sorted(task_type_stats.keys()):
            stats = task_type_stats[task_type]
            before_rate = 100.0 * stats["before"] / stats["total"]
            after_rate = 100.0 * stats["after"] / stats["total"]
            improvement = stats["after"] - stats["before"]
            print(f"{task_type:8s}: {stats['before']:3d}/{stats['total']:3d} ({before_rate:5.1f}%) -> "
                  f"{stats['after']:3d}/{stats['total']:3d} ({after_rate:5.1f}%)  "
                  f"  [+{improvement:2d} fixed]")

        overall_before = sum(s["before"] for s in task_type_stats.values())
        overall_after = sum(s["after"] for s in task_type_stats.values())
        overall_total = sum(s["total"] for s in task_type_stats.values())
        overall_before_rate = 100.0 * overall_before / overall_total
        overall_after_rate = 100.0 * overall_after / overall_total

        print("-" * 90)
        print(f"{'OVERALL':8s}: {overall_before:3d}/{overall_total:3d} ({overall_before_rate:5.1f}%) -> "
              f"{overall_after:3d}/{overall_total:3d} ({overall_after_rate:5.1f}%)  "
              f"  [+{overall_after - overall_before:2d} fixed]")

        print("\n" + "-" * 90)
        print("Changed Tasks (sample of first 20):")
        print("-" * 90)
        for task_id, old, new, task_type in changes[:20]:
            direction = "FAIL->PASS" if new else "PASS->FAIL"
            print(f"  Task {task_id:3s} ({task_type:8s}): {old} -> {new}  {direction}")

        if len(changes) > 20:
            print(f"  ... and {len(changes) - 20} more")
    else:
        print("\nNo changes detected. Evaluation logic may be identical.")

    # Save output if requested
    if output_file:
        print(f"\nSaving re-assessed trajectories to {output_file}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"[OK] Saved {len(records)} records to {output_file}")


if __name__ == "__main__":
    main()
