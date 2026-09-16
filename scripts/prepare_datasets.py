"""
Dataset Preparation Pipeline — Calibrated Agency
=================================================

Downloads, analyses, and transforms benchmark datasets into a unified JSONL schema
under data/<benchmark>/. All data files are gitignored.

By default prepares the three final thesis benchmarks: GSM8K, AgentBench-DB, HotpotQA.
AgentBench-OS and ToolBench (G1/G2) were dropped from the final thesis scope; their prep
functions remain here (opt in via `--only agentbench_os` / `toolbench_g1` / `toolbench_g2`)
to support the archived experiment code under legacy/; see legacy/README.md.

Unified schema per record
--------------------------
{
    "idx":        int,            # global zero-based index within the split
    "task_id":    str,            # unique stable identifier
    "benchmark":  str,            # "gsm8k" | "agentbench_os" | "agentbench_db" | "toolbench_g1" | "toolbench_g2" | "hotpotqa"
    "split":      str,            # "train" | "test" | "validation"
    "instruction": str,           # the problem / task description shown to the agent
    "gold_answer": str | None,    # expected final answer (normalised string)
    "metadata":   dict,           # benchmark-specific extra fields
}

Output
------
data/
├── gsm8k/                    (final thesis benchmark)
│   ├── train.jsonl          (7,473 records)
│   ├── test.jsonl           (1,319 records)
│   └── stats.json
├── agentbench_db/            (final thesis benchmark)
│   ├── test.jsonl
│   └── stats.json
├── hotpotqa/                 (final thesis benchmark)
│   ├── test.jsonl
│   └── stats.json
├── agentbench_os/            (excluded from final thesis, opt in explicitly)
│   ├── test.jsonl
│   └── stats.json
├── toolbench_g1/              (excluded from final thesis, opt in explicitly)
│   ├── test.jsonl
│   └── stats.json
├── toolbench_g2/              (excluded from final thesis, opt in explicitly)
│   ├── test.jsonl
│   └── stats.json
└── summary.json              (cross-benchmark statistics table)

Usage
-----
    python scripts/prepare_datasets.py                        # 3 final benchmarks
    python scripts/prepare_datasets.py --only gsm8k            # single benchmark
    python scripts/prepare_datasets.py --only agentbench_os    # opt into an excluded benchmark
    python scripts/prepare_datasets.py --force                 # re-download / overwrite
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

# `datasets` (HuggingFace) is imported lazily inside the gsm8k/hotpotqa prep functions
# below; it's only needed for those two benchmarks, not for AgentBench/ToolBench prep.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"


# =============================================================================
# Shared utilities
# =============================================================================

def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("  Saved %d records → %s", len(records), path)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    logger.info("  Stats saved → %s", path)


# =============================================================================
# GSM8K
# =============================================================================

def _extract_gsm8k_steps(answer_text: str) -> list[str]:
    """Extract the chain-of-thought reasoning steps from a GSM8K answer."""
    # Gold answer format: reasoning steps separated by newlines, final answer after ####
    lines = answer_text.split("\n")
    steps = [l.strip() for l in lines if l.strip() and not l.strip().startswith("####")]
    return steps


def _extract_gsm8k_gold(answer_text: str) -> Optional[str]:
    """Extract the numeric gold answer after ####."""
    m = re.search(r"####\s*([\d,.\-]+)", answer_text)
    return m.group(1).replace(",", "").strip() if m else None


def _count_calculations(answer_text: str) -> int:
    """Count the number of arithmetic calculation expressions in a GSM8K solution."""
    # Look for expressions containing operators
    return len(re.findall(r"<<[^>]+>>", answer_text))


def _difficulty_label(n_steps: int) -> str:
    """Map solution length to a difficulty tier used throughout the thesis.

    Bins match Section 5.3 stratification:
      Easy:   1-2 reasoning steps  (short, single-calculation problems)
      Medium: 3-4 reasoning steps
      Hard:   5+ reasoning steps   (multi-step chains, higher error risk)
    """
    if n_steps <= 2:
        return "Easy"
    elif n_steps <= 4:
        return "Medium"
    else:
        return "Hard"


def prepare_gsm8k(force: bool = False) -> dict:
    """Download and transform GSM8K. Returns statistics dict."""
    from datasets import load_dataset

    out_dir = DATA_DIR / "gsm8k"
    if not force and (out_dir / "test.jsonl").exists():
        logger.info("GSM8K already prepared (use --force to re-download).")
        return json.loads((out_dir / "stats.json").read_text())

    logger.info("Preparing GSM8K ...")
    ds = load_dataset("gsm8k", "main")

    all_stats: dict[str, dict] = {}

    for split_name in ["train", "test"]:
        split = ds[split_name]
        records: list[dict] = []

        gold_answers: list[float] = []
        step_counts:  list[int]   = []
        calc_counts:  list[int]   = []
        q_lengths:    list[int]   = []

        for idx, row in enumerate(split):
            gold = _extract_gsm8k_gold(row["answer"])
            steps = _extract_gsm8k_steps(row["answer"])
            n_calcs = _count_calculations(row["answer"])

            n_steps = len(steps)
            records.append({
                "idx":         idx,
                "task_id":     f"gsm8k_{split_name}_{idx}",
                "benchmark":   "gsm8k",
                "split":       split_name,
                "instruction": row["question"],
                "gold_answer": gold,
                "metadata": {
                    "raw_answer":     row["answer"],
                    "n_steps":        n_steps,
                    "n_calculations": n_calcs,
                    "question_len":   len(row["question"].split()),
                    "difficulty":     _difficulty_label(n_steps),
                },
            })

            if gold:
                try:
                    gold_answers.append(float(gold))
                except ValueError:
                    pass
            step_counts.append(len(steps))
            calc_counts.append(n_calcs)
            q_lengths.append(len(row["question"].split()))

        save_jsonl(records, out_dir / f"{split_name}.jsonl")


        # Difficulty distribution
        difficulty_counts: dict[str, int] = {"Easy": 0, "Medium": 0, "Hard": 0}
        for r in records:
            difficulty_counts[r["metadata"]["difficulty"]] += 1

        all_stats[split_name] = {
            "n_examples":           len(records),
            "answer_numeric_count": len(gold_answers),
            "difficulty_counts":    difficulty_counts,
            "answer_min":           float(np.min(gold_answers)) if gold_answers else None,
            "answer_max":           float(np.max(gold_answers)) if gold_answers else None,
            "answer_median":        float(np.median(gold_answers)) if gold_answers else None,
            "answer_p25":           float(np.percentile(gold_answers, 25)) if gold_answers else None,
            "answer_p75":           float(np.percentile(gold_answers, 75)) if gold_answers else None,
            "steps_mean":           float(np.mean(step_counts)),
            "steps_std":            float(np.std(step_counts)),
            "steps_min":            int(np.min(step_counts)),
            "steps_max":            int(np.max(step_counts)),
            "steps_median":         float(np.median(step_counts)),
            "calculations_mean":    float(np.mean(calc_counts)),
            "calculations_std":     float(np.std(calc_counts)),
            "calculations_median":  float(np.median(calc_counts)),
            "question_len_mean":    float(np.mean(q_lengths)),
            "question_len_std":     float(np.std(q_lengths)),
        }

    stats = {
        "benchmark":   "gsm8k",
        "description": "Grade-school multi-step arithmetic (Cobbe et al., 2021)",
        "hf_id":       "gsm8k / main",
        "tool":        "calculator",
        "splits":      all_stats,
    }
    save_stats(stats, out_dir / "stats.json")
    logger.info("GSM8K preparation complete.")
    return stats


# =============================================================================
# AgentBench OS + DB
# =============================================================================

def _load_agentbench_raw(env_type: str, agentbench_root: Path) -> list[dict]:
    path = agentbench_root / "data" / env_type / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"AgentBench {env_type} data not found at {path}.\n"
            "Run: bash scripts/setup_agentbench.sh"
        )
    return load_jsonl(path)


def prepare_agentbench(env_type: str, agentbench_root: Path,
                       force: bool = False) -> dict:
    """Transform AgentBench OS or DB tasks into unified schema."""

    bname  = f"agentbench_{env_type}"
    out_dir = DATA_DIR / bname

    if not force and (out_dir / "test.jsonl").exists():
        logger.info("%s already prepared.", bname.upper())
        return json.loads((out_dir / "stats.json").read_text())

    logger.info("Preparing AgentBench-%s ...", env_type.upper())
    raw = _load_agentbench_raw(env_type, agentbench_root)

    records: list[dict] = []
    instr_lengths: list[int] = []
    has_gold: int = 0

    for idx, row in enumerate(raw):
        instruction = row.get("instruction", row.get("question", ""))
        gold        = row.get("answer", row.get("gold_answer"))

        records.append({
            "idx":         idx,
            "task_id":     row.get("task_id", f"{bname}_{idx}"),
            "benchmark":   bname,
            "split":       "test",
            "instruction": instruction,
            "gold_answer": str(gold) if gold is not None else None,
            "metadata": {
                "env_type":      env_type,
                "initial_state": row.get("initial_state", {}),
                "tags":          row.get("tags", []),
                "instruction_len": len(instruction.split()),
            },
        })

        instr_lengths.append(len(instruction.split()))
        if gold is not None:
            has_gold += 1

    save_jsonl(records, out_dir / "test.jsonl")

    tool = "bash (Docker)" if env_type == "os" else "SQLite (MCP)"
    stats = {
        "benchmark":        bname,
        "description":      f"AgentBench interactive {env_type.upper()} environment (Liu et al., 2024)",
        "tool":             tool,
        "splits": {
            "test": {
                "n_examples":           len(records),
                "has_gold_answer_count": has_gold,
                "instruction_len_mean": float(np.mean(instr_lengths)),
                "instruction_len_std":  float(np.std(instr_lengths)),
                "instruction_len_min":  int(np.min(instr_lengths)),
                "instruction_len_max":  int(np.max(instr_lengths)),
                "instruction_len_median": float(np.median(instr_lengths)),
            }
        },
    }
    save_stats(stats, out_dir / "stats.json")
    logger.info("AgentBench-%s preparation complete.", env_type.upper())
    return stats


# =============================================================================
# ToolBench G1 / G2
# =============================================================================

def _load_toolbench_raw(split: str, toolbench_root: Path) -> list[dict]:
    path = toolbench_root / "data" / "test_instructions" / f"{split}_instruction.json"
    if not path.exists():
        raise FileNotFoundError(
            f"ToolBench {split} data not found at {path}.\n"
            "Run: bash legacy/setup_toolbench.sh"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def prepare_toolbench(split: str, toolbench_root: Path,
                      force: bool = False) -> dict:
    """Transform ToolBench G1 or G2 queries into unified schema."""

    bname   = f"toolbench_{split.lower()}"
    out_dir = DATA_DIR / bname

    if not force and (out_dir / "test.jsonl").exists():
        logger.info("%s already prepared.", bname.upper())
        return json.loads((out_dir / "stats.json").read_text())

    logger.info("Preparing ToolBench-%s ...", split)
    raw = _load_toolbench_raw(split, toolbench_root)

    records: list[dict] = []
    q_lengths:   list[int] = []
    has_gold: int = 0
    categories: dict[str, int] = {}

    for idx, row in enumerate(raw):
        query    = row.get("query", row.get("instruction", ""))
        gold     = row.get("answer")
        category = row.get("category", "unknown")

        records.append({
            "idx":         idx,
            "task_id":     row.get("query_id", f"{bname}_{idx}"),
            "benchmark":   bname,
            "split":       "test",
            "instruction": query,
            "gold_answer": str(gold) if gold is not None else None,
            "metadata": {
                "category":    category,
                "api_list":    row.get("api_list", []),
                "query_len":   len(query.split()),
                "n_apis":      len(row.get("api_list", [])),
            },
        })

        q_lengths.append(len(query.split()))
        if gold is not None:
            has_gold += 1
        categories[category] = categories.get(category, 0) + 1

    save_jsonl(records, out_dir / "test.jsonl")

    top_cats = sorted(categories.items(), key=lambda x: -x[1])[:10]
    stats = {
        "benchmark":   bname,
        "description": f"ToolBench {split} ({('single-tool' if split=='G1' else 'intra-category multi-tool')}) — Qin et al. (2023)",
        "tool":        "REST APIs (mock server)",
        "splits": {
            "test": {
                "n_examples":           len(records),
                "has_gold_answer_count": has_gold,
                "query_len_mean":       float(np.mean(q_lengths)),
                "query_len_std":        float(np.std(q_lengths)),
                "query_len_min":        int(np.min(q_lengths)),
                "query_len_max":        int(np.max(q_lengths)),
                "query_len_median":     float(np.median(q_lengths)),
                "n_categories":         len(categories),
                "top_10_categories":    dict(top_cats),
            }
        },
    }
    save_stats(stats, out_dir / "stats.json")
    logger.info("ToolBench-%s preparation complete.", split)
    return stats


# =============================================================================
# HotpotQA
# =============================================================================

def prepare_hotpotqa(force: bool = False) -> dict:
    """
    Download and transform HotpotQA (distractor setting) into unified schema.

    Stores full context paragraphs and supporting-fact titles in metadata so
    that experiments can run without reloading the HuggingFace dataset.

    Splits prepared
    ---------------
    validation : 7,405 records, used for both calib and test phases
    train      : 90,564 records, available for larger-scale calibration studies

    The agent experiment defaults to the validation split (answers are public).
    The test split withholds answers for the HotpotQA leaderboard and is skipped.
    """
    from datasets import load_dataset

    out_dir = DATA_DIR / "hotpotqa"
    if not force and (out_dir / "validation.jsonl").exists():
        logger.info("HotpotQA already prepared (use --force to re-download).")
        return json.loads((out_dir / "stats.json").read_text())

    logger.info("Preparing HotpotQA (distractor) ...")
    ds = load_dataset("hotpot_qa", "distractor")

    all_stats: dict[str, dict] = {}

    for split_name in ["train", "validation"]:
        if split_name not in ds:
            logger.warning("Split '%s' not found in hotpot_qa dataset; skipping.", split_name)
            continue

        split = ds[split_name]
        records:      list[dict] = []
        q_lengths:    list[int]  = []
        n_paragraphs: list[int]  = []
        type_counts:  dict[str, int] = {}
        level_counts: dict[str, int] = {}

        for idx, row in enumerate(split):
            # Reconstruct context as list of dicts for easy downstream use
            ctx = row["context"]
            paragraphs = [
                {"title": t, "sentences": s}
                for t, s in zip(ctx["title"], ctx["sentences"])
            ]

            # Deduplicate supporting fact titles (preserve order)
            sf_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))

            q_type = row.get("type", "unknown")
            level  = row.get("level", "unknown")

            records.append({
                "idx":         idx,
                "task_id":     f"hotpotqa_{split_name}_{idx}",
                "benchmark":   "hotpotqa",
                "split":       split_name,
                "instruction": row["question"],
                "gold_answer": row["answer"],
                "metadata": {
                    "id":                    row.get("id", f"hotpotqa_{idx}"),
                    "type":                  q_type,
                    "level":                 level,
                    "context":               paragraphs,
                    "supporting_fact_titles": sf_titles,
                    "supporting_facts": [
                        {"title": t, "sent_id": s}
                        for t, s in zip(
                            row["supporting_facts"]["title"],
                            row["supporting_facts"]["sent_id"],
                        )
                    ],
                    "question_len": len(row["question"].split()),
                    "n_paragraphs": len(paragraphs),
                    "answer_len":   len(row["answer"].split()),
                },
            })

            q_lengths.append(len(row["question"].split()))
            n_paragraphs.append(len(paragraphs))
            type_counts[q_type]  = type_counts.get(q_type, 0) + 1
            level_counts[level]  = level_counts.get(level, 0) + 1

        save_jsonl(records, out_dir / f"{split_name}.jsonl")

        all_stats[split_name] = {
            "n_examples":          len(records),
            "type_counts":         type_counts,
            "level_counts":        level_counts,
            "question_len_mean":   float(np.mean(q_lengths)),
            "question_len_std":    float(np.std(q_lengths)),
            "question_len_min":    int(np.min(q_lengths)),
            "question_len_max":    int(np.max(q_lengths)),
            "question_len_median": float(np.median(q_lengths)),
            "n_paragraphs_mean":   float(np.mean(n_paragraphs)),
        }

    stats = {
        "benchmark":   "hotpotqa",
        "description": (
            "HotpotQA multi-hop QA (distractor setting, Yang et al., 2018). "
            "Each question requires reasoning over 2 supporting paragraphs "
            "selected from 10 candidates (2 gold + 8 distractors)."
        ),
        "hf_id":       "hotpot_qa / distractor",
        "tool":        "in-memory context search (search + lookup)",
        "splits":      all_stats,
    }
    save_stats(stats, out_dir / "stats.json")
    logger.info("HotpotQA preparation complete.")
    return stats


# =============================================================================
# Cross-benchmark summary
# =============================================================================

def build_summary(all_stats: dict[str, dict]) -> None:
    """Build and save a cross-benchmark summary table for the thesis."""

    rows = []
    for bname, stats in all_stats.items():
        splits = stats.get("splits", {})
        test   = splits.get("test", {})
        train  = splits.get("train", {})
        # HotpotQA uses "validation" instead of "test"
        val    = splits.get("validation", {})
        n_test  = test.get("n_examples") or val.get("n_examples", "—")
        n_train = train.get("n_examples", "—")

        rows.append({
            "benchmark":   bname,
            "description": stats.get("description", ""),
            "tool":        stats.get("tool", ""),
            "n_train":     n_train,
            "n_test":      n_test,
            # Instruction/question length in words (key differs per benchmark)
            "avg_instruction_len_words": (
                test.get("instruction_len_mean")
                or test.get("query_len_mean")
                or test.get("question_len_mean")
                or val.get("question_len_mean")  # HotpotQA uses validation key
                or None
            ),
            "avg_tool_calls": (
                test.get("calculations_mean")          # GSM8K
                or None
            ),
            "avg_reasoning_steps": (
                test.get("steps_mean")                 # GSM8K
                or None
            ),
            "has_gold_answer": test.get("has_gold_answer_count", n_test),
        })

    summary = {
        "generated_by": "scripts/prepare_datasets.py",
        "benchmarks":   rows,
        "thesis_table": (
            "Use this table for Chapter 4.2 (Benchmarks and Tasks).\n"
            "Key columns: benchmark, n_test, avg_instruction_len, tool, has_gold_answer."
        ),
    }

    out_path = DATA_DIR / "summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Cross-benchmark summary → %s", out_path)

    # Pretty-print to console
    print("\n" + "="*72)
    print("  DATASET SUMMARY  (for thesis Chapter 4.2)")
    print("="*72)
    hdr = f"{'Benchmark':<22} {'N test':>7} {'N train':>8} {'Instr (words)':>14} {'Avg steps':>10} {'Avg tool calls':>15} {'Tool'}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        instr = row.get("avg_instruction_len_words")
        steps = row.get("avg_reasoning_steps")
        calls = row.get("avg_tool_calls")
        fmt   = lambda v: f"{v:.1f}" if isinstance(v, float) else "—"
        print(
            f"{row['benchmark']:<22} "
            f"{str(row['n_test']):>7} "
            f"{str(row['n_train']):>8} "
            f"{fmt(instr):>14} "
            f"{fmt(steps):>10} "
            f"{fmt(calls):>15} "
            f"{row['tool']}"
        )
    print()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and analyse datasets for Calibrated Agency experiments."
    )
    parser.add_argument(
        "--only",
        choices=["gsm8k", "agentbench_os", "agentbench_db",
                 "toolbench_g1", "toolbench_g2", "hotpotqa"],
        nargs="+",
        default=None,
        help="Run only these benchmarks (default: the three final thesis benchmarks: "
             "gsm8k, agentbench_db, hotpotqa. agentbench_os/toolbench_g1/toolbench_g2 "
             "were dropped from the final thesis scope but remain available here for "
             "the archived experiment code under legacy/, see legacy/README.md).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing data files.",
    )
    parser.add_argument(
        "--agentbench-root",
        type=Path,
        default=ROOT / "AgentBench",
        help="Path to cloned AgentBench repository.",
    )
    parser.add_argument(
        "--toolbench-root",
        type=Path,
        default=ROOT / "ToolBench",
        help="Path to cloned ToolBench repository.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Default: the three final thesis benchmarks only. Pass --only agentbench_os /
    # toolbench_g1 / toolbench_g2 explicitly to prepare data for the archived
    # (non-final) benchmarks; see legacy/README.md.
    todo = args.only or ["gsm8k", "agentbench_db", "hotpotqa"]
    all_stats: dict[str, dict] = {}

    if "gsm8k" in todo:
        all_stats["gsm8k"] = prepare_gsm8k(force=args.force)

    for env in ["os", "db"]:
        key = f"agentbench_{env}"
        if key in todo:
            try:
                all_stats[key] = prepare_agentbench(
                    env, args.agentbench_root, force=args.force
                )
            except FileNotFoundError as exc:
                logger.warning("Skipping %s: %s", key, exc)

    for split in ["G1", "G2"]:
        key = f"toolbench_{split.lower()}"
        if key in todo:
            try:
                all_stats[key] = prepare_toolbench(
                    split, args.toolbench_root, force=args.force
                )
            except FileNotFoundError as exc:
                logger.warning("Skipping %s: %s", key, exc)

    if "hotpotqa" in todo:
        all_stats["hotpotqa"] = prepare_hotpotqa(force=args.force)

    if all_stats:
        build_summary(all_stats)
        logger.info("Done. Data written to %s/", DATA_DIR)
    else:
        logger.warning("No datasets were prepared.")


if __name__ == "__main__":
    main()
