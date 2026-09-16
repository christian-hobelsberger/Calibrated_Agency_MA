# Lazy imports. Heavy dependencies (vllm, datasets, torch) are only loaded
# when the specific experiment function is actually called, not at import time.
#
# Covers the three final thesis benchmarks (GSM8K, AgentBench-DB, HotpotQA).
# ToolBench was dropped from the final scope; its module was archived to
# legacy/toolbench_agent.py; see legacy/README.md.

__all__ = [
    "run_gsm8k_experiment",
    "run_agentbench_experiment",
    "run_hotpotqa_experiment",
]


def __getattr__(name):
    if name == "run_gsm8k_experiment":
        from .gsm8k_agent import run_gsm8k_experiment
        return run_gsm8k_experiment
    if name == "run_agentbench_experiment":
        from .agentbench_agent import run_agentbench_experiment
        return run_agentbench_experiment
    if name == "run_hotpotqa_experiment":
        from .hotpotqa_agent import run_hotpotqa_experiment
        return run_hotpotqa_experiment
    raise AttributeError(f"module 'experiments' has no attribute {name!r}")
