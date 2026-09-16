"""
LRZ prerequisite check for Calibrated Agency.

Run this inside a SLURM job (with GPU allocation) before submitting
the full baseline experiments. It verifies:
  1. Python / CUDA environment
  2. All required packages import correctly
  3. GPU is visible and has sufficient VRAM
  4. Local model paths exist
  5. GSM8K dataset is reachable (HuggingFace or local JSONL)
  6. Calculator tool executes correctly
  7. NLI model (cross-encoder) loads
  8. vLLM short inference round-trip with the 8B model

Exit code 0 = all checks passed.

Note: each check below imports its own heavy dependency (datasets, sentence_transformers,
vllm, ...) inline rather than at module top level. This is deliberate, so that one
missing/broken package produces a single isolated FAIL for that check instead of an
ImportError that aborts the whole script before any other check can run.
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

LRZ_MODEL_DIR = os.getenv("LRZ_MODEL_DIR", "/path/to/models")
MODEL_8B  = f"{LRZ_MODEL_DIR}/Llama-3.1-8B-Instruct"
MODEL_70B = f"{LRZ_MODEL_DIR}/Llama-3.1-70B-Instruct"

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"
INFO = "\033[94m INFO\033[0m"

failures: list[str] = []


def check(label: str, fn):
    try:
        result = fn()
        print(f"[{PASS}] {label}" + (f": {result}" if result is not None else ""))
        return True
    except Exception as exc:
        print(f"[{FAIL}] {label}: {exc}")
        failures.append(label)
        return False


# ---------------------------------------------------------------------------
# 1. Python & environment
# ---------------------------------------------------------------------------
print("\n=== 1. Python & Environment ===")

def _python_version():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Need Python ≥ 3.10, got {v.major}.{v.minor}")
    return f"{v.major}.{v.minor}.{v.micro}"

check("Python version ≥ 3.10", _python_version)
check("PROJECT_DIR env",
      lambda: os.environ.get("PROJECT_DIR", "not set (using cwd)"))

# ---------------------------------------------------------------------------
# 2. Core imports
# ---------------------------------------------------------------------------
print("\n=== 2. Package Imports ===")
check("torch",          lambda: __import__("torch").__version__)
check("transformers",   lambda: __import__("transformers").__version__)
check("datasets",       lambda: __import__("datasets").__version__)
check("vllm",           lambda: __import__("vllm").__version__)
check("accelerate",     lambda: __import__("accelerate").__version__)
check("sentence_transformers",
      lambda: __import__("sentence_transformers").__version__)
check("lm_polygraph",   lambda: __import__("lm_polygraph").__version__)
check("sklearn",        lambda: __import__("sklearn").__version__)
check("mlflow",         lambda: __import__("mlflow").__version__)
check("jsonlines",      lambda: __import__("jsonlines").__version__)
check("rich",           lambda: __import__("rich").__version__)
check("tqdm",           lambda: __import__("tqdm").__version__)
check("mcp",            lambda: __import__("mcp").__version__)

# ---------------------------------------------------------------------------
# 3. CUDA / GPU
# ---------------------------------------------------------------------------
print("\n=== 3. CUDA / GPU ===")
import torch


def _cuda_available():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible")
    return "yes"


check("CUDA available", _cuda_available)

if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    check("GPU count", lambda: n_gpus)
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1e9

        def _gpu_name(p=props, v=vram_gb):
            return f"{v:.1f} GB VRAM"

        def _gpu_vram(v=vram_gb):
            if v < 40:
                raise RuntimeError(
                    f"Only {v:.1f} GB, 8B model needs >= 16 GB, recommend >= 40 GB"
                )
            return "OK"

        check(f"GPU {i} ({props.name})", _gpu_name)
        check(f"GPU {i} VRAM ≥ 40 GB", _gpu_vram)


# ---------------------------------------------------------------------------
# 4. Local model paths
# ---------------------------------------------------------------------------
print("\n=== 4. Local Model Paths ===")


def _check_dir(path: str):
    if not Path(path).is_dir():
        raise FileNotFoundError(path)
    return path


def _check_config(path: str):
    cfg = Path(path) / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(str(cfg))
    return "OK"


check("8B model dir exists",  lambda: _check_dir(MODEL_8B))
check("70B model dir exists", lambda: _check_dir(MODEL_70B))
check("8B config.json present",  lambda: _check_config(MODEL_8B))
check("70B config.json present", lambda: _check_config(MODEL_70B))

# ---------------------------------------------------------------------------
# 5. Dataset access
# ---------------------------------------------------------------------------
print("\n=== 5. Dataset (GSM8K) ===")
local_jsonl = Path("data/gsm8k/test.jsonl")
check("Local GSM8K JSONL",
      lambda: f"{sum(1 for _ in open(local_jsonl))} examples" if local_jsonl.exists()
      else "not found, will fall back to HuggingFace")

def _hf_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test[:3]")
    return f"{len(ds)} examples from HuggingFace"

check("HuggingFace GSM8K reachable", _hf_gsm8k)

# ---------------------------------------------------------------------------
# 6. Calculator tool
# ---------------------------------------------------------------------------
print("\n=== 6. Calculator Tool ===")

def _calc():
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from agent.mcp_client import DirectCalculator
    calc = DirectCalculator()
    result_str = calc.calculate("(12 * 3 + 7) / 5")
    result = float(result_str)
    expected = (12 * 3 + 7) / 5
    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"
    return f"{result}"

check("DirectCalculator arithmetic", _calc)

# ---------------------------------------------------------------------------
# 7. NLI model (cross-encoder for UQ)
# ---------------------------------------------------------------------------
print("\n=== 7. NLI Cross-Encoder ===")

def _nli():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/nli-roberta-base")
    score = model.predict([("The cat sat on the mat.", "There is a cat.")])
    return f"score shape OK ({score.shape})"

check("cross-encoder/nli-roberta-base loads & runs", _nli)

# ---------------------------------------------------------------------------
# 8. vLLM quick inference with 8B model
# ---------------------------------------------------------------------------
print("\n=== 8. vLLM Inference (8B model) ===")

def _vllm_inference():
    if not Path(MODEL_8B).is_dir():
        raise FileNotFoundError(f"Model not found: {MODEL_8B}")
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(
        model=MODEL_8B,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,   # conservative for sanity check
        max_model_len=512,
        tensor_parallel_size=1,
        enforce_eager=True,           # skip CUDA graph for faster startup
    )
    params = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(["What is 1 + 1?"], params)
    answer = outputs[0].outputs[0].text.strip()
    elapsed = time.time() - t0
    del llm
    torch.cuda.empty_cache()
    return f'"{answer}" ({elapsed:.1f}s)'

check("vLLM 8B inference (1-shot)", _vllm_inference)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if failures:
    print(f"\n[{FAIL}] {len(failures)} check(s) failed:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"\n[{PASS}] All checks passed, ready to run experiments!")
    sys.exit(0)
