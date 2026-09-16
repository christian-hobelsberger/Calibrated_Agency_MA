# Quick Start: Calibration/Test Split Commands

## Full Pipeline Commands (Recommended)

Run **calib → test** sequentially in one command. Examples use GSM8K, AgentBench-DB, and
HotpotQA (the three final thesis benchmarks, N=500/360/500, 50/50 split).

### GSM8K (500 samples, 50/50 split = 250/250)

```bash
python -m experiments.gsm8k_agent \
  --n-samples 500 \
  --split-phase full \
  --output-dir results/gsm8k_8B_500_split_50-50 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --cocoa-m 5 --branch-m 5 --temperature 0.7 --calib-ratio 0.5
```

**Output:**
```
results/gsm8k_8B_500_split_50-50/
├── calib/
│   ├── trajectories.jsonl       # 250 samples
│   └── calib_params.json        # Fitted q98, u_min
└── test/
    └── trajectories.jsonl       # 250 samples (CALIBRATED) ← EVALUATE THIS
```

### AgentBench-DB (360 tasks, 50/50 split = 180/180, final thesis benchmark)

```bash
python -m experiments.agentbench_agent \
  --env-type db \
  --n-samples 360 \
  --split-phase full \
  --dataset-mode full \
  --output-dir results/agentbench_db_8B_full_split_50-50 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --calib-ratio 0.5
```

### HotpotQA (500 samples, 50/50 split = 250/250, final thesis benchmark)

```bash
python -m experiments.hotpotqa_agent \
  --n-samples 500 \
  --split-phase full \
  --output-dir results/hotpotqa_8B_500_full_split_50-50 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --cocoa-m 5 --branch-m 5 --temperature 0.7 --calib-ratio 0.5
```

---

## Evaluate Test Phase

Run `notebooks/comprehensive_results_analysis.ipynb`. It picks up every `results/<run>/test/`
directory listed in its `EXPERIMENTS` config cell and produces the thesis's actual reported
numbers: trajectory-level ACC, ECE, AUROC, SelAcc@0.8, Cov@0.8, snowball detection (TR/PPV/FER),
reliability diagrams, selective-prediction curves, and the LaTeX/CSV summary tables, across all
benchmarks and both model scales in one pass.

Add a new run to the notebook by adding its output directory (e.g.
`results/hotpotqa_8B_500_full_split_50-50`) to the `EXPERIMENTS` dict near the top, then
re-run the notebook top to bottom.

---

## Individual Phase Runs (if needed)

### Run calib only

```bash
python -m experiments.gsm8k_agent \
  --n-samples 500 \
  --split-phase calib \
  --output-dir results/gsm8k/calib \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --cocoa-m 5 --branch-m 5 --temperature 0.7 --calib-ratio 0.5
```

### Run test only (with pre-fitted params)

```bash
python -m experiments.gsm8k_agent \
  --n-samples 500 \
  --split-phase test \
  --output-dir results/gsm8k/test \
  --calib-params-path results/gsm8k/calib/calib_params.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --cocoa-m 5 --branch-m 5 --temperature 0.7 --calib-ratio 0.5
```

---

## Key Points

✅ **Do:**
- Use `--split-phase full` to orchestrate automatically
- Keep model and hyperparameters the same in both phases
- Evaluate ONLY the test phase
- Use fixed seed 42 (default, ensures reproducibility)
- Use `--calib-ratio 0.5` to match the thesis's final methodology

❌ **Don't:**
- Evaluate the calib phase
- Change model between calib and test
- Change `--cocoa-m` or `--temperature` between phases
- Use `--no-resume` unless starting fresh

---

## Expected Output

```
================================================================================
CALIBRATION → TEST PIPELINE COMPLETE
================================================================================
Calib phase (50%):  250/250 samples
Test phase  (50%):  250/250 samples (CALIBRATED)

Results location:
  Calib (used for fitting):  results/gsm8k/calib/trajectories.jsonl
  Calib params:              results/gsm8k/calib/calib_params.json
  Test (for evaluation):     results/gsm8k/test/trajectories.jsonl ← USE THIS
```

---

## Split Summary

| Benchmark | Total | Calib (50%) | Test (50%) | In final thesis? |
|-----------|-------|-----------|-----------|------|
| GSM8K | 500 | 250 | 250 | ✅ Yes |
| AgentBench-DB | 360 | 180 | 180 | ✅ Yes |
| HotpotQA | 500 | 250 | 250 | ✅ Yes |

See `CALIBRATION_WORKFLOW.md` for detailed documentation.
