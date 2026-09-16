# Calibration Workflow Guide (50/50 Calib/Test Split)

## Overview

The calibration workflow uses a **50/50 split** to properly calibrate step-wise
uncertainty estimates. This is the split used for the thesis's final reported results:

- **Calib (50%):** Collect raw uncertainty scores and fit normaliser parameters
- **Test (50%):** Apply fitted parameters and evaluate metrics (ECE, AUROC, etc.)

This guide covers what happens, why it matters, and how to use it. Examples below use
GSM8K (N=500) and AgentBench-DB (N=360), the final thesis benchmarks. AgentBench-OS is
shown for reference only; it was evaluated during experimentation but dropped from the
final thesis scope (see `legacy/README.md`).

---

## Quick Start: Run Full Pipeline

### GSM8K (500 samples)

```bash
python -m experiments.gsm8k_agent \
  --n-samples 500 \
  --split-phase full \
  --output-dir results/gsm8k_8B_500_split_50-50 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --cocoa-m 5 --branch-m 5 --temperature 0.7 --calib-ratio 0.5
```

**Takes:** several hours on GPU

**Output structure:**
```
results/gsm8k_8B_500_split_50-50/
├── calib/
│   ├── trajectories.jsonl       # 250 samples (used to fit normalizer)
│   └── calib_params.json        # Fitted q98, u_min
└── test/
    └── trajectories.jsonl       # 250 samples (CALIBRATED C*, for evaluation)
```

### AgentBench-DB (360 tasks, final thesis benchmark)

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

**Output structure:**
```
results/agentbench_db_8B_full_split_50-50/
├── calib/
│   ├── trajectories.jsonl       # 180 tasks (used to fit normalizer)
│   └── calib_params.json        # Fitted q98, u_min
└── test/
    └── trajectories.jsonl       # 180 tasks (CALIBRATED C*, for evaluation)
```

### AgentBench-OS (26 dev tasks, excluded from the final thesis, kept for reference)

```bash
python -m experiments.agentbench_agent \
  --env-type os \
  --n-samples 26 \
  --split-phase full \
  --output-dir results/agentbench_os \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --calib-ratio 0.5
```

---

## What Happens in Each Phase

### Phase 1: Calibration (50%)

**Goal:** Fit normaliser parameters (q98, u_min) from raw uncertainty scores

**What happens:**
1. Load dataset with fixed seed 42
2. Run the first `--calib-ratio` (default 0.5) of samples through the agent
3. At each step, compute:
   - `c_star_cocoa`: Raw uncertainty (uncalibrated)
   - `branch_consistency`: Semantic consistency (already [0,1], sigmoid-normalized NLI logits)
   - `is_correct`: heuristic step-level correctness annotation (qualitative illustration
     only, never used for a reported metric; see `uq/annotation.py`)
4. Collect ALL raw u_cocoa scores
5. After the calib phase completes:
   ```python
   cocoa.fit_normalisation()  # Compute q98 (98th percentile) and u_min
   ```

**Output:**
- `calib/trajectories.jsonl`: 250 GSM8K / 13 OS / 180 DB samples (at N=500/26/360)
- `calib/calib_params.json`: Fitted parameters
  ```json
  {
    "q98": 2.3456,              # 98th percentile of raw u_cocoa
    "u_min": 0.0123,            # Minimum of clipped scores
    "n_raw_scores": 1250,       # Number of step-level scores collected
    "split_phase": "calib",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "cocoa_m": 5,
    "temperature": 0.7
  }
  ```

**C* formula (calib phase):**
```
u_raw = u_cocoa  (unbounded, from logprobs + consistency)
u_clipped = u_raw / (|u_raw| + 1)  [soft sigmoid, poor calibration]
C* = 1 - u_clipped ∈ [0, 1)
```

**Use for:** Parameter fitting only (do NOT evaluate on this)

### Phase 2: Test (50%)

**Goal:** Evaluate calibration quality on held-out test set

**What happens:**
1. Load dataset with same seed 42 (same shuffle)
2. Load fitted parameters from `calib/calib_params.json`
3. Run the remaining `(1 - calib_ratio)` of samples through the agent
4. At each step, apply the **calibrated formula**:
   ```
   u_clipped = min(u_raw, q98)                    # Hard clip at quantile
   normalised = (u_clipped - u_min) / (q98 - u_min)
   C* = 1 - normalised ∈ [0, 1]
   ```
5. Collect results with calibrated C* scores

**Output:**
- `test/trajectories.jsonl`: 250 GSM8K / 13 OS / 180 DB samples
- Each step has **CALIBRATED** C* in [0, 1]

**C* formula (test phase):**
```
Fitted from calib: q98 = 2.3456, u_min = 0.0123
u_clipped = min(u_raw, 2.3456)
normalised = (u_clipped - 0.0123) / (2.3456 - 0.0123)
C* = 1 - normalised ∈ [0, 1]
```

**Use for:** Evaluation only (this is your evaluation set!)

---

## Metrics in Each Phase

### Computed During Experiment

For **every step** in **every sample**:

| Metric | Calib | Test | Format | Notes |
|--------|-------|------|--------|-------|
| `branch_consistency` | ✅ | ✅ | [0, 1] | Semantic NLI-based, already normalized |
| `c_star_cocoa` | ✅ uncal | ✅ **cal** | [0,1) / [0,1] | Online approx / Calibrated |
| `is_correct` | ✅ | ✅ | bool | Heuristic step-level label, diagnostic/illustration only |
| `consistency_scores` | ✅ | ✅ | dict | Raw u_belief, u_cons, u_cocoa |

### Computed Post-Hoc (During Analysis)

After collecting test trajectories, compute trajectory-level metrics via
`notebooks/comprehensive_results_analysis.ipynb` (the thesis's reported numbers) or the
diagnostic tool `scripts/analyze_baseline.py`:

| Metric | Computation | Purpose |
|--------|-----------|---------|
| **ACC** | Trajectory-level accuracy (task_success) | Overall task success |
| **ECE** | Binning + calibration error | Measure calibration quality |
| **AUROC** | ROC curve from aggregated C* vs task_success | Discrimination ability |
| **SelAcc@0.8 / Cov@0.8** | Accuracy / coverage at C* > 0.8 | Coverage vs accuracy tradeoff |
| **TR / PPV / FER** | Snowball early-warning detection | Trigger rate / precision / false escalation rate |

---

## Example: Full GSM8K Run

### Run the pipeline

```bash
python -m experiments.gsm8k_agent \
  --n-samples 500 \
  --split-phase full \
  --output-dir results/gsm8k_8B_500_split_50-50 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --cocoa-m 5 --branch-m 5 --temperature 0.7 --calib-ratio 0.5
```

**Log output:**
```
INFO - ================================================================================
INFO - PHASE 1: CALIBRATION (50% of 500 samples)
INFO - ================================================================================
INFO - Loaded 500 examples (with difficulty labels).
INFO - Calib phase: running 250 samples (0-50%)
INFO - Loading model: meta-llama/Llama-3.1-8B-Instruct
...
INFO - GSM8K-CALIB done. Task accuracy: 0.723
INFO - Fitting normalisation parameters on 1250 raw scores...
INFO - Saved calibration parameters to results/gsm8k_8B_500_split_50-50/calib/calib_params.json
INFO - Phase 1 complete: 250 samples processed

INFO - ================================================================================
INFO - PHASE 2: TEST (50% of 500 samples)
INFO - Using calibration params from: results/gsm8k_8B_500_split_50-50/calib/calib_params.json
INFO - Loaded calibration parameters: q98=2.3456, u_min=0.0123
...
INFO - GSM8K-TEST done. Task accuracy: 0.730
INFO - Phase 2 complete: 250 samples processed

INFO - ================================================================================
INFO - CALIBRATION → TEST PIPELINE COMPLETE
INFO - ================================================================================
INFO - Calib phase (50%):  250/250 samples
INFO - Test phase  (50%):  250/250 samples (CALIBRATED)
INFO - Results location:
INFO -   Calib (used for fitting):  results/gsm8k_8B_500_split_50-50/calib/trajectories.jsonl
INFO -   Calib params:              results/gsm8k_8B_500_split_50-50/calib/calib_params.json
INFO -   Test (for evaluation):     results/gsm8k_8B_500_split_50-50/test/trajectories.jsonl ← USE THIS
```

### Evaluate test phase (diagnostic tool)

```bash
python scripts/analyze_baseline.py \
  --results-dir results/gsm8k_8B_500_split_50-50/test \
  --save-plots
```

This prints a step-level diagnostic table plus trajectory-level min/mean comparison and
saves reliability/selective-accuracy plots. For the thesis's actual reported numbers, use
`notebooks/comprehensive_results_analysis.ipynb`, which evaluates trajectory-level
metrics (never step-level heuristic labels) across all benchmarks/models.

---

## Individual Phase Runs

If you want to run phases separately:

### Calib phase only

```bash
python -m experiments.gsm8k_agent \
  --n-samples 500 \
  --split-phase calib \
  --output-dir results/gsm8k/calib \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --cocoa-m 5 --branch-m 5 --temperature 0.7 --calib-ratio 0.5
```

**Output:**
- `results/gsm8k/calib/trajectories.jsonl` (250 samples)
- `results/gsm8k/calib/calib_params.json` (fitted parameters)

### Test phase (using pre-fitted params)

```bash
python -m experiments.gsm8k_agent \
  --n-samples 500 \
  --split-phase test \
  --output-dir results/gsm8k/test \
  --calib-params-path results/gsm8k/calib/calib_params.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --cocoa-m 5 --branch-m 5 --temperature 0.7 --calib-ratio 0.5
```

**Output:**
- `results/gsm8k/test/trajectories.jsonl` (250 samples with calibrated C*)

---

## Why This Design?

### Why 50/50 and not 80/20?

**50/50 was chosen for the final thesis methodology:**
- Gives a large enough test set for reliable trajectory-level metrics at the smaller
  benchmark sizes (e.g. AgentBench-DB, N=360 → 180 test samples)
- Symmetric split simplifies cross-benchmark comparison
- An 80/20 split was used during early development/exploration (the `calib_ratio`
  parameter defaults to 0.5 but remains configurable if you want to reproduce that)

### Why separate calib and test sets?

**Data leakage prevention:**
- If you fit and evaluate on the same data, parameters overfit
- You get optimistically biased ECE (appears better than reality)
- Separate sets ensure unbiased evaluation

Example:
```
❌ WRONG: Fit normalizer on set A, evaluate on set A
   → ECE = 0.015 (biased, too good!)

✅ RIGHT: Fit normalizer on set A (calib), evaluate on set B (test)
   → ECE = 0.042 (unbiased, realistic)
```

### Why keep BranchConsistency as-is (no quantile-clip calibration)?

**It's already [0, 1] normalized:**
- Computed as `sigmoid(mean(nli_logits))`
- Probabilities don't need the quantile-clip + min-max normalization that MSP/StepCoCoA use
- A quantile-clip normalized variant ("BranchConsistency-raw") was evaluated as an ablation
  during experimentation but is excluded from the thesis entirely

---

## FAQ

### Q: Can I run just the test phase without calib?

**A:** No, you need calib first to fit parameters. But you can rerun test:

```bash
# First time: run full pipeline
python -m experiments.gsm8k_agent --n-samples 500 --split-phase full --calib-ratio 0.5 ...

# Later: rerun just test
python -m experiments.gsm8k_agent --n-samples 500 --split-phase test \
  --calib-params-path results/gsm8k/calib/calib_params.json ...
```

### Q: What if I want to use different models in calib vs test?

**A:** Not recommended. The normalizer is fitted for specific model outputs. Changing the model makes the fitted parameters invalid.

**Correct approach:**
```bash
# ✅ Same model in both phases
--model meta-llama/Llama-3.1-8B-Instruct  # in both calib and test
```

**Wrong approach:**
```bash
# ❌ Different models
# Calib: --model meta-llama/Llama-3.1-8B-Instruct
# Test:  --model meta-llama/Llama-3.1-70B-Instruct  ← Will break!
```

### Q: Can I change cocoa_m or temperature between phases?

**A:** No, keep them constant:

```bash
# ✅ CORRECT: Same hyperparameters
--cocoa-m 5 --temperature 0.7  # in both phases

# ❌ WRONG: Different hyperparameters
# Calib: --cocoa-m 5 --temperature 0.7
# Test:  --cocoa-m 3 --temperature 0.8  ← Parameters now invalid!
```

### Q: Can I use a different calib/test ratio?

**A:** Yes, via `--calib-ratio` (default 0.5, matching the thesis's final methodology):

```bash
--calib-ratio 0.5   # thesis default
--calib-ratio 0.8   # used during early development/exploration
```

### Q: What is "calib_params.json" and can I reuse it?

**A:** Yes! The parameters are model/hyperparameter specific but dataset-agnostic:

```json
{
  "q98": 2.3456,
  "u_min": 0.0123,
  "n_raw_scores": 1250,
  "split_phase": "calib",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "cocoa_m": 5,
  "temperature": 0.7
}
```

You can reuse these params for other datasets with the **same model and hyperparameters**:

```bash
# Run test on a different dataset
python -m experiments.gsm8k_agent \
  --n-samples 1000 \
  --split-phase test \
  --calib-params-path results/gsm8k/calib/calib_params.json \
  ...
```

---

## Summary Table

| Aspect | Calib (50%) | Test (50%) |
|--------|-----------|-----------|
| **Purpose** | Fit normalizer | Evaluate |
| **Samples (at N=500/360)** | 250 GSM8K, 180 DB | 250 GSM8K, 180 DB |
| **C* formula** | Online (uncalibrated) | **Calibrated** |
| **Evaluate?** | ❌ No | ✅ **Yes** |
| **Use case** | Parameter fitting | Report metrics |

---

## Next Steps

1. **Run full pipeline:** `--split-phase full --calib-ratio 0.5`
2. **Wait for completion**
3. **Evaluate test set:** for the thesis's reported numbers, use
   `notebooks/comprehensive_results_analysis.ipynb`; for a quick single-run diagnostic,
   `scripts/analyze_baseline.py --results-dir <output_dir>/test`
4. **Report metrics:** ACC, ECE, AUROC, SelAcc@0.8, Cov@0.8, TR, PPV, FER from the test phase
