# Legacy / Archived Code

This directory holds code that is **not part of the final thesis pipeline** but is kept
working (not deleted) in case it's useful later. None of it produces or feeds any number
reported in the thesis. See the top-level [README.md](../README.md) for what the active
pipeline (three benchmarks, two models, four estimators) actually is.

## Why each file is here

**Excluded benchmark: AgentBench-OS** (bash/Docker interaction, evaluated during
experimentation, then dropped from the final thesis scope in favor of AgentBench-DB only):
- `bash_server.py`: MCP stdio server exposing the sandboxed `bash` tool, used only by
  AgentBench-OS.
- `Dockerfile.bash`: sandbox image for `bash_server.py`.

(AgentBench-OS's *agent* code itself was **not** moved here. It's still live inline in
[`experiments/agentbench_agent.py`](../experiments/agentbench_agent.py) behind
`--env-type os`, since the file wasn't worth physically splitting; see the module
docstring there for the scope note.)

**Excluded benchmark: ToolBench** (REST-API tool use via a mock server, evaluated during
experimentation, then dropped from the final thesis scope entirely):
- `toolbench_agent.py`: full ToolBench G1/G2 agent, moved from `experiments/`.
- `setup_toolbench.sh`: clones ToolBench and launches the mock REST server.

**Never actually used to produce a thesis result.** Per the author's confirmation, final
experiments were run via direct `python -m experiments.<benchmark>_agent` CLI
invocations, and evaluation went through the notebooks, not these scripts:
- `run_all_experiments.py`: a combined experiment-runner/evaluation script. Its
  evaluation path also computes every method's metrics via `evaluate_step_uq` (heuristic
  step-level labels), which the thesis methodology never reports. One more reason this
  was never the real pipeline.
- `reassess_task_success.py`: a one-off AgentBench-DB task-success re-evaluation/bugfix
  utility. Whatever correction it made has already been applied to the results it touched.
- `01_baseline_replication.ipynb`: an early single-experiment deep-dive notebook,
  superseded by the (also-unused, see below) proper-split version.

**Ad-hoc dev smoke tests and the pytest suite.** Per the author's confirmation, these were
never part of the real workflow used to produce thesis results:
- `tests/`: pytest suite (`test_annotation.py`, `test_calculator.py`, `test_evaluate.py`,
  `test_gsm8k_parsing.py`) and `pytest.ini`. Run with `pytest` from inside `legacy/`.
- `local_test/`: manual CPU/TinyLlama smoke-test scripts
  (`test_local_basic.py`, `test_local_gsm8k.py`, `test_local_gsm8k_simple.py`).
- `test_agentbench_setup.py`: AgentBench dual-mode (OS + DB) setup smoke test.

## Running archived code

Everything here still imports and runs (verified at archive time). `toolbench_agent.py`
and `run_all_experiments.py` import from the project root (`config`, `uq`, `experiments`),
so run them from the repo root:

```bash
python -m legacy.run_all_experiments --dry-run
python -m legacy.toolbench_agent --split G1 --n-samples 10
pytest legacy/
```
