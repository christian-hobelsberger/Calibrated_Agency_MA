"""
AgentBench OS + DB Experiment with dual execution modes and Calibration/Test split.

SCOPE NOTE: only --env-type db (AgentBench-DB, SQL interaction via SQLite) is used in the
final thesis results. --env-type os (bash/Docker interaction) was evaluated during
experimentation but dropped from the final thesis scope. Both modes are kept in this single
file (not physically split) since the OS code is still functional and self-contained behind
--env-type; see legacy/README.md for the supporting OS-only tooling (tools/bash_server.py,
docker/Dockerfile.bash) that has been archived.

Supports two execution strategies:
1. Python API (default) - Subprocess-based task execution, no containers needed
2. Docker/Enroot (fallback) - Full containerization for higher fidelity; OS mode only

The Python API mode works immediately on any system (including LRZ login nodes).
Docker mode is available as an option for future full-fidelity OS runs.

Split Workflow (default 50/50, configurable via --calib-ratio):
  Phase 1 (calib, --calib-ratio):   Generate trajectories, fit normalisation parameters
  Phase 2 (test, remainder):    Generate trajectories with calibrated C* scores
  All (none):             Legacy mode: generate all samples without splitting

Usage (final thesis benchmark, AgentBench-DB):
    python -m experiments.agentbench_agent --env-type db --n-samples 360 --split-phase full

Usage (AgentBench-OS, excluded from final thesis, kept working for reference):
    python -m experiments.agentbench_agent --env-type os --n-samples 26 --split-phase train
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

OS_SYSTEM_PROMPT = """\
You are an expert Linux system administrator.

CRITICAL RULES - READ CAREFULLY:

1. Each response must be ONLY ONE of these:
   - A single Thought (1-2 sentences max)
   - A single Action (just the command, no extra text)
   - FINAL ANSWER (when you have solved the task)
2. Do NOT output multiple steps in one response.
3. Do NOT mix Thought and Action in the same response.
4. Stop immediately after outputting one step.

STEP FORMAT (pick ONE per response):

THOUGHT FORMAT:
Thought: <1-2 sentence reasoning about what to do next>

ACTION FORMAT:
Action: bash("<your_command>")

FINAL FORMAT:
FINAL ANSWER: <description of what was accomplished>

RULES FOR COMMANDS:
- One command per step — no chaining with && or ;
- Always verify before modifying: use ls, cat, grep first
- If command fails, read error and try corrected version

EXAMPLE OF CORRECT SEQUENCE:
Action: bash("find /home -name 'target.txt'")
Observation: /home/user/target.txt
Thought: Found it. Now reading the contents.
Action: bash("cat /home/user/target.txt")
Observation: Hello World
FINAL ANSWER: The file contains "Hello World".\
"""

DB_SYSTEM_PROMPT = """\
You are an expert database analyst operating on a SQLite database.

CRITICAL RULES - READ CAREFULLY:

1. Each response must be ONLY ONE of:
   - A single Thought (1-2 sentences max)
   - A single Action (just the call, no extra text)
   - FINAL ANSWER (when you have the result)
2. Do NOT output multiple steps in one response.
3. Do NOT mix Thought and Action in the same response.
4. Stop immediately after outputting one step.
5. Your FIRST response must ALWAYS be: Action: list_tables()
6. NEVER skip steps — always call list_tables() then describe_table() before writing SQL.
7. If you have tried 3 queries and still get 0 rows, output FINAL ANSWER: None.

STEP FORMAT (pick ONE per response):

THOUGHT FORMAT:
Thought: <1-2 sentence reasoning about what to do next>

ACTION FORMAT:
Action: list_tables()
Action: describe_table("table_name")
Action: sql_query("SELECT ...")            ← for SELECT questions only
Action: sql_execute("INSERT/UPDATE/DELETE ...")  ← for write operations only

FINAL FORMAT:
FINAL ANSWER: <value>    ← for SELECT tasks
FINAL ANSWER: Done       ← for INSERT/UPDATE tasks

TASK TYPE — decide before acting:
  Question ("what is", "how many", "which", "find") → sql_query(), then FINAL ANSWER: <value>
  Insert ("add", "insert", "record a new row")      → sql_execute("INSERT ..."), then FINAL ANSWER: Done
  Modify ("update", "change", "set X to Y")         → sql_execute("UPDATE ..."), then FINAL ANSWER: Done

EXAMPLE OF CORRECT MULTI-STEP SEQUENCE:

[Model Response 1]
Action: list_tables()

[Model Response 2]
Action: describe_table("Game Schedule")

[Model Response 3]
Action: sql_query("SELECT `Game` FROM `Game Schedule` WHERE `Opponent` = 'Arsenal FC'")

[Model Response 4]
FINAL ANSWER: 6

EXAMPLE OF WRONG BEHAVIOR (DO NOT DO THIS):
"Action: list_tables() Action: describe_table("Game Schedule") Action: sql_query("SELECT ...") FINAL ANSWER: 6"
^ This is WRONG - multiple steps in one response

MANDATORY DATABASE RULES:
- Copy the table name CHARACTER-BY-CHARACTER from list_tables() into describe_table().
  list_tables() → "OK: Military Leaders Table"
  CORRECT: describe_table("Military Leaders Table")
  WRONG:   describe_table("Military Leaders")   ← NEVER truncate!
- Column names must EXACTLY match describe_table() output (case-sensitive).
- Use backticks for names with spaces: `My Table`, `My Column`
- All values stored as TEXT — always use single quotes: WHERE `Year` = '2015'
- describe_table() shows SAMPLE ROWS — copy value formats EXACTLY.
  Sample shows Date='March 8, 2007' → use WHERE `Date` = 'March 8, 2007'
- If query returns 0 rows → retry with LIKE '%value%'\
"""


# ---------------------------------------------------------------------------
# Task data
# ---------------------------------------------------------------------------

@dataclass
class AgentBenchTask:
    task_id: str
    env_type: str  # "os" or "db"
    instruction: str
    gold_answer: Optional[Union[str, list[str]]]  # String (OS) or list (DB)
    initial_state: dict


# ---------------------------------------------------------------------------
# Abstract environment interface
# ---------------------------------------------------------------------------

class TaskEnvironment(ABC):
    """Abstract base class for task execution environments."""

    def __init__(self, env_type: str, task: AgentBenchTask):
        self.env_type = env_type
        self.task = task
        self.timeout_s = DEFAULT_CONFIG.experiment.docker_timeout_s

    @abstractmethod
    def start(self) -> None:
        """Initialize the environment."""
        pass

    @abstractmethod
    def execute(self, command: str) -> str:
        """Execute a command and return output."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Clean up the environment."""
        pass


# ---------------------------------------------------------------------------
# Python API Environment (Default, No Containers)
# ---------------------------------------------------------------------------

class PythonAPIEnvironment(TaskEnvironment):
    """Execute tasks using Python subprocess without containers.

    Works on any system: LRZ login nodes, local machines, etc.
    Uses direct bash/sqlite execution rather than containers.
    """

    def __init__(self, env_type: str, task: AgentBenchTask):
        super().__init__(env_type, task)
        self.db_conn = None
        self.db_path = None

    def start(self) -> None:
        """Initialize execution environment."""
        logger.debug("Starting Python API environment for %s task: %s",
                     self.env_type, self.task.task_id)

        if self.env_type == "db":
            # Create in-memory SQLite database
            self._init_database()
            # Populate schema and pre-existing rows
            self._apply_initial_db_state()

    def _init_database(self) -> None:
        """Initialize SQLite database for DB tasks."""
        self.db_conn = sqlite3.connect(":memory:")
        self.db_conn.row_factory = sqlite3.Row
        logger.debug("Initialized in-memory SQLite database")

    def _apply_initial_db_state(self) -> None:
        """Execute CREATE TABLE and INSERT statements to set up the initial DB.

        Prefers the new 'schema_sqls' list (one statement each) over the legacy
        single 'schema' string so that row data is properly populated.
        """
        state = self.task.initial_state
        if "schema_sqls" in state:
            for sql in state["schema_sqls"]:
                if not sql or not sql.strip():
                    continue
                try:
                    self.db_conn.execute(sql)
                except Exception as exc:
                    logger.warning("Schema SQL skipped (%s): %.120s", exc, sql)
            self.db_conn.commit()
            logger.debug(
                "Applied %d schema SQL statements for task %s",
                len(state["schema_sqls"]),
                self.task.task_id,
            )
        elif "schema" in state:
            # Legacy single-string fallback (CREATE TABLE only, no row data)
            self.execute(state["schema"])

    def execute(self, command: str) -> str:
        """Execute bash or SQL command."""
        if self.env_type == "os":
            return self._execute_bash(command)
        elif self.env_type == "db":
            return self._execute_sql(command)
        else:
            return f"ERROR: Unknown env_type {self.env_type}"

    def _execute_bash(self, command: str) -> str:
        """Execute bash command safely."""
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                cwd="/tmp",  # Run in /tmp for safety
            )
            output = result.stdout + result.stderr
            return output[:2000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {self.timeout_s}s"
        except Exception as exc:
            return f"ERROR: {exc}"

    def _execute_sql(self, command: str) -> str:
        """Execute SQL command with helpful error messages."""
        if not self.db_conn:
            return "ERROR: Database not initialized"

        try:
            # Handle special commands
            if command.strip().lower() == "list_tables()":
                cursor = self.db_conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in cursor.fetchall()]
                if tables:
                    return "OK: " + ", ".join(tables)
                else:
                    return "(no tables found)"

            if command.strip().lower().startswith("describe_table("):
                # Use regex to extract the table name so that names containing
                # parentheses (e.g. "IME Exchange (Including ...)") are handled
                # correctly.  The naive split("(") approach truncates at the first
                # inner parenthesis.
                _dt_inner = command[command.index("(") + 1 : command.rindex(")")]
                _dt_m = re.match(r'\s*(["\'])(.*?)\1\s*$', _dt_inner, re.DOTALL)
                if _dt_m:
                    table_name = _dt_m.group(2)
                else:
                    table_name = _dt_inner.strip().strip("\"'")
                # Clean up any residual escape sequences the LLM might have produced
                table_name = (table_name.replace('\\"', '"')
                                        .replace("\\'", "'")
                                        .replace("\\`", ""))
                cursor = self.db_conn.cursor()
                # Use backticks to handle table names with spaces
                cursor.execute(f"PRAGMA table_info(`{table_name}`)")
                columns = cursor.fetchall()
                actual_table = table_name  # track which name actually worked
                note = ""
                if not columns:
                    # Fuzzy fallback: find a table whose name contains or starts with
                    # what the agent provided (handles truncation like "Military Leaders"
                    # when actual name is "Military Leaders Table").
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    actual_tables = [row[0] for row in cursor.fetchall()]
                    matched = None
                    tname_lower = table_name.lower()
                    for actual in actual_tables:
                        if (actual.lower().startswith(tname_lower)
                                or tname_lower in actual.lower()):
                            matched = actual
                            break
                    if matched:
                        cursor.execute(f"PRAGMA table_info(`{matched}`)")
                        columns = cursor.fetchall()
                        actual_table = matched
                        note = (f"  (WARNING: '{table_name}' not found. "
                                f"Using closest match '{matched}'; "
                                f"use this EXACT name in all queries)\n")
                    else:
                        avail = ", ".join(f"'{t}'" for t in actual_tables[:5])
                        return (f"ERROR: Table '{table_name}' not found. "
                                f"Available tables: {avail}. "
                                f"Call list_tables() to see all tables.")
                result = "OK: Column schema:\n" + note
                result += "\n".join(f"  {col[1]}: {col[2]}" for col in columns)
                # Append sample rows so the agent can see exact value formats
                # (dates, names, numbers) before writing WHERE clauses.
                try:
                    cursor.execute(f"SELECT * FROM `{actual_table}` LIMIT 3")
                    sample_rows = cursor.fetchall()
                    if sample_rows:
                        col_names = [col[1] for col in columns]
                        result += "\n\nSample rows (use these EXACT value formats in WHERE clauses):"
                        for row in sample_rows:
                            row_dict = {col_names[i]: str(row[i]) for i in range(min(len(col_names), len(row)))}
                            # Limit each value to 40 chars to keep output compact
                            compact = {k: v[:40] for k, v in row_dict.items()}
                            result += "\n  " + ", ".join(f"{k}='{v}'" for k, v in compact.items())
                except Exception:
                    pass  # sample rows are best-effort
                return result

            # Execute SQL query or update.
            # Unescape sequences the LLM sometimes produces from the prompt examples:
            #   \" → "    \' → '    \` → `  (backticks must NOT have a leading backslash)
            command = (command.replace('\\"', '"')
                              .replace("\\'", "'")
                              .replace("\\`", "`"))
            cursor = self.db_conn.cursor()

            _cmd_upper = command.strip().upper()
            if _cmd_upper.startswith("SELECT") or _cmd_upper.startswith("WITH"):
                cursor.execute(command)
                rows = cursor.fetchall()
                if not rows:
                    # Give the agent a hint: show 3 rows from the table so it can
                    # see what value formats are actually stored (dates, names, etc.)
                    hint = ""
                    try:
                        # Extract table name from the FROM clause.
                        # Try quoted forms first (backtick or double-quote) so that
                        # names containing commas, dashes, parentheses etc. are
                        # captured in full.  Fall back to unquoted word-chars.
                        _from_m = (
                            re.search(r'\bFROM\s+`([^`]+)`', command, re.IGNORECASE)
                            or re.search(r'\bFROM\s+"([^"]+)"', command, re.IGNORECASE)
                            or re.search(
                                r'\bFROM\s+(\w[\w\s]*?)(?:\s+WHERE|\s+ORDER|\s+LIMIT|\s+GROUP|\s*$)',
                                command, re.IGNORECASE,
                            )
                        )
                        m = _from_m  # keep variable name consistent below
                        if m:
                            tbl_hint = m.group(1).strip()
                            cur2 = self.db_conn.cursor()
                            cur2.execute(f"SELECT * FROM `{tbl_hint}` LIMIT 3")
                            sample = cur2.fetchall()
                            if sample:
                                cols_cur = self.db_conn.cursor()
                                cols_cur.execute(f"PRAGMA table_info(`{tbl_hint}`)")
                                col_names = [c[1] for c in cols_cur.fetchall()]
                                hint_rows = []
                                for r in sample:
                                    row_str = ", ".join(
                                        f"{col_names[i]}='{str(r[i])[:30]}'"
                                        for i in range(min(len(col_names), len(r)))
                                    )
                                    hint_rows.append("  " + row_str)
                                hint = ("\nHint: Table has data. Sample rows:\n"
                                        + "\n".join(hint_rows)
                                        + "\nAdjust your WHERE clause to match these exact value formats.")
                    except Exception:
                        pass
                    return "OK: (query executed but returned 0 rows)" + hint
                # Format results
                result_lines = ["OK: Query returned results:"]
                for row in rows[:100]:  # Limit output
                    result_lines.append("  " + str(dict(row)))
                if len(rows) > 100:
                    result_lines.append(f"  ... ({len(rows) - 100} more rows)")
                return "\n".join(result_lines)
            else:
                # INSERT, UPDATE, DELETE
                cursor.execute(command)
                self.db_conn.commit()
                return f"OK: Statement executed. {cursor.rowcount} rows affected"

        except sqlite3.OperationalError as exc:
            error_msg = str(exc).lower()
            if "no such table" in error_msg:
                # Show which tables actually exist so the agent can self-correct
                try:
                    cur2 = self.db_conn.cursor()
                    cur2.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    actual = [row[0] for row in cur2.fetchall()]
                    avail = ", ".join(f"'{t}'" for t in actual[:5])
                    return (f"ERROR: Table not found. "
                            f"Available tables: {avail}. "
                            f"Use list_tables() for the full list.")
                except Exception:
                    return "ERROR: Table not found. Use list_tables() to see available tables"
            elif "no such column" in error_msg:
                # Show available columns for context so the agent can self-correct
                try:
                    cur2 = self.db_conn.cursor()
                    cur2.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    tables = [row[0] for row in cur2.fetchall()]
                    hints = []
                    for tname in tables[:3]:
                        cur2.execute(f"PRAGMA table_info(`{tname}`)")
                        cols = [row[1] for row in cur2.fetchall()]
                        hints.append(f"'{tname}': {', '.join(cols[:10])}")
                    return (f"ERROR: Column not found ({exc}). "
                            f"Actual columns: {'; '.join(hints)}. "
                            f"Use describe_table() to verify exact names.")
                except Exception:
                    return "ERROR: Column not found. Use describe_table() to check column names"
            else:
                return f"ERROR: SQL error: {exc}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    def stop(self) -> None:
        """Clean up environment."""
        if self.db_conn:
            self.db_conn.close()
            logger.debug("Closed SQLite database")


# ---------------------------------------------------------------------------
# Container Environment (Docker/Enroot Fallback)
# ---------------------------------------------------------------------------

class ContainerEnvironment(TaskEnvironment):
    """Execute tasks in Docker or Enroot containers.

    Provides higher fidelity than Python API by running in isolated sandboxes.
    Requires Docker or Enroot to be available.
    """

    def __init__(self, env_type: str, task: AgentBenchTask):
        super().__init__(env_type, task)
        self.container_id: Optional[str] = None
        self.container_runtime = self._detect_runtime()

    @staticmethod
    def _detect_runtime() -> Optional[str]:
        """Detect Docker or Enroot availability."""
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, timeout=5, check=True
            )
            logger.debug("Docker detected")
            return "docker"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        try:
            subprocess.run(
                ["enroot", "list"],
                capture_output=True, text=True, timeout=5, check=True
            )
            logger.debug("Enroot detected")
            return "enroot"
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("Neither Docker nor Enroot found")
            return None

    def start(self) -> None:
        """Start the container."""
        if not self.container_runtime:
            raise RuntimeError(
                "No container runtime available (Docker or Enroot required). "
                "To use container mode, install Docker or run on an HPC system with Enroot. "
                "Use Python API mode (default) for immediate execution."
            )

        self.container_id = f"agentbench-{uuid.uuid4().hex[:8]}"

        if self.container_runtime == "docker":
            self._start_docker()
        elif self.container_runtime == "enroot":
            self._start_enroot()

        logger.debug("Started container %s", self.container_id)

    def _start_docker(self) -> None:
        """Start Docker container."""
        image = f"local-os/{self.env_type}"  # Use locally-built images
        cfg = DEFAULT_CONFIG.experiment

        result = subprocess.run(
            ["docker", "run", "-d", "--rm",
             f"--memory={cfg.docker_memory}",
             f"--cpus={cfg.docker_cpus}",
             image, "sleep", "300"],
            capture_output=True, text=True, check=True,
        )
        self.container_id = result.stdout.strip()

    def _start_enroot(self) -> None:
        """Start Enroot container."""
        cfg = DEFAULT_CONFIG.experiment

        # Create container
        create_result = subprocess.run(
            ["enroot", "create", "--name", self.container_id,
             f"docker://local-os/{self.env_type}"],
            capture_output=True, text=True, timeout=600,
        )

        if create_result.returncode != 0:
            raise RuntimeError(
                f"Failed to create Enroot container: {create_result.stderr}"
            )

        # Start container
        subprocess.run(
            ["enroot", "start", "--rw", self.container_id, "sleep", "300"],
            capture_output=True, text=True, check=True, timeout=60,
        )

    def execute(self, command: str) -> str:
        """Execute command in container."""
        if not self.container_id:
            return "ERROR: Container not started"

        try:
            if self.container_runtime == "docker":
                result = subprocess.run(
                    ["docker", "exec", self.container_id, "bash", "-c", command],
                    capture_output=True, text=True, timeout=self.timeout_s,
                )
            elif self.container_runtime == "enroot":
                result = subprocess.run(
                    ["enroot", "exec", self.container_id, "bash", "-c", command],
                    capture_output=True, text=True, timeout=self.timeout_s,
                )
            else:
                return "ERROR: Unknown runtime"

            output = result.stdout + result.stderr
            return output[:2000] if output else "(no output)"

        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {self.timeout_s}s"
        except Exception as exc:
            return f"ERROR: {exc}"

    def stop(self) -> None:
        """Stop and remove container."""
        if not self.container_id:
            return

        try:
            if self.container_runtime == "docker":
                subprocess.run(
                    ["docker", "stop", self.container_id],
                    capture_output=True, timeout=10,
                )
            elif self.container_runtime == "enroot":
                subprocess.run(
                    ["enroot", "remove", self.container_id],
                    capture_output=True, timeout=10,
                )
        except Exception as exc:
            logger.warning("Error stopping container: %s", exc)
        finally:
            self.container_id = None


# ---------------------------------------------------------------------------
# Task Loading
# ---------------------------------------------------------------------------

def load_agentbench_tasks(
    env_type: str,
    n: int = 200,
    agentbench_root: str = "AgentBench",
    dataset: str = "full",
) -> list[AgentBenchTask]:
    """Load AgentBench tasks from JSON/JSONL files.

    Parameters
    ----------
    env_type : str
        "os" or "db"
    n : int
        Number of tasks to load
    agentbench_root : str
        Path to AgentBench root directory
    dataset : str
        "full" (default): use train_0317 for OS (1000 samples), dev+standard for DB (360)
        "dev": use dev for OS (26 samples), dev+standard for DB (360)

    Returns
    -------
    list[AgentBenchTask]
        List of parsed tasks
    """
    tasks: list[AgentBenchTask] = []

    if env_type == "os":
        # OS datasets
        if dataset == "full":
            # Use train_0317/training.json (1000 samples)
            task_file = Path(agentbench_root) / "data" / "os_interaction" / "train_0317" / "training.json"
            if not task_file.exists():
                logger.warning(
                    f"train_0317 not found, falling back to dev.json. "
                    f"Expected: {task_file}"
                )
                task_file = Path(agentbench_root) / "data" / "os_interaction" / "data" / "dev.json"
        else:  # dev
            # Use dev.json (26 samples)
            task_file = Path(agentbench_root) / "data" / "os_interaction" / "data" / "dev.json"

        if not task_file.exists():
            raise FileNotFoundError(
                f"OS tasks not found: {task_file}. "
                f"Available: dev.json (26) or train_0317/training.json (1000)"
            )

        logger.info(f"Loading OS tasks from {task_file.name} (dataset={dataset})")

    elif env_type == "db":
        # DB datasets: combine dev + standard (60 + 300 = 360 total)
        base_dir = Path(agentbench_root) / "data" / "dbbench"
        dev_file = base_dir / "dev.jsonl"
        standard_file = base_dir / "standard.jsonl"

        if not dev_file.exists() or not standard_file.exists():
            raise FileNotFoundError(
                f"DB datasets not found in {base_dir}. "
                f"Expected: dev.jsonl (60) and standard.jsonl (300)"
            )

        logger.info(
            f"Loading DB tasks from dev.jsonl ({dev_file}) + standard.jsonl ({standard_file}) "
            f"(dataset={dataset}, always combined for DB)"
        )

        # Load and combine dev + standard
        all_data = []
        for file_path in [dev_file, standard_file]:
            with open(file_path, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if len(all_data) >= n:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    all_data.append(d)

        for i, d in enumerate(all_data[:n]):
            tasks.append(_parse_task(d, env_type, i))

        logger.info(f"Loaded {len(tasks)} DB tasks (dev+standard combined)")
        return tasks

    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    # Load OS tasks (single file)
    with open(task_file, encoding="utf-8") as fh:
        content = fh.read().strip()

    # Try JSON array format
    if task_file.suffix == ".json":
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for i, d in enumerate(data):
                    if i >= n:
                        break
                    tasks.append(_parse_task(d, env_type, i))
                logger.info(f"Loaded {len(tasks)} OS tasks from {task_file.name}")
                return tasks
        except json.JSONDecodeError:
            pass

    # Parse as JSONL
    for i, line in enumerate(content.split("\n")):
        if i >= n:
            break
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        tasks.append(_parse_task(d, env_type, i))

    logger.info(f"Loaded {len(tasks)} OS tasks from {task_file.name}")
    return tasks


def _generate_create_table_sql(table_info: dict) -> str:
    """Generate CREATE TABLE SQL from AgentBench table_info.

    All columns are created as TEXT to avoid type-affinity mismatches when
    values from the JSONL rows (always strings) are inserted and later compared
    with string literals in WHERE clauses.

    Parameters
    ----------
    table_info : dict
        Dict with 'table_name' and 'table_info' keys.
        table_info contains 'columns' list with 'name' and 'type' keys.

    Returns
    -------
    str
        CREATE TABLE statement.
    """
    table_name = table_info.get("table_name", "")
    table_info_data = table_info.get("table_info", {})
    columns = table_info_data.get("columns", [])

    if not table_name or not columns:
        return ""

    # Use TEXT for every column so that row values (always strings in the JSONL)
    # can be inserted and compared consistently without SQLite type-coercion issues.
    col_defs = [f'`{col.get("name", "")}` TEXT' for col in columns if col.get("name")]

    if not col_defs:
        return ""

    return f'CREATE TABLE `{table_name}` ({", ".join(col_defs)})'


def _generate_insert_rows_sql(table_info: dict) -> list[str]:
    """Generate INSERT statements for all existing rows in table_info.

    This populates the table with the data rows from the JSONL so that:
    - SELECT tasks can actually retrieve values
    - UPDATE/DELETE tasks have rows to match in WHERE clauses

    Parameters
    ----------
    table_info : dict
        Dict with 'table_name' and 'table_info' keys.

    Returns
    -------
    list[str]
        One INSERT statement per data row (empty list if no rows).
    """
    table_name = table_info.get("table_name", "")
    table_data = table_info.get("table_info", {})
    columns = table_data.get("columns", [])
    rows = table_data.get("rows", [])

    if not table_name or not columns or not rows:
        return []

    # Build a list of (original_index, name) keeping only columns with non-empty
    # names: symmetric with _generate_create_table_sql which uses `if col.get("name")`.
    # A column with an empty name cannot exist in the CREATE TABLE, so attempting
    # to INSERT into it produces "no such column named " errors that flood the logs
    # and cause every row to be skipped, leaving the table completely empty.
    valid_cols: list[tuple[int, str]] = [
        (i, col.get("name", ""))
        for i, col in enumerate(columns)
        if col.get("name", "").strip()
    ]

    if not valid_cols:
        return []

    col_indices = [i for i, _ in valid_cols]
    col_names   = [n for _, n in valid_cols]

    # Skip the header row that some datasets include as row[0]
    data_rows = rows
    if rows and [str(v) for v in rows[0]] == [col.get("name", "") for col in columns]:
        data_rows = rows[1:]

    col_defs_q = ", ".join(f"`{col}`" for col in col_names)

    insert_sqls: list[str] = []
    for row in data_rows:
        # Row must have at least enough entries to cover the highest valid column index
        if not col_indices or len(row) <= max(col_indices):
            logger.warning(
                "Skipping row for table '%s': expected ≥%d columns, got %d  row=%s",
                table_name, max(col_indices) + 1, len(row), list(row)[:6],
            )
            continue
        values = []
        for idx in col_indices:
            val = row[idx]
            if val is None:
                values.append("NULL")
            else:
                val_str = str(val).replace("'", "''")
                values.append(f"'{val_str}'")
        insert_sqls.append(
            f"INSERT INTO `{table_name}` ({col_defs_q}) VALUES ({', '.join(values)})"
        )
    return insert_sqls


def _parse_task(d: dict, env_type: str, idx: int) -> AgentBenchTask:
    """Parse task dictionary into AgentBenchTask.

    Handles multiple AgentBench data formats:
    - instruction / question (standard)
    - description (OS and DB tasks)
    - evaluation.match (AgentBench OS format)
    - label (AgentBench DB format - stores list of possible answers)
    - table (AgentBench DB format - table schema for creating database)
    """
    # Extract instruction from various possible fields
    instruction = (
        d.get("instruction")
        or d.get("question")
        or d.get("description")
        or ""
    )

    # Extract answer from various possible fields
    gold_answer = (
        d.get("answer")
        or d.get("expected_result")
        or d.get("expected")
    )

    # For AgentBench OS tasks: answer is in evaluation.match
    if not gold_answer and "evaluation" in d:
        evaluation = d["evaluation"]
        if isinstance(evaluation, dict):
            gold_answer = evaluation.get("match")

    # For AgentBench DB tasks: answer is in label field (list of possible answers)
    # Store the entire list for evaluation logic to handle
    if not gold_answer and "label" in d:
        label = d.get("label")
        # label is a list of possible correct answers for DB tasks
        if label:
            gold_answer = label if isinstance(label, list) else [label]

    # For DB tasks: generate CREATE TABLE + INSERT rows from table_info.
    # We store a list of SQL statements so that the environment can execute
    # them one by one (sqlite3 does not support multi-statement strings).
    initial_state = d.get("initial_state", {})
    if env_type == "db" and "table" in d and "schema_sqls" not in initial_state:
        create_sql = _generate_create_table_sql(d["table"])
        if create_sql:
            insert_sqls = _generate_insert_rows_sql(d["table"])
            initial_state = {
                **initial_state,
                "schema_sqls": [create_sql] + insert_sqls,
            }

    return AgentBenchTask(
        task_id=d.get("task_id", str(idx)),
        env_type=env_type,
        instruction=instruction,
        gold_answer=gold_answer,
        initial_state=initial_state,
    )


# ---------------------------------------------------------------------------
# Answer Extraction (similar to GSM8K)
# ---------------------------------------------------------------------------

def extract_final_answer(text: str) -> Optional[str]:
    """Extract FINAL ANSWER from agent response.

    Looks for "FINAL ANSWER: <answer>" format used by both OS and DB tasks.
    Falls back to "TASK COMPLETE:" format for backward compatibility.
    """
    # Try FINAL ANSWER format first
    m = re.search(r"FINAL\s+ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback to TASK COMPLETE format
    m = re.search(r"TASK\s+COMPLETE:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Action Parsing
# ---------------------------------------------------------------------------

def parse_agentbench_action(response: str, env_type: str) -> tuple[Optional[str], Optional[str]]:
    """Parse Action from ReAct response - flexible pattern matching.

    Handles variations like:
    - Action: bash("command")
    - bash("command")
    - Action: bash(command)
    - bash(command)
    - Thought: Action: bash("command")   ← strips leading 'Thought:' prefix
    """
    # Strip leading "Thought:" prefix that the model sometimes emits on action lines,
    # e.g. "Thought: Action: sql_query(...)" → "Action: sql_query(...)"
    response_clean = re.sub(r'^\s*Thought:\s*', '', response, flags=re.IGNORECASE | re.MULTILINE)

    if env_type == "os":
        # Try multiple patterns for bash actions
        patterns = [
            r'Action:\s*bash\(["\']?(.*?)["\']?\)',  # Action: bash("cmd")
            r'bash\(["\']?(.*?)["\']?\)',              # bash("cmd") without Action:
            r'Action:\s*bash\s+["\']?(.*?)["\']?\s*$', # Action: bash "cmd"
        ]
        for pattern in patterns:
            m = re.search(pattern, response_clean, re.DOTALL | re.MULTILINE)
            if m:
                cmd = m.group(1).strip()
                if cmd:  # Only return if command is non-empty
                    return ("bash", cmd)
        return (None, None)

    # DB actions: sql_query, sql_execute, list_tables, describe_table
    for fn in ["sql_query", "sql_execute", "list_tables", "describe_table"]:
        # list_tables takes no argument: match it before trying arg patterns
        if fn == "list_tables":
            if re.search(r'(?:Action:\s*)?list_tables\(\s*\)', response_clean):
                return ("list_tables", "")
            continue

        # describe_table takes a table-name argument that may contain parentheses,
        # e.g. describe_table("IME Exchange (Including spot, credit and forward transactions)")
        # Strategy: find describe_table(" then capture up to the closing ") or ')
        if fn == "describe_table":
            # Double-quoted arg: allow any chars including parentheses inside quotes
            m = re.search(
                r'(?:Action:\s*)?describe_table\(\s*"([^"]+)"\s*\)',
                response_clean
            )
            if m:
                return ("describe_table", m.group(1).strip())
            # Single-quoted arg
            m = re.search(
                r"(?:Action:\s*)?describe_table\(\s*'([^']+)'\s*\)",
                response_clean
            )
            if m:
                return ("describe_table", m.group(1).strip())
            # Bare unquoted arg (no parens allowed: old fallback for simple names)
            m = re.search(
                r'(?:Action:\s*)?describe_table\(\s*([^()"\'\s][^()]*?)\s*\)',
                response_clean
            )
            if m:
                return ("describe_table", m.group(1).strip().strip('"\''))
            continue

        # sql_query / sql_execute: use a GREEDY match so inner parens don't truncate the SQL.
        # The argument is almost always double-quoted; try double then single then bare.
        prefix = rf'(?:Action:\s*)?{fn}\('

        # Double-quoted: fn("...SQL..."): greedy (.+) finds the LAST " before closing )
        m = re.search(prefix + r'\s*"(.+)"\s*\)', response_clean, re.DOTALL)
        if m:
            return (fn, m.group(1))

        # Single-quoted: fn('...SQL...')
        m = re.search(prefix + r"\s*'(.+)'\s*\)", response_clean, re.DOTALL)
        if m:
            return (fn, m.group(1))

        # Bare (no surrounding quotes): fn(SELECT ...): take everything up to last )
        m = re.search(prefix + r'\s*([A-Z].+)\s*\)', response_clean, re.DOTALL | re.IGNORECASE)
        if m:
            return (fn, m.group(1).strip())

    return (None, None)


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for loose comparison: lowercase, collapse whitespace, strip quotes.

    Also strips backslashes so that \\`col\\` and `col` normalise identically,
    since the LLM sometimes emits backslash-escaped backticks copied from prompt examples.

    Normalizes spaces around = so that `col = val` and `col=val` compare as equal.
    """
    sql = sql.lower()
    sql = sql.replace("\\`", "`")      # unescape \` → ` before stripping
    sql = re.sub(r'[`"\'\\]', '', sql) # remove quoting chars AND stray backslashes
    sql = re.sub(r'\s*=\s*', '=', sql) # col = val -> col=val (purely cosmetic diff)
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql


def evaluate_agentbench_task(task: AgentBenchTask, trajectory: list) -> bool:
    """Check if task was successfully completed.

    Handles both string gold answers (OS tasks) and list gold answers (DB tasks).

    For DB SELECT tasks (gold answer is a scalar value like "7" or "Chelsea"):
      Returns True if the gold value appears in a sql_query/sql_execute observation
      (NOT describe_table or list_tables: those now contain sample rows that could
      accidentally match the gold answer), OR in any FINAL ANSWER line in a thought.
      Also handles numeric gold answers stored as floats (e.g. '1.0' matches '1').

    For DB INSERT/UPDATE tasks (gold answer is a SQL string):
      Returns True if the agent executed a SQL command that normalized-matches the gold SQL.

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
            # The stricter threshold (was 85%) prevents false positives when the
            # agent uses the right structure but a wrong column name or value.
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
        # the gold label stores a plain float ("2859.0"): strip commas before
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


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

async def run_agentbench_experiment(
    env_type: str = "os",
    n_samples: int = 200,
    output_dir: str = "results/agentbench_os",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cocoa_m: int = 5,
    branch_m: int = 5,
    temperature: float = 0.7,
    resume: bool = True,
    tensor_parallel_size: int = 1,
    split_phase: str = "all",
    calib_params_path: Optional[str] = None,
    dataset: str = "full",
    calib_ratio: float = 0.5,
    _cocoa=None,   # optional pre-built StepwiseCoCoA: keeps vLLM workers alive across phases
) -> tuple[list[dict], any]:
    """Run AgentBench experiment with UQ instrumentation and optional train/calib/test split.

    Parameters
    ----------
    env_type            : "os" or "db" task type.
    n_samples           : Total number of tasks to sample from.
    output_dir          : Directory for JSONL output and calibration params.
    model_name          : HuggingFace model ID or local path.
    cocoa_m             : M samples for step-wise CoCoA.
    branch_m            : M alternatives for branching consistency.
    temperature         : Sampling temperature for consistency sampling.
    resume              : If True, skip already-completed tasks.
    tensor_parallel_size: Number of GPUs for vLLM tensor parallelism.
    split_phase         : One of "calib", "test", or "all" (default: "all").
                          - "all": Run all n_samples without splitting (legacy mode).
                          - "calib": Run first calib_ratio of n_samples, fit normalisation params.
                          - "test": Run remaining (1-calib_ratio) of n_samples, use fitted params.
    calib_params_path   : Path to JSON file with fitted calibration params.
                          Required for split_phase="test" when called standalone.
    dataset             : "full" (default, 1000 OS / 360 DB) or "dev" (26 OS / 360 DB).
    _cocoa              : Pre-built StepwiseCoCoA instance whose .model is reused.
                          Keeping the model reference alive prevents vLLM worker
                          termination between calib and test phases.
    """
    # Deliberately lazy: these pull in vLLM/torch, so importing this module for its
    # dataclasses/CLI parsing alone doesn't require a GPU environment.
    from tqdm import tqdm
    from agent.instrumented_agent import InstrumentedVLLMModel, StepRecord
    from uq.annotation import annotate_steps_by_task_success
    from uq.branching import BranchingConsistency
    from uq.stepwise_cocoa import StepwiseCoCoA

    cfg = DEFAULT_CONFIG.experiment
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"

    logger.info("Loading AgentBench %s tasks (%d), dataset=%s", env_type.upper(), n_samples, dataset)
    all_tasks = load_agentbench_tasks(env_type, n=n_samples, dataset=dataset)

    # --- Split dataset based on split_phase ---
    if split_phase == "all":
        tasks = all_tasks
        logger.info("Running all %d tasks (no splitting)", len(tasks))
    else:
        # Use a local RNG with a fixed seed so the global numpy state is
        # not polluted and downstream code remains reproducible.
        indices = np.arange(len(all_tasks))
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

        calib_end = int(calib_ratio * len(all_tasks))

        if split_phase == "calib":
            selected_indices = indices[:calib_end]
            logger.info("Calib phase: running %d tasks (0-%.0f%%)", len(selected_indices), calib_ratio * 100)
        elif split_phase == "test":
            selected_indices = indices[calib_end:]
            logger.info("Test phase: running %d tasks (%.0f-100%%)", len(selected_indices), calib_ratio * 100)
        else:
            raise ValueError(f"Invalid split_phase: {split_phase!r}. "
                           f"Must be one of: 'calib', 'test', 'all'")

        tasks = [all_tasks[i] for i in selected_indices]

    # Resume support
    completed: set[str] = set()
    if resume and traj_path.exists():
        with open(traj_path) as fh:
            for line in fh:
                try:
                    completed.add(json.loads(line)["task_id"])
                except Exception:
                    pass
        logger.info("Resuming: %d tasks already done", len(completed))

    if _cocoa is not None:
        model = _cocoa.model
        cocoa = _cocoa
        logger.info("Reusing shared model instance for %s phase.", split_phase)
    else:
        logger.info("Loading model: %s", model_name)
        model = InstrumentedVLLMModel(model_name, tensor_parallel_size=tensor_parallel_size)
        cocoa = StepwiseCoCoA(model=model, M=cocoa_m, temperature=temperature)
    brancher = BranchingConsistency(model=model, M=branch_m, temperature=temperature)

    # --- Load calibration parameters if test phase (standalone only) ---
    # When _cocoa is provided the params were already fitted and set by the caller.
    if split_phase == "test" and _cocoa is None:
        if calib_params_path is None:
            raise ValueError("split_phase='test' requires --calib-params-path")
        with open(calib_params_path) as f:
            params = json.load(f)
        cocoa._q98 = params["q98"]
        cocoa._u_min = params["u_min"]
        cocoa._fitted = True
        logger.info("Loaded calibration parameters: q98=%.4f, u_min=%.4f",
                   cocoa._q98, cocoa._u_min)

    system_prompt = OS_SYSTEM_PROMPT if env_type == "os" else DB_SYSTEM_PROMPT
    results: list[dict] = []

    # Determine execution mode
    use_python_api = cfg.use_python_api
    use_docker = cfg.use_docker

    logger.info("Execution mode: %s", "Python API" if use_python_api else "Docker/Enroot")

    for task in tqdm(tasks, desc=f"AgentBench-{env_type.upper()}"):
        if task.task_id in completed:
            continue

        # Choose environment
        if use_python_api:
            env = PythonAPIEnvironment(env_type, task)
        elif use_docker:
            env = ContainerEnvironment(env_type, task)
        else:
            raise RuntimeError("No execution mode configured")

        try:
            env.start()
        except Exception as exc:
            logger.error("Failed to start environment: %s", exc)
            continue

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.instruction},
        ]
        trajectory: list[StepRecord] = []

        try:
            for step_id in range(cfg.agentbench_max_steps):
                response = model(messages, temperature=0.0, max_tokens=512)[0]
                raw_logprob = model.last_seq_logprob

                action_name, action_cmd = parse_agentbench_action(response, env_type)

                # ── Early-exit on FINAL ANSWER / TASK COMPLETE ──────────────
                # Check BEFORE calling env.execute() so the terminal step gets a
                # clean "Task complete" observation rather than "No valid action
                # parsed", and we don't waste an extra environment call.
                is_terminal = (
                    "TASK COMPLETE" in response
                    or "FINAL ANSWER" in response.upper()
                )

                # Reconstruct the proper command string for the environment.
                # The parser returns (action_name, bare_arg); _execute_sql expects
                # the full function-call form for list_tables / describe_table.
                if is_terminal:
                    exec_cmd: Optional[str] = None
                    observation = "[Task ended: FINAL ANSWER given]"
                elif action_name == "list_tables":
                    exec_cmd = "list_tables()"
                    observation = env.execute(exec_cmd)
                elif action_name == "describe_table":
                    exec_cmd = f"describe_table({action_cmd!r})" if action_cmd else "describe_table()"
                    observation = env.execute(exec_cmd)
                elif action_name in ("sql_query", "sql_execute"):
                    exec_cmd = action_cmd  # raw SQL passed straight to SQLite
                    observation = env.execute(exec_cmd)
                else:
                    exec_cmd = None
                    observation = "No valid action parsed"

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
                    action_name=action_name or "none",
                    action_args={"command": action_cmd or ""},
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

                if is_terminal:
                    break

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user",
                                 "content": f"Observation: {observation}"})

        finally:
            env.stop()

        task_success = evaluate_agentbench_task(task, [
            {
                "thought": s.thought,
                "observation": s.observation,
                # action_name lets the evaluator distinguish sql_query observations
                # (valid answer sources) from describe_table/list_tables observations
                # (which contain sample rows that can accidentally match gold answers).
                "action_name": s.action_name,
                "action_cmd": s.action_args.get("command", ""),
            }
            for s in trajectory
        ])
        annotated = annotate_steps_by_task_success(trajectory, task_success)

        record = {
            "task_id": task.task_id,
            "env_type": env_type,
            "instruction": task.instruction,
            "gold_answer": task.gold_answer,
            "task_success": task_success,
            "n_steps": len(annotated),
            "trajectory": [
                {
                    "step_id": s.step_id,
                    "thought": s.thought[:1000],  # 1000 chars: avoids truncating FINAL ANSWER
                    "action_name": s.action_name,
                    # action_cmd stores the raw SQL / table-name arg for evaluation and analysis.
                    # action_args ({"command": ...}) was redundant and has been removed.
                    "action_cmd": s.action_args.get("command", ""),
                    "observation": s.observation[:1000],
                    "seq_logprob": s.seq_logprob,
                    "consistency_scores": s.consistency_scores,
                    "c_star_cocoa": s.c_star_cocoa,
                    "branch_consistency_raw": s.branch_consistency_raw,
                    "branch_consistency": s.branch_consistency,
                    "escalated": s.escalated,
                    "is_correct": s.is_correct,
                }
                for s in annotated
            ],
        }
        results.append(record)
        with open(traj_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    logger.info(
        "AgentBench-%s done. Success rate: %.3f",
        env_type.upper(),
        sum(r["task_success"] for r in results) / max(len(results), 1),
    )

    # --- Fit and save calibration parameters (calib phase only) ---
    if split_phase == "calib":
        if not cocoa._raw_scores:
            # All calib samples were resumed: params are already on disk.
            logger.info(
                "All calib samples were resumed; skipping fit "
                "(calibration params already saved on disk)."
            )
        else:
            logger.info("Fitting normalisation parameters on %d raw scores...",
                       len(cocoa._raw_scores))
            cocoa.fit_normalisation()

            calib_params = {
                "q98": float(cocoa._q98),
                "u_min": float(cocoa._u_min),
                "n_raw_scores": len(cocoa._raw_scores),
                "split_phase": "calib",
                "env_type": env_type,
                "model": model_name,
                "cocoa_m": cocoa_m,
                "temperature": temperature,
            }
            calib_params_file = Path(output_dir) / "calib_params.json"
            calib_params_file.write_text(json.dumps(calib_params, indent=2))
            logger.info("Saved calibration parameters to %s", calib_params_file)

    return results, cocoa


async def run_agentbench_full_pipeline(
    env_type: str = "os",
    n_samples: int = 200,
    output_base: str = "results/agentbench_os",
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    cocoa_m: int = 5,
    branch_m: int = 5,
    temperature: float = 0.7,
    resume: bool = True,
    tensor_parallel_size: int = 1,
    dataset: str = "full",
    calib_ratio: float = 0.5,
) -> dict[str, list[dict]]:
    """
    Run calibration-to-test pipeline sequentially inside one vLLM lifetime.

    By passing the StepwiseCoCoA object (which holds the model reference) from
    the calib call into the test call, the vLLM backend never idles between
    phases, so its worker processes stay alive and there is no CUDA re-init hang.

    Creates subdirectories for each phase and passes calibration params automatically.

    Parameters
    ----------
    dataset : str
        "full" (default, 1000 OS / 360 DB) or "dev" (26 OS / 360 DB).

    Returns: dict with keys "calib", "test" containing results from each phase.
    """
    base = Path(output_base)
    calib_dir = base / "calib"
    test_dir = base / "test"
    calib_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ------------------------------------------------------------------
    # Phase 1: Calibration (calib_ratio of n_samples)
    # The cocoa object returned here keeps the vLLM model alive.
    # ------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("PHASE 1: CALIBRATION (%.0f%% of %d %s tasks)", calib_ratio * 100, n_samples, env_type.upper())
    logger.info("=" * 80)
    calib_results, cocoa = await run_agentbench_experiment(
        env_type=env_type,
        n_samples=n_samples,
        output_dir=str(calib_dir),
        model_name=model_name,
        cocoa_m=cocoa_m,
        branch_m=branch_m,
        temperature=temperature,
        resume=resume,
        tensor_parallel_size=tensor_parallel_size,
        split_phase="calib",
        dataset=dataset,
        calib_ratio=calib_ratio,
    )
    all_results["calib"] = calib_results
    logger.info("Phase 1 complete: %d tasks processed", len(calib_results))

    # Ensure calib params are fitted on cocoa before handing it to the test phase.
    calib_params_file = calib_dir / "calib_params.json"
    if not cocoa._fitted:
        # All calib samples were resumed: load params from disk and set on cocoa.
        if not calib_params_file.exists():
            raise FileNotFoundError(
                f"Calibration params file not found: {calib_params_file}\n"
                f"Cannot proceed to test phase without fitted params."
            )
        with open(calib_params_file) as f:
            params = json.load(f)
        cocoa._q98 = params["q98"]
        cocoa._u_min = params["u_min"]
        cocoa._fitted = True
        logger.info(
            "Loaded calib params from disk (all calib samples were resumed): "
            "q98=%.4f, u_min=%.2e", cocoa._q98, cocoa._u_min
        )

    # Reset the raw-score buffer so test-phase scores don't mix with calib scores.
    cocoa._raw_scores = []

    # ------------------------------------------------------------------
    # Phase 2: Test (remaining fraction): reuse cocoa/model, no re-init
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: TEST (%.0f%% of %d %s tasks)", (1 - calib_ratio) * 100, n_samples, env_type.upper())
    logger.info("=" * 80)
    test_results, _ = await run_agentbench_experiment(
        env_type=env_type,
        n_samples=n_samples,
        output_dir=str(test_dir),
        model_name=model_name,
        cocoa_m=cocoa_m,
        branch_m=branch_m,
        temperature=temperature,
        resume=resume,
        tensor_parallel_size=tensor_parallel_size,
        split_phase="test",
        dataset=dataset,
        calib_ratio=calib_ratio,
        _cocoa=cocoa,   # ← keeps vLLM workers alive; params already set on cocoa
    )
    all_results["test"] = test_results
    logger.info("Phase 2 complete: %d tasks processed", len(test_results))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION -> TEST PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info("Calib phase (%.0f%%):  %d/%d tasks, %d trajectory records",
               calib_ratio * 100,
               sum(1 for r in calib_results if r["task_success"]),
               len(calib_results),
               sum(len(r["trajectory"]) for r in calib_results))
    logger.info("Test phase  (%.0f%%):  %d/%d tasks, %d trajectory records (CALIBRATED)",
               (1 - calib_ratio) * 100,
               sum(1 for r in test_results if r["task_success"]),
               len(test_results),
               sum(len(r["trajectory"]) for r in test_results))

    logger.info("\nResults location:")
    logger.info("  Calib (used for fitting):  %s/trajectories.jsonl", calib_dir)
    logger.info("  Calib params:              %s", calib_params_file)
    logger.info("  Test (for evaluation):     %s/trajectories.jsonl ← USE THIS", test_dir)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AgentBench Experiment with UQ instrumentation and Train/Calib/Test split"
    )
    parser.add_argument(
        "--env-type",
        choices=["os", "db"],
        default="os",
        help="AgentBench environment type (default: os)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Total number of tasks to sample from (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/agentbench_os",
        help="Results directory for trajectories and calibration params",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--cocoa-m",
        type=int,
        default=5,
        help="StepwiseCoCoA samples (default: 5)",
    )
    parser.add_argument(
        "--branch-m",
        type=int,
        default=5,
        help="BranchingConsistency alternatives (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoint",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Use Docker/Enroot instead of Python API",
    )
    parser.add_argument(
        "--split-phase",
        type=str,
        default="all",
        choices=["calib", "test", "all", "full"],
        help="Which phase to run: 'calib'/'test' (individual), 'all' (no split, legacy), or 'full' (run calib-to-test sequentially)",
    )
    parser.add_argument(
        "--calib-params-path",
        type=str,
        default=None,
        help="Path to JSON file with calibration parameters (required for --split-phase test)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="full",
        choices=["full", "dev"],
        help="Dataset version: 'full' (default, 1000 OS / 360 DB) or 'dev' (26 OS / 360 DB). DB always uses dev+standard combined.",
    )
    parser.add_argument(
        "--calib-ratio",
        type=float,
        default=0.5,
        help="Fraction of tasks used for calibration phase (default: 0.5)",
    )
    args = parser.parse_args()

    # Override config if requested
    if args.use_docker:
        DEFAULT_CONFIG.experiment.use_python_api = False
        DEFAULT_CONFIG.experiment.use_docker = True

    if args.split_phase == "full":
        # Run all three phases sequentially
        asyncio.run(
            run_agentbench_full_pipeline(
                env_type=args.env_type,
                n_samples=args.n_samples,
                output_base=args.output_dir,
                model_name=args.model,
                cocoa_m=args.cocoa_m,
                branch_m=args.branch_m,
                temperature=args.temperature,
                resume=not args.no_resume,
                tensor_parallel_size=args.tensor_parallel_size,
                dataset=args.dataset,
                calib_ratio=args.calib_ratio,
            )
        )
    else:
        # Run a single phase (unpack the (results, cocoa) tuple)
        async def _run_single():
            results, _ = await run_agentbench_experiment(
                env_type=args.env_type,
                n_samples=args.n_samples,
                output_dir=args.output_dir,
                model_name=args.model,
                cocoa_m=args.cocoa_m,
                branch_m=args.branch_m,
                temperature=args.temperature,
                tensor_parallel_size=args.tensor_parallel_size,
                resume=not args.no_resume,
                split_phase=args.split_phase,
                calib_params_path=args.calib_params_path,
                dataset=args.dataset,
                calib_ratio=args.calib_ratio,
            )
            return results

        asyncio.run(_run_single())
