#!/usr/bin/env python3
"""
Quick test script to verify AgentBench setup and dual execution modes.

Only AgentBench-DB (SQL interaction) is used in the final thesis. The OS-mode tests
here (Python API OS, container detection) cover AgentBench-OS, which was dropped from
the final thesis scope. Kept working for the archived tooling in legacy/, see
legacy/README.md.

Tests:
1. Task loading from JSONL/JSON files
2. Python API environment: OS mode (excluded from final thesis)
3. Python API environment: DB mode (final thesis benchmark)
4. Container runtime detection (Docker/Enroot; OS mode only)
5. Configuration settings
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from config import DEFAULT_CONFIG
from experiments.agentbench_agent import (
    load_agentbench_tasks,
    PythonAPIEnvironment,
    ContainerEnvironment,
    AgentBenchTask,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_task_loading():
    """Test loading AgentBench tasks."""
    logger.info("=" * 70)
    logger.info("TEST 1: Task Loading")
    logger.info("=" * 70)

    try:
        # Try to load a few OS tasks
        logger.info("Loading OS tasks...")
        os_tasks = load_agentbench_tasks("os", n=3)
        logger.info(f"✓ Loaded {len(os_tasks)} OS tasks")
        for task in os_tasks:
            logger.info(f"  - Task {task.task_id}: {task.instruction[:60]}...")

        # Try to load a few DB tasks
        logger.info("Loading DB tasks...")
        db_tasks = load_agentbench_tasks("db", n=3)
        logger.info(f"✓ Loaded {len(db_tasks)} DB tasks")
        for task in db_tasks:
            logger.info(f"  - Task {task.task_id}: {task.instruction[:60]}...")

        return True
    except FileNotFoundError as e:
        logger.error(f"✗ Task loading failed: {e}")
        logger.info("   (This is OK if AgentBench data not downloaded yet)")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False


def test_python_api_environment():
    """Test Python API execution environment."""
    logger.info("=" * 70)
    logger.info("TEST 2: Python API Environment")
    logger.info("=" * 70)

    try:
        # Create a mock OS task
        task = AgentBenchTask(
            task_id="test_os_001",
            env_type="os",
            instruction="List files in current directory",
            gold_answer="test_agentbench_setup.py",
            initial_state={},
        )

        logger.info("Creating Python API environment...")
        env = PythonAPIEnvironment("os", task)
        env.start()
        logger.info("✓ Environment started")

        # Test bash execution
        logger.info("Testing bash execution...")
        result = env.execute("ls -la | head -5")
        logger.info(f"✓ Bash execution successful")
        logger.info(f"  Output preview: {result[:100]}...")

        env.stop()
        logger.info("✓ Environment stopped")
        return True

    except Exception as e:
        logger.error(f"✗ Python API test failed: {e}")
        return False


def test_python_api_database():
    """Test Python API SQLite environment."""
    logger.info("=" * 70)
    logger.info("TEST 3: Python API Database Environment")
    logger.info("=" * 70)

    try:
        # Create a mock DB task
        task = AgentBenchTask(
            task_id="test_db_001",
            env_type="db",
            instruction="Create a table and insert data",
            gold_answer=None,
            initial_state={
                "schema": "CREATE TABLE test (id INTEGER, name TEXT)"
            },
        )

        logger.info("Creating Python API DB environment...")
        env = PythonAPIEnvironment("db", task)
        env.start()
        logger.info("✓ Database environment started")

        # Test SQL execution
        logger.info("Testing SQL execution...")
        result = env.execute("INSERT INTO test VALUES (1, 'Alice')")
        logger.info(f"✓ INSERT successful: {result}")

        result = env.execute("SELECT * FROM test")
        logger.info(f"✓ SELECT successful: {result}")

        # Test list tables
        result = env.execute("list_tables()")
        logger.info(f"✓ List tables: {result}")

        env.stop()
        logger.info("✓ Database environment stopped")
        return True

    except Exception as e:
        logger.error(f"✗ Database test failed: {e}")
        return False


def test_container_detection():
    """Test Docker/Enroot detection."""
    logger.info("=" * 70)
    logger.info("TEST 4: Container Runtime Detection")
    logger.info("=" * 70)

    try:
        task = AgentBenchTask(
            task_id="test_container",
            env_type="os",
            instruction="Test",
            gold_answer=None,
            initial_state={},
        )

        env = ContainerEnvironment("os", task)
        runtime = env.container_runtime

        if runtime:
            logger.info(f"✓ Container runtime detected: {runtime.upper()}")
            logger.info(f"  Note: Docker/Enroot is available as a fallback option")
            return True
        else:
            logger.warning("⚠ No container runtime found (this is OK)")
            logger.info("  Python API mode will be used (default)")
            return True  # Not a failure, just no containers available

    except Exception as e:
        logger.error(f"✗ Container detection failed: {e}")
        return False


def test_configuration():
    """Test configuration settings."""
    logger.info("=" * 70)
    logger.info("TEST 5: Configuration")
    logger.info("=" * 70)

    try:
        cfg = DEFAULT_CONFIG.experiment

        logger.info(f"Execution Mode Configuration:")
        logger.info(f"  use_python_api: {cfg.use_python_api} ✓")
        logger.info(f"  use_docker: {cfg.use_docker}")
        logger.info(f"  docker_timeout_s: {cfg.docker_timeout_s}s")

        if cfg.use_python_api:
            logger.info("✓ Python API mode is ENABLED (default)")
        if cfg.use_docker:
            logger.info("✓ Docker mode is ENABLED")

        if not cfg.use_python_api and not cfg.use_docker:
            logger.error("✗ No execution mode enabled!")
            return False

        return True

    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 15 + "AGENTBENCH SETUP VERIFICATION" + " " * 25 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    logger.info("\n")

    results = {
        "Configuration": test_configuration(),
        "Task Loading": test_task_loading(),
        "Python API (OS)": test_python_api_environment(),
        "Python API (DB)": test_python_api_database(),
        "Container Detection": test_container_detection(),
    }

    # Summary
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} {test_name}")

    logger.info("=" * 70)
    logger.info(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✓ All systems ready! You can now run AgentBench experiments:")
        logger.info("  python -m experiments.agentbench_agent --env-type db --n-samples 10")
        return 0
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) failed. See details above.")
        if results["Task Loading"] is False:
            logger.info("\nNote: Task loading failed because AgentBench data is not downloaded.")
            logger.info("Run setup if needed: bash scripts/setup_agentbench.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
