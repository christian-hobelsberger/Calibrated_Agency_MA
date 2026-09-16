"""
MCP server providing sandboxed bash execution for the AgentBench OS experiment.

IMPORTANT: This server must be run INSIDE a Docker container for real isolation.
See docker/Dockerfile.bash and the AgentBench experiment setup.

Run as a subprocess (stdio transport):
    python tools/bash_server.py

Exposed tools: bash
"""
import os
import subprocess

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("bash-executor")

TIMEOUT_SECONDS: int = int(os.getenv("BASH_TIMEOUT", "15"))
MAX_OUTPUT_CHARS: int = 2000

# Allow-list is the last defence; Docker provides real isolation.
ALLOWED_COMMANDS: frozenset[str] = frozenset({
    "awk", "cat", "chmod", "cp", "cut", "echo", "find", "grep",
    "head", "ls", "mkdir", "mv", "python3", "pip", "pwd",
    "rm", "sed", "sort", "tail", "wc",
})


@mcp.tool()
def bash(command: str) -> str:
    """
    Execute a bash command in a sandboxed environment.

    Returns combined stdout + stderr (truncated to 2000 characters).
    Example: bash("ls -la /home/user")

    Rules:
    - Only commands in the ALLOWED_COMMANDS list are permitted.
    - Commands time out after BASH_TIMEOUT seconds (default 15).
    - The working directory is /tmp/agent_home.
    """
    stripped = command.strip()
    if not stripped:
        return "ERROR: Empty command."

    # Extract the base command (handles paths like /usr/bin/ls → ls)
    base_cmd = stripped.split()[0].split("/")[-1]
    if base_cmd not in ALLOWED_COMMANDS:
        return (
            f"BLOCKED: '{base_cmd}' is not in the allowed command list.\n"
            f"Allowed: {sorted(ALLOWED_COMMANDS)}"
        )

    env = {**os.environ, "HOME": "/tmp/agent_home"}
    try:
        result = subprocess.run(
            stripped,
            shell=True,        # noqa: S602 (Docker provides real isolation)
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd="/tmp/agent_home",
            env=env,
        )
        output = result.stdout + result.stderr
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n[TRUNCATED]"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {TIMEOUT_SECONDS}s."
    except Exception as exc:
        return f"ERROR: {exc}"


if __name__ == "__main__":
    os.makedirs("/tmp/agent_home", exist_ok=True)
    mcp.run(transport="stdio")
