"""
MCPToolProvider — starts an MCP server subprocess and wraps its tools
for use in experiments, with full call-level logging for UQ annotation.

Usage
-----
    provider = MCPToolProvider("tools/calculator_server.py", "calculator")
    await provider.connect()
    result = await provider.call("calculate", expression="2 + 2")
    await provider.close()

Or use as an async context manager:
    async with MCPToolProvider("tools/calculator_server.py") as provider:
        result = await provider.call("calculate", expression="2 + 2")
"""
from __future__ import annotations

import ast
import asyncio
import logging
import operator
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MCPToolProvider:
    """
    Manages a single MCP server subprocess and exposes its tools.

    Each tool call is logged with full args, result, latency, and error flag
    so the annotation protocol can correlate tool invocations with UQ scores.
    """

    def __init__(self, server_script: str | Path, server_name: str = ""):
        self.server_script = str(server_script)
        self.server_name = server_name or Path(server_script).stem
        self._session = None
        self._client_cm = None
        self._call_log: list[dict] = []

    async def connect(self) -> None:
        """Start the MCP server process and initialise the session."""
        # Deliberately lazy: the `mcp` SDK is an optional dependency, and code paths that
        # use DirectCalculator instead of MCPToolProvider don't need it installed.
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise ImportError(
                "MCP SDK not installed. Run: pip install mcp"
            ) from exc

        params = StdioServerParameters(
            command="python",
            args=[self.server_script],
        )
        self._client_cm = stdio_client(params)
        read, write = await self._client_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        logger.debug("Connected to MCP server: %s", self.server_name)

    async def close(self) -> None:
        """Shut down the MCP session and server process."""
        if self._session:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._client_cm:
            try:
                await self._client_cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._session = None

    async def __aenter__(self) -> "MCPToolProvider":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def call(self, tool_name: str, **kwargs: Any) -> str:
        """
        Invoke a tool on the MCP server. Returns the text result.

        All calls are logged to `self.call_log` for later annotation.
        """
        if self._session is None:
            raise RuntimeError("Not connected. Call await provider.connect() first.")

        t0 = time.monotonic()
        error = False
        text = ""
        try:
            result = await self._session.call_tool(tool_name, kwargs)
            text = result.content[0].text if result.content else ""
            error = bool(result.isError)
        except Exception as exc:
            text = f"ERROR: {exc}"
            error = True
        finally:
            elapsed = time.monotonic() - t0
            self._call_log.append({
                "server":   self.server_name,
                "tool":     tool_name,
                "args":     kwargs,
                "result":   text,
                "elapsed_s": round(elapsed, 4),
                "error":    error,
            })

        return text

    async def list_tools(self) -> list[str]:
        """Return the names of tools exposed by this server."""
        if self._session is None:
            raise RuntimeError("Not connected.")
        resp = await self._session.list_tools()
        return [t.name for t in resp.tools]

    @property
    def call_log(self) -> list[dict]:
        """Full list of all calls made through this provider."""
        return self._call_log

    def clear_log(self) -> None:
        self._call_log.clear()


# ---------------------------------------------------------------------------
# Lightweight synchronous fallback (no MCP required)
# Used in unit tests and when the MCP server is unavailable.
# ---------------------------------------------------------------------------

class DirectCalculator:
    """
    Pure-Python calculator that mirrors the calculator MCP server.
    Used for local development and CI without a running MCP subprocess.
    """

    _SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
    }

    def _validate_and_fix_parentheses(self, expr: str) -> tuple[str, Optional[str]]:
        """
        Validate expression and auto-fix mismatched parentheses.

        Returns: (fixed_expr, note)
            note is None if no fixing was needed, otherwise describes what was fixed.
        """
        expr = expr.strip()
        open_count = expr.count('(')
        close_count = expr.count(')')

        if open_count == close_count:
            return expr, None

        if open_count > close_count:
            # Auto-fix by adding closing parens
            fixed = expr + ')' * (open_count - close_count)
            return fixed, f"auto-fixed: added {open_count - close_count} closing paren(s)"

        # close_count > open_count: can't safely auto-fix
        raise ValueError(f"Too many closing parentheses: {close_count} close vs {open_count} open")

    def calculate(self, expression: str) -> str:
        try:
            # Validate and auto-fix parenthesis mismatches (silently, no annotation)
            fixed_expr, fix_note = self._validate_and_fix_parentheses(expression)

            tree = ast.parse(fixed_expr, mode="eval")
            result = self._eval(tree.body)

            # Avoid floating-point noise on integers
            if result == int(result) and abs(result) < 1e15:
                result_str = str(int(result))
            else:
                result_str = str(round(result, 8))

            # Return result without annotation to prevent repetition loops
            return result_str
        except Exception as e:
            return f"ERROR: {e}"

    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return self._SAFE_OPS[type(node.op)](
                self._eval(node.left), self._eval(node.right)
            )
        elif isinstance(node, ast.UnaryOp):
            return self._SAFE_OPS[type(node.op)](self._eval(node.operand))
        raise ValueError(f"Unsafe operation: {type(node).__name__}")
