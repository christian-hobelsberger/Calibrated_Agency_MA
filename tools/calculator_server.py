"""
MCP server exposing a safe arithmetic calculator for the GSM8K experiment.

Run as a subprocess (stdio transport):
    python tools/calculator_server.py

Exposed tools: calculate, sqrt, round_number
"""
import ast
import math
import operator

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator")

_SAFE_OPS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.Pow:      operator.pow,
    ast.Mod:      operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub:     operator.neg,
}


def _validate_and_fix_parentheses(expr: str) -> tuple[str, str | None]:
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


def _safe_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op.__name__}")
        return _SAFE_OPS[op](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary operator: {op.__name__}")
        return _SAFE_OPS[op](_safe_eval(node.operand))
    raise ValueError(f"Unsafe expression element: {type(node).__name__}")


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.

    Supports: +, -, *, /, **, %, //
    Automatically fixes unmatched parentheses silently (e.g., "(7 * 1.5" -> "(7 * 1.5)")

    Example: calculate("(80 / 100) * 10 + 10")  -> "18.0"
    Example: calculate("(7 * 1.5")  -> "10.5"  (parenthesis auto-fixed silently)
    """
    try:
        # Validate and auto-fix parenthesis mismatches (silently, no annotation)
        fixed_expr, fix_note = _validate_and_fix_parentheses(expression)

        tree = ast.parse(fixed_expr, mode="eval")
        result = _safe_eval(tree.body)

        # Avoid floating-point noise on integers
        if result == int(result) and abs(result) < 1e15:
            result_str = str(int(result))
        else:
            result_str = str(round(result, 8))

        # Return result without annotation to prevent repetition loops
        return result_str
    except ZeroDivisionError:
        return "ERROR: Division by zero"
    except Exception as exc:
        return f"ERROR: {exc}"


@mcp.tool()
def sqrt(x: float) -> str:
    """
    Return the square root of x.

    Example: sqrt(16)  -> "4.0"
    """
    if x < 0:
        return "ERROR: Cannot take sqrt of a negative number"
    return str(round(math.sqrt(x), 8))


@mcp.tool()
def round_number(x: float, decimals: int = 0) -> str:
    """
    Round x to the given number of decimal places (default 0).

    Example: round_number(3.14159, 2)  -> "3.14"
    """
    return str(round(x, decimals))


if __name__ == "__main__":
    mcp.run(transport="stdio")
