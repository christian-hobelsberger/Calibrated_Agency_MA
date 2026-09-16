"""
MCP server providing read/write access to a SQLite database.

Run as a subprocess (stdio transport):
    python tools/sqlite_server.py [--db-path /path/to/db.sqlite]

Exposed tools: sql_query, sql_execute, list_tables, describe_table
"""
import argparse
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Default path, override via CLI argument or DB_PATH env var
import os

_DEFAULT_DB = Path(os.getenv("AGENTBENCH_DB_PATH", "/tmp/agentbench_db.sqlite"))

mcp = FastMCP("sqlite-db")


def _get_db_path() -> Path:
    return _DEFAULT_DB


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def sql_query(query: str) -> str:
    """
    Execute a SQL SELECT query and return results formatted as a table.

    Use for read-only lookups only.
    Example: sql_query("SELECT * FROM orders WHERE status = 'pending'")
    """
    if not query.strip().upper().startswith("SELECT"):
        return "ERROR: sql_query only accepts SELECT statements. Use sql_execute for writes."
    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "(no rows returned)"
        header = " | ".join(rows[0].keys())
        sep    = "-" * len(header)
        body   = "\n".join(" | ".join(str(v) for v in row) for row in rows)
        return f"{header}\n{sep}\n{body}"
    except sqlite3.Error as exc:
        return f"SQL ERROR: {exc}"


@mcp.tool()
def sql_execute(statement: str) -> str:
    """
    Execute a SQL INSERT, UPDATE, or DELETE statement.

    Returns the number of rows affected.
    Example: sql_execute("UPDATE orders SET status='shipped' WHERE id=42")
    """
    forbidden = {"SELECT", "DROP", "TRUNCATE", "ALTER"}
    first_word = statement.strip().split()[0].upper() if statement.strip() else ""
    if first_word in {"DROP", "TRUNCATE", "ALTER"}:
        return f"ERROR: '{first_word}' is not permitted via sql_execute."
    try:
        conn = sqlite3.connect(str(_get_db_path()))
        cursor = conn.cursor()
        cursor.execute(statement)
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return f"OK: {affected} row(s) affected."
    except sqlite3.Error as exc:
        return f"SQL ERROR: {exc}"


@mcp.tool()
def list_tables() -> str:
    """Return all table names in the current database."""
    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return ", ".join(tables) if tables else "(no tables found)"
    except sqlite3.Error as exc:
        return f"SQL ERROR: {exc}"


@mcp.tool()
def describe_table(table_name: str) -> str:
    """
    Return the schema (column names and types) of a table.

    Example: describe_table("orders")
    """
    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")  # noqa: S608
        cols = cursor.fetchall()
        conn.close()
        if not cols:
            return f"Table '{table_name}' not found or has no columns."
        return "\n".join(f"{c[1]} ({c[2]})" for c in cols)
    except sqlite3.Error as exc:
        return f"SQL ERROR: {exc}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=_DEFAULT_DB)
    args = parser.parse_args()
    _DEFAULT_DB = args.db_path
    mcp.run(transport="stdio")
