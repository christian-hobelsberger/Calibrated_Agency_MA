"""
Unit tests for the calculator MCP server (tool logic only, no MCP transport).
"""
import pytest
from agent.mcp_client import DirectCalculator


@pytest.fixture
def calc():
    return DirectCalculator()


class TestDirectCalculator:

    def test_addition(self, calc):
        assert calc.calculate("2 + 3") == "5"

    def test_subtraction(self, calc):
        assert calc.calculate("10 - 4") == "6"

    def test_multiplication(self, calc):
        assert calc.calculate("3 * 7") == "21"

    def test_division(self, calc):
        result = calc.calculate("10 / 4")
        assert float(result) == pytest.approx(2.5)

    def test_float_division(self, calc):
        result = calc.calculate("1 / 3")
        assert float(result) == pytest.approx(1 / 3, rel=1e-5)

    def test_nested_expression(self, calc):
        result = calc.calculate("(80 / 100) * 10 + 10")
        assert float(result) == pytest.approx(18.0)

    def test_power(self, calc):
        result = calc.calculate("2 ** 10")
        assert float(result) == pytest.approx(1024.0)

    def test_modulo(self, calc):
        result = calc.calculate("17 % 5")
        assert float(result) == pytest.approx(2.0)

    def test_division_by_zero(self, calc):
        result = calc.calculate("5 / 0")
        assert "ERROR" in result

    def test_invalid_expression(self, calc):
        result = calc.calculate("import os")
        assert "ERROR" in result

    def test_empty_string(self, calc):
        result = calc.calculate("")
        assert "ERROR" in result or result == "0"

    def test_unmatched_open_paren_simple(self, calc):
        """Test auto-fix of unmatched opening parenthesis (from idx 12 trajectory)."""
        result = calc.calculate("(7 * 1.5")
        # Auto-fix is silent (no annotation), just verify correct result
        assert float(result) == pytest.approx(10.5)

    def test_unmatched_open_paren_division(self, calc):
        """Test auto-fix of unmatched opening parenthesis (from idx 13 trajectory)."""
        result = calc.calculate("(1/2")
        # Auto-fix is silent (no annotation), just verify correct result
        assert float(result) == pytest.approx(0.5)

    def test_multiple_unmatched_parens(self, calc):
        """Test auto-fix with multiple unmatched opening parentheses."""
        result = calc.calculate("((3 + 2")
        # Auto-fix is silent (no annotation), just verify correct result
        assert float(result) == pytest.approx(5.0)

    def test_too_many_closing_parens(self, calc):
        """Test error when there are too many closing parentheses."""
        result = calc.calculate("3 + 2)")
        assert "ERROR" in result
