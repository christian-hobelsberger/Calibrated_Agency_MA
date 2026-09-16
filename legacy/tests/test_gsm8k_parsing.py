"""
Unit tests for GSM8K answer parsing and action parsing.
No GPU required.
"""
import pytest
from experiments.gsm8k_agent import (
    extract_gsm8k_answer,
    answers_match,
    parse_react_output,
)


class TestExtractAnswer:

    def test_explicit_final_answer(self):
        text = "Thought: done\nFINAL ANSWER: 42"
        assert extract_gsm8k_answer(text) == "42"

    def test_gsm8k_gold_format(self):
        text = "She earned #### 18 dollars"
        assert extract_gsm8k_answer(text) == "18"

    def test_fallback_last_number(self):
        text = "The result is 3.14 and the answer is 99"
        assert extract_gsm8k_answer(text) == "99"

    def test_comma_separated_number(self):
        text = "FINAL ANSWER: 1,234"
        assert extract_gsm8k_answer(text) == "1234"

    def test_no_number(self):
        text = "I cannot determine the answer."
        assert extract_gsm8k_answer(text) is None


class TestAnswersMatch:

    def test_exact_integer_match(self):
        assert answers_match("42", "42") is True

    def test_float_tolerance(self):
        assert answers_match("3.14159", "3.14160") is True

    def test_mismatch(self):
        assert answers_match("42", "43") is False

    def test_none_pred(self):
        assert answers_match(None, "42") is False

    def test_none_gold(self):
        assert answers_match("42", None) is False

    def test_integer_vs_float(self):
        assert answers_match("18.0", "18") is True


class TestParseReActOutput:

    def test_parses_calculate_action(self):
        text = 'Thought: I need to compute 3*4.\nAction: calculate("3*4")'
        thought, action, expr, is_final = parse_react_output(text)
        assert action == "calculate"
        assert expr == "3*4"
        assert is_final is False

    def test_detects_final_answer(self):
        text = "Thought: done.\nFINAL ANSWER: 42"
        _, _, _, is_final = parse_react_output(text)
        assert is_final is True

    def test_no_action_returns_none(self):
        text = "Thought: I'm thinking..."
        thought, action, expr, is_final = parse_react_output(text)
        assert action is None
        assert expr is None
        assert is_final is False
