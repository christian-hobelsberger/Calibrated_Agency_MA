"""
Unit tests for the step-level annotation protocol.
These tests do NOT require a GPU or vLLM.
"""
import pytest
from agent.instrumented_agent import StepRecord
from uq.annotation import annotate_steps_by_task_success, annotate_gsm8k_steps


def _make_step(step_id, observation, action_name="calculate", expression="1+1"):
    return StepRecord(
        step_id=step_id,
        thought="some thought",
        action_name=action_name,
        action_args={"expression": expression},
        observation=observation,
    )


# ---------------------------------------------------------------------------
# annotate_steps_by_task_success
# ---------------------------------------------------------------------------

class TestAnnotateByTaskSuccess:

    def test_all_correct_on_success(self):
        steps = [_make_step(i, "ok") for i in range(4)]
        result = annotate_steps_by_task_success(steps, task_success=True)
        assert all(s.is_correct for s in result)

    def test_all_after_first_error_incorrect(self):
        steps = [
            _make_step(0, "2"),
            _make_step(1, "4"),
            _make_step(2, "ERROR: division by zero"),  # first error
            _make_step(3, "10"),
        ]
        result = annotate_steps_by_task_success(steps, task_success=False)
        assert result[0].is_correct is True
        assert result[1].is_correct is True
        assert result[2].is_correct is False
        assert result[3].is_correct is False

    def test_unparsed_action_triggers_error(self):
        steps = [
            _make_step(0, "3", action_name="calculate"),
            _make_step(1, "ok", action_name="none"),  # no action = suspicious
        ]
        result = annotate_steps_by_task_success(steps, task_success=False)
        assert result[0].is_correct is True
        assert result[1].is_correct is False

    def test_no_error_found_all_correct(self):
        steps = [_make_step(i, "ok") for i in range(3)]
        result = annotate_steps_by_task_success(steps, task_success=False)
        # No error signal found: all steps labelled correct (conservative)
        assert all(s.is_correct for s in result)


# ---------------------------------------------------------------------------
# annotate_gsm8k_steps
# ---------------------------------------------------------------------------

class TestAnnotateGSM8K:

    def test_correct_arithmetic_step(self):
        steps = [_make_step(0, "6", expression="2*3")]
        result = annotate_gsm8k_steps(steps, task_success=True)
        assert result[0].is_correct is True

    def test_incorrect_arithmetic_step(self):
        steps = [_make_step(0, "7", expression="2*3")]  # 2*3 = 6, not 7
        result = annotate_gsm8k_steps(steps, task_success=True)
        assert result[0].is_correct is False

    def test_error_observation_is_incorrect(self):
        steps = [_make_step(0, "ERROR: Division by zero", expression="5/0")]
        result = annotate_gsm8k_steps(steps, task_success=False)
        assert result[0].is_correct is False

    def test_non_numeric_observation_uses_task_success(self):
        steps = [_make_step(0, "The answer is 42.", expression="")]
        result = annotate_gsm8k_steps(steps, task_success=True)
        assert result[0].is_correct is True

    def test_float_tolerance(self):
        # 10 / 3 ≈ 3.333333; model might return 3.33 (within 1e-3? No, 0.003 > 1e-3)
        steps = [_make_step(0, "3.333333", expression="10/3")]
        result = annotate_gsm8k_steps(steps, task_success=True)
        expected = 10 / 3  # ≈ 3.3333333
        actual = 3.333333
        within_tol = abs(expected - actual) < 1e-3
        assert result[0].is_correct == within_tol
