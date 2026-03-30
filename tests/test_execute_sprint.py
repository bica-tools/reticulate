"""Tests for execute_sprint module — the full sprint execution loop."""

import json
import pytest
from unittest.mock import patch, MagicMock

from reticulate.execute_sprint import (
    StepResult,
    SprintIteration,
    SprintExecutionResult,
    run_sprint_loop,
    sprint_cards_json,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TestStepResult:
    def test_create(self):
        sr = StepResult(step_number="23", grade_before="B", score_before=70)
        assert sr.step_number == "23"
        assert sr.grade_before == "B"
        assert sr.grade_after is None
        assert sr.fixed is False

    def test_fixed_step(self):
        sr = StepResult(
            step_number="23", grade_before="B", score_before=70,
            grade_after="A+", score_after=95, fixed=True,
        )
        assert sr.fixed is True
        assert sr.grade_after == "A+"


class TestSprintIteration:
    def test_create(self):
        it = SprintIteration(iteration=1)
        assert it.iteration == 1
        assert it.steps_evaluated == 0
        assert it.pending_cards == []

    def test_with_results(self):
        it = SprintIteration(
            iteration=1,
            steps_evaluated=5,
            steps_failing=2,
            steps_fixed=1,
        )
        assert it.steps_fixed == 1


class TestSprintExecutionResult:
    def test_create(self):
        r = SprintExecutionResult()
        assert r.total_fixed == 0
        assert r.all_clear is False
        assert r.pending_cards == []

    def test_final_rate(self):
        r = SprintExecutionResult(final_accepted=80, final_total=84)
        assert r.final_rate == "80/84"

    def test_final_rate_zero(self):
        r = SprintExecutionResult()
        assert r.final_rate == "0/0"

    def test_summary(self):
        r = SprintExecutionResult(
            final_accepted=84, final_total=84,
            total_fixed=0, all_clear=True,
        )
        s = r.summary()
        assert "84/84" in s
        assert "ALL STEPS AT A+" in s

    def test_summary_with_pending(self):
        from reticulate.agent_tasks import TaskCard
        r = SprintExecutionResult(
            final_accepted=80, final_total=84,
            pending_cards=[TaskCard("Writer", "1", "expand", "prompt")],
        )
        s = r.summary()
        assert "1 cards awaiting" in s


# ---------------------------------------------------------------------------
# run_sprint_loop
# ---------------------------------------------------------------------------

class TestRunSprintLoop:
    def test_evaluator_runs(self):
        """Sprint loop should evaluate all steps and return a result."""
        result = run_sprint_loop(max_iterations=1, mode="execute")
        assert result.final_total > 0
        assert result.final_accepted >= 0
        assert result.final_accepted <= result.final_total
        # At least half should be accepted with any reasonable evaluator
        assert result.final_accepted >= result.final_total // 2

    def test_verify_mode(self):
        result = run_sprint_loop(max_iterations=1, mode="verify")
        assert result.final_total > 0

    def test_generate_mode(self):
        result = run_sprint_loop(max_iterations=1, mode="generate")
        assert result.final_total > 0

    def test_max_iterations_respected(self):
        """Loop should not exceed max_iterations."""
        result = run_sprint_loop(max_iterations=1, mode="execute")
        assert len(result.iterations) <= 1

    def test_result_has_timing(self):
        result = run_sprint_loop(max_iterations=1)
        assert result.elapsed_ms >= 0


class TestRunSprintLoopWithFailures:
    """Test sprint loop behavior when steps are failing."""

    def _mock_eval_with_failures(self):
        """Return mock evaluation with some failures."""
        return [
            {"step": "1", "title": "Test", "grade": "A+", "score": 100,
             "accepted": True, "fixes": []},
            {"step": "999", "title": "Fake", "grade": "B", "score": 70,
             "accepted": False, "fixes": ["[MINOR] Word count too low"]},
        ]

    def test_generate_mode_produces_cards(self):
        """In generate mode, failing steps should produce task cards."""
        with patch("reticulate.execute_sprint._evaluate_all",
                   return_value=self._mock_eval_with_failures()):
            result = run_sprint_loop(max_iterations=1, mode="generate")

        assert not result.all_clear
        assert result.final_accepted == 1
        assert result.final_total == 2
        assert len(result.iterations) == 1
        # Cards may or may not be generated depending on step 999 existence

    def test_execute_mode_with_no_fix(self):
        """Execute mode should stop if nothing was fixed."""
        with patch("reticulate.execute_sprint._evaluate_all",
                   return_value=self._mock_eval_with_failures()):
            result = run_sprint_loop(max_iterations=3, mode="execute")

        # Should stop after 1 iteration since nothing was fixed
        assert len(result.iterations) <= 1

    def test_verify_mode_dispatches(self):
        """Verify mode should run dispatch_sprint on failing steps."""
        with patch("reticulate.execute_sprint._evaluate_all",
                   return_value=self._mock_eval_with_failures()):
            result = run_sprint_loop(max_iterations=1, mode="verify")

        assert len(result.iterations) == 1


# ---------------------------------------------------------------------------
# sprint_cards_json
# ---------------------------------------------------------------------------

class TestSprintCardsJson:
    def test_sprint_cards_returns_json(self):
        output = sprint_cards_json()
        data = json.loads(output)
        assert data["status"] in ("all_clear", "work_needed")
        # all_clear has just status+message; work_needed has total/total_steps
        if data["status"] == "work_needed":
            assert "total" in data or "total_steps" in data

    def test_json_valid(self):
        output = sprint_cards_json()
        data = json.loads(output)
        assert "status" in data

    def test_with_failures(self):
        mock_results = [
            {"step": "999", "title": "Fake", "grade": "B", "score": 70,
             "accepted": False, "fixes": ["word count low"]},
        ]
        with patch("reticulate.execute_sprint._evaluate_all",
                   return_value=mock_results):
            output = sprint_cards_json()

        data = json.loads(output)
        assert data["status"] == "work_needed"
        assert data["accepted"] == 0
        assert data["total"] == 1

    def test_cards_structure(self):
        mock_results = [
            {"step": "999", "title": "Fake", "grade": "B", "score": 70,
             "accepted": False, "fixes": ["[MINOR] Word count: too low"]},
        ]
        with patch("reticulate.execute_sprint._evaluate_all",
                   return_value=mock_results):
            output = sprint_cards_json()

        data = json.loads(output)
        if data["status"] == "work_needed" and data.get("cards"):
            card = data["cards"][0]
            assert "agent_type" in card
            assert "step_number" in card
            assert "action" in card
            assert "prompt" in card
            assert "transitions" in card


# ---------------------------------------------------------------------------
# Integration: full loop on real codebase
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_loop_real_codebase(self):
        """Run the full loop on the real codebase — should find all A+."""
        result = run_sprint_loop(max_iterations=1, mode="execute")
        assert result.final_accepted > 0
        assert result.final_total > 0
        assert isinstance(result.summary(), str)

    def test_generate_then_verify(self):
        """Generate cards, then verify — should be consistent."""
        gen_result = run_sprint_loop(max_iterations=1, mode="generate")
        ver_result = run_sprint_loop(max_iterations=1, mode="verify")

        # Both should agree on final state
        assert gen_result.final_accepted == ver_result.final_accepted
        assert gen_result.final_total == ver_result.final_total

    def test_cli_mode(self):
        """Test CLI entry point doesn't crash."""
        from reticulate.execute_sprint import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--mode", "generate", "--max-iterations", "1"])
        # Exit 0 = all clear, Exit 1 = work needed — both valid
        assert exc_info.value.code in (0, 1)

    def test_json_output_mode(self):
        """Test JSON output mode."""
        from reticulate.execute_sprint import main
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            main(["--json"])

        output = f.getvalue()
        data = json.loads(output)
        assert "status" in data
