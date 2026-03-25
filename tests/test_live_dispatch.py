"""Tests for live_dispatch module — bridge to Claude Code Agent tool."""

import pytest
from unittest.mock import patch, MagicMock

from reticulate.live_dispatch import (
    DispatchBatch,
    DispatchPlan,
    plan_dispatch,
    format_agent_call,
    format_batch_description,
    post_dispatch,
    dispatch_now,
)
from reticulate.agent_tasks import TaskCard


# ---------------------------------------------------------------------------
# DispatchBatch
# ---------------------------------------------------------------------------

class TestDispatchBatch:
    def test_create(self):
        card = TaskCard("Writer", "1", "expand_paper", "prompt")
        batch = DispatchBatch(batch_number=1, cards=(card,), description="test")
        assert batch.size == 1
        assert batch.batch_number == 1

    def test_frozen(self):
        batch = DispatchBatch(batch_number=1, cards=(), description="test")
        with pytest.raises(AttributeError):
            batch.batch_number = 2  # type: ignore

    def test_multiple_cards(self):
        cards = (
            TaskCard("Writer", "1", "expand", "p1"),
            TaskCard("Writer", "2", "expand", "p2"),
            TaskCard("Prover", "3", "create", "p3"),
        )
        batch = DispatchBatch(batch_number=1, cards=cards, description="content")
        assert batch.size == 3


# ---------------------------------------------------------------------------
# DispatchPlan
# ---------------------------------------------------------------------------

class TestDispatchPlan:
    def test_empty_plan(self):
        plan = DispatchPlan()
        assert plan.all_clear is True
        assert plan.total_cards == 0

    def test_plan_with_cards(self):
        plan = DispatchPlan(total_cards=5, steps_failing=["1", "2"])
        assert plan.all_clear is False

    def test_summary_all_clear(self):
        plan = DispatchPlan(accepted_before=84, total_steps=84)
        s = plan.summary()
        assert "All clear" in s
        assert "84/84" in s

    def test_summary_with_work(self):
        card = TaskCard("Writer", "1", "expand", "prompt")
        batch = DispatchBatch(1, (card,), "test batch")
        plan = DispatchPlan(
            batches=[batch],
            total_cards=1,
            accepted_before=83,
            total_steps=84,
            steps_failing=["1"],
        )
        s = plan.summary()
        assert "1 cards" in s or "1 card" in s
        assert "83/84" in s
        assert "Writer" in s


# ---------------------------------------------------------------------------
# plan_dispatch (integration)
# ---------------------------------------------------------------------------

class TestPlanDispatch:
    def test_all_clear(self):
        """Real codebase: all steps should be A+."""
        plan = plan_dispatch()
        assert plan.all_clear is True
        assert plan.accepted_before == plan.total_steps
        assert len(plan.batches) == 0

    def test_with_failures(self):
        """Simulate failing steps to test batch grouping."""
        mock_results = [
            {"step": "1", "grade": "A+", "score": 100, "accepted": True, "fixes": []},
            {"step": "999", "grade": "B", "score": 70, "accepted": False,
             "fixes": ["[MINOR] Word count too low"]},
        ]
        fake_cards = [TaskCard("Writer", "999", "expand", "prompt",
                              transitions=("writePaper", "paperReady"))]
        with patch("reticulate.live_dispatch._evaluate_all", return_value=mock_results):
            with patch("reticulate.live_dispatch.generate_task_cards", return_value=fake_cards):
                plan = plan_dispatch()

        assert not plan.all_clear
        assert plan.accepted_before == 1
        assert plan.total_steps == 2
        assert "999" in plan.steps_failing

    def test_batch_grouping(self):
        """Cards should be grouped by type: writers first, then impl, then test."""
        cards = [
            TaskCard("Writer", "1", "expand", "p", transitions=("writePaper", "paperReady")),
            TaskCard("Prover", "1", "create", "p", transitions=("writeProofs", "proofsReady")),
            TaskCard("Implementer", "1", "create", "p", transitions=("implement", "moduleReady")),
            TaskCard("Tester", "1", "create", "p", transitions=("writeTests", "testsPass")),
        ]
        mock_results = [
            {"step": "1", "grade": "B", "score": 70, "accepted": False,
             "fixes": ["word count low", "proofs missing", "module missing", "tests missing"]},
        ]
        with patch("reticulate.live_dispatch._evaluate_all", return_value=mock_results):
            with patch("reticulate.live_dispatch.generate_task_cards", return_value=cards):
                plan = plan_dispatch()

        # Should have 3 batches: Writer+Prover, Implementer, Tester
        assert len(plan.batches) == 3
        assert plan.batches[0].cards[0].agent_type in ("Writer", "Prover")
        assert plan.batches[1].cards[0].agent_type == "Implementer"
        assert plan.batches[2].cards[0].agent_type == "Tester"


# ---------------------------------------------------------------------------
# format_agent_call
# ---------------------------------------------------------------------------

class TestFormatAgentCall:
    def test_basic_format(self):
        card = TaskCard(
            agent_type="Writer",
            step_number="23",
            action="expand_paper",
            prompt="Expand the paper for Step 23...",
            context_files=("papers/steps/step23/main.tex",),
            output_files=("papers/steps/step23/main.tex",),
            acceptance_criteria=("5000+ words",),
            transitions=("writePaper", "paperReady"),
        )
        result = format_agent_call(card)
        assert "Expand the paper" in result
        assert "Files to read" in result
        assert "Files to create" in result
        assert "Acceptance criteria" in result
        assert "5000+ words" in result
        assert "writePaper → paperReady" in result

    def test_minimal_card(self):
        card = TaskCard("Writer", "1", "expand", "Do the work")
        result = format_agent_call(card)
        assert "Do the work" in result
        assert "Important" in result

    def test_includes_guidelines(self):
        card = TaskCard("Writer", "1", "expand", "prompt")
        result = format_agent_call(card)
        assert "placeholder" in result.lower()
        assert "CLAUDE.md" in result


# ---------------------------------------------------------------------------
# format_batch_description
# ---------------------------------------------------------------------------

class TestFormatBatchDescription:
    def test_single_agent(self):
        cards = (TaskCard("Writer", "1", "expand", "p"),)
        batch = DispatchBatch(1, cards, "test")
        desc = format_batch_description(batch)
        assert "Writer" in desc
        assert "1" in desc

    def test_multiple_agents(self):
        cards = (
            TaskCard("Writer", "1", "expand", "p"),
            TaskCard("Prover", "2", "create", "p"),
        )
        batch = DispatchBatch(1, cards, "test")
        desc = format_batch_description(batch)
        assert "2" in desc  # 2 agents


# ---------------------------------------------------------------------------
# post_dispatch
# ---------------------------------------------------------------------------

class TestPostDispatch:
    def test_post_dispatch_returns_report(self):
        report = post_dispatch(commit=False, push=False)
        assert "accepted" in report
        assert "total" in report
        assert "all_clear" in report
        assert report["accepted"] > 0

    def test_post_dispatch_still_failing(self):
        mock_results = [
            {"step": "999", "grade": "B", "score": 70, "accepted": False,
             "fixes": ["fix me"]},
        ]
        with patch("reticulate.live_dispatch._evaluate_all", return_value=mock_results):
            report = post_dispatch(commit=False, push=False)

        assert report["all_clear"] is False
        assert len(report["still_failing"]) == 1


# ---------------------------------------------------------------------------
# dispatch_now (main entry point)
# ---------------------------------------------------------------------------

class TestDispatchNow:
    def test_all_clear(self):
        result = dispatch_now()
        assert result["all_clear"] is True
        assert "summary" in result

    def test_with_work(self):
        mock_results = [
            {"step": "999", "grade": "B", "score": 70, "accepted": False,
             "fixes": ["[MINOR] Word count too low"]},
        ]
        fake_cards = [TaskCard("Writer", "999", "expand", "prompt",
                              transitions=("writePaper", "paperReady"))]
        with patch("reticulate.live_dispatch._evaluate_all", return_value=mock_results):
            with patch("reticulate.live_dispatch.generate_task_cards", return_value=fake_cards):
                result = dispatch_now()

        assert result["all_clear"] is False
        assert result["accepted"] == 0
        assert result["total"] == 1
        assert "summary" in result

    def test_batches_have_prompts(self):
        cards = [
            TaskCard("Writer", "1", "expand", "Write stuff",
                    context_files=("a.tex",), output_files=("a.tex",),
                    acceptance_criteria=("5000 words",),
                    transitions=("writePaper", "paperReady")),
        ]
        mock_results = [
            {"step": "1", "grade": "B", "score": 70, "accepted": False,
             "fixes": ["word count low"]},
        ]
        with patch("reticulate.live_dispatch._evaluate_all", return_value=mock_results):
            with patch("reticulate.live_dispatch.generate_task_cards", return_value=cards):
                result = dispatch_now()

        assert "batches" in result
        batch = result["batches"][0]
        assert len(batch["agents"]) == 1
        agent = batch["agents"][0]
        assert "prompt" in agent
        assert "Write stuff" in agent["prompt"]
        assert agent["agent_type"] == "Writer"
        assert agent["step_number"] == "1"
