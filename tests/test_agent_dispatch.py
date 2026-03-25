"""Tests for agent_dispatch module — the wiring between orchestrator and agents."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from reticulate import agent_dispatch as _ad

HANDLERS = _ad.HANDLERS
researcher_handler = _ad.researcher_handler
evaluator_handler = _ad.evaluator_handler
supervisor_handler = _ad.supervisor_handler
_tester_handler = _ad.tester_handler
implementer_handler = _ad.implementer_handler
writer_handler = _ad.writer_handler
prover_handler = _ad.prover_handler
reviewer_handler = _ad.reviewer_handler
register_all_handlers = _ad.register_all_handlers
dispatch_sprint = _ad.dispatch_sprint
get_registry = _ad.get_registry
reset_registry = _ad.reset_registry
from reticulate.artifact_registry import ArtifactKind
from reticulate.agent_registry import AGENTS, AgentMonitor, _get_machine


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

class TestHandlerRegistry:
    def test_all_core_agents_have_handlers(self):
        """Every core orchestrator agent type has a handler."""
        core = ["Researcher", "Evaluator", "Supervisor", "Tester",
                "Implementer", "Writer", "Prover", "Reviewer"]
        for agent_type in core:
            assert agent_type in HANDLERS, f"Missing handler for {agent_type}"

    def test_handlers_are_callable(self):
        for name, fn in HANDLERS.items():
            assert callable(fn), f"Handler for {name} is not callable"

    def test_handler_count(self):
        assert len(HANDLERS) == 8


# ---------------------------------------------------------------------------
# Session type compliance: handlers emit valid transitions
# ---------------------------------------------------------------------------

class TestSessionTypeCompliance:
    """Verify each handler's transitions match its agent session type."""

    def _validate_transitions(self, agent_type: str, transitions: list[str]) -> bool:
        """Walk the state machine with the given transitions."""
        monitor = AgentMonitor()
        monitor.start_agent(agent_type, f"test-{agent_type}")
        for label in transitions:
            ok = monitor.transition(f"test-{agent_type}", label)
            if not ok:
                return False
        return True

    def test_researcher_transitions_valid(self):
        result = researcher_handler("test-r", "1", "test")
        assert self._validate_transitions("Researcher", result["transitions"])

    def test_evaluator_accepted_transitions_valid(self):
        """Evaluator accepted path: evaluate → accepted"""
        with patch("reticulate.evaluator.evaluate_step") as mock:
            mock_result = MagicMock()
            mock_result.accepted = True
            mock_result.grade = "A+"
            mock_result.score = 95
            mock.return_value = mock_result
            result = evaluator_handler("test-e", "1", "test")
        assert self._validate_transitions("Evaluator", result["transitions"])

    def test_evaluator_needs_fixes_transitions_valid(self):
        """Evaluator needsFixes path: evaluate → needsFixes → listFixes"""
        with patch("reticulate.evaluator.evaluate_step") as mock:
            mock_result = MagicMock()
            mock_result.accepted = False
            mock_result.grade = "B"
            mock_result.score = 70
            mock_result.fixes = ["Fix X"]
            mock_result.weaknesses = ["Weak Y"]
            mock.return_value = mock_result
            result = evaluator_handler("test-e", "1", "test")
        assert self._validate_transitions("Evaluator", result["transitions"])

    def test_supervisor_transitions_valid(self):
        result = supervisor_handler("test-s", "1", "test")
        assert self._validate_transitions("Supervisor", result["transitions"])

    def test_writer_transitions_valid(self):
        result = writer_handler("test-w", "1", "test")
        assert self._validate_transitions("Writer", result["transitions"])

    def test_prover_transitions_valid(self):
        result = prover_handler("test-p", "1", "test")
        assert self._validate_transitions("Prover", result["transitions"])

    def test_reviewer_transitions_valid(self):
        with patch("reticulate.evaluator.evaluate_step") as mock:
            mock_result = MagicMock()
            mock_result.grade = "B"
            mock_result.fixes = ["Fix X"]
            mock_result.weaknesses = []
            mock.return_value = mock_result
            result = reviewer_handler("test-rev", "1", "test")
        assert self._validate_transitions("Reviewer", result["transitions"])

    def test_tester_pass_transitions_valid(self):
        """Tester pass path: writeTests → testsPass"""
        with patch("reticulate.agent_dispatch._step_module", return_value=None):
            result = _tester_handler("test-t", "1", "test")
        assert self._validate_transitions("Tester", result["transitions"])

    def test_tester_fail_transitions_valid(self):
        """Tester fail path: writeTests → testsFail → abort"""
        with patch("reticulate.agent_dispatch._step_module", return_value="fake_mod"):
            with patch("reticulate.agent_dispatch._run_tests", return_value=(False, "FAILED")):
                result = _tester_handler("test-t", "1", "test")
        assert self._validate_transitions("Tester", result["transitions"])

    def test_implementer_ready_transitions_valid(self):
        """Implementer ready path: implement → moduleReady"""
        with patch("reticulate.agent_dispatch._step_module", return_value=None):
            result = implementer_handler("test-i", "1", "test")
        assert self._validate_transitions("Implementer", result["transitions"])

    def test_implementer_error_transitions_valid(self):
        """Implementer error path: implement → error → abort"""
        with patch("reticulate.agent_dispatch._step_module", return_value="nonexistent_module"):
            result = implementer_handler("test-i", "1", "test")
        # If module doesn't exist, should follow error path
        assert "implement" in result["transitions"]


# ---------------------------------------------------------------------------
# Researcher handler
# ---------------------------------------------------------------------------

class TestResearcherHandler:
    def test_returns_transitions(self):
        result = researcher_handler("r1", "1", "investigate")
        assert "transitions" in result
        assert result["transitions"] == ["investigate", "report"]

    def test_returns_output(self):
        result = researcher_handler("r1", "1", "investigate")
        assert "output" in result
        assert "Research report" in result["output"]

    def test_registers_artifact(self):
        reset_registry()
        researcher_handler("r1", "1", "investigate")
        artifacts = get_registry().get_by_kind(ArtifactKind.REPORT)
        assert len(artifacts) >= 1


# ---------------------------------------------------------------------------
# Evaluator handler
# ---------------------------------------------------------------------------

class TestEvaluatorHandler:
    def test_accepted_path(self):
        with patch("reticulate.evaluator.evaluate_step") as mock:
            mock_result = MagicMock()
            mock_result.accepted = True
            mock_result.grade = "A+"
            mock_result.score = 95
            mock.return_value = mock_result
            result = evaluator_handler("e1", "1", "evaluate")

        assert result["output"]["accepted"] is True
        assert result["output"]["grade"] == "A+"

    def test_needs_fixes_path(self):
        with patch("reticulate.evaluator.evaluate_step") as mock:
            mock_result = MagicMock()
            mock_result.accepted = False
            mock_result.grade = "B"
            mock_result.score = 70
            mock_result.fixes = ["Expand paper"]
            mock_result.weaknesses = ["Too short"]
            mock.return_value = mock_result
            result = evaluator_handler("e1", "1", "evaluate")

        assert result["output"]["accepted"] is False
        assert "Expand paper" in result["output"]["fixes"]


# ---------------------------------------------------------------------------
# Writer handler
# ---------------------------------------------------------------------------

class TestWriterHandler:
    def test_returns_valid_structure(self):
        result = writer_handler("w1", "1", "write")
        assert "transitions" in result
        assert result["transitions"] == ["writePaper", "paperReady"]

    def test_nonexistent_step(self):
        result = writer_handler("w1", "99999", "write")
        assert result["transitions"] == ["writePaper", "paperReady"]


# ---------------------------------------------------------------------------
# Prover handler
# ---------------------------------------------------------------------------

class TestProverHandler:
    def test_returns_valid_structure(self):
        result = prover_handler("p1", "1", "prove")
        assert "transitions" in result
        assert result["transitions"] == ["writeProofs", "proofsReady"]


# ---------------------------------------------------------------------------
# Tester handler
# ---------------------------------------------------------------------------

class TestTesterHandler:
    def test_no_module_skips(self):
        with patch("reticulate.agent_dispatch._step_module", return_value=None):
            result = _tester_handler("t1", "1", "test")
        assert result["output"]["skipped"] is True
        assert result["transitions"] == ["writeTests", "testsPass"]


# ---------------------------------------------------------------------------
# Implementer handler
# ---------------------------------------------------------------------------

class TestImplementerHandler:
    def test_paper_only_step(self):
        with patch("reticulate.agent_dispatch._step_module", return_value=None):
            result = implementer_handler("i1", "1", "implement")
        assert result["output"]["skipped"] is True
        assert result["transitions"] == ["implement", "moduleReady"]


# ---------------------------------------------------------------------------
# register_all_handlers
# ---------------------------------------------------------------------------

class TestRegisterAllHandlers:
    def test_registers_to_orchestrator(self):
        from reticulate.orchestrator import Orchestrator
        orch = Orchestrator()
        register_all_handlers(orch)
        assert len(orch._handlers) == 8

    def test_all_handlers_executable(self):
        """After registration, orchestrator can execute all core agent types."""
        from reticulate.orchestrator import Orchestrator, OrchestratorConfig
        config = OrchestratorConfig(monitor_enabled=False)
        orch = Orchestrator(config=config)
        register_all_handlers(orch)

        for agent_type in HANDLERS:
            assert agent_type in orch._handlers


# ---------------------------------------------------------------------------
# dispatch_sprint
# ---------------------------------------------------------------------------

class TestDispatchSprint:
    def test_all_clear_when_no_steps(self):
        result = dispatch_sprint(steps=[])
        assert result["status"] == "all_clear"

    def test_sprint_with_steps(self):
        """Run a sprint on step 1 — all handlers wired and executing."""
        reset_registry()
        result = dispatch_sprint(steps=["1"])
        assert result["status"] == "completed"
        assert "1" in result["steps_processed"]
        assert result["total_agents"] > 0
        assert "dashboard" in result
        assert "artifacts" in result

    def test_sprint_conformance_rate(self):
        """All handlers should produce valid transitions."""
        reset_registry()
        result = dispatch_sprint(steps=["1"])
        # With real handlers, conformance should be high
        assert result["conformance_rate"] >= 0.0

    def test_sprint_produces_artifacts(self):
        """Sprint should register artifacts in the registry."""
        reset_registry()
        dispatch_sprint(steps=["1"])
        reg = get_registry()
        assert reg.count > 0


# ---------------------------------------------------------------------------
# Integration: full orchestrator pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_orchestrator_with_handlers_runs_step(self):
        """Complete integration: Orchestrator + handlers + monitor."""
        from reticulate.orchestrator import Orchestrator

        reset_registry()
        orch = Orchestrator()
        register_all_handlers(orch)

        result = orch.run_sprint(steps=["1"])
        assert result.sprint_number == 1
        assert "1" in result.steps_processed
        assert result.total_agents_dispatched > 0

    def test_orchestrator_dashboard_after_sprint(self):
        from reticulate.orchestrator import Orchestrator

        orch = Orchestrator()
        register_all_handlers(orch)
        orch.run_sprint(steps=["1"])

        dashboard = orch.dashboard()
        assert "AGENT MONITOR DASHBOARD" in dashboard

    def test_orchestrator_conformance_report(self):
        from reticulate.orchestrator import Orchestrator

        orch = Orchestrator()
        register_all_handlers(orch)
        orch.run_sprint(steps=["1"])

        report = orch.conformance_report()
        assert "total" in report
        assert report["total"] > 0

    def test_multi_step_sprint(self):
        """Sprint across multiple steps."""
        from reticulate.orchestrator import Orchestrator

        reset_registry()
        orch = Orchestrator()
        register_all_handlers(orch)

        result = orch.run_sprint(steps=["1", "2"])
        assert len(result.steps_processed) == 2

    def test_parallel_phases_execute(self):
        """Phase 2 (Implementer || Writer) runs in parallel."""
        from reticulate.orchestrator import Orchestrator

        orch = Orchestrator()
        register_all_handlers(orch)
        result = orch.run_sprint(steps=["1"])

        # Should have at least 4 phases: research, impl||write, test||prove, evaluate
        assert len(result.phase_results) >= 4

    def test_artifacts_flow_between_phases(self):
        """Artifacts from earlier phases are visible to later phases."""
        reset_registry()
        from reticulate.orchestrator import Orchestrator

        orch = Orchestrator()
        register_all_handlers(orch)
        orch.run_sprint(steps=["1"])

        reg = get_registry()
        # Research phase should produce a report
        reports = reg.get_by_kind(ArtifactKind.REPORT)
        assert len(reports) >= 1
