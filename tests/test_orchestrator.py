"""Tests for the monitored parallel orchestrator.

Covers:
- OrchestratorConfig defaults and customization
- Monitored agent lifecycle (start -> transitions -> complete)
- Parallel phase dispatch
- Violation detection and rejection
- Sprint execution with mock agents
- Dashboard output
- Updated ORCHESTRATOR_TYPE lattice verification
"""

import pytest

from reticulate.agent_registry import (
    AGENTS,
    AgentMonitor,
    ORCHESTRATOR_TYPE,
    _build_state_machine,
    _get_initial_state,
)
from reticulate.lattice import check_lattice
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.orchestrator import (
    AgentResult,
    AgentTask,
    Orchestrator,
    OrchestratorConfig,
    ParallelPhaseResult,
    SprintResult,
)


# ---------------------------------------------------------------------------
# Helper: mock agent handlers
# ---------------------------------------------------------------------------

def _make_handler(transitions: list[str], output: dict | None = None):
    """Return a handler that records the given transitions."""
    def handler(agent_id: str, step_number: str, task: str):
        result = {"transitions": transitions}
        if output:
            result.update(output)
        return result
    return handler


def _make_failing_handler(error_msg: str = "boom"):
    """Return a handler that raises an exception."""
    def handler(agent_id: str, step_number: str, task: str):
        raise RuntimeError(error_msg)
    return handler


def _make_violating_handler(good_transitions: list[str], bad_label: str):
    """Return a handler that sends valid transitions then a violation."""
    def handler(agent_id: str, step_number: str, task: str):
        return {"transitions": good_transitions + [bad_label]}
    return handler


# ---------------------------------------------------------------------------
# ORCHESTRATOR_TYPE lattice verification
# ---------------------------------------------------------------------------

class TestOrchestratorType:
    """Verify the updated ORCHESTRATOR_TYPE with parallel constructors."""

    def test_parses(self):
        ast = parse(ORCHESTRATOR_TYPE)
        assert ast is not None

    def test_builds_statespace(self):
        ast = parse(ORCHESTRATOR_TYPE)
        ss = build_statespace(ast)
        assert len(ss.states) > 0
        assert len(ss.transitions) > 0

    def test_is_lattice(self):
        ast = parse(ORCHESTRATOR_TYPE)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice, f"ORCHESTRATOR_TYPE is not a lattice: {lr.counterexample}"

    def test_has_parallel_states(self):
        """The new type should have more states than a purely sequential version."""
        ast = parse(ORCHESTRATOR_TYPE)
        ss = build_statespace(ast)
        # With parallel constructors, we get product states
        assert len(ss.states) >= 20

    def test_contains_parallel_keyword(self):
        assert "||" in ORCHESTRATOR_TYPE


# ---------------------------------------------------------------------------
# OrchestratorConfig
# ---------------------------------------------------------------------------

class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.max_parallel == 4
        assert cfg.monitor_enabled is True
        assert cfg.reject_violations is True
        assert cfg.max_retry == 3

    def test_custom(self):
        cfg = OrchestratorConfig(max_parallel=8, reject_violations=False)
        assert cfg.max_parallel == 8
        assert cfg.reject_violations is False


# ---------------------------------------------------------------------------
# AgentResult / ParallelPhaseResult / SprintResult dataclasses
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_agent_result_basic(self):
        r = AgentResult(
            agent_id="test-1", agent_type="Researcher",
            step_number="1", success=True,
        )
        assert r.success
        assert not r.violated

    def test_agent_result_violated(self):
        r = AgentResult(
            agent_id="test-1", agent_type="Researcher",
            step_number="1", success=False, violated=True,
            violation_detail="bad transition",
        )
        assert r.violated
        assert "bad" in r.violation_detail

    def test_parallel_phase_all_completed(self):
        pr = ParallelPhaseResult(
            phase_name="test", agents=["A", "B"],
            results=[
                AgentResult("a", "A", "1", success=True),
                AgentResult("b", "B", "1", success=True),
            ],
        )
        assert pr.all_completed
        assert not pr.any_violated

    def test_parallel_phase_with_failure(self):
        pr = ParallelPhaseResult(
            phase_name="test", agents=["A", "B"],
            results=[
                AgentResult("a", "A", "1", success=True),
                AgentResult("b", "B", "1", success=False),
            ],
        )
        assert not pr.all_completed

    def test_parallel_phase_with_violation(self):
        pr = ParallelPhaseResult(
            phase_name="test", agents=["A", "B"],
            results=[
                AgentResult("a", "A", "1", success=True),
                AgentResult("b", "B", "1", success=False, violated=True),
            ],
        )
        assert pr.any_violated
        assert len(pr.succeeded_results) == 1

    def test_parallel_phase_empty(self):
        pr = ParallelPhaseResult(phase_name="empty", agents=[])
        assert pr.all_completed
        assert not pr.any_violated

    def test_sprint_result_conformance_rate(self):
        sr = SprintResult(sprint_number=1)
        sr.phase_results = [
            ParallelPhaseResult("p1", ["A"], [
                AgentResult("a", "A", "1", success=True),
            ]),
            ParallelPhaseResult("p2", ["B"], [
                AgentResult("b", "B", "1", success=False, violated=True),
            ]),
        ]
        assert sr.conformance_rate == 0.5
        assert sr.total_agents_dispatched == 2

    def test_sprint_result_no_agents(self):
        sr = SprintResult(sprint_number=1)
        assert sr.conformance_rate == 1.0
        assert sr.total_agents_dispatched == 0


# ---------------------------------------------------------------------------
# Monitored agent lifecycle
# ---------------------------------------------------------------------------

class TestMonitoredLifecycle:
    def test_start_agent(self):
        orch = Orchestrator()
        aid = orch._start_monitored_agent("Researcher", "1", "test")
        assert "researcher" in aid
        assert "1" in aid
        assert aid in orch.monitor.instances

    def test_start_agent_monitoring_disabled(self):
        cfg = OrchestratorConfig(monitor_enabled=False)
        orch = Orchestrator(cfg)
        aid = orch._start_monitored_agent("Researcher", "1", "test")
        assert aid  # still returns an id
        assert len(orch.monitor.instances) == 0

    def test_execute_agent_with_valid_transitions(self):
        orch = Orchestrator()
        orch.register_handler("Researcher", _make_handler(["investigate", "report"]))
        result = orch._execute_agent("Researcher", "1", "test")
        assert result.success
        assert not result.violated
        assert result.elapsed_ms >= 0

    def test_execute_agent_with_violation(self):
        orch = Orchestrator()
        orch.register_handler("Researcher", _make_handler(["investigate", "INVALID"]))
        result = orch._execute_agent("Researcher", "1", "test")
        assert not result.success
        assert result.violated
        assert result.violation_detail is not None

    def test_execute_agent_exception(self):
        orch = Orchestrator()
        orch.register_handler("Researcher", _make_failing_handler("kaboom"))
        result = orch._execute_agent("Researcher", "1", "test")
        assert not result.success
        assert "kaboom" in str(result.output)

    def test_execute_agent_no_handler(self):
        orch = Orchestrator()
        result = orch._execute_agent("Researcher", "1", "test")
        assert not result.success
        assert "No handler" in str(result.output)

    def test_violation_not_rejected_when_disabled(self):
        cfg = OrchestratorConfig(reject_violations=False)
        orch = Orchestrator(cfg)
        orch.register_handler("Researcher", _make_handler(["investigate", "INVALID"]))
        result = orch._execute_agent("Researcher", "1", "test")
        # violated but success=True because reject_violations=False
        assert result.success
        assert result.violated


# ---------------------------------------------------------------------------
# Parallel phase dispatch
# ---------------------------------------------------------------------------

class TestParallelPhase:
    def test_empty_phase(self):
        orch = Orchestrator()
        pr = orch._parallel_phase("empty", [])
        assert pr.all_completed
        assert len(pr.results) == 0

    def test_single_agent_phase(self):
        orch = Orchestrator()
        orch.register_handler("Writer", _make_handler(["writePaper", "paperReady"]))
        pr = orch._parallel_phase("write", [
            AgentTask("Writer", "1", "Write paper"),
        ])
        assert len(pr.results) == 1
        assert pr.all_completed

    def test_two_agents_parallel(self):
        orch = Orchestrator()
        orch.register_handler("Implementer", _make_handler(["implement", "moduleReady"]))
        orch.register_handler("Writer", _make_handler(["writePaper", "paperReady"]))
        pr = orch._parallel_phase("impl-write", [
            AgentTask("Implementer", "1", "Implement"),
            AgentTask("Writer", "1", "Write"),
        ])
        assert len(pr.results) == 2
        assert pr.all_completed
        assert not pr.any_violated

    def test_parallel_with_one_violation(self):
        orch = Orchestrator()
        orch.register_handler("Implementer", _make_handler(["implement", "moduleReady"]))
        orch.register_handler("Writer", _make_handler(["writePaper", "BADLABEL"]))
        pr = orch._parallel_phase("impl-write", [
            AgentTask("Implementer", "1", "Implement"),
            AgentTask("Writer", "1", "Write"),
        ])
        assert len(pr.results) == 2
        assert not pr.all_completed
        assert pr.any_violated
        assert len(pr.succeeded_results) == 1

    def test_parallel_respects_max_parallel(self):
        cfg = OrchestratorConfig(max_parallel=1)
        orch = Orchestrator(cfg)
        orch.register_handler("Implementer", _make_handler(["implement", "moduleReady"]))
        orch.register_handler("Writer", _make_handler(["writePaper", "paperReady"]))
        pr = orch._parallel_phase("impl-write", [
            AgentTask("Implementer", "1", "Implement"),
            AgentTask("Writer", "1", "Write"),
        ])
        # Still completes, just serialized
        assert len(pr.results) == 2
        assert pr.all_completed


# ---------------------------------------------------------------------------
# Sequential phase
# ---------------------------------------------------------------------------

class TestSequentialPhase:
    def test_sequential_phase(self):
        orch = Orchestrator()
        orch.register_handler("Evaluator", _make_handler(["evaluate", "accepted"]))
        pr = orch._sequential_phase(
            "eval", AgentTask("Evaluator", "1", "Grade"),
        )
        assert len(pr.results) == 1
        assert pr.all_completed


# ---------------------------------------------------------------------------
# Sprint execution
# ---------------------------------------------------------------------------

class TestSprintExecution:
    def _wire_all_agents(self, orch: Orchestrator) -> None:
        """Wire up all agents with valid protocol-conforming handlers."""
        orch.register_handler("Researcher", _make_handler(["investigate", "report"]))
        orch.register_handler("Implementer", _make_handler(["implement", "moduleReady"]))
        orch.register_handler("Writer", _make_handler(["writePaper", "paperReady"]))
        orch.register_handler("Tester", _make_handler(["writeTests", "testsPass"]))
        orch.register_handler("Prover", _make_handler(["writeProofs", "proofsReady"]))
        orch.register_handler("Evaluator", _make_handler(
            ["evaluate", "accepted"], {"accepted": True},
        ))
        orch.register_handler("Reviewer", _make_handler(["review", "fixes"]))

    def test_sprint_no_steps(self):
        orch = Orchestrator()
        sr = orch.run_sprint(steps=[])
        assert sr.sprint_number == 1
        assert sr.total_agents_dispatched == 0
        assert sr.conformance_rate == 1.0

    def test_sprint_increments_counter(self):
        orch = Orchestrator()
        sr1 = orch.run_sprint(steps=[])
        sr2 = orch.run_sprint(steps=[])
        assert sr1.sprint_number == 1
        assert sr2.sprint_number == 2

    def test_sprint_single_step_all_agents(self):
        orch = Orchestrator()
        self._wire_all_agents(orch)
        sr = orch.run_sprint(steps=["23"])
        assert sr.sprint_number == 1
        assert "23" in sr.steps_processed
        # Should have at least 4 phases (research, impl||write, test||prove, eval)
        assert len(sr.phase_results) >= 4
        assert len(sr.violations) == 0
        assert sr.conformance_rate == 1.0

    def test_sprint_multiple_steps(self):
        orch = Orchestrator()
        self._wire_all_agents(orch)
        sr = orch.run_sprint(steps=["10", "11"])
        assert len(sr.steps_processed) == 2
        # Each step produces at least 4 phases
        assert len(sr.phase_results) >= 8

    def test_sprint_with_violation(self):
        orch = Orchestrator()
        # Researcher violates protocol
        orch.register_handler("Researcher", _make_handler(["investigate", "WRONG"]))
        sr = orch.run_sprint(steps=["5"])
        assert len(sr.violations) > 0

    def test_sprint_stops_on_phase_failure_when_rejecting(self):
        cfg = OrchestratorConfig(reject_violations=True)
        orch = Orchestrator(cfg)
        # Research succeeds
        orch.register_handler("Researcher", _make_handler(["investigate", "report"]))
        # Implementer violates, Writer succeeds
        orch.register_handler("Implementer", _make_handler(["implement", "INVALID"]))
        orch.register_handler("Writer", _make_handler(["writePaper", "paperReady"]))
        sr = orch.run_sprint(steps=["7"])
        # Should stop after phase 2 (impl||write) because it has violation
        # Phase 1 (research) + Phase 2 (impl||write) = 2 phases
        assert len(sr.phase_results) == 2

    def test_sprint_with_review_phase(self):
        orch = Orchestrator()
        orch.register_handler("Researcher", _make_handler(["investigate", "report"]))
        orch.register_handler("Implementer", _make_handler(["implement", "moduleReady"]))
        orch.register_handler("Writer", _make_handler(["writePaper", "paperReady"]))
        orch.register_handler("Tester", _make_handler(["writeTests", "testsPass"]))
        orch.register_handler("Prover", _make_handler(["writeProofs", "proofsReady"]))
        # Evaluator says needs fixes
        orch.register_handler("Evaluator", _make_handler(
            ["evaluate", "needsFixes", "listFixes"],
            {"needs_fixes": True},
        ))
        orch.register_handler("Reviewer", _make_handler(["review", "fixes"]))
        sr = orch.run_sprint(steps=["8"])
        # Should have 5 phases: research, impl||write, test||prove, eval, review
        assert len(sr.phase_results) == 5

    def test_sprint_elapsed_ms(self):
        orch = Orchestrator()
        sr = orch.run_sprint(steps=[])
        assert sr.elapsed_ms >= 0


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class TestDashboard:
    def test_dashboard_with_agents(self):
        orch = Orchestrator()
        orch.register_handler("Researcher", _make_handler(["investigate", "report"]))
        orch._execute_agent("Researcher", "1", "test")
        dash = orch.dashboard()
        assert "AGENT MONITOR DASHBOARD" in dash
        assert "researcher" in dash.lower()

    def test_dashboard_monitoring_disabled(self):
        cfg = OrchestratorConfig(monitor_enabled=False)
        orch = Orchestrator(cfg)
        dash = orch.dashboard()
        assert "disabled" in dash.lower()

    def test_conformance_report(self):
        orch = Orchestrator()
        orch.register_handler("Researcher", _make_handler(["investigate", "report"]))
        orch._execute_agent("Researcher", "1", "test")
        report = orch.conformance_report()
        assert "total" in report
        assert report["total"] >= 1

    def test_conformance_report_disabled(self):
        cfg = OrchestratorConfig(monitor_enabled=False)
        orch = Orchestrator(cfg)
        report = orch.conformance_report()
        assert report.get("monitoring") is False


# ---------------------------------------------------------------------------
# Complete agent (low-level)
# ---------------------------------------------------------------------------

class TestCompleteAgent:
    def test_complete_valid_agent(self):
        orch = Orchestrator()
        aid = orch._start_monitored_agent("Researcher", "1", "test")
        orch.monitor.transition(aid, "investigate")
        orch.monitor.transition(aid, "report")
        assert orch._complete_agent(aid, None)

    def test_complete_violated_agent(self):
        orch = Orchestrator()
        aid = orch._start_monitored_agent("Researcher", "1", "test")
        orch.monitor.transition(aid, "investigate")
        orch.monitor.transition(aid, "WRONG")  # violation
        assert not orch._complete_agent(aid, None)

    def test_complete_unknown_agent(self):
        orch = Orchestrator()
        assert not orch._complete_agent("nonexistent", None)

    def test_complete_monitoring_disabled(self):
        cfg = OrchestratorConfig(monitor_enabled=False)
        orch = Orchestrator(cfg)
        assert orch._complete_agent("anything", None)
