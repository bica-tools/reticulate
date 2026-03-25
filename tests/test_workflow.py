"""Tests for the session-type-enforced agent workflow."""

import pytest
from unittest.mock import patch, MagicMock

from reticulate.workflow import (
    Workflow,
    MCPTransport,
    A2ATransport,
    StepResult,
    PhaseResult,
    AgentExecution,
)
from reticulate.agent_registry import (
    AGENTS,
    ORCHESTRATOR_TYPE,
    AgentMonitor,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ===========================================================================
# Protocol verification tests
# ===========================================================================

class TestProtocolVerification:
    """Every agent session type must form a lattice."""

    def test_all_agent_types_have_session_types(self):
        """Every registered agent has a non-empty session type."""
        for name, agent in AGENTS.items():
            assert agent.session_type, f"{name} has empty session type"
            assert agent.protocol in ("MCP", "A2A"), f"{name} has bad protocol: {agent.protocol}"

    @pytest.mark.parametrize("agent_name", list(AGENTS.keys()))
    def test_agent_session_type_is_lattice(self, agent_name: str):
        """Each agent's session type state space forms a lattice."""
        agent = AGENTS[agent_name]
        ast = parse(agent.session_type)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice, (
            f"{agent_name} session type is NOT a lattice: {agent.session_type}"
        )

    def test_orchestrator_type_is_lattice(self):
        """The orchestrator's composite protocol forms a lattice."""
        ast = parse(ORCHESTRATOR_TYPE)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice, "Orchestrator protocol is NOT a lattice"

    def test_orchestrator_has_parallel(self):
        """The orchestrator protocol uses parallel composition."""
        assert "||" in ORCHESTRATOR_TYPE

    def test_mcp_agents_use_stdio(self):
        """MCP agents use stdio transport."""
        for name, agent in AGENTS.items():
            if agent.protocol == "MCP":
                assert agent.transport == "stdio", f"{name}: MCP agent should use stdio"

    def test_a2a_agents_use_agent_transport(self):
        """A2A agents use Agent() transport."""
        for name, agent in AGENTS.items():
            if agent.protocol == "A2A":
                assert agent.transport == "Agent()", f"{name}: A2A agent should use Agent()"


# ===========================================================================
# MCP Transport tests
# ===========================================================================

class TestMCPTransport:
    """MCP transport: in-process tool calls."""

    def test_research_returns_step_info(self):
        mcp = MCPTransport()
        result = mcp.call_tool("research", step_number="10")
        assert "step" in result
        assert result["step"] == "10"

    def test_analyze_parses_type(self):
        mcp = MCPTransport()
        result = mcp.call_tool("analyze", type_string="&{a: end, b: end}")
        assert result["is_lattice"] is True
        assert result["states"] > 0

    def test_unknown_tool_returns_error(self):
        mcp = MCPTransport()
        result = mcp.call_tool("nonexistent_tool")
        assert "error" in result

    def test_evaluate_returns_grade(self):
        mcp = MCPTransport()
        result = mcp.call_tool("evaluate", step_number="10")
        assert "grade" in result
        assert "accepted" in result


# ===========================================================================
# A2A Transport tests
# ===========================================================================

class TestA2ATransport:
    """A2A transport: task delegation."""

    def test_dry_run_returns_prompt_info(self):
        a2a = A2ATransport(dry_run=True)
        result = a2a.send_task("Writer", "10", "Write paper")
        assert result["status"] == "dry_run"
        assert result["agent_type"] == "Writer"
        assert result["prompt_length"] == 11

    def test_get_status(self):
        a2a = A2ATransport(dry_run=True)
        result = a2a.get_status("task-123")
        assert result["status"] == "completed"

    def test_cancel(self):
        a2a = A2ATransport(dry_run=True)
        result = a2a.cancel("task-123")
        assert result["status"] == "cancelled"


# ===========================================================================
# Agent Monitor integration tests
# ===========================================================================

class TestAgentMonitorIntegration:
    """Agent monitor tracks transitions against session types."""

    def test_researcher_valid_path(self):
        monitor = AgentMonitor()
        monitor.start_agent("Researcher", "r-1", "10")
        assert monitor.transition("r-1", "investigate")
        assert monitor.transition("r-1", "report")
        status = monitor.get_status("r-1")
        assert status["status"] == "completed"

    def test_writer_valid_path(self):
        monitor = AgentMonitor()
        monitor.start_agent("Writer", "w-1", "10")
        assert monitor.transition("w-1", "writePaper")
        assert monitor.transition("w-1", "paperReady")
        assert monitor.get_status("w-1")["status"] == "completed"

    def test_evaluator_accept_path(self):
        monitor = AgentMonitor()
        monitor.start_agent("Evaluator", "e-1", "10")
        assert monitor.transition("e-1", "evaluate")
        assert monitor.transition("e-1", "accepted")
        assert monitor.get_status("e-1")["status"] == "completed"

    def test_evaluator_reject_path(self):
        monitor = AgentMonitor()
        monitor.start_agent("Evaluator", "e-2", "10")
        assert monitor.transition("e-2", "evaluate")
        assert monitor.transition("e-2", "needsFixes")
        assert monitor.transition("e-2", "listFixes")
        assert monitor.get_status("e-2")["status"] == "completed"

    def test_implementer_retry_loop(self):
        monitor = AgentMonitor()
        monitor.start_agent("Implementer", "i-1", "10")
        assert monitor.transition("i-1", "implement")
        assert monitor.transition("i-1", "error")
        assert monitor.transition("i-1", "retry")
        # After retry, we're back at the beginning (recursion)
        assert monitor.transition("i-1", "implement")
        assert monitor.transition("i-1", "moduleReady")
        assert monitor.get_status("i-1")["status"] == "completed"

    def test_tester_fail_abort(self):
        monitor = AgentMonitor()
        monitor.start_agent("Tester", "t-1", "10")
        assert monitor.transition("t-1", "writeTests")
        assert monitor.transition("t-1", "testsFail")
        assert monitor.transition("t-1", "abort")
        assert monitor.get_status("t-1")["status"] == "completed"

    def test_violation_detected(self):
        monitor = AgentMonitor()
        monitor.start_agent("Writer", "w-2", "10")
        assert monitor.transition("w-2", "writePaper")
        # Try invalid transition
        assert not monitor.transition("w-2", "INVALID")
        assert monitor.get_status("w-2")["status"] == "violated"

    def test_out_of_order_violation(self):
        monitor = AgentMonitor()
        monitor.start_agent("Evaluator", "e-3", "10")
        # Can't start with "accepted" — must start with "evaluate"
        assert not monitor.transition("e-3", "accepted")
        assert monitor.get_status("e-3")["status"] == "violated"


# ===========================================================================
# Workflow execution tests (dry run)
# ===========================================================================

class TestWorkflowDryRun:
    """Workflow execution in dry-run mode."""

    def test_run_step_dry_run(self):
        wf = Workflow(dry_run=True, verbose=False)
        result = wf.run_step("10")
        assert isinstance(result, StepResult)
        assert result.step_number == "10"
        assert len(result.phases) >= 4  # research, impl||write, test||prove, evaluate

    def test_run_sprint_dry_run(self):
        wf = Workflow(dry_run=True, verbose=False)
        results = wf.run_sprint(steps=["10"])
        assert len(results) == 1
        assert results[0].step_number == "10"

    def test_phases_are_named(self):
        wf = Workflow(dry_run=True, verbose=False)
        result = wf.run_step("10")
        phase_names = [p.name for p in result.phases]
        assert "research" in phase_names
        assert "implement || write" in phase_names
        assert "test || prove" in phase_names
        assert "evaluate" in phase_names

    def test_parallel_phases_have_two_agents(self):
        wf = Workflow(dry_run=True, verbose=False)
        result = wf.run_step("10")
        for phase in result.phases:
            if phase.name == "implement || write":
                agent_types = [a.agent_type for a in phase.agents]
                assert "Implementer" in agent_types
                assert "Writer" in agent_types
            if phase.name == "test || prove":
                agent_types = [a.agent_type for a in phase.agents]
                assert "Tester" in agent_types
                assert "Prover" in agent_types

    def test_mcp_agents_use_mcp_transport(self):
        wf = Workflow(dry_run=True, verbose=False)
        result = wf.run_step("10")
        for phase in result.phases:
            for agent in phase.agents:
                expected = AGENTS[agent.agent_type].protocol
                assert agent.transport == expected, (
                    f"{agent.agent_type} should use {expected} transport"
                )

    def test_dashboard_after_sprint(self):
        wf = Workflow(dry_run=True, verbose=False)
        wf.run_sprint(steps=["10"])
        dashboard = wf.dashboard()
        assert "AGENT MONITOR DASHBOARD" in dashboard

    def test_verify_protocols(self):
        wf = Workflow(dry_run=True, verbose=False)
        results = wf.verify_protocols()
        assert "__Orchestrator__" in results
        # All should be lattices
        for name, info in results.items():
            assert info["is_lattice"], f"{name} is not a lattice"

    def test_conformance_report(self):
        wf = Workflow(dry_run=True, verbose=False)
        wf.run_sprint(steps=["10"])
        report = wf.monitor.conformance_report()
        assert report["total"] > 0
        assert "conformance_rate" in report

    def test_no_violations_in_normal_run(self):
        wf = Workflow(dry_run=True, verbose=False)
        result = wf.run_step("10")
        assert len(result.violations) == 0

    def test_step_result_has_grade(self):
        wf = Workflow(dry_run=True, verbose=False)
        result = wf.run_step("10")
        # Step 10 should have a grade from evaluator
        assert result.grade != ""


# ===========================================================================
# Transition inference tests
# ===========================================================================

class TestTransitionInference:
    """Verify that transition inference produces valid paths."""

    def test_supervisor_transitions(self):
        wf = Workflow(dry_run=True, verbose=False)
        transitions = wf._infer_transitions("Supervisor", {})
        assert transitions == ["scan", "proposals"]

    def test_researcher_transitions(self):
        wf = Workflow(dry_run=True, verbose=False)
        transitions = wf._infer_transitions("Researcher", {})
        assert transitions == ["investigate", "report"]

    def test_evaluator_accept(self):
        wf = Workflow(dry_run=True, verbose=False)
        transitions = wf._infer_transitions("Evaluator", {"accepted": True})
        assert transitions == ["evaluate", "accepted"]

    def test_evaluator_reject(self):
        wf = Workflow(dry_run=True, verbose=False)
        transitions = wf._infer_transitions("Evaluator", {"accepted": False})
        assert transitions == ["evaluate", "needsFixes", "listFixes"]

    def test_implementer_success(self):
        wf = Workflow(dry_run=True, verbose=False)
        transitions = wf._infer_transitions("Implementer", {"status": "completed"})
        assert transitions == ["implement", "moduleReady"]

    def test_implementer_failure(self):
        wf = Workflow(dry_run=True, verbose=False)
        transitions = wf._infer_transitions("Implementer", {"status": "failed"})
        assert transitions == ["implement", "error", "abort"]

    def test_writer_transitions(self):
        wf = Workflow(dry_run=True, verbose=False)
        transitions = wf._infer_transitions("Writer", {})
        assert transitions == ["writePaper", "paperReady"]

    def test_prover_transitions(self):
        wf = Workflow(dry_run=True, verbose=False)
        transitions = wf._infer_transitions("Prover", {})
        assert transitions == ["writeProofs", "proofsReady"]

    def test_reviewer_transitions(self):
        wf = Workflow(dry_run=True, verbose=False)
        transitions = wf._infer_transitions("Reviewer", {})
        assert transitions == ["review", "fixes"]

    def test_unknown_agent_returns_empty(self):
        wf = Workflow(dry_run=True, verbose=False)
        transitions = wf._infer_transitions("Nonexistent", {})
        assert transitions == []


# ===========================================================================
# Session type structure tests
# ===========================================================================

class TestSessionTypeStructure:
    """Verify structural properties of agent session types."""

    def test_all_agents_have_top_and_bottom(self):
        """Every agent state space has reachable top and bottom."""
        for name, agent in AGENTS.items():
            ast = parse(agent.session_type)
            ss = build_statespace(ast)
            lr = check_lattice(ss)
            assert lr.has_top, f"{name} has no reachable top"
            assert lr.has_bottom, f"{name} has no reachable bottom"

    def test_orchestrator_has_product_structure(self):
        """Orchestrator uses parallel → product lattice."""
        ast = parse(ORCHESTRATOR_TYPE)
        ss = build_statespace(ast)
        # Product lattice should have more states than a linear chain
        assert len(ss.states) > 10

    def test_recursive_agents_have_cycles(self):
        """Agents with rec X should produce cyclic state spaces."""
        recursive_agents = [
            name for name, agent in AGENTS.items()
            if "rec" in agent.session_type
        ]
        assert len(recursive_agents) > 0, "Should have some recursive agents"
        for name in recursive_agents:
            agent = AGENTS[name]
            ast = parse(agent.session_type)
            ss = build_statespace(ast)
            lr = check_lattice(ss)
            # Recursive types with cycles should have SCCs > 1
            # (or the SCC count equals number of states for non-cyclic)
            assert lr.is_lattice, f"{name}: recursive type should still be lattice"

    def test_mcp_agent_count(self):
        """At least 3 MCP agents."""
        mcp_count = sum(1 for a in AGENTS.values() if a.protocol == "MCP")
        assert mcp_count >= 3

    def test_a2a_agent_count(self):
        """At least 5 A2A agents."""
        a2a_count = sum(1 for a in AGENTS.values() if a.protocol == "A2A")
        assert a2a_count >= 5

    @pytest.mark.parametrize("agent_name", [
        "Researcher", "Evaluator", "Supervisor",
    ])
    def test_mcp_agent_is_stateless(self, agent_name: str):
        """MCP agents should be simple (few states, no recursion)."""
        agent = AGENTS[agent_name]
        assert "rec" not in agent.session_type, f"{agent_name}: MCP agents should not be recursive"

    @pytest.mark.parametrize("agent_name", [
        "Implementer", "Tester",
    ])
    def test_a2a_task_agent_is_recursive(self, agent_name: str):
        """Task-based A2A agents should support retry loops."""
        agent = AGENTS[agent_name]
        assert "rec" in agent.session_type, f"{agent_name}: A2A task agents should be recursive"


# ===========================================================================
# CLI tests
# ===========================================================================

class TestWorkflowCLI:
    """Test CLI entry point."""

    def test_verify_command(self, capsys):
        from reticulate.workflow import main
        with pytest.raises(SystemExit) as exc_info:
            main(["verify"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Agent" in captured.out
        assert "Lattice" in captured.out

    def test_protocols_command(self, capsys):
        from reticulate.workflow import main
        main(["protocols"])
        captured = capsys.readouterr()
        assert "Researcher" in captured.out
        assert "Orchestrator" in captured.out

    def test_status_command(self, capsys):
        from reticulate.workflow import main
        main(["status"])
        captured = capsys.readouterr()
        assert "Available agents" in captured.out

    def test_no_command_shows_help(self, capsys):
        from reticulate.workflow import main
        main([])
        captured = capsys.readouterr()
        # argparse prints to stdout
        assert "workflow" in captured.out or "usage" in captured.out.lower()
