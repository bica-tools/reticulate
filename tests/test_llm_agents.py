"""Tests for LLM Agent Protocol Verification (Step 82)."""

import pytest

from reticulate.global_types import (
    GEnd,
    GMessage,
    GParallel,
    GRec,
    GVar,
    build_global_statespace,
    pretty_global,
    roles,
)
from reticulate.lattice import check_lattice
from reticulate.llm_agents import (
    AgentOrchestration,
    OrchestrationResult,
    RoleProjection,
    a2a_broadcast,
    a2a_chain,
    format_orchestration_result,
    mcp_multi_agent,
    multi_model_consensus,
    rag_pipeline,
    tool_use_loop,
    verify_orchestration,
)


# ---------------------------------------------------------------------------
# AgentOrchestration dataclass
# ---------------------------------------------------------------------------


class TestAgentOrchestration:
    """Test the AgentOrchestration dataclass."""

    def test_frozen(self):
        orch = rag_pipeline()
        with pytest.raises(AttributeError):
            orch.name = "changed"  # type: ignore

    def test_agents_are_tuple(self):
        orch = rag_pipeline()
        assert isinstance(orch.agents, tuple)

    def test_channels_are_tuple(self):
        orch = rag_pipeline()
        assert isinstance(orch.channels, tuple)

    def test_global_type_is_global_type(self):
        orch = rag_pipeline()
        assert orch.global_type is not None


# ---------------------------------------------------------------------------
# MCP Multi-Agent
# ---------------------------------------------------------------------------


class TestMCPMultiAgent:
    """Test MCP multi-agent orchestration."""

    def test_single_server(self):
        orch = mcp_multi_agent(1)
        assert len(orch.agents) == 2
        assert "host" in orch.agents
        assert "server1" in orch.agents

    def test_two_servers(self):
        orch = mcp_multi_agent(2)
        assert len(orch.agents) == 3
        assert "server2" in orch.agents

    def test_three_servers(self):
        orch = mcp_multi_agent(3)
        assert len(orch.agents) == 4

    def test_channels_per_server(self):
        orch = mcp_multi_agent(2)
        # 2 channels per server (host->server, server->host)
        assert len(orch.channels) == 4

    def test_global_type_has_roles(self):
        orch = mcp_multi_agent(2)
        r = roles(orch.global_type)
        assert "host" in r
        assert "server1" in r
        assert "server2" in r

    def test_invalid_num_servers(self):
        with pytest.raises(ValueError):
            mcp_multi_agent(0)

    def test_single_server_verification(self):
        orch = mcp_multi_agent(1)
        result = verify_orchestration(orch)
        assert result.global_is_lattice
        assert result.all_projections_defined
        assert result.is_verified


# ---------------------------------------------------------------------------
# A2A Chain
# ---------------------------------------------------------------------------


class TestA2AChain:
    """Test A2A chain orchestration."""

    def test_two_agents(self):
        orch = a2a_chain(2)
        assert len(orch.agents) == 2

    def test_three_agents(self):
        orch = a2a_chain(3)
        assert len(orch.agents) == 3

    def test_channels(self):
        orch = a2a_chain(3)
        # 2 channels per link, 2 links
        assert len(orch.channels) == 4

    def test_invalid_num_agents(self):
        with pytest.raises(ValueError):
            a2a_chain(1)

    def test_roles_match_agents(self):
        orch = a2a_chain(3)
        r = roles(orch.global_type)
        for a in orch.agents:
            assert a in r

    def test_verification(self):
        orch = a2a_chain(2)
        result = verify_orchestration(orch)
        assert result.global_is_lattice
        assert result.all_projections_defined
        assert result.is_verified

    def test_three_agent_verification(self):
        orch = a2a_chain(3)
        result = verify_orchestration(orch)
        assert result.global_is_lattice
        assert result.all_projections_defined
        assert result.is_verified


# ---------------------------------------------------------------------------
# A2A Broadcast
# ---------------------------------------------------------------------------


class TestA2ABroadcast:
    """Test A2A broadcast orchestration."""

    def test_single_worker(self):
        orch = a2a_broadcast(1)
        assert len(orch.agents) == 2

    def test_three_workers(self):
        orch = a2a_broadcast(3)
        assert len(orch.agents) == 4
        assert "orchestrator" in orch.agents

    def test_invalid_num_agents(self):
        with pytest.raises(ValueError):
            a2a_broadcast(0)

    def test_single_worker_verification(self):
        orch = a2a_broadcast(1)
        result = verify_orchestration(orch)
        assert result.global_is_lattice
        assert result.is_verified

    def test_two_worker_verification(self):
        orch = a2a_broadcast(2)
        result = verify_orchestration(orch)
        assert result.global_is_lattice
        assert result.all_projections_defined


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    """Test RAG pipeline orchestration."""

    def test_agents(self):
        orch = rag_pipeline()
        assert "user" in orch.agents
        assert "retriever" in orch.agents
        assert "ranker" in orch.agents
        assert "generator" in orch.agents
        assert "validator" in orch.agents

    def test_five_agents(self):
        orch = rag_pipeline()
        assert len(orch.agents) == 5

    def test_channels(self):
        orch = rag_pipeline()
        assert len(orch.channels) == 5

    def test_verification(self):
        orch = rag_pipeline()
        result = verify_orchestration(orch)
        assert result.global_is_lattice
        assert result.all_projections_defined
        assert result.is_verified

    def test_all_projections_are_lattices(self):
        orch = rag_pipeline()
        result = verify_orchestration(orch)
        assert result.all_local_lattices


# ---------------------------------------------------------------------------
# Tool Use Loop
# ---------------------------------------------------------------------------


class TestToolUseLoop:
    """Test tool use loop orchestration."""

    def test_agents(self):
        orch = tool_use_loop()
        assert "agent" in orch.agents
        assert "tool" in orch.agents

    def test_has_recursion(self):
        orch = tool_use_loop()
        assert isinstance(orch.global_type, GRec)

    def test_verification(self):
        orch = tool_use_loop()
        result = verify_orchestration(orch)
        assert result.global_is_lattice
        assert result.all_projections_defined
        assert result.is_verified

    def test_global_state_space_finite(self):
        orch = tool_use_loop()
        ss = build_global_statespace(orch.global_type)
        assert len(ss.states) > 0
        assert len(ss.states) < 100


# ---------------------------------------------------------------------------
# Multi-Model Consensus
# ---------------------------------------------------------------------------


class TestMultiModelConsensus:
    """Test multi-model consensus orchestration."""

    def test_two_models(self):
        orch = multi_model_consensus(2)
        assert "coordinator" in orch.agents
        assert "aggregator" in orch.agents
        assert "user" in orch.agents
        assert "model1" in orch.agents
        assert "model2" in orch.agents

    def test_three_models(self):
        orch = multi_model_consensus(3)
        assert len(orch.agents) == 6  # coordinator, aggregator, user, model1-3

    def test_invalid_num_models(self):
        with pytest.raises(ValueError):
            multi_model_consensus(1)

    def test_verification(self):
        orch = multi_model_consensus(2)
        result = verify_orchestration(orch)
        assert result.global_is_lattice
        assert result.all_projections_defined


# ---------------------------------------------------------------------------
# verify_orchestration
# ---------------------------------------------------------------------------


class TestVerifyOrchestration:
    """Test the verification function itself."""

    def test_result_type(self):
        orch = rag_pipeline()
        result = verify_orchestration(orch)
        assert isinstance(result, OrchestrationResult)

    def test_result_frozen(self):
        result = verify_orchestration(rag_pipeline())
        with pytest.raises(AttributeError):
            result.is_verified = False  # type: ignore

    def test_global_type_str_not_empty(self):
        result = verify_orchestration(rag_pipeline())
        assert len(result.global_type_str) > 0

    def test_projections_count_matches_roles(self):
        orch = rag_pipeline()
        result = verify_orchestration(orch)
        all_roles = roles(orch.global_type)
        assert len(result.projections) == len(all_roles)

    def test_projection_has_local_type_str(self):
        result = verify_orchestration(rag_pipeline())
        for p in result.projections:
            assert isinstance(p.local_type_str, str)
            assert len(p.local_type_str) > 0

    def test_chain_projections_have_states(self):
        result = verify_orchestration(a2a_chain(2))
        for p in result.projections:
            assert p.state_count > 0


# ---------------------------------------------------------------------------
# format_orchestration_result
# ---------------------------------------------------------------------------


class TestFormatResult:
    """Test report formatting."""

    def test_format_rag(self):
        result = verify_orchestration(rag_pipeline())
        text = format_orchestration_result(result)
        assert "VERIFIED" in text
        assert "RAG Pipeline" in text

    def test_format_contains_agents(self):
        result = verify_orchestration(rag_pipeline())
        text = format_orchestration_result(result)
        assert "retriever" in text
        assert "ranker" in text

    def test_format_contains_verdict(self):
        result = verify_orchestration(a2a_chain(2))
        text = format_orchestration_result(result)
        assert "Verdict" in text or "VERIFIED" in text


# ---------------------------------------------------------------------------
# Cross-pattern tests
# ---------------------------------------------------------------------------


class TestCrossPattern:
    """Test properties across all patterns."""

    @pytest.mark.parametrize("factory", [
        lambda: mcp_multi_agent(1),
        lambda: a2a_chain(2),
        lambda: a2a_broadcast(1),
        lambda: rag_pipeline(),
        lambda: tool_use_loop(),
        lambda: multi_model_consensus(2),
    ])
    def test_all_patterns_build_global_statespace(self, factory):
        orch = factory()
        ss = build_global_statespace(orch.global_type)
        assert len(ss.states) > 0

    @pytest.mark.parametrize("factory", [
        lambda: mcp_multi_agent(1),
        lambda: a2a_chain(2),
        lambda: a2a_broadcast(1),
        lambda: rag_pipeline(),
        lambda: tool_use_loop(),
        lambda: multi_model_consensus(2),
    ])
    def test_all_patterns_global_is_lattice(self, factory):
        orch = factory()
        ss = build_global_statespace(orch.global_type)
        lr = check_lattice(ss)
        assert lr.is_lattice

    @pytest.mark.parametrize("factory", [
        lambda: mcp_multi_agent(1),
        lambda: a2a_chain(2),
        lambda: a2a_broadcast(1),
        lambda: rag_pipeline(),
        lambda: tool_use_loop(),
        lambda: multi_model_consensus(2),
    ])
    def test_all_patterns_verify(self, factory):
        orch = factory()
        result = verify_orchestration(orch)
        assert result.all_projections_defined
