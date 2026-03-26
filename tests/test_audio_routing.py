"""Tests for audio_routing.py — verified audio routing via session types (Step 57a).

Tests cover:
  - Data type construction and validation
  - Session type generation from routing graphs
  - Routing verification (lattice property)
  - Dead signal detection
  - Feedback loop detection and safety
  - Pre-built DAW scenarios
  - Benchmark: all routing graphs produce lattices
"""

from __future__ import annotations

import pytest

from reticulate.audio_routing import (
    ALL_ROUTING_SCENARIOS,
    AudioGraph,
    AudioNode,
    AudioRoute,
    RoutingResult,
    audio_graph_to_session_type,
    basic_mix,
    dead_signal_graph,
    feedback_delay,
    find_dead_signals,
    find_feedback_loops,
    format_routing_report,
    is_feedback_safe,
    live_mix,
    mastering_chain,
    multiband_split,
    recording_setup,
    send_return,
    sidechain_compress,
    unsafe_feedback,
    verify_routing,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Data type tests
# ---------------------------------------------------------------------------

class TestAudioNode:
    """AudioNode dataclass validation."""

    def test_valid_source(self) -> None:
        node = AudioNode("kick", "source", "mono")
        assert node.name == "kick"
        assert node.node_type == "source"
        assert node.channels == "mono"

    def test_valid_effect(self) -> None:
        node = AudioNode("eq", "effect", "stereo")
        assert node.node_type == "effect"

    def test_valid_bus(self) -> None:
        node = AudioNode("reverb_bus", "bus", "stereo")
        assert node.node_type == "bus"

    def test_valid_output(self) -> None:
        node = AudioNode("master", "output", "surround")
        assert node.channels == "surround"

    def test_invalid_node_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid node_type"):
            AudioNode("x", "mixer")

    def test_invalid_channels(self) -> None:
        with pytest.raises(ValueError, match="Invalid channels"):
            AudioNode("x", "effect", "quad")

    def test_frozen(self) -> None:
        node = AudioNode("x", "source")
        with pytest.raises(AttributeError):
            node.name = "y"  # type: ignore[misc]


class TestAudioRoute:
    """AudioRoute dataclass."""

    def test_basic(self) -> None:
        route = AudioRoute("a", "b", "send")
        assert route.src == "a"
        assert route.dst == "b"
        assert route.label == "send"

    def test_default_label(self) -> None:
        route = AudioRoute("a", "b")
        assert route.label == "send"


class TestAudioGraph:
    """AudioGraph construction and helpers."""

    def test_invalid_master(self) -> None:
        with pytest.raises(ValueError, match="master_output"):
            AudioGraph(
                nodes=[AudioNode("a", "source")],
                routes=[],
                master_output="nonexistent",
            )

    def test_node_map(self) -> None:
        g = basic_mix()
        nm = g.node_map
        assert "kick" in nm
        assert nm["kick"].node_type == "source"

    def test_successors(self) -> None:
        g = basic_mix()
        succs = g.successors("kick")
        assert len(succs) == 1
        assert succs[0] == ("master", "send")

    def test_predecessors(self) -> None:
        g = basic_mix()
        preds = g.predecessors("master")
        assert len(preds) == 3

    def test_sources(self) -> None:
        g = basic_mix()
        assert sorted(g.sources()) == ["kick", "snare", "vocal"]

    def test_outputs(self) -> None:
        g = basic_mix()
        assert g.outputs() == ["master"]


# ---------------------------------------------------------------------------
# Session type generation tests
# ---------------------------------------------------------------------------

class TestSessionTypeGeneration:
    """Test audio_graph_to_session_type."""

    def test_empty_sources(self) -> None:
        g = AudioGraph(
            nodes=[AudioNode("master", "output")],
            routes=[],
            master_output="master",
        )
        assert audio_graph_to_session_type(g) == "end"

    def test_single_chain(self) -> None:
        g = AudioGraph(
            nodes=[
                AudioNode("input", "source"),
                AudioNode("eq", "effect"),
                AudioNode("master", "output"),
            ],
            routes=[
                AudioRoute("input", "eq", "send"),
                AudioRoute("eq", "master", "send"),
            ],
            master_output="master",
        )
        st = audio_graph_to_session_type(g)
        # Should parse without error
        ast = parse(st)
        assert ast is not None

    def test_parallel_mix(self) -> None:
        g = basic_mix()
        st = audio_graph_to_session_type(g)
        # Should contain parallel composition
        assert "||" in st
        ast = parse(st)
        assert ast is not None

    def test_mastering_chain_parseable(self) -> None:
        g = mastering_chain()
        st = audio_graph_to_session_type(g)
        ast = parse(st)
        assert ast is not None

    def test_multiband_parseable(self) -> None:
        g = multiband_split()
        st = audio_graph_to_session_type(g)
        ast = parse(st)
        assert ast is not None


# ---------------------------------------------------------------------------
# Verification tests
# ---------------------------------------------------------------------------

class TestVerifyRouting:
    """Test verify_routing — the main verification function."""

    def test_basic_mix_verified(self) -> None:
        result = verify_routing(basic_mix())
        assert result.is_verified
        assert result.lattice_result.is_lattice
        assert result.dead_signals == []
        assert result.num_nodes == 4
        assert result.num_routes == 3

    def test_mastering_chain_verified(self) -> None:
        result = verify_routing(mastering_chain())
        assert result.is_verified
        assert result.lattice_result.is_lattice
        assert result.dead_signals == []

    def test_recording_setup_verified(self) -> None:
        result = verify_routing(recording_setup())
        assert result.is_verified
        assert result.dead_signals == []

    def test_live_mix_verified(self) -> None:
        result = verify_routing(live_mix())
        assert result.is_verified
        assert result.dead_signals == []

    def test_send_return_verified(self) -> None:
        result = verify_routing(send_return())
        assert result.is_verified

    def test_sidechain_verified(self) -> None:
        result = verify_routing(sidechain_compress())
        assert result.is_verified

    def test_multiband_verified(self) -> None:
        result = verify_routing(multiband_split())
        assert result.is_verified


# ---------------------------------------------------------------------------
# Dead signal tests
# ---------------------------------------------------------------------------

class TestDeadSignals:
    """Test find_dead_signals."""

    def test_no_dead_signals_basic(self) -> None:
        assert find_dead_signals(basic_mix()) == []

    def test_no_dead_signals_mastering(self) -> None:
        assert find_dead_signals(mastering_chain()) == []

    def test_dead_signal_detected(self) -> None:
        g = dead_signal_graph()
        dead = find_dead_signals(g)
        assert "orphan_bus" in dead
        assert "orphan_fx" in dead

    def test_dead_signal_graph_not_verified(self) -> None:
        g = dead_signal_graph()
        result = verify_routing(g)
        # Dead signals detected
        assert len(result.dead_signals) > 0
        assert not result.is_verified

    def test_isolated_node(self) -> None:
        """A node with no routes at all is dead."""
        g = AudioGraph(
            nodes=[
                AudioNode("input", "source"),
                AudioNode("orphan", "effect"),
                AudioNode("master", "output"),
            ],
            routes=[
                AudioRoute("input", "master", "send"),
            ],
            master_output="master",
        )
        dead = find_dead_signals(g)
        assert "orphan" in dead


# ---------------------------------------------------------------------------
# Feedback loop tests
# ---------------------------------------------------------------------------

class TestFeedbackLoops:
    """Test find_feedback_loops and is_feedback_safe."""

    def test_no_loops_basic(self) -> None:
        assert find_feedback_loops(basic_mix()) == []

    def test_no_loops_mastering(self) -> None:
        assert find_feedback_loops(mastering_chain()) == []

    def test_feedback_delay_has_loop(self) -> None:
        loops = find_feedback_loops(feedback_delay())
        assert len(loops) >= 1
        # The loop should include "delay" and "feedback"
        all_nodes = [n for loop in loops for n in loop]
        assert "delay" in all_nodes
        assert "feedback" in all_nodes

    def test_feedback_delay_is_safe(self) -> None:
        assert is_feedback_safe(feedback_delay())

    def test_unsafe_feedback_detected(self) -> None:
        loops = find_feedback_loops(unsafe_feedback())
        assert len(loops) >= 1
        assert not is_feedback_safe(unsafe_feedback())

    def test_no_loops_trivially_safe(self) -> None:
        assert is_feedback_safe(basic_mix())


# ---------------------------------------------------------------------------
# Pre-built scenarios tests
# ---------------------------------------------------------------------------

class TestPrebuiltScenarios:
    """Test all pre-built audio routing scenarios."""

    def test_basic_mix_creates_graph(self) -> None:
        g = basic_mix()
        assert len(g.nodes) == 4
        assert len(g.routes) == 3

    def test_mastering_chain_creates_graph(self) -> None:
        g = mastering_chain()
        assert len(g.nodes) == 5
        assert len(g.routes) == 4

    def test_multiband_creates_graph(self) -> None:
        g = multiband_split()
        assert len(g.nodes) == 7

    def test_sidechain_creates_graph(self) -> None:
        g = sidechain_compress()
        assert len(g.nodes) == 5

    def test_feedback_delay_creates_graph(self) -> None:
        g = feedback_delay()
        assert len(g.nodes) == 4

    def test_send_return_creates_graph(self) -> None:
        g = send_return()
        assert len(g.nodes) == 5

    def test_live_mix_creates_graph(self) -> None:
        g = live_mix()
        assert len(g.nodes) == 9

    def test_recording_setup_creates_graph(self) -> None:
        g = recording_setup()
        assert len(g.nodes) == 5


# ---------------------------------------------------------------------------
# Format report test
# ---------------------------------------------------------------------------

class TestFormatReport:
    """Test format_routing_report."""

    def test_report_basic(self) -> None:
        result = verify_routing(basic_mix())
        report = format_routing_report(result)
        assert "AUDIO ROUTING VERIFICATION REPORT" in report
        assert "Lattice" in report
        assert "YES" in report

    def test_report_dead_signals(self) -> None:
        result = verify_routing(dead_signal_graph())
        report = format_routing_report(result)
        assert "orphan" in report


# ---------------------------------------------------------------------------
# Benchmark: all scenarios produce lattices
# ---------------------------------------------------------------------------

class TestBenchmark:
    """Benchmark: verify all pre-built routing scenarios."""

    @pytest.mark.parametrize("name,graph", ALL_ROUTING_SCENARIOS)
    def test_scenario_produces_lattice(self, name: str, graph: AudioGraph) -> None:
        """Every well-formed routing graph should produce a lattice."""
        st = audio_graph_to_session_type(graph)
        ast = parse(st)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice, f"Scenario '{name}' did not produce a lattice"

    @pytest.mark.parametrize("name,graph", ALL_ROUTING_SCENARIOS)
    def test_scenario_no_dead_signals(self, name: str, graph: AudioGraph) -> None:
        """No pre-built scenario should have dead signals."""
        dead = find_dead_signals(graph)
        assert dead == [], f"Scenario '{name}' has dead signals: {dead}"

    @pytest.mark.parametrize("name,graph", ALL_ROUTING_SCENARIOS)
    def test_scenario_verifiable(self, name: str, graph: AudioGraph) -> None:
        """Every pre-built scenario should be fully verified."""
        result = verify_routing(graph)
        assert result.is_verified, f"Scenario '{name}' verification failed"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and corner scenarios."""

    def test_single_source_direct_to_master(self) -> None:
        g = AudioGraph(
            nodes=[
                AudioNode("input", "source"),
                AudioNode("master", "output"),
            ],
            routes=[AudioRoute("input", "master", "send")],
            master_output="master",
        )
        result = verify_routing(g)
        assert result.is_verified

    def test_many_parallel_sources(self) -> None:
        """10 sources all going to master — large parallel composition."""
        nodes = [AudioNode(f"src_{i}", "source") for i in range(10)]
        nodes.append(AudioNode("master", "output"))
        routes = [AudioRoute(f"src_{i}", "master", "send") for i in range(10)]
        g = AudioGraph(nodes=nodes, routes=routes, master_output="master")
        result = verify_routing(g)
        assert result.is_verified
        assert result.num_nodes == 11

    def test_diamond_topology(self) -> None:
        """Diamond: input → (A, B) → output. Both paths reach master."""
        g = AudioGraph(
            nodes=[
                AudioNode("input", "source"),
                AudioNode("path_a", "effect"),
                AudioNode("path_b", "effect"),
                AudioNode("master", "output"),
            ],
            routes=[
                AudioRoute("input", "path_a", "split_a"),
                AudioRoute("input", "path_b", "split_b"),
                AudioRoute("path_a", "master", "send"),
                AudioRoute("path_b", "master", "send"),
            ],
            master_output="master",
        )
        dead = find_dead_signals(g)
        assert dead == []
