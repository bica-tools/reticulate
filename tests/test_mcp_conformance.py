"""Tests for MCP/A2A conformance testing (Step 70)."""

import pytest

from reticulate.parser import parse, ParseError
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.mcp_conformance import (
    MCP_TYPE_STRING,
    A2A_TYPE_STRING,
    ConformanceReport,
    ProtocolModel,
    TransitionAnnotation,
    a2a_protocol,
    conformance_report,
    custom_protocol,
    format_report,
    generate_conformance_tests,
    mcp_protocol,
)


# ---------------------------------------------------------------------------
# Protocol models
# ---------------------------------------------------------------------------


class TestProtocolModels:
    """Test protocol model construction."""

    def test_mcp_protocol_parses(self):
        p = mcp_protocol()
        assert isinstance(p, ProtocolModel)
        assert p.ast is not None
        assert p.name == "MCP"

    def test_a2a_protocol_parses(self):
        p = a2a_protocol()
        assert isinstance(p, ProtocolModel)
        assert p.ast is not None
        assert p.name == "A2A"

    def test_mcp_type_string_matches_benchmark(self):
        from tests.benchmarks.protocols import BENCHMARKS
        mcp_bench = [b for b in BENCHMARKS if b.name == "MCP"][0]
        assert MCP_TYPE_STRING == mcp_bench.type_string

    def test_a2a_type_string_matches_benchmark(self):
        from tests.benchmarks.protocols import BENCHMARKS
        a2a_bench = [b for b in BENCHMARKS if b.name == "A2A"][0]
        assert A2A_TYPE_STRING == a2a_bench.type_string

    def test_mcp_has_version(self):
        assert mcp_protocol().version == "2024-11-05"

    def test_a2a_has_version(self):
        assert a2a_protocol().version == "1.0"

    def test_custom_protocol(self):
        p = custom_protocol("Test", "&{a: end}")
        assert p.name == "Test"
        assert p.version == "custom"
        assert p.annotations == ()

    def test_custom_protocol_bad_parse(self):
        with pytest.raises(ParseError):
            custom_protocol("Bad", "&{a: }")


# ---------------------------------------------------------------------------
# MCP state space
# ---------------------------------------------------------------------------


class TestMcpStateSpace:
    """Verify MCP state space properties."""

    @pytest.fixture
    def ss(self):
        return build_statespace(parse(MCP_TYPE_STRING))

    def test_mcp_states(self, ss):
        assert len(ss.states) == 7

    def test_mcp_transitions(self, ss):
        assert len(ss.transitions) == 17

    def test_mcp_uses_parallel(self, ss):
        assert ss.product_coords is not None

    def test_mcp_top_bottom(self, ss):
        assert ss.top != ss.bottom

    def test_mcp_lattice(self, ss):
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_mcp_reachable(self, ss):
        reachable = ss.reachable_from(ss.top)
        assert ss.bottom in reachable


# ---------------------------------------------------------------------------
# A2A state space
# ---------------------------------------------------------------------------


class TestA2aStateSpace:
    """Verify A2A state space properties."""

    @pytest.fixture
    def ss(self):
        return build_statespace(parse(A2A_TYPE_STRING))

    def test_a2a_states(self, ss):
        assert len(ss.states) == 5

    def test_a2a_transitions(self, ss):
        assert len(ss.transitions) == 7

    def test_a2a_no_parallel(self, ss):
        assert ss.product_coords is None

    def test_a2a_top_bottom(self, ss):
        assert ss.top != ss.bottom

    def test_a2a_lattice(self, ss):
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_a2a_reachable(self, ss):
        reachable = ss.reachable_from(ss.top)
        assert ss.bottom in reachable


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


class TestAnnotations:
    """Verify protocol transition annotations."""

    def test_mcp_annotations_cover_all_transitions(self):
        p = mcp_protocol()
        ss = build_statespace(p.ast)
        assert len(p.annotations) == len(ss.transitions)

    def test_mcp_initialize_is_client_request(self):
        p = mcp_protocol()
        init_anns = [a for a in p.annotations if a.label == "initialize"]
        assert len(init_anns) >= 1
        assert init_anns[0].initiator == "client"
        assert init_anns[0].message_kind == "request"

    def test_mcp_notification_is_server(self):
        p = mcp_protocol()
        notif_anns = [a for a in p.annotations if a.label == "NOTIFICATION"]
        assert all(a.initiator == "server" for a in notif_anns)
        assert all(a.message_kind == "notification" for a in notif_anns)

    def test_a2a_annotations_cover_all_transitions(self):
        p = a2a_protocol()
        ss = build_statespace(p.ast)
        assert len(p.annotations) == len(ss.transitions)

    def test_a2a_sendtask_is_client_request(self):
        p = a2a_protocol()
        send_anns = [a for a in p.annotations if a.label == "sendTask"]
        assert len(send_anns) == 1
        assert send_anns[0].initiator == "client"

    def test_a2a_working_is_server_response(self):
        p = a2a_protocol()
        working_anns = [a for a in p.annotations if a.label == "WORKING"]
        assert all(a.initiator == "server" for a in working_anns)


# ---------------------------------------------------------------------------
# Conformance report
# ---------------------------------------------------------------------------


class TestConformanceReport:
    """Test full conformance report generation."""

    @pytest.fixture
    def mcp_report(self):
        return conformance_report(mcp_protocol())

    @pytest.fixture
    def a2a_report(self):
        return conformance_report(a2a_protocol())

    def test_mcp_report_complete(self, mcp_report):
        r = mcp_report
        assert isinstance(r, ConformanceReport)
        assert r.protocol.name == "MCP"
        assert r.lattice_result is not None
        assert r.enumeration is not None
        assert r.coverage is not None
        assert r.petri is not None
        assert r.invariants is not None
        assert r.marking is not None

    def test_a2a_report_complete(self, a2a_report):
        assert isinstance(a2a_report, ConformanceReport)
        assert a2a_report.protocol.name == "A2A"

    def test_mcp_is_lattice(self, mcp_report):
        assert mcp_report.lattice_result.is_lattice

    def test_mcp_coverage_positive(self, mcp_report):
        assert mcp_report.coverage.transition_coverage > 0

    def test_mcp_petri_isomorphic(self, mcp_report):
        assert mcp_report.petri.reachability_isomorphic

    def test_mcp_invariants(self, mcp_report):
        assert mcp_report.invariants.num_states == 7

    def test_mcp_marking_isomorphic(self, mcp_report):
        assert mcp_report.marking.is_isomorphic_to_statespace

    def test_custom_report(self):
        p = custom_protocol("Simple", "&{a: end, b: end}")
        r = conformance_report(p)
        assert r.lattice_result.is_lattice
        assert r.num_valid_paths >= 1


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------


class TestTestGeneration:
    """Test conformance test source generation."""

    def test_mcp_generates_tests(self):
        src = generate_conformance_tests(mcp_protocol())
        assert len(src) > 0

    def test_a2a_generates_tests(self):
        src = generate_conformance_tests(a2a_protocol())
        assert len(src) > 0

    def test_mcp_tests_contain_class(self):
        src = generate_conformance_tests(mcp_protocol())
        assert "MCPProtocolTest" in src

    def test_mcp_valid_paths_exist(self):
        r = conformance_report(mcp_protocol())
        assert r.num_valid_paths > 0

    def test_mcp_violations_exist(self):
        r = conformance_report(mcp_protocol())
        assert r.num_violations > 0


# ---------------------------------------------------------------------------
# Format report
# ---------------------------------------------------------------------------


class TestFormatReport:
    """Test text report formatting."""

    @pytest.fixture
    def mcp_text(self):
        r = conformance_report(mcp_protocol())
        return format_report(r)

    def test_contains_protocol_name(self, mcp_text):
        assert "MCP" in mcp_text

    def test_contains_lattice_verdict(self, mcp_text):
        assert "WELL-FORMED" in mcp_text

    def test_contains_coverage(self, mcp_text):
        assert "coverage" in mcp_text.lower()

    def test_contains_invariants(self, mcp_text):
        assert "Mobius" in mcp_text or "mobius" in mcp_text.lower()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCliIntegration:
    """Test CLI entry point."""

    def test_cli_mcp_report(self, capsys):
        from reticulate.agent_test_cli import main
        main(["--protocol", "mcp", "--report"])
        captured = capsys.readouterr()
        assert "MCP" in captured.out
        assert "WELL-FORMED" in captured.out

    def test_cli_a2a_test_gen(self, capsys):
        from reticulate.agent_test_cli import main
        main(["--protocol", "a2a", "--test-gen"])
        captured = capsys.readouterr()
        assert "A2AProtocolTest" in captured.out

    def test_cli_custom(self, capsys):
        from reticulate.agent_test_cli import main
        main(["--protocol", "custom", "--type", "&{a: end}", "--invariants"])
        captured = capsys.readouterr()
        assert "States" in captured.out

    def test_cli_custom_missing_type(self):
        from reticulate.agent_test_cli import main
        with pytest.raises(SystemExit):
            main(["--protocol", "custom"])
