"""Tests for protocol_mining module (Step 83).

Tests cover:
- MiningResult dataclass
- Prefix tree construction and merging
- mine_from_traces with various trace patterns
- mine_from_statespace round-trip
- mine_from_logs with various log formats
- Log parsing
- Edge cases (empty traces, single traces, etc.)
- Coverage and confidence computation
- Lattice property of mined state spaces
"""

from __future__ import annotations

import pytest

from reticulate.parser import Branch, End, parse, pretty
from reticulate.statespace import build_statespace
from reticulate.csp import extract_traces, extract_complete_traces
from reticulate.protocol_mining import (
    MiningResult,
    mine_from_traces,
    mine_from_statespace,
    mine_from_logs,
    parse_log_lines,
    _build_pta,
    _pta_to_statespace,
    _accepts_trace,
    _compute_confidence,
)


# ---------------------------------------------------------------------------
# MiningResult dataclass
# ---------------------------------------------------------------------------

class TestMiningResult:
    """Tests for the MiningResult dataclass."""

    def test_mining_result_fields(self) -> None:
        result = MiningResult(
            inferred_type=End(),
            inferred_type_str="end",
            confidence=1.0,
            num_traces=5,
            coverage=1.0,
            is_lattice=True,
            num_states=1,
            num_transitions=0,
        )
        assert result.inferred_type == End()
        assert result.confidence == 1.0
        assert result.num_traces == 5
        assert result.coverage == 1.0
        assert result.is_lattice is True

    def test_mining_result_frozen(self) -> None:
        result = MiningResult(
            inferred_type=None,
            inferred_type_str="end",
            confidence=0.0,
            num_traces=0,
            coverage=0.0,
            is_lattice=True,
            num_states=0,
            num_transitions=0,
        )
        with pytest.raises(AttributeError):
            result.confidence = 0.5  # type: ignore


# ---------------------------------------------------------------------------
# mine_from_traces
# ---------------------------------------------------------------------------

class TestMineFromTraces:
    """Tests for mine_from_traces."""

    def test_empty_traces(self) -> None:
        result = mine_from_traces([])
        assert result.num_traces == 0
        assert result.confidence == 0.0
        assert result.inferred_type is None

    def test_all_empty_traces(self) -> None:
        result = mine_from_traces([(), ()])
        assert result.inferred_type == End()
        assert result.coverage == 1.0
        assert result.num_traces == 2

    def test_single_method_trace(self) -> None:
        traces = [("open",)]
        result = mine_from_traces(traces)
        assert result.inferred_type is not None
        assert result.num_traces == 1
        assert result.coverage == 1.0
        assert result.num_states >= 2  # at least start and end

    def test_linear_protocol(self) -> None:
        """A linear protocol: open -> read -> close."""
        traces = [("open", "read", "close")]
        result = mine_from_traces(traces)
        assert result.inferred_type is not None
        assert result.coverage == 1.0
        assert result.num_states >= 4  # open, read, close, end

    def test_branching_protocol(self) -> None:
        """Two traces sharing a prefix: open->read->close, open->write->close."""
        traces = [
            ("open", "read", "close"),
            ("open", "write", "close"),
        ]
        result = mine_from_traces(traces)
        assert result.inferred_type is not None
        assert result.coverage == 1.0
        assert "open" in result.inferred_type_str

    def test_multiple_branches(self) -> None:
        """Multiple branches from the initial state."""
        traces = [
            ("a",),
            ("b",),
            ("c",),
        ]
        result = mine_from_traces(traces)
        assert result.inferred_type is not None
        assert result.coverage == 1.0
        assert result.num_traces == 3

    def test_repeated_traces(self) -> None:
        """Repeated identical traces should not affect the result."""
        traces = [("a", "b"), ("a", "b"), ("a", "b")]
        result = mine_from_traces(traces)
        assert result.inferred_type is not None
        assert result.coverage == 1.0

    def test_coverage_computed(self) -> None:
        """All input traces should be accepted by the mined type."""
        traces = [
            ("init", "process", "done"),
            ("init", "skip", "done"),
        ]
        result = mine_from_traces(traces)
        assert result.coverage == 1.0

    def test_confidence_range(self) -> None:
        """Confidence should be in [0, 1]."""
        traces = [("a", "b"), ("a", "c")]
        result = mine_from_traces(traces)
        assert 0.0 <= result.confidence <= 1.0

    def test_state_space_attached(self) -> None:
        """The mined state space should be attached to the result."""
        traces = [("a",), ("b",)]
        result = mine_from_traces(traces)
        assert result.state_space is not None
        assert len(result.state_space.states) == result.num_states


# ---------------------------------------------------------------------------
# mine_from_statespace (round-trip)
# ---------------------------------------------------------------------------

class TestMineFromStatespace:
    """Tests for mine_from_statespace."""

    def test_simple_branch(self) -> None:
        """Round-trip: parse -> build_statespace -> mine_from_statespace."""
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        result = mine_from_statespace(ss)
        assert result.inferred_type is not None
        assert result.coverage == 1.0
        assert result.is_lattice is True

    def test_nested_branch(self) -> None:
        ast = parse("&{open: &{read: end, write: end}}")
        ss = build_statespace(ast)
        result = mine_from_statespace(ss)
        assert result.inferred_type is not None
        assert result.is_lattice is True

    def test_selection(self) -> None:
        ast = parse("+{OK: end, ERROR: end}")
        ss = build_statespace(ast)
        result = mine_from_statespace(ss)
        assert result.inferred_type is not None
        assert result.is_lattice is True

    def test_recursive_type(self) -> None:
        ast = parse("rec X . &{next: X, stop: end}")
        ss = build_statespace(ast)
        result = mine_from_statespace(ss)
        assert result.inferred_type is not None
        assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# mine_from_logs
# ---------------------------------------------------------------------------

class TestMineFromLogs:
    """Tests for mine_from_logs and log parsing."""

    def test_simple_method_calls(self) -> None:
        logs = ["CALL open", "CALL read", "CALL close"]
        result = mine_from_logs(logs)
        assert result.num_traces == 1
        assert result.inferred_type is not None

    def test_dotted_method_calls(self) -> None:
        logs = ["obj.open()", "obj.read()", "obj.close()"]
        result = mine_from_logs(logs)
        assert result.num_traces == 1

    def test_arrow_method_calls(self) -> None:
        logs = ["-> open", "-> read", "=> close"]
        result = mine_from_logs(logs)
        assert result.num_traces == 1

    def test_session_delimiters(self) -> None:
        """Blank lines and separators split sessions."""
        logs = [
            "CALL open", "CALL read", "CALL close",
            "",
            "CALL open", "CALL write", "CALL close",
        ]
        result = mine_from_logs(logs)
        assert result.num_traces == 2

    def test_separator_markers(self) -> None:
        logs = [
            "CALL a", "CALL b",
            "---",
            "CALL c", "CALL d",
        ]
        result = mine_from_logs(logs)
        assert result.num_traces == 2

    def test_empty_logs(self) -> None:
        result = mine_from_logs([])
        assert result.num_traces == 0

    def test_word_per_line(self) -> None:
        """Simple format: one method name per line."""
        logs = ["open", "read", "close"]
        result = mine_from_logs(logs)
        assert result.num_traces == 1
        assert result.inferred_type is not None


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

class TestParseLogLines:
    """Tests for parse_log_lines."""

    def test_call_format(self) -> None:
        traces = parse_log_lines(["CALL open", "CALL close"])
        assert len(traces) == 1
        assert traces[0] == ("open", "close")

    def test_invoke_format(self) -> None:
        traces = parse_log_lines(["INVOKE read"])
        assert len(traces) == 1
        assert traces[0] == ("read",)

    def test_method_format(self) -> None:
        traces = parse_log_lines(["METHOD process"])
        assert traces[0] == ("process",)

    def test_multi_session(self) -> None:
        traces = parse_log_lines(["CALL a", "", "CALL b"])
        assert len(traces) == 2
        assert traces[0] == ("a",)
        assert traces[1] == ("b",)

    def test_begin_end_delimiters(self) -> None:
        traces = parse_log_lines([
            "SESSION START",
            "CALL open",
            "CALL close",
            "SESSION END",
            "SESSION START",
            "CALL open",
            "SESSION END",
        ])
        assert len(traces) == 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for internal helper functions."""

    def test_build_pta_single_trace(self) -> None:
        root = _build_pta([("a", "b")])
        assert "a" in root.children
        assert "b" in root.children["a"].children
        assert root.children["a"].children["b"].is_terminal

    def test_build_pta_shared_prefix(self) -> None:
        root = _build_pta([("a", "b"), ("a", "c")])
        assert "a" in root.children
        a_node = root.children["a"]
        assert "b" in a_node.children
        assert "c" in a_node.children

    def test_pta_to_statespace(self) -> None:
        root = _build_pta([("a",), ("b",)])
        ss = _pta_to_statespace(root)
        assert ss.top in ss.states
        assert ss.bottom in ss.states
        assert len(ss.transitions) >= 2

    def test_accepts_trace(self) -> None:
        root = _build_pta([("a", "b"), ("a", "c")])
        ss = _pta_to_statespace(root)
        assert _accepts_trace(ss, ("a", "b"))
        assert _accepts_trace(ss, ("a", "c"))
        assert not _accepts_trace(ss, ("x",))

    def test_confidence_full_coverage(self) -> None:
        conf = _compute_confidence(1.0, True, 10, 5)
        assert conf == 1.0

    def test_confidence_no_lattice(self) -> None:
        conf = _compute_confidence(1.0, False, 10, 5)
        assert conf < 1.0

    def test_confidence_few_traces(self) -> None:
        conf_few = _compute_confidence(1.0, True, 1, 5)
        conf_many = _compute_confidence(1.0, True, 10, 5)
        assert conf_few < conf_many


# ---------------------------------------------------------------------------
# Integration: round-trip with CSP traces
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Test that mining from extracted traces recovers the original type."""

    def test_roundtrip_simple_branch(self) -> None:
        """mine_from_traces(extract_traces(build_statespace(S))) should
        produce a type that accepts all the same traces."""
        original = "&{a: end, b: end}"
        ast = parse(original)
        ss = build_statespace(ast)
        traces = list(extract_complete_traces(ss))
        result = mine_from_traces(traces)
        assert result.coverage == 1.0

    def test_roundtrip_deeper(self) -> None:
        original = "&{open: &{read: end, write: end}}"
        ast = parse(original)
        ss = build_statespace(ast)
        traces = list(extract_complete_traces(ss))
        result = mine_from_traces(traces)
        assert result.coverage == 1.0

    def test_roundtrip_with_selection(self) -> None:
        """Selection types should also round-trip."""
        original = "&{login: +{OK: end, FAIL: end}}"
        ast = parse(original)
        ss = build_statespace(ast)
        traces = list(extract_complete_traces(ss))
        result = mine_from_traces(traces)
        assert result.coverage == 1.0
