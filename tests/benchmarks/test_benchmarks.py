"""Parametrized tests for the 30 benchmark protocols.

Each protocol is parsed, its state space is built, and the lattice
properties are verified against expected metrics.
"""

from __future__ import annotations

import pytest

from reticulate import build_statespace, check_lattice, parse, pretty
from tests.benchmarks.protocols import BENCHMARKS, BenchmarkProtocol

# ---------------------------------------------------------------------------
# Parametrized tests — one run per protocol
# ---------------------------------------------------------------------------

BENCHMARK_IDS = [p.name for p in BENCHMARKS]


@pytest.fixture(params=BENCHMARKS, ids=BENCHMARK_IDS)
def protocol(request: pytest.FixtureRequest) -> BenchmarkProtocol:
    return request.param


def test_parse_roundtrip(protocol: BenchmarkProtocol) -> None:
    """Parsing the type string succeeds and pretty-printing round-trips."""
    ast = parse(protocol.type_string)
    ast2 = parse(pretty(ast))
    assert ast == ast2


def test_state_count(protocol: BenchmarkProtocol) -> None:
    ss = build_statespace(parse(protocol.type_string))
    assert len(ss.states) == protocol.expected_states


def test_transition_count(protocol: BenchmarkProtocol) -> None:
    ss = build_statespace(parse(protocol.type_string))
    assert len(ss.transitions) == protocol.expected_transitions


def test_is_lattice(protocol: BenchmarkProtocol) -> None:
    ss = build_statespace(parse(protocol.type_string))
    result = check_lattice(ss)
    assert result.is_lattice is True


def test_scc_count(protocol: BenchmarkProtocol) -> None:
    ss = build_statespace(parse(protocol.type_string))
    result = check_lattice(ss)
    assert result.num_scc == protocol.expected_sccs


def test_has_top(protocol: BenchmarkProtocol) -> None:
    ss = build_statespace(parse(protocol.type_string))
    result = check_lattice(ss)
    assert result.has_top is True


def test_has_bottom(protocol: BenchmarkProtocol) -> None:
    ss = build_statespace(parse(protocol.type_string))
    result = check_lattice(ss)
    assert result.has_bottom is True


def test_all_meets(protocol: BenchmarkProtocol) -> None:
    ss = build_statespace(parse(protocol.type_string))
    result = check_lattice(ss)
    assert result.all_meets_exist is True


def test_all_joins(protocol: BenchmarkProtocol) -> None:
    ss = build_statespace(parse(protocol.type_string))
    result = check_lattice(ss)
    assert result.all_joins_exist is True


def test_no_counterexample(protocol: BenchmarkProtocol) -> None:
    ss = build_statespace(parse(protocol.type_string))
    result = check_lattice(ss)
    assert result.counterexample is None


def test_top_reaches_bottom(protocol: BenchmarkProtocol) -> None:
    """The top (initial) state can reach the bottom (end) state."""
    ss = build_statespace(parse(protocol.type_string))
    assert ss.bottom in ss.reachable_from(ss.top)


# ---------------------------------------------------------------------------
# Structural tests — not parametrized
# ---------------------------------------------------------------------------


class TestBenchmarkSuite:
    """Tests about the benchmark suite itself."""

    def test_all_83_present(self) -> None:
        assert len(BENCHMARKS) == 83  # 79 original + 4 dialogue (Step 18)

    def test_unique_names(self) -> None:
        names = [p.name for p in BENCHMARKS]
        assert len(names) == len(set(names))

    def test_required_protocols_present(self) -> None:
        names = {p.name for p in BENCHMARKS}
        required = {
            "SMTP",
            "OAuth 2.0",
            "Two-Buyer",
            "MCP",
            "A2A",
            "HTTP Connection",
            "DB Transaction",
            "Java Iterator",
            "TLS Handshake",
            "Raft Leader Election",
            "Circuit Breaker",
            "Two-Phase Commit",
            "Saga Orchestrator",
            "Kafka Consumer",
            "Failover",
        }
        assert required.issubset(names)

    def test_parallel_count(self) -> None:
        parallel_count = sum(1 for p in BENCHMARKS if p.uses_parallel)
        assert parallel_count >= 16
