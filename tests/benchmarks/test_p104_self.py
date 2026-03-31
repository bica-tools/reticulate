"""Tests for P104 self-referencing benchmarks.

Every P104 component's protocol is analyzed by the tool it specifies.
This is the self-referencing dogfooding suite.
"""

from __future__ import annotations

import pytest

from reticulate.lattice import check_distributive, check_lattice
from reticulate.modularity import analyze_modularity
from reticulate.parser import parse
from reticulate.statespace import build_statespace

from tests.benchmarks.p104_self import P104_BENCHMARKS, P104Benchmark


# ---------------------------------------------------------------------------
# Parametrized: verify state-space metrics for every benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bench",
    P104_BENCHMARKS,
    ids=[b.name for b in P104_BENCHMARKS],
)
class TestP104SelfBenchmarks:
    """Verify each P104 component protocol matches its expected metrics."""

    def test_parses(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        assert ast is not None

    def test_state_count(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        assert len(ss.states) == bench.expected_states, (
            f"{bench.name}: expected {bench.expected_states} states, "
            f"got {len(ss.states)}"
        )

    def test_transition_count(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        assert len(ss.transitions) == bench.expected_transitions, (
            f"{bench.name}: expected {bench.expected_transitions} transitions, "
            f"got {len(ss.transitions)}"
        )

    def test_is_lattice(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.is_lattice == bench.expected_lattice, (
            f"{bench.name}: expected lattice={bench.expected_lattice}"
        )

    def test_scc_count(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.num_scc == bench.expected_sccs, (
            f"{bench.name}: expected {bench.expected_sccs} SCCs, "
            f"got {result.num_scc}"
        )

    def test_distributivity(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        dr = check_distributive(ss)
        assert dr.is_distributive == bench.expected_distributive, (
            f"{bench.name}: expected distributive={bench.expected_distributive}, "
            f"got {dr.is_distributive} (classification={dr.classification})"
        )


# ---------------------------------------------------------------------------
# Modularity analysis: deeper checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bench",
    P104_BENCHMARKS,
    ids=[b.name for b in P104_BENCHMARKS],
)
class TestP104ModularityAnalysis:
    """Run full modularity analysis on each P104 component protocol."""

    def test_modularity_runs(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        analysis = analyze_modularity(ss)
        assert analysis.modularity.is_lattice

    def test_modularity_matches_distributivity(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        analysis = analyze_modularity(ss)
        assert analysis.modularity.is_distributive == bench.expected_distributive

    def test_coupling_score_finite(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        analysis = analyze_modularity(ss)
        assert 0.0 <= analysis.coupling <= 1.0

    def test_interface_width_non_negative(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        analysis = analyze_modularity(ss)
        assert analysis.interface_width >= 0.0


# ---------------------------------------------------------------------------
# Specific: Importer non-modularity diagnosis
# ---------------------------------------------------------------------------


class TestP104ImporterNonModularity:
    """The Importer protocol is non-distributive.  Verify diagnosis."""

    def test_importer_has_m3(self) -> None:
        from tests.benchmarks.p104_self import P104_BY_COMPONENT
        bench = P104_BY_COMPONENT["Importer"]
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        analysis = analyze_modularity(ss)
        assert analysis.diagnosis is not None
        assert analysis.diagnosis.has_m3

    def test_importer_refactoring_suggestions(self) -> None:
        from tests.benchmarks.p104_self import P104_BY_COMPONENT
        bench = P104_BY_COMPONENT["Importer"]
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        analysis = analyze_modularity(ss)
        assert len(analysis.refactorings) > 0


# ---------------------------------------------------------------------------
# Self-reference: all P104 protocols form lattices
# ---------------------------------------------------------------------------


class TestP104AllLattices:
    """Every P104 component protocol forms a bounded lattice."""

    def test_all_are_lattices(self) -> None:
        for bench in P104_BENCHMARKS:
            ast = parse(bench.type_string)
            ss = build_statespace(ast)
            result = check_lattice(ss)
            assert result.is_lattice, f"{bench.name} is not a lattice"

    def test_exactly_one_non_distributive(self) -> None:
        non_dist = []
        for bench in P104_BENCHMARKS:
            ast = parse(bench.type_string)
            ss = build_statespace(ast)
            dr = check_distributive(ss)
            if not dr.is_distributive:
                non_dist.append(bench.name)
        assert non_dist == ["P104-Importer"]
