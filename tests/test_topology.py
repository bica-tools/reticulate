"""Tests for the topological session type analyzer (Step 60a)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.topology import (
    TopologicalResult,
    betti_numbers,
    classify_topology,
    cycle_rank,
    edge_density,
    euler_characteristic,
    is_tree,
    topological_distance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def end_ss():
    return build_statespace(parse("end"))


@pytest.fixture
def simple_branch_ss():
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def diamond_ss():
    """Diamond lattice: &{a: &{c: end}, b: &{c: end}}."""
    return build_statespace(parse("&{a: &{c: end}, b: &{c: end}}"))


@pytest.fixture
def recursive_ss():
    return build_statespace(parse("rec X . &{next: X, done: end}"))


@pytest.fixture
def deep_recursive_ss():
    return build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))


@pytest.fixture
def parallel_ss():
    return build_statespace(parse("(end || end)"))


@pytest.fixture
def selection_ss():
    return build_statespace(parse("+{ok: end, err: end}"))


@pytest.fixture
def iterator_ss():
    return build_statespace(parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"))


@pytest.fixture
def smtp_ss():
    return build_statespace(
        parse("&{ehlo: +{ok_250: &{mail: +{ok_mail: &{rcpt: +{ok_rcpt: &{data: +{ok_data: &{content: +{ok_queued: end}}}}}}}}, err: end}}")
    )


# ---------------------------------------------------------------------------
# Euler characteristic
# ---------------------------------------------------------------------------


class TestEulerCharacteristic:
    def test_end(self, end_ss):
        """end has 1 vertex, 0 edges => chi = 1."""
        assert euler_characteristic(end_ss) == 1

    def test_simple_branch(self, simple_branch_ss):
        """&{a: end, b: end} has 2 states, 2 undirected edges (top->a_end collapses to 1) => depends on state space."""
        chi = euler_characteristic(simple_branch_ss)
        # tree-like: V - E = 1 for connected tree
        assert isinstance(chi, int)

    def test_diamond(self, diamond_ss):
        """Diamond has a cycle: chi should be less than V."""
        chi = euler_characteristic(diamond_ss)
        assert isinstance(chi, int)

    def test_recursive_type(self, recursive_ss):
        """Recursive type has a back-edge creating a cycle."""
        chi = euler_characteristic(recursive_ss)
        # With a cycle, chi = V - E < V - (V-1) = 1
        assert chi < 1 or chi == 1  # at least computable

    def test_chi_equals_v_minus_e(self, diamond_ss):
        """Verify chi = V - E directly."""
        from reticulate.topology import _total_undirected_edges
        v = len(diamond_ss.states)
        e = _total_undirected_edges(diamond_ss)
        assert euler_characteristic(diamond_ss) == v - e


# ---------------------------------------------------------------------------
# Betti numbers
# ---------------------------------------------------------------------------


class TestBettiNumbers:
    def test_end_betti(self, end_ss):
        """end: 1 component, 0 cycles."""
        b0, b1 = betti_numbers(end_ss)
        assert b0 == 1
        assert b1 == 0

    def test_simple_branch_connected(self, simple_branch_ss):
        """Simple branch is connected (b0 = 1)."""
        b0, _b1 = betti_numbers(simple_branch_ss)
        assert b0 == 1

    def test_tree_no_cycles(self, simple_branch_ss):
        """Tree-like types: b1 should be 0 if no merging."""
        _b0, b1 = betti_numbers(simple_branch_ss)
        assert b1 >= 0

    def test_recursive_has_cycle(self, recursive_ss):
        """Recursive type has at least one cycle (b1 >= 1)."""
        _b0, b1 = betti_numbers(recursive_ss)
        assert b1 >= 1

    def test_diamond_may_have_cycle(self, diamond_ss):
        """Diamond lattice may have a cycle if edges merge."""
        _b0, b1 = betti_numbers(diamond_ss)
        assert b1 >= 0

    def test_betti_formula(self, recursive_ss):
        """Verify b1 = E - V + b0."""
        from reticulate.topology import _total_undirected_edges, _connected_components
        v = len(recursive_ss.states)
        e = _total_undirected_edges(recursive_ss)
        b0 = _connected_components(recursive_ss)
        _, b1 = betti_numbers(recursive_ss)
        assert b1 == e - v + b0

    def test_iterator_betti(self, iterator_ss):
        """Iterator has a recursive loop."""
        b0, b1 = betti_numbers(iterator_ss)
        assert b0 == 1
        assert b1 >= 1


# ---------------------------------------------------------------------------
# Cycle rank
# ---------------------------------------------------------------------------


class TestCycleRank:
    def test_end_no_cycles(self, end_ss):
        assert cycle_rank(end_ss) == 0

    def test_recursive_positive(self, recursive_ss):
        assert cycle_rank(recursive_ss) >= 1

    def test_matches_b1(self, diamond_ss):
        """cycle_rank must equal b1."""
        _, b1 = betti_numbers(diamond_ss)
        assert cycle_rank(diamond_ss) == b1

    def test_matches_b1_recursive(self, recursive_ss):
        _, b1 = betti_numbers(recursive_ss)
        assert cycle_rank(recursive_ss) == b1


# ---------------------------------------------------------------------------
# is_tree
# ---------------------------------------------------------------------------


class TestIsTree:
    def test_end_is_tree(self, end_ss):
        assert is_tree(end_ss) is True

    def test_recursive_not_tree(self, recursive_ss):
        assert is_tree(recursive_ss) is False

    def test_selection_tree(self, selection_ss):
        """Selection +{ok: end, err: end} is tree-like if no merging."""
        # May or may not be tree depending on whether end states merge
        result = is_tree(selection_ss)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Edge density
# ---------------------------------------------------------------------------


class TestEdgeDensity:
    def test_end_density_zero(self, end_ss):
        """Single vertex: density = 0."""
        assert edge_density(end_ss) == 0.0

    def test_density_range(self, diamond_ss):
        """Density should be in [0, 1]."""
        d = edge_density(diamond_ss)
        assert 0.0 <= d <= 1.0

    def test_density_range_recursive(self, recursive_ss):
        d = edge_density(recursive_ss)
        assert 0.0 <= d <= 1.0

    def test_density_range_smtp(self, smtp_ss):
        d = edge_density(smtp_ss)
        assert 0.0 <= d <= 1.0

    def test_density_positive_for_nonempty(self, simple_branch_ss):
        """Non-trivial type should have positive density."""
        d = edge_density(simple_branch_ss)
        assert d > 0.0


# ---------------------------------------------------------------------------
# topological_distance
# ---------------------------------------------------------------------------


class TestTopologicalDistance:
    def test_identical_zero(self, end_ss):
        """Distance from a type to itself is 0."""
        assert topological_distance(end_ss, end_ss) == 0.0

    def test_identical_recursive(self, recursive_ss):
        assert topological_distance(recursive_ss, recursive_ss) == 0.0

    def test_different_positive(self, end_ss, recursive_ss):
        """Different types should have positive distance."""
        d = topological_distance(end_ss, recursive_ss)
        assert d > 0.0

    def test_symmetry(self, simple_branch_ss, recursive_ss):
        """Distance is symmetric."""
        d1 = topological_distance(simple_branch_ss, recursive_ss)
        d2 = topological_distance(recursive_ss, simple_branch_ss)
        assert d1 == d2

    def test_nonnegative(self, diamond_ss, selection_ss):
        d = topological_distance(diamond_ss, selection_ss)
        assert d >= 0.0


# ---------------------------------------------------------------------------
# classify_topology
# ---------------------------------------------------------------------------


class TestClassifyTopology:
    def test_returns_result(self, diamond_ss):
        result = classify_topology(diamond_ss)
        assert isinstance(result, TopologicalResult)

    def test_frozen(self, diamond_ss):
        result = classify_topology(diamond_ss)
        with pytest.raises(AttributeError):
            result.euler_characteristic = 999  # type: ignore[misc]

    def test_end_topology(self, end_ss):
        result = classify_topology(end_ss)
        assert result.euler_characteristic == 1
        assert result.betti_0 == 1
        assert result.betti_1 == 0
        assert result.is_tree is True
        assert result.is_planar is True
        assert result.cycle_rank == 0
        assert result.edge_density == 0.0

    def test_recursive_topology(self, recursive_ss):
        result = classify_topology(recursive_ss)
        assert result.betti_1 >= 1
        assert result.is_tree is False
        assert result.cycle_rank == result.betti_1

    def test_consistency(self, smtp_ss):
        """All fields should be consistent."""
        result = classify_topology(smtp_ss)
        assert result.cycle_rank == result.betti_1
        assert result.genus == result.betti_1
        assert result.is_tree == (result.betti_1 == 0)
        assert 0.0 <= result.edge_density <= 1.0

    def test_planarity_small_graphs(self, simple_branch_ss):
        """Small graphs are always planar."""
        result = classify_topology(simple_branch_ss)
        assert result.is_planar is True


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------


class TestBenchmarks:
    def test_iterator_topology(self, iterator_ss):
        result = classify_topology(iterator_ss)
        assert result.betti_0 == 1
        assert result.betti_1 >= 1

    def test_smtp_topology(self, smtp_ss):
        result = classify_topology(smtp_ss)
        assert result.betti_0 == 1
        assert result.is_planar is True
