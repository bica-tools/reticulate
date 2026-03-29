"""Tests for Ramanujan optimality of session type lattices (Step 31f)."""

import math
import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.ramanujan import (
    DegreeStats,
    RamanujanResult,
    adjacency_eigenvalues,
    alon_boppana_bound,
    degree_stats,
    is_ramanujan,
    optimal_expansion_check,
    ramanujan_gap,
    ramanujan_ratio,
)


def _ss(type_string: str):
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Degree statistics
# ---------------------------------------------------------------------------


class TestDegreeStats:
    def test_end_stats(self):
        stats = degree_stats(_ss("end"))
        assert stats.min_degree == 0
        assert stats.max_degree == 0
        assert stats.is_regular

    def test_simple_branch_stats(self):
        stats = degree_stats(_ss("&{a: end}"))
        assert stats.min_degree >= 1
        assert stats.max_degree >= 1
        assert len(stats.degree_sequence) == 2

    def test_chain_stats(self):
        stats = degree_stats(_ss("&{a: &{b: end}}"))
        # Path graph: endpoints have degree 1, middle has degree 2
        assert stats.min_degree == 1
        assert stats.max_degree == 2
        assert not stats.is_regular

    def test_diamond_stats(self):
        stats = degree_stats(_ss("&{a: &{c: end}, b: &{c: end}}"))
        assert stats.max_degree >= 2
        assert stats.avg_degree > 0

    def test_parallel_stats(self):
        stats = degree_stats(_ss("(&{a: end} || &{b: end})"))
        assert stats.max_degree >= 2


# ---------------------------------------------------------------------------
# Adjacency eigenvalues
# ---------------------------------------------------------------------------


class TestAdjacencyEigenvalues:
    def test_end_eigenvalues(self):
        eigs = adjacency_eigenvalues(_ss("end"))
        assert len(eigs) == 1
        assert abs(eigs[0]) < 1e-10

    def test_simple_branch_eigenvalues(self):
        eigs = adjacency_eigenvalues(_ss("&{a: end}"))
        assert len(eigs) == 2
        # Path P_2 eigenvalues: -1, 1
        assert abs(eigs[0] - (-1.0)) < 0.1
        assert abs(eigs[1] - 1.0) < 0.1

    def test_chain_eigenvalues(self):
        eigs = adjacency_eigenvalues(_ss("&{a: &{b: end}}"))
        assert len(eigs) == 3
        # Should be sorted
        for i in range(len(eigs) - 1):
            assert eigs[i] <= eigs[i + 1] + 1e-10

    def test_eigenvalues_sorted(self):
        eigs = adjacency_eigenvalues(_ss("&{a: &{c: end}, b: &{c: end}}"))
        for i in range(len(eigs) - 1):
            assert eigs[i] <= eigs[i + 1] + 1e-10


# ---------------------------------------------------------------------------
# Alon-Boppana bound
# ---------------------------------------------------------------------------


class TestAlonBoppanaBound:
    def test_degree_1(self):
        assert alon_boppana_bound(1) == 0.0

    def test_degree_0(self):
        assert alon_boppana_bound(0) == 0.0

    def test_degree_2(self):
        assert abs(alon_boppana_bound(2) - 2.0) < 1e-10

    def test_degree_3(self):
        expected = 2.0 * math.sqrt(2)
        assert abs(alon_boppana_bound(3) - expected) < 1e-10

    def test_degree_5(self):
        expected = 2.0 * math.sqrt(4)
        assert abs(alon_boppana_bound(5) - expected) < 1e-10
        assert abs(alon_boppana_bound(5) - 4.0) < 1e-10

    def test_monotone(self):
        for d in range(2, 10):
            assert alon_boppana_bound(d) <= alon_boppana_bound(d + 1)


# ---------------------------------------------------------------------------
# Ramanujan check
# ---------------------------------------------------------------------------


class TestIsRamanujan:
    def test_end_is_ramanujan(self):
        assert is_ramanujan(_ss("end"))

    def test_simple_branch_ramanujan(self):
        assert is_ramanujan(_ss("&{a: end}"))

    def test_chain_ramanujan(self):
        # Path graphs are typically Ramanujan
        result = is_ramanujan(_ss("&{a: &{b: end}}"))
        assert isinstance(result, bool)

    def test_diamond_ramanujan(self):
        result = is_ramanujan(_ss("&{a: &{c: end}, b: &{c: end}}"))
        assert isinstance(result, bool)

    def test_parallel_ramanujan(self):
        result = is_ramanujan(_ss("(&{a: end} || &{b: end})"))
        assert isinstance(result, bool)

    def test_recursive_ramanujan(self):
        result = is_ramanujan(_ss("rec X . &{a: X, b: end}"))
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Ramanujan gap
# ---------------------------------------------------------------------------


class TestRamanujanGap:
    def test_end_gap_zero(self):
        assert ramanujan_gap(_ss("end")) == 0.0

    def test_simple_branch_gap(self):
        gap = ramanujan_gap(_ss("&{a: end}"))
        assert isinstance(gap, float)

    def test_chain_gap(self):
        gap = ramanujan_gap(_ss("&{a: &{b: end}}"))
        assert isinstance(gap, float)

    def test_ramanujan_has_nonpositive_gap(self):
        ss = _ss("&{a: end}")
        if is_ramanujan(ss):
            assert ramanujan_gap(ss) <= 1e-9


# ---------------------------------------------------------------------------
# Ramanujan ratio
# ---------------------------------------------------------------------------


class TestRamanujanRatio:
    def test_end_ratio_zero(self):
        assert ramanujan_ratio(_ss("end")) == 0.0

    def test_ratio_nonnegative(self):
        ratio = ramanujan_ratio(_ss("&{a: &{b: end}}"))
        assert ratio >= 0.0

    def test_ramanujan_ratio_at_most_one(self):
        ss = _ss("&{a: end}")
        if is_ramanujan(ss):
            assert ramanujan_ratio(ss) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


class TestOptimalExpansionCheck:
    def test_end_analysis(self):
        result = optimal_expansion_check(_ss("end"))
        assert isinstance(result, RamanujanResult)
        assert result.num_states == 1
        assert result.is_ramanujan

    def test_simple_branch_analysis(self):
        result = optimal_expansion_check(_ss("&{a: end}"))
        assert result.num_states == 2
        assert result.num_edges >= 1

    def test_chain_analysis(self):
        result = optimal_expansion_check(_ss("&{a: &{b: end}}"))
        assert result.num_states == 3
        assert result.degree_stats.max_degree >= 1
        assert len(result.eigenvalues) == 3

    def test_diamond_analysis(self):
        result = optimal_expansion_check(_ss("&{a: &{c: end}, b: &{c: end}}"))
        assert result.alon_boppana >= 0.0
        assert isinstance(result.optimal_expansion, bool)

    def test_parallel_analysis(self):
        result = optimal_expansion_check(_ss("(&{a: end} || &{b: end})"))
        assert result.num_states >= 3

    def test_recursive_analysis(self):
        result = optimal_expansion_check(_ss("rec X . &{a: X, b: end}"))
        assert isinstance(result, RamanujanResult)

    def test_gap_consistent_with_is_ramanujan(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        result = optimal_expansion_check(ss)
        if result.is_ramanujan:
            assert result.ramanujan_gap <= 1e-9
        else:
            assert result.ramanujan_gap > -1e-9

    def test_ratio_consistent_with_is_ramanujan(self):
        ss = _ss("&{a: &{b: end}}")
        result = optimal_expansion_check(ss)
        if result.is_ramanujan and result.alon_boppana > 1e-15:
            assert result.ramanujan_ratio <= 1.0 + 1e-9
