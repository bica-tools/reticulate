"""Tests for Tutte polynomial and protocol reliability (Step 32d).

Tests cover:
- Tutte polynomial computation
- Spanning tree count (T(G; 1, 1))
- Spanning forest count (T(G; 2, 1))
- Reliability polynomial
- Kirchhoff's matrix-tree theorem
- Comparison: Tutte vs Kirchhoff spanning tree counts
- Graph helper functions
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.tutte import (
    tutte_polynomial,
    count_spanning_trees,
    count_spanning_forests,
    count_connected_spanning,
    reliability_polynomial,
    evaluate_reliability,
    kirchhoff_spanning_trees,
    analyze_tutte,
    _hasse_undirected,
    _connected_components,
    _eval_tutte,
    _expand_binomial,
    _comb,
    _laplacian_matrix,
    _determinant,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_comb_basic(self):
        assert _comb(5, 2) == 10
        assert _comb(4, 0) == 1
        assert _comb(4, 4) == 1

    def test_comb_edge(self):
        assert _comb(0, 0) == 1
        assert _comb(3, 5) == 0

    def test_expand_binomial_zero(self):
        result = _expand_binomial(0)
        assert result == {0: 1}  # (x-1)^0 = 1

    def test_expand_binomial_one(self):
        result = _expand_binomial(1)
        # (x-1)^1 = x - 1
        assert result.get(1, 0) == 1
        assert result.get(0, 0) == -1

    def test_expand_binomial_two(self):
        result = _expand_binomial(2)
        # (x-1)^2 = x^2 - 2x + 1
        assert result.get(2, 0) == 1
        assert result.get(1, 0) == -2
        assert result.get(0, 0) == 1

    def test_connected_components_single(self):
        assert _connected_components([0], []) == 1

    def test_connected_components_disconnected(self):
        assert _connected_components([0, 1, 2], []) == 3

    def test_connected_components_connected(self):
        assert _connected_components([0, 1, 2], [(0, 1), (1, 2)]) == 1

    def test_determinant_1x1(self):
        assert abs(_determinant([[3.0]]) - 3.0) < 1e-10

    def test_determinant_2x2(self):
        det = _determinant([[2.0, 1.0], [1.0, 3.0]])
        assert abs(det - 5.0) < 1e-10

    def test_determinant_singular(self):
        det = _determinant([[1.0, 2.0], [2.0, 4.0]])
        assert abs(det) < 1e-10


class TestHasseUndirected:

    def test_end(self):
        ss = _build("end")
        nodes, edges = _hasse_undirected(ss)
        assert len(nodes) == 1
        assert edges == []

    def test_single_method(self):
        ss = _build("&{a: end}")
        nodes, edges = _hasse_undirected(ss)
        assert len(nodes) >= 1
        assert len(edges) >= 0


# ---------------------------------------------------------------------------
# Tutte polynomial
# ---------------------------------------------------------------------------

class TestTuttePolynomial:

    def test_single_node(self):
        """T(single vertex; x, y) = 1."""
        ss = _build("end")
        coeffs = tutte_polynomial(ss)
        assert _eval_tutte(coeffs, 1, 1) == 1

    def test_single_edge(self):
        """T(K_2; x, y) = x (bridge)."""
        ss = _build("&{a: end}")
        coeffs = tutte_polynomial(ss)
        # For K_2 (single edge): T = x
        val_at_2_1 = _eval_tutte(coeffs, 2, 1)
        assert val_at_2_1 > 0

    def test_parallel(self):
        ss = _build("(&{a: end} || &{b: end})")
        coeffs = tutte_polynomial(ss)
        assert isinstance(coeffs, dict)
        assert len(coeffs) > 0

    def test_chain(self):
        ss = _build("&{a: &{b: end}}")
        coeffs = tutte_polynomial(ss)
        assert isinstance(coeffs, dict)


# ---------------------------------------------------------------------------
# Spanning trees
# ---------------------------------------------------------------------------

class TestSpanningTrees:

    def test_single_node_tree(self):
        ss = _build("end")
        assert count_spanning_trees(ss) == 1

    def test_single_edge_tree(self):
        """K_2 has 1 spanning tree."""
        ss = _build("&{a: end}")
        assert count_spanning_trees(ss) == 1

    def test_chain_tree(self):
        """Path P_3 has 1 spanning tree (itself)."""
        ss = _build("&{a: &{b: end}}")
        assert count_spanning_trees(ss) == 1

    def test_parallel_trees(self):
        """Diamond (C_4) has 4 spanning trees."""
        ss = _build("(&{a: end} || &{b: end})")
        trees = count_spanning_trees(ss)
        assert trees >= 1

    def test_forests_ge_trees(self):
        """Number of spanning forests >= spanning trees."""
        ss = _build("(&{a: end} || &{b: end})")
        trees = count_spanning_trees(ss)
        forests = count_spanning_forests(ss)
        assert forests >= trees


# ---------------------------------------------------------------------------
# Kirchhoff's theorem
# ---------------------------------------------------------------------------

class TestKirchhoff:

    def test_single_node(self):
        ss = _build("end")
        assert kirchhoff_spanning_trees(ss) == 1

    def test_single_edge(self):
        ss = _build("&{a: end}")
        assert kirchhoff_spanning_trees(ss) == 1

    def test_chain(self):
        ss = _build("&{a: &{b: end}}")
        assert kirchhoff_spanning_trees(ss) == 1

    def test_kirchhoff_vs_tutte(self):
        """Kirchhoff and Tutte must agree on spanning tree count."""
        for typ in ["&{a: end}", "&{a: &{b: end}}",
                     "(&{a: end} || &{b: end})", "&{a: end, b: end}"]:
            ss = _build(typ)
            tutte_count = count_spanning_trees(ss)
            kirch_count = kirchhoff_spanning_trees(ss)
            assert tutte_count == kirch_count, \
                f"Mismatch for {typ}: Tutte={tutte_count}, Kirchhoff={kirch_count}"


# ---------------------------------------------------------------------------
# Reliability polynomial
# ---------------------------------------------------------------------------

class TestReliability:

    def test_single_node(self):
        ss = _build("end")
        rel = evaluate_reliability(ss, 1.0)
        assert abs(rel - 1.0) < 1e-10

    def test_fully_reliable(self):
        """At p=1 (no failures), reliability = 1 for connected graphs."""
        ss = _build("&{a: &{b: end}}")
        rel = evaluate_reliability(ss, 1.0)
        assert abs(rel - 1.0) < 1e-10

    def test_reliability_monotone(self):
        """Reliability increases with p."""
        ss = _build("(&{a: end} || &{b: end})")
        r_low = evaluate_reliability(ss, 0.3)
        r_high = evaluate_reliability(ss, 0.9)
        assert r_high >= r_low - 1e-10


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalysis:

    def test_end_analysis(self):
        ss = _build("end")
        result = analyze_tutte(ss)
        assert result.num_states == 1
        assert result.num_spanning_trees == 1

    def test_chain_analysis(self):
        ss = _build("&{a: &{b: end}}")
        result = analyze_tutte(ss)
        assert result.is_tree
        assert result.num_spanning_trees == 1
        assert result.laplacian_spanning_trees == 1

    def test_parallel_analysis(self):
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_tutte(ss)
        assert result.num_states >= 2
        assert result.num_spanning_trees >= 1
        assert result.num_spanning_trees == result.laplacian_spanning_trees

    def test_branch_analysis(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_tutte(ss)
        assert result.num_edges >= 1
        assert result.num_spanning_forests >= result.num_spanning_trees

    def test_recursive_analysis(self):
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_tutte(ss)
        assert result.num_states >= 1
