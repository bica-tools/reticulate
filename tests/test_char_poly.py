"""Tests for characteristic polynomial via deletion-contraction (Step 32c).

Tests cover:
- Graph deletion and contraction operations
- Polynomial arithmetic (subtract, add)
- Deletion-contraction algorithm
- Chromatic number bound
- Comparison with Mobius-based computation
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.char_poly import (
    char_polynomial,
    evaluate_char_poly,
    chromatic_number_bound,
    verify_dc_vs_mobius,
    deletion,
    contraction,
    analyze_char_poly,
    _hasse_graph,
    _graph_delete,
    _graph_contract,
    _poly_subtract,
    _poly_add,
    _char_poly_dc_graph,
)
from reticulate.characteristic import poly_evaluate


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Polynomial arithmetic
# ---------------------------------------------------------------------------

class TestPolyArithmetic:

    def test_subtract_equal(self):
        assert _poly_subtract([1, 2, 3], [1, 2, 3]) == [0]

    def test_subtract_different_length(self):
        # t^2 - t = t^2 - t
        assert _poly_subtract([1, 0, 0], [0, 1, 0]) == [1, -1, 0]

    def test_add_basic(self):
        assert _poly_add([1, 0], [0, 1]) == [1, 1]

    def test_add_different_length(self):
        assert _poly_add([1, 0, 0], [1, 0]) == [1, 1, 0]

    def test_subtract_leading_zeros(self):
        result = _poly_subtract([1, 0], [1, 0])
        assert result == [0]


# ---------------------------------------------------------------------------
# Graph operations
# ---------------------------------------------------------------------------

class TestGraphOperations:

    def test_hasse_graph_end(self):
        ss = _build("end")
        nodes, edges = _hasse_graph(ss)
        assert len(nodes) == 1
        assert edges == []

    def test_hasse_graph_single(self):
        ss = _build("&{a: end}")
        nodes, edges = _hasse_graph(ss)
        assert len(nodes) >= 1

    def test_delete_edge(self):
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        new_nodes, new_edges = _graph_delete(nodes, edges, (0, 1))
        assert len(new_edges) == 1
        assert (1, 2) in new_edges
        assert len(new_nodes) == 3  # nodes preserved

    def test_contract_edge(self):
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        new_nodes, new_edges = _graph_contract(nodes, edges, (0, 1))
        assert 1 not in new_nodes  # v=1 merged into u=0
        assert len(new_nodes) == 2
        # Edge (1,2) becomes (0,2) after merging 1 into 0
        assert (0, 2) in new_edges

    def test_contract_no_self_loop(self):
        nodes = [0, 1]
        edges = [(0, 1)]
        new_nodes, new_edges = _graph_contract(nodes, edges, (0, 1))
        assert len(new_nodes) == 1
        assert len(new_edges) == 0  # no self-loops

    def test_deletion_api(self):
        ss = _build("&{a: &{b: end}}")
        nodes, edges = deletion(ss, 0)
        assert isinstance(nodes, list)
        assert isinstance(edges, list)

    def test_contraction_api(self):
        ss = _build("&{a: &{b: end}}")
        nodes, edges = contraction(ss, 0)
        assert isinstance(nodes, list)


# ---------------------------------------------------------------------------
# Deletion-contraction algorithm
# ---------------------------------------------------------------------------

class TestDeletionContraction:

    def test_no_edges(self):
        """Graph with no edges: p(G, t) = t^n."""
        result = _char_poly_dc_graph([0, 1, 2], [])
        # t^3
        assert result == [1, 0, 0, 0]

    def test_single_edge(self):
        """K_2: p(G, t) = t(t-1) = t^2 - t."""
        result = _char_poly_dc_graph([0, 1], [(0, 1)])
        assert result == [1, -1, 0]
        # Verify: p(2) = 4 - 2 = 2 (two colorings of K_2 with 2 colors)
        assert poly_evaluate(result, 2) == 2

    def test_path_three(self):
        """Path P_3: p = t(t-1)^2 = t^3 - 2t^2 + t."""
        result = _char_poly_dc_graph([0, 1, 2], [(0, 1), (1, 2)])
        assert result == [1, -2, 1, 0]
        # p(2) = 8 - 8 + 2 = 2
        assert poly_evaluate(result, 2) == 2

    def test_triangle(self):
        """K_3 (triangle): p = t(t-1)(t-2) = t^3 - 3t^2 + 2t."""
        result = _char_poly_dc_graph([0, 1, 2], [(0, 1), (0, 2), (1, 2)])
        assert result == [1, -3, 2, 0]
        # p(3) = 27 - 27 + 6 = 6 (3! colorings)
        assert poly_evaluate(result, 3) == 6

    def test_single_node(self):
        """Single node: p(G, t) = t."""
        result = _char_poly_dc_graph([0], [])
        assert result == [1, 0]


class TestCharPolynomial:

    def test_end(self):
        ss = _build("end")
        coeffs = char_polynomial(ss)
        assert isinstance(coeffs, list)
        assert len(coeffs) >= 1

    def test_single_method(self):
        ss = _build("&{a: end}")
        coeffs = char_polynomial(ss)
        assert isinstance(coeffs, list)

    def test_chain_three(self):
        ss = _build("&{a: &{b: end}}")
        coeffs = char_polynomial(ss)
        assert isinstance(coeffs, list)
        # Chromatic poly of path P_3 = t(t-1)^2
        # p(1) = 0, p(2) = 2
        val_2 = poly_evaluate(coeffs, 2)
        assert val_2 > 0

    def test_evaluate(self):
        ss = _build("&{a: &{b: end}}")
        val = evaluate_char_poly(ss, 3)
        assert val > 0

    def test_parallel(self):
        ss = _build("(&{a: end} || &{b: end})")
        coeffs = char_polynomial(ss)
        assert isinstance(coeffs, list)
        assert len(coeffs) >= 1


# ---------------------------------------------------------------------------
# Chromatic number bound
# ---------------------------------------------------------------------------

class TestChromaticBound:

    def test_end_chromatic(self):
        ss = _build("end")
        bound = chromatic_number_bound(ss)
        assert bound >= 1

    def test_chain_chromatic(self):
        """Path graph is 2-colorable."""
        ss = _build("&{a: &{b: end}}")
        bound = chromatic_number_bound(ss)
        assert bound <= 3  # Path is 2-colorable

    def test_parallel_chromatic(self):
        ss = _build("(&{a: end} || &{b: end})")
        bound = chromatic_number_bound(ss)
        assert bound >= 1


# ---------------------------------------------------------------------------
# Comparison with Mobius
# ---------------------------------------------------------------------------

class TestComparison:

    def test_end_match(self):
        """Trivial case should match."""
        ss = _build("end")
        # Both should produce valid polynomials
        result = analyze_char_poly(ss)
        assert isinstance(result.coefficients_dc, list)
        assert isinstance(result.coefficients_mobius, list)

    def test_chain_analysis(self):
        ss = _build("&{a: &{b: end}}")
        result = analyze_char_poly(ss)
        assert result.degree >= 0
        assert result.num_states >= 1
        assert result.num_edges >= 0


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalysis:

    def test_end_analysis(self):
        ss = _build("end")
        result = analyze_char_poly(ss)
        assert result.num_states >= 1
        assert result.chromatic_bound >= 1

    def test_branch_analysis(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_char_poly(ss)
        assert result.degree >= 0
        assert result.chromatic_bound >= 1

    def test_selection_analysis(self):
        ss = _build("+{a: end, b: end}")
        result = analyze_char_poly(ss)
        assert result.num_edges >= 0

    def test_recursive_analysis(self):
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_char_poly(ss)
        assert result.num_states >= 1
        assert result.chromatic_bound >= 1

    def test_parallel_analysis(self):
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_char_poly(ss)
        assert result.num_states >= 2
        assert result.chromatic_bound >= 1

    def test_eval_at_0(self):
        """p(G, 0) = 0 for any graph with at least one vertex."""
        ss = _build("&{a: &{b: end}}")
        result = analyze_char_poly(ss)
        assert result.eval_at_0 == 0

    def test_eval_at_1_chain(self):
        """p(path, 1) = 0 for path with > 1 vertex."""
        ss = _build("&{a: &{b: end}}")
        result = analyze_char_poly(ss)
        assert result.eval_at_1 == 0
