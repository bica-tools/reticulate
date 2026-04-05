"""Tests for LGV Lemma analysis (Step 30ac).

Tests cover:
- Path counting on simple structures (end, chain, diamond, parallel)
- Path count matrix dimensions and values
- Determinant computation (Bareiss and Leibniz)
- LGV verification (det = signed non-intersecting count)
- Auto source/sink detection via rank layers
- Full analysis on benchmark protocols
- Edge cases (single state, recursive types)
- Permutation sign computation
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lgv import (
    LGVResult,
    _count_paths,
    _determinant,
    _determinant_leibniz,
    _determinant_safe,
    _perm_sign,
    _signed_non_intersecting_count,
    analyze_lgv,
    find_rank_layers,
    lgv_determinant,
    non_intersecting_count,
    path_count_matrix,
    verify_lgv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str):
    """Parse and build state space."""
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Permutation sign
# ---------------------------------------------------------------------------

class TestPermSign:
    """Tests for permutation sign computation."""

    def test_identity_1(self):
        assert _perm_sign((0,)) == 1

    def test_identity_3(self):
        assert _perm_sign((0, 1, 2)) == 1

    def test_transposition(self):
        assert _perm_sign((1, 0)) == -1

    def test_cycle_3(self):
        # (0 1 2) = even permutation
        assert _perm_sign((1, 2, 0)) == 1

    def test_transposition_3(self):
        # (0 1) applied to 3 elements
        assert _perm_sign((1, 0, 2)) == -1

    def test_identity_4(self):
        assert _perm_sign((0, 1, 2, 3)) == 1

    def test_double_transposition(self):
        # Two transpositions = even
        assert _perm_sign((1, 0, 3, 2)) == 1


# ---------------------------------------------------------------------------
# Determinant
# ---------------------------------------------------------------------------

class TestDeterminant:
    """Tests for exact integer determinant computation."""

    def test_empty(self):
        assert _determinant([]) == 0

    def test_1x1(self):
        assert _determinant([[5]]) == 5

    def test_2x2(self):
        assert _determinant([[1, 2], [3, 4]]) == -2

    def test_2x2_identity(self):
        assert _determinant([[1, 0], [0, 1]]) == 1

    def test_3x3(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert _determinant_safe(M) == 0  # Singular

    def test_3x3_nonsingular(self):
        M = [[1, 0, 2], [0, 1, 0], [3, 0, 1]]
        assert _determinant_safe(M) == -5

    def test_leibniz_agrees_with_bareiss(self):
        M = [[2, 1, 3], [0, 4, 1], [1, 0, 2]]
        assert _determinant(M) == _determinant_leibniz(M)

    def test_diagonal(self):
        M = [[2, 0, 0], [0, 3, 0], [0, 0, 5]]
        assert _determinant_safe(M) == 30

    def test_upper_triangular(self):
        M = [[1, 2, 3], [0, 4, 5], [0, 0, 6]]
        assert _determinant_safe(M) == 24

    def test_zero_matrix(self):
        M = [[0, 0], [0, 0]]
        assert _determinant(M) == 0


# ---------------------------------------------------------------------------
# Path counting
# ---------------------------------------------------------------------------

class TestPathCounting:
    """Tests for directed path counting."""

    def test_end_type_self_path(self):
        """end: single state, path from it to itself = 1."""
        ss = _build("end")
        assert _count_paths(ss, ss.top, ss.bottom) == 1

    def test_single_branch_one_path(self):
        """&{a: end}: one path from top to bottom."""
        ss = _build("&{a: end}")
        assert _count_paths(ss, ss.top, ss.bottom) == 1

    def test_two_branch_one_path(self):
        """&{a: end, b: end}: two edges but one path in quotient DAG (2 states)."""
        ss = _build("&{a: end, b: end}")
        # Only 2 states: top and bottom — one path in the DAG
        assert _count_paths(ss, ss.top, ss.bottom) == 1

    def test_diamond_paths(self):
        """&{a: &{c: end}, b: &{c: end}}: diamond shape."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        count = _count_paths(ss, ss.top, ss.bottom)
        assert count >= 2  # At least paths through a→c and b→c

    def test_chain_path(self):
        """&{a: &{b: end}}: single chain, one path."""
        ss = _build("&{a: &{b: end}}")
        assert _count_paths(ss, ss.top, ss.bottom) == 1

    def test_three_branches_one_path(self):
        """&{a: end, b: end, c: end}: three edges but one DAG path (2 states)."""
        ss = _build("&{a: end, b: end, c: end}")
        assert _count_paths(ss, ss.top, ss.bottom) == 1

    def test_unreachable_returns_zero(self):
        """Path count from bottom to top is 0 (DAG goes top→bottom)."""
        ss = _build("&{a: end}")
        assert _count_paths(ss, ss.bottom, ss.top) == 0

    def test_recursive_type_path_count(self):
        """rec X . &{a: X, b: end}: quotient yields a DAG."""
        ss = _build("rec X . &{a: X, b: end}")
        # After SCC quotient, there should be at least one path top→bottom
        count = _count_paths(ss, ss.top, ss.bottom)
        assert count >= 1


# ---------------------------------------------------------------------------
# Path count matrix
# ---------------------------------------------------------------------------

class TestPathCountMatrix:
    """Tests for path count matrix construction."""

    def test_1x1_end(self):
        """end: 1×1 matrix [[1]]."""
        ss = _build("end")
        M = path_count_matrix(ss, [ss.top], [ss.bottom])
        assert M == [[1]]

    def test_1x1_branch(self):
        """&{a: end}: 1×1 matrix [[1]]."""
        ss = _build("&{a: end}")
        M = path_count_matrix(ss, [ss.top], [ss.bottom])
        assert M == [[1]]

    def test_dimensions(self):
        """Matrix has correct dimensions."""
        ss = _build("&{a: end, b: end}")
        # 1 source (top), 1 sink (bottom)
        M = path_count_matrix(ss, [ss.top], [ss.bottom])
        assert len(M) == 1
        assert len(M[0]) == 1

    def test_matrix_values_positive(self):
        """All entries in path matrix are non-negative."""
        ss = _build("&{a: &{c: end}, b: &{d: end}}")
        M = path_count_matrix(ss, [ss.top], [ss.bottom])
        for row in M:
            for val in row:
                assert val >= 0


# ---------------------------------------------------------------------------
# Rank layers
# ---------------------------------------------------------------------------

class TestRankLayers:
    """Tests for rank layer detection."""

    def test_end_single_layer(self):
        """end: one layer with one state."""
        ss = _build("end")
        layers = find_rank_layers(ss)
        assert len(layers) == 1
        assert 0 in layers

    def test_branch_two_layers(self):
        """&{a: end}: two layers — rank 1 (top) and rank 0 (bottom)."""
        ss = _build("&{a: end}")
        layers = find_rank_layers(ss)
        assert len(layers) == 2
        max_rank = max(layers.keys())
        assert ss.top in layers[max_rank]
        assert ss.bottom in layers[0]

    def test_chain_three_layers(self):
        """&{a: &{b: end}}: three layers."""
        ss = _build("&{a: &{b: end}}")
        layers = find_rank_layers(ss)
        assert len(layers) == 3

    def test_parallel_layers(self):
        """(end || end): product creates layers."""
        ss = _build("(&{a: end} || &{b: end})")
        layers = find_rank_layers(ss)
        assert len(layers) >= 2

    def test_layers_sorted(self):
        """States in each layer are sorted."""
        ss = _build("&{a: end, b: end, c: end}")
        layers = find_rank_layers(ss)
        for rank, states in layers.items():
            assert states == sorted(states)


# ---------------------------------------------------------------------------
# LGV verification
# ---------------------------------------------------------------------------

class TestLGVVerification:
    """Tests verifying the LGV Lemma identity."""

    def test_end_lgv(self):
        """end: trivial 1×1 case, det = 1 = non-intersecting."""
        ss = _build("end")
        assert verify_lgv(ss, [ss.top], [ss.bottom])

    def test_single_branch_lgv(self):
        """&{a: end}: det([1]) = 1."""
        ss = _build("&{a: end}")
        assert verify_lgv(ss, [ss.top], [ss.bottom])

    def test_two_branch_lgv(self):
        """&{a: end, b: end}: 1×1 matrix, det = path count."""
        ss = _build("&{a: end, b: end}")
        assert verify_lgv(ss, [ss.top], [ss.bottom])

    def test_chain_lgv(self):
        """&{a: &{b: end}}: chain, det = 1."""
        ss = _build("&{a: &{b: end}}")
        assert verify_lgv(ss, [ss.top], [ss.bottom])

    def test_mismatched_lengths_raises(self):
        """Mismatched source/sink lengths raise ValueError."""
        ss = _build("&{a: end}")
        with pytest.raises(ValueError):
            verify_lgv(ss, [ss.top], [])

    def test_lgv_determinant_mismatched(self):
        """lgv_determinant also raises on mismatch."""
        ss = _build("&{a: end}")
        with pytest.raises(ValueError):
            lgv_determinant(ss, [ss.top, ss.bottom], [ss.bottom])


# ---------------------------------------------------------------------------
# Non-intersecting count
# ---------------------------------------------------------------------------

class TestNonIntersecting:
    """Tests for non-intersecting path system counting."""

    def test_single_path(self):
        """One source, one sink: non-intersecting = total paths."""
        ss = _build("&{a: end}")
        assert non_intersecting_count(ss, [ss.top], [ss.bottom]) == 1

    def test_empty(self):
        """No sources/sinks: 1 (empty system)."""
        ss = _build("end")
        assert non_intersecting_count(ss, [], []) == 1

    def test_two_branch_single_system(self):
        """&{a: end, b: end}: single source → single sink, one path in DAG."""
        ss = _build("&{a: end, b: end}")
        count = non_intersecting_count(ss, [ss.top], [ss.bottom])
        assert count == 1  # One path (top→bottom) in the quotient DAG


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeLGV:
    """Tests for the full analysis pipeline."""

    def test_end_analysis(self):
        """end: trivial analysis."""
        ss = _build("end")
        r = analyze_lgv(ss)
        assert isinstance(r, LGVResult)
        assert r.num_sources >= 1
        assert r.num_sinks >= 1
        assert r.lgv_verified

    def test_single_branch_analysis(self):
        """&{a: end}: standard analysis."""
        ss = _build("&{a: end}")
        r = analyze_lgv(ss)
        assert r.lgv_verified
        assert r.determinant >= 0

    def test_two_branch_analysis(self):
        """&{a: end, b: end}: analysis with branching."""
        ss = _build("&{a: end, b: end}")
        r = analyze_lgv(ss)
        assert r.lgv_verified

    def test_chain_analysis(self):
        """&{a: &{b: end}}: chain analysis."""
        ss = _build("&{a: &{b: end}}")
        r = analyze_lgv(ss)
        assert r.lgv_verified
        assert r.determinant == 1

    def test_selection_analysis(self):
        """+{a: end, b: end}: selection type."""
        ss = _build("+{a: end, b: end}")
        r = analyze_lgv(ss)
        assert r.lgv_verified

    def test_parallel_analysis(self):
        """(&{a: end} || &{b: end}): parallel composition."""
        ss = _build("(&{a: end} || &{b: end})")
        r = analyze_lgv(ss)
        assert r.lgv_verified

    def test_recursive_analysis(self):
        """rec X . &{a: X, b: end}: recursive type."""
        ss = _build("rec X . &{a: X, b: end}")
        r = analyze_lgv(ss)
        assert r.lgv_verified

    def test_result_fields(self):
        """Result has all expected fields."""
        ss = _build("&{a: end}")
        r = analyze_lgv(ss)
        assert hasattr(r, 'path_matrix')
        assert hasattr(r, 'determinant')
        assert hasattr(r, 'num_sources')
        assert hasattr(r, 'num_sinks')
        assert hasattr(r, 'sources')
        assert hasattr(r, 'sinks')
        assert hasattr(r, 'lgv_verified')
        assert hasattr(r, 'num_non_intersecting')

    def test_result_frozen(self):
        """LGVResult is frozen."""
        ss = _build("end")
        r = analyze_lgv(ss)
        with pytest.raises(AttributeError):
            r.determinant = 42  # type: ignore[misc]

    def test_path_matrix_dimensions(self):
        """Path matrix is square with dimension = min(sources, sinks)."""
        ss = _build("&{a: end}")
        r = analyze_lgv(ss)
        n = r.num_sources
        assert len(r.path_matrix) == n
        for row in r.path_matrix:
            assert len(row) == r.num_sinks


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """LGV analysis on benchmark-like protocols."""

    def test_iterator_lgv(self):
        """Java Iterator: rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        r = analyze_lgv(ss)
        assert r.lgv_verified

    def test_file_object_lgv(self):
        """File Object protocol."""
        ss = _build("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")
        r = analyze_lgv(ss)
        assert r.lgv_verified

    def test_two_branch_protocol_lgv(self):
        """Two-branch protocol."""
        ss = _build("&{a: &{c: end}, b: &{d: end}}")
        r = analyze_lgv(ss)
        assert r.lgv_verified

    def test_nested_selection_lgv(self):
        """Nested selection/branch."""
        ss = _build("&{open: +{OK: &{use: end}, ERR: end}}")
        r = analyze_lgv(ss)
        assert r.lgv_verified

    def test_deep_chain_lgv(self):
        """Deep chain: a.b.c.d — single path."""
        ss = _build("&{a: &{b: &{c: &{d: end}}}}")
        r = analyze_lgv(ss)
        assert r.lgv_verified
        assert r.determinant == 1

    def test_wide_branch_lgv(self):
        """Wide branch: many alternatives."""
        ss = _build("&{a: end, b: end, c: end, d: end, e: end}")
        r = analyze_lgv(ss)
        assert r.lgv_verified

    def test_parallel_deep_lgv(self):
        """Parallel with deeper branches."""
        ss = _build("(&{a: &{b: end}} || &{c: end})")
        r = analyze_lgv(ss)
        assert r.lgv_verified
