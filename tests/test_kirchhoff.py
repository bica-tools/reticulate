"""Tests for Kirchhoff's matrix-tree theorem (Step 30h).

Tests cover:
- Spanning tree count via cofactor and eigenvalue methods
- Known tree counts for specific graphs
- Normalized count and complexity ratio
- Tree detection
- Composition under parallel
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.kirchhoff import (
    spanning_tree_count,
    spanning_tree_count_cofactor,
    spanning_tree_count_eigenvalue,
    normalized_tree_count,
    complexity_ratio,
    is_tree,
    verify_tree_count_product,
    analyze_kirchhoff,
    _determinant,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Determinant tests
# ---------------------------------------------------------------------------

class TestDeterminant:

    def test_1x1(self):
        assert _determinant([[5.0]]) == 5.0

    def test_2x2(self):
        assert abs(_determinant([[1.0, 2.0], [3.0, 4.0]]) - (-2.0)) < 1e-10

    def test_identity(self):
        I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert abs(_determinant(I) - 1.0) < 1e-10

    def test_singular(self):
        M = [[1.0, 2.0], [2.0, 4.0]]
        assert abs(_determinant(M)) < 1e-10


# ---------------------------------------------------------------------------
# Spanning tree count tests
# ---------------------------------------------------------------------------

class TestSpanningTreeCount:

    def test_end(self):
        """Single state: τ = 1 (trivial tree)."""
        assert spanning_tree_count(_build("end")) == 1

    def test_single_edge(self):
        """K₂: τ = 1."""
        assert spanning_tree_count(_build("&{a: end}")) == 1

    def test_chain_3(self):
        """P₃ (path of 3): τ = 1 (it's already a tree)."""
        assert spanning_tree_count(_build("&{a: &{b: end}}")) == 1

    def test_chain_4(self):
        """P₄: τ = 1."""
        assert spanning_tree_count(_build("&{a: &{b: &{c: end}}}")) == 1

    def test_parallel_4cycle(self):
        """(&{a:end} || &{b:end}): Hasse is C₄ (4-cycle), τ = 4."""
        tau = spanning_tree_count(_build("(&{a: end} || &{b: end})"))
        assert tau == 4

    def test_cofactor_matches_eigenvalue(self):
        """Both methods should agree."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            tau_c = spanning_tree_count_cofactor(ss)
            tau_e = spanning_tree_count_eigenvalue(ss)
            assert abs(tau_c - tau_e) < 1.0, \
                f"Cofactor={tau_c} vs eigenvalue={tau_e} for {typ}"


# ---------------------------------------------------------------------------
# Tree detection tests
# ---------------------------------------------------------------------------

class TestIsTree:

    def test_chain_is_tree(self):
        """Chains (paths) are trees."""
        assert is_tree(_build("&{a: end}")) is True
        assert is_tree(_build("&{a: &{b: end}}")) is True

    def test_cycle_not_tree(self):
        """C₄ (from parallel) is not a tree."""
        assert is_tree(_build("(&{a: end} || &{b: end})")) is False


# ---------------------------------------------------------------------------
# Normalized count tests
# ---------------------------------------------------------------------------

class TestNormalized:

    def test_chain_normalized(self):
        """Chain of n: τ=1, Cayley = n^{n-2}, ratio = 1/n^{n-2}."""
        ss = _build("&{a: &{b: end}}")
        norm = normalized_tree_count(ss)
        assert 0 < norm <= 1.0

    def test_parallel_normalized(self):
        ss = _build("(&{a: end} || &{b: end})")
        norm = normalized_tree_count(ss)
        # C₄: τ=4, Cayley(4)=4²=16, ratio=4/16=0.25
        assert abs(norm - 0.25) < 0.1


# ---------------------------------------------------------------------------
# Complexity ratio tests
# ---------------------------------------------------------------------------

class TestComplexityRatio:

    def test_chain_ratio(self):
        """Chain of n: τ=1, |E|=n-1, ratio=1/(n-1)."""
        ss = _build("&{a: &{b: end}}")
        cr = complexity_ratio(ss)
        assert abs(cr - 0.5) < 0.01  # τ=1, |E|=2

    def test_parallel_ratio(self):
        """C₄: τ=4, |E|=4, ratio=1.0."""
        ss = _build("(&{a: end} || &{b: end})")
        cr = complexity_ratio(ss)
        assert abs(cr - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Composition tests
# ---------------------------------------------------------------------------

class TestComposition:

    def test_product_has_more_trees(self):
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ss_prod = _build("(&{a: end} || &{b: end})")
        assert verify_tree_count_product(ss1, ss2, ss_prod)


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyze:

    def test_end(self):
        r = analyze_kirchhoff(_build("end"))
        assert r.num_states == 1
        assert r.spanning_tree_count == 1
        assert r.is_tree is True

    def test_chain(self):
        r = analyze_kirchhoff(_build("&{a: end}"))
        assert r.spanning_tree_count == 1
        assert r.is_tree is True

    def test_parallel(self):
        r = analyze_kirchhoff(_build("(&{a: end} || &{b: end})"))
        assert r.spanning_tree_count == 4
        assert r.is_tree is False


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmarks:
    BENCHMARKS = [
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("Simple SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_kirchhoff_properties(self, name, typ):
        """Kirchhoff analysis runs correctly."""
        ss = _build(typ)
        r = analyze_kirchhoff(ss)
        assert r.spanning_tree_count >= 1
        assert r.normalized_count >= 0
        assert r.normalized_count <= 1.0 + 0.01

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_methods_agree(self, name, typ):
        """Cofactor and eigenvalue methods agree."""
        ss = _build(typ)
        tau_c = spanning_tree_count_cofactor(ss)
        tau_e = spanning_tree_count_eigenvalue(ss)
        assert abs(tau_c - tau_e) < 1.0, \
            f"{name}: cofactor={tau_c:.2f} vs eigenvalue={tau_e:.2f}"

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_tree_iff_no_cycles(self, name, typ):
        """Tree iff |E| = n-1."""
        ss = _build(typ)
        r = analyze_kirchhoff(ss)
        expected_tree = (r.num_edges == r.num_states - 1)
        assert r.is_tree == expected_tree, \
            f"{name}: is_tree={r.is_tree}, edges={r.num_edges}, states={r.num_states}"
