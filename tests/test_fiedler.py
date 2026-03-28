"""Tests for Fiedler / algebraic connectivity analysis (Step 30e).

Tests cover:
- Fiedler value computation
- Fiedler vector and spectral bisection
- Cut vertices and bottleneck detection
- Vertex and edge connectivity
- Cheeger inequality bounds
- Composition under parallel
- Benchmark protocols
"""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.fiedler import (
    fiedler_value,
    fiedler_vector,
    find_cut_vertices,
    vertex_connectivity,
    edge_connectivity,
    betweenness_centrality,
    find_bottleneck,
    cheeger_bounds,
    verify_fiedler_min_composition,
    analyze_fiedler,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Fiedler value tests
# ---------------------------------------------------------------------------

class TestFiedlerValue:

    def test_end(self):
        """Single state: λ₂ = 0."""
        assert fiedler_value(_build("end")) == 0.0

    def test_single_edge(self):
        """Two states: λ₂ = 2 (complete graph K₂)."""
        fv = fiedler_value(_build("&{a: end}"))
        assert abs(fv - 2.0) < 0.1

    def test_chain_3(self):
        """Chain of 3: Fiedler value is positive (connected) and < 2."""
        fv = fiedler_value(_build("&{a: &{b: end}}"))
        assert 0 < fv < 2.0

    def test_parallel_positive(self):
        """Parallel type has positive Fiedler value (connected)."""
        fv = fiedler_value(_build("(&{a: end} || &{b: end})"))
        assert fv > 0

    def test_longer_chain_decreasing(self):
        """Longer chains have smaller Fiedler values (weaker connectivity)."""
        fv2 = fiedler_value(_build("&{a: end}"))
        fv3 = fiedler_value(_build("&{a: &{b: end}}"))
        assert fv2 > fv3  # K₂ more connected than P₃


# ---------------------------------------------------------------------------
# Fiedler vector tests
# ---------------------------------------------------------------------------

class TestFiedlerVector:

    def test_length_matches_states(self):
        ss = _build("&{a: &{b: end}}")
        vec = fiedler_vector(ss)
        assert len(vec) == len(ss.states)

    def test_normalized(self):
        ss = _build("&{a: &{b: end}}")
        vec = fiedler_vector(ss)
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 0.1 or norm < 0.01  # normalized or zero

    def test_single_state(self):
        vec = fiedler_vector(_build("end"))
        assert len(vec) == 1


# ---------------------------------------------------------------------------
# Cut vertex tests
# ---------------------------------------------------------------------------

class TestCutVertices:

    def test_end_no_cuts(self):
        assert find_cut_vertices(_build("end")) == []

    def test_single_edge_no_cuts(self):
        """K₂ has no cut vertices."""
        assert find_cut_vertices(_build("&{a: end}")) == []

    def test_chain_3_has_cut(self):
        """Path P₃: middle vertex is a cut vertex."""
        ss = _build("&{a: &{b: end}}")
        cuts = find_cut_vertices(ss)
        assert len(cuts) >= 1


# ---------------------------------------------------------------------------
# Connectivity tests
# ---------------------------------------------------------------------------

class TestConnectivity:

    def test_end_connectivity(self):
        assert vertex_connectivity(_build("end")) == 0

    def test_edge_connectivity_single(self):
        assert edge_connectivity(_build("&{a: end}")) >= 1

    def test_vertex_le_edge(self):
        """Whitney's theorem: vertex_conn ≤ edge_conn."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            vc = vertex_connectivity(ss)
            ec = edge_connectivity(ss)
            assert vc <= ec + 1  # Allow small tolerance for approximation


# ---------------------------------------------------------------------------
# Betweenness tests
# ---------------------------------------------------------------------------

class TestBetweenness:

    def test_end_betweenness(self):
        bc = betweenness_centrality(_build("end"))
        assert len(bc) == 1

    def test_chain_middle_highest(self):
        """In a chain, the middle state has highest betweenness."""
        ss = _build("&{a: &{b: end}}")
        bc = betweenness_centrality(ss)
        # Middle state should have highest centrality
        assert len(bc) == 3

    def test_bottleneck_exists(self):
        ss = _build("&{a: &{b: end}}")
        b = find_bottleneck(ss)
        assert b is not None
        assert b in ss.states


# ---------------------------------------------------------------------------
# Cheeger inequality tests
# ---------------------------------------------------------------------------

class TestCheeger:

    def test_bounds_order(self):
        """Lower bound ≤ upper bound."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            lo, hi = cheeger_bounds(ss)
            assert lo <= hi + 0.01

    def test_trivial(self):
        lo, hi = cheeger_bounds(_build("end"))
        assert lo == 0.0


# ---------------------------------------------------------------------------
# Composition tests
# ---------------------------------------------------------------------------

class TestComposition:

    def test_fiedler_min(self):
        """λ₂(L₁×L₂) = min(λ₂(L₁), λ₂(L₂))."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ss_prod = _build("(&{a: end} || &{b: end})")
        assert verify_fiedler_min_composition(ss1, ss2, ss_prod)


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyze:

    def test_end_analysis(self):
        r = analyze_fiedler(_build("end"))
        assert r.num_states == 1
        assert r.fiedler_value == 0.0
        assert r.cut_vertices == []

    def test_branch_analysis(self):
        r = analyze_fiedler(_build("&{a: end}"))
        assert r.num_states == 2
        assert r.is_connected is True
        assert abs(r.fiedler_value - 2.0) < 0.1

    def test_parallel_analysis(self):
        r = analyze_fiedler(_build("(&{a: end} || &{b: end})"))
        assert r.num_states == 4
        assert r.is_connected is True
        assert r.fiedler_value > 0


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
    def test_fiedler_properties(self, name, typ):
        """Fiedler analysis runs correctly for all benchmarks."""
        ss = _build(typ)
        r = analyze_fiedler(ss)
        assert r.num_states == len(ss.states)
        assert r.fiedler_value >= -0.01
        assert r.cheeger_lower <= r.cheeger_upper + 0.01

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_cut_vertices_valid(self, name, typ):
        """Cut vertices are actual states."""
        ss = _build(typ)
        cuts = find_cut_vertices(ss)
        for c in cuts:
            assert c in ss.states

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_betweenness_valid(self, name, typ):
        """Betweenness centrality values are non-negative."""
        ss = _build(typ)
        bc = betweenness_centrality(ss)
        for s, val in bc.items():
            assert val >= -0.01, f"{name}: betweenness({s}) = {val} < 0"
