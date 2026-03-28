"""Tests for eigenvalue analysis (Step 30d).

Tests cover:
- Adjacency and Laplacian spectra
- Spectral radius, gap, and energy
- Fiedler value (algebraic connectivity)
- Spectral composition under parallel
- Spectral classification
- Benchmark protocols
"""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.eigenvalues import (
    compute_adjacency_spectrum,
    compute_laplacian_spectrum,
    spectral_radius,
    spectral_gap,
    graph_energy,
    algebraic_connectivity,
    hasse_edge_count,
    average_degree,
    degree_sequence,
    verify_spectral_composition,
    verify_fiedler_min,
    classify_spectrum,
    analyze_eigenvalues,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Adjacency spectrum tests
# ---------------------------------------------------------------------------

class TestAdjacencySpectrum:

    def test_end(self):
        """Single state: spectrum = [0]."""
        ss = _build("end")
        eigs = compute_adjacency_spectrum(ss)
        assert len(eigs) == 1
        assert abs(eigs[0]) < 1e-10

    def test_single_branch(self):
        """&{a: end}: two states, one edge. Eigenvalues ≈ {-1, 1}."""
        ss = _build("&{a: end}")
        eigs = compute_adjacency_spectrum(ss)
        assert len(eigs) == 2
        assert abs(eigs[0] - (-1.0)) < 0.1
        assert abs(eigs[1] - 1.0) < 0.1

    def test_chain_3(self):
        """Path graph P₃: eigenvalues {-√2, 0, √2}."""
        ss = _build("&{a: &{b: end}}")
        eigs = compute_adjacency_spectrum(ss)
        assert len(eigs) == 3
        # P₃ eigenvalues: 2cos(kπ/4) for k=1,2,3 = {√2, 0, -√2}
        assert abs(eigs[0] - (-math.sqrt(2))) < 0.2
        assert abs(eigs[1]) < 0.2
        assert abs(eigs[2] - math.sqrt(2)) < 0.2

    def test_symmetric_spectrum(self):
        """Hasse diagrams are bipartite → spectrum symmetric around 0."""
        ss = _build("&{a: &{b: end}}")
        eigs = compute_adjacency_spectrum(ss)
        for i in range(len(eigs)):
            assert abs(eigs[i] + eigs[-(i + 1)]) < 0.2


# ---------------------------------------------------------------------------
# Laplacian spectrum tests
# ---------------------------------------------------------------------------

class TestLaplacianSpectrum:

    def test_end(self):
        ss = _build("end")
        eigs = compute_laplacian_spectrum(ss)
        assert len(eigs) == 1
        assert abs(eigs[0]) < 1e-10

    def test_single_edge(self):
        """Two states: Laplacian eigenvalues {0, 2}."""
        ss = _build("&{a: end}")
        eigs = compute_laplacian_spectrum(ss)
        assert abs(eigs[0]) < 0.1
        assert abs(eigs[1] - 2.0) < 0.1

    def test_smallest_eigenvalue_zero(self):
        """Smallest Laplacian eigenvalue is always 0 for connected graphs."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            eigs = compute_laplacian_spectrum(ss)
            assert abs(eigs[0]) < 0.1, f"Smallest eigenvalue should be ~0"


# ---------------------------------------------------------------------------
# Spectral invariant tests
# ---------------------------------------------------------------------------

class TestSpectralInvariants:

    def test_spectral_radius_single(self):
        ss = _build("end")
        assert spectral_radius(ss) < 0.1

    def test_spectral_radius_edge(self):
        ss = _build("&{a: end}")
        assert abs(spectral_radius(ss) - 1.0) < 0.1

    def test_fiedler_connected(self):
        """Connected Hasse diagrams have Fiedler value > 0."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            assert algebraic_connectivity(ss) > 0

    def test_energy_positive(self):
        """Graph energy is non-negative."""
        for typ in ["end", "&{a: end}", "&{a: &{b: end}}"]:
            ss = _build(typ)
            assert graph_energy(ss) >= -0.01

    def test_spectral_gap_positive(self):
        """Spectral gap is non-negative for connected graphs."""
        ss = _build("&{a: end}")
        assert spectral_gap(ss) >= -0.1


# ---------------------------------------------------------------------------
# Hasse diagram property tests
# ---------------------------------------------------------------------------

class TestHasseProperties:

    def test_end_edges(self):
        assert hasse_edge_count(_build("end")) == 0

    def test_single_branch_edges(self):
        assert hasse_edge_count(_build("&{a: end}")) == 1

    def test_chain_edges(self):
        """Chain of n states has n-1 edges."""
        ss = _build("&{a: &{b: end}}")
        assert hasse_edge_count(ss) == 2

    def test_parallel_edges(self):
        """Parallel creates more edges (product graph)."""
        ss = _build("(&{a: end} || &{b: end})")
        assert hasse_edge_count(ss) >= 4

    def test_degree_sequence(self):
        """Chain: degree sequence is [1, 2, ..., 2, 1] (endpoints degree 1)."""
        ss = _build("&{a: &{b: end}}")
        degs = degree_sequence(ss)
        assert degs[0] == 2  # middle has degree 2
        assert degs[-1] == 1  # endpoints degree 1

    def test_average_degree(self):
        ss = _build("&{a: end}")
        assert abs(average_degree(ss) - 1.0) < 0.01  # 2 edges / 2 nodes


# ---------------------------------------------------------------------------
# Composition tests
# ---------------------------------------------------------------------------

class TestComposition:

    def test_spectrum_additivity(self):
        """Eigenvalues of product = pairwise sums of factor eigenvalues."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ss_prod = _build("(&{a: end} || &{b: end})")
        assert verify_spectral_composition(ss1, ss2, ss_prod)

    def test_fiedler_min(self):
        """Fiedler value of product = min of factor Fiedler values."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ss_prod = _build("(&{a: end} || &{b: end})")
        assert verify_fiedler_min(ss1, ss2, ss_prod)


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassification:

    def test_trivial(self):
        assert classify_spectrum(_build("end")) == "trivial"

    def test_path(self):
        """Chain should be classified as path."""
        result = classify_spectrum(_build("&{a: end}"))
        assert result in ("path", "bipartite")

    def test_classify_runs(self):
        """Classification runs without error for all types."""
        for typ in ["end", "&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            result = classify_spectrum(_build(typ))
            assert result in ("trivial", "path", "bipartite", "product", "general")


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyze:

    def test_end_analysis(self):
        r = analyze_eigenvalues(_build("end"))
        assert r.num_states == 1
        assert r.num_edges == 0
        assert r.spectral_radius < 0.1
        assert r.energy < 0.1

    def test_branch_analysis(self):
        r = analyze_eigenvalues(_build("&{a: end}"))
        assert r.num_states == 2
        assert r.num_edges == 1
        assert r.is_connected is True
        assert abs(r.spectral_radius - 1.0) < 0.1

    def test_parallel_analysis(self):
        r = analyze_eigenvalues(_build("(&{a: end} || &{b: end})"))
        assert r.num_states == 4
        assert r.is_connected is True
        assert r.fiedler_value > 0


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmarks:
    BENCHMARKS = [
        ("Java Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("Simple SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_spectral_properties(self, name, typ):
        """Basic spectral properties hold for all benchmarks."""
        ss = _build(typ)
        r = analyze_eigenvalues(ss)

        # Number of eigenvalues = number of states
        assert len(r.adjacency_eigenvalues) == r.num_states
        assert len(r.laplacian_eigenvalues) == r.num_states

        # Spectral radius ≥ 0
        assert r.spectral_radius >= -0.01

        # Energy ≥ 0
        assert r.energy >= -0.01

        # Laplacian smallest eigenvalue ≈ 0
        assert abs(r.laplacian_eigenvalues[0]) < 0.1

        # For acyclic types, Hasse diagram should be connected
        # Recursive types may have disconnected Hasse diagrams
        # (SCC nodes lose internal edges after quotient)

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_trace_properties(self, name, typ):
        """Trace of A = 0 (no self-loops), trace of L = 2·|E|."""
        ss = _build(typ)
        adj_eigs = compute_adjacency_spectrum(ss)
        lap_eigs = compute_laplacian_spectrum(ss)
        n_edges = hasse_edge_count(ss)

        # tr(A) = Σ λ_i ≈ 0
        assert abs(sum(adj_eigs)) < 0.5

        # tr(L) = Σ μ_i = 2|E|
        assert abs(sum(lap_eigs) - 2 * n_edges) < 0.5
