"""Tests for heat kernel analysis (Step 30q).

Tests cover:
- Graph Laplacian: symmetry, row sums zero, PSD
- Eigendecomposition: sorted eigenvalues, orthogonal eigenvectors
- Heat kernel matrix: H_0 = I, symmetry, positive entries
- Heat trace: Z(0) = n, monotone decreasing
- Heat kernel signature: decreasing in t, local descriptor
- Diffusion distance: metric properties (symmetry, triangle, d(x,x)=0)
- Total heat content: Q(0) = n
- Heat kernel PageRank: non-negative, sums to 1
- Return probability, spectral gap, effective resistance, Kemeny constant
- Full analysis on session type state spaces
- Benchmark protocols
"""

import math

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.heat_kernel import (
    graph_laplacian,
    laplacian_eigendecomposition,
    heat_kernel_matrix,
    heat_trace,
    heat_kernel_signature,
    diffusion_distance,
    total_heat_content,
    heat_kernel_pagerank,
    return_probability,
    spectral_gap,
    effective_resistance,
    kemeny_constant,
    analyze_heat_kernel,
    HeatKernelResult,
    DEFAULT_SAMPLE_TIMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str):
    """Parse and build state space."""
    return build_statespace(parse(s))


_TOL = 1e-6


def _approx(a: float, b: float, tol: float = _TOL) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Test Laplacian
# ---------------------------------------------------------------------------

class TestLaplacian:
    """Tests for graph Laplacian construction."""

    def test_single_state(self):
        ss = _build("end")
        L = graph_laplacian(ss)
        assert len(L) == 1
        assert L[0][0] == 0.0

    def test_two_state_chain(self):
        ss = _build("&{a: end}")
        L = graph_laplacian(ss)
        n = len(L)
        assert n >= 2
        # Row sums must be zero
        for i in range(n):
            assert _approx(sum(L[i]), 0.0)

    def test_symmetric(self):
        ss = _build("&{a: end, b: end}")
        L = graph_laplacian(ss)
        n = len(L)
        for i in range(n):
            for j in range(n):
                assert _approx(L[i][j], L[j][i]), f"L[{i}][{j}] != L[{j}][{i}]"

    def test_row_sums_zero(self):
        ss = _build("&{a: &{b: end}, c: end}")
        L = graph_laplacian(ss)
        for i in range(len(L)):
            assert _approx(sum(L[i]), 0.0)

    def test_diagonal_non_negative(self):
        ss = _build("&{a: end, b: &{c: end}}")
        L = graph_laplacian(ss)
        for i in range(len(L)):
            assert L[i][i] >= -_TOL

    def test_off_diagonal_non_positive(self):
        ss = _build("&{a: end, b: end}")
        L = graph_laplacian(ss)
        n = len(L)
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert L[i][j] <= _TOL

    def test_psd_eigenvalues(self):
        """Laplacian eigenvalues must be non-negative (PSD)."""
        ss = _build("&{a: &{b: end}, c: end}")
        eigenvalues, _ = laplacian_eigendecomposition(ss)
        for lam in eigenvalues:
            assert lam >= -_TOL

    def test_smallest_eigenvalue_zero(self):
        """Connected graph has exactly one zero eigenvalue."""
        ss = _build("&{a: end, b: end}")
        eigenvalues, _ = laplacian_eigendecomposition(ss)
        assert _approx(eigenvalues[0], 0.0)

    def test_parallel_laplacian(self):
        ss = _build("(&{a: end} || &{b: end})")
        L = graph_laplacian(ss)
        n = len(L)
        for i in range(n):
            assert _approx(sum(L[i]), 0.0)


# ---------------------------------------------------------------------------
# Test Eigendecomposition
# ---------------------------------------------------------------------------

class TestEigendecomposition:
    """Tests for Laplacian eigendecomposition."""

    def test_eigenvalues_sorted(self):
        ss = _build("&{a: end, b: end}")
        eigenvalues, _ = laplacian_eigendecomposition(ss)
        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] <= eigenvalues[i + 1] + _TOL

    def test_eigenvectors_orthogonal(self):
        ss = _build("&{a: &{b: end}, c: end}")
        eigenvalues, eigenvectors = laplacian_eigendecomposition(ss)
        n = len(eigenvalues)
        for i in range(n):
            for j in range(n):
                dot = sum(eigenvectors[i][k] * eigenvectors[j][k] for k in range(n))
                expected = 1.0 if i == j else 0.0
                assert _approx(dot, expected, tol=1e-4), \
                    f"dot(v{i}, v{j}) = {dot}, expected {expected}"

    def test_eigenvectors_normalized(self):
        ss = _build("&{a: end, b: &{c: end}}")
        _, eigenvectors = laplacian_eigendecomposition(ss)
        for vec in eigenvectors:
            norm = math.sqrt(sum(x * x for x in vec))
            assert _approx(norm, 1.0, tol=1e-4)

    def test_single_state_eigendecomposition(self):
        ss = _build("end")
        eigenvalues, eigenvectors = laplacian_eigendecomposition(ss)
        assert eigenvalues == [0.0]
        assert eigenvectors == [[1.0]]

    def test_eigenvalue_count_matches_states(self):
        ss = _build("&{a: end, b: end, c: end}")
        L = graph_laplacian(ss)
        eigenvalues, eigenvectors = laplacian_eigendecomposition(ss)
        assert len(eigenvalues) == len(L)
        assert len(eigenvectors) == len(L)


# ---------------------------------------------------------------------------
# Test Heat Kernel Matrix
# ---------------------------------------------------------------------------

class TestHeatKernel:
    """Tests for heat kernel matrix H_t = exp(-tL)."""

    def test_h0_is_identity(self):
        """H_0 = exp(0) = I."""
        ss = _build("&{a: end, b: end}")
        H = heat_kernel_matrix(ss, 0.0)
        n = len(H)
        for i in range(n):
            for j in range(n):
                expected = 1.0 if i == j else 0.0
                assert _approx(H[i][j], expected, tol=1e-4), \
                    f"H_0[{i}][{j}] = {H[i][j]}, expected {expected}"

    def test_symmetric(self):
        ss = _build("&{a: &{b: end}, c: end}")
        H = heat_kernel_matrix(ss, 1.0)
        n = len(H)
        for i in range(n):
            for j in range(n):
                assert _approx(H[i][j], H[j][i], tol=1e-4)

    def test_positive_entries(self):
        """H_t has non-negative entries for t >= 0."""
        ss = _build("&{a: end, b: end}")
        H = heat_kernel_matrix(ss, 1.0)
        for i in range(len(H)):
            for j in range(len(H)):
                assert H[i][j] >= -_TOL

    def test_diagonal_decreasing_with_t(self):
        """H_t(x,x) is non-increasing in t for each x."""
        ss = _build("&{a: end, b: end}")
        n = len(graph_laplacian(ss))
        times = [0.0, 0.1, 1.0, 10.0]
        for i in range(n):
            prev = None
            for t in times:
                H = heat_kernel_matrix(ss, t)
                val = H[i][i]
                if prev is not None:
                    assert val <= prev + _TOL
                prev = val

    def test_single_state_heat_kernel(self):
        ss = _build("end")
        H = heat_kernel_matrix(ss, 5.0)
        assert len(H) == 1
        assert _approx(H[0][0], 1.0)

    def test_row_sums_constant(self):
        """Row sums of H_t are 1 for all t (stochastic if L is symmetric)."""
        ss = _build("&{a: end, b: end}")
        H = heat_kernel_matrix(ss, 1.0)
        n = len(H)
        # For graph Laplacian, each row sum of exp(-tL) = 1
        # because L*1 = 0, so exp(-tL)*1 = 1
        for i in range(n):
            row_sum = sum(H[i][j] for j in range(n))
            assert _approx(row_sum, 1.0, tol=1e-3), f"row {i} sum = {row_sum}"


# ---------------------------------------------------------------------------
# Test Heat Trace
# ---------------------------------------------------------------------------

class TestHeatTrace:
    """Tests for heat trace Z(t) = tr(H_t)."""

    def test_z0_equals_n(self):
        """Z(0) = n (number of quotient states)."""
        ss = _build("&{a: end, b: end}")
        L = graph_laplacian(ss)
        n = len(L)
        z = heat_trace(ss, 0.0)
        assert _approx(z, float(n))

    def test_trace_decreasing(self):
        """Z(t) is non-increasing in t for t >= 0."""
        ss = _build("&{a: end, b: end}")
        times = [0.0, 0.1, 1.0, 5.0, 10.0]
        prev = None
        for t in times:
            z = heat_trace(ss, t)
            if prev is not None:
                assert z <= prev + _TOL
            prev = z

    def test_trace_positive(self):
        """Z(t) > 0 for all t."""
        ss = _build("&{a: &{b: end}, c: end}")
        for t in [0.01, 0.1, 1.0, 10.0]:
            assert heat_trace(ss, t) > 0.0

    def test_single_state_trace(self):
        ss = _build("end")
        assert _approx(heat_trace(ss, 0.0), 1.0)
        assert _approx(heat_trace(ss, 100.0), 1.0)

    def test_long_time_limit(self):
        """Z(t) -> multiplicity of 0 eigenvalue as t -> inf.
        For connected graph, this is 1."""
        ss = _build("&{a: end, b: end}")
        z = heat_trace(ss, 100.0)
        assert _approx(z, 1.0, tol=0.01)


# ---------------------------------------------------------------------------
# Test HKS (Heat Kernel Signature)
# ---------------------------------------------------------------------------

class TestHKS:
    """Tests for heat kernel signature."""

    def test_hks_decreasing_in_t(self):
        """HKS(x, t) is non-increasing in t for each x."""
        ss = _build("&{a: end, b: end}")
        times = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        for state in ss.states:
            vals = heat_kernel_signature(ss, state, times)
            for i in range(len(vals) - 1):
                assert vals[i] >= vals[i + 1] - _TOL

    def test_hks_positive(self):
        ss = _build("&{a: end, b: end}")
        times = [0.1, 1.0, 5.0]
        for state in ss.states:
            vals = heat_kernel_signature(ss, state, times)
            for v in vals:
                assert v >= -_TOL

    def test_hks_at_zero(self):
        """HKS(x, 0) = 1 for all x (since H_0 = I)."""
        ss = _build("&{a: end, b: end}")
        for state in ss.states:
            vals = heat_kernel_signature(ss, state, [0.0])
            assert _approx(vals[0], 1.0, tol=1e-4)

    def test_hks_single_state(self):
        ss = _build("end")
        vals = heat_kernel_signature(ss, ss.top, [0.0, 1.0, 10.0])
        for v in vals:
            assert _approx(v, 1.0)


# ---------------------------------------------------------------------------
# Test Diffusion Distance
# ---------------------------------------------------------------------------

class TestDiffusionDistance:
    """Tests for diffusion distance."""

    def test_self_distance_zero(self):
        ss = _build("&{a: end, b: end}")
        D = diffusion_distance(ss, 1.0)
        for i in range(len(D)):
            assert _approx(D[i][i], 0.0)

    def test_symmetric(self):
        ss = _build("&{a: &{b: end}, c: end}")
        D = diffusion_distance(ss, 1.0)
        n = len(D)
        for i in range(n):
            for j in range(n):
                assert _approx(D[i][j], D[j][i], tol=1e-4)

    def test_non_negative(self):
        ss = _build("&{a: end, b: end}")
        D = diffusion_distance(ss, 1.0)
        for i in range(len(D)):
            for j in range(len(D)):
                assert D[i][j] >= -_TOL

    def test_triangle_inequality(self):
        ss = _build("&{a: &{b: end}, c: end}")
        D = diffusion_distance(ss, 1.0)
        n = len(D)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert D[i][k] <= D[i][j] + D[j][k] + _TOL

    def test_single_state(self):
        ss = _build("end")
        D = diffusion_distance(ss, 1.0)
        assert D == [[0.0]]

    def test_distinct_states_positive_distance(self):
        """Non-identical states should have positive diffusion distance."""
        ss = _build("&{a: end, b: end}")
        D = diffusion_distance(ss, 1.0)
        n = len(D)
        if n > 1:
            assert D[0][1] > 0.0


# ---------------------------------------------------------------------------
# Test Total Heat Content
# ---------------------------------------------------------------------------

class TestTotalHeatContent:
    """Tests for total heat content Q(t)."""

    def test_q0_equals_n(self):
        """Q(0) = n since H_0 = I and sum of identity = n."""
        ss = _build("&{a: end, b: end}")
        L = graph_laplacian(ss)
        n = len(L)
        q = total_heat_content(ss, 0.0)
        assert _approx(q, float(n))

    def test_content_positive(self):
        ss = _build("&{a: end, b: end}")
        for t in [0.01, 0.1, 1.0, 10.0]:
            assert total_heat_content(ss, t) > 0.0

    def test_single_state(self):
        ss = _build("end")
        assert _approx(total_heat_content(ss, 0.0), 1.0)
        assert _approx(total_heat_content(ss, 100.0), 1.0)

    def test_long_time_connected(self):
        """For a connected graph, Q(t) -> n as t -> inf.
        (Heat spreads uniformly, each entry -> 1/n, sum = n.)"""
        ss = _build("&{a: end, b: end}")
        L = graph_laplacian(ss)
        n = len(L)
        q = total_heat_content(ss, 100.0)
        assert _approx(q, float(n), tol=0.1)


# ---------------------------------------------------------------------------
# Test Heat Kernel PageRank
# ---------------------------------------------------------------------------

class TestPageRank:
    """Tests for heat kernel PageRank."""

    def test_non_negative(self):
        ss = _build("&{a: end, b: end}")
        pr = heat_kernel_pagerank(ss, 1.0)
        for v in pr:
            assert v >= -_TOL

    def test_sums_to_one(self):
        """PageRank vector sums to 1."""
        ss = _build("&{a: end, b: end}")
        pr = heat_kernel_pagerank(ss, 1.0)
        assert _approx(sum(pr), 1.0, tol=1e-4)

    def test_uniform_at_t0(self):
        """At t=0, H_0 = I so PageRank = uniform."""
        ss = _build("&{a: end, b: end}")
        L = graph_laplacian(ss)
        n = len(L)
        pr = heat_kernel_pagerank(ss, 0.0)
        for v in pr:
            assert _approx(v, 1.0 / n, tol=1e-4)

    def test_single_state(self):
        ss = _build("end")
        pr = heat_kernel_pagerank(ss, 1.0)
        assert len(pr) == 1
        assert _approx(pr[0], 1.0)


# ---------------------------------------------------------------------------
# Test Return Probability
# ---------------------------------------------------------------------------

class TestReturnProbability:
    """Tests for return probability."""

    def test_at_t0(self):
        """Return probability at t=0 is 1 (walker is at start)."""
        ss = _build("&{a: end, b: end}")
        for state in ss.states:
            rp = return_probability(ss, state, 0.0)
            assert _approx(rp, 1.0, tol=1e-4)

    def test_decreasing_in_t(self):
        ss = _build("&{a: end, b: end}")
        times = [0.0, 0.1, 1.0, 10.0]
        for state in ss.states:
            prev = None
            for t in times:
                rp = return_probability(ss, state, t)
                if prev is not None:
                    assert rp <= prev + _TOL
                prev = rp

    def test_positive(self):
        ss = _build("&{a: end, b: end}")
        for state in ss.states:
            assert return_probability(ss, state, 1.0) > 0.0


# ---------------------------------------------------------------------------
# Test Spectral Gap
# ---------------------------------------------------------------------------

class TestSpectralGap:
    """Tests for spectral gap."""

    def test_positive_for_connected(self):
        ss = _build("&{a: end, b: end}")
        gap = spectral_gap(ss)
        assert gap > 0.0

    def test_single_state(self):
        ss = _build("end")
        gap = spectral_gap(ss)
        assert _approx(gap, 0.0)

    def test_gap_is_smallest_nonzero_eigenvalue(self):
        ss = _build("&{a: &{b: end}, c: end}")
        eigenvalues, _ = laplacian_eigendecomposition(ss)
        gap = spectral_gap(ss)
        nonzero = [lam for lam in eigenvalues if lam > 1e-10]
        if nonzero:
            assert _approx(gap, min(nonzero), tol=1e-4)


# ---------------------------------------------------------------------------
# Test Effective Resistance
# ---------------------------------------------------------------------------

class TestEffectiveResistance:
    """Tests for effective resistance."""

    def test_self_resistance_zero(self):
        ss = _build("&{a: end, b: end}")
        R = effective_resistance(ss)
        for i in range(len(R)):
            assert _approx(R[i][i], 0.0)

    def test_symmetric(self):
        ss = _build("&{a: end, b: end}")
        R = effective_resistance(ss)
        n = len(R)
        for i in range(n):
            for j in range(n):
                assert _approx(R[i][j], R[j][i], tol=1e-4)

    def test_non_negative(self):
        ss = _build("&{a: end, b: end}")
        R = effective_resistance(ss)
        for i in range(len(R)):
            for j in range(len(R)):
                assert R[i][j] >= -_TOL


# ---------------------------------------------------------------------------
# Test Kemeny Constant
# ---------------------------------------------------------------------------

class TestKemenyConstant:
    """Tests for Kemeny constant."""

    def test_positive_for_nontrivial(self):
        ss = _build("&{a: end, b: end}")
        k = kemeny_constant(ss)
        assert k > 0.0

    def test_single_state(self):
        ss = _build("end")
        k = kemeny_constant(ss)
        assert _approx(k, 0.0)

    def test_increases_with_size(self):
        """Larger graphs generally have larger Kemeny constant."""
        ss_small = _build("&{a: end}")
        ss_large = _build("&{a: &{b: end}, c: end}")
        k_small = kemeny_constant(ss_small)
        k_large = kemeny_constant(ss_large)
        assert k_large >= k_small - _TOL


# ---------------------------------------------------------------------------
# Test Full Analysis
# ---------------------------------------------------------------------------

class TestAnalyze:
    """Tests for full heat kernel analysis."""

    def test_result_type(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_heat_kernel(ss)
        assert isinstance(result, HeatKernelResult)

    def test_sample_times_default(self):
        ss = _build("&{a: end}")
        result = analyze_heat_kernel(ss)
        assert result.sample_times == DEFAULT_SAMPLE_TIMES

    def test_custom_sample_times(self):
        ss = _build("&{a: end}")
        times = [0.5, 2.0]
        result = analyze_heat_kernel(ss, sample_times=times)
        assert result.sample_times == times
        assert len(result.heat_trace) == 2

    def test_laplacian_in_result(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_heat_kernel(ss)
        n = len(result.laplacian)
        assert n > 0
        for i in range(n):
            assert _approx(sum(result.laplacian[i]), 0.0)

    def test_eigenvalues_in_result(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_heat_kernel(ss)
        assert len(result.eigenvalues) == len(result.laplacian)
        assert _approx(result.eigenvalues[0], 0.0)

    def test_hks_diagonal_shape(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_heat_kernel(ss)
        n = len(result.laplacian)
        assert len(result.hks_diagonal) == n
        for row in result.hks_diagonal:
            assert len(row) == len(result.sample_times)

    def test_diffusion_distance_in_result(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_heat_kernel(ss)
        n = len(result.laplacian)
        assert len(result.diffusion_distance) == n
        for row in result.diffusion_distance:
            assert len(row) == n

    def test_empty_state_space(self):
        """Edge case: single state space."""
        ss = _build("end")
        result = analyze_heat_kernel(ss)
        assert len(result.laplacian) == 1
        assert result.eigenvalues == [0.0]

    def test_parallel_analysis(self):
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_heat_kernel(ss)
        assert len(result.laplacian) > 0
        assert all(lam >= -_TOL for lam in result.eigenvalues)

    def test_selection_analysis(self):
        ss = _build("+{a: end, b: end}")
        result = analyze_heat_kernel(ss)
        assert len(result.laplacian) > 0

    def test_deeper_tree(self):
        ss = _build("&{a: &{b: end, c: end}, d: end}")
        result = analyze_heat_kernel(ss)
        n = len(result.laplacian)
        # Quotient may merge end states; at least 3 quotient nodes
        assert n >= 3
        # Verify all eigenvalues non-negative
        for lam in result.eigenvalues:
            assert lam >= -_TOL


# ---------------------------------------------------------------------------
# Test Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on various session type protocols."""

    def test_smtp_like(self):
        """SMTP-like: init -> branch with multiple outcomes."""
        ss = _build("&{ehlo: &{mail: &{rcpt: &{data: end}}, quit: end}}")
        result = analyze_heat_kernel(ss)
        assert result.eigenvalues[0] < _TOL
        assert all(lam >= -_TOL for lam in result.eigenvalues)

    def test_iterator(self):
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_heat_kernel(ss)
        assert len(result.laplacian) > 0

    def test_binary_choice(self):
        ss = _build("+{ok: end, error: end}")
        result = analyze_heat_kernel(ss)
        n = len(result.laplacian)
        assert n > 0
        # Trace at t=0 should equal n
        assert _approx(result.heat_trace[0], float(n), tol=0.1)

    def test_nested_branch(self):
        ss = _build("&{a: &{b: end}, c: &{d: end}}")
        result = analyze_heat_kernel(ss)
        assert len(result.eigenvalues) > 0
        gap = spectral_gap(ss)
        assert gap > 0.0

    def test_parallel_protocol(self):
        ss = _build("(&{read: end} || &{write: end})")
        result = analyze_heat_kernel(ss)
        # Product lattice should give 4 quotient states
        n = len(result.laplacian)
        assert n > 0
        # Diffusion distance should be a metric
        D = result.diffusion_distance
        for i in range(n):
            assert _approx(D[i][i], 0.0)

    def test_recursive_protocol(self):
        ss = _build("rec X . &{step: X, done: end}")
        result = analyze_heat_kernel(ss)
        assert len(result.laplacian) > 0

    def test_heat_trace_consistency(self):
        """Heat trace from analyze should match direct computation."""
        ss = _build("&{a: end, b: &{c: end}}")
        result = analyze_heat_kernel(ss)
        for i, t in enumerate(result.sample_times):
            direct = heat_trace(ss, t)
            assert _approx(result.heat_trace[i], direct, tol=1e-3)
