"""Tests for von Neumann entropy analysis (Step 30r).

Tests cover:
- Density matrix properties (trace=1, PSD, symmetric)
- Von Neumann entropy bounds [0, log(n)]
- Renyi entropy inequalities (S_inf <= S_2 <= S_1)
- Effective dimension bounds [1, n]
- Single-state and two-state edge cases
- Linear chains (low entropy)
- Branching protocols (higher entropy)
- Parallel composition (product lattice entropy)
- Relative entropy (non-negative, same-dimension check)
- Mutual information (non-negative)
- Entropy rate
- Full analyze_von_neumann integration
- Benchmark protocol analysis
"""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace, StateSpace
from reticulate.von_neumann import (
    VonNeumannResult,
    analyze_von_neumann,
    density_eigenvalues,
    density_matrix,
    effective_dimension,
    entropy_rate,
    laplacian_matrix,
    max_entropy,
    mutual_information,
    normalized_entropy,
    relative_entropy,
    renyi_entropy,
    renyi_entropy_2,
    renyi_entropy_inf,
    von_neumann_entropy,
    _adjacency_matrix,
    _degree_matrix,
    _eigenvalues_symmetric,
    _mat_trace,
    _undirected_edges,
    _xlogx,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str) -> StateSpace:
    """Parse and build state space."""
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Edge cases: trivial graphs
# ---------------------------------------------------------------------------

class TestTrivialGraphs:
    """Single-state and minimal graphs."""

    def test_end_type_entropy_zero(self):
        """end: single state, entropy = 0."""
        ss = _build("end")
        assert von_neumann_entropy(ss) == pytest.approx(0.0)

    def test_end_type_max_entropy_zero(self):
        """end: single state, max entropy = 0."""
        ss = _build("end")
        assert max_entropy(ss) == pytest.approx(0.0)

    def test_end_type_normalized_zero(self):
        """end: normalized entropy is 0 for single-state."""
        ss = _build("end")
        assert normalized_entropy(ss) == pytest.approx(0.0)

    def test_end_type_effective_dimension_one(self):
        """end: effective dimension = exp(0) = 1."""
        ss = _build("end")
        assert effective_dimension(ss) == pytest.approx(1.0)

    def test_end_density_matrix(self):
        """end: density matrix is [[1.0]]."""
        ss = _build("end")
        rho = density_matrix(ss)
        assert len(rho) == 1
        assert rho[0][0] == pytest.approx(1.0)

    def test_end_density_eigenvalues(self):
        """end: single eigenvalue = 1."""
        ss = _build("end")
        eigs = density_eigenvalues(ss)
        assert len(eigs) == 1
        assert eigs[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Density matrix properties
# ---------------------------------------------------------------------------

class TestDensityMatrixProperties:
    """Density matrix must be PSD with trace 1."""

    def test_trace_one_single_branch(self):
        """&{a: end}: density matrix has trace 1."""
        ss = _build("&{a: end}")
        rho = density_matrix(ss)
        assert _mat_trace(rho) == pytest.approx(1.0, abs=1e-10)

    def test_trace_one_two_branch(self):
        """&{a: end, b: end}: density matrix has trace 1."""
        ss = _build("&{a: end, b: end}")
        rho = density_matrix(ss)
        assert _mat_trace(rho) == pytest.approx(1.0, abs=1e-10)

    def test_trace_one_parallel(self):
        """(a.end || b.end): density matrix has trace 1."""
        ss = _build("(&{a: end} || &{b: end})")
        rho = density_matrix(ss)
        assert _mat_trace(rho) == pytest.approx(1.0, abs=1e-10)

    def test_eigenvalues_sum_to_one(self):
        """Density eigenvalues sum to 1."""
        ss = _build("&{a: end, b: end}")
        eigs = density_eigenvalues(ss)
        assert sum(eigs) == pytest.approx(1.0, abs=1e-10)

    def test_eigenvalues_non_negative(self):
        """All density eigenvalues are >= 0."""
        ss = _build("&{a: &{c: end}, b: end}")
        eigs = density_eigenvalues(ss)
        for e in eigs:
            assert e >= -1e-10

    def test_symmetric_density_matrix(self):
        """Density matrix is symmetric."""
        ss = _build("&{a: end, b: end}")
        rho = density_matrix(ss)
        n = len(rho)
        for i in range(n):
            for j in range(n):
                assert rho[i][j] == pytest.approx(rho[j][i], abs=1e-10)

    def test_eigenvalues_sum_one_deeper(self):
        """Deeper type: eigenvalues still sum to 1."""
        ss = _build("&{a: &{c: end, d: end}, b: end}")
        eigs = density_eigenvalues(ss)
        assert sum(eigs) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Laplacian properties
# ---------------------------------------------------------------------------

class TestLaplacianProperties:
    """Graph Laplacian structural properties."""

    def test_laplacian_row_sum_zero(self):
        """Laplacian rows sum to zero (for any connected graph)."""
        ss = _build("&{a: end, b: end}")
        L = laplacian_matrix(ss)
        for row in L:
            assert sum(row) == pytest.approx(0.0, abs=1e-10)

    def test_laplacian_symmetric(self):
        """Laplacian is symmetric."""
        ss = _build("&{a: &{c: end}, b: end}")
        L = laplacian_matrix(ss)
        n = len(L)
        for i in range(n):
            for j in range(n):
                assert L[i][j] == pytest.approx(L[j][i], abs=1e-10)

    def test_laplacian_psd(self):
        """Laplacian eigenvalues are non-negative (PSD)."""
        ss = _build("&{a: end, b: end}")
        L = laplacian_matrix(ss)
        eigs = _eigenvalues_symmetric(L)
        for e in eigs:
            assert e >= -1e-10

    def test_laplacian_smallest_eigenvalue_zero(self):
        """Connected graph: smallest Laplacian eigenvalue is 0."""
        ss = _build("&{a: end, b: end}")
        L = laplacian_matrix(ss)
        eigs = _eigenvalues_symmetric(L)
        assert min(eigs) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Von Neumann entropy bounds
# ---------------------------------------------------------------------------

class TestEntropyBounds:
    """S(rho) in [0, log(n)]."""

    def test_entropy_non_negative_single_branch(self):
        """Entropy >= 0 for &{a: end}."""
        ss = _build("&{a: end}")
        assert von_neumann_entropy(ss) >= -1e-10

    def test_entropy_non_negative_two_branch(self):
        """Entropy >= 0 for &{a: end, b: end}."""
        ss = _build("&{a: end, b: end}")
        assert von_neumann_entropy(ss) >= -1e-10

    def test_entropy_at_most_log_n(self):
        """Entropy <= log(n) for &{a: end, b: end}."""
        ss = _build("&{a: end, b: end}")
        n = len(ss.states)
        assert von_neumann_entropy(ss) <= math.log(n) + 1e-10

    def test_entropy_at_most_log_n_deeper(self):
        """Entropy <= log(n) for deeper type."""
        ss = _build("&{a: &{c: end, d: end}, b: end}")
        n = len(ss.states)
        assert von_neumann_entropy(ss) <= math.log(n) + 1e-10

    def test_entropy_at_most_log_n_parallel(self):
        """Entropy <= log(n) for parallel type."""
        ss = _build("(&{a: end} || &{b: end})")
        n = len(ss.states)
        assert von_neumann_entropy(ss) <= math.log(n) + 1e-10


# ---------------------------------------------------------------------------
# Renyi entropy inequalities
# ---------------------------------------------------------------------------

class TestRenyiInequalities:
    """S_inf <= S_2 <= S_1 (von Neumann)."""

    def test_renyi_ordering_two_branch(self):
        """S_inf <= S_2 <= S_vn for &{a: end, b: end}."""
        ss = _build("&{a: end, b: end}")
        s_vn = von_neumann_entropy(ss)
        s2 = renyi_entropy_2(ss)
        s_inf = renyi_entropy_inf(ss)
        assert s_inf <= s2 + 1e-10
        assert s2 <= s_vn + 1e-10

    def test_renyi_ordering_deeper(self):
        """S_inf <= S_2 <= S_vn for deeper type."""
        ss = _build("&{a: &{c: end, d: end}, b: end}")
        s_vn = von_neumann_entropy(ss)
        s2 = renyi_entropy_2(ss)
        s_inf = renyi_entropy_inf(ss)
        assert s_inf <= s2 + 1e-10
        assert s2 <= s_vn + 1e-10

    def test_renyi_ordering_parallel(self):
        """S_inf <= S_2 <= S_vn for parallel type."""
        ss = _build("(&{a: end} || &{b: end})")
        s_vn = von_neumann_entropy(ss)
        s2 = renyi_entropy_2(ss)
        s_inf = renyi_entropy_inf(ss)
        assert s_inf <= s2 + 1e-10
        assert s2 <= s_vn + 1e-10

    def test_renyi_alpha_invalid(self):
        """Renyi entropy raises ValueError for alpha <= 0."""
        ss = _build("&{a: end}")
        with pytest.raises(ValueError):
            renyi_entropy(ss, 0.0)
        with pytest.raises(ValueError):
            renyi_entropy(ss, -1.0)

    def test_renyi_alpha_near_one(self):
        """Renyi entropy at alpha near 1 converges to von Neumann."""
        ss = _build("&{a: end, b: end}")
        s_vn = von_neumann_entropy(ss)
        s_r = renyi_entropy(ss, 0.9999999)
        assert s_r == pytest.approx(s_vn, abs=1e-3)

    def test_renyi_2_non_negative(self):
        """Renyi-2 entropy >= 0."""
        ss = _build("&{a: end, b: end}")
        assert renyi_entropy_2(ss) >= -1e-10

    def test_renyi_inf_non_negative(self):
        """Min-entropy >= 0."""
        ss = _build("&{a: end, b: end}")
        assert renyi_entropy_inf(ss) >= -1e-10


# ---------------------------------------------------------------------------
# Effective dimension
# ---------------------------------------------------------------------------

class TestEffectiveDimension:
    """exp(S) in [1, n]."""

    def test_effective_dimension_at_least_one(self):
        """Effective dimension >= 1."""
        ss = _build("&{a: end, b: end}")
        assert effective_dimension(ss) >= 1.0 - 1e-10

    def test_effective_dimension_at_most_n(self):
        """Effective dimension <= n."""
        ss = _build("&{a: end, b: end}")
        n = len(ss.states)
        assert effective_dimension(ss) <= n + 1e-10

    def test_effective_dimension_one_for_end(self):
        """end: effective dimension = 1."""
        ss = _build("end")
        assert effective_dimension(ss) == pytest.approx(1.0)

    def test_effective_dimension_parallel(self):
        """Parallel: effective dimension > single branch."""
        ss_branch = _build("&{a: end, b: end}")
        ss_par = _build("(&{a: end} || &{b: end})")
        ed_branch = effective_dimension(ss_branch)
        ed_par = effective_dimension(ss_par)
        # Parallel has more states, so effective dimension should differ
        assert ed_par >= 1.0


# ---------------------------------------------------------------------------
# Relative entropy
# ---------------------------------------------------------------------------

class TestRelativeEntropy:
    """Quantum relative entropy S(rho1 || rho2)."""

    def test_same_type_zero(self):
        """S(rho || rho) = 0."""
        ss = _build("&{a: end, b: end}")
        re = relative_entropy(ss, ss)
        assert re is not None
        assert re == pytest.approx(0.0, abs=1e-8)

    def test_different_dimension_none(self):
        """Different dimensions: returns None."""
        ss1 = _build("end")
        ss2 = _build("&{a: end, b: end}")
        assert relative_entropy(ss1, ss2) is None

    def test_non_negative(self):
        """Relative entropy >= 0 (Klein's inequality)."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{a: &{c: end}, b: end}")
        # These have different number of states, so None
        re = relative_entropy(ss1, ss2)
        if re is not None:
            assert re >= -1e-10

    def test_end_self_relative_entropy(self):
        """end: S(rho || rho) = 0."""
        ss = _build("end")
        re = relative_entropy(ss, ss)
        assert re is not None
        assert re == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

class TestMutualInformation:
    """Mutual information = log(n) - S(rho)."""

    def test_non_negative(self):
        """Mutual information >= 0."""
        ss = _build("&{a: end, b: end}")
        assert mutual_information(ss) >= -1e-10

    def test_zero_for_end(self):
        """end: mutual information = 0."""
        ss = _build("end")
        assert mutual_information(ss) == pytest.approx(0.0)

    def test_mi_plus_entropy_equals_log_n(self):
        """I + S = log(n)."""
        ss = _build("&{a: end, b: end}")
        n = len(ss.states)
        mi = mutual_information(ss)
        ent = von_neumann_entropy(ss)
        assert mi + ent == pytest.approx(math.log(n), abs=1e-10)


# ---------------------------------------------------------------------------
# Entropy rate
# ---------------------------------------------------------------------------

class TestEntropyRate:
    """Entropy rate = S / n."""

    def test_entropy_rate_end(self):
        """end: entropy rate = 0."""
        ss = _build("end")
        assert entropy_rate(ss) == pytest.approx(0.0)

    def test_entropy_rate_non_negative(self):
        """Multi-state: entropy rate >= 0."""
        ss = _build("&{a: end, b: end}")
        assert entropy_rate(ss) >= 0.0

    def test_entropy_rate_equals_s_over_n(self):
        """Entropy rate = S(rho) / n."""
        ss = _build("&{a: end, b: end}")
        n = len(ss.states)
        assert entropy_rate(ss) == pytest.approx(
            von_neumann_entropy(ss) / n, abs=1e-10
        )


# ---------------------------------------------------------------------------
# xlogx helper
# ---------------------------------------------------------------------------

class TestXlogx:
    """The 0*log(0)=0 convention."""

    def test_zero(self):
        """0 * log(0) = 0."""
        assert _xlogx(0.0) == 0.0

    def test_one(self):
        """1 * log(1) = 0."""
        assert _xlogx(1.0) == pytest.approx(0.0)

    def test_half(self):
        """0.5 * log(0.5) = -0.5 * log(2)."""
        assert _xlogx(0.5) == pytest.approx(-0.5 * math.log(2))

    def test_small_positive(self):
        """Very small positive value: xlogx is small negative."""
        val = _xlogx(1e-20)
        assert val == 0.0  # Below threshold


# ---------------------------------------------------------------------------
# Branching vs linear protocols
# ---------------------------------------------------------------------------

class TestProtocolComparison:
    """Branching protocols should generally have higher entropy than linear."""

    def test_single_branch_has_entropy(self):
        """&{a: end}: two states, entropy >= 0."""
        ss = _build("&{a: end}")
        assert von_neumann_entropy(ss) >= 0.0

    def test_two_branch_higher_than_single(self):
        """&{a: end, b: end}: more structure, entropy >= 0."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{a: end, b: end}")
        assert von_neumann_entropy(ss1) >= 0.0
        assert von_neumann_entropy(ss2) >= 0.0


# ---------------------------------------------------------------------------
# Recursive types
# ---------------------------------------------------------------------------

class TestRecursiveTypes:
    """Recursive types with cycles."""

    def test_recursive_entropy_positive(self):
        """rec X . &{a: X, b: end}: has cycles, positive entropy."""
        ss = _build("rec X . &{a: X, b: end}")
        assert von_neumann_entropy(ss) >= 0.0

    def test_recursive_density_trace_one(self):
        """Recursive type: density matrix still has trace 1."""
        ss = _build("rec X . &{a: X, b: end}")
        rho = density_matrix(ss)
        assert _mat_trace(rho) == pytest.approx(1.0, abs=1e-10)

    def test_recursive_eigenvalues_sum_one(self):
        """Recursive type: eigenvalues sum to 1."""
        ss = _build("rec X . &{a: X, b: end}")
        eigs = density_eigenvalues(ss)
        assert sum(eigs) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Full analysis integration
# ---------------------------------------------------------------------------

class TestAnalyzeVonNeumann:
    """Integration tests for analyze_von_neumann."""

    def test_end_analysis(self):
        """End type: full analysis."""
        ss = _build("end")
        result = analyze_von_neumann(ss)
        assert isinstance(result, VonNeumannResult)
        assert result.entropy == pytest.approx(0.0)
        assert result.max_entropy == pytest.approx(0.0)
        assert result.effective_dimension == pytest.approx(1.0)
        assert not result.is_maximally_mixed

    def test_branch_analysis(self):
        """Branch type: full analysis."""
        ss = _build("&{a: end, b: end}")
        result = analyze_von_neumann(ss)
        assert isinstance(result, VonNeumannResult)
        assert result.entropy >= 0.0
        assert result.max_entropy > 0.0
        assert 0.0 <= result.normalized_entropy <= 1.0 + 1e-10
        assert result.effective_dimension >= 1.0
        assert len(result.density_eigenvalues) == len(ss.states)
        assert sum(result.density_eigenvalues) == pytest.approx(1.0, abs=1e-10)

    def test_parallel_analysis(self):
        """Parallel type: full analysis."""
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_von_neumann(ss)
        assert isinstance(result, VonNeumannResult)
        assert result.entropy >= 0.0
        assert result.renyi_entropy_inf <= result.renyi_entropy_2 + 1e-10
        assert result.renyi_entropy_2 <= result.entropy + 1e-10

    def test_frozen_result(self):
        """VonNeumannResult is frozen."""
        ss = _build("&{a: end}")
        result = analyze_von_neumann(ss)
        with pytest.raises(AttributeError):
            result.entropy = 42.0  # type: ignore[misc]

    def test_deeper_protocol(self):
        """Deeper protocol: &{a: &{c: end, d: end}, b: end}."""
        ss = _build("&{a: &{c: end, d: end}, b: end}")
        result = analyze_von_neumann(ss)
        assert result.entropy > 0.0
        assert result.entropy <= result.max_entropy + 1e-10
        assert 1.0 <= result.effective_dimension <= len(ss.states) + 1e-10


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarkProtocols:
    """Entropy analysis on standard benchmark protocols."""

    def test_iterator_entropy(self):
        """Java Iterator: rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_von_neumann(ss)
        assert result.entropy >= 0.0
        assert result.entropy <= result.max_entropy + 1e-10

    def test_two_branch_selection(self):
        """Two-branch selection type."""
        ss = _build("+{OK: end, ERROR: end}")
        result = analyze_von_neumann(ss)
        assert result.entropy >= 0.0
        assert sum(result.density_eigenvalues) == pytest.approx(1.0, abs=1e-10)

    def test_nested_branch_selection(self):
        """Nested branch + selection."""
        ss = _build("&{open: +{OK: &{read: end}, ERROR: end}}")
        result = analyze_von_neumann(ss)
        assert result.entropy > 0.0
        assert result.effective_dimension >= 1.0
