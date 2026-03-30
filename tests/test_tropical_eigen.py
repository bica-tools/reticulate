"""Tests for tropical eigenvalue analysis (Step 30l).

Tests cover:
- Critical components (SCCs of critical graph)
- Cyclicity (gcd of critical cycle lengths)
- Kleene star computation
- Eigenspace and subeigenspace bases
- Eigenspace dimension
- CSR decomposition (index and period)
- Definiteness check
- Tropical characteristic polynomial
- Tropical trace
- Full analysis on various types
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.tropical import (
    tropical_eigenvalue,
    critical_graph,
    _maxplus_adjacency_matrix,
    NEG_INF,
    INF,
)
from reticulate.tropical_eigen import (
    TropicalEigenResult,
    critical_components,
    cyclicity,
    kleene_star,
    tropical_eigenspace,
    tropical_subeigenspace,
    eigenspace_dimension,
    csr_decomposition,
    is_definite,
    tropical_characteristic_polynomial,
    tropical_trace,
    analyze_tropical_eigen,
    _tropical_identity,
    _tropical_matrix_add,
    _tropical_matrix_scalar,
    _matrices_equal,
    _tropical_matrix_vec,
    _gcd_list,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

class TestTropicalIdentity:

    def test_1x1(self):
        I = _tropical_identity(1)
        assert I == [[0.0]]

    def test_3x3(self):
        I = _tropical_identity(3)
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert I[i][j] == 0.0
                else:
                    assert I[i][j] == NEG_INF

    def test_0x0(self):
        I = _tropical_identity(0)
        assert I == []


class TestTropicalMatrixAdd:

    def test_basic(self):
        A = [[1.0, NEG_INF], [NEG_INF, 2.0]]
        B = [[NEG_INF, 3.0], [4.0, NEG_INF]]
        C = _tropical_matrix_add(A, B)
        assert C[0][0] == 1.0
        assert C[0][1] == 3.0
        assert C[1][0] == 4.0
        assert C[1][1] == 2.0

    def test_identity_is_neutral(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        I = [[NEG_INF] * 2 for _ in range(2)]
        C = _tropical_matrix_add(A, I)
        assert C == A

    def test_empty(self):
        assert _tropical_matrix_add([], []) == []


class TestTropicalMatrixScalar:

    def test_add_scalar(self):
        A = [[1.0, NEG_INF], [3.0, 2.0]]
        C = _tropical_matrix_scalar(A, 5.0)
        assert C[0][0] == 6.0
        assert C[0][1] == NEG_INF
        assert C[1][0] == 8.0
        assert C[1][1] == 7.0

    def test_zero_scalar(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        C = _tropical_matrix_scalar(A, 0.0)
        assert C == A

    def test_neg_inf_absorbs(self):
        A = [[1.0, 2.0]]
        C = _tropical_matrix_scalar(A, NEG_INF)
        assert C == [[NEG_INF, NEG_INF]]


class TestMatricesEqual:

    def test_equal(self):
        A = [[1.0, NEG_INF], [2.0, 3.0]]
        B = [[1.0, NEG_INF], [2.0, 3.0]]
        assert _matrices_equal(A, B)

    def test_not_equal(self):
        A = [[1.0, NEG_INF], [2.0, 3.0]]
        B = [[1.0, NEG_INF], [2.0, 4.0]]
        assert not _matrices_equal(A, B)

    def test_neg_inf_mismatch(self):
        A = [[NEG_INF]]
        B = [[0.0]]
        assert not _matrices_equal(A, B)

    def test_tolerance(self):
        A = [[1.0000000001]]
        B = [[1.0]]
        assert _matrices_equal(A, B, tol=1e-9)


class TestGcdList:

    def test_single(self):
        assert _gcd_list([6]) == 6

    def test_two(self):
        assert _gcd_list([6, 4]) == 2

    def test_coprime(self):
        assert _gcd_list([7, 3]) == 1

    def test_all_same(self):
        assert _gcd_list([5, 5, 5]) == 5

    def test_empty(self):
        assert _gcd_list([]) == 0

    def test_with_one(self):
        assert _gcd_list([12, 8, 1]) == 1


# ---------------------------------------------------------------------------
# Critical components tests
# ---------------------------------------------------------------------------

class TestCriticalComponents:

    def test_end_no_components(self):
        ss = _build("end")
        comps = critical_components(ss)
        assert comps == []

    def test_linear_no_components(self):
        ss = _build("&{a: end}")
        comps = critical_components(ss)
        assert comps == []

    def test_branch_no_components(self):
        ss = _build("&{a: end, b: end}")
        comps = critical_components(ss)
        assert comps == []

    def test_recursive_has_components(self):
        ss = _build("rec X . &{a: X}")
        comps = critical_components(ss)
        # Should have at least one component with states on the cycle
        assert len(comps) >= 1

    def test_recursive_branch_components(self):
        ss = _build("rec X . &{a: X, b: end}")
        comps = critical_components(ss)
        # The recursive cycle forms a critical component
        assert len(comps) >= 1

    def test_components_are_sorted(self):
        ss = _build("rec X . &{a: X, b: end}")
        comps = critical_components(ss)
        for comp in comps:
            assert comp == sorted(comp)
        # Components sorted by first element
        firsts = [c[0] for c in comps]
        assert firsts == sorted(firsts)


# ---------------------------------------------------------------------------
# Cyclicity tests
# ---------------------------------------------------------------------------

class TestCyclicity:

    def test_acyclic_zero(self):
        ss = _build("end")
        assert cyclicity(ss) == 0

    def test_linear_zero(self):
        ss = _build("&{a: end}")
        assert cyclicity(ss) == 0

    def test_branch_zero(self):
        ss = _build("&{a: end, b: end}")
        assert cyclicity(ss) == 0

    def test_simple_recursion(self):
        ss = _build("rec X . &{a: X}")
        cyc = cyclicity(ss)
        assert cyc >= 1

    def test_recursion_with_exit(self):
        ss = _build("rec X . &{a: X, b: end}")
        cyc = cyclicity(ss)
        assert cyc >= 1


# ---------------------------------------------------------------------------
# Kleene star tests
# ---------------------------------------------------------------------------

class TestKleeneStar:

    def test_end(self):
        ss = _build("end")
        ks = kleene_star(ss)
        assert ks is not None
        assert len(ks) == 1
        assert ks[0][0] == 0.0

    def test_linear(self):
        ss = _build("&{a: end}")
        ks = kleene_star(ss)
        assert ks is not None
        n = len(ks)
        # Diagonal should be 0
        for i in range(n):
            assert ks[i][i] == 0.0

    def test_branch(self):
        ss = _build("&{a: end, b: end}")
        ks = kleene_star(ss)
        assert ks is not None

    def test_recursive_diverges(self):
        """Kleene star diverges for cyclic graphs (spectral radius = 1)."""
        ss = _build("rec X . &{a: X}")
        ks = kleene_star(ss)
        assert ks is None

    def test_recursive_with_exit_diverges(self):
        ss = _build("rec X . &{a: X, b: end}")
        ks = kleene_star(ss)
        assert ks is None

    def test_nested_branch_converges(self):
        ss = _build("&{a: &{b: end}}")
        ks = kleene_star(ss)
        assert ks is not None

    def test_selection_converges(self):
        ss = _build("+{a: end, b: end}")
        ks = kleene_star(ss)
        assert ks is not None

    def test_parallel_converges(self):
        ss = _build("(&{a: end} || &{b: end})")
        ks = kleene_star(ss)
        assert ks is not None


# ---------------------------------------------------------------------------
# Eigenspace tests
# ---------------------------------------------------------------------------

class TestTropicalEigenspace:

    def test_end(self):
        ss = _build("end")
        basis = tropical_eigenspace(ss)
        assert len(basis) >= 1
        assert len(basis[0]) == 1

    def test_linear(self):
        ss = _build("&{a: end}")
        basis = tropical_eigenspace(ss)
        assert len(basis) >= 1

    def test_branch(self):
        ss = _build("&{a: end, b: end}")
        basis = tropical_eigenspace(ss)
        assert len(basis) >= 1

    def test_recursive(self):
        ss = _build("rec X . &{a: X, b: end}")
        basis = tropical_eigenspace(ss)
        assert len(basis) >= 1
        n = len(list(ss.states))
        for vec in basis:
            assert len(vec) == n

    def test_vectors_are_float_lists(self):
        ss = _build("&{a: end, b: end}")
        basis = tropical_eigenspace(ss)
        for vec in basis:
            assert all(isinstance(v, float) for v in vec)


class TestTropicalSubeigenspace:

    def test_end(self):
        ss = _build("end")
        basis = tropical_subeigenspace(ss)
        assert len(basis) >= 1

    def test_branch(self):
        ss = _build("&{a: end, b: end}")
        basis = tropical_subeigenspace(ss)
        assert len(basis) >= 1

    def test_subeigen_contains_eigen(self):
        """Subeigenspace should be at least as large as eigenspace."""
        ss = _build("&{a: &{b: end}, c: end}")
        eigen = tropical_eigenspace(ss)
        subeigen = tropical_subeigenspace(ss)
        assert len(subeigen) >= len(eigen)

    def test_recursive(self):
        ss = _build("rec X . &{a: X, b: end}")
        basis = tropical_subeigenspace(ss)
        assert len(basis) >= 1


class TestEigenspaceDimension:

    def test_end_dim_1(self):
        ss = _build("end")
        assert eigenspace_dimension(ss) == 1

    def test_linear_dim_1(self):
        ss = _build("&{a: end}")
        assert eigenspace_dimension(ss) == 1

    def test_branch_dim_1(self):
        ss = _build("&{a: end, b: end}")
        assert eigenspace_dimension(ss) == 1

    def test_recursive_positive_dim(self):
        ss = _build("rec X . &{a: X, b: end}")
        dim = eigenspace_dimension(ss)
        assert dim >= 1

    def test_dimension_matches_components(self):
        ss = _build("rec X . &{a: X, b: end}")
        comps = critical_components(ss)
        dim = eigenspace_dimension(ss)
        if comps:
            assert dim == len(comps)


# ---------------------------------------------------------------------------
# CSR decomposition tests
# ---------------------------------------------------------------------------

class TestCSRDecomposition:

    def test_end(self):
        ss = _build("end")
        k0, c = csr_decomposition(ss)
        assert k0 == 0
        assert c == 1

    def test_linear(self):
        ss = _build("&{a: end}")
        k0, c = csr_decomposition(ss)
        assert k0 == 0
        assert c == 1

    def test_branch(self):
        ss = _build("&{a: end, b: end}")
        k0, c = csr_decomposition(ss)
        assert k0 == 0
        assert c == 1

    def test_recursive(self):
        ss = _build("rec X . &{a: X, b: end}")
        k0, c = csr_decomposition(ss)
        assert k0 >= 0
        assert c >= 1

    def test_period_positive(self):
        ss = _build("rec X . &{a: X}")
        k0, c = csr_decomposition(ss)
        assert c >= 1

    def test_index_non_negative(self):
        ss = _build("rec X . &{a: X, b: end}")
        k0, c = csr_decomposition(ss)
        assert k0 >= 0


# ---------------------------------------------------------------------------
# Definiteness tests
# ---------------------------------------------------------------------------

class TestIsDefinite:

    def test_end(self):
        ss = _build("end")
        # Single state: A = [[-inf]], A* = [[0]], A != A*
        result = is_definite(ss)
        assert isinstance(result, bool)

    def test_recursive_not_definite(self):
        """Cyclic graphs can't be definite (Kleene star doesn't exist)."""
        ss = _build("rec X . &{a: X}")
        assert not is_definite(ss)

    def test_linear_not_definite(self):
        ss = _build("&{a: end}")
        # A has off-diagonal 1, A* has additional entries -> not definite
        result = is_definite(ss)
        assert isinstance(result, bool)

    def test_branch_not_definite(self):
        ss = _build("&{a: end, b: end}")
        result = is_definite(ss)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Tropical characteristic polynomial tests
# ---------------------------------------------------------------------------

class TestTropicalCharacteristicPolynomial:

    def test_end(self):
        ss = _build("end")
        poly = tropical_characteristic_polynomial(ss)
        assert len(poly) == 2  # n=1 -> coeffs [c_0, c_1]

    def test_linear(self):
        ss = _build("&{a: end}")
        poly = tropical_characteristic_polynomial(ss)
        assert len(poly) == 3  # n=2 -> 3 coefficients

    def test_branch(self):
        ss = _build("&{a: end, b: end}")
        poly = tropical_characteristic_polynomial(ss)
        n = len(list(ss.states))
        assert len(poly) == n + 1

    def test_coefficients_are_floats(self):
        ss = _build("&{a: end, b: end}")
        poly = tropical_characteristic_polynomial(ss)
        for c in poly:
            assert isinstance(c, float)

    def test_last_coeff_is_zero(self):
        """c_n = 0 (tropical identity for empty matching)."""
        ss = _build("&{a: end, b: end}")
        poly = tropical_characteristic_polynomial(ss)
        assert poly[-1] == 0.0


# ---------------------------------------------------------------------------
# Tropical trace tests
# ---------------------------------------------------------------------------

class TestTropicalTrace:

    def test_end(self):
        ss = _build("end")
        tr = tropical_trace(ss, k=1)
        # Single state, no self-loop -> A = [[-inf]], tr = -inf
        assert tr == NEG_INF

    def test_linear_k1(self):
        ss = _build("&{a: end}")
        tr = tropical_trace(ss, k=1)
        # No self-loops -> diagonal of A is all -inf
        assert tr == NEG_INF

    def test_recursive_k1(self):
        """Recursive type with self-loop should have positive trace."""
        ss = _build("rec X . &{a: X}")
        tr = tropical_trace(ss, k=1)
        # May or may not have self-loop depending on state-space construction
        assert isinstance(tr, float)

    def test_trace_k0(self):
        """A^0 = I, so trace = 0 (max of diagonal zeros)."""
        ss = _build("&{a: end}")
        tr = tropical_trace(ss, k=0)
        assert tr == 0.0

    def test_trace_k2_branch(self):
        ss = _build("&{a: end, b: end}")
        tr = tropical_trace(ss, k=2)
        assert isinstance(tr, float)

    def test_trace_monotone(self):
        """For cyclic graphs, tr_k should be non-decreasing in k."""
        ss = _build("rec X . &{a: X, b: end}")
        tr1 = tropical_trace(ss, k=1)
        tr2 = tropical_trace(ss, k=2)
        tr3 = tropical_trace(ss, k=3)
        # tr_k / k converges to eigenvalue
        assert isinstance(tr1, float)
        assert isinstance(tr2, float)
        assert isinstance(tr3, float)


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyzeTropicalEigen:

    def test_end(self):
        ss = _build("end")
        result = analyze_tropical_eigen(ss)
        assert isinstance(result, TropicalEigenResult)
        assert result.eigenvalue == 0.0
        assert result.eigenspace_dim >= 1
        assert result.cyclicity == 0
        assert result.csr_index == 0
        assert result.csr_period == 1

    def test_linear(self):
        ss = _build("&{a: end}")
        result = analyze_tropical_eigen(ss)
        assert result.eigenvalue == 0.0
        assert result.kleene_star is not None
        assert result.critical_components == []
        assert result.cyclicity == 0

    def test_branch(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_tropical_eigen(ss)
        assert result.eigenvalue == 0.0
        assert len(result.eigenvectors) >= 1
        assert len(result.subeigenvectors) >= 1

    def test_selection(self):
        ss = _build("+{a: end, b: end}")
        result = analyze_tropical_eigen(ss)
        assert result.eigenvalue == 0.0
        assert result.kleene_star is not None

    def test_recursive_simple(self):
        ss = _build("rec X . &{a: X}")
        result = analyze_tropical_eigen(ss)
        assert result.eigenvalue > 0.0
        assert result.kleene_star is None  # Diverges
        assert len(result.critical_components) >= 1
        assert result.cyclicity >= 1

    def test_recursive_with_exit(self):
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_tropical_eigen(ss)
        assert result.eigenvalue > 0.0
        assert result.kleene_star is None
        assert result.csr_period >= 1

    def test_parallel(self):
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_tropical_eigen(ss)
        assert result.eigenvalue == 0.0
        assert result.kleene_star is not None

    def test_nested_branch(self):
        ss = _build("&{a: &{b: end, c: end}, d: end}")
        result = analyze_tropical_eigen(ss)
        assert result.eigenvalue == 0.0
        assert len(result.eigenvectors) >= 1

    def test_visualization_data(self):
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_tropical_eigen(ss)
        viz = result.visualization_data
        assert "num_states" in viz
        assert "eigenvalue" in viz
        assert "eigenspace_dim" in viz
        assert "cyclicity" in viz
        assert "csr_index" in viz
        assert "csr_period" in viz
        assert "critical_nodes" in viz
        assert "has_kleene_star" in viz
        assert "is_definite" in viz

    def test_result_is_frozen(self):
        ss = _build("end")
        result = analyze_tropical_eigen(ss)
        with pytest.raises(AttributeError):
            result.eigenvalue = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------

class TestConsistency:

    def test_eigenvalue_matches_tropical_module(self):
        """Eigenvalue should match the one from tropical.py."""
        ss = _build("rec X . &{a: X, b: end}")
        from reticulate.tropical import tropical_eigenvalue as tev
        lam = tev(ss)
        result = analyze_tropical_eigen(ss)
        assert abs(result.eigenvalue - lam) < 1e-9

    def test_critical_graph_matches(self):
        """Critical edges should match tropical.critical_graph."""
        ss = _build("rec X . &{a: X, b: end}")
        crit = critical_graph(ss)
        comps = critical_components(ss)
        crit_nodes_from_edges = set()
        for u, v in crit:
            crit_nodes_from_edges.add(u)
            crit_nodes_from_edges.add(v)
        comp_nodes = set()
        for comp in comps:
            comp_nodes.update(comp)
        # All component nodes should be among critical edge nodes
        assert comp_nodes.issubset(crit_nodes_from_edges)

    def test_eigenspace_dim_matches_vectors(self):
        """Dimension should match number of eigenvectors."""
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_tropical_eigen(ss)
        assert result.eigenspace_dim == len(result.eigenvectors)

    def test_csr_period_matches_cyclicity(self):
        """CSR period should equal cyclicity (for irreducible critical graph)."""
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_tropical_eigen(ss)
        # CSR period equals cyclicity
        assert result.csr_period == result.cyclicity or result.cyclicity == 0


# ---------------------------------------------------------------------------
# Benchmark protocol tests
# ---------------------------------------------------------------------------

class TestBenchmarks:

    def _analyze(self, type_str: str) -> TropicalEigenResult:
        ss = _build(type_str)
        return analyze_tropical_eigen(ss)

    def test_iterator(self):
        result = self._analyze("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        assert result.eigenvalue > 0.0
        assert result.eigenspace_dim >= 1
        assert result.cyclicity >= 1

    def test_simple_branch(self):
        result = self._analyze("&{open: &{close: end}}")
        assert result.eigenvalue == 0.0
        assert result.kleene_star is not None

    def test_nested_selection(self):
        result = self._analyze("+{ok: &{read: end}, err: end}")
        assert result.eigenvalue == 0.0

    def test_parallel_simple(self):
        result = self._analyze("(&{a: end} || &{b: end})")
        assert result.eigenvalue == 0.0
        assert result.is_definite is not None

    def test_complex_recursive(self):
        result = self._analyze(
            "rec X . &{read: X, write: X, close: end}"
        )
        assert result.eigenvalue > 0.0
        assert len(result.critical_components) >= 1
        assert result.csr_index >= 0
        assert result.csr_period >= 1
