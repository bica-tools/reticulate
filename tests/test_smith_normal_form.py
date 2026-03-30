"""Tests for Smith Normal Form analysis (Step 30ae).

Tests cover:
- Adjacency matrix construction on SCC quotient
- Laplacian matrix construction (L = D_out - A)
- Signed incidence matrix construction
- Smith Normal Form computation (D = U @ A @ V)
- Invariant factors extraction and divisibility
- Critical group (sandpile group) from reduced Laplacian
- Full analysis via analyze_smith_normal_form
- Benchmark protocols (iterator, file object, diamond, parallel)
- Extended GCD and determinant helpers

Target: 60+ tests across 9 test classes.
"""

from __future__ import annotations

import math

import pytest
from reticulate.parser import parse
from reticulate.smith_normal_form import (
    SmithNormalFormResult,
    _deep_copy,
    _determinant,
    _extended_gcd,
    _identity,
    adjacency_matrix,
    analyze_smith_normal_form,
    critical_group,
    critical_group_order,
    incidence_matrix,
    invariant_factors,
    laplacian_matrix,
    smith_normal_form,
)
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str) -> StateSpace:
    """Parse and build state space."""
    return build_statespace(parse(s))


def _matmul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Integer matrix multiplication."""
    m = len(A)
    p = len(B)
    n = len(B[0]) if B else 0
    C = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
    return C


def _is_diagonal(M: list[list[int]]) -> bool:
    """Check if a matrix is diagonal (off-diagonal entries are zero)."""
    m = len(M)
    n = len(M[0]) if M else 0
    for i in range(m):
        for j in range(n):
            if i != j and M[i][j] != 0:
                return False
    return True


# ---------------------------------------------------------------------------
# TestExtendedGCD
# ---------------------------------------------------------------------------

class TestExtendedGCD:
    """Tests for the extended GCD helper."""

    def test_basic(self) -> None:
        g, x, y = _extended_gcd(12, 8)
        assert g == 4
        assert 12 * x + 8 * y == 4

    def test_coprime(self) -> None:
        g, x, y = _extended_gcd(7, 5)
        assert g == 1
        assert 7 * x + 5 * y == 1

    def test_zero_second(self) -> None:
        g, x, y = _extended_gcd(5, 0)
        assert g == 5
        assert 5 * x + 0 * y == 5

    def test_both_zero(self) -> None:
        g, x, y = _extended_gcd(0, 0)
        assert g == 0

    def test_negative_first(self) -> None:
        g, x, y = _extended_gcd(-6, 4)
        assert g == 2
        assert -6 * x + 4 * y == 2

    def test_gcd_always_positive(self) -> None:
        g, _, _ = _extended_gcd(-15, -10)
        assert g > 0


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for identity and deep copy helpers."""

    def test_identity_3x3(self) -> None:
        I = _identity(3)
        assert I == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_identity_1x1(self) -> None:
        assert _identity(1) == [[1]]

    def test_identity_0x0(self) -> None:
        assert _identity(0) == []

    def test_deep_copy_independent(self) -> None:
        M = [[1, 2], [3, 4]]
        C = _deep_copy(M)
        C[0][0] = 99
        assert M[0][0] == 1


# ---------------------------------------------------------------------------
# TestDeterminant
# ---------------------------------------------------------------------------

class TestDeterminant:
    """Tests for the internal determinant computation."""

    def test_det_1x1(self) -> None:
        assert _determinant([[7]]) == 7

    def test_det_2x2(self) -> None:
        assert _determinant([[1, 2], [3, 4]]) == -2

    def test_det_3x3_identity(self) -> None:
        assert _determinant([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 1

    def test_det_singular(self) -> None:
        assert _determinant([[1, 2], [2, 4]]) == 0

    def test_det_empty(self) -> None:
        assert _determinant([]) == 1

    def test_det_negative(self) -> None:
        assert _determinant([[-3]]) == -3


# ---------------------------------------------------------------------------
# TestAdjacencyMatrix
# ---------------------------------------------------------------------------

class TestAdjacencyMatrix:
    """Tests for adjacency matrix construction on SCC quotient."""

    def test_end_single_state(self) -> None:
        """end has a single state, so 1x1 zero matrix."""
        ss = _build("end")
        A = adjacency_matrix(ss)
        assert A == [[0]]

    def test_branch_two_states(self) -> None:
        """&{a: end} has 2 quotient states, one edge top -> bottom."""
        ss = _build("&{a: end}")
        A = adjacency_matrix(ss)
        assert len(A) == 2
        assert len(A[0]) == 2
        total = sum(A[i][j] for i in range(2) for j in range(2))
        assert total == 1

    def test_two_branch_same_target(self) -> None:
        """&{a: end, b: end}: both methods lead to same bottom, one quotient edge."""
        ss = _build("&{a: end, b: end}")
        A = adjacency_matrix(ss)
        assert len(A) == 2
        # Quotient: at most one edge per pair
        total = sum(A[i][j] for i in range(2) for j in range(2))
        assert total == 1

    def test_chain_three_states(self) -> None:
        """&{a: &{b: end}} has 3 states in a chain."""
        ss = _build("&{a: &{b: end}}")
        A = adjacency_matrix(ss)
        assert len(A) == 3
        total = sum(A[i][j] for i in range(3) for j in range(3))
        assert total == 2

    def test_dimensions_match_quotient(self) -> None:
        """Matrix dimensions match number of SCC representatives."""
        ss = _build("&{a: end, b: &{c: end}}")
        A = adjacency_matrix(ss)
        n = len(A)
        assert n > 0
        assert all(len(row) == n for row in A)

    def test_entries_nonnegative(self) -> None:
        """All entries are non-negative."""
        ss = _build("&{a: &{b: end}, c: end}")
        A = adjacency_matrix(ss)
        for row in A:
            for val in row:
                assert val >= 0

    def test_no_self_loops(self) -> None:
        """Diagonal entries are zero (no self-loops in quotient DAG)."""
        ss = _build("&{a: &{b: end}}")
        A = adjacency_matrix(ss)
        for i in range(len(A)):
            assert A[i][i] == 0

    def test_diamond_out_degree(self) -> None:
        """Diamond top has out-degree 2."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        A = adjacency_matrix(ss)
        assert len(A) == 4
        out_degrees = [sum(row) for row in A]
        assert max(out_degrees) == 2


# ---------------------------------------------------------------------------
# TestLaplacianMatrix
# ---------------------------------------------------------------------------

class TestLaplacianMatrix:
    """Tests for Laplacian matrix L = D_out - A."""

    def test_end_zero(self) -> None:
        """end: single state with no edges, L = [[0]]."""
        ss = _build("end")
        assert laplacian_matrix(ss) == [[0]]

    def test_diagonal_equals_out_degree(self) -> None:
        """Diagonal entries equal out-degree."""
        ss = _build("&{a: end, b: end}")
        A = adjacency_matrix(ss)
        L = laplacian_matrix(ss)
        n = len(A)
        for i in range(n):
            assert L[i][i] == sum(A[i])

    def test_off_diagonal_negative_adjacency(self) -> None:
        """Off-diagonal L[i][j] = -A[i][j]."""
        ss = _build("&{a: &{b: end}}")
        A = adjacency_matrix(ss)
        L = laplacian_matrix(ss)
        n = len(A)
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert L[i][j] == -A[i][j]

    def test_row_sums_zero(self) -> None:
        """Each row of the Laplacian sums to zero."""
        ss = _build("&{a: end, b: &{c: end}}")
        L = laplacian_matrix(ss)
        for row in L:
            assert sum(row) == 0

    def test_dimensions_match_adjacency(self) -> None:
        """Laplacian has same dimensions as adjacency."""
        ss = _build("&{a: end}")
        A = adjacency_matrix(ss)
        L = laplacian_matrix(ss)
        assert len(L) == len(A)
        assert len(L[0]) == len(A[0])

    def test_diagonal_nonnegative(self) -> None:
        """Diagonal entries (out-degrees) are non-negative."""
        ss = _build("&{a: end, b: end}")
        L = laplacian_matrix(ss)
        for i in range(len(L)):
            assert L[i][i] >= 0

    def test_diamond_row_sums(self) -> None:
        """Diamond Laplacian has zero row sums."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        L = laplacian_matrix(ss)
        for row in L:
            assert sum(row) == 0


# ---------------------------------------------------------------------------
# TestIncidenceMatrix
# ---------------------------------------------------------------------------

class TestIncidenceMatrix:
    """Tests for signed incidence matrix (states x edges)."""

    def test_end_no_edges(self) -> None:
        """end: no edges, incidence matrix has zero columns."""
        ss = _build("end")
        B = incidence_matrix(ss)
        assert len(B) == 1
        assert len(B[0]) == 0

    def test_branch_single_edge(self) -> None:
        """&{a: end}: one edge, incidence matrix is 2 x 1."""
        ss = _build("&{a: end}")
        B = incidence_matrix(ss)
        assert len(B) == 2
        assert len(B[0]) == 1

    def test_chain_two_edges(self) -> None:
        """&{a: &{b: end}}: two edges in chain."""
        ss = _build("&{a: &{b: end}}")
        B = incidence_matrix(ss)
        assert len(B[0]) == 2

    def test_column_sums_zero(self) -> None:
        """Each column sums to zero (+1 at source, -1 at target)."""
        ss = _build("&{a: &{b: end}, c: end}")
        B = incidence_matrix(ss)
        if B and B[0]:
            n_edges = len(B[0])
            for j in range(n_edges):
                col_sum = sum(B[i][j] for i in range(len(B)))
                assert col_sum == 0

    def test_column_exactly_one_plus_one_minus(self) -> None:
        """Each column has exactly one +1 and one -1."""
        ss = _build("&{a: &{b: end}, c: end}")
        B = incidence_matrix(ss)
        if B and B[0]:
            n_states = len(B)
            n_edges = len(B[0])
            for j in range(n_edges):
                col = [B[i][j] for i in range(n_states)]
                assert col.count(1) == 1
                assert col.count(-1) == 1
                assert col.count(0) == n_states - 2

    def test_dimensions_states_by_edges(self) -> None:
        """Rows = states, columns = edges."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        A = adjacency_matrix(ss)
        B = incidence_matrix(ss)
        n_states = len(A)
        assert len(B) == n_states
        n_edges_from_adj = sum(sum(row) for row in A)
        n_edges_from_inc = len(B[0]) if B[0] else 0
        assert n_edges_from_inc == n_edges_from_adj


# ---------------------------------------------------------------------------
# TestSmithNormalForm
# ---------------------------------------------------------------------------

class TestSmithNormalForm:
    """Tests for the Smith Normal Form computation."""

    def test_identity_snf(self) -> None:
        """SNF of identity is identity."""
        U, D, V = smith_normal_form([[1, 0], [0, 1]])
        assert D == [[1, 0], [0, 1]]

    def test_zero_matrix_snf(self) -> None:
        """SNF of zero matrix is zero matrix."""
        U, D, V = smith_normal_form([[0, 0], [0, 0]])
        assert D == [[0, 0], [0, 0]]

    def test_1x1_positive(self) -> None:
        """SNF of [[5]] is [[5]]."""
        U, D, V = smith_normal_form([[5]])
        assert D == [[5]]

    def test_1x1_negative(self) -> None:
        """SNF of [[-3]] has positive diagonal."""
        U, D, V = smith_normal_form([[-3]])
        assert D == [[3]]

    def test_d_is_diagonal_2x2(self) -> None:
        """D is diagonal for a 2x2 matrix."""
        U, D, V = smith_normal_form([[1, 2], [3, 4]])
        assert _is_diagonal(D)

    def test_d_is_diagonal_3x3(self) -> None:
        """D is diagonal for a 3x3 matrix."""
        U, D, V = smith_normal_form([[2, 4, 4], [0, 6, 12], [0, 0, 8]])
        assert _is_diagonal(D)

    def test_factorization_2x2(self) -> None:
        """D = U @ A @ V for a 2x2 matrix."""
        A = [[2, 4], [6, 8]]
        U, D, V = smith_normal_form(A)
        assert _matmul(_matmul(U, A), V) == D

    def test_factorization_3x3(self) -> None:
        """D = U @ A @ V for a 3x3 matrix."""
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        U, D, V = smith_normal_form(A)
        assert _matmul(_matmul(U, A), V) == D

    def test_factorization_rectangular_3x2(self) -> None:
        """D = U @ A @ V for a 3x2 matrix."""
        A = [[0, -1], [1, 0], [-1, 1]]
        U, D, V = smith_normal_form(A)
        assert _matmul(_matmul(U, A), V) == D

    def test_factorization_rectangular_2x3(self) -> None:
        """D = U @ A @ V for a 2x3 matrix."""
        A = [[1, 2, 3], [4, 5, 6]]
        U, D, V = smith_normal_form(A)
        assert _matmul(_matmul(U, A), V) == D

    def test_u_unimodular(self) -> None:
        """U is unimodular (|det| = 1)."""
        A = [[2, 4], [6, 8]]
        U, D, V = smith_normal_form(A)
        assert abs(_determinant(U)) == 1

    def test_v_unimodular(self) -> None:
        """V is unimodular (|det| = 1)."""
        A = [[2, 4], [6, 8]]
        U, D, V = smith_normal_form(A)
        assert abs(_determinant(V)) == 1

    def test_unimodular_3x3(self) -> None:
        """U, V unimodular for 3x3."""
        A = [[3, 1, 0], [1, 2, 1], [0, 1, 3]]
        U, D, V = smith_normal_form(A)
        assert abs(_determinant(U)) == 1
        assert abs(_determinant(V)) == 1

    def test_diagonal_nonnegative(self) -> None:
        """Diagonal entries of D are non-negative."""
        A = [[3, 6], [9, 12]]
        U, D, V = smith_normal_form(A)
        n = min(len(D), len(D[0]))
        for i in range(n):
            assert D[i][i] >= 0

    def test_divisibility_chain(self) -> None:
        """d_1 | d_2 | ... | d_k on the diagonal."""
        A = [[2, 4], [6, 8]]
        U, D, V = smith_normal_form(A)
        diag = [D[i][i] for i in range(min(len(D), len(D[0]))) if D[i][i] != 0]
        for i in range(len(diag) - 1):
            assert diag[i + 1] % diag[i] == 0

    def test_known_factors_2_4(self) -> None:
        """[[2,4],[6,8]] has invariant factors [2, 4]."""
        assert invariant_factors([[2, 4], [6, 8]]) == [2, 4]

    def test_empty_matrix(self) -> None:
        """Empty matrix returns empty decomposition."""
        U, D, V = smith_normal_form([])
        assert D == []

    def test_preserves_rank(self) -> None:
        """Rank (non-zero diagonal count) equals matrix rank."""
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        U, D, V = smith_normal_form(A)
        rank = sum(1 for i in range(3) if D[i][i] != 0)
        assert rank == 2  # rank-deficient matrix

    def test_factorization_variety(self) -> None:
        """D = U @ A @ V for a variety of matrices."""
        matrices = [
            [[1, 0], [0, 1]],
            [[2, 3], [4, 5]],
            [[6, 0, 0], [0, 3, 0], [0, 0, 15]],
            [[1, 2], [3, 4], [5, 6]],
        ]
        for M in matrices:
            U, D, V = smith_normal_form(M)
            assert _matmul(_matmul(U, M), V) == D, f"Failed for {M}"

    def test_diagonal_coprime_becomes_1_and_product(self) -> None:
        """diag(2, 3) -> SNF diag(1, 6) since gcd(2,3)=1."""
        U, D, V = smith_normal_form([[2, 0], [0, 3]])
        diag = sorted([D[i][i] for i in range(2)])
        assert diag == [1, 6]


# ---------------------------------------------------------------------------
# TestInvariantFactors
# ---------------------------------------------------------------------------

class TestInvariantFactors:
    """Tests for invariant factor extraction."""

    def test_identity_factors(self) -> None:
        assert invariant_factors([[1, 0], [0, 1]]) == [1, 1]

    def test_zero_matrix_no_factors(self) -> None:
        assert invariant_factors([[0, 0], [0, 0]]) == []

    def test_factors_positive(self) -> None:
        """All invariant factors are positive."""
        factors = invariant_factors([[2, 4], [6, 8]])
        assert all(f > 0 for f in factors)

    def test_factors_divisibility(self) -> None:
        """Each factor divides the next."""
        factors = invariant_factors([[3, 6], [9, 12]])
        for i in range(len(factors) - 1):
            assert factors[i + 1] % factors[i] == 0

    def test_count_equals_rank(self) -> None:
        """Number of factors = rank."""
        factors = invariant_factors([[1, 2], [3, 4]])
        assert len(factors) == 2

    def test_empty_matrix(self) -> None:
        assert invariant_factors([]) == []

    def test_scalar(self) -> None:
        assert invariant_factors([[7]]) == [7]

    def test_product_equals_abs_det(self) -> None:
        """Product of factors = |det| for full-rank square matrices."""
        A = [[2, 4], [6, 8]]
        factors = invariant_factors(A)
        product = math.prod(factors)
        det = abs(A[0][0] * A[1][1] - A[0][1] * A[1][0])
        assert product == det

    def test_3x3_diagonal_factors(self) -> None:
        """Diagonal 3x3 gets SNF factors with divisibility."""
        factors = invariant_factors([[6, 0, 0], [0, 12, 0], [0, 0, 60]])
        for i in range(len(factors) - 1):
            assert factors[i + 1] % factors[i] == 0


# ---------------------------------------------------------------------------
# TestCriticalGroup
# ---------------------------------------------------------------------------

class TestCriticalGroup:
    """Tests for critical group (sandpile group)."""

    def test_end_trivial(self) -> None:
        """end: single state, trivial critical group."""
        assert critical_group(_build("end")) == []

    def test_branch_trivial(self) -> None:
        """&{a: end}: tree, trivial critical group."""
        assert critical_group(_build("&{a: end}")) == []

    def test_chain_trivial(self) -> None:
        """Chain is a tree, trivial critical group."""
        assert critical_group(_build("&{a: &{b: end}}")) == []

    def test_diamond_nontrivial(self) -> None:
        """Diamond &{a: &{c: end}, b: &{c: end}} has critical group [2]."""
        cg = critical_group(_build("&{a: &{c: end}, b: &{c: end}}"))
        assert cg == [2]

    def test_parallel_nontrivial(self) -> None:
        """Parallel creates non-trivial critical group."""
        cg = critical_group(_build("(&{a: end} || &{b: end})"))
        assert len(cg) > 0

    def test_group_order_positive(self) -> None:
        """Critical group order is always positive."""
        for s in ["end", "&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            assert critical_group_order(_build(s)) > 0

    def test_order_is_product_of_factors(self) -> None:
        """Order = product of factors for non-trivial group."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        cg = critical_group(ss)
        order = critical_group_order(ss)
        if cg:
            assert order == math.prod(cg)

    def test_tree_order_one(self) -> None:
        """Trees have critical group order 1."""
        for s in ["end", "&{a: end}", "&{a: &{b: end}}", "&{a: &{b: &{c: end}}}"]:
            assert critical_group_order(_build(s)) == 1


# ---------------------------------------------------------------------------
# TestAnalyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    """Tests for full analyze_smith_normal_form."""

    def test_end_analysis(self) -> None:
        r = analyze_smith_normal_form(_build("end"))
        assert isinstance(r, SmithNormalFormResult)
        assert r.adjacency_snf == []
        assert r.matrix_rank == 0
        assert r.nullity == 1

    def test_branch_analysis(self) -> None:
        r = analyze_smith_normal_form(_build("&{a: end}"))
        assert r.adjacency_snf == [1]
        assert r.matrix_rank == 1
        assert r.nullity == 1

    def test_chain_analysis(self) -> None:
        r = analyze_smith_normal_form(_build("&{a: &{b: end}}"))
        assert r.adjacency_snf == [1, 1]
        assert r.matrix_rank == 2
        assert r.nullity == 1
        assert r.critical_group == []
        assert r.critical_group_order == 1

    def test_parallel_analysis(self) -> None:
        r = analyze_smith_normal_form(_build("(&{a: end} || &{b: end})"))
        assert r.matrix_rank > 0
        assert r.critical_group_order > 0
        assert len(r.laplacian_snf) > 0

    def test_diamond_analysis(self) -> None:
        r = analyze_smith_normal_form(_build("&{a: &{c: end}, b: &{c: end}}"))
        assert r.critical_group == [2]
        assert r.critical_group_order == 2

    def test_result_is_frozen(self) -> None:
        r = analyze_smith_normal_form(_build("end"))
        with pytest.raises(AttributeError):
            r.matrix_rank = 42  # type: ignore[misc]

    def test_all_fields_populated(self) -> None:
        r = analyze_smith_normal_form(_build("&{a: end}"))
        assert isinstance(r.adjacency_snf, list)
        assert isinstance(r.laplacian_snf, list)
        assert isinstance(r.incidence_snf, list)
        assert isinstance(r.critical_group, list)
        assert isinstance(r.critical_group_order, int)
        assert isinstance(r.matrix_rank, int)
        assert isinstance(r.nullity, int)

    def test_rank_plus_nullity_equals_n(self) -> None:
        """rank + nullity = n for various types."""
        for s in ["end", "&{a: end}", "&{a: &{b: end}}", "&{a: end, b: end}"]:
            ss = _build(s)
            r = analyze_smith_normal_form(ss)
            n = len(adjacency_matrix(ss))
            assert r.matrix_rank + r.nullity == n

    def test_selection_analysis(self) -> None:
        r = analyze_smith_normal_form(_build("+{a: end, b: end}"))
        assert isinstance(r.adjacency_snf, list)
        assert isinstance(r.laplacian_snf, list)

    def test_recursive_type(self) -> None:
        """Recursive types with SCCs produce valid results."""
        r = analyze_smith_normal_form(_build("rec X . &{a: X, b: end}"))
        assert isinstance(r, SmithNormalFormResult)
        assert r.matrix_rank >= 0

    def test_invariant_factors_divisibility(self) -> None:
        """All factor lists satisfy divisibility chain."""
        r = analyze_smith_normal_form(_build("&{a: &{b: end}, c: end}"))
        for factors in [r.adjacency_snf, r.laplacian_snf, r.incidence_snf]:
            for i in range(len(factors) - 1):
                if factors[i] != 0:
                    assert factors[i + 1] % factors[i] == 0


# ---------------------------------------------------------------------------
# TestBenchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """SNF analysis on benchmark protocols."""

    BENCHMARKS = [
        ("end", "end"),
        ("branch", "&{a: end}"),
        ("two-branch", "&{a: end, b: end}"),
        ("selection", "+{ok: end, err: end}"),
        ("mixed-depth", "&{a: end, b: &{c: end}}"),
        ("open-select", "&{open: +{ok: &{read: end}, err: end}}"),
        ("simple-rec", "rec X . &{next: X, stop: end}"),
        ("iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("parallel-basic", "(&{a: end} || &{b: end})"),
        ("parallel-multi", "(&{a: end, b: end} || &{c: end})"),
        ("file-object", "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"),
        ("smtp-like", "&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: +{OK: X, ERR: X}}}, quit: end}}}"),
    ]

    @pytest.mark.parametrize("name,type_str", BENCHMARKS, ids=[b[0] for b in BENCHMARKS])
    def test_analysis_valid(self, name: str, type_str: str) -> None:
        """Full analysis produces valid SmithNormalFormResult."""
        ss = _build(type_str)
        r = analyze_smith_normal_form(ss)
        assert isinstance(r, SmithNormalFormResult)
        n = len(adjacency_matrix(ss))
        assert r.matrix_rank + r.nullity == n
        assert r.critical_group_order >= 1

    @pytest.mark.parametrize("name,type_str", BENCHMARKS, ids=[b[0] for b in BENCHMARKS])
    def test_laplacian_row_sums(self, name: str, type_str: str) -> None:
        """Laplacian rows sum to zero."""
        ss = _build(type_str)
        L = laplacian_matrix(ss)
        for row in L:
            assert sum(row) == 0

    @pytest.mark.parametrize("name,type_str", BENCHMARKS, ids=[b[0] for b in BENCHMARKS])
    def test_incidence_col_sums(self, name: str, type_str: str) -> None:
        """Incidence columns sum to zero."""
        ss = _build(type_str)
        B = incidence_matrix(ss)
        if not B or not B[0]:
            return
        for j in range(len(B[0])):
            assert sum(B[i][j] for i in range(len(B))) == 0

    @pytest.mark.parametrize("name,type_str", BENCHMARKS, ids=[b[0] for b in BENCHMARKS])
    def test_adjacency_snf_factorization(self, name: str, type_str: str) -> None:
        """D = U @ A @ V for adjacency matrix."""
        ss = _build(type_str)
        A = adjacency_matrix(ss)
        if not A or not A[0]:
            return
        U, D, V = smith_normal_form(A)
        assert _matmul(_matmul(U, A), V) == D

    def test_parallel_chain_critical_group(self) -> None:
        """Parallel with chain has richer structure."""
        r = analyze_smith_normal_form(_build("(&{a: &{b: end}} || &{c: end})"))
        assert r.critical_group_order >= 1
        assert r.matrix_rank >= 2

    def test_nested_parallel(self) -> None:
        """Nested parallel."""
        r = analyze_smith_normal_form(_build("((&{a: end} || &{b: end}) || &{c: end})"))
        assert r.matrix_rank >= 1
        assert r.critical_group_order >= 1
