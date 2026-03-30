"""Tests for smith_normal_form module (Step 30ae)."""

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

def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Integer matrix multiplication."""
    m = len(A)
    n = len(B[0]) if B else 0
    p = len(B)
    C = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
    return C


def _build(type_str: str) -> StateSpace:
    """Parse and build state space."""
    return build_statespace(parse(type_str))


# ---------------------------------------------------------------------------
# Test extended GCD
# ---------------------------------------------------------------------------

class TestExtendedGCD:
    def test_basic(self) -> None:
        g, x, y = _extended_gcd(12, 8)
        assert g == 4
        assert 12 * x + 8 * y == 4

    def test_coprime(self) -> None:
        g, x, y = _extended_gcd(7, 5)
        assert g == 1
        assert 7 * x + 5 * y == 1

    def test_zero(self) -> None:
        g, x, y = _extended_gcd(5, 0)
        assert g == 5
        assert 5 * x + 0 * y == 5

    def test_both_zero(self) -> None:
        g, x, y = _extended_gcd(0, 0)
        assert g == 0

    def test_negative(self) -> None:
        g, x, y = _extended_gcd(-6, 4)
        assert g == 2
        assert -6 * x + 4 * y == 2


# ---------------------------------------------------------------------------
# Test identity and deep copy
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_identity(self) -> None:
        I = _identity(3)
        assert I == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_deep_copy(self) -> None:
        M = [[1, 2], [3, 4]]
        C = _deep_copy(M)
        C[0][0] = 99
        assert M[0][0] == 1


# ---------------------------------------------------------------------------
# Test SNF on known matrices
# ---------------------------------------------------------------------------

class TestSmithNormalForm:
    def test_identity_matrix(self) -> None:
        I = [[1, 0], [0, 1]]
        U, D, V = smith_normal_form(I)
        assert D == [[1, 0], [0, 1]]

    def test_zero_matrix(self) -> None:
        Z = [[0, 0], [0, 0]]
        U, D, V = smith_normal_form(Z)
        assert D == [[0, 0], [0, 0]]

    def test_single_entry(self) -> None:
        M = [[5]]
        U, D, V = smith_normal_form(M)
        assert D == [[5]]

    def test_diagonal_matrix(self) -> None:
        M = [[2, 0], [0, 6]]
        U, D, V = smith_normal_form(M)
        diag = [D[i][i] for i in range(2)]
        diag.sort()
        assert diag == [2, 6]

    def test_2x2_simple(self) -> None:
        M = [[2, 4], [6, 8]]
        U, D, V = smith_normal_form(M)
        # D = U @ M @ V, check factorization
        product = _mat_mul(_mat_mul(U, M), V)
        assert product == D
        # Check D is diagonal with divisibility
        assert D[0][1] == 0
        assert D[1][0] == 0
        d1, d2 = abs(D[0][0]), abs(D[1][1])
        if d1 > d2:
            d1, d2 = d2, d1
        if d1 != 0:
            assert d2 % d1 == 0

    def test_3x3(self) -> None:
        M = [[2, 4, 4], [0, 6, 12], [0, 0, 8]]
        U, D, V = smith_normal_form(M)
        product = _mat_mul(_mat_mul(U, M), V)
        assert product == D
        # Check diagonal
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert D[i][j] == 0

    def test_non_square(self) -> None:
        M = [[1, 2, 3], [4, 5, 6]]
        U, D, V = smith_normal_form(M)
        product = _mat_mul(_mat_mul(U, M), V)
        assert product == D

    def test_unimodular_U_V(self) -> None:
        """U and V should be unimodular (det = +/-1)."""
        M = [[3, 1], [1, 2]]
        U, D, V = smith_normal_form(M)
        det_u = _determinant(U)
        det_v = _determinant(V)
        assert abs(det_u) == 1
        assert abs(det_v) == 1

    def test_empty_matrix(self) -> None:
        U, D, V = smith_normal_form([])
        assert U == []
        assert D == []
        assert V == []

    def test_factorization_holds(self) -> None:
        """For a variety of matrices, verify U @ A @ V = D."""
        matrices = [
            [[1, 0], [0, 1]],
            [[2, 3], [4, 5]],
            [[6, 0, 0], [0, 3, 0], [0, 0, 15]],
            [[1, 2], [3, 4], [5, 6]],
        ]
        for M in matrices:
            U, D, V = smith_normal_form(M)
            product = _mat_mul(_mat_mul(U, M), V)
            assert product == D, f"Factorization failed for {M}"


# ---------------------------------------------------------------------------
# Test invariant factors
# ---------------------------------------------------------------------------

class TestInvariantFactors:
    def test_identity(self) -> None:
        assert invariant_factors([[1, 0], [0, 1]]) == [1, 1]

    def test_zero(self) -> None:
        assert invariant_factors([[0, 0], [0, 0]]) == []

    def test_diagonal(self) -> None:
        factors = invariant_factors([[2, 0], [0, 6]])
        assert factors == [2, 6]

    def test_divisibility_chain(self) -> None:
        factors = invariant_factors([[6, 0, 0], [0, 12, 0], [0, 0, 60]])
        for i in range(len(factors) - 1):
            assert factors[i + 1] % factors[i] == 0

    def test_empty(self) -> None:
        assert invariant_factors([]) == []


# ---------------------------------------------------------------------------
# Test determinant
# ---------------------------------------------------------------------------

class TestDeterminant:
    def test_1x1(self) -> None:
        assert _determinant([[7]]) == 7

    def test_2x2(self) -> None:
        assert _determinant([[1, 2], [3, 4]]) == -2

    def test_3x3_identity(self) -> None:
        assert _determinant([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 1

    def test_singular(self) -> None:
        assert _determinant([[1, 2], [2, 4]]) == 0

    def test_empty(self) -> None:
        assert _determinant([]) == 1


# ---------------------------------------------------------------------------
# Test matrix construction from session types
# ---------------------------------------------------------------------------

class TestAdjacencyMatrix:
    def test_end(self) -> None:
        ss = _build("end")
        A = adjacency_matrix(ss)
        assert len(A) == 1
        assert A == [[0]]

    def test_branch_two(self) -> None:
        ss = _build("&{a: end, b: end}")
        A = adjacency_matrix(ss)
        n = len(A)
        assert n == 2  # top + end
        # Top has edges to end (both a and b lead to end)
        total_edges = sum(A[i][j] for i in range(n) for j in range(n))
        assert total_edges >= 1

    def test_chain(self) -> None:
        ss = _build("&{a: &{b: end}}")
        A = adjacency_matrix(ss)
        n = len(A)
        assert n == 3  # top -> a_state -> end


class TestLaplacianMatrix:
    def test_end(self) -> None:
        ss = _build("end")
        L = laplacian_matrix(ss)
        assert L == [[0]]

    def test_row_sums_zero(self) -> None:
        """Each row of the Laplacian sums to zero."""
        ss = _build("&{a: end, b: &{c: end}}")
        L = laplacian_matrix(ss)
        n = len(L)
        for i in range(n):
            assert sum(L[i]) == 0

    def test_diagonal_nonnegative(self) -> None:
        ss = _build("&{a: end, b: end}")
        L = laplacian_matrix(ss)
        n = len(L)
        for i in range(n):
            assert L[i][i] >= 0


class TestIncidenceMatrix:
    def test_end(self) -> None:
        ss = _build("end")
        M = incidence_matrix(ss)
        assert M == [[]]  # 1 state, 0 edges

    def test_column_sums_zero(self) -> None:
        """Each column of the signed incidence matrix sums to zero."""
        ss = _build("&{a: end, b: &{c: end}}")
        M = incidence_matrix(ss)
        n = len(M)
        if n == 0 or not M[0]:
            return
        m = len(M[0])
        for j in range(m):
            col_sum = sum(M[i][j] for i in range(n))
            assert col_sum == 0

    def test_branch_edges(self) -> None:
        ss = _build("&{a: &{b: end}}")
        M = incidence_matrix(ss)
        n = len(M)
        m = len(M[0]) if M and M[0] else 0
        assert m == 2  # two edges in the chain


# ---------------------------------------------------------------------------
# Test critical group
# ---------------------------------------------------------------------------

class TestCriticalGroup:
    def test_end_trivial(self) -> None:
        ss = _build("end")
        cg = critical_group(ss)
        assert cg == []

    def test_simple_branch(self) -> None:
        ss = _build("&{a: end}")
        cg = critical_group(ss)
        # Single edge: reduced Laplacian is 1x1 = [1], trivial group
        assert cg == []

    def test_order_positive(self) -> None:
        ss = _build("&{a: end, b: &{c: end}}")
        order = critical_group_order(ss)
        assert order >= 1


# ---------------------------------------------------------------------------
# Test full analysis on session types
# ---------------------------------------------------------------------------

class TestAnalyzeSmithNormalForm:
    def test_end(self) -> None:
        ss = _build("end")
        result = analyze_smith_normal_form(ss)
        assert isinstance(result, SmithNormalFormResult)
        assert result.matrix_rank == 0
        assert result.nullity == 1

    def test_simple_branch(self) -> None:
        ss = _build("&{a: end}")
        result = analyze_smith_normal_form(ss)
        assert result.matrix_rank >= 1
        assert result.nullity >= 0
        assert result.matrix_rank + result.nullity == len(adjacency_matrix(ss))

    def test_two_branch(self) -> None:
        ss = _build("&{a: end, b: end}")
        result = analyze_smith_normal_form(ss)
        assert result.matrix_rank + result.nullity == len(adjacency_matrix(ss))
        assert result.critical_group_order >= 1

    def test_chain(self) -> None:
        ss = _build("&{a: &{b: end}}")
        result = analyze_smith_normal_form(ss)
        assert result.matrix_rank >= 1

    def test_selection(self) -> None:
        ss = _build("+{a: end, b: end}")
        result = analyze_smith_normal_form(ss)
        assert isinstance(result.adjacency_snf, list)
        assert isinstance(result.laplacian_snf, list)
        assert isinstance(result.incidence_snf, list)

    def test_parallel(self) -> None:
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_smith_normal_form(ss)
        assert result.matrix_rank + result.nullity == len(adjacency_matrix(ss))

    def test_recursive(self) -> None:
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_smith_normal_form(ss)
        # Recursive types have SCCs; quotient should still work
        assert isinstance(result, SmithNormalFormResult)
        assert result.matrix_rank >= 0

    def test_invariant_factors_divisibility(self) -> None:
        """All invariant factor lists satisfy the divisibility chain."""
        ss = _build("&{a: &{b: end}, c: end}")
        result = analyze_smith_normal_form(ss)
        for factors in [result.adjacency_snf, result.laplacian_snf, result.incidence_snf]:
            for i in range(len(factors) - 1):
                if factors[i] != 0:
                    assert factors[i + 1] % factors[i] == 0

    def test_frozen_result(self) -> None:
        ss = _build("end")
        result = analyze_smith_normal_form(ss)
        with pytest.raises(AttributeError):
            result.matrix_rank = 42  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test against benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Run SNF analysis on benchmark protocols to verify no crashes."""

    BENCHMARKS = [
        "end",
        "&{a: end}",
        "&{a: end, b: end}",
        "+{ok: end, err: end}",
        "&{a: &{b: end}, c: end}",
        "&{open: +{ok: &{read: end}, err: end}}",
        "rec X . &{next: X, stop: end}",
        "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
        "(&{a: end} || &{b: end})",
        "(&{a: end, b: end} || &{c: end})",
    ]

    @pytest.mark.parametrize("type_str", BENCHMARKS)
    def test_benchmark(self, type_str: str) -> None:
        ss = _build(type_str)
        result = analyze_smith_normal_form(ss)
        assert isinstance(result, SmithNormalFormResult)
        n = len(adjacency_matrix(ss))
        assert result.matrix_rank + result.nullity == n
        assert result.critical_group_order >= 1

    @pytest.mark.parametrize("type_str", BENCHMARKS)
    def test_laplacian_row_sums_zero(self, type_str: str) -> None:
        ss = _build(type_str)
        L = laplacian_matrix(ss)
        for row in L:
            assert sum(row) == 0

    @pytest.mark.parametrize("type_str", BENCHMARKS)
    def test_incidence_col_sums_zero(self, type_str: str) -> None:
        ss = _build(type_str)
        M = incidence_matrix(ss)
        if not M or not M[0]:
            return
        m = len(M[0])
        for j in range(m):
            assert sum(M[i][j] for i in range(len(M))) == 0
