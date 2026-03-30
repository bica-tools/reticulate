"""Tests for tropical determinant analysis (Step 30m).

Tests cover:
- Tropical determinant computation (identity, zero, small examples)
- Optimal permutations (unique vs multiple)
- Tropical singularity detection
- Tropical adjugate and minor operations
- Tropical rank
- Hungarian algorithm for optimal assignment
- Tropical Cramer's rule
- Full analysis on session type state spaces
- Benchmark protocols (iterator, file, parallel)
"""

import math
import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.tropical import NEG_INF, INF, _maxplus_adjacency_matrix
from reticulate.tropical_determinant import (
    TropicalDetResult,
    tropical_det,
    tropical_det_from_ss,
    optimal_permutations,
    is_tropically_singular,
    tropical_adjugate,
    tropical_cramer,
    tropical_rank,
    tropical_minor,
    all_tropical_minors,
    hungarian_assignment,
    analyze_tropical_determinant,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# TestTropicalDet
# ---------------------------------------------------------------------------

class TestTropicalDet:
    """Tests for the core tropical_det function."""

    def test_empty_matrix(self):
        assert tropical_det([]) == 0.0

    def test_1x1(self):
        assert tropical_det([[5.0]]) == 5.0

    def test_1x1_neg_inf(self):
        assert tropical_det([[NEG_INF]]) == NEG_INF

    def test_2x2_identity(self):
        """Tropical identity: 0 on diagonal, -inf elsewhere."""
        I = [[0.0, NEG_INF], [NEG_INF, 0.0]]
        assert tropical_det(I) == 0.0

    def test_2x2_all_ones(self):
        """All entries 1 => tdet = max(1+1, 1+1) = 2."""
        A = [[1.0, 1.0], [1.0, 1.0]]
        assert tropical_det(A) == 2.0

    def test_2x2_known(self):
        A = [[3.0, 1.0], [2.0, 4.0]]
        # Perms: (0,1)->3+4=7, (1,0)->1+2=3 => max=7
        assert tropical_det(A) == 7.0

    def test_2x2_anti_diagonal(self):
        A = [[NEG_INF, 5.0], [3.0, NEG_INF]]
        # Only (1,0) perm is feasible: 5+3=8
        assert tropical_det(A) == 8.0

    def test_3x3_identity(self):
        I = [
            [0.0, NEG_INF, NEG_INF],
            [NEG_INF, 0.0, NEG_INF],
            [NEG_INF, NEG_INF, 0.0],
        ]
        assert tropical_det(I) == 0.0

    def test_3x3_example(self):
        A = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        # Best: pick 3 + 5 + 7 = 15? No — maximize:
        # (0,1,2): 1+5+9=15, (0,2,1): 1+6+8=15, (1,0,2): 2+4+9=15,
        # (1,2,0): 2+6+7=15, (2,0,1): 3+4+8=15, (2,1,0): 3+5+7=15
        # All equal 15!
        assert tropical_det(A) == 15.0

    def test_no_feasible_permutation(self):
        """Matrix with no feasible permutation."""
        A = [[1.0, NEG_INF], [NEG_INF, NEG_INF]]
        assert tropical_det(A) == NEG_INF

    def test_from_statespace_end(self):
        ss = _build("end")
        det = tropical_det_from_ss(ss)
        # 1 state, 1x1 matrix with -inf (no self-loop)
        # Actually _maxplus_adjacency_matrix has NEG_INF on diagonal
        assert isinstance(det, float)

    def test_from_statespace_branch(self):
        ss = _build("&{a: end, b: end}")
        det = tropical_det_from_ss(ss)
        assert isinstance(det, float)


# ---------------------------------------------------------------------------
# TestOptimalPermutations
# ---------------------------------------------------------------------------

class TestOptimalPermutations:
    """Tests for finding all optimal permutations."""

    def test_empty(self):
        opts = optimal_permutations([])
        assert opts == [()]

    def test_1x1(self):
        opts = optimal_permutations([[7.0]])
        assert opts == [(0,)]

    def test_identity_unique(self):
        I = [[0.0, NEG_INF], [NEG_INF, 0.0]]
        opts = optimal_permutations(I)
        assert len(opts) == 1
        assert opts[0] == (0, 1)

    def test_anti_diagonal_unique(self):
        A = [[NEG_INF, 5.0], [3.0, NEG_INF]]
        opts = optimal_permutations(A)
        assert len(opts) == 1
        assert opts[0] == (1, 0)

    def test_multiple_optimal(self):
        """All 1's: every permutation is optimal."""
        A = [[1.0, 1.0], [1.0, 1.0]]
        opts = optimal_permutations(A)
        assert len(opts) == 2

    def test_3x3_all_same_sum(self):
        """Arithmetic progression rows => all permutations have same sum."""
        A = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        opts = optimal_permutations(A)
        assert len(opts) == 6  # all 3! permutations

    def test_no_feasible(self):
        A = [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
        opts = optimal_permutations(A)
        assert len(opts) == 0

    def test_unique_optimal_3x3(self):
        A = [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
        # Identity perm gives 30, all others give at most 20
        opts = optimal_permutations(A)
        assert len(opts) == 1
        assert opts[0] == (0, 1, 2)


# ---------------------------------------------------------------------------
# TestSingularity
# ---------------------------------------------------------------------------

class TestSingularity:
    """Tests for tropical singularity detection."""

    def test_identity_not_singular(self):
        I = [[0.0, NEG_INF], [NEG_INF, 0.0]]
        assert not is_tropically_singular(I)

    def test_all_ones_singular(self):
        A = [[1.0, 1.0], [1.0, 1.0]]
        assert is_tropically_singular(A)

    def test_strong_diagonal_not_singular(self):
        A = [[10.0, 1.0], [1.0, 10.0]]
        assert not is_tropically_singular(A)

    def test_equal_diag_anti_diag_singular(self):
        A = [[5.0, 5.0], [5.0, 5.0]]
        assert is_tropically_singular(A)

    def test_no_feasible_singular(self):
        """No feasible permutation counts as singular (0 != 1 optimal)."""
        A = [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
        assert is_tropically_singular(A)

    def test_3x3_regular(self):
        A = [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
        assert not is_tropically_singular(A)

    def test_3x3_singular(self):
        A = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        assert is_tropically_singular(A)


# ---------------------------------------------------------------------------
# TestMinors
# ---------------------------------------------------------------------------

class TestMinors:
    """Tests for tropical minor operations."""

    def test_minor_2x2(self):
        A = [[3.0, 1.0], [2.0, 4.0]]
        # minor(0,0) = det([[4.0]]) = 4.0
        assert tropical_minor(A, 0, 0) == 4.0
        # minor(0,1) = det([[2.0]]) = 2.0
        assert tropical_minor(A, 0, 1) == 2.0
        # minor(1,0) = det([[1.0]]) = 1.0
        assert tropical_minor(A, 1, 0) == 1.0
        # minor(1,1) = det([[3.0]]) = 3.0
        assert tropical_minor(A, 1, 1) == 3.0

    def test_minor_3x3(self):
        A = [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
        # minor(0,0) removes row 0, col 0 => [[10,0],[0,10]] => det=20
        assert tropical_minor(A, 0, 0) == 20.0

    def test_all_minors_1x1(self):
        A = [[5.0]]
        m1 = all_tropical_minors(A, 1)
        assert len(m1) == 1
        assert m1[0] == 5.0

    def test_all_minors_2x2(self):
        A = [[3.0, 1.0], [2.0, 4.0]]
        # 1x1 minors: each element
        m1 = all_tropical_minors(A, 1)
        assert len(m1) == 4
        assert sorted(m1) == [1.0, 2.0, 3.0, 4.0]

    def test_all_minors_invalid_k(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        assert all_tropical_minors(A, 0) == []
        assert all_tropical_minors(A, 3) == []

    def test_all_minors_full(self):
        A = [[3.0, 1.0], [2.0, 4.0]]
        m2 = all_tropical_minors(A, 2)
        # Only one 2x2 submatrix: the full matrix
        assert len(m2) == 1
        assert m2[0] == tropical_det(A)


# ---------------------------------------------------------------------------
# TestAdjugate
# ---------------------------------------------------------------------------

class TestAdjugate:
    """Tests for tropical adjugate matrix."""

    def test_empty(self):
        assert tropical_adjugate([]) == []

    def test_1x1(self):
        adj = tropical_adjugate([[7.0]])
        assert adj == [[0.0]]

    def test_2x2(self):
        A = [[3.0, 1.0], [2.0, 4.0]]
        adj = tropical_adjugate(A)
        # adj[0][0] = minor(0,0) = 4.0
        # adj[0][1] = minor(1,0) = 1.0
        # adj[1][0] = minor(0,1) = 2.0
        # adj[1][1] = minor(1,1) = 3.0
        assert adj[0][0] == 4.0
        assert adj[0][1] == 1.0
        assert adj[1][0] == 2.0
        assert adj[1][1] == 3.0

    def test_adjugate_identity(self):
        I = [[0.0, NEG_INF], [NEG_INF, 0.0]]
        adj = tropical_adjugate(I)
        # minor(0,0)=0, minor(1,0)=NEG_INF, minor(0,1)=NEG_INF, minor(1,1)=0
        assert adj[0][0] == 0.0
        assert adj[1][1] == 0.0


# ---------------------------------------------------------------------------
# TestTropicalRank
# ---------------------------------------------------------------------------

class TestTropicalRank:
    """Tests for tropical rank computation."""

    def test_empty(self):
        assert tropical_rank([]) == 0

    def test_1x1_finite(self):
        assert tropical_rank([[5.0]]) == 1

    def test_1x1_neg_inf(self):
        assert tropical_rank([[NEG_INF]]) == 0

    def test_2x2_full_rank(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        assert tropical_rank(A) == 2

    def test_2x2_rank_1(self):
        """Only 1x1 submatrices are finite."""
        A = [[1.0, NEG_INF], [NEG_INF, NEG_INF]]
        # 2x2 det requires a feasible perm: (0,0)&(1,1) or (0,1)&(1,0)
        # (0,0)&(1,1): 1 + NEG_INF = NEG_INF
        # (0,1)&(1,0): NEG_INF + NEG_INF = NEG_INF
        # So rank < 2. 1x1 submatrix [[1.0]] is finite, so rank = 1.
        assert tropical_rank(A) == 1

    def test_2x2_rank_0(self):
        A = [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
        assert tropical_rank(A) == 0

    def test_identity_full_rank(self):
        I = [
            [0.0, NEG_INF, NEG_INF],
            [NEG_INF, 0.0, NEG_INF],
            [NEG_INF, NEG_INF, 0.0],
        ]
        assert tropical_rank(I) == 3


# ---------------------------------------------------------------------------
# TestHungarian
# ---------------------------------------------------------------------------

class TestHungarian:
    """Tests for the Hungarian algorithm."""

    def test_empty(self):
        w, a = hungarian_assignment([])
        assert w == 0.0
        assert a == []

    def test_1x1(self):
        w, a = hungarian_assignment([[7.0]])
        assert w == 7.0
        assert a == [0]

    def test_2x2_identity(self):
        I = [[0.0, NEG_INF], [NEG_INF, 0.0]]
        w, a = hungarian_assignment(I)
        assert w == 0.0
        assert a == [0, 1]

    def test_2x2_known(self):
        A = [[3.0, 1.0], [2.0, 4.0]]
        w, a = hungarian_assignment(A)
        assert w == 7.0  # 3+4 or equivalently max assignment
        assert a[0] == 0  # row 0 -> col 0
        assert a[1] == 1  # row 1 -> col 1

    def test_2x2_anti_diagonal(self):
        A = [[NEG_INF, 5.0], [3.0, NEG_INF]]
        w, a = hungarian_assignment(A)
        assert w == 8.0
        assert a == [1, 0]

    def test_3x3_strong_diagonal(self):
        A = [
            [10.0, 1.0, 1.0],
            [1.0, 10.0, 1.0],
            [1.0, 1.0, 10.0],
        ]
        w, a = hungarian_assignment(A)
        assert w == 30.0
        assert a == [0, 1, 2]

    def test_infeasible(self):
        A = [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
        w, a = hungarian_assignment(A)
        assert w == NEG_INF

    def test_agrees_with_brute_force(self):
        """Hungarian should agree with brute-force on a small matrix."""
        A = [
            [2.0, 5.0, 3.0],
            [4.0, 1.0, 6.0],
            [7.0, 3.0, 2.0],
        ]
        w_hungarian, _ = hungarian_assignment(A)
        w_det = tropical_det(A)
        assert math.isclose(w_hungarian, w_det, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# TestCramer
# ---------------------------------------------------------------------------

class TestCramer:
    """Tests for tropical Cramer's rule."""

    def test_empty(self):
        x = tropical_cramer([], [])
        assert x == []

    def test_dimension_mismatch(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        assert tropical_cramer(A, [1.0]) is None

    def test_infeasible(self):
        A = [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
        assert tropical_cramer(A, [1.0, 2.0]) is None

    def test_identity_system(self):
        """A = tropical identity, b = [a, b] => x = b."""
        I = [[0.0, NEG_INF], [NEG_INF, 0.0]]
        b = [3.0, 5.0]
        x = tropical_cramer(I, b)
        assert x is not None
        assert len(x) == 2
        # For identity: adj is also identity-like, so x[j] = b[j]
        assert math.isclose(x[0], 3.0, abs_tol=1e-12)
        assert math.isclose(x[1], 5.0, abs_tol=1e-12)

    def test_returns_list(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        b = [5.0, 6.0]
        x = tropical_cramer(A, b)
        assert x is not None
        assert len(x) == 2


# ---------------------------------------------------------------------------
# TestAnalyze — full analysis on session type state spaces
# ---------------------------------------------------------------------------

class TestAnalyze:
    """Tests for full tropical determinant analysis."""

    def test_end(self):
        ss = _build("end")
        result = analyze_tropical_determinant(ss)
        assert isinstance(result, TropicalDetResult)
        assert isinstance(result.determinant, float)
        assert isinstance(result.tropical_rank, int)

    def test_simple_branch(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_tropical_determinant(ss)
        assert result.tropical_rank >= 0
        assert result.assignment_weight == result.determinant

    def test_selection(self):
        ss = _build("+{x: end, y: end}")
        result = analyze_tropical_determinant(ss)
        assert isinstance(result.is_singular, bool)
        assert result.num_optimal == len(result.optimal_permutations)

    def test_nested_branch(self):
        ss = _build("&{a: &{c: end, d: end}, b: end}")
        result = analyze_tropical_determinant(ss)
        assert result.tropical_rank >= 1
        assert isinstance(result.adjugate, list)

    def test_parallel(self):
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_tropical_determinant(ss)
        assert isinstance(result.determinant, float)

    def test_result_consistency(self):
        """Check that assignment_weight == determinant always."""
        for ty in ["end", "&{a: end}", "+{x: end, y: end}"]:
            ss = _build(ty)
            r = analyze_tropical_determinant(ss)
            assert r.assignment_weight == r.determinant

    def test_optimal_perms_count_matches(self):
        ss = _build("&{a: end, b: end}")
        r = analyze_tropical_determinant(ss)
        assert r.num_optimal == len(r.optimal_permutations)


# ---------------------------------------------------------------------------
# TestBenchmarks — protocols from the benchmark suite
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on benchmark protocol types."""

    def test_iterator(self):
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_tropical_determinant(ss)
        assert isinstance(result.determinant, float)
        assert result.tropical_rank >= 1

    def test_file_object(self):
        ss = _build(
            "&{open: +{OK: &{read: end, close: end}, ERROR: end}}"
        )
        result = analyze_tropical_determinant(ss)
        assert isinstance(result.determinant, float)

    def test_simple_parallel(self):
        ss = _build("(&{a: end} || &{b: end})")
        r = analyze_tropical_determinant(ss)
        assert r.tropical_rank >= 1

    def test_deep_branch(self):
        ss = _build("&{a: &{b: &{c: end}}}")
        r = analyze_tropical_determinant(ss)
        assert r.tropical_rank >= 1
        # 4 states in a chain => adjacency has rank properties
        A = _maxplus_adjacency_matrix(ss)
        assert len(A) == 4

    def test_wide_branch(self):
        ss = _build("&{a: end, b: end, c: end, d: end}")
        r = analyze_tropical_determinant(ss)
        assert isinstance(r.adjugate, list)
        assert len(r.adjugate) == len(_maxplus_adjacency_matrix(ss))

    def test_selection_branch_mix(self):
        ss = _build("+{x: &{a: end, b: end}, y: end}")
        r = analyze_tropical_determinant(ss)
        assert isinstance(r.is_singular, bool)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and mathematical properties."""

    def test_det_equals_hungarian_small(self):
        """tropical_det and hungarian_assignment agree on small matrices."""
        matrices = [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, NEG_INF], [NEG_INF, 0.0]],
            [[5.0, 5.0], [5.0, 5.0]],
        ]
        for A in matrices:
            det = tropical_det(A)
            w, _ = hungarian_assignment(A)
            if det != NEG_INF:
                assert math.isclose(det, w, abs_tol=1e-12), f"Mismatch on {A}"

    def test_rank_leq_n(self):
        """Tropical rank never exceeds matrix size."""
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        r = tropical_rank(A)
        assert 0 <= r <= 3

    def test_adjugate_size(self):
        """Adjugate has same dimensions as input."""
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        adj = tropical_adjugate(A)
        assert len(adj) == 3
        assert all(len(row) == 3 for row in adj)

    def test_minor_reduces_size(self):
        """Minor of n×n is value (scalar), not matrix."""
        A = [[1.0, 2.0], [3.0, 4.0]]
        m = tropical_minor(A, 0, 0)
        assert isinstance(m, float)

    def test_singular_iff_multiple_optimal(self):
        """Singularity <=> more than one optimal permutation."""
        cases = [
            ([[0.0, NEG_INF], [NEG_INF, 0.0]], False),
            ([[1.0, 1.0], [1.0, 1.0]], True),
        ]
        for A, expected in cases:
            assert is_tropically_singular(A) == expected
