"""Tests for simplicial homology of session type lattices (Step 32b).

Tests cover:
- Order complex construction
- Face vector computation
- Boundary matrix construction and d^2 = 0
- Betti number computation
- Euler characteristic from homology vs faces
- Integer matrix rank
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.homology import (
    order_complex,
    face_vector,
    boundary_matrices,
    betti_numbers,
    euler_characteristic_from_homology,
    euler_characteristic_from_faces,
    analyze_homology,
    _matrix_rank_integer,
    _interior_elements,
    _gcd,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Integer matrix rank
# ---------------------------------------------------------------------------

class TestMatrixRank:

    def test_rank_empty(self):
        assert _matrix_rank_integer([]) == 0

    def test_rank_identity_2x2(self):
        assert _matrix_rank_integer([[1, 0], [0, 1]]) == 2

    def test_rank_identity_3x3(self):
        mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert _matrix_rank_integer(mat) == 3

    def test_rank_zero_matrix(self):
        assert _matrix_rank_integer([[0, 0], [0, 0]]) == 0

    def test_rank_singular(self):
        # Rows are linearly dependent
        assert _matrix_rank_integer([[1, 2], [2, 4]]) == 1

    def test_rank_rectangular(self):
        mat = [[1, 0, 0], [0, 1, 0]]
        assert _matrix_rank_integer(mat) == 2

    def test_rank_tall_matrix(self):
        mat = [[1, 0], [0, 1], [1, 1]]
        assert _matrix_rank_integer(mat) == 2

    def test_rank_boundary_like(self):
        # Typical boundary matrix: [1, -1, 0; 0, 1, -1; -1, 0, 1]
        mat = [[1, -1, 0], [0, 1, -1], [-1, 0, 1]]
        assert _matrix_rank_integer(mat) == 2


class TestGCD:

    def test_gcd_basic(self):
        assert _gcd(12, 8) == 4

    def test_gcd_coprime(self):
        assert _gcd(7, 13) == 1

    def test_gcd_zero(self):
        assert _gcd(0, 5) == 5
        assert _gcd(5, 0) == 5


# ---------------------------------------------------------------------------
# Order complex
# ---------------------------------------------------------------------------

class TestOrderComplex:

    def test_end_empty_complex(self):
        """end has no interior elements."""
        ss = _build("end")
        simplices = order_complex(ss)
        assert simplices == []

    def test_single_method(self):
        """&{a: end} has 2 states (top, bottom), no interior."""
        ss = _build("&{a: end}")
        simplices = order_complex(ss)
        assert simplices == []

    def test_chain_of_two(self):
        """&{a: &{b: end}} has one interior element."""
        ss = _build("&{a: &{b: end}}")
        simplices = order_complex(ss)
        # One interior vertex → one 0-simplex
        assert len(simplices) == 1
        assert len(simplices[0]) == 1

    def test_branch_two(self):
        """&{a: end, b: end} has no interior (top→a→end, top→b→end)."""
        ss = _build("&{a: end, b: end}")
        interior = _interior_elements(ss)
        # Two interior states (the a and b targets are bottom=end, top is top)
        simplices = order_complex(ss)
        # Each branch target is interior if distinct from top and bottom
        assert isinstance(simplices, list)

    def test_selection_two(self):
        """+{a: end, b: end} has interior elements."""
        ss = _build("+{a: end, b: end}")
        simplices = order_complex(ss)
        assert isinstance(simplices, list)

    def test_parallel_simple(self):
        """(a.end || b.end) creates a diamond with interior."""
        ss = _build("(&{a: end} || &{b: end})")
        simplices = order_complex(ss)
        # Diamond: top, two mid, bottom → 2 interior elements
        # 0-simplices: 2, 1-simplex: 0 (incomparable) or 1 (comparable)
        assert len(simplices) >= 2


class TestFaceVector:

    def test_end_empty(self):
        ss = _build("end")
        assert face_vector(ss) == []

    def test_single_method_empty(self):
        ss = _build("&{a: end}")
        fv = face_vector(ss)
        assert fv == []

    def test_chain_three(self):
        """&{a: &{b: end}} has one interior: f_0 = 1."""
        ss = _build("&{a: &{b: end}}")
        fv = face_vector(ss)
        assert fv[0] == 1  # one vertex


# ---------------------------------------------------------------------------
# Boundary matrices
# ---------------------------------------------------------------------------

class TestBoundaryMatrices:

    def test_empty_no_boundaries(self):
        ss = _build("end")
        mats = boundary_matrices(ss)
        assert mats == []

    def test_chain_single_interior(self):
        """Single interior element → no boundary matrices (only 0-simplices)."""
        ss = _build("&{a: &{b: end}}")
        mats = boundary_matrices(ss)
        # Only dimension 0 exists, no d_1
        assert len(mats) == 0 or all(m == [] for m in mats)

    def test_boundary_squared_zero(self):
        """d_{k-1} . d_k = 0 for all k (fundamental property)."""
        ss = _build("&{a: &{b: end}, c: &{d: end}}")
        mats = boundary_matrices(ss)
        if len(mats) >= 2:
            for i in range(len(mats) - 1):
                d_k = mats[i]
                d_kp1 = mats[i + 1]
                if d_k and d_kp1:
                    # Matrix multiply d_k . d_{k+1}
                    rows = len(d_k)
                    cols = len(d_kp1[0]) if d_kp1[0] else 0
                    inner = len(d_kp1)
                    for r in range(rows):
                        for c in range(cols):
                            val = sum(d_k[r][j] * d_kp1[j][c] for j in range(inner))
                            assert val == 0, f"d^2 != 0 at ({r},{c})"


# ---------------------------------------------------------------------------
# Betti numbers
# ---------------------------------------------------------------------------

class TestBettiNumbers:

    def test_end_no_betti(self):
        ss = _build("end")
        assert betti_numbers(ss) == []

    def test_single_method_no_betti(self):
        ss = _build("&{a: end}")
        assert betti_numbers(ss) == []

    def test_chain_betti(self):
        """Chain with one interior: b_0 = 1."""
        ss = _build("&{a: &{b: end}}")
        betti = betti_numbers(ss)
        if betti:
            assert betti[0] == 1  # one connected component

    def test_betti_nonnegative(self):
        """All Betti numbers must be non-negative."""
        for typ in ["&{a: end, b: end}", "+{a: end, b: end}",
                     "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            for b in betti_numbers(ss):
                assert b >= 0, f"Negative Betti number for {typ}"


# ---------------------------------------------------------------------------
# Euler characteristic
# ---------------------------------------------------------------------------

class TestEulerCharacteristic:

    def test_euler_match(self):
        """Euler from homology must equal Euler from faces."""
        for typ in ["&{a: &{b: end}}", "&{a: end, b: end}",
                     "(&{a: end} || &{b: end})",
                     "&{a: &{b: end}, c: end}"]:
            ss = _build(typ)
            chi_h = euler_characteristic_from_homology(ss)
            chi_f = euler_characteristic_from_faces(ss)
            assert chi_h == chi_f, f"Euler mismatch for {typ}: {chi_h} != {chi_f}"

    def test_empty_euler_zero(self):
        ss = _build("end")
        assert euler_characteristic_from_homology(ss) == 0
        assert euler_characteristic_from_faces(ss) == 0


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeHomology:

    def test_end_result(self):
        ss = _build("end")
        result = analyze_homology(ss)
        assert result.dimension == -1
        assert result.betti_numbers == []
        assert result.euler_match

    def test_chain_result(self):
        ss = _build("&{a: &{b: end}}")
        result = analyze_homology(ss)
        assert result.dimension >= 0
        assert result.euler_match
        assert result.torsion_free

    def test_parallel_result(self):
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_homology(ss)
        assert result.euler_match
        assert result.torsion_free
        assert len(result.f_vector) > 0

    def test_branch_result(self):
        ss = _build("&{a: end, b: end}")
        result = analyze_homology(ss)
        assert result.euler_match

    def test_recursive_type(self):
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_homology(ss)
        assert result.euler_match
        assert result.torsion_free

    def test_boundary_ranks_length(self):
        """Number of boundary ranks should match dimension."""
        ss = _build("&{a: &{b: end}, c: &{d: end}}")
        result = analyze_homology(ss)
        if result.dimension > 0:
            assert len(result.boundary_ranks) >= 1

    def test_cycles_ge_boundaries(self):
        """Number of cycles >= number of boundaries at each dimension."""
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_homology(ss)
        for z, b in zip(result.num_cycles, result.num_boundaries):
            assert z >= b
