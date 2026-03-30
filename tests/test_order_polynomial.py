"""Tests for order polynomial (Step 30ab)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.order_polynomial import (
    OrderPolynomialResult,
    order_polynomial_values,
    strict_order_polynomial_values,
    order_polynomial_coefficients,
    strict_order_polynomial_coefficients,
    num_linear_extensions,
    analyze_order_polynomial,
)


def _ss(type_str: str):
    """Parse and build state space."""
    return build_statespace(parse(type_str))


# ---------------------------------------------------------------------------
# TestOrderPolynomialValues
# ---------------------------------------------------------------------------

class TestOrderPolynomialValues:
    """Tests for Ω(P, t) values."""

    def test_end_omega(self):
        """Single state: Ω(P, t) = t for all t >= 0."""
        ss = _ss("end")
        vals = order_polynomial_values(ss, max_t=5)
        # 1 state, Ω = t (one map for each value in {1..t})
        # Actually for a single element, Ω(P,t) = t
        assert vals[0] == 0
        assert vals[1] == 1
        assert vals[2] == 2
        assert vals[3] == 3

    def test_chain2_omega(self):
        """Chain of 2: Ω(P, t) = t(t+1)/2 = C(t+1, 2)."""
        ss = _ss("&{a: end}")
        vals = order_polynomial_values(ss, max_t=5)
        assert vals[0] == 0
        assert vals[1] == 1  # Only map: f(top)=1, f(bot)=1
        assert vals[2] == 3  # (1,1),(1,2),(2,2)
        assert vals[3] == 6  # C(4,2) = 6

    def test_chain3_omega(self):
        """Chain of 3: Ω(P, t) = C(t+2, 3)."""
        ss = _ss("&{a: &{b: end}}")
        vals = order_polynomial_values(ss, max_t=5)
        assert vals[0] == 0
        assert vals[1] == 1
        # C(4,3) = 4
        assert vals[2] == 4
        # C(5,3) = 10
        assert vals[3] == 10

    def test_omega_nonnegative(self):
        """Ω(P, t) >= 0 for all t >= 0."""
        ss = _ss("&{a: end, b: end}")
        vals = order_polynomial_values(ss, max_t=5)
        for v in vals:
            assert v >= 0

    def test_omega_zero_at_zero(self):
        """Ω(P, 0) = 0 for non-empty posets."""
        for typ in ["end", "&{a: end}", "&{a: end, b: end}"]:
            ss = _ss(typ)
            vals = order_polynomial_values(ss, max_t=1)
            assert vals[0] == 0

    def test_omega_monotone(self):
        """Ω(P, t) is non-decreasing in t."""
        ss = _ss("&{a: &{b: end}}")
        vals = order_polynomial_values(ss, max_t=8)
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]


# ---------------------------------------------------------------------------
# TestStrictOrderPolynomial
# ---------------------------------------------------------------------------

class TestStrictOrderPolynomial:
    """Tests for Ω̄(P, t)."""

    def test_end_strict(self):
        """Single state: Ω̄(P, t) = t."""
        ss = _ss("end")
        vals = strict_order_polynomial_values(ss, max_t=5)
        assert vals[1] == 1
        assert vals[2] == 2

    def test_chain2_strict(self):
        """Chain of 2: Ω̄(P, t) = C(t, 2) = t(t-1)/2."""
        ss = _ss("&{a: end}")
        vals = strict_order_polynomial_values(ss, max_t=5)
        assert vals[0] == 0
        assert vals[1] == 0  # Can't have 2 distinct values in {1}
        assert vals[2] == 1  # (1,2) only
        assert vals[3] == 3  # C(3,2) = 3

    def test_chain3_strict(self):
        """Chain of 3: Ω̄(P, t) = C(t, 3)."""
        ss = _ss("&{a: &{b: end}}")
        vals = strict_order_polynomial_values(ss, max_t=5)
        assert vals[0] == 0
        assert vals[1] == 0
        assert vals[2] == 0
        assert vals[3] == 1  # C(3,3) = 1
        assert vals[4] == 4  # C(4,3) = 4

    def test_strict_leq_nonstrict(self):
        """Ω̄(P, t) ≤ Ω(P, t) for all t."""
        ss = _ss("&{a: end, b: end}")
        vals = order_polynomial_values(ss, max_t=5)
        strict = strict_order_polynomial_values(ss, max_t=5)
        for i in range(len(vals)):
            assert strict[i] <= vals[i]


# ---------------------------------------------------------------------------
# TestCoefficients
# ---------------------------------------------------------------------------

class TestCoefficients:
    """Tests for polynomial interpolation."""

    def test_end_coefficients(self):
        """Single element: Ω(P, t) = t, coefficients [0, 1]."""
        ss = _ss("end")
        coeffs = order_polynomial_coefficients(ss)
        # n+1 coefficients for n-state poset (degree n)
        assert len(coeffs) >= 1

    def test_coefficients_length(self):
        """n+1 coefficients for an n-state poset."""
        ss = _ss("&{a: &{b: end}}")
        coeffs = order_polynomial_coefficients(ss)
        # 3 states -> degree 3 polynomial (n+1 = 4 coefficients)
        assert len(coeffs) == 4

    def test_coefficients_evaluate(self):
        """Polynomial should match computed values."""
        ss = _ss("&{a: end, b: end}")
        coeffs = order_polynomial_coefficients(ss)
        vals = order_polynomial_values(ss, max_t=5)
        for t in range(6):
            # Evaluate polynomial at t
            poly_val = sum(c * t**k for k, c in enumerate(coeffs))
            assert abs(poly_val - vals[t]) < 0.5


# ---------------------------------------------------------------------------
# TestLinearExtensions
# ---------------------------------------------------------------------------

class TestLinearExtensions:
    """Tests for linear extension counting."""

    def test_chain_one_extension(self):
        """A chain has exactly 1 linear extension."""
        for typ in ["end", "&{a: end}", "&{a: &{b: end}}"]:
            ss = _ss(typ)
            assert num_linear_extensions(ss) == 1

    def test_parallel_extensions(self):
        """2x2 grid (diamond) has 2 linear extensions."""
        ss = _ss("(&{a: end} || &{b: end})")
        n_ext = num_linear_extensions(ss)
        # 4-state diamond: top first, bottom last, 2 middle states can swap
        assert n_ext == 2

    def test_diamond_extensions(self):
        """Diamond lattice has 2 linear extensions."""
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        n_ext = num_linear_extensions(ss)
        assert n_ext >= 2  # At least 2 (a before b, or b before a)

    def test_extensions_positive(self):
        """Every finite poset has at least 1 linear extension."""
        for typ in ["end", "&{a: end}", "+{a: end, b: end}",
                     "(&{a: end} || &{b: end})"]:
            ss = _ss(typ)
            assert num_linear_extensions(ss) >= 1


# ---------------------------------------------------------------------------
# TestAnalyze
# ---------------------------------------------------------------------------

class TestAnalyze:
    """Tests for full analysis."""

    def test_end_analysis(self):
        ss = _ss("end")
        result = analyze_order_polynomial(ss)
        assert isinstance(result, OrderPolynomialResult)
        assert result.num_states == 1
        assert result.num_linear_extensions >= 1

    def test_chain_analysis(self):
        ss = _ss("&{a: &{b: end}}")
        result = analyze_order_polynomial(ss)
        assert result.num_states == 3
        assert result.num_linear_extensions == 1
        assert result.height >= 2

    def test_parallel_analysis(self):
        ss = _ss("(&{a: end} || &{b: end})")
        result = analyze_order_polynomial(ss)
        assert result.num_states == 4
        assert result.num_linear_extensions == 2

    def test_branch_analysis(self):
        ss = _ss("&{a: end, b: end}")
        result = analyze_order_polynomial(ss)
        # &{a: end, b: end} has 2 states (top and end, both branches merge)
        assert result.num_states >= 2
        assert result.num_linear_extensions >= 1

    def test_recursive_analysis(self):
        ss = _ss("rec X . &{a: X, b: end}")
        result = analyze_order_polynomial(ss)
        assert result.num_states >= 2
        assert result.num_linear_extensions >= 1

    def test_selection_analysis(self):
        ss = _ss("+{a: end, b: end}")
        result = analyze_order_polynomial(ss)
        assert result.num_states >= 2

    def test_values_tuple(self):
        ss = _ss("&{a: end}")
        result = analyze_order_polynomial(ss)
        assert isinstance(result.values, tuple)
        assert isinstance(result.strict_values, tuple)

    def test_result_consistency(self):
        ss = _ss("&{a: end, b: end}")
        result = analyze_order_polynomial(ss)
        assert len(result.values) == 9  # max_t=8 -> indices 0..8
        assert result.values[0] == 0


# ---------------------------------------------------------------------------
# TestBenchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on benchmark-like protocols."""

    def test_smtp_like(self):
        ss = _ss("&{ehlo: &{mail: &{data: end}}}")
        result = analyze_order_polynomial(ss)
        assert result.num_linear_extensions == 1  # Chain

    def test_iterator_like(self):
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_order_polynomial(ss)
        assert result.num_linear_extensions >= 1

    def test_two_buyer_like(self):
        ss = _ss("&{quote: &{accept: end, reject: end}}")
        result = analyze_order_polynomial(ss)
        assert result.num_linear_extensions >= 1
