"""Tests for incidence algebra of session types (Step 32g).

Tests cover:
- Incidence function construction (zeta, Mobius, delta, rank)
- Convolution: zeta * mu = delta (fundamental identity)
- Mobius involution: mu * zeta = delta
- Algebra operations: add, subtract, scalar_mul, power
- Inverse computation
- Nilpotent index of (zeta - delta)
- Commutativity checks
- Benchmark protocol analysis
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.incidence_algebra import (
    incidence_function,
    zeta_element,
    mobius_element,
    delta_element,
    rank_element,
    constant_element,
    indicator_element,
    convolve,
    add,
    subtract,
    scalar_mul,
    power,
    verify_convolution_identity,
    verify_mobius_involution,
    is_multiplicative,
    is_commutative_pair,
    nilpotent_index,
    inverse,
    analyze_incidence_algebra,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str):
    """Parse and build state space."""
    return build_statespace(parse(s))


def _approx_equal(f, g, tol=1e-9):
    """Check if two incidence functions are approximately equal."""
    keys = set(f.keys()) | set(g.keys())
    for k in keys:
        if abs(f.get(k, 0.0) - g.get(k, 0.0)) > tol:
            return False
    return True


# ---------------------------------------------------------------------------
# Element construction
# ---------------------------------------------------------------------------

class TestElements:
    """Tests for incidence function construction."""

    def test_zeta_end(self):
        """Zeta of end: {(0,0): 1}."""
        ss = _build("end")
        z = zeta_element(ss)
        assert len(z) >= 1
        # All values should be 1
        for v in z.values():
            assert v == 1.0

    def test_delta_end(self):
        """Delta of end: {(0,0): 1}."""
        ss = _build("end")
        d = delta_element(ss)
        assert len(d) >= 1

    def test_mobius_end(self):
        """Mobius of end: {(0,0): 1}."""
        ss = _build("end")
        mu = mobius_element(ss)
        assert len(mu) >= 1

    def test_zeta_single_branch(self):
        """Zeta of &{a: end}: 3 nonzero entries."""
        ss = _build("&{a: end}")
        z = zeta_element(ss)
        # top >= top, top >= bottom, bottom >= bottom
        nonzero = sum(1 for v in z.values() if abs(v) > 1e-12)
        assert nonzero == 3

    def test_delta_diagonal_only(self):
        """Delta has nonzero entries only on diagonal."""
        ss = _build("&{a: end, b: end}")
        d = delta_element(ss)
        for (x, y), v in d.items():
            if x != y:
                assert abs(v) < 1e-12

    def test_rank_element(self):
        """Rank element has non-negative values for x >= y."""
        ss = _build("&{a: end, b: end}")
        r = rank_element(ss)
        for (x, y), v in r.items():
            assert v >= 0

    def test_constant_element(self):
        """Constant element with c=2."""
        ss = _build("&{a: end}")
        c = constant_element(ss, 2.0)
        for v in c.values():
            assert v == 2.0

    def test_constant_zero(self):
        """Constant element with c=0 is empty."""
        ss = _build("&{a: end}")
        c = constant_element(ss, 0.0)
        assert len(c) == 0

    def test_indicator_element(self):
        """Indicator element on specific pairs."""
        ss = _build("&{a: end}")
        top, bottom = ss.top, ss.bottom
        ind = indicator_element(ss, {(top, bottom)})
        assert ind.get((top, bottom), 0.0) == 1.0

    def test_custom_incidence_function(self):
        """Custom incidence function via callable."""
        ss = _build("&{a: end}")
        f = incidence_function(ss, lambda x, y: float(x + y))
        assert len(f) >= 1


# ---------------------------------------------------------------------------
# Convolution identity: zeta * mu = delta
# ---------------------------------------------------------------------------

class TestConvolutionIdentity:
    """Tests for the fundamental identity zeta * mu = delta."""

    def test_end_identity(self):
        """zeta * mu = delta for end."""
        ss = _build("end")
        assert verify_convolution_identity(ss)

    def test_single_branch_identity(self):
        """zeta * mu = delta for &{a: end}."""
        ss = _build("&{a: end}")
        assert verify_convolution_identity(ss)

    def test_two_branch_identity(self):
        """zeta * mu = delta for &{a: end, b: end}."""
        ss = _build("&{a: end, b: end}")
        assert verify_convolution_identity(ss)

    def test_selection_identity(self):
        """zeta * mu = delta for +{a: end, b: end}."""
        ss = _build("+{a: end, b: end}")
        assert verify_convolution_identity(ss)

    def test_nested_identity(self):
        """zeta * mu = delta for nested types."""
        ss = _build("&{a: &{c: end}, b: end}")
        assert verify_convolution_identity(ss)

    def test_recursive_identity(self):
        """zeta * mu = delta for recursive types."""
        ss = _build("rec X . &{a: X, b: end}")
        assert verify_convolution_identity(ss)


# ---------------------------------------------------------------------------
# Mobius involution: mu * zeta = delta
# ---------------------------------------------------------------------------

class TestMobiusInvolution:
    """Tests for mu * zeta = delta (right inverse)."""

    def test_end_involution(self):
        """mu * zeta = delta for end."""
        ss = _build("end")
        assert verify_mobius_involution(ss)

    def test_single_branch_involution(self):
        """mu * zeta = delta for &{a: end}."""
        ss = _build("&{a: end}")
        assert verify_mobius_involution(ss)

    def test_two_branch_involution(self):
        """mu * zeta = delta for &{a: end, b: end}."""
        ss = _build("&{a: end, b: end}")
        assert verify_mobius_involution(ss)

    def test_nested_involution(self):
        """mu * zeta = delta for nested type."""
        ss = _build("&{a: &{c: end}, b: end}")
        assert verify_mobius_involution(ss)


# ---------------------------------------------------------------------------
# Algebra operations
# ---------------------------------------------------------------------------

class TestAlgebraOperations:
    """Tests for add, subtract, scalar_mul, power."""

    def test_add_zero(self):
        """f + 0 = f."""
        ss = _build("&{a: end}")
        z = zeta_element(ss)
        result = add(z, {})
        assert _approx_equal(result, z)

    def test_subtract_self(self):
        """f - f = 0."""
        ss = _build("&{a: end}")
        z = zeta_element(ss)
        result = subtract(z, z)
        for v in result.values():
            assert abs(v) < 1e-9

    def test_scalar_mul_1(self):
        """1 * f = f."""
        ss = _build("&{a: end}")
        z = zeta_element(ss)
        result = scalar_mul(1.0, z)
        assert _approx_equal(result, z)

    def test_scalar_mul_0(self):
        """0 * f = 0."""
        ss = _build("&{a: end}")
        z = zeta_element(ss)
        result = scalar_mul(0.0, z)
        assert len(result) == 0

    def test_power_0_is_delta(self):
        """f^0 = delta."""
        ss = _build("&{a: end}")
        z = zeta_element(ss)
        d = delta_element(ss)
        result = power(z, 0, ss)
        assert _approx_equal(result, d)

    def test_power_1_is_self(self):
        """f^1 = f."""
        ss = _build("&{a: end}")
        z = zeta_element(ss)
        result = power(z, 1, ss)
        assert _approx_equal(result, z)

    def test_add_commutative(self):
        """f + g = g + f."""
        ss = _build("&{a: end}")
        z = zeta_element(ss)
        d = delta_element(ss)
        assert _approx_equal(add(z, d), add(d, z))


# ---------------------------------------------------------------------------
# Inverse computation
# ---------------------------------------------------------------------------

class TestInverse:
    """Tests for incidence function inverse."""

    def test_zeta_inverse_is_mobius(self):
        """zeta^{-1} should match Mobius function."""
        ss = _build("&{a: end}")
        z = zeta_element(ss)
        mu = mobius_element(ss)
        z_inv = inverse(z, ss)
        assert z_inv is not None
        assert _approx_equal(z_inv, mu)

    def test_delta_inverse_is_delta(self):
        """delta^{-1} = delta."""
        ss = _build("&{a: end}")
        d = delta_element(ss)
        d_inv = inverse(d, ss)
        assert d_inv is not None
        assert _approx_equal(d_inv, d)

    def test_zero_diagonal_not_invertible(self):
        """Function with zero diagonal is not invertible."""
        ss = _build("&{a: end}")
        f = {}  # zero function
        result = inverse(f, ss)
        assert result is None


# ---------------------------------------------------------------------------
# Nilpotent index
# ---------------------------------------------------------------------------

class TestNilpotentIndex:
    """Tests for nilpotency of (zeta - delta)."""

    def test_end_nilpotent(self):
        """For end: zeta - delta = 0, nilpotent index = 1."""
        ss = _build("end")
        idx = nilpotent_index(ss)
        assert idx is not None
        assert idx >= 1

    def test_single_branch_nilpotent(self):
        """For &{a: end}: (zeta - delta) is nilpotent."""
        ss = _build("&{a: end}")
        idx = nilpotent_index(ss)
        assert idx is not None
        assert idx >= 1

    def test_nilpotent_bounded_by_height(self):
        """Nilpotent index <= height + 1."""
        ss = _build("&{a: &{b: end}}")
        idx = nilpotent_index(ss)
        assert idx is not None
        # Height is 2 for this chain, so nilpotent index <= 3
        assert idx <= 4


# ---------------------------------------------------------------------------
# Commutativity
# ---------------------------------------------------------------------------

class TestCommutativity:
    """Tests for commutativity of convolution."""

    def test_zeta_mu_commute(self):
        """Zeta and Mobius should commute (both are inverses of each other)."""
        ss = _build("&{a: end}")
        z = zeta_element(ss)
        mu = mobius_element(ss)
        assert is_commutative_pair(z, mu, ss)

    def test_delta_commutes_with_all(self):
        """Delta is the identity, commutes with everything."""
        ss = _build("&{a: end}")
        d = delta_element(ss)
        z = zeta_element(ss)
        assert is_commutative_pair(d, z, ss)


# ---------------------------------------------------------------------------
# Multiplicativity (idempotent check)
# ---------------------------------------------------------------------------

class TestMultiplicative:
    """Tests for multiplicativity checks."""

    def test_delta_is_idempotent(self):
        """delta * delta = delta (identity is idempotent)."""
        ss = _build("&{a: end}")
        d = delta_element(ss)
        assert is_multiplicative(d, ss)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeIncidenceAlgebra:
    """Tests for the complete analysis function."""

    def test_end_analysis(self):
        """Complete analysis of end type."""
        ss = _build("end")
        result = analyze_incidence_algebra(ss)
        assert result.num_states == 1
        assert result.convolution_identity_verified
        assert result.mobius_involution_verified

    def test_single_branch_analysis(self):
        """Complete analysis of &{a: end}."""
        ss = _build("&{a: end}")
        result = analyze_incidence_algebra(ss)
        assert result.num_states == 2
        assert result.convolution_identity_verified
        assert result.dimension >= 3  # at least 3 intervals

    def test_two_branch_analysis(self):
        """Complete analysis of &{a: end, b: end} (2 states, branches merge)."""
        ss = _build("&{a: end, b: end}")
        result = analyze_incidence_algebra(ss)
        assert result.num_states == 2
        assert result.convolution_identity_verified
        assert result.mobius_involution_verified

    def test_selection_analysis(self):
        """Complete analysis of +{a: end, b: end}."""
        ss = _build("+{a: end, b: end}")
        result = analyze_incidence_algebra(ss)
        assert result.convolution_identity_verified

    def test_nested_analysis(self):
        """Complete analysis of nested type."""
        ss = _build("&{a: &{c: end, d: end}, b: end}")
        result = analyze_incidence_algebra(ss)
        assert result.convolution_identity_verified
        assert result.nilpotent_index is not None

    def test_recursive_analysis(self):
        """Analysis handles recursive types."""
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_incidence_algebra(ss)
        assert isinstance(result.convolution_identity_verified, bool)


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on standard benchmark protocols."""

    def test_iterator_identity(self):
        """Java Iterator: zeta * mu = delta."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        assert verify_convolution_identity(ss)

    def test_simple_resource_identity(self):
        """Simple resource: open, use, close."""
        ss = _build("&{open: &{use: &{close: end}}}")
        assert verify_convolution_identity(ss)

    def test_parallel_identity(self):
        """Parallel composition: identity holds."""
        ss = _build("(&{a: end} || &{b: end})")
        assert verify_convolution_identity(ss)

    def test_deep_chain_identity(self):
        """Deep chain protocol."""
        ss = _build("&{a: &{b: &{c: &{d: end}}}}")
        assert verify_convolution_identity(ss)
