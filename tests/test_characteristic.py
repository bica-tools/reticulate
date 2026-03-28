"""Tests for Rota characteristic polynomial (Step 30c).

Tests cover:
- Polynomial computation and evaluation
- Whitney numbers (first and second kind)
- Log-concavity
- Factorization under parallel
- Polynomial arithmetic
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.characteristic import (
    characteristic_polynomial,
    whitney_numbers_first,
    whitney_numbers_second,
    check_log_concave,
    is_whitney_log_concave,
    verify_factorization,
    analyze_characteristic,
    poly_evaluate,
    poly_multiply,
    poly_derivative,
    poly_to_string,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Polynomial arithmetic
# ---------------------------------------------------------------------------

class TestPolyArithmetic:

    def test_evaluate_constant(self):
        assert poly_evaluate([5], 3) == 5

    def test_evaluate_linear(self):
        assert poly_evaluate([2, 1], 3) == 7  # 2t + 1 at t=3

    def test_evaluate_quadratic(self):
        assert poly_evaluate([1, 0, -1], 2) == 3  # t² - 1 at t=2

    def test_multiply_constants(self):
        assert poly_multiply([3], [4]) == [12]

    def test_multiply_linear(self):
        # (t - 1)(t + 1) = t² - 1
        assert poly_multiply([1, -1], [1, 1]) == [1, 0, -1]

    def test_multiply_quadratic(self):
        # (t - 1)² = t² - 2t + 1
        assert poly_multiply([1, -1], [1, -1]) == [1, -2, 1]

    def test_derivative_constant(self):
        assert poly_derivative([5]) == [0]

    def test_derivative_linear(self):
        assert poly_derivative([3, 2]) == [3]  # d/dt(3t + 2) = 3

    def test_derivative_quadratic(self):
        assert poly_derivative([1, -2, 1]) == [2, -2]  # d/dt(t² - 2t + 1) = 2t - 2

    def test_to_string(self):
        assert poly_to_string([1, -1]) == "t - 1"
        assert poly_to_string([1, 0, -1]) == "t^2 - 1"
        assert poly_to_string([0]) == "0"


# ---------------------------------------------------------------------------
# Characteristic polynomial tests
# ---------------------------------------------------------------------------

class TestCharacteristicPoly:

    def test_end(self):
        """end: single state, χ(t) = 1."""
        ss = _build("end")
        coeffs = characteristic_polynomial(ss)
        assert coeffs == [1]

    def test_single_branch(self):
        """&{a: end}: chain of 2, χ(t) = t - 1."""
        ss = _build("&{a: end}")
        coeffs = characteristic_polynomial(ss)
        assert coeffs == [1, -1]

    def test_chain_length_2(self):
        """&{a: &{b: end}}: chain of 3, χ(t) = t² - t."""
        ss = _build("&{a: &{b: end}}")
        coeffs = characteristic_polynomial(ss)
        # μ(⊤,⊤)=1 at rank 2 → t², μ(⊤,mid)=-1 at rank 1 → -t, μ(⊤,⊥)=0 at rank 0
        assert poly_evaluate(coeffs, 0) == 0  # χ(0) = μ(⊤,⊥) = 0

    def test_eval_at_0(self):
        """χ(0) = μ(⊤,⊥) always."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            coeffs = characteristic_polynomial(ss)
            from reticulate.mobius import mobius_value
            assert poly_evaluate(coeffs, 0) == mobius_value(ss)

    def test_leading_coefficient(self):
        """Leading coefficient is always 1 (monic polynomial)."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            coeffs = characteristic_polynomial(ss)
            assert coeffs[0] == 1

    def test_parallel_polynomial(self):
        """Parallel: χ should factor."""
        ss = _build("(&{a: end} || &{b: end})")
        coeffs = characteristic_polynomial(ss)
        # Product of two (t-1): (t-1)² = t² - 2t + 1
        assert coeffs == [1, -2, 1]


# ---------------------------------------------------------------------------
# Whitney number tests
# ---------------------------------------------------------------------------

class TestWhitneyNumbers:

    def test_chain_whitney_first(self):
        """Chain: Whitney first kind from Möbius values."""
        ss = _build("&{a: end}")
        w = whitney_numbers_first(ss)
        # corank 0 (top): μ(⊤,⊤) = 1, corank 1 (bottom): μ(⊤,⊥) = -1
        assert w.get(0, 0) == 1
        assert w.get(1, 0) == -1

    def test_chain_whitney_second(self):
        """Chain: Whitney second kind = 1 per corank level."""
        ss = _build("&{a: &{b: end}}")
        W = whitney_numbers_second(ss)
        assert W.get(0, 0) == 1  # top
        assert W.get(1, 0) == 1  # middle
        assert W.get(2, 0) == 1  # bottom

    def test_parallel_whitney_second(self):
        """Parallel: width shows in Whitney second kind."""
        ss = _build("(&{a: end} || &{b: end})")
        W = whitney_numbers_second(ss)
        # Product of two chains: ranks 0,1,2. Rank 1 has 2 elements.
        assert W.get(1, 0) == 2


# ---------------------------------------------------------------------------
# Log-concavity tests
# ---------------------------------------------------------------------------

class TestLogConcavity:

    def test_constant_sequence(self):
        assert check_log_concave([1, 1, 1]) is True

    def test_unimodal(self):
        assert check_log_concave([1, 2, 1]) is True

    def test_geometric(self):
        assert check_log_concave([1, 2, 4]) is True

    def test_not_log_concave(self):
        assert check_log_concave([1, 0, 1]) is False  # 0² < 1·1

    def test_single_element(self):
        assert check_log_concave([5]) is True

    def test_two_elements(self):
        assert check_log_concave([3, 7]) is True

    def test_benchmark_log_concavity(self):
        """All benchmarks should have log-concave Whitney numbers."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            assert is_whitney_log_concave(ss)


# ---------------------------------------------------------------------------
# Factorization tests
# ---------------------------------------------------------------------------

class TestFactorization:

    def test_parallel_factors(self):
        """χ(L₁ × L₂) = χ(L₁) · χ(L₂)."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ss_prod = _build("(&{a: end} || &{b: end})")
        assert verify_factorization(ss1, ss2, ss_prod)

    def test_parallel_chain_factors(self):
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: &{c: end}}")
        ss_prod = _build("(&{a: end} || &{b: &{c: end}})")
        assert verify_factorization(ss1, ss2, ss_prod)

    def test_parallel_both_chains(self):
        ss1 = _build("&{a: &{b: end}}")
        ss2 = _build("&{c: end}")
        ss_prod = _build("(&{a: &{b: end}} || &{c: end})")
        assert verify_factorization(ss1, ss2, ss_prod)


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyze:

    def test_end_analysis(self):
        r = analyze_characteristic(_build("end"))
        assert r.degree == 0
        assert r.coefficients == [1]
        assert r.eval_at_0 == 1

    def test_branch_analysis(self):
        r = analyze_characteristic(_build("&{a: end}"))
        assert r.degree == 1
        assert r.eval_at_0 == -1
        assert r.eval_at_1 == 0

    def test_parallel_analysis(self):
        r = analyze_characteristic(_build("(&{a: end} || &{b: end})"))
        assert r.degree == 2
        assert r.eval_at_0 == 1  # (-1)²
        assert r.eval_at_1 == 0  # (1-1)²
        assert r.is_log_concave is True


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
    def test_polynomial_properties(self, name, typ):
        """Characteristic polynomial has correct properties."""
        ss = _build(typ)
        r = analyze_characteristic(ss)
        # Degree = height
        assert r.degree == r.height
        # Leading coefficient is 1 (monic)
        assert r.coefficients[0] == 1
        # χ(0) = μ(⊤,⊥)
        from reticulate.mobius import mobius_value
        assert r.eval_at_0 == mobius_value(ss)

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_log_concavity(self, name, typ):
        """Whitney numbers should be log-concave."""
        ss = _build(typ)
        assert is_whitney_log_concave(ss), f"{name}: Whitney numbers not log-concave"

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_whitney_sum(self, name, typ):
        """Sum of Whitney second kind = number of quotient states."""
        ss = _build(typ)
        W = whitney_numbers_second(ss)
        r = analyze_characteristic(ss)
        assert sum(W.values()) == r.num_states
