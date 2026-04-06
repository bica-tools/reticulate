"""Tests for Step 32l: Session Dirichlet zeta function and Euler product."""

from fractions import Fraction

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.session_zeta import (
    ZetaSeries,
    downset_sizes,
    session_zeta,
    session_zeta_series,
    verify_euler_product,
    is_zeta_irreducible,
    euler_factor,
    _series_irreducible,
    _try_split,
    _compositions,
)


def build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Downset sizes
# ---------------------------------------------------------------------------

def test_downset_sizes_end():
    sizes = downset_sizes(build("end"))
    assert sizes == [1]


def test_downset_sizes_single_branch():
    sizes = downset_sizes(build("&{a: end}"))
    assert sorted(sizes) == [1, 2]


def test_downset_sizes_binary_branch():
    ss = build("&{a: end, b: end}")
    sizes = downset_sizes(ss)
    # Top has downset {top, end} size 2; end has size 1
    assert 1 in sizes
    assert max(sizes) == len(ss.states)


def test_downset_sizes_chain():
    # linear protocol a -> b -> end
    ss = build("&{a: &{b: end}}")
    sizes = downset_sizes(ss)
    assert sizes == [1, 2, 3]


# ---------------------------------------------------------------------------
# ZetaSeries basic operations
# ---------------------------------------------------------------------------

def test_zeta_series_from_sizes():
    z = ZetaSeries.from_sizes([1, 2, 2, 3])
    assert z.as_dict() == {1: 1, 2: 2, 3: 1}
    assert z.total_mass() == 4


def test_zeta_series_evaluate_at_1():
    z = ZetaSeries.from_sizes([1, 2, 4])
    # 1/1 + 1/2 + 1/4 = 7/4
    assert z.evaluate(1) == Fraction(7, 4)


def test_zeta_series_evaluate_at_2():
    z = ZetaSeries.from_sizes([1, 2])
    assert z.evaluate(2) == Fraction(1, 1) + Fraction(1, 4)


def test_zeta_series_evaluate_at_0():
    # At s=0, each term contributes 1 -> total_mass
    z = ZetaSeries.from_sizes([1, 2, 3])
    assert z.evaluate(0) == 3


def test_zeta_series_multiplication_commutative():
    a = ZetaSeries.from_sizes([1, 2])
    b = ZetaSeries.from_sizes([1, 3])
    assert a * b == b * a


def test_zeta_series_multiplication_dirichlet():
    # [1,2] * [1,3] should have keys {1*1, 1*3, 2*1, 2*3} = {1,3,2,6}
    a = ZetaSeries.from_sizes([1, 2])
    b = ZetaSeries.from_sizes([1, 3])
    prod = a * b
    d = prod.as_dict()
    assert d == {1: 1, 2: 1, 3: 1, 6: 1}


def test_zeta_series_equality():
    a = ZetaSeries.from_sizes([1, 2, 3])
    b = ZetaSeries.from_sizes([3, 2, 1])
    assert a == b


# ---------------------------------------------------------------------------
# session_zeta
# ---------------------------------------------------------------------------

def test_session_zeta_end():
    assert session_zeta(build("end"), 1) == Fraction(1, 1)


def test_session_zeta_binary_branch():
    ss = build("&{a: end, b: end}")
    # shared end: states {top, end}; sizes [1, 2]; 1 + 1/2 = 3/2
    assert session_zeta(ss, 1) == Fraction(3, 2)


def test_session_zeta_at_zero_is_cardinality():
    ss = build("&{a: &{b: end, c: end}}")
    assert session_zeta(ss, 0) == len(ss.states)


# ---------------------------------------------------------------------------
# Euler product (the main theorem)
# ---------------------------------------------------------------------------

def test_euler_product_simple_parallel():
    S1 = "&{a: end}"
    S2 = "&{b: end}"
    left = build(S1)
    right = build(S2)
    prod = build(f"({S1} || {S2})")
    result = verify_euler_product(left, right, prod)
    assert result.series_equal
    assert result.all_points_equal


def test_euler_product_branching_parallel():
    S1 = "&{a: end, b: end}"
    S2 = "&{x: end}"
    left = build(S1)
    right = build(S2)
    prod = build(f"({S1} || {S2})")
    result = verify_euler_product(left, right, prod)
    assert result.series_equal
    assert result.all_points_equal


def test_euler_product_both_branching():
    S1 = "&{a: end, b: end}"
    S2 = "&{c: end, d: end}"
    left = build(S1)
    right = build(S2)
    prod = build(f"({S1} || {S2})")
    result = verify_euler_product(left, right, prod)
    assert result.series_equal
    assert result.all_points_equal
    # Lattice cardinalities multiply
    assert result.product_sizes[-1] == result.left_sizes[-1] * result.right_sizes[-1]


def test_euler_product_deep_chain_parallel():
    S1 = "&{a: &{b: end}}"
    S2 = "&{x: end}"
    left = build(S1)
    right = build(S2)
    prod = build(f"({S1} || {S2})")
    result = verify_euler_product(left, right, prod)
    assert result.series_equal


def test_euler_product_point_evaluations():
    S1 = "&{a: end, b: end}"
    S2 = "&{x: end}"
    left = build(S1)
    right = build(S2)
    prod = build(f"({S1} || {S2})")
    result = verify_euler_product(left, right, prod, test_values=(1, 2, 3, 5))
    for s, lhs, rhs, ok in result.point_checks:
        assert ok, f"mismatch at s={s}: {lhs} vs {rhs}"


# ---------------------------------------------------------------------------
# Irreducibility
# ---------------------------------------------------------------------------

def test_end_is_irreducible():
    assert is_zeta_irreducible(build("end"))


def test_single_method_is_irreducible():
    # Two-element chain: sizes [1, 2], mass 2
    assert is_zeta_irreducible(build("&{a: end}"))


def test_chain_three_irreducible():
    # sizes [1,2,3], mass 3 -- no non-trivial split
    assert is_zeta_irreducible(build("&{a: &{b: end}}"))


def test_binary_branch_reducibility():
    # sizes for &{a:end, b:end}: top size 3, a->end size 1, b->end size 1
    # series: {1:2, 3:1}, mass 3 -> irreducible
    assert is_zeta_irreducible(build("&{a: end, b: end}"))


def test_parallel_is_reducible_when_both_nontrivial():
    ss = build("(&{a: end} || &{b: end})")
    assert not is_zeta_irreducible(ss)


def test_series_irreducible_singleton():
    assert _series_irreducible(ZetaSeries.from_sizes([1]))


def test_series_reducible_product_of_two_elem():
    # [1,2] * [1,2] = {1:1, 2:2, 4:1}
    z = ZetaSeries.from_sizes([1, 2]) * ZetaSeries.from_sizes([1, 2])
    assert not _series_irreducible(z)


def test_compositions_basic():
    assert _compositions(3, 1) == [(3,)]
    assert sorted(_compositions(3, 2)) == [(1, 2), (2, 1)]
    assert len(_compositions(4, 2)) == 3


def test_try_split_product():
    A = ZetaSeries.from_sizes([1, 2])
    B = ZetaSeries.from_sizes([1, 3])
    Z = A * B
    rA, rB = _try_split(Z, 2, 2)
    assert rA is not None and rB is not None
    assert rA * rB == Z


# ---------------------------------------------------------------------------
# Euler factorization
# ---------------------------------------------------------------------------

def test_euler_factor_irreducible():
    ss = build("&{a: &{b: end}}")
    f = euler_factor(ss)
    assert f.matches
    assert len(f.factors) == 1


def test_euler_factor_parallel():
    ss = build("(&{a: end} || &{b: end})")
    f = euler_factor(ss)
    assert f.matches
    assert len(f.factors) >= 2


def test_euler_factor_product_reconstructs():
    ss = build("(&{a: end, b: end} || &{c: end})")
    f = euler_factor(ss)
    assert f.matches
    assert f.product == f.original


# ---------------------------------------------------------------------------
# Benchmark-style multi-parallel protocols
# ---------------------------------------------------------------------------

def test_triple_parallel_euler():
    S = "((&{a: end} || &{b: end}) || &{c: end})"
    ss = build(S)
    f = euler_factor(ss)
    assert f.matches
    # Should split into three 2-element chains
    assert len(f.factors) >= 2


def test_session_zeta_multiplicative_under_parallel_at_s2():
    S1 = "&{a: &{b: end}}"
    S2 = "&{c: end, d: end}"
    left = build(S1)
    right = build(S2)
    prod = build(f"({S1} || {S2})")
    lhs = session_zeta(prod, 2)
    rhs = session_zeta(left, 2) * session_zeta(right, 2)
    assert lhs == rhs
