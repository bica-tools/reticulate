"""Tests for product construction (product.py).

Covers product_statespace, power_statespace, and power_type.
"""

import pytest

from reticulate.parser import parse, pretty
from reticulate.statespace import StateSpace, build_statespace
from reticulate.product import product_statespace, power_statespace, power_type
from reticulate.lattice import check_lattice


# ===================================================================
# Helper utilities
# ===================================================================

def _transitions_from(ss: StateSpace, state: int) -> dict[str, int]:
    """Return {label: target} for transitions from *state*."""
    return {l: t for s, l, t in ss.transitions if s == state}


def _all_labels(ss: StateSpace) -> set[str]:
    """Return all transition labels in the state space."""
    return {l for _, l, _ in ss.transitions}


def _build(type_str: str) -> StateSpace:
    return build_statespace(parse(type_str))


# ===================================================================
# product_statespace — basic properties
# ===================================================================

class TestProductBasic:
    def test_end_times_end(self) -> None:
        """end × end = single state (trivial product)."""
        left = _build("end")
        right = _build("end")
        prod = product_statespace(left, right)
        assert len(prod.states) == 1
        assert prod.top == prod.bottom
        assert len(prod.transitions) == 0

    def test_state_count(self) -> None:
        """Product of 2-state × 2-state = 4 states."""
        left = _build("&{a: end}")
        right = _build("&{b: end}")
        prod = product_statespace(left, right)
        assert len(prod.states) == 4

    def test_state_count_asymmetric(self) -> None:
        """Product of 2-state × 3-state = 6 states."""
        left = _build("&{a: end}")
        right = _build("&{b: &{c: end}}")
        prod = product_statespace(left, right)
        assert len(prod.states) == 6

    def test_top_bottom_distinct(self) -> None:
        """In non-trivial product, top != bottom."""
        left = _build("&{a: end}")
        right = _build("&{b: end}")
        prod = product_statespace(left, right)
        assert prod.top != prod.bottom

    def test_transitions_interleave(self) -> None:
        """Product allows interleaved transitions from both components."""
        left = _build("&{a: end}")
        right = _build("&{b: end}")
        prod = product_statespace(left, right)
        labels = _all_labels(prod)
        assert "a" in labels
        assert "b" in labels

    def test_transition_count(self) -> None:
        """2-state × 2-state: from (top,top) we get a and b transitions."""
        left = _build("&{a: end}")
        right = _build("&{b: end}")
        prod = product_statespace(left, right)
        top_trans = _transitions_from(prod, prod.top)
        assert len(top_trans) == 2


# ===================================================================
# product_statespace — lattice properties
# ===================================================================

class TestProductLattice:
    def test_product_is_lattice(self) -> None:
        """Product of two lattices is a lattice."""
        left = _build("&{a: end}")
        right = _build("&{b: end}")
        prod = product_statespace(left, right)
        lr = check_lattice(prod)
        assert lr.is_lattice

    def test_product_of_chains_is_lattice(self) -> None:
        """Product of two chains is a lattice."""
        left = _build("&{a: &{b: end}}")
        right = _build("&{c: &{d: end}}")
        prod = product_statespace(left, right)
        lr = check_lattice(prod)
        assert lr.is_lattice

    def test_product_three_branch_is_lattice(self) -> None:
        """Product involving a branch with 3 choices is a lattice."""
        left = _build("&{a: end, b: end, c: end}")
        right = _build("&{d: end}")
        prod = product_statespace(left, right)
        lr = check_lattice(prod)
        assert lr.is_lattice


# ===================================================================
# product_statespace — labels and coordinates
# ===================================================================

class TestProductLabels:
    def test_labels_are_pairs(self) -> None:
        """Product state labels have pair format (l1, l2)."""
        left = _build("&{a: end}")
        right = _build("&{b: end}")
        prod = product_statespace(left, right)
        for sid in prod.states:
            label = prod.labels[sid]
            assert label.startswith("(")
            assert label.endswith(")")

    def test_product_coords_populated(self) -> None:
        """Product coordinate map is populated for all states."""
        left = _build("&{a: end}")
        right = _build("&{b: end}")
        prod = product_statespace(left, right)
        assert prod.product_coords is not None
        assert len(prod.product_coords) == len(prod.states)

    def test_product_coords_are_pairs(self) -> None:
        """Each coordinate is a 2-tuple for binary product."""
        left = _build("&{a: end}")
        right = _build("&{b: end}")
        prod = product_statespace(left, right)
        for coord in prod.product_coords.values():
            assert len(coord) == 2


# ===================================================================
# product_statespace — selection propagation
# ===================================================================

class TestProductSelections:
    def test_selection_propagated_from_left(self) -> None:
        """Selection transitions from left component appear in product."""
        left = _build("+{ok: end, err: end}")
        right = _build("&{b: end}")
        prod = product_statespace(left, right)
        assert len(prod.selection_transitions) > 0

    def test_selection_propagated_from_right(self) -> None:
        """Selection transitions from right component appear in product."""
        left = _build("&{a: end}")
        right = _build("+{ok: end, err: end}")
        prod = product_statespace(left, right)
        assert len(prod.selection_transitions) > 0


# ===================================================================
# power_statespace
# ===================================================================

class TestPowerStatespace:
    def test_power_0_is_trivial(self) -> None:
        """S^0 = single-state trivial lattice."""
        ss = _build("&{a: end}")
        p0 = power_statespace(ss, 0)
        assert len(p0.states) == 1
        assert p0.top == p0.bottom

    def test_power_1_is_identity(self) -> None:
        """S^1 = S (same object)."""
        ss = _build("&{a: end}")
        p1 = power_statespace(ss, 1)
        assert p1 is ss

    def test_power_2_state_count(self) -> None:
        """S^2 has |S|^2 states."""
        ss = _build("&{a: end}")
        p2 = power_statespace(ss, 2)
        assert len(p2.states) == len(ss.states) ** 2

    def test_power_3_state_count(self) -> None:
        """S^3 has |S|^3 states."""
        ss = _build("&{a: end}")
        p3 = power_statespace(ss, 3)
        assert len(p3.states) == len(ss.states) ** 3

    def test_power_2_is_lattice(self) -> None:
        """S^2 is a lattice when S is a lattice."""
        ss = _build("&{a: end, b: end}")
        p2 = power_statespace(ss, 2)
        lr = check_lattice(p2)
        assert lr.is_lattice

    def test_power_negative_raises(self) -> None:
        """Negative exponent raises ValueError."""
        ss = _build("&{a: end}")
        with pytest.raises(ValueError, match="non-negative"):
            power_statespace(ss, -1)


# ===================================================================
# power_type
# ===================================================================

class TestPowerType:
    def test_power_type_1_identity(self) -> None:
        """power_type(S, 1) returns the original string."""
        result = power_type("&{a: end}", 1)
        assert result == "&{a: end}"

    def test_power_type_2_has_parallel(self) -> None:
        """power_type(S, 2) produces a parallel type."""
        result = power_type("&{a: end}", 2)
        assert "||" in result or "∥" in result

    def test_power_type_labels_disambiguated(self) -> None:
        """power_type relabels to satisfy WF-Par disjointness."""
        result = power_type("&{a: end}", 2)
        # Should contain suffixed labels like a_1 and a_2
        assert "a_1" in result
        assert "a_2" in result

    def test_power_type_3_copies(self) -> None:
        """power_type(S, 3) produces 3 parallel copies."""
        result = power_type("&{a: end}", 3)
        assert "a_1" in result
        assert "a_2" in result
        assert "a_3" in result

    def test_power_type_negative_raises(self) -> None:
        """Exponent < 1 raises ValueError."""
        with pytest.raises(ValueError, match="≥ 1"):
            power_type("&{a: end}", 0)

    def test_power_type_roundtrip(self) -> None:
        """power_type output can be parsed back."""
        result = power_type("&{a: end, b: end}", 2)
        ast = parse(result)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice


# ===================================================================
# product_statespace — product factors
# ===================================================================

class TestProductFactors:
    def test_binary_product_has_two_factors(self) -> None:
        """Binary product stores two factors."""
        left = _build("&{a: end}")
        right = _build("&{b: end}")
        prod = product_statespace(left, right)
        assert prod.product_factors is not None
        assert len(prod.product_factors) == 2

    def test_nested_product_accumulates_factors(self) -> None:
        """(A × B) × C has three factors."""
        a = _build("&{a: end}")
        b = _build("&{b: end}")
        c = _build("&{c: end}")
        ab = product_statespace(a, b)
        abc = product_statespace(ab, c)
        assert abc.product_factors is not None
        assert len(abc.product_factors) == 3

    def test_nested_product_coords_are_triples(self) -> None:
        """(A × B) × C has 3-tuples as coordinates."""
        a = _build("&{a: end}")
        b = _build("&{b: end}")
        c = _build("&{c: end}")
        ab = product_statespace(a, b)
        abc = product_statespace(ab, c)
        for coord in abc.product_coords.values():
            assert len(coord) == 3
