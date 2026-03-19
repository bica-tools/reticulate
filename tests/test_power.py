"""Tests for protocol power construction S^n."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.product import power_statespace, power_type


def _ss(type_str: str):
    return build_statespace(parse(type_str))


# ---------------------------------------------------------------------------
# power_statespace: basic properties
# ---------------------------------------------------------------------------

class TestPowerBasic:
    def test_power_0_single_state(self):
        ss = _ss("&{a: end}")
        p = power_statespace(ss, 0)
        assert len(p.states) == 1
        assert p.top == p.bottom

    def test_power_1_identity(self):
        ss = _ss("&{a: end, b: end}")
        p = power_statespace(ss, 1)
        assert len(p.states) == len(ss.states)
        assert len(p.transitions) == len(ss.transitions)

    def test_power_2_squared(self):
        ss = _ss("&{a: end}")
        p = power_statespace(ss, 2)
        assert len(p.states) == len(ss.states) ** 2  # 2^2 = 4

    def test_power_3_cubed(self):
        ss = _ss("&{a: end}")
        p = power_statespace(ss, 3)
        assert len(p.states) == len(ss.states) ** 3  # 2^3 = 8

    def test_power_negative_raises(self):
        ss = _ss("&{a: end}")
        with pytest.raises(ValueError, match="non-negative"):
            power_statespace(ss, -1)


# ---------------------------------------------------------------------------
# power_statespace: state counts for various protocols
# ---------------------------------------------------------------------------

class TestPowerStateCounts:
    """S^n should have |S|^n states."""

    def test_fork_squared(self):
        """&{a: end, b: end} has 2 states (top, end). S^2 = 4."""
        ss = _ss("&{a: end, b: end}")
        assert len(ss.states) == 2
        p = power_statespace(ss, 2)
        assert len(p.states) == 4  # 2^2

    def test_fork_cubed(self):
        """&{a: end, b: end} has 2 states. S^3 = 8."""
        ss = _ss("&{a: end, b: end}")
        p = power_statespace(ss, 3)
        assert len(p.states) == 8  # 2^3

    def test_chain_squared(self):
        """&{a: &{b: end}} has 3 states. S^2 = 9."""
        ss = _ss("&{a: &{b: end}}")
        p = power_statespace(ss, 2)
        assert len(p.states) == 9

    def test_selection_squared(self):
        """&{a: +{OK: end, ERR: end}} has 3 states. S^2 = 9."""
        ss = _ss("&{a: +{OK: end, ERR: end}}")
        p = power_statespace(ss, 2)
        assert len(p.states) == 9

    def test_recursive_squared(self):
        """rec X . &{a: X, done: end} has 2 states. S^2 = 4."""
        ss = _ss("rec X . &{a: X, done: end}")
        p = power_statespace(ss, 2)
        assert len(p.states) == 4


# ---------------------------------------------------------------------------
# power_statespace: lattice preservation
# ---------------------------------------------------------------------------

class TestPowerLattice:
    """S^n should always be a lattice if S is a lattice."""

    def test_fork_squared_is_lattice(self):
        ss = _ss("&{a: end, b: end}")
        p = power_statespace(ss, 2)
        result = check_lattice(p)
        assert result.is_lattice

    def test_fork_cubed_is_lattice(self):
        ss = _ss("&{a: end, b: end}")
        p = power_statespace(ss, 3)
        result = check_lattice(p)
        assert result.is_lattice

    def test_selection_squared_is_lattice(self):
        ss = _ss("&{a: +{OK: end, ERR: end}}")
        p = power_statespace(ss, 2)
        result = check_lattice(p)
        assert result.is_lattice

    def test_recursive_squared_is_lattice(self):
        ss = _ss("rec X . &{a: X, done: end}")
        p = power_statespace(ss, 2)
        result = check_lattice(p)
        assert result.is_lattice

    def test_power_4_is_lattice(self):
        ss = _ss("&{a: end}")
        p = power_statespace(ss, 4)
        result = check_lattice(p)
        assert result.is_lattice


# ---------------------------------------------------------------------------
# power_statespace: structural properties
# ---------------------------------------------------------------------------

class TestPowerStructure:
    def test_top_bottom_exist(self):
        ss = _ss("&{a: end, b: end}")
        p = power_statespace(ss, 3)
        assert p.top in p.states
        assert p.bottom in p.states

    def test_coordinates_have_n_components(self):
        ss = _ss("&{a: end}")
        p = power_statespace(ss, 4)
        assert p.product_coords is not None
        for coord in p.product_coords.values():
            assert len(coord) == 4

    def test_factors_have_n_copies(self):
        ss = _ss("&{a: end}")
        p = power_statespace(ss, 3)
        assert p.product_factors is not None
        assert len(p.product_factors) == 3

    def test_selections_propagated(self):
        """Selection edges from each copy should appear in the product."""
        ss = _ss("&{a: +{OK: end, ERR: end}}")
        p = power_statespace(ss, 2)
        sel_count = sum(1 for s, l, t in p.transitions
                        if p.is_selection(s, l, t))
        # Each copy has 2 selection edges, in a 3×3 grid
        # selections at each coordinate position
        assert sel_count > 0

    def test_transition_count(self):
        """Transitions grow as |S| * |T| * (|T_left| + |T_right|)."""
        ss = _ss("&{a: end}")  # 2 states, 1 transition
        p2 = power_statespace(ss, 2)
        p3 = power_statespace(ss, 3)
        # S^2: 4 states, each inner state has 2 outgoing (1 per component)
        assert len(p2.transitions) >= 2
        assert len(p3.transitions) >= len(p2.transitions)


# ---------------------------------------------------------------------------
# power_statespace: exponent laws
# ---------------------------------------------------------------------------

class TestPowerExponentLaws:
    """S^(m+n) ≅ S^m × S^n in state count."""

    def test_additive_exponent_2_plus_1(self):
        ss = _ss("&{a: end, b: end}")
        p3 = power_statespace(ss, 3)
        p2 = power_statespace(ss, 2)
        p1 = power_statespace(ss, 1)
        from reticulate.product import product_statespace
        p2x1 = product_statespace(p2, p1)
        assert len(p3.states) == len(p2x1.states)

    def test_additive_exponent_2_plus_2(self):
        ss = _ss("&{a: end}")
        p4 = power_statespace(ss, 4)
        p2 = power_statespace(ss, 2)
        from reticulate.product import product_statespace
        p2x2 = product_statespace(p2, p2)
        assert len(p4.states) == len(p2x2.states)


# ---------------------------------------------------------------------------
# power_type: string construction
# ---------------------------------------------------------------------------

class TestPowerType:
    def test_power_1_unchanged(self):
        result = power_type("&{a: end}", 1)
        assert "a" in result

    def test_power_2_has_parallel(self):
        result = power_type("&{a: end}", 2)
        assert "||" in result or "∥" in result

    def test_power_2_labels_distinct(self):
        result = power_type("&{a: end}", 2)
        assert "a_1" in result
        assert "a_2" in result

    def test_power_3_labels_distinct(self):
        result = power_type("&{a: end, b: end}", 3)
        assert "a_1" in result
        assert "a_2" in result
        assert "a_3" in result

    def test_power_type_parseable(self):
        """The generated type string should be parseable."""
        result = power_type("&{a: end, b: end}", 2)
        ast = parse(result)
        ss = build_statespace(ast)
        assert len(ss.states) == 4  # 2^2

    def test_power_type_recursive(self):
        result = power_type("rec X . &{a: X, done: end}", 2)
        ast = parse(result)
        ss = build_statespace(ast)
        assert len(ss.states) == 4  # 2^2

    def test_power_type_negative_raises(self):
        with pytest.raises(ValueError, match="≥ 1"):
            power_type("&{a: end}", 0)

    def test_power_type_lattice(self):
        """S^n from power_type should always produce a lattice."""
        result = power_type("&{a: +{OK: end, ERR: end}}", 2)
        ss = build_statespace(parse(result))
        assert check_lattice(ss).is_lattice
