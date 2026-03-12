"""Tests for channel duality as role-restricted parallel (Step 157a)."""

import pytest

from reticulate.parser import (
    Branch, Continuation, End, Parallel, Rec, Select, Var, Wait,
    parse, pretty,
)
from reticulate.duality import dual
from reticulate.statespace import build_statespace
from reticulate.product import product_statespace
from reticulate.channel import (
    ChannelResult,
    RoleView,
    build_channel,
    check_branch_complementarity,
    check_role_embedding,
    role_view,
)


# ---------------------------------------------------------------------------
# Sprint 1: Core data types + build_channel
# ---------------------------------------------------------------------------


class TestBuildChannelBasic:
    """build_channel on basic session types."""

    def test_end(self):
        """Channel for 'end': trivial 1-state product."""
        r = build_channel(End())
        assert r.role_a_states == 1
        assert r.role_b_states == 1
        assert r.channel_states == 1
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True

    def test_single_branch(self):
        """Channel for '&{a: end}'."""
        s = parse("&{a: end}")
        r = build_channel(s)
        assert r.role_a_states == 2
        assert r.role_b_states == 2
        assert r.channel_states == 4
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True

    def test_single_select(self):
        """Channel for '+{a: end}'."""
        s = parse("+{a: end}")
        r = build_channel(s)
        assert r.role_a_states == 2
        assert r.role_b_states == 2
        assert r.channel_states == 4
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True

    def test_branch_select(self):
        """Channel for '&{a: +{x: end}}'."""
        s = parse("&{a: +{x: end}}")
        r = build_channel(s)
        assert r.role_a_states == 3
        assert r.role_b_states == 3
        assert r.channel_states == 9
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True

    def test_multi_branch(self):
        """Channel for '&{a: end, b: end}' — both go to end, so 2 states."""
        s = parse("&{a: end, b: end}")
        r = build_channel(s)
        assert r.role_a_states == 2
        assert r.role_b_states == 2
        assert r.channel_states == 4
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True

    def test_multi_select(self):
        """Channel for '+{a: end, b: end}' — both go to end, so 2 states."""
        s = parse("+{a: end, b: end}")
        r = build_channel(s)
        assert r.role_a_states == 2
        assert r.role_b_states == 2
        assert r.channel_states == 4
        assert r.is_product_lattice is True

    def test_recursive_type(self):
        """Channel for 'rec X . &{a: X, b: end}'."""
        s = parse("rec X . &{a: X, b: end}")
        r = build_channel(s)
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True

    def test_nested_branch_select(self):
        """Channel for '&{a: +{x: end, y: end}, b: end}'."""
        s = parse("&{a: +{x: end, y: end}, b: end}")
        r = build_channel(s)
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True


class TestChannelResultFields:
    """Verify ChannelResult field values."""

    def test_type_strings(self):
        s = parse("&{a: end}")
        r = build_channel(s)
        assert r.type_str == "&{a: end}"
        assert r.dual_str == "+{a: end}"

    def test_states_product_formula(self):
        """channel_states == role_a_states * role_b_states."""
        s = parse("&{a: +{x: end}}")
        r = build_channel(s)
        assert r.channel_states == r.role_a_states * r.role_b_states

    def test_frozen_dataclass(self):
        r = build_channel(End())
        with pytest.raises(AttributeError):
            r.is_product_lattice = False  # type: ignore[misc]


class TestChannelProductAlwaysLattice:
    """The product of two lattices is always a lattice."""

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "+{a: end}",
        "&{a: end, b: end}",
        "+{a: end, b: end}",
        "&{a: +{x: end}}",
        "rec X . &{a: X, b: end}",
        "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}",
        "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
    ])
    def test_product_is_lattice(self, type_str: str):
        s = parse(type_str)
        r = build_channel(s)
        assert r.is_product_lattice is True, f"Channel product not lattice for: {type_str}"
