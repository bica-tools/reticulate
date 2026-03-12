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


# ---------------------------------------------------------------------------
# Sprint 2: Branch complementarity + role embedding + role_view
# ---------------------------------------------------------------------------


class TestBranchComplementarity:
    """Branch in L(S) ↔ Selection in L(dual(S))."""

    def test_single_branch_complementary(self):
        """&{a: end}: branch in S, selection in dual(S)."""
        s = parse("&{a: end}")
        r = build_channel(s)
        assert r.selection_complementary is True

    def test_single_select_complementary(self):
        """+{a: end}: selection in S, branch in dual(S)."""
        s = parse("+{a: end}")
        r = build_channel(s)
        assert r.selection_complementary is True

    def test_nested_complementary(self):
        """&{a: +{x: end}}: alternating branch/select."""
        s = parse("&{a: +{x: end}}")
        r = build_channel(s)
        assert r.selection_complementary is True

    def test_recursive_complementary(self):
        s = parse("rec X . &{a: X, b: end}")
        r = build_channel(s)
        assert r.selection_complementary is True

    def test_end_trivially_complementary(self):
        """end has no transitions, so complementarity holds vacuously."""
        r = build_channel(End())
        assert r.selection_complementary is True

    def test_direct_check(self):
        """Use check_branch_complementarity directly."""
        s = parse("&{a: +{x: end}}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)
        from reticulate.morphism import find_isomorphism
        iso = find_isomorphism(ss_s, ss_d)
        assert iso is not None
        assert check_branch_complementarity(ss_s, ss_d, iso.mapping) is True

    def test_complementarity_with_multi_branch(self):
        s = parse("&{a: end, b: end}")
        r = build_channel(s)
        assert r.selection_complementary is True

    def test_deep_nesting(self):
        s = parse("&{a: +{x: &{y: end}}}")
        r = build_channel(s)
        assert r.selection_complementary is True


class TestRoleEmbedding:
    """Inclusion ι_A: L(S) → L(S) × L(dual(S)) is order-embedding."""

    def test_end_embedding(self):
        r = build_channel(End())
        assert r.role_a_embedding is True
        assert r.role_b_embedding is True

    def test_single_branch_embedding(self):
        s = parse("&{a: end}")
        r = build_channel(s)
        assert r.role_a_embedding is True
        assert r.role_b_embedding is True

    def test_single_select_embedding(self):
        s = parse("+{a: end}")
        r = build_channel(s)
        assert r.role_a_embedding is True
        assert r.role_b_embedding is True

    def test_recursive_embedding(self):
        s = parse("rec X . &{a: X, b: end}")
        r = build_channel(s)
        assert r.role_a_embedding is True
        assert r.role_b_embedding is True

    def test_direct_check_left(self):
        """Use check_role_embedding directly for left component."""
        s = parse("&{a: end}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)
        ch = product_statespace(ss_s, ss_d)
        assert check_role_embedding(ch, ss_s, ss_d, "left") is True

    def test_direct_check_right(self):
        """Use check_role_embedding directly for right component."""
        s = parse("&{a: end}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)
        ch = product_statespace(ss_s, ss_d)
        assert check_role_embedding(ch, ss_d, ss_s, "right") is True


class TestRoleView:
    """Extract role's restricted view from product."""

    def test_role_a_view_branch(self):
        """Role A's view of &{a: end} channel."""
        s = parse("&{a: end}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)
        ch = product_statespace(ss_s, ss_d)
        rv = role_view(ch, ss_s, ss_d, s, "left", "A")
        assert rv.role == "A"
        assert rv.local_type == s
        assert rv.local_statespace is ss_s
        # Role A has branch transitions (not selections)
        assert len(rv.visible_transitions) > 0
        # Branch transitions should not be selections for role A
        assert len(rv.selection_transitions) == 0

    def test_role_b_view_becomes_select(self):
        """Role B's view of &{a: end} channel — dual is +{a: end}."""
        s = parse("&{a: end}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)
        ch = product_statespace(ss_s, ss_d)
        rv = role_view(ch, ss_d, ss_s, d, "right", "B")
        assert rv.role == "B"
        assert rv.local_type == d
        # Role B has selection transitions
        assert len(rv.visible_transitions) > 0
        assert len(rv.selection_transitions) > 0

    def test_view_transition_count(self):
        """Each role's visible transitions = local transitions * other_states."""
        s = parse("&{a: +{x: end}}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)
        ch = product_statespace(ss_s, ss_d)
        rv_a = role_view(ch, ss_s, ss_d, s, "left", "A")
        # Each local transition appears for each state of the other component
        expected = len(ss_s.transitions) * len(ss_d.states)
        assert len(rv_a.visible_transitions) == expected

    def test_views_partition_transitions(self):
        """Role A + Role B visible transitions = all channel transitions."""
        s = parse("&{a: end}")
        d = dual(s)
        ss_s = build_statespace(s)
        ss_d = build_statespace(d)
        ch = product_statespace(ss_s, ss_d)
        rv_a = role_view(ch, ss_s, ss_d, s, "left", "A")
        rv_b = role_view(ch, ss_d, ss_s, d, "right", "B")
        all_visible = set(rv_a.visible_transitions) | set(rv_b.visible_transitions)
        assert all_visible == set(ch.transitions)
