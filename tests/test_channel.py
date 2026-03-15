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


# ---------------------------------------------------------------------------
# Sprint 3: Global type bridge + benchmark sweep
# ---------------------------------------------------------------------------


def _get_benchmarks():
    """Load benchmarks with ≤ 15 local states (product ≤ 225 states)."""
    from tests.benchmarks.protocols import BENCHMARKS
    from reticulate.parser import parse as _parse
    from reticulate.statespace import build_statespace as _build
    result = []
    for bp in BENCHMARKS:
        s = _parse(bp.type_string)
        ss = _build(s)
        if len(ss.states) <= 15:
            result.append(bp)
    return result


def _get_binary_multiparty():
    """Load multiparty benchmarks with exactly 2 roles."""
    from tests.benchmarks.multiparty_protocols import MULTIPARTY_BENCHMARKS
    return [mb for mb in MULTIPARTY_BENCHMARKS if len(mb.expected_roles) == 2]


class TestChannelFromGlobal:
    """Build channel from binary global types via projection."""

    def test_request_response(self):
        """Client -> Server : {request: Server -> Client : {response: end}}."""
        from reticulate.global_types import parse_global
        from reticulate.channel import channel_from_global
        g = parse_global(
            "Client -> Server : {request: Server -> Client : {response: end}}"
        )
        r = channel_from_global(g, "Client", "Server")
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True
        assert r.selection_complementary is True

    def test_two_phase_commit(self):
        from reticulate.global_types import parse_global
        from reticulate.channel import channel_from_global
        g = parse_global(
            "Coord -> P : {prepare: "
            "P -> Coord : {yes: Coord -> P : {commit: end}, "
            "no: Coord -> P : {abort: end}}}"
        )
        r = channel_from_global(g, "Coord", "P")
        assert r.is_product_lattice is True
        assert r.selection_complementary is True

    def test_streaming(self):
        from reticulate.global_types import parse_global
        from reticulate.channel import channel_from_global
        g = parse_global(
            "rec X . Producer -> Consumer : {data: X, done: end}"
        )
        r = channel_from_global(g, "Producer", "Consumer")
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True

    def test_negotiation(self):
        from reticulate.global_types import parse_global
        from reticulate.channel import channel_from_global
        g = parse_global(
            "rec X . Buyer -> Seller : {offer: "
            "Seller -> Buyer : {accept: end, counter: X}}"
        )
        r = channel_from_global(g, "Buyer", "Seller")
        assert r.is_product_lattice is True

    def test_auth_service(self):
        from reticulate.global_types import parse_global
        from reticulate.channel import channel_from_global
        g = parse_global(
            "Client -> Server : {login: "
            "Server -> Client : {granted: "
            "rec X . Client -> Server : {request: "
            "Server -> Client : {response: X}, "
            "logout: end}, "
            "denied: end}}"
        )
        r = channel_from_global(g, "Client", "Server")
        assert r.is_product_lattice is True
        assert r.selection_complementary is True

    def test_raft_consensus(self):
        from reticulate.global_types import parse_global
        from reticulate.channel import channel_from_global
        g = parse_global(
            "Candidate -> Voter : {requestVote: "
            "Voter -> Candidate : {granted: "
            "Candidate -> Voter : {appendEntries: "
            "Voter -> Candidate : {ack: end}}, "
            "rejected: end}}"
        )
        r = channel_from_global(g, "Candidate", "Voter")
        assert r.is_product_lattice is True


class TestBenchmarkSweep:
    """Sweep all 79 binary benchmarks: channel is lattice + complementary."""

    @pytest.mark.parametrize("bp", _get_benchmarks(), ids=lambda bp: bp.name)
    def test_benchmark_channel_is_lattice(self, bp):
        s = parse(bp.type_string)
        r = build_channel(s)
        assert r.is_product_lattice is True, f"{bp.name}: channel not lattice"
        assert r.is_isomorphic is True, f"{bp.name}: not isomorphic"
        assert r.selection_complementary is True, f"{bp.name}: not complementary"
        assert r.role_a_embedding is True, f"{bp.name}: role A not embedded"
        assert r.role_b_embedding is True, f"{bp.name}: role B not embedded"

    @pytest.mark.parametrize("mb", _get_binary_multiparty(), ids=lambda mb: mb.name)
    def test_binary_multiparty_channel(self, mb):
        """Binary multiparty benchmarks: projections yield channel."""
        from reticulate.global_types import parse_global
        from reticulate.channel import channel_from_global
        g = parse_global(mb.global_type_string)
        roles = sorted(mb.expected_roles)
        r = channel_from_global(g, roles[0], roles[1])
        assert r.is_product_lattice is True, f"{mb.name}: channel not lattice"


# ---------------------------------------------------------------------------
# Sprint 4: Edge cases + __init__.py re-exports
# ---------------------------------------------------------------------------


class TestChannelEdgeCases:
    """Edge cases: parallel types, deeply nested, recursive with selection."""

    def test_parallel_type_dual(self):
        """Channel for '(S1 || S2)' — dual of parallel is parallel of duals."""
        s = parse("(&{a: end} || &{b: end})")
        r = build_channel(s)
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True

    def test_deeply_nested_branch_select(self):
        """Three levels of alternating branch/select."""
        s = parse("&{a: +{x: &{y: end}}}")
        r = build_channel(s)
        assert r.is_product_lattice is True
        assert r.selection_complementary is True
        assert r.role_a_embedding is True
        assert r.role_b_embedding is True

    def test_recursive_with_select(self):
        """rec X . +{a: X, b: end} — recursive selection."""
        s = parse("rec X . +{a: X, b: end}")
        r = build_channel(s)
        assert r.is_product_lattice is True
        assert r.selection_complementary is True

    def test_continuation_type(self):
        """Parallel with continuation: (&{a: end} || &{b: end}) . &{c: end}."""
        s = parse("(&{a: end} || &{b: end}) . &{c: end}")
        r = build_channel(s)
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True

    def test_dual_of_parallel_is_parallel_of_duals(self):
        """Verify dual(S1 || S2) = dual(S1) || dual(S2)."""
        s = parse("(&{a: end} || +{b: end})")
        d = dual(s)
        assert isinstance(d, Parallel)
        # dual(&{a: end}) = +{a: end}
        assert isinstance(d.branches[0], Select)
        # dual(+{b: end}) = &{b: end}
        assert isinstance(d.branches[1], Branch)

    def test_channel_import_from_init(self):
        """Verify re-exports from __init__.py."""
        import reticulate
        assert hasattr(reticulate, 'ChannelResult')
        assert hasattr(reticulate, 'RoleView')
        assert hasattr(reticulate, 'build_channel')
        assert hasattr(reticulate, 'channel_from_global')
        assert hasattr(reticulate, 'check_branch_complementarity')
        assert hasattr(reticulate, 'check_role_embedding')
        assert hasattr(reticulate, 'role_view')

    def test_symmetric_channel(self):
        """Self-dual type: end is its own dual."""
        r = build_channel(End())
        assert r.type_str == "end"
        assert r.dual_str == "end"
        assert r.is_isomorphic is True

    def test_wait_type(self):
        """Channel for 'wait' — dual(wait) = wait."""
        r = build_channel(Wait())
        assert r.is_product_lattice is True
        assert r.is_isomorphic is True

    def test_multi_branch_select_mix(self):
        """&{a: +{x: end}, b: +{y: end}} — each branch has a selection."""
        s = parse("&{a: +{x: end}, b: +{y: end}}")
        r = build_channel(s)
        assert r.is_product_lattice is True
        assert r.selection_complementary is True

    def test_channel_states_product_invariant(self):
        """channel_states == role_a_states * role_b_states always holds."""
        for ts in ["end", "&{a: end}", "+{a: end}", "&{a: +{x: end}}",
                    "rec X . &{a: X, b: end}"]:
            s = parse(ts)
            r = build_channel(s)
            assert r.channel_states == r.role_a_states * r.role_b_states, ts
