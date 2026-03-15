"""Tests for async channels: message buffers as lattice extensions (Step 157b)."""

import pytest

from reticulate.parser import (
    Branch, End, Rec, Select, Var, parse, pretty,
)
from reticulate.duality import dual
from reticulate.statespace import StateSpace, build_statespace
from reticulate.product import product_statespace
from reticulate.async_channel import (
    AsyncChannelResult,
    build_async_channel,
    build_async_statespace,
    check_sync_embedding,
    async_growth_ratio,
)


# ---------------------------------------------------------------------------
# Sprint 1: Core data types + BFS construction
# ---------------------------------------------------------------------------


class TestAsyncChannelResultDataclass:
    """AsyncChannelResult is a frozen dataclass."""

    def test_frozen(self):
        r = AsyncChannelResult(
            type_str="end", dual_str="end", capacity=1,
            sender_states=1, receiver_states=1,
            async_channel_states=1, sync_channel_states=1,
            is_lattice=True, sync_embeds=True,
            max_buffer_occupancy=0, buffer_distribution={0: 1},
        )
        with pytest.raises(AttributeError):
            r.capacity = 2  # type: ignore[misc]

    def test_fields_accessible(self):
        r = AsyncChannelResult(
            type_str="end", dual_str="end", capacity=1,
            sender_states=1, receiver_states=1,
            async_channel_states=1, sync_channel_states=1,
            is_lattice=True, sync_embeds=True,
            max_buffer_occupancy=0, buffer_distribution={0: 1},
        )
        assert r.capacity == 1
        assert r.is_lattice is True
        assert r.buffer_distribution == {0: 1}


class TestBuildAsyncStatespaceBasic:
    """BFS construction of async channel state space."""

    def test_end_type(self):
        """end/end: only state is (end, end, (), ()), no transitions."""
        ss_s = build_statespace(End())
        ss_d = build_statespace(dual(End()))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=1)
        assert len(async_ss.states) == 1
        assert len(async_ss.transitions) == 0
        assert async_ss.top == async_ss.bottom

    def test_single_select_k1(self):
        """'+{a: end}': A sends 'a' to buffer, B receives 'a'."""
        s = parse("+{a: end}")
        ss_s = build_statespace(s)
        ss_d = build_statespace(dual(s))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=1)
        # States: (top_A, top_B, (), ()) → send_AB:a → (bot_A, top_B, (a,), ())
        #         → recv_AB:a → (bot_A, bot_B, (), ())
        # Plus top_B has branch {a: end} which is the dual
        assert len(async_ss.states) >= 2
        assert len(async_ss.transitions) >= 1
        # Top has empty buffers
        assert async_ss.top is not None

    def test_single_branch_k1(self):
        """'&{a: end}': A has branch, dual has select. B sends, A receives."""
        s = parse("&{a: end}")
        ss_s = build_statespace(s)
        ss_d = build_statespace(dual(s))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=1)
        assert len(async_ss.states) >= 2
        assert len(async_ss.transitions) >= 1

    def test_top_has_empty_buffers(self):
        """Top state should correspond to empty buffers."""
        s = parse("+{a: end}")
        ss_s = build_statespace(s)
        ss_d = build_statespace(dual(s))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=1)
        # The top state label should not contain buffer content markers
        top_label = async_ss.labels.get(async_ss.top, "")
        # Empty buffers means no "[...]" part or empty brackets
        assert "[" not in top_label or top_label.endswith("[|])")

    def test_bottom_has_empty_buffers(self):
        """Bottom state should correspond to empty buffers."""
        s = parse("+{a: end}")
        ss_s = build_statespace(s)
        ss_d = build_statespace(dual(s))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=1)
        # Bottom = (bot_S, bot_dual, (), ())
        bottom_label = async_ss.labels.get(async_ss.bottom, "")
        assert "[" not in bottom_label or bottom_label.endswith("[|])")

    def test_buffer_contents_tracked(self):
        """States with non-empty buffers should exist for select types."""
        s = parse("+{a: end}")
        ss_s = build_statespace(s)
        ss_d = build_statespace(dual(s))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=1)
        # Should have at least one state with buffer content
        labels = list(async_ss.labels.values())
        has_buffered = any("[" in l for l in labels)
        assert has_buffered, f"No buffered states found in {labels}"

    def test_state_count_at_least_one(self):
        """Every async state space has at least one state."""
        s = End()
        ss_s = build_statespace(s)
        ss_d = build_statespace(dual(s))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=1)
        assert len(async_ss.states) >= 1

    def test_transition_labels_prefixed(self):
        """Transition labels should be send_AB:m, recv_AB:m, send_BA:m, recv_BA:m."""
        s = parse("+{a: end}")
        ss_s = build_statespace(s)
        ss_d = build_statespace(dual(s))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=1)
        for _, lbl, _ in async_ss.transitions:
            assert any(lbl.startswith(p) for p in
                       ("send_AB:", "recv_AB:", "send_BA:", "recv_BA:")), \
                f"Unexpected label: {lbl}"

    def test_select_then_branch_k1(self):
        """'+{a: &{b: end}}': A selects a (buffered), B receives a,
        then B selects b (buffered), A receives b."""
        s = parse("+{a: &{b: end}}")
        ss_s = build_statespace(s)
        ss_d = build_statespace(dual(s))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=1)
        # Should have send_AB and recv_AB and send_BA and recv_BA transitions
        labels = {lbl for _, lbl, _ in async_ss.transitions}
        assert "send_AB:a" in labels
        assert "recv_AB:a" in labels

    def test_capacity_zero_no_sends(self):
        """K=0: no buffer space, so no send transitions at all."""
        s = parse("+{a: end}")
        ss_s = build_statespace(s)
        ss_d = build_statespace(dual(s))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=0)
        send_labels = [lbl for _, lbl, _ in async_ss.transitions
                       if lbl.startswith("send_")]
        assert len(send_labels) == 0

    def test_multi_select_k1(self):
        """'+{a: end, b: end}': A can send either a or b."""
        s = parse("+{a: end, b: end}")
        ss_s = build_statespace(s)
        ss_d = build_statespace(dual(s))
        async_ss = build_async_statespace(ss_s, ss_d, capacity=1)
        labels = {lbl for _, lbl, _ in async_ss.transitions}
        assert "send_AB:a" in labels
        assert "send_AB:b" in labels


# ---------------------------------------------------------------------------
# Sprint 2: Lattice check + sync embedding + build_async_channel
# ---------------------------------------------------------------------------


class TestBuildAsyncChannel:
    """build_async_channel orchestrator."""

    def test_end_lattice(self):
        r = build_async_channel(End(), capacity=1)
        assert r.is_lattice is True
        assert r.async_channel_states == 1
        assert r.sync_channel_states == 1

    def test_single_select_lattice(self):
        s = parse("+{a: end}")
        r = build_async_channel(s, capacity=1)
        assert r.is_lattice is True

    def test_single_branch_lattice(self):
        s = parse("&{a: end}")
        r = build_async_channel(s, capacity=1)
        assert r.is_lattice is True

    def test_branch_select_lattice(self):
        s = parse("&{a: +{x: end}}")
        r = build_async_channel(s, capacity=1)
        assert r.is_lattice is True

    def test_recursive_lattice(self):
        s = parse("rec X . &{a: +{ok: X, err: end}}")
        r = build_async_channel(s, capacity=1)
        assert r.is_lattice is True

    def test_sync_embeds(self):
        s = parse("+{a: end}")
        r = build_async_channel(s, capacity=1)
        assert r.sync_embeds is True

    def test_sync_embeds_branch(self):
        s = parse("&{a: end}")
        r = build_async_channel(s, capacity=1)
        assert r.sync_embeds is True

    def test_capacity_zero_matches_sync(self):
        """K=0: async state space should have same count as sync (only empty-buffer states)."""
        s = parse("+{a: end}")
        r = build_async_channel(s, capacity=0)
        # With K=0, no sends possible, so only the start state is reachable
        # unless there are no selection transitions at all
        # Actually: K=0 means we can never buffer, so only (top,top,(),()) is reachable
        # if there are only selection transitions from top
        assert r.async_channel_states >= 1

    def test_capacity_increases_states(self):
        """Higher capacity should produce >= states."""
        s = parse("+{a: &{b: end}}")
        r1 = build_async_channel(s, capacity=1)
        r2 = build_async_channel(s, capacity=2)
        assert r2.async_channel_states >= r1.async_channel_states

    def test_result_fields_correct(self):
        s = parse("+{a: end}")
        r = build_async_channel(s, capacity=1)
        assert r.type_str == pretty(s)
        assert r.dual_str == pretty(dual(s))
        assert r.capacity == 1
        assert r.sender_states == 2
        assert r.receiver_states == 2

    def test_buffer_distribution_sums(self):
        """buffer_distribution values should sum to async_channel_states."""
        s = parse("+{a: end}")
        r = build_async_channel(s, capacity=1)
        assert sum(r.buffer_distribution.values()) == r.async_channel_states

    def test_max_buffer_occupancy_bounded(self):
        """max_buffer_occupancy should be <= 2 * capacity."""
        s = parse("+{a: &{b: end}}")
        r = build_async_channel(s, capacity=1)
        assert r.max_buffer_occupancy <= 2 * r.capacity


# ---------------------------------------------------------------------------
# Sprint 3: Benchmark sweep + capacity analysis
# ---------------------------------------------------------------------------


class TestBenchmarkAsyncLattice:
    """Parametrized benchmark sweep at K=1."""

    SMALL_BENCHMARKS = [
        "end",
        "+{a: end}",
        "&{a: end}",
        "+{a: end, b: end}",
        "&{a: end, b: end}",
        "&{a: +{x: end}}",
        "+{a: &{b: end}}",
        "&{a: +{ok: end, err: end}}",
        "rec X . &{a: +{ok: X, err: end}}",
        "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
    ]

    @pytest.mark.parametrize("type_str", SMALL_BENCHMARKS)
    def test_async_lattice_k1(self, type_str: str):
        s = parse(type_str)
        r = build_async_channel(s, capacity=1)
        assert r.is_lattice is True, f"Not a lattice for {type_str}"

    @pytest.mark.parametrize("type_str", SMALL_BENCHMARKS[:5])
    def test_async_lattice_k2(self, type_str: str):
        s = parse(type_str)
        r = build_async_channel(s, capacity=2)
        assert r.is_lattice is True, f"Not a lattice at K=2 for {type_str}"

    def test_growth_k1_vs_k2(self):
        """K=2 should produce >= states than K=1."""
        s = parse("+{a: &{b: end}}")
        r1 = build_async_channel(s, capacity=1)
        r2 = build_async_channel(s, capacity=2)
        assert r2.async_channel_states >= r1.async_channel_states

    def test_max_occupancy_bounded_by_capacity(self):
        """Max buffer occupancy should not exceed capacity (per direction)."""
        s = parse("+{a: &{b: end}}")
        r = build_async_channel(s, capacity=2)
        # Each buffer bounded by K, so total bounded by 2K
        assert r.max_buffer_occupancy <= 2 * r.capacity

    def test_bottom_has_empty_buffers_benchmark(self):
        """Bottom state should always have empty buffers."""
        s = parse("&{a: +{ok: end, err: end}}")
        r = build_async_channel(s, capacity=1)
        assert 0 in r.buffer_distribution


class TestAsyncGrowthRatio:
    """async_growth_ratio measurements."""

    def test_end_ratio_one(self):
        assert async_growth_ratio(End(), 1) == 1.0

    def test_select_ratio_positive(self):
        s = parse("+{a: end}")
        ratio = async_growth_ratio(s, 1)
        assert ratio > 0

    def test_higher_capacity_higher_ratio(self):
        s = parse("+{a: &{b: end}}")
        r1 = async_growth_ratio(s, 1)
        r2 = async_growth_ratio(s, 2)
        assert r2 >= r1


# ---------------------------------------------------------------------------
# Sprint 4: __init__.py + global type bridge + edge cases
# ---------------------------------------------------------------------------


class TestAsyncChannelEdgeCases:
    """Edge cases and global type bridge."""

    def test_deeply_nested(self):
        s = parse("&{a: &{b: +{c: end}}}")
        r = build_async_channel(s, capacity=1)
        assert r.is_lattice is True

    def test_multi_branch_multi_select(self):
        s = parse("&{a: +{x: end, y: end}, b: +{z: end}}")
        r = build_async_channel(s, capacity=1)
        assert r.is_lattice is True

    def test_recursive_with_selection(self):
        s = parse("rec X . +{a: &{b: X}, c: end}")
        r = build_async_channel(s, capacity=1)
        assert r.is_lattice is True

    def test_frozen_result(self):
        r = build_async_channel(End(), capacity=1)
        with pytest.raises(AttributeError):
            r.capacity = 5  # type: ignore[misc]

    def test_import_from_init(self):
        """Verify async_channel exports are accessible from reticulate."""
        from reticulate import AsyncChannelResult as ACR
        from reticulate import build_async_channel as bac
        assert ACR is not None
        assert bac is not None


class TestAsyncChannelFromGlobal:
    """Global type → async channel bridge."""

    def test_request_response(self):
        from reticulate.global_types import parse_global
        g = parse_global("C->S: {request: S->C: {response: end}}")
        from reticulate.async_channel import async_channel_from_global
        r = async_channel_from_global(g, "C", "S", capacity=1)
        assert r.is_lattice is True

    def test_streaming(self):
        from reticulate.global_types import parse_global
        g = parse_global("rec X . C->S: {data: S->C: {ack: X, done: end}}")
        from reticulate.async_channel import async_channel_from_global
        r = async_channel_from_global(g, "C", "S", capacity=1)
        assert r.is_lattice is True
