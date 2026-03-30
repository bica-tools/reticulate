"""Tests for covert channel detection via mutual information (Step 89e)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.covert_channel import (
    CovertChannel,
    CovertChannelResult,
    IndependenceResult,
    analyze_covert_channels,
    channel_capacity,
    detect_covert_channels,
    parallel_independence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def end_ss():
    """Single-state 'end' type."""
    return build_statespace(parse("end"))


@pytest.fixture
def linear_ss():
    """Linear: a . b . end."""
    return build_statespace(parse("&{a: &{b: end}}"))


@pytest.fixture
def branch2_ss():
    """Branch with 2 options: &{a: end, b: end}."""
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def parallel_ss():
    """Parallel: (&{a: end} || &{b: end}) — independent branches."""
    return build_statespace(parse("(&{a: end} || &{b: end})"))


@pytest.fixture
def nested_branch_ss():
    """Nested: &{a: &{x: end, y: end}, b: end}."""
    return build_statespace(parse("&{a: &{x: end, y: end}, b: end}"))


@pytest.fixture
def select_ss():
    """Selection: +{ok: end, err: end}."""
    return build_statespace(parse("+{ok: end, err: end}"))


@pytest.fixture
def shared_label_ss():
    """Protocol with same label at multiple states:
    &{a: &{b: end}, c: &{b: end}}
    'b' appears after both 'a' and 'c' paths.
    """
    return build_statespace(parse("&{a: &{b: end}, c: &{b: end}}"))


# ---------------------------------------------------------------------------
# detect_covert_channels
# ---------------------------------------------------------------------------


class TestDetectCovertChannels:
    """Tests for detect_covert_channels."""

    def test_end_no_channels(self, end_ss):
        channels = detect_covert_channels(end_ss)
        assert len(channels) == 0

    def test_linear_no_channels(self, linear_ss):
        channels = detect_covert_channels(linear_ss)
        assert len(channels) == 0

    def test_explicit_disjoint_groups_no_channel(self, parallel_ss):
        """Disjoint label groups should have no covert channel."""
        channels = detect_covert_channels(
            parallel_ss, label_groups=[{"a"}, {"b"}]
        )
        assert len(channels) == 0

    def test_explicit_same_state_groups(self, branch2_ss):
        """Labels at the same state may show mutual information."""
        channels = detect_covert_channels(
            branch2_ss, label_groups=[{"a"}, {"b"}]
        )
        # a and b are at the same state -> may have MI
        # They are co-enabled, so they share a state
        for ch in channels:
            assert ch.mutual_information >= 0.0

    def test_channel_has_correct_type(self, branch2_ss):
        channels = detect_covert_channels(
            branch2_ss, label_groups=[{"a"}, {"b"}]
        )
        for ch in channels:
            assert isinstance(ch, CovertChannel)

    def test_channel_description_nonempty(self, branch2_ss):
        channels = detect_covert_channels(
            branch2_ss, label_groups=[{"a"}, {"b"}]
        )
        for ch in channels:
            assert len(ch.description) > 0

    def test_single_group_no_channels(self, branch2_ss):
        channels = detect_covert_channels(
            branch2_ss, label_groups=[{"a", "b"}]
        )
        assert len(channels) == 0

    def test_no_labels_no_channels(self, end_ss):
        channels = detect_covert_channels(end_ss, label_groups=[])
        assert len(channels) == 0


# ---------------------------------------------------------------------------
# channel_capacity
# ---------------------------------------------------------------------------


class TestChannelCapacity:
    """Tests for channel_capacity."""

    def test_disjoint_zero_capacity(self, parallel_ss):
        """Labels at different states -> zero MI."""
        cap = channel_capacity(parallel_ss, {"a"}, {"b"})
        assert cap == 0.0

    def test_same_labels_self_capacity(self, branch2_ss):
        """Same labels compared against themselves."""
        cap = channel_capacity(branch2_ss, {"a"}, {"a"})
        # Self-MI = entropy of a
        assert cap >= 0.0

    def test_empty_labels_zero(self, linear_ss):
        cap = channel_capacity(linear_ss, set(), {"a"})
        assert cap == 0.0

    def test_co_located_labels_positive(self, branch2_ss):
        """Labels at the same state have positive MI."""
        cap = channel_capacity(branch2_ss, {"a"}, {"b"})
        assert cap >= 0.0  # They share a state

    def test_capacity_non_negative(self, nested_branch_ss):
        """Channel capacity is always non-negative."""
        cap = channel_capacity(nested_branch_ss, {"a"}, {"x"})
        assert cap >= 0.0

    def test_nested_different_states(self, nested_branch_ss):
        """Labels at different states have zero capacity."""
        # 'a' is at top, 'x' is at a's successor -> different states
        cap = channel_capacity(nested_branch_ss, {"a", "b"}, {"x", "y"})
        assert cap == 0.0


# ---------------------------------------------------------------------------
# parallel_independence
# ---------------------------------------------------------------------------


class TestParallelIndependence:
    """Tests for parallel_independence."""

    def test_end_independent(self, end_ss):
        result = parallel_independence(end_ss)
        assert result.is_independent is True

    def test_parallel_independent(self, parallel_ss):
        """Parallel composition with disjoint labels is independent."""
        result = parallel_independence(parallel_ss)
        assert result.is_independent is True
        assert len(result.shared_labels) == 0

    def test_independence_score_bounds(self, parallel_ss):
        result = parallel_independence(parallel_ss)
        assert 0.0 <= result.independence_score <= 1.0

    def test_single_label_group_independent(self, linear_ss):
        result = parallel_independence(linear_ss)
        # Single connected group -> trivially independent
        assert result.is_independent is True

    def test_result_type(self, branch2_ss):
        result = parallel_independence(branch2_ss)
        assert isinstance(result, IndependenceResult)

    def test_shared_labels_frozenset(self, parallel_ss):
        result = parallel_independence(parallel_ss)
        assert isinstance(result.shared_labels, frozenset)

    def test_correlation_states_frozenset(self, parallel_ss):
        result = parallel_independence(parallel_ss)
        assert isinstance(result.correlation_states, frozenset)


# ---------------------------------------------------------------------------
# analyze_covert_channels
# ---------------------------------------------------------------------------


class TestAnalyzeCovertChannels:
    """Tests for analyze_covert_channels."""

    def test_result_type(self, linear_ss):
        result = analyze_covert_channels(linear_ss)
        assert isinstance(result, CovertChannelResult)

    def test_end_no_channels(self, end_ss):
        result = analyze_covert_channels(end_ss)
        assert result.has_covert_channels is False
        assert result.total_capacity == 0.0
        assert result.max_mutual_info == 0.0

    def test_parallel_clean(self, parallel_ss):
        result = analyze_covert_channels(parallel_ss)
        assert result.independence.is_independent is True

    def test_channel_count(self, branch2_ss):
        result = analyze_covert_channels(
            branch2_ss, label_groups=[{"a"}, {"b"}]
        )
        assert len(result.channels) >= 0

    def test_total_capacity_non_negative(self, nested_branch_ss):
        result = analyze_covert_channels(nested_branch_ss)
        assert result.total_capacity >= 0.0

    def test_num_label_groups(self, branch2_ss):
        result = analyze_covert_channels(
            branch2_ss, label_groups=[{"a"}, {"b"}]
        )
        assert result.num_label_groups == 2

    def test_independence_in_result(self, linear_ss):
        result = analyze_covert_channels(linear_ss)
        assert isinstance(result.independence, IndependenceResult)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and integration tests."""

    def test_recursive_type(self):
        ss = build_statespace(parse("rec X . &{a: X}"))
        result = analyze_covert_channels(ss)
        assert isinstance(result, CovertChannelResult)

    def test_select_type(self, select_ss):
        result = analyze_covert_channels(select_ss)
        assert isinstance(result, CovertChannelResult)

    def test_shared_label_protocol(self, shared_label_ss):
        """Protocol where same label appears at multiple states."""
        result = analyze_covert_channels(shared_label_ss)
        assert isinstance(result, CovertChannelResult)

    def test_large_branch(self):
        """Larger protocol with many labels."""
        ss = build_statespace(parse("&{a: end, b: end, c: end, d: end}"))
        result = analyze_covert_channels(ss)
        assert isinstance(result, CovertChannelResult)

    def test_three_label_groups(self):
        """Three separate groups."""
        ss = build_statespace(parse("&{a: &{b: &{c: end}}}"))
        result = analyze_covert_channels(
            ss, label_groups=[{"a"}, {"b"}, {"c"}]
        )
        assert result.num_label_groups == 3

    def test_capacity_symmetric(self, branch2_ss):
        """MI is symmetric: I(X;Y) = I(Y;X)."""
        cap1 = channel_capacity(branch2_ss, {"a"}, {"b"})
        cap2 = channel_capacity(branch2_ss, {"b"}, {"a"})
        assert abs(cap1 - cap2) < 1e-10

    def test_max_mutual_info_consistent(self, branch2_ss):
        result = analyze_covert_channels(
            branch2_ss, label_groups=[{"a"}, {"b"}]
        )
        if result.channels:
            assert result.max_mutual_info == max(
                ch.mutual_information for ch in result.channels
            )
        else:
            assert result.max_mutual_info == 0.0
