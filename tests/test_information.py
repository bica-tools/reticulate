"""Tests for the information-theoretic session type module (Step 60b)."""

from __future__ import annotations

import math

import pytest

from reticulate import build_statespace, parse
from reticulate.information import (
    InformationResult,
    analyze_information,
    branching_entropy,
    channel_capacity,
    information_density,
    is_deterministic,
    label_entropy,
    path_entropy,
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
    """Linear chain: a . b . end (no branching)."""
    return build_statespace(parse("&{a: &{b: end}}"))


@pytest.fixture
def branch2_ss():
    """Simple branch with 2 options."""
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def branch3_ss():
    """Branch with 3 options."""
    return build_statespace(parse("&{a: end, b: end, c: end}"))


@pytest.fixture
def nested_branch_ss():
    """Nested branches: different path lengths."""
    return build_statespace(parse("&{a: &{x: end, y: end}, b: end}"))


@pytest.fixture
def select_ss():
    """Selection with 2 options."""
    return build_statespace(parse("+{ok: end, err: end}"))


@pytest.fixture
def recursive_ss():
    """Recursive type: rec X . &{next: X, done: end}."""
    return build_statespace(parse("rec X . &{next: X, done: end}"))


@pytest.fixture
def parallel_ss():
    """Parallel composition."""
    return build_statespace(parse("(&{a: end} || &{b: end})"))


@pytest.fixture
def complex_ss():
    """More complex type with branching and nesting."""
    return build_statespace(
        parse("&{open: +{OK: &{read: end, write: end}, ERROR: end}}")
    )


# ---------------------------------------------------------------------------
# branching_entropy tests
# ---------------------------------------------------------------------------


class TestBranchingEntropy:
    def test_end_type_zero(self, end_ss):
        assert branching_entropy(end_ss) == 0.0

    def test_linear_chain_zero(self, linear_ss):
        """Linear chain: each state has exactly 1 outgoing transition => H=0."""
        assert branching_entropy(linear_ss) == 0.0

    def test_branch2_positive(self, branch2_ss):
        """2-way branch has H = log2(2) = 1.0 at the branch state."""
        h = branching_entropy(branch2_ss)
        assert h > 0.0
        # Only one non-terminal state (the branch), so average = log2(2) = 1.0
        assert math.isclose(h, 1.0, rel_tol=1e-9)

    def test_branch3_higher(self, branch3_ss):
        """3-way branch has H = log2(3) > log2(2)."""
        h = branching_entropy(branch3_ss)
        assert h > branching_entropy(build_statespace(parse("&{a: end, b: end}")))
        assert math.isclose(h, math.log2(3), rel_tol=1e-9)

    def test_nested_branch_averaged(self, nested_branch_ss):
        """Nested branches: average of top (2 choices) and inner (2 choices)."""
        h = branching_entropy(nested_branch_ss)
        assert h > 0.0
        # Top has 2 outgoing, inner branch has 2 outgoing; average = 1.0
        assert math.isclose(h, 1.0, rel_tol=1e-9)

    def test_select_positive(self, select_ss):
        """Selection also has entropy > 0."""
        assert branching_entropy(select_ss) > 0.0

    def test_recursive_positive(self, recursive_ss):
        """Recursive type with branch has entropy > 0."""
        assert branching_entropy(recursive_ss) > 0.0

    def test_nonnegative(self, complex_ss):
        assert branching_entropy(complex_ss) >= 0.0


# ---------------------------------------------------------------------------
# path_entropy tests
# ---------------------------------------------------------------------------


class TestPathEntropy:
    def test_end_type_zero(self, end_ss):
        assert path_entropy(end_ss) == 0.0

    def test_single_path_zero(self, linear_ss):
        """Only one path => entropy = 0."""
        assert path_entropy(linear_ss) == 0.0

    def test_same_length_paths_zero(self, branch2_ss):
        """Two paths both of length 1 => single length bucket => H = 0."""
        assert path_entropy(branch2_ss) == 0.0

    def test_different_lengths_positive(self, nested_branch_ss):
        """Paths of length 1 and 2 => H > 0."""
        h = path_entropy(nested_branch_ss)
        assert h > 0.0

    def test_nonnegative(self, complex_ss):
        assert path_entropy(complex_ss) >= 0.0


# ---------------------------------------------------------------------------
# label_entropy tests
# ---------------------------------------------------------------------------


class TestLabelEntropy:
    def test_end_type_zero(self, end_ss):
        assert label_entropy(end_ss) == 0.0

    def test_single_label_zero(self):
        """Only one distinct label => H = 0."""
        ss = build_statespace(parse("&{a: &{a: end}}"))
        assert label_entropy(ss) == 0.0

    def test_two_distinct_labels(self, branch2_ss):
        """Two distinct labels each appearing once => H = 1.0."""
        h = label_entropy(branch2_ss)
        assert math.isclose(h, 1.0, rel_tol=1e-9)

    def test_three_labels(self, branch3_ss):
        """Three labels: H = log2(3)."""
        h = label_entropy(branch3_ss)
        assert math.isclose(h, math.log2(3), rel_tol=1e-9)

    def test_uneven_distribution(self, nested_branch_ss):
        """Labels: a, b, x, y — not all same frequency necessarily."""
        h = label_entropy(nested_branch_ss)
        assert h > 0.0

    def test_nonnegative(self, complex_ss):
        assert label_entropy(complex_ss) >= 0.0


# ---------------------------------------------------------------------------
# channel_capacity tests
# ---------------------------------------------------------------------------


class TestChannelCapacity:
    def test_end_type(self, end_ss):
        """Single-state end: 1 path (trivial), log2(1) = 0."""
        assert channel_capacity(end_ss) == 0.0

    def test_linear_single_path(self, linear_ss):
        """One path => log2(1) = 0."""
        assert channel_capacity(linear_ss) == 0.0

    def test_branch2_one_bit(self, branch2_ss):
        """Two paths => log2(2) = 1.0."""
        cap = channel_capacity(branch2_ss)
        assert math.isclose(cap, 1.0, rel_tol=1e-9)

    def test_branch3_log3(self, branch3_ss):
        """Three paths => log2(3)."""
        cap = channel_capacity(branch3_ss)
        assert math.isclose(cap, math.log2(3), rel_tol=1e-9)

    def test_nested_increases(self, nested_branch_ss):
        """Nested branch has 3 paths => log2(3)."""
        cap = channel_capacity(nested_branch_ss)
        assert math.isclose(cap, math.log2(3), rel_tol=1e-9)

    def test_nonnegative(self, complex_ss):
        assert channel_capacity(complex_ss) >= 0.0


# ---------------------------------------------------------------------------
# is_deterministic tests
# ---------------------------------------------------------------------------


class TestIsDeterministic:
    def test_end_deterministic(self, end_ss):
        assert is_deterministic(end_ss) is True

    def test_linear_deterministic(self, linear_ss):
        assert is_deterministic(linear_ss) is True

    def test_branch_deterministic(self, branch2_ss):
        """Branch with distinct labels is deterministic."""
        assert is_deterministic(branch2_ss) is True

    def test_select_deterministic(self, select_ss):
        assert is_deterministic(select_ss) is True

    def test_recursive_deterministic(self, recursive_ss):
        assert is_deterministic(recursive_ss) is True

    def test_parallel_may_have_duplicate_labels(self, parallel_ss):
        """Parallel composition might introduce duplicate labels at a state."""
        # This is still deterministic if labels are distinct across factors
        result = is_deterministic(parallel_ss)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# information_density tests
# ---------------------------------------------------------------------------


class TestInformationDensity:
    def test_end_type_zero(self, end_ss):
        assert information_density(end_ss) == 0.0

    def test_nonnegative(self, complex_ss):
        assert information_density(complex_ss) >= 0.0

    def test_branch2(self, branch2_ss):
        """2 paths, 2 transitions => 1.0 / 2 = 0.5."""
        d = information_density(branch2_ss)
        assert math.isclose(d, 0.5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# analyze_information tests
# ---------------------------------------------------------------------------


class TestAnalyzeInformation:
    def test_returns_information_result(self, branch2_ss):
        result = analyze_information(branch2_ss)
        assert isinstance(result, InformationResult)

    def test_frozen(self, branch2_ss):
        result = analyze_information(branch2_ss)
        with pytest.raises(AttributeError):
            result.branching_entropy = 42.0  # type: ignore[misc]

    def test_end_type_all_zero(self, end_ss):
        result = analyze_information(end_ss)
        assert result.branching_entropy == 0.0
        assert result.path_entropy == 0.0
        assert result.label_entropy == 0.0
        assert result.channel_capacity == 0.0
        assert result.information_density == 0.0
        assert result.is_deterministic is True
        assert result.state_complexity >= 1
        assert result.transition_complexity == 0

    def test_state_complexity(self, branch2_ss):
        result = analyze_information(branch2_ss)
        assert result.state_complexity == len(branch2_ss.states)

    def test_transition_complexity(self, branch2_ss):
        result = analyze_information(branch2_ss)
        assert result.transition_complexity == len(branch2_ss.transitions)

    def test_consistency(self, complex_ss):
        """Fields in InformationResult are consistent with standalone calls."""
        result = analyze_information(complex_ss)
        assert math.isclose(result.branching_entropy, branching_entropy(complex_ss))
        assert math.isclose(result.path_entropy, path_entropy(complex_ss))
        assert math.isclose(result.label_entropy, label_entropy(complex_ss))
        assert math.isclose(result.channel_capacity, channel_capacity(complex_ss))
        assert result.is_deterministic == is_deterministic(complex_ss)
        assert math.isclose(result.information_density, information_density(complex_ss))


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------


class TestBenchmarks:
    def test_all_benchmarks_nonnegative(self):
        """All benchmark protocols produce non-negative entropy values."""
        from tests.benchmarks.protocols import BENCHMARKS

        for bp in BENCHMARKS[:10]:  # first 10 to keep test fast
            ss = build_statespace(parse(bp.type_string))
            result = analyze_information(ss)
            assert result.branching_entropy >= 0.0, bp.name
            assert result.path_entropy >= 0.0, bp.name
            assert result.label_entropy >= 0.0, bp.name
            assert result.channel_capacity >= 0.0, bp.name
            assert result.information_density >= 0.0, bp.name
            assert result.state_complexity > 0, bp.name
            assert result.transition_complexity >= 0, bp.name
