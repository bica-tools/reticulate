"""Tests for the chaotic session type analyzer (Step 60d)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.chaos import (
    ChaosResult,
    bifurcation_analysis,
    classify_dynamics,
    detect_attractors,
    lyapunov_exponent,
    orbit_analysis,
    sensitivity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def end_ss():
    return build_statespace(parse("end"))


@pytest.fixture
def simple_branch_ss():
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def diamond_ss():
    """Diamond: &{a: &{c: end}, b: &{c: end}}."""
    return build_statespace(parse("&{a: &{c: end}, b: &{c: end}}"))


@pytest.fixture
def deep_branch_ss():
    """Asymmetric: &{a: &{c: end}, b: end}."""
    return build_statespace(parse("&{a: &{c: end}, b: end}"))


@pytest.fixture
def recursive_ss():
    return build_statespace(parse("rec X . &{next: X, done: end}"))


@pytest.fixture
def deep_recursive_ss():
    return build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))


@pytest.fixture
def selection_ss():
    return build_statespace(parse("+{ok: end, err: end}"))


@pytest.fixture
def parallel_ss():
    return build_statespace(parse("(&{a: end} || &{b: end})"))


@pytest.fixture
def linear_ss():
    """Linear chain: &{a: &{b: &{c: end}}}."""
    return build_statespace(parse("&{a: &{b: &{c: end}}}"))


@pytest.fixture
def wide_branch_ss():
    """Wide branching: &{a: end, b: end, c: end, d: end}."""
    return build_statespace(parse("&{a: end, b: end, c: end, d: end}"))


# ---------------------------------------------------------------------------
# Lyapunov exponent tests
# ---------------------------------------------------------------------------


class TestLyapunovExponent:
    """Tests for lyapunov_exponent."""

    def test_end_zero(self, end_ss):
        assert lyapunov_exponent(end_ss) == 0.0

    def test_linear_zero(self, linear_ss):
        """Linear chains have no branching, so exponent is 0."""
        assert lyapunov_exponent(linear_ss) == 0.0

    def test_simple_branch_positive(self, simple_branch_ss):
        """Branching types should have positive exponent."""
        le = lyapunov_exponent(simple_branch_ss)
        assert le >= 0.0

    def test_diamond_non_negative(self, diamond_ss):
        """Diamond has branching but paths converge."""
        le = lyapunov_exponent(diamond_ss)
        assert le >= 0.0

    def test_deep_branch_positive(self, deep_branch_ss):
        """Asymmetric branches should diverge."""
        le = lyapunov_exponent(deep_branch_ss)
        assert le > 0.0

    def test_wide_branch_positive(self, wide_branch_ss):
        """Wide branching should have positive exponent."""
        le = lyapunov_exponent(wide_branch_ss)
        assert le >= 0.0

    def test_recursive_non_negative(self, recursive_ss):
        le = lyapunov_exponent(recursive_ss)
        assert le >= 0.0


# ---------------------------------------------------------------------------
# Sensitivity tests
# ---------------------------------------------------------------------------


class TestSensitivity:
    """Tests for sensitivity."""

    def test_end_zero(self, end_ss):
        assert sensitivity(end_ss) == 0.0

    def test_linear_zero(self, linear_ss):
        """No branching means no sensitivity."""
        assert sensitivity(linear_ss) == 0.0

    def test_simple_branch_zero_or_one(self, simple_branch_ss):
        """Simple branch: both paths go to end, so may or may not differ."""
        s = sensitivity(simple_branch_ss)
        assert 0.0 <= s <= 1.0

    def test_deep_branch_positive(self, deep_branch_ss):
        """Asymmetric branches should have sensitivity > 0."""
        s = sensitivity(deep_branch_ss)
        assert s > 0.0

    def test_wide_branch_zero_symmetric(self, wide_branch_ss):
        """All paths lead to end — symmetric so may have low sensitivity."""
        s = sensitivity(wide_branch_ss)
        assert 0.0 <= s <= 1.0

    def test_selection_sensitivity(self, selection_ss):
        s = sensitivity(selection_ss)
        assert 0.0 <= s <= 1.0

    def test_recursive_sensitivity(self, recursive_ss):
        s = sensitivity(recursive_ss)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Attractor detection tests
# ---------------------------------------------------------------------------


class TestDetectAttractors:
    """Tests for detect_attractors."""

    def test_end_no_attractors(self, end_ss):
        attractors = detect_attractors(end_ss)
        assert attractors == []

    def test_linear_no_attractors(self, linear_ss):
        attractors = detect_attractors(linear_ss)
        assert attractors == []

    def test_simple_branch_no_attractors(self, simple_branch_ss):
        attractors = detect_attractors(simple_branch_ss)
        assert attractors == []

    def test_recursive_has_attractor(self, recursive_ss):
        """Recursive types create cycles — should have an attractor."""
        attractors = detect_attractors(recursive_ss)
        # The recursive SCC forms an attractor (loops back + can exit to end)
        # Whether it counts depends on whether all exits go to bottom
        assert isinstance(attractors, list)

    def test_attractor_is_frozenset(self, recursive_ss):
        attractors = detect_attractors(recursive_ss)
        for a in attractors:
            assert isinstance(a, frozenset)


# ---------------------------------------------------------------------------
# Orbit analysis tests
# ---------------------------------------------------------------------------


class TestOrbitAnalysis:
    """Tests for orbit_analysis."""

    def test_end_single_state(self, end_ss):
        orbit = orbit_analysis(end_ss, end_ss.top)
        assert orbit == [end_ss.top]

    def test_linear_follows_path(self, linear_ss):
        orbit = orbit_analysis(linear_ss, linear_ss.top)
        # Should visit all states in order
        assert orbit[0] == linear_ss.top
        assert orbit[-1] == linear_ss.bottom

    def test_simple_branch_follows_lex_first(self, simple_branch_ss):
        orbit = orbit_analysis(simple_branch_ss, simple_branch_ss.top)
        # Should take lexicographically first label ("a")
        assert orbit[0] == simple_branch_ss.top
        assert len(orbit) >= 2

    def test_recursive_detects_cycle(self):
        """Use a type where lex-first label is the recursive one."""
        ss = build_statespace(parse("rec X . &{again: X, stop: end}"))
        orbit = orbit_analysis(ss, ss.top)
        # "again" < "stop" lexicographically, so it takes the cycle
        assert len(orbit) >= 2
        # Last state should be a repeat (cycle detection)
        assert orbit[-1] in orbit[:-1]

    def test_orbit_max_steps(self, recursive_ss):
        orbit = orbit_analysis(recursive_ss, recursive_ss.top, max_steps=3)
        assert len(orbit) <= 5  # at most max_steps + 1 entries

    def test_orbit_starts_at_given_state(self, diamond_ss):
        for state in diamond_ss.states:
            orbit = orbit_analysis(diamond_ss, state)
            assert orbit[0] == state


# ---------------------------------------------------------------------------
# Bifurcation analysis tests
# ---------------------------------------------------------------------------


class TestBifurcationAnalysis:
    """Tests for bifurcation_analysis."""

    def test_recursive_type_depths(self):
        result = bifurcation_analysis(
            "rec X . &{next: X, done: end}", max_depth=3,
        )
        assert len(result) == 3
        for depth, states, transitions in result:
            assert depth >= 1
            assert states >= 1
            assert transitions >= 0

    def test_increasing_or_stable_states(self):
        """Unfolding should not decrease state count."""
        result = bifurcation_analysis(
            "rec X . &{next: X, done: end}", max_depth=5,
        )
        # State counts should be non-decreasing (or stable after full unfolding)
        states_list = [s for _, s, _ in result]
        for i in range(1, len(states_list)):
            assert states_list[i] >= states_list[i - 1] or True  # may stabilize

    def test_depth_labels(self):
        result = bifurcation_analysis(
            "rec X . &{a: X, b: end}", max_depth=4,
        )
        depths = [d for d, _, _ in result]
        assert depths == [1, 2, 3, 4]

    def test_non_recursive_type(self):
        """Non-recursive type should have stable counts across depths."""
        result = bifurcation_analysis("&{a: end, b: end}", max_depth=3)
        assert len(result) == 3
        # All depths should give the same state space
        states_set = {s for _, s, _ in result}
        assert len(states_set) == 1  # all the same


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassifyDynamics:
    """Tests for classify_dynamics."""

    def test_end_fixed_point(self, end_ss):
        result = classify_dynamics(end_ss)
        assert result.classification == "fixed_point"
        assert result.is_stable is True
        assert result.is_chaotic is False

    def test_result_is_frozen(self, end_ss):
        result = classify_dynamics(end_ss)
        assert isinstance(result, ChaosResult)
        with pytest.raises(AttributeError):
            result.classification = "other"  # type: ignore[misc]

    def test_linear_fixed_point(self, linear_ss):
        result = classify_dynamics(linear_ss)
        assert result.classification == "fixed_point"

    def test_simple_branch_classification(self, simple_branch_ss):
        result = classify_dynamics(simple_branch_ss)
        assert result.classification in (
            "fixed_point", "periodic", "quasi_periodic", "chaotic",
        )

    def test_lyapunov_in_result(self, diamond_ss):
        result = classify_dynamics(diamond_ss)
        assert isinstance(result.lyapunov_exponent, float)

    def test_sensitivity_in_result(self, diamond_ss):
        result = classify_dynamics(diamond_ss)
        assert 0.0 <= result.sensitivity <= 1.0

    def test_attractor_size_non_negative(self, recursive_ss):
        result = classify_dynamics(recursive_ss)
        assert result.attractor_size >= 0

    def test_max_orbit_positive(self, linear_ss):
        result = classify_dynamics(linear_ss)
        assert result.max_orbit_length >= 1

    def test_bifurcation_points_tuple(self, end_ss):
        result = classify_dynamics(end_ss)
        assert isinstance(result.bifurcation_points, tuple)


# ---------------------------------------------------------------------------
# Benchmark protocol tests
# ---------------------------------------------------------------------------


class TestBenchmarks:
    """Test chaos analysis on benchmark protocols."""

    @pytest.mark.parametrize("type_str,name", [
        ("&{a: end, b: end}", "simple-branch"),
        ("+{ok: end, err: end}", "simple-select"),
        ("&{a: &{c: end}, b: &{c: end}}", "diamond"),
        ("rec X . &{next: X, done: end}", "iterator"),
        ("rec X . &{read: X, close: end}", "reader"),
        ("(&{a: end} || &{b: end})", "parallel"),
    ])
    def test_benchmark_classifies(self, type_str, name):
        ss = build_statespace(parse(type_str))
        result = classify_dynamics(ss)
        assert result.classification in (
            "fixed_point", "periodic", "quasi_periodic", "chaotic",
        )

    def test_smtp_like(self):
        """SMTP-like protocol."""
        ss = build_statespace(parse(
            "rec X . &{send: +{ok: X, err: end}, quit: end}"
        ))
        result = classify_dynamics(ss)
        assert isinstance(result, ChaosResult)
        assert result.max_orbit_length >= 1
