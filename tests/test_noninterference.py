"""Tests for non-interference as lattice morphism (Step 89b)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.noninterference import (
    NonInterferenceResult,
    TransitionClassification,
    InformationFlowLattice,
    analyze_noninterference,
    check_noninterference,
    classify_transitions,
    information_flow_lattice,
    leakage_score,
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
def select2_ss():
    """Selection with 2 options: +{ok: end, err: end}."""
    return build_statespace(parse("+{ok: end, err: end}"))


@pytest.fixture
def high_low_ss():
    """Protocol with high selection then low branch:
    +{secret: &{public: end}, hidden: &{public: end}}
    Non-interfering because both high choices lead to same low view.
    """
    return build_statespace(parse("+{secret: &{public: end}, hidden: &{public: end}}"))


@pytest.fixture
def leaking_ss():
    """Protocol that leaks high info to low level:
    +{secret: &{a: end}, hidden: &{b: end}}
    Interfering because different high choices lead to different low labels.
    """
    return build_statespace(parse("+{secret: &{a: end}, hidden: &{b: end}}"))


@pytest.fixture
def mixed_ss():
    """Protocol with both high and low transitions at same state:
    &{low_read: +{high_ok: end, high_err: end}}
    """
    return build_statespace(parse("&{low_read: +{high_ok: end, high_err: end}}"))


@pytest.fixture
def parallel_ss():
    """Parallel protocol: (a.end || b.end)."""
    return build_statespace(parse("(&{a: end} || &{b: end})"))


# ---------------------------------------------------------------------------
# classify_transitions
# ---------------------------------------------------------------------------


class TestClassifyTransitions:
    """Tests for classify_transitions."""

    def test_empty_end(self, end_ss):
        cls = classify_transitions(end_ss, set(), set())
        assert len(cls.high) == 0
        assert len(cls.low) == 0
        assert len(cls.mixed) == 0

    def test_all_low(self, linear_ss):
        cls = classify_transitions(linear_ss, set(), {"a", "b"})
        assert len(cls.low) == 2
        assert len(cls.high) == 0

    def test_all_high(self, linear_ss):
        cls = classify_transitions(linear_ss, {"a", "b"}, set())
        assert len(cls.high) == 2
        assert len(cls.low) == 0

    def test_mixed_partition(self, linear_ss):
        cls = classify_transitions(linear_ss, {"a"}, {"b"})
        assert len(cls.high) == 1
        assert len(cls.low) == 1
        assert len(cls.mixed) == 0

    def test_unclassified_goes_to_mixed(self, linear_ss):
        cls = classify_transitions(linear_ss, set(), set())
        assert len(cls.mixed) == 2

    def test_branch2_partition(self, branch2_ss):
        cls = classify_transitions(branch2_ss, {"a"}, {"b"})
        assert len(cls.high) == 1
        assert len(cls.low) == 1

    def test_labels_stored(self, branch2_ss):
        cls = classify_transitions(branch2_ss, {"a"}, {"b"})
        assert cls.high_labels == frozenset({"a"})
        assert cls.low_labels == frozenset({"b"})

    def test_both_high_and_low_goes_to_mixed(self, linear_ss):
        """Label in both sets goes to mixed."""
        cls = classify_transitions(linear_ss, {"a"}, {"a"})
        # 'a' is in both, so goes to mixed
        mixed_labels = {label for _, label, _ in cls.mixed}
        assert "a" in mixed_labels


# ---------------------------------------------------------------------------
# information_flow_lattice
# ---------------------------------------------------------------------------


class TestInformationFlowLattice:
    """Tests for information_flow_lattice."""

    def test_end_all_low(self, end_ss):
        fl = information_flow_lattice(end_ss, set(), set())
        assert fl.high_state == 1
        assert fl.low_state == 0
        # end state has no transitions, maps to low
        for state, level in fl.mapping.items():
            assert level == 0

    def test_high_labels_map_states(self, select2_ss):
        fl = information_flow_lattice(select2_ss, {"ok", "err"}, set())
        # The initial state has ok/err transitions -> High
        assert fl.mapping[select2_ss.top] == 1

    def test_low_only_maps_low(self, linear_ss):
        fl = information_flow_lattice(linear_ss, set(), {"a", "b"})
        # No high labels, so all states map to Low
        for state, level in fl.mapping.items():
            assert level == 0

    def test_default_uses_selections(self, select2_ss):
        fl = information_flow_lattice(select2_ss)
        # Selections are high by default
        assert fl.mapping[select2_ss.top] == 1

    def test_mapping_covers_all_states(self, branch2_ss):
        fl = information_flow_lattice(branch2_ss, {"a"}, {"b"})
        assert set(fl.mapping.keys()) == branch2_ss.states


# ---------------------------------------------------------------------------
# check_noninterference
# ---------------------------------------------------------------------------


class TestCheckNoninterference:
    """Tests for check_noninterference."""

    def test_end_is_noninterfering(self, end_ss):
        assert check_noninterference(end_ss, set(), set()) is True

    def test_linear_is_noninterfering(self, linear_ss):
        assert check_noninterference(linear_ss, {"a"}, {"b"}) is True

    def test_no_high_is_noninterfering(self, branch2_ss):
        assert check_noninterference(branch2_ss, set(), {"a", "b"}) is True

    def test_same_low_continuation_noninterfering(self, high_low_ss):
        """Both high choices lead to same low behavior."""
        assert check_noninterference(
            high_low_ss, {"secret", "hidden"}, {"public"}
        ) is True

    def test_different_low_continuation_interfering(self, leaking_ss):
        """Different high choices lead to different low labels."""
        assert check_noninterference(
            leaking_ss, {"secret", "hidden"}, {"a", "b"}
        ) is False

    def test_single_high_choice_noninterfering(self):
        """Only one high option -> no interference possible."""
        ss = build_statespace(parse("+{secret: &{a: end}}"))
        assert check_noninterference(ss, {"secret"}, {"a"}) is True

    def test_branch_all_low_noninterfering(self, branch2_ss):
        """All transitions low -> trivially non-interfering."""
        assert check_noninterference(branch2_ss, set(), {"a", "b"}) is True

    def test_parallel_noninterfering(self, parallel_ss):
        """Parallel branches with independent labels."""
        assert check_noninterference(parallel_ss, {"a"}, {"b"}) is True


# ---------------------------------------------------------------------------
# leakage_score
# ---------------------------------------------------------------------------


class TestLeakageScore:
    """Tests for leakage_score."""

    def test_end_zero_leakage(self, end_ss):
        assert leakage_score(end_ss, set(), set()) == 0.0

    def test_noninterfering_zero_leakage(self, high_low_ss):
        score = leakage_score(high_low_ss, {"secret", "hidden"}, {"public"})
        assert score == 0.0

    def test_interfering_positive_leakage(self, leaking_ss):
        score = leakage_score(leaking_ss, {"secret", "hidden"}, {"a", "b"})
        assert score > 0.0
        assert score <= 1.0

    def test_full_leakage_is_one(self, leaking_ss):
        """When all high-choice states leak, score = 1.0."""
        score = leakage_score(leaking_ss, {"secret", "hidden"}, {"a", "b"})
        assert score == 1.0

    def test_no_high_transitions_zero_leakage(self, branch2_ss):
        score = leakage_score(branch2_ss, set(), {"a", "b"})
        assert score == 0.0

    def test_leakage_between_zero_and_one(self):
        """Mixed protocol: some states leak, some don't."""
        # +{h1: &{lo: +{h2: &{lo: end}, h3: &{lo: end}}},
        #   h4: &{lo: +{h2: &{lo: end}, h3: &{lo: end}}}}
        # The outer high choice doesn't leak (same continuation),
        # inner high choice doesn't leak either (same lo)
        ss = build_statespace(parse(
            "+{h1: &{lo: end}, h4: &{lo: end}}"
        ))
        score = leakage_score(ss, {"h1", "h4"}, {"lo"})
        assert score == 0.0


# ---------------------------------------------------------------------------
# analyze_noninterference
# ---------------------------------------------------------------------------


class TestAnalyzeNoninterference:
    """Tests for analyze_noninterference."""

    def test_result_type(self, linear_ss):
        result = analyze_noninterference(linear_ss, {"a"}, {"b"})
        assert isinstance(result, NonInterferenceResult)

    def test_noninterfering_result(self, high_low_ss):
        result = analyze_noninterference(
            high_low_ss, {"secret", "hidden"}, {"public"}
        )
        assert result.is_noninterfering is True
        assert result.leakage == 0.0
        assert len(result.leaking_states) == 0
        assert result.witness_paths is None

    def test_interfering_result(self, leaking_ss):
        result = analyze_noninterference(
            leaking_ss, {"secret", "hidden"}, {"a", "b"}
        )
        assert result.is_noninterfering is False
        assert result.leakage > 0.0
        assert len(result.leaking_states) > 0
        assert result.witness_paths is not None

    def test_transition_counts(self, leaking_ss):
        result = analyze_noninterference(
            leaking_ss, {"secret", "hidden"}, {"a", "b"}
        )
        assert result.num_high_transitions == 2
        assert result.num_low_transitions == 2

    def test_classification_in_result(self, branch2_ss):
        result = analyze_noninterference(branch2_ss, {"a"}, {"b"})
        assert isinstance(result.classification, TransitionClassification)

    def test_flow_lattice_in_result(self, branch2_ss):
        result = analyze_noninterference(branch2_ss, {"a"}, {"b"})
        assert isinstance(result.flow_lattice, InformationFlowLattice)

    def test_end_type(self, end_ss):
        result = analyze_noninterference(end_ss, set(), set())
        assert result.is_noninterfering is True
        assert result.num_high_transitions == 0
        assert result.num_low_transitions == 0

    def test_witness_has_two_paths(self, leaking_ss):
        result = analyze_noninterference(
            leaking_ss, {"secret", "hidden"}, {"a", "b"}
        )
        assert result.witness_paths is not None
        path1, path2 = result.witness_paths
        assert len(path1) > 0
        assert len(path2) > 0


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and integration tests."""

    def test_recursive_type(self):
        """Recursive type: rec X . &{a: X}."""
        ss = build_statespace(parse("rec X . &{a: X}"))
        result = analyze_noninterference(ss, set(), {"a"})
        assert result.is_noninterfering is True

    def test_nested_high_low(self):
        """Nested: high choice then low branch then high choice."""
        ss = build_statespace(parse(
            "+{h1: &{lo: +{h2: end, h3: end}}, h4: &{lo: +{h2: end, h3: end}}}"
        ))
        result = analyze_noninterference(
            ss, {"h1", "h4", "h2", "h3"}, {"lo"}
        )
        assert result.is_noninterfering is True

    def test_smtp_like_protocol(self):
        """SMTP-like: connect . +{auth: &{ok: send . end, err: end}}."""
        ss = build_statespace(parse(
            "&{connect: +{auth: &{ok: &{send: end}, err: end}}}"
        ))
        result = analyze_noninterference(
            ss, {"auth"}, {"connect", "ok", "err", "send"}
        )
        # Single auth option -> no interference
        assert result.is_noninterfering is True

    def test_all_labels_same_set(self, branch2_ss):
        """All labels in both sets -> mixed classification."""
        cls = classify_transitions(branch2_ss, {"a", "b"}, {"a", "b"})
        assert len(cls.mixed) == 2

    def test_disjoint_labels_full_partition(self, branch2_ss):
        cls = classify_transitions(branch2_ss, {"a"}, {"b"})
        total = len(cls.high) + len(cls.low) + len(cls.mixed)
        assert total == len(branch2_ss.transitions)
