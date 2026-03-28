"""Tests for beauty as lattice symmetry (Step 202)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.aesthetics import (
    AestheticProfile,
    analyze_aesthetics,
    beauty_score,
    classify_aesthetic,
    compare_beauty,
    golden_ratio_check,
    most_beautiful,
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
    """Diamond lattice: &{a: &{c: end}, b: &{c: end}}."""
    return build_statespace(parse("&{a: &{c: end}, b: &{c: end}}"))


@pytest.fixture
def recursive_ss():
    return build_statespace(parse("rec X . &{next: X, done: end}"))


@pytest.fixture
def pendular_ss():
    """A pendular protocol with alternating select/branch."""
    return build_statespace(parse(
        "&{request: +{accept: &{process: end}, reject: end}}"
    ))


@pytest.fixture
def symmetric_ss():
    """A symmetric branch with equal children."""
    return build_statespace(parse("&{a: &{c: end}, b: &{d: end}}"))


@pytest.fixture
def complex_ss():
    """A more complex protocol."""
    return build_statespace(parse(
        "&{init: +{ok: &{read: end, write: end}, err: end}}"
    ))


# ---------------------------------------------------------------------------
# Aesthetic profile tests
# ---------------------------------------------------------------------------


class TestAnalyzeAesthetics:
    """Test the analyze_aesthetics function."""

    def test_returns_profile(self, simple_branch_ss):
        result = analyze_aesthetics(simple_branch_ss)
        assert isinstance(result, AestheticProfile)

    def test_symmetry_between_0_and_1(self, diamond_ss):
        result = analyze_aesthetics(diamond_ss)
        assert 0.0 <= result.symmetry <= 1.0

    def test_balance_between_0_and_1(self, diamond_ss):
        result = analyze_aesthetics(diamond_ss)
        assert 0.0 <= result.balance <= 1.0

    def test_complexity_non_negative(self, diamond_ss):
        result = analyze_aesthetics(diamond_ss)
        assert result.complexity >= 0.0

    def test_order_between_0_and_1(self, diamond_ss):
        result = analyze_aesthetics(diamond_ss)
        assert 0.0 <= result.order <= 1.0

    def test_harmony_between_0_and_1(self, diamond_ss):
        result = analyze_aesthetics(diamond_ss)
        assert 0.0 <= result.harmony <= 1.0

    def test_beauty_score_non_negative(self, diamond_ss):
        result = analyze_aesthetics(diamond_ss)
        assert result.beauty_score >= 0.0

    def test_classification_valid(self, diamond_ss):
        result = analyze_aesthetics(diamond_ss)
        assert result.classification in (
            "sublime", "beautiful", "pleasant", "neutral", "ugly"
        )

    def test_end_type_classification(self, end_ss):
        """End type has minimal structure — should be neutral or ugly."""
        result = analyze_aesthetics(end_ss)
        assert result.classification in ("neutral", "ugly")


# ---------------------------------------------------------------------------
# Beauty score tests
# ---------------------------------------------------------------------------


class TestBeautyScore:
    """Test the beauty_score convenience function."""

    def test_returns_float(self, simple_branch_ss):
        score = beauty_score(simple_branch_ss)
        assert isinstance(score, float)

    def test_non_negative(self, simple_branch_ss):
        assert beauty_score(simple_branch_ss) >= 0.0

    def test_end_is_low(self, end_ss):
        assert beauty_score(end_ss) <= 0.5

    def test_diamond_higher_than_end(self, end_ss, diamond_ss):
        """A diamond lattice should be more beautiful than bare end."""
        assert beauty_score(diamond_ss) >= beauty_score(end_ss)


# ---------------------------------------------------------------------------
# Compare beauty tests
# ---------------------------------------------------------------------------


class TestCompareBeauty:
    """Test compare_beauty function."""

    def test_returns_valid_value(self, end_ss, diamond_ss):
        result = compare_beauty(end_ss, diamond_ss)
        assert result in (-1, 0, 1)

    def test_same_type_equal(self, end_ss):
        assert compare_beauty(end_ss, end_ss) == 0

    def test_antisymmetric(self, end_ss, diamond_ss):
        """If compare_beauty(a, b) = 1, then compare_beauty(b, a) = -1."""
        ab = compare_beauty(end_ss, diamond_ss)
        ba = compare_beauty(diamond_ss, end_ss)
        if ab != 0:
            assert ab == -ba


# ---------------------------------------------------------------------------
# Most beautiful tests
# ---------------------------------------------------------------------------


class TestMostBeautiful:
    """Test most_beautiful function."""

    def test_returns_name(self, end_ss, diamond_ss):
        entries = {"end": end_ss, "diamond": diamond_ss}
        result = most_beautiful(entries)
        assert result in entries

    def test_single_entry(self, end_ss):
        result = most_beautiful({"only": end_ss})
        assert result == "only"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            most_beautiful({})


# ---------------------------------------------------------------------------
# Golden ratio tests
# ---------------------------------------------------------------------------


class TestGoldenRatioCheck:
    """Test golden_ratio_check function."""

    def test_returns_bool(self, diamond_ss):
        result = golden_ratio_check(diamond_ss)
        assert isinstance(result, bool)

    def test_end_no_golden_ratio(self, end_ss):
        """Single-state type cannot have golden ratio between levels."""
        assert golden_ratio_check(end_ss) is False


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassifyAesthetic:
    """Test the classify_aesthetic convenience function."""

    def test_returns_string(self, simple_branch_ss):
        result = classify_aesthetic(simple_branch_ss)
        assert isinstance(result, str)

    def test_valid_classification(self, simple_branch_ss):
        result = classify_aesthetic(simple_branch_ss)
        assert result in ("sublime", "beautiful", "pleasant", "neutral", "ugly")

    def test_consistent_with_profile(self, diamond_ss):
        profile = analyze_aesthetics(diamond_ss)
        cls = classify_aesthetic(diamond_ss)
        assert cls == profile.classification
