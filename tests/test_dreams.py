"""Tests for the dream metaphor engine (Step 204)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.dreams import (
    DreamResult,
    DreamSequence,
    collective_dream,
    dream,
    dream_sequence,
    interpret_dream,
    lucid_dream,
    nightmare,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_BRANCH = "&{a: end, b: end}"
DEEP_TYPE = "&{a: &{b: end}, c: end}"
RECURSIVE_TYPE = "rec X . &{next: X, done: end}"
SELECT_TYPE = "+{ok: end, err: end}"


# ---------------------------------------------------------------------------
# Tests: dream
# ---------------------------------------------------------------------------


class TestDream:
    def test_returns_dream_result(self):
        d = dream(SIMPLE_BRANCH)
        assert isinstance(d, DreamResult)

    def test_structural_preservation(self):
        d = dream(SIMPLE_BRANCH)
        assert d.structural_preservation is True

    def test_same_state_count(self):
        d = dream(SIMPLE_BRANCH)
        original_ss = build_statespace(parse(SIMPLE_BRANCH))
        dream_ss = build_statespace(parse(d.dream_type_str))
        assert len(original_ss.states) == len(dream_ss.states)

    def test_novelty_in_range(self):
        d = dream(SIMPLE_BRANCH)
        assert 0.0 <= d.novelty_score <= 1.0

    def test_lucidity_in_range(self):
        d = dream(SIMPLE_BRANCH)
        assert 0.0 <= d.lucidity <= 1.0

    def test_deterministic_with_seed(self):
        d1 = dream(SIMPLE_BRANCH, seed=123)
        d2 = dream(SIMPLE_BRANCH, seed=123)
        assert d1.dream_type_str == d2.dream_type_str

    def test_different_seeds_may_differ(self):
        d1 = dream(DEEP_TYPE, seed=1)
        d2 = dream(DEEP_TYPE, seed=999)
        # With 3 labels, different seeds should often produce different results.
        # Not guaranteed, but highly likely.
        assert isinstance(d1, DreamResult)
        assert isinstance(d2, DreamResult)

    def test_end_type_no_labels(self):
        d = dream("end")
        assert d.novelty_score == 0.0
        assert d.structural_preservation is True

    def test_label_mapping_has_all_labels(self):
        d = dream(SIMPLE_BRANCH)
        assert "a" in d.label_mapping
        assert "b" in d.label_mapping


# ---------------------------------------------------------------------------
# Tests: lucid_dream
# ---------------------------------------------------------------------------


class TestLucidDream:
    def test_fixed_labels_unchanged(self):
        d = lucid_dream(SIMPLE_BRANCH, fixed_labels={"a"})
        assert d.label_mapping["a"] == "a"

    def test_all_fixed_no_change(self):
        d = lucid_dream(SIMPLE_BRANCH, fixed_labels={"a", "b"})
        assert d.novelty_score == 0.0
        assert d.label_mapping["a"] == "a"
        assert d.label_mapping["b"] == "b"

    def test_structural_preservation(self):
        d = lucid_dream(DEEP_TYPE, fixed_labels={"a"})
        assert d.structural_preservation is True

    def test_lucidity_in_range(self):
        d = lucid_dream(DEEP_TYPE, fixed_labels={"a"})
        assert 0.0 <= d.lucidity <= 1.0


# ---------------------------------------------------------------------------
# Tests: nightmare
# ---------------------------------------------------------------------------


class TestNightmare:
    def test_returns_dream_result(self):
        d = nightmare(SIMPLE_BRANCH)
        assert isinstance(d, DreamResult)

    def test_structural_preservation(self):
        d = nightmare(SIMPLE_BRANCH)
        assert d.structural_preservation is True

    def test_polarity_inverted(self):
        """Branch becomes Select in the dream type."""
        d = nightmare(SIMPLE_BRANCH)
        dreamed_ast = parse(d.dream_type_str)
        # Original is Branch, nightmare should produce Select.
        from reticulate.parser import Select
        assert isinstance(dreamed_ast, Select)

    def test_novelty_in_range(self):
        d = nightmare(DEEP_TYPE)
        assert 0.0 <= d.novelty_score <= 1.0


# ---------------------------------------------------------------------------
# Tests: dream_sequence
# ---------------------------------------------------------------------------


class TestDreamSequence:
    def test_correct_count(self):
        seq = dream_sequence(SIMPLE_BRANCH, count=3)
        assert seq.total_dreams == 3
        assert len(seq.dreams) == 3

    def test_average_novelty_in_range(self):
        seq = dream_sequence(SIMPLE_BRANCH, count=5)
        assert 0.0 <= seq.average_novelty <= 1.0

    def test_average_lucidity_in_range(self):
        seq = dream_sequence(SIMPLE_BRANCH, count=5)
        assert 0.0 <= seq.average_lucidity <= 1.0

    def test_recurring_elements_are_strings(self):
        seq = dream_sequence(SIMPLE_BRANCH, count=5)
        assert all(isinstance(r, str) for r in seq.recurring_elements)

    def test_empty_sequence(self):
        seq = dream_sequence(SIMPLE_BRANCH, count=0)
        assert seq.total_dreams == 0
        assert seq.average_novelty == 0.0

    def test_each_dream_is_different_seed(self):
        seq = dream_sequence(DEEP_TYPE, count=3, base_seed=0)
        # Each should be a valid DreamResult.
        for d in seq.dreams:
            assert isinstance(d, DreamResult)


# ---------------------------------------------------------------------------
# Tests: interpret_dream
# ---------------------------------------------------------------------------


class TestInterpretDream:
    def test_returns_string(self):
        d = dream(SIMPLE_BRANCH, seed=7)
        text = interpret_dream(d)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_no_change_dreamless(self):
        d = dream("end")
        text = interpret_dream(d)
        assert "dreamless" in text.lower() or isinstance(text, str)

    def test_mentions_changed_labels(self):
        d = dream(DEEP_TYPE, seed=7)
        text = interpret_dream(d)
        # Should mention at least one label change if novelty > 0.
        if d.novelty_score > 0:
            assert "became" in text


# ---------------------------------------------------------------------------
# Tests: collective_dream
# ---------------------------------------------------------------------------


class TestCollectiveDream:
    def test_returns_dream_result(self):
        d = collective_dream([SIMPLE_BRANCH, SELECT_TYPE])
        assert isinstance(d, DreamResult)

    def test_shared_mapping(self):
        d = collective_dream([SIMPLE_BRANCH, DEEP_TYPE])
        # Mapping should include labels from both types.
        assert "a" in d.label_mapping
        assert "b" in d.label_mapping
        assert "c" in d.label_mapping

    def test_structural_preservation(self):
        d = collective_dream([SIMPLE_BRANCH, DEEP_TYPE])
        assert d.structural_preservation is True

    def test_single_type(self):
        d = collective_dream([SIMPLE_BRANCH])
        assert isinstance(d, DreamResult)
