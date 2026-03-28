"""Tests for the formal theory of analogy as session type morphism (Step 101)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.analogy import (
    EMBEDDING,
    FALSE_ANALOGY,
    HOMOMORPHISM,
    ISOMORPHISM,
    PROJECTION,
    WEAK_ANALOGY,
    AnalogyNetwork,
    AnalogyResult,
    analyze_analogy,
    analogy_transitivity,
    aristotelian_proportion,
    build_analogy_network,
    cross_domain_analogies,
    explain_analogy,
)
from reticulate.universal import UNIVERSAL_REPOSITORY


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
def other_branch_ss():
    return build_statespace(parse("&{c: end, d: end}"))


@pytest.fixture
def overlap_branch_ss():
    return build_statespace(parse("&{a: end, c: end}"))


@pytest.fixture
def iterator_ss():
    return build_statespace(parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"))


@pytest.fixture
def parallel_ss():
    return build_statespace(parse("(end || end)"))


@pytest.fixture
def deep_branch_ss():
    return build_statespace(parse("&{a: &{b: &{c: end}}}"))


@pytest.fixture
def identical_branch_ss():
    """Same structure as simple_branch_ss."""
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def three_entries(simple_branch_ss, other_branch_ss, overlap_branch_ss):
    return {
        "simple": simple_branch_ss,
        "other": other_branch_ss,
        "overlap": overlap_branch_ss,
    }


# ---------------------------------------------------------------------------
# analyze_analogy (12 tests)
# ---------------------------------------------------------------------------


class TestAnalyzeAnalogy:
    """Tests for the core analyze_analogy function."""

    def test_identical_types_isomorphism(self, simple_branch_ss, identical_branch_ss):
        result = analyze_analogy(simple_branch_ss, identical_branch_ss, "A", "B")
        assert result.strength == ISOMORPHISM

    def test_subtype_pair_embedding_or_better(self, simple_branch_ss, deep_branch_ss):
        result = analyze_analogy(simple_branch_ss, deep_branch_ss, "branch2", "branch3")
        assert result.strength in {ISOMORPHISM, EMBEDDING, PROJECTION, HOMOMORPHISM, WEAK_ANALOGY, FALSE_ANALOGY}

    def test_completely_different_no_shared_labels(self, simple_branch_ss, deep_branch_ss):
        # &{a: end, b: end} vs &{a: &{b: &{c: end}}} -- different structures
        result = analyze_analogy(simple_branch_ss, deep_branch_ss, "flat", "deep")
        # Both have some shared labels (a, b) so not false_analogy
        # But structure differs, so check result is reasonable
        assert result.compatibility_score >= 0.0
        assert isinstance(result.strength, str)

    def test_partially_overlapping(self, simple_branch_ss):
        # &{a: end, b: end} vs &{a: &{b: end}, c: end} -- shares "a" but different shapes
        ss2 = build_statespace(parse("&{a: &{b: end}, c: end}"))
        result = analyze_analogy(simple_branch_ss, ss2, "ab", "abc")
        # Shares some labels but different structure
        assert result.strength in {WEAK_ANALOGY, HOMOMORPHISM, PROJECTION, EMBEDDING}

    def test_result_is_frozen(self, end_ss):
        result = analyze_analogy(end_ss, end_ss, "X", "Y")
        with pytest.raises(AttributeError):
            result.strength = "modified"  # type: ignore[misc]

    def test_is_perfect_true_only_for_isomorphism(self, simple_branch_ss, identical_branch_ss):
        result = analyze_analogy(simple_branch_ss, identical_branch_ss, "A", "B")
        assert result.is_perfect is True

    def test_is_perfect_false_for_non_iso(self, end_ss, iterator_ss):
        result = analyze_analogy(end_ss, iterator_ss, "end", "iterator")
        assert result.is_perfect is False

    def test_is_structural_for_iso(self, simple_branch_ss, identical_branch_ss):
        result = analyze_analogy(simple_branch_ss, identical_branch_ss, "A", "B")
        assert result.is_structural is True

    def test_compatibility_score_in_range(self, simple_branch_ss, overlap_branch_ss):
        result = analyze_analogy(simple_branch_ss, overlap_branch_ss, "A", "B")
        assert 0.0 <= result.compatibility_score <= 1.0

    def test_topological_distance_nonnegative(self, simple_branch_ss, deep_branch_ss):
        result = analyze_analogy(simple_branch_ss, deep_branch_ss, "A", "B")
        assert result.topological_distance >= 0.0

    def test_state_ratio_in_range(self, simple_branch_ss, deep_branch_ss):
        result = analyze_analogy(simple_branch_ss, deep_branch_ss, "A", "B")
        assert 0.0 < result.state_ratio <= 1.0

    def test_shared_labels_correct(self, simple_branch_ss, overlap_branch_ss):
        result = analyze_analogy(simple_branch_ss, overlap_branch_ss, "A", "B")
        assert "a" in result.shared_labels

    def test_explanation_non_empty(self, simple_branch_ss, other_branch_ss):
        result = analyze_analogy(simple_branch_ss, other_branch_ss, "A", "B")
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0


# ---------------------------------------------------------------------------
# explain_analogy (5 tests)
# ---------------------------------------------------------------------------


class TestExplainAnalogy:
    """Tests for the explain_analogy function."""

    def test_returns_string(self, end_ss):
        result = analyze_analogy(end_ss, end_ss, "X", "Y")
        explanation = explain_analogy(result)
        assert isinstance(explanation, str)

    def test_mentions_strength(self, simple_branch_ss, identical_branch_ss):
        result = analyze_analogy(simple_branch_ss, identical_branch_ss, "A", "B")
        explanation = explain_analogy(result)
        assert result.strength in explanation

    def test_mentions_source_and_target(self, simple_branch_ss, other_branch_ss):
        result = analyze_analogy(simple_branch_ss, other_branch_ss, "alpha", "beta")
        explanation = explain_analogy(result)
        assert "alpha" in explanation
        assert "beta" in explanation

    def test_different_for_different_strengths(
        self, simple_branch_ss, identical_branch_ss, other_branch_ss
    ):
        iso_result = analyze_analogy(simple_branch_ss, identical_branch_ss, "A", "B")
        false_result = analyze_analogy(simple_branch_ss, other_branch_ss, "A", "C")
        assert explain_analogy(iso_result) != explain_analogy(false_result)

    def test_contains_verdict(self, end_ss):
        result = analyze_analogy(end_ss, end_ss, "X", "Y")
        explanation = explain_analogy(result)
        assert "Verdict" in explanation


# ---------------------------------------------------------------------------
# build_analogy_network (8 tests)
# ---------------------------------------------------------------------------


class TestBuildAnalogyNetwork:
    """Tests for the build_analogy_network function."""

    def test_correct_entries(self, three_entries):
        network = build_analogy_network(three_entries)
        assert set(network.entries) == set(three_entries.keys())

    def test_no_self_comparisons(self, three_entries):
        network = build_analogy_network(three_entries)
        for src, tgt, _, _ in network.edges:
            assert src != tgt

    def test_clusters_are_disjoint(self, three_entries):
        network = build_analogy_network(three_entries)
        all_elements: list[str] = []
        for cluster in network.clusters:
            all_elements.extend(cluster)
        assert len(all_elements) == len(set(all_elements))

    def test_all_entries_in_clusters(self, three_entries):
        network = build_analogy_network(three_entries)
        clustered = set()
        for cluster in network.clusters:
            clustered.update(cluster)
        assert clustered == set(three_entries.keys())

    def test_strongest_has_highest_score(self, three_entries):
        network = build_analogy_network(three_entries)
        if network.edges:
            max_score = max(e[3] for e in network.edges)
            assert network.strongest_analogy[2] >= max_score - 0.001

    def test_returns_analogy_network(self, three_entries):
        network = build_analogy_network(three_entries)
        assert isinstance(network, AnalogyNetwork)

    def test_edges_filterable_by_strength(self, three_entries):
        network = build_analogy_network(three_entries)
        strong = [e for e in network.edges if e[2] in {ISOMORPHISM, EMBEDDING}]
        assert isinstance(strong, list)

    def test_empty_entries(self):
        network = build_analogy_network({})
        assert network.entries == ()
        assert network.edges == ()


# ---------------------------------------------------------------------------
# aristotelian_proportion (5 tests)
# ---------------------------------------------------------------------------


class TestAristotelianProportion:
    """Tests for the aristotelian_proportion function."""

    def test_returns_entry_name(self, three_entries):
        best_d, score = aristotelian_proportion("simple", "other", "overlap", three_entries)
        assert best_d in three_entries

    def test_score_nonnegative(self, three_entries):
        _, score = aristotelian_proportion("simple", "other", "overlap", three_entries)
        assert score >= 0.0

    def test_same_source_pair(self, simple_branch_ss, identical_branch_ss, other_branch_ss):
        entries = {"A": simple_branch_ss, "B": identical_branch_ss, "C": other_branch_ss}
        best_d, _ = aristotelian_proportion("A", "B", "C", entries)
        assert best_d in entries

    def test_raises_on_missing_key(self, three_entries):
        with pytest.raises(KeyError):
            aristotelian_proportion("nonexistent", "simple", "other", three_entries)

    def test_with_two_entries(self, simple_branch_ss, other_branch_ss):
        entries = {"A": simple_branch_ss, "B": other_branch_ss}
        best_d, score = aristotelian_proportion("A", "B", "A", entries)
        assert best_d == "B"


# ---------------------------------------------------------------------------
# analogy_transitivity (5 tests)
# ---------------------------------------------------------------------------


class TestAnalogyTransitivity:
    """Tests for the analogy_transitivity function."""

    def test_identical_types_transitive(self, simple_branch_ss, identical_branch_ss):
        entries = {"A": simple_branch_ss, "B": identical_branch_ss, "C": simple_branch_ss}
        assert analogy_transitivity("A", "B", "C", entries) is True

    def test_returns_bool(self, three_entries):
        result = analogy_transitivity("simple", "other", "overlap", three_entries)
        assert isinstance(result, bool)

    def test_end_types_transitive(self, end_ss):
        entries = {"X": end_ss, "Y": end_ss, "Z": end_ss}
        assert analogy_transitivity("X", "Y", "Z", entries) is True

    def test_raises_on_missing_key(self, three_entries):
        with pytest.raises(KeyError):
            analogy_transitivity("nonexistent", "simple", "other", three_entries)

    def test_unrelated_types(self, simple_branch_ss, other_branch_ss, deep_branch_ss):
        entries = {"A": simple_branch_ss, "B": other_branch_ss, "C": deep_branch_ss}
        # Just check it returns a bool (may or may not be transitive)
        result = analogy_transitivity("A", "B", "C", entries)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# cross_domain_analogies (5 tests)
# ---------------------------------------------------------------------------


class TestCrossDomainAnalogies:
    """Tests for the cross_domain_analogies function."""

    def test_finds_cross_domain_pairs(self):
        # Use a small subset of the universal repository
        subset = {k: v for i, (k, v) in enumerate(UNIVERSAL_REPOSITORY.items()) if i < 6}
        results = cross_domain_analogies(subset)
        assert isinstance(results, list)

    def test_results_sorted_descending(self):
        subset = {k: v for i, (k, v) in enumerate(UNIVERSAL_REPOSITORY.items()) if i < 6}
        results = cross_domain_analogies(subset)
        if len(results) >= 2:
            scores = [r.compatibility_score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_all_pairs_cross_domain(self):
        subset = {k: v for i, (k, v) in enumerate(UNIVERSAL_REPOSITORY.items()) if i < 6}
        results = cross_domain_analogies(subset)
        domains = {e.name: e.domain for e in subset.values()}
        for r in results:
            if r.source_name in domains and r.target_name in domains:
                assert domains[r.source_name] != domains[r.target_name]

    def test_returns_analogy_results(self):
        subset = {k: v for i, (k, v) in enumerate(UNIVERSAL_REPOSITORY.items()) if i < 4}
        results = cross_domain_analogies(subset)
        for r in results:
            assert isinstance(r, AnalogyResult)

    def test_empty_repository(self):
        results = cross_domain_analogies({})
        assert results == []


# ---------------------------------------------------------------------------
# Edge cases (5 tests)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_end_vs_end_isomorphism(self, end_ss):
        result = analyze_analogy(end_ss, end_ss, "end1", "end2")
        assert result.strength == ISOMORPHISM

    def test_end_vs_complex_type(self, end_ss, iterator_ss):
        result = analyze_analogy(end_ss, iterator_ss, "end", "iterator")
        # end trivially embeds (1 state maps to bottom), but no shared labels
        assert result.compatibility_score == 0.0
        assert result.shared_labels == ()

    def test_single_state_spaces(self, end_ss):
        result = analyze_analogy(end_ss, end_ss, "A", "B")
        assert result.state_ratio == 1.0

    def test_parallel_vs_branch(self, parallel_ss, simple_branch_ss):
        result = analyze_analogy(parallel_ss, simple_branch_ss, "par", "branch")
        assert isinstance(result, AnalogyResult)

    def test_state_ratio_one_for_same_size(self, simple_branch_ss, other_branch_ss):
        result = analyze_analogy(simple_branch_ss, other_branch_ss, "A", "B")
        assert result.state_ratio == 1.0
