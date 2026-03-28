"""Tests for entry_analyzer — automated research pipeline."""

from __future__ import annotations

import pytest

from reticulate.entry_analyzer import (
    EntryAnalysis,
    analyze_all_entries,
    analyze_entry,
    cluster_by_fingerprint,
    compare_entries,
    generate_latex_section,
    generate_report,
    summary_statistics,
    _classify_complexity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_END = "end"
SIMPLE_BRANCH = "&{a: end, b: end}"
SIMPLE_SELECT = "+{x: end, y: end}"
RECURSIVE = "rec X . &{a: X, b: end}"
PARALLEL = "(&{a: end} || &{b: end})"
DEEP = "&{a: &{b: &{c: end}}}"


# ---------------------------------------------------------------------------
# EntryAnalysis dataclass
# ---------------------------------------------------------------------------


class TestEntryAnalysisDataclass:
    """Test that EntryAnalysis is a proper frozen dataclass."""

    def test_frozen(self) -> None:
        a = analyze_entry("test", SIMPLE_BRANCH)
        with pytest.raises(AttributeError):
            a.name = "other"  # type: ignore[misc]

    def test_fields_exist(self) -> None:
        a = analyze_entry("test", SIMPLE_BRANCH)
        # Spot-check key fields
        assert isinstance(a.name, str)
        assert isinstance(a.state_count, int)
        assert isinstance(a.transition_count, int)
        assert isinstance(a.is_lattice, bool)
        assert isinstance(a.euler_characteristic, int)
        assert isinstance(a.betti_1, int)
        assert isinstance(a.is_tree, bool)
        assert isinstance(a.edge_density, float)
        assert isinstance(a.branching_entropy, float)
        assert isinstance(a.channel_capacity, float)
        assert isinstance(a.is_deterministic, bool)
        assert isinstance(a.is_pendular, bool)
        assert isinstance(a.is_graded, bool)
        assert isinstance(a.select_ratio, float)
        assert isinstance(a.branch_ratio, float)
        assert isinstance(a.classification_alternation, str)
        assert isinstance(a.binding_energy, float)
        assert isinstance(a.dynamics_classification, str)
        assert isinstance(a.beauty_score, float)
        assert isinstance(a.aesthetic_classification, str)
        assert isinstance(a.shell_count, int)
        assert isinstance(a.symmetry_ratio, float)
        assert isinstance(a.orbital_classification, str)
        assert isinstance(a.top_analogies, tuple)
        assert isinstance(a.complexity_class, str)
        assert isinstance(a.fingerprint, tuple)


# ---------------------------------------------------------------------------
# Complexity classification
# ---------------------------------------------------------------------------


class TestComplexityClassification:
    def test_trivial(self) -> None:
        assert _classify_complexity(1) == "trivial"
        assert _classify_complexity(2) == "trivial"

    def test_simple(self) -> None:
        assert _classify_complexity(3) == "simple"
        assert _classify_complexity(5) == "simple"

    def test_moderate(self) -> None:
        assert _classify_complexity(6) == "moderate"
        assert _classify_complexity(10) == "moderate"

    def test_complex(self) -> None:
        assert _classify_complexity(11) == "complex"
        assert _classify_complexity(20) == "complex"

    def test_highly_complex(self) -> None:
        assert _classify_complexity(21) == "highly_complex"
        assert _classify_complexity(100) == "highly_complex"


# ---------------------------------------------------------------------------
# analyze_entry
# ---------------------------------------------------------------------------


class TestAnalyzeEntry:
    def test_end(self) -> None:
        a = analyze_entry("end_type", SIMPLE_END)
        assert a.name == "end_type"
        assert a.state_count >= 1
        assert a.is_lattice is True
        assert a.complexity_class == "trivial"

    def test_simple_branch(self) -> None:
        a = analyze_entry("branch", SIMPLE_BRANCH, domain="test", description="A branch")
        assert a.name == "branch"
        assert a.domain == "test"
        assert a.description == "A branch"
        assert a.state_count == 2  # end states merge
        assert a.transition_count == 2
        assert a.is_lattice is True

    def test_simple_select(self) -> None:
        a = analyze_entry("select", SIMPLE_SELECT)
        assert a.state_count == 2  # end states merge
        assert a.is_lattice is True

    def test_recursive(self) -> None:
        a = analyze_entry("recursive", RECURSIVE)
        assert a.state_count >= 2
        assert a.dynamics_classification in {
            "fixed_point", "periodic", "quasi_periodic", "chaotic",
        }

    def test_deep(self) -> None:
        a = analyze_entry("deep", DEEP)
        assert a.state_count == 4
        assert a.is_graded is True

    def test_domain_description_optional(self) -> None:
        a = analyze_entry("x", SIMPLE_END)
        assert a.domain == ""
        assert a.description == ""

    def test_fingerprint_is_tuple_of_ints(self) -> None:
        a = analyze_entry("fp", SIMPLE_BRANCH)
        assert all(isinstance(x, int) for x in a.fingerprint)

    def test_fingerprint_length(self) -> None:
        a = analyze_entry("fp", SIMPLE_BRANCH)
        assert len(a.fingerprint) == 6

    def test_with_repository(self) -> None:
        repo = {"other": "&{a: end}", "another": "+{x: end}"}
        a = analyze_entry("test", SIMPLE_BRANCH, repository=repo)
        assert isinstance(a.top_analogies, tuple)
        # Each analogy is (name, score)
        for name, score in a.top_analogies:
            assert isinstance(name, str)
            assert isinstance(score, float)

    def test_no_repository(self) -> None:
        a = analyze_entry("test", SIMPLE_BRANCH)
        assert a.top_analogies == ()


# ---------------------------------------------------------------------------
# analyze_all_entries
# ---------------------------------------------------------------------------


class TestAnalyzeAllEntries:
    def test_batch(self) -> None:
        entries = {
            "alpha": (SIMPLE_BRANCH, "test", "alpha desc"),
            "beta": (SIMPLE_SELECT, "test", "beta desc"),
        }
        results = analyze_all_entries(entries)
        assert "alpha" in results
        assert "beta" in results
        assert results["alpha"].name == "alpha"

    def test_empty(self) -> None:
        results = analyze_all_entries({})
        assert results == {}

    def test_invalid_entry_skipped(self) -> None:
        entries = {
            "good": (SIMPLE_BRANCH, "test", ""),
            "bad": ("NOT A VALID TYPE !!!", "test", ""),
        }
        results = analyze_all_entries(entries)
        assert "good" in results
        assert "bad" not in results


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_non_empty(self) -> None:
        a = analyze_entry("report_test", SIMPLE_BRANCH, "test", "A test")
        report = generate_report(a)
        assert len(report) > 0

    def test_contains_sections(self) -> None:
        a = analyze_entry("report_test", SIMPLE_BRANCH, "test", "A test")
        report = generate_report(a)
        assert "# Analysis: report_test" in report
        assert "## State Space" in report
        assert "## Topology" in report
        assert "## Information Theory" in report
        assert "## Dynamics" in report
        assert "## Aesthetics" in report
        assert "## Orbital" in report
        assert "## Analogies" in report
        assert "## Fingerprint" in report

    def test_contains_values(self) -> None:
        a = analyze_entry("rv", SIMPLE_BRANCH)
        report = generate_report(a)
        assert "States: 2" in report
        assert "Transitions: 2" in report


# ---------------------------------------------------------------------------
# generate_latex_section
# ---------------------------------------------------------------------------


class TestGenerateLatexSection:
    def test_non_empty(self) -> None:
        a = analyze_entry("latex_test", SIMPLE_BRANCH)
        latex = generate_latex_section(a)
        assert len(latex) > 0

    def test_contains_subsection(self) -> None:
        a = analyze_entry("latex_test", SIMPLE_BRANCH)
        latex = generate_latex_section(a)
        assert "\\subsection{" in latex

    def test_contains_paragraphs(self) -> None:
        a = analyze_entry("latex_test", SIMPLE_BRANCH, "test_domain", "A desc")
        latex = generate_latex_section(a)
        assert "\\paragraph{Session type.}" in latex
        assert "\\paragraph{State space.}" in latex
        assert "\\paragraph{Topology.}" in latex
        assert "\\paragraph{Information theory.}" in latex
        assert "\\paragraph{Dynamics.}" in latex
        assert "\\paragraph{Aesthetics.}" in latex
        assert "\\paragraph{Orbital.}" in latex

    def test_underscores_escaped(self) -> None:
        a = analyze_entry("my_entry", "&{a: end}", "my_domain")
        latex = generate_latex_section(a)
        assert "my\\_entry" in latex
        assert "my\\_domain" in latex


# ---------------------------------------------------------------------------
# compare_entries
# ---------------------------------------------------------------------------


class TestCompareEntries:
    def test_returns_dict(self) -> None:
        a = analyze_entry("a", SIMPLE_BRANCH)
        b = analyze_entry("b", SIMPLE_SELECT)
        result = compare_entries(a, b)
        assert isinstance(result, dict)

    def test_has_expected_keys(self) -> None:
        a = analyze_entry("a", SIMPLE_BRANCH)
        b = analyze_entry("b", SIMPLE_SELECT)
        result = compare_entries(a, b)
        assert "names" in result
        assert "state_count" in result
        assert "beauty_score" in result
        assert "same_fingerprint" in result

    def test_same_entry(self) -> None:
        a = analyze_entry("a", SIMPLE_BRANCH)
        result = compare_entries(a, a)
        assert result["same_fingerprint"] is True


# ---------------------------------------------------------------------------
# cluster_by_fingerprint
# ---------------------------------------------------------------------------


class TestClusterByFingerprint:
    def test_same_type_clusters_together(self) -> None:
        a1 = analyze_entry("x", SIMPLE_BRANCH)
        a2 = analyze_entry("y", SIMPLE_BRANCH)
        clusters = cluster_by_fingerprint({"x": a1, "y": a2})
        # They have the same fingerprint, so one cluster with both
        found = False
        for names in clusters.values():
            if "x" in names and "y" in names:
                found = True
        assert found

    def test_different_types_separate(self) -> None:
        a1 = analyze_entry("branch", SIMPLE_BRANCH)
        a2 = analyze_entry("deep", DEEP)
        clusters = cluster_by_fingerprint({"branch": a1, "deep": a2})
        assert len(clusters) >= 2

    def test_empty(self) -> None:
        clusters = cluster_by_fingerprint({})
        assert clusters == {}


# ---------------------------------------------------------------------------
# summary_statistics
# ---------------------------------------------------------------------------


class TestSummaryStatistics:
    def test_empty(self) -> None:
        stats = summary_statistics({})
        assert stats["total_entries"] == 0

    def test_has_expected_keys(self) -> None:
        a = analyze_entry("a", SIMPLE_BRANCH)
        stats = summary_statistics({"a": a})
        assert "total_entries" in stats
        assert "lattice_fraction" in stats
        assert "avg_states" in stats
        assert "avg_entropy" in stats
        assert "complexity_distribution" in stats
        assert "dynamics_distribution" in stats
        assert "most_common_fingerprint" in stats

    def test_correct_total(self) -> None:
        entries = {
            "a": (SIMPLE_BRANCH, "t", ""),
            "b": (SIMPLE_SELECT, "t", ""),
        }
        analyses = analyze_all_entries(entries)
        stats = summary_statistics(analyses)
        assert stats["total_entries"] == 2

    def test_lattice_fraction(self) -> None:
        a = analyze_entry("a", SIMPLE_BRANCH)
        stats = summary_statistics({"a": a})
        assert stats["lattice_fraction"] == 1.0
