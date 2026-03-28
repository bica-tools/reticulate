"""Tests for the Universal Session Type Repository (Step 100)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.universal import (
    UNIVERSAL_REPOSITORY,
    AnalogResult,
    CompositionResult,
    OntologyEntry,
    all_form_lattices,
    compose,
    domains,
    find_analogies,
    lookup,
    repository_stats,
    search_by_domain,
    search_by_tag,
    tags,
)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_lookup_known_entry(self) -> None:
        entry = lookup("photosynthesis")
        assert entry.name == "photosynthesis"
        assert entry.domain == "biology"

    def test_lookup_returns_ontology_entry(self) -> None:
        entry = lookup("gravity")
        assert isinstance(entry, OntologyEntry)

    def test_lookup_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown entry"):
            lookup("nonexistent_entry")

    def test_lookup_all_entries(self) -> None:
        for name in UNIVERSAL_REPOSITORY:
            entry = lookup(name)
            assert entry.name == name


# ---------------------------------------------------------------------------
# Search by tag
# ---------------------------------------------------------------------------


class TestSearchByTag:
    def test_recursive_tag_finds_multiple(self) -> None:
        results = search_by_tag("recursive")
        assert len(results) >= 5
        names = {e.name for e in results}
        assert "scientific_method" in names
        assert "dialectic" in names

    def test_biology_tag_finds_nature_entries(self) -> None:
        results = search_by_tag("biology")
        names = {e.name for e in results}
        assert "photosynthesis" in names
        assert "cell_division" in names

    def test_energy_tag_cross_domain(self) -> None:
        results = search_by_tag("energy")
        domains_found = {e.domain for e in results}
        assert "biology" in domains_found
        assert "physics" in domains_found

    def test_nonexistent_tag_returns_empty(self) -> None:
        assert search_by_tag("nonexistent_tag_xyz") == []

    def test_results_sorted_by_name(self) -> None:
        results = search_by_tag("recursive")
        names = [e.name for e in results]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Search by domain
# ---------------------------------------------------------------------------


class TestSearchByDomain:
    def test_biology_domain(self) -> None:
        results = search_by_domain("biology")
        assert len(results) == 2
        names = {e.name for e in results}
        assert names == {"photosynthesis", "cell_division"}

    def test_physics_domain(self) -> None:
        results = search_by_domain("physics")
        assert len(results) == 3

    def test_philosophy_domain(self) -> None:
        results = search_by_domain("philosophy")
        names = {e.name for e in results}
        assert "dialectic" in names
        assert "socratic_method" in names

    def test_unknown_domain_returns_empty(self) -> None:
        assert search_by_domain("alchemy") == []

    def test_computing_domain(self) -> None:
        results = search_by_domain("computing")
        names = {e.name for e in results}
        assert "compile" in names
        assert "search" in names


# ---------------------------------------------------------------------------
# Domains and tags
# ---------------------------------------------------------------------------


class TestDomainsAndTags:
    def test_domains_returns_sorted(self) -> None:
        d = domains()
        assert d == sorted(d)
        assert len(d) >= 10

    def test_domains_includes_expected(self) -> None:
        d = domains()
        for expected in ["biology", "physics", "philosophy", "computing", "law"]:
            assert expected in d

    def test_tags_returns_sorted(self) -> None:
        t = tags()
        assert t == sorted(t)
        assert len(t) >= 15

    def test_tags_includes_expected(self) -> None:
        t = tags()
        for expected in ["recursive", "energy", "biology", "philosophy", "music"]:
            assert expected in t


# ---------------------------------------------------------------------------
# Parsing — all entries parse correctly
# ---------------------------------------------------------------------------


class TestAllEntriesParse:
    @pytest.mark.parametrize("name", list(UNIVERSAL_REPOSITORY.keys()))
    def test_entry_parses(self, name: str) -> None:
        entry = UNIVERSAL_REPOSITORY[name]
        ast = parse(entry.session_type_str)
        assert ast is not None

    @pytest.mark.parametrize("name", list(UNIVERSAL_REPOSITORY.keys()))
    def test_entry_builds_statespace(self, name: str) -> None:
        entry = UNIVERSAL_REPOSITORY[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2  # At least initial + end

    @pytest.mark.parametrize("name", list(UNIVERSAL_REPOSITORY.keys()))
    def test_entry_has_nonempty_description(self, name: str) -> None:
        entry = UNIVERSAL_REPOSITORY[name]
        assert entry.description
        assert len(entry.description) > 0


# ---------------------------------------------------------------------------
# All entries form lattices (CRITICAL)
# ---------------------------------------------------------------------------


class TestAllFormLattices:
    @pytest.mark.parametrize("name", list(UNIVERSAL_REPOSITORY.keys()))
    def test_entry_forms_lattice(self, name: str) -> None:
        entry = UNIVERSAL_REPOSITORY[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.is_lattice, (
            f"{name} does not form a lattice: {result.counterexample}"
        )

    def test_all_form_lattices_function(self) -> None:
        assert all_form_lattices() is True


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestCompose:
    def test_compose_two_entries(self) -> None:
        result = compose(["photosynthesis", "thermodynamics"])
        assert isinstance(result, CompositionResult)
        assert len(result.components) == 2
        assert result.state_count >= 1
        assert isinstance(result.is_lattice, bool)

    def test_compose_three_entries(self) -> None:
        result = compose(["haiku", "sonata_form", "compile"])
        assert len(result.components) == 3
        assert ("haiku", "sonata_form") in result.compatibility_matrix

    def test_compose_returns_lattice(self) -> None:
        result = compose(["grief", "cell_division"])
        assert result.is_lattice

    def test_compose_too_few_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            compose(["photosynthesis"])

    def test_compose_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            compose(["photosynthesis", "nonexistent"])

    def test_compose_compatibility_matrix_keys(self) -> None:
        result = compose(["haiku", "sonata_form", "compile"])
        assert ("haiku", "sonata_form") in result.compatibility_matrix
        assert ("haiku", "compile") in result.compatibility_matrix
        assert ("sonata_form", "compile") in result.compatibility_matrix


# ---------------------------------------------------------------------------
# Find analogies
# ---------------------------------------------------------------------------


class TestFindAnalogies:
    def test_find_analogies_returns_list(self) -> None:
        results = find_analogies("photosynthesis", threshold=0.0)
        assert isinstance(results, list)
        assert all(isinstance(r, AnalogResult) for r in results)

    def test_find_analogies_sorted_descending(self) -> None:
        results = find_analogies("photosynthesis", threshold=0.0)
        scores = [r.similarity for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_find_analogies_excludes_self(self) -> None:
        results = find_analogies("photosynthesis", threshold=0.0)
        sources = {r.target for r in results}
        assert "photosynthesis" not in sources

    def test_find_analogies_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            find_analogies("nonexistent")

    def test_find_analogies_high_threshold_fewer_results(self) -> None:
        low = find_analogies("compile", threshold=0.0)
        high = find_analogies("compile", threshold=0.9)
        assert len(high) <= len(low)


# ---------------------------------------------------------------------------
# Repository stats
# ---------------------------------------------------------------------------


class TestRepositoryStats:
    def test_stats_has_expected_keys(self) -> None:
        stats = repository_stats()
        for key in ["total", "domains", "tags", "recursive", "non_recursive"]:
            assert key in stats

    def test_stats_total_matches_repo(self) -> None:
        stats = repository_stats()
        assert stats["total"] == len(UNIVERSAL_REPOSITORY)

    def test_stats_recursive_plus_nonrecursive_equals_total(self) -> None:
        stats = repository_stats()
        assert stats["recursive"] + stats["non_recursive"] == stats["total"]

    def test_stats_has_recursive_entries(self) -> None:
        stats = repository_stats()
        assert stats["recursive"] >= 5

    def test_stats_domains_matches(self) -> None:
        stats = repository_stats()
        assert stats["domains"] == len(domains())

    def test_stats_tags_matches(self) -> None:
        stats = repository_stats()
        assert stats["tags"] == len(tags())


# ---------------------------------------------------------------------------
# Data type checks
# ---------------------------------------------------------------------------


class TestDataTypes:
    def test_ontology_entry_is_frozen(self) -> None:
        entry = lookup("haiku")
        with pytest.raises(AttributeError):
            entry.name = "modified"  # type: ignore[misc]

    def test_composition_result_is_frozen(self) -> None:
        result = compose(["haiku", "sonata_form"])
        with pytest.raises(AttributeError):
            result.state_count = 999  # type: ignore[misc]

    def test_analog_result_is_frozen(self) -> None:
        results = find_analogies("haiku", threshold=0.0)
        if results:
            with pytest.raises(AttributeError):
                results[0].similarity = 0.0  # type: ignore[misc]
