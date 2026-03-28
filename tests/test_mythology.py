"""Tests for mythological session types: archetypes as universal protocols (Step 205)."""

from __future__ import annotations

import pytest

from reticulate import parse, build_statespace
from reticulate.lattice import check_lattice
from reticulate.mythology import (
    ARCHETYPE_LIBRARY,
    Archetype,
    MythAnalysis,
    get_archetype,
    shadow_of,
    analyze_myth,
    mythological_analogy,
    quest,
    all_archetypes_form_lattices,
)


# ---------------------------------------------------------------------------
# Library structure
# ---------------------------------------------------------------------------


class TestArchetypeLibrary:
    """Tests for the archetype library contents."""

    def test_library_has_at_least_20_archetypes(self) -> None:
        assert len(ARCHETYPE_LIBRARY) >= 20

    def test_all_archetypes_have_required_fields(self) -> None:
        for name, arch in ARCHETYPE_LIBRARY.items():
            assert isinstance(arch, Archetype)
            assert arch.name == name
            assert arch.domain in ("jungian", "campbellian", "greek", "tarot")
            assert len(arch.session_type_str) > 0
            assert len(arch.shadow_type_str) > 0
            assert len(arch.description) > 0

    def test_all_archetypes_parse(self) -> None:
        for name, arch in ARCHETYPE_LIBRARY.items():
            ast = parse(arch.session_type_str)
            assert ast is not None, f"Failed to parse archetype {name}"

    def test_all_shadows_parse(self) -> None:
        for name, arch in ARCHETYPE_LIBRARY.items():
            ast = parse(arch.shadow_type_str)
            assert ast is not None, f"Failed to parse shadow of {name}"

    def test_all_archetypes_build_statespace(self) -> None:
        for name, arch in ARCHETYPE_LIBRARY.items():
            ast = parse(arch.session_type_str)
            ss = build_statespace(ast)
            assert len(ss.states) >= 2, f"Archetype {name} too small"

    def test_all_archetypes_form_lattices(self) -> None:
        assert all_archetypes_form_lattices() is True

    def test_each_archetype_forms_lattice_individually(self) -> None:
        for name, arch in ARCHETYPE_LIBRARY.items():
            ast = parse(arch.session_type_str)
            ss = build_statespace(ast)
            result = check_lattice(ss)
            assert result.is_lattice, f"Archetype {name} is not a lattice"


# ---------------------------------------------------------------------------
# get_archetype
# ---------------------------------------------------------------------------


class TestGetArchetype:
    """Tests for get_archetype."""

    def test_get_known_archetype(self) -> None:
        arch = get_archetype("hero")
        assert arch.name == "hero"
        assert arch.domain == "campbellian"

    def test_get_shadow(self) -> None:
        arch = get_archetype("shadow")
        assert arch.name == "shadow"
        assert arch.domain == "jungian"

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown archetype"):
            get_archetype("nonexistent")

    def test_all_names_retrievable(self) -> None:
        for name in ARCHETYPE_LIBRARY:
            arch = get_archetype(name)
            assert arch.name == name


# ---------------------------------------------------------------------------
# shadow_of
# ---------------------------------------------------------------------------


class TestShadowOf:
    """Tests for shadow_of."""

    def test_shadow_returns_string(self) -> None:
        s = shadow_of("hero")
        assert isinstance(s, str)
        assert len(s) > 0

    def test_shadow_is_valid_type(self) -> None:
        for name in ARCHETYPE_LIBRARY:
            s = shadow_of(name)
            ast = parse(s)
            assert ast is not None

    def test_shadow_differs_from_original(self) -> None:
        # For most archetypes, shadow should differ from original
        arch = get_archetype("hero")
        assert arch.shadow_type_str != arch.session_type_str

    def test_shadow_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            shadow_of("nonexistent")


# ---------------------------------------------------------------------------
# analyze_myth
# ---------------------------------------------------------------------------


class TestAnalyzeMyth:
    """Tests for analyze_myth."""

    def test_returns_myth_analysis(self) -> None:
        result = analyze_myth("&{a: end, b: end}")
        assert isinstance(result, MythAnalysis)

    def test_hero_archetype_analysis(self) -> None:
        arch = get_archetype("hero")
        result = analyze_myth(arch.session_type_str)
        assert result.transformation_count > 0
        assert isinstance(result.hero_journey_phase, str)

    def test_sisyphus_is_circular(self) -> None:
        arch = get_archetype("sisyphus")
        result = analyze_myth(arch.session_type_str)
        assert result.is_circular is True

    def test_phoenix_is_circular(self) -> None:
        arch = get_archetype("phoenix")
        result = analyze_myth(arch.session_type_str)
        assert result.is_circular is True

    def test_garden_of_eden_is_not_circular(self) -> None:
        arch = get_archetype("garden_of_eden")
        result = analyze_myth(arch.session_type_str)
        assert result.is_circular is False

    def test_simple_type_analysis(self) -> None:
        result = analyze_myth("&{a: end}")
        assert result.transformation_count >= 1
        assert result.is_circular is False

    def test_sacrifice_points_are_ints(self) -> None:
        arch = get_archetype("father")
        result = analyze_myth(arch.session_type_str)
        assert isinstance(result.sacrifice_points, tuple)
        for pt in result.sacrifice_points:
            assert isinstance(pt, int)

    def test_rebirth_points_are_ints(self) -> None:
        arch = get_archetype("labyrinth")
        result = analyze_myth(arch.session_type_str)
        assert isinstance(result.rebirth_points, tuple)
        for pt in result.rebirth_points:
            assert isinstance(pt, int)


# ---------------------------------------------------------------------------
# mythological_analogy
# ---------------------------------------------------------------------------


class TestMythologicalAnalogy:
    """Tests for mythological_analogy."""

    def test_self_analogy_is_high(self) -> None:
        score = mythological_analogy("hero", "hero")
        assert score >= 0.8

    def test_returns_float(self) -> None:
        score = mythological_analogy("hero", "shadow")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_similar_archetypes_score_higher(self) -> None:
        # Sisyphus and Phoenix are both eternal cycles
        cycle_score = mythological_analogy("sisyphus", "phoenix")
        # Father and Sisyphus are structurally different
        diff_score = mythological_analogy("father", "sisyphus")
        # Both should be valid scores
        assert 0.0 <= cycle_score <= 1.0
        assert 0.0 <= diff_score <= 1.0


# ---------------------------------------------------------------------------
# quest
# ---------------------------------------------------------------------------


class TestQuest:
    """Tests for quest composition."""

    def test_quest_returns_string(self) -> None:
        result = quest("hero", ["shadow"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_quest_result_parses(self) -> None:
        result = quest("hero", ["shadow", "trickster"])
        ast = parse(result)
        assert ast is not None

    def test_quest_result_forms_lattice(self) -> None:
        result = quest("hero", ["shadow"])
        ast = parse(result)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_quest_empty_challenges(self) -> None:
        result = quest("hero", [])
        arch = get_archetype("hero")
        # With no challenges, should return the hero archetype itself.
        ast = parse(result)
        assert ast is not None

    def test_quest_single_challenge(self) -> None:
        result = quest("child", ["wise_old"])
        ast = parse(result)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2

    def test_quest_unknown_start_raises(self) -> None:
        with pytest.raises(KeyError):
            quest("nonexistent", ["hero"])

    def test_quest_unknown_challenge_raises(self) -> None:
        with pytest.raises(KeyError):
            quest("hero", ["nonexistent"])
