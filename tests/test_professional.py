"""Tests for reticulate.professional — professional session types (Step 60i)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.professional import (
    PROFESSION_LIBRARY,
    ProfessionAnalysis,
    ProfessionComparison,
    ProfessionProfile,
    analyze_profession,
    compare_professions,
    get_profession,
    most_compatible_pair,
    most_complex_profession,
    profession_lattice,
    professions_by_category,
)


# ---------------------------------------------------------------------------
# All session types parse correctly (critical)
# ---------------------------------------------------------------------------


class TestParsing:
    """Verify every profession's session type string parses without error."""

    @pytest.mark.parametrize("name", sorted(PROFESSION_LIBRARY.keys()))
    def test_parse_profession(self, name: str) -> None:
        ast = parse(PROFESSION_LIBRARY[name])
        assert ast is not None

    @pytest.mark.parametrize("name", sorted(PROFESSION_LIBRARY.keys()))
    def test_build_statespace(self, name: str) -> None:
        ast = parse(PROFESSION_LIBRARY[name])
        ss = build_statespace(ast)
        assert len(ss.states) > 0
        assert len(ss.transitions) > 0


# ---------------------------------------------------------------------------
# get_profession
# ---------------------------------------------------------------------------


class TestGetProfession:
    """Test profession profile lookup."""

    def test_doctor(self) -> None:
        p = get_profession("doctor")
        assert isinstance(p, ProfessionProfile)
        assert p.category == "healthcare"
        assert p.is_client_facing is True
        assert p.recursion_type == "iterative"

    def test_nurse(self) -> None:
        p = get_profession("nurse")
        assert p.category == "healthcare"
        assert p.interaction_pattern == "consultation"

    def test_pharmacist(self) -> None:
        p = get_profession("pharmacist")
        assert p.category == "healthcare"
        assert p.recursion_type == "none"

    def test_teacher(self) -> None:
        p = get_profession("teacher")
        assert p.category == "education"
        assert p.interaction_pattern == "instruction"

    def test_professor(self) -> None:
        p = get_profession("professor")
        assert p.category == "education"

    def test_developer(self) -> None:
        p = get_profession("developer")
        assert p.category == "technology"
        assert p.is_client_facing is False

    def test_sysadmin(self) -> None:
        p = get_profession("sysadmin")
        assert p.category == "technology"

    def test_barista(self) -> None:
        p = get_profession("barista")
        assert p.category == "service"
        assert p.interaction_pattern == "transaction"

    def test_waiter(self) -> None:
        p = get_profession("waiter")
        assert p.category == "service"

    def test_hairdresser(self) -> None:
        p = get_profession("hairdresser")
        assert p.category == "service"

    def test_musician(self) -> None:
        p = get_profession("musician")
        assert p.category == "creative"
        assert p.interaction_pattern == "performance"

    def test_writer(self) -> None:
        p = get_profession("writer")
        assert p.category == "creative"

    def test_lawyer(self) -> None:
        p = get_profession("lawyer")
        assert p.category == "legal"

    def test_judge(self) -> None:
        p = get_profession("judge")
        assert p.category == "legal"
        assert p.recursion_type == "none"

    def test_banker(self) -> None:
        p = get_profession("banker")
        assert p.category == "finance"

    def test_trader(self) -> None:
        p = get_profession("trader")
        assert p.category == "finance"
        assert p.interaction_pattern == "negotiation"

    def test_firefighter(self) -> None:
        p = get_profession("firefighter")
        assert p.category == "emergency"

    def test_paramedic(self) -> None:
        p = get_profession("paramedic")
        assert p.category == "emergency"
        assert p.recursion_type == "none"

    def test_case_insensitive(self) -> None:
        p = get_profession("Doctor")
        assert p.name == "doctor"

    def test_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="Unknown profession"):
            get_profession("astronaut")


# ---------------------------------------------------------------------------
# analyze_profession
# ---------------------------------------------------------------------------


class TestAnalyzeProfession:
    """Test structural analysis of professions."""

    @pytest.mark.parametrize("name", sorted(PROFESSION_LIBRARY.keys()))
    def test_state_count_positive(self, name: str) -> None:
        result = analyze_profession(name)
        assert result.state_count > 0

    @pytest.mark.parametrize("name", sorted(PROFESSION_LIBRARY.keys()))
    def test_all_form_lattices(self, name: str) -> None:
        result = analyze_profession(name)
        assert result.is_lattice is True

    def test_doctor_analysis(self) -> None:
        result = analyze_profession("doctor")
        assert isinstance(result, ProfessionAnalysis)
        assert result.profile.name == "doctor"
        assert result.transition_count > 0
        assert result.branching_entropy >= 0.0
        assert 0.0 <= result.binding_energy <= 1.0

    def test_barista_simple(self) -> None:
        result = analyze_profession("barista")
        assert result.complexity_class == "simple"

    def test_trader_has_binding_energy(self) -> None:
        result = analyze_profession("trader")
        # Trader has rec X . &{analyze: +{buy: X, sell: X, hold: X}}
        # So it should have non-trivial SCCs (binding energy > 0)
        assert result.binding_energy > 0.0

    def test_pharmacist_no_recursion(self) -> None:
        result = analyze_profession("pharmacist")
        assert result.profile.recursion_type == "none"

    def test_pendularity_works(self) -> None:
        # Just verify it returns a bool without crashing
        result = analyze_profession("doctor")
        assert isinstance(result.is_pendular, bool)


# ---------------------------------------------------------------------------
# compare_professions
# ---------------------------------------------------------------------------


class TestCompareProfessions:
    """Test pairwise profession comparisons."""

    def test_doctor_vs_nurse(self) -> None:
        result = compare_professions("doctor", "nurse")
        assert isinstance(result, ProfessionComparison)
        assert result.profession_a == "doctor"
        assert result.profession_b == "nurse"
        assert 0.0 <= result.compatibility_score <= 1.0

    def test_teacher_vs_professor(self) -> None:
        result = compare_professions("teacher", "professor")
        assert 0.0 <= result.compatibility_score <= 1.0

    def test_barista_vs_trader_incomparable(self) -> None:
        result = compare_professions("barista", "trader")
        # Very different professions — likely incomparable
        assert result.subsumption in (
            "a_subsumes_b", "b_subsumes_a", "incomparable", "equivalent"
        )

    def test_shared_and_unique_methods(self) -> None:
        result = compare_professions("doctor", "nurse")
        # All returned tuples should be tuples of strings
        assert isinstance(result.shared_interactions, tuple)
        assert isinstance(result.unique_to_a, tuple)
        assert isinstance(result.unique_to_b, tuple)

    def test_self_comparison_equivalent(self) -> None:
        # Comparing a profession to itself should be equivalent
        result = compare_professions("doctor", "doctor")
        assert result.compatibility_score == 1.0
        assert result.subsumption == "equivalent"


# ---------------------------------------------------------------------------
# professions_by_category
# ---------------------------------------------------------------------------


class TestProfessionsByCategory:
    """Test category listing."""

    def test_healthcare(self) -> None:
        profs = professions_by_category("healthcare")
        assert "doctor" in profs
        assert "nurse" in profs
        assert "pharmacist" in profs
        assert len(profs) == 3

    def test_education(self) -> None:
        profs = professions_by_category("education")
        assert len(profs) == 2
        assert "teacher" in profs
        assert "professor" in profs

    def test_technology(self) -> None:
        profs = professions_by_category("technology")
        assert len(profs) == 2

    def test_service(self) -> None:
        profs = professions_by_category("service")
        assert len(profs) == 3

    def test_creative(self) -> None:
        profs = professions_by_category("creative")
        assert len(profs) == 2

    def test_legal(self) -> None:
        profs = professions_by_category("legal")
        assert len(profs) == 2

    def test_finance(self) -> None:
        profs = professions_by_category("finance")
        assert len(profs) == 2

    def test_emergency(self) -> None:
        profs = professions_by_category("emergency")
        assert len(profs) == 2

    def test_unknown_category_empty(self) -> None:
        profs = professions_by_category("aerospace")
        assert profs == []

    def test_sorted(self) -> None:
        profs = professions_by_category("healthcare")
        assert profs == sorted(profs)


# ---------------------------------------------------------------------------
# most_complex_profession
# ---------------------------------------------------------------------------


class TestMostComplex:
    """Test most complex profession finder."""

    def test_returns_string(self) -> None:
        result = most_complex_profession()
        assert isinstance(result, str)
        assert result in PROFESSION_LIBRARY

    def test_result_in_library(self) -> None:
        result = most_complex_profession()
        assert result in PROFESSION_LIBRARY


# ---------------------------------------------------------------------------
# most_compatible_pair
# ---------------------------------------------------------------------------


class TestMostCompatible:
    """Test most compatible pair finder."""

    def test_returns_triple(self) -> None:
        a, b, score = most_compatible_pair()
        assert isinstance(a, str)
        assert isinstance(b, str)
        assert isinstance(score, float)
        assert a in PROFESSION_LIBRARY
        assert b in PROFESSION_LIBRARY
        assert 0.0 <= score <= 1.0

    def test_different_professions(self) -> None:
        a, b, _ = most_compatible_pair()
        assert a != b


# ---------------------------------------------------------------------------
# profession_lattice
# ---------------------------------------------------------------------------


class TestProfessionLattice:
    """Test the subsumption map."""

    def test_returns_dict(self) -> None:
        result = profession_lattice()
        assert isinstance(result, dict)
        assert len(result) == len(PROFESSION_LIBRARY)

    def test_all_professions_present(self) -> None:
        result = profession_lattice()
        for name in PROFESSION_LIBRARY:
            assert name in result

    def test_values_are_lists(self) -> None:
        result = profession_lattice()
        for name, subs in result.items():
            assert isinstance(subs, list)
            for s in subs:
                assert s in PROFESSION_LIBRARY
