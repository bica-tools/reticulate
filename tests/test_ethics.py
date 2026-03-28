"""Tests for ethics as subtype checking (Step 201)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.ethics import (
    ETHICAL_FRAMEWORKS,
    EthicalFramework,
    EthicalJudgment,
    all_frameworks_agree,
    ethical_dilemma,
    framework_lattice,
    judge_action,
    judge_all_frameworks,
)
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(ETHICAL_FRAMEWORKS.keys()))
def framework_name(request):
    return request.param


# ---------------------------------------------------------------------------
# Framework parsing tests
# ---------------------------------------------------------------------------


class TestFrameworkParsing:
    """All predefined frameworks must parse and produce valid state spaces."""

    def test_all_frameworks_parse(self, framework_name):
        fw = ETHICAL_FRAMEWORKS[framework_name]
        ast = parse(fw.principle_type_str)
        assert ast is not None

    def test_all_frameworks_build_statespace(self, framework_name):
        fw = ETHICAL_FRAMEWORKS[framework_name]
        ss = build_statespace(parse(fw.principle_type_str))
        assert len(ss.states) > 0
        assert len(ss.transitions) > 0

    def test_all_frameworks_form_lattices(self, framework_name):
        fw = ETHICAL_FRAMEWORKS[framework_name]
        ss = build_statespace(parse(fw.principle_type_str))
        result = check_lattice(ss)
        assert result.is_lattice, (
            f"Framework {framework_name} does not form a lattice: "
            f"{result.counterexample}"
        )

    def test_framework_has_name(self, framework_name):
        fw = ETHICAL_FRAMEWORKS[framework_name]
        assert fw.name == framework_name

    def test_framework_has_description(self, framework_name):
        fw = ETHICAL_FRAMEWORKS[framework_name]
        assert len(fw.description) > 0


class TestFrameworkCount:
    """Verify the number of predefined frameworks."""

    def test_six_frameworks(self):
        assert len(ETHICAL_FRAMEWORKS) == 6

    def test_expected_names(self):
        expected = {"kantian", "utilitarian", "virtue", "care", "deontological", "contractarian"}
        assert set(ETHICAL_FRAMEWORKS.keys()) == expected


# ---------------------------------------------------------------------------
# Judgment tests
# ---------------------------------------------------------------------------


class TestJudgeAction:
    """Test the judge_action function."""

    def test_end_is_ethical_under_all(self, framework_name):
        """end (empty branch) should be ethical — it makes no demands."""
        j = judge_action("end", framework_name)
        assert isinstance(j, EthicalJudgment)
        assert j.framework == framework_name

    def test_judgment_returns_correct_action(self):
        j = judge_action("end", "kantian")
        assert j.action == "end"

    def test_unknown_framework_raises(self):
        with pytest.raises(KeyError):
            judge_action("end", "nonexistent")

    def test_invalid_type_raises(self):
        with pytest.raises(Exception):
            judge_action("not a valid type !!!", "kantian")

    def test_alignment_score_between_0_and_1(self, framework_name):
        j = judge_action("&{act: end}", framework_name)
        assert 0.0 <= j.alignment_score <= 1.0

    def test_violation_methods_are_strings(self):
        j = judge_action("&{weird_method: end}", "kantian")
        assert all(isinstance(m, str) for m in j.violation_methods)

    def test_exact_principle_is_ethical(self):
        """An action that exactly matches the principle type should be ethical."""
        fw = ETHICAL_FRAMEWORKS["utilitarian"]
        j = judge_action(fw.principle_type_str, "utilitarian")
        assert j.is_ethical

    def test_subtype_is_ethical(self):
        """An action matching the principle exactly should be ethical."""
        fw = ETHICAL_FRAMEWORKS["deontological"]
        j = judge_action(fw.principle_type_str, "deontological")
        assert j.is_ethical


class TestJudgeAllFrameworks:
    """Test judge_all_frameworks."""

    def test_returns_dict(self):
        result = judge_all_frameworks("end")
        assert isinstance(result, dict)
        assert len(result) == 6

    def test_all_framework_names_present(self):
        result = judge_all_frameworks("end")
        for name in ETHICAL_FRAMEWORKS:
            assert name in result

    def test_each_value_is_judgment(self):
        result = judge_all_frameworks("end")
        for j in result.values():
            assert isinstance(j, EthicalJudgment)


# ---------------------------------------------------------------------------
# Ethical dilemma tests
# ---------------------------------------------------------------------------


class TestEthicalDilemma:
    """Test the ethical_dilemma function."""

    def test_returns_valid_value(self):
        result = ethical_dilemma("end", "&{act: end}", "kantian")
        assert result in ("a", "b", "equal")

    def test_identical_actions_equal(self):
        result = ethical_dilemma("end", "end", "kantian")
        assert result == "equal"

    def test_dilemma_with_different_actions(self):
        result = ethical_dilemma(
            "&{calculate_consequences: +{net_positive: &{act: end}, net_negative: &{abstain: end}}}",
            "&{random_act: end}",
            "utilitarian",
        )
        # The first action matches the utilitarian principle exactly
        assert result in ("a", "b", "equal")


# ---------------------------------------------------------------------------
# Agreement tests
# ---------------------------------------------------------------------------


class TestAllFrameworksAgree:
    """Test the all_frameworks_agree function."""

    def test_end_agreement(self):
        """end should produce agreement (all ethical or all unethical)."""
        result = all_frameworks_agree("end")
        assert isinstance(result, bool)

    def test_returns_bool(self):
        result = all_frameworks_agree("&{act: end}")
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Framework lattice tests
# ---------------------------------------------------------------------------


class TestFrameworkLattice:
    """Test the framework_lattice function."""

    def test_returns_dict(self):
        result = framework_lattice()
        assert isinstance(result, dict)

    def test_all_frameworks_present(self):
        result = framework_lattice()
        for name in ETHICAL_FRAMEWORKS:
            assert name in result

    def test_values_are_sorted_lists(self):
        result = framework_lattice()
        for name, subsumes in result.items():
            assert isinstance(subsumes, list)
            assert subsumes == sorted(subsumes)

    def test_no_self_subsumption(self):
        result = framework_lattice()
        for name, subsumes in result.items():
            assert name not in subsumes
