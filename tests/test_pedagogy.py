"""Tests for reticulate.pedagogy — pedagogical session types (Step 60e)."""

from __future__ import annotations

import pytest

from reticulate import parse, build_statespace
from reticulate.parser import Branch, End, Select, Rec, Var
from reticulate.pedagogy import (
    ScaffoldingResult,
    DialogueResult,
    analyze_scaffolding,
    zone_of_proximal_development,
    analyze_dialogue,
    developmental_stages,
    learning_path,
    classify_pedagogy,
    _get_choices,
    _classify_stage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Teacher: offers open, close, read, write
TEACHER_STR = "&{open: end, close: end, read: end, write: end}"
# Learner (novice): only knows open
LEARNER_NOVICE_STR = "&{open: end}"
# Learner (intermediate): knows open, close
LEARNER_INTER_STR = "&{open: end, close: end}"
# Learner (advanced): knows open, close, read
LEARNER_ADV_STR = "&{open: end, close: end, read: end}"
# Expert: knows all
LEARNER_EXPERT_STR = "&{open: end, close: end, read: end, write: end}"

# Pendular (turn-taking) type: branch then select then end
PENDULAR_STR = "&{ask: +{answer: end}}"
# Non-pendular: branch -> branch
NON_PENDULAR_STR = "&{a: &{b: end}}"
# Socratic: more branches than selects
SOCRATIC_STR = "&{q1: +{a1: &{q2: end}}, q2: +{a2: end}}"

# Iterator progression
ITER_NOVICE_STR = "&{hasNext: end}"
ITER_INTER_STR = "&{hasNext: end, next: end}"
ITER_TARGET_STR = "&{hasNext: end, next: end, remove: end}"

# SMTP-like dialogue
SMTP_STR = "&{EHLO: +{OK: &{MAIL: +{OK: &{RCPT: +{OK: end}}}}}}"

# Incompatible types (branch vs select)
SELECT_TYPE_STR = "+{a: end, b: end}"
BRANCH_TYPE_STR = "&{c: end, d: end}"


# ---------------------------------------------------------------------------
# Tests: _get_choices helper
# ---------------------------------------------------------------------------


class TestGetChoices:
    def test_branch_choices(self) -> None:
        t = parse(TEACHER_STR)
        choices = _get_choices(t)
        assert set(choices.keys()) == {"open", "close", "read", "write"}

    def test_select_choices(self) -> None:
        t = parse(SELECT_TYPE_STR)
        choices = _get_choices(t)
        assert set(choices.keys()) == {"a", "b"}

    def test_end_no_choices(self) -> None:
        t = parse("end")
        assert _get_choices(t) == {}

    def test_rec_unwraps(self) -> None:
        t = parse("rec X . &{a: X, b: end}")
        choices = _get_choices(t)
        assert set(choices.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Tests: analyze_scaffolding
# ---------------------------------------------------------------------------


class TestAnalyzeScaffolding:
    def test_teacher_superset_is_scaffold(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse(LEARNER_NOVICE_STR)
        result = analyze_scaffolding(teacher, learner)
        assert result.is_scaffold is True
        assert result.zpd_size == 3
        assert "open" in result.shared_methods
        assert set(result.scaffold_methods) == {"close", "read", "write"}

    def test_equal_types_not_scaffold(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse(LEARNER_EXPERT_STR)
        result = analyze_scaffolding(teacher, learner)
        assert result.is_scaffold is False
        assert result.zpd_size == 0
        assert result.learning_progress == 1.0
        assert result.stage == "expert"

    def test_incompatible_not_scaffold(self) -> None:
        teacher = parse(BRANCH_TYPE_STR)
        learner = parse("&{x: end, y: end}")
        result = analyze_scaffolding(teacher, learner)
        # learner methods {x, y} not subset of teacher {c, d}
        assert result.is_scaffold is False

    def test_novice_stage(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse("&{z: end}")  # no shared methods
        result = analyze_scaffolding(teacher, learner)
        assert result.stage == "novice"
        assert result.learning_progress == 0.0

    def test_intermediate_stage(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse(LEARNER_INTER_STR)
        result = analyze_scaffolding(teacher, learner)
        # 2/4 = 0.5 => advanced (>= 0.50)
        assert result.learning_progress == 0.5
        assert result.stage == "advanced"

    def test_advanced_stage(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse(LEARNER_ADV_STR)
        result = analyze_scaffolding(teacher, learner)
        assert result.learning_progress == 0.75
        assert result.stage == "expert"  # 0.75 >= 0.75

    def test_end_teacher_end_learner(self) -> None:
        teacher = parse("end")
        learner = parse("end")
        result = analyze_scaffolding(teacher, learner)
        assert result.is_scaffold is False
        assert result.zpd_size == 0
        # No methods: 0/1 = 0.0 => novice
        assert result.stage == "novice"

    def test_scaffold_result_is_frozen(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse(LEARNER_NOVICE_STR)
        result = analyze_scaffolding(teacher, learner)
        with pytest.raises(AttributeError):
            result.is_scaffold = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: zone_of_proximal_development
# ---------------------------------------------------------------------------


class TestZPD:
    def test_gap_methods_returned(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse(LEARNER_NOVICE_STR)
        zpd = zone_of_proximal_development(teacher, learner)
        assert len(zpd) == 3  # close, read, write

    def test_no_gap_equal_types(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse(LEARNER_EXPERT_STR)
        zpd = zone_of_proximal_development(teacher, learner)
        assert len(zpd) == 0

    def test_end_types_empty_zpd(self) -> None:
        zpd = zone_of_proximal_development(parse("end"), parse("end"))
        assert zpd == ()

    def test_zpd_continuations_are_session_types(self) -> None:
        teacher = parse("&{a: &{x: end}, b: end}")
        learner = parse("&{a: &{x: end}}")
        zpd = zone_of_proximal_development(teacher, learner)
        assert len(zpd) == 1  # method "b"
        assert isinstance(zpd[0], End)


# ---------------------------------------------------------------------------
# Tests: analyze_dialogue
# ---------------------------------------------------------------------------


class TestAnalyseDialogue:
    def test_pendular_is_turn_taking(self) -> None:
        ss = build_statespace(parse(PENDULAR_STR))
        result = analyze_dialogue(ss)
        assert result.is_turn_taking is True

    def test_non_pendular_branch_branch(self) -> None:
        ss = build_statespace(parse(NON_PENDULAR_STR))
        result = analyze_dialogue(ss)
        assert result.is_turn_taking is False

    def test_socratic_more_branches(self) -> None:
        # &{q1: +{a1: &{q2: &{q3: end}}}, q2: +{a2: end}} — 3 branch, 2 select
        socratic = "&{q1: +{a1: &{q2: &{q3: end}}}, q2: +{a2: end}}"
        ss = build_statespace(parse(socratic))
        result = analyze_dialogue(ss)
        assert result.is_socratic is True
        assert result.teacher_turns > result.learner_turns

    def test_dialogue_depth_pendular(self) -> None:
        ss = build_statespace(parse(PENDULAR_STR))
        result = analyze_dialogue(ss)
        assert result.dialogue_depth >= 2

    def test_end_only_balanced(self) -> None:
        ss = build_statespace(parse("end"))
        result = analyze_dialogue(ss)
        assert result.balance == 0.5
        assert result.teacher_turns == 0
        assert result.learner_turns == 0

    def test_smtp_dialogue(self) -> None:
        ss = build_statespace(parse(SMTP_STR))
        result = analyze_dialogue(ss)
        assert result.is_turn_taking is True
        assert result.teacher_turns > 0
        assert result.learner_turns > 0
        assert result.dialogue_depth >= 6

    def test_dialogue_result_is_frozen(self) -> None:
        ss = build_statespace(parse(PENDULAR_STR))
        result = analyze_dialogue(ss)
        with pytest.raises(AttributeError):
            result.is_turn_taking = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: developmental_stages
# ---------------------------------------------------------------------------


class TestDevelopmentalStages:
    def test_progression_classification(self) -> None:
        types = [
            parse(ITER_NOVICE_STR),
            parse(ITER_INTER_STR),
            parse(ITER_TARGET_STR),
        ]
        stages = developmental_stages(types)
        assert len(stages) == 3
        # First type: 1/3 methods => intermediate (0.333)
        assert stages[0][0] == "intermediate"
        # Second type: 2/3 methods => advanced (0.667)
        assert stages[1][0] == "advanced"
        # Last type: 3/3 methods => expert (1.0)
        assert stages[2][0] == "expert"
        assert stages[2][2] == 1.0

    def test_empty_list(self) -> None:
        assert developmental_stages([]) == []

    def test_single_type_is_expert(self) -> None:
        t = parse(TEACHER_STR)
        stages = developmental_stages([t])
        assert len(stages) == 1
        assert stages[0][0] == "expert"
        assert stages[0][2] == 1.0

    def test_all_end_types(self) -> None:
        types = [parse("end"), parse("end")]
        stages = developmental_stages(types)
        # No methods in target, so progress = 0/1 = 0 for all
        # Both have 0 methods, target has 0 => shared=0, total_target=1
        assert all(s[2] == 0.0 for s in stages)

    def test_progression_scores_increase(self) -> None:
        types = [
            parse("&{a: end}"),
            parse("&{a: end, b: end}"),
            parse("&{a: end, b: end, c: end}"),
            parse("&{a: end, b: end, c: end, d: end}"),
        ]
        stages = developmental_stages(types)
        scores = [s[2] for s in stages]
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# Tests: learning_path
# ---------------------------------------------------------------------------


class TestLearningPath:
    def test_gap_methods_ordered(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse(LEARNER_NOVICE_STR)
        path = learning_path(teacher, learner)
        assert len(path) == 3
        assert set(path) == {"close", "read", "write"}

    def test_no_gap_empty_path(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse(LEARNER_EXPERT_STR)
        path = learning_path(teacher, learner)
        assert path == []

    def test_simpler_methods_first(self) -> None:
        # "a" has deeper continuation than "b"
        teacher = parse("&{a: &{x: &{y: end}}, b: end, c: &{z: end}}")
        learner = parse("&{d: end}")  # no overlap with teacher
        path = learning_path(teacher, learner)
        # b (depth 0) < c (depth 1) < a (depth 2)
        assert path[0] == "b"
        assert path[-1] == "a"

    def test_end_teacher_empty_path(self) -> None:
        path = learning_path(parse("end"), parse("end"))
        assert path == []


# ---------------------------------------------------------------------------
# Tests: classify_pedagogy
# ---------------------------------------------------------------------------


class TestClassifyPedagogy:
    def test_combined_analysis(self) -> None:
        teacher = parse(TEACHER_STR)
        learner = parse(LEARNER_NOVICE_STR)
        ss = build_statespace(teacher)
        result = classify_pedagogy(teacher, learner, ss)
        assert "scaffolding_result" in result
        assert "dialogue_result" in result
        assert "zpd_methods" in result
        assert "learning_path" in result
        assert isinstance(result["scaffolding_result"], ScaffoldingResult)
        assert isinstance(result["dialogue_result"], DialogueResult)
        assert isinstance(result["zpd_methods"], tuple)
        assert isinstance(result["learning_path"], list)

    def test_combined_identical_types(self) -> None:
        t = parse(TEACHER_STR)
        ss = build_statespace(t)
        result = classify_pedagogy(t, t, ss)
        assert result["scaffolding_result"].is_scaffold is False
        assert result["zpd_methods"] == ()
        assert result["learning_path"] == []


# ---------------------------------------------------------------------------
# Tests: _classify_stage
# ---------------------------------------------------------------------------


class TestClassifyStage:
    def test_novice(self) -> None:
        assert _classify_stage(0.0) == "novice"
        assert _classify_stage(0.24) == "novice"

    def test_intermediate(self) -> None:
        assert _classify_stage(0.25) == "intermediate"
        assert _classify_stage(0.49) == "intermediate"

    def test_advanced(self) -> None:
        assert _classify_stage(0.50) == "advanced"
        assert _classify_stage(0.74) == "advanced"

    def test_expert(self) -> None:
        assert _classify_stage(0.75) == "expert"
        assert _classify_stage(1.0) == "expert"
