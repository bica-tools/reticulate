"""Tests for natural language as session type composition (Step 206)."""

from __future__ import annotations

import pytest

from reticulate import parse, build_statespace
from reticulate.lattice import check_lattice
from reticulate.language_theory import (
    SPEECH_ACT_LIBRARY,
    ConversationType,
    SentenceType,
    TranslationResult,
    classify_speech_act,
    compose_conversation,
    grice_check,
    poetic_score,
    rhetoric_power,
    translate,
)


# ---------------------------------------------------------------------------
# Speech Act Library
# ---------------------------------------------------------------------------


class TestSpeechActLibrary:
    """Tests for the speech act library contents."""

    def test_library_has_at_least_12_acts(self) -> None:
        assert len(SPEECH_ACT_LIBRARY) >= 12

    def test_all_acts_parse(self) -> None:
        for name, (type_str, _sa, _il) in SPEECH_ACT_LIBRARY.items():
            ast = parse(type_str)
            assert ast is not None, f"Failed to parse speech act {name}"

    def test_all_acts_build_statespace(self) -> None:
        for name, (type_str, _sa, _il) in SPEECH_ACT_LIBRARY.items():
            ast = parse(type_str)
            ss = build_statespace(ast)
            assert len(ss.states) >= 2, f"Speech act {name} too small"

    def test_all_acts_form_lattices(self) -> None:
        for name, (type_str, _sa, _il) in SPEECH_ACT_LIBRARY.items():
            ast = parse(type_str)
            ss = build_statespace(ast)
            result = check_lattice(ss)
            assert result.is_lattice, f"Speech act {name} is not a lattice"

    def test_all_acts_have_valid_speech_act(self) -> None:
        valid = {"assertion", "question", "command", "promise", "greeting",
                 "apology", "negotiation", "story", "argument", "lecture",
                 "confession", "prayer"}
        for name, (_ts, sa, _il) in SPEECH_ACT_LIBRARY.items():
            assert sa in valid, f"Speech act {name} has invalid type {sa}"

    def test_all_acts_have_valid_force(self) -> None:
        valid = {"inform", "request", "commit", "direct", "express"}
        for name, (_ts, _sa, il) in SPEECH_ACT_LIBRARY.items():
            assert il in valid, f"Speech act {name} has invalid force {il}"


# ---------------------------------------------------------------------------
# classify_speech_act
# ---------------------------------------------------------------------------


class TestClassifySpeechAct:
    """Tests for classify_speech_act."""

    def test_branch_classified_as_question(self) -> None:
        result = classify_speech_act("&{a: end, b: end}")
        assert result == "question"

    def test_select_classified_as_command(self) -> None:
        result = classify_speech_act("+{a: end, b: end}")
        assert result == "command"

    def test_recursive_classified_as_conversation(self) -> None:
        result = classify_speech_act("rec X . &{a: X, b: end}")
        assert result == "conversation"

    def test_end_classified_as_silence(self) -> None:
        result = classify_speech_act("end")
        assert result == "silence"

    def test_assertion_type_classified(self) -> None:
        type_str = SPEECH_ACT_LIBRARY["assertion"][0]
        result = classify_speech_act(type_str)
        assert result == "question"  # Branch at top

    def test_command_type_classified(self) -> None:
        type_str = SPEECH_ACT_LIBRARY["command"][0]
        result = classify_speech_act(type_str)
        assert result == "command"  # Select at top

    def test_negotiation_classified_as_conversation(self) -> None:
        type_str = SPEECH_ACT_LIBRARY["negotiation"][0]
        result = classify_speech_act(type_str)
        assert result == "conversation"  # Recursive


# ---------------------------------------------------------------------------
# compose_conversation
# ---------------------------------------------------------------------------


class TestComposeConversation:
    """Tests for compose_conversation."""

    def test_empty_returns_end(self) -> None:
        result = compose_conversation([])
        assert "end" in result.lower()

    def test_single_act(self) -> None:
        result = compose_conversation(["greeting"])
        ast = parse(result)
        assert ast is not None

    def test_two_acts_compose(self) -> None:
        result = compose_conversation(["greeting", "assertion"])
        ast = parse(result)
        ss = build_statespace(ast)
        assert len(ss.states) >= 3

    def test_composed_forms_lattice(self) -> None:
        result = compose_conversation(["greeting", "apology"])
        ast = parse(result)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_unknown_act_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown speech act"):
            compose_conversation(["nonexistent"])

    def test_three_acts_compose(self) -> None:
        result = compose_conversation(["greeting", "question", "apology"])
        ast = parse(result)
        assert ast is not None


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------


class TestTranslate:
    """Tests for translate."""

    def test_returns_translation_result(self) -> None:
        result = translate("greeting", {"hello": "bonjour", "hello_back": "bonjour_retour"})
        assert isinstance(result, TranslationResult)

    def test_source_preserved(self) -> None:
        result = translate("greeting", {"hello": "hola"})
        assert result.source.speech_act == "greeting"

    def test_target_has_new_labels(self) -> None:
        result = translate("greeting", {"hello": "bonjour"})
        assert "bonjour" in result.target.session_type_str

    def test_fidelity_is_float(self) -> None:
        result = translate("greeting", {"hello": "hi"})
        assert isinstance(result.fidelity, float)
        assert result.fidelity >= 0.0

    def test_faithful_when_high_fidelity(self) -> None:
        # Identity mapping should be perfectly faithful.
        result = translate("greeting", {})
        assert result.fidelity >= 0.8
        assert result.is_faithful is True

    def test_unknown_act_raises(self) -> None:
        with pytest.raises(KeyError):
            translate("nonexistent", {})


# ---------------------------------------------------------------------------
# grice_check
# ---------------------------------------------------------------------------


class TestGriceCheck:
    """Tests for grice_check."""

    def test_returns_dict_with_four_keys(self) -> None:
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        result = grice_check(ss)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"quantity", "quality", "relation", "manner"}

    def test_values_are_booleans(self) -> None:
        ast = parse("&{a: end}")
        ss = build_statespace(ast)
        result = grice_check(ss)
        for v in result.values():
            assert isinstance(v, bool)

    def test_simple_type_is_cooperative(self) -> None:
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        result = grice_check(ss)
        assert result["quality"] is True
        assert result["relation"] is True

    def test_all_speech_acts_have_grice_results(self) -> None:
        for name, (type_str, _sa, _il) in SPEECH_ACT_LIBRARY.items():
            ast = parse(type_str)
            ss = build_statespace(ast)
            result = grice_check(ss)
            assert len(result) == 4, f"Grice check failed for {name}"


# ---------------------------------------------------------------------------
# poetic_score
# ---------------------------------------------------------------------------


class TestPoeticScore:
    """Tests for poetic_score."""

    def test_returns_non_negative_float(self) -> None:
        score = poetic_score("&{a: end, b: end}")
        assert isinstance(score, float)
        assert score >= 0.0

    def test_end_has_score(self) -> None:
        score = poetic_score("end")
        assert score >= 0.0

    def test_symmetric_type_scores_well(self) -> None:
        score = poetic_score("&{a: end, b: end}")
        assert score > 0.0

    def test_all_speech_acts_have_poetic_score(self) -> None:
        for name, (type_str, _sa, _il) in SPEECH_ACT_LIBRARY.items():
            score = poetic_score(type_str)
            assert score >= 0.0, f"Poetic score failed for {name}"


# ---------------------------------------------------------------------------
# rhetoric_power
# ---------------------------------------------------------------------------


class TestRhetoricPower:
    """Tests for rhetoric_power."""

    def test_returns_non_negative_float(self) -> None:
        power = rhetoric_power("&{a: end, b: end}")
        assert isinstance(power, float)
        assert power >= 0.0

    def test_more_branches_more_power(self) -> None:
        p1 = rhetoric_power("&{a: end}")
        p2 = rhetoric_power("&{a: end, b: end, c: end}")
        # More branches = more paths = more rhetorical options
        assert p2 >= p1

    def test_all_speech_acts_have_rhetoric_power(self) -> None:
        for name, (type_str, _sa, _il) in SPEECH_ACT_LIBRARY.items():
            power = rhetoric_power(type_str)
            assert power >= 0.0, f"Rhetoric power failed for {name}"
