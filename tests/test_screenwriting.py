"""Tests for screenwriting module (Step 55)."""

from __future__ import annotations

import pytest

from reticulate.screenwriting import (
    Act,
    Beat,
    BeatSheetResult,
    CharacterArc,
    ComparisonResult,
    Scene,
    Screenplay,
    StructureAnalysis,
    analyze_structure,
    beat_sheet,
    compare_screenplays,
    dialogue_protocol,
    encode_character_arc,
    encode_screenplay,
    three_act_template,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Data type tests
# ---------------------------------------------------------------------------

class TestBeat:
    def test_creation(self) -> None:
        b = Beat("hook", "Opening moment", 0.5)
        assert b.name == "hook"
        assert b.description == "Opening moment"
        assert b.emotional_value == 0.5

    def test_frozen(self) -> None:
        b = Beat("hook")
        with pytest.raises(AttributeError):
            b.name = "other"  # type: ignore[misc]

    def test_defaults(self) -> None:
        b = Beat("test")
        assert b.description == ""
        assert b.emotional_value == 0.0


class TestScene:
    def test_creation(self) -> None:
        s = Scene("opening", (Beat("hook"),), "Interior", ("Alice",))
        assert s.name == "opening"
        assert len(s.beats) == 1
        assert s.location == "Interior"
        assert s.characters == ("Alice",)


class TestScreenplay:
    def test_creation(self) -> None:
        sp = Screenplay("Test Film", (Act("act1", ()),))
        assert sp.title == "Test Film"
        assert len(sp.acts) == 1


# ---------------------------------------------------------------------------
# encode_screenplay tests
# ---------------------------------------------------------------------------

class TestEncodeScreenplay:
    def test_empty_screenplay(self) -> None:
        sp = Screenplay("Empty", ())
        result = encode_screenplay(sp)
        assert result == "end"

    def test_single_beat(self) -> None:
        sp = Screenplay("Short", (
            Act("act1", (Scene("s1", (Beat("hook"),)),)),
        ))
        result = encode_screenplay(sp)
        assert "hook" in result

    def test_three_act(self) -> None:
        sp = three_act_template("My Film")
        result = encode_screenplay(sp)
        assert "act1" in result
        assert "act2" in result
        assert "act3" in result

    def test_parseable(self) -> None:
        sp = three_act_template()
        type_str = encode_screenplay(sp)
        st = parse(type_str)
        ss = build_statespace(st)
        assert len(ss.states) > 1

    def test_is_lattice(self) -> None:
        sp = three_act_template()
        type_str = encode_screenplay(sp)
        st = parse(type_str)
        ss = build_statespace(st)
        lr = check_lattice(ss)
        assert lr.is_lattice


# ---------------------------------------------------------------------------
# analyze_structure tests
# ---------------------------------------------------------------------------

class TestAnalyzeStructure:
    def test_basic(self) -> None:
        sp = three_act_template("Test")
        result = analyze_structure(sp)
        assert isinstance(result, StructureAnalysis)
        assert result.title == "Test"
        assert result.num_acts == 3
        assert result.num_beats > 0

    def test_state_count(self) -> None:
        sp = three_act_template()
        result = analyze_structure(sp)
        assert result.state_count > 1
        assert result.transition_count > 0

    def test_emotional_arc(self) -> None:
        sp = three_act_template()
        result = analyze_structure(sp)
        assert len(result.emotional_arc) == result.num_beats

    def test_structural_balance(self) -> None:
        sp = three_act_template()
        result = analyze_structure(sp)
        assert 0.0 <= result.structural_balance <= 1.0


# ---------------------------------------------------------------------------
# compare_screenplays tests
# ---------------------------------------------------------------------------

class TestCompareScreenplays:
    def test_identical(self) -> None:
        sp = three_act_template("A")
        result = compare_screenplays(sp, sp)
        assert result.structural_similarity == 1.0
        assert result.both_lattices

    def test_different(self) -> None:
        sp_a = Screenplay("A", (Act("act1", (Scene("s1", (Beat("hook"),)),)),))
        sp_b = Screenplay("B", (Act("act1", (Scene("s1", (Beat("conflict"),)),)),))
        result = compare_screenplays(sp_a, sp_b)
        assert isinstance(result, ComparisonResult)
        assert result.structural_similarity < 1.0

    def test_shared_beats(self) -> None:
        sp_a = Screenplay("A", (Act("act1", (
            Scene("s1", (Beat("hook"), Beat("setup"))),)),))
        sp_b = Screenplay("B", (Act("act1", (
            Scene("s1", (Beat("hook"), Beat("conflict"))),)),))
        result = compare_screenplays(sp_a, sp_b)
        assert result.shared_beat_count == 1  # "hook"


# ---------------------------------------------------------------------------
# Character arc tests
# ---------------------------------------------------------------------------

class TestCharacterArc:
    def test_positive_arc(self) -> None:
        arc = encode_character_arc("Hero", "positive")
        assert arc.character_name == "Hero"
        assert arc.arc_type == "positive"
        assert "ordinary_world" in arc.session_type_str
        assert len(arc.beats) == 10

    def test_negative_arc(self) -> None:
        arc = encode_character_arc("Villain", "negative")
        assert arc.arc_type == "negative"
        assert "fall" in arc.beats

    def test_flat_arc(self) -> None:
        arc = encode_character_arc("Mentor", "flat")
        assert len(arc.beats) == 3

    def test_circular_arc(self) -> None:
        arc = encode_character_arc("Wanderer", "circular")
        assert "return_home" in arc.beats

    def test_invalid_arc_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown arc type"):
            encode_character_arc("X", "invalid")

    def test_parseable(self) -> None:
        arc = encode_character_arc("Hero", "positive")
        st = parse(arc.session_type_str)
        ss = build_statespace(st)
        lr = check_lattice(ss)
        assert lr.is_lattice


# ---------------------------------------------------------------------------
# Beat sheet tests
# ---------------------------------------------------------------------------

class TestBeatSheet:
    def test_save_the_cat(self) -> None:
        result = beat_sheet()
        assert isinstance(result, BeatSheetResult)
        assert len(result.beats) == 15
        assert "opening_image" in result.beats
        assert "final_image" in result.beats

    def test_is_lattice(self) -> None:
        result = beat_sheet()
        assert result.is_lattice

    def test_state_count(self) -> None:
        result = beat_sheet()
        assert result.state_count == 16  # 15 beats + end state


# ---------------------------------------------------------------------------
# Dialogue protocol tests
# ---------------------------------------------------------------------------

class TestDialogueProtocol:
    def test_zero_exchanges(self) -> None:
        result = dialogue_protocol("Alice", "Bob", 0)
        assert result == "end"

    def test_single_exchange(self) -> None:
        result = dialogue_protocol("Alice", "Bob", 1)
        assert "Alice_speaks_0" in result
        assert "Bob_speaks_0" in result

    def test_parseable(self) -> None:
        result = dialogue_protocol("Alice", "Bob", 2)
        st = parse(result)
        ss = build_statespace(st)
        assert len(ss.states) > 1


# ---------------------------------------------------------------------------
# Three-act template tests
# ---------------------------------------------------------------------------

class TestThreeActTemplate:
    def test_structure(self) -> None:
        sp = three_act_template("Test")
        assert sp.title == "Test"
        assert len(sp.acts) == 3
        assert sp.acts[0].act_number == 1
        assert sp.acts[1].act_number == 2
        assert sp.acts[2].act_number == 3

    def test_scenes(self) -> None:
        sp = three_act_template()
        total_scenes = sum(len(a.scenes) for a in sp.acts)
        assert total_scenes == 5  # 2 + 2 + 1
