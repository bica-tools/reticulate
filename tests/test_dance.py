"""Tests for dance module (Step 57)."""

from __future__ import annotations

import pytest

from reticulate.dance import (
    BodyPart,
    ChoreographyResult,
    DanceScore,
    Direction,
    Duration,
    EffortFlow,
    EffortShape,
    EffortSpace,
    EffortTime,
    EffortWeight,
    Level,
    Movement,
    MovementPhrase,
    arabesque,
    choreography_lattice,
    dance_score,
    encode_movement,
    encode_phrase,
    lma_effort,
    parallel_body_parts,
    plie_sequence,
    waltz_basic,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Movement data type tests
# ---------------------------------------------------------------------------

class TestMovement:
    def test_creation(self) -> None:
        m = Movement(BodyPart.LEFT_ARM, Direction.FORWARD, Level.HIGH)
        assert m.body_part == BodyPart.LEFT_ARM
        assert m.direction == Direction.FORWARD
        assert m.level == Level.HIGH

    def test_label(self) -> None:
        m = Movement(BodyPart.RIGHT_LEG, Direction.UP, Level.MIDDLE)
        assert m.label == "R_leg_up_mid"

    def test_short_label(self) -> None:
        m = Movement(BodyPart.RIGHT_LEG, Direction.UP, Level.MIDDLE)
        assert m.short_label == "up_mid"

    def test_frozen(self) -> None:
        m = Movement(BodyPart.LEFT_ARM, Direction.FORWARD)
        with pytest.raises(AttributeError):
            m.direction = Direction.BACKWARD  # type: ignore[misc]

    def test_defaults(self) -> None:
        m = Movement(BodyPart.HEAD, Direction.PLACE)
        assert m.level == Level.MIDDLE
        assert m.duration == Duration.NORMAL


# ---------------------------------------------------------------------------
# EffortShape tests
# ---------------------------------------------------------------------------

class TestEffortShape:
    def test_effort_name_punch(self) -> None:
        e = EffortShape(EffortWeight.STRONG, EffortTime.SUDDEN,
                        EffortSpace.DIRECT, EffortFlow.FREE)
        assert e.effort_name == "punch"

    def test_effort_name_float(self) -> None:
        e = EffortShape(EffortWeight.LIGHT, EffortTime.SUSTAINED,
                        EffortSpace.INDIRECT, EffortFlow.FREE)
        assert e.effort_name == "float"

    def test_effort_name_glide(self) -> None:
        e = EffortShape(EffortWeight.LIGHT, EffortTime.SUSTAINED,
                        EffortSpace.DIRECT, EffortFlow.BOUND)
        assert e.effort_name == "glide"

    def test_label(self) -> None:
        e = EffortShape(EffortWeight.STRONG, EffortTime.SUDDEN,
                        EffortSpace.DIRECT, EffortFlow.FREE)
        assert "strong" in e.label
        assert "sudden" in e.label


# ---------------------------------------------------------------------------
# encode_movement tests
# ---------------------------------------------------------------------------

class TestEncodeMovement:
    def test_single_movement(self) -> None:
        m = Movement(BodyPart.LEFT_ARM, Direction.FORWARD, Level.HIGH)
        result = encode_movement(m)
        assert "L_arm_fwd_high" in result

    def test_parseable(self) -> None:
        m = Movement(BodyPart.RIGHT_ARM, Direction.UP, Level.MIDDLE)
        result = encode_movement(m)
        st = parse(result)
        ss = build_statespace(st)
        assert len(ss.states) == 2  # one transition + end


# ---------------------------------------------------------------------------
# encode_phrase tests
# ---------------------------------------------------------------------------

class TestEncodePhrase:
    def test_empty_phrase(self) -> None:
        phrase = MovementPhrase(BodyPart.LEFT_ARM, ())
        result = encode_phrase(phrase)
        assert result == "end"

    def test_single_movement(self) -> None:
        phrase = MovementPhrase(BodyPart.LEFT_ARM, (
            Movement(BodyPart.LEFT_ARM, Direction.FORWARD),
        ))
        result = encode_phrase(phrase)
        assert "fwd_mid" in result

    def test_multi_movement(self) -> None:
        phrase = MovementPhrase(BodyPart.LEFT_ARM, (
            Movement(BodyPart.LEFT_ARM, Direction.FORWARD),
            Movement(BodyPart.LEFT_ARM, Direction.UP),
        ))
        result = encode_phrase(phrase)
        st = parse(result)
        ss = build_statespace(st)
        assert len(ss.states) == 3  # 2 transitions + end


# ---------------------------------------------------------------------------
# parallel_body_parts tests
# ---------------------------------------------------------------------------

class TestParallelBodyParts:
    def test_empty(self) -> None:
        result = parallel_body_parts(())
        assert result == "end"

    def test_single_part(self) -> None:
        phrase = MovementPhrase(BodyPart.LEFT_ARM, (
            Movement(BodyPart.LEFT_ARM, Direction.FORWARD),
        ))
        result = parallel_body_parts((phrase,))
        assert "fwd_mid" in result

    def test_two_parts_parallel(self) -> None:
        p1 = MovementPhrase(BodyPart.LEFT_ARM, (
            Movement(BodyPart.LEFT_ARM, Direction.FORWARD),
        ))
        p2 = MovementPhrase(BodyPart.RIGHT_ARM, (
            Movement(BodyPart.RIGHT_ARM, Direction.UP),
        ))
        result = parallel_body_parts((p1, p2))
        assert "||" in result

    def test_parseable(self) -> None:
        p1 = MovementPhrase(BodyPart.LEFT_ARM, (
            Movement(BodyPart.LEFT_ARM, Direction.FORWARD),
        ))
        p2 = MovementPhrase(BodyPart.RIGHT_ARM, (
            Movement(BodyPart.RIGHT_ARM, Direction.UP),
        ))
        result = parallel_body_parts((p1, p2))
        st = parse(result)
        ss = build_statespace(st)
        assert len(ss.states) > 1


# ---------------------------------------------------------------------------
# choreography_lattice tests
# ---------------------------------------------------------------------------

class TestChoreographyLattice:
    def test_basic(self) -> None:
        p1 = MovementPhrase(BodyPart.LEFT_ARM, (
            Movement(BodyPart.LEFT_ARM, Direction.FORWARD),
        ))
        result = choreography_lattice((p1,))
        assert isinstance(result, ChoreographyResult)
        assert result.num_body_parts == 1
        assert result.is_lattice

    def test_parallel_lattice(self) -> None:
        p1 = MovementPhrase(BodyPart.LEFT_ARM, (
            Movement(BodyPart.LEFT_ARM, Direction.FORWARD),
            Movement(BodyPart.LEFT_ARM, Direction.UP),
        ))
        p2 = MovementPhrase(BodyPart.RIGHT_ARM, (
            Movement(BodyPart.RIGHT_ARM, Direction.BACKWARD),
        ))
        result = choreography_lattice((p1, p2))
        assert result.is_lattice
        assert result.num_body_parts == 2

    def test_body_part_types(self) -> None:
        p1 = MovementPhrase(BodyPart.LEFT_ARM, (
            Movement(BodyPart.LEFT_ARM, Direction.FORWARD),
        ))
        result = choreography_lattice((p1,))
        assert "L_arm" in result.body_part_types


# ---------------------------------------------------------------------------
# lma_effort tests
# ---------------------------------------------------------------------------

class TestLmaEffort:
    def test_parseable(self) -> None:
        result = lma_effort()
        st = parse(result)
        ss = build_statespace(st)
        assert len(ss.states) > 1

    def test_is_lattice(self) -> None:
        result = lma_effort()
        st = parse(result)
        ss = build_statespace(st)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_contains_efforts(self) -> None:
        result = lma_effort()
        assert "punch" in result
        assert "float" in result


# ---------------------------------------------------------------------------
# Pre-built pattern tests
# ---------------------------------------------------------------------------

class TestWaltzBasic:
    def test_two_parts(self) -> None:
        phrases = waltz_basic()
        assert len(phrases) == 2

    def test_lattice(self) -> None:
        phrases = waltz_basic()
        result = choreography_lattice(phrases)
        assert result.is_lattice


class TestPlieSequence:
    def test_two_parts(self) -> None:
        phrases = plie_sequence()
        assert len(phrases) == 2

    def test_lattice(self) -> None:
        phrases = plie_sequence()
        result = choreography_lattice(phrases)
        assert result.is_lattice


class TestArabesque:
    def test_three_parts(self) -> None:
        phrases = arabesque()
        assert len(phrases) == 3

    def test_lattice(self) -> None:
        phrases = arabesque()
        result = choreography_lattice(phrases)
        assert result.is_lattice


# ---------------------------------------------------------------------------
# Dance score tests
# ---------------------------------------------------------------------------

class TestDanceScore:
    def test_basic(self) -> None:
        phrases = waltz_basic()
        score = DanceScore("Waltz", phrases, "moderate", "classical")
        result = dance_score(score)
        assert isinstance(result, ChoreographyResult)
        assert result.is_lattice
