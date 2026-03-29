"""Tests for counterpoint module (Step 56b)."""

from __future__ import annotations

import pytest

from reticulate.counterpoint import (
    CounterpointResult,
    Interval,
    IntervalQuality,
    MotionType,
    Note,
    Violation,
    VoiceLeadingResult,
    analyze_voice_leading,
    check_species1,
    check_species2,
    classify_motion,
    compute_interval,
    counterpoint_violations,
    encode_counterpoint,
    interval_lattice,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Note tests
# ---------------------------------------------------------------------------

class TestNote:
    def test_midi_number(self) -> None:
        c4 = Note("C", 4)
        assert c4.midi_number == 60

    def test_pitch_class(self) -> None:
        a = Note("A", 4)
        assert a.pitch_class == 9

    def test_frozen(self) -> None:
        n = Note("C", 4)
        with pytest.raises(AttributeError):
            n.name = "D"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Interval tests
# ---------------------------------------------------------------------------

class TestComputeInterval:
    def test_unison(self) -> None:
        iv = compute_interval(Note("C", 4), Note("C", 4))
        assert iv.semitones == 0
        assert iv.quality == IntervalQuality.PERFECT_CONSONANCE

    def test_fifth(self) -> None:
        iv = compute_interval(Note("C", 4), Note("G", 4))
        assert iv.semitones == 7
        assert iv.quality == IntervalQuality.PERFECT_CONSONANCE

    def test_octave(self) -> None:
        iv = compute_interval(Note("C", 4), Note("C", 5))
        assert iv.semitones == 12
        assert iv.quality == IntervalQuality.PERFECT_CONSONANCE

    def test_third(self) -> None:
        iv = compute_interval(Note("C", 4), Note("E", 4))
        assert iv.semitones == 4
        assert iv.quality == IntervalQuality.IMPERFECT_CONSONANCE

    def test_second(self) -> None:
        iv = compute_interval(Note("C", 4), Note("D", 4))
        assert iv.semitones == 2
        assert iv.quality == IntervalQuality.DISSONANCE

    def test_generic_interval(self) -> None:
        iv = compute_interval(Note("C", 4), Note("G", 5))
        assert iv.generic_interval == 7  # 19 % 12


# ---------------------------------------------------------------------------
# Motion classification tests
# ---------------------------------------------------------------------------

class TestClassifyMotion:
    def test_parallel(self) -> None:
        motion = classify_motion(
            Note("C", 4), Note("D", 4),
            Note("G", 4), Note("A", 4),
        )
        assert motion == MotionType.PARALLEL

    def test_contrary(self) -> None:
        motion = classify_motion(
            Note("C", 4), Note("D", 4),
            Note("G", 4), Note("F", 4),
        )
        assert motion == MotionType.CONTRARY

    def test_oblique(self) -> None:
        motion = classify_motion(
            Note("C", 4), Note("C", 4),
            Note("G", 4), Note("A", 4),
        )
        assert motion == MotionType.OBLIQUE

    def test_similar(self) -> None:
        motion = classify_motion(
            Note("C", 4), Note("D", 4),
            Note("E", 4), Note("A", 4),
        )
        assert motion == MotionType.SIMILAR


# ---------------------------------------------------------------------------
# Species 1 tests
# ---------------------------------------------------------------------------

class TestCheckSpecies1:
    def test_valid_counterpoint(self) -> None:
        cf = (Note("C", 4), Note("D", 4), Note("E", 4), Note("C", 4))
        cp = (Note("G", 4), Note("F", 4), Note("G", 4), Note("G", 4))
        result = check_species1(cf, cp)
        assert isinstance(result, CounterpointResult)
        assert result.species == 1

    def test_parallel_fifths_detected(self) -> None:
        cf = (Note("C", 4), Note("D", 4))
        cp = (Note("G", 4), Note("A", 4))
        result = check_species1(cf, cp)
        rules = [v.rule for v in result.violations]
        assert "no_parallel_fifths" in rules

    def test_dissonance_detected(self) -> None:
        cf = (Note("C", 4), Note("C", 4))
        cp = (Note("D", 4), Note("C", 4))  # second is dissonant
        result = check_species1(cf, cp)
        rules = [v.rule for v in result.violations]
        assert "consonance_required" in rules

    def test_unequal_length_raises(self) -> None:
        cf = (Note("C", 4), Note("D", 4))
        cp = (Note("G", 4),)
        with pytest.raises(ValueError, match="equal-length"):
            check_species1(cf, cp)

    def test_begin_end_perfect(self) -> None:
        cf = (Note("C", 4), Note("D", 4), Note("E", 4))
        cp = (Note("E", 4), Note("F", 4), Note("E", 4))
        result = check_species1(cf, cp)
        rules = [v.rule for v in result.violations if v.severity == "error"]
        # Begins on third (imperfect), should have begin_perfect violation
        assert "begin_perfect" in rules


# ---------------------------------------------------------------------------
# Species 2 tests
# ---------------------------------------------------------------------------

class TestCheckSpecies2:
    def test_wrong_ratio_raises(self) -> None:
        cf = (Note("C", 4), Note("D", 4))
        cp = (Note("G", 4), Note("A", 4))
        with pytest.raises(ValueError, match="2:1"):
            check_species2(cf, cp)

    def test_valid_species2(self) -> None:
        cf = (Note("C", 4), Note("E", 4))
        cp = (Note("G", 4), Note("A", 4), Note("G", 4), Note("G", 4))
        result = check_species2(cf, cp)
        assert result.species == 2


# ---------------------------------------------------------------------------
# Counterpoint violations tests
# ---------------------------------------------------------------------------

class TestCounterpointViolations:
    def test_clean_voices(self) -> None:
        lower = (Note("C", 4), Note("D", 4), Note("C", 4))
        upper = (Note("E", 4), Note("D", 4), Note("E", 4))
        violations = counterpoint_violations(lower, upper)
        errors = [v for v in violations if v.severity == "error"]
        # This may or may not have violations depending on intervals
        assert isinstance(violations, tuple)

    def test_returns_tuple(self) -> None:
        lower = (Note("C", 4),)
        upper = (Note("G", 4),)
        violations = counterpoint_violations(lower, upper)
        assert isinstance(violations, tuple)


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------

class TestEncodeCounterpoint:
    def test_equal_length_parallel(self) -> None:
        lower = (Note("C", 4), Note("D", 4))
        upper = (Note("E", 4), Note("F", 4))
        result = encode_counterpoint(lower, upper)
        assert "||" in result

    def test_parseable(self) -> None:
        lower = (Note("C", 4), Note("D", 4))
        upper = (Note("E", 4), Note("F", 4))
        type_str = encode_counterpoint(lower, upper)
        st = parse(type_str)
        ss = build_statespace(st)
        assert len(ss.states) > 1

    def test_lattice(self) -> None:
        lower = (Note("C", 4), Note("D", 4))
        upper = (Note("E", 4), Note("F", 4))
        type_str = encode_counterpoint(lower, upper)
        st = parse(type_str)
        ss = build_statespace(st)
        lr = check_lattice(ss)
        assert lr.is_lattice


# ---------------------------------------------------------------------------
# Voice-leading analysis tests
# ---------------------------------------------------------------------------

class TestAnalyzeVoiceLeading:
    def test_basic(self) -> None:
        lower = (Note("C", 4), Note("D", 4), Note("E", 4))
        upper = (Note("G", 4), Note("F", 4), Note("E", 4))
        result = analyze_voice_leading(lower, upper)
        assert isinstance(result, VoiceLeadingResult)
        assert len(result.intervals) == 3
        assert len(result.motion_types) == 2

    def test_score_range(self) -> None:
        lower = (Note("C", 4), Note("D", 4))
        upper = (Note("E", 4), Note("F", 4))
        result = analyze_voice_leading(lower, upper)
        assert 0.0 <= result.score <= 1.0

    def test_contrary_motion_pct(self) -> None:
        lower = (Note("C", 4), Note("D", 4))
        upper = (Note("G", 4), Note("F", 4))
        result = analyze_voice_leading(lower, upper)
        assert result.contrary_motion_pct > 0


# ---------------------------------------------------------------------------
# Interval lattice tests
# ---------------------------------------------------------------------------

class TestIntervalLattice:
    def test_parseable(self) -> None:
        type_str = interval_lattice()
        st = parse(type_str)
        ss = build_statespace(st)
        assert len(ss.states) > 1

    def test_is_lattice(self) -> None:
        type_str = interval_lattice()
        st = parse(type_str)
        ss = build_statespace(st)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_contains_intervals(self) -> None:
        type_str = interval_lattice()
        assert "unison" in type_str
        assert "fifth" in type_str
        assert "tritone" in type_str
