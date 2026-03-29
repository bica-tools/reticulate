"""Counterpoint type-checker via session types (Step 56b).

Species counterpoint has strict rules governing how two or more
melodic voices move simultaneously.  This module encodes counterpoint
rules as session type constraints, enabling formal verification of
musical compositions against classical counterpoint theory.

Key rules modelled:

- **No parallel fifths**: Two voices must not move in parallel motion
  to a perfect fifth interval.
- **No parallel octaves**: Similarly forbidden for octaves/unisons.
- **Stepwise motion preferred**: Conjunct motion (steps) preferred over
  disjunct (leaps).
- **Contrary motion preferred**: Voices should generally move in
  opposite directions.

Species:
1. **First species**: Note-against-note (1:1 ratio).
2. **Second species**: Two notes against one (2:1).
3. **Third species**: Four notes against one (4:1).
4. **Fourth species**: Syncopation / suspensions.
5. **Fifth species**: Florid counterpoint (free combination).

This module provides:
    ``check_species1()``          -- verify first species rules.
    ``check_species2()``          -- verify second species rules.
    ``counterpoint_violations()`` -- list all rule violations.
    ``encode_counterpoint()``     -- build session type from voice pair.
    ``analyze_voice_leading()``   -- full voice-leading analysis.
    ``interval_lattice()``        -- lattice of consonance/dissonance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from reticulate.parser import (
    Branch,
    End,
    Parallel,
    Select,
    SessionType,
    parse,
    pretty,
)
from reticulate.statespace import build_statespace, StateSpace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Musical data types
# ---------------------------------------------------------------------------

# Chromatic pitch classes (0 = C, 1 = C#, ..., 11 = B)
_NOTE_TO_PITCH = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}
_PITCH_TO_NOTE = {v: k for k, v in _NOTE_TO_PITCH.items()}


class MotionType(Enum):
    """Type of motion between two voices."""
    PARALLEL = auto()
    SIMILAR = auto()
    CONTRARY = auto()
    OBLIQUE = auto()


class IntervalQuality(Enum):
    """Consonance classification of an interval."""
    PERFECT_CONSONANCE = auto()   # unison, fifth, octave
    IMPERFECT_CONSONANCE = auto() # third, sixth
    DISSONANCE = auto()           # second, fourth, seventh, tritone


@dataclass(frozen=True)
class Note:
    """A note with pitch class and octave."""
    name: str       # "C", "D", etc.
    octave: int     # 3, 4, 5, etc.

    @property
    def midi_number(self) -> int:
        return _NOTE_TO_PITCH[self.name] + (self.octave + 1) * 12

    @property
    def pitch_class(self) -> int:
        return _NOTE_TO_PITCH[self.name]


@dataclass(frozen=True)
class Interval:
    """An interval between two notes."""
    semitones: int
    quality: IntervalQuality

    @property
    def generic_interval(self) -> int:
        """Simple interval in scale steps (mod 12)."""
        return self.semitones % 12


@dataclass(frozen=True)
class Violation:
    """A counterpoint rule violation."""
    position: int          # beat index
    rule: str              # rule name
    description: str       # human-readable
    severity: str = "error"  # "error" or "warning"


@dataclass(frozen=True)
class VoiceLeadingResult:
    """Result of voice-leading analysis."""
    motion_types: tuple[MotionType, ...]
    intervals: tuple[Interval, ...]
    violations: tuple[Violation, ...]
    parallel_motion_pct: float
    contrary_motion_pct: float
    conjunct_motion_pct: float
    score: float  # 0-1, higher = better voice leading


@dataclass(frozen=True)
class CounterpointResult:
    """Result of counterpoint checking."""
    is_valid: bool
    species: int
    violations: tuple[Violation, ...]
    session_type_str: str
    state_count: int
    is_lattice: bool


# ---------------------------------------------------------------------------
# Interval computation
# ---------------------------------------------------------------------------

def compute_interval(n1: Note, n2: Note) -> Interval:
    """Compute the interval between two notes."""
    semitones = abs(n2.midi_number - n1.midi_number)
    simple = semitones % 12
    if simple in (0, 7):  # unison, fifth
        quality = IntervalQuality.PERFECT_CONSONANCE
    elif simple in (3, 4, 8, 9):  # minor/major third, minor/major sixth
        quality = IntervalQuality.IMPERFECT_CONSONANCE
    else:  # 1, 2, 5, 6, 10, 11 = seconds, fourth, tritone, seventh
        quality = IntervalQuality.DISSONANCE
    return Interval(semitones=semitones, quality=quality)


def classify_motion(
    lower1: Note, lower2: Note,
    upper1: Note, upper2: Note,
) -> MotionType:
    """Classify the type of motion between two voice pairs."""
    lower_dir = upper1.midi_number - lower1.midi_number
    upper_dir = upper2.midi_number - lower2.midi_number
    d_lower = lower2.midi_number - lower1.midi_number
    d_upper = upper2.midi_number - upper1.midi_number

    if d_lower == 0 and d_upper == 0:
        return MotionType.OBLIQUE
    if d_lower == 0 or d_upper == 0:
        return MotionType.OBLIQUE
    if d_lower == d_upper:
        return MotionType.PARALLEL
    if (d_lower > 0) == (d_upper > 0):
        return MotionType.SIMILAR
    return MotionType.CONTRARY


def _is_step(n1: Note, n2: Note) -> bool:
    """Check if motion between notes is stepwise (<=2 semitones)."""
    return abs(n2.midi_number - n1.midi_number) <= 2


# ---------------------------------------------------------------------------
# Species checking
# ---------------------------------------------------------------------------

def _check_parallel_perfects(
    lower: tuple[Note, ...],
    upper: tuple[Note, ...],
) -> list[Violation]:
    """Check for parallel fifths and octaves."""
    violations: list[Violation] = []
    for i in range(len(lower) - 1):
        iv1 = compute_interval(lower[i], upper[i])
        iv2 = compute_interval(lower[i + 1], upper[i + 1])
        motion = classify_motion(lower[i], lower[i + 1],
                                 upper[i], upper[i + 1])
        if motion == MotionType.PARALLEL:
            if iv1.generic_interval == 7 and iv2.generic_interval == 7:
                violations.append(Violation(
                    position=i,
                    rule="no_parallel_fifths",
                    description=f"Parallel fifths at beat {i}-{i+1}",
                ))
            if iv1.generic_interval == 0 and iv2.generic_interval == 0:
                violations.append(Violation(
                    position=i,
                    rule="no_parallel_octaves",
                    description=f"Parallel octaves/unisons at beat {i}-{i+1}",
                ))
    return violations


def _check_dissonances(
    lower: tuple[Note, ...],
    upper: tuple[Note, ...],
) -> list[Violation]:
    """Check for dissonant intervals (species 1: all must be consonant)."""
    violations: list[Violation] = []
    for i in range(len(lower)):
        iv = compute_interval(lower[i], upper[i])
        if iv.quality == IntervalQuality.DISSONANCE:
            violations.append(Violation(
                position=i,
                rule="consonance_required",
                description=f"Dissonant interval ({iv.semitones} semitones) at beat {i}",
            ))
    return violations


def _check_stepwise(voice: tuple[Note, ...]) -> list[Violation]:
    """Check for excessive leaps (warn on non-stepwise motion)."""
    violations: list[Violation] = []
    for i in range(len(voice) - 1):
        if not _is_step(voice[i], voice[i + 1]):
            leap = abs(voice[i + 1].midi_number - voice[i].midi_number)
            violations.append(Violation(
                position=i,
                rule="stepwise_preferred",
                description=f"Leap of {leap} semitones at beat {i}-{i+1}",
                severity="warning",
            ))
    return violations


def check_species1(
    cantus_firmus: tuple[Note, ...],
    counterpoint_voice: tuple[Note, ...],
) -> CounterpointResult:
    """Verify first species counterpoint (note-against-note).

    Rules:
    - All intervals must be consonant.
    - No parallel fifths or octaves.
    - Begin and end on perfect consonance.
    - Stepwise motion preferred.
    """
    if len(cantus_firmus) != len(counterpoint_voice):
        raise ValueError("First species requires equal-length voices")

    violations: list[Violation] = []
    violations.extend(_check_parallel_perfects(cantus_firmus, counterpoint_voice))
    violations.extend(_check_dissonances(cantus_firmus, counterpoint_voice))
    violations.extend(_check_stepwise(counterpoint_voice))

    # Begin/end on perfect consonance
    if cantus_firmus:
        first_iv = compute_interval(cantus_firmus[0], counterpoint_voice[0])
        if first_iv.quality != IntervalQuality.PERFECT_CONSONANCE:
            violations.append(Violation(
                position=0,
                rule="begin_perfect",
                description="Must begin on perfect consonance",
            ))
        last_iv = compute_interval(cantus_firmus[-1], counterpoint_voice[-1])
        if last_iv.quality != IntervalQuality.PERFECT_CONSONANCE:
            violations.append(Violation(
                position=len(cantus_firmus) - 1,
                rule="end_perfect",
                description="Must end on perfect consonance",
            ))

    st_str = encode_counterpoint(cantus_firmus, counterpoint_voice)
    st = parse(st_str)
    ss = build_statespace(st)
    lr = check_lattice(ss)

    errors = [v for v in violations if v.severity == "error"]
    return CounterpointResult(
        is_valid=len(errors) == 0,
        species=1,
        violations=tuple(violations),
        session_type_str=st_str,
        state_count=len(ss.states),
        is_lattice=lr.is_lattice,
    )


def check_species2(
    cantus_firmus: tuple[Note, ...],
    counterpoint_voice: tuple[Note, ...],
) -> CounterpointResult:
    """Verify second species counterpoint (2:1 ratio).

    The counterpoint voice has twice as many notes as the cantus firmus.
    Dissonances allowed on weak beats only (passing tones).
    """
    if len(counterpoint_voice) != 2 * len(cantus_firmus):
        raise ValueError("Second species requires 2:1 note ratio")

    violations: list[Violation] = []

    # Check strong beats (consonance required)
    strong_lower = cantus_firmus
    strong_upper = tuple(counterpoint_voice[i * 2] for i in range(len(cantus_firmus)))
    violations.extend(_check_dissonances(strong_lower, strong_upper))
    violations.extend(_check_parallel_perfects(strong_lower, strong_upper))

    # Weak beats: dissonance OK if stepwise (passing tone)
    for i in range(len(cantus_firmus)):
        weak_idx = i * 2 + 1
        if weak_idx < len(counterpoint_voice):
            iv = compute_interval(cantus_firmus[i], counterpoint_voice[weak_idx])
            if iv.quality == IntervalQuality.DISSONANCE:
                # Must be stepwise approach and departure
                prev_step = _is_step(counterpoint_voice[weak_idx - 1],
                                     counterpoint_voice[weak_idx])
                next_step = (weak_idx + 1 < len(counterpoint_voice) and
                             _is_step(counterpoint_voice[weak_idx],
                                      counterpoint_voice[weak_idx + 1]))
                if not (prev_step and next_step):
                    violations.append(Violation(
                        position=weak_idx,
                        rule="passing_tone",
                        description=f"Dissonance at weak beat {weak_idx} not a passing tone",
                    ))

    st_str = encode_counterpoint(cantus_firmus, counterpoint_voice)
    st = parse(st_str)
    ss = build_statespace(st)
    lr = check_lattice(ss)

    errors = [v for v in violations if v.severity == "error"]
    return CounterpointResult(
        is_valid=len(errors) == 0,
        species=2,
        violations=tuple(violations),
        session_type_str=st_str,
        state_count=len(ss.states),
        is_lattice=lr.is_lattice,
    )


def counterpoint_violations(
    lower: tuple[Note, ...],
    upper: tuple[Note, ...],
) -> tuple[Violation, ...]:
    """List all counterpoint rule violations for a voice pair."""
    violations: list[Violation] = []
    violations.extend(_check_parallel_perfects(lower, upper))
    violations.extend(_check_dissonances(lower, upper))
    violations.extend(_check_stepwise(lower))
    violations.extend(_check_stepwise(upper))
    return tuple(violations)


# ---------------------------------------------------------------------------
# Encoding as session types
# ---------------------------------------------------------------------------

def encode_counterpoint(
    lower: tuple[Note, ...],
    upper: tuple[Note, ...],
) -> str:
    """Encode a two-voice counterpoint as a session type.

    Each voice becomes a branch chain; the two voices run in parallel
    if they have equal length. Otherwise, they are sequential branches.
    """
    def _voice_chain(voice: tuple[Note, ...]) -> SessionType:
        current: SessionType = End()
        for note in reversed(voice):
            label = f"{note.name}{note.octave}"
            current = Branch(((label, current),))
        return current

    lower_st = _voice_chain(lower)
    upper_st = _voice_chain(upper)

    if len(lower) == len(upper):
        st = Parallel((lower_st, upper_st))
    else:
        # Different lengths: sequential
        choices = [("lower", lower_st), ("upper", upper_st)]
        st = Branch(tuple(choices))

    return pretty(st)


# ---------------------------------------------------------------------------
# Voice-leading analysis
# ---------------------------------------------------------------------------

def analyze_voice_leading(
    lower: tuple[Note, ...],
    upper: tuple[Note, ...],
) -> VoiceLeadingResult:
    """Full voice-leading analysis of two voices."""
    motions: list[MotionType] = []
    intervals: list[Interval] = []

    n = min(len(lower), len(upper))
    for i in range(n):
        intervals.append(compute_interval(lower[i], upper[i]))
    for i in range(n - 1):
        motions.append(classify_motion(lower[i], lower[i + 1],
                                       upper[i], upper[i + 1]))

    violations = list(counterpoint_violations(lower[:n], upper[:n]))

    parallel_count = sum(1 for m in motions if m == MotionType.PARALLEL)
    contrary_count = sum(1 for m in motions if m == MotionType.CONTRARY)
    total_motions = len(motions) if motions else 1

    # Conjunct motion in upper voice
    conjunct = 0
    for i in range(len(upper) - 1):
        if _is_step(upper[i], upper[i + 1]):
            conjunct += 1
    total_steps = max(len(upper) - 1, 1)

    # Score: penalize violations, reward contrary motion
    error_count = sum(1 for v in violations if v.severity == "error")
    warn_count = sum(1 for v in violations if v.severity == "warning")
    raw_score = 1.0 - (error_count * 0.2 + warn_count * 0.05)
    raw_score += contrary_count / total_motions * 0.1
    score = max(0.0, min(1.0, raw_score))

    return VoiceLeadingResult(
        motion_types=tuple(motions),
        intervals=tuple(intervals),
        violations=tuple(violations),
        parallel_motion_pct=parallel_count / total_motions,
        contrary_motion_pct=contrary_count / total_motions,
        conjunct_motion_pct=conjunct / total_steps,
        score=score,
    )


# ---------------------------------------------------------------------------
# Interval lattice
# ---------------------------------------------------------------------------

def interval_lattice() -> str:
    """Build a session type whose state space models the consonance lattice.

    Perfect consonances < imperfect consonances < dissonances.
    Represented as branch choices ordered by consonance.
    """
    st = Branch((
        ("perfect", Branch((
            ("unison", End()),
            ("fifth", End()),
            ("octave", End()),
        ))),
        ("imperfect", Branch((
            ("third", End()),
            ("sixth", End()),
        ))),
        ("dissonance", Branch((
            ("second", End()),
            ("fourth", End()),
            ("seventh", End()),
            ("tritone", End()),
        ))),
    ))
    return pretty(st)
