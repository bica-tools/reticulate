"""Labanotation to session types (Step 57).

Labanotation is the standard system for recording human movement,
used extensively in dance, martial arts, and physical therapy.
This module encodes Labanotation constructs as session types,
treating body parts as parallel voices and movements as transitions.

Key mappings:

- **Body parts** (left arm, right arm, torso, left leg, right leg)
  run in parallel composition, reflecting simultaneous movement.
- **Directions** (forward, backward, left, right, up, down) become
  transition labels within each body part's voice.
- **Levels** (high, middle, low) modify the direction labels.
- **Duration** is encoded in the label suffix (quick, slow, sustained).
- **Effort qualities** (Laban Movement Analysis) map to selection:
  the dancer chooses from available dynamics.

This module provides:
    ``encode_movement()``       -- single movement as session type.
    ``parallel_body_parts()``   -- multi-limb movement as parallel type.
    ``choreography_lattice()``  -- analyze choreography lattice structure.
    ``encode_phrase()``         -- movement phrase as branch chain.
    ``lma_effort()``            -- Laban effort qualities as session type.
    ``dance_score()``           -- complete dance score encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from reticulate.parser import (
    Branch,
    End,
    Parallel,
    Rec,
    Select,
    SessionType,
    Var,
    Wait,
    parse,
    pretty,
)
from reticulate.statespace import build_statespace, StateSpace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Labanotation data types
# ---------------------------------------------------------------------------

class Direction(Enum):
    """Spatial direction in Labanotation."""
    FORWARD = "fwd"
    BACKWARD = "bwd"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    PLACE = "place"        # movement in place


class Level(Enum):
    """Height level of movement."""
    HIGH = "high"
    MIDDLE = "mid"
    LOW = "low"


class BodyPart(Enum):
    """Body parts in Labanotation staff."""
    LEFT_ARM = "L_arm"
    RIGHT_ARM = "R_arm"
    LEFT_LEG = "L_leg"
    RIGHT_LEG = "R_leg"
    TORSO = "torso"
    HEAD = "head"


class Duration(Enum):
    """Movement duration."""
    QUICK = "quick"
    NORMAL = "normal"
    SLOW = "slow"
    SUSTAINED = "sustained"


class EffortWeight(Enum):
    """Laban effort weight quality."""
    LIGHT = "light"
    STRONG = "strong"


class EffortTime(Enum):
    """Laban effort time quality."""
    SUDDEN = "sudden"
    SUSTAINED = "sustained"


class EffortSpace(Enum):
    """Laban effort space quality."""
    DIRECT = "direct"
    INDIRECT = "indirect"


class EffortFlow(Enum):
    """Laban effort flow quality."""
    BOUND = "bound"
    FREE = "free"


@dataclass(frozen=True)
class Movement:
    """A single movement instruction."""
    body_part: BodyPart
    direction: Direction
    level: Level = Level.MIDDLE
    duration: Duration = Duration.NORMAL

    @property
    def label(self) -> str:
        return f"{self.body_part.value}_{self.direction.value}_{self.level.value}"

    @property
    def short_label(self) -> str:
        return f"{self.direction.value}_{self.level.value}"


@dataclass(frozen=True)
class MovementPhrase:
    """A sequence of movements for one body part."""
    body_part: BodyPart
    movements: tuple[Movement, ...]


@dataclass(frozen=True)
class EffortShape:
    """Laban Movement Analysis effort qualities."""
    weight: EffortWeight
    time: EffortTime
    space: EffortSpace
    flow: EffortFlow

    @property
    def label(self) -> str:
        return f"{self.weight.value}_{self.time.value}_{self.space.value}_{self.flow.value}"

    @property
    def effort_name(self) -> str:
        """The classical eight effort actions."""
        mapping = {
            (EffortWeight.LIGHT, EffortTime.SUDDEN, EffortSpace.DIRECT): "dab",
            (EffortWeight.LIGHT, EffortTime.SUDDEN, EffortSpace.INDIRECT): "flick",
            (EffortWeight.LIGHT, EffortTime.SUSTAINED, EffortSpace.DIRECT): "glide",
            (EffortWeight.LIGHT, EffortTime.SUSTAINED, EffortSpace.INDIRECT): "float",
            (EffortWeight.STRONG, EffortTime.SUDDEN, EffortSpace.DIRECT): "punch",
            (EffortWeight.STRONG, EffortTime.SUDDEN, EffortSpace.INDIRECT): "slash",
            (EffortWeight.STRONG, EffortTime.SUSTAINED, EffortSpace.DIRECT): "press",
            (EffortWeight.STRONG, EffortTime.SUSTAINED, EffortSpace.INDIRECT): "wring",
        }
        return mapping.get((self.weight, self.time, self.space), "unknown")


@dataclass(frozen=True)
class ChoreographyResult:
    """Result of choreography lattice analysis."""
    session_type_str: str
    num_body_parts: int
    total_movements: int
    state_count: int
    transition_count: int
    is_lattice: bool
    body_part_types: dict[str, str]  # body_part -> its type string


@dataclass(frozen=True)
class DanceScore:
    """A complete dance score."""
    title: str
    phrases: tuple[MovementPhrase, ...]
    tempo: str = "moderate"
    style: str = ""


# ---------------------------------------------------------------------------
# Core encoding
# ---------------------------------------------------------------------------

def encode_movement(movement: Movement) -> str:
    """Encode a single movement as a session type string.

    A single movement maps to a single-choice branch: &{label: end}.
    """
    st = Branch(((movement.label, End()),))
    return pretty(st)


def encode_phrase(phrase: MovementPhrase) -> str:
    """Encode a movement phrase as a session type.

    Sequential movements become nested branches.
    """
    if not phrase.movements:
        return pretty(End())

    current: SessionType = End()
    for mvt in reversed(phrase.movements):
        current = Branch(((mvt.short_label, current),))
    return pretty(current)


def _phrase_to_session_type(phrase: MovementPhrase) -> SessionType:
    """Convert a phrase to a session type AST."""
    if not phrase.movements:
        return End()
    current: SessionType = End()
    for mvt in reversed(phrase.movements):
        current = Branch(((mvt.short_label, current),))
    return current


def parallel_body_parts(phrases: tuple[MovementPhrase, ...]) -> str:
    """Encode simultaneous body part movements as parallel composition.

    Each body part's phrase becomes one branch of a parallel type.
    """
    if not phrases:
        return pretty(End())

    types: list[SessionType] = []
    for phrase in phrases:
        types.append(_phrase_to_session_type(phrase))

    if len(types) == 1:
        return pretty(types[0])

    st = Parallel(tuple(types))
    return pretty(st)


def choreography_lattice(
    phrases: tuple[MovementPhrase, ...],
) -> ChoreographyResult:
    """Analyze the lattice structure of a choreography.

    Builds the state space from parallel body parts and checks
    whether it forms a lattice.
    """
    type_str = parallel_body_parts(phrases)
    st = parse(type_str)
    ss = build_statespace(st)
    lr = check_lattice(ss)

    total_movements = sum(len(p.movements) for p in phrases)

    body_part_types: dict[str, str] = {}
    for phrase in phrases:
        bp_str = encode_phrase(phrase)
        body_part_types[phrase.body_part.value] = bp_str

    return ChoreographyResult(
        session_type_str=type_str,
        num_body_parts=len(phrases),
        total_movements=total_movements,
        state_count=len(ss.states),
        transition_count=len(ss.transitions),
        is_lattice=lr.is_lattice,
        body_part_types=body_part_types,
    )


# ---------------------------------------------------------------------------
# Laban Movement Analysis: Effort
# ---------------------------------------------------------------------------

def lma_effort() -> str:
    """Build a session type representing the eight basic effort actions.

    The dancer selects effort qualities (weight, time, space, flow),
    which combine into the eight fundamental effort actions:
    dab, flick, glide, float, punch, slash, press, wring.
    """
    efforts = [
        EffortShape(w, t, s, EffortFlow.FREE)
        for w in EffortWeight
        for t in EffortTime
        for s in EffortSpace
    ]
    choices: list[tuple[str, SessionType]] = []
    for e in efforts:
        choices.append((e.effort_name, End()))

    st = Select(tuple(choices))
    return pretty(st)


# ---------------------------------------------------------------------------
# Complete dance score
# ---------------------------------------------------------------------------

def dance_score(score: DanceScore) -> ChoreographyResult:
    """Encode and analyze a complete dance score.

    Delegates to choreography_lattice for the lattice analysis.
    """
    return choreography_lattice(score.phrases)


# ---------------------------------------------------------------------------
# Pre-built dance patterns
# ---------------------------------------------------------------------------

def waltz_basic() -> tuple[MovementPhrase, ...]:
    """Basic waltz step pattern (3/4 time)."""
    right_leg = MovementPhrase(
        body_part=BodyPart.RIGHT_LEG,
        movements=(
            Movement(BodyPart.RIGHT_LEG, Direction.FORWARD, Level.LOW),
            Movement(BodyPart.RIGHT_LEG, Direction.RIGHT, Level.MIDDLE),
            Movement(BodyPart.RIGHT_LEG, Direction.PLACE, Level.MIDDLE),
        ),
    )
    left_leg = MovementPhrase(
        body_part=BodyPart.LEFT_LEG,
        movements=(
            Movement(BodyPart.LEFT_LEG, Direction.PLACE, Level.MIDDLE),
            Movement(BodyPart.LEFT_LEG, Direction.LEFT, Level.MIDDLE),
            Movement(BodyPart.LEFT_LEG, Direction.PLACE, Level.LOW),
        ),
    )
    return (right_leg, left_leg)


def plie_sequence() -> tuple[MovementPhrase, ...]:
    """Classical ballet plie sequence."""
    legs = MovementPhrase(
        body_part=BodyPart.RIGHT_LEG,
        movements=(
            Movement(BodyPart.RIGHT_LEG, Direction.DOWN, Level.MIDDLE),
            Movement(BodyPart.RIGHT_LEG, Direction.DOWN, Level.LOW),
            Movement(BodyPart.RIGHT_LEG, Direction.UP, Level.MIDDLE),
        ),
    )
    arms = MovementPhrase(
        body_part=BodyPart.RIGHT_ARM,
        movements=(
            Movement(BodyPart.RIGHT_ARM, Direction.FORWARD, Level.LOW),
            Movement(BodyPart.RIGHT_ARM, Direction.UP, Level.MIDDLE),
            Movement(BodyPart.RIGHT_ARM, Direction.RIGHT, Level.MIDDLE),
        ),
    )
    return (legs, arms)


def arabesque() -> tuple[MovementPhrase, ...]:
    """Classical ballet arabesque position."""
    right_leg = MovementPhrase(
        body_part=BodyPart.RIGHT_LEG,
        movements=(
            Movement(BodyPart.RIGHT_LEG, Direction.BACKWARD, Level.HIGH),
        ),
    )
    left_leg = MovementPhrase(
        body_part=BodyPart.LEFT_LEG,
        movements=(
            Movement(BodyPart.LEFT_LEG, Direction.PLACE, Level.LOW),
        ),
    )
    right_arm = MovementPhrase(
        body_part=BodyPart.RIGHT_ARM,
        movements=(
            Movement(BodyPart.RIGHT_ARM, Direction.FORWARD, Level.HIGH),
        ),
    )
    return (right_leg, left_leg, right_arm)
