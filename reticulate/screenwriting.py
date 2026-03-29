"""Screenwriting models as session types (Step 55).

Screenplays follow highly structured protocols: three-act structure,
scene beats, character arcs, and dialogue exchanges.  This module
encodes screenplay structure as session types, enabling formal
analysis of dramatic structure through lattice theory.

Key concepts:

- **Act**: A major structural division (typically three acts).
- **Scene**: A continuous block of action in one location.
- **Beat**: The smallest unit of dramatic action (a shift in value).
- **Character arc**: A session type describing one character's journey.
- **Dialogue exchange**: Branch/select interaction between characters.

The three-act structure maps naturally to sequencing:

    rec X . &{act1: &{inciting_incident: ...},
              act2: &{midpoint: ..., crisis: ...},
              act3: &{climax: ..., resolution: end}}

This module provides:
    ``encode_screenplay()``      -- build session type from screenplay spec.
    ``analyze_structure()``      -- structural analysis of a screenplay type.
    ``compare_screenplays()``    -- compare two screenplays via morphism.
    ``encode_character_arc()``   -- single character's journey as session type.
    ``beat_sheet()``             -- Save the Cat beat sheet as session type.
    ``dialogue_protocol()``     -- dialogue exchange as multiparty protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from reticulate.lattice import check_lattice, LatticeResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Beat:
    """A dramatic beat -- the smallest unit of action."""
    name: str
    description: str = ""
    emotional_value: float = 0.0  # -1 (negative) to +1 (positive)


@dataclass(frozen=True)
class Scene:
    """A scene comprising a sequence of beats."""
    name: str
    beats: tuple[Beat, ...]
    location: str = ""
    characters: tuple[str, ...] = ()


@dataclass(frozen=True)
class Act:
    """An act comprising scenes."""
    name: str
    scenes: tuple[Scene, ...]
    act_number: int = 1


@dataclass(frozen=True)
class Screenplay:
    """A complete screenplay structure."""
    title: str
    acts: tuple[Act, ...]
    characters: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructureAnalysis:
    """Result of analyzing screenplay structure."""
    title: str
    session_type_str: str
    num_acts: int
    num_scenes: int
    num_beats: int
    state_count: int
    transition_count: int
    is_lattice: bool
    emotional_arc: tuple[float, ...]
    structural_balance: float  # 0-1, how balanced acts are


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing two screenplays."""
    title_a: str
    title_b: str
    structural_similarity: float  # 0-1
    shared_beat_count: int
    type_a_str: str
    type_b_str: str
    both_lattices: bool


@dataclass(frozen=True)
class CharacterArc:
    """A character's journey encoded as session type."""
    character_name: str
    arc_type: str  # "positive", "negative", "flat", "circular"
    session_type_str: str
    beats: tuple[str, ...]


@dataclass(frozen=True)
class BeatSheetResult:
    """Save the Cat beat sheet as session type."""
    session_type_str: str
    beats: tuple[str, ...]
    state_count: int
    is_lattice: bool


# ---------------------------------------------------------------------------
# Core encoding
# ---------------------------------------------------------------------------

def _beats_to_session_type(beats: tuple[Beat, ...]) -> SessionType:
    """Encode a sequence of beats as nested branches."""
    if not beats:
        return End()
    current: SessionType = End()
    for beat in reversed(beats):
        current = Branch(((beat.name, current),))
    return current


def _scene_to_session_type(scene: Scene) -> SessionType:
    """Encode a scene as a session type."""
    return _beats_to_session_type(scene.beats)


def _act_to_session_type(act: Act) -> SessionType:
    """Encode an act as a branch over its scenes."""
    if not act.scenes:
        return End()
    if len(act.scenes) == 1:
        return _scene_to_session_type(act.scenes[0])
    choices: list[tuple[str, SessionType]] = []
    for scene in act.scenes:
        choices.append((scene.name, _scene_to_session_type(scene)))
    return Branch(tuple(choices))


def encode_screenplay(screenplay: Screenplay) -> str:
    """Encode a screenplay as a session type string.

    Each act becomes a branch; scenes are sub-branches; beats are
    sequential method calls.  Returns the pretty-printed session type.
    """
    if not screenplay.acts:
        return pretty(End())

    if len(screenplay.acts) == 1:
        st = _act_to_session_type(screenplay.acts[0])
        return pretty(st)

    choices: list[tuple[str, SessionType]] = []
    for act in screenplay.acts:
        choices.append((act.name, _act_to_session_type(act)))

    st = Branch(tuple(choices))
    return pretty(st)


def analyze_structure(screenplay: Screenplay) -> StructureAnalysis:
    """Analyze the structure of a screenplay via its session type.

    Builds the state space and checks lattice properties.
    Returns a StructureAnalysis with metrics.
    """
    type_str = encode_screenplay(screenplay)
    st = parse(type_str)
    ss = build_statespace(st)
    lr = check_lattice(ss)

    num_scenes = sum(len(act.scenes) for act in screenplay.acts)
    all_beats: list[Beat] = []
    for act in screenplay.acts:
        for scene in act.scenes:
            all_beats.extend(scene.beats)

    emotional_arc = tuple(b.emotional_value for b in all_beats)

    # Structural balance: ratio of smallest act (by beats) to largest
    act_sizes = []
    for act in screenplay.acts:
        size = sum(len(s.beats) for s in act.scenes)
        act_sizes.append(size)
    if act_sizes and max(act_sizes) > 0:
        balance = min(act_sizes) / max(act_sizes)
    else:
        balance = 1.0

    return StructureAnalysis(
        title=screenplay.title,
        session_type_str=type_str,
        num_acts=len(screenplay.acts),
        num_scenes=num_scenes,
        num_beats=len(all_beats),
        state_count=len(ss.states),
        transition_count=len(ss.transitions),
        is_lattice=lr.is_lattice,
        emotional_arc=emotional_arc,
        structural_balance=balance,
    )


def compare_screenplays(sp_a: Screenplay, sp_b: Screenplay) -> ComparisonResult:
    """Compare two screenplays via their session type state spaces.

    Measures structural similarity based on shared beats and
    state space characteristics.
    """
    type_a = encode_screenplay(sp_a)
    type_b = encode_screenplay(sp_b)

    st_a = parse(type_a)
    st_b = parse(type_b)
    ss_a = build_statespace(st_a)
    ss_b = build_statespace(st_b)
    lr_a = check_lattice(ss_a)
    lr_b = check_lattice(ss_b)

    # Collect beat names
    beats_a: set[str] = set()
    beats_b: set[str] = set()
    for act in sp_a.acts:
        for scene in act.scenes:
            for beat in scene.beats:
                beats_a.add(beat.name)
    for act in sp_b.acts:
        for scene in act.scenes:
            for beat in scene.beats:
                beats_b.add(beat.name)

    shared = beats_a & beats_b
    total = beats_a | beats_b
    similarity = len(shared) / len(total) if total else 1.0

    return ComparisonResult(
        title_a=sp_a.title,
        title_b=sp_b.title,
        structural_similarity=similarity,
        shared_beat_count=len(shared),
        type_a_str=type_a,
        type_b_str=type_b,
        both_lattices=lr_a.is_lattice and lr_b.is_lattice,
    )


# ---------------------------------------------------------------------------
# Character arcs
# ---------------------------------------------------------------------------

_ARC_TEMPLATES: dict[str, tuple[str, ...]] = {
    "positive": ("ordinary_world", "call_to_adventure", "refusal",
                 "mentor", "crossing_threshold", "tests",
                 "ordeal", "reward", "resurrection", "return"),
    "negative": ("status_quo", "temptation", "descent",
                 "corruption", "crisis", "fall"),
    "flat": ("conviction", "challenge", "reaffirmation"),
    "circular": ("beginning", "departure", "journey", "return_home"),
}


def encode_character_arc(name: str, arc_type: str = "positive") -> CharacterArc:
    """Encode a character arc as a session type.

    Args:
        name: Character name.
        arc_type: One of 'positive', 'negative', 'flat', 'circular'.

    Returns:
        CharacterArc with session type string.
    """
    if arc_type not in _ARC_TEMPLATES:
        raise ValueError(f"Unknown arc type: {arc_type}. "
                         f"Choose from {list(_ARC_TEMPLATES.keys())}")

    beats = _ARC_TEMPLATES[arc_type]
    beat_objs = tuple(Beat(name=b) for b in beats)
    st = _beats_to_session_type(beat_objs)
    return CharacterArc(
        character_name=name,
        arc_type=arc_type,
        session_type_str=pretty(st),
        beats=beats,
    )


# ---------------------------------------------------------------------------
# Beat sheet (Save the Cat)
# ---------------------------------------------------------------------------

_SAVE_THE_CAT_BEATS = (
    "opening_image", "theme_stated", "setup", "catalyst",
    "debate", "break_into_two", "b_story", "fun_and_games",
    "midpoint", "bad_guys_close_in", "all_is_lost",
    "dark_night_of_the_soul", "break_into_three",
    "finale", "final_image",
)


def beat_sheet() -> BeatSheetResult:
    """Generate the Save the Cat beat sheet as a session type.

    The 15-beat structure maps to a linear chain of branches.
    """
    beat_objs = tuple(Beat(name=b) for b in _SAVE_THE_CAT_BEATS)
    st = _beats_to_session_type(beat_objs)
    type_str = pretty(st)
    ss = build_statespace(parse(type_str))
    lr = check_lattice(ss)

    return BeatSheetResult(
        session_type_str=type_str,
        beats=_SAVE_THE_CAT_BEATS,
        state_count=len(ss.states),
        is_lattice=lr.is_lattice,
    )


# ---------------------------------------------------------------------------
# Dialogue protocol
# ---------------------------------------------------------------------------

def dialogue_protocol(
    character_a: str,
    character_b: str,
    exchanges: int = 3,
) -> str:
    """Build a dialogue protocol between two characters.

    Models dialogue as alternating branch/select between two characters.
    Returns the session type string from character_a's perspective.
    """
    if exchanges <= 0:
        return pretty(End())

    current: SessionType = End()
    for i in range(exchanges - 1, -1, -1):
        speak_label = f"{character_a}_speaks_{i}"
        listen_label = f"{character_b}_speaks_{i}"
        # Character A selects what to say, then branches on B's response
        inner = Branch(((listen_label, current),))
        current = Select(((speak_label, inner),))

    return pretty(current)


# ---------------------------------------------------------------------------
# Three-act template
# ---------------------------------------------------------------------------

def three_act_template(title: str = "Untitled") -> Screenplay:
    """Create a standard three-act screenplay template.

    Act 1: Setup (25%), Act 2: Confrontation (50%), Act 3: Resolution (25%).
    """
    act1 = Act(
        name="act1",
        scenes=(
            Scene("opening", (Beat("hook", emotional_value=0.3),
                              Beat("setup", emotional_value=0.0))),
            Scene("inciting_incident", (Beat("catalyst", emotional_value=0.5),
                                        Beat("debate", emotional_value=-0.2))),
        ),
        act_number=1,
    )
    act2 = Act(
        name="act2",
        scenes=(
            Scene("rising_action", (Beat("obstacles", emotional_value=-0.3),
                                    Beat("midpoint", emotional_value=0.4))),
            Scene("crisis", (Beat("reversal", emotional_value=-0.6),
                             Beat("dark_moment", emotional_value=-0.8))),
        ),
        act_number=2,
    )
    act3 = Act(
        name="act3",
        scenes=(
            Scene("climax", (Beat("confrontation", emotional_value=0.7),
                             Beat("resolution", emotional_value=0.9))),
        ),
        act_number=3,
    )
    return Screenplay(title=title, acts=(act1, act2, act3))
