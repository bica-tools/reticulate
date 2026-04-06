"""Theatrical stage blocking as multiparty session types (Step 58).

*Blocking* is the craft of deciding where actors stand and move on stage,
when they enter and exit, and on what cues they deliver lines or
gestures.  A rehearsal script annotated with blocking is, operationally,
a concurrent protocol between the actors and the stage manager.  This
module models such a protocol as a :class:`GlobalType` in which:

- **Roles** are the actors (characters), plus an implicit ``SM`` role
  for the stage manager who issues go-cues.
- **Messages** ``SM -> Actor : {cue_X}`` carry lighting / sound / entry
  cues; messages ``Actor -> SM : {at_PosY}`` report arrival at a
  blocking position.
- **Sequencing** is the rehearsal order: enter, cross, speak, exit.

The running example is the opening of **Act I, Scene i of
Shakespeare's \"Macbeth\"** (the three witches on the heath), which is
short enough to be fully enumerated yet has genuine concurrency
(three witches speak in rotation, with thunder and a final exit
together).

Practical payoffs (cross-domain must be practical):

- **Rehearsal planning**: the lattice gives a canonical order for
  walk-throughs; the top element is the curtain, the bottom is the
  closing tableau.
- **Safety**: an actor who tries to enter when another is already in
  their blocking position violates the session type.  We detect this
  before the dress rehearsal — no collisions, no missed cues.
- **Cue verification**: each lattice state knows which cues have fired.
  Light board operators can cross-check their cue sheet against the
  lattice's Hasse diagram for completeness.
- **Understudy onboarding**: projection yields the local type for one
  role, which is exactly the script-plus-blocking an understudy needs.

The bidirectional morphism pair

.. math::

    \\varphi : \\mathcal{L}(S_{\\text{scene}}) \\to \\mathrm{BlockingPlan}
    \\qquad
    \\psi   : \\mathrm{BlockingPlan} \\to \\mathcal{L}(S_{\\text{scene}})

maps each lattice state to a *blocking snapshot* (who is on stage, at
what position, which cues have fired) and back.  We classify the pair
as an order-preserving **section--retract**: ``psi`` is a section of
``phi`` whose image is exactly the reachable blocking plans.

See ``papers/steps/step58-stage-blocking/main.tex`` for the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from reticulate.global_types import (
    GEnd,
    GMessage,
    GlobalType,
    build_global_statespace,
    roles as gt_roles,
)
from reticulate.projection import ProjectionError, project
from reticulate.parser import SessionType, pretty
from reticulate.statespace import StateSpace
from reticulate.lattice import check_lattice


STAGE_MANAGER = "SM"


# ---------------------------------------------------------------------------
# Data types: cues, positions, beats
# ---------------------------------------------------------------------------


class CueKind(Enum):
    """Kinds of cues issued by the stage manager."""

    ENTRANCE = "entrance"
    LIGHT = "light"
    SOUND = "sound"
    LINE = "line"
    EXIT = "exit"


@dataclass(frozen=True)
class Cue:
    """A single cue issued by ``SM`` to an actor."""

    kind: CueKind
    actor: str
    label: str  # e.g. ``thunder``, ``enterUL``, ``speak1``


@dataclass(frozen=True)
class Position:
    """A blocking position on stage (stage-left / centre / etc.)."""

    code: str  # ``DSL``, ``DSC``, ``DSR``, ``USL``, ``USC``, ``USR``, ``OFF``

    def __post_init__(self) -> None:
        allowed = {"DSL", "DSC", "DSR", "USL", "USC", "USR", "CS", "OFF"}
        if self.code not in allowed:
            raise ValueError(f"unknown stage position: {self.code}")


@dataclass(frozen=True)
class Beat:
    """A single beat of the scene.

    A beat is either a cue from ``SM`` to an actor, or a position
    report from an actor back to ``SM``.  Beats are totally ordered
    within a :class:`Scene`.
    """

    sender: str
    receiver: str
    label: str
    cue_kind: CueKind | None = None
    position: Position | None = None


@dataclass(frozen=True)
class Scene:
    """A theatrical scene specified as a totally ordered list of beats.

    Attributes:
        title: human-readable scene title.
        actors: tuple of actor (role) names.  The stage manager
            ``SM`` is implicit and always present.
        beats: the beats in performance order.
    """

    title: str
    actors: tuple[str, ...]
    beats: tuple[Beat, ...]

    def __post_init__(self) -> None:
        if not self.actors:
            raise ValueError("scene must have at least one actor")
        if STAGE_MANAGER in self.actors:
            raise ValueError(f"actor name {STAGE_MANAGER!r} is reserved")
        all_roles = set(self.actors) | {STAGE_MANAGER}
        for b in self.beats:
            if b.sender not in all_roles:
                raise ValueError(f"unknown sender in beat: {b.sender}")
            if b.receiver not in all_roles:
                raise ValueError(f"unknown receiver in beat: {b.receiver}")
            if b.sender == b.receiver:
                raise ValueError(f"self-loop beat forbidden: {b.sender}")


# ---------------------------------------------------------------------------
# Encoding: Scene -> GlobalType
# ---------------------------------------------------------------------------


def encode_scene(scene: Scene) -> GlobalType:
    """Build a :class:`GlobalType` from a :class:`Scene`.

    The encoding is sequential: beats become unary-choice
    ``GMessage`` nodes right-associated and terminated by ``GEnd``.
    Concurrency between actors is recovered at *projection* time —
    each actor's local type sees only the beats where they appear,
    exactly as an actor's cue sheet is a private view of the scene.
    """
    if not scene.beats:
        return GEnd()
    g: GlobalType = GEnd()
    for b in reversed(scene.beats):
        g = GMessage(sender=b.sender, receiver=b.receiver, choices=((b.label, g),))
    return g


# ---------------------------------------------------------------------------
# Well-formedness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneWellFormedness:
    """Well-formedness report for a scene."""

    is_well_formed: bool
    global_is_lattice: bool
    roles: tuple[str, ...]
    local_types: dict[str, str]
    global_size: int
    errors: tuple[str, ...]


def check_scene(scene: Scene) -> SceneWellFormedness:
    """Project the scene onto every role and check lattice property."""
    g = encode_scene(scene)
    errors: list[str] = []
    locals_: dict[str, str] = {}
    all_roles = tuple(sorted(set(scene.actors) | {STAGE_MANAGER}))

    for r in all_roles:
        try:
            lt: SessionType = project(g, r)
            locals_[r] = pretty(lt)
        except ProjectionError as exc:
            errors.append(f"projection onto {r} failed: {exc}")

    try:
        gss = build_global_statespace(g)
        size = len(gss.states)
    except Exception as exc:  # pragma: no cover
        errors.append(f"state-space build failed: {exc}")
        size = 0

    is_wf = not errors
    if is_wf:
        lat = check_lattice(gss)
        is_lat = lat.is_lattice
    else:
        is_lat = False

    return SceneWellFormedness(
        is_well_formed=is_wf,
        global_is_lattice=is_lat,
        roles=all_roles,
        local_types=locals_,
        global_size=size,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# Blocking plan: snapshot of who-is-where-with-what-cues
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockingSnapshot:
    """Snapshot of the stage at a single lattice state.

    ``positions`` maps actor -> current :class:`Position` (``OFF``
    means off-stage).  ``fired_cues`` is the set of cue labels that
    have been issued up to and including this state.
    """

    positions: tuple[tuple[str, str], ...]  # (actor, position-code)
    fired_cues: tuple[str, ...]

    def actors_on_stage(self) -> tuple[str, ...]:
        return tuple(a for (a, p) in self.positions if p != "OFF")


@dataclass(frozen=True)
class BlockingPlan:
    """Full blocking plan: lattice-state-id -> snapshot."""

    snapshots: dict[int, BlockingSnapshot]
    scene: Scene


# ---------------------------------------------------------------------------
# Bidirectional morphism
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockingMorphism:
    """Bidirectional morphism between the scene lattice and blocking plans.

    ``phi`` maps each state id to a :class:`BlockingSnapshot`.
    ``psi`` maps each snapshot (hashable) back to the *first* state
    id that realises it in BFS order.  The pair is an
    order-preserving section--retract: ``phi(psi(sn)) == sn`` for
    every reachable snapshot ``sn``.
    """

    phi: dict[int, BlockingSnapshot]
    psi: dict[BlockingSnapshot, int]
    state_space: StateSpace
    plan: BlockingPlan


def _bfs_order(ss: StateSpace) -> dict[int, int]:
    order: dict[int, int] = {}
    frontier = [ss.top]
    seen = {ss.top}
    idx = 0
    while frontier:
        nxt: list[int] = []
        for s in frontier:
            order[s] = idx
            idx += 1
            for (src, _lbl, tgt) in ss.transitions:
                if src == s and tgt not in seen:
                    seen.add(tgt)
                    nxt.append(tgt)
        frontier = nxt
    for s in ss.states:
        if s not in order:
            order[s] = idx
            idx += 1
    return order


def build_morphism(scene: Scene) -> BlockingMorphism:
    """Construct ``phi`` and ``psi`` for a scene.

    Walks the global state space in BFS order.  After the ``k``-th
    beat, the snapshot is obtained by folding the first ``k`` beats
    onto the initial (all ``OFF``) stage picture and accumulating
    fired cues.
    """
    g = encode_scene(scene)
    ss = build_global_statespace(g)
    order = _bfs_order(ss)

    # Initial positions: every actor OFF.
    init_positions: dict[str, str] = {a: "OFF" for a in scene.actors}

    # Precompute per-index snapshot by folding beats in order.
    snapshots_by_index: list[BlockingSnapshot] = []
    positions = dict(init_positions)
    fired: list[str] = []
    snapshots_by_index.append(
        BlockingSnapshot(
            positions=tuple(sorted(positions.items())),
            fired_cues=(),
        )
    )
    for b in scene.beats:
        if b.position is not None and b.sender in positions:
            positions[b.sender] = b.position.code
        if b.position is not None and b.receiver in positions:
            positions[b.receiver] = b.position.code
        if b.cue_kind == CueKind.EXIT and b.receiver in positions:
            positions[b.receiver] = "OFF"
        fired.append(b.label)
        snapshots_by_index.append(
            BlockingSnapshot(
                positions=tuple(sorted(positions.items())),
                fired_cues=tuple(fired),
            )
        )

    phi: dict[int, BlockingSnapshot] = {}
    for state, k in order.items():
        idx = min(k, len(snapshots_by_index) - 1)
        phi[state] = snapshots_by_index[idx]

    psi: dict[BlockingSnapshot, int] = {}
    for state, k in sorted(order.items(), key=lambda kv: kv[1]):
        sn = phi[state]
        if sn not in psi:
            psi[sn] = state

    plan = BlockingPlan(snapshots=dict(phi), scene=scene)
    return BlockingMorphism(phi=phi, psi=psi, state_space=ss, plan=plan)


def round_trip_blocking(m: BlockingMorphism) -> bool:
    """Verify phi(psi(sn)) == sn for all snapshots in the image of psi."""
    for sn, s in m.psi.items():
        if m.phi[s] != sn:
            return False
    return True


def classify_state(m: BlockingMorphism, state: int) -> BlockingSnapshot:
    return m.phi[state]


# ---------------------------------------------------------------------------
# Safety / cue verification helpers
# ---------------------------------------------------------------------------


def detect_collisions(scene: Scene) -> tuple[str, ...]:
    """Return a tuple of human-readable collision diagnostics.

    A *collision* is when two distinct on-stage actors are scripted
    into the same non-``OFF`` stage position at the same beat.
    """
    positions: dict[str, str] = {a: "OFF" for a in scene.actors}
    errors: list[str] = []
    for i, b in enumerate(scene.beats):
        if b.position is not None and b.receiver in positions:
            target = b.position.code
            if target != "OFF":
                for other, p in positions.items():
                    if other != b.receiver and p == target:
                        errors.append(
                            f"beat {i} ({b.label}): {b.receiver} moving to "
                            f"{target} but {other} is already there"
                        )
            positions[b.receiver] = target
        if b.cue_kind == CueKind.EXIT and b.receiver in positions:
            positions[b.receiver] = "OFF"
    return tuple(errors)


def verify_cue_sheet(
    scene: Scene, expected_cues: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Compare scripted cues to an expected cue sheet.

    Returns ``(missing, extra)`` where ``missing`` is the set of
    cues the board operator expects but the script does not fire,
    and ``extra`` is the set of script cues not on the sheet.
    """
    scripted = [b.label for b in scene.beats if b.sender == STAGE_MANAGER]
    expected = list(expected_cues)
    missing = tuple(c for c in expected if c not in scripted)
    extra = tuple(c for c in scripted if c not in expected)
    return missing, extra


# ---------------------------------------------------------------------------
# Running example: Macbeth, Act I Scene i (the three witches)
# ---------------------------------------------------------------------------


def macbeth_witches_scene() -> Scene:
    """Opening scene of *Macbeth*: three witches on a heath.

    Structural approximation (not a line-by-line reduction):

    1. SM fires ``thunder`` cue to witch W1.
    2. SM cues W1 to enter at ``DSL``.
    3. SM cues W2 to enter at ``DSC``.
    4. SM cues W3 to enter at ``DSR``.
    5. W1 speaks (``when_shall_we_three``).
    6. W2 speaks (``when_hurlyburly``).
    7. W3 speaks (``ere_set_of_sun``).
    8. SM cues all three to exit together (``fair_is_foul``).
    """
    actors = ("W1", "W2", "W3")
    beats = (
        Beat(STAGE_MANAGER, "W1", "thunder", CueKind.SOUND),
        Beat(STAGE_MANAGER, "W1", "enter_DSL", CueKind.ENTRANCE, Position("DSL")),
        Beat(STAGE_MANAGER, "W2", "enter_DSC", CueKind.ENTRANCE, Position("DSC")),
        Beat(STAGE_MANAGER, "W3", "enter_DSR", CueKind.ENTRANCE, Position("DSR")),
        Beat("W1", STAGE_MANAGER, "when_shall_we_three", CueKind.LINE),
        Beat("W2", STAGE_MANAGER, "when_hurlyburly", CueKind.LINE),
        Beat("W3", STAGE_MANAGER, "ere_set_of_sun", CueKind.LINE),
        Beat(STAGE_MANAGER, "W1", "exit_W1", CueKind.EXIT),
        Beat(STAGE_MANAGER, "W2", "exit_W2", CueKind.EXIT),
        Beat(STAGE_MANAGER, "W3", "fair_is_foul", CueKind.EXIT),
    )
    return Scene(title="Macbeth I.i", actors=actors, beats=beats)


__all__ = [
    "STAGE_MANAGER",
    "CueKind",
    "Cue",
    "Position",
    "Beat",
    "Scene",
    "SceneWellFormedness",
    "BlockingSnapshot",
    "BlockingPlan",
    "BlockingMorphism",
    "encode_scene",
    "check_scene",
    "build_morphism",
    "round_trip_blocking",
    "classify_state",
    "detect_collisions",
    "verify_cue_sheet",
    "macbeth_witches_scene",
]
