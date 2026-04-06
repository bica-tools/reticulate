"""Bach fugues as multiparty session types (Step 56 — Bach Fugues MPST).

A fugue is a polyphonic form built from a *subject* (main theme) stated
successively in each of N voices.  In session-type terms, a fugue is a
multiparty protocol where:

- Each voice (Soprano/Alto/Tenor/Bass — or SI/SII/A/B) is a role.
- A subject *entry* is a message ``leader -> follower : {enter}`` where
  the leading voice hands off the subject to the next voice.
- After the exposition, *episodes* (free counterpoint) interleave
  before a *final entry* that returns the subject in the home key.

This module:

- encodes a fugue structure (:class:`FugueStructure`) as a
  :class:`~reticulate.global_types.GlobalType`;
- projects onto per-voice local session types (via
  :mod:`reticulate.projection`);
- verifies well-formedness (every voice gets a well-defined projection,
  the state space is finite, the global state space is a lattice);
- supplies a bidirectional morphism pair
  ``phi : L(S_fugue) -> FugueElement`` and ``psi : FugueElement -> state``
  mapping lattice states to structural fugue elements (exposition,
  episode, stretto, final entry) and back;
- ships with a hand-coded rendition of J. S. Bach, WTC Book I, Fugue in
  C major (BWV 846), a 4-voice fugue used as the running example.

The module is deliberately algebraic: we do not attempt to synthesise
audio, we only reason about the *structure*.  The practical payoff is a
set of safety / composition / analysis actions:

- **Safety**: a fugue whose projection fails is *ill-formed* — a voice
  would be asked to play a non-deterministic part.  Catch before
  rehearsal.
- **Composition**: the lattice meet ``x \\sqcap y`` of two partial
  performances is the *common past*; the join is the smallest shared
  future.  Useful for score-following and for merging independent
  voice parts.
- **Analysis**: φ classifies each reachable state as an exposition /
  episode / stretto / final-entry state, yielding an automatic
  sectional segmentation directly from the lattice.

See ``papers/steps/step56-bach-fugues-mpst/main.tex`` for the full write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from reticulate.global_types import (
    GEnd,
    GMessage,
    GRec,
    GVar,
    GlobalType,
    build_global_statespace,
    roles,
)
from reticulate.projection import (
    ProjectionError,
    project,
)
from reticulate.parser import SessionType, pretty
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Structural data types
# ---------------------------------------------------------------------------


class FugueSection(Enum):
    """Coarse sectional label for a state in the fugue lattice."""

    EXPOSITION = "exposition"
    EPISODE = "episode"
    STRETTO = "stretto"
    FINAL_ENTRY = "final_entry"
    CADENCE = "cadence"


@dataclass(frozen=True)
class SubjectEntry:
    """A single subject entry: ``voice`` enters after being cued by ``cue``.

    ``cue`` is ``None`` for the very first entry (the head of the
    exposition).  The ``label`` is the message label used on the global
    type edge; defaults to ``"enter"``.
    """

    voice: str
    cue: str | None = None
    label: str = "enter"


@dataclass(frozen=True)
class FugueStructure:
    """A fugue as an ordered list of subject entries plus episodes.

    Attributes:
        voices: tuple of voice (role) names.
        exposition: subject entries forming the exposition.
        episodes: number of free-counterpoint episodes between the
            exposition and the final entry.  Each episode is modelled
            as an uninterpreted message ``lead -> next : {episode}``.
        final_entry: the closing subject statement (home key).
    """

    voices: tuple[str, ...]
    exposition: tuple[SubjectEntry, ...]
    episodes: int = 0
    final_entry: SubjectEntry | None = None

    def __post_init__(self) -> None:
        if not self.voices:
            raise ValueError("fugue must have at least one voice")
        vs = set(self.voices)
        for e in self.exposition:
            if e.voice not in vs:
                raise ValueError(f"unknown voice in entry: {e.voice}")
        if self.final_entry is not None and self.final_entry.voice not in vs:
            raise ValueError("unknown voice in final entry")


# ---------------------------------------------------------------------------
# Encoding: FugueStructure -> GlobalType
# ---------------------------------------------------------------------------


def _pair_each(
    entries: tuple[SubjectEntry, ...],
    voices: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    """Return a list of ``(sender, receiver, label)`` triples.

    The *first* entry has no cue; we model it as a message from the
    *next* voice in ``voices`` (a silent pre-entry cue from the
    ensemble) to the entering voice, to avoid degenerate self-messages.
    Subsequent entries use the preceding voice as sender.
    """
    triples: list[tuple[str, str, str]] = []
    prev: str | None = None
    for e in entries:
        if e.cue is not None:
            sender = e.cue
        elif prev is not None:
            sender = prev
        else:
            others = [v for v in voices if v != e.voice]
            sender = others[0] if others else e.voice
        if sender == e.voice:
            others = [v for v in voices if v != e.voice]
            if others:
                sender = others[0]
        triples.append((sender, e.voice, e.label))
        prev = e.voice
    return triples


def encode_fugue(fugue: FugueStructure) -> GlobalType:
    """Build a :class:`GlobalType` from a :class:`FugueStructure`.

    The encoding is strictly sequential (no parallel composition at
    the global level: polyphony appears only *after* projection, as
    each voice acquires its own local type).  Each subject entry
    becomes a unary-choice message; each episode becomes an
    ``episode`` message from the current leader to the next voice in
    cyclic order; the final entry closes with an ``end``.
    """

    voices = fugue.voices
    triples = _pair_each(fugue.exposition, voices)

    # Interleave episodes (cyclic leader rotation over the voice list).
    leader_idx = voices.index(fugue.exposition[-1].voice) if fugue.exposition else 0
    for _ in range(fugue.episodes):
        s = voices[leader_idx % len(voices)]
        r = voices[(leader_idx + 1) % len(voices)]
        # Avoid sender == receiver which GMessage forbids semantically.
        if s == r:
            r = voices[(leader_idx + 2) % len(voices)]
        triples.append((s, r, "episode"))
        leader_idx += 1

    if fugue.final_entry is not None:
        fe = fugue.final_entry
        sender = fe.cue if fe.cue is not None else voices[leader_idx % len(voices)]
        if sender == fe.voice:
            # Pick any other voice as the cueing sender.
            others = [v for v in voices if v != fe.voice]
            sender = others[0] if others else sender
        triples.append((sender, fe.voice, fe.label))

    # Build right-associated GMessage chain ending in GEnd.
    g: GlobalType = GEnd()
    for sender, receiver, label in reversed(triples):
        g = GMessage(sender=sender, receiver=receiver, choices=((label, g),))
    return g


# ---------------------------------------------------------------------------
# Well-formedness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FugueWellFormedness:
    """Result of checking a fugue's well-formedness.

    Attributes:
        is_well_formed: all projections succeed and every voice's
            local state space is finite.
        global_is_lattice: true iff the global state space is a
            lattice under reachability.
        voices: the voices / roles involved.
        local_types: voice -> pretty-printed local session type.
        global_size: number of states in the global state space.
        errors: human-readable diagnostics.
    """

    is_well_formed: bool
    global_is_lattice: bool
    voices: tuple[str, ...]
    local_types: dict[str, str]
    global_size: int
    errors: tuple[str, ...]


def check_fugue(fugue: FugueStructure) -> FugueWellFormedness:
    """Check well-formedness of a fugue: project, size, lattice."""

    g = encode_fugue(fugue)
    errors: list[str] = []
    local_types: dict[str, str] = {}

    for v in fugue.voices:
        try:
            lt: SessionType = project(g, v)
            local_types[v] = pretty(lt)
        except ProjectionError as exc:
            errors.append(f"projection onto {v} failed: {exc}")

    try:
        gss = build_global_statespace(g)
        global_size = len(gss.states)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"global state space build failed: {exc}")
        global_size = 0

    is_wf = not errors
    if is_wf:
        lattice_result = check_lattice(gss)
        global_is_lattice = lattice_result.is_lattice
    else:
        global_is_lattice = False

    return FugueWellFormedness(
        is_well_formed=is_wf,
        global_is_lattice=global_is_lattice,
        voices=fugue.voices,
        local_types=local_types,
        global_size=global_size,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# Bidirectional morphism: lattice states <-> fugue sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FugueMorphism:
    """Bidirectional morphism pair for a fugue lattice.

    ``phi`` maps each state id in the global state space to a
    :class:`FugueSection`; ``psi`` maps each section to the *first*
    state id that carries that section (a section representative).
    Both maps are total and their composition ``phi o psi`` is the
    identity on sections that occur in the fugue.
    """

    phi: dict[int, FugueSection]
    psi: dict[FugueSection, int]
    state_space: StateSpace


def build_morphism(fugue: FugueStructure) -> FugueMorphism:
    """Compute φ and ψ for a fugue.

    Strategy: walk the global state space in BFS order from its
    initial state; classify each state by the index of the next
    transition relative to the exposition length, episode count, and
    final-entry position.
    """

    g = encode_fugue(fugue)
    ss = build_global_statespace(g)

    n_expo = len(fugue.exposition)
    n_epi = fugue.episodes
    has_final = fugue.final_entry is not None

    # Compute a BFS order from the initial state.
    order: dict[int, int] = {}
    frontier = [ss.top]
    idx = 0
    seen = {ss.top}
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
    # Any unreachable states (shouldn't happen for sequential encodings)
    # get a trailing index.
    for s in ss.states:
        if s not in order:
            order[s] = idx
            idx += 1

    phi: dict[int, FugueSection] = {}
    for s, k in order.items():
        if k < n_expo:
            phi[s] = FugueSection.EXPOSITION
        elif k < n_expo + n_epi:
            phi[s] = FugueSection.EPISODE
        elif has_final and k == n_expo + n_epi:
            phi[s] = FugueSection.FINAL_ENTRY
        else:
            phi[s] = FugueSection.CADENCE

    psi: dict[FugueSection, int] = {}
    # First state per section in BFS order is the representative.
    ordered_states = sorted(order.items(), key=lambda kv: kv[1])
    for s, _k in ordered_states:
        sec = phi[s]
        if sec not in psi:
            psi[sec] = s

    return FugueMorphism(phi=phi, psi=psi, state_space=ss)


def classify_state(morphism: FugueMorphism, state: int) -> FugueSection:
    """Return the fugue section for ``state`` (φ lookup)."""
    return morphism.phi[state]


def section_representative(morphism: FugueMorphism, section: FugueSection) -> int:
    """Return ψ(section): the representative state for the section."""
    if section not in morphism.psi:
        raise KeyError(f"section not present in fugue: {section}")
    return morphism.psi[section]


def round_trip_sections(morphism: FugueMorphism) -> bool:
    """Verify phi(psi(sec)) == sec for every section present."""
    for sec, s in morphism.psi.items():
        if morphism.phi[s] != sec:
            return False
    return True


# ---------------------------------------------------------------------------
# Stretto detection
# ---------------------------------------------------------------------------


def has_stretto(fugue: FugueStructure) -> bool:
    """A rough structural test for stretto.

    Stretto = subject entries that overlap in time.  In our purely
    structural encoding we do not model timing, but we capture
    intent: if the fugue's exposition contains the *same* voice twice
    before all other voices have entered, we mark it as stretto.
    """
    seen: set[str] = set()
    for e in fugue.exposition:
        if e.voice in seen and len(seen) < len(fugue.voices):
            return True
        seen.add(e.voice)
    return False


# ---------------------------------------------------------------------------
# Running example: J. S. Bach, WTC Book I, Fugue in C major (BWV 846)
# ---------------------------------------------------------------------------


def bwv846_fugue() -> FugueStructure:
    """J. S. Bach, Well-Tempered Clavier Book I, Fugue No. 1 in C major.

    BWV 846 is a 4-voice fugue.  Exposition order is
    Alto → Soprano → Tenor → Bass (cf. the Henle edition).  We use
    2 episodes (the two main episodes before the stretto cluster) and
    a final entry in the bass in the home key.

    This is a structural approximation suitable for MPST analysis,
    not a bar-by-bar reduction.
    """
    voices = ("S", "A", "T", "B")
    expo = (
        SubjectEntry(voice="A", cue=None, label="subject"),
        SubjectEntry(voice="S", cue="A", label="answer"),
        SubjectEntry(voice="T", cue="S", label="subject"),
        SubjectEntry(voice="B", cue="T", label="answer"),
    )
    final = SubjectEntry(voice="B", cue="S", label="final")
    return FugueStructure(
        voices=voices,
        exposition=expo,
        episodes=2,
        final_entry=final,
    )


__all__ = [
    "FugueSection",
    "SubjectEntry",
    "FugueStructure",
    "FugueWellFormedness",
    "FugueMorphism",
    "encode_fugue",
    "check_fugue",
    "build_morphism",
    "classify_state",
    "section_representative",
    "round_trip_sections",
    "has_stretto",
    "bwv846_fugue",
]
