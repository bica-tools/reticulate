"""Event structure morphisms for session types (Step 19).

An **ES morphism** h: ES_1 -> ES_2 is a partial function on events satisfying:
  (M1) Configuration preservation: for every configuration C of ES_1,
       h(C) = {h(e) : e in C, h(e) defined} is a configuration of ES_2.
  (M2) Local injectivity: for every configuration C of ES_1,
       h restricted to C is injective.

Equivalently (Winskel): h preserves causality and reflects conflict.

Morphism classes:
  - **Rigid**: total + globally injective (embedding).
  - **Folding**: total + surjective (abstraction/projection).
  - **Isomorphism**: rigid + folding (bijection preserving & reflecting all structure).
  - **Label-preserving**: h(e).label == e.label for all mapped events.

Key functions:
  - ``find_es_morphism(es1, es2)`` -- find a morphism if one exists
  - ``find_es_embedding(es1, es2)`` -- find a rigid morphism (embedding)
  - ``check_es_isomorphism(es1, es2)`` -- check if isomorphic
  - ``is_label_preserving(f, es1, es2)`` -- check label preservation
  - ``is_rigid(f, es1, es2)`` -- check rigidity (total + injective)
  - ``classify_es_morphism(f, es1, es2)`` -- classify morphism type
  - ``analyze_es_morphisms(ss1, ss2)`` -- full comparison via state spaces
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.event_structures import (
    Event,
    EventStructure,
    Configuration,
    build_event_structure,
    configurations,
)

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ESMorphism:
    """A morphism between event structures.

    Attributes:
        source: Source event structure.
        target: Target event structure.
        mapping: Partial function from source events to target events.
                 Events not in the dict are unmapped (h undefined).
        kind: Classification string: "isomorphism", "rigid", "folding",
              "morphism", or "none" if invalid.
    """
    source: EventStructure
    target: EventStructure
    mapping: dict[Event, Event]
    kind: str


@dataclass(frozen=True)
class ESMorphismAnalysis:
    """Full morphism analysis between two event structures.

    Attributes:
        es1: First event structure.
        es2: Second event structure.
        morphism_1_to_2: Morphism from es1 to es2 (or None).
        morphism_2_to_1: Morphism from es2 to es1 (or None).
        are_isomorphic: Whether ES1 and ES2 are isomorphic.
        label_preserving_1_to_2: Whether the 1->2 morphism preserves labels.
        label_preserving_2_to_1: Whether the 2->1 morphism preserves labels.
    """
    es1: EventStructure
    es2: EventStructure
    morphism_1_to_2: ESMorphism | None
    morphism_2_to_1: ESMorphism | None
    are_isomorphic: bool
    label_preserving_1_to_2: bool
    label_preserving_2_to_1: bool


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _causality_set(es: EventStructure) -> set[tuple[Event, Event]]:
    """Return causality as a mutable set for fast lookup."""
    return set(es.causality)


def _conflict_set(es: EventStructure) -> set[tuple[Event, Event]]:
    """Return conflict as a mutable set for fast lookup."""
    return set(es.conflict)


def _is_valid_morphism(
    f: dict[Event, Event],
    es1: EventStructure,
    es2: EventStructure,
) -> bool:
    """Check if f satisfies (M1) config preservation and (M2) local injectivity.

    Equivalent formulation: preserves causality and reflects conflict.
    We check both the direct axioms for efficiency on small structures.
    """
    causal1 = _causality_set(es1)
    causal2 = _causality_set(es2)
    conflict1 = _conflict_set(es1)
    conflict2 = _conflict_set(es2)

    # Check causality preservation:
    # If e1 <=_1 e2 and both f(e1), f(e2) defined, then f(e1) <=_2 f(e2)
    for e1, e2 in causal1:
        if e1 in f and e2 in f:
            if (f[e1], f[e2]) not in causal2:
                return False

    # Check conflict reflection:
    # If f(e1) #_2 f(e2), then e1 #_1 e2
    # Equivalently: if e1 NOT #_1 e2, then f(e1) NOT #_2 f(e2)
    mapped_events = [e for e in es1.events if e in f]
    for i, e1 in enumerate(mapped_events):
        for e2 in mapped_events[i + 1:]:
            if (e1, e2) not in conflict1:
                if (f[e1], f[e2]) in conflict2:
                    return False

    # Check local injectivity: no two non-conflicting events map to same target
    for i, e1 in enumerate(mapped_events):
        for e2 in mapped_events[i + 1:]:
            if f[e1] == f[e2] and (e1, e2) not in conflict1:
                return False

    # Verify mapped events land in es2
    for e in f.values():
        if e not in es2.events:
            return False

    return True


def _is_valid_morphism_via_configs(
    f: dict[Event, Event],
    es1: EventStructure,
    es2: EventStructure,
) -> bool:
    """Check morphism validity via configuration axioms (M1) + (M2).

    Slower but serves as an independent check.
    """
    configs1 = configurations(es1)
    conflict2 = _conflict_set(es2)
    causal2 = _causality_set(es2)

    for config in configs1:
        # Compute image
        image_events: list[Event] = []
        for e in config.events:
            if e in f:
                image_events.append(f[e])

        # (M2) Local injectivity
        if len(image_events) != len(set(image_events)):
            return False

        image_set = frozenset(image_events)

        # (M1) Image must be a configuration of es2
        # Check conflict-free
        for i, a in enumerate(image_events):
            for b in image_events[i + 1:]:
                if (a, b) in conflict2:
                    return False

        # Check downward-closed
        for a in image_events:
            for e2, a2 in causal2:
                if a2 == a and e2 != a and e2 not in image_set:
                    # e2 causes a in es2, but e2 is not in the image
                    # This is OK only if e2 is not strictly below a
                    # (i.e., not a proper predecessor)
                    pass
            # Proper check: all predecessors of a in es2 must be in image
            for pred in es2.events:
                if pred != a and (pred, a) in causal2 and pred not in image_set:
                    return False

    return True


# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

def is_label_preserving(
    f: dict[Event, Event],
    es1: EventStructure,
    es2: EventStructure,
) -> bool:
    """Check if the morphism preserves event labels.

    For every e in dom(f): f(e).label == e.label.
    """
    for e, fe in f.items():
        if e.label != fe.label:
            return False
    return True


def is_rigid(
    f: dict[Event, Event],
    es1: EventStructure,
    es2: EventStructure,
) -> bool:
    """Check if the morphism is rigid (total + globally injective).

    - Total: every event in es1 is in dom(f).
    - Injective: no two events map to the same target.
    """
    # Total
    if set(f.keys()) != set(es1.events):
        return False
    # Injective
    image = list(f.values())
    return len(image) == len(set(image))


def is_folding(
    f: dict[Event, Event],
    es1: EventStructure,
    es2: EventStructure,
) -> bool:
    """Check if the morphism is a folding (total + surjective).

    - Total: every event in es1 is in dom(f).
    - Surjective: every event in es2 is in the image.
    """
    if set(f.keys()) != set(es1.events):
        return False
    image = set(f.values())
    return image == set(es2.events)


def is_isomorphism(
    f: dict[Event, Event],
    es1: EventStructure,
    es2: EventStructure,
) -> bool:
    """Check if the morphism is an ES isomorphism (rigid + folding = bijection).

    Also requires preserving AND reflecting both causality and conflict.
    """
    if not is_rigid(f, es1, es2):
        return False
    if not is_folding(f, es1, es2):
        return False

    causal1 = _causality_set(es1)
    causal2 = _causality_set(es2)
    conflict1 = _conflict_set(es1)
    conflict2 = _conflict_set(es2)

    # Causality reflection: f(e1) <=_2 f(e2) => e1 <=_1 e2
    inv = {v: k for k, v in f.items()}
    for fe1, fe2 in causal2:
        if fe1 in inv and fe2 in inv:
            if (inv[fe1], inv[fe2]) not in causal1:
                return False

    # Conflict preservation: e1 #_1 e2 => f(e1) #_2 f(e2)
    for e1, e2 in conflict1:
        if (f[e1], f[e2]) not in conflict2:
            return False

    return True


def preserves_causality(
    f: dict[Event, Event],
    es1: EventStructure,
    es2: EventStructure,
) -> bool:
    """Check if f preserves causality: e1 <=_1 e2 => f(e1) <=_2 f(e2)."""
    causal1 = _causality_set(es1)
    causal2 = _causality_set(es2)
    for e1, e2 in causal1:
        if e1 in f and e2 in f:
            if (f[e1], f[e2]) not in causal2:
                return False
    return True


def reflects_conflict(
    f: dict[Event, Event],
    es1: EventStructure,
    es2: EventStructure,
) -> bool:
    """Check if f reflects conflict: f(e1) #_2 f(e2) => e1 #_1 e2."""
    conflict1 = _conflict_set(es1)
    conflict2 = _conflict_set(es2)
    mapped = [e for e in es1.events if e in f]
    for i, e1 in enumerate(mapped):
        for e2 in mapped[i + 1:]:
            if (f[e1], f[e2]) in conflict2:
                if (e1, e2) not in conflict1:
                    return False
    return True


def reflects_causality(
    f: dict[Event, Event],
    es1: EventStructure,
    es2: EventStructure,
) -> bool:
    """Check if f reflects causality: f(e1) <=_2 f(e2) => e1 <=_1 e2.

    This is an additional property of rigid morphisms / isomorphisms.
    """
    causal1 = _causality_set(es1)
    causal2 = _causality_set(es2)
    inv: dict[Event, list[Event]] = {}
    for e, fe in f.items():
        inv.setdefault(fe, []).append(e)

    for fe1, fe2 in causal2:
        for e1 in inv.get(fe1, []):
            for e2 in inv.get(fe2, []):
                if (e1, e2) not in causal1:
                    return False
    return True


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_es_morphism(
    f: dict[Event, Event],
    es1: EventStructure,
    es2: EventStructure,
) -> str:
    """Classify an ES morphism.

    Returns one of: "isomorphism", "rigid", "folding", "morphism", "none".
    """
    if not _is_valid_morphism(f, es1, es2):
        return "none"
    if is_isomorphism(f, es1, es2):
        return "isomorphism"
    if is_rigid(f, es1, es2):
        return "rigid"
    if is_folding(f, es1, es2):
        return "folding"
    return "morphism"


# ---------------------------------------------------------------------------
# Search: find morphisms by backtracking
# ---------------------------------------------------------------------------

def find_es_morphism(
    es1: EventStructure,
    es2: EventStructure,
    *,
    label_preserving: bool = False,
    require_total: bool = False,
) -> dict[Event, Event] | None:
    """Find an ES morphism h: es1 -> es2 if one exists.

    Uses backtracking search with pruning.

    Args:
        es1: Source event structure.
        es2: Target event structure.
        label_preserving: If True, require h(e).label == e.label.
        require_total: If True, require h defined on all events.

    Returns:
        Mapping dict, or None if no morphism exists.
    """
    events1 = sorted(es1.events, key=lambda e: (e.source, e.label, e.target))
    events2 = sorted(es2.events, key=lambda e: (e.source, e.label, e.target))

    causal1 = _causality_set(es1)
    causal2 = _causality_set(es2)
    conflict1 = _conflict_set(es1)
    conflict2 = _conflict_set(es2)

    # For each event in es1, compute candidate targets in es2
    def candidates(e: Event) -> list[Event | None]:
        cands: list[Event | None] = []
        for e2 in events2:
            if label_preserving and e.label != e2.label:
                continue
            cands.append(e2)
        if not require_total:
            cands.append(None)  # unmapped
        return cands

    mapping: dict[Event, Event] = {}

    def is_consistent(e1: Event, e2: Event | None) -> bool:
        """Check if mapping e1 -> e2 is consistent with current partial mapping."""
        if e2 is None:
            return True

        # Check causality preservation for already-mapped pairs
        for prev_e1, prev_e2 in mapping.items():
            # If prev_e1 <= e1, then prev_e2 <= e2
            if (prev_e1, e1) in causal1 and (prev_e2, e2) not in causal2:
                return False
            # If e1 <= prev_e1, then e2 <= prev_e2
            if (e1, prev_e1) in causal1 and (e2, prev_e2) not in causal2:
                return False

            # Conflict reflection: if prev_e2 #_2 e2, then prev_e1 #_1 e1
            if (prev_e2, e2) in conflict2 and (prev_e1, e1) not in conflict1:
                return False

            # Local injectivity: if prev_e2 == e2 and NOT prev_e1 #_1 e1, fail
            if prev_e2 == e2 and (prev_e1, e1) not in conflict1:
                return False

        return True

    def backtrack(idx: int) -> bool:
        if idx == len(events1):
            return True
        e1 = events1[idx]
        for e2 in candidates(e1):
            if is_consistent(e1, e2):
                if e2 is not None:
                    mapping[e1] = e2
                if backtrack(idx + 1):
                    return True
                if e2 is not None:
                    del mapping[e1]
        return False

    if backtrack(0):
        return dict(mapping)
    return None


def find_es_embedding(
    es1: EventStructure,
    es2: EventStructure,
    *,
    label_preserving: bool = False,
) -> dict[Event, Event] | None:
    """Find a rigid (total + injective) ES morphism h: es1 -> es2.

    Returns:
        Mapping dict, or None if no embedding exists.
    """
    if len(es1.events) > len(es2.events):
        return None  # Can't inject into smaller set

    events1 = sorted(es1.events, key=lambda e: (e.source, e.label, e.target))
    events2 = sorted(es2.events, key=lambda e: (e.source, e.label, e.target))

    causal1 = _causality_set(es1)
    causal2 = _causality_set(es2)
    conflict1 = _conflict_set(es1)
    conflict2 = _conflict_set(es2)

    mapping: dict[Event, Event] = {}
    used: set[Event] = set()

    def candidates(e: Event) -> list[Event]:
        cands: list[Event] = []
        for e2 in events2:
            if e2 in used:
                continue
            if label_preserving and e.label != e2.label:
                continue
            cands.append(e2)
        return cands

    def is_consistent(e1: Event, e2: Event) -> bool:
        for prev_e1, prev_e2 in mapping.items():
            if (prev_e1, e1) in causal1 and (prev_e2, e2) not in causal2:
                return False
            if (e1, prev_e1) in causal1 and (e2, prev_e2) not in causal2:
                return False
            if (prev_e2, e2) in conflict2 and (prev_e1, e1) not in conflict1:
                return False
            # Injectivity already ensured by 'used' set
        return True

    def backtrack(idx: int) -> bool:
        if idx == len(events1):
            return True
        e1 = events1[idx]
        for e2 in candidates(e1):
            if is_consistent(e1, e2):
                mapping[e1] = e2
                used.add(e2)
                if backtrack(idx + 1):
                    return True
                del mapping[e1]
                used.discard(e2)
        return False

    if backtrack(0):
        return dict(mapping)
    return None


def check_es_isomorphism(
    es1: EventStructure,
    es2: EventStructure,
    *,
    label_preserving: bool = False,
) -> dict[Event, Event] | None:
    """Check if es1 and es2 are isomorphic event structures.

    Returns the isomorphism mapping, or None if not isomorphic.
    """
    if len(es1.events) != len(es2.events):
        return None
    if es1.num_conflicts != es2.num_conflicts:
        return None
    if es1.num_causal_pairs != es2.num_causal_pairs:
        return None

    # Try to find a bijective rigid morphism
    f = find_es_embedding(es1, es2, label_preserving=label_preserving)
    if f is None:
        return None

    # Verify it's surjective (should be since same size + injective)
    if set(f.values()) != set(es2.events):
        return None

    # Verify it's a proper isomorphism (reflects causality and conflict)
    if not is_isomorphism(f, es1, es2):
        return None

    return f


# ---------------------------------------------------------------------------
# High-level analysis
# ---------------------------------------------------------------------------

def analyze_es_morphisms(
    ss1: "StateSpace",
    ss2: "StateSpace",
) -> ESMorphismAnalysis:
    """Full morphism analysis between event structures of two state spaces.

    Builds event structures from both state spaces, then searches for
    morphisms in both directions. Classifies each morphism found.

    Args:
        ss1: First state space.
        ss2: Second state space.

    Returns:
        ESMorphismAnalysis with all results.
    """
    es1 = build_event_structure(ss1)
    es2 = build_event_structure(ss2)

    # Search morphisms in both directions
    f12 = find_es_morphism(es1, es2, label_preserving=True)
    f21 = find_es_morphism(es2, es1, label_preserving=True)

    morph12: ESMorphism | None = None
    morph21: ESMorphism | None = None
    lp12 = False
    lp21 = False

    if f12 is not None:
        kind12 = classify_es_morphism(f12, es1, es2)
        morph12 = ESMorphism(source=es1, target=es2, mapping=f12, kind=kind12)
        lp12 = is_label_preserving(f12, es1, es2)

    if f21 is not None:
        kind21 = classify_es_morphism(f21, es2, es1)
        morph21 = ESMorphism(source=es2, target=es1, mapping=f21, kind=kind21)
        lp21 = is_label_preserving(f21, es2, es1)

    # Check isomorphism
    iso_map = check_es_isomorphism(es1, es2, label_preserving=True)
    are_iso = iso_map is not None

    return ESMorphismAnalysis(
        es1=es1,
        es2=es2,
        morphism_1_to_2=morph12,
        morphism_2_to_1=morph21,
        are_isomorphic=are_iso,
        label_preserving_1_to_2=lp12,
        label_preserving_2_to_1=lp21,
    )
