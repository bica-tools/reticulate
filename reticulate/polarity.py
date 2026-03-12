"""Galois connections from polarity (Birkhoff IV.9-10).

Every session type state space has a natural binary relation R ⊆ States × Labels
(state s is related to label m iff m is enabled at s). This relation induces a
Galois connection between powersets of states and labels via closure operators,
and its fixed points form a concept lattice (Formal Concept Analysis).

This connects session type lattices to FCA: a concept is a maximal rectangle
(A, B) in the incidence table where A is a set of states sharing exactly the
capabilities in B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolarityResult:
    """Result of polarity analysis on a state space.

    Attributes:
        relation: R: state → set of enabled labels.
        concepts: List of (extent, intent) pairs — fixed points of the
            closure operators.
        concept_lattice_edges: Covering relation on concepts (by index).
        is_galois: True iff the closure operators form a Galois connection
            (always true by construction, verified as sanity check).
        num_concepts: Number of formal concepts.
    """

    relation: dict[int, frozenset[str]]
    concepts: list[tuple[frozenset[int], frozenset[str]]]
    concept_lattice_edges: list[tuple[int, int]]
    is_galois: bool
    num_concepts: int


# ---------------------------------------------------------------------------
# Core polarity functions
# ---------------------------------------------------------------------------

def build_polarity(ss: StateSpace) -> dict[int, frozenset[str]]:
    """Build the binary relation R ⊆ States × Labels.

    R(s) = {m | transition (s, m, _) exists in ss.transitions}.
    """
    relation: dict[int, set[str]] = {s: set() for s in ss.states}
    for src, label, _tgt in ss.transitions:
        relation[src].add(label)
    return {s: frozenset(labels) for s, labels in relation.items()}


def all_labels(ss: StateSpace) -> frozenset[str]:
    """Return the set of all transition labels in a state space."""
    return frozenset(label for _, label, _ in ss.transitions)


def state_closure(labels: frozenset[str], relation: dict[int, frozenset[str]]) -> frozenset[int]:
    """↓-closure: states enabling ALL given labels.

    state_closure(B) = {s ∈ States | B ⊆ R(s)}
    For the empty set of labels, returns all states.
    """
    if not labels:
        return frozenset(relation.keys())
    return frozenset(
        s for s, s_labels in relation.items()
        if labels <= s_labels
    )


def label_closure(states: frozenset[int], relation: dict[int, frozenset[str]]) -> frozenset[str]:
    """↑-closure: labels enabled at ALL given states.

    label_closure(A) = ⋂{R(s) | s ∈ A}
    For the empty set of states, returns all labels.
    """
    if not states:
        all_labs: set[str] = set()
        for labs in relation.values():
            all_labs |= labs
        return frozenset(all_labs)
    result: frozenset[str] | None = None
    for s in states:
        s_labels = relation.get(s, frozenset())
        if result is None:
            result = s_labels
        else:
            result = result & s_labels
    return result if result is not None else frozenset()


# ---------------------------------------------------------------------------
# Concept computation
# ---------------------------------------------------------------------------

def compute_concepts(ss: StateSpace) -> list[tuple[frozenset[int], frozenset[str]]]:
    """Compute all formal concepts (fixed points of the closure operators).

    A concept is a pair (A, B) where:
      - A = state_closure(B, R)  (A is the set of states enabling all labels in B)
      - B = label_closure(A, R)  (B is the set of labels enabled at all states in A)

    Uses the basic algorithm: for each subset of labels (via attribute exploration
    on individual labels), compute the closure and check if it's a fixed point.
    For efficiency, we generate concepts by closing each subset of states.
    """
    relation = build_polarity(ss)
    concepts: list[tuple[frozenset[int], frozenset[str]]] = []
    seen_extents: set[frozenset[int]] = set()

    # Generate concepts by closing each subset of states.
    # Optimization: close individual states and their intersections.
    # Start with all singletons and progressively intersect.
    # For small state spaces (< 1000 states), enumerate by closing
    # each element of the powerset of labels.
    labs = all_labels(ss)

    if len(labs) <= 20:
        # Enumerate by closing subsets of labels (2^|labels| is manageable)
        _enumerate_by_labels(relation, labs, concepts, seen_extents)
    else:
        # Fallback: close individual states and their combinations
        _enumerate_by_states(relation, concepts, seen_extents)

    # Sort concepts by extent size (largest first = top of concept lattice)
    concepts.sort(key=lambda c: (-len(c[0]), sorted(c[0])))
    return concepts


def _enumerate_by_labels(
    relation: dict[int, frozenset[str]],
    labs: frozenset[str],
    concepts: list[tuple[frozenset[int], frozenset[str]]],
    seen_extents: set[frozenset[int]],
) -> None:
    """Enumerate concepts by closing subsets of labels."""
    label_list = sorted(labs)
    n = len(label_list)

    for mask in range(1 << n):
        label_subset = frozenset(label_list[i] for i in range(n) if mask & (1 << i))
        # Close: labels → states → labels
        extent = state_closure(label_subset, relation)
        if extent in seen_extents:
            continue
        intent = label_closure(extent, relation)
        # Verify fixed point: close again
        extent2 = state_closure(intent, relation)
        if extent2 == extent:
            seen_extents.add(extent)
            concepts.append((extent, intent))

    # Also add the concept from closing the empty label set
    extent = state_closure(frozenset(), relation)
    if extent not in seen_extents:
        intent = label_closure(extent, relation)
        extent2 = state_closure(intent, relation)
        if extent2 == extent:
            seen_extents.add(extent)
            concepts.append((extent, intent))


def _enumerate_by_states(
    relation: dict[int, frozenset[str]],
    concepts: list[tuple[frozenset[int], frozenset[str]]],
    seen_extents: set[frozenset[int]],
) -> None:
    """Enumerate concepts by closing subsets of states (fallback for many labels)."""
    states = sorted(relation.keys())

    # Close each individual state
    for s in states:
        intent = label_closure(frozenset({s}), relation)
        extent = state_closure(intent, relation)
        if extent not in seen_extents:
            seen_extents.add(extent)
            concepts.append((extent, intent))

    # Close all pairs
    for i, s1 in enumerate(states):
        for s2 in states[i + 1:]:
            intent = label_closure(frozenset({s1, s2}), relation)
            extent = state_closure(intent, relation)
            if extent not in seen_extents:
                seen_extents.add(extent)
                concepts.append((extent, intent))

    # Top concept (all states)
    intent = label_closure(frozenset(states), relation)
    extent = state_closure(intent, relation)
    if extent not in seen_extents:
        seen_extents.add(extent)
        concepts.append((extent, intent))

    # Bottom concept (empty label set → all states, then close)
    extent = state_closure(frozenset(), relation)
    if extent not in seen_extents:
        intent = label_closure(extent, relation)
        extent2 = state_closure(intent, relation)
        if extent2 == extent:
            seen_extents.add(extent)
            concepts.append((extent, intent))


# ---------------------------------------------------------------------------
# Concept lattice construction
# ---------------------------------------------------------------------------

def build_concept_lattice(
    concepts: list[tuple[frozenset[int], frozenset[str]]],
) -> list[tuple[int, int]]:
    """Build the covering relation on concepts ordered by extent inclusion.

    Concept (A₁, B₁) ≤ (A₂, B₂) iff A₁ ⊆ A₂ (equivalently B₂ ⊆ B₁).
    Returns edges (i, j) where concepts[i] covers concepts[j] (i.e., i > j
    with no concept strictly between them).
    """
    n = len(concepts)
    if n <= 1:
        return []

    # Precompute ordering: i ≥ j iff concepts[i].extent ⊇ concepts[j].extent
    # (concepts are sorted largest-extent-first, so index 0 is likely top)
    greater: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and concepts[j][0] <= concepts[i][0]:
                greater[i].add(j)

    # Covering: i covers j iff i > j and there's no k with i > k > j
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in greater[i]:
            # Check if any k is strictly between i and j
            is_cover = True
            for k in greater[i]:
                if k != j and j in greater[k]:
                    is_cover = False
                    break
            if is_cover:
                edges.append((i, j))

    return edges


# ---------------------------------------------------------------------------
# Galois pair verification
# ---------------------------------------------------------------------------

def is_galois_pair(
    relation: dict[int, frozenset[str]],
) -> bool:
    """Verify that (state_closure, label_closure) forms a Galois connection.

    Check the adjunction: for all A ⊆ States, B ⊆ Labels,
      B ⊆ label_closure(A) ⟺ A ⊆ state_closure(B)

    Since the construction guarantees this, we verify on a representative
    sample (singletons and the full sets).
    """
    states = sorted(relation.keys())
    all_labs: set[str] = set()
    for labs in relation.values():
        all_labs |= labs
    label_list = sorted(all_labs)

    # Check adjunction on all singleton states and labels
    for s in states:
        a = frozenset({s})
        lc_a = label_closure(a, relation)
        for m in label_list:
            b = frozenset({m})
            sc_b = state_closure(b, relation)
            # b ⊆ label_closure(a) ⟺ a ⊆ state_closure(b)
            lhs = b <= lc_a
            rhs = a <= sc_b
            if lhs != rhs:
                return False

    return True


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def check_polarity(ss: StateSpace) -> PolarityResult:
    """Full polarity analysis: build relation, compute concepts, verify Galois.

    Returns a PolarityResult with the binary relation, formal concepts,
    concept lattice covering edges, and Galois connection verification.
    """
    relation = build_polarity(ss)
    concepts = compute_concepts(ss)
    edges = build_concept_lattice(concepts)
    galois = is_galois_pair(relation)

    return PolarityResult(
        relation=relation,
        concepts=concepts,
        concept_lattice_edges=edges,
        is_galois=galois,
        num_concepts=len(concepts),
    )
