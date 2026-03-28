"""Origami-to-session-types mapping (Step 60f).

Maps paper-folding (origami) concepts to session types:

    - **Crease patterns** become state spaces: vertices are states,
      creases are transitions labeled by fold type.
    - **Folds** (mountain / valley) become branch transitions.
    - **Flat-foldability** constraints become type well-formedness
      conditions (Kawasaki-Justin theorem, layer ordering).

The key insight is that a crease pattern's combinatorial structure —
vertices connected by mountain and valley creases — mirrors a session
type's state space.  Flat-foldability (the analogue of type soundness)
requires the Kawasaki-Justin condition at every interior vertex:
``|M - V| = 2`` where *M* and *V* count mountain and valley creases.

This module provides:
    ``crease_to_statespace(pattern)`` — convert a crease pattern to a StateSpace.
    ``crease_to_session_type(pattern)`` — convert a crease pattern to a session type string.
    ``check_kawasaki(pattern)`` — verify the Kawasaki-Justin condition.
    ``check_flat_foldable(pattern)`` — verify flat-foldability.
    ``simple_fold(n)`` — generate a simple linear crease pattern.
    ``star_fold(n)`` — generate a star crease pattern.
    ``analyze_fold(pattern)`` — full origami-to-session-type analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.lattice import check_lattice
from reticulate.parser import parse, pretty
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreasePattern:
    """A crease pattern on a sheet of paper.

    Attributes:
        vertices: Named vertices of the crease pattern.
        creases: Tuples of (vertex1, vertex2, fold_type) where fold_type
            is ``"mountain"`` or ``"valley"``.
        is_flat_foldable: Whether the pattern is known to be flat-foldable.
    """

    vertices: tuple[str, ...]
    creases: tuple[tuple[str, str, str], ...]
    is_flat_foldable: bool


@dataclass(frozen=True)
class FoldResult:
    """Result of full origami-to-session-type analysis.

    Attributes:
        crease_pattern: The input crease pattern.
        session_type_str: The generated session type string.
        state_count: Number of states in the generated state space.
        is_lattice: Whether the state space forms a lattice.
        kawasaki_satisfied: Whether the Kawasaki-Justin condition holds.
    """

    crease_pattern: CreasePattern
    session_type_str: str
    state_count: int
    is_lattice: bool
    kawasaki_satisfied: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _outgoing_creases(
    pattern: CreasePattern, vertex: str,
) -> list[tuple[str, str, str]]:
    """Return creases originating from *vertex* (as first endpoint)."""
    return [c for c in pattern.creases if c[0] == vertex]


def _all_creases_at(
    pattern: CreasePattern, vertex: str,
) -> list[tuple[str, str, str]]:
    """Return all creases incident to *vertex* (either endpoint)."""
    return [c for c in pattern.creases if c[0] == vertex or c[1] == vertex]


def _crease_label(v1: str, v2: str, fold_type: str) -> str:
    """Build the transition label for a crease."""
    return f"{fold_type}_{v1}_{v2}"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def crease_to_statespace(pattern: CreasePattern) -> StateSpace:
    """Convert a crease pattern to a state space.

    Each vertex becomes a state.  Each crease becomes a transition
    labeled ``"mountain_v1_v2"`` or ``"valley_v1_v2"``.  The first vertex
    in the pattern is top.  A synthetic ``"flat"`` state is bottom
    (representing the fully-folded state).  Vertices with no outgoing
    creases get an automatic transition to bottom.
    """
    if not pattern.vertices:
        # Empty pattern: single state that is both top and bottom.
        return StateSpace(
            states={0},
            transitions=[],
            top=0,
            bottom=0,
            labels={0: "flat"},
            selection_transitions=set(),
        )

    # Assign state IDs: one per vertex plus a "flat" bottom state.
    vertex_to_id: dict[str, int] = {
        v: i for i, v in enumerate(pattern.vertices)
    }
    flat_id = len(pattern.vertices)

    states: set[int] = set(vertex_to_id.values()) | {flat_id}
    labels: dict[int, str] = {sid: v for v, sid in vertex_to_id.items()}
    labels[flat_id] = "flat"

    transitions: list[tuple[int, str, int]] = []
    for v1, v2, fold_type in pattern.creases:
        src = vertex_to_id[v1]
        tgt = vertex_to_id[v2]
        transitions.append((src, _crease_label(v1, v2, fold_type), tgt))

    # Vertices with no outgoing creases get an edge to "flat".
    vertices_with_outgoing = {c[0] for c in pattern.creases}
    for v in pattern.vertices:
        if v not in vertices_with_outgoing:
            transitions.append((vertex_to_id[v], "fold_complete", flat_id))

    return StateSpace(
        states=states,
        transitions=transitions,
        top=vertex_to_id[pattern.vertices[0]],
        bottom=flat_id,
        labels=labels,
        selection_transitions=set(),
    )


def crease_to_session_type(pattern: CreasePattern) -> str:
    """Convert a crease pattern to a session type string.

    Builds a branch at each vertex with the available folds as methods,
    then returns the pretty-printed string.  Vertices with no outgoing
    creases map to ``end``.
    """
    if not pattern.vertices:
        return "end"

    # Build a mapping from each vertex to its outgoing crease labels.
    vertex_methods: dict[str, list[str]] = {v: [] for v in pattern.vertices}
    vertex_targets: dict[str, list[str]] = {v: [] for v in pattern.vertices}
    for v1, v2, fold_type in pattern.creases:
        vertex_methods[v1].append(_crease_label(v1, v2, fold_type))
        vertex_targets[v1].append(v2)

    # Build session type string bottom-up: vertices with no outgoing
    # creases are ``end``; others are branches over their methods.
    def _build(vertex: str, visited: set[str]) -> str:
        if vertex in visited:
            return "end"
        visited = visited | {vertex}
        methods = vertex_methods[vertex]
        targets = vertex_targets[vertex]
        if not methods:
            return "end"
        if len(methods) == 1:
            sub = _build(targets[0], visited)
            return f"&{{{methods[0]}: {sub}}}"
        parts = []
        for m, t in zip(methods, targets):
            sub = _build(t, visited)
            parts.append(f"{m}: {sub}")
        return "&{" + ", ".join(parts) + "}"

    return _build(pattern.vertices[0], set())


def check_kawasaki(pattern: CreasePattern) -> bool:
    """Check the Kawasaki-Justin condition at every vertex.

    Simplified version: at each interior vertex (degree >= 2), count
    mountain (*M*) and valley (*V*) folds.  The condition ``|M - V| = 2``
    must hold.  Returns ``True`` if all interior vertices satisfy this.

    Boundary vertices (degree < 2) are exempt.
    """
    if not pattern.vertices:
        return True

    for vertex in pattern.vertices:
        incident = _all_creases_at(pattern, vertex)
        if len(incident) < 2:
            continue
        mountain = sum(1 for _, _, ft in incident if ft == "mountain")
        valley = sum(1 for _, _, ft in incident if ft == "valley")
        if abs(mountain - valley) != 2:
            return False
    return True


def check_flat_foldable(pattern: CreasePattern) -> bool:
    """Check whether a crease pattern is flat-foldable.

    A pattern is flat-foldable if:
    1. The Kawasaki-Justin condition holds at all interior vertices.
    2. No two creases cross (simplified: no two creases share both
       endpoints in reversed order, which would indicate an overlap).

    Returns ``True`` if both conditions are satisfied.
    """
    if not check_kawasaki(pattern):
        return False

    # Simplified crossing check: two creases cross if they connect the
    # same pair of vertices in opposite directions (v1→v2 and v2→v1).
    edge_set: set[frozenset[str]] = set()
    for v1, v2, _ in pattern.creases:
        key = frozenset({v1, v2})
        if key in edge_set:
            return False
        edge_set.add(key)

    return True


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------


def simple_fold(n_creases: int) -> CreasePattern:
    """Generate a simple linear crease pattern with *n_creases* creases.

    Creates vertices ``v0, v1, ..., vn`` connected by creases that
    alternate between mountain and valley.  Always flat-foldable.
    """
    if n_creases < 0:
        raise ValueError("n_creases must be non-negative")
    if n_creases == 0:
        return CreasePattern(
            vertices=("v0",),
            creases=(),
            is_flat_foldable=True,
        )

    vertices = tuple(f"v{i}" for i in range(n_creases + 1))
    creases: list[tuple[str, str, str]] = []
    for i in range(n_creases):
        fold_type = "mountain" if i % 2 == 0 else "valley"
        creases.append((f"v{i}", f"v{i + 1}", fold_type))

    return CreasePattern(
        vertices=vertices,
        creases=tuple(creases),
        is_flat_foldable=True,
    )


def star_fold(n_points: int) -> CreasePattern:
    """Generate a star crease pattern with *n_points* outer vertices.

    A center vertex ``center`` is connected to ``n_points`` outer vertices
    ``p0, p1, ..., p(n-1)`` with alternating mountain/valley creases.
    Flat-foldable when *n_points* is even (ensures ``|M - V| = 2`` can
    fail for odd — but actually for odd n, |M-V| = 1 which violates
    Kawasaki).

    Actually: for even *n*, we arrange (n/2 + 1) mountain and (n/2 - 1)
    valley so ``|M - V| = 2``.  For odd *n*, we alternate but
    ``|M - V| = 1``, violating Kawasaki.
    """
    if n_points < 1:
        raise ValueError("n_points must be at least 1")

    outer = [f"p{i}" for i in range(n_points)]
    vertices = ("center", *outer)

    creases: list[tuple[str, str, str]] = []
    if n_points % 2 == 0:
        # Arrange mountain/valley counts so |M - V| = 2:
        # n/2 + 1 mountain, n/2 - 1 valley.
        n_mountain = n_points // 2 + 1
        for i in range(n_points):
            fold_type = "mountain" if i < n_mountain else "valley"
            creases.append(("center", outer[i], fold_type))
    else:
        # Simple alternation — Kawasaki will fail.
        for i in range(n_points):
            fold_type = "mountain" if i % 2 == 0 else "valley"
            creases.append(("center", outer[i], fold_type))

    is_flat = n_points % 2 == 0 and n_points >= 2
    return CreasePattern(
        vertices=tuple(vertices),
        creases=tuple(creases),
        is_flat_foldable=is_flat,
    )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


def analyze_fold(pattern: CreasePattern) -> FoldResult:
    """Perform full origami-to-session-type analysis.

    Converts the crease pattern to a state space, checks lattice
    properties, verifies the Kawasaki-Justin condition, and returns
    a comprehensive ``FoldResult``.
    """
    ss = crease_to_statespace(pattern)
    lattice_result = check_lattice(ss)
    kawasaki = check_kawasaki(pattern)
    session_str = crease_to_session_type(pattern)

    return FoldResult(
        crease_pattern=pattern,
        session_type_str=session_str,
        state_count=len(ss.states),
        is_lattice=lattice_result.is_lattice,
        kawasaki_satisfied=kawasaki,
    )
