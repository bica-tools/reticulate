"""Orphaned session detection for session types (Step 80e).

An orphaned state is one from which the bottom (terminal) state is
unreachable.  These states represent protocol paths that can never
complete -- the session is started but can never be properly closed.

Unlike livelock (cyclic trapping), orphan states may be acyclic dead
ends or parts of cycles.  The key criterion is simple: can bottom be
reached?

This module provides:
- **Detection**: Find all orphaned states (bottom-unreachable).
- **Depth analysis**: How deep in the state space orphans appear.
- **Repair**: Suggest transitions to add to connect orphans to bottom.
- **Comprehensive analysis**: Combine all techniques.

The relationship to other safety properties:
- Deadlock: a state with no outgoing transitions (subset of orphans).
- Livelock: a trapped cycle (subset of orphans if cycle has no exit).
- Orphan: any state that cannot reach bottom, regardless of mechanism.

Public API:
    detect_orphans(ss)       -- states from which bottom is unreachable
    orphan_depth(ss, state)  -- depth of an orphan in the state space
    is_orphan_free(ss)       -- True iff all states can reach bottom
    repair_orphans(ss)       -- suggest transitions to make all reach bottom
    analyze_orphans(ss)      -- comprehensive OrphanResult
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrphanState:
    """A single orphaned state.

    Attributes:
        state: The state ID.
        label: Human-readable state label.
        depth_from_top: Shortest path from top to this state (-1 if unreachable).
        has_outgoing: True if the state has outgoing transitions.
        is_cyclic: True if the state is part of a cycle.
        nearest_safe: Closest non-orphan state reachable in reverse (-1 if none).
    """
    state: int
    label: str
    depth_from_top: int
    has_outgoing: bool
    is_cyclic: bool
    nearest_safe: int


@dataclass(frozen=True)
class RepairSuggestion:
    """A suggested transition to repair an orphan.

    Attributes:
        src: Source state (the orphan or a state near it).
        label: Suggested transition label.
        tgt: Target state (bottom or a state that can reach bottom).
        rationale: Why this repair is suggested.
    """
    src: int
    label: str
    tgt: int
    rationale: str


@dataclass(frozen=True)
class OrphanResult:
    """Comprehensive orphan analysis result.

    Attributes:
        is_orphan_free: True iff all states can reach bottom.
        orphaned_states: Details of each orphaned state.
        num_orphans: Number of orphaned states.
        num_total_states: Total states.
        orphan_ratio: Fraction of states that are orphaned.
        max_orphan_depth: Maximum depth of any orphan from top.
        repairs: Suggested transitions to fix orphans.
        num_repairs_needed: Minimum repairs needed.
        reachable_from_top_orphans: Orphans reachable from top.
        unreachable_from_top_orphans: Orphans not even reachable from top.
        summary: Human-readable summary.
    """
    is_orphan_free: bool
    orphaned_states: tuple[OrphanState, ...]
    num_orphans: int
    num_total_states: int
    orphan_ratio: float
    max_orphan_depth: int
    repairs: tuple[RepairSuggestion, ...]
    num_repairs_needed: int
    reachable_from_top_orphans: frozenset[int]
    unreachable_from_top_orphans: frozenset[int]
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _outgoing(ss: "StateSpace", state: int) -> list[tuple[str, int]]:
    """Return outgoing (label, target) pairs for a state."""
    return [(label, tgt) for src, label, tgt in ss.transitions if src == state]


def _bfs_distances(ss: "StateSpace", start: int) -> dict[int, int]:
    """BFS shortest distances from *start*."""
    dist: dict[int, int] = {start: 0}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        for label, t in _outgoing(ss, s):
            if t not in dist:
                dist[t] = dist[s] + 1
                queue.append(t)
    return dist


def _reverse_reachable(ss: "StateSpace", target: int) -> set[int]:
    """States that can reach *target* via reverse edges."""
    rev: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        rev.setdefault(tgt, []).append(src)
    visited: set[int] = set()
    queue = deque([target])
    visited.add(target)
    while queue:
        s = queue.popleft()
        for pred in rev.get(s, []):
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)
    return visited


def _reachable_from(ss: "StateSpace", start: int) -> set[int]:
    """States reachable from *start* via forward edges."""
    visited: set[int] = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        s = queue.popleft()
        for label, t in _outgoing(ss, s):
            if t not in visited:
                visited.add(t)
                queue.append(t)
    return visited


def _is_cyclic_state(ss: "StateSpace", state: int) -> bool:
    """Check if *state* is part of a cycle (can reach itself)."""
    visited: set[int] = set()
    queue = deque()
    # Start from successors of state
    for label, t in _outgoing(ss, state):
        if t == state:
            return True  # Self-loop
        if t not in visited:
            visited.add(t)
            queue.append(t)

    while queue:
        s = queue.popleft()
        if s == state:
            return True
        for label, t in _outgoing(ss, s):
            if t not in visited:
                visited.add(t)
                queue.append(t)

    return False


def _find_nearest_safe(
    ss: "StateSpace",
    state: int,
    safe_states: set[int],
) -> int:
    """Find the nearest safe state reachable via reverse edges.

    Returns -1 if no safe state can reach this orphan.
    """
    # BFS in reverse from the orphan
    rev: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        rev.setdefault(tgt, []).append(src)

    visited: set[int] = {state}
    queue: deque[tuple[int, int]] = deque()
    for pred in rev.get(state, []):
        if pred not in visited:
            visited.add(pred)
            queue.append((pred, 1))

    while queue:
        s, dist = queue.popleft()
        if s in safe_states:
            return s
        for pred in rev.get(s, []):
            if pred not in visited:
                visited.add(pred)
                queue.append((pred, dist + 1))

    return -1


def _compute_repair_suggestions(
    ss: "StateSpace",
    orphans: set[int],
    safe_states: set[int],
) -> list[RepairSuggestion]:
    """Suggest transitions to repair orphaned states.

    Strategy: for each orphan, suggest adding a transition to the nearest
    safe state or directly to bottom.  Group orphans in SCCs to minimize
    the number of repairs needed.
    """
    repairs: list[RepairSuggestion] = []

    # Group orphans that are mutually reachable (same SCC-like component)
    processed: set[int] = set()

    for orphan in sorted(orphans):
        if orphan in processed:
            continue

        # Find all orphans reachable from this one
        reachable_orphans = _reachable_from(ss, orphan) & orphans
        processed |= reachable_orphans

        # Find the orphan closest to a safe state
        best_orphan = orphan
        best_label = "_repair"

        # Try to find an orphan with outgoing transitions
        for o in sorted(reachable_orphans):
            out = _outgoing(ss, o)
            if out:
                best_orphan = o
                break

        # Suggest connecting to bottom
        repairs.append(RepairSuggestion(
            src=best_orphan,
            label=best_label,
            tgt=ss.bottom,
            rationale=(
                f"Connect orphan state {best_orphan} to bottom. "
                f"This repairs {len(reachable_orphans)} orphan(s) "
                f"in the same connected component."
            ),
        ))

    return repairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_orphans(ss: "StateSpace") -> frozenset[int]:
    """Find all orphaned states: states from which bottom is unreachable.

    A state is orphaned if no directed path leads from it to the bottom
    state.  This includes dead-end states, trapped cycles, and states
    that are entirely disconnected from bottom.

    Returns:
        Frozenset of orphaned state IDs.
    """
    can_reach_bottom = _reverse_reachable(ss, ss.bottom)
    return frozenset(ss.states - can_reach_bottom)


def orphan_depth(ss: "StateSpace", state: int) -> int:
    """Compute how deep an orphan is from the top state.

    Returns the shortest path length from top to the given state.
    Returns -1 if the state is not reachable from top.
    """
    distances = _bfs_distances(ss, ss.top)
    return distances.get(state, -1)


def is_orphan_free(ss: "StateSpace") -> bool:
    """Return True iff all states can reach bottom.

    A state space is orphan-free if every state has at least one
    directed path to the bottom state.
    """
    return len(detect_orphans(ss)) == 0


def repair_orphans(ss: "StateSpace") -> list[RepairSuggestion]:
    """Suggest transitions to add to make all states reach bottom.

    Returns a list of RepairSuggestion objects, each suggesting a
    transition to add.  The suggestions aim to minimize the number
    of new transitions while ensuring all orphans can reach bottom.
    """
    orphans = detect_orphans(ss)
    if not orphans:
        return []

    safe_states = set(ss.states) - orphans
    return _compute_repair_suggestions(ss, orphans, safe_states)


def analyze_orphans(ss: "StateSpace") -> OrphanResult:
    """Comprehensive orphan analysis.

    Combines orphan detection, depth analysis, cycle detection,
    and repair suggestion.

    Returns:
        OrphanResult with complete analysis.
    """
    can_reach_bottom = _reverse_reachable(ss, ss.bottom)
    orphans = frozenset(ss.states - can_reach_bottom)
    safe_states = can_reach_bottom

    from_top = _reachable_from(ss, ss.top)
    distances = _bfs_distances(ss, ss.top)

    orphan_details: list[OrphanState] = []
    max_depth = -1
    reachable_from_top_orphans: set[int] = set()
    unreachable_from_top_orphans: set[int] = set()

    for s in sorted(orphans):
        depth = distances.get(s, -1)
        has_out = len(_outgoing(ss, s)) > 0
        is_cyclic = _is_cyclic_state(ss, s)
        nearest = _find_nearest_safe(ss, s, safe_states)
        label = ss.labels.get(s, f"state_{s}")

        if depth >= 0:
            max_depth = max(max_depth, depth)
            reachable_from_top_orphans.add(s)
        else:
            unreachable_from_top_orphans.add(s)

        orphan_details.append(OrphanState(
            state=s,
            label=label,
            depth_from_top=depth,
            has_outgoing=has_out,
            is_cyclic=is_cyclic,
            nearest_safe=nearest,
        ))

    # Repairs
    repairs = _compute_repair_suggestions(ss, orphans, safe_states)

    num_total = len(ss.states)
    num_orphans = len(orphans)
    ratio = num_orphans / num_total if num_total > 0 else 0.0

    is_free = num_orphans == 0

    if is_free:
        summary = "Orphan-free: all states can reach bottom."
    else:
        summary = (
            f"ORPHANS DETECTED: {num_orphans} of {num_total} states "
            f"({ratio:.0%}) cannot reach bottom. "
            f"{len(reachable_from_top_orphans)} reachable from top, "
            f"{len(unreachable_from_top_orphans)} not reachable from top. "
            f"{len(repairs)} repair(s) suggested."
        )

    return OrphanResult(
        is_orphan_free=is_free,
        orphaned_states=tuple(orphan_details),
        num_orphans=num_orphans,
        num_total_states=num_total,
        orphan_ratio=ratio,
        max_orphan_depth=max_depth,
        repairs=tuple(repairs),
        num_repairs_needed=len(repairs),
        reachable_from_top_orphans=frozenset(reachable_from_top_orphans),
        unreachable_from_top_orphans=frozenset(unreachable_from_top_orphans),
        summary=summary,
    )
