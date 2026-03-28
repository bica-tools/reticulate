"""Time as session type unfolding (Step 203).

The core insight: time doesn't "flow" — it IS the unfolding of recursive
session types.  The present is the current state, the future is the reachable
set, the past is the trace so far.  Temporal logic operators (◇, □, U, ○)
map directly to session type state-space properties.

Key concepts:

- **Present**: the current state in the state space.
- **Future**: the set of states reachable from the present (BFS).
- **Past**: the trace of (state, label) pairs traversed to reach the present.
- **Arrow of time**: irreversibility = the state space is a DAG (no cycles).
- **Time remaining**: minimum transitions to the bottom state.
- **Branching futures**: the number of distinct paths from present to bottom.

Temporal logic operators:

- **◇ (eventually)**: a label will be available at some reachable state.
- **□ (always)**: a label is available at every reachable state.
- **U (until)**: one label holds until another becomes available.
- **○ (next)**: labels enabled at the current state.

This module provides:
    ``now(ss, state)``               — temporal view from a given state.
    ``step(ss, view, label)``        — take a transition, advance time.
    ``eventually(ss, state, label)`` — ◇: label reachable somewhere ahead.
    ``always(ss, state, label)``     — □: label available at every future state.
    ``until(ss, state, hold, goal)`` — U: hold until goal.
    ``next_possible(ss, state)``     — ○: enabled labels at state.
    ``time_to_end(ss, state)``       — min transitions to bottom.
    ``arrow_of_time(ss)``            — True if state space is a DAG.
    ``time_branches(ss, state)``     — number of distinct paths to bottom.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalView:
    """Snapshot of a session type execution at a moment in time.

    Attributes:
        present: current state ID.
        past: trace of (state, label) pairs taken so far.
        future_states: states reachable from present (excluding present).
        future_labels: labels enabled at present.
        time_remaining: minimum transitions to bottom from present.
        time_elapsed: number of steps taken (len of past).
        is_terminal: True if present is the bottom state.
        branching_futures: number of distinct paths from present to bottom.
        deterministic_future: True if at most one outgoing transition.
    """

    present: int
    past: tuple[tuple[int, str], ...]
    future_states: frozenset[int]
    future_labels: frozenset[str]
    time_remaining: int
    time_elapsed: int
    is_terminal: bool
    branching_futures: int
    deterministic_future: bool


@dataclass(frozen=True)
class TemporalProperty:
    """Result of checking a temporal logic property.

    Attributes:
        name: name of the temporal operator (eventually, always, until).
        holds: whether the property holds.
        witness: explanation or counterexample.
    """

    name: str
    holds: bool
    witness: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _adjacency(ss: StateSpace) -> dict[int, list[tuple[str, int]]]:
    """Build adjacency list: state -> [(label, target)]."""
    adj: dict[int, list[tuple[str, int]]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        adj[src].append((label, tgt))
    return adj


def _reachable_bfs(
    adj: dict[int, list[tuple[str, int]]], start: int,
) -> frozenset[int]:
    """All states reachable from *start* (excluding start itself)."""
    visited: set[int] = set()
    queue = deque[int]()
    for _, tgt in adj.get(start, []):
        if tgt not in visited:
            visited.add(tgt)
            queue.append(tgt)
    while queue:
        s = queue.popleft()
        for _, tgt in adj.get(s, []):
            if tgt not in visited:
                visited.add(tgt)
                queue.append(tgt)
    return frozenset(visited)


def _enabled_labels(
    adj: dict[int, list[tuple[str, int]]], state: int,
) -> frozenset[str]:
    """Labels enabled at *state*."""
    return frozenset(label for label, _ in adj.get(state, []))


def _min_distance_to(
    adj: dict[int, list[tuple[str, int]]], start: int, target: int,
) -> int:
    """Minimum number of transitions from *start* to *target* (BFS).

    Returns -1 if target is unreachable.
    """
    if start == target:
        return 0
    visited: set[int] = {start}
    queue = deque[tuple[int, int]]()
    queue.append((start, 0))
    while queue:
        s, dist = queue.popleft()
        for _, tgt in adj.get(s, []):
            if tgt == target:
                return dist + 1
            if tgt not in visited:
                visited.add(tgt)
                queue.append((tgt, dist + 1))
    return -1


def _count_paths_to_bottom(
    adj: dict[int, list[tuple[str, int]]], start: int, bottom: int,
    bound: int = 10000,
) -> int:
    """Count distinct paths from *start* to *bottom*, bounded by *bound*.

    Uses memoized DFS.  Cycles are handled by tracking the current path
    to avoid infinite recursion.
    """
    cache: dict[int, int] = {}

    def dfs(state: int, on_path: frozenset[int]) -> int:
        if state == bottom:
            return 1
        if state in cache:
            return cache[state]
        total = 0
        for _, tgt in adj.get(state, []):
            if tgt not in on_path:
                total += dfs(tgt, on_path | {tgt})
                if total >= bound:
                    total = bound
                    break
        cache[state] = total
        return total

    return dfs(start, frozenset({start}))


def _has_cycle(adj: dict[int, list[tuple[str, int]]], states: set[int]) -> bool:
    """Check if the graph contains any cycle (DFS-based)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[int, int] = {s: WHITE for s in states}

    def dfs(u: int) -> bool:
        color[u] = GRAY
        for _, v in adj.get(u, []):
            if color.get(v, WHITE) == GRAY:
                return True
            if color.get(v, WHITE) == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for s in states:
        if color[s] == WHITE:
            if dfs(s):
                return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def now(ss: StateSpace, state: int) -> TemporalView:
    """Compute the temporal view from a given state.

    The past is empty (we don't know how we arrived here).
    The future is computed by BFS reachability from *state*.
    """
    adj = _adjacency(ss)
    future = _reachable_bfs(adj, state)
    labels = _enabled_labels(adj, state)
    t_remaining = _min_distance_to(adj, state, ss.bottom)
    paths = _count_paths_to_bottom(adj, state, ss.bottom)
    outgoing = adj.get(state, [])

    return TemporalView(
        present=state,
        past=(),
        future_states=future,
        future_labels=labels,
        time_remaining=t_remaining if t_remaining >= 0 else -1,
        time_elapsed=0,
        is_terminal=(state == ss.bottom),
        branching_futures=paths,
        deterministic_future=(len(outgoing) <= 1),
    )


def step(ss: StateSpace, view: TemporalView, label: str) -> TemporalView:
    """Take a transition labeled *label*, advancing time.

    Appends the old (present, label) to the past and recomputes the future.
    Raises ``ValueError`` if *label* is not enabled at the current state.
    """
    adj = _adjacency(ss)
    targets = [tgt for lbl, tgt in adj.get(view.present, []) if lbl == label]
    if not targets:
        enabled = sorted(_enabled_labels(adj, view.present))
        raise ValueError(
            f"Label '{label}' not enabled at state {view.present}; "
            f"enabled: {enabled}"
        )
    # Take the first matching transition (deterministic resolution).
    new_present = targets[0]
    new_past = view.past + ((view.present, label),)
    future = _reachable_bfs(adj, new_present)
    labels = _enabled_labels(adj, new_present)
    t_remaining = _min_distance_to(adj, new_present, ss.bottom)
    paths = _count_paths_to_bottom(adj, new_present, ss.bottom)
    outgoing = adj.get(new_present, [])

    return TemporalView(
        present=new_present,
        past=new_past,
        future_states=future,
        future_labels=labels,
        time_remaining=t_remaining if t_remaining >= 0 else -1,
        time_elapsed=len(new_past),
        is_terminal=(new_present == ss.bottom),
        branching_futures=paths,
        deterministic_future=(len(outgoing) <= 1),
    )


def eventually(ss: StateSpace, state: int, label: str) -> TemporalProperty:
    """Temporal logic ◇: 'eventually *label* will be available'.

    Checks whether any state reachable from *state* (including *state*
    itself) enables a transition labeled *label*.
    """
    adj = _adjacency(ss)
    # Check state itself first.
    if label in _enabled_labels(adj, state):
        return TemporalProperty(
            name="eventually",
            holds=True,
            witness=f"'{label}' is available at state {state} (now).",
        )
    # BFS over reachable states.
    visited: set[int] = {state}
    queue = deque[int]([state])
    while queue:
        s = queue.popleft()
        for _, tgt in adj.get(s, []):
            if tgt not in visited:
                if label in _enabled_labels(adj, tgt):
                    return TemporalProperty(
                        name="eventually",
                        holds=True,
                        witness=f"'{label}' becomes available at state {tgt}.",
                    )
                visited.add(tgt)
                queue.append(tgt)
    return TemporalProperty(
        name="eventually",
        holds=False,
        witness=f"'{label}' is never available from state {state}.",
    )


def always(ss: StateSpace, state: int, label: str) -> TemporalProperty:
    """Temporal logic □: 'label is always available'.

    Checks that every non-bottom state reachable from *state* (including
    *state* itself) enables *label*.
    """
    adj = _adjacency(ss)
    visited: set[int] = {state}
    queue = deque[int]([state])
    while queue:
        s = queue.popleft()
        # Skip bottom — it has no outgoing transitions by definition.
        if s == ss.bottom:
            continue
        if label not in _enabled_labels(adj, s):
            return TemporalProperty(
                name="always",
                holds=False,
                witness=f"'{label}' not available at state {s}.",
            )
        for _, tgt in adj.get(s, []):
            if tgt not in visited:
                visited.add(tgt)
                queue.append(tgt)
    return TemporalProperty(
        name="always",
        holds=True,
        witness=f"'{label}' is available at every reachable state from {state}.",
    )


def until(
    ss: StateSpace, state: int, hold_label: str, goal_label: str,
) -> TemporalProperty:
    """Temporal logic U: '*hold_label* until *goal_label*'.

    Checks that *hold_label* is enabled at every non-bottom reachable state
    from *state* until some state enables *goal_label*.  Uses BFS: every
    path must have *hold_label* enabled until *goal_label* is reached.
    """
    adj = _adjacency(ss)

    # BFS ensuring hold_label at every state before goal_label.
    visited: set[int] = {state}
    queue = deque[int]([state])
    goal_found = False

    while queue:
        s = queue.popleft()
        if s == ss.bottom:
            continue
        enabled = _enabled_labels(adj, s)
        if goal_label in enabled:
            goal_found = True
            continue  # Goal reached on this path — don't need to go further.
        if hold_label not in enabled:
            return TemporalProperty(
                name="until",
                holds=False,
                witness=(
                    f"'{hold_label}' not available at state {s} "
                    f"before '{goal_label}' was reached."
                ),
            )
        for _, tgt in adj.get(s, []):
            if tgt not in visited:
                visited.add(tgt)
                queue.append(tgt)

    if not goal_found:
        return TemporalProperty(
            name="until",
            holds=False,
            witness=f"'{goal_label}' is never reached from state {state}.",
        )
    return TemporalProperty(
        name="until",
        holds=True,
        witness=(
            f"'{hold_label}' holds at every state until "
            f"'{goal_label}' is reached."
        ),
    )


def next_possible(ss: StateSpace, state: int) -> list[str]:
    """Temporal logic ○: return all labels enabled at *state*."""
    adj = _adjacency(ss)
    return sorted(label for label, _ in adj.get(state, []))


def time_to_end(ss: StateSpace, state: int) -> int:
    """Minimum number of transitions from *state* to bottom.

    Returns -1 if bottom is unreachable.
    """
    adj = _adjacency(ss)
    return _min_distance_to(adj, state, ss.bottom)


def arrow_of_time(ss: StateSpace) -> bool:
    """True if the state space is a DAG (no cycles).

    A DAG means time is irreversible — you can never return to a previous
    state.  Cycles (from recursive types) mean time can loop.
    """
    adj = _adjacency(ss)
    return not _has_cycle(adj, ss.states)


def time_branches(ss: StateSpace, state: int) -> int:
    """Number of distinct paths from *state* to bottom.

    Bounded at 10000 to avoid combinatorial explosion.
    """
    adj = _adjacency(ss)
    return _count_paths_to_bottom(adj, state, ss.bottom)
