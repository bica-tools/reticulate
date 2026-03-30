"""Replay prevention via lattice monotonicity (Step 89d).

Verifies that the lattice ordering of a session type state space prevents
replay attacks.  A replay attack consists of re-executing a previously
observed transition, potentially returning to an earlier protocol state.

Key insight: in a lattice-ordered state space, valid transitions always
move from a higher state to a lower state (closer to bottom).  A replay
attack attempts to move "upward" --- back to a state already visited.
If the state space is acyclic (a strict partial order), replays are
impossible because the ordering enforces monotone progress.  Cycles
(from recursive types) introduce potential replay vulnerabilities.

This module provides:
    ``check_replay_safety(ss)``
        — verify that lattice ordering prevents replay.
    ``monotone_monitor(ss)``
        — generate a monitor enforcing "state only decreases".
    ``replay_vulnerable_states(ss)``
        — identify states where replay could succeed.
    ``analyze_replay(ss)``
        — full analysis as ReplayResult.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonotoneMonitor:
    """A monitor that enforces monotone state progression.

    Tracks the current state and the set of visited states.
    Rejects any transition that would return to a previously
    visited state (replay detection).

    Attributes:
        initial_state: Starting state of the monitor.
        terminal_state: Terminal state of the monitor.
        state_machine: Mapping state -> [(label, target)].
        allowed_at: For each state, the transitions that maintain monotonicity.
        blocked_at: For each state, the transitions that would violate monotonicity.
    """

    initial_state: int
    terminal_state: int
    state_machine: dict[int, list[tuple[str, int]]]
    allowed_at: dict[int, list[tuple[str, int]]]
    blocked_at: dict[int, list[tuple[str, int]]]


@dataclass(frozen=True)
class ReplayResult:
    """Result of replay safety analysis.

    Attributes:
        is_replay_safe: True iff no replay is possible (no cycles).
        vulnerable_states: States involved in cycles where replay could occur.
        cycle_count: Number of distinct cycles found.
        cycles: List of cycles as lists of state IDs.
        monitor: A monotone monitor for enforcing replay prevention.
        scc_sizes: Size of each SCC (SCCs with size > 1 indicate cycles).
        max_scc_size: Size of the largest SCC.
        acyclic_fraction: Fraction of states not in any cycle.
    """

    is_replay_safe: bool
    vulnerable_states: frozenset[int]
    cycle_count: int
    cycles: list[list[int]]
    monitor: MonotoneMonitor
    scc_sizes: list[int]
    max_scc_size: int
    acyclic_fraction: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_adjacency(
    ss: StateSpace,
) -> dict[int, list[tuple[str, int]]]:
    """Build adjacency list: state -> [(label, target)]."""
    adj: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for src, label, tgt in ss.transitions:
        adj[src].append((label, tgt))
    return dict(adj)


def _find_sccs(ss: StateSpace) -> list[set[int]]:
    """Find strongly connected components using iterative Tarjan's algorithm."""
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    sccs: list[set[int]] = []

    adj: dict[int, list[int]] = defaultdict(list)
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    def strongconnect(v: int) -> None:
        # Iterative version to avoid stack overflow
        call_stack: list[tuple[int, int]] = [(v, 0)]
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        while call_stack:
            node, neighbor_idx = call_stack[-1]
            neighbors = adj.get(node, [])

            if neighbor_idx < len(neighbors):
                call_stack[-1] = (node, neighbor_idx + 1)
                w = neighbors[neighbor_idx]
                if w not in index:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, 0))
                elif w in on_stack:
                    lowlink[node] = min(lowlink[node], index[w])
            else:
                # All neighbors processed
                if lowlink[node] == index[node]:
                    scc: set[int] = set()
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.add(w)
                        if w == node:
                            break
                    sccs.append(scc)

                call_stack.pop()
                if call_stack:
                    parent, _ = call_stack[-1]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])

    for state in ss.states:
        if state not in index:
            strongconnect(state)

    return sccs


def _find_cycles(
    ss: StateSpace, *, max_cycles: int = 100
) -> list[list[int]]:
    """Find simple cycles in the state space via DFS.

    Returns up to max_cycles distinct cycles as lists of state IDs.
    """
    adj: dict[int, list[int]] = defaultdict(list)
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    cycles: list[list[int]] = []

    for start in sorted(ss.states):
        # DFS from start looking for cycles back to start
        stack: list[tuple[int, list[int], set[int]]] = [
            (start, [start], {start})
        ]
        while stack and len(cycles) < max_cycles:
            state, path, visited = stack.pop()
            for neighbor in adj.get(state, []):
                if neighbor == start and len(path) > 1:
                    cycles.append(path + [start])
                    if len(cycles) >= max_cycles:
                        break
                elif neighbor not in visited and neighbor > start:
                    # Only explore neighbors > start to avoid duplicates
                    stack.append(
                        (neighbor, path + [neighbor], visited | {neighbor})
                    )

    return cycles


def _has_self_loop(ss: StateSpace, state: int) -> bool:
    """Check if a state has a self-loop transition."""
    for src, _, tgt in ss.transitions:
        if src == state and tgt == state:
            return True
    return False


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def check_replay_safety(ss: StateSpace) -> bool:
    """Check whether the state space is replay-safe.

    A state space is replay-safe if it contains no cycles ---
    the lattice ordering enforces strict monotone decrease in
    state (toward bottom), making it impossible to replay a
    transition and return to a previously visited state.

    Self-loops count as cycles (replaying a transition that returns
    to the same state).

    Args:
        ss: The state space to check.

    Returns:
        True iff the state space is acyclic (replay-safe).
    """
    # Check for self-loops
    for src, _, tgt in ss.transitions:
        if src == tgt:
            return False

    # Check for multi-state cycles via SCC
    sccs = _find_sccs(ss)
    for scc in sccs:
        if len(scc) > 1:
            return False

    return True


def monotone_monitor(ss: StateSpace) -> MonotoneMonitor:
    """Generate a monitor that enforces monotone state progression.

    The monitor tracks visited states and blocks any transition that
    would return to a previously visited state.  For acyclic state
    spaces, no transitions are blocked.  For cyclic spaces, the
    back edges (transitions creating cycles) are blocked.

    Args:
        ss: The state space to monitor.

    Returns:
        MonotoneMonitor with allowed and blocked transitions per state.
    """
    adj = _build_adjacency(ss)

    # Find back edges via DFS
    visited: set[int] = set()
    in_progress: set[int] = set()
    back_edges: set[tuple[int, str, int]] = set()

    def dfs(state: int) -> None:
        visited.add(state)
        in_progress.add(state)
        for label, tgt in adj.get(state, []):
            if tgt in in_progress:
                back_edges.add((state, label, tgt))
            elif tgt not in visited:
                dfs(tgt)
        in_progress.discard(state)

    # Use iterative DFS to avoid recursion limit
    iter_visited: set[int] = set()
    iter_in_progress: set[int] = set()
    call_stack: list[tuple[int, int, bool]] = []  # (state, neighbor_idx, entering)

    for start in sorted(ss.states):
        if start in iter_visited:
            continue
        call_stack.append((start, 0, True))

        while call_stack:
            state, n_idx, entering = call_stack[-1]
            neighbors = adj.get(state, [])

            if entering:
                iter_visited.add(state)
                iter_in_progress.add(state)
                call_stack[-1] = (state, 0, False)

            if n_idx < len(neighbors):
                call_stack[-1] = (state, n_idx + 1, False)
                label, tgt = neighbors[n_idx]
                if tgt in iter_in_progress:
                    back_edges.add((state, label, tgt))
                elif tgt not in iter_visited:
                    call_stack.append((tgt, 0, True))
            else:
                iter_in_progress.discard(state)
                call_stack.pop()

    # Build allowed/blocked per state
    allowed: dict[int, list[tuple[str, int]]] = {}
    blocked: dict[int, list[tuple[str, int]]] = {}

    for state in ss.states:
        state_allowed: list[tuple[str, int]] = []
        state_blocked: list[tuple[str, int]] = []
        for label, tgt in adj.get(state, []):
            if (state, label, tgt) in back_edges:
                state_blocked.append((label, tgt))
            else:
                state_allowed.append((label, tgt))
        allowed[state] = state_allowed
        blocked[state] = state_blocked

    return MonotoneMonitor(
        initial_state=ss.top,
        terminal_state=ss.bottom,
        state_machine=adj,
        allowed_at=allowed,
        blocked_at=blocked,
    )


def replay_vulnerable_states(ss: StateSpace) -> frozenset[int]:
    """Identify states where a replay attack could succeed.

    A state is replay-vulnerable if it is part of a cycle ---
    a transition from that state could eventually lead back to it.

    Args:
        ss: The state space to analyze.

    Returns:
        Frozenset of state IDs that are part of cycles.
    """
    sccs = _find_sccs(ss)
    vulnerable: set[int] = set()

    for scc in sccs:
        if len(scc) > 1:
            vulnerable |= scc
        elif len(scc) == 1:
            state = next(iter(scc))
            if _has_self_loop(ss, state):
                vulnerable.add(state)

    return frozenset(vulnerable)


def analyze_replay(ss: StateSpace) -> ReplayResult:
    """Full replay safety analysis.

    Checks for replay safety, identifies vulnerable states and cycles,
    and generates a monotone monitor.

    Args:
        ss: The state space to analyze.

    Returns:
        ReplayResult with comprehensive analysis.
    """
    is_safe = check_replay_safety(ss)
    vulnerable = replay_vulnerable_states(ss)
    cycles = _find_cycles(ss)
    monitor = monotone_monitor(ss)

    sccs = _find_sccs(ss)
    scc_sizes = sorted([len(scc) for scc in sccs], reverse=True)
    max_size = scc_sizes[0] if scc_sizes else 0

    total_states = len(ss.states)
    acyclic_states = total_states - len(vulnerable)
    acyclic_frac = acyclic_states / total_states if total_states > 0 else 1.0

    return ReplayResult(
        is_replay_safe=is_safe,
        vulnerable_states=vulnerable,
        cycle_count=len(cycles),
        cycles=cycles,
        monitor=monitor,
        scc_sizes=scc_sizes,
        max_scc_size=max_size,
        acyclic_fraction=acyclic_frac,
    )
