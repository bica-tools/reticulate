"""Gravitational session types: potential and force analysis (Step 59l).

Models session type state spaces as gravitational fields where:
    - **Potential** V(s) = longest path from s to bottom (height above ground).
    - **Force** F(s) = number of outgoing transitions (choices = acceleration).
    - **Kinetic branching** K(s) = potential drop per transition averaged
      over successors (how fast the protocol descends).
    - **Orbits**: strongly connected components (recursive cycles that resist
      termination — stable orbits at constant potential).
    - **Black holes**: states with no outgoing transitions that are NOT the
      bottom (deadlocked states — nothing escapes).
    - **Escape velocity**: minimum path length from a state to bottom
      (fastest route to termination).

The gravitational potential is a Lyapunov function on acyclic state spaces:
it strictly decreases along every transition, guaranteeing termination.
For recursive types, cycles create stable orbits at constant potential.

This module provides:
    ``gravitational_potential(ss)`` — compute V(s) for each state.
    ``gravitational_force(ss)`` — compute F(s) for each state.
    ``detect_orbits(ss)`` — find stable orbits (SCCs of size > 1).
    ``detect_black_holes(ss)`` — find deadlocked states.
    ``analyze_gravity(ss)`` — full gravitational analysis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GravitationalField:
    """Complete gravitational analysis of a state space.

    Attributes:
        potential: V(s) = longest acyclic path from s to bottom. 0 at bottom, -1 if unreachable.
        escape_velocity: Shortest path from s to bottom. 0 at bottom, -1 if unreachable.
        force: Number of outgoing transitions from each state.
        kinetic_branching: Average potential drop per transition from each state.
        orbits: List of sets of states forming stable orbits (SCCs of size > 1).
        black_holes: States with no outgoing transitions that are not bottom.
        total_energy: Sum of potentials across all states.
        is_lyapunov: True if potential strictly decreases on every non-orbit transition.
        max_potential: Maximum potential (always at top for well-formed types).
        gradient: For each transition (s, label, t), the potential drop V(s) - V(t).
    """

    potential: dict[int, int]
    escape_velocity: dict[int, int]
    force: dict[int, int]
    kinetic_branching: dict[int, float]
    orbits: tuple[frozenset[int], ...]
    black_holes: tuple[int, ...]
    total_energy: int
    is_lyapunov: bool
    max_potential: int
    gradient: dict[tuple[int, str, int], int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _outgoing(ss: StateSpace, state: int) -> list[tuple[str, int]]:
    """Return outgoing (label, target) pairs for a state."""
    return [(label, tgt) for src, label, tgt in ss.transitions if src == state]


def _longest_paths_to_bottom(ss: StateSpace) -> dict[int, int]:
    """Compute longest acyclic path from each state to bottom via reverse BFS + DP.

    Returns dict mapping state -> longest path length. -1 if unreachable.
    """
    # Build reverse adjacency
    reverse_adj: dict[int, list[int]] = {s: [] for s in ss.states}
    forward_adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _label, tgt in ss.transitions:
        reverse_adj[tgt].append(src)
        forward_adj[src].append(tgt)

    # Detect SCC membership for cycle handling
    orbit_states = _find_orbit_states(ss)

    # DP with memoization, DFS-based
    memo: dict[int, int] = {ss.bottom: 0}

    def _dp(s: int, visiting: frozenset[int]) -> int:
        if s in memo:
            return memo[s]
        if s in visiting:
            return -1  # cycle — don't count

        successors = forward_adj.get(s, [])
        if not successors:
            memo[s] = -1  # no outgoing, not bottom = black hole
            return -1

        best = -1
        new_visiting = visiting | {s}
        for t in successors:
            sub = _dp(t, new_visiting)
            if sub >= 0:
                best = max(best, 1 + sub)

        memo[s] = best
        return best

    for s in ss.states:
        if s not in memo:
            _dp(s, frozenset())

    return memo


def _shortest_paths_to_bottom(ss: StateSpace) -> dict[int, int]:
    """BFS from bottom backwards to compute shortest path to bottom."""
    # Reverse BFS from bottom
    reverse_adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _label, tgt in ss.transitions:
        reverse_adj[tgt].append(src)

    dist: dict[int, int] = {ss.bottom: 0}
    queue: deque[int] = deque([ss.bottom])
    while queue:
        s = queue.popleft()
        d = dist[s]
        for pred in reverse_adj[s]:
            if pred not in dist:
                dist[pred] = d + 1
                queue.append(pred)

    # States not reached get -1
    for s in ss.states:
        if s not in dist:
            dist[s] = -1

    return dist


def _find_orbit_states(ss: StateSpace) -> set[int]:
    """Find all states that belong to SCCs of size > 1 (orbits)."""
    # Iterative Tarjan's SCC
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    sccs: list[set[int]] = []

    forward_adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _label, tgt in ss.transitions:
        forward_adj[src].append(tgt)

    def strongconnect(v: int) -> None:
        # Iterative version
        work_stack: list[tuple[int, int]] = [(v, 0)]
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        while work_stack:
            node, idx = work_stack[-1]
            successors = forward_adj.get(node, [])

            if idx < len(successors):
                work_stack[-1] = (node, idx + 1)
                w = successors[idx]
                if w not in index:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    work_stack.append((w, 0))
                elif w in on_stack:
                    lowlink[node] = min(lowlink[node], index[w])
            else:
                if lowlink[node] == index[node]:
                    scc: set[int] = set()
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.add(w)
                        if w == node:
                            break
                    sccs.append(scc)
                work_stack.pop()
                if work_stack:
                    parent, _ = work_stack[-1]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])

    for s in ss.states:
        if s not in index:
            strongconnect(s)

    orbit_states: set[int] = set()
    for scc in sccs:
        if len(scc) > 1:
            orbit_states.update(scc)

    return orbit_states


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def gravitational_potential(ss: StateSpace) -> dict[int, int]:
    """Compute gravitational potential V(s) for each state.

    V(s) = longest acyclic path from s to bottom.
    V(bottom) = 0.  V(s) = -1 if s cannot reach bottom.
    """
    return _longest_paths_to_bottom(ss)


def escape_velocity(ss: StateSpace) -> dict[int, int]:
    """Compute escape velocity (shortest path to bottom) for each state.

    The minimum number of transitions to reach termination.
    """
    return _shortest_paths_to_bottom(ss)


def gravitational_force(ss: StateSpace) -> dict[int, int]:
    """Compute gravitational force F(s) = number of outgoing transitions.

    More transitions = more choices = more "acceleration" away from
    the current state.
    """
    force: dict[int, int] = {s: 0 for s in ss.states}
    for src, _label, _tgt in ss.transitions:
        force[src] += 1
    return force


def detect_orbits(ss: StateSpace) -> tuple[frozenset[int], ...]:
    """Find stable orbits: SCCs of size > 1 (recursive cycles)."""
    orbit_states = _find_orbit_states(ss)
    if not orbit_states:
        return ()

    # Group into individual SCCs
    # Re-run Tarjan's to get individual SCCs
    forward_adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _label, tgt in ss.transitions:
        forward_adj[src].append(tgt)

    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    orbits: list[frozenset[int]] = []

    def strongconnect(v: int) -> None:
        work_stack: list[tuple[int, int]] = [(v, 0)]
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        while work_stack:
            node, idx = work_stack[-1]
            successors = forward_adj.get(node, [])

            if idx < len(successors):
                work_stack[-1] = (node, idx + 1)
                w = successors[idx]
                if w not in index:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    work_stack.append((w, 0))
                elif w in on_stack:
                    lowlink[node] = min(lowlink[node], index[w])
            else:
                if lowlink[node] == index[node]:
                    scc: set[int] = set()
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.add(w)
                        if w == node:
                            break
                    if len(scc) > 1:
                        orbits.append(frozenset(scc))
                work_stack.pop()
                if work_stack:
                    parent, _ = work_stack[-1]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])

    for s in ss.states:
        if s not in index:
            strongconnect(s)

    return tuple(orbits)


def detect_black_holes(ss: StateSpace) -> tuple[int, ...]:
    """Find black holes: states with no outgoing transitions that are not bottom."""
    force = gravitational_force(ss)
    return tuple(sorted(s for s, f in force.items() if f == 0 and s != ss.bottom))


def analyze_gravity(ss: StateSpace) -> GravitationalField:
    """Full gravitational analysis of a state space."""
    potential = gravitational_potential(ss)
    esc_vel = escape_velocity(ss)
    force = gravitational_force(ss)
    orbits = detect_orbits(ss)
    black_holes = detect_black_holes(ss)

    orbit_states = set()
    for orbit in orbits:
        orbit_states.update(orbit)

    # Kinetic branching: average potential drop per transition
    kinetic: dict[int, float] = {}
    for s in ss.states:
        out = _outgoing(ss, s)
        if not out:
            kinetic[s] = 0.0
            continue
        v_s = potential[s]
        if v_s < 0:
            kinetic[s] = 0.0
            continue
        drops = []
        for _label, t in out:
            v_t = potential[t]
            if v_t >= 0:
                drops.append(v_s - v_t)
        kinetic[s] = sum(drops) / len(drops) if drops else 0.0

    # Gradient: potential drop per transition
    gradient: dict[tuple[int, str, int], int] = {}
    for src, label, tgt in ss.transitions:
        v_s = potential[src]
        v_t = potential[tgt]
        if v_s >= 0 and v_t >= 0:
            gradient[(src, label, tgt)] = v_s - v_t
        else:
            gradient[(src, label, tgt)] = 0

    # Lyapunov check: potential strictly decreases on all non-orbit transitions
    is_lyapunov = True
    for src, label, tgt in ss.transitions:
        if src in orbit_states and tgt in orbit_states:
            continue  # Skip orbit-internal transitions
        v_s = potential[src]
        v_t = potential[tgt]
        if v_s >= 0 and v_t >= 0 and v_s <= v_t:
            is_lyapunov = False
            break

    # Total energy and max potential
    valid_potentials = [v for v in potential.values() if v >= 0]
    total_energy = sum(valid_potentials)
    max_potential = max(valid_potentials) if valid_potentials else 0

    return GravitationalField(
        potential=potential,
        escape_velocity=esc_vel,
        force=force,
        kinetic_branching={s: round(k, 3) for s, k in kinetic.items()},
        orbits=orbits,
        black_holes=black_holes,
        total_energy=total_energy,
        is_lyapunov=is_lyapunov,
        max_potential=max_potential,
        gradient=gradient,
    )
