"""Lagrangian mechanics of session types (Step 60q).

Complete mechanical formulation of session type dynamics:

  L(x) = T(x) - V(x)

where T is kinetic energy (transition availability) and V is potential
energy (distance to termination).

Circular mechanics for recursive types:
- Orbital period, angular velocity, centripetal acceleration
- Inertia = SCC size (resistance to change)
- Forces: centripetal (recursion), tangential (branch), gravitational (end)

Path mechanics:
- Action = Σ L(x) along a path
- Least-action path = most natural execution
- Hamilton's principle for protocol optimization

Conservation laws:
- Energy conservation along deterministic paths
- Momentum = transition count (conserved in linear protocols)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _state_list,
    _adjacency,
    _reachability,
    _compute_sccs,
    compute_rank,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CircularMechanics:
    """Mechanics of a single recursive cycle (SCC).

    Attributes:
        scc_id: Representative state of the SCC.
        states: States in the cycle.
        period: Number of transitions in one full cycle.
        angular_velocity: 2π / period.
        inertia: Number of states in the SCC (resistance to change).
        centripetal_accel: v² / r ≈ period / inertia.
        internal_branches: Branch points within the cycle.
        tangential_accel: Number of internal branches (direction changes per orbit).
        escape_transitions: Transitions leaving the SCC (exit velocity).
    """
    scc_id: int
    states: frozenset[int]
    period: int
    angular_velocity: float
    inertia: int
    centripetal_accel: float
    internal_branches: int
    tangential_accel: int
    escape_transitions: int


@dataclass(frozen=True)
class LagrangianResult:
    """Complete Lagrangian mechanics analysis.

    Attributes:
        num_states: Number of states.
        kinetic_energy: T(x) for each state (velocity²/2).
        potential_energy: V(x) for each state (rank = distance to bottom).
        lagrangian: L(x) = T(x) - V(x) for each state.
        total_energy: Σ (T + V) over all states.
        least_action_path: Path minimizing total action.
        least_action_value: Action along the least-action path.
        max_action_path: Path maximizing total action.
        max_action_value: Action along the max-action path.
        circular_mechanics: Mechanics of each recursive cycle.
        gravitational_field: "Pull" toward end state for each state.
        momentum: Transition count per state.
    """
    num_states: int
    kinetic_energy: dict[int, float]
    potential_energy: dict[int, float]
    lagrangian: dict[int, float]
    total_energy: float
    least_action_path: list[int]
    least_action_value: float
    max_action_path: list[int]
    max_action_value: float
    circular_mechanics: list[CircularMechanics]
    gravitational_field: dict[int, float]
    momentum: dict[int, int]


# ---------------------------------------------------------------------------
# Energy functions
# ---------------------------------------------------------------------------

def kinetic_energy(ss: "StateSpace") -> dict[int, float]:
    """Kinetic energy T(x) = |outgoing transitions|² / 2.

    States with many available transitions have high kinetic energy.
    Bottom state (no transitions) has T = 0.
    """
    out_count: dict[int, int] = {s: 0 for s in ss.states}
    for src, _, _ in ss.transitions:
        out_count[src] += 1
    return {s: (c * c) / 2.0 for s, c in out_count.items()}


def potential_energy(ss: "StateSpace") -> dict[int, float]:
    """Potential energy V(x) = rank(x) = distance to bottom.

    The bottom (end) state has V = 0 (ground state).
    Higher states have more potential energy (further from termination).
    """
    rank = compute_rank(ss)
    return {s: float(r) for s, r in rank.items()}


def lagrangian_field(ss: "StateSpace") -> dict[int, float]:
    """Lagrangian L(x) = T(x) - V(x) at each state."""
    T = kinetic_energy(ss)
    V = potential_energy(ss)
    return {s: T[s] - V[s] for s in ss.states}


def hamiltonian_field(ss: "StateSpace") -> dict[int, float]:
    """Hamiltonian H(x) = T(x) + V(x) (total energy at each state)."""
    T = kinetic_energy(ss)
    V = potential_energy(ss)
    return {s: T[s] + V[s] for s in ss.states}


def total_energy(ss: "StateSpace") -> float:
    """Total system energy: Σ H(x) over all states."""
    H = hamiltonian_field(ss)
    return sum(H.values())


# ---------------------------------------------------------------------------
# Gravitational field (attraction toward end)
# ---------------------------------------------------------------------------

def gravitational_field(ss: "StateSpace") -> dict[int, float]:
    """Gravitational pull toward end state.

    g(x) = 1 / distance_to_bottom(x).
    States closer to end feel stronger gravity.
    Bottom state has g = ∞ (represented as a large value).
    """
    rank = compute_rank(ss)
    field: dict[int, float] = {}
    for s in ss.states:
        r = rank[s]
        if r == 0:
            field[s] = 100.0  # At bottom: maximum gravity
        else:
            field[s] = 1.0 / r
    return field


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def momentum(ss: "StateSpace") -> dict[int, int]:
    """Momentum p(x) = number of outgoing transitions.

    In linear (non-branching) protocols, momentum is conserved at 1.
    Branch points have higher momentum (multiple directions).
    Bottom state has momentum 0 (at rest).
    """
    p: dict[int, int] = {s: 0 for s in ss.states}
    for src, _, _ in ss.transitions:
        p[src] += 1
    return p


# ---------------------------------------------------------------------------
# Action and least-action paths
# ---------------------------------------------------------------------------

def path_action(ss: "StateSpace", path: list[int]) -> float:
    """Compute the action S = Σ L(x) along a path.

    The action is the sum of the Lagrangian at each state visited.
    """
    L = lagrangian_field(ss)
    return sum(L.get(s, 0.0) for s in path)


def _enumerate_top_to_bottom_paths(ss: "StateSpace", max_paths: int = 1000) -> list[list[int]]:
    """Enumerate all directed paths from top to bottom."""
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        if tgt not in adj[src]:
            adj[src].append(tgt)

    paths: list[list[int]] = []

    def _dfs(current: int, path: list[int], visited: set[int]) -> None:
        if len(paths) >= max_paths:
            return
        if current == ss.bottom:
            paths.append(path[:])
            return
        if current in visited:
            return
        visited.add(current)
        for nxt in adj[current]:
            path.append(nxt)
            _dfs(nxt, path, visited)
            path.pop()
        visited.discard(current)

    _dfs(ss.top, [ss.top], set())
    return paths


def least_action_path(ss: "StateSpace") -> tuple[list[int], float]:
    """Find the path from top to bottom that minimizes the action.

    Hamilton's principle: the natural execution follows the path
    of least action.
    """
    paths = _enumerate_top_to_bottom_paths(ss)
    if not paths:
        return [ss.top], 0.0

    L = lagrangian_field(ss)
    best_path = paths[0]
    best_action = path_action(ss, paths[0])

    for p in paths[1:]:
        a = path_action(ss, p)
        if a < best_action:
            best_action = a
            best_path = p

    return best_path, best_action


def max_action_path(ss: "StateSpace") -> tuple[list[int], float]:
    """Find the path from top to bottom that maximizes the action.

    This is the "most energetic" execution — the one that visits
    the highest-energy states.
    """
    paths = _enumerate_top_to_bottom_paths(ss)
    if not paths:
        return [ss.top], 0.0

    best_path = paths[0]
    best_action = path_action(ss, paths[0])

    for p in paths[1:]:
        a = path_action(ss, p)
        if a > best_action:
            best_action = a
            best_path = p

    return best_path, best_action


# ---------------------------------------------------------------------------
# Circular mechanics (recursive cycles)
# ---------------------------------------------------------------------------

def analyze_circular(ss: "StateSpace") -> list[CircularMechanics]:
    """Analyze the mechanics of each recursive cycle (SCC with size > 1)."""
    scc_map, scc_members = _compute_sccs(ss)
    adj = _adjacency(ss)
    results: list[CircularMechanics] = []

    for rep, members in scc_members.items():
        if len(members) <= 1:
            continue

        # Period: count internal transitions (edges within SCC)
        internal_edges = 0
        internal_branches = 0
        escape_count = 0

        for s in members:
            out_in_scc = sum(1 for t in adj[s] if scc_map.get(t) == rep)
            out_outside = sum(1 for t in adj[s] if scc_map.get(t) != rep)
            internal_edges += out_in_scc
            escape_count += out_outside
            if out_in_scc > 1:
                internal_branches += 1

        # Period ≈ internal edges (transitions per full orbit)
        period = max(1, internal_edges)
        inertia = len(members)
        angular_vel = 2 * math.pi / period
        centripetal = (period * period) / inertia if inertia > 0 else 0.0

        results.append(CircularMechanics(
            scc_id=rep,
            states=frozenset(members),
            period=period,
            angular_velocity=angular_vel,
            inertia=inertia,
            centripetal_accel=centripetal,
            internal_branches=internal_branches,
            tangential_accel=internal_branches,
            escape_transitions=escape_count,
        ))

    return results


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------

def check_energy_conservation(ss: "StateSpace", path: list[int]) -> list[float]:
    """Check energy conservation along a path.

    Returns the Hamiltonian H(x) at each step. In a deterministic
    (non-branching) path, H should be approximately constant.
    """
    H = hamiltonian_field(ss)
    return [H.get(s, 0.0) for s in path]


def is_energy_conserved(ss: "StateSpace", path: list[int], tolerance: float = 0.5) -> bool:
    """Check if energy is approximately conserved along a path."""
    energies = check_energy_conservation(ss, path)
    if len(energies) < 2:
        return True
    max_e = max(energies)
    min_e = min(energies)
    return (max_e - min_e) <= tolerance


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_lagrangian(ss: "StateSpace") -> LagrangianResult:
    """Complete Lagrangian mechanics analysis."""
    T = kinetic_energy(ss)
    V = potential_energy(ss)
    L = lagrangian_field(ss)
    E = total_energy(ss)
    lap, lav = least_action_path(ss)
    map_, mav = max_action_path(ss)
    circ = analyze_circular(ss)
    gf = gravitational_field(ss)
    p = momentum(ss)

    return LagrangianResult(
        num_states=len(ss.states),
        kinetic_energy=T,
        potential_energy=V,
        lagrangian=L,
        total_energy=E,
        least_action_path=lap,
        least_action_value=lav,
        max_action_path=map_,
        max_action_value=mav,
        circular_mechanics=circ,
        gravitational_field=gf,
        momentum=p,
    )
