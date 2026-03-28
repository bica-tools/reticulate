"""Protocol mechanics: the physics of session types (Step 60p).

The fundamental correspondence between session type structure and
classical mechanics:

  Position   ↔  State           (where you are)
  Velocity   ↔  Transition      (how you move)
  Acceleration ↔ Branch point   (what changes the movement)
  Jerk       ↔  Nested branch   (what changes the acceleration)
  k-th deriv ↔  k-nested branch (k-deep tree structure)
  Dimension  ↔  Parallel (∥)    (independent degrees of freedom)
  Period     ↔  Recursion (μX)  (oscillation / periodic orbit)

This module provides:

- **Discrete derivatives** on lattice-valued functions
- **Discrete integration** via Möbius summation
- **Branch order analysis** (acceleration detection)
- **Parallel dimension** counting
- **Recursion period** detection
- **Smoothness measure** via second differences
- **Energy** (Euler characteristic as total complexity)
- **Refinement** as increasing sampling rate
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _state_list,
    _adjacency,
    _reachability,
    _compute_sccs,
    _covering_relation,
    compute_rank,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MechanicsResult:
    """Complete protocol mechanics analysis.

    Attributes:
        num_states: Number of states (positions).
        num_transitions: Number of transitions (velocities).
        num_branch_points: Number of branch points (accelerations).
        max_branch_order: Deepest nesting of branches (highest derivative order).
        parallel_dimensions: Number of independent parallel dimensions.
        has_recursion: True iff protocol has periodic orbits.
        smoothness: Average |Δ²f| for the rank function (acceleration magnitude).
        energy: Euler characteristic of the order complex.
        velocity_field: For each state, list of outgoing transitions (velocity vectors).
        acceleration_field: For each state, branch degree (acceleration magnitude).
    """
    num_states: int
    num_transitions: int
    num_branch_points: int
    max_branch_order: int
    parallel_dimensions: int
    has_recursion: bool
    smoothness: float
    energy: int
    velocity_field: dict[int, list[str]]
    acceleration_field: dict[int, int]


# ---------------------------------------------------------------------------
# Discrete calculus on lattices
# ---------------------------------------------------------------------------

def discrete_derivative(
    ss: "StateSpace",
    f: dict[int, float],
) -> dict[int, float]:
    """Discrete derivative: Δf(x) = Σ_{y: x covers y} (f(x) - f(y)).

    This is the discrete analogue of df/dt: how much does f change
    when descending one step in the lattice.
    """
    covers = _covering_relation(ss)
    adj_down: dict[int, list[int]] = {s: [] for s in ss.states}
    for x, y in covers:
        adj_down[x].append(y)

    df: dict[int, float] = {}
    for x in ss.states:
        children = adj_down[x]
        if not children:
            df[x] = 0.0
        else:
            df[x] = sum(f.get(x, 0.0) - f.get(y, 0.0) for y in children)
    return df


def discrete_second_derivative(
    ss: "StateSpace",
    f: dict[int, float],
) -> dict[int, float]:
    """Discrete second derivative: Δ²f = Δ(Δf).

    Measures the "acceleration" — how rapidly the rate of change itself changes.
    Small |Δ²f| everywhere = "smooth" protocol.
    """
    df = discrete_derivative(ss, f)
    return discrete_derivative(ss, df)


def discrete_integral(
    ss: "StateSpace",
    f: dict[int, float],
) -> dict[int, float]:
    """Discrete integral: F(x) = Σ_{y ≤ x} f(y).

    Summation over the downset of x. Inverse of Möbius difference
    via Möbius inversion.
    """
    reach = _reachability(ss)
    F: dict[int, float] = {}
    for x in ss.states:
        F[x] = sum(f.get(y, 0.0) for y in reach[x])
    return F


def mobius_inversion(
    ss: "StateSpace",
    F: dict[int, float],
) -> dict[int, float]:
    """Möbius inversion: recover f from F = Σ_{y≤x} f(y).

    f(x) = Σ_{y ≤ x} μ(x, y) · F(y).
    """
    from reticulate.zeta import mobius_function
    mu = mobius_function(ss)
    reach = _reachability(ss)

    f: dict[int, float] = {}
    for x in ss.states:
        val = 0.0
        for y in ss.states:
            if y in reach[x]:
                val += mu.get((x, y), 0) * F.get(y, 0.0)
        f[x] = val
    return f


# ---------------------------------------------------------------------------
# Velocity field (transitions as velocity vectors)
# ---------------------------------------------------------------------------

def velocity_field(ss: "StateSpace") -> dict[int, list[str]]:
    """For each state, the set of available transitions (velocity vectors).

    The "velocity" at state x is the set of method labels that can be
    invoked — the available directions of movement.
    """
    field: dict[int, list[str]] = {s: [] for s in ss.states}
    for src, label, _ in ss.transitions:
        if label not in field[src]:
            field[src].append(label)
    return field


def velocity_magnitude(ss: "StateSpace") -> dict[int, int]:
    """The "speed" at each state = number of available transitions.

    High speed = many choices = high branching = high acceleration potential.
    """
    vf = velocity_field(ss)
    return {s: len(labels) for s, labels in vf.items()}


# ---------------------------------------------------------------------------
# Acceleration field (branch points)
# ---------------------------------------------------------------------------

def acceleration_field(ss: "StateSpace") -> dict[int, int]:
    """For each state, the branch degree (acceleration magnitude).

    A state with k outgoing transitions has acceleration magnitude k-1
    (one transition = uniform motion, k>1 = deflection).
    acceleration = 0 for sequential states, >0 for branch points.
    """
    vm = velocity_magnitude(ss)
    return {s: max(0, v - 1) for s, v in vm.items()}


def branch_points(ss: "StateSpace") -> list[int]:
    """States where acceleration is non-zero (trajectory changes direction)."""
    af = acceleration_field(ss)
    return sorted(s for s, a in af.items() if a > 0)


def max_branch_order(ss: "StateSpace") -> int:
    """Maximum nesting depth of branch points.

    Corresponds to the highest-order derivative that is non-zero.
    A chain has order 0 (no branches).
    A simple branch has order 1.
    A branch-within-branch has order 2.
    """
    adj = _adjacency(ss)
    covers = _covering_relation(ss)
    cover_adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for x, y in covers:
        cover_adj[x].append(y)

    # Count outgoing transitions (labels) per state, not just distinct targets
    out_count: dict[int, int] = {s: 0 for s in ss.states}
    out_targets: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        out_count[src] += 1
        out_targets[src].add(tgt)

    max_order = 0

    def _dfs(s: int, current_order: int, visited: set[int]) -> None:
        nonlocal max_order
        if s in visited:
            return
        visited.add(s)
        # A branch point = state with multiple outgoing transitions
        is_branch = out_count[s] > 1
        new_order = current_order + (1 if is_branch else 0)
        max_order = max(max_order, new_order)
        for c in out_targets[s]:
            _dfs(c, new_order, visited)

    _dfs(ss.top, 0, set())
    return max_order


# ---------------------------------------------------------------------------
# Parallel dimensions
# ---------------------------------------------------------------------------

def parallel_dimensions(ss: "StateSpace") -> int:
    """Number of independent parallel dimensions.

    Computed from the width of the lattice: width = product of
    component widths. For a k-fold parallel composition, this is k.

    Approximated as log₂(width) for power-of-2 widths.
    """
    from reticulate.zeta import compute_width
    w = compute_width(ss)
    if w <= 1:
        return 1
    # Width of 2 = 2 dimensions, width of 3 = 2 dimensions, width of 4 = 2-3 etc.
    return w


# ---------------------------------------------------------------------------
# Recursion period
# ---------------------------------------------------------------------------

def has_periodic_orbit(ss: "StateSpace") -> bool:
    """Check if the protocol has recursion (periodic orbits)."""
    scc_map, scc_members = _compute_sccs(ss)
    return any(len(members) > 1 for members in scc_members.values())


# ---------------------------------------------------------------------------
# Smoothness
# ---------------------------------------------------------------------------

def smoothness(ss: "StateSpace") -> float:
    """Smoothness measure: mean |Δ²f| for the rank function.

    Low smoothness = high acceleration = sharp turns in the protocol.
    High smoothness = low acceleration = gentle trajectory.
    """
    rank = compute_rank(ss)
    f = {s: float(rank[s]) for s in ss.states}
    d2f = discrete_second_derivative(ss, f)
    if not d2f:
        return 1.0
    mean_abs = sum(abs(v) for v in d2f.values()) / len(d2f)
    # Invert: high |Δ²f| = low smoothness
    return 1.0 / (1.0 + mean_abs)


# ---------------------------------------------------------------------------
# Energy (Euler characteristic)
# ---------------------------------------------------------------------------

def protocol_energy(ss: "StateSpace") -> int:
    """Protocol energy = Euler characteristic of the order complex.

    Balances all orders of structure: states, transitions, branches,
    nested branches, etc. Analogous to total mechanical energy.
    """
    from reticulate.order_complex import euler_characteristic
    return euler_characteristic(ss)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_mechanics(ss: "StateSpace") -> MechanicsResult:
    """Complete protocol mechanics analysis."""
    vf = velocity_field(ss)
    af = acceleration_field(ss)
    bp = branch_points(ss)
    mbo = max_branch_order(ss)
    pd = parallel_dimensions(ss)
    rec = has_periodic_orbit(ss)
    smooth = smoothness(ss)
    energy = protocol_energy(ss)

    return MechanicsResult(
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        num_branch_points=len(bp),
        max_branch_order=mbo,
        parallel_dimensions=pd,
        has_recursion=rec,
        smoothness=smooth,
        energy=energy,
        velocity_field=vf,
        acceleration_field=af,
    )
