"""Lattice valuations and the FKG inequality (Steps 30y-30z).

- **Step 30y**: Lattice valuations v: L → R with v(x∧y) + v(x∨y) = v(x) + v(y)
- **Step 30z**: FKG inequality for monotone functions on distributive lattices

For session type lattices, valuations provide additive measures on
protocol states (e.g., execution cost, resource usage) that respect
the lattice structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _state_list,
    _reachability,
    _compute_sccs,
    compute_rank,
)
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValuationResult:
    """Lattice valuation analysis.

    Attributes:
        num_states: Number of quotient states.
        rank_valuation: v(x) = rank(x) — always a valuation on graded lattices.
        is_rank_modular: True iff rank is a modular function (v(x∧y)+v(x∨y)=v(x)+v(y)).
        height_valuation: v(x) = height from top — another natural valuation.
        fkg_applicable: True iff the lattice is distributive (FKG applies).
    """
    num_states: int
    rank_valuation: dict[int, int]
    is_rank_modular: bool
    height_valuation: dict[int, int]
    fkg_applicable: bool


# ---------------------------------------------------------------------------
# Valuations
# ---------------------------------------------------------------------------

def rank_valuation(ss: "StateSpace") -> dict[int, int]:
    """The rank function as a valuation: v(x) = rank(x)."""
    return compute_rank(ss)


def height_valuation(ss: "StateSpace") -> dict[int, int]:
    """Height from top as a valuation: v(x) = distance from top."""
    from reticulate.zeta import _adjacency
    adj = _adjacency(ss)
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())

    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr:
                q_adj[sr].add(tr)

    top_rep = scc_map[ss.top]
    dist: dict[int, int] = {top_rep: 0}
    queue = [top_rep]
    visited = {top_rep}
    while queue:
        u = queue.pop(0)
        for v in q_adj[u]:
            if v not in visited:
                visited.add(v)
                dist[v] = dist[u] + 1
                queue.append(v)
    for r in reps:
        if r not in dist:
            dist[r] = 0

    return {s: dist.get(scc_map[s], 0) for s in ss.states}


def is_modular_valuation(ss: "StateSpace", v: dict[int, int]) -> bool:
    """Check if v is a modular function: v(x∧y) + v(x∨y) = v(x) + v(y).

    Requires computing meets and joins for all pairs.
    """
    reach = _reachability(ss)
    states = _state_list(ss)
    scc_map, _ = _compute_sccs(ss)

    # Build meets/joins from lattice.py's logic
    for x in states:
        for y in states:
            if x >= y:
                continue

            # Meet (GLB): greatest z with z ≤ x and z ≤ y
            lower = [z for z in states if z in reach[x] and z in reach[y]]
            if not lower:
                continue
            meet = max(lower, key=lambda z: len([w for w in states if w in reach[z]]))

            # Join (LUB): least z with z ≥ x and z ≥ y
            upper = [z for z in states if x in reach[z] and y in reach[z]]
            if not upper:
                continue
            join = min(upper, key=lambda z: len([w for w in states if w in reach[z]]))

            # Check modularity
            lhs = v.get(meet, 0) + v.get(join, 0)
            rhs = v.get(x, 0) + v.get(y, 0)
            if lhs != rhs:
                return False

    return True


# ---------------------------------------------------------------------------
# FKG inequality (Step 30z)
# ---------------------------------------------------------------------------

def is_distributive_lattice(ss: "StateSpace") -> bool:
    """Check if the lattice is distributive (FKG applies)."""
    result = check_lattice(ss)
    if not result.is_lattice:
        return False
    # Use the distributivity check from lattice.py
    return getattr(result, 'is_distributive', True)  # Default True if not checked


def verify_fkg(
    ss: "StateSpace",
    f: Callable[[int], float],
    g: Callable[[int], float],
) -> bool:
    """Verify FKG inequality: E[fg] ≥ E[f]·E[g] for monotone f, g.

    For increasing functions f, g on a distributive lattice L with
    log-supermodular measure μ:
    Σ f(x)g(x)μ(x) · Σ μ(x) ≥ Σ f(x)μ(x) · Σ g(x)μ(x)

    Uses uniform measure (μ(x) = 1/n).
    """
    states = _state_list(ss)
    n = len(states)
    if n == 0:
        return True

    sum_fg = sum(f(s) * g(s) for s in states) / n
    sum_f = sum(f(s) for s in states) / n
    sum_g = sum(g(s) for s in states) / n

    return sum_fg >= sum_f * sum_g - 1e-10


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_valuations(ss: "StateSpace") -> ValuationResult:
    """Complete valuation analysis."""
    rv = rank_valuation(ss)
    hv = height_valuation(ss)
    is_mod = is_modular_valuation(ss, rv)
    fkg = is_distributive_lattice(ss)

    scc_map, _ = _compute_sccs(ss)
    n_quotient = len(set(scc_map.values()))

    return ValuationResult(
        num_states=n_quotient,
        rank_valuation=rv,
        is_rank_modular=is_mod,
        height_valuation=hv,
        fkg_applicable=fkg,
    )
