"""Ramanujan optimality for session type lattice graphs (Step 31f).

Checks whether session type lattice Hasse diagrams are Ramanujan graphs,
i.e., whether their spectral radius satisfies the Alon-Boppana bound
|lambda| <= 2*sqrt(d-1) for d-regular graphs.  For irregular graphs we
use the generalized notion: the largest non-trivial eigenvalue of the
adjacency matrix is at most 2*sqrt(d_max - 1).

Ramanujan graphs are optimal expanders: they achieve the best possible
spectral gap for their degree.  If a session type lattice is Ramanujan,
its expansion (and hence error correction capacity from Step 31e) is
provably optimal.

Main API:
  - ``adjacency_eigenvalues(ss)``     sorted eigenvalues of Hasse adjacency
  - ``degree_stats(ss)``              min/max/avg degree of Hasse diagram
  - ``alon_boppana_bound(d)``         the theoretical bound 2*sqrt(d-1)
  - ``is_ramanujan(ss)``              check the Ramanujan property
  - ``ramanujan_gap(ss)``             gap between actual and Ramanujan bound
  - ``optimal_expansion_check(ss)``   full optimality analysis
  - ``ramanujan_ratio(ss)``           how close to Ramanujan (0=exact, >0=worse)
  - ``RamanujanResult``               aggregate analysis result
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DegreeStats:
    """Degree statistics of the undirected Hasse diagram.

    Attributes:
        min_degree: Minimum vertex degree.
        max_degree: Maximum vertex degree.
        avg_degree: Average vertex degree.
        is_regular: Whether all vertices have the same degree.
        degree_sequence: Sorted list of degrees.
    """
    min_degree: int
    max_degree: int
    avg_degree: float
    is_regular: bool
    degree_sequence: list[int]


@dataclass(frozen=True)
class RamanujanResult:
    """Complete Ramanujan optimality analysis.

    Attributes:
        num_states: Number of states.
        num_edges: Number of undirected Hasse edges.
        degree_stats: Degree statistics.
        eigenvalues: Sorted adjacency eigenvalues.
        spectral_radius: Largest absolute eigenvalue.
        nontrivial_radius: Largest absolute non-trivial eigenvalue.
        alon_boppana: The Alon-Boppana bound 2*sqrt(d_max - 1).
        is_ramanujan: Whether the graph satisfies the bound.
        ramanujan_gap: nontrivial_radius - alon_boppana (negative = Ramanujan).
        ramanujan_ratio: nontrivial_radius / alon_boppana (< 1 = Ramanujan).
        optimal_expansion: Whether expansion is provably optimal.
    """
    num_states: int
    num_edges: int
    degree_stats: DegreeStats
    eigenvalues: list[float]
    spectral_radius: float
    nontrivial_radius: float
    alon_boppana: float
    is_ramanujan: bool
    ramanujan_gap: float
    ramanujan_ratio: float
    optimal_expansion: bool


# ---------------------------------------------------------------------------
# Hasse graph utilities
# ---------------------------------------------------------------------------

def _hasse_edges(ss: "StateSpace") -> list[tuple[int, int]]:
    """Compute covering relation edges of the Hasse diagram."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    edges: list[tuple[int, int]] = []
    for a in ss.states:
        direct = adj[a]
        for b in direct:
            is_cover = True
            for c in direct:
                if c == b:
                    continue
                visited: set[int] = set()
                stack = [c]
                while stack:
                    u = stack.pop()
                    if u == b:
                        is_cover = False
                        break
                    if u in visited:
                        continue
                    visited.add(u)
                    for v in adj[u]:
                        if v not in visited:
                            stack.append(v)
                if not is_cover:
                    break
            if is_cover:
                edges.append((a, b))
    return edges


def _undirected_adj(ss: "StateSpace") -> dict[int, set[int]]:
    """Build undirected adjacency from Hasse edges."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for a, b in _hasse_edges(ss):
        adj[a].add(b)
        adj[b].add(a)
    return adj


# ---------------------------------------------------------------------------
# Degree statistics
# ---------------------------------------------------------------------------

def degree_stats(ss: "StateSpace") -> DegreeStats:
    """Compute degree statistics of the undirected Hasse diagram.

    Args:
        ss: A session type state space.

    Returns:
        DegreeStats with min, max, avg degree and regularity check.
    """
    adj = _undirected_adj(ss)
    degrees = sorted(len(adj[s]) for s in sorted(ss.states))

    if not degrees:
        return DegreeStats(
            min_degree=0, max_degree=0, avg_degree=0.0,
            is_regular=True, degree_sequence=[],
        )

    min_d = degrees[0]
    max_d = degrees[-1]
    avg_d = sum(degrees) / len(degrees)
    regular = min_d == max_d

    return DegreeStats(
        min_degree=min_d,
        max_degree=max_d,
        avg_degree=avg_d,
        is_regular=regular,
        degree_sequence=degrees,
    )


# ---------------------------------------------------------------------------
# Adjacency eigenvalues
# ---------------------------------------------------------------------------

def adjacency_eigenvalues(ss: "StateSpace") -> list[float]:
    """Compute sorted eigenvalues of the undirected Hasse adjacency matrix.

    Args:
        ss: A session type state space.

    Returns:
        Sorted list of real eigenvalues.
    """
    from reticulate.matrix import adjacency_spectrum
    return adjacency_spectrum(ss)


# ---------------------------------------------------------------------------
# Alon-Boppana bound
# ---------------------------------------------------------------------------

def alon_boppana_bound(d: int) -> float:
    """Compute the Alon-Boppana bound for a d-regular graph.

    For a d-regular graph, the second-largest eigenvalue satisfies
    lambda_2 >= 2*sqrt(d-1) - o(1) as n -> infinity.
    A graph meeting this bound with equality is Ramanujan.

    Args:
        d: The degree (must be >= 1).

    Returns:
        The bound 2*sqrt(d-1), or 0.0 if d <= 1.
    """
    if d <= 1:
        return 0.0
    return 2.0 * math.sqrt(d - 1)


# ---------------------------------------------------------------------------
# Ramanujan check
# ---------------------------------------------------------------------------

def _nontrivial_radius(eigenvalues: list[float], d_max: int) -> float:
    """Compute the largest absolute non-trivial eigenvalue.

    For a d-regular graph, the trivial eigenvalue is d itself.
    For irregular graphs, we exclude the largest eigenvalue.

    Args:
        eigenvalues: Sorted eigenvalues.
        d_max: Maximum degree.

    Returns:
        Largest |lambda| among non-trivial eigenvalues.
    """
    if len(eigenvalues) <= 1:
        return 0.0

    # The largest eigenvalue is trivial (equals spectral radius)
    # Non-trivial = all others
    nontrivial = eigenvalues[:-1]  # exclude the largest
    if not nontrivial:
        return 0.0

    return max(abs(e) for e in nontrivial)


def is_ramanujan(ss: "StateSpace") -> bool:
    """Check whether the Hasse diagram satisfies the Ramanujan property.

    A graph is Ramanujan if every non-trivial eigenvalue lambda satisfies
    |lambda| <= 2*sqrt(d_max - 1).

    For trivial graphs (0 or 1 states, or single edge), returns True.

    Args:
        ss: A session type state space.

    Returns:
        True if the graph is Ramanujan.
    """
    n = len(ss.states)
    if n <= 2:
        return True

    eigs = adjacency_eigenvalues(ss)
    stats = degree_stats(ss)
    d_max = stats.max_degree

    if d_max <= 1:
        return True

    bound = alon_boppana_bound(d_max)
    ntr = _nontrivial_radius(eigs, d_max)

    # Allow small numerical tolerance
    return ntr <= bound + 1e-9


def ramanujan_gap(ss: "StateSpace") -> float:
    """Compute the gap between actual nontrivial radius and Ramanujan bound.

    gap = nontrivial_radius - 2*sqrt(d_max - 1)

    Negative values mean the graph is Ramanujan (better than bound).
    Zero means exactly Ramanujan.
    Positive means not Ramanujan.

    Args:
        ss: A session type state space.

    Returns:
        The Ramanujan gap (float).
    """
    n = len(ss.states)
    if n <= 2:
        return 0.0

    eigs = adjacency_eigenvalues(ss)
    stats = degree_stats(ss)
    d_max = stats.max_degree

    if d_max <= 1:
        return 0.0

    bound = alon_boppana_bound(d_max)
    ntr = _nontrivial_radius(eigs, d_max)

    return ntr - bound


def ramanujan_ratio(ss: "StateSpace") -> float:
    """Compute how close the graph is to Ramanujan optimality.

    ratio = nontrivial_radius / alon_boppana_bound

    ratio < 1.0  =>  Ramanujan (better than bound)
    ratio = 1.0  =>  exactly at bound
    ratio > 1.0  =>  not Ramanujan

    Args:
        ss: A session type state space.

    Returns:
        The ratio (0.0 if bound is 0).
    """
    n = len(ss.states)
    if n <= 2:
        return 0.0

    eigs = adjacency_eigenvalues(ss)
    stats = degree_stats(ss)
    d_max = stats.max_degree

    bound = alon_boppana_bound(d_max)
    if bound < 1e-15:
        return 0.0

    ntr = _nontrivial_radius(eigs, d_max)
    return ntr / bound


# ---------------------------------------------------------------------------
# Optimal expansion check
# ---------------------------------------------------------------------------

def optimal_expansion_check(ss: "StateSpace") -> RamanujanResult:
    """Full Ramanujan optimality analysis.

    Computes degree statistics, adjacency eigenvalues, the Alon-Boppana
    bound, and determines whether the lattice achieves optimal expansion.

    Args:
        ss: A session type state space.

    Returns:
        RamanujanResult with all metrics.
    """
    stats = degree_stats(ss)
    eigs = adjacency_eigenvalues(ss)
    n = len(ss.states)
    edges = _hasse_edges(ss)
    n_edges = len(edges)

    sp_radius = max(abs(e) for e in eigs) if eigs else 0.0
    ntr = _nontrivial_radius(eigs, stats.max_degree)
    bound = alon_boppana_bound(stats.max_degree)

    is_ram = ntr <= bound + 1e-9 if n > 2 and stats.max_degree > 1 else True
    gap = ntr - bound if n > 2 and stats.max_degree > 1 else 0.0
    ratio = ntr / bound if bound > 1e-15 else 0.0

    # Optimal expansion: Ramanujan AND connected
    from reticulate.matrix import fiedler_value
    fiedler = fiedler_value(ss)
    optimal = is_ram and fiedler > 1e-9

    return RamanujanResult(
        num_states=n,
        num_edges=n_edges,
        degree_stats=stats,
        eigenvalues=eigs,
        spectral_radius=sp_radius,
        nontrivial_radius=ntr,
        alon_boppana=bound,
        is_ramanujan=is_ram,
        ramanujan_gap=gap,
        ramanujan_ratio=ratio,
        optimal_expansion=optimal,
    )
