"""Laplacian optimization for protocol connectivity (Step 31h).

Gradient flow on the Laplacian: optimize protocol structure for better
connectivity by suggesting edges (transitions) to add.  The Fiedler
value (algebraic connectivity) is the objective; we compute its gradient
with respect to edge additions and propose improvements.

Main API:
  - ``laplacian_gradient(ss)``         gradient of Fiedler value w.r.t. edges
  - ``optimize_connectivity(ss, k)``   suggest k best edges to add
  - ``suggest_improvements(ss)``       full improvement analysis
  - ``connectivity_score(ss)``         normalized connectivity metric
  - ``bottleneck_edges(ss)``           edges whose removal hurts most
  - ``ImprovementResult``              aggregate result
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
class EdgeCandidate:
    """A candidate edge to add for connectivity improvement.

    Attributes:
        src: Source state ID.
        tgt: Target state ID.
        fiedler_gain: Increase in Fiedler value if this edge is added.
        relative_gain: Fiedler gain as fraction of current Fiedler value.
    """
    src: int
    tgt: int
    fiedler_gain: float
    relative_gain: float


@dataclass(frozen=True)
class BottleneckEdge:
    """An edge whose removal most degrades connectivity.

    Attributes:
        src: Source state.
        tgt: Target state.
        fiedler_loss: Decrease in Fiedler value if this edge is removed.
        relative_loss: Loss as fraction of current Fiedler value.
        is_bridge: Whether removal disconnects the graph.
    """
    src: int
    tgt: int
    fiedler_loss: float
    relative_loss: float
    is_bridge: bool


@dataclass(frozen=True)
class ImprovementResult:
    """Complete Laplacian optimization analysis.

    Attributes:
        num_states: Number of states.
        num_edges: Number of current Hasse edges.
        fiedler_current: Current Fiedler value.
        connectivity_score: Normalized connectivity (0 to 1).
        candidates: Top edge candidates for addition, sorted by gain.
        bottlenecks: Most critical edges, sorted by loss.
        fiedler_after_best: Fiedler value after adding the best candidate.
        max_possible_fiedler: Upper bound on achievable Fiedler value.
        improvement_potential: (max_possible - current) / max_possible.
    """
    num_states: int
    num_edges: int
    fiedler_current: float
    connectivity_score: float
    candidates: list[EdgeCandidate]
    bottlenecks: list[BottleneckEdge]
    fiedler_after_best: float
    max_possible_fiedler: float
    improvement_potential: float


# ---------------------------------------------------------------------------
# Hasse graph utilities
# ---------------------------------------------------------------------------

def _hasse_edges(ss: "StateSpace") -> list[tuple[int, int]]:
    """Compute covering relation edges."""
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


def _undirected_edge_set(ss: "StateSpace") -> set[tuple[int, int]]:
    """Get undirected edge set from Hasse edges (normalized: smaller first)."""
    result: set[tuple[int, int]] = set()
    for a, b in _hasse_edges(ss):
        result.add((min(a, b), max(a, b)))
    return result


def _laplacian_from_edges(
    states: list[int], edges: list[tuple[int, int]],
) -> list[list[float]]:
    """Build Laplacian from undirected edge list."""
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    L = [[0.0] * n for _ in range(n)]

    for a, b in edges:
        i, j = idx[a], idx[b]
        L[i][i] += 1.0
        L[j][j] += 1.0
        L[i][j] -= 1.0
        L[j][i] -= 1.0

    return L


def _eigenvalues_symmetric(A: list[list[float]]) -> list[float]:
    """Compute eigenvalues of a symmetric matrix."""
    from reticulate.matrix import _eigenvalues_symmetric
    return _eigenvalues_symmetric(A)


def _fiedler_from_edges(states: list[int], edges: list[tuple[int, int]]) -> float:
    """Compute Fiedler value from edge list."""
    L = _laplacian_from_edges(states, edges)
    n = len(L)
    if n <= 1:
        return 0.0
    eigs = sorted(_eigenvalues_symmetric(L))
    if len(eigs) < 2:
        return 0.0
    return max(0.0, eigs[1])


def _fiedler_vector(states: list[int], edges: list[tuple[int, int]]) -> list[float]:
    """Compute the Fiedler vector (eigenvector for lambda_1).

    Uses power iteration on (L_max * I - L) to find the second-smallest
    eigenvector.
    """
    n = len(states)
    if n <= 1:
        return [0.0] * n

    L = _laplacian_from_edges(states, edges)
    eigs = sorted(_eigenvalues_symmetric(L))
    lambda_max = eigs[-1] if eigs else 1.0

    # Shifted matrix: M = lambda_max * I - L has largest eigenvalue
    # corresponding to smallest eigenvalue of L
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = -L[i][j]
        M[i][i] += lambda_max

    # Power iteration to find dominant eigenvector of M (= Fiedler vector of L)
    # But dominant eigenvector of M is the all-ones vector (lambda_0 of L).
    # We need the second eigenvector, so deflate.

    # Start with a random-ish vector orthogonal to all-ones
    v = [float(i) - (n - 1) / 2.0 for i in range(n)]
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-15:
        return [0.0] * n
    v = [x / norm for x in v]

    for _ in range(200):
        # Multiply by M
        w = [0.0] * n
        for i in range(n):
            for j in range(n):
                w[i] += M[i][j] * v[j]

        # Project out the all-ones direction
        avg = sum(w) / n
        w = [x - avg for x in w]

        # Normalize
        norm = math.sqrt(sum(x * x for x in w))
        if norm < 1e-15:
            break
        v = [x / norm for x in w]

    return v


# ---------------------------------------------------------------------------
# Laplacian gradient
# ---------------------------------------------------------------------------

def laplacian_gradient(ss: "StateSpace") -> dict[tuple[int, int], float]:
    """Compute the gradient of the Fiedler value with respect to edge additions.

    For each non-existing edge (u, v), the gain in Fiedler value from
    adding it is approximately (f_u - f_v)^2 where f is the Fiedler vector.

    This is the first-order perturbation theory result for symmetric
    eigenvalues: d(lambda_1)/d(w_e) = (f_u - f_v)^2 for edge e = (u,v).

    Args:
        ss: A session type state space.

    Returns:
        Dict mapping (u, v) pairs (normalized: smaller first) to gradient values.
    """
    states = sorted(ss.states)
    n = len(states)
    if n <= 2:
        return {}

    edges = _hasse_edges(ss)
    existing = _undirected_edge_set(ss)

    f = _fiedler_vector(states, edges)
    idx = {s: i for i, s in enumerate(states)}

    gradient: dict[tuple[int, int], float] = {}
    for i, s in enumerate(states):
        for j, t in enumerate(states):
            if i >= j:
                continue
            edge = (min(s, t), max(s, t))
            if edge in existing:
                continue
            grad = (f[idx[s]] - f[idx[t]]) ** 2
            gradient[edge] = grad

    return gradient


# ---------------------------------------------------------------------------
# Optimize connectivity
# ---------------------------------------------------------------------------

def optimize_connectivity(
    ss: "StateSpace", k: int = 3,
) -> list[EdgeCandidate]:
    """Suggest the k best edges to add for connectivity improvement.

    Uses the Fiedler vector gradient to rank candidate edges, then
    verifies the actual Fiedler gain by recomputation.

    Args:
        ss: A session type state space.
        k: Number of candidates to return.

    Returns:
        List of EdgeCandidate sorted by fiedler_gain (descending).
    """
    states = sorted(ss.states)
    n = len(states)
    if n <= 2:
        return []

    edges = _hasse_edges(ss)
    fiedler_current = _fiedler_from_edges(states, edges)

    grad = laplacian_gradient(ss)
    if not grad:
        return []

    # Sort by gradient (descending)
    sorted_candidates = sorted(grad.items(), key=lambda x: -x[1])

    results: list[EdgeCandidate] = []
    for (u, v), _ in sorted_candidates[:k * 2]:
        # Verify actual gain
        new_edges = edges + [(u, v)]
        fiedler_new = _fiedler_from_edges(states, new_edges)
        gain = fiedler_new - fiedler_current
        rel_gain = gain / fiedler_current if fiedler_current > 1e-10 else gain

        results.append(EdgeCandidate(
            src=u, tgt=v,
            fiedler_gain=gain,
            relative_gain=rel_gain,
        ))

    # Sort by actual gain and return top k
    results.sort(key=lambda c: -c.fiedler_gain)
    return results[:k]


# ---------------------------------------------------------------------------
# Connectivity score
# ---------------------------------------------------------------------------

def connectivity_score(ss: "StateSpace") -> float:
    """Compute a normalized connectivity score in [0, 1].

    The score is the Fiedler value divided by the maximum possible
    Fiedler value for a graph on n vertices (which is n for K_n).

    Args:
        ss: A session type state space.

    Returns:
        Normalized connectivity score.
    """
    n = len(ss.states)
    if n <= 1:
        return 1.0

    from reticulate.matrix import fiedler_value
    fiedler = fiedler_value(ss)

    # Max Fiedler for K_n is n
    max_fiedler = float(n)

    return min(1.0, fiedler / max_fiedler)


# ---------------------------------------------------------------------------
# Bottleneck edges
# ---------------------------------------------------------------------------

def bottleneck_edges(ss: "StateSpace", k: int = 3) -> list[BottleneckEdge]:
    """Find the k edges whose removal most degrades connectivity.

    For each edge, computes the Fiedler value of the graph without it.
    Edges causing the largest Fiedler drop are bottlenecks.

    Args:
        ss: A session type state space.
        k: Number of bottlenecks to return.

    Returns:
        List of BottleneckEdge sorted by fiedler_loss (descending).
    """
    states = sorted(ss.states)
    n = len(states)
    if n <= 2:
        return []

    edges = _hasse_edges(ss)
    fiedler_current = _fiedler_from_edges(states, edges)

    results: list[BottleneckEdge] = []
    for idx_e, (a, b) in enumerate(edges):
        # Remove this edge
        remaining = [e for i, e in enumerate(edges) if i != idx_e]
        fiedler_without = _fiedler_from_edges(states, remaining)

        loss = fiedler_current - fiedler_without
        rel_loss = loss / fiedler_current if fiedler_current > 1e-10 else loss
        is_bridge = fiedler_without < 1e-10

        results.append(BottleneckEdge(
            src=a, tgt=b,
            fiedler_loss=loss,
            relative_loss=rel_loss,
            is_bridge=is_bridge,
        ))

    results.sort(key=lambda b: -b.fiedler_loss)
    return results[:k]


# ---------------------------------------------------------------------------
# Full improvement analysis
# ---------------------------------------------------------------------------

def suggest_improvements(
    ss: "StateSpace", num_candidates: int = 5, num_bottlenecks: int = 5,
) -> ImprovementResult:
    """Complete Laplacian optimization analysis.

    Computes current connectivity, suggests edges to add, identifies
    bottlenecks, and estimates improvement potential.

    Args:
        ss: A session type state space.
        num_candidates: Number of edge candidates to suggest.
        num_bottlenecks: Number of bottleneck edges to identify.

    Returns:
        ImprovementResult with all metrics.
    """
    states = sorted(ss.states)
    n = len(states)
    edges = _hasse_edges(ss)
    m = len(edges)

    from reticulate.matrix import fiedler_value
    fiedler_current = fiedler_value(ss)

    score = connectivity_score(ss)

    candidates = optimize_connectivity(ss, num_candidates)
    bottlenecks = bottleneck_edges(ss, num_bottlenecks)

    # Fiedler after adding best candidate
    if candidates:
        best = candidates[0]
        fiedler_after = fiedler_current + best.fiedler_gain
    else:
        fiedler_after = fiedler_current

    # Max possible Fiedler (complete graph K_n has Fiedler = n)
    max_fiedler = float(n) if n > 1 else 0.0

    potential = (max_fiedler - fiedler_current) / max_fiedler if max_fiedler > 1e-10 else 0.0

    return ImprovementResult(
        num_states=n,
        num_edges=m,
        fiedler_current=fiedler_current,
        connectivity_score=score,
        candidates=candidates,
        bottlenecks=bottlenecks,
        fiedler_after_best=fiedler_after,
        max_possible_fiedler=max_fiedler,
        improvement_potential=potential,
    )
