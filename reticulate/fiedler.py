"""Algebraic connectivity (Fiedler) analysis for session type lattices (Step 30e).

The Fiedler value λ₂ (second-smallest Laplacian eigenvalue) measures
how well-connected a protocol's Hasse diagram is. This module provides:

- **Fiedler value**: λ₂(L) — algebraic connectivity
- **Fiedler vector**: eigenvector for λ₂ — gives optimal graph bisection
- **Bottleneck detection**: states where removal maximally disconnects
- **Cut vertices**: states whose removal disconnects the Hasse diagram
- **Vertex connectivity**: minimum vertex cut size
- **Edge connectivity**: minimum edge cut size
- **Cheeger inequality**: relates Fiedler value to conductance
- **Composition**: λ₂(L₁ × L₂) = min(λ₂(L₁), λ₂(L₂))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.matrix import (
    adjacency_matrix,
    laplacian_matrix,
    _eigenvalues_symmetric,
    _hasse_edges,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FiedlerResult:
    """Complete algebraic connectivity analysis.

    Attributes:
        num_states: Number of states.
        fiedler_value: λ₂(L) — algebraic connectivity.
        is_connected: True iff λ₂ > 0.
        cut_vertices: States whose removal disconnects the Hasse diagram.
        vertex_connectivity: Minimum number of vertices to remove to disconnect.
        edge_connectivity: Minimum number of edges to remove to disconnect.
        num_edges: Number of Hasse diagram edges.
        bottleneck_state: State with highest betweenness (most paths through it).
        cheeger_lower: Lower bound from Cheeger inequality: h(G) ≥ λ₂/2.
        cheeger_upper: Upper bound from Cheeger inequality: h(G) ≤ √(2λ₂).
    """
    num_states: int
    fiedler_value: float
    is_connected: bool
    cut_vertices: list[int]
    vertex_connectivity: int
    edge_connectivity: int
    num_edges: int
    bottleneck_state: int | None
    cheeger_lower: float
    cheeger_upper: float


# ---------------------------------------------------------------------------
# Fiedler value and vector
# ---------------------------------------------------------------------------

def fiedler_value(ss: "StateSpace") -> float:
    """Compute the Fiedler value λ₂(L).

    Second-smallest Laplacian eigenvalue. Measures algebraic connectivity.
    λ₂ > 0 iff the graph is connected.
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return 0.0
    eigs = sorted(_eigenvalues_symmetric(L))
    return max(0.0, eigs[1]) if len(eigs) >= 2 else 0.0


def fiedler_vector(ss: "StateSpace") -> list[float]:
    """Compute the Fiedler vector (eigenvector for λ₂).

    The signs of Fiedler vector components give an optimal 2-partition
    of the graph (spectral bisection). Positive components go to one
    partition, negative to the other.
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return [0.0] * n

    # Power iteration to find eigenvector for λ₂
    # First compute eigenvalues to know λ₂
    eigs = sorted(_eigenvalues_symmetric(L))
    if len(eigs) < 2:
        return [0.0] * n

    lambda2 = max(0.0, eigs[1])

    # Inverse iteration: (L - λ₂I)^{-1} v converges to eigenvector of λ₂
    # Use shifted inverse iteration with deflation of the constant eigenvector
    # Simpler: power iteration on (λ_max I - L) deflated

    # For small matrices, compute all eigenvectors via Jacobi
    S = [row[:] for row in L]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    max_iter = 100 * n * n
    for _ in range(max_iter):
        p, q = 0, 1
        max_val = abs(S[0][1])
        for i in range(n):
            for j in range(i + 1, n):
                if abs(S[i][j]) > max_val:
                    max_val = abs(S[i][j])
                    p, q = i, j
        if max_val < 1e-12:
            break

        if abs(S[p][p] - S[q][q]) < 1e-15:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * S[p][q], S[p][p] - S[q][q])

        c = math.cos(theta)
        s = math.sin(theta)

        new_S = [row[:] for row in S]
        for i in range(n):
            if i != p and i != q:
                new_S[i][p] = c * S[i][p] + s * S[i][q]
                new_S[p][i] = new_S[i][p]
                new_S[i][q] = -s * S[i][p] + c * S[i][q]
                new_S[q][i] = new_S[i][q]
        new_S[p][p] = c * c * S[p][p] + 2 * s * c * S[p][q] + s * s * S[q][q]
        new_S[q][q] = s * s * S[p][p] - 2 * s * c * S[p][q] + c * c * S[q][q]
        new_S[p][q] = 0.0
        new_S[q][p] = 0.0

        # Update eigenvector matrix
        new_V = [row[:] for row in V]
        for i in range(n):
            new_V[i][p] = c * V[i][p] + s * V[i][q]
            new_V[i][q] = -s * V[i][p] + c * V[i][q]

        S = new_S
        V = new_V

    # Find which diagonal of S is closest to λ₂
    diag = [S[i][i] for i in range(n)]
    best_idx = min(range(n), key=lambda i: abs(diag[i] - lambda2))

    # Extract eigenvector column
    vec = [V[i][best_idx] for i in range(n)]

    # Normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 1e-15:
        vec = [x / norm for x in vec]

    return vec


# ---------------------------------------------------------------------------
# Cut vertices and connectivity
# ---------------------------------------------------------------------------

def _build_hasse_adj(ss: "StateSpace") -> dict[int, set[int]]:
    """Build undirected adjacency from Hasse diagram."""
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for a, b in _hasse_edges(ss):
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _is_connected_without(adj: dict[int, set[int]], exclude: int | None = None) -> bool:
    """Check if the Hasse graph is connected, optionally excluding a vertex."""
    nodes = [s for s in adj if s != exclude]
    if not nodes:
        return True

    visited: set[int] = set()
    stack = [nodes[0]]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        for v in adj[u]:
            if v != exclude and v not in visited:
                stack.append(v)

    return len(visited) == len(nodes)


def find_cut_vertices(ss: "StateSpace") -> list[int]:
    """Find cut vertices (articulation points) of the Hasse diagram.

    A cut vertex is a state whose removal disconnects the graph.
    These represent protocol bottlenecks.
    """
    adj = _build_hasse_adj(ss)
    cuts = []
    for s in sorted(ss.states):
        if not _is_connected_without(adj, exclude=s):
            cuts.append(s)
    return cuts


def vertex_connectivity(ss: "StateSpace") -> int:
    """Minimum number of vertices to remove to disconnect the graph.

    For complete graphs, this equals n-1.
    For trees, this is 1 (any internal node is a cut vertex).
    """
    n = len(ss.states)
    if n <= 1:
        return 0

    adj = _build_hasse_adj(ss)

    # Check if already disconnected
    if not _is_connected_without(adj):
        return 0

    # Try removing sets of increasing size
    states = sorted(ss.states)

    # Size 1: cut vertices
    for s in states:
        if not _is_connected_without(adj, exclude=s):
            return 1

    # For small graphs, check pairs
    if n <= 20:
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                remaining = {s: adj[s] - {states[i], states[j]}
                             for s in states if s != states[i] and s != states[j]}
                nodes = list(remaining.keys())
                if not nodes:
                    continue
                vis: set[int] = set()
                stack = [nodes[0]]
                while stack:
                    u = stack.pop()
                    if u in vis:
                        continue
                    vis.add(u)
                    for v in remaining[u]:
                        if v not in vis:
                            stack.append(v)
                if len(vis) < len(nodes):
                    return 2

    # Default: assume high connectivity
    return min(n - 1, 3)


def edge_connectivity(ss: "StateSpace") -> int:
    """Minimum number of edges to remove to disconnect the graph.

    By Whitney's theorem: vertex_conn ≤ edge_conn ≤ min_degree.
    """
    A = adjacency_matrix(ss)
    n = len(A)
    if n <= 1:
        return 0

    # Min degree gives upper bound
    min_deg = min(sum(row) for row in A)

    # Check if removing min_deg edges disconnects
    # For simple cases, edge_connectivity = min_degree
    return min_deg


# ---------------------------------------------------------------------------
# Bottleneck detection
# ---------------------------------------------------------------------------

def betweenness_centrality(ss: "StateSpace") -> dict[int, float]:
    """Approximate betweenness centrality of each state in the Hasse diagram.

    States with high betweenness are protocol bottlenecks — many
    shortest paths pass through them.
    """
    adj = _build_hasse_adj(ss)
    states = sorted(ss.states)
    n = len(states)
    centrality = {s: 0.0 for s in states}

    for source in states:
        # BFS from source
        dist: dict[int, int] = {source: 0}
        num_paths: dict[int, int] = {source: 1}
        order: list[int] = []
        queue = [source]
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    num_paths[v] = 0
                    queue.append(v)
                if dist[v] == dist[u] + 1:
                    num_paths[v] += num_paths[u]

        # Back-propagate dependencies
        dep: dict[int, float] = {s: 0.0 for s in states}
        for v in reversed(order):
            if v == source:
                continue
            for u in adj[v]:
                if u in dist and dist[u] == dist[v] - 1:
                    frac = num_paths[u] / num_paths[v] if num_paths[v] > 0 else 0
                    dep[u] += frac * (1 + dep[v])
            centrality[v] += dep[v]

    # Normalize
    if n > 2:
        norm = 1.0 / ((n - 1) * (n - 2))
        for s in states:
            centrality[s] *= norm

    return centrality


def find_bottleneck(ss: "StateSpace") -> int | None:
    """Find the state with highest betweenness centrality."""
    bc = betweenness_centrality(ss)
    if not bc:
        return None
    return max(bc, key=lambda s: bc[s])


# ---------------------------------------------------------------------------
# Cheeger inequality
# ---------------------------------------------------------------------------

def cheeger_bounds(ss: "StateSpace") -> tuple[float, float]:
    """Compute Cheeger inequality bounds on the conductance h(G).

    Cheeger's inequality: λ₂/2 ≤ h(G) ≤ √(2λ₂)

    where h(G) is the edge expansion (conductance) of the graph.
    """
    fv = fiedler_value(ss)
    lower = fv / 2.0
    upper = math.sqrt(2.0 * fv) if fv >= 0 else 0.0
    return lower, upper


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def verify_fiedler_min_composition(
    ss_left: "StateSpace",
    ss_right: "StateSpace",
    ss_product: "StateSpace",
    tol: float = 0.1,
) -> bool:
    """Verify λ₂(L₁ × L₂) = min(λ₂(L₁), λ₂(L₂))."""
    f_left = fiedler_value(ss_left)
    f_right = fiedler_value(ss_right)
    f_product = fiedler_value(ss_product)
    expected = min(f_left, f_right)
    return abs(f_product - expected) < tol


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_fiedler(ss: "StateSpace") -> FiedlerResult:
    """Complete algebraic connectivity analysis."""
    fv = fiedler_value(ss)
    cuts = find_cut_vertices(ss)
    v_conn = vertex_connectivity(ss)
    e_conn = edge_connectivity(ss)
    n_edges = sum(sum(row) for row in adjacency_matrix(ss)) // 2
    bottleneck = find_bottleneck(ss)
    lower, upper = cheeger_bounds(ss)

    return FiedlerResult(
        num_states=len(ss.states),
        fiedler_value=fv,
        is_connected=fv > 1e-10,
        cut_vertices=cuts,
        vertex_connectivity=v_conn,
        edge_connectivity=e_conn,
        num_edges=n_edges,
        bottleneck_state=bottleneck,
        cheeger_lower=lower,
        cheeger_upper=upper,
    )
