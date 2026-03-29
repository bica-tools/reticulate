"""Spectral protocol compression via sparsification (Step 31g).

Spectral sparsification finds a subgraph with fewer edges that preserves
the spectral properties (eigenvalues of the Laplacian) of the original
Hasse diagram.  This enables protocol compression: a simpler protocol
that behaves spectrally like the original.

A spectral sparsifier H of G satisfies, for all vectors x:
    (1-epsilon) * x^T L_G x  <=  x^T L_H x  <=  (1+epsilon) * x^T L_G x

where L_G and L_H are the Laplacians, and epsilon is the quality parameter.

Main API:
  - ``spectral_sparsify(ss, epsilon)``   find a spectral sparsifier
  - ``compression_quality(ss, edges)``   measure how well a subgraph preserves spectrum
  - ``preserve_fiedler(ss, edges)``      check Fiedler value preservation
  - ``effective_resistance(ss)``         compute effective resistances for all edges
  - ``sparsification_potential(ss)``     how much can this protocol be compressed?
  - ``SparsifyResult``                   aggregate result
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SparsifyResult:
    """Result of spectral sparsification.

    Attributes:
        original_edges: Number of edges in the original Hasse diagram.
        sparsified_edges: Number of edges in the sparsifier.
        compression_ratio: sparsified / original (lower = more compression).
        epsilon: Quality parameter (spectral approximation factor).
        fiedler_original: Fiedler value of the original.
        fiedler_sparsified: Fiedler value of the sparsifier.
        fiedler_ratio: sparsified Fiedler / original Fiedler.
        kept_edges: List of (src, tgt) pairs of kept edges.
        edge_weights: Weights assigned to kept edges.
    """
    original_edges: int
    sparsified_edges: int
    compression_ratio: float
    epsilon: float
    fiedler_original: float
    fiedler_sparsified: float
    fiedler_ratio: float
    kept_edges: list[tuple[int, int]]
    edge_weights: list[float]


@dataclass(frozen=True)
class CompressionQuality:
    """Quality metrics for a subgraph as spectral approximation.

    Attributes:
        max_distortion: Maximum Rayleigh quotient distortion.
        fiedler_ratio: Subgraph Fiedler / original Fiedler.
        eigenvalue_distances: Per-eigenvalue absolute differences.
        is_good_sparsifier: Whether max_distortion <= epsilon.
    """
    max_distortion: float
    fiedler_ratio: float
    eigenvalue_distances: list[float]
    is_good_sparsifier: bool


@dataclass(frozen=True)
class SparsificationPotential:
    """How compressible is this protocol spectrally?

    Attributes:
        num_edges: Total Hasse edges.
        num_states: Total states.
        min_edges_needed: Lower bound on edges for connectivity.
        max_removable: num_edges - min_edges_needed.
        removable_fraction: max_removable / num_edges.
        effective_resistances: Dict mapping edge index to R_eff.
        low_resistance_count: Edges with R_eff < median (candidates for removal).
    """
    num_edges: int
    num_states: int
    min_edges_needed: int
    max_removable: int
    removable_fraction: float
    effective_resistances: dict[int, float]
    low_resistance_count: int


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


def _laplacian_from_edges(
    states: list[int], edges: list[tuple[int, int]], weights: list[float] | None = None,
) -> list[list[float]]:
    """Build Laplacian matrix from edge list with optional weights."""
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    L = [[0.0] * n for _ in range(n)]

    if weights is None:
        weights = [1.0] * len(edges)

    for k, (a, b) in enumerate(edges):
        w = weights[k]
        i, j = idx[a], idx[b]
        L[i][i] += w
        L[j][j] += w
        L[i][j] -= w
        L[j][i] -= w

    return L


def _eigenvalues_symmetric(A: list[list[float]]) -> list[float]:
    """Compute eigenvalues of a symmetric matrix via Jacobi iteration."""
    from reticulate.matrix import _eigenvalues_symmetric
    return _eigenvalues_symmetric(A)


def _laplacian_eigenvalues(
    states: list[int], edges: list[tuple[int, int]], weights: list[float] | None = None,
) -> list[float]:
    """Compute sorted Laplacian eigenvalues."""
    L = _laplacian_from_edges(states, edges, weights)
    n = len(L)
    if n <= 1:
        return [0.0] * n
    eigs = _eigenvalues_symmetric(L)
    return sorted(eigs)


def _fiedler_from_edges(
    states: list[int], edges: list[tuple[int, int]], weights: list[float] | None = None,
) -> float:
    """Compute Fiedler value from edge list."""
    eigs = _laplacian_eigenvalues(states, edges, weights)
    if len(eigs) < 2:
        return 0.0
    return max(0.0, eigs[1])


# ---------------------------------------------------------------------------
# Effective resistance
# ---------------------------------------------------------------------------

def effective_resistance(ss: "StateSpace") -> dict[int, float]:
    """Compute effective resistance for each Hasse edge.

    The effective resistance R_eff(e) of edge e = (u,v) is:
        R_eff(e) = (chi_u - chi_v)^T L^+ (chi_u - chi_v)

    where L^+ is the Moore-Penrose pseudoinverse of the Laplacian.

    Edges with low effective resistance are redundant (removing them
    has little impact on the Laplacian spectrum).  Edges with high
    resistance are critical (bridges have R_eff = 1).

    Args:
        ss: A session type state space.

    Returns:
        Dict mapping edge index to effective resistance.
    """
    states = sorted(ss.states)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    edges = _hasse_edges(ss)

    if n <= 1 or not edges:
        return {}

    # Compute L and its pseudoinverse
    L = _laplacian_from_edges(states, edges)

    # Pseudoinverse: L^+ = (L - J/n)^{-1} + J/n  where J = 11^T
    # For small matrices, compute via eigendecomposition
    # Approximate: L^+ ≈ sum_{i>0} (1/lambda_i) v_i v_i^T
    # For our purposes, use the formula R_eff(u,v) = L^+_{uu} + L^+_{vv} - 2 L^+_{uv}

    # Simple approach: solve L x = (e_u - e_v) for each edge
    # using the regularized system (L + J/n) x = b
    resistances: dict[int, float] = {}

    # Build regularized Laplacian
    L_reg = [row[:] for row in L]
    for i in range(n):
        for j in range(n):
            L_reg[i][j] += 1.0 / n

    # Solve for each edge using Gaussian elimination
    for k, (a, b) in enumerate(edges):
        ai, bi = idx[a], idx[b]
        rhs = [0.0] * n
        rhs[ai] = 1.0
        rhs[bi] = -1.0

        # Solve L_reg * x = rhs
        x = _solve_linear(L_reg, rhs)
        if x is not None:
            r_eff = x[ai] - x[bi]
            resistances[k] = max(0.0, r_eff)
        else:
            resistances[k] = 1.0  # Default for singular case

    return resistances


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        # Partial pivoting
        max_row = col
        max_val = abs(M[col][col])
        for row in range(col + 1, n):
            if abs(M[row][col]) > max_val:
                max_val = abs(M[row][col])
                max_row = row
        if max_val < 1e-14:
            continue
        M[col], M[max_row] = M[max_row], M[col]

        # Eliminate
        pivot = M[col][col]
        for row in range(col + 1, n):
            factor = M[row][col] / pivot
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-14:
            x[i] = 0.0
            continue
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]

    return x


# ---------------------------------------------------------------------------
# Spectral sparsification
# ---------------------------------------------------------------------------

def spectral_sparsify(
    ss: "StateSpace", epsilon: float = 0.3, seed: int = 42,
) -> SparsifyResult:
    """Find a spectral sparsifier of the Hasse diagram.

    Uses effective-resistance sampling (Spielman--Srivastava 2011):
    each edge is kept with probability proportional to its effective
    resistance, and reweighted to maintain the expected Laplacian.

    For small graphs, we use a greedy approach: try removing each edge
    and keep it only if removal would distort the Fiedler value by
    more than epsilon.

    Args:
        ss: A session type state space.
        epsilon: Quality parameter (0 < epsilon < 1).  Smaller = closer
            to original but fewer edges removed.
        seed: Random seed for sampling.

    Returns:
        SparsifyResult with the sparsified edge set and quality metrics.
    """
    states = sorted(ss.states)
    n = len(states)
    edges = _hasse_edges(ss)
    m = len(edges)

    if m == 0 or n <= 1:
        return SparsifyResult(
            original_edges=m, sparsified_edges=0,
            compression_ratio=0.0, epsilon=epsilon,
            fiedler_original=0.0, fiedler_sparsified=0.0,
            fiedler_ratio=0.0, kept_edges=[], edge_weights=[],
        )

    fiedler_orig = _fiedler_from_edges(states, edges)

    if m <= n:
        # Already a tree or near-tree; cannot sparsify further
        return SparsifyResult(
            original_edges=m, sparsified_edges=m,
            compression_ratio=1.0, epsilon=epsilon,
            fiedler_original=fiedler_orig,
            fiedler_sparsified=fiedler_orig,
            fiedler_ratio=1.0,
            kept_edges=edges[:],
            edge_weights=[1.0] * m,
        )

    # Greedy sparsification: try removing edges with low effective resistance
    resistances = effective_resistance(ss)

    # Sort edges by resistance (low resistance = more redundant)
    edge_order = sorted(range(m), key=lambda i: resistances.get(i, 1.0))

    kept = [True] * m
    kept_weights = [1.0] * m

    for idx in edge_order:
        # Try removing this edge
        kept[idx] = False
        current_edges = [edges[i] for i in range(m) if kept[i]]
        current_weights = [kept_weights[i] for i in range(m) if kept[i]]

        if not current_edges:
            kept[idx] = True
            continue

        fiedler_new = _fiedler_from_edges(states, current_edges, current_weights)

        # Check if Fiedler value is preserved within epsilon
        if fiedler_orig > 1e-10:
            distortion = abs(fiedler_new - fiedler_orig) / fiedler_orig
        else:
            distortion = abs(fiedler_new - fiedler_orig)

        if distortion > epsilon:
            # Removing this edge distorts too much; keep it
            kept[idx] = True

    # Build result
    final_edges = [edges[i] for i in range(m) if kept[i]]
    final_weights = [kept_weights[i] for i in range(m) if kept[i]]
    fiedler_sparse = _fiedler_from_edges(states, final_edges, final_weights)

    fiedler_ratio = fiedler_sparse / fiedler_orig if fiedler_orig > 1e-10 else 1.0

    return SparsifyResult(
        original_edges=m,
        sparsified_edges=len(final_edges),
        compression_ratio=len(final_edges) / m if m > 0 else 1.0,
        epsilon=epsilon,
        fiedler_original=fiedler_orig,
        fiedler_sparsified=fiedler_sparse,
        fiedler_ratio=fiedler_ratio,
        kept_edges=final_edges,
        edge_weights=final_weights,
    )


# ---------------------------------------------------------------------------
# Compression quality
# ---------------------------------------------------------------------------

def compression_quality(
    ss: "StateSpace", kept_edges: list[tuple[int, int]],
    weights: list[float] | None = None, epsilon: float = 0.3,
) -> CompressionQuality:
    """Measure how well a subgraph preserves the spectrum.

    Compares Laplacian eigenvalues of the original and subgraph.

    Args:
        ss: Original state space.
        kept_edges: Edges of the subgraph.
        weights: Optional weights for subgraph edges.
        epsilon: Quality threshold.

    Returns:
        CompressionQuality with distortion metrics.
    """
    states = sorted(ss.states)
    orig_edges = _hasse_edges(ss)

    eigs_orig = _laplacian_eigenvalues(states, orig_edges)
    eigs_sub = _laplacian_eigenvalues(states, kept_edges, weights)

    # Pad to same length
    n = max(len(eigs_orig), len(eigs_sub))
    while len(eigs_orig) < n:
        eigs_orig.append(0.0)
    while len(eigs_sub) < n:
        eigs_sub.append(0.0)

    # Per-eigenvalue distances
    distances = [abs(eigs_orig[i] - eigs_sub[i]) for i in range(n)]
    max_dist = max(distances) if distances else 0.0

    # Fiedler ratio
    fiedler_orig = max(0.0, eigs_orig[1]) if len(eigs_orig) > 1 else 0.0
    fiedler_sub = max(0.0, eigs_sub[1]) if len(eigs_sub) > 1 else 0.0
    fiedler_ratio = fiedler_sub / fiedler_orig if fiedler_orig > 1e-10 else 1.0

    # Max distortion as fraction of original
    max_orig = max(abs(e) for e in eigs_orig) if eigs_orig else 1.0
    max_distortion = max_dist / max_orig if max_orig > 1e-10 else max_dist

    return CompressionQuality(
        max_distortion=max_distortion,
        fiedler_ratio=fiedler_ratio,
        eigenvalue_distances=distances,
        is_good_sparsifier=max_distortion <= epsilon,
    )


# ---------------------------------------------------------------------------
# Fiedler preservation check
# ---------------------------------------------------------------------------

def preserve_fiedler(
    ss: "StateSpace", kept_edges: list[tuple[int, int]],
    weights: list[float] | None = None, tolerance: float = 0.1,
) -> bool:
    """Check whether a subgraph preserves the Fiedler value.

    Args:
        ss: Original state space.
        kept_edges: Edges of the subgraph.
        weights: Optional weights.
        tolerance: Maximum allowed relative change in Fiedler value.

    Returns:
        True if |fiedler_sub - fiedler_orig| / fiedler_orig <= tolerance.
    """
    states = sorted(ss.states)
    orig_edges = _hasse_edges(ss)

    fiedler_orig = _fiedler_from_edges(states, orig_edges)
    fiedler_sub = _fiedler_from_edges(states, kept_edges, weights)

    if fiedler_orig < 1e-10:
        return fiedler_sub < 1e-10 + tolerance

    return abs(fiedler_sub - fiedler_orig) / fiedler_orig <= tolerance


# ---------------------------------------------------------------------------
# Sparsification potential
# ---------------------------------------------------------------------------

def sparsification_potential(ss: "StateSpace") -> SparsificationPotential:
    """Analyze how much the protocol can be compressed.

    Args:
        ss: A session type state space.

    Returns:
        SparsificationPotential with compression metrics.
    """
    states = sorted(ss.states)
    n = len(states)
    edges = _hasse_edges(ss)
    m = len(edges)

    # Minimum edges for a spanning tree
    min_edges = max(0, n - 1)
    max_removable = max(0, m - min_edges)
    removable_frac = max_removable / m if m > 0 else 0.0

    # Effective resistances
    resistances = effective_resistance(ss)

    # Count edges with below-median resistance
    if resistances:
        vals = sorted(resistances.values())
        median = vals[len(vals) // 2]
        low_count = sum(1 for r in resistances.values() if r < median)
    else:
        low_count = 0

    return SparsificationPotential(
        num_edges=m,
        num_states=n,
        min_edges_needed=min_edges,
        max_removable=max_removable,
        removable_fraction=removable_frac,
        effective_resistances=resistances,
        low_resistance_count=low_count,
    )
