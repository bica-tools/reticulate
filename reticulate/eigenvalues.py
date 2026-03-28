"""Eigenvalue analysis of session type lattices (Step 30d).

Spectral analysis of the Hasse diagram viewed as an undirected graph:

- **Adjacency spectrum**: eigenvalues of the adjacency matrix A
- **Laplacian spectrum**: eigenvalues of L = D - A
- **Spectral radius**: max |λ_i| from adjacency spectrum
- **Spectral gap**: λ_1 - λ_2 (largest minus second-largest eigenvalue)
- **Fiedler value**: second-smallest Laplacian eigenvalue (algebraic connectivity)
- **Spectral composition**: for product L₁ × L₂, adjacency eigenvalues are {λᵢ + μⱼ}
- **Spectral classification**: categorize protocols by spectral properties

All eigenvalue computations use the Jacobi iteration from matrix.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.matrix import (
    adjacency_matrix,
    adjacency_spectrum,
    laplacian_matrix,
    fiedler_value as _fiedler_value,
    _eigenvalues_symmetric,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralResult:
    """Complete spectral analysis of a session type lattice.

    Attributes:
        num_states: Number of states.
        adjacency_eigenvalues: Sorted eigenvalues of A (Hasse adjacency).
        laplacian_eigenvalues: Sorted eigenvalues of L = D - A.
        spectral_radius: max |λ_i| from adjacency spectrum.
        spectral_gap: λ_max - λ_second from adjacency spectrum.
        fiedler_value: Second-smallest Laplacian eigenvalue.
        algebraic_connectivity: Same as Fiedler value.
        is_connected: True iff Fiedler value > 0.
        num_edges: Number of edges in Hasse diagram.
        avg_degree: Average degree of Hasse diagram.
        energy: Graph energy = Σ |λ_i|.
    """
    num_states: int
    adjacency_eigenvalues: list[float]
    laplacian_eigenvalues: list[float]
    spectral_radius: float
    spectral_gap: float
    fiedler_value: float
    algebraic_connectivity: float
    is_connected: bool
    num_edges: int
    avg_degree: float
    energy: float


# ---------------------------------------------------------------------------
# Core spectral computations
# ---------------------------------------------------------------------------

def compute_adjacency_spectrum(ss: "StateSpace") -> list[float]:
    """Compute eigenvalues of the undirected Hasse adjacency matrix."""
    return adjacency_spectrum(ss)


def compute_laplacian_spectrum(ss: "StateSpace") -> list[float]:
    """Compute eigenvalues of the Laplacian L = D - A."""
    L = laplacian_matrix(ss)
    n = len(L)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    return sorted(_eigenvalues_symmetric(L))


def spectral_radius(ss: "StateSpace") -> float:
    """Maximum absolute eigenvalue of the adjacency matrix."""
    eigs = compute_adjacency_spectrum(ss)
    if not eigs:
        return 0.0
    return max(abs(e) for e in eigs)


def spectral_gap(ss: "StateSpace") -> float:
    """Spectral gap: λ_max - λ_second.

    Large spectral gap indicates a well-connected protocol with
    rapid mixing / fast convergence.
    """
    eigs = compute_adjacency_spectrum(ss)
    if len(eigs) < 2:
        return 0.0
    sorted_desc = sorted(eigs, reverse=True)
    return sorted_desc[0] - sorted_desc[1]


def graph_energy(ss: "StateSpace") -> float:
    """Graph energy: E(G) = Σ |λ_i|.

    Measures total spectral content. Higher energy indicates more
    complex graph structure.
    """
    eigs = compute_adjacency_spectrum(ss)
    return sum(abs(e) for e in eigs)


def algebraic_connectivity(ss: "StateSpace") -> float:
    """Algebraic connectivity = Fiedler value = λ₂(L).

    The second-smallest Laplacian eigenvalue. λ₂ > 0 iff graph is connected.
    """
    return _fiedler_value(ss)


# ---------------------------------------------------------------------------
# Hasse diagram properties
# ---------------------------------------------------------------------------

def hasse_edge_count(ss: "StateSpace") -> int:
    """Number of edges in the Hasse diagram."""
    A = adjacency_matrix(ss)
    n = len(A)
    return sum(A[i][j] for i in range(n) for j in range(i + 1, n))


def average_degree(ss: "StateSpace") -> float:
    """Average degree of the Hasse diagram."""
    A = adjacency_matrix(ss)
    n = len(A)
    if n == 0:
        return 0.0
    total_degree = sum(sum(row) for row in A)
    return total_degree / n


def degree_sequence(ss: "StateSpace") -> list[int]:
    """Sorted degree sequence of the Hasse diagram (non-increasing)."""
    A = adjacency_matrix(ss)
    return sorted([sum(row) for row in A], reverse=True)


# ---------------------------------------------------------------------------
# Spectral composition under parallel
# ---------------------------------------------------------------------------

def verify_spectral_composition(
    ss_left: "StateSpace",
    ss_right: "StateSpace",
    ss_product: "StateSpace",
    tol: float = 0.1,
) -> bool:
    """Verify spectrum additivity: eigenvalues of L₁×L₂ = {λᵢ + μⱼ}.

    For the Cartesian product of graphs, the adjacency eigenvalues of
    G₁ □ G₂ are {λᵢ(G₁) + μⱼ(G₂)} for all pairs (i, j).
    """
    eigs_left = compute_adjacency_spectrum(ss_left)
    eigs_right = compute_adjacency_spectrum(ss_right)
    eigs_product = compute_adjacency_spectrum(ss_product)

    # Expected eigenvalues: all pairwise sums
    expected = sorted(a + b for a in eigs_left for b in eigs_right)
    actual = sorted(eigs_product)

    if len(expected) != len(actual):
        return False

    return all(abs(a - e) < tol for a, e in zip(actual, expected))


def verify_fiedler_min(
    ss_left: "StateSpace",
    ss_right: "StateSpace",
    ss_product: "StateSpace",
    tol: float = 0.1,
) -> bool:
    """Verify Fiedler value takes minimum: λ₂(L₁×L₂) = min(λ₂(L₁), λ₂(L₂))."""
    f_left = algebraic_connectivity(ss_left)
    f_right = algebraic_connectivity(ss_right)
    f_product = algebraic_connectivity(ss_product)
    expected = min(f_left, f_right)
    return abs(f_product - expected) < tol


# ---------------------------------------------------------------------------
# Spectral classification
# ---------------------------------------------------------------------------

def classify_spectrum(ss: "StateSpace") -> str:
    """Classify the protocol by spectral properties.

    Returns one of:
    - "trivial": single state
    - "path": chain (spectral radius ≈ 2cos(π/(n+1)))
    - "bipartite": all eigenvalues symmetric around 0
    - "product": Fiedler value matches product structure
    - "general": none of the above
    """
    n = len(ss.states)
    if n <= 1:
        return "trivial"

    eigs = compute_adjacency_spectrum(ss)

    # Check bipartiteness: eigenvalues symmetric around 0
    # (for each λ, -λ is also an eigenvalue)
    sorted_eigs = sorted(eigs)
    is_bipartite = True
    for i in range(len(sorted_eigs)):
        if abs(sorted_eigs[i] + sorted_eigs[-(i + 1)]) > 0.1:
            is_bipartite = False
            break

    # Check path graph: eigenvalues should be 2cos(kπ/(n+1))
    if n >= 2:
        expected_path = sorted(2 * math.cos(k * math.pi / (n + 1)) for k in range(1, n + 1))
        is_path = all(abs(a - e) < 0.1 for a, e in zip(sorted_eigs, expected_path))
        if is_path:
            return "path"

    if is_bipartite:
        return "bipartite"

    return "general"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_eigenvalues(ss: "StateSpace") -> SpectralResult:
    """Complete spectral analysis of a session type state space."""
    adj_eigs = compute_adjacency_spectrum(ss)
    lap_eigs = compute_laplacian_spectrum(ss)
    sr = spectral_radius(ss)
    sg = spectral_gap(ss)
    fv = algebraic_connectivity(ss)
    n_edges = hasse_edge_count(ss)
    avg_deg = average_degree(ss)
    energy = graph_energy(ss)

    return SpectralResult(
        num_states=len(ss.states),
        adjacency_eigenvalues=adj_eigs,
        laplacian_eigenvalues=lap_eigs,
        spectral_radius=sr,
        spectral_gap=sg,
        fiedler_value=fv,
        algebraic_connectivity=fv,
        is_connected=fv > 1e-10,
        num_edges=n_edges,
        avg_degree=avg_deg,
        energy=energy,
    )
