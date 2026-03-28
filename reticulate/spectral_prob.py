"""Spectral-probabilistic analysis for session type lattices (Steps 30o-30r).

Probabilistic and information-theoretic invariants derived from the
Hasse diagram's spectral structure:

- **Step 30o**: Cheeger constant (edge expansion / conductance)
- **Step 30p**: Random walk mixing time
- **Step 30q**: Heat kernel trace and diagonal
- **Step 30r**: Von Neumann entropy of the Laplacian

These invariants capture how "well-mixed" a protocol is — whether
all states are equally accessible or some regions are isolated.
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
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralProbResult:
    """Spectral-probabilistic analysis results.

    Attributes:
        num_states: Number of states.
        cheeger_constant: Edge expansion h(G).
        cheeger_lower: λ₂/2 (lower bound from Cheeger inequality).
        cheeger_upper: √(2λ₂) (upper bound from Cheeger inequality).
        mixing_time_bound: Upper bound on mixing time from spectral gap.
        heat_trace: Trace of heat kernel at t=1: Σ exp(-λᵢ).
        von_neumann_entropy: S = -Σ (λᵢ/Σλⱼ) log(λᵢ/Σλⱼ).
        normalized_entropy: S / log(n) (0 to 1 scale).
        stationary_distribution: Stationary dist of random walk (degree-proportional).
    """
    num_states: int
    cheeger_constant: float
    cheeger_lower: float
    cheeger_upper: float
    mixing_time_bound: float
    heat_trace: float
    von_neumann_entropy: float
    normalized_entropy: float
    stationary_distribution: dict[int, float]


# ---------------------------------------------------------------------------
# Step 30o: Cheeger constant
# ---------------------------------------------------------------------------

def cheeger_constant(ss: "StateSpace") -> float:
    """Compute the Cheeger constant (edge expansion) h(G).

    h(G) = min over subsets S with |S| <= n/2 of |boundary(S)| / |S|

    where boundary(S) = edges between S and V minus S.
    For small graphs, we compute exactly by trying all subsets.
    """
    A = adjacency_matrix(ss)
    n = len(A)
    if n <= 1:
        return 0.0

    min_expansion = float('inf')

    # Try all non-empty subsets of size ≤ n/2
    for mask in range(1, 1 << n):
        size = bin(mask).count('1')
        if size > n // 2:
            continue
        if size == 0:
            continue

        # Count boundary edges
        boundary = 0
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                boundary += A[i][j]

        expansion = boundary / size
        min_expansion = min(min_expansion, expansion)

    return min_expansion if min_expansion < float('inf') else 0.0


def cheeger_bounds(ss: "StateSpace") -> tuple[float, float]:
    """Cheeger inequality bounds: λ₂/2 ≤ h(G) ≤ √(2λ₂)."""
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return (0.0, 0.0)
    eigs = sorted(_eigenvalues_symmetric(L))
    lambda2 = max(0.0, eigs[1]) if len(eigs) >= 2 else 0.0
    return (lambda2 / 2.0, math.sqrt(2.0 * lambda2) if lambda2 > 0 else 0.0)


# ---------------------------------------------------------------------------
# Step 30p: Random walk mixing time
# ---------------------------------------------------------------------------

def mixing_time_bound(ss: "StateSpace") -> float:
    """Upper bound on random walk mixing time from spectral gap.

    t_mix ≤ (1/gap) · log(n) where gap = 1 - λ₂'/λ_max' for
    the normalized transition matrix.

    For the lazy random walk on the Hasse diagram:
    t_mix ≈ (1/λ₂) · log(n).
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return 0.0

    eigs = sorted(_eigenvalues_symmetric(L))
    lambda2 = max(eigs[1], 1e-15) if len(eigs) >= 2 else 1e-15

    return (1.0 / lambda2) * math.log(max(n, 2))


def stationary_distribution(ss: "StateSpace") -> dict[int, float]:
    """Stationary distribution of the random walk on the Hasse diagram.

    For an undirected graph, π(v) = deg(v) / (2|E|).
    """
    A = adjacency_matrix(ss)
    n = len(A)
    states = sorted(ss.states)

    total_degree = sum(sum(row) for row in A)
    if total_degree == 0:
        return {s: 1.0 / n for s in states}

    return {states[i]: sum(A[i]) / total_degree for i in range(n)}


# ---------------------------------------------------------------------------
# Step 30q: Heat kernel
# ---------------------------------------------------------------------------

def heat_kernel_trace(ss: "StateSpace", t: float = 1.0) -> float:
    """Heat kernel trace: Z(t) = Σ exp(-λᵢ · t).

    The trace of the heat kernel e^{-tL} encodes spectral information.
    At t=0: Z = n. As t → ∞: Z → 1 (contribution from λ₀=0).
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return float(n)

    eigs = _eigenvalues_symmetric(L)
    return sum(math.exp(-e * t) for e in eigs)


def heat_kernel_diagonal(ss: "StateSpace", t: float = 1.0) -> dict[int, float]:
    """Diagonal of the heat kernel: h_t(v,v) = Σ_k φ_k(v)² exp(-λ_k t).

    This measures the "return probability" of the random walk at time t.
    Approximated via the spectral decomposition.
    """
    # For exact computation we'd need eigenvectors.
    # Approximation: uniform over n gives h_t(v,v) ≈ Z(t)/n.
    Z = heat_kernel_trace(ss, t)
    n = len(ss.states)
    states = sorted(ss.states)
    return {s: Z / n for s in states}


# ---------------------------------------------------------------------------
# Step 30r: Von Neumann entropy
# ---------------------------------------------------------------------------

def von_neumann_entropy(ss: "StateSpace") -> float:
    """Von Neumann entropy of the normalized Laplacian.

    S(G) = -Σ_k (λ_k / Σ_j λ_j) · log(λ_k / Σ_j λ_j)

    where the sum is over non-zero Laplacian eigenvalues.
    Measures spectral complexity / information content.
    """
    L = laplacian_matrix(ss)
    n = len(L)
    if n <= 1:
        return 0.0

    eigs = _eigenvalues_symmetric(L)
    # Filter out near-zero eigenvalues
    positive_eigs = [e for e in eigs if e > 1e-10]
    if not positive_eigs:
        return 0.0

    total = sum(positive_eigs)
    if total < 1e-15:
        return 0.0

    entropy = 0.0
    for e in positive_eigs:
        p = e / total
        if p > 1e-15:
            entropy -= p * math.log(p)

    return entropy


def normalized_entropy(ss: "StateSpace") -> float:
    """Normalized Von Neumann entropy: S / log(n-1).

    Ranges from 0 (one dominant eigenvalue) to 1 (uniform spectrum).
    """
    n = len(ss.states)
    if n <= 2:
        return 0.0
    S = von_neumann_entropy(ss)
    max_S = math.log(n - 1)  # Maximum for n-1 positive eigenvalues
    return S / max_S if max_S > 0 else 0.0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_spectral_prob(ss: "StateSpace") -> SpectralProbResult:
    """Complete spectral-probabilistic analysis."""
    n = len(ss.states)

    # Cheeger (expensive for large n, skip if > 20)
    if n <= 20:
        h = cheeger_constant(ss)
    else:
        h = 0.0  # Would need approximation

    lo, hi = cheeger_bounds(ss)
    t_mix = mixing_time_bound(ss)
    Z = heat_kernel_trace(ss)
    S = von_neumann_entropy(ss)
    S_norm = normalized_entropy(ss)
    pi = stationary_distribution(ss)

    return SpectralProbResult(
        num_states=n,
        cheeger_constant=h,
        cheeger_lower=lo,
        cheeger_upper=hi,
        mixing_time_bound=t_mix,
        heat_trace=Z,
        von_neumann_entropy=S,
        normalized_entropy=S_norm,
        stationary_distribution=pi,
    )
