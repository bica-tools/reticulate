"""Spectral GNN features for session type lattices (Step 31b).

Graph Neural Network feature extraction from spectral properties of
session type state spaces. This module computes feature vectors
suitable for protocol classification and similarity search:

- **Spectral features**: eigenvalues, spectral moments, spectral gap
- **Structural features**: degree distribution, diameter, width, height
- **Algebraic features**: Möbius value, Whitney numbers, Fiedler value
- **Feature vector**: fixed-length vector combining all invariants
- **Similarity**: cosine and Euclidean distance between feature vectors
- **Classification**: protocol family assignment from features
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.matrix import adjacency_spectrum, fiedler_value as _fiedler
from reticulate.zeta import compute_rank, _compute_sccs
from reticulate.mobius import mobius_value
from reticulate.characteristic import characteristic_polynomial, whitney_numbers_second
from reticulate.eigenvalues import graph_energy, spectral_radius as _spectral_radius


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralFeatures:
    """Spectral GNN feature vector for a session type lattice.

    Attributes:
        num_states: Number of states.
        feature_vector: Fixed-length feature vector (20 dimensions).
        feature_names: Names for each dimension.
        spectral_moments: First 4 spectral moments.
        protocol_family: Predicted family (chain/branch/parallel/recursive/mixed).
    """
    num_states: int
    feature_vector: list[float]
    feature_names: list[str]
    spectral_moments: list[float]
    protocol_family: str


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "num_states",           # 0
    "num_transitions",      # 1
    "height",               # 2
    "width",                # 3
    "spectral_radius",      # 4
    "fiedler_value",        # 5
    "graph_energy",         # 6
    "mobius_value",         # 7
    "spectral_moment_1",    # 8  (mean eigenvalue)
    "spectral_moment_2",    # 9  (variance)
    "spectral_moment_3",    # 10 (skewness proxy)
    "spectral_moment_4",    # 11 (kurtosis proxy)
    "max_degree",           # 12
    "min_degree",           # 13
    "avg_degree",           # 14
    "density",              # 15
    "has_parallel",         # 16 (binary: width > 1)
    "has_recursion",        # 17 (binary: has cycles)
    "whitney_max",          # 18 (max Whitney number)
    "char_poly_degree",     # 19
]


def spectral_moments(eigs: list[float], k: int = 4) -> list[float]:
    """Compute first k spectral moments: m_j = (1/n) Σ λ_i^j."""
    n = len(eigs)
    if n == 0:
        return [0.0] * k
    return [sum(e ** j for e in eigs) / n for j in range(1, k + 1)]


def extract_features(ss: "StateSpace") -> list[float]:
    """Extract a 20-dimensional feature vector from a session type state space."""
    n = len(ss.states)
    m = len(ss.transitions)

    rank = compute_rank(ss)
    height = rank.get(ss.top, 0)

    # Width (approximate: count max same-rank elements)
    scc_map, _ = _compute_sccs(ss)
    rank_counts: dict[int, int] = {}
    seen: set[int] = set()
    for s in ss.states:
        rep = scc_map[s]
        if rep not in seen:
            seen.add(rep)
            r = rank[s]
            rank_counts[r] = rank_counts.get(r, 0) + 1
    width = max(rank_counts.values()) if rank_counts else 1

    eigs = adjacency_spectrum(ss)
    sr = _spectral_radius(ss)
    fv = _fiedler(ss)
    energy = graph_energy(ss)
    mu = mobius_value(ss)
    moments = spectral_moments(eigs, 4)

    # Degree info from adjacency
    from reticulate.matrix import adjacency_matrix
    A = adjacency_matrix(ss)
    degrees = [sum(row) for row in A]
    max_deg = max(degrees) if degrees else 0
    min_deg = min(degrees) if degrees else 0
    avg_deg = sum(degrees) / len(degrees) if degrees else 0.0

    # Density
    max_edges = n * (n - 1) / 2
    density = m / max_edges if max_edges > 0 else 0.0

    # Binary features
    has_parallel = 1.0 if width > 1 else 0.0
    # Recursion detection: any SCC with more than one state
    has_rec = any(
        sum(1 for s2 in ss.states if scc_map[s2] == rep) > 1
        for rep in set(scc_map.values())
    )
    has_recursion = 1.0 if has_rec else 0.0

    # Whitney
    W = whitney_numbers_second(ss)
    whitney_max = max(W.values()) if W else 1

    # Characteristic polynomial degree
    cp = characteristic_polynomial(ss)
    cp_deg = len(cp) - 1

    return [
        float(n), float(m), float(height), float(width),
        sr, fv, energy, float(mu),
        moments[0], moments[1], moments[2], moments[3],
        float(max_deg), float(min_deg), avg_deg, density,
        has_parallel, has_recursion,
        float(whitney_max), float(cp_deg),
    ]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity between two feature vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 < 1e-15 or norm2 < 1e-15:
        return 0.0
    return dot / (norm1 * norm2)


def euclidean_distance(v1: list[float], v2: list[float]) -> float:
    """Euclidean distance between two feature vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_protocol(ss: "StateSpace") -> str:
    """Classify protocol into a family based on spectral features.

    Families:
    - "chain": purely sequential (width 1, no recursion)
    - "branch": branching but no parallel (width 1, multiple transitions from some state)
    - "parallel": uses parallel composition (width > 1)
    - "recursive": has cycles from recursion
    - "mixed": combination of above
    """
    features = extract_features(ss)
    width = features[3]
    has_parallel = features[16] > 0.5
    has_recursion = features[17] > 0.5
    n_trans = features[1]
    n_states = features[0]

    if has_recursion and has_parallel:
        return "mixed"
    if has_recursion:
        return "recursive"
    if has_parallel:
        return "parallel"

    # Check branching: more transitions than a chain would have
    if n_trans > n_states - 0.5:  # Chain has exactly n-1 transitions
        return "branch"

    return "chain"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_spectral_gnn(ss: "StateSpace") -> SpectralFeatures:
    """Complete spectral GNN feature extraction."""
    features = extract_features(ss)
    eigs = adjacency_spectrum(ss)
    moments = spectral_moments(eigs, 4)
    family = classify_protocol(ss)

    return SpectralFeatures(
        num_states=len(ss.states),
        feature_vector=features,
        feature_names=FEATURE_NAMES,
        spectral_moments=moments,
        protocol_family=family,
    )
