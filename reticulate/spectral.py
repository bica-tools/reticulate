"""Spectral clustering of session type lattices (Step 31a).

Given a collection of session type state spaces (lattices), this module
extracts spectral feature vectors from each, computes pairwise distances
in feature space, and clusters protocols by structural similarity.

Feature vector components:
  1. spectral_radius   — largest |eigenvalue| of Hasse adjacency matrix
  2. fiedler_value     — algebraic connectivity (2nd smallest Laplacian eigenvalue)
  3. von_neumann_entropy — spectral complexity of the Laplacian
  4. width             — maximum antichain size (Dilworth width)
  5. height            — longest chain from top to bottom
  6. num_states        — number of states in the state space
  7. num_transitions   — number of transitions

Key properties:
  - Feature extraction is deterministic and compositional under parallel.
  - Euclidean distance in feature space provides a metric on protocol structure.
  - k-means clustering groups protocols with similar lattice geometry.
  - Similarity search finds the n closest protocols to a query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralFeatures:
    """Feature vector extracted from a session type lattice.

    Attributes:
        spectral_radius: Largest absolute eigenvalue of the Hasse adjacency matrix.
        fiedler_value: Second-smallest Laplacian eigenvalue (algebraic connectivity).
        von_neumann_entropy: Spectral entropy of the Laplacian.
        width: Maximum antichain size (Dilworth width).
        height: Length of longest chain (number of edges, top to bottom).
        num_states: Number of states in the state space.
        num_transitions: Number of transitions in the state space.
    """
    spectral_radius: float
    fiedler_value: float
    von_neumann_entropy: float
    width: int
    height: int
    num_states: int
    num_transitions: int

    def as_vector(self) -> list[float]:
        """Return the feature vector as a list of floats."""
        return [
            self.spectral_radius,
            self.fiedler_value,
            self.von_neumann_entropy,
            float(self.width),
            float(self.height),
            float(self.num_states),
            float(self.num_transitions),
        ]


@dataclass(frozen=True)
class SpectralCluster:
    """A cluster of protocols grouped by spectral similarity.

    Attributes:
        cluster_id: Integer identifier for the cluster (0-indexed).
        centroid: The centroid feature vector of this cluster.
        members: List of (index, name) pairs for protocols in this cluster.
        inertia: Sum of squared distances from members to centroid.
    """
    cluster_id: int
    centroid: list[float]
    members: list[tuple[int, str]]
    inertia: float


@dataclass(frozen=True)
class ClusteringResult:
    """Result of spectral clustering on a collection of protocols.

    Attributes:
        k: Number of clusters.
        clusters: List of SpectralCluster objects.
        assignments: Mapping from protocol index to cluster ID.
        total_inertia: Sum of inertia across all clusters.
        features: List of (name, SpectralFeatures) pairs for all protocols.
        silhouette_avg: Average silhouette coefficient (-1 to 1), or 0 if k=1.
    """
    k: int
    clusters: list[SpectralCluster]
    assignments: dict[int, int]
    total_inertia: float
    features: list[tuple[str, SpectralFeatures]]
    silhouette_avg: float


@dataclass(frozen=True)
class SimilarityResult:
    """Result of a similarity search.

    Attributes:
        query_features: Features of the query protocol.
        neighbors: List of (distance, index, name) triples, sorted ascending.
    """
    query_features: SpectralFeatures
    neighbors: list[tuple[float, int, str]]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _compute_width(ss: "StateSpace") -> int:
    """Compute the width (maximum antichain size) of the state space poset."""
    from reticulate.marking_lattice import compute_width
    return compute_width(ss)


def _compute_height(ss: "StateSpace") -> int:
    """Compute the height (longest chain length) of the state space poset."""
    from reticulate.marking_lattice import compute_height
    return compute_height(ss)


def spectral_features(ss: "StateSpace") -> SpectralFeatures:
    """Extract the spectral feature vector from a session type state space.

    Computes seven features that characterise the lattice geometry:
    spectral radius, Fiedler value, von Neumann entropy, width, height,
    number of states, and number of transitions.

    Args:
        ss: A session type state space (from build_statespace).

    Returns:
        A SpectralFeatures dataclass with all seven features.
    """
    from reticulate.matrix import (
        adjacency_spectrum,
        fiedler_value as compute_fiedler,
        von_neumann_entropy as compute_entropy,
    )

    # Spectral invariants from matrix.py
    eigs = adjacency_spectrum(ss)
    sp_radius = max(abs(e) for e in eigs) if eigs else 0.0
    fiedler = compute_fiedler(ss)
    entropy = compute_entropy(ss)

    # Poset metrics
    width = _compute_width(ss)
    height = _compute_height(ss)

    return SpectralFeatures(
        spectral_radius=sp_radius,
        fiedler_value=fiedler,
        von_neumann_entropy=entropy,
        width=width,
        height=height,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
    )


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def _normalize_vectors(
    vectors: list[list[float]],
) -> tuple[list[list[float]], list[float], list[float]]:
    """Min-max normalize each feature dimension to [0, 1].

    Returns:
        (normalized_vectors, mins, ranges) where ranges[i] = max_i - min_i.
    """
    if not vectors:
        return [], [], []

    dim = len(vectors[0])
    mins = [min(v[d] for v in vectors) for d in range(dim)]
    maxs = [max(v[d] for v in vectors) for d in range(dim)]
    ranges = [maxs[d] - mins[d] if maxs[d] > mins[d] else 1.0 for d in range(dim)]

    normalized = []
    for v in vectors:
        normalized.append([(v[d] - mins[d]) / ranges[d] for d in range(dim)])

    return normalized, mins, ranges


def _normalize_single(
    v: list[float], mins: list[float], ranges: list[float],
) -> list[float]:
    """Normalize a single vector using precomputed min/range."""
    return [(v[d] - mins[d]) / ranges[d] for d in range(len(v))]


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def spectral_distance(ss1: "StateSpace", ss2: "StateSpace") -> float:
    """Compute Euclidean distance between two state spaces in feature space.

    Features are not normalized here (raw Euclidean distance).
    For comparing within a collection, use cluster_benchmarks or find_similar
    which apply min-max normalization.

    Args:
        ss1: First state space.
        ss2: Second state space.

    Returns:
        Euclidean distance in the 7-dimensional feature space.
    """
    f1 = spectral_features(ss1)
    f2 = spectral_features(ss2)
    return _euclidean_distance(f1.as_vector(), f2.as_vector())


# ---------------------------------------------------------------------------
# k-means clustering
# ---------------------------------------------------------------------------

def _kmeans(
    vectors: list[list[float]],
    k: int,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple[list[int], list[list[float]]]:
    """Simple k-means clustering (Lloyd's algorithm).

    Uses deterministic initialization: pick k evenly spaced points.

    Args:
        vectors: List of feature vectors (already normalized).
        k: Number of clusters.
        max_iter: Maximum iterations.
        seed: Not used (deterministic init), kept for API consistency.

    Returns:
        (assignments, centroids) where assignments[i] is the cluster ID
        for vectors[i], and centroids[j] is the centroid of cluster j.
    """
    n = len(vectors)
    if n == 0:
        return [], []
    if k >= n:
        # Each point is its own cluster
        return list(range(n)), [v[:] for v in vectors]

    dim = len(vectors[0])

    # Deterministic initialization: pick evenly spaced indices
    centroids = [vectors[i * n // k][:] for i in range(k)]

    assignments = [0] * n

    for _ in range(max_iter):
        # Assignment step
        new_assignments = [0] * n
        for i, v in enumerate(vectors):
            best_c = 0
            best_d = _euclidean_distance(v, centroids[0])
            for c in range(1, k):
                d = _euclidean_distance(v, centroids[c])
                if d < best_d:
                    best_d = d
                    best_c = c
            new_assignments[i] = best_c

        if new_assignments == assignments:
            break
        assignments = new_assignments

        # Update step
        for c in range(k):
            members = [vectors[i] for i in range(n) if assignments[i] == c]
            if members:
                centroids[c] = [
                    sum(m[d] for m in members) / len(members) for d in range(dim)
                ]

    return assignments, centroids


def _silhouette(
    vectors: list[list[float]], assignments: list[int], k: int,
) -> float:
    """Compute average silhouette coefficient.

    For each point, silhouette = (b - a) / max(a, b) where:
      a = mean distance to same-cluster points
      b = mean distance to nearest other cluster
    """
    n = len(vectors)
    if n <= 1 or k <= 1 or k >= n:
        return 0.0

    sil_sum = 0.0
    for i in range(n):
        ci = assignments[i]

        # a(i) = mean distance to same-cluster points
        same = [j for j in range(n) if j != i and assignments[j] == ci]
        if not same:
            continue
        a_i = sum(_euclidean_distance(vectors[i], vectors[j]) for j in same) / len(same)

        # b(i) = min over other clusters of mean distance
        b_i = float("inf")
        for c in range(k):
            if c == ci:
                continue
            others = [j for j in range(n) if assignments[j] == c]
            if not others:
                continue
            mean_d = sum(
                _euclidean_distance(vectors[i], vectors[j]) for j in others
            ) / len(others)
            b_i = min(b_i, mean_d)

        if b_i == float("inf"):
            continue

        denom = max(a_i, b_i)
        if denom > 1e-15:
            sil_sum += (b_i - a_i) / denom

    return sil_sum / n


def cluster_benchmarks(
    benchmarks: list[tuple[str, "StateSpace"]],
    k: int = 3,
) -> ClusteringResult:
    """Cluster a collection of protocol state spaces by spectral features.

    Extracts feature vectors, applies min-max normalization, and runs
    k-means clustering.

    Args:
        benchmarks: List of (name, state_space) pairs.
        k: Number of clusters (default 3).

    Returns:
        A ClusteringResult with cluster assignments, centroids, and metrics.

    Raises:
        ValueError: If benchmarks is empty or k < 1.
    """
    if not benchmarks:
        raise ValueError("benchmarks list must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")

    # Extract features
    features: list[tuple[str, SpectralFeatures]] = []
    for name, ss in benchmarks:
        f = spectral_features(ss)
        features.append((name, f))

    # Build raw vectors
    raw_vectors = [f.as_vector() for _, f in features]

    # Normalize
    norm_vectors, mins, ranges = _normalize_vectors(raw_vectors)

    # Clamp k to number of distinct vectors
    k = min(k, len(benchmarks))

    # Cluster
    assignments_list, centroids = _kmeans(norm_vectors, k)

    # Build cluster objects
    clusters: list[SpectralCluster] = []
    assignment_map: dict[int, int] = {}
    for i, c in enumerate(assignments_list):
        assignment_map[i] = c

    for c in range(k):
        member_indices = [i for i in range(len(benchmarks)) if assignments_list[i] == c]
        members = [(i, features[i][0]) for i in member_indices]
        inertia = sum(
            _euclidean_distance(norm_vectors[i], centroids[c]) ** 2
            for i in member_indices
        )
        clusters.append(SpectralCluster(
            cluster_id=c,
            centroid=centroids[c] if c < len(centroids) else [0.0] * 7,
            members=members,
            inertia=inertia,
        ))

    total_inertia = sum(cl.inertia for cl in clusters)
    sil = _silhouette(norm_vectors, assignments_list, k)

    return ClusteringResult(
        k=k,
        clusters=clusters,
        assignments=assignment_map,
        total_inertia=total_inertia,
        features=features,
        silhouette_avg=sil,
    )


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def find_similar(
    ss: "StateSpace",
    benchmarks: list[tuple[str, "StateSpace"]],
    n: int = 5,
) -> SimilarityResult:
    """Find the n most similar protocols to a query state space.

    Computes spectral features for the query and all benchmarks, normalizes
    using the combined feature ranges, and returns the n nearest neighbors
    by Euclidean distance.

    Args:
        ss: The query state space.
        benchmarks: List of (name, state_space) pairs to search.
        n: Number of nearest neighbors to return (default 5).

    Returns:
        A SimilarityResult with the query features and sorted neighbors.

    Raises:
        ValueError: If benchmarks is empty or n < 1.
    """
    if not benchmarks:
        raise ValueError("benchmarks list must not be empty")
    if n < 1:
        raise ValueError("n must be at least 1")

    query_feat = spectral_features(ss)
    query_vec = query_feat.as_vector()

    bench_feats: list[tuple[str, SpectralFeatures]] = []
    for name, bss in benchmarks:
        bench_feats.append((name, spectral_features(bss)))

    # Combine all vectors for normalization
    all_vecs = [query_vec] + [f.as_vector() for _, f in bench_feats]
    _, mins, ranges = _normalize_vectors(all_vecs)

    norm_query = _normalize_single(query_vec, mins, ranges)

    # Compute distances
    distances: list[tuple[float, int, str]] = []
    for i, (name, feat) in enumerate(bench_feats):
        norm_bench = _normalize_single(feat.as_vector(), mins, ranges)
        d = _euclidean_distance(norm_query, norm_bench)
        distances.append((d, i, name))

    distances.sort()
    n = min(n, len(distances))

    return SimilarityResult(
        query_features=query_feat,
        neighbors=distances[:n],
    )
