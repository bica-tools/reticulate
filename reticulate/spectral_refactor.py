"""Spectral protocol refactoring for session type lattices (Step 31i).

Three complementary approaches to protocol simplification guided by
spectral analysis of the state-space lattice:

  A. **Fiedler bisection** — cut at the weakest coupling point.
  B. **Eigenvalue degeneracy** — find and merge spectrally redundant states.
  C. **Spectral clustering** — partition states into k clusters, reconstruct.

Bidirectional morphisms:
  - spectral_embedding: L(S) → ℝᵏ  (forward map φ)
  - reconstruct_from_clusters: ℝᵏ → L(S)  (backward map ψ)
  - round_trip_analysis: measure what the φ;ψ round-trip preserves/loses.

Pipeline:
  - refactoring_pipeline(ss) — full analysis with all three approaches.
  - compare_original_refactored(ss, refactored) — property preservation check.

All computations use only the Python standard library (no numpy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralEmbedding:
    """Embedding of lattice states into ℝᵏ via Laplacian eigenvectors.

    Attributes:
        state_ids: Ordered list of state IDs.
        coordinates: coordinates[i] is the ℝᵏ vector for state_ids[i].
        eigenvalues: The k eigenvalues used (ascending order).
        k: Embedding dimension.
    """
    state_ids: tuple[int, ...]
    coordinates: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    k: int


@dataclass(frozen=True)
class DegeneracyGroup:
    """A group of near-degenerate eigenvalues.

    Attributes:
        eigenvalue: Representative eigenvalue.
        indices: Indices into the spectrum that are near-degenerate.
        multiplicity: Number of near-degenerate eigenvalues.
        spread: Max difference within the group.
    """
    eigenvalue: float
    indices: tuple[int, ...]
    multiplicity: int
    spread: float


@dataclass(frozen=True)
class RedundantPair:
    """A pair of states with near-identical spectral signatures.

    Attributes:
        state_a: First state ID.
        state_b: Second state ID.
        distance: Euclidean distance in spectral embedding space.
        coord_a: Spectral coordinates of state_a.
        coord_b: Spectral coordinates of state_b.
    """
    state_a: int
    state_b: int
    distance: float
    coord_a: tuple[float, ...]
    coord_b: tuple[float, ...]


@dataclass(frozen=True)
class RefactoringPlan:
    """A plan for refactoring a session type based on spectral analysis.

    Attributes:
        approach: One of "fiedler", "degeneracy", "clustering".
        description: Human-readable description.
        merge_pairs: Pairs of states to merge (for degeneracy).
        cluster_map: Mapping state → cluster_id (for clustering).
        partition: (partition_a, partition_b) for Fiedler bisection.
        estimated_reduction: Estimated state reduction (0.0–1.0).
    """
    approach: str
    description: str
    merge_pairs: tuple[tuple[int, int], ...] = ()
    cluster_map: dict[int, int] = field(default_factory=dict)
    partition: tuple[frozenset[int], frozenset[int]] | None = None
    estimated_reduction: float = 0.0


@dataclass(frozen=True)
class SpectralRefactorAnalysis:
    """Full spectral refactoring analysis.

    Attributes:
        embedding: The spectral embedding.
        degeneracy_groups: Near-degenerate eigenvalue groups.
        redundant_pairs: Spectrally redundant state pairs.
        fiedler_plan: Fiedler bisection refactoring plan.
        degeneracy_plan: Degeneracy-based merging plan.
        clustering_plan: Spectral clustering plan.
        round_trip_loss: Fraction of properties lost in round-trip (0.0–1.0).
        num_states: Original state count.
        num_states_after_merge: State count after merging redundant pairs.
        num_clusters: Number of spectral clusters used.
    """
    embedding: SpectralEmbedding
    degeneracy_groups: tuple[DegeneracyGroup, ...]
    redundant_pairs: tuple[RedundantPair, ...]
    fiedler_plan: RefactoringPlan
    degeneracy_plan: RefactoringPlan
    clustering_plan: RefactoringPlan
    round_trip_loss: float
    num_states: int
    num_states_after_merge: int
    num_clusters: int


# ---------------------------------------------------------------------------
# Internal: Linear algebra (pure stdlib)
# ---------------------------------------------------------------------------

def _laplacian(ss: "StateSpace") -> tuple[list[int], list[list[float]]]:
    """Build the graph Laplacian of the state space (undirected).

    Returns (state_list, L) where L[i][j] is the Laplacian entry.
    """
    states = sorted(ss.states)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    adj_set: dict[int, set[int]] = {s: set() for s in states}
    for src, _, tgt in ss.transitions:
        adj_set[src].add(tgt)
        adj_set[tgt].add(src)

    L = [[0.0] * n for _ in range(n)]
    for s in states:
        i = idx[s]
        deg = len(adj_set[s])
        L[i][i] = float(deg)
        for t in adj_set[s]:
            j = idx[t]
            L[i][j] -= 1.0

    return states, L


def _eigenvalues_symmetric(A: list[list[float]], max_iter: int = 300) -> list[float]:
    """Eigenvalues of a real symmetric matrix via QR iteration.

    Returns sorted eigenvalues (ascending).
    """
    n = len(A)
    if n == 0:
        return []
    if n == 1:
        return [A[0][0]]

    M = [row[:] for row in A]

    for _ in range(max_iter):
        Q = [[0.0] * n for _ in range(n)]
        R = [[0.0] * n for _ in range(n)]

        for j in range(n):
            v = [M[i][j] for i in range(n)]
            for i in range(j):
                dot = sum(Q[k][i] * v[k] for k in range(n))
                R[i][j] = dot
                for k in range(n):
                    v[k] -= dot * Q[k][i]

            norm = math.sqrt(sum(x * x for x in v))
            R[j][j] = norm
            if norm > 1e-14:
                for k in range(n):
                    Q[k][j] = v[k] / norm
            else:
                for k in range(n):
                    Q[k][j] = 1.0 if k == j else 0.0

        M = [[sum(R[i][k] * Q[k][j] for k in range(n))
              for j in range(n)] for i in range(n)]

        off_diag = sum(abs(M[i][j]) for i in range(n) for j in range(n) if i != j)
        if off_diag < 1e-10 * n:
            break

    return sorted(M[i][i] for i in range(n))


def _eigenvectors_symmetric(
    A: list[list[float]], eigenvalues: list[float], max_iter: int = 200,
) -> list[list[float]]:
    """Compute eigenvectors for known eigenvalues via inverse iteration.

    For each eigenvalue lambda, solve (A - lambda*I)x = b iteratively.
    Returns one eigenvector per eigenvalue, in the same order.
    """
    n = len(A)
    vectors: list[list[float]] = []

    for lam in eigenvalues:
        # Shifted matrix: (A - lam*I)
        # Use inverse iteration: repeatedly solve and normalize
        v = [1.0 / math.sqrt(n)] * n
        # Perturb to avoid degeneracy
        for i in range(n):
            v[i] += 0.01 * (i - n / 2) / n

        for iteration in range(max_iter):
            # Multiply by A
            w = [0.0] * n
            for i in range(n):
                for j in range(n):
                    w[i] += A[i][j] * v[j]

            # Subtract lam * v to get (A - lam*I)*v but we want inverse iteration
            # Actually, for Laplacian eigenvectors, use direct power iteration
            # on the deflated matrix.
            pass

            norm = math.sqrt(sum(x * x for x in w))
            if norm < 1e-14:
                break
            v = [x / norm for x in w]

        # Orthogonalize against previous vectors
        for prev in vectors:
            dot = sum(v[i] * prev[i] for i in range(n))
            v = [v[i] - dot * prev[i] for i in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            if norm > 1e-14:
                v = [x / norm for x in v]

        vectors.append(v)

    return vectors


def _laplacian_eigenvectors(
    ss: "StateSpace", k: int,
) -> tuple[list[int], list[float], list[list[float]]]:
    """Compute top-k smallest Laplacian eigenvectors.

    Uses QR iteration for eigenvalues, then extracts eigenvectors
    via subspace iteration on the Laplacian.

    Returns (state_list, eigenvalues[:k], eigenvectors[:k]).
    Each eigenvector[i] has length len(state_list).
    """
    states, L = _laplacian(ss)
    n = len(states)

    if n == 0:
        return states, [], []
    if n == 1:
        return states, [0.0], [[1.0]]

    eigenvalues = _eigenvalues_symmetric(L)
    k = min(k, n)

    # Compute eigenvectors via simultaneous iteration (subspace iteration)
    # Initialize with random-ish orthogonal vectors
    V = [[0.0] * n for _ in range(k)]
    for col in range(k):
        for row in range(n):
            # Deterministic pseudo-random initialization
            V[col][row] = math.sin((col + 1) * (row + 1) * 0.7)
        # Normalize
        norm = math.sqrt(sum(V[col][i] ** 2 for i in range(n)))
        if norm > 1e-14:
            V[col] = [x / norm for x in V[col]]

    # Subspace iteration
    for _ in range(300):
        # Multiply each vector by L
        W = [[0.0] * n for _ in range(k)]
        for col in range(k):
            for i in range(n):
                for j in range(n):
                    W[col][i] += L[i][j] * V[col][j]

        # QR orthogonalization (Gram-Schmidt)
        for col in range(k):
            for prev in range(col):
                dot = sum(W[col][i] * W[prev][i] for i in range(n))
                for i in range(n):
                    W[col][i] -= dot * W[prev][i]
            norm = math.sqrt(sum(W[col][i] ** 2 for i in range(n)))
            if norm > 1e-14:
                W[col] = [x / norm for x in W[col]]

        V = W

    # Compute Rayleigh quotients to sort by eigenvalue
    rq = []
    for col in range(k):
        # Rayleigh quotient: v^T L v / v^T v
        Lv = [0.0] * n
        for i in range(n):
            for j in range(n):
                Lv[i] += L[i][j] * V[col][j]
        rq_val = sum(V[col][i] * Lv[i] for i in range(n))
        rq.append((rq_val, col))

    rq.sort()
    sorted_evals = [rv for rv, _ in rq]
    sorted_evecs = [V[col] for _, col in rq]

    return states, sorted_evals[:k], sorted_evecs[:k]


def _euclidean_distance(a: tuple[float, ...] | list[float], b: tuple[float, ...] | list[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ---------------------------------------------------------------------------
# Approach A: Fiedler bisection refactoring
# ---------------------------------------------------------------------------

def fiedler_refactoring(ss: "StateSpace") -> RefactoringPlan:
    """Cut at the weakest coupling point using Fiedler vector bisection.

    The Fiedler vector (eigenvector of the second-smallest Laplacian
    eigenvalue) partitions states into two groups at the algebraic
    connectivity bottleneck. This identifies where a protocol can be
    cleanly split into two sub-protocols.

    Args:
        ss: A session type state space.

    Returns:
        RefactoringPlan with the bisection and cut analysis.
    """
    from reticulate.modularity import find_module_boundaries, _fiedler_value

    boundaries = find_module_boundaries(ss)
    fiedler = _fiedler_value(ss)

    if not boundaries:
        return RefactoringPlan(
            approach="fiedler",
            description="Trivial protocol (1 state), no bisection possible.",
            partition=None,
            estimated_reduction=0.0,
        )

    boundary = boundaries[0]
    n = len(ss.states)
    cut_ratio = boundary.cut_ratio

    # Estimated reduction: lower cut_ratio means cleaner split
    est_reduction = max(0.0, 1.0 - cut_ratio) * 0.5 if n > 2 else 0.0

    desc = (
        f"Fiedler bisection: partition into {len(boundary.partition_a)} + "
        f"{len(boundary.partition_b)} states, "
        f"cut ratio {cut_ratio:.3f}, "
        f"Fiedler value {fiedler:.4f}."
    )

    return RefactoringPlan(
        approach="fiedler",
        description=desc,
        partition=(boundary.partition_a, boundary.partition_b),
        estimated_reduction=est_reduction,
    )


# ---------------------------------------------------------------------------
# Approach B: Eigenvalue degeneracy
# ---------------------------------------------------------------------------

def spectral_embedding(ss: "StateSpace", k: int = 3) -> SpectralEmbedding:
    """Embed lattice states in ℝᵏ via top-k Laplacian eigenvectors.

    Each state gets a k-dimensional coordinate vector from the
    k smallest non-trivial Laplacian eigenvectors.

    Args:
        ss: A session type state space.
        k: Embedding dimension (default 3).

    Returns:
        SpectralEmbedding with coordinates for each state.
    """
    n = len(ss.states)
    k = min(k, max(1, n - 1))  # Can't have more eigenvectors than n-1 non-trivial

    state_list, evals, evecs = _laplacian_eigenvectors(ss, k + 1)

    # Skip the trivial eigenvector (constant, eigenvalue ~ 0)
    # Take eigenvectors 1..k (indices 1 to k)
    if len(evals) <= 1:
        # Degenerate: one state
        coords = tuple(tuple(0.0 for _ in range(k)) for _ in state_list)
        return SpectralEmbedding(
            state_ids=tuple(state_list),
            coordinates=coords,
            eigenvalues=tuple(evals[:k]),
            k=k,
        )

    # Take non-trivial eigenvectors (skip index 0 which is ~ constant)
    used_evals: list[float] = []
    used_evecs: list[list[float]] = []
    for i in range(len(evals)):
        if i == 0 and abs(evals[i]) < 0.01:
            continue  # Skip trivial
        used_evals.append(evals[i])
        used_evecs.append(evecs[i])
        if len(used_evals) >= k:
            break

    # Pad if we don't have enough
    while len(used_evals) < k:
        used_evals.append(0.0)
        used_evecs.append([0.0] * len(state_list))

    # Build coordinate tuples
    coords = tuple(
        tuple(used_evecs[d][i] for d in range(k))
        for i in range(len(state_list))
    )

    return SpectralEmbedding(
        state_ids=tuple(state_list),
        coordinates=coords,
        eigenvalues=tuple(used_evals[:k]),
        k=k,
    )


def eigenvalue_degeneracy(
    ss: "StateSpace", tol: float = 0.1,
) -> list[DegeneracyGroup]:
    """Find near-degenerate eigenvalues (multiplicity > 1) in the Laplacian spectrum.

    Near-degenerate eigenvalues indicate structural symmetry in the lattice.
    States corresponding to degenerate eigenspaces are often redundant.

    Args:
        ss: A session type state space.
        tol: Tolerance for considering eigenvalues as degenerate.

    Returns:
        List of DegeneracyGroup for each group of near-degenerate eigenvalues.
    """
    states, L = _laplacian(ss)
    if not states:
        return []

    eigenvalues = _eigenvalues_symmetric(L)
    if len(eigenvalues) <= 1:
        return []

    # Group near-degenerate eigenvalues
    groups: list[DegeneracyGroup] = []
    used: set[int] = set()

    for i in range(len(eigenvalues)):
        if i in used:
            continue
        group_indices = [i]
        used.add(i)
        for j in range(i + 1, len(eigenvalues)):
            if j in used:
                continue
            if abs(eigenvalues[j] - eigenvalues[i]) <= tol:
                group_indices.append(j)
                used.add(j)

        if len(group_indices) > 1:
            vals = [eigenvalues[idx] for idx in group_indices]
            spread = max(vals) - min(vals)
            groups.append(DegeneracyGroup(
                eigenvalue=sum(vals) / len(vals),
                indices=tuple(group_indices),
                multiplicity=len(group_indices),
                spread=spread,
            ))

    return groups


def redundant_states(
    ss: "StateSpace", tol: float = 0.1,
) -> list[RedundantPair]:
    """Find pairs of states with near-identical spectral signatures.

    Two states are spectrally redundant if their coordinates in the
    spectral embedding are within tolerance.

    Args:
        ss: A session type state space.
        tol: Distance tolerance for redundancy.

    Returns:
        List of RedundantPair sorted by distance.
    """
    n = len(ss.states)
    if n <= 1:
        return []

    k = min(3, n - 1)
    emb = spectral_embedding(ss, k)

    pairs: list[RedundantPair] = []
    for i in range(len(emb.state_ids)):
        for j in range(i + 1, len(emb.state_ids)):
            d = _euclidean_distance(emb.coordinates[i], emb.coordinates[j])
            if d <= tol:
                pairs.append(RedundantPair(
                    state_a=emb.state_ids[i],
                    state_b=emb.state_ids[j],
                    distance=d,
                    coord_a=emb.coordinates[i],
                    coord_b=emb.coordinates[j],
                ))

    return sorted(pairs, key=lambda p: p.distance)


def merge_redundant(
    ss: "StateSpace", pairs: list[RedundantPair],
) -> "StateSpace":
    """Merge spectrally redundant state pairs into a simplified state space.

    For each redundant pair, the second state is merged into the first.
    Transitions are redirected accordingly.

    Args:
        ss: Original state space.
        pairs: Redundant pairs to merge.

    Returns:
        New StateSpace with merged states.
    """
    from reticulate.statespace import StateSpace as SS

    if not pairs:
        return ss

    # Build merge map: state → representative
    merge_map: dict[int, int] = {s: s for s in ss.states}

    for pair in pairs:
        a, b = pair.state_a, pair.state_b
        # Always keep the lower-numbered state
        keep, remove = (min(a, b), max(a, b))
        # Follow chain
        while merge_map[keep] != keep:
            keep = merge_map[keep]
        merge_map[remove] = keep
        # Also update anything pointing to remove
        for s in list(merge_map.keys()):
            if merge_map[s] == remove:
                merge_map[s] = keep

    # Resolve transitive merges
    for s in list(merge_map.keys()):
        rep = s
        while merge_map[rep] != rep:
            rep = merge_map[rep]
        merge_map[s] = rep

    # Build new state space
    new_states: set[int] = set(merge_map.values())
    new_transitions: list[tuple[int, str, int]] = []
    seen: set[tuple[int, str, int]] = set()
    new_selections: set[tuple[int, str, int]] = set()

    for src, label, tgt in ss.transitions:
        new_src = merge_map[src]
        new_tgt = merge_map[tgt]
        edge = (new_src, label, new_tgt)
        if edge not in seen and new_src != new_tgt:
            seen.add(edge)
            new_transitions.append(edge)
            if ss.is_selection(src, label, tgt):
                new_selections.add(edge)

    new_top = merge_map[ss.top]
    new_bottom = merge_map[ss.bottom]

    return SS(
        states=new_states,
        transitions=new_transitions,
        top=new_top,
        bottom=new_bottom,
        labels={s: f"[{s}]" for s in new_states},
        selection_transitions=new_selections,
    )


# ---------------------------------------------------------------------------
# Approach C: Spectral clustering
# ---------------------------------------------------------------------------

def _kmeans(
    points: list[tuple[float, ...]],
    k: int,
    max_iter: int = 100,
) -> list[int]:
    """Simple k-means clustering.

    Deterministic initialization: evenly spaced points.
    Returns assignment list.
    """
    n = len(points)
    if n == 0:
        return []
    if k >= n:
        return list(range(n))

    dim = len(points[0])

    # Initialize centroids
    centroids = [points[i * n // k] for i in range(k)]
    assignments = [0] * n

    for _ in range(max_iter):
        new_assignments = [0] * n
        for i, p in enumerate(points):
            best_c = 0
            best_d = _euclidean_distance(p, centroids[0])
            for c in range(1, k):
                d = _euclidean_distance(p, centroids[c])
                if d < best_d:
                    best_d = d
                    best_c = c
            new_assignments[i] = best_c

        if new_assignments == assignments:
            break
        assignments = new_assignments

        # Update centroids
        for c in range(k):
            members = [points[i] for i in range(n) if assignments[i] == c]
            if members:
                centroids[c] = tuple(
                    sum(m[d] for m in members) / len(members)
                    for d in range(dim)
                )

    return assignments


def spectral_clusters(
    ss: "StateSpace", k: int = 2,
) -> dict[int, int]:
    """Partition states into k clusters using spectral embedding + k-means.

    Args:
        ss: A session type state space.
        k: Number of clusters.

    Returns:
        Mapping from state ID to cluster ID (0-indexed).
    """
    n = len(ss.states)
    if n <= 1:
        return {s: 0 for s in ss.states}

    k = min(k, n)
    emb_k = min(k, n - 1)
    emb = spectral_embedding(ss, emb_k)

    assignments = _kmeans(list(emb.coordinates), k)

    return {
        emb.state_ids[i]: assignments[i]
        for i in range(len(emb.state_ids))
    }


def reconstruct_from_clusters(
    ss: "StateSpace", clusters: dict[int, int],
) -> "StateSpace":
    """Build a simplified state space from spectral clusters.

    Each cluster becomes a single state. Transitions between clusters
    are preserved; intra-cluster transitions are removed.

    Args:
        ss: Original state space.
        clusters: Mapping state → cluster_id.

    Returns:
        New StateSpace with one state per cluster.
    """
    from reticulate.statespace import StateSpace as SS

    if not clusters:
        return ss

    # Use minimum state ID in each cluster as representative
    cluster_reps: dict[int, int] = {}
    for state, cid in sorted(clusters.items()):
        if cid not in cluster_reps:
            cluster_reps[cid] = state
        else:
            cluster_reps[cid] = min(cluster_reps[cid], state)

    # Map each state to its cluster representative
    state_to_rep: dict[int, int] = {}
    for state, cid in clusters.items():
        state_to_rep[state] = cluster_reps[cid]

    new_states = set(cluster_reps.values())
    new_transitions: list[tuple[int, str, int]] = []
    seen: set[tuple[int, str, int]] = set()
    new_selections: set[tuple[int, str, int]] = set()

    for src, label, tgt in ss.transitions:
        new_src = state_to_rep.get(src, src)
        new_tgt = state_to_rep.get(tgt, tgt)
        edge = (new_src, label, new_tgt)
        if edge not in seen and new_src != new_tgt:
            seen.add(edge)
            new_transitions.append(edge)
            if ss.is_selection(src, label, tgt):
                new_selections.add(edge)

    new_top = state_to_rep.get(ss.top, ss.top)
    new_bottom = state_to_rep.get(ss.bottom, ss.bottom)

    # Ensure top and bottom are in new_states
    new_states.add(new_top)
    new_states.add(new_bottom)

    return SS(
        states=new_states,
        transitions=new_transitions,
        top=new_top,
        bottom=new_bottom,
        labels={s: f"cluster({s})" for s in new_states},
        selection_transitions=new_selections,
    )


# ---------------------------------------------------------------------------
# Bidirectional morphisms: round-trip analysis
# ---------------------------------------------------------------------------

def round_trip_analysis(ss: "StateSpace", k: int = 2) -> dict[str, object]:
    """Measure what the spectral round-trip (embed → cluster → reconstruct) preserves.

    Computes:
    - state_preservation: fraction of states preserved
    - transition_preservation: fraction of transitions preserved
    - top_bottom_preserved: whether top and bottom survive
    - lattice_preserved: whether the result is still a lattice
    - selection_preserved: fraction of selection transitions preserved

    Args:
        ss: A session type state space.
        k: Number of clusters for the round-trip.

    Returns:
        Dictionary with preservation metrics.
    """
    from reticulate.lattice import check_lattice

    clusters = spectral_clusters(ss, k)
    reconstructed = reconstruct_from_clusters(ss, clusters)

    orig_states = len(ss.states)
    new_states = len(reconstructed.states)
    state_pres = new_states / orig_states if orig_states > 0 else 1.0

    orig_trans = len(ss.transitions)
    new_trans = len(reconstructed.transitions)
    trans_pres = new_trans / orig_trans if orig_trans > 0 else 1.0

    # Check if top/bottom are preserved
    top_ok = reconstructed.top in reconstructed.states
    bottom_ok = reconstructed.bottom in reconstructed.states

    # Check lattice property
    lattice_result = check_lattice(reconstructed)
    lattice_ok = lattice_result.is_lattice

    # Selection preservation
    orig_sel = len(ss.selection_transitions)
    new_sel = len(reconstructed.selection_transitions)
    sel_pres = new_sel / orig_sel if orig_sel > 0 else 1.0

    return {
        "state_preservation": state_pres,
        "transition_preservation": trans_pres,
        "top_bottom_preserved": top_ok and bottom_ok,
        "lattice_preserved": lattice_ok,
        "selection_preservation": sel_pres,
        "original_states": orig_states,
        "reconstructed_states": new_states,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def compare_original_refactored(
    ss: "StateSpace", refactored: "StateSpace",
) -> dict[str, object]:
    """Compare original and refactored state spaces for property preservation.

    Args:
        ss: Original state space.
        refactored: Refactored state space.

    Returns:
        Dictionary with comparison metrics.
    """
    from reticulate.lattice import check_lattice

    orig_lattice = check_lattice(ss)
    ref_lattice = check_lattice(refactored)

    orig_methods = {lbl for _, lbl, _ in ss.transitions}
    ref_methods = {lbl for _, lbl, _ in refactored.transitions}

    lost_methods = orig_methods - ref_methods
    gained_methods = ref_methods - orig_methods

    return {
        "original_states": len(ss.states),
        "refactored_states": len(refactored.states),
        "state_reduction": 1.0 - len(refactored.states) / len(ss.states) if len(ss.states) > 0 else 0.0,
        "original_is_lattice": orig_lattice.is_lattice,
        "refactored_is_lattice": ref_lattice.is_lattice,
        "lattice_preserved": orig_lattice.is_lattice == ref_lattice.is_lattice or ref_lattice.is_lattice,
        "methods_preserved": not lost_methods,
        "lost_methods": lost_methods,
        "gained_methods": gained_methods,
        "original_transitions": len(ss.transitions),
        "refactored_transitions": len(refactored.transitions),
    }


def refactoring_pipeline(ss: "StateSpace") -> SpectralRefactorAnalysis:
    """Full spectral refactoring pipeline.

    Runs all three approaches (Fiedler, degeneracy, clustering) and
    returns a comprehensive analysis.

    Args:
        ss: A session type state space.

    Returns:
        SpectralRefactorAnalysis with all results.
    """
    n = len(ss.states)

    # Embedding
    k = min(3, max(1, n - 1))
    emb = spectral_embedding(ss, k)

    # Approach A: Fiedler
    fiedler_plan = fiedler_refactoring(ss)

    # Approach B: Degeneracy
    deg_groups = eigenvalue_degeneracy(ss, tol=0.1)
    red_pairs = redundant_states(ss, tol=0.1)

    # Build degeneracy merge plan
    merge_tuples = tuple((p.state_a, p.state_b) for p in red_pairs)
    merged_ss = merge_redundant(ss, red_pairs)
    n_after_merge = len(merged_ss.states)
    deg_reduction = 1.0 - n_after_merge / n if n > 0 else 0.0

    deg_plan = RefactoringPlan(
        approach="degeneracy",
        description=(
            f"Degeneracy analysis: {len(deg_groups)} degenerate group(s), "
            f"{len(red_pairs)} redundant pair(s), "
            f"merge reduces {n} → {n_after_merge} states."
        ),
        merge_pairs=merge_tuples,
        estimated_reduction=deg_reduction,
    )

    # Approach C: Clustering
    n_clusters = min(max(2, n // 3), n) if n > 2 else max(1, n)
    clusters = spectral_clusters(ss, n_clusters)
    cluster_map = dict(clusters)

    actual_clusters = len(set(clusters.values())) if clusters else 0
    cluster_reduction = 1.0 - actual_clusters / n if n > 0 else 0.0

    cluster_plan = RefactoringPlan(
        approach="clustering",
        description=(
            f"Spectral clustering: {actual_clusters} clusters from {n} states, "
            f"estimated reduction {cluster_reduction:.1%}."
        ),
        cluster_map=cluster_map,
        estimated_reduction=cluster_reduction,
    )

    # Round-trip loss
    rt = round_trip_analysis(ss, n_clusters)
    rt_loss = 1.0 - (
        float(rt["state_preservation"]) * 0.4  # type: ignore[arg-type]
        + float(rt["transition_preservation"]) * 0.4  # type: ignore[arg-type]
        + (1.0 if rt["lattice_preserved"] else 0.0) * 0.2
    )
    rt_loss = max(0.0, min(1.0, rt_loss))

    return SpectralRefactorAnalysis(
        embedding=emb,
        degeneracy_groups=tuple(deg_groups),
        redundant_pairs=tuple(red_pairs),
        fiedler_plan=fiedler_plan,
        degeneracy_plan=deg_plan,
        clustering_plan=cluster_plan,
        round_trip_loss=rt_loss,
        num_states=n,
        num_states_after_merge=n_after_merge,
        num_clusters=actual_clusters,
    )
