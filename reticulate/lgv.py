"""LGV Lemma analysis for session type lattices (Step 30ac).

The Lindström–Gessel–Viennot (LGV) Lemma relates the determinant of a
path-count matrix to the signed count of non-intersecting path systems
in a directed acyclic graph.

For a DAG with designated sources s₁, …, sₙ and sinks t₁, …, tₙ, define
the path-count matrix M where M[i,j] = number of directed paths from sᵢ
to tⱼ.  The LGV Lemma states:

    det(M) = Σ_{σ ∈ Sₙ} sgn(σ) · (# path systems from (s₁,…,sₙ) to (t_{σ(1)},…,t_{σ(n)}))

When the DAG is *planar* (or more generally has the Lindström property),
only the identity permutation contributes non-zero terms to the sum, so
det(M) equals the number of non-intersecting path systems.

Session type state spaces — after SCC quotient — are DAGs.  This module
applies the LGV Lemma to these DAGs, using rank layers to identify
natural source and sink sets.

All computations use exact integer arithmetic (no floating point).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _adjacency,
    _compute_sccs,
    _covering_relation,
    _reachability,
    _state_list,
    compute_rank,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LGVResult:
    """Complete LGV Lemma analysis for a session type state space.

    Attributes:
        path_matrix: M[i][j] = number of directed paths from source_i to sink_j.
        determinant: det(M), computed via the Bareiss algorithm.
        num_sources: Number of source states used.
        num_sinks: Number of sink states used.
        sources: List of source state IDs.
        sinks: List of sink state IDs.
        lgv_verified: True iff det(M) equals the signed non-intersecting count.
        num_non_intersecting: Count of non-intersecting path systems (identity perm).
    """
    path_matrix: list[list[int]]
    determinant: int
    num_sources: int
    num_sinks: int
    sources: list[int]
    sinks: list[int]
    lgv_verified: bool
    num_non_intersecting: int


# ---------------------------------------------------------------------------
# Path counting (DP on DAG)
# ---------------------------------------------------------------------------

def _topological_order(adj: dict[int, list[int]], states: set[int]) -> list[int]:
    """Compute a topological ordering of states using Kahn's algorithm.

    Returns a list of states in topological order (sources first).
    Ties are broken by state ID for determinism.
    """
    in_degree: dict[int, int] = {s: 0 for s in states}
    for s in states:
        for t in adj.get(s, []):
            if t in states:
                in_degree[t] = in_degree.get(t, 0) + 1

    import heapq
    queue: list[int] = []
    for s in states:
        if in_degree[s] == 0:
            heapq.heappush(queue, s)

    order: list[int] = []
    while queue:
        u = heapq.heappop(queue)
        order.append(u)
        for v in adj.get(u, []):
            if v in states:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    heapq.heappush(queue, v)

    return order


def _count_paths_dag(
    adj: dict[int, list[int]],
    states: set[int],
    src: int,
    tgt: int,
) -> int:
    """Count directed paths from src to tgt in a DAG using DP.

    Uses reverse topological order: dp[v] = number of paths from v to tgt.
    dp[tgt] = 1, dp[v] = sum(dp[w] for w in adj[v]).
    """
    if src not in states or tgt not in states:
        return 0

    topo = _topological_order(adj, states)
    # Process in reverse topological order
    dp: dict[int, int] = {s: 0 for s in states}
    dp[tgt] = 1
    for v in reversed(topo):
        if v == tgt:
            continue
        total = 0
        for w in adj.get(v, []):
            if w in states:
                total += dp[w]
        dp[v] = total

    return dp.get(src, 0)


def _count_paths(ss: "StateSpace", src: int, tgt: int) -> int:
    """Count directed paths from src to tgt in the state space.

    For state spaces with cycles (from recursion), first quotients by SCCs
    to obtain a DAG, then counts paths on the quotient.
    """
    scc_map, scc_members = _compute_sccs(ss)
    adj = _adjacency(ss)

    # Build quotient DAG
    reps = set(scc_members.keys())
    q_adj: dict[int, list[int]] = {r: [] for r in reps}
    seen_edges: set[tuple[int, int]] = set()
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr and (sr, tr) not in seen_edges:
                q_adj[sr].append(tr)
                seen_edges.add((sr, tr))

    src_rep = scc_map.get(src, src)
    tgt_rep = scc_map.get(tgt, tgt)

    return _count_paths_dag(q_adj, reps, src_rep, tgt_rep)


# ---------------------------------------------------------------------------
# Path count matrix
# ---------------------------------------------------------------------------

def path_count_matrix(
    ss: "StateSpace",
    sources: list[int],
    sinks: list[int],
) -> list[list[int]]:
    """Build path-count matrix M[i][j] = #paths from sources[i] to sinks[j].

    Args:
        ss: The state space (may have cycles from recursion).
        sources: List of source state IDs.
        sinks: List of sink state IDs.

    Returns:
        An n×m matrix where n = len(sources), m = len(sinks).
    """
    n = len(sources)
    m = len(sinks)
    M: list[list[int]] = [[0] * m for _ in range(n)]
    for i, s in enumerate(sources):
        for j, t in enumerate(sinks):
            M[i][j] = _count_paths(ss, s, t)
    return M


# ---------------------------------------------------------------------------
# Determinant (Bareiss algorithm — exact integer arithmetic)
# ---------------------------------------------------------------------------

def _determinant(matrix: list[list[int]]) -> int:
    """Compute determinant of an integer matrix using the Bareiss algorithm.

    Bareiss algorithm performs exact integer division at each step,
    avoiding floating-point errors entirely.

    Returns 0 for empty matrices.
    """
    n = len(matrix)
    if n == 0:
        return 0
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Work on a copy
    M = [row[:] for row in matrix]
    sign = 1

    for k in range(n - 1):
        # Pivot: find non-zero entry in column k, rows k..n-1
        pivot_row = -1
        for i in range(k, n):
            if M[i][k] != 0:
                pivot_row = i
                break
        if pivot_row == -1:
            return 0  # Singular

        if pivot_row != k:
            M[k], M[pivot_row] = M[pivot_row], M[k]
            sign = -sign

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = M[k][k] * M[i][j] - M[i][k] * M[k][j]
                if k > 0:
                    M[i][j] //= M[k - 1][k - 1] if k > 0 else 1
            M[i][k] = 0

    return sign * M[n - 1][n - 1]


def _determinant_safe(matrix: list[list[int]]) -> int:
    """Compute determinant using Leibniz formula for small matrices,
    Bareiss for larger ones.

    For matrices up to 6x6, uses the permutation expansion directly
    to avoid any integer division issues with Bareiss.
    """
    n = len(matrix)
    if n == 0:
        return 0
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if n <= 6:
        return _determinant_leibniz(matrix)
    return _determinant(matrix)


def _determinant_leibniz(matrix: list[list[int]]) -> int:
    """Compute determinant using Leibniz formula (permutation expansion).

    Exact for any integer matrix, but O(n!) complexity.
    Only suitable for small matrices (n ≤ 8 or so).
    """
    n = len(matrix)
    if n == 0:
        return 0

    total = 0
    for perm in permutations(range(n)):
        # Compute sign of permutation
        sgn = _perm_sign(perm)
        prod = 1
        for i, j in enumerate(perm):
            prod *= matrix[i][j]
            if prod == 0:
                break
        total += sgn * prod
    return total


def _perm_sign(perm: tuple[int, ...]) -> int:
    """Compute the sign of a permutation: +1 for even, -1 for odd."""
    n = len(perm)
    visited = [False] * n
    sign = 1
    for i in range(n):
        if visited[i]:
            continue
        j = i
        cycle_len = 0
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        if cycle_len % 2 == 0:
            sign = -sign
    return sign


# ---------------------------------------------------------------------------
# LGV determinant
# ---------------------------------------------------------------------------

def lgv_determinant(
    ss: "StateSpace",
    sources: list[int],
    sinks: list[int],
) -> int:
    """Compute det(M) where M is the path-count matrix.

    By the LGV Lemma, this equals the signed count of non-intersecting
    path systems from sources to sinks.

    Args:
        ss: The state space.
        sources: Source state IDs (must have same length as sinks).
        sinks: Sink state IDs.

    Returns:
        The determinant of the path-count matrix.

    Raises:
        ValueError: If len(sources) != len(sinks).
    """
    if len(sources) != len(sinks):
        raise ValueError(
            f"sources and sinks must have same length, "
            f"got {len(sources)} and {len(sinks)}"
        )
    M = path_count_matrix(ss, sources, sinks)
    return _determinant_safe(M)


# ---------------------------------------------------------------------------
# Rank layers (source/sink detection)
# ---------------------------------------------------------------------------

def find_rank_layers(ss: "StateSpace") -> dict[int, list[int]]:
    """Find states at each rank level.

    Returns a dict mapping rank → sorted list of state IDs at that rank.
    Rank 0 = bottom states (sinks), max rank = top states (sources).
    """
    ranks = compute_rank(ss)
    layers: dict[int, list[int]] = {}
    for state, rank in ranks.items():
        if rank not in layers:
            layers[rank] = []
        layers[rank].append(state)

    for rank in layers:
        layers[rank].sort()

    return layers


# ---------------------------------------------------------------------------
# Non-intersecting path systems (direct enumeration)
# ---------------------------------------------------------------------------

def _enumerate_path_systems(
    ss: "StateSpace",
    sources: list[int],
    sinks: list[int],
) -> list[tuple[int, list[list[int]]]]:
    """Enumerate all path systems from sources to sinks with signs.

    Returns list of (sign, paths) where sign is the permutation sign
    and paths[i] is the path from sources[i] to sinks[perm[i]].

    A path system is non-intersecting if no two paths share a vertex.
    """
    n = len(sources)
    if n == 0:
        return [(1, [])]

    scc_map, scc_members = _compute_sccs(ss)
    adj = _adjacency(ss)

    # Build quotient DAG
    reps = set(scc_members.keys())
    q_adj: dict[int, list[int]] = {r: [] for r in reps}
    seen_edges: set[tuple[int, int]] = set()
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr and (sr, tr) not in seen_edges:
                q_adj[sr].append(tr)
                seen_edges.add((sr, tr))

    def all_paths(src_rep: int, tgt_rep: int) -> list[list[int]]:
        """Find all directed paths from src_rep to tgt_rep in quotient DAG."""
        if src_rep == tgt_rep:
            return [[src_rep]]
        result: list[list[int]] = []
        stack: list[tuple[int, list[int]]] = [(src_rep, [src_rep])]
        while stack:
            u, path = stack.pop()
            for v in q_adj.get(u, []):
                if v == tgt_rep:
                    result.append(path + [v])
                elif v not in path:  # avoid revisiting in path
                    stack.append((v, path + [v]))
        return result

    systems: list[tuple[int, list[list[int]]]] = []

    for perm in permutations(range(n)):
        sgn = _perm_sign(perm)
        # For each assignment of sources to sinks via perm,
        # find all path combinations
        src_reps = [scc_map.get(sources[i], sources[i]) for i in range(n)]
        tgt_reps = [scc_map.get(sinks[perm[i]], sinks[perm[i]]) for i in range(n)]

        paths_per_pair: list[list[list[int]]] = []
        for i in range(n):
            paths_per_pair.append(all_paths(src_reps[i], tgt_reps[i]))

        # Generate all combinations
        def combine(idx: int, current: list[list[int]]) -> None:
            if idx == n:
                systems.append((sgn, [p[:] for p in current]))
                return
            for p in paths_per_pair[idx]:
                current.append(p)
                combine(idx + 1, current)
                current.pop()

        combine(0, [])

    return systems


def non_intersecting_count(
    ss: "StateSpace",
    sources: list[int],
    sinks: list[int],
) -> int:
    """Count non-intersecting path systems from sources to sinks.

    A path system is non-intersecting if no two paths share a vertex
    (except possibly at endpoints if a source coincides with a sink).

    This counts systems for the identity permutation only (sources[i] → sinks[i]).
    For the full LGV signed sum, use lgv_determinant().

    Returns:
        Number of vertex-disjoint path systems (identity permutation).
    """
    n = len(sources)
    if n == 0:
        return 1
    if n == 1:
        return _count_paths(ss, sources[0], sinks[0])

    scc_map, scc_members = _compute_sccs(ss)
    adj = _adjacency(ss)

    # Build quotient DAG
    reps = set(scc_members.keys())
    q_adj: dict[int, list[int]] = {r: [] for r in reps}
    seen_edges: set[tuple[int, int]] = set()
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr and (sr, tr) not in seen_edges:
                q_adj[sr].append(tr)
                seen_edges.add((sr, tr))

    def all_paths(src_rep: int, tgt_rep: int) -> list[list[int]]:
        """All directed paths from src_rep to tgt_rep."""
        if src_rep == tgt_rep:
            return [[src_rep]]
        result: list[list[int]] = []
        stack: list[tuple[int, list[int]]] = [(src_rep, [src_rep])]
        while stack:
            u, path = stack.pop()
            for v in q_adj.get(u, []):
                if v == tgt_rep:
                    result.append(path + [v])
                elif v not in path:
                    stack.append((v, path + [v]))
        return result

    src_reps = [scc_map.get(sources[i], sources[i]) for i in range(n)]
    tgt_reps = [scc_map.get(sinks[i], sinks[i]) for i in range(n)]

    paths_per_pair: list[list[list[int]]] = []
    for i in range(n):
        paths_per_pair.append(all_paths(src_reps[i], tgt_reps[i]))

    count = 0

    def _count_non_intersecting(
        idx: int,
        used_vertices: set[int],
    ) -> int:
        if idx == n:
            return 1
        total = 0
        for p in paths_per_pair[idx]:
            # Interior vertices (exclude endpoints for this path)
            interior = set(p[1:-1]) if len(p) > 2 else set()
            all_verts = set(p)
            # Check if any vertex in this path collides with used vertices
            if all_verts & used_vertices:
                continue
            total += _count_non_intersecting(idx + 1, used_vertices | all_verts)
        return total

    return _count_non_intersecting(0, set())


def _signed_non_intersecting_count(
    ss: "StateSpace",
    sources: list[int],
    sinks: list[int],
) -> int:
    """Signed count of non-intersecting path systems (full LGV sum).

    Sums over all permutations σ: sgn(σ) × (# non-intersecting systems
    from sources[i] to sinks[σ(i)]).
    """
    n = len(sources)
    if n == 0:
        return 1

    scc_map, scc_members = _compute_sccs(ss)
    adj = _adjacency(ss)

    reps = set(scc_members.keys())
    q_adj: dict[int, list[int]] = {r: [] for r in reps}
    seen_edges: set[tuple[int, int]] = set()
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr and (sr, tr) not in seen_edges:
                q_adj[sr].append(tr)
                seen_edges.add((sr, tr))

    def all_paths(src_rep: int, tgt_rep: int) -> list[list[int]]:
        if src_rep == tgt_rep:
            return [[src_rep]]
        result: list[list[int]] = []
        stack: list[tuple[int, list[int]]] = [(src_rep, [src_rep])]
        while stack:
            u, path = stack.pop()
            for v in q_adj.get(u, []):
                if v == tgt_rep:
                    result.append(path + [v])
                elif v not in path:
                    stack.append((v, path + [v]))
        return result

    total = 0

    for perm in permutations(range(n)):
        sgn = _perm_sign(perm)

        src_reps = [scc_map.get(sources[i], sources[i]) for i in range(n)]
        tgt_reps = [scc_map.get(sinks[perm[i]], sinks[perm[i]]) for i in range(n)]

        paths_per_pair: list[list[list[int]]] = []
        for i in range(n):
            paths_per_pair.append(all_paths(src_reps[i], tgt_reps[i]))

        def _count_ni(idx: int, used: set[int], ppp: list[list[list[int]]]) -> int:
            if idx == n:
                return 1
            t = 0
            for p in ppp[idx]:
                verts = set(p)
                if verts & used:
                    continue
                t += _count_ni(idx + 1, used | verts, ppp)
            return t

        total += sgn * _count_ni(0, set(), paths_per_pair)

    return total


# ---------------------------------------------------------------------------
# LGV verification
# ---------------------------------------------------------------------------

def verify_lgv(
    ss: "StateSpace",
    sources: list[int],
    sinks: list[int],
) -> bool:
    """Verify the LGV Lemma: det(M) == signed non-intersecting count.

    Args:
        ss: The state space.
        sources: Source state IDs.
        sinks: Sink state IDs (same length as sources).

    Returns:
        True if the LGV identity holds.

    Raises:
        ValueError: If len(sources) != len(sinks).
    """
    if len(sources) != len(sinks):
        raise ValueError(
            f"sources and sinks must have same length, "
            f"got {len(sources)} and {len(sinks)}"
        )
    det = lgv_determinant(ss, sources, sinks)
    signed_ni = _signed_non_intersecting_count(ss, sources, sinks)
    return det == signed_ni


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_lgv(ss: "StateSpace") -> LGVResult:
    """Full LGV analysis: auto-detect sources/sinks, compute everything.

    Sources are states at the maximum rank (top layer).
    Sinks are states at the minimum rank (rank 0, bottom layer).

    For typical session types with a single top and single bottom, this
    gives a 1x1 matrix.  Parallel compositions and branching create
    richer rank structures with multiple states per layer.
    """
    layers = find_rank_layers(ss)
    if not layers:
        return LGVResult(
            path_matrix=[],
            determinant=0,
            num_sources=0,
            num_sinks=0,
            sources=[],
            sinks=[],
            lgv_verified=True,
            num_non_intersecting=0,
        )

    max_rank = max(layers.keys())
    min_rank = min(layers.keys())

    sources = layers[max_rank]
    sinks = layers[min_rank]

    # LGV requires square matrix: take min(n_sources, n_sinks)
    n = min(len(sources), len(sinks))
    sources_used = sources[:n]
    sinks_used = sinks[:n]

    M = path_count_matrix(ss, sources_used, sinks_used)

    if n == 0:
        return LGVResult(
            path_matrix=M,
            determinant=0,
            num_sources=0,
            num_sinks=0,
            sources=[],
            sinks=[],
            lgv_verified=True,
            num_non_intersecting=0,
        )

    det = _determinant_safe(M)
    ni = _signed_non_intersecting_count(ss, sources_used, sinks_used)
    verified = (det == ni)

    return LGVResult(
        path_matrix=M,
        determinant=det,
        num_sources=n,
        num_sinks=n,
        sources=sources_used,
        sinks=sinks_used,
        lgv_verified=verified,
        num_non_intersecting=ni,
    )
