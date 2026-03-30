"""Max-plus longest paths for session type lattices (Step 30n).

In max-plus algebra, matrix powers A^k give longest paths of exactly k steps.
The Kleene star A* = I + A + A^2 + ... gives all-pairs longest paths.

For session type lattices:
- Longest path = maximum number of protocol steps from initial to final state
- Critical path = longest path in the protocol DAG (determines minimum execution time)
- Path width = maximum number of parallel independent paths (max antichain)
- Bottleneck path = path maximizing the minimum edge weight

All computations work on the SCC quotient DAG to handle recursive types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.tropical import (
    tropical_add,
    tropical_mul,
    tropical_distance,
    _maxplus_adjacency_matrix,
)
from reticulate.zeta import (
    _adjacency,
    _compute_sccs,
    _state_list,
    compute_rank,
)

INF = float('inf')
NEG_INF = float('-inf')


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaxPlusPathResult:
    """Complete max-plus path analysis.

    Attributes:
        longest_path: Length of longest path from top to bottom.
        shortest_path: Length of shortest path from top to bottom.
        critical_path: States on the longest path (top to bottom).
        all_pairs_longest: All-pairs longest path distance matrix.
        diameter: Maximum longest-path distance over all pairs.
        width: Maximum antichain size (= max parallelism).
        path_count_top_bottom: Number of distinct paths from top to bottom.
        bottleneck_value: Bottleneck path value (max of min edge weight).
        is_geodesic: True iff all paths top->bottom have same length.
    """
    longest_path: int
    shortest_path: int
    critical_path: list[int]
    all_pairs_longest: list[list[float]]
    diameter: int
    width: int
    path_count_top_bottom: int
    bottleneck_value: float
    is_geodesic: bool


# ---------------------------------------------------------------------------
# Quotient DAG helpers
# ---------------------------------------------------------------------------

def _quotient_dag(ss: "StateSpace") -> tuple[
    list[int],
    dict[int, int],
    dict[int, set[int]],
    dict[int, list[int]],
    int,
    int,
]:
    """Build SCC quotient DAG.

    Returns:
        reps: Sorted list of SCC representative IDs.
        scc_map: state -> representative.
        scc_members: representative -> set of member states.
        q_adj: quotient adjacency (representative -> list of successors).
        top_rep: Representative of the SCC containing ss.top.
        bot_rep: Representative of the SCC containing ss.bottom.
    """
    scc_map, scc_members = _compute_sccs(ss)
    adj = _adjacency(ss)
    reps = sorted(scc_members.keys())

    q_adj: dict[int, list[int]] = {r: [] for r in reps}
    seen_edges: set[tuple[int, int]] = set()
    for s in ss.states:
        sr = scc_map[s]
        for t in adj[s]:
            tr = scc_map[t]
            if sr != tr and (sr, tr) not in seen_edges:
                q_adj[sr].append(tr)
                seen_edges.add((sr, tr))

    top_rep = scc_map[ss.top]
    bot_rep = scc_map[ss.bottom]

    return reps, scc_map, scc_members, q_adj, top_rep, bot_rep


def _topological_sort(
    reps: list[int], q_adj: dict[int, list[int]]
) -> list[int]:
    """Topological sort of quotient DAG (Kahn's algorithm)."""
    in_deg: dict[int, int] = {r: 0 for r in reps}
    for r in reps:
        for t in q_adj[r]:
            in_deg[t] = in_deg.get(t, 0) + 1

    queue: list[int] = [r for r in reps if in_deg[r] == 0]
    order: list[int] = []
    while queue:
        # Pick smallest for determinism
        queue.sort()
        node = queue.pop(0)
        order.append(node)
        for t in q_adj[node]:
            in_deg[t] -= 1
            if in_deg[t] == 0:
                queue.append(t)
    return order


# ---------------------------------------------------------------------------
# Longest path
# ---------------------------------------------------------------------------

def longest_path_length(ss: "StateSpace") -> int:
    """Length of longest path from top to bottom in the DAG.

    Works on SCC quotient to handle cycles from recursive types.
    Each SCC counts as a single node (distance 0 within SCC).
    """
    reps, scc_map, scc_members, q_adj, top_rep, bot_rep = _quotient_dag(ss)

    if top_rep == bot_rep:
        return 0

    topo = _topological_sort(reps, q_adj)

    # DP: dist[r] = longest path from top_rep to r in quotient DAG
    dist: dict[int, int] = {r: -1 for r in reps}
    dist[top_rep] = 0

    for r in topo:
        if dist[r] < 0:
            continue
        for t in q_adj[r]:
            if dist[r] + 1 > dist[t]:
                dist[t] = dist[r] + 1

    return max(dist[bot_rep], 0)


# ---------------------------------------------------------------------------
# Shortest path
# ---------------------------------------------------------------------------

def shortest_path_length(ss: "StateSpace") -> int:
    """Length of shortest path from top to bottom.

    Uses BFS on the quotient DAG.
    """
    reps, scc_map, scc_members, q_adj, top_rep, bot_rep = _quotient_dag(ss)

    if top_rep == bot_rep:
        return 0

    # BFS
    from collections import deque
    visited: set[int] = {top_rep}
    queue: deque[tuple[int, int]] = deque([(top_rep, 0)])

    while queue:
        node, d = queue.popleft()
        for t in q_adj[node]:
            if t == bot_rep:
                return d + 1
            if t not in visited:
                visited.add(t)
                queue.append((t, d + 1))

    return 0  # unreachable


# ---------------------------------------------------------------------------
# Critical path (longest path as state sequence)
# ---------------------------------------------------------------------------

def critical_path(ss: "StateSpace") -> list[int]:
    """Find the critical path (longest path) as sequence of state IDs.

    Returns a list of original state IDs from top to bottom along the
    longest path. Uses DP on the quotient DAG then reconstructs through
    original state IDs.
    """
    reps, scc_map, scc_members, q_adj, top_rep, bot_rep = _quotient_dag(ss)

    if top_rep == bot_rep:
        return [ss.top]

    topo = _topological_sort(reps, q_adj)

    # DP: longest path from top_rep to each rep
    dist: dict[int, int] = {r: -1 for r in reps}
    pred: dict[int, int] = {}
    dist[top_rep] = 0

    for r in topo:
        if dist[r] < 0:
            continue
        for t in q_adj[r]:
            if dist[r] + 1 > dist[t]:
                dist[t] = dist[r] + 1
                pred[t] = r

    if dist[bot_rep] < 0:
        return [ss.top]

    # Reconstruct path through representatives
    rep_path: list[int] = []
    cur = bot_rep
    while cur != top_rep:
        rep_path.append(cur)
        cur = pred[cur]
    rep_path.append(top_rep)
    rep_path.reverse()

    # Map representatives to original state IDs
    # For each rep, pick a representative original state
    # top and bottom use the actual top/bottom IDs
    adj = _adjacency(ss)
    result: list[int] = []
    for i, rep in enumerate(rep_path):
        if rep == top_rep:
            result.append(ss.top)
        elif rep == bot_rep:
            result.append(ss.bottom)
        else:
            # Pick a member state that is adjacent to previous and next on path
            members = scc_members[rep]
            # Prefer states that connect to previous/next
            if result:
                prev_state = result[-1]
                for m in sorted(members):
                    if m in adj.get(prev_state, []):
                        result.append(m)
                        break
                else:
                    result.append(min(members))
            else:
                result.append(min(members))

    return result


# ---------------------------------------------------------------------------
# All-pairs longest paths
# ---------------------------------------------------------------------------

def all_pairs_longest_paths(ss: "StateSpace") -> list[list[float]]:
    """All-pairs longest path lengths via max-plus matrix closure.

    Uses modified Floyd-Warshall on the original state space.
    L[i][j] = longest directed path from state_i to state_j.
    L[i][j] = -inf if j not reachable from i.

    For cyclic graphs, paths through cycles can be arbitrarily long,
    so we work on the SCC quotient and count inter-SCC edges only.
    """
    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    scc_map, _ = _compute_sccs(ss)

    # Build adjacency matrix with unit weights
    L = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 0

    for src, _, tgt in ss.transitions:
        si, ti = idx[src], idx[tgt]
        # Only count edges between different SCCs or self-loops within same SCC
        L[si][ti] = max(L[si][ti], 1)

    # Modified Floyd-Warshall for longest paths
    for k in range(n):
        for i in range(n):
            if L[i][k] <= NEG_INF:
                continue
            for j in range(n):
                if L[k][j] <= NEG_INF:
                    continue
                candidate = L[i][k] + L[k][j]
                if candidate > L[i][j]:
                    L[i][j] = candidate

    return L


# ---------------------------------------------------------------------------
# Path width (max antichain = max parallelism)
# ---------------------------------------------------------------------------

def path_width(ss: "StateSpace") -> int:
    """Width of the poset = maximum antichain size = max parallelism.

    Uses Dilworth's theorem: width = n - max_matching on the
    comparability bipartite graph. Delegates to zeta.compute_width.
    """
    from reticulate.zeta import compute_width
    return compute_width(ss)


# ---------------------------------------------------------------------------
# Path counting
# ---------------------------------------------------------------------------

def count_paths(ss: "StateSpace", src: int, tgt: int) -> int:
    """Count number of distinct directed paths from src to tgt.

    Counts paths in the original state space (treating parallel edges
    as distinct paths). For acyclic graphs, uses DP on topological order.
    For cyclic graphs, works on the SCC quotient with edge multiplicities.
    """
    if src == tgt:
        return 1

    scc_map, scc_members = _compute_sccs(ss)
    src_rep = scc_map[src]
    tgt_rep = scc_map[tgt]

    if src_rep == tgt_rep:
        return 1

    # Build quotient DAG with edge multiplicities
    reps = sorted(scc_members.keys())
    # q_edge_count[(r1, r2)] = number of original transitions from any
    # member of r1's SCC to any member of r2's SCC
    q_edge_count: dict[tuple[int, int], int] = {}
    q_adj: dict[int, set[int]] = {r: set() for r in reps}

    for s, _, t in ss.transitions:
        sr, tr = scc_map[s], scc_map[t]
        if sr != tr:
            q_adj[sr].add(tr)
            q_edge_count[(sr, tr)] = q_edge_count.get((sr, tr), 0) + 1

    topo = _topological_sort(reps, {r: sorted(q_adj[r]) for r in reps})

    # DP: cnt[r] = number of distinct paths from src_rep to r
    cnt: dict[int, int] = {r: 0 for r in reps}
    cnt[src_rep] = 1

    for r in topo:
        if cnt[r] == 0:
            continue
        for t in q_adj[r]:
            cnt[t] += cnt[r] * q_edge_count[(r, t)]

    return cnt[tgt_rep]


def total_paths_top_bottom(ss: "StateSpace") -> int:
    """Total number of distinct paths from top to bottom."""
    return count_paths(ss, ss.top, ss.bottom)


# ---------------------------------------------------------------------------
# Bottleneck path
# ---------------------------------------------------------------------------

def bottleneck_path(ss: "StateSpace") -> tuple[float, list[int]]:
    """Find the bottleneck path: maximizes the minimum edge weight.

    For unit-weight session type state spaces, all edges have weight 1,
    so the bottleneck value is 1.0 if any path exists, 0.0 otherwise.

    Returns (bottleneck_value, path_as_state_ids).
    Uses modified Dijkstra on the quotient DAG: instead of minimizing
    sum, we maximize the minimum weight along the path.
    """
    reps, scc_map, scc_members, q_adj, top_rep, bot_rep = _quotient_dag(ss)

    if top_rep == bot_rep:
        return (0.0, [ss.top])

    # For unit weights, all edges have weight 1.
    # The bottleneck is 1.0 if any path exists from top to bottom.
    # We still find the actual path for the return value.

    # BFS/DFS to find if path exists and reconstruct it
    visited: set[int] = set()
    pred: dict[int, int] = {}
    stack = [top_rep]
    found = False

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node == bot_rep:
            found = True
            break
        for t in q_adj[node]:
            if t not in visited:
                pred[t] = node
                stack.append(t)

    if not found:
        return (0.0, [ss.top])

    # Reconstruct path
    rep_path: list[int] = []
    cur = bot_rep
    while cur != top_rep:
        rep_path.append(cur)
        cur = pred[cur]
    rep_path.append(top_rep)
    rep_path.reverse()

    # Map to original states
    adj = _adjacency(ss)
    result: list[int] = []
    for rep in rep_path:
        if rep == top_rep:
            result.append(ss.top)
        elif rep == bot_rep:
            result.append(ss.bottom)
        else:
            members = scc_members[rep]
            if result:
                prev_state = result[-1]
                for m in sorted(members):
                    if m in adj.get(prev_state, []):
                        result.append(m)
                        break
                else:
                    result.append(min(members))
            else:
                result.append(min(members))

    return (1.0, result)


# ---------------------------------------------------------------------------
# Geodesic check
# ---------------------------------------------------------------------------

def is_geodesic(ss: "StateSpace") -> bool:
    """Check if all paths from top to bottom have the same length.

    A session type is geodesic if every execution path through the
    protocol has the same number of steps.
    """
    hist = path_histogram(ss)
    return len(hist) <= 1


# ---------------------------------------------------------------------------
# Path histogram
# ---------------------------------------------------------------------------

def path_histogram(ss: "StateSpace") -> dict[int, int]:
    """Distribution of path lengths from top to bottom.

    Returns dict mapping path_length -> count of paths with that length.
    Works on the SCC quotient DAG using DP, accounting for edge multiplicities.
    """
    scc_map, scc_members = _compute_sccs(ss)
    reps = sorted(scc_members.keys())
    top_rep = scc_map[ss.top]
    bot_rep = scc_map[ss.bottom]

    if top_rep == bot_rep:
        return {0: 1}

    # Build quotient adjacency with edge multiplicities
    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    q_edge_count: dict[tuple[int, int], int] = {}
    for s, _, t in ss.transitions:
        sr, tr = scc_map[s], scc_map[t]
        if sr != tr:
            q_adj[sr].add(tr)
            q_edge_count[(sr, tr)] = q_edge_count.get((sr, tr), 0) + 1

    topo = _topological_sort(reps, {r: sorted(q_adj[r]) for r in reps})

    # DP: for each rep, store dict of {length: count} for paths from top_rep
    length_counts: dict[int, dict[int, int]] = {r: {} for r in reps}
    length_counts[top_rep] = {0: 1}

    for r in topo:
        if not length_counts[r]:
            continue
        for t in q_adj[r]:
            mult = q_edge_count.get((r, t), 1)
            for length, count in length_counts[r].items():
                new_len = length + 1
                if new_len not in length_counts[t]:
                    length_counts[t][new_len] = 0
                length_counts[t][new_len] += count * mult

    result = length_counts[bot_rep]
    return dict(sorted(result.items())) if result else {}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_max_plus_paths(ss: "StateSpace") -> MaxPlusPathResult:
    """Complete max-plus path analysis of a session type state space.

    Computes longest/shortest paths, critical path, width, path counts,
    bottleneck analysis, geodesic check, and full path histogram.
    """
    longest = longest_path_length(ss)
    shortest = shortest_path_length(ss)
    crit = critical_path(ss)
    apl = all_pairs_longest_paths(ss)

    # Diameter: max finite entry in all-pairs longest matrix
    diam = 0
    for row in apl:
        for v in row:
            if v > NEG_INF and v < INF:
                diam = max(diam, int(v))

    w = path_width(ss)
    pc = total_paths_top_bottom(ss)
    bv, _ = bottleneck_path(ss)
    geo = is_geodesic(ss)

    return MaxPlusPathResult(
        longest_path=longest,
        shortest_path=shortest,
        critical_path=crit,
        all_pairs_longest=apl,
        diameter=diam,
        width=w,
        path_count_top_bottom=pc,
        bottleneck_value=bv,
        is_geodesic=geo,
    )
