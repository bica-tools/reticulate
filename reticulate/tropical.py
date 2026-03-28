"""Tropical (min-plus / max-plus) algebra for session type lattices (Steps 30k-30n).

Tropical algebra replaces (+, ×) with (min, +) or (max, +), providing
shortest/longest path analysis through matrix operations:

- **Step 30k**: Tropical distance matrix (all-pairs shortest paths)
- **Step 30l**: Tropical eigenvalue (max cycle mean / critical circuit)
- **Step 30m**: Tropical determinant (permanent over min-plus)
- **Step 30n**: Max-plus longest paths (protocol depth analysis)

For session type lattices:
- Shortest paths = minimum transitions between states
- Longest paths = maximum protocol depth
- Tropical eigenvalue = throughput bottleneck (cycle mean)
- Tropical determinant = optimal assignment weight
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import _state_list, _adjacency

INF = float('inf')


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TropicalResult:
    """Complete tropical algebra analysis.

    Attributes:
        num_states: Number of states.
        distance_matrix: All-pairs shortest path distances.
        diameter: Maximum finite distance (graph diameter).
        tropical_eigenvalue: Maximum cycle mean (0 if acyclic).
        tropical_determinant: Min-plus permanent.
        longest_path_top_bottom: Longest directed path from top to bottom.
        shortest_path_top_bottom: Shortest directed path from top to bottom.
        eccentricity: Maximum distance from each state.
        radius: Minimum eccentricity.
        center: States with eccentricity = radius.
    """
    num_states: int
    distance_matrix: list[list[float]]
    diameter: int
    tropical_eigenvalue: float
    tropical_determinant: float
    longest_path_top_bottom: int
    shortest_path_top_bottom: int
    eccentricity: dict[int, int]
    radius: int
    center: list[int]


# ---------------------------------------------------------------------------
# Step 30k: Tropical distance matrix (shortest paths)
# ---------------------------------------------------------------------------

def tropical_distance(ss: "StateSpace") -> list[list[float]]:
    """All-pairs shortest path matrix via Floyd-Warshall.

    D[i][j] = length of shortest directed path from state_i to state_j.
    D[i][j] = INF if j not reachable from i.
    """
    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    D = [[INF] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = 0

    for src, _, tgt in ss.transitions:
        si, ti = idx[src], idx[tgt]
        D[si][ti] = min(D[si][ti], 1)  # unit edge weights

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i][k] + D[k][j] < D[i][j]:
                    D[i][j] = D[i][k] + D[k][j]

    return D


def diameter(ss: "StateSpace") -> int:
    """Graph diameter: maximum finite distance between any two reachable states."""
    D = tropical_distance(ss)
    n = len(D)
    max_dist = 0
    for i in range(n):
        for j in range(n):
            if D[i][j] < INF:
                max_dist = max(max_dist, int(D[i][j]))
    return max_dist


def eccentricity(ss: "StateSpace") -> dict[int, int]:
    """Eccentricity of each state: max distance to any reachable state."""
    states = _state_list(ss)
    D = tropical_distance(ss)
    n = len(states)
    ecc: dict[int, int] = {}
    for i in range(n):
        max_d = 0
        for j in range(n):
            if D[i][j] < INF:
                max_d = max(max_d, int(D[i][j]))
        ecc[states[i]] = max_d
    return ecc


def radius(ss: "StateSpace") -> int:
    """Graph radius: minimum eccentricity."""
    ecc = eccentricity(ss)
    return min(ecc.values()) if ecc else 0


def center(ss: "StateSpace") -> list[int]:
    """Graph center: states with eccentricity = radius."""
    ecc = eccentricity(ss)
    r = min(ecc.values()) if ecc else 0
    return sorted(s for s, e in ecc.items() if e == r)


# ---------------------------------------------------------------------------
# Step 30l: Tropical eigenvalue (max cycle mean)
# ---------------------------------------------------------------------------

def tropical_eigenvalue(ss: "StateSpace") -> float:
    """Maximum cycle mean (tropical eigenvalue).

    λ_trop = max over all cycles C of (weight(C) / length(C)).
    For unit-weight edges: λ_trop = 1 if any cycle exists, 0 if acyclic.

    Uses Karp's algorithm on the distance matrix.
    """
    states = _state_list(ss)
    n = len(states)
    if n == 0:
        return 0.0

    adj = _adjacency(ss)
    idx = {s: i for i, s in enumerate(states)}

    # Karp's algorithm: compute D[k][v] = min weight path of length k ending at v
    # Then λ = max_v min_{0≤k<n} (D[n][v] - D[k][v]) / (n - k)
    D_karp = [[INF] * n for _ in range(n + 1)]

    # Source: try each state as source
    max_mean = 0.0
    for src_i in range(n):
        for k in range(n + 1):
            for v in range(n):
                D_karp[k][v] = INF
        D_karp[0][src_i] = 0

        for k in range(n):
            for v_idx in range(n):
                if D_karp[k][v_idx] >= INF:
                    continue
                v = states[v_idx]
                for w in adj[v]:
                    w_idx = idx[w]
                    new_dist = D_karp[k][v_idx] + 1  # unit weight
                    if new_dist < D_karp[k + 1][w_idx]:
                        D_karp[k + 1][w_idx] = new_dist

        # Karp's formula for max cycle mean reachable from src
        for v in range(n):
            if D_karp[n][v] >= INF:
                continue
            min_ratio = INF
            for k in range(n):
                if D_karp[k][v] < INF:
                    ratio = (D_karp[n][v] - D_karp[k][v]) / (n - k)
                    min_ratio = min(min_ratio, ratio)
            if min_ratio < INF:
                max_mean = max(max_mean, min_ratio)

    return max_mean


# ---------------------------------------------------------------------------
# Step 30m: Tropical determinant (min-plus permanent)
# ---------------------------------------------------------------------------

def tropical_determinant(ss: "StateSpace") -> float:
    """Tropical determinant: min-plus permanent of the distance matrix.

    tdet(D) = min over permutations σ of Σ_i D[i][σ(i)].

    This is the weight of an optimal assignment — the minimum total
    cost of assigning each state to a unique target state.

    Uses the Hungarian algorithm approximation for small matrices.
    """
    D = tropical_distance(ss)
    n = len(D)
    if n == 0:
        return 0.0
    if n == 1:
        return D[0][0]

    # For small n, try all permutations (n! feasible for n ≤ 10)
    if n <= 10:
        return _min_perm_sum(D, n)

    # For larger n, use greedy approximation
    return _greedy_assignment(D, n)


def _min_perm_sum(D: list[list[float]], n: int) -> float:
    """Find minimum-weight perfect matching (brute force for small n)."""
    min_sum = INF

    def _search(row: int, used: int, current_sum: float) -> None:
        nonlocal min_sum
        if current_sum >= min_sum:
            return  # prune
        if row == n:
            min_sum = min(min_sum, current_sum)
            return
        for col in range(n):
            if used & (1 << col):
                continue
            _search(row + 1, used | (1 << col), current_sum + D[row][col])

    _search(0, 0, 0.0)
    return min_sum


def _greedy_assignment(D: list[list[float]], n: int) -> float:
    """Greedy approximation for minimum assignment."""
    used_cols: set[int] = set()
    total = 0.0
    for i in range(n):
        best_j = -1
        best_val = INF
        for j in range(n):
            if j not in used_cols and D[i][j] < best_val:
                best_val = D[i][j]
                best_j = j
        if best_j >= 0:
            used_cols.add(best_j)
            total += best_val
        else:
            total += INF
    return total


# ---------------------------------------------------------------------------
# Step 30n: Max-plus longest paths
# ---------------------------------------------------------------------------

def longest_path_matrix(ss: "StateSpace") -> list[list[float]]:
    """All-pairs longest path matrix (max-plus closure).

    L[i][j] = length of longest directed path from state_i to state_j.
    L[i][j] = -INF if j not reachable from i.
    Only well-defined for acyclic graphs (DAGs).
    """
    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    NEG_INF = float('-inf')
    L = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 0

    for src, _, tgt in ss.transitions:
        si, ti = idx[src], idx[tgt]
        L[si][ti] = max(L[si][ti], 1)

    # Modified Floyd-Warshall for longest paths (max-plus)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if L[i][k] > NEG_INF and L[k][j] > NEG_INF:
                    candidate = L[i][k] + L[k][j]
                    if candidate > L[i][j]:
                        L[i][j] = candidate

    return L


def longest_path_top_bottom(ss: "StateSpace") -> int:
    """Longest directed path from top to bottom."""
    states = _state_list(ss)
    idx = {s: i for i, s in enumerate(states)}
    L = longest_path_matrix(ss)
    top_i = idx[ss.top]
    bot_i = idx[ss.bottom]
    val = L[top_i][bot_i]
    return int(val) if val > float('-inf') else 0


def shortest_path_top_bottom(ss: "StateSpace") -> int:
    """Shortest directed path from top to bottom."""
    states = _state_list(ss)
    idx = {s: i for i, s in enumerate(states)}
    D = tropical_distance(ss)
    top_i = idx[ss.top]
    bot_i = idx[ss.bottom]
    val = D[top_i][bot_i]
    return int(val) if val < INF else 0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_tropical(ss: "StateSpace") -> TropicalResult:
    """Complete tropical algebra analysis."""
    D = tropical_distance(ss)
    states = _state_list(ss)
    n = len(states)

    diam = diameter(ss)
    trop_eig = tropical_eigenvalue(ss)
    trop_det = tropical_determinant(ss)
    longest = longest_path_top_bottom(ss)
    shortest = shortest_path_top_bottom(ss)
    ecc = eccentricity(ss)
    rad = radius(ss)
    ctr = center(ss)

    return TropicalResult(
        num_states=n,
        distance_matrix=D,
        diameter=diam,
        tropical_eigenvalue=trop_eig,
        tropical_determinant=trop_det,
        longest_path_top_bottom=longest,
        shortest_path_top_bottom=shortest,
        eccentricity=ecc,
        radius=rad,
        center=ctr,
    )
