"""Transfer matrix analysis for session type lattices (Step 30f).

The transfer matrix T encodes the weighted adjacency of the Hasse diagram.
T^k[i,j] counts the number of paths of length exactly k from state i to j.
This module provides:

- **Transfer matrix**: T[i,j] = 1 if i covers j in Hasse diagram
- **Path counting**: T^k[i,j] = number of length-k directed paths
- **Generating function**: Σ_k T^k z^k = (I - zT)^{-1} (resolvent)
- **Total path count**: number of all directed paths from top to bottom
- **Expected path length**: average path length from top to bottom
- **Transfer matrix spectrum**: eigenvalues of the directed transfer matrix
- **Composition under parallel**: T(L₁×L₂) structure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _state_list,
    _covering_relation,
    _compute_sccs,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransferResult:
    """Complete transfer matrix analysis.

    Attributes:
        num_states: Number of states.
        transfer_matrix: The directed Hasse (covering) matrix.
        states: State ID list matching matrix indices.
        total_paths: Number of directed paths from top to bottom.
        path_length_distribution: k → number of paths of length k.
        expected_path_length: Average path length (weighted by count).
        height: Length of longest path from top to bottom.
        is_acyclic: True iff the covering relation has no cycles.
    """
    num_states: int
    transfer_matrix: list[list[int]]
    states: list[int]
    total_paths: int
    path_length_distribution: dict[int, int]
    expected_path_length: float
    height: int
    is_acyclic: bool


# ---------------------------------------------------------------------------
# Transfer matrix computation
# ---------------------------------------------------------------------------

def transfer_matrix(ss: "StateSpace") -> tuple[list[list[int]], list[int]]:
    """Compute the directed transfer (covering) matrix.

    T[i,j] = 1 if state_i covers state_j in the Hasse diagram.
    Returns (matrix, states) where states maps indices to IDs.
    """
    states = _state_list(ss)
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    covers = _covering_relation(ss)

    T = [[0] * n for _ in range(n)]
    for x, y in covers:
        if x in idx and y in idx:
            T[idx[x]][idx[y]] = 1

    return T, states


def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Integer matrix multiplication."""
    n = len(A)
    m = len(B[0]) if B else 0
    p = len(B)
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


# ---------------------------------------------------------------------------
# Path counting
# ---------------------------------------------------------------------------

def count_paths_by_length(ss: "StateSpace") -> dict[int, int]:
    """Count directed paths from top to bottom by length.

    path_counts[k] = number of directed paths of length k from top to bottom,
    where each step follows a covering relation.
    """
    T, states = transfer_matrix(ss)
    n = len(states)
    if n == 0:
        return {}

    idx = {s: i for i, s in enumerate(states)}
    top_i = idx.get(ss.top, 0)
    bot_i = idx.get(ss.bottom, 0)

    if top_i == bot_i:
        return {0: 1}

    counts: dict[int, int] = {}
    power = [row[:] for row in T]  # T^1

    for k in range(1, n + 1):
        val = power[top_i][bot_i]
        if val > 0:
            counts[k] = val

        # Check if all zeros
        all_zero = all(power[i][j] == 0 for i in range(n) for j in range(n))
        if all_zero:
            break

        power = _mat_mul(power, T)

    return counts


def total_path_count(ss: "StateSpace") -> int:
    """Total number of directed paths from top to bottom (all lengths)."""
    counts = count_paths_by_length(ss)
    return sum(counts.values())


def expected_path_length(ss: "StateSpace") -> float:
    """Expected (average) path length from top to bottom.

    Weighted average: Σ k·c_k / Σ c_k.
    """
    counts = count_paths_by_length(ss)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(k * c for k, c in counts.items()) / total


# ---------------------------------------------------------------------------
# Transfer matrix powers
# ---------------------------------------------------------------------------

def transfer_power(ss: "StateSpace", k: int) -> list[list[int]]:
    """Compute T^k — the k-th power of the transfer matrix.

    T^k[i,j] counts paths of length exactly k from i to j.
    """
    T, states = transfer_matrix(ss)
    n = len(states)
    # Identity for k=0
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in T]
    exp = k
    while exp > 0:
        if exp % 2 == 1:
            result = _mat_mul(result, base)
        base = _mat_mul(base, base)
        exp //= 2
    return result


# ---------------------------------------------------------------------------
# Path enumeration (explicit)
# ---------------------------------------------------------------------------

def enumerate_paths(ss: "StateSpace", max_paths: int = 1000) -> list[list[int]]:
    """Enumerate all directed paths from top to bottom.

    Returns list of paths, where each path is a list of state IDs.
    Stops after max_paths to avoid explosion.
    """
    T, states = transfer_matrix(ss)
    idx = {s: i for i, s in enumerate(states)}

    # Build adjacency from T
    adj: dict[int, list[int]] = {s: [] for s in states}
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if T[i][j] > 0:
                adj[si].append(sj)

    paths: list[list[int]] = []

    def _dfs(current: int, path: list[int]) -> None:
        if len(paths) >= max_paths:
            return
        if current == ss.bottom:
            paths.append(path[:])
            return
        for nxt in adj[current]:
            path.append(nxt)
            _dfs(nxt, path)
            path.pop()

    _dfs(ss.top, [ss.top])
    return paths


# ---------------------------------------------------------------------------
# Resolvent (generating function)
# ---------------------------------------------------------------------------

def resolvent_at(ss: "StateSpace", z: float) -> list[list[float]] | None:
    """Compute (I - zT)^{-1} — the resolvent / generating function.

    Returns None if the matrix is singular.
    For |z| < 1/spectral_radius(T), this converges and equals Σ_k (zT)^k.
    """
    T, states = transfer_matrix(ss)
    n = len(states)

    # Build I - zT
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = (1.0 if i == j else 0.0) - z * T[i][j]

    # Gaussian elimination to find inverse
    # Augment with identity
    aug = [[0.0] * (2 * n) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            aug[i][j] = M[i][j]
            aug[i][n + j] = 1.0 if i == j else 0.0

    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(col, n):
            if abs(aug[row][col]) > 1e-12:
                pivot = row
                break
        if pivot == -1:
            return None  # Singular

        # Swap
        aug[col], aug[pivot] = aug[pivot], aug[col]

        # Scale
        scale = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= scale

        # Eliminate
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    # Extract inverse
    inv = [[aug[i][n + j] for j in range(n)] for i in range(n)]
    return inv


# ---------------------------------------------------------------------------
# Acyclicity check
# ---------------------------------------------------------------------------

def is_transfer_acyclic(ss: "StateSpace") -> bool:
    """Check if the transfer (covering) matrix defines an acyclic graph.

    The covering relation of a poset is always acyclic, but we verify
    this computationally.
    """
    T, states = transfer_matrix(ss)
    n = len(states)

    # Check if T^n is zero (no paths of length n → no cycles)
    power = [row[:] for row in T]
    for _ in range(n):
        power = _mat_mul(power, T)

    # After n multiplications, T^{n+1} should be zero for acyclic
    return all(power[i][j] == 0 for i in range(n) for j in range(n))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_transfer(ss: "StateSpace") -> TransferResult:
    """Complete transfer matrix analysis."""
    T, states = transfer_matrix(ss)
    counts = count_paths_by_length(ss)
    total = sum(counts.values())
    avg_len = expected_path_length(ss)
    height = max(counts.keys()) if counts else 0
    acyclic = is_transfer_acyclic(ss)

    return TransferResult(
        num_states=len(states),
        transfer_matrix=T,
        states=states,
        total_paths=total,
        path_length_distribution=counts,
        expected_path_length=avg_len,
        height=height,
        is_acyclic=acyclic,
    )
